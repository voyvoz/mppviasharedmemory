//! IPC endpoint over CXL shared memory, generalised to N hosts.
//!
//! Topology: N*(N-1) directed inbox rings. Each ring is 1 producer and
//! 1 consumer.
//!
//!   index 0 : 1 -> 2
//!   index 1:  1 -> 3
//!   index 2:  2 -> 1
//!   index 3:  2 -> 3
//!   index 4:  3 -> 1
//!   index 5:  3 -> 2
//!

use crate::core::host::{self, MAX_HOSTS, PeerID, PeerMachineID, get_machine_id};
use crate::core::structs::{CXLRingBuffer, RingError};
use crate::core::{CXLRBParams, Cacheline, RBSlice};
use crate::cxl::cxl_typed_new_sync;
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync};
use crate::cxl_v2::CXLSizedObject;
use crate::utils::timestamps::SpinCounter;
use allocator_api2::alloc::Allocator;
use anyhow::{Result, anyhow};
use log::{debug, info, trace};
use std::alloc::Layout;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ------------------------------------------------------
// CXLIpcOptions
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct CXLIpcOptions {
    // How many hosts are expected
    pub hosts: usize,
    // Used for all created ringbuffer
    pub rb_opts: CXLRBParams,
}
impl CXLIpcOptions {
    pub fn default() -> Self {
        CXLIpcOptions {
            hosts: MAX_HOSTS,
            rb_opts: CXLRBParams::default(),
        }
    }
    pub fn with_hosts(mut self, num_hosts: usize) -> Self {
        self.hosts = num_hosts;
        self
    }
    pub fn with_ringbuffer_options(mut self, rb_opts: CXLRBParams) -> Self {
        self.rb_opts = rb_opts;
        self
    }
}

impl CXLSizedObject for CXLIpcOptions {
    fn required_cachelines(&self) -> usize {
        // Header + N*(N-1) * RB
        1 + ((self.hosts * (self.hosts - 1)) * self.rb_opts.required_cachelines())
    }
}

// ------------------------------------------------------
// CXLIpc
// ------------------------------------------------------

/// Sentinel meaning "this message has no accociated CXL object".
pub const IPC_NO_OFFSET: u64 = u64::MAX;

///
#[derive(Debug)]
pub struct CXLIpc {
    base_cl: *mut Cacheline,

    n_hosts: usize, // Assumed to be constant during lifetime

    rr_cursor: usize, // Round-robin cursor for fairness across multiple peers.

    writers: Vec<Option<CXLRingBuffer>>,
    readers: Vec<Option<CXLRingBuffer>>,

    pub options: CXLIpcOptions,
}

impl CXLIpc {
    #[inline]
    pub fn get_base_cl(&self) -> *mut Cacheline {
        self.base_cl
    }
    #[inline]
    fn rb_start(base: *mut Cacheline) -> *mut Cacheline {
        // Need to skip the header
        unsafe { base.add(1) }
    }
    #[inline]
    fn ring_index(src: PeerMachineID, dst: PeerMachineID, n: PeerMachineID) -> usize {
        debug_assert!(src >= 1 && dst >= 1, "machine ids start at 1");
        debug_assert!(src <= n && dst <= n, "machine id out of range");
        debug_assert!(src != dst, "self ring forbidden");
        let s = (src - 1) as usize;
        let d = (dst - 1) as usize;
        let nn = n as usize;
        s * (nn - 1) + if d > s { d - 1 } else { d }
    }
    /// `id` starts at 1 but when used as an index we need to convert
    #[inline]
    fn peer_to_index(id: PeerMachineID) -> usize {
        (id - 1) as usize
    }
    #[inline]
    fn is_this_machine(peer: PeerMachineID) -> bool {
        PeerID::whoami().machine_id == peer
    }
    #[inline]
    fn peer_to_writer(&mut self, peer: PeerMachineID) -> &mut CXLRingBuffer {
        self.writers[CXLIpc::peer_to_index(peer)].as_mut().unwrap()
    }
    #[inline]
    fn peer_to_reader(&mut self, peer: PeerMachineID) -> &mut CXLRingBuffer {
        self.readers[CXLIpc::peer_to_index(peer)].as_mut().unwrap()
    }

    fn ring_base_cl(
        base: *mut Cacheline,
        src: PeerMachineID,
        dst: PeerMachineID,
        config: &CXLIpcOptions,
    ) -> *mut Cacheline {
        let idx = CXLIpc::ring_index(src, dst, config.hosts as u16);
        let stride = config.rb_opts.required_cachelines();
        let result = unsafe { CXLIpc::rb_start(base).add(idx * stride) };
        trace!(
            "ring_base_cl(base={base:p}, src={src}, dst={dst}, n={}) -> {idx} @ {:p}",
            config.hosts, result,
        );
        result
    }

    /// Create a new IPC strucutre to interconnect `n_hosts` using `alloc` for memory allocations.
    /// Once created this structure remains static.
    pub fn new(ipc_opts: CXLIpcOptions, alloc: Arc<impl Allocator>) -> Self {
        debug!("CXLIpc::new(ipc_opts={ipc_opts:?})");

        let mem_layout = Layout::from_size_align(ipc_opts.required_bytes(), 64).unwrap();
        let mem = alloc.allocate(mem_layout).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        CXLIpc::init(base_cl, ipc_opts)
    }

    /// Creates all necessary data structures for the given number of `n_hosts` starting at `base_cl`.
    pub fn init(base_cl: *mut Cacheline, ipc_opts: CXLIpcOptions) -> Self {
        let n_hosts = ipc_opts.hosts;
        assert!(n_hosts >= 2, "ipc: n_hosts must be >= 2, got {n_hosts}");
        let host_id = PeerID::whoami().machine_id as usize;

        debug!("CXLIpc::init(base_cl={base_cl:?}, options={ipc_opts:?}) -> host_id:{host_id}");

        cxl_typed_new_sync::<Self>(base_cl);

        let mut writers: Vec<Option<CXLRingBuffer>> = (0..n_hosts).map(|_| None).collect();
        let mut readers: Vec<Option<CXLRingBuffer>> = (0..n_hosts).map(|_| None).collect();

        for src in 1..=n_hosts {
            for dst in 1..=n_hosts {
                if src == dst {
                    continue;
                }
                let ring_addr = CXLIpc::ring_base_cl(
                    base_cl,
                    src as PeerMachineID,
                    dst as PeerMachineID,
                    &ipc_opts,
                );
                let rb = CXLRingBuffer::init(ring_addr, ipc_opts.rb_opts);
                trace!("src:{src} -> dst:{dst} @ {ring_addr:?}: {rb:?}");
            }
        }
        // TODO this is not correct... dont use from_cl here
        // Problem is the increasing ref counter when calling from_cl
        for src in 1..=n_hosts {
            if src != host_id {
                let peer_id = src as u16;
                let idx = src - 1;
                writers[idx] = Some(
                    CXLRingBuffer::from_cl_sync(
                        CXLIpc::ring_base_cl(base_cl, host_id as u16, peer_id, &ipc_opts),
                        ipc_opts.rb_opts,
                    )
                    .unwrap(),
                );
                trace!("Writer self {} -> {peer_id} peer", host_id);
                readers[idx] = Some(
                    CXLRingBuffer::from_cl_sync(
                        CXLIpc::ring_base_cl(base_cl, peer_id, host_id as u16, &ipc_opts),
                        ipc_opts.rb_opts,
                    )
                    .unwrap(),
                );
                trace!("Reader peer {peer_id} -> {} self", host_id);
            }
        }

        CXLIpc {
            base_cl,
            n_hosts,
            readers,
            writers,
            rr_cursor: 0,
            options: ipc_opts,
        }
    }

    /// Creates the Endpoint using `base` assuming that all shared state has already been created.
    /// Uses `base` as a header to derive necessary parameters.
    pub fn from_cl_sync(base_cl: *mut Cacheline, ipc_opts: CXLIpcOptions) -> Option<Self> {
        let host_id = PeerID::whoami();

        debug!(
            "CXLIpc::from_cl(base_cl={base_cl:?}) -> host_id:{}",
            host_id.machine_id
        );

        let res = cxl_typed_from_cl_sync::<Self>(base_cl, |_meta| {
            let n_hosts = ipc_opts.hosts;
            CXLIpc {
                base_cl,
                n_hosts,
                writers: (0..n_hosts).map(|_| None).collect(),
                readers: (0..n_hosts).map(|_| None).collect(),
                rr_cursor: 0,
                options: ipc_opts,
            }
        });
        if res.is_none() {
            return res;
        }
        let mut ipc = res.unwrap();

        for peer_id_m in 1..(ipc.n_hosts as PeerMachineID) + 1 {
            if peer_id_m == host_id.machine_id {
                continue;
            }
            let peer_id = PeerID::from_machine_id(peer_id_m).machine_id;
            let idx = (peer_id_m - 1) as usize;

            ipc.writers[idx] = Some(
                CXLRingBuffer::from_cl_sync(
                    CXLIpc::ring_base_cl(base_cl, host_id.machine_id, peer_id, &ipc_opts),
                    ipc_opts.rb_opts,
                )
                .unwrap(),
            );
            trace!("Writer self {} -> {peer_id} peer", host_id.machine_id);

            ipc.readers[idx] = Some(
                CXLRingBuffer::from_cl_sync(
                    CXLIpc::ring_base_cl(base_cl, peer_id, host_id.machine_id, &ipc_opts),
                    ipc_opts.rb_opts,
                )
                .unwrap(),
            );
            trace!("Reader peer {peer_id} -> {} self", host_id.machine_id);
        }
        Some(ipc)
    }

    // -------------------------------------

    /// Send `Payload` to the `target_peer`.
    pub fn try_send_to(&mut self, target_peer: PeerMachineID, payload: &Cacheline) -> Result<()> {
        if target_peer == host::MACHINE_ID_INVALID {
            return Err(anyhow!("ipc send: cannot send invalid machine id"));
        }
        if target_peer as usize > self.n_hosts {
            return Err(anyhow!(
                "ipc send: target_host={} out of range (n_hosts={})",
                target_peer,
                self.n_hosts
            ));
        }
        if CXLIpc::is_this_machine(target_peer) {
            return Err(anyhow!("ipc send: cannot send to self"));
        }

        let rb = self.peer_to_writer(target_peer);
        trace!("CXLIpc::try_send_to({target_peer}, {payload:?}) -> {rb:?}");
        match rb.try_push(&payload) {
            Ok(()) => Ok(()),
            Err(RingError::Full) => Err(anyhow!(
                "ipc send: inbox[{}->{}] full",
                PeerID::whoami().machine_id,
                target_peer
            )),
            Err(e) => Err(anyhow!("ipc send: {:?}", e)),
        }
    }

    /// Send `Payload` to each registered host that is not this machine.
    pub fn try_broadcast(&mut self, payload: &Cacheline) -> Result<()> {
        for i in 0..self.n_hosts {
            let target = i as u16 + 1u16;
            // Dont send to self
            if target == get_machine_id() {
                continue;
            }
            self.try_send_to(target, payload)?;
        }
        Ok(())
    }

    /// Block until a message arrives from `peer` specifically or `timeout` runs out.
    pub fn try_pop_from(
        &mut self,
        target_peer: PeerMachineID,
        timeout: Option<Duration>,
    ) -> Result<(PeerID, Cacheline)> {
        trace!("CXLIpc::try_pop_from(target_peer={target_peer}, timeout={timeout:?})");

        if target_peer == host::MACHINE_ID_INVALID {
            return Err(anyhow!("try_pop_from: cannot pop from invalid machine id"));
        }
        if target_peer as usize > self.n_hosts {
            return Err(anyhow!(
                "try_pop_from: peer={target_peer} out of range (n_hosts={})",
                self.n_hosts
            ));
        }
        if CXLIpc::is_this_machine(target_peer) {
            return Err(anyhow!("try_pop_from: cannot pop on self"));
        }

        let rb = self.peer_to_reader(target_peer);
        let start = Instant::now();
        let mut spins = SpinCounter::new();
        loop {
            match rb.try_pop() {
                Ok(cl) => {
                    trace!("try_pop_from({target_peer:?}) -> {rb:?} -> {cl:?}");
                    return Ok((PeerID::from_machine_id(target_peer), cl));
                }
                Err(RingError::Empty) => {
                    if let Some(t) = timeout {
                        if start.elapsed() >= t {
                            return Err(anyhow!("try_pop_from: timeout"));
                        }
                    }
                    spins.wait();
                }
                Err(e) => return Err(anyhow!("ipc await_from: {:?}", e)),
            }
        }
    }

    ///  Non-blocking. Polls all inbox readers round-robin.
    pub fn try_pop_any(&mut self) -> Result<Option<(PeerID, Cacheline)>> {
        trace!("CXLIpc::try_pop_any()");
        let n = self.n_hosts as usize;
        for i in 0..n {
            let peer = (self.rr_cursor + i) % n;
            if CXLIpc::is_this_machine((peer + 1) as u16) {
                continue;
            }
            let reader = match self.readers[peer].as_mut() {
                Some(r) => r,
                None => continue,
            };
            match reader.try_pop() {
                Ok(cl) => {
                    self.rr_cursor = (peer + 1) % n;
                    return Ok(Some((PeerID::from_machine_id(peer as u16), cl)));
                }
                Err(RingError::Empty) => continue,
                Err(e) => return Err(anyhow!("ipc try_recv: {:?}", e)),
            }
        }
        Ok(None)
    }

    pub fn try_pop_all_from_any_with_state_machine(&mut self) -> Result<Option<(PeerID, Vec<Cacheline>)>> {
        trace!("CXLIpc::try_pop_any()");
        let n = self.n_hosts as usize;
        for i in 0..n {
            let peer = (self.rr_cursor + i) % n;
            if CXLIpc::is_this_machine((peer + 1) as u16) {
                continue;
            }
            let reader = match self.readers[peer].as_mut() {
                Some(r) => r,
                None => continue,
            };
            if let Some(res) = reader.try_pop_all_with_state_machine() {
                match res {
                    Ok(cl) => {
                        self.rr_cursor = (peer + 1) % n;
                        return Ok(Some((PeerID::from_machine_id(peer as u16), cl)));
                    }
                    Err(RingError::Empty) => continue,
                    Err(e) => return Err(anyhow!("ipc try_recv: {:?}", e)),
                }
            }
        }
        Ok(None)
    }

    /// Block until any message arrives, or `timeout` elapses.
    pub fn await_any_msg(&mut self, timeout: Option<Duration>) -> Result<(PeerID, Cacheline)> {
        trace!("CXLIpc::await_any_msg(timeout={timeout:?})");
        let start = Instant::now();
        let mut spins = SpinCounter::new();
        loop {
            if let Some(r) = self.try_pop_any()? {
                return Ok(r);
            }

            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    return Err(anyhow!("ipc await: timeout"));
                }
            }
            spins.wait();
        }
    }

    //
    // Slice variants for messages of arbitrary size
    //

    pub fn try_send_slice_to(&mut self, target_peer: PeerMachineID, payload: &[u8]) -> Result<()> {
        if target_peer == host::MACHINE_ID_INVALID {
            return Err(anyhow!("ipc send: cannot send invalid machine id"));
        }
        if target_peer as usize > self.n_hosts {
            return Err(anyhow!(
                "ipc send: target_host={} out of range (n_hosts={})",
                target_peer,
                self.n_hosts
            ));
        }
        if CXLIpc::is_this_machine(target_peer) {
            return Err(anyhow!("ipc send: cannot send to self"));
        }

        let rb = self.peer_to_writer(target_peer);
        let rbslice = RBSlice::from_slice(payload);
        match rb.try_push_slice(rbslice) {
            Ok(()) => Ok(()),
            Err(RingError::Full) => Err(anyhow!(
                "ipc send: inbox[{}->{}] full",
                PeerID::whoami().machine_id,
                target_peer
            )),
            Err(e) => Err(anyhow!("ipc send: {:?}", e)),
        }
    }

    pub fn try_pop_any_slice(&mut self) -> Result<Option<(PeerID, RBSlice)>> {
        trace!("CXLIpc::try_pop_any_slice()");
        let n = self.n_hosts as usize;
        for i in 0..n {
            let peer = (self.rr_cursor + i) % n;
            if CXLIpc::is_this_machine((peer + 1) as u16) {
                continue;
            }
            let reader = match self.readers[peer].as_mut() {
                Some(r) => r,
                None => continue,
            };
            match reader.try_pop_slice() {
                Ok(slice) => {
                    self.rr_cursor = (peer + 1) % n;
                    return Ok(Some((PeerID::from_machine_id(peer as u16), slice)));
                }
                Err(RingError::Empty) => continue,
                Err(e) => return Err(anyhow!("ipc try_recv: {:?}", e)),
            }
        }
        Ok(None)
    }

    /// Block until a slice arrives from one specific peer, or `timeout`
    /// elapses. A dedicated bridge worker can use this instead of repeatedly
    /// scanning every peer ring, which keeps receive latency independent of
    /// the number of hosts.
    pub fn await_slice_from(
        &mut self,
        target_peer: PeerMachineID,
        timeout: Option<Duration>,
    ) -> Result<(PeerID, RBSlice)> {
        if target_peer == host::MACHINE_ID_INVALID {
            return Err(anyhow!("await_slice_from: invalid machine id"));
        }
        if target_peer as usize > self.n_hosts {
            return Err(anyhow!(
                "await_slice_from: peer={target_peer} out of range (n_hosts={})",
                self.n_hosts
            ));
        }
        if CXLIpc::is_this_machine(target_peer) {
            return Err(anyhow!("await_slice_from: cannot receive from self"));
        }

        let reader = self.peer_to_reader(target_peer);
        let start = Instant::now();
        let mut spins = SpinCounter::new();
        loop {
            match reader.try_pop_slice() {
                Ok(slice) => return Ok((PeerID::from_machine_id(target_peer), slice)),
                Err(RingError::Empty) => {
                    if let Some(limit) = timeout
                        && start.elapsed() >= limit
                    {
                        return Err(anyhow!("await_slice_from: timeout"));
                    }
                    spins.wait();
                }
                Err(error) => return Err(anyhow!("await_slice_from: {error:?}")),
            }
        }
    }

    pub fn await_any_slice(&mut self, timeout: Option<Duration>) -> Result<(PeerID, RBSlice)> {
        trace!("CXLIpc::try_pop_any_slice(timeout={timeout:?})");
        let start = Instant::now();
        let mut spins = SpinCounter::new();
        loop {
            if let Some(r) = self.try_pop_any_slice()? {
                return Ok(r);
            }

            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    return Err(anyhow!("ipc await: timeout"));
                }
            }
            spins.wait();
        }
    }

    pub fn try_pop_any_all_slice(&mut self) -> Result<Option<(PeerID, Vec<RBSlice>)>> {
        trace!("CXLIpc::try_pop_any_slice()");
        let n = self.n_hosts as usize;
        for i in 0..n {
            let peer = (self.rr_cursor + i) % n;
            if CXLIpc::is_this_machine((peer + 1) as u16) {
                continue;
            }
            let reader = match self.readers[peer].as_mut() {
                Some(r) => r,
                None => continue,
            };
            match reader.try_pop_all_slice() {
                Ok(slice) => {
                    self.rr_cursor = (peer + 1) % n;
                    return Ok(Some((PeerID::from_machine_id(peer as u16), slice)));
                }
                Err(RingError::Empty) => continue,
                Err(e) => return Err(anyhow!("ipc try_recv: {:?}", e)),
            }
        }
        Ok(None)
    }

    pub fn await_any_all_slice(&mut self, timeout: Option<Duration>) -> Result<(PeerID, Vec<RBSlice>)> {
        trace!("CXLIpc::await_any_slice_all(timeout={timeout:?})");
        let start = Instant::now();
        let mut spins = SpinCounter::new();
        loop {
            if let Some(r) = self.try_pop_any_all_slice()? {
                return Ok(r);
            }

            if let Some(t) = timeout {
                if start.elapsed() >= t {
                    return Err(anyhow!("ipc await: timeout"));
                }
            }
            spins.wait();
        }
    }
}

unsafe impl Send for CXLIpc {}

impl CXLSharedState for CXLIpc {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        self.options.required_cachelines()
    }
    fn get_cxl_type_id() -> u64 {
        1234
    }
}

#[cfg(test)]
mod test {
    use crate::allocator::os::OSAlloc;
    use crate::com::ipc::{CXLIpc, CXLIpcOptions};
    use crate::core::Cacheline;
    use crate::core::host::{PeerID, PeerMachineID, set_machine_id};
    use crate::core::set_global_offset;
    use crate::utils::open_test_cxl_dev;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn base() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);
        set_machine_id(1);

        let hosts: usize = 2;
        let config = CXLIpcOptions::default().with_hosts(hosts);

        let ipc = CXLIpc::new(config, Arc::new(os_alloc));

        let base_ptr = ipc.base_cl as u64;
        let mut threads = vec![];
        for machine_id in 1..(hosts as PeerMachineID) + 1 {
            threads.push(thread::spawn(move || {
                set_machine_id(machine_id);
                let my_id = PeerID::whoami();
                assert_eq!(my_id.machine_id, machine_id);

                let mut my_ipc_ep =
                    CXLIpc::from_cl_sync(base_ptr as *mut Cacheline, config).unwrap();
                assert_eq!(hosts, my_ipc_ep.n_hosts);

                let mut dummy_data = Cacheline::empty();

                // Send a message to all hosts using their machine_id
                for i in 1..(hosts as PeerMachineID) + 1 {
                    let target_machine = PeerID::from_machine_id(i);
                    // Store target in payload -> helps us check if we got the correct message back
                    dummy_data.d[0] = target_machine.machine_id as u64;
                    dummy_data.d[1] = my_id.machine_id as u64;
                    let res = my_ipc_ep.try_send_to(target_machine.machine_id, &dummy_data);
                    if target_machine.eq(&my_id) {
                        assert!(res.is_err());
                    } else {
                        res.unwrap();
                    }
                }

                // Check that we recieved the correct message from all hosts
                for i in 1..(hosts as PeerMachineID) + 1 {
                    let target_machine = PeerID::from_machine_id(i);
                    let res = my_ipc_ep.try_pop_from(target_machine.machine_id, None);
                    // Expected to fail for i == my_id as we cant wait on messges from our own host
                    if target_machine.eq(&my_id) {
                        assert!(res.is_err());
                    } else {
                        let (res_peer, res_msg) = res.unwrap();
                        assert_eq!(res_peer.machine_id, target_machine.machine_id);
                        // Make sure that payload contains correct host id
                        assert_eq!(res_msg.d[0], my_id.machine_id as u64);
                        assert_eq!(res_msg.d[1], target_machine.machine_id as u64);
                    }
                }
            }));
        }
        for i in threads {
            i.join().unwrap();
        }
    }

    #[test]
    fn base_cxl() {
        let (mut base, _) = open_test_cxl_dev();

        base = unsafe { base.add(4096) };

        let hosts: PeerMachineID = 5;
        let config = CXLIpcOptions::default().with_hosts(hosts as usize);

        set_machine_id(1);
        CXLIpc::init(base, config);

        let base_ptr = base as usize;
        let mut threads = vec![];
        for machine_id in 1..hosts + 1 {
            threads.push(thread::spawn(move || {
                set_machine_id(machine_id);
                let my_id = PeerID::whoami();
                assert_eq!(my_id.machine_id, machine_id);

                let mut my_ipc_ep =
                    CXLIpc::from_cl_sync(base_ptr as *mut Cacheline, config).unwrap();
                assert_eq!(hosts as usize, my_ipc_ep.n_hosts);

                let mut dummy_data = Cacheline::empty();

                // Send a message to all hosts using their machine_id
                for i in 1..hosts + 1 {
                    let target_machine = PeerID::from_machine_id(i);
                    // Store target in payload -> helps us check if we got the correct message back
                    dummy_data.d[0] = target_machine.machine_id as u64;
                    dummy_data.d[1] = my_id.machine_id as u64;
                    let res = my_ipc_ep.try_send_to(target_machine.machine_id, &dummy_data);
                    if target_machine.eq(&my_id) {
                        assert!(res.is_err());
                    } else {
                        res.unwrap();
                    }
                }

                // Check that we recieved the correct message from all hosts
                for i in 1..hosts + 1 {
                    let target_machine = PeerID::from_machine_id(i);
                    let res = my_ipc_ep.try_pop_from(target_machine.machine_id, None);
                    // Expected to fail for i == my_id as we cant wait on messges from our own host
                    if target_machine.eq(&my_id) {
                        assert!(res.is_err());
                    } else {
                        let (res_peer, res_msg) = res.unwrap();
                        assert_eq!(res_peer.machine_id, target_machine.machine_id);
                        // Make sure that payload contains correct host id
                        assert_eq!(res_msg.d[0], my_id.machine_id as u64);
                        assert_eq!(res_msg.d[1], target_machine.machine_id as u64);
                    }
                }
            }));
        }
        for i in threads {
            i.join().unwrap();
        }
    }

    #[test]
    fn ring_index_layout_is_consistent() {
        // For N=3, init() and ring_base_cl() must hit the same 6 distinct positions
        // contiguously starting at 0.
        let mut seen = [false; 6];
        for src in 1..=3 {
            for dst in 1..=3 {
                if src == dst {
                    continue;
                }
                let idx = CXLIpc::ring_index(src, dst, 3);
                assert!(idx < 6, "({src},{dst}) -> {idx}");
                assert!(!seen[idx], "({src},{dst}) collided at {idx}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|b| *b));
    }
    /*
    const N: usize = 3;

    fn setup() -> (IpcEndpoint, IpcEndpoint, IpcEndpoint) {
        let (base, _) = open_test_cxl_dev();
        let pds = PersistantDataStructs::load(base as *mut Cacheline, true);
        let ep0 = IpcEndpoint::init(&pds, 0, N, true).unwrap();
        let ep1 = IpcEndpoint::init(&pds, 1, N, false).unwrap();
        let ep2 = IpcEndpoint::init(&pds, 2, N, false).unwrap();
        (ep0, ep1, ep2)
    }

    #[test]
    fn ipc3_directed_send() {
        let (mut ep0, mut ep1, mut ep2) = setup();

        ep0.send(1, None, 0xA).unwrap();
        ep0.send(2, None, 0xB).unwrap();
        ep1.send(2, None, 0xC).unwrap();

        let r1 = ep1.await_msg(Some(Duration::from_secs(2))).unwrap();
        assert_eq!(r1.sender_host, 0);
        assert_eq!(r1.user_tag, 0xA);

        let r2a = ep2.await_msg(Some(Duration::from_secs(2))).unwrap();
        let r2b = ep2.await_msg(Some(Duration::from_secs(2))).unwrap();
        let got: std::collections::BTreeSet<(usize, u64)> = [
            (r2a.sender_host, r2a.user_tag),
            (r2b.sender_host, r2b.user_tag),
        ]
        .into();
        let want: std::collections::BTreeSet<(usize, u64)> = [(0, 0xB), (1, 0xC)].into();
        assert_eq!(got, want);
    }

    #[test]
    fn ipc_self_send_rejected() {
        let (mut ep0, _ep1, _ep2) = setup();
        assert!(ep0.send(0, None, 1).is_err());
        assert!(ep0.send(99, None, 1).is_err());
    }

    #[test]
    fn ipc_three_way_relay() {
        // A -> B -> C -> A
        let (mut ep0, mut ep1, mut ep2) = setup();
        ep0.send(1, Some(0x1000), 0xAA).unwrap();
        let r_b = ep1.await_msg(Some(Duration::from_secs(1))).unwrap();
        assert_eq!(r_b.sender_host, 0);
        assert_eq!(r_b.obj_offset, Some(0x1000));
        assert_eq!(r_b.user_tag, 0xAA);
        ep1.send(2, Some(0x2000), 0xBB).unwrap();
        let r_c = ep2.await_msg(Some(Duration::from_secs(1))).unwrap();
        assert_eq!(r_c.sender_host, 1);
        assert_eq!(r_c.obj_offset, Some(0x2000));
        ep2.send(0, Some(0x3000), 0xCC).unwrap();
        let r_a = ep0.await_msg(Some(Duration::from_secs(1))).unwrap();
        assert_eq!(r_a.sender_host, 2);
        assert_eq!(r_a.obj_offset, Some(0x3000));
        assert_eq!(r_a.user_tag, 0xCC);
    }

    #[test]
    fn ipc_await_from_specific() {
        let (mut ep0, mut ep1, mut ep2) = setup();
        ep1.send(0, None, 0xB).unwrap();
        ep2.send(0, None, 0xC).unwrap();
        // Drain peer 2 first even though peer 1's message arrived first in absolute time.
        let r2 = ep0.await_from(2, Some(Duration::from_secs(1))).unwrap();
        assert_eq!(r2.sender_host, 2);
        assert_eq!(r2.user_tag, 0xC);
        let r1 = ep0.await_from(1, Some(Duration::from_secs(1))).unwrap();
        assert_eq!(r1.sender_host, 1);
        assert_eq!(r1.user_tag, 0xB);
    }

    #[test]
    fn ipc_queue_full_per_pair() {
        let (mut ep0, mut ep1, _ep2) = setup();
        for i in 0..RING_CAPACITY as u64 {
            ep0.send(1, None, i).unwrap();
        }
        // 0 -> 1 ring is full now.
        assert!(ep0.send(1, None, 9999).is_err());
        // But 0 -> 2 must still be empty and usable.
        ep0.send(2, None, 1234).unwrap();
        for i in 0..RING_CAPACITY as u64 {
            let r = ep1.await_msg(Some(Duration::from_secs(1))).unwrap();
            assert_eq!(r.user_tag, i);
        }
    }

    #[test]
    fn ipc_no_msg_returns_none() {
        let (mut ep0, mut ep1, mut ep2) = setup();

        assert!(matches!(ep0.try_recv(), Ok(None)));
        assert!(matches!(ep1.try_recv(), Ok(None)));
        assert!(matches!(ep2.try_recv(), Ok(None)));

        ep1.send(0, None, 1).unwrap();
        let r = ep0.try_recv().unwrap();
        assert!(r.is_some());

        assert!(matches!(ep0.try_recv(), Ok(None))); // empty after drain
    }
    */
}
