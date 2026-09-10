use crate::core::host::{MAX_HOSTS, get_machine_cnt, get_machine_id};
use crate::core::{Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync};
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync, cxl_typed_new_sync};
use crate::cxl_v2::CXLSizedObject;
use crate::cxl_write_sync;
use crate::utils::timestamps::SpinCounter;
use allocator_api2::alloc::Allocator;
use log::{debug, trace};
use std::alloc::Layout;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

static GLOBAL_BARRIER: OnceLock<CXLBarrier> = OnceLock::new();

pub fn barrier_wait_all() {
    let nhosts = get_machine_cnt();
    debug!(
        "barrier_wait_all() -> id:{} / nhosts:{nhosts}",
        get_machine_id()
    );
    GLOBAL_BARRIER
        .get()
        .expect(
            "lib not correctly initialized. Must use `set_barrier` before calling this function.",
        )
        .wait_for(nhosts);
}
pub fn barrier_wait_for(nhosts: u16) {
    debug!(
        "barrier_wait_for(nhosts={nhosts}) -> id:{}",
        get_machine_id()
    );
    GLOBAL_BARRIER
        .get()
        .expect(
            "lib not correctly initialized. Must use `set_barrier` before calling this function.",
        )
        .wait_for(nhosts);
}

// overwrite hostname
pub fn set_global_barrier(b: CXLBarrier) {
    GLOBAL_BARRIER.set(b).expect("Barrier can be set only once");
}

// ------------------------------------------------------
// CXLBarrierOptions
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct CXLBarrierOptions {
    pub registered_hosts: usize,
}
impl CXLBarrierOptions {
    pub fn from_hosts(num_hosts: usize) -> Self {
        CXLBarrierOptions {
            registered_hosts: num_hosts,
        }
    }
}

impl CXLSizedObject for CXLBarrierOptions {
    fn required_cachelines(&self) -> usize {
        // Meta data + 1 * hosts
        1 + self.registered_hosts
    }
}

// ------------------------------------------------------
// CXLBarrier
// ------------------------------------------------------

#[derive(Debug)]
pub struct CXLBarrier {
    base_cl: *mut Cacheline,

    cached_idx: u16, // The cacheline index used by this host

    generation: AtomicU64,

    pub options: CXLBarrierOptions,
}

// Send + Sync is fine -> state is saved in CXL shared memory

unsafe impl Send for CXLBarrier {}
unsafe impl Sync for CXLBarrier {}

impl CXLBarrier {
    #[inline]
    fn start_cl(&self) -> *mut Cacheline {
        unsafe { self.base_cl.add(1) }
    }

    /// Creates a new CXLBarrier allocating memory from `alloc`.
    pub fn new(alloc: Arc<impl Allocator>, options: CXLBarrierOptions) -> Self {
        debug!("CXLBarrier::init()");

        let layout = Layout::from_size_align(options.required_bytes(), 64).unwrap();
        let mem = alloc.allocate(layout).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        CXLBarrier::init(base_cl, options)
    }

    /// Treats `base_cl` as a `CXLBarrier` for MAX_HOSTS participants.
    pub fn init(base_cl: *mut Cacheline, options: CXLBarrierOptions) -> Self {
        let id = get_machine_id() - 1;
        debug!(
            "[{:#}] CXLBarrier::init(base_cl={base_cl:p}) -> id:{id}",
            Cacheline::ptr_to_offset(base_cl)
        );

        let p = CXLBarrier {
            base_cl,
            cached_idx: id,
            generation: AtomicU64::new(0),
            options,
        };
        // CL[0]
        cxl_typed_new_sync::<Self>(base_cl);

        // CL[1:MAX_HOSTS]
        let hosts_cl = p.start_cl();
        for n in 0..MAX_HOSTS {
            cl_set_zero_sync(unsafe { hosts_cl.add(n as usize) });
        }
        p
    }

    /// Tries to create a CXLBarrier from the given `base_cl`.
    pub fn from_cl(base_cl: *mut Cacheline, options: CXLBarrierOptions) -> Option<Self> {
        let id = get_machine_id() - 1;
        debug!(
            "[{:#}] CXLBarrier::from_cl(base_cl={base_cl:p}) -> id:{id}",
            Cacheline::ptr_to_offset(base_cl)
        );

        cxl_typed_from_cl_sync::<Self>(base_cl, |_m| CXLBarrier {
            base_cl,
            cached_idx: id,
            generation: AtomicU64::new(0),
            options,
        })
    }

    /// Blocks until all hosts registered within this `context` have entered this barrier.
    pub fn wait_for_all(&self) {
        self.wait_for(self.options.registered_hosts as u16);
    }

    /// Blocks until `nhosts` have entered this barrier.
    pub fn wait_for(&self, nhosts: u16) {
        let g = self.generation.fetch_add(1, Ordering::Relaxed) + 1; // This breaks when
        // nhost!=hosts in the system!
        if self.cached_idx == 0 {
            // I'm the coordinator
            if nhosts == 1 {
                // We are the only one in this barrier!
                return;
            }
            debug!(
                "[{:#}] CXLBarrier::waitfor(nhosts={nhosts}) -> Coordinator idx:{}",
                self.cached_idx,
                Cacheline::ptr_to_offset(self.base_cl)
            );
            let mut spins = SpinCounter::new();
            /*
            loop {
                // wait on all others
                let mut cnt = 0;
                // Machine IDs are assigned densely in `0..nhosts`. Ignore
                // inactive slots: a reused CXL window may still contain a
                // previous generation there, and counting it can make `cnt`
                // exceed `nhosts - 1` forever.
                for idx in 1..nhosts as usize {
                    if cxl_read_cacheline_sync(unsafe { self.start_cl().add(idx) }).d[0] == g {
                        cnt += 1;
                    }
                }
                if cnt == (nhosts - 1) as u64 {
                    break;
                }
                spins.wait();
            }
            */
            let mut waitbits = 0u64;
            for idx in 1..nhosts as usize {
                waitbits |= 1 << idx;
            }

            loop {
                for idx in 1..nhosts as usize {
                    if ((waitbits & (1 << idx)) > 0)
                        && (cxl_read_cacheline_sync(unsafe { self.start_cl().add(idx) }).d[0] == g)
                    {
                        waitbits &= !(1 << idx);
                    }
                }
                if waitbits == 0 {
                    break;
                }
                spins.wait();
            }

            cxl_write_sync!((self.start_cl(), 0, g)); // release all waiting
        } else {
            debug!(
                "[{:#}] CXLBarrier::waitfor(nhosts={nhosts}) -> idx:{}",
                self.cached_idx,
                Cacheline::ptr_to_offset(self.base_cl)
            );
            // I'm another node
            let my_barrier_cl = unsafe { self.start_cl().add(self.cached_idx.into()) };
            cxl_write_sync!((my_barrier_cl, 0, g)); // I'm here
            let mut spins = SpinCounter::new();
            while cxl_read_cacheline_sync(self.start_cl()).d[0] < g {
                spins.wait();
            } //wait on coordinator
        }
    }
}

impl CXLSharedState for CXLBarrier {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        self.options.required_cachelines()
    }
    fn get_cxl_type_id() -> u64 {
        0xb4331e3
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::allocator::os::OSAlloc;
    use crate::core::{
        host::{PeerMachineID, set_machine_cnt, set_machine_id},
        set_global_offset,
    };
    use log::info;
    use std::sync::Arc;

    #[test]
    fn base() {
        // This test does not verify correct cross host behaviour!

        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        set_machine_id(1);
        set_machine_cnt(MAX_HOSTS as u16);

        let options = CXLBarrierOptions::from_hosts(MAX_HOSTS);
        let barrier = CXLBarrier::new(Arc::new(os_alloc), options);
        let base_cl = barrier.base_cl as u64;
        set_global_barrier(barrier);

        let mut join_handles = vec![];
        for i in 1..MAX_HOSTS {
            join_handles.push(std::thread::spawn(move || {
                set_machine_id(i as PeerMachineID + 1);
                let b = CXLBarrier::from_cl(base_cl as *mut Cacheline, options).unwrap();
                info!("[{}] Now starting to wait in barrier...", get_machine_id());
                b.wait_for(MAX_HOSTS as u16);
                info!("[{}] Through the barrier", get_machine_id());
            }));
        }

        info!("[{}] Now starting to wait in barrier...", get_machine_id());
        barrier_wait_all();
        info!("[{}] Through the barrier", get_machine_id());

        for i in join_handles {
            i.join().unwrap();
        }

        CXLBarrier::from_cl(base_cl as *mut Cacheline, options).unwrap();
    }
}
