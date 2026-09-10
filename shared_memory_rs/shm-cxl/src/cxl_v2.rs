use crate::allocator::MemOptions;
use crate::allocator::opts::MemManager;
use crate::com::ComOptions;
use crate::com::opts::ComManager;
use crate::core::host::{LEADER_ID, PeerID, set_machine_id};
use crate::core::{Cacheline, ShmOffset, set_global_offset};
use crate::cxl::{CXLSharedState, CXLSharedStateHeader};
use crate::utils::init_n_host_ids;
use log::debug;

// ------------------------------------------------------
// CXLSizedObject trait
// ------------------------------------------------------

/// A trait that allows calcuation of the actual size of a CXL resident data structure
pub trait CXLSizedObject: std::fmt::Debug + Copy + Clone {
    //
    fn required_cachelines(&self) -> usize;

    fn required_bytes(&self) -> usize {
        self.required_cachelines() * 64
    }

    // Type ID
    // During recovery or from_cl this should be checked
    //fn type_id() -> u64;

    // During recover of from_cl this should be validated as well
    //fn checksum()->

    //fn create() calls new or from_cl depending on id
    //fn new(base_cl, options) -> T
    //fn from_cl(base_cl) -> Option<T>
}

// ------------------------------------------------------
// Context for an instance of a CXLSharedMemory context
// ------------------------------------------------------

#[derive(Debug, Copy, Clone)]
pub struct RuntimeContext {
    //wait_timeout:
    pub mem_offset: ShmOffset,
    pub machine_id: PeerID,
    //
    pub recover_magic: u64,
}

impl RuntimeContext {
    pub fn default() -> Self {
        RuntimeContext {
            mem_offset: 0,
            machine_id: PeerID::from_machine_id(LEADER_ID as u16),
            recover_magic: u64::MAX,
        }
    }
    /// Instead of calling init oder from_cl we try to recover our state after a crash
    pub fn with_recovery(magic: u64) -> Self {
        let mut cfg = RuntimeContext::default();
        cfg.recover_magic = magic;
        cfg
    }
    #[inline]
    pub fn cl_to_offset(&self, cl: *mut Cacheline) -> ShmOffset {
        (cl as u64 - self.mem_offset) / 64
    }
    #[inline]
    pub fn offset_to_cl(&self, offset: ShmOffset) -> *mut Cacheline {
        unsafe { (self.mem_offset as *mut Cacheline).add(offset as usize) }
    }

    #[inline]
    pub fn is_leader(&self) -> bool {
        self.machine_id.machine_id == LEADER_ID as u16
    }
}

// ------------------------------------------------------
// SharedMemContxtOptions
// ------------------------------------------------------

/*
 * ENV Vars
 *
 * CXL_NUM_HOSTS = <INT>   How many hosts to expect when running init
 *
 */

// TODO add SerDe here?
#[derive(Debug, Clone, Copy)]
pub struct SharedMemSegmentOptions {
    pub num_hosts: usize,

    pub com: ComOptions,

    pub mem: MemOptions,
}
impl SharedMemSegmentOptions {
    fn hosts_from_env_or_default(self) -> Self {
        let num_hosts = std::env::var("CXL_NUM_HOSTS")
            .unwrap_or("2".to_string())
            .parse::<usize>()
            .unwrap();
        self.with_hosts(num_hosts)
    }
    pub fn default() -> Self {
        let s = SharedMemSegmentOptions {
            com: ComOptions::default(),
            mem: MemOptions::default(),
            num_hosts: 0, // invalid here cuz we need to set using the with_hosts so its set correctly in all places
        };
        s.hosts_from_env_or_default()
    }
    /// Default but without any mem related capabilities
    pub fn default_no_mem() -> Self {
        let s = SharedMemSegmentOptions {
            com: ComOptions::default(),
            mem: MemOptions::with_no_mem(),
            num_hosts: 0, // invalid here cuz we need to set using the with_hosts so its set correctly in all places
        };
        s.hosts_from_env_or_default()
    }
    /// Sets the number of hosts on all relevant option structs
    pub fn with_hosts(mut self, hosts: usize) -> Self {
        self.num_hosts = hosts;
        self.com.ipc_opts.hosts = hosts;
        self
    }
    pub fn with_com_opts(mut self, com_opts: ComOptions) -> Self {
        self.com = com_opts;
        self
    }
    pub fn with_mem_opts(mut self, mem_opts: MemOptions) -> Self {
        self.mem = mem_opts;
        self
    }

    ///
    pub fn checksum() -> u64 {
        0x42
    }
}

impl CXLSizedObject for SharedMemSegmentOptions {
    fn required_cachelines(&self) -> usize {
        self.com.required_cachelines() + self.mem.required_cachelines()
    }
}

// ------------------------------------------------------
// CXLSharedMemContext
// ------------------------------------------------------

///
pub struct CXLSharedMemSegment {
    // Start address of this context
    pub base_cl: *mut Cacheline,

    // The configuration of this context
    pub context: RuntimeContext,
    pub config: SharedMemSegmentOptions,

    pub header: CXLSharedStateHeader,

    //
    pub com: ComManager,
    //
    pub mem: Option<MemManager>,
}

///
pub fn create_shared_memory_segment(
    base_cl: *mut Cacheline,
    size: usize,
    options: SharedMemSegmentOptions,
) -> CXLSharedMemSegment {
    debug!("create_shared_memory_segment(base_cl={base_cl:p}, size={size}, options={options:?})");
    //. Check if structs already present
    //      Check if they match what we expect
    // If not
    //  ->  init_host_ids
    //  -> Host1 -> Create shared state
    //  -> !Host1 -> Wait for init
    //

    // TODO implement recovery path

    let machine_id = if options.num_hosts == 1 {
        set_machine_id(LEADER_ID as u16); // TODO remove once all internal structs use runtime context to get machine_id
        LEADER_ID as u16
    } else {
        init_n_host_ids(base_cl, options.num_hosts as u16, Some(30_000))
    };
    let context = RuntimeContext {
        mem_offset: 0, // TODO
        machine_id: PeerID::from_machine_id(machine_id),
        recover_magic: 0, // TODO
    };
    set_global_offset(base_cl as u64);

    // Next is com followed by mem
    let (header, com, mem) = if context.is_leader() {
        // First Element is the header
        let header_cl = base_cl;
        let header = CXLSharedStateHeader::init(header_cl, size, options.num_hosts as u16);
        let mut next_cl = unsafe { header_cl.add(header.required_cxl_shared_state_cachelines()) };

        // Init com
        let com = ComManager::init(next_cl, options, &context);

        // Init mem
        let mem = if options.mem.enable {
            next_cl = unsafe { next_cl.add(com.required_cxl_shared_state_cachelines()) };
            Some(MemManager::init(next_cl, options))
        } else {
            None
        };

        // Mark as cxl shared state header as correctly initialized
        CXLSharedStateHeader::set_finished(header_cl);

        (header, com, mem)
    } else {
        // this is still race condition prone...
        std::thread::sleep(std::time::Duration::from_millis(50));

        let header = CXLSharedStateHeader::blocking_from_cl_sync(base_cl);
        let mut next_cl = unsafe { base_cl.add(header.required_cxl_shared_state_cachelines()) };

        // Init com
        let com = ComManager::from_cl_sync(next_cl, options).unwrap();

        // Init mem
        let mem = if options.mem.enable {
            next_cl = unsafe { next_cl.add(com.required_cxl_shared_state_cachelines()) };
            Some(MemManager::from_cl_sync(next_cl, options).unwrap())
        } else {
            None
        };

        (header, com, mem)
    };

    let ctx = CXLSharedMemSegment {
        base_cl,
        context,
        config: options,
        header,
        com,
        mem,
    };

    ctx.com.barrier.wait_for_all();

    ctx
}

impl CXLSharedMemSegment {
    /// Thread local ID
    pub fn id(&self) -> u16 {
        self.context.machine_id.machine_id
    }

    /// How many hosts are activly participating here
    pub fn num_hosts(&self) -> usize {
        self.config.num_hosts
    }

    /// Use this to check if the shared state structure has the expected magic value when
    /// trying to recover.
    pub fn recovery_magic(&self) -> u64 {
        //self.header.init_timestamp.
        0x69
    }

    /// Instead of sharing u64 and calling from_cl we could offer this to make cross thread sharing easier?
    pub fn local_clone() {
        // Call from_cl on all active elements?
    }

    /// Returns base_cl + size
    pub fn next_free_cl(&self) -> *mut Cacheline {
        unsafe { self.base_cl.add(self.config.required_cachelines()) }
    }
}

impl CXLSharedState for CXLSharedMemSegment {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        self.header.required_cxl_shared_state_cachelines()
            + self.com.required_cxl_shared_state_cachelines()
        //+ match self.mem {
        //    Some(m) => m.required_cxl_shared_state_cachelines(),
        //    None => 0,
        //}
    }
    fn get_cxl_type_id() -> u64 {
        0x11211
    }
}

#[cfg(test)]
mod test {
    use crate::allocator::os::OSAlloc;
    use crate::core::Cacheline;
    use crate::cxl_v2::{SharedMemSegmentOptions, create_shared_memory_segment};
    use allocator_api2::alloc::Allocator;

    #[test]
    fn base() {
        //
        // NOTE To run this test the from_cl_sync need to not try to set global stuff...
        //
        let _ = env_logger::try_init();

        let mem_size = 0x1_000_000_000;
        let mem_request = std::alloc::Layout::from_size_align(mem_size, 64).unwrap();
        let os_alloc = OSAlloc::create();
        let mem = os_alloc.allocate(mem_request).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        let options = SharedMemSegmentOptions::default().with_hosts(1);
        let cxl_seg = create_shared_memory_segment(base_cl, mem_size, options);

        /*
        let cxl_seg_2 = CXLSharedMemSegment::from_cl_sync(base_cl);

        assert_eq!(cxl_seg.base, cxl_seg_2.base);
        assert_eq!(cxl_seg.cl_alloc, cxl_seg_2.cl_alloc);
        assert_eq!(cxl_seg.cl_barrier, cxl_seg_2.cl_barrier);
        assert_eq!(cxl_seg.cl_ipc, cxl_seg_2.cl_ipc);
        assert_eq!(cxl_seg.cl_named_pipes_base, cxl_seg_2.cl_named_pipes_base);
        */
    }
}
