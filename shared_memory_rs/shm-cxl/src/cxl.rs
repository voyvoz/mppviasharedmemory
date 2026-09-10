use crate::allocator::CXL_ALLOC_MIN_SIZE;
use crate::allocator::bitmap::CXLBitMapAlloc;
use crate::com::barrier::{CXLBarrier, barrier_wait_all, set_global_barrier};
use crate::com::pipe::NamedPipeOptions;
use crate::com::{ipc::CXLIpc, pipe::NamedPipes};
use crate::core::core::set_global_offset;
use crate::core::host::{LEADER_ID, MACHINE_ID_INVALID, MAX_HOSTS, PeerMachineID, get_machine_id};
use crate::core::{Cacheline, cxl_read_cacheline_sync};
use crate::cxl_v2::SharedMemSegmentOptions;
use crate::cxl_write_sync;
use crate::utils::timestamps::SpinCounter;
use log::debug;
use std::mem::transmute;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ------------------------------------------------------
// CXL Shared Memory Data Structures
// ------------------------------------------------------

/// A trait that enables CXL functionality for shared data structures
/// Implementing it allows using cxl_typed_* functions to have
/// ref counter behaviour.
pub trait CXLSharedState {
    /// Returns amount of Cacheline are required for the static version
    fn required_cxl_shared_state_cachelines_static() -> usize;
    /// Returns how many Cachelines this structure requires when used with dynamic size `n`
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize;

    /// Returns how many Cachelines this structure currently requires.
    fn required_cxl_shared_state_cachelines(&self) -> usize;

    /// A unique ID to identify the type
    fn get_cxl_type_id() -> u64;
}

// ------------------------------------------------------
// CXL Shared Memory Data Structures
// ------------------------------------------------------

const TYPE_ID_IDX: usize = 0;
const REF_COUNTER_IDX: usize = 1;
const OBJ_SIZE_IDX: usize = 2;

pub const CUSTOM_FIELD_IDX_START: usize = 3;
pub const CUSTOM_FIELD_IDX_END: usize = 7;

/// Creates the allocation metadata for a newly created data structure.
/// Requires external locking.
pub fn cxl_typed_new_sync<S: CXLSharedState>(base_cl: *mut Cacheline) {
    cxl_write_sync!((base_cl => [
        (TYPE_ID_IDX, S::get_cxl_type_id()),
        (REF_COUNTER_IDX, 1)
    ]));
}

pub fn cxl_new_sized_obj<S: CXLSharedState>(base_cl: *mut Cacheline, obj_size: u64) {
    cxl_write_sync!((base_cl => [
        (TYPE_ID_IDX, S::get_cxl_type_id()),
        (REF_COUNTER_IDX, 1),
        (OBJ_SIZE_IDX, obj_size)
    ]));
}

///Increments the reference counter resident in shared memory.
/// `build` allows using the unused fields in the `base_cl`.
/// Requires external locking.
pub fn cxl_typed_from_cl_sync<S: CXLSharedState>(
    base_cl: *mut Cacheline,
    build: impl FnOnce(Cacheline) -> S,
) -> Option<S> {
    let meta = cxl_read_cacheline_sync(base_cl);
    if meta.d[TYPE_ID_IDX] != S::get_cxl_type_id() {
        log::error!(
            "cxl type mismatch: expected {:x}, found {:x}",
            S::get_cxl_type_id(),
            meta.d[TYPE_ID_IDX]
        );
        return None;
    }
    cxl_write_sync!((base_cl, REF_COUNTER_IDX, meta.d[REF_COUNTER_IDX] + 1));
    Some(build(meta))
}

///A ref counter implementation
pub fn cxl_typed_drop_guard<S: CXLSharedState>(base_cl: *mut Cacheline) -> bool {
    let meta = cxl_read_cacheline_sync(base_cl);
    let updated_ref_counter = meta.d[REF_COUNTER_IDX];
    if updated_ref_counter == 0 {
        true
    } else {
        let updated_ref_counter = meta.d[REF_COUNTER_IDX] - 1;
        cxl_write_sync!((base_cl, REF_COUNTER_IDX, updated_ref_counter));
        if updated_ref_counter > 1 { false } else { true }
    }
}

/// How many cachelines this object requires at runtime
pub fn cxl_typed_get_size_cl<S: CXLSharedState>(base_cl: *mut Cacheline) -> u64 {
    cxl_read_cacheline_sync(base_cl).d[OBJ_SIZE_IDX]
}

/// Allows interaction with the shared state stored in CXL shared memory
pub struct CXLSharedObjectState {
    base_cl: *mut Cacheline,
}
impl CXLSharedObjectState {
    pub fn new_sync(base_cl: *mut Cacheline, cxl_type_id: u64) -> Self {
        cxl_write_sync!((base_cl => [
            (TYPE_ID_IDX, cxl_type_id),
            (REF_COUNTER_IDX, 1)
        ]));
        CXLSharedObjectState { base_cl }
    }
    pub fn from_cl_sync(base_cl: *mut Cacheline, cxl_type_id: u64) -> Option<Self> {
        let meta = cxl_read_cacheline_sync(base_cl);
        if meta.d[TYPE_ID_IDX] != cxl_type_id {
            log::error!(
                "cxl type mismatch: expected {:x}, found {:x}",
                cxl_type_id,
                meta.d[TYPE_ID_IDX]
            );
            return None;
        }
        cxl_write_sync!((base_cl, REF_COUNTER_IDX, meta.d[REF_COUNTER_IDX] + 1));
        Some(CXLSharedObjectState { base_cl })
    }
}

// ------------------------------------------------------
// CXL Shared State Header
// ------------------------------------------------------

/// Header to describe persistent allocation state stored within CXL shared memory.
/// Using this struct will yield a copy which represents the state during intial read.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct CXLSharedStateHeader {
    /// A magic number used to check if the persistent memory contains the allocator state
    magic: u64,
    /// could be the id of the machine that currently tries to init or did the init?
    random: u64,
    /// Marker for successfull init
    magic_valid: u64,
    /// The size used when this structure was initliazed
    pub init_param_size: u64,
    ///Number of hosts used when this structire was created
    pub num_hosts: u64,
    /// The unix timestamp when this structure was created
    pub init_timestamp: SystemTime,
}

impl CXLSharedStateHeader {
    /// Magic marking the meta frame.
    const MAGIC: u64 = 0x_dead_beef;
    const MAGIC_2: u64 = 0xCAFE_BABE;

    /// Check if the Header values are valid
    pub fn is_valid(&self) -> bool {
        self.magic == CXLSharedStateHeader::MAGIC
    }
    /// Check if the init has been finished
    pub fn is_init_finished(&self) -> bool {
        self.magic_valid == CXLSharedStateHeader::MAGIC_2
    }
    pub fn set_finished(base_cl: *mut Cacheline) {
        cxl_write_sync!((base_cl, 2, CXLSharedStateHeader::MAGIC_2));
    }

    /// Transmute Cacheline into the CXLSharedStateHeader
    /// Blocks until init has been finished.
    pub fn blocking_from_cl_sync(base_cl: *mut Cacheline) -> Self {
        let mut spins = SpinCounter::new();
        loop {
            let cl = cxl_read_cacheline_sync(base_cl);
            let header = unsafe { transmute::<Cacheline, CXLSharedStateHeader>(cl) };
            if header.is_init_finished() {
                return header;
            }
            spins.wait();
        }
    }

    /// Treats `base_cl` as the base addr where CXL shared memory structures are located
    pub fn init(
        base_cl: *mut Cacheline,
        shared_memory_size: usize,
        num_hosts: PeerMachineID,
    ) -> CXLSharedStateHeader {
        assert_ne!(num_hosts, MACHINE_ID_INVALID);

        // TODO this needs to be way more sophisitcated!
        // When we read the line and determine it to be invalid we gotta make sure that its still that before we write?
        // Or we write and put something random in there, then read again -> still the same random thing then we can be
        // sure that we won the init?

        // Reset shared state to known inital state
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

        cxl_write_sync!(
            (base_cl => [
                (0, CXLSharedStateHeader::MAGIC),
                (1, LEADER_ID), // ID of the machine that runs the init.
                (2, 0), // Magic2
                (3, shared_memory_size),
                (4, num_hosts),
                (5, t.as_secs()), // tv_sec
                (6, t.subsec_nanos()), // tv_nsec
                (7, 7) // TODO use this as a version information?
            ])
        );

        let cl = cxl_read_cacheline_sync(base_cl);
        let header = unsafe { transmute::<Cacheline, CXLSharedStateHeader>(cl) };
        match header.is_valid() {
            false => panic!("CXL Header::load: Failed validity check"),
            true => header,
        }
    }
}

impl CXLSharedState for CXLSharedStateHeader {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        1
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        CXLSharedStateHeader::required_cxl_shared_state_cachelines_static()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        CXLSharedStateHeader::required_cxl_shared_state_cachelines_static()
    }
    fn get_cxl_type_id() -> u64 {
        CXLSharedStateHeader::MAGIC
    }
}

///
/// PersistantDataStructs stored in CXL shared memory.
///
/// Layout:
///
/// |-------------------------------
/// | [0] Header
/// |-------------------------------
/// | [1] Barrier
/// |-------------------------------
/// | [m] IPC
/// |-------------------------------
/// | [p] lock + Named Pipes
/// |-------------------------------
/// | [l] lock + Alloc
/// |-------------------------------
/// | < Free Space > (Managed by Alloc)
/// |-------------------------------
///
#[derive(Debug)]
pub struct PersistantDataStructs {
    // Header
    base: *mut Cacheline,
    pub header: CXLSharedStateHeader, // A snapshot of the header when this structure was created.

    // Barrier
    cl_barrier: *mut Cacheline,

    // IPC
    cl_ipc: *mut Cacheline,
    pub ipc: CXLIpc,

    // Named pipes
    cl_named_pipes_base: *mut Cacheline,
    pub named_pipes: NamedPipes,

    // Allocator
    // Tree for big allocations
    // Owns all remaining memory
    cl_alloc: *mut Cacheline,
    pub alloc: Arc<CXLBitMapAlloc>,

    // To ease the use of options for now until we switch to v2
    pub options: SharedMemSegmentOptions,
}
impl PersistantDataStructs {
    pub fn print_state(&self) {
        debug!("<------------------------------------------->");
        debug!("[{:p}] CXL Shared Memory Content", self.base);

        debug!("---------------------------------------------");
        debug!("MetaData Header");
        debug!(
            "[{:#x?}] -> {:#x?}",
            self.base as usize - self.base as usize,
            self.header
        );

        debug!("---------------------------------------------");
        debug!("IPC");
        debug!(
            "[{:#x?}] -> {:?}",
            Cacheline::ptr_to_offset(self.cl_ipc),
            self.ipc.required_cxl_shared_state_cachelines()
        );

        debug!("---------------------------------------------");
        debug!("Alloc");
        debug!(
            "[{:#x?}] -> {:?}",
            Cacheline::ptr_to_offset(self.cl_alloc),
            self.alloc.required_cxl_shared_state_cachelines()
        );
        debug!(
            "Managed mem size: [{:#x?}]",
            self.alloc.get_owned_memory_size()
        );

        debug!("<------------------------------------------->");
    }

    ///
    pub fn init(base_cl: *mut Cacheline, cxl_shm_size: usize) -> PersistantDataStructs {
        debug!("PersistantDataStructs::init(base_cl={base_cl:p}, cxl_shm_size={cxl_shm_size})");

        set_global_offset(base_cl as u64);

        let options = SharedMemSegmentOptions::default().with_hosts(MAX_HOSTS);

        // First Element is the header
        let header_cl = base_cl;
        let header = CXLSharedStateHeader::init(header_cl, cxl_shm_size, options.num_hosts as u16);
        let mut next_cl = unsafe {
            header_cl.add(CXLSharedStateHeader::required_cxl_shared_state_cachelines_static())
        };

        // Barrier state structure
        let cl_barrier = next_cl;
        let barrier = CXLBarrier::init(cl_barrier, options.com.barrier_opts);
        next_cl = unsafe { next_cl.add(barrier.required_cxl_shared_state_cachelines()) };
        set_global_barrier(barrier);
        debug!(
            "PersistantDataStructs::init()   Barrier: [{}] <-> [{}]",
            Cacheline::ptr_to_offset(cl_barrier),
            Cacheline::ptr_to_offset(next_cl)
        );

        // IPC state structure
        let cl_ipc = next_cl;
        let ipc = CXLIpc::init(cl_ipc, options.com.ipc_opts);
        next_cl = unsafe { next_cl.add(ipc.required_cxl_shared_state_cachelines()) };
        debug!(
            "PersistantDataStructs::init()   IPC: [{}] <-> [{}]",
            Cacheline::ptr_to_offset(cl_ipc),
            Cacheline::ptr_to_offset(next_cl)
        );

        // Named pipes stuff
        let cl_named_pipes_base = next_cl;
        let np_config = NamedPipeOptions::default();
        let named_pipes = NamedPipes::init(cl_named_pipes_base, np_config);
        next_cl = unsafe { next_cl.add(named_pipes.required_cxl_shared_state_cachelines()) };
        debug!(
            "PersistantDataStructs::init()   NamedPipes: [{}] <-> [{}]",
            Cacheline::ptr_to_offset(cl_named_pipes_base),
            Cacheline::ptr_to_offset(next_cl)
        );

        // Last is the Allocator
        //next_cl = unsafe{next_cl.add(next_cl.align_offset(CXL_ALLOC_MIN_SIZE))};

        let remaining_space = cxl_shm_size - (next_cl as usize - base_cl as usize);
        //println!("ALLOC_MIN_SIZE {CXL_ALLOC_MIN_SIZE}");
        //println!("before remaining_space {remaining_space}");
        //remaining_space -= remaining_space % CXL_ALLOC_MIN_SIZE;
        //println!(" after remaining_space {remaining_space}");

        let cl_alloc = next_cl;
        //let alloc = CXLAlloc::init(cl_alloc, remaining_space, os_alloc);
        let alloc = CXLBitMapAlloc::try_new(
            cl_alloc,
            CXL_ALLOC_MIN_SIZE,
            remaining_space,
            header.num_hosts as usize,
        )
        .unwrap();

        // Mark as cxl shared state header as correctly initialized
        CXLSharedStateHeader::set_finished(header_cl);

        PersistantDataStructs {
            base: base_cl,
            header,
            cl_barrier,
            cl_ipc,
            ipc,
            named_pipes,
            cl_named_pipes_base,
            alloc: Arc::new(alloc),
            cl_alloc,
            options,
        }
    }

    /// Treats `base_cl` as a pointer to a PersistantDataStructs.
    /// Blocks until header is marked as correctly init.
    pub fn from_cl_sync(base_cl: *mut Cacheline) -> PersistantDataStructs {
        debug!("PersistantDataStructs::from_cl_sync(base_cl={base_cl:p})");

        let options = SharedMemSegmentOptions::default().with_hosts(MAX_HOSTS);

        //#[cfg(not(test))]
        set_global_offset(base_cl as u64);

        let header = CXLSharedStateHeader::blocking_from_cl_sync(base_cl);
        let mut next_cl = unsafe { base_cl.add(header.required_cxl_shared_state_cachelines()) };

        // Barrier state structure
        let cl_barrier = next_cl;
        let barrier = CXLBarrier::from_cl(cl_barrier, options.com.barrier_opts).unwrap();
        next_cl = unsafe { next_cl.add(barrier.required_cxl_shared_state_cachelines()) };
        //#[cfg(not(test))]
        set_global_barrier(barrier);
        debug!("PersistantDataStructs::from_cl_sync()   Barrier: {cl_barrier:p} <-> {next_cl:p}");

        // IPC state
        let cl_ipc = next_cl;
        let ipc = CXLIpc::from_cl_sync(cl_ipc, options.com.ipc_opts).unwrap();
        next_cl = unsafe { next_cl.add(ipc.required_cxl_shared_state_cachelines()) };
        debug!("PersistantDataStructs::from_cl_sync()   IPC: {cl_ipc:p} <-> {next_cl:p}");

        // Named pipes stuff
        let cl_named_pipes_base = next_cl;
        let np_config = NamedPipeOptions::default();
        let named_pipes = NamedPipes::from_cl_sync(cl_named_pipes_base, np_config).unwrap();
        next_cl = unsafe { next_cl.add(named_pipes.required_cxl_shared_state_cachelines()) };
        debug!(
            "PersistantDataStructs::from_cl_sync()   NamedPipes: {cl_named_pipes_base:p} <-> {next_cl:p}"
        );

        // alloc state
        let cl_alloc = next_cl;
        let alloc = CXLBitMapAlloc::from_cl(cl_alloc).unwrap();

        PersistantDataStructs {
            base: base_cl,
            header,
            cl_barrier,
            cl_ipc,
            ipc,
            named_pipes,
            cl_named_pipes_base,
            alloc: Arc::new(alloc),
            cl_alloc,
            options,
        }
    }
}

/// Uses /dev/dax0.0 as a CXL shared memory device and create PersistantDataStructs on it.
/// `expected_hosts` determines on how many other machines this function will wait until
/// it returns.
/// To set a fixed offset within the CXL shared memory use environment variable CXL_MAPOFFSET.
/// `open_test_cxl_dev` uses that. Keep in mind that the offset value but be 2MB aligned.
pub fn cxl_init(expected_hosts: u16) -> PersistantDataStructs {
    if expected_hosts < 2 {
        panic!("Must use at least 2 hosts");
    }

    let (dev, size) = crate::utils::open_test_cxl_dev();

    crate::utils::init_n_host_ids(dev, expected_hosts as u16, Some(30_000));

    let persist: PersistantDataStructs;
    let this_machine_id = get_machine_id();
    if this_machine_id == 1 {
        persist = PersistantDataStructs::init(dev, size);
    } else {
        // Peers synchronise on the header's "init finished" marker inside
        // `from_cl_sync`; this stagger only keeps them off the link while the
        // leader does its initialising writes.
        std::thread::sleep(std::time::Duration::from_millis(50));
        persist = PersistantDataStructs::from_cl_sync(dev);
    }

    barrier_wait_all();

    persist
}

#[cfg(test)]
mod test {
    use crate::allocator::os::OSAlloc;
    use crate::core::Cacheline;
    use crate::core::host::set_machine_id;
    use crate::cxl::PersistantDataStructs;
    use allocator_api2::alloc::Allocator;

    #[test]
    fn base() {
        //
        // NOTE To run this test the from_cl_sync need to not try to set global stuff...
        //
        let _ = env_logger::try_init();
        set_machine_id(1);

        let mem_request = std::alloc::Layout::from_size_align(0x1_000_000_000, 64).unwrap();
        let os_alloc = OSAlloc::create();
        let mem = os_alloc.allocate(mem_request).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        let pds = PersistantDataStructs::init(base_cl, mem.len());

        let pds2 = PersistantDataStructs::from_cl_sync(base_cl);

        assert_eq!(pds.base, pds2.base);
        assert_eq!(pds.cl_alloc, pds2.cl_alloc);
        assert_eq!(pds.cl_barrier, pds2.cl_barrier);
        assert_eq!(pds.cl_ipc, pds2.cl_ipc);
        assert_eq!(pds.cl_named_pipes_base, pds2.cl_named_pipes_base);
    }

    #[test]
    fn base_cxl() {
        set_machine_id(1);

        let (dev, size) = crate::utils::open_test_cxl_dev();

        let pds = PersistantDataStructs::init(dev, size);
        println!("{pds:?}");

        let pds2 = PersistantDataStructs::from_cl_sync(dev);
        println!("{pds2:?}");

        assert_eq!(pds.base, pds2.base);
        // TODO assert all equal!
    }
}
