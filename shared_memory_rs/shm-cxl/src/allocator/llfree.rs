use crate::allocator::common::CXL_ALLOC_MIN_SIZE;
use crate::core::host::{MAX_HOSTS, PeerID};
use crate::core::{Cacheline, cxl_read_cacheline_sync, cxl_write_cl_sync};
use crate::cxl::CXLSharedState;
use crate::mutex::CXLMutexImpl;
use crate::utils::align_down_mut;
use crate::utils::align_up_mut;

use allocator_api2::alloc::Allocator;
use allocator_api2::alloc::Layout;
use llfree::frame::Frame;
use llfree::{
    Alloc, FrameId, Init, LLFree, MetaData, Policy, Request, TREE_FRAMES, Tier, Tiering, TreesLock,
};
use log::{debug, error, warn};
use std::slice;

/// Based on the NvmAlloc which is a Persistent memory allocator, that is able to recover its state from the memory it manages.
/// Adapted to work on CXL shared memory.
pub struct CXLAlloc<'a> {
    header: *mut Cacheline,

    // Allocator shared lock
    cl_mutex: *mut Cacheline,
    pub mutex: Box<dyn TreesLock>,

    // The allocator
    alloc: LLFree<'a>,
    free_space_start: *mut Cacheline, // First cacheline that can be allocated
}

struct CXLLayout {
    header: *mut Cacheline,
    lock: *mut Cacheline,
    tree_start: *mut Cacheline,
    free_space_start: *mut Cacheline,
    pages: usize,
}

impl<'a> CXLAlloc<'a> {
    /// Returns amount of bytes this allocator manages
    pub fn get_owned_memory_size(&self) -> usize {
        self.alloc.frames() * Frame::SIZE
    }

    fn node_policy() -> Tiering {
        let tiers = [
            (Tier(0), 0), // any, not used by anyone
            (Tier(1), 1), // node
            (Tier(2), 1), // node
            (Tier(3), 1), // node
            (Tier(4), 1), // node
        ];

        #[track_caller]
        fn policy(requested: Tier, target: Tier, free: usize) -> Policy {
            warn!(
                "Policy: {requested:?} -> {target:?} {free} at {:?}",
                core::panic::Location::caller()
            );
            if target.0 == 0 {
                return Policy::Demote;
            }
            if requested == target {
                return match free {
                    f if f >= TREE_FRAMES / 2 => Policy::Match(1), // half free
                    f if f >= TREE_FRAMES / 64 => Policy::Match(u8::MAX), // almost allocated
                    _ => Policy::Match(0), // low free count -> causes frequent reservations
                };
            }
            Policy::Invalid
        }

        Tiering::new(&tiers, Tier(0), policy)
    }

    fn calc_layout(base: *mut Cacheline, cxl_mem_size: usize) -> CXLLayout {
        assert!(cxl_mem_size >= CXL_ALLOC_MIN_SIZE);
        let end = unsafe { base.add(cxl_mem_size / size_of::<Cacheline>()) };
        // Align down to CXL_ALLOC_MIN_SIZE
        let end = align_down_mut(end, CXL_ALLOC_MIN_SIZE);
        assert!((end as usize).is_multiple_of(CXL_ALLOC_MIN_SIZE));

        // Header for metadata
        let mut next_cl = unsafe { base.add(1) };

        // Lock
        let cl_tpl = next_cl;
        let lock_cachelines = 32; //;CXLMutexImpl::calc_size(MAX_HOSTS);
        next_cl = unsafe { next_cl.add(lock_cachelines) };

        // Cast all required pointers for metadata management
        // We manage multiples of CXL_ALLOC_MIN_SIZE so we skip all cachelines that dont fit
        let next_cl = align_up_mut(next_cl, CXL_ALLOC_MIN_SIZE);
        assert!((next_cl as usize).is_multiple_of(CXL_ALLOC_MIN_SIZE));
        let remaining_size_bytes = unsafe { end.byte_offset_from(next_cl) };
        assert!(remaining_size_bytes >= 0);
        let pages = remaining_size_bytes as usize / Frame::SIZE;

        debug!("max pages: {pages}");
        let tiering = CXLAlloc::node_policy();
        let meta = LLFree::metadata_size(&tiering, pages);

        // we need to skip the tree itself for the actual free space
        let trees_start = next_cl;
        let next_cl = unsafe { next_cl.add(meta.trees.div_ceil(size_of::<Cacheline>())) };
        let next_cl = align_up_mut(next_cl, CXL_ALLOC_MIN_SIZE);
        let remaining_size_bytes = unsafe { end.byte_offset_from(next_cl) };
        assert!(remaining_size_bytes >= 0);
        let pages = remaining_size_bytes as usize / Frame::SIZE;

        debug!("actual pages: {pages}");

        CXLLayout {
            header: base,
            lock: cl_tpl,
            tree_start: trees_start,
            free_space_start: next_cl,
            pages,
        }
    }

    pub fn from_cl(base: *mut Cacheline, os_alloc: impl Allocator) -> Self {
        let header = cxl_read_cacheline_sync(base);
        let cxl_mem_size = header.d[0] as usize;
        let CXLLayout {
            header,
            lock,
            tree_start,
            free_space_start,
            pages,
        } = Self::calc_layout(base, cxl_mem_size);

        let tpl = Box::new(CXLMutexImpl::from_cl_sync(lock).unwrap());

        let tiering = CXLAlloc::node_policy();
        let meta = LLFree::metadata_size(&tiering, pages);
        let trees = unsafe { std::slice::from_raw_parts_mut(tree_start as *mut u8, meta.trees) };
        let lower = unsafe {
            os_alloc
                .allocate(Layout::from_size_align(meta.lower, 64).unwrap())
                .unwrap()
                .as_mut()
        };
        let local = unsafe {
            os_alloc
                .allocate(Layout::from_size_align(meta.local, 64).unwrap())
                .unwrap()
                .as_mut()
        };
        let metadata = MetaData {
            local,
            trees,
            lower,
        };

        // ............ FIX (later)
        let cxl_global_lock: Box<dyn TreesLock> = tpl;
        let lock_ptr = cxl_global_lock.as_ref() as &dyn TreesLock as *const dyn TreesLock;
        let alloc = LLFree::new(
            pages,
            Init::VerifyTrees,
            &tiering,
            unsafe { &*lock_ptr },
            metadata,
        )
        .unwrap();
        Self {
            header,
            alloc,
            cl_mutex: lock,
            mutex: cxl_global_lock,
            free_space_start,
        }
    }

    /// `base` is the base addresss within the CXL shared memory where we place our structures.
    /// `cxl_mem_size` size of the complete CXL shared memory segment that this allocator can use.
    /// `cxl_global_lock` is used to coordinate concurrent shared state updates
    /// `os_alloc`
    pub fn init(base: *mut Cacheline, cxl_mem_size: usize, os_alloc: impl Allocator) -> Self {
        let CXLLayout {
            header,
            lock,
            tree_start,
            free_space_start,
            pages,
        } = Self::calc_layout(base, cxl_mem_size);

        // Header for metadata
        let mut init_data = Cacheline::empty();
        init_data.d[0] = cxl_mem_size as u64;

        // Lock
        let tpl = Box::new(CXLMutexImpl::init(lock, MAX_HOSTS));

        let tiering = CXLAlloc::node_policy();
        let meta = LLFree::metadata_size(&tiering, pages);
        init_data.d[1] = meta.trees as u64;
        let trees = unsafe { std::slice::from_raw_parts_mut(tree_start as *mut u8, meta.trees) };
        let lower = unsafe {
            os_alloc
                .allocate(Layout::from_size_align(meta.lower, 64).unwrap())
                .unwrap()
                .as_mut()
        };
        let local = unsafe {
            os_alloc
                .allocate(Layout::from_size_align(meta.local, 64).unwrap())
                .unwrap()
                .as_mut()
        };
        let metadata = MetaData {
            local,
            trees,
            lower,
        };

        // ............ FIX (later)
        let cxl_global_lock: Box<dyn TreesLock> = tpl;
        let lock_ptr = cxl_global_lock.as_ref() as &dyn TreesLock as *const dyn TreesLock;
        let alloc = LLFree::new(
            pages,
            Init::FreeAll,
            &tiering,
            unsafe { &*lock_ptr },
            metadata,
        )
        .unwrap();
        debug!("{alloc:?}");

        // Write header
        cxl_write_cl_sync(header, &init_data);

        Self {
            header,
            alloc,
            cl_mutex: lock,
            mutex: cxl_global_lock,
            free_space_start,
        }
    }
}

impl CXLSharedState for CXLAlloc<'_> {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(n: usize) -> usize {
        //(self.free_space_start as usize - self.header as usize) / 64
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        unimplemented!()
    }
    fn get_cxl_type_id() -> u64 {
        0x876543210
    }
}

unsafe impl<'a> Allocator for CXLAlloc<'a> {
    fn allocate(
        &self,
        layout: Layout,
    ) -> std::result::Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        let size_align = layout.size().max(layout.align());
        let pages = size_align.div_ceil(Frame::SIZE);
        // 2^oder == pages
        let order = pages.ilog2() as usize;

        let node = PeerID::whoami().machine_id;

        let request = Request::new(order, Tier(node as _), Some(0)); // was Some(0) -> Fails
        debug!("get {request:?}");

        let (page, tier) = self
            .alloc
            .get(None, request)
            .map_err(|_| allocator_api2::alloc::AllocError)?;

        assert!(tier == Tier(node as _) || tier == Tier(4));
        if tier == Tier(4) {
            // TODO: mmap!
        }

        let pointer = unsafe { self.free_space_start.cast::<u8>().add(page.0 * Frame::SIZE) };
        assert!(!pointer.is_null());

        unsafe {
            Ok(std::ptr::NonNull::new_unchecked(slice::from_raw_parts_mut(
                pointer,
                layout.size(),
            )))
        }
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        let byte_offset = unsafe { ptr.as_ptr().offset_from(self.free_space_start.cast()) };
        if byte_offset < 0 {
            error!("Out of range");
            return;
        }
        let frame = byte_offset as usize / Frame::SIZE;

        let size_align = layout.size().max(layout.align());
        let pages = size_align.div_ceil(Frame::SIZE);
        // 2^oder == pages
        let order = pages.ilog2() as usize;
        let node = PeerID::whoami().machine_id;

        let request = Request::new(order, Tier(node as _), Some(0));
        debug!("put {request:?}");

        if let Err(e) = self.alloc.put(FrameId(frame), request) {
            error!("Free failed {e:?}");
        }
    }
}

//##########################
//  Tests
//#########################

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::memory_management::{ExternalMemProvider, Flags, MemoryManager};
    use crate::allocator::os::OSAlloc;
    use crate::utils::{DEFAULT_DEVDAX_MEM_SIZE, DEFAULT_DEVDAX_PATH, open_cxl_daxdev};
    use allocator_api2::alloc::{Allocator, Layout};
    use llfree::{Alloc, TREE_FRAMES, frame::Frame};
    use log::{debug, error, info};
    use std::os::fd::IntoRawFd;
    use std::ptr::NonNull;

    #[test]
    fn base() {
        let (cxl_dev, cxl_mem_size) = crate::utils::open_test_cxl_dev();

        let os_alloc = OSAlloc::create();
        let cxl_alloc = CXLAlloc::init(cxl_dev, cxl_mem_size, os_alloc);
        info!("CXLAlloc::create");

        assert_eq!(
            cxl_alloc.alloc.frames(),
            cxl_alloc.alloc.stats().free_frames
        );

        let alloc_1_layout = Layout::from_size_align(8, 1).unwrap();
        let alloc_res = cxl_alloc.allocate(alloc_1_layout).unwrap();
        info!("cxl_alloc.allocate(1)");

        let ptr = NonNull::new(alloc_res.as_ptr() as *mut u8).unwrap();
        unsafe { cxl_alloc.deallocate(ptr, alloc_1_layout) };
        info!("cxl_alloc.deallocate(1)");

        assert_eq!(
            cxl_alloc.alloc.frames(),
            cxl_alloc.alloc.stats().free_frames
        );

        let alloc_2_layout = Layout::from_size_align(8, 2 << 20).unwrap();
        let alloc_res2 = cxl_alloc.allocate(alloc_2_layout).unwrap();
        info!("cxl_alloc.allocate(2)");

        let ptr2 = NonNull::new(alloc_res2.as_ptr() as *mut u8).unwrap();
        unsafe { cxl_alloc.deallocate(ptr2, alloc_2_layout) };
        info!("cxl_alloc.deallocate(2)");

        assert_eq!(
            cxl_alloc.alloc.frames(),
            cxl_alloc.alloc.stats().free_frames
        );

        let mut allocs = Vec::new();
        let byte_size = TREE_FRAMES * Frame::SIZE;
        let layout = Layout::from_size_align(byte_size, byte_size).unwrap();
        for _ in 0..cxl_alloc.alloc.frames() / TREE_FRAMES {
            match cxl_alloc.allocate(layout) {
                Ok(ptr) => allocs.push(ptr.as_ptr() as *mut u8 as usize),
                Err(e) => error!("{e:?} -> {:?}", cxl_alloc.alloc),
            }
        }
        allocs.sort();
        // Unique check
        assert!(allocs.array_windows::<2>().all(|[a, b]| a != b));

        for alloc in allocs {
            unsafe { cxl_alloc.deallocate(NonNull::new(alloc as *mut u8).unwrap(), layout) };
        }

        assert_eq!(
            cxl_alloc.alloc.frames(),
            cxl_alloc.alloc.stats().free_frames
        );
    }

    /// Tests the API integration of the OSAlloc.
    #[test]
    fn api() {
        let fd = open_cxl_daxdev(DEFAULT_DEVDAX_PATH).unwrap();

        let mem_prov = vec![ExternalMemProvider::cxl(
            fd.into_raw_fd(),
            DEFAULT_DEVDAX_MEM_SIZE,
        )];
        let mm = MemoryManager::mm_init(mem_prov).unwrap();
        info!("> MemoryManager::mm_init");

        let mem_obj = mm
            .mm_malloc(Layout::from_size_align(8, 1).unwrap(), Flags::LOCATION_CXL)
            .unwrap();
        info!("> os_alloc.mm_malloc");

        debug!("{:?}", mem_obj);

        assert!(mem_obj.owned_by_self());
    }
}
