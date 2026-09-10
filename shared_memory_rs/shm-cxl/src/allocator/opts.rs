use crate::allocator::opts::{CXLAllocVariant::BitMap, CXLAllocVariant::None};
use crate::allocator::{CXL_ALLOC_MIN_SIZE, CXLBitMapAlloc};
use crate::core::Cacheline;
use crate::cxl::{CXLSharedState, cxl_new_sized_obj, cxl_typed_get_size_cl};
use crate::cxl_v2::{CXLSizedObject, SharedMemSegmentOptions};

// ------------------------------------------------------
// MemOptions
// ------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct MemOptions {
    pub enable: bool,
    size_cl: usize, // Indicates how many cachelines are under control of the MemoryManager
    alloc: CXLAllocVariant,
}
impl MemOptions {
    pub fn default() -> Self {
        MemOptions {
            enable: true,
            size_cl: CXL_ALLOC_MIN_SIZE,
            alloc: BitMap,
        }
    }
    pub fn with_no_mem() -> Self {
        MemOptions {
            enable: false,
            size_cl: 0,
            alloc: None,
        }
    }
    pub fn with_allocator(mut self, alloc: CXLAllocVariant) -> Self {
        self.alloc = alloc;
        self
    }
}

impl CXLSizedObject for MemOptions {
    fn required_cachelines(&self) -> usize {
        self.size_cl
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CXLAllocVariant {
    None,
    BitMap,
    LLfree,
}

// ------------------------------------------------------
// MemManager
// ------------------------------------------------------

pub struct MemManager {
    //
    base_cl: *mut Cacheline,

    pub alloc: CXLBitMapAlloc,

    pub options: SharedMemSegmentOptions,
}

impl MemManager {
    pub fn init(base_cl: *mut Cacheline, options: SharedMemSegmentOptions) -> Self {
        let next_cl = base_cl;
        let remaining_space = options.mem.size_cl - (next_cl as usize - base_cl as usize);

        let cl_alloc = next_cl;
        let alloc = CXLBitMapAlloc::try_new(
            cl_alloc,
            CXL_ALLOC_MIN_SIZE,
            remaining_space,
            options.num_hosts,
        )
        .unwrap();

        cxl_new_sized_obj::<MemManager>(base_cl, remaining_space as u64);

        MemManager {
            base_cl,
            options,
            alloc,
        }
    }

    pub fn from_cl_sync(base_cl: *mut Cacheline, options: SharedMemSegmentOptions) -> Option<Self> {
        let alloc = CXLBitMapAlloc::from_cl(base_cl).unwrap();

        Some(MemManager {
            base_cl,
            options,
            alloc,
        })
    }

    // Future API

    pub fn mem_reserve() {}
    pub fn mem_commit() {}
    // OR
    pub fn mem_alloc() {}

    pub fn mem_copy() {}

    pub fn mem_set_owner() {}

    // Built from the previous API calls
    pub fn mem_move() {
        // copy
        // set_owner
    }

    pub fn mem_clone() {}
}

impl CXLSharedState for MemManager {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        cxl_typed_get_size_cl::<MemManager>(self.base_cl) as usize
    }
    fn get_cxl_type_id() -> u64 {
        0x1337
    }
}
