use std::alloc::System;

use allocator_api2::alloc::Allocator;

#[global_allocator]
static A: System = System;

#[derive(Clone, Copy)]
pub struct OSAlloc {
    a: System,
}

impl OSAlloc {
    pub fn create() -> Self {
        OSAlloc { a: System }
    }
}

unsafe impl Allocator for OSAlloc {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        self.a.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        unsafe { self.a.deallocate(ptr, layout) };
    }
}

//##########################
//  Tests
//#########################

#[cfg(test)]
mod tests {
    use crate::allocator::memory_management::{Flags, MemoryManager};
    use crate::allocator::os::OSAlloc;
    use allocator_api2::alloc::{Allocator, Layout};
    use log::{debug, info};

    /// Tests the base functions of the OSAlloc.
    #[test]
    fn base() {
        let os_alloc = OSAlloc::create();
        info!("> OSAlloc::create");

        let alloc_res = os_alloc
            .allocate(Layout::from_size_align(8, 1).unwrap())
            .unwrap();
        info!("> os_alloc.allocate");
    }

    /// Tests the API integration of the OSAlloc.
    #[test]
    fn api() {
        let mm = MemoryManager::mm_init(vec![]).unwrap();
        info!("> MemoryManager::mm_init");

        let mem_obj = mm
            .mm_malloc(Layout::from_size_align(8, 1).unwrap(), Flags::LOCATION_OS)
            .unwrap();
        info!("> os_alloc.mm_malloc");

        debug!("{:?}", mem_obj);

        assert!(mem_obj.owned_by_self());
    }
}
