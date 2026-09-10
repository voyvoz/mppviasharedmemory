use crate::core::{
    Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync, invalidate, range_flush, range_invalidate,
};
use crate::cxl::{
    CXLSharedState, cxl_typed_drop_guard, cxl_typed_from_cl_sync, cxl_typed_new_sync,
};
use crate::cxl_write_flush;
use allocator_api2::alloc::Allocator;
use log::trace;
use std::alloc::Layout;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

/// A CXL shared memory native Vector implementation.
/// Once created this can be shared with any other host.
/// If a grow is called the metaData Cacheline stays in place and just the backing memory gets transfered if required.
///
/// CXL State data:
///------------------------------------------------------------------
/// | MetaData (marker, len, cap, cl_start*)                        |
/// -----------------------------------------------------------------
///
/// *cl_start:
/// -----------------------------------------------------------------
/// | T ...                                                         |
/// -----------------------------------------------------------------
#[derive(Debug)]
pub struct CXLVector<T, A>
where
    A: Allocator,
    T: Sized,
{
    // Local
    alloc: Arc<A>,

    // Pointer to the immovable CXL memory shared state
    base_cl: *mut Cacheline,

    // T is needed to determine width of a single Element
    _t: PhantomData<T>,
}

unsafe impl<T: Send, A: Allocator> Send for CXLVector<T, A> {}
unsafe impl<T: Sync, A: Allocator> Sync for CXLVector<T, A> {}

// How many elements are currently in the vector
const VEC_LEN_IDX: usize = 4;
// Capacity of this vector with the current allocation
const VEC_CAP_IDX: usize = 5;
// Offset to the data
const VEC_CL_IDX: usize = 6;

pub const VEC_DEFAULT_CAPACITY: usize = 1;

impl<T, A> CXLVector<T, A>
where
    A: Allocator,
    T: Sized,
{
    /// Pointer to the meta data cacheline of this vector
    #[inline]
    pub fn get_base_cl(&self) -> *mut Cacheline {
        self.base_cl
    }
    #[inline]
    pub fn len(&self) -> usize {
        unsafe { *self.base_cl }.d[VEC_LEN_IDX] as usize
    }
    #[inline]
    pub fn cap(&self) -> usize {
        unsafe { *self.base_cl }.d[VEC_CAP_IDX] as usize
    }
    /// Pointer to the start of the actual memory of this vector.
    /// Should only be called after reloading shared state
    #[inline]
    fn get_data_start(&self) -> *mut T {
        let cl_offset = unsafe { *self.base_cl }.d[VEC_CL_IDX];
        let ptr = Cacheline::ptr_from_offset(cl_offset);
        ptr as *mut T
    }
    /// Retrieve most recent shared state from CXL shared memory
    #[inline]
    fn load_metadata_sync(&mut self) {
        cxl_read_cacheline_sync(self.base_cl);
    }

    /// `alloc` is the Allocator to use.
    pub fn new(alloc: Arc<A>) -> Self {
        CXLVector::new_with_capacity(0, alloc)
    }

    /// `default_capacity` space that is allocated by default for this CXLVector instance.
    /// `alloc` is the Allocator to use.
    pub fn new_with_capacity(mut default_capacity: usize, alloc: Arc<A>) -> Self {
        // Maybe we do two allocations here:
        // First for the meta data anchor Cacheline.
        // Second for the moving data storage.
        if default_capacity == 0 {
            default_capacity = VEC_DEFAULT_CAPACITY;
        }
        let cap = 64 + (default_capacity * size_of::<T>()); // 1 CL of meta data + Actual vector data
        // Allocate backing memory
        let mem = alloc
            .allocate(Layout::from_size_align(cap, 64).unwrap())
            .unwrap();
        // Write meta data cacheline
        let base_cl = mem.as_ptr() as *mut Cacheline;
        let alloc_size = mem.len();

        trace!(
            "[{:#}] CXLVec::init(cap={default_capacity}) -> cl:{base_cl:p} size:{alloc_size}",
            Cacheline::ptr_to_offset(base_cl)
        );

        cxl_typed_new_sync::<Self>(base_cl);

        cxl_write_flush!((base_cl => [
            (VEC_LEN_IDX, 0),
            (VEC_CAP_IDX, alloc_size),
            (VEC_CL_IDX, Cacheline::ptr_to_offset(base_cl.add(1))) // TODO for now we place the data directly ahead of the anchor cl
        ]));
        CXLVector {
            base_cl,
            alloc,
            _t: PhantomData,
        }
    }

    /// Recreate a CXLVector from a cacheline.
    pub fn from_cl_sync(base_cl: *mut Cacheline, alloc: Arc<A>) -> Option<Self> {
        trace!(
            "[{:#}] CXLVec::from_cl_sync(addr={base_cl:p})",
            Cacheline::ptr_to_offset(base_cl)
        );
        cxl_typed_from_cl_sync::<Self>(base_cl, |_meta| CXLVector {
            alloc,
            base_cl,
            _t: PhantomData,
        })
    }

    // Should only be called internally -> Meta has been synced already
    //
    // Previous allocation will be copied to new location.
    //
    // Previous allocation gets an update so that shared references know
    // about the new location.
    //
    // Then calls drop on the previous allocation.
    //
    fn grow_sync(&mut self) {
        let old_data_ptr = self.get_data_start();

        let new_mem = self
            .alloc
            .allocate(Layout::from_size_align(self.cap() * 2, 64).unwrap())
            .unwrap();

        let new_data_cl = new_mem.as_ptr() as *mut Cacheline;
        let new_alloc_size = new_mem.len() as u64;
        // Update remote state
        cxl_write_flush!((self.get_base_cl() => [
            (VEC_CAP_IDX, new_alloc_size),
            (VEC_CL_IDX, Cacheline::ptr_to_offset(new_data_cl))
        ]));

        // Copy data to new vector and make sure that new data gets written to CXL shared memory without local cache eviction
        unsafe {
            std::ptr::copy(old_data_ptr as *mut Cacheline, new_data_cl, self.len());
        }
        range_flush(new_data_cl, self.len());

        // TODO Call drop on previous vector elements?
        // do this once we actually use a double alloc

        trace!(
            "CXLVec::grow_sync() -> new_data_cl:{new_data_cl:p} new_alloc_size:{new_alloc_size}"
        );
    }

    /// Pushes a `T`. Grows the backing memory if required.
    /// _sync Variants directly push their state changes to backing memory and invalidate local caches.
    /// In general these require a surrounding lock for correct syncronization.
    pub fn push_sync(&mut self, e: T) {
        self.load_metadata_sync();

        trace!(
            "CXLVec::push_sync() -> marker:{:x} refs:{} | len:{} cap:{}",
            (unsafe { *self.base_cl }).d[0],
            (unsafe { *self.base_cl }).d[1],
            self.len(),
            self.cap()
        );

        if self.len() == self.cap() {
            self.grow_sync();
        }

        // Write element to slot -> sync
        unsafe {
            let data_ptr = self.get_data_start().add(self.len());
            //trace!("{:p} {} -> {data_ptr:p}", self.get_data_start(), self.len());
            *data_ptr = e
        };

        // Update shared state
        cxl_write_flush!((self.get_base_cl(), VEC_LEN_IDX, self.len() + 1));
    }

    /// Pops a T if possible.
    pub fn pop_sync(&mut self) -> Option<T> {
        self.load_metadata_sync();

        trace!(
            "CXLVec::pop_sync() ->  marker:{:x} refs:{} | len:{} cap:{}",
            (unsafe { *self.base_cl }).d[0],
            (unsafe { *self.base_cl }).d[1],
            self.len(),
            self.cap()
        );

        if self.len() == 0 {
            return None;
        }

        // Get the data pointer
        let data_ptr = unsafe { self.get_data_start().add(self.len() - 1) };
        // Invalidate data pointer to enforce reloading from shared state
        invalidate(Cacheline::to_containing_cl(data_ptr as *mut u8));
        // Update shared state
        cxl_write_flush!((self.get_base_cl(), VEC_LEN_IDX, self.len() - 1));
        //trace!("{:p} {} -> {data_ptr:p}", self.get_data_start(), self.len()-1);
        Some(unsafe { std::ptr::read(data_ptr) })
    }

    /// Invalidates all cached entries based on local knowledge of the data
    pub fn invalidate(&self) {
        range_invalidate(
            self.get_data_start() as *mut Cacheline,
            self.len() / (size_of::<T>() / 8),
        );
    }
}

// Sync

impl<T: Sized, A: Allocator> Drop for CXLVector<T, A> {
    fn drop(&mut self) {
        let full_drop = cxl_typed_drop_guard::<Self>(self.get_base_cl());
        trace!(
            "[{:#}] CXLVec::drop(addr={:p}) -> full_drop:{full_drop}",
            Cacheline::ptr_to_offset(self.get_base_cl()),
            self.get_base_cl()
        );
        if full_drop {
            if self.cap() != 0 {
                // Drop data allocation
                let capa = self.cap();
                // Drop each individual element
                while let Some(_) = self.pop_sync() {}
                // Drop the allocation
                let layout = Layout::array::<T>(capa).unwrap();
                unsafe {
                    self.alloc.deallocate(
                        std::ptr::NonNull::new_unchecked(self.get_base_cl() as *mut u8),
                        layout,
                    );
                }

                // TODO drop anchor
                // Kill the metadata
                cl_set_zero_sync(self.get_base_cl());
            }
        }
    }
}

// Does not enforce sync

use std::ops::Deref;
impl<T: Sized + Debug, A: Allocator> Deref for CXLVector<T, A> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.get_data_start(), self.len()) }
    }
}

use std::ops::DerefMut;
impl<T: Sized + Debug, A: Allocator> DerefMut for CXLVector<T, A> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.get_data_start(), self.len()) }
    }
}

impl<T, A: Allocator> CXLSharedState for CXLVector<T, A> {
    /// The actual shared state that must be synced is just a single cacheline that contains
    /// all required information.
    fn required_cxl_shared_state_cachelines_static() -> usize {
        1 // only the meta line is required for construction
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        CXLVector::<T, A>::required_cxl_shared_state_cachelines_static()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        // static + buckets * bucket::rt_size?
        unimplemented!()
    }
    fn get_cxl_type_id() -> u64 {
        0xb00b4269
    }
}

#[cfg(test)]
mod test {
    use super::CXLVector;
    use crate::allocator::os::OSAlloc;
    use crate::core::set_global_offset;
    use log::trace;
    use std::sync::Arc;

    #[test]
    fn base() {
        // This test does not verify correct cross host behaviour!

        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        let mut v: CXLVector<u64, OSAlloc> = CXLVector::new(Arc::new(os_alloc));

        assert!(v.pop_sync() == None);

        v.push_sync(1u64);
        let v_pop_1 = v.pop_sync();
        assert!(v_pop_1.is_some());
        let n = v_pop_1.unwrap();
        assert!(n == 1u64);

        assert!(v.pop_sync().is_none());
        assert!(v.is_empty());

        // Force a grow
        for i in 0..=v.cap() {
            v.push_sync(i as u64 + 1);
        }
        // Capacity should have increased
        assert!(v.cap() >= 10);
        assert!(v.len() < v.cap());
        assert!(v.len() != 0);

        let org_cap = v.cap();
        let org_len = v.len();

        // Recreate the vec again and check it shows the same entries
        let mut v2: CXLVector<u64, OSAlloc> =
            CXLVector::from_cl_sync(v.get_base_cl(), Arc::new(os_alloc)).unwrap();
        assert!(v2.cap() == org_cap);
        assert!(v2.len() == org_len);

        // Make sure both vector point to the same underlying structure
        assert!(v.get_base_cl() == v2.get_base_cl());
        assert!(v.len() == v2.len());
        assert!(v.cap() == v2.cap());

        // Changes should be reflected by both vectors
        v2.pop_sync();

        assert!(v.len() == v2.len());
        assert!(v.cap() == v2.cap());

        for i in v.iter() {
            let value = *i;
            trace!("value {:}", *i);
            assert!(value != 0);
        }
        //drop(v);
        //drop(v2);

        // from_sync -> grow on org -> check that new from_sync on old ptr yields new state
        let mut new_vec: CXLVector<u64, OSAlloc> = CXLVector::new(Arc::new(os_alloc));
        let org_cap = new_vec.cap();
        for i in 0..org_cap {
            new_vec.push_sync(i as u64);
        }
        let current_base = new_vec.get_base_cl();
        let mut new_vec_ref2 = CXLVector::from_cl_sync(current_base, Arc::new(os_alloc)).unwrap();
        new_vec_ref2.push_sync(1 as u64);
        // Reconstruct from old base_cl -> should use the update base_cl instead!
        let new_vec_ref3: CXLVector<u64, OSAlloc> =
            CXLVector::from_cl_sync(current_base, Arc::new(os_alloc)).unwrap();
        assert!(
            new_vec.get_base_cl() == new_vec_ref2.get_base_cl()
                && new_vec_ref2.get_base_cl() == new_vec_ref3.get_base_cl()
        );
        // Ensure that the new meta data is used!
        assert!(new_vec.get_data_start() == new_vec_ref3.get_data_start());
        assert!(new_vec_ref3.cap() != org_cap);
    }
}
