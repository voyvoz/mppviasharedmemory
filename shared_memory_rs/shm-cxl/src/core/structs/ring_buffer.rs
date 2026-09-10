//! SPSC ring buffer over CXL shared memory.
//!
//! Layout per ring:
//!     [CL 0: Meta data]
//!     [CL 1: head_cl]
//!     [CL 2: tail_cl]
//!     [CL 4 - CL (Capacity * Slot size): Slots]
//!

use crate::core::{
    Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync, cxl_write_cl_flush, invalidate, load_and_store_fence, prefetch_into_l0, range_flush, range_invalidate, range_invalidate_sync, store_fence,
};
use crate::cxl::{CXLSharedState, cxl_new_sized_obj, cxl_typed_from_cl_sync};
use crate::cxl_v2::CXLSizedObject;
use crate::cxl_write_flush;
use allocator_api2::alloc::Allocator;
use log::{debug, trace};
use std::alloc::Layout;
use std::slice;
use std::sync::Arc;

// ------------------------------------------------------
// CXLRBParams
// ------------------------------------------------------

pub const DEFAULT_RING_CAPACITY: usize = 128;
pub const DEFAULT_SLOT_SIZE: usize = 1;

#[derive(Debug, Clone, Copy)]
pub struct CXLRBParams {
    pub capacity: usize,     // How many slots are available
    pub slot_size_cl: usize, // How many cachelines a single slot occupies
}

impl CXLRBParams {
    pub fn default() -> Self {
        CXLRBParams {
            capacity: DEFAULT_RING_CAPACITY,
            slot_size_cl: DEFAULT_SLOT_SIZE,
        }
    }
    pub fn with_slot_size_cl(mut self, slot_size_cl: usize) -> Self {
        self.slot_size_cl = slot_size_cl;
        self
    }
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
}

impl CXLSizedObject for CXLRBParams {
    fn required_cachelines(&self) -> usize {
        // Meta + Head_idx + Tail_idx + (capacity * slot size)
        3 + (self.capacity * self.slot_size_cl)
    }
}

// ------------------------------------------------------
// CXLRingBuffer
// ------------------------------------------------------

// TODO if i would be better at rust i could find out how i could use [u8] instead of this thing...
pub struct RBSlice {
    ptr: *const u8,
    len: usize,
}
impl RBSlice {
    pub fn from_slice(slice: &[u8]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }
    pub fn ptr(&self) -> *const u8 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingError {
    Full,
    Empty,
}

/// This RingBuffer variant stores all state in CXL shared memory.
/// Once allocated its stays static.
#[derive(Debug)]
pub struct CXLRingBuffer {
    // Stats
    msg_counter: u64,

    // Shared state
    base_cl: *mut Cacheline,

    local_tail_counter: u64,
    local_head_counter: u64,

    pub options: CXLRBParams,

    internal_state: u32,
}

impl CXLRingBuffer {
    #[inline]
    fn head_cl(&self) -> *mut Cacheline {
        unsafe { self.base_cl.add(1) }
    }
    #[inline]
    fn tail_cl(&self) -> *mut Cacheline {
        unsafe { self.base_cl.add(2) }
    }
    #[inline]
    fn slots(&self) -> *mut Cacheline {
        unsafe { self.base_cl.add(3) }
    }
    #[inline]
    pub fn clear_sync(&self) {
        cl_set_zero_sync(self.head_cl());
        cl_set_zero_sync(self.tail_cl());
    }
    #[inline]
    fn idx_head(&self) -> usize {
        (self.local_head_counter as usize) % self.options.capacity
    }
    #[inline]
    fn idx_tail(&self) -> usize {
        (self.local_tail_counter as usize) % self.options.capacity
    }

    /// Create a ringbuffer structue allocating memory from `alloc`.
    /// Uses default parameter values.
    pub fn new_with_defaults(alloc: Arc<impl Allocator>) -> Self {
        debug!("CXLRingBuffer::new_with_defaults()");
        let opts = CXLRBParams::default();

        let mem_layout = Layout::from_size_align(opts.required_bytes(), 64).unwrap();
        let mem = alloc.allocate(mem_layout).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        CXLRingBuffer::init(base_cl, opts)
    }

    /// Create a ringbuffer structue allocating memory from `alloc`.
    pub fn new(alloc: Arc<impl Allocator>, options: CXLRBParams) -> Self {
        debug!("CXLRingBuffer::new(params={options:?})");

        let mem_layout = Layout::from_size_align(options.required_bytes(), 64).unwrap();
        let mem = alloc.allocate(mem_layout).unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        CXLRingBuffer::init(base_cl, options)
    }

    /// Treat `base_cl` as the start of a CXLRingBuffer.
    /// Uses default parameters.
    pub fn init_with_defaults(base_cl: *mut Cacheline) -> Self {
        CXLRingBuffer::init(base_cl, CXLRBParams::default())
    }

    /// Treat `base_cl` as the start of a CXLRingBuffer.
    /// `capacity` indicates how many slots are available
    /// `slot_size` indicates how many cachelines a single slot occupies
    pub fn init(base_cl: *mut Cacheline, options: CXLRBParams) -> Self {
        debug!(
            "[{:#}] CXLRingBuffer::init(base_cl={base_cl:p}, params={options:?})",
            Cacheline::ptr_to_offset(base_cl)
        );
        cxl_new_sized_obj::<Self>(base_cl, options.required_cachelines() as u64);
        let rb = CXLRingBuffer {
            base_cl,
            msg_counter: 0,
            local_tail_counter: 0,
            local_head_counter: 0,
            options,
            internal_state: 0,
        };
        rb.clear_sync();
        rb
    }

    /// Recreate a RingBuffer from its shared state.
    /// Increments the reference counter.
    pub fn from_cl_sync(base_cl: *mut Cacheline, options: CXLRBParams) -> Option<Self> {
        debug!(
            "[{:#}] CXLRingBuffer::from_cl_sync(base_cl={base_cl:p})",
            Cacheline::ptr_to_offset(base_cl)
        );
        cxl_typed_from_cl_sync::<Self>(base_cl, |_m| {
            let mut rb = CXLRingBuffer {
                base_cl,
                msg_counter: 0,
                local_tail_counter: 0,
                local_head_counter: 0,
                options,
                internal_state: 0,
            };
            rb.local_head_counter = cxl_read_cacheline_sync(rb.head_cl()).d[0];
            rb.local_tail_counter = cxl_read_cacheline_sync(rb.tail_cl()).d[0];
            rb
        })
    }

    /// Check local state first, then reads from remote state to be sure that its really empty
    pub fn is_empty(&mut self) -> bool {
        if self.local_tail_counter == self.local_head_counter {
            let counter = cxl_read_cacheline_sync(self.tail_cl()).d[0];
            self.local_tail_counter = counter;
            if counter == self.local_head_counter {
                return true;
            }
        }
        false
    }

    /// Check local state first, then reads from remote state to be sure that its really full
    pub fn is_full(&mut self) -> bool {
        if (self
            .local_tail_counter
            .wrapping_sub(self.local_head_counter)) as usize
            >= self.options.capacity
        {
            self.local_head_counter = cxl_read_cacheline_sync(self.head_cl()).d[0];
            if (self
                .local_tail_counter
                .wrapping_sub(self.local_head_counter)) as usize
                >= self.options.capacity
            {
                return true;
            }
        }
        false
    }

    /// Add an element into the buffer if there is an empty slot
    pub fn try_push(&mut self, entry: &Cacheline) -> Result<(), RingError> {
        //trace!(
        //  "[{:#}] CXLRingBuffer::try_push()",
        //  Cacheline::ptr_to_offset(self.base_cl)
        //);
        if self.is_full() {
            return Err(RingError::Full);
        }

        let tail_idx = self.idx_tail();
        let slot_cl = unsafe { self.slots().add(tail_idx) };
        cxl_write_cl_flush(slot_cl, entry);

        // Update state
        self.local_tail_counter = self.local_tail_counter.wrapping_add(1);
        cxl_write_flush!((self.tail_cl(), self.local_tail_counter));

        self.msg_counter = self.msg_counter.wrapping_add(1);

        Ok(())
    }

    ///
    pub fn try_pop(&mut self) -> Result<Cacheline, RingError> {
        //trace!(
        //    "[{:#}] CXLRingBuffer::try_pop()",
        //    Cacheline::ptr_to_offset(self.base_cl)
        //);
        if self.is_empty() {
            return Err(RingError::Empty);
        }

        let head_idx = self.idx_head();
        let slot_cl = unsafe { self.slots().add(head_idx) };
        let cl = cxl_read_cacheline_sync(slot_cl);

        // Update state
        self.local_head_counter = self.local_head_counter.wrapping_add(1);
        cxl_write_flush!((self.head_cl(), self.local_head_counter));

        Ok(cl)
    }


    /// EXPERIMENTAL
    /// Returns all available messages as a single slice.
    /// Race conditions will happen when writes are still faster
    pub fn try_pop_all(&mut self) -> Result<&[Cacheline], RingError> {
        //trace!(
        //    "[{:#}] CXLRingBuffer::try_pop()",
        //    Cacheline::ptr_to_offset(self.base_cl)
        //);

        if self.is_empty() {
            return Err(RingError::Empty);
        }

        // Assemble slice
        let start_slot_cl = unsafe { self.slots().add(self.idx_head()) };
        let elems = self.idx_tail() - self.idx_head();
        range_invalidate(start_slot_cl, elems);

        // Update state
        self.local_head_counter = self.local_head_counter.wrapping_add(elems as u64);
        cxl_write_flush!((self.head_cl(), self.local_head_counter));

        // This could lead to race conditions for sure but avoids copies...
        let s = unsafe { slice::from_raw_parts(start_slot_cl, elems) };

        load_and_store_fence();

        Ok(s)
    }

    /// Experimental
    /// A different version that goes through an internal state machine instead of doing a blocking read
    /// Upon reaching pop state copies the available data
    pub fn try_pop_all_with_state_machine(&mut self) -> Option<Result<Vec<Cacheline>, RingError>> {

        match self.internal_state {
            // State 1 - issue invalidate
            0 => {
                invalidate(self.tail_cl());
                self.internal_state = 1;
            }
            // State 2 - mfence and prefetch
            1 => {
                load_and_store_fence();
                prefetch_into_l0(self.tail_cl());
                self.internal_state = 2;
            }
            // State 3 - Check
            _ => {
                self.internal_state = 0;
                // Empty check
                if (unsafe{*self.tail_cl()}).d[0] == self.local_head_counter {
                    return Some(Err(RingError::Empty));
                }

                let start_slot_cl = unsafe { self.slots().add(self.idx_head()) };
                let elems = self.idx_tail() - self.idx_head();
                range_invalidate(start_slot_cl, elems);

                let mut res = vec![];
                for i in 0..elems {
                    res.push(
                        unsafe {
                            *(start_slot_cl.add(i))
                        }
                    );
                }

                // Update state
                self.local_head_counter = self.local_head_counter.wrapping_add(elems as u64);
                cxl_write_flush!((self.head_cl(), self.local_head_counter));

                load_and_store_fence();

                return Some(Ok(res))
            }
        }
        None
    }

    // -- slice API

    /// Push a slice of size up to `params.slot_size_cl` into the rb if able
    pub fn try_push_slice(&mut self, entry: RBSlice) -> Result<(), RingError> {
        // Slice API requires the first Cacheline as a metadata field
        assert!(64 + entry.len() <= self.options.slot_size_cl * 64);
        //trace!(
        //    "[{:#}] CXLRingBuffer::try_push()",
        //    Cacheline::ptr_to_offset(self.base_cl)
        //);
        if self.is_full() {
            return Err(RingError::Full);
        }

        let tail_idx = self.idx_tail() * self.options.slot_size_cl;
        let slot_cl = unsafe { self.slots().add(tail_idx) };
        let mut meta = Cacheline::empty();
        meta.d[0] = 0x1010;
        meta.d[1] = entry.len as u64;
        meta.d[2] = self.msg_counter;
        cxl_write_cl_flush(slot_cl, &meta);

        let dst_data_ptr = unsafe { slot_cl.add(1) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                entry.ptr as *const u8,
                dst_data_ptr as *mut u8,
                entry.len,
            )
        };
        range_flush(dst_data_ptr, entry.len.div_ceil(64));

        store_fence();

        //debug!("tail_idx {tail_idx} -> [{}] {:02X?}...", Cacheline::ptr_to_offset(slot_cl), &entry.as_slice()[..8]);

        // Update state
        self.local_tail_counter = self.local_tail_counter.wrapping_add(1);
        cxl_write_flush!((self.tail_cl(), self.local_tail_counter));

        Ok(())
    }

    /// Pop a slice of size up to `params.slot_size_cl` from the rb if able
    pub fn try_pop_slice(&mut self) -> Result<RBSlice, RingError> {
        //trace!(
        //    "[{:#}] CXLRingBuffer::try_pop_slice()",
        //    Cacheline::ptr_to_offset(self.base_cl)
        //);
        if self.is_empty() {
            return Err(RingError::Empty);
        }

        let head_idx = self.idx_head() * self.options.slot_size_cl;
        let slot_cl = unsafe { self.slots().add(head_idx) };
        let cl = cxl_read_cacheline_sync(slot_cl);
        if cl.d[0] != 0x1010 {
            panic!("Something went wrong. Packet header is broken");
        }
        // First CL contains packet information
        let len = cl.d[1] as usize;

        let data_cl = unsafe { slot_cl.add(1) };

        range_invalidate(data_cl, len.div_ceil(64));

        let entry = RBSlice {
            ptr: data_cl as *const u8,
            len,
        };

        //debug!("head_idx {head_idx} -> [{}] {:02X?}...", Cacheline::ptr_to_offset(slot_cl), &entry.as_slice()[..8]);

        // Update state
        self.local_head_counter = self.local_head_counter.wrapping_add(1);
        cxl_write_flush!((self.head_cl(), self.local_head_counter));

        Ok(entry)
    }

    /// Pop a slice of size up to `params.slot_size_cl` from the rb if able
    pub fn try_pop_all_slice(&mut self) -> Result<Vec<RBSlice>, RingError> {
        //trace!(
        //    "[{:#}] CXLRingBuffer::try_pop_slice()",
        //    Cacheline::ptr_to_offset(self.base_cl)
        //);
        if self.is_empty() {
            return Err(RingError::Empty);
        }

        let elems = (self.idx_tail() - self.idx_head()) % self.options.slot_size_cl;
        let mut v = Vec::with_capacity(elems);

        for _i in 0..elems {
            let head_idx = self.idx_head() * self.options.slot_size_cl;
            let slot_cl = unsafe { self.slots().add(head_idx) };
            let cl = cxl_read_cacheline_sync(slot_cl);
            if cl.d[0] != 0x1010 {
                panic!("Something went wrong. Packet header is broken");
            }
            // First CL contains packet information
            let len = cl.d[1] as usize;

            let data_cl = unsafe { slot_cl.add(1) };

            range_invalidate(data_cl, len.div_ceil(64));

            v.push(RBSlice {
                ptr: data_cl as *const u8,
                len,
            });
        }

        //debug!("head_idx {head_idx} -> [{}] {:02X?}...", Cacheline::ptr_to_offset(slot_cl), &entry.as_slice()[..8]);

        // Update state
        self.local_head_counter = self.local_head_counter.wrapping_add(1);
        cxl_write_flush!((self.head_cl(), self.local_head_counter));

        Ok(v)
    }
}

impl CXLSharedState for CXLRingBuffer {
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
        0x12120013
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::allocator::os::OSAlloc;
    use crate::core::Cacheline;
    use crate::core::set_global_offset;
    use crate::cxl_write_sync;
    use crate::utils::open_test_cxl_dev;
    use std::thread;
    use std::time::SystemTime;

    #[test]
    fn base() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        let mut rb = CXLRingBuffer::new_with_defaults(Arc::new(os_alloc));

        let mut dummy_data = Cacheline {
            d: [1, 2, 3, 4, 5, 6, 7, 8],
        };

        assert!(rb.is_empty());

        rb.try_push(&dummy_data).unwrap();

        assert!(!rb.is_empty());

        let res = rb.try_pop().unwrap();

        assert_eq!(dummy_data, res);

        assert!(rb.is_empty());

        // Fill
        let start_val = 10;
        for i in 0..DEFAULT_RING_CAPACITY {
            dummy_data.d[0] = (start_val + i) as u64;
            rb.try_push(&dummy_data).unwrap();
        }
        assert!(!rb.is_empty());
        assert!(rb.is_full());
        let should_fail = rb.try_push(&dummy_data);
        assert!(should_fail.is_err());
        let err = should_fail.unwrap_err();
        assert_eq!(err, RingError::Full);

        // Drain again and check for correctness
        for i in 0..DEFAULT_RING_CAPACITY {
            let res = rb.try_pop().unwrap();
            assert_eq!(res.d[0], (start_val + i) as u64);
        }
        assert!(rb.is_empty());
        let res = rb.try_pop();
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err, RingError::Empty);
    }

    #[test]
    fn base_slice() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        let mut rb = CXLRingBuffer::new(
            Arc::new(os_alloc),
            CXLRBParams::default().with_slot_size_cl(1024),
        );

        let entry_size_bytes = 65;
        let mut data = os_alloc
            .allocate(std::alloc::Layout::from_size_align(entry_size_bytes, 64).unwrap())
            .unwrap();
        let ptr: *mut u8 = unsafe { data.as_mut().as_mut_ptr().add(8) };
        for i in 0..entry_size_bytes {
            unsafe { *(ptr.add(i)) = i as u8 };
        }
        let entry = RBSlice {
            ptr,
            len: entry_size_bytes,
        };
        rb.try_push_slice(entry).unwrap();

        let res = rb.try_pop_slice().unwrap();
        assert!(res.len == entry_size_bytes);
        let result = res.ptr();
        for i in 0..entry_size_bytes {
            let v = unsafe { *result.add(i) };
            assert!(v == i as u8)
        }

        let icmp_vec = vec![
            45, 00, 00, 54, 0x3D, 97, 40, 00, 40, 01, 0xE9, 0x0F, 0x0A, 00, 00, 00, 0x0A, 00, 00,
            03, 08, 00, 53, 0xF8, 27, 0xCD, 00, 04, 33, 0xBE, 77, 0x6A, 00, 00, 00, 00, 03, 0x3B,
            0x0F, 00, 00, 00, 00, 00, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0x1A, 0x1B, 0x1C,
            0x1D, 0x1E, 0x1F, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E,
            0x2F, 30, 31, 32, 33, 34, 35, 36, 37,
        ];
        let icmp_slice = icmp_vec.as_slice();

        for _i in 0..2 {
            let entry2 = RBSlice::from_slice(icmp_slice);
            rb.try_push_slice(entry2).unwrap();

            let res = rb.try_pop_slice().unwrap();
            let icmp_msg_slice = res.as_slice();

            assert!(icmp_msg_slice.eq(icmp_msg_slice));
        }
    }

    #[test]
    fn base_pop_all() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        let mut rb = CXLRingBuffer::new(
            Arc::new(os_alloc),
            CXLRBParams::default().with_slot_size_cl(1024),
        );

        rb.try_push(&Cacheline::from_b0(1)).unwrap();
        rb.try_push(&Cacheline::from_b0(2)).unwrap();

        let element_slice = rb.try_pop_all().unwrap();
        assert!(element_slice.len() == 2);

        assert!(element_slice[0].d[0] == 1);
        assert!(element_slice[1].d[0] == 2);
    }

    #[test]
    fn base_pop_all_state() {
        let _ = env_logger::try_init();

        let os_alloc = OSAlloc::create();

        set_global_offset(0);

        let mut rb = CXLRingBuffer::new(
            Arc::new(os_alloc),
            CXLRBParams::default().with_slot_size_cl(1024),
        );

        rb.try_push(&Cacheline::from_b0(1)).unwrap();
        rb.try_push(&Cacheline::from_b0(2)).unwrap();
        rb.try_push(&Cacheline::from_b0(3)).unwrap();

        let s0 = rb.try_pop_all_with_state_machine();
        assert!(s0 == None);
        let s1 = rb.try_pop_all_with_state_machine();
        assert!(s1 == None);
        let element_slice = rb.try_pop_all_with_state_machine().unwrap().unwrap();
        assert!(element_slice.len() == 3);

        assert!(element_slice[0].d[0] == 1);
        assert!(element_slice[1].d[0] == 2);
        assert!(element_slice[2].d[0] == 3);

        let s0_again = rb.try_pop_all_with_state_machine();
        assert!(s0_again == None);
    }

    #[test]
    fn cxl_base() {
        let (mem, _) = open_test_cxl_dev();

        let mut rb = CXLRingBuffer::init_with_defaults(mem);

        let mut dummy_data = Cacheline {
            d: [1, 2, 3, 4, 5, 6, 7, 8],
        };

        assert!(rb.is_empty());

        rb.try_push(&dummy_data).unwrap();

        assert!(!rb.is_empty());

        let res = rb.try_pop().unwrap();

        assert_eq!(dummy_data, res);

        assert!(rb.is_empty());

        // Fill
        let start_val = 10;
        for i in 0..DEFAULT_RING_CAPACITY {
            dummy_data.d[0] = (start_val + i) as u64;
            rb.try_push(&dummy_data).unwrap();
        }
        assert!(!rb.is_empty());
        assert!(rb.is_full());
        let should_fail = rb.try_push(&dummy_data);
        assert!(should_fail.is_err());
        let err = should_fail.unwrap_err();
        assert_eq!(err, RingError::Full);

        // Drain again and check for correctness
        for i in 0..DEFAULT_RING_CAPACITY {
            let res = rb.try_pop().unwrap();
            assert_eq!(res.d[0], (start_val + i) as u64);
        }
        assert!(rb.is_empty());
        let res = rb.try_pop();
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert_eq!(err, RingError::Empty);
    }

    #[test]
    fn cxl_edge_case() {
        let (mem, _) = open_test_cxl_dev();

        let mut rb = CXLRingBuffer::init_with_defaults(mem);

        assert!(rb.is_empty());

        rb.local_head_counter = u64::MAX - 1;
        rb.local_tail_counter = u64::MAX - 1;
        cxl_write_sync!(
            (rb.head_cl(), rb.local_head_counter),
            (rb.tail_cl(), rb.local_tail_counter)
        );
        assert!(rb.is_empty());

        let mut d = Cacheline::from_b0(1);
        rb.try_push(&d).unwrap();
        println!("{rb:?} <- {d:?}");
        assert!(!rb.is_full());
        assert!(!rb.is_empty());

        d.d[0] = 2;
        rb.try_push(&d).unwrap();
        println!("{rb:?} <- {d:?}");
        assert!(!rb.is_full());
        assert!(!rb.is_empty());

        let d_res = rb.try_pop().unwrap();
        println!("{rb:?} -> {d_res:?}");
        assert_eq!(d_res.d[0], 1);
        assert!(!rb.is_full());
        assert!(!rb.is_empty());

        let d_res2 = rb.try_pop().unwrap();
        println!("{rb:?} -> {d_res2:?}");
        assert_eq!(d_res2.d[0], 2);
        assert!(!rb.is_full());
        assert!(rb.is_empty());

        let d_res3 = rb.try_pop();
        println!("{rb:?} -> {d_res3:?}");
        assert!(d_res3.is_err());
        assert!(!rb.is_full());
        assert!(rb.is_empty());
    }

    #[test]
    fn cxl_bench() {
        let (mem, _) = open_test_cxl_dev();
        let mem_ptr = mem as usize;

        let config = CXLRBParams::default();

        let target = 1000000;

        let prod = thread::spawn(move || {
            let mut rb = CXLRingBuffer::init_with_defaults(mem_ptr as *mut Cacheline);

            let mut stalled = 0;
            let mut dummy_data = Cacheline {
                d: [1, 2, 3, 4, 5, 6, 7, 8],
            };
            let start = SystemTime::now();
            for i in 0..target {
                dummy_data.d[0] = i;
                loop {
                    let res = rb.try_push(&dummy_data);
                    if res.is_ok() {
                        break;
                    }
                    stalled = stalled + 1;
                }
            }
            let dur = start.elapsed().unwrap();
            println!(
                "Producer took {:?} -> {:?}/msg for {target} messages (stalls: {stalled})",
                dur,
                dur / target as u32
            );
        });
        let con = thread::spawn(move || {
            let mut rb = CXLRingBuffer::from_cl_sync(mem_ptr as *mut Cacheline, config).unwrap();
            let mut stalled = 0;
            let start = SystemTime::now();
            for _i in 0..target {
                loop {
                    let res = rb.try_pop();
                    if res.is_ok() {
                        break;
                    }
                    stalled = stalled + 1;
                }
            }
            let dur = start.elapsed().unwrap();
            println!(
                "Consumer took {:?} -> {:?}/msg for {target} messages (stalls: {stalled})",
                dur,
                dur / target as u32
            );
        });

        prod.join().unwrap();
        con.join().unwrap();
    }
}
