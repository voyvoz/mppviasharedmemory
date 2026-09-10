//! Primitives to read and write CXL shared memory.
//! Using them guarantess cache bypassing behavouir.
use crate::core::isa::primitives::{flush, invalidate, store_fence};
///
use crate::core::{Cacheline, load_and_store_fence};
use crate::utils::timestamps::CxlMetricTimer;

#[inline]
pub fn cxl_read_cacheline_sync(addr: *const Cacheline) -> Cacheline {
    let _timer = CxlMetricTimer::load();
    unsafe {
        // We must explicitly purge from the cache and await that purge
        // before we read the value from remote memory.
        invalidate_sync(addr);
        Cacheline { d: (*addr).d }
    }
}

/// `ptr` must be a Cacheline!
/// Example usage:
///
/// Write value 1 to first u64 as CL pointed to by ptr_a.
///
/// `
/// cxl_write!((ptr_a, 1));
/// `
///
/// Write value 42 to fourth u64 in the CL pointed to by ptr_b.
///
///`
/// cxl_write!(((ptr_b, 3, 42)));
/// `
///
/// Group multiple writes to same cacheline.
///
/// `
/// cxl_write!((ptr_c => [(0, 1), (3, 42)]));
/// `
/// Writing to memory backed by CXL shared memory and guaranteeing that data is
/// propagated as soon as possible and local caches dont retain stale data
/// is ensured when using this macro.
///
/// MOV + CLFLUSHOPT + MFENCE
///
/// This macro allows batching stores if possible.
///
///Further ideas:
/// group by cacheline, only append flush when leaving a cacheline boundary
#[macro_export]
macro_rules! cxl_write_sync {
    // base case that issues the MFENCE after all MOV + CLWB groups
    ( $( $item:tt ),* ) => {{
        let _cxl_store_timer = $crate::utils::timestamps::CxlMetricTimer::store();
        unsafe {
            $( cxl_write_sync!(@item $item); )*
            $crate::core::isa::load_and_store_fence();
        }
    }};

    // cacheline group: ptr => [(index, value), ...]
    (@item ($ptr:expr => [ $( ($i:expr, $val:expr) ),* ])) => {
        $(
            (*$ptr).d[$i] = $val as u64;
        )*
        // CLWB is fine here as we know that the write will set the
        // state to our local value.
        $crate::core::isa::flush($ptr);
    };

    // convenience with default u64 index 0: (ptr, val)
    (@item ($ptr:expr, $val:expr)) => {
        cxl_write_sync!(@item ($ptr => [(0, $val)]));
    };

    // write val at idx at addr ptr
    (@item ($ptr:expr, $idx:expr, $val:expr)) => {
        cxl_write_sync!(@item ($ptr => [($idx, $val)]));
    };
}

/// Writes the given cacheline `cl` to `addr`
#[inline]
pub fn cxl_write_cl_sync(addr: *mut Cacheline, cl: &Cacheline) {
    cxl_write_sync!((addr => [
        (0, cl.d[0]), (1, cl.d[1]), (2, cl.d[2]), (3, cl.d[3]),
        (4, cl.d[4]), (5, cl.d[5]), (6, cl.d[6]), (7, cl.d[7])
    ]));
}

/// Sets each u64 of the indicated Cacheline to v
#[inline]
pub fn cl_set_sync(addr: *mut Cacheline, v: u64) {
    cxl_write_sync!((addr => [
            (0, v),
            (1, v),
            (2, v),
            (3, v),
            (4, v),
            (5, v),
            (6, v),
            (7, v)
    ]));
}

/// Sets the whole Cacheline to zero
#[inline]
pub fn cl_set_zero_sync(addr: *mut Cacheline) {
    cl_set_sync(addr, 0)
}

// sync
#[inline]
pub fn flush_sync(ptr: *const Cacheline) {
    let _timer = CxlMetricTimer::store();
    flush(ptr);
    store_fence();
}
pub fn range_flush_sync(ptr: *const Cacheline, len: usize) {
    let _timer = CxlMetricTimer::store();
    for i in 0..len {
        flush(unsafe { ptr.add(i) });
    }
    store_fence();
}

#[inline]
pub fn invalidate_sync(ptr: *const Cacheline) {
    let _timer = CxlMetricTimer::load();
    invalidate(ptr);
    load_and_store_fence();
}
pub fn range_invalidate_sync(ptr: *const Cacheline, len: usize) {
    let _timer = CxlMetricTimer::load();
    for i in 0..len {
        invalidate(unsafe { ptr.add(i) });
    }
    load_and_store_fence();
}
