use crate::core::Cacheline;

// Call a flush on the given address to ensure data is written to the backing memory
#[inline]
pub fn flush(ptr: *const Cacheline) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::asm!(
                "clwb [{ptr}]",
                ptr = in(reg) ptr,
            );
        }
    }
}

// Call an invalidate to ensure data currently present in caches will be loaded from the backing memory on next access
#[inline]
pub fn invalidate(ptr: *const Cacheline) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::asm!(
                "clflushopt [{ptr}]",
                ptr = in(reg) ptr,
            );
        }
    }
}

pub fn prefetch_into_l0(ptr: *const Cacheline) {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe {
            std::arch::x86_64::_mm_prefetch::<3>(ptr as *const i8)
        }
    }
}

// Fence
// https://stackoverflow.com/a/27608800
// Might not be usefull at all...
#[inline]
pub fn load_fence() {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_mm_lfence() }
    }
}
#[inline]
pub fn store_fence() {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_mm_sfence() }
    }
}
#[inline]
pub fn load_and_store_fence() {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { std::arch::x86_64::_mm_mfence() }
    }
}

/// Copy `len` u64 values into CXL memory with non-temporal stores.
///
/// Ordinary stores to write-back memory first fetch the target cacheline for
/// ownership, so publishing a large buffer costs one read plus one write of the
/// same bytes over the CXL link. `movnti` writes through the write-combining
/// buffers instead, halving that traffic and leaving the caches free for the
/// application's working set.
///
/// # Safety
/// `dst` must be writable for `len` u64 values, must not alias `src`, and both
/// pointers must be 8-byte aligned. Non-temporal stores are weakly ordered:
/// the calling thread must execute [`store_fence`] before the written range is
/// published to other hosts, and any line that may still be cached locally must
/// be flushed with [`flush`] or [`range_flush`] afterwards.
#[inline]
pub unsafe fn stream_copy_u64(dst: *mut u64, src: *const u64, len: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in 0..len {
            // `_mm_stream_si64` is baseline SSE2 on x86_64, so no runtime
            // feature detection is needed on any supported host.
            std::arch::x86_64::_mm_stream_si64(dst.add(i) as *mut i64, *src.add(i) as i64);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, len);
    }
}

// range variants

pub fn range_flush(ptr: *const Cacheline, len: usize) {
    for i in 0..len {
        flush(unsafe { ptr.add(i) });
    }
}
pub fn range_invalidate(ptr: *const Cacheline, len: usize) {
    for i in 0..len {
        invalidate(unsafe { ptr.add(i) });
    }
}

//

/// `ptr` must be a `*mut Cacheline`!
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
///
/// Writing to memory backed by CXL shared memory and guaranteeing that data is
/// propagated as soon as possible and local caches dont retain stale data
/// is ensured when using this macro.
///
/// MOV + CLWB
///
/// This macro allows batching stores if possible.
#[macro_export]
macro_rules! cxl_write_flush {
    // base case that issues the MFENCE after all MOV + CLWB groups
    ( $( $item:tt ),* ) => {{
        let _cxl_store_timer = $crate::utils::timestamps::CxlMetricTimer::store();
        unsafe {
            $( cxl_write_flush!(@item $item); )*
        }
    }};

    // cacheline group: ptr => [(index, value), ...]
    (@item ($ptr:expr => [ $( ($i:expr, $val:expr) ),* ])) => {
        $(
            (*$ptr).d[$i] = $val as u64;
        )*
        $crate::core::isa::flush($ptr);
    };

    // convenience with default u64 index 0: (ptr, val)
    (@item ($ptr:expr, $val:expr)) => {
        cxl_write_flush!(@item ($ptr => [(0, $val)]));
    };

    // write val at idx at addr ptr
    (@item ($ptr:expr, $idx:expr, $val:expr)) => {
        cxl_write_flush!(@item ($ptr => [($idx, $val)]));
    };
}

#[inline]
pub fn cxl_write_cl_flush(addr: *mut Cacheline, cl: &Cacheline) {
    cxl_write_flush!((addr => [
        (0, cl.d[0]), (1, cl.d[1]), (2, cl.d[2]), (3, cl.d[3]),
        (4, cl.d[4]), (5, cl.d[5]), (6, cl.d[6]), (7, cl.d[7])
    ]));
}
