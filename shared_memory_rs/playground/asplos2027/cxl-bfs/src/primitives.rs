//! Graph helpers for this benchmark, and the shared CXL primitives it builds
//! on, re-exported from the library.

pub use shm_cxl::core::structs::shm_array::{
    ShmSegmentedU64Array, ShmU64Array, alloc_cachelines, alloc_u64_region,
};
pub use shm_cxl::mutex::{HostSharedGuard, HostSharedLock};

/// Segment length for the shared frontier arrays. A frontier is allocated twice
/// for double buffering and reallocated per run, so short segments waste less of
/// the window on the last, partly filled one.
pub const FRONTIER_SEGMENT_U64S: usize = 4 * 1024 * 1024;

pub const UNVISITED: u64 = u64::MAX;

pub fn filter_unvisited(dist_ptr: *const u64, nbrs: *const u64, len: usize, out: &mut Vec<u64>) {
    for k in 0..len {
        let nb = unsafe { *nbrs.add(k) };
        if unsafe { *dist_ptr.add(nb as usize) } == UNVISITED {
            out.push(nb);
        }
    }
}

pub fn gather_sum_f64(vals: *const f64, nbrs: *const u64, len: usize) -> f64 {
    let mut s = 0.0f64;
    for k in 0..len {
        s += unsafe { *vals.add(*nbrs.add(k) as usize) };
    }
    s
}
