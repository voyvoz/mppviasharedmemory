//! Batched, multi-threaded bulk transfer of whole buckets between a host-local
//! buffer and the shared CXL arrays, which are re-exported here from the
//! library.

pub use shm_cxl::core::structs::shm_array::{
    ShmSegmentedU64Array, ShmU64Array, alloc_cachelines, alloc_u64_region,
};

use shm_cxl::core::{
    Cacheline, load_and_store_fence, range_flush, range_invalidate, store_fence, stream_copy_u64,
};

/// One contiguous move between a host-local buffer and the shared array.
#[derive(Clone, Copy, Debug)]
pub struct Transfer {
    pub shared_begin: usize,
    pub local_begin: usize,
    pub len: usize,
}

/// Chunk size for one unit of copy work. Large enough that the per-chunk fence
/// and dispatch cost disappear, small enough that unequal bucket sizes still
/// balance across the workers.
const TRANSFER_CHUNK_U64S: usize = 1 << 19;

struct SharedSlicePtr(*mut u64);

unsafe impl Send for SharedSlicePtr {}
unsafe impl Sync for SharedSlicePtr {}

fn split_transfers(transfers: &[Transfer]) -> Vec<Transfer> {
    let mut chunks = Vec::new();
    for transfer in transfers {
        let mut done = 0usize;
        while done < transfer.len {
            let len = (transfer.len - done).min(TRANSFER_CHUNK_U64S);
            chunks.push(Transfer {
                shared_begin: transfer.shared_begin + done,
                local_begin: transfer.local_begin + done,
                len,
            });
            done += len;
        }
    }
    chunks
}

fn run_chunks<F>(chunks: Vec<Transfer>, threads: usize, apply: F)
where
    F: Fn(&Transfer) + Sync,
{
    if chunks.is_empty() {
        return;
    }
    let workers = threads.max(1).min(chunks.len());
    if workers == 1 {
        chunks.iter().for_each(&apply);
        return;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (chunks, next, apply) = (&chunks, &next, &apply);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    match chunks.get(index) {
                        Some(chunk) => apply(chunk),
                        None => break,
                    }
                }
            });
        }
    });
}

/// Copy host-local ranges of `src` into the shared array with non-temporal
/// stores, in parallel.
///
/// Every `shared_begin` should be cacheline aligned so that concurrent writers
/// never share a line, which non-coherent CXL 2.0 cannot arbitrate.
pub fn publish_transfers(
    array: &ShmSegmentedU64Array,
    src: &[u64],
    transfers: &[Transfer],
    threads: usize,
) {
    run_chunks(split_transfers(transfers), threads, |chunk| {
        let mut local = chunk.local_begin;
        for (pointer, len) in array.contiguous_pieces(chunk.shared_begin, chunk.len) {
            unsafe { stream_copy_u64(pointer, src[local..].as_ptr(), len) };
            // Non-temporal stores that hit a line still cached from an earlier
            // phase update that line instead of memory, so write it back.
            range_flush(
                pointer as *const Cacheline,
                cacheline_span_len(pointer, len),
            );
            local += len;
        }
        store_fence();
    });
}

/// Copy shared ranges into host-local `dst`, in parallel, dropping any locally
/// cached copy of the source lines first.
pub fn fetch_transfers(
    array: &ShmSegmentedU64Array,
    dst: &mut [u64],
    transfers: &[Transfer],
    threads: usize,
) {
    let dst = SharedSlicePtr(dst.as_mut_ptr());
    let dst = &dst;
    run_chunks(split_transfers(transfers), threads, move |chunk| {
        let mut local = chunk.local_begin;
        for (pointer, len) in array.contiguous_pieces(chunk.shared_begin, chunk.len) {
            range_invalidate(
                pointer as *const Cacheline,
                cacheline_span_len(pointer, len),
            );
            // `clflushopt` is weakly ordered against the loads that follow it.
            load_and_store_fence();
            unsafe { std::ptr::copy_nonoverlapping(pointer, dst.0.add(local), len) };
            local += len;
        }
    });
}

/// Number of cachelines touched by `len` u64 values starting at `pointer`.
#[inline]
fn cacheline_span_len(pointer: *mut u64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let first = pointer as usize / 64;
    let last = (pointer as usize + (len - 1) * 8) / 64;
    last - first + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use shm_cxl::allocator_api2::alloc::Global;
    use shm_cxl::core::set_global_offset;

    /// Shared offsets are byte offsets from the mapped CXL base. Heap-backed
    /// test arrays use a zero base, which makes them their own address space.
    fn init_offsets() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| set_global_offset(0));
    }

    fn round_up_to_cacheline(value: usize) -> usize {
        value.div_ceil(8) * 8
    }

    /// Exercises chunk splitting, segment addressing and the non-temporal
    /// store path against a heap-backed array; the CXL cache maintenance
    /// instructions are valid on ordinary memory too.
    #[test]
    fn bulk_transfers_round_trip_through_the_shared_array() {
        init_offsets();
        let len = 3 * TRANSFER_CHUNK_U64S + 1_234;
        let (array, _table) = ShmSegmentedU64Array::create_uninit(&Global, len);
        let source: Vec<u64> = (0..len as u64)
            .map(|value| value.wrapping_mul(31))
            .collect();

        // Two cacheline-aligned regions with an unaligned length each, as the
        // exchange produces for padded per-destination buckets.
        let first = TRANSFER_CHUNK_U64S + 13;
        let transfers = [
            Transfer {
                shared_begin: 0,
                local_begin: 0,
                len: first,
            },
            Transfer {
                shared_begin: round_up_to_cacheline(first),
                local_begin: first,
                len: len - round_up_to_cacheline(first),
            },
        ];
        let copied: usize = transfers.iter().map(|transfer| transfer.len).sum();
        for threads in [1usize, 4] {
            publish_transfers(&array, &source, &transfers, threads);
            let mut fetched = vec![0u64; len];
            fetch_transfers(&array, &mut fetched, &transfers, threads);
            assert_eq!(fetched[..copied], source[..copied]);
        }
    }

    #[test]
    fn contiguous_pieces_cover_the_range_exactly() {
        init_offsets();
        let (array, _table) = ShmSegmentedU64Array::create_uninit(&Global, 4096);
        let pieces: Vec<_> = array.contiguous_pieces(7, 1_000).collect();
        assert_eq!(pieces.iter().map(|&(_, len)| len).sum::<usize>(), 1_000);
        // 4096 elements is a single segment, so the range is one piece and it
        // starts where a one-element view of the same index starts.
        assert_eq!(pieces.len(), 1);
        let head: Vec<_> = array.contiguous_pieces(7, 1).collect();
        assert_eq!(pieces[0].0, head[0].0);
    }
}
