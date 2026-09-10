pub const RADIX_BITS: usize = 8;
pub const BUCKETS: usize = 1 << RADIX_BITS;
pub const PASSES: usize = u64::BITS as usize / RADIX_BITS;

struct SharedMutPtr(*mut u64);

unsafe impl Send for SharedMutPtr {}
unsafe impl Sync for SharedMutPtr {}

impl SharedMutPtr {
    #[inline]
    unsafe fn write(&self, index: usize, value: u64) {
        unsafe { *self.0.add(index) = value };
    }
}

#[inline]
pub fn digit(value: u64, pass: usize) -> usize {
    ((value >> (pass * RADIX_BITS)) & (BUCKETS as u64 - 1)) as usize
}

/// One stable, threaded LSD radix pass.
///
/// Each worker counts a contiguous input range. Bucket offsets include all
/// earlier buckets and all earlier workers in the same bucket, which gives
/// every worker a disjoint output range while retaining global stability.
pub fn radix_pass_parallel(
    input: &[u64],
    pass: usize,
    requested_threads: usize,
) -> (Vec<u64>, [usize; BUCKETS]) {
    assert!(pass < PASSES);
    if input.is_empty() {
        return (Vec::new(), [0; BUCKETS]);
    }

    let threads = requested_threads.max(1).min(input.len());
    let histograms: Vec<[usize; BUCKETS]> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|worker| {
                let begin = input.len() * worker / threads;
                let end = input.len() * (worker + 1) / threads;
                scope.spawn(move || {
                    let mut histogram = [0usize; BUCKETS];
                    for &value in &input[begin..end] {
                        histogram[digit(value, pass)] += 1;
                    }
                    histogram
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut totals = [0usize; BUCKETS];
    for histogram in &histograms {
        for bucket in 0..BUCKETS {
            totals[bucket] += histogram[bucket];
        }
    }

    let mut offsets = vec![[0usize; BUCKETS]; threads];
    let mut bucket_begin = 0usize;
    for bucket in 0..BUCKETS {
        let mut cursor = bucket_begin;
        for worker in 0..threads {
            offsets[worker][bucket] = cursor;
            cursor += histograms[worker][bucket];
        }
        bucket_begin = cursor;
    }

    let mut output = vec![0u64; input.len()];
    let output_ptr = SharedMutPtr(output.as_mut_ptr());
    std::thread::scope(|scope| {
        let output_ptr = &output_ptr;
        for (worker, worker_offsets) in offsets.into_iter().enumerate() {
            let begin = input.len() * worker / threads;
            let end = input.len() * (worker + 1) / threads;
            scope.spawn(move || {
                let mut positions = worker_offsets;
                for &value in &input[begin..end] {
                    let bucket = digit(value, pass);
                    let position = positions[bucket];
                    unsafe { output_ptr.write(position, value) };
                    positions[bucket] += 1;
                }
            });
        }
    });

    (output, totals)
}

/// A stable threaded radix pass with a runtime-selected digit width.
///
/// The CXL implementation uses wider digits than the local baseline to reduce
/// the number of global exchange and synchronization rounds.
pub fn radix_pass_parallel_bits(
    input: &[u64],
    shift: usize,
    bits: usize,
    requested_threads: usize,
) -> (Vec<u64>, Vec<usize>) {
    let mut output = vec![0u64; input.len()];
    let totals = radix_pass_into(input, &mut output, shift, bits, requested_threads);
    (output, totals)
}

/// The same pass, scattering into a caller-owned buffer.
///
/// Multi-pass sorts alternate between two buffers instead of allocating and
/// first-touching a fresh one per pass, which for partitions of hundreds of
/// megabytes costs more in page faults than the pass itself.
pub fn radix_pass_into(
    input: &[u64],
    output: &mut [u64],
    shift: usize,
    bits: usize,
    requested_threads: usize,
) -> Vec<usize> {
    assert!(bits > 0 && bits <= 16);
    assert!(shift + bits <= u64::BITS as usize);
    assert_eq!(input.len(), output.len());
    if input.is_empty() {
        return vec![0; 1 << bits];
    }

    let buckets = 1usize << bits;
    let mask = buckets as u64 - 1;
    let threads = requested_threads.max(1).min(input.len());
    let histograms: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|worker| {
                let begin = input.len() * worker / threads;
                let end = input.len() * (worker + 1) / threads;
                scope.spawn(move || {
                    let mut histogram = vec![0usize; buckets];
                    for &value in &input[begin..end] {
                        histogram[((value >> shift) & mask) as usize] += 1;
                    }
                    histogram
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut totals = vec![0usize; buckets];
    for histogram in &histograms {
        for bucket in 0..buckets {
            totals[bucket] += histogram[bucket];
        }
    }

    let mut offsets = vec![vec![0usize; buckets]; threads];
    let mut bucket_begin = 0usize;
    for bucket in 0..buckets {
        let mut cursor = bucket_begin;
        for worker in 0..threads {
            offsets[worker][bucket] = cursor;
            cursor += histograms[worker][bucket];
        }
        bucket_begin = cursor;
    }

    let output_ptr = SharedMutPtr(output.as_mut_ptr());
    std::thread::scope(|scope| {
        let output_ptr = &output_ptr;
        for (worker, worker_offsets) in offsets.into_iter().enumerate() {
            let begin = input.len() * worker / threads;
            let end = input.len() * (worker + 1) / threads;
            scope.spawn(move || {
                let mut positions = worker_offsets;
                for &value in &input[begin..end] {
                    let bucket = ((value >> shift) & mask) as usize;
                    let position = positions[bucket];
                    unsafe { output_ptr.write(position, value) };
                    positions[bucket] += 1;
                }
            });
        }
    });

    totals
}

pub fn radix_sort_parallel(mut values: Vec<u64>, threads: usize) -> Vec<u64> {
    for pass in 0..PASSES {
        values = radix_pass_parallel(&values, pass, threads).0;
    }
    values
}

/// Digit width used by the distributed variants for their host-local sort.
/// Wider digits than the sequential baseline mean six passes instead of eight.
pub const LOCAL_RADIX_BITS: usize = 11;

/// Sort a host-local partition with alternating buffers.
///
/// The input is borrowed: the first pass scatters out of it into the first
/// owned buffer, and the remaining passes alternate between the two owned
/// buffers, so callers keep their partitioned buffer for the concurrent
/// exchange.
pub fn sort_local_partition(values: &[u64], threads: usize) -> Vec<u64> {
    if values.len() < 2 {
        return values.to_vec();
    }
    let mut front = vec![0u64; values.len()];
    let mut back = vec![0u64; values.len()];
    let mut shift = LOCAL_RADIX_BITS.min(u64::BITS as usize);
    radix_pass_into(values, &mut front, 0, shift, threads);
    while shift < u64::BITS as usize {
        let bits = LOCAL_RADIX_BITS.min(u64::BITS as usize - shift);
        radix_pass_into(&front, &mut back, shift, bits, threads);
        std::mem::swap(&mut front, &mut back);
        shift += bits;
    }
    front
}

/// Merge two sorted runs in parallel.
///
/// `left` is cut into equal pieces and each cut value's lower bound in `right`
/// gives the matching cut there, so the workers own disjoint, correctly ordered
/// output ranges and need no coordination. Every element of a piece is at least
/// the value at its own cut, which keeps the concatenation globally sorted.
pub fn merge_sorted(left: &[u64], right: &[u64], threads: usize) -> Vec<u64> {
    let mut merged = vec![0u64; left.len() + right.len()];
    if left.is_empty() || right.is_empty() {
        merged[..left.len()].copy_from_slice(left);
        merged[left.len()..].copy_from_slice(right);
        return merged;
    }

    let workers = threads.max(1).min(left.len());
    let bounds: Vec<(usize, usize)> = (0..=workers)
        .map(|worker| {
            let left_cut = left.len() * worker / workers;
            let right_cut = match worker {
                0 => 0,
                worker if worker == workers => right.len(),
                _ => right.partition_point(|&value| value < left[left_cut]),
            };
            (left_cut, right_cut)
        })
        .collect();

    let output = SharedMutPtr(merged.as_mut_ptr());
    std::thread::scope(|scope| {
        let output = &output;
        for worker in 0..workers {
            let (left_begin, right_begin) = bounds[worker];
            let (left_end, right_end) = bounds[worker + 1];
            scope.spawn(move || {
                let (mut i, mut j) = (left_begin, right_begin);
                let mut out = left_begin + right_begin;
                while i < left_end && j < right_end {
                    let value = if left[i] <= right[j] {
                        i += 1;
                        left[i - 1]
                    } else {
                        j += 1;
                        right[j - 1]
                    };
                    unsafe { output.write(out, value) };
                    out += 1;
                }
                for &value in &left[i..left_end] {
                    unsafe { output.write(out, value) };
                    out += 1;
                }
                for &value in &right[j..right_end] {
                    unsafe { output.write(out, value) };
                    out += 1;
                }
            });
        }
    });
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threaded_radix_matches_standard_sort() {
        let mut state = 0x1234_5678_9abc_def1u64;
        let values: Vec<u64> = (0..20_003)
            .map(|index| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if index % 17 == 0 { 42 } else { state }
            })
            .collect();
        let mut expected = values.clone();
        expected.sort_unstable();
        assert_eq!(radix_sort_parallel(values, 7), expected);
    }

    #[test]
    fn handles_empty_and_small_inputs() {
        assert_eq!(radix_sort_parallel(Vec::new(), 8), Vec::<u64>::new());
        assert_eq!(radix_sort_parallel(vec![3, 1, 2, 1], 16), vec![1, 1, 2, 3]);
    }

    #[test]
    fn wider_runtime_radix_matches_standard_sort() {
        let mut values = (0..50_003)
            .map(|index| {
                let value = (index as u64)
                    .wrapping_mul(6364136223846793005)
                    .rotate_left((index % 63) as u32);
                if index % 19 == 0 { 7 } else { value }
            })
            .collect::<Vec<_>>();
        let mut expected = values.clone();
        expected.sort_unstable();

        let mut shift = 0usize;
        while shift < u64::BITS as usize {
            let bits = 11usize.min(u64::BITS as usize - shift);
            values = radix_pass_parallel_bits(&values, shift, bits, 7).0;
            shift += bits;
        }
        assert_eq!(values, expected);
    }

    #[test]
    fn alternating_buffer_sort_matches_standard_sort() {
        for len in [0usize, 1, 2, 1_001, 50_003] {
            let values: Vec<u64> = (0..len)
                .map(|index| {
                    let value = (index as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left((index % 63) as u32);
                    if index % 23 == 0 { 11 } else { value }
                })
                .collect();
            let mut expected = values.clone();
            expected.sort_unstable();
            assert_eq!(sort_local_partition(&values, 7), expected);
        }
    }

    #[test]
    fn parallel_merge_matches_sequential_merge() {
        let left: Vec<u64> = {
            let mut values: Vec<u64> = (0..10_007).map(|index| (index * 7 % 977) as u64).collect();
            values.sort_unstable();
            values
        };
        let right: Vec<u64> = {
            let mut values: Vec<u64> = (0..7_919).map(|index| (index * 13 % 977) as u64).collect();
            values.sort_unstable();
            values
        };
        let mut expected = left.clone();
        expected.extend_from_slice(&right);
        expected.sort_unstable();

        for threads in [1usize, 2, 5, 16] {
            assert_eq!(merge_sorted(&left, &right, threads), expected);
        }
        assert_eq!(merge_sorted(&left, &[], 8), left);
        assert_eq!(merge_sorted(&[], &right, 8), right);
        assert_eq!(merge_sorted(&[], &[], 8), Vec::<u64>::new());
    }
}
