use crate::radix::{merge_sorted, sort_local_partition};
#[cfg(test)]
use crate::utils::gen_keys_range;
use crate::utils::{Fingerprint, sort_key_at};
use mpi::Count;
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use shm_cxl::utils::timestamps::{IpCxlMetricsGuard, print_bench_metrics_value};
use std::time::{Duration, Instant};

pub struct MpiRadixResult {
    pub rank: usize,
    pub ranks: usize,
    pub kernel: Duration,
    pub validation: Option<bool>,
    pub total_keys: usize,
}

struct SharedMutPtr(*mut u64);

unsafe impl Send for SharedMutPtr {}
unsafe impl Sync for SharedMutPtr {}

impl SharedMutPtr {
    #[inline]
    unsafe fn write(&self, index: usize, value: u64) {
        unsafe { *self.0.add(index) = value };
    }
}

fn displacements(counts: &[Count]) -> Vec<Count> {
    let mut result = Vec::with_capacity(counts.len());
    let mut cursor = 0 as Count;
    for &count in counts {
        result.push(cursor);
        cursor += count;
    }
    result
}

fn rank_range(rank: usize, block: usize, n: usize) -> (usize, usize) {
    ((rank * block).min(n), ((rank + 1) * block).min(n))
}

/// Map the complete `u64` key space onto equal, contiguous rank ranges.
/// Multiplication avoids the overflow and uneven final range of
/// `value / (u64::MAX / ranks)`.
#[inline]
fn key_owner(value: u64, ranks: usize) -> usize {
    (((value as u128) * (ranks as u128)) >> u64::BITS) as usize
}

/// Arrange keys destination-major for one Alltoallv. Counting and scattering
/// are threaded and use disjoint output offsets, avoiding per-destination
/// vectors and the additional flattening copy they require.
#[cfg(test)]
fn partition_by_owner(
    input: &[u64],
    ranks: usize,
    requested_threads: usize,
) -> (Vec<u64>, Vec<Count>) {
    if input.is_empty() {
        return (Vec::new(), vec![0 as Count; ranks]);
    }

    let threads = requested_threads.max(1).min(input.len());
    let histograms: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|worker| {
                let begin = input.len() * worker / threads;
                let end = input.len() * (worker + 1) / threads;
                scope.spawn(move || {
                    let mut counts = vec![0usize; ranks];
                    for &value in &input[begin..end] {
                        counts[key_owner(value, ranks)] += 1;
                    }
                    counts
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut totals = vec![0usize; ranks];
    let mut offsets = vec![vec![0usize; ranks]; threads];
    let mut cursor = 0usize;
    for destination in 0..ranks {
        for worker in 0..threads {
            offsets[worker][destination] = cursor;
            let count = histograms[worker][destination];
            totals[destination] += count;
            cursor += count;
        }
    }
    debug_assert_eq!(cursor, input.len());

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
                    let destination = key_owner(value, ranks);
                    let position = positions[destination];
                    unsafe { output_ptr.write(position, value) };
                    positions[destination] += 1;
                }
            });
        }
    });

    let counts = totals
        .into_iter()
        .map(|count| Count::try_from(count).expect("MPI send count overflow"))
        .collect();
    (output, counts)
}

/// Generate directly into a destination-major MPI buffer. Recomputing the
/// inexpensive deterministic key for the scatter pass avoids materializing a
/// second, gigabyte-scale input vector and then reading it twice.
fn generate_partition_by_owner(
    seed: u64,
    input_begin: usize,
    input_len: usize,
    ranks: usize,
    requested_threads: usize,
) -> (Vec<u64>, Vec<Count>) {
    if input_len == 0 {
        return (Vec::new(), vec![0 as Count; ranks]);
    }

    let threads = requested_threads.max(1).min(input_len);
    let histograms: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|worker| {
                let begin = input_len * worker / threads;
                let end = input_len * (worker + 1) / threads;
                scope.spawn(move || {
                    let mut counts = vec![0usize; ranks];
                    for offset in begin..end {
                        let value = sort_key_at((input_begin + offset) as u64, seed);
                        counts[key_owner(value, ranks)] += 1;
                    }
                    counts
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut totals = vec![0usize; ranks];
    let mut offsets = vec![vec![0usize; ranks]; threads];
    let mut cursor = 0usize;
    for destination in 0..ranks {
        for worker in 0..threads {
            offsets[worker][destination] = cursor;
            let count = histograms[worker][destination];
            totals[destination] += count;
            cursor += count;
        }
    }
    debug_assert_eq!(cursor, input_len);

    let mut output = vec![0u64; input_len];
    let output_ptr = SharedMutPtr(output.as_mut_ptr());
    std::thread::scope(|scope| {
        let output_ptr = &output_ptr;
        for (worker, worker_offsets) in offsets.into_iter().enumerate() {
            let begin = input_len * worker / threads;
            let end = input_len * (worker + 1) / threads;
            scope.spawn(move || {
                let mut positions = worker_offsets;
                for offset in begin..end {
                    let value = sort_key_at((input_begin + offset) as u64, seed);
                    let destination = key_owner(value, ranks);
                    let position = positions[destination];
                    unsafe { output_ptr.write(position, value) };
                    positions[destination] += 1;
                }
            });
        }
    });

    let counts = totals
        .into_iter()
        .map(|count| Count::try_from(count).expect("MPI send count overflow"))
        .collect();
    (output, counts)
}

/// Distributed range-partitioned radix sort.
///
/// Every rank initially generates an equal index range. Keys are then routed
/// once to the rank owning their contiguous value range, and each received
/// partition is radix-sorted in local DRAM. Concatenating rank order is the
/// complete sorted sequence, so the keys cross the network once rather than
/// once per radix pass.
pub fn run_mpi_radix(n: usize, seed: u64, threads: usize) -> MpiRadixResult {
    let universe = mpi::initialize().expect("MPI initialize failed");
    let world = universe.world();
    let rank = world.rank() as usize;
    let ranks = world.size() as usize;
    let block = n.div_ceil(ranks).max(1);
    let (owned_begin, owned_end) = rank_range(rank, block, n);

    world.barrier();
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    let start = Instant::now();

    let (send_buffer, send_counts) =
        generate_partition_by_owner(seed, owned_begin, owned_end - owned_begin, ranks, threads);
    let partition_done = Instant::now();

    let send_displacements = displacements(&send_counts);
    let local_begin = send_displacements[rank] as usize;
    let local_len = send_counts[rank] as usize;
    let mut exchange_send_counts = send_counts;
    // The rank-local bucket is already in DRAM. Excluding it from Alltoallv
    // avoids a large implementation-dependent self-copy through MPI.
    exchange_send_counts[rank] = 0;
    let mut receive_counts = vec![0 as Count; ranks];
    world.all_to_all_into(&exchange_send_counts[..], &mut receive_counts[..]);
    let receive_displacements = displacements(&receive_counts);
    let receive_len: usize = receive_counts.iter().map(|&count| count as usize).sum();
    let mut receive_buffer = vec![0u64; receive_len];
    // Sort the local bucket while the main thread blocks in the network
    // exchange, hiding a substantial fraction of the local work behind the
    // transfer. Both the sorter and Alltoallv only read `send_buffer`.
    let (local_sorted, local_sort_elapsed, exchange_done) = std::thread::scope(|scope| {
        let local_values = &send_buffer[local_begin..local_begin + local_len];
        let local_sort = scope.spawn(move || {
            let started = Instant::now();
            let sorted = sort_local_partition(local_values, threads);
            (sorted, started.elapsed())
        });
        {
            let send_partition =
                Partition::new(&send_buffer[..], exchange_send_counts, send_displacements);
            let mut receive_partition = PartitionMut::new(
                &mut receive_buffer[..],
                receive_counts,
                receive_displacements,
            );
            world.all_to_all_varcount_into(&send_partition, &mut receive_partition);
        }
        let exchange_done = Instant::now();
        let (sorted, elapsed) = local_sort.join().unwrap();
        (sorted, elapsed, exchange_done)
    });
    drop(send_buffer);

    let remote_sort_started = Instant::now();
    let remote_sorted = sort_local_partition(&receive_buffer, threads);
    let local = merge_sorted(&local_sorted, &remote_sorted, threads);
    let remote_sort_merge_elapsed = remote_sort_started.elapsed();

    world.barrier();
    let kernel = start.elapsed();
    if let Some(metrics) = ip_cxl_metrics.finish() {
        print_bench_metrics_value(metrics);
    }
    println!(
        "[mpi_radix_phase] rank {rank}/{ranks} partition={:?} exchange={:?} hidden_local_sort={local_sort_elapsed:?} remote_sort_merge={remote_sort_merge_elapsed:?}",
        partition_done.duration_since(start),
        exchange_done.duration_since(partition_done),
    );

    // Validate without gathering the complete output back over the network.
    // This remains outside BENCH_KERNEL_NS. Ordering is checked locally and at
    // every rank boundary; count, sum, xor and an independent mixed-hash sum
    // establish that the distributed output is the generated input multiset.
    let local_sorted = u64::from(local.windows(2).all(|pair| pair[0] <= pair[1]));
    let mut globally_sorted = 0u64;
    world.all_reduce_into(&local_sorted, &mut globally_sorted, SystemOperation::min());
    let bounds = [
        u64::from(!local.is_empty()),
        local.first().copied().unwrap_or(0),
        local.last().copied().unwrap_or(0),
    ];
    let mut all_bounds = vec![0u64; ranks * bounds.len()];
    world.all_gather_into(&bounds[..], &mut all_bounds[..]);
    let mut previous_max = None;
    for other in 0..ranks {
        let fields = &all_bounds[other * 3..other * 3 + 3];
        if fields[0] == 0 {
            continue;
        }
        if previous_max.is_some_and(|maximum| maximum > fields[1]) {
            globally_sorted = 0;
        }
        previous_max = Some(fields[2]);
    }

    let local_input =
        Fingerprint::of((owned_begin..owned_end).map(|index| sort_key_at(index as u64, seed)));
    let local_output = Fingerprint::of(local.iter().copied());
    let (local_input_sums, local_input_xor) = (local_input.sums(), local_input.xor);
    let (local_output_sums, local_output_xor) = (local_output.sums(), local_output.xor);
    let mut input_sums = [0u64; 3];
    let mut output_sums = [0u64; 3];
    let mut input_xor = 0u64;
    let mut output_xor = 0u64;
    world.all_reduce_into(
        &local_input_sums[..],
        &mut input_sums[..],
        SystemOperation::sum(),
    );
    world.all_reduce_into(
        &local_output_sums[..],
        &mut output_sums[..],
        SystemOperation::sum(),
    );
    world.all_reduce_into(
        &local_input_xor,
        &mut input_xor,
        SystemOperation::bitwise_xor(),
    );
    world.all_reduce_into(
        &local_output_xor,
        &mut output_xor,
        SystemOperation::bitwise_xor(),
    );
    let valid = globally_sorted != 0
        && input_sums == output_sums
        && input_xor == output_xor
        && output_sums[0] == n as u64;

    MpiRadixResult {
        rank,
        ranks,
        kernel,
        validation: (rank == 0).then_some(valid),
        total_keys: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_ranges_are_monotonic_and_bounded() {
        for ranks in 1..=17 {
            let samples = [0, 1, u64::MAX / 3, u64::MAX / 2, u64::MAX - 1, u64::MAX];
            let owners: Vec<_> = samples
                .into_iter()
                .map(|value| key_owner(value, ranks))
                .collect();
            assert!(owners.windows(2).all(|pair| pair[0] <= pair[1]));
            assert!(owners.iter().all(|&owner| owner < ranks));
            assert_eq!(owners[0], 0);
            assert_eq!(*owners.last().unwrap(), ranks - 1);
        }
    }

    #[test]
    fn partition_is_destination_major_and_preserves_keys() {
        let input = gen_keys_range(0x1234, 0, 50_003);
        let (partitioned, counts) = partition_by_owner(&input, 3, 7);
        let mut cursor = 0usize;
        for (owner, count) in counts.into_iter().enumerate() {
            let end = cursor + count as usize;
            assert!(
                partitioned[cursor..end]
                    .iter()
                    .all(|&value| key_owner(value, 3) == owner)
            );
            cursor = end;
        }
        assert_eq!(cursor, input.len());
        let mut expected = input;
        let mut actual = partitioned;
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    #[test]
    fn generated_partition_matches_materialized_partition() {
        let input = gen_keys_range(0x2345, 113, 50_003);
        let (materialized, materialized_counts) = partition_by_owner(&input, 3, 7);
        let (generated, generated_counts) = generate_partition_by_owner(0x2345, 113, 50_003, 3, 7);
        assert_eq!(generated_counts, materialized_counts);
        for owner in 0..3 {
            let begin = materialized_counts[..owner]
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>();
            let end = begin + materialized_counts[owner] as usize;
            let mut expected = materialized[begin..end].to_vec();
            let mut actual = generated[begin..end].to_vec();
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn shared_sort_and_merge_reproduce_the_rank_pipeline() {
        let values = gen_keys_range(0x5678, 0, 50_003);
        let mut expected = values.clone();
        expected.sort_unstable();
        let (left, right) = values.split_at(20_011);
        let left = sort_local_partition(left, 7);
        let right = sort_local_partition(right, 7);
        assert_eq!(merge_sorted(&left, &right, 7), expected);
    }
}
