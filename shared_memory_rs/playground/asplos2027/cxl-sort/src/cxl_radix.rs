use crate::primitives::{
    ShmSegmentedU64Array, ShmU64Array, Transfer, alloc_u64_region, fetch_transfers,
    publish_transfers,
};
use crate::radix::{merge_sorted, sort_local_partition};
use crate::utils::sort_key_at;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use shm_cxl::utils::timestamps::CxlMetricTimer;
use std::time::Instant;

fn round_up8(value: usize) -> usize {
    value.div_ceil(8) * 8
}

/// Threads used for the bulk CXL copies. The link saturates well below the
/// host's core count, so the remaining cores stay with the overlapping local
/// sort instead of contending for the same write-combining path.
const CXL_COPY_THREADS: usize = 8;

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
fn key_owner(value: u64, hosts: usize) -> usize {
    (((value as u128) * (hosts as u128)) >> u64::BITS) as usize
}

fn displacements(counts: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(counts.len());
    let mut cursor = 0usize;
    for &count in counts {
        result.push(cursor);
        cursor += count;
    }
    result
}

/// Generate directly into a destination-major buffer for the single CXL
/// exchange. Recomputing deterministic keys avoids a second input allocation.
fn generate_partition_by_owner(
    seed: u64,
    input_begin: usize,
    input_len: usize,
    hosts: usize,
    requested_threads: usize,
) -> (Vec<u64>, Vec<usize>) {
    if input_len == 0 {
        return (Vec::new(), vec![0; hosts]);
    }
    let threads = requested_threads.max(1).min(input_len);
    let histograms: Vec<Vec<usize>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|worker| {
                let begin = input_len * worker / threads;
                let end = input_len * (worker + 1) / threads;
                scope.spawn(move || {
                    let mut counts = vec![0usize; hosts];
                    for offset in begin..end {
                        let value = sort_key_at((input_begin + offset) as u64, seed);
                        counts[key_owner(value, hosts)] += 1;
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

    let mut totals = vec![0usize; hosts];
    let mut offsets = vec![vec![0usize; hosts]; threads];
    let mut cursor = 0usize;
    for destination in 0..hosts {
        for worker in 0..threads {
            offsets[worker][destination] = cursor;
            let count = histograms[worker][destination];
            totals[destination] += count;
            cursor += count;
        }
    }

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
                    let destination = key_owner(value, hosts);
                    unsafe { output_ptr.write(positions[destination], value) };
                    positions[destination] += 1;
                }
            });
        }
    });
    (output, totals)
}

/// Cacheline-aligned start of every bucket a host writes, so that concurrent
/// writers and readers never share a line on non-coherent CXL 2.0.
fn padded_displacements(counts: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(counts.len());
    let mut cursor = 0usize;
    for &count in counts {
        result.push(cursor);
        cursor += round_up8(count);
    }
    result
}

/// Placement of every shared range the algorithm uses.
///
/// The staging allocation serves two successive phases. During the exchange
/// each host owns one region of `send_stride` u64s and writes its outgoing
/// buckets there, cacheline padded. After the exchange the same allocation
/// holds the final value-range partitions, again cacheline padded and ordered
/// by owning host, which is never larger than the exchange layout.
#[derive(Clone, Copy)]
struct StagingLayout {
    n: usize,
    nhosts: usize,
    block: usize,
}

impl StagingLayout {
    fn new(n: usize, nhosts: usize) -> Self {
        Self {
            n,
            nhosts,
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
    }

    fn from_block(n: usize, nhosts: usize, block: usize) -> Self {
        Self { n, nhosts, block }
    }

    fn capacity(&self) -> usize {
        (self.nhosts * self.send_stride()).max(1)
    }

    fn send_stride(&self) -> usize {
        self.block + self.nhosts * 8
    }

    /// Input index range a host generates. Ranges are clamped to `n`, so the
    /// final host's block needs no padding.
    fn owned_range(&self, host: usize) -> (usize, usize) {
        let begin = (host * self.block).min(self.n);
        let end = ((host + 1) * self.block).min(self.n);
        (begin, end)
    }

    /// Where `host` writes the bucket it sends to `destination`, given that
    /// host's row of the shared count matrix.
    fn send_begin(&self, host: usize, row: &[usize], destination: usize) -> usize {
        host * self.send_stride() + padded_displacements(row)[destination]
    }

    /// Where `host` writes its final sorted value range.
    fn output_begin(&self, destination_totals: &[usize], host: usize) -> usize {
        destination_totals[..host]
            .iter()
            .map(|&count| round_up8(count))
            .sum()
    }
}

/// Row `sender`, column `destination` of the shared send-count matrix, padded
/// so each host owns whole cachelines of it.
fn read_count_matrix(histograms: &ShmU64Array, nhosts: usize) -> Vec<usize> {
    let count_stride = round_up8(nhosts);
    histograms.invalidate_range(nhosts * count_stride);
    (0..nhosts * count_stride)
        .map(|index| histograms.load_local(index) as usize)
        .collect()
}

fn destination_totals(all_counts: &[usize], nhosts: usize) -> Vec<usize> {
    let count_stride = round_up8(nhosts);
    (0..nhosts)
        .map(|destination| {
            (0..nhosts)
                .map(|sender| all_counts[sender * count_stride + destination])
                .sum()
        })
        .collect()
}

pub struct CxlRadixSort {
    staging: ShmSegmentedU64Array,
    histograms: ShmU64Array,
    layout: StagingLayout,
    n: usize,
    nhosts: usize,
    myid: u16,
    seed: u64,
}

unsafe impl Send for CxlRadixSort {}
unsafe impl Sync for CxlRadixSort {}

impl CxlRadixSort {
    pub fn create<A: Allocator>(alloc: &A, n: usize, seed: u64) -> (Self, Cacheline) {
        let nhosts = get_machine_cnt() as usize;
        let layout = StagingLayout::new(n, nhosts);
        // The staging array is segmented by its wrapper to stay within the
        // bitmap allocator's per-allocation limit.
        let (staging, staging_table) =
            ShmSegmentedU64Array::create_uninit(alloc, layout.capacity());
        let histograms = ShmU64Array::from_cl(alloc_u64_region(alloc, nhosts * round_up8(nhosts)));

        let descriptor = Cacheline {
            d: [
                staging_table,
                histograms.offset(),
                n as u64,
                nhosts as u64,
                seed,
                layout.block as u64,
                0,
                0,
            ],
        };
        (
            Self {
                staging,
                histograms,
                layout,
                n,
                nhosts,
                myid: get_machine_id(),
                seed,
            },
            descriptor,
        )
    }

    pub fn attach(descriptor: &Cacheline) -> Self {
        let n = descriptor.d[2] as usize;
        let nhosts = descriptor.d[3] as usize;
        Self {
            staging: ShmSegmentedU64Array::attach(descriptor.d[0]),
            histograms: ShmU64Array::from_offset(descriptor.d[1]),
            layout: StagingLayout::from_block(n, nhosts, descriptor.d[5] as usize),
            n,
            nhosts,
            myid: get_machine_id(),
            seed: descriptor.d[4],
        }
    }

    fn count_matrix(&self) -> Vec<usize> {
        read_count_matrix(&self.histograms, self.nhosts)
    }

    fn destination_totals(&self, all_counts: &[usize]) -> Vec<usize> {
        destination_totals(all_counts, self.nhosts)
    }

    /// Sort this host's value range and return it in host-local memory.
    ///
    /// The kernel ends with the result distributed across the hosts; making it
    /// readable by everyone is [`publish`]'s job.
    ///
    /// [`publish`]: CxlRadixSort::publish
    pub fn run(&self, threads: usize) -> Vec<u64> {
        let me = (self.myid - 1) as usize;
        let (owned_begin, owned_end) = self.layout.owned_range(me);
        let owned_len = owned_end - owned_begin;

        let partition_started = Instant::now();
        let count_stride = round_up8(self.nhosts);
        let (ordered, send_counts) =
            generate_partition_by_owner(self.seed, owned_begin, owned_len, self.nhosts, threads);
        let send_displacements = displacements(&send_counts);
        let local_begin = send_displacements[me];
        // Sorted straight out of the partitioned buffer: the publisher only
        // reads the other buckets, so both borrows are shared.
        let local_values = &ordered[local_begin..local_begin + send_counts[me]];
        let partition_done = Instant::now();

        let copy_threads = CXL_COPY_THREADS.min(threads.max(1));
        let sort_threads = threads.saturating_sub(copy_threads).max(1);

        // Sort the host-local value range concurrently with publishing and
        // reading the remote ranges. Only peer-owned buckets touch CXL.
        let (local_sorted, remote_values, all_counts, exchange_done) =
            std::thread::scope(|scope| {
                let local_sort =
                    scope.spawn(move || sort_local_partition(local_values, sort_threads));

                {
                    let _store_timer = CxlMetricTimer::store();
                    let outgoing: Vec<Transfer> = (0..self.nhosts)
                        .filter(|&destination| destination != me && send_counts[destination] != 0)
                        .map(|destination| Transfer {
                            shared_begin: self.layout.send_begin(me, &send_counts, destination),
                            local_begin: send_displacements[destination],
                            len: send_counts[destination],
                        })
                        .collect();
                    publish_transfers(&self.staging, &ordered, &outgoing, copy_threads);
                    let counts_begin = me * count_stride;
                    for (destination, &count) in send_counts.iter().enumerate() {
                        self.histograms
                            .store_local(counts_begin + destination, count as u64);
                    }
                    self.histograms.flush_slice(counts_begin, self.nhosts);
                }
                barrier_wait_all();

                let all_counts = {
                    let _load_timer = CxlMetricTimer::load();
                    self.count_matrix()
                };
                let mut incoming = Vec::with_capacity(self.nhosts);
                let mut remote_len = 0usize;
                for sender in 0..self.nhosts {
                    if sender == me {
                        continue;
                    }
                    let row =
                        &all_counts[sender * count_stride..sender * count_stride + self.nhosts];
                    let count = row[me];
                    if count == 0 {
                        continue;
                    }
                    incoming.push(Transfer {
                        shared_begin: self.layout.send_begin(sender, row, me),
                        local_begin: remote_len,
                        len: count,
                    });
                    remote_len += count;
                }
                let mut remote_values = vec![0u64; remote_len];
                {
                    let _load_timer = CxlMetricTimer::load();
                    fetch_transfers(&self.staging, &mut remote_values, &incoming, copy_threads);
                }
                // All remote reads must finish before any host reuses staging
                // for its final sorted value-range partition.
                barrier_wait_all();
                let local_sorted = local_sort.join().unwrap();
                (local_sorted, remote_values, all_counts, Instant::now())
            });
        drop(ordered);

        let remote_sorted = sort_local_partition(&remote_values, threads);
        let local = merge_sorted(&local_sorted, &remote_sorted, threads);
        debug_assert_eq!(
            self.destination_totals(&all_counts).iter().sum::<usize>(),
            self.n
        );
        debug_assert_eq!(local.len(), self.destination_totals(&all_counts)[me]);
        println!(
            "[cxl_radix_phase] host {}/{} partition={:?} exchange={:?} remote_sort_merge={:?}",
            self.myid,
            self.nhosts,
            partition_done.duration_since(partition_started),
            exchange_done.duration_since(partition_done),
            Instant::now().duration_since(exchange_done),
        );
        local
    }

    /// Make this host's sorted value range readable by every host.
    ///
    /// The distributed result exists once [`run`] returns; this is the gather
    /// step on top of it. Output ranges are disjoint, cacheline padded and
    /// ordered by value owner.
    ///
    /// [`run`]: CxlRadixSort::run
    pub fn publish(&self, local: &[u64], threads: usize) {
        let _store_timer = CxlMetricTimer::store();
        let me = (self.myid - 1) as usize;
        let all_counts = self.count_matrix();
        let output_begin = self
            .layout
            .output_begin(&self.destination_totals(&all_counts), me);
        let transfer = [Transfer {
            shared_begin: output_begin,
            local_begin: 0,
            len: local.len(),
        }];
        publish_transfers(
            &self.staging,
            local,
            &transfer,
            CXL_COPY_THREADS.min(threads.max(1)),
        );
    }

    pub fn collect(&self) -> Vec<u64> {
        let _load_timer = CxlMetricTimer::load();
        let destination_totals = self.destination_totals(&self.count_matrix());
        let mut result = vec![0u64; destination_totals.iter().sum()];
        let mut shared_begin = 0usize;
        let mut local_begin = 0usize;
        let mut transfers = Vec::with_capacity(destination_totals.len());
        for count in destination_totals {
            transfers.push(Transfer {
                shared_begin,
                local_begin,
                len: count,
            });
            shared_begin += round_up8(count);
            local_begin += count;
        }
        fetch_transfers(&self.staging, &mut result, &transfers, CXL_COPY_THREADS);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_partition_is_destination_major_and_complete() {
        let seed = 0x1234;
        let begin = 113;
        let len = 50_003;
        let (partitioned, counts) = generate_partition_by_owner(seed, begin, len, 3, 7);
        assert_eq!(counts.iter().sum::<usize>(), len);
        let offsets = displacements(&counts);
        for owner in 0..3 {
            assert!(
                partitioned[offsets[owner]..offsets[owner] + counts[owner]]
                    .iter()
                    .all(|&value| key_owner(value, 3) == owner)
            );
        }
        let mut actual = partitioned;
        actual.sort_unstable();
        let mut expected: Vec<_> = (begin..begin + len)
            .map(|index| sort_key_at(index as u64, seed))
            .collect();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }

    /// Counts of a plausible exchange: `matrix[sender][destination]`.
    fn count_matrix(n: usize, nhosts: usize) -> Vec<Vec<usize>> {
        let layout = StagingLayout::new(n, nhosts);
        (0..nhosts)
            .map(|sender| {
                let (begin, end) = layout.owned_range(sender);
                let mut row = vec![0usize; nhosts];
                for index in begin..end {
                    row[key_owner(sort_key_at(index as u64, 0xC0FFEE), nhosts)] += 1;
                }
                row
            })
            .collect()
    }

    /// Both phases must stay inside the allocation, keep every host on its own
    /// cachelines, and reassemble into exactly `n` keys in value order.
    #[test]
    fn staging_ranges_are_disjoint_aligned_and_complete() {
        for (n, nhosts) in [
            (1usize, 2usize),
            (17, 3),
            (1_000, 2),
            (50_003, 3),
            (65_536, 5),
        ] {
            let layout = StagingLayout::new(n, nhosts);
            let matrix = count_matrix(n, nhosts);
            assert_eq!(
                matrix.iter().flatten().sum::<usize>(),
                n,
                "the exchange must route every key"
            );

            let mut occupied = vec![None; layout.capacity()];
            for (sender, row) in matrix.iter().enumerate() {
                for destination in 0..nhosts {
                    let begin = layout.send_begin(sender, row, destination);
                    assert_eq!(begin % 8, 0, "buckets start on a cacheline");
                    for index in begin..begin + row[destination] {
                        assert!(occupied[index].is_none(), "exchange ranges overlap");
                        occupied[index] = Some(sender);
                    }
                }
            }

            let destination_totals: Vec<usize> = (0..nhosts)
                .map(|destination| matrix.iter().map(|row| row[destination]).sum())
                .collect();
            let mut published = 0usize;
            for host in 0..nhosts {
                let begin = layout.output_begin(&destination_totals, host);
                assert_eq!(begin % 8, 0, "output partitions start on a cacheline");
                assert_eq!(begin, published, "the gather leaves no unread gap");
                assert!(
                    begin + destination_totals[host] <= layout.capacity(),
                    "the gather must fit the exchange allocation"
                );
                published += round_up8(destination_totals[host]);
            }
            assert_eq!(destination_totals.iter().sum::<usize>(), n);
        }
    }

    #[test]
    fn padded_bucket_starts_are_cacheline_aligned_and_ordered() {
        let counts = vec![3usize, 0, 17, 8, 1];
        let offsets = padded_displacements(&counts);
        assert!(offsets.iter().all(|offset| offset % 8 == 0));
        for index in 1..counts.len() {
            assert!(offsets[index] >= offsets[index - 1] + counts[index - 1]);
        }
        // The padding a host's region must reserve stays within `nhosts * 8`.
        let padded_total = offsets.last().unwrap() + round_up8(*counts.last().unwrap());
        assert!(padded_total <= counts.iter().sum::<usize>() + 8 * counts.len());
    }
}
