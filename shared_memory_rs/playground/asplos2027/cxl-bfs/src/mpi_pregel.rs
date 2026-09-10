use crate::primitives::UNVISITED;
use crate::utils::Csr;
use mpi::Count;
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::*;
use shm_cxl::utils::timestamps::{IpCxlMetricsGuard, print_bench_metrics_value};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

const CYCLIC_OWNER_CHUNK: usize = 1024;
const CYCLIC_MIN_AVERAGE_DEGREE: usize = 8;
const EDGES_PER_THREAD: usize = 32768;

/// Dense supersteps exchange the frontier as an n-bit bitmap per rank
/// (Allgather) instead of u64 id lists (Alltoallv). Per-rank dedup still
/// lets every rank propose the same vertex, so list volume grows with the
/// rank count while the 1 GbE pipe does not -- that is exactly what made
/// np3 slower than np2 on the large graphs. The bitmap caps a superstep's
/// exchange at ranks * n/8 bytes regardless of how many ranks propose each
/// vertex. Sparse supersteps (e.g. every level of a road graph) keep the
/// id-list path. Both sides of the switch derive it from the count matrix,
/// which every rank already holds, so the collective schedule stays agreed.
fn use_bitmap_exchange(total_remote: usize, n: usize, ranks: usize) -> bool {
    total_remote * 8 > (ranks - 1) * n.div_ceil(64) * 8
}

/// Inline-proposal slots in the sweep engine's per-round Allgather block.
const SWEEP_INLINE: usize = 64;

pub struct MpiPregelResult {
    pub rank: usize,
    pub ranks: usize,
    pub kernel: Duration,
    pub distances: Option<Vec<u64>>,
}

enum Ownership {
    /// Per-vertex owner table cut from a BFS visit order of the graph, so
    /// each rank owns a connected, locality-tight region. Vertex ids in road
    /// graphs interleave geographically, and id-contiguous cuts made the
    /// BFS wavefront cross partition boundaries thousands of times; region
    /// partitions bring the sweep engine's exchange rounds down to roughly
    /// the number of regions a shortest path traverses. Every rank derives
    /// the identical table from the CSR before the kernel clock starts.
    Locality(Vec<u8>),
    Cyclic,
}

impl Ownership {
    fn new(csr: &Csr, ranks: usize) -> Self {
        let n = csr.row_ptr.len() - 1;
        let edges = csr.row_ptr[n] as usize;
        if edges >= n.saturating_mul(CYCLIC_MIN_AVERAGE_DEGREE) {
            return Self::Cyclic;
        }
        assert!(ranks <= u8::MAX as usize + 1, "owner table stores u8 ranks");

        // Deterministic BFS visit order over all components, then cut it
        // into `ranks` edge-balanced blocks.
        let mut order = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut queue = std::collections::VecDeque::new();
        for root in 0..n {
            if seen[root] {
                continue;
            }
            seen[root] = true;
            queue.push_back(root);
            while let Some(u) = queue.pop_front() {
                order.push(u);
                let row_begin = csr.row_ptr[u] as usize;
                let row_end = csr.row_ptr[u + 1] as usize;
                for &v in &csr.col_idx[row_begin..row_end] {
                    if !seen[v as usize] {
                        seen[v as usize] = true;
                        queue.push_back(v as usize);
                    }
                }
            }
        }

        let mut table = vec![0u8; n];
        let mut accumulated = 0usize;
        let mut rank = 0usize;
        for &u in &order {
            while rank + 1 < ranks && accumulated * ranks >= edges * (rank + 1) {
                rank += 1;
            }
            table[u] = rank as u8;
            accumulated += (csr.row_ptr[u + 1] - csr.row_ptr[u]) as usize;
        }
        Self::Locality(table)
    }

    #[inline]
    fn owner(&self, vertex: u64, ranks: usize) -> usize {
        let vertex = vertex as usize;
        match self {
            Self::Cyclic => (vertex / CYCLIC_OWNER_CHUNK) % ranks,
            Self::Locality(table) => table[vertex] as usize,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Locality(_) => "bfs-locality",
            Self::Cyclic => "cyclic-1024",
        }
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

fn expand_parallel(
    csr: &Csr,
    active: &[u64],
    ownership: &Ownership,
    ranks: usize,
    requested_threads: usize,
    sent: &[AtomicBool],
) -> Vec<Vec<u64>> {
    if active.is_empty() {
        return vec![Vec::new(); ranks];
    }
    let edges: usize = active
        .iter()
        .map(|&vertex| (csr.row_ptr[vertex as usize + 1] - csr.row_ptr[vertex as usize]) as usize)
        .sum();
    let threads = requested_threads
        .max(1)
        .min(edges / EDGES_PER_THREAD + 1)
        .min(active.len());
    let chunk = active.len().div_ceil(threads * 4).max(16);
    let cursor = AtomicUsize::new(0);

    let parts: Vec<Vec<Vec<u64>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut outgoing = vec![Vec::<u64>::new(); ranks];
                    loop {
                        let begin = cursor.fetch_add(chunk, Ordering::Relaxed);
                        if begin >= active.len() {
                            break;
                        }
                        let end = (begin + chunk).min(active.len());
                        for &vertex in &active[begin..end] {
                            let row_begin = csr.row_ptr[vertex as usize] as usize;
                            let row_end = csr.row_ptr[vertex as usize + 1] as usize;
                            for &neighbour in &csr.col_idx[row_begin..row_end] {
                                // A rank only needs to propose a vertex once. The
                                // first proposal is necessarily at its earliest
                                // local superstep; the owner filters duplicates
                                // proposed by other ranks after the exchange.
                                if sent[neighbour as usize]
                                    .compare_exchange(
                                        false,
                                        true,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    outgoing[ownership.owner(neighbour, ranks)].push(neighbour);
                                }
                            }
                        }
                    }
                    outgoing
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    let mut outgoing = vec![Vec::<u64>::new(); ranks];
    for part in parts {
        for (destination, messages) in part.into_iter().enumerate() {
            outgoing[destination].extend(messages);
        }
    }
    outgoing
}

/// Label-correcting BFS for contiguous edge-balanced partitions (road-like
/// graphs). Each rank drains its local frontier to exhaustion in distance
/// order with a bucket queue, so the collectives cost one round per
/// partition-boundary crossing on shortest paths instead of one per BFS
/// level -- a ~6200-level road graph over a ~50 us/collective wire needs
/// tens of rounds, not thousands. A relaxation is applied only when it
/// improves a distance, so the fixpoint is exactly the BFS distances.
fn sweep_bfs(
    world: &mpi::topology::SimpleCommunicator,
    csr: &Csr,
    source: u64,
    owner_table: &[u8],
    rank: usize,
    ranks: usize,
    distances: &mut [u64],
) {
    let me = rank as u8;
    let n = distances.len();
    debug_assert!(n < (1 << 32), "sweep packs (distance, vertex) into one u64");

    // Best distance already proposed per remote vertex; only improvements are
    // re-sent, so wire volume is bounded by improvements, not by sweeps.
    let mut sent_best = vec![UNVISITED; n];
    let mut buckets: Vec<Vec<u64>> = Vec::new();
    let mut cursor = 0usize;
    if owner_table[source as usize] == me {
        // distances[source] was set by the caller.
        buckets.push(vec![source]);
    }

    loop {
        let mut outgoing: Vec<Vec<u64>> = vec![Vec::new(); ranks];
        while cursor < buckets.len() {
            if buckets[cursor].is_empty() {
                cursor += 1;
                continue;
            }
            let bucket = std::mem::take(&mut buckets[cursor]);
            let distance = cursor as u64;
            for vertex in bucket {
                if distances[vertex as usize] != distance {
                    continue; // improved since it was queued
                }
                let next = distance + 1;
                let row_begin = csr.row_ptr[vertex as usize] as usize;
                let row_end = csr.row_ptr[vertex as usize + 1] as usize;
                for &neighbour in &csr.col_idx[row_begin..row_end] {
                    let v = neighbour as usize;
                    if owner_table[v] == me {
                        if distances[v] > next {
                            distances[v] = next;
                            let slot = next as usize;
                            if buckets.len() <= slot {
                                buckets.resize(slot + 1, Vec::new());
                            }
                            buckets[slot].push(neighbour);
                        }
                    } else if sent_best[v] > next {
                        sent_best[v] = next;
                        outgoing[owner_table[v] as usize].push((next << 32) | neighbour);
                    }
                }
            }
        }

        // The round latency, not the payload, dominates on road graphs, so
        // each round is a single fixed-size Allgather: every rank's block
        // carries its per-destination counts plus up to SWEEP_INLINE inline
        // proposals. That covers termination, receive counts, and (for the
        // typical few-proposal round) the data itself. Only when some rank
        // overflows the inline slots does an Alltoallv follow, sized from
        // the counts already gathered.
        let block_len = ranks + SWEEP_INLINE;
        let mut block = vec![0u64; block_len];
        let mut total = 0usize;
        for (destination, messages) in outgoing.iter().enumerate() {
            block[destination] = messages.len() as u64;
            total += messages.len();
        }
        if total <= SWEEP_INLINE {
            let mut slot = ranks;
            for messages in &outgoing {
                for &packed in messages {
                    block[slot] = packed;
                    slot += 1;
                }
            }
        }
        let mut all = vec![0u64; block_len * ranks];
        world.all_gather_into(&block[..], &mut all[..]);

        let totals: Vec<usize> = (0..ranks)
            .map(|source| {
                (0..ranks)
                    .map(|destination| all[source * block_len + destination] as usize)
                    .sum()
            })
            .collect();
        if totals.iter().all(|&count| count == 0) {
            break;
        }

        let mut apply = |packed: u64, buckets: &mut Vec<Vec<u64>>, cursor: &mut usize| {
            let vertex = packed & 0xFFFF_FFFF;
            let proposal = packed >> 32;
            let v = vertex as usize;
            if owner_table[v] == me && distances[v] > proposal {
                distances[v] = proposal;
                let slot = proposal as usize;
                if buckets.len() <= slot {
                    buckets.resize(slot + 1, Vec::new());
                }
                buckets[slot].push(vertex);
                *cursor = (*cursor).min(slot);
            }
        };

        if totals.iter().all(|&count| count <= SWEEP_INLINE) {
            for source in 0..ranks {
                if source == rank {
                    continue;
                }
                let base = source * block_len + ranks;
                for offset in 0..totals[source] {
                    apply(all[base + offset], &mut buckets, &mut cursor);
                }
            }
            continue;
        }

        let send_counts: Vec<Count> = outgoing
            .iter()
            .map(|messages| Count::try_from(messages.len()).expect("MPI send count overflow"))
            .collect();
        let send_displacements = displacements(&send_counts);
        let send_buffer: Vec<u64> = outgoing.into_iter().flatten().collect();
        let receive_counts: Vec<Count> = (0..ranks)
            .map(|source| {
                Count::try_from(all[source * block_len + rank]).expect("MPI recv count overflow")
            })
            .collect();
        let receive_displacements = displacements(&receive_counts);
        let receive_len: usize = receive_counts.iter().map(|&count| count as usize).sum();
        let mut receive_buffer = vec![0u64; receive_len];
        {
            let send_partition = Partition::new(&send_buffer[..], send_counts, send_displacements);
            let mut receive_partition = PartitionMut::new(
                &mut receive_buffer[..],
                receive_counts,
                receive_displacements,
            );
            world.all_to_all_varcount_into(&send_partition, &mut receive_partition);
        }
        for packed in receive_buffer {
            apply(packed, &mut buckets, &mut cursor);
        }
    }
}

/// Pregel-style level-synchronous BFS over rsmpi. Each superstep expands the
/// rank's active vertices in parallel, gathers the message-count matrix for
/// termination, and exchanges owner-bucketed messages with Alltoallv.
pub fn pregel_bfs_mpi(csr: &Csr, source: u64, requested_threads: usize) -> MpiPregelResult {
    let universe = mpi::initialize().expect("MPI initialize failed");
    let world = universe.world();
    let rank = world.rank() as usize;
    let ranks = world.size() as usize;
    let local_world = world.split_shared(rank as i32);
    let local_ranks = local_world.size().max(1) as usize;
    let available_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1);
    let threads = if requested_threads == 0 {
        available_threads.div_ceil(local_ranks).max(1)
    } else {
        requested_threads.max(1)
    };
    let n = csr.row_ptr.len() - 1;
    let ownership = Ownership::new(csr, ranks);
    if rank == 0 {
        println!(
            "[pregel_mpi] ownership={} threads_per_rank={} local_ranks={}",
            ownership.name(),
            threads,
            local_ranks
        );
    }

    let mut distances = vec![UNVISITED; n];
    // Once this rank has proposed a vertex, proposing it again in a later
    // superstep cannot discover anything new. Atomic claims also deduplicate
    // candidates produced by this rank's worker threads without sorting.
    let sent: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
    sent[source as usize].store(true, Ordering::Relaxed);
    let mut active = Vec::new();
    if ownership.owner(source, ranks) == rank {
        distances[source as usize] = 0;
        active.push(source);
    }

    world.barrier();
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    let start = Instant::now();
    if let Ownership::Locality(table) = &ownership {
        // Region partitions of a road-like graph cross rank boundaries
        // rarely, so the label-correcting sweep engine pays collectives per
        // boundary crossing, not per BFS level.
        sweep_bfs(&world, csr, source, table, rank, ranks, &mut distances);
    } else {
        let mut level = 0u64;
        loop {
            let mut outgoing = expand_parallel(csr, &active, &ownership, ranks, threads, &sent);

            let mut local = std::mem::take(&mut outgoing[rank]);
            local.retain(|&vertex| {
                if distances[vertex as usize] == UNVISITED {
                    distances[vertex as usize] = level + 1;
                    true
                } else {
                    false
                }
            });
            let send_counts: Vec<Count> = outgoing
                .iter()
                .map(|messages| Count::try_from(messages.len()).expect("MPI send count overflow"))
                .collect();
            // One Allgather replaces the old Allreduce(activity) +
            // Alltoall(send_counts) pair. Every rank sees the complete count
            // matrix, so it can both detect termination and derive its receive
            // counts. The diagonal carries local-only activity for termination;
            // self messages have already been handled above and are not sent.
            let mut activity_counts = send_counts.clone();
            activity_counts[rank] =
                Count::try_from(local.len()).expect("MPI local activity count overflow");
            let mut count_matrix = vec![0 as Count; ranks * ranks];
            world.all_gather_into(&activity_counts[..], &mut count_matrix[..]);
            if count_matrix.iter().all(|&count| count == 0) {
                break;
            }
            let mut total_remote = 0usize;
            for source in 0..ranks {
                for destination in 0..ranks {
                    if source != destination {
                        total_remote += count_matrix[source * ranks + destination] as usize;
                    }
                }
            }
            if total_remote == 0 {
                active = local;
                level += 1;
                continue;
            }

            if use_bitmap_exchange(total_remote, n, ranks) {
                let words = n.div_ceil(64);
                let mut mine = vec![0u64; words];
                for messages in &outgoing {
                    for &vertex in messages {
                        mine[vertex as usize / 64] |= 1 << (vertex % 64);
                    }
                }
                let mut all = vec![0u64; words * ranks];
                world.all_gather_into(&mine[..], &mut all[..]);

                let mut next = local;
                for source in 0..ranks {
                    if source == rank {
                        continue;
                    }
                    let base = source * words;
                    for word_index in 0..words {
                        let mut bits = all[base + word_index];
                        while bits != 0 {
                            let vertex = (word_index * 64 + bits.trailing_zeros() as usize) as u64;
                            bits &= bits - 1;
                            if ownership.owner(vertex, ranks) == rank
                                && distances[vertex as usize] == UNVISITED
                            {
                                distances[vertex as usize] = level + 1;
                                next.push(vertex);
                            }
                        }
                    }
                }
                active = next;
                level += 1;
                continue;
            }

            let send_displacements = displacements(&send_counts);
            let send_buffer: Vec<u64> = outgoing.into_iter().flatten().collect();
            let mut receive_counts: Vec<Count> = (0..ranks)
                .map(|source| count_matrix[source * ranks + rank])
                .collect();
            receive_counts[rank] = 0;
            let receive_displacements = displacements(&receive_counts);
            let receive_len: usize = receive_counts.iter().map(|&count| count as usize).sum();
            let mut receive_buffer = vec![0u64; receive_len];
            {
                let send_partition = Partition::new(&send_buffer[..], send_counts, send_displacements);
                let mut receive_partition = PartitionMut::new(
                    &mut receive_buffer[..],
                    receive_counts,
                    receive_displacements,
                );
                world.all_to_all_varcount_into(&send_partition, &mut receive_partition);
            }

            let mut next = local;
            for vertex in receive_buffer {
                if distances[vertex as usize] == UNVISITED {
                    distances[vertex as usize] = level + 1;
                    next.push(vertex);
                }
            }
            active = next;
            level += 1;
        }
    }
    world.barrier();
    let kernel = start.elapsed();
    if let Some(metrics) = ip_cxl_metrics.finish() {
        print_bench_metrics_value(metrics);
    }

    // Ownership may be cyclic, so reduce sparse per-rank distance vectors
    // rather than assuming each rank owns one contiguous Gatherv slice.
    let root = world.process_at_rank(0);
    let full = if rank == 0 {
        let mut full = vec![UNVISITED; n];
        root.reduce_into_root(&distances[..], &mut full[..], SystemOperation::min());
        Some(full)
    } else {
        root.reduce_into(&distances[..], SystemOperation::min());
        None
    };

    MpiPregelResult {
        rank,
        ranks,
        kernel,
        distances: full,
    }
}
