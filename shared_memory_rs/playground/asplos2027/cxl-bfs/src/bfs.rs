use crate::frontier::{FrontierSegment, SharedFrontier};
use crate::primitives::{
    FRONTIER_SEGMENT_U64S, ShmSegmentedU64Array, ShmU64Array, UNVISITED, alloc_u64_region,
};
use crate::utils::Csr;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use shm_cxl::utils::timestamps::CxlMetricTimer;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const EDGES_PER_THREAD: usize = 32768;
const PAR_MIN: usize = 16384;
const CLAIM_CHUNK: usize = 8192;
// Expansion is split by vertex-id chunk, not by frontier position, so hosts
// need no globally identical frontier ordering (and therefore no sorting):
// each host expands exactly the claimed vertices in its stripes. 1024-vertex
// chunks spread hub-heavy id ranges of power-law graphs across hosts.
const EXPAND_CHUNK: usize = 1024;

fn round_up8(value: usize) -> usize {
    value.div_ceil(8) * 8
}

/// Shared-frontier BFS for non-coherent CXL 2.0.
///
/// The immutable CSR and distance vector are replicated host-locally. Every
/// level each host publishes its candidates into its own lane through CXL,
/// then all hosts read the complete frontier (every lane), claim it against
/// their local distance replica, and expand their vertex-stripe of it. This
/// preserves an unpartitioned shared frontier without pretending that a
/// cacheable CXL distance array is coherent across hosts.
pub struct SharedBfs<'a> {
    frontier: SharedFrontier,
    distances: Vec<AtomicU64>,
    /// Once this host has proposed a vertex it never proposes it again: any
    /// vertex proposed at level L is claimed by every host at L+1, so a later
    /// proposal cannot discover anything. This bounds each lane to n entries
    /// and replaces the old sort+dedup of the candidate list.
    sent: Vec<AtomicBool>,
    csr: &'a Csr,
    myid: u16,
    nhosts: u16,
}

unsafe impl Send for SharedBfs<'_> {}
unsafe impl Sync for SharedBfs<'_> {}

impl<'a> SharedBfs<'a> {
    pub fn create<A: Allocator>(alloc: &A, csr: &'a Csr, source: u64) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;
        let lane_stride = round_up8(n).max(8);
        let capacity = lane_stride * nhosts;

        let ctrl = ShmU64Array::from_cl(alloc_u64_region(alloc, SharedFrontier::ctrl_len(nhosts)));
        let (a, a_offset) =
            ShmSegmentedU64Array::create_uninit_with(alloc, capacity, FRONTIER_SEGMENT_U64S);
        let (b, b_offset) =
            ShmSegmentedU64Array::create_uninit_with(alloc, capacity, FRONTIER_SEGMENT_U64S);

        let offsets = Cacheline {
            d: [
                ctrl.offset(),
                lane_stride as u64,
                a_offset,
                b_offset,
                0,
                0,
                0,
                0,
            ],
        };

        let frontier = SharedFrontier::new(ctrl, a, b, lane_stride, nhosts);
        frontier.seed(source, n);
        (Self::from_parts(frontier, csr), offsets)
    }

    pub fn attach(offsets: &Cacheline, csr: &'a Csr) -> Self {
        let nhosts = get_machine_cnt() as usize;
        let ctrl = ShmU64Array::from_offset(offsets.d[0]);
        let n = SharedFrontier::n_from_ctrl(&ctrl);
        assert_eq!(n, csr.row_ptr.len() - 1, "graph mismatch across hosts");
        let frontier = SharedFrontier::new(
            ctrl,
            ShmSegmentedU64Array::attach(offsets.d[2]),
            ShmSegmentedU64Array::attach(offsets.d[3]),
            offsets.d[1] as usize,
            nhosts,
        );
        Self::from_parts(frontier, csr)
    }

    fn from_parts(frontier: SharedFrontier, csr: &'a Csr) -> Self {
        let n = csr.row_ptr.len() - 1;
        Self {
            frontier,
            distances: (0..n).map(|_| AtomicU64::new(UNVISITED)).collect(),
            sent: (0..n).map(|_| AtomicBool::new(false)).collect(),
            csr,
            myid: get_machine_id(),
            nhosts: get_machine_cnt(),
        }
    }

    #[inline]
    fn expander(&self, vertex: u64) -> usize {
        (vertex as usize / EXPAND_CHUNK) % self.nhosts as usize
    }

    /// Read every lane segment, claim each vertex against the local distance
    /// replica, and return the claimed vertices in this host's expansion
    /// stripe. Exactly one copy of a vertex wins the local claim no matter
    /// how many hosts proposed it, so the returned stripes are disjoint and
    /// complete across hosts without any global ordering.
    fn gather_claim(
        &self,
        level: u64,
        parity: usize,
        segments: &[FrontierSegment],
        total: usize,
        threads: usize,
    ) -> Vec<u64> {
        let buffer = self.frontier.buffer(parity);
        // One main-thread timer covers the whole read+claim phase, matching
        // the partitioned engine's accounting.
        let _load_timer = CxlMetricTimer::load();
        for segment in segments {
            buffer.invalidate_slice(segment.base, segment.len);
        }

        let claim = |base: usize, len: usize, mine: &mut Vec<u64>| {
            for offset in 0..len {
                let vertex = buffer.load_local(base + offset);
                if self.distances[vertex as usize]
                    .compare_exchange(UNVISITED, level, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                    && self.expander(vertex) == (self.myid - 1) as usize
                {
                    mine.push(vertex);
                }
            }
        };

        if total < PAR_MIN || threads <= 1 {
            let mut mine = Vec::new();
            for segment in segments {
                claim(segment.base, segment.len, &mut mine);
            }
            return mine;
        }

        let tasks: Vec<(usize, usize)> = segments
            .iter()
            .flat_map(|segment| {
                (0..segment.len)
                    .step_by(CLAIM_CHUNK)
                    .map(move |k| (segment.base + k, CLAIM_CHUNK.min(segment.len - k)))
            })
            .collect();
        let workers = threads.min(tasks.len());
        let cursor = AtomicUsize::new(0);
        let parts: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let claim = &claim;
            let tasks = &tasks;
            let cursor = &cursor;
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(move || {
                        let mut mine = Vec::new();
                        loop {
                            let index = cursor.fetch_add(1, Ordering::Relaxed);
                            if index >= tasks.len() {
                                break;
                            }
                            let (base, len) = tasks[index];
                            claim(base, len, &mut mine);
                        }
                        mine
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        parts.into_iter().flatten().collect()
    }

    fn expand(&self, active: &[u64], threads: usize) -> Vec<u64> {
        let edges: usize = active
            .iter()
            .map(|&node| {
                (self.csr.row_ptr[node as usize + 1] - self.csr.row_ptr[node as usize]) as usize
            })
            .sum();
        let workers = threads
            .max(1)
            .min(edges / EDGES_PER_THREAD + 1)
            .min(active.len().max(1));

        let walk = |begin: usize, end: usize, candidates: &mut Vec<u64>| {
            for &node in &active[begin..end] {
                let row_begin = self.csr.row_ptr[node as usize] as usize;
                let row_end = self.csr.row_ptr[node as usize + 1] as usize;
                for &neighbour in &self.csr.col_idx[row_begin..row_end] {
                    if self.distances[neighbour as usize].load(Ordering::Relaxed) == UNVISITED
                        && self.sent[neighbour as usize]
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                    {
                        candidates.push(neighbour);
                    }
                }
            }
        };

        if workers <= 1 {
            let mut candidates = Vec::new();
            walk(0, active.len(), &mut candidates);
            return candidates;
        }

        let chunk = active.len().div_ceil(workers * 4).max(16);
        let cursor = AtomicUsize::new(0);
        let parts: Vec<Vec<u64>> = std::thread::scope(|scope| {
            let walk = &walk;
            let cursor = &cursor;
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(move || {
                        let mut candidates = Vec::new();
                        loop {
                            let begin = cursor.fetch_add(chunk, Ordering::Relaxed);
                            if begin >= active.len() {
                                break;
                            }
                            walk(begin, (begin + chunk).min(active.len()), &mut candidates);
                        }
                        candidates
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        parts.into_iter().flatten().collect()
    }

    pub fn run(&self, requested_threads: usize) {
        let me = (self.myid - 1) as usize;
        let threads = requested_threads.max(1);
        barrier_wait_all();

        let mut level = 0u64;
        loop {
            let parity = (level % 2) as usize;
            let (segments, total) = self.frontier.snapshot(parity);
            if total == 0 {
                break;
            }

            let active = self.gather_claim(level, parity, &segments, total, threads);
            let candidates = self.expand(&active, threads);

            let next_parity = 1 - parity;
            if !candidates.is_empty() {
                let base = self.frontier.lane_base(me);
                let lane = self.frontier.buffer(next_parity);
                let _store_timer = CxlMetricTimer::store();
                for (offset, &vertex) in candidates.iter().enumerate() {
                    lane.store_local(base + offset, vertex);
                }
                lane.flush_slice(base, candidates.len());
            }
            self.frontier.publish(next_parity, me, candidates.len());

            // The barrier both releases this level's publishes and guarantees
            // no host still reads this parity's lanes when they are reused
            // two levels later.
            barrier_wait_all();
            level += 1;
        }
    }

    pub fn collect_distances(&self) -> Vec<u64> {
        self.distances
            .iter()
            .map(|distance| distance.load(Ordering::Relaxed))
            .collect()
    }
}
