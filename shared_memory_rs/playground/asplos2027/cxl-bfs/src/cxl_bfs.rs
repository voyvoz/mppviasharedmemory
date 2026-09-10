use crate::primitives::{ShmU64Array, UNVISITED, alloc_cachelines, alloc_u64_region};
use crate::utils::Csr;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use shm_cxl::utils::timestamps::CxlMetricTimer;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

// One barrier per level: `counts` and `ids` are double-buffered by level
// parity, so a host may write level L+1 while a peer still reads level L.
// The barrier bounds the skew to one level, which is exactly what the two
// buffers absorb. Termination is derived from the exchanged counts (each
// sender also publishes its local-next size in `LOCAL_NEXT_SLOT`), so no
// separate `active` exchange/barrier is needed.
const LOCAL_NEXT_SLOT: usize = 7;

// Thread spawns cost ~tens of us each and every level ends at a barrier, so
// spawning must be proportional to the work: high-diameter graphs run
// thousands of near-empty levels where any fixed per-level spawn cost
// dominates the whole kernel.
const EDGES_PER_THREAD: usize = 32768;
const EXCHANGE_PAR_MIN: usize = 16384;
const RECV_CHUNK: usize = 8192;
const CYCLIC_OWNER_CHUNK: usize = 1024;
const CYCLIC_MIN_AVG_DEGREE: usize = 8;

pub struct CxlBfs<'a> {
    csr: &'a Csr,
    counts: ShmU64Array,
    ids: [ShmU64Array; 2],
    result: ShmU64Array,
    dist: Vec<AtomicU64>,
    sent: Vec<AtomicBool>,
    n: usize,
    nhosts: usize,
    myid: u16,
    source: u64,
    cuts: Vec<usize>,
    owner_chunk: usize,
    lane_cap: usize,
    ids_stride: usize,
}

unsafe impl Send for CxlBfs<'_> {}
unsafe impl Sync for CxlBfs<'_> {}

impl<'a> CxlBfs<'a> {
    /// Contiguous vertex ranges balanced by edge count, not vertex count:
    /// datasets ordered hubs-first (as-Skitter) put nearly all edges in the
    /// first vertex-count block, and the level barrier makes every host pay
    /// the slowest host's expansion. Boundaries are rounded to 8 so hosts
    /// never share a `result` cacheline. Both sides derive the same cuts from
    /// the same CSR, so they need not be exchanged.
    fn edge_cuts(csr: &Csr, nhosts: usize) -> Vec<usize> {
        let n = csr.row_ptr.len() - 1;
        let total = csr.row_ptr[n] as usize;
        let mut cuts = vec![0usize; nhosts + 1];
        for k in 1..nhosts {
            let target = (total * k / nhosts) as u64;
            let c = csr.row_ptr.partition_point(|&x| x < target);
            cuts[k] = round_up8(c).min(n).max(cuts[k - 1]);
        }
        cuts[nhosts] = n;
        cuts
    }

    /// Road graphs benefit from contiguous ranges because most edges stay
    /// local. Power-law graphs often encode hubs and BFS layers in vertex-id
    /// order, which makes contiguous ownership level-imbalanced; distribute
    /// cacheline-aligned chunks cyclically for those graphs instead.
    fn layout(csr: &Csr, nhosts: usize) -> (Vec<usize>, usize, usize, usize) {
        let n = csr.row_ptr.len() - 1;
        let total = csr.row_ptr[n] as usize;
        let cuts = Self::edge_cuts(csr, nhosts);
        if total >= n.saturating_mul(CYCLIC_MIN_AVG_DEGREE) {
            let chunks = n.div_ceil(CYCLIC_OWNER_CHUNK);
            let lane_cap = round_up8(chunks.div_ceil(nhosts) * CYCLIC_OWNER_CHUNK).max(1);
            let ids_stride = nhosts * lane_cap;
            (cuts, CYCLIC_OWNER_CHUNK, lane_cap, ids_stride)
        } else {
            (cuts, 0, 0, round_up8(n).max(1))
        }
    }

    fn fresh_dist(n: usize, source: u64, owned: bool) -> Vec<AtomicU64> {
        let dist: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(UNVISITED)).collect();
        if owned {
            dist[source as usize].store(0, Ordering::Relaxed);
        }
        dist
    }

    fn fresh_sent(n: usize, source: u64) -> Vec<AtomicBool> {
        let sent: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();
        sent[source as usize].store(true, Ordering::Relaxed);
        sent
    }

    pub fn create<A: Allocator>(alloc: &A, csr: &'a Csr, source: u64) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;
        assert!(
            nhosts <= LOCAL_NEXT_SLOT,
            "counts row has no free slot for local-next size"
        );
        let (cuts, owner_chunk, lane_cap, ids_stride) = Self::layout(csr, nhosts);

        let counts = ShmU64Array::from_cl(alloc_cachelines(alloc, 2 * nhosts));
        // Keep the parity buffers in separate allocations. The bitmap
        // allocator currently caps one allocation at 64 blocks; combining
        // both buffers exceeds that cap for three-host as-Skitter even though
        // each individual buffer is comfortably below it.
        // A sender row is partitioned by the same vertex cuts as ownership:
        // destination r uses [cuts[r], cuts[r+1]). This is sufficient because
        // global send dedup means a sender can emit each owned vertex at most
        // once, and avoids padding every destination lane to the widest range.
        let ids_len = nhosts * ids_stride;
        let ids = [
            ShmU64Array::from_cl(alloc_u64_region(alloc, ids_len)),
            ShmU64Array::from_cl(alloc_u64_region(alloc, ids_len)),
        ];
        let result = ShmU64Array::from_cl(alloc_u64_region(alloc, n));

        counts.fill_flush(2 * nhosts * 8, 0);
        result.fill_flush(n, UNVISITED);

        let offsets = Cacheline {
            d: [
                counts.offset(),
                ids[0].offset(),
                ids[1].offset(),
                result.offset(),
                n as u64,
                source,
                nhosts as u64,
                0,
            ],
        };

        let myid = get_machine_id();
        let me = (myid - 1) as usize;
        let owned = Self::owner_of(source, &cuts, owner_chunk, nhosts) == me;

        (
            CxlBfs {
                csr,
                counts,
                ids,
                result,
                dist: Self::fresh_dist(n, source, owned),
                sent: Self::fresh_sent(n, source),
                n,
                nhosts,
                myid,
                source,
                cuts,
                owner_chunk,
                lane_cap,
                ids_stride,
            },
            offsets,
        )
    }

    pub fn attach(offsets: &Cacheline, csr: &'a Csr) -> Self {
        let n = offsets.d[4] as usize;
        let source = offsets.d[5];
        let nhosts = offsets.d[6] as usize;
        let (cuts, owner_chunk, lane_cap, ids_stride) = Self::layout(csr, nhosts);

        let myid = get_machine_id();
        let me = (myid - 1) as usize;
        let owned = Self::owner_of(source, &cuts, owner_chunk, nhosts) == me;

        CxlBfs {
            csr,
            counts: ShmU64Array::from_offset(offsets.d[0]),
            ids: [
                ShmU64Array::from_offset(offsets.d[1]),
                ShmU64Array::from_offset(offsets.d[2]),
            ],
            result: ShmU64Array::from_offset(offsets.d[3]),
            dist: Self::fresh_dist(n, source, owned),
            sent: Self::fresh_sent(n, source),
            n,
            nhosts,
            myid,
            source,
            cuts,
            owner_chunk,
            lane_cap,
            ids_stride,
        }
    }

    #[inline]
    fn owner_of(v: u64, cuts: &[usize], owner_chunk: usize, nhosts: usize) -> usize {
        let v = v as usize;
        if owner_chunk != 0 {
            return (v / owner_chunk) % nhosts;
        }
        let mut k = 0;
        while cuts[k + 1] <= v {
            k += 1;
        }
        k
    }

    #[inline]
    fn owner(&self, v: u64) -> usize {
        Self::owner_of(v, &self.cuts, self.owner_chunk, self.nhosts)
    }

    #[inline]
    fn lane_base(&self, sender: usize, dest: usize) -> usize {
        let offset = if self.owner_chunk == 0 {
            self.cuts[dest]
        } else {
            dest * self.lane_cap
        };
        sender * self.ids_stride + offset
    }

    #[inline]
    fn lane_capacity(&self, dest: usize) -> usize {
        if self.owner_chunk == 0 {
            self.cuts[dest + 1] - self.cuts[dest]
        } else {
            self.lane_cap
        }
    }

    fn expand(&self, frontier: &[u64], level: u64, threads: usize) -> (Vec<u64>, Vec<Vec<u64>>) {
        let nt = threads.max(1).min(frontier.len().max(1));
        let me = (self.myid - 1) as usize;
        let row_ptr = &self.csr.row_ptr;
        let col_idx = &self.csr.col_idx;
        let dist = &self.dist;
        let nhosts = self.nhosts;
        let cuts = self.cuts.as_slice();
        let owner_chunk = self.owner_chunk;
        let flen = frontier.len();
        let chunk = flen.div_ceil(nt * 4).max(16);
        let cursor = AtomicUsize::new(0);

        let walk = || {
            let mut local_next: Vec<u64> = Vec::new();
            let mut out: Vec<Vec<u64>> = vec![Vec::new(); nhosts];
            loop {
                let lo = cursor.fetch_add(chunk, Ordering::Relaxed);
                if lo >= flen {
                    break;
                }
                let hi = (lo + chunk).min(flen);
                for &u in &frontier[lo..hi] {
                    let rs = row_ptr[u as usize] as usize;
                    let re = row_ptr[u as usize + 1] as usize;
                    for &v in &col_idx[rs..re] {
                        let o = Self::owner_of(v, cuts, owner_chunk, nhosts);
                        if o == me {
                            if dist[v as usize]
                                .compare_exchange(
                                    UNVISITED,
                                    level + 1,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                            {
                                local_next.push(v);
                            }
                        } else if self.sent[v as usize]
                            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                        {
                            // Once this host has proposed `v`, its owner will
                            // process it at the earliest possible level. Any
                            // later proposal from this host is redundant.
                            out[o].push(v);
                        }
                    }
                }
            }
            (local_next, out)
        };

        if nt == 1 {
            return walk();
        }

        let parts: Vec<(Vec<u64>, Vec<Vec<u64>>)> = std::thread::scope(|s| {
            let walk = &walk;
            let mut handles = Vec::with_capacity(nt);
            for _ in 0..nt {
                handles.push(s.spawn(walk));
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut local_next: Vec<u64> = Vec::new();
        let mut out: Vec<Vec<u64>> = vec![Vec::new(); nhosts];
        for (ln, o) in parts {
            local_next.extend(ln);
            for (r, list) in o.into_iter().enumerate() {
                out[r].extend(list);
            }
        }
        (local_next, out)
    }

    pub fn run(&mut self, threads: usize) {
        let me = (self.myid - 1) as usize;
        let mut frontier: Vec<u64> = Vec::new();
        if self.owner(self.source) == me {
            frontier.push(self.source);
        }

        barrier_wait_all();

        let profile_mode = std::env::var("CXL_BFS_PROFILE").ok();
        let profile = profile_mode.is_some();
        let profile_levels = profile_mode.as_deref() == Some("levels");
        let mut t_expand = std::time::Duration::ZERO;
        let mut t_send = std::time::Duration::ZERO;
        let mut t_barrier = std::time::Duration::ZERO;
        let mut t_recv = std::time::Duration::ZERO;
        let mut sent_sum = 0usize;
        let mut recv_sum = 0usize;

        let mut level = 0u64;
        loop {
            let parity = (level % 2) as usize;
            let counts_base = parity * self.nhosts * 8;
            let ids = &self.ids[parity];

            let row_ptr = &self.csr.row_ptr;
            let edges: usize = frontier
                .iter()
                .map(|&u| (row_ptr[u as usize + 1] - row_ptr[u as usize]) as usize)
                .sum();
            let nt = (edges / EDGES_PER_THREAD + 1).min(threads.max(1));

            let t0 = std::time::Instant::now();
            let (mut next, out) = self.expand(&frontier, level, nt);
            let level_expand = t0.elapsed();
            t_expand += level_expand;
            let t0 = std::time::Instant::now();
            let store_timer = CxlMetricTimer::store();

            let write = |list: Vec<u64>, r: usize| -> (usize, usize) {
                let w = list.len();
                assert!(w <= self.lane_capacity(r), "send buffer overflow");

                let base = self.lane_base(me, r);
                for (i, &v) in list.iter().enumerate() {
                    ids.store_local(base + i, v);
                }
                ids.flush_slice(base, w);
                (r, w)
            };

            let sent_total: usize = out
                .iter()
                .enumerate()
                .filter(|(r, _)| *r != me)
                .map(|(_, l)| l.len())
                .sum();
            let widths: Vec<(usize, usize)> = if sent_total >= EXCHANGE_PAR_MIN && self.nhosts > 2 {
                std::thread::scope(|s| {
                    let write = &write;
                    let handles: Vec<_> = out
                        .into_iter()
                        .enumerate()
                        .filter(|(r, _)| *r != me)
                        .map(|(r, list)| s.spawn(move || write(list, r)))
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                })
            } else {
                out.into_iter()
                    .enumerate()
                    .filter(|(r, _)| *r != me)
                    .map(|(r, list)| write(list, r))
                    .collect()
            };
            let sent_level: usize = widths.iter().map(|(_, w)| *w).sum();
            for (r, w) in widths {
                sent_sum += w;
                self.counts.store_local(counts_base + me * 8 + r, w as u64);
            }
            self.counts
                .store_local(counts_base + me * 8 + LOCAL_NEXT_SLOT, next.len() as u64);
            self.counts.flush_slice(counts_base + me * 8, 8);
            drop(store_timer);
            let level_send = t0.elapsed();
            t_send += level_send;

            let t0 = profile.then(std::time::Instant::now);
            barrier_wait_all();
            let level_barrier = t0.map(|t0| t0.elapsed()).unwrap_or_default();
            t_barrier += level_barrier;
            let t0 = std::time::Instant::now();
            let load_timer = CxlMetricTimer::load();

            self.counts.invalidate_slice(counts_base, self.nhosts * 8);
            let mut segs: Vec<(usize, usize)> = Vec::new();
            for s in 0..self.nhosts {
                if s == me {
                    continue;
                }
                let c = self.counts.load_local(counts_base + s * 8 + me) as usize;
                if c == 0 {
                    continue;
                }
                let base = self.lane_base(s, me);
                ids.invalidate_slice(base, c);
                segs.push((base, c));
            }

            let claim = |base: usize, len: usize, next: &mut Vec<u64>| {
                for i in 0..len {
                    let v = ids.load_local(base + i);
                    if self.dist[v as usize]
                        .compare_exchange(
                            UNVISITED,
                            level + 1,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        next.push(v);
                    }
                }
            };

            let recv_total: usize = segs.iter().map(|s| s.1).sum();
            if recv_total >= EXCHANGE_PAR_MIN && nt > 1 {
                let tasks: Vec<(usize, usize)> = segs
                    .iter()
                    .flat_map(|&(base, c)| {
                        (0..c)
                            .step_by(RECV_CHUNK)
                            .map(move |k| (base + k, RECV_CHUNK.min(c - k)))
                    })
                    .collect();
                let nthr = nt.min(tasks.len());
                let parts: Vec<Vec<u64>> = std::thread::scope(|s| {
                    let claim = &claim;
                    let tasks = &tasks;
                    let handles: Vec<_> = (0..nthr)
                        .map(|t| {
                            s.spawn(move || {
                                let mut got: Vec<u64> = Vec::new();
                                let mut i = t;
                                while i < tasks.len() {
                                    let (base, len) = tasks[i];
                                    claim(base, len, &mut got);
                                    i += nthr;
                                }
                                got
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for p in parts {
                    next.extend(p);
                }
            } else {
                for &(base, c) in &segs {
                    claim(base, c, &mut next);
                }
            }
            recv_sum += recv_total;
            drop(load_timer);
            let level_recv = t0.elapsed();
            t_recv += level_recv;
            if profile_levels {
                println!(
                    "LEVEL host={} level={} frontier={} edges={} threads={} next={} sent={} recvd={} expand={:?} send={:?} barrier={:?} recv={:?}",
                    self.myid,
                    level,
                    frontier.len(),
                    edges,
                    nt,
                    next.len(),
                    sent_level,
                    recv_total,
                    level_expand,
                    level_send,
                    level_barrier,
                    level_recv
                );
            }
            // Any next frontier is a subset of local-next sizes plus sent ids,
            // so if both sum to zero everywhere, every host's next is empty.
            // Sent ids that all turn out already visited cost one extra empty
            // level before this reaches zero.
            let mut total = 0u64;
            for s in 0..self.nhosts {
                total += self
                    .counts
                    .load_local(counts_base + s * 8 + LOCAL_NEXT_SLOT);
                for r in 0..self.nhosts {
                    total += self.counts.load_local(counts_base + s * 8 + r);
                }
            }
            if total == 0 {
                break;
            }

            frontier = next;
            level += 1;
        }

        if profile {
            println!(
                "PHASES host {} levels={} expand={:?} send={:?} barrier={:?} recv={:?} sent={} recvd={}",
                self.myid, level, t_expand, t_send, t_barrier, t_recv, sent_sum, recv_sum
            );
        }
    }

    /// Publish the distributed result for validation after kernel timing.
    /// MPI likewise excludes its final distance reduction from BENCH_KERNEL_NS.
    pub fn publish_distances(&self) {
        let me = (self.myid - 1) as usize;
        {
            let _store_timer = CxlMetricTimer::store();
            if self.owner_chunk == 0 {
                let lo = self.cuts[me];
                let hi = self.cuts[me + 1];
                for v in lo..hi {
                    self.result
                        .store_local(v, self.dist[v].load(Ordering::Relaxed));
                }
                self.result.flush_slice(lo, hi - lo);
            } else {
                let chunks = self.n.div_ceil(self.owner_chunk);
                for chunk in (me..chunks).step_by(self.nhosts) {
                    let lo = chunk * self.owner_chunk;
                    let hi = (lo + self.owner_chunk).min(self.n);
                    for v in lo..hi {
                        self.result
                            .store_local(v, self.dist[v].load(Ordering::Relaxed));
                    }
                    self.result.flush_slice(lo, hi - lo);
                }
            }
        }
        barrier_wait_all();
    }

    pub fn collect_distances(&self) -> Vec<u64> {
        let _load_timer = CxlMetricTimer::load();
        self.result.invalidate_range(self.n);
        (0..self.n).map(|i| self.result.load_local(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr_with_degrees(degrees: &[u64]) -> Csr {
        let mut row_ptr = Vec::with_capacity(degrees.len() + 1);
        row_ptr.push(0);
        for &degree in degrees {
            row_ptr.push(row_ptr.last().unwrap() + degree);
        }
        Csr {
            row_ptr,
            col_idx: Vec::new(),
        }
    }

    #[test]
    fn low_degree_graphs_use_contiguous_edge_balanced_ranges() {
        let csr = csr_with_degrees(&vec![2; 4096]);
        let (cuts, owner_chunk, lane_cap, ids_stride) = CxlBfs::layout(&csr, 3);
        assert_eq!(owner_chunk, 0);
        assert_eq!(lane_cap, 0);
        assert_eq!(ids_stride, 4096);
        assert_eq!(cuts.first(), Some(&0));
        assert_eq!(cuts.last(), Some(&4096));
        assert!(cuts.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn high_degree_graphs_cycle_cacheline_aligned_chunks() {
        let csr = csr_with_degrees(&vec![8; 4096]);
        let (cuts, owner_chunk, lane_cap, ids_stride) = CxlBfs::layout(&csr, 3);
        assert_eq!(owner_chunk, CYCLIC_OWNER_CHUNK);
        assert_eq!(lane_cap, 2048);
        assert_eq!(ids_stride, 6144);
        assert_eq!(CxlBfs::owner_of(0, &cuts, owner_chunk, 3), 0);
        assert_eq!(CxlBfs::owner_of(1023, &cuts, owner_chunk, 3), 0);
        assert_eq!(CxlBfs::owner_of(1024, &cuts, owner_chunk, 3), 1);
        assert_eq!(CxlBfs::owner_of(2048, &cuts, owner_chunk, 3), 2);
        assert_eq!(CxlBfs::owner_of(3072, &cuts, owner_chunk, 3), 0);
    }
}
