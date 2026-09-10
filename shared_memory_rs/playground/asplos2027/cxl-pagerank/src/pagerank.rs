use crate::primitives::{ShmU64Array, alloc_u64_region, gather_sum_f64};
use crate::utils::Csr;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

// Shared-memory PageRank over CXL
// in CXL;
// each iteration is two barrier-separated phases:
//   1. contrib[owned] = rank[owned] / deg(owned)
//   2. rank[owned]    = (1-d)/n + d * SUM contrib[nbrs]
pub struct SharedPageRank {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    rank: ShmU64Array,
    contrib: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    block: usize,
}

unsafe impl Send for SharedPageRank {}
unsafe impl Sync for SharedPageRank {}

impl SharedPageRank {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;

        let row_ptr = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.row_ptr.len()));
        let col_idx = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.col_idx.len().max(1)));
        let rank = ShmU64Array::from_cl(alloc_u64_region(alloc, n));
        let contrib = ShmU64Array::from_cl(alloc_u64_region(alloc, n));

        row_ptr.store_slice_flush(&csr.row_ptr);
        col_idx.store_slice_flush(&csr.col_idx);

        let offsets = Cacheline {
            d: [
                row_ptr.offset(),
                col_idx.offset(),
                rank.offset(),
                contrib.offset(),
                n as u64,
                0,
                0,
                0,
            ],
        };

        (
            SharedPageRank {
                row_ptr,
                col_idx,
                rank,
                contrib,
                n,
                nhosts,
                myid: get_machine_id(),
                block: round_up8(n.div_ceil(nhosts).max(1)),
            },
            offsets,
        )
    }

    pub fn attach(offsets: &Cacheline) -> Self {
        let n = offsets.d[4] as usize;
        let nhosts = get_machine_cnt() as usize;
        SharedPageRank {
            row_ptr: ShmU64Array::from_offset(offsets.d[0]),
            col_idx: ShmU64Array::from_offset(offsets.d[1]),
            rank: ShmU64Array::from_offset(offsets.d[2]),
            contrib: ShmU64Array::from_offset(offsets.d[3]),
            n,
            nhosts,
            myid: get_machine_id(),
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
    }

    #[inline]
    fn deg(&self, v: usize) -> usize {
        (self.row_ptr.load_local(v + 1) - self.row_ptr.load_local(v)) as usize
    }

    fn par_range<F: Fn(usize) + Sync>(&self, a: usize, b: usize, nt: usize, f: F) {
        let len = b - a;
        std::thread::scope(|s| {
            for t in 0..nt {
                let lo = a + len * t / nt;
                let hi = a + len * (t + 1) / nt;
                let fr = &f;
                s.spawn(move || {
                    for v in lo..hi {
                        fr(v);
                    }
                });
            }
        });
    }

    pub fn run(&self, iters: usize, damping: f64, threads: usize) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let nt = threads.max(1);
        let me = (self.myid - 1) as usize;
        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);
        let nf = self.n as f64;
        let base = (1.0 - damping) / nf;

        for v in lo..hi {
            self.rank.store_f64_local(v, 1.0 / nf);
        }
        self.rank.flush_slice(lo, hi - lo);
        barrier_wait_all();

        for _ in 0..iters {
            // phase 1: contrib for owned vertices
            self.par_range(lo, hi, nt, |v| {
                let d = self.deg(v);
                let c = if d > 0 {
                    self.rank.load_f64_local(v) / d as f64
                } else {
                    0.0
                };
                self.contrib.store_f64_local(v, c);
            });
            self.contrib.flush_slice(lo, hi - lo);
            barrier_wait_all();

            // everyone needs every host's contributions
            self.contrib.invalidate_range(self.n);
            let contrib_addr = self.contrib.as_ptr() as usize;

            // phase 2: new rank for owned vertices
            self.par_range(lo, hi, nt, |v| {
                let contrib_ptr = contrib_addr as *const f64;
                let rs = self.row_ptr.load_local(v) as usize;
                let re = self.row_ptr.load_local(v + 1) as usize;
                let col = unsafe { self.col_idx.as_ptr().add(rs) };
                let s = gather_sum_f64(contrib_ptr, col, re - rs);
                self.rank.store_f64_local(v, base + damping * s);
            });
            self.rank.flush_slice(lo, hi - lo);
            barrier_wait_all();
        }
    }

    pub fn collect(&self) -> Vec<f64> {
        self.rank.invalidate_range(self.n);
        (0..self.n).map(|i| self.rank.load_f64_local(i)).collect()
    }
}
