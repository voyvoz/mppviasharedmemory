use crate::primitives::{ShmU64Array, UNVISITED, alloc_cachelines, alloc_u64_region};
use crate::utils::{Csr, edge_weight};
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use std::sync::atomic::{AtomicBool, Ordering};

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

pub struct SharedSssp {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    dist: ShmU64Array,
    flags: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    block: usize,
}

unsafe impl Send for SharedSssp {}
unsafe impl Sync for SharedSssp {}

impl SharedSssp {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr, source: u64) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;

        let row_ptr = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.row_ptr.len()));
        let col_idx = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.col_idx.len().max(1)));
        let dist = ShmU64Array::from_cl(alloc_u64_region(alloc, n));
        let flags = ShmU64Array::from_cl(alloc_cachelines(alloc, nhosts));

        row_ptr.store_slice_flush(&csr.row_ptr);
        col_idx.store_slice_flush(&csr.col_idx);
        dist.fill_flush(n, UNVISITED);
        dist.set(source as usize, 0);
        flags.fill_flush(nhosts * 8, 0);

        let offsets = Cacheline {
            d: [
                row_ptr.offset(),
                col_idx.offset(),
                dist.offset(),
                flags.offset(),
                n as u64,
                nhosts as u64,
                0,
                0,
            ],
        };

        (
            SharedSssp {
                row_ptr,
                col_idx,
                dist,
                flags,
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
        let nhosts = offsets.d[5] as usize;
        SharedSssp {
            row_ptr: ShmU64Array::from_offset(offsets.d[0]),
            col_idx: ShmU64Array::from_offset(offsets.d[1]),
            dist: ShmU64Array::from_offset(offsets.d[2]),
            flags: ShmU64Array::from_offset(offsets.d[3]),
            n,
            nhosts,
            myid: get_machine_id(),
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
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

    pub fn run(&self, threads: usize) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let nt = threads.max(1);
        let me = (self.myid - 1) as usize;
        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);

        barrier_wait_all();

        loop {
            self.dist.invalidate_range(self.n);
            let changed = AtomicBool::new(false);

            self.par_range(lo, hi, nt, |v| {
                let rs = self.row_ptr.load_local(v) as usize;
                let re = self.row_ptr.load_local(v + 1) as usize;
                let cur = self.dist.load_local(v);
                let mut best = cur;
                for e in rs..re {
                    let u = self.col_idx.load_local(e);
                    let du = self.dist.load_local(u as usize);
                    if du != UNVISITED {
                        let cand = du + edge_weight(u, v as u64);
                        if cand < best {
                            best = cand;
                        }
                    }
                }
                if best < cur {
                    self.dist.store_local(v, best);
                    changed.store(true, Ordering::Relaxed);
                }
            });
            self.dist.flush_slice(lo, hi - lo);

            self.flags.set(
                me * 8,
                if changed.load(Ordering::Relaxed) {
                    1
                } else {
                    0
                },
            );
            barrier_wait_all();

            let any = (0..self.nhosts).any(|h| self.flags.get(h * 8) != 0);
            barrier_wait_all();
            if !any {
                break;
            }
        }

        barrier_wait_all();
    }

    pub fn collect_distances(&self) -> Vec<u64> {
        self.dist.invalidate_range(self.n);
        (0..self.n).map(|i| self.dist.load_local(i)).collect()
    }
}
