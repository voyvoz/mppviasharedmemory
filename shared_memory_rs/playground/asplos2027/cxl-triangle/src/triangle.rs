use crate::primitives::{ShmU64Array, alloc_cachelines, alloc_u64_region};
use crate::utils::Csr;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use std::sync::atomic::{AtomicU64, Ordering};

const CHUNK: usize = 512;

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

pub struct SharedTriangle {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    partial: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    block: usize,
}

unsafe impl Send for SharedTriangle {}
unsafe impl Sync for SharedTriangle {}

impl SharedTriangle {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;

        let mut sorted = csr.col_idx.clone();
        for v in 0..n {
            let rs = csr.row_ptr[v] as usize;
            let re = csr.row_ptr[v + 1] as usize;
            sorted[rs..re].sort_unstable();
        }

        let row_ptr = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.row_ptr.len()));
        let col_idx = ShmU64Array::from_cl(alloc_u64_region(alloc, sorted.len().max(1)));
        let partial = ShmU64Array::from_cl(alloc_cachelines(alloc, nhosts));

        row_ptr.store_slice_flush(&csr.row_ptr);
        col_idx.store_slice_flush(&sorted);
        partial.fill_flush(nhosts * 8, 0);

        let offsets = Cacheline {
            d: [
                row_ptr.offset(),
                col_idx.offset(),
                partial.offset(),
                n as u64,
                nhosts as u64,
                0,
                0,
                0,
            ],
        };

        (
            SharedTriangle {
                row_ptr,
                col_idx,
                partial,
                n,
                nhosts,
                myid: get_machine_id(),
                block: round_up8(n.div_ceil(nhosts).max(1)),
            },
            offsets,
        )
    }

    pub fn attach(offsets: &Cacheline) -> Self {
        let n = offsets.d[3] as usize;
        let nhosts = offsets.d[4] as usize;
        SharedTriangle {
            row_ptr: ShmU64Array::from_offset(offsets.d[0]),
            col_idx: ShmU64Array::from_offset(offsets.d[1]),
            partial: ShmU64Array::from_offset(offsets.d[2]),
            n,
            nhosts,
            myid: get_machine_id(),
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
    }

    #[inline]
    fn count_vertex(&self, u: usize) -> u64 {
        let urs = self.row_ptr.load_local(u) as usize;
        let ure = self.row_ptr.load_local(u + 1) as usize;
        let mut total = 0u64;
        for e in urs..ure {
            let v = self.col_idx.load_local(e);
            if v as usize <= u {
                continue;
            }
            let vrs = self.row_ptr.load_local(v as usize) as usize;
            let vre = self.row_ptr.load_local(v as usize + 1) as usize;
            let mut i = urs;
            let mut j = vrs;
            while i < ure && j < vre {
                let a = self.col_idx.load_local(i);
                let b = self.col_idx.load_local(j);
                if a == b {
                    if a > v {
                        total += 1;
                    }
                    i += 1;
                    j += 1;
                } else if a < b {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
        total
    }

    pub fn run(&self, threads: usize) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let me = (self.myid - 1) as usize;
        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);

        barrier_wait_all();

        let total = AtomicU64::new(0);
        let cursor = AtomicU64::new(lo as u64);
        let total_ref = &total;
        let cursor_ref = &cursor;

        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(move || {
                    let mut local = 0u64;
                    loop {
                        let start = cursor_ref.fetch_add(CHUNK as u64, Ordering::Relaxed) as usize;
                        if start >= hi {
                            break;
                        }
                        let end = (start + CHUNK).min(hi);
                        for u in start..end {
                            local += self.count_vertex(u);
                        }
                    }
                    total_ref.fetch_add(local, Ordering::Relaxed);
                });
            }
        });

        self.partial.set(me * 8, total.into_inner());
        barrier_wait_all();
    }

    pub fn collect(&self) -> u64 {
        (0..self.nhosts).map(|h| self.partial.get(h * 8)).sum()
    }
}
