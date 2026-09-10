use crate::primitives::{ShmU64Array, alloc_cachelines, alloc_u64_region};
use crate::utils::Csr;
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};

const BATCH: usize = 3;

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

pub struct PregelPageRank {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    sent_matrix: ShmU64Array,
    result: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    block: usize,
}

unsafe impl Send for PregelPageRank {}
unsafe impl Sync for PregelPageRank {}

impl PregelPageRank {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;
        assert!(nhosts <= 8, "pregel: sent-matrix row holds at most 8 hosts");

        let row_ptr = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.row_ptr.len()));
        let col_idx = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.col_idx.len().max(1)));
        let sent_matrix = ShmU64Array::from_cl(alloc_cachelines(alloc, nhosts));
        let result = ShmU64Array::from_cl(alloc_u64_region(alloc, n));

        row_ptr.store_slice_flush(&csr.row_ptr);
        col_idx.store_slice_flush(&csr.col_idx);
        sent_matrix.fill_flush(nhosts * 8, 0);

        let offsets = Cacheline {
            d: [
                row_ptr.offset(),
                col_idx.offset(),
                sent_matrix.offset(),
                result.offset(),
                n as u64,
                nhosts as u64,
                0,
                0,
            ],
        };

        (
            PregelPageRank {
                row_ptr,
                col_idx,
                sent_matrix,
                result,
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
        PregelPageRank {
            row_ptr: ShmU64Array::from_offset(offsets.d[0]),
            col_idx: ShmU64Array::from_offset(offsets.d[1]),
            sent_matrix: ShmU64Array::from_offset(offsets.d[2]),
            result: ShmU64Array::from_offset(offsets.d[3]),
            n,
            nhosts,
            myid: get_machine_id(),
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
    }

    #[inline]
    fn owner(&self, v: u64) -> usize {
        (v as usize / self.block).min(self.nhosts - 1)
    }

    fn apply_msg(cl: &Cacheline, inc: &mut [f64]) {
        let count = cl.d[0] as usize;
        for k in 0..count {
            let v = cl.d[1 + 2 * k] as usize;
            let val = f64::from_bits(cl.d[2 + 2 * k]);
            inc[v] += val;
        }
    }

    pub fn run(&self, ipc: &mut CXLIpc, iters: usize, damping: f64) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let me = (self.myid - 1) as usize;
        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);
        let nf = self.n as f64;
        let base = (1.0 - damping) / nf;

        let mut rank = vec![0.0f64; self.n];
        for v in lo..hi {
            rank[v] = 1.0 / nf;
        }
        let mut acc = vec![0.0f64; self.n];
        let mut touched: Vec<u64> = Vec::new();
        let mut inc = vec![0.0f64; self.n];

        barrier_wait_all();

        for _ in 0..iters {
            touched.clear();
            for u in lo..hi {
                let rs = self.row_ptr.load_local(u) as usize;
                let re = self.row_ptr.load_local(u + 1) as usize;
                let d = re - rs;
                if d == 0 {
                    continue;
                }
                let c = rank[u] / d as f64;
                for e in rs..re {
                    let v = self.col_idx.load_local(e) as usize;
                    if acc[v] == 0.0 {
                        touched.push(v as u64);
                    }
                    acc[v] += c;
                }
            }

            for v in lo..hi {
                inc[v] = 0.0;
            }
            let mut out: Vec<Vec<(u64, f64)>> = vec![Vec::new(); self.nhosts];
            for &vv in &touched {
                let v = vv as usize;
                let h = self.owner(vv);
                if h == me {
                    inc[v] += acc[v];
                } else {
                    out[h].push((vv, acc[v]));
                }
            }

            let mut sent_cls = vec![0u64; self.nhosts];
            for h in 0..self.nhosts {
                if h != me {
                    sent_cls[h] = out[h].len().div_ceil(BATCH) as u64;
                }
            }
            for r in 0..self.nhosts {
                self.sent_matrix.set(me * 8 + r, sent_cls[r]);
            }
            barrier_wait_all();

            let mut expected = 0u64;
            for sender in 0..self.nhosts {
                if sender != me {
                    expected += self.sent_matrix.get(sender * 8 + me);
                }
            }

            let mut recv = 0u64;
            for h in 0..self.nhosts {
                if h == me {
                    continue;
                }
                let target = (h + 1) as PeerMachineID;
                let ids = &out[h];
                let mut i = 0;
                while i < ids.len() {
                    let end = (i + BATCH).min(ids.len());
                    let mut cl = Cacheline::empty();
                    cl.d[0] = (end - i) as u64;
                    for k in i..end {
                        cl.d[1 + 2 * (k - i)] = ids[k].0;
                        cl.d[2 + 2 * (k - i)] = ids[k].1.to_bits();
                    }
                    while ipc.try_send_to(target, &cl).is_err() {
                        if let Ok(Some((_, rc))) = ipc.try_pop_any() {
                            Self::apply_msg(&rc, &mut inc);
                            recv += 1;
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                    i = end;
                }
            }
            while recv < expected {
                match ipc.try_pop_any() {
                    Ok(Some((_, cl))) => {
                        Self::apply_msg(&cl, &mut inc);
                        recv += 1;
                    }
                    Ok(None) => std::hint::spin_loop(),
                    Err(e) => panic!("pregel pagerank: ipc error {e}"),
                }
            }

            for v in lo..hi {
                rank[v] = base + damping * inc[v];
            }
            for &vv in &touched {
                acc[vv as usize] = 0.0;
            }

            barrier_wait_all();
        }

        for v in lo..hi {
            self.result.store_f64_local(v, rank[v]);
        }
        self.result.flush_slice(lo, hi - lo);
        barrier_wait_all();
    }

    pub fn collect(&self) -> Vec<f64> {
        self.result.invalidate_range(self.n);
        (0..self.n).map(|i| self.result.load_f64_local(i)).collect()
    }
}
