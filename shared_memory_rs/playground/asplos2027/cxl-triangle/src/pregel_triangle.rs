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

pub struct PregelTriangle {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    sent_matrix: ShmU64Array,
    result: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    block: usize,
}

unsafe impl Send for PregelTriangle {}
unsafe impl Sync for PregelTriangle {}

impl PregelTriangle {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr) -> (Self, Cacheline) {
        let n = csr.row_ptr.len() - 1;
        let nhosts = get_machine_cnt() as usize;
        assert!(nhosts <= 8, "pregel: sent-matrix row holds at most 8 hosts");

        let mut sorted = csr.col_idx.clone();
        for v in 0..n {
            let rs = csr.row_ptr[v] as usize;
            let re = csr.row_ptr[v + 1] as usize;
            sorted[rs..re].sort_unstable();
        }

        let row_ptr = ShmU64Array::from_cl(alloc_u64_region(alloc, csr.row_ptr.len()));
        let col_idx = ShmU64Array::from_cl(alloc_u64_region(alloc, sorted.len().max(1)));
        let sent_matrix = ShmU64Array::from_cl(alloc_cachelines(alloc, nhosts));
        let result = ShmU64Array::from_cl(alloc_cachelines(alloc, nhosts));

        row_ptr.store_slice_flush(&csr.row_ptr);
        col_idx.store_slice_flush(&sorted);
        sent_matrix.fill_flush(nhosts * 8, 0);
        result.fill_flush(nhosts * 8, 0);

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
            PregelTriangle {
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
        PregelTriangle {
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

    fn contains(&self, u: usize, target: u64) -> bool {
        let mut lo = self.row_ptr.load_local(u) as usize;
        let mut hi = self.row_ptr.load_local(u + 1) as usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let x = self.col_idx.load_local(mid);
            if x == target {
                return true;
            } else if x < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        false
    }

    fn count_pairs(&self, cl: &Cacheline) -> u64 {
        let count = cl.d[0] as usize;
        let mut c = 0u64;
        for k in 0..count {
            let a = cl.d[1 + 2 * k];
            let b = cl.d[2 + 2 * k];
            if self.contains(a as usize, b) {
                c += 1;
            }
        }
        c
    }

    pub fn run(&self, ipc: &mut CXLIpc) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let me = (self.myid - 1) as usize;
        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);

        let mut out: Vec<Vec<(u64, u64)>> = vec![Vec::new(); self.nhosts];
        let mut local: Vec<(u64, u64)> = Vec::new();
        for u in lo..hi {
            let rs = self.row_ptr.load_local(u) as usize;
            let re = self.row_ptr.load_local(u + 1) as usize;
            let mut higher: Vec<u64> = Vec::new();
            for e in rs..re {
                let w = self.col_idx.load_local(e);
                if w as usize > u {
                    higher.push(w);
                }
            }
            for i in 0..higher.len() {
                for j in (i + 1)..higher.len() {
                    let a = higher[i];
                    let b = higher[j];
                    let h = self.owner(a);
                    if h == me {
                        local.push((a, b));
                    } else {
                        out[h].push((a, b));
                    }
                }
            }
        }

        let mut count = 0u64;
        for &(a, b) in &local {
            if self.contains(a as usize, b) {
                count += 1;
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

        let mut recv_count = 0u64;
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
                    cl.d[2 + 2 * (k - i)] = ids[k].1;
                }
                while ipc.try_send_to(target, &cl).is_err() {
                    if let Ok(Some((_, rc))) = ipc.try_pop_any() {
                        count += self.count_pairs(&rc);
                        recv_count += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                i = end;
            }
        }
        while recv_count < expected {
            match ipc.try_pop_any() {
                Ok(Some((_, cl))) => {
                    count += self.count_pairs(&cl);
                    recv_count += 1;
                }
                Ok(None) => std::hint::spin_loop(),
                Err(e) => panic!("pregel triangle: ipc error {e}"),
            }
        }

        self.result.set(me * 8, count);
        barrier_wait_all();
    }

    pub fn collect(&self) -> u64 {
        (0..self.nhosts).map(|h| self.result.get(h * 8)).sum()
    }
}
