use crate::primitives::{ShmU64Array, UNVISITED, alloc_cachelines, alloc_u64_region};
use crate::utils::{Csr, edge_weight};
use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};

const BATCH: usize = 3;

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

pub struct PregelSssp {
    row_ptr: ShmU64Array,
    col_idx: ShmU64Array,
    sent_matrix: ShmU64Array,
    result: ShmU64Array,
    n: usize,
    nhosts: usize,
    myid: u16,
    source: u64,
    block: usize,
}

unsafe impl Send for PregelSssp {}
unsafe impl Sync for PregelSssp {}

impl PregelSssp {
    pub fn create<A: Allocator>(alloc: &A, csr: &Csr, source: u64) -> (Self, Cacheline) {
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
        result.fill_flush(n, UNVISITED);

        let offsets = Cacheline {
            d: [
                row_ptr.offset(),
                col_idx.offset(),
                sent_matrix.offset(),
                result.offset(),
                n as u64,
                source,
                nhosts as u64,
                0,
            ],
        };

        (
            PregelSssp {
                row_ptr,
                col_idx,
                sent_matrix,
                result,
                n,
                nhosts,
                myid: get_machine_id(),
                source,
                block: round_up8(n.div_ceil(nhosts).max(1)),
            },
            offsets,
        )
    }

    pub fn attach(offsets: &Cacheline) -> Self {
        let n = offsets.d[4] as usize;
        let nhosts = offsets.d[6] as usize;
        PregelSssp {
            row_ptr: ShmU64Array::from_offset(offsets.d[0]),
            col_idx: ShmU64Array::from_offset(offsets.d[1]),
            sent_matrix: ShmU64Array::from_offset(offsets.d[2]),
            result: ShmU64Array::from_offset(offsets.d[3]),
            n,
            nhosts,
            myid: get_machine_id(),
            source: offsets.d[5],
            block: round_up8(n.div_ceil(nhosts).max(1)),
        }
    }

    #[inline]
    fn owner(&self, v: u64) -> usize {
        (v as usize / self.block).min(self.nhosts - 1)
    }

    fn relax(
        cl: &Cacheline,
        dist: &mut [u64],
        improved: &mut Vec<u64>,
        stamp: &mut [u64],
        stampgen: u64,
    ) {
        let count = cl.d[0] as usize;
        for k in 0..count {
            let v = cl.d[1 + 2 * k] as usize;
            let cand = cl.d[2 + 2 * k];
            if cand < dist[v] {
                dist[v] = cand;
                if stamp[v] != stampgen {
                    stamp[v] = stampgen;
                    improved.push(v as u64);
                }
            }
        }
    }

    pub fn run(&self, ipc: &mut CXLIpc) {
        self.row_ptr.invalidate_range(self.n + 1);
        let m = self.row_ptr.load_local(self.n) as usize;
        self.col_idx.invalidate_range(m.max(1));

        let me = (self.myid - 1) as usize;
        let mut dist = vec![UNVISITED; self.n];
        let mut active: Vec<u64> = Vec::new();
        if self.owner(self.source) == me {
            dist[self.source as usize] = 0;
            active.push(self.source);
        }
        let mut stamp = vec![u64::MAX; self.n];

        barrier_wait_all();

        let mut stampgen = 0u64;
        loop {
            let mut out: Vec<Vec<(u64, u64)>> = vec![Vec::new(); self.nhosts];
            let mut local: Vec<(u64, u64)> = Vec::new();
            for &u in &active {
                let du = dist[u as usize];
                let rs = self.row_ptr.load_local(u as usize) as usize;
                let re = self.row_ptr.load_local(u as usize + 1) as usize;
                for e in rs..re {
                    let v = self.col_idx.load_local(e);
                    let cand = du + edge_weight(u, v);
                    let h = self.owner(v);
                    if h == me {
                        local.push((v, cand));
                    } else {
                        out[h].push((v, cand));
                    }
                }
            }

            let mut sent_cls = vec![0u64; self.nhosts];
            for h in 0..self.nhosts {
                if h != me {
                    sent_cls[h] = out[h].len().div_ceil(BATCH) as u64;
                }
            }
            sent_cls[me] = local.len() as u64;
            for r in 0..self.nhosts {
                self.sent_matrix.set(me * 8 + r, sent_cls[r]);
            }

            barrier_wait_all();

            let mut global = 0u64;
            let mut expected = 0u64;
            for sender in 0..self.nhosts {
                for r in 0..self.nhosts {
                    let c = self.sent_matrix.get(sender * 8 + r);
                    global += c;
                    if r == me && sender != me {
                        expected += c;
                    }
                }
            }
            if global == 0 {
                break;
            }

            stampgen = stampgen.wrapping_add(1);
            let mut improved: Vec<u64> = Vec::new();
            for &(v, cand) in &local {
                if cand < dist[v as usize] {
                    dist[v as usize] = cand;
                    if stamp[v as usize] != stampgen {
                        stamp[v as usize] = stampgen;
                        improved.push(v);
                    }
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
                            Self::relax(&rc, &mut dist, &mut improved, &mut stamp, stampgen);
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
                        Self::relax(&cl, &mut dist, &mut improved, &mut stamp, stampgen);
                        recv_count += 1;
                    }
                    Ok(None) => std::hint::spin_loop(),
                    Err(e) => panic!("pregel sssp: ipc error {e}"),
                }
            }

            active = improved;
            barrier_wait_all();
        }

        let lo = me * self.block;
        let hi = ((me + 1) * self.block).min(self.n);
        for v in lo..hi {
            self.result.set(v, dist[v]);
        }
        barrier_wait_all();
    }

    pub fn collect(&self) -> Vec<u64> {
        self.result.invalidate_range(self.n);
        (0..self.n).map(|i| self.result.load_local(i)).collect()
    }
}
