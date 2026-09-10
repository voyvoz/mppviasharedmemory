use crate::net::{NetComm, deser_u64, for_each_pair, ser_pairs, ser_u64};
use crate::utils::Csr;

fn round_up8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

fn ser_f64(v: &[f64]) -> Vec<u8> {
    ser_u64(&v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>())
}
fn deser_f64(b: &[u8]) -> Vec<f64> {
    deser_u64(b).into_iter().map(f64::from_bits).collect()
}

// Partitioned, Pregel-style PageRank over standard TCP. Each rank owns a vertex
// block, scatters contributions to neighbor owners once per iteration, and
// applies received contributions. Returns the full rank vector on rank 0.
pub fn net_pagerank(comm: &mut NetComm, csr: &Csr, iters: usize, damping: f64) -> Option<Vec<f64>> {
    let n = csr.row_ptr.len() - 1;
    let nranks = comm.nranks;
    let me = comm.rank;
    let block = round_up8(n.div_ceil(nranks).max(1));
    let owner = |v: u64| (v as usize / block).min(nranks - 1);
    let lo = me * block;
    let hi = ((me + 1) * block).min(n);

    let nf = n as f64;
    let base = (1.0 - damping) / nf;

    let mut rank = vec![0.0f64; n];
    for v in lo..hi {
        rank[v] = 1.0 / nf;
    }
    let mut acc = vec![0.0f64; n];
    let mut touched: Vec<u64> = Vec::new();

    for _ in 0..iters {
        touched.clear();
        for u in lo..hi {
            let rs = csr.row_ptr[u] as usize;
            let re = csr.row_ptr[u + 1] as usize;
            let d = re - rs;
            if d == 0 {
                continue;
            }
            let c = rank[u] / d as f64;
            for e in rs..re {
                let v = csr.col_idx[e] as usize;
                if acc[v] == 0.0 {
                    touched.push(v as u64);
                }
                acc[v] += c;
            }
        }

        let mut out: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nranks];
        let mut inc = vec![0.0f64; n];
        for &vv in &touched {
            let v = vv as usize;
            let h = owner(vv);
            if h == me {
                inc[v] += acc[v];
            } else {
                out[h].push((vv, acc[v]));
            }
        }

        let outb: Vec<Vec<u8>> = out.iter().map(|o| ser_pairs(o)).collect();
        let inb = comm.exchange(&outb);
        for j in 0..nranks {
            if j == me {
                continue;
            }
            for_each_pair(&inb[j], |v, val| inc[v as usize] += val);
        }

        for v in lo..hi {
            rank[v] = base + damping * inc[v];
        }
        for &vv in &touched {
            acc[vv as usize] = 0.0;
        }
    }

    let owned = ser_f64(&rank[lo..hi]);
    let parts = comm.gather_to_root(&owned);
    if me == 0 {
        let mut full = vec![0.0f64; n];
        for r in 0..nranks {
            let rlo = r * block;
            for (k, val) in deser_f64(&parts[r]).into_iter().enumerate() {
                if rlo + k < n {
                    full[rlo + k] = val;
                }
            }
        }
        Some(full)
    } else {
        None
    }
}
