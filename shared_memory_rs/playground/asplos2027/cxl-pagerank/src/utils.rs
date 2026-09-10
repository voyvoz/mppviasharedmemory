use crate::primitives::{UNVISITED, filter_unvisited, gather_sum_f64};
use clap::Parser;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

const CHUNK: usize = 512;

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    #[arg(short, long, default_value_t = 0)]
    pub num_hosts: usize,

    #[arg(short, long, default_value_t = 0)]
    pub threads: usize,

    #[arg(long, default_value_t = 4096)]
    pub nodes: usize,

    #[arg(long, default_value_t = 8)]
    pub degree: usize,

    #[arg(long, default_value_t = 0)]
    pub source: u64,

    #[arg(long, default_value_t = 0xC0FFEE)]
    pub seed: u64,

    #[arg(long)]
    pub mtx: Option<String>,

    #[arg(long)]
    pub pregel: bool,

    #[arg(long)]
    pub partitioned: bool,

    /// Use the standard-network (MPI-style TCP) engine instead of CXL. Only
    /// applies to --algo bfs and pagerank. Requires --net-peers.
    #[arg(long)]
    pub net: bool,

    /// Comma-separated host:port list of all ranks, e.g.
    /// "dbcxla:9100,dbcxlb:9100,dbcxlc:9100". This rank is inferred from the
    /// local hostname unless --net-rank is given.
    #[arg(long)]
    pub net_peers: Option<String>,

    /// Explicit rank of this process within --net-peers (0-based).
    #[arg(long)]
    pub net_rank: Option<usize>,

    #[arg(long, default_value = "bfs")]
    pub algo: String,

    #[arg(long, default_value_t = 20)]
    pub iters: usize,

    #[arg(long, default_value_t = 0.85)]
    pub damping: f64,
}

pub fn resolve_threads(requested: usize) -> usize {
    if requested != 0 {
        requested
    } else {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    }
}

pub struct Csr {
    pub row_ptr: Vec<u64>,
    pub col_idx: Vec<u64>,
}

pub fn gen_csr(nodes: usize, degree: usize, seed: u64) -> Csr {
    let mut state = seed | 1;
    let mut next_rand = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 1
    };

    let mut adj: Vec<Vec<u64>> = vec![Vec::new(); nodes];
    fn add_edge(a: usize, b: usize, adj: &mut Vec<Vec<u64>>) {
        if a != b && !adj[a].contains(&(b as u64)) {
            adj[a].push(b as u64);
            adj[b].push(a as u64);
        }
    }

    for i in 0..nodes {
        add_edge(i, (i + 1) % nodes, &mut adj);
    }
    for i in 0..nodes {
        for _ in 0..degree {
            let j = (next_rand() as usize) % nodes;
            add_edge(i, j, &mut adj);
        }
    }

    let mut row_ptr = Vec::with_capacity(nodes + 1);
    let mut col_idx = Vec::new();
    row_ptr.push(0u64);
    for i in 0..nodes {
        for nb in &adj[i] {
            col_idx.push(*nb);
        }
        row_ptr.push(col_idx.len() as u64);
    }
    Csr { row_ptr, col_idx }
}

fn parse_mtx_err(path: &str, msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{path}: {msg}"))
}

struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

#[inline]
fn scan_u64(bytes: &[u8], pos: &mut usize) -> Option<u64> {
    let n = bytes.len();
    while *pos < n && !bytes[*pos].is_ascii_digit() {
        if bytes[*pos] == b'%' {
            return None;
        }
        *pos += 1;
    }
    if *pos >= n {
        return None;
    }
    let mut v = 0u64;
    while *pos < n && bytes[*pos].is_ascii_digit() {
        v = v * 10 + (bytes[*pos] - b'0') as u64;
        *pos += 1;
    }
    Some(v)
}

#[inline]
fn next_line_start(data: &[u8], mut p: usize) -> usize {
    let len = data.len();
    if p >= len {
        return len;
    }
    while p < len && data[p - 1] != b'\n' {
        p += 1;
    }
    p
}

/// Loads an undirected graph from a Matrix Market coordinate (.mtx) file
pub fn load_mtx(path: &str, threads: usize) -> io::Result<Csr> {
    let data = std::fs::read(path)?;
    let len = data.len();

    let mut pos = 0usize;
    let mut nodes = 0usize;
    let mut data_start = None;
    while pos < len {
        let line_end = match data[pos..].iter().position(|&c| c == b'\n') {
            Some(off) => pos + off,
            None => len,
        };
        let line = &data[pos..line_end];
        let trimmed = line.strip_prefix(b" ").unwrap_or(line);
        if !line.is_empty()
            && trimmed.first() != Some(&b'%')
            && line.iter().any(|c| c.is_ascii_digit())
        {
            let mut lp = 0usize;
            let rows =
                scan_u64(line, &mut lp).ok_or_else(|| parse_mtx_err(path, "bad row count"))?;
            let cols =
                scan_u64(line, &mut lp).ok_or_else(|| parse_mtx_err(path, "bad col count"))?;
            nodes = (rows.max(cols)) as usize;
            data_start = Some((line_end + 1).min(len));
            break;
        }
        pos = line_end + 1;
    }

    let data_start = data_start.ok_or_else(|| parse_mtx_err(path, "missing header line"))?;

    if nodes == 0 {
        return Ok(Csr {
            row_ptr: vec![0u64],
            col_idx: Vec::new(),
        });
    }

    let nthreads = threads.max(1);

    let mut bounds = Vec::with_capacity(nthreads + 1);
    let span = len - data_start;
    for k in 0..nthreads {
        bounds.push(next_line_start(&data, data_start + span * k / nthreads));
    }
    bounds.push(len);

    let deg: Vec<AtomicU32> = (0..nodes).map(|_| AtomicU32::new(0)).collect();
    let data_ref = &data;
    let deg_ref = &deg;
    let bounds_ref = &bounds;

    // Phase 1: parse edges and count degrees in parallel.
    let per_thread: Vec<Vec<(u32, u32)>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(nthreads);
        for t in 0..nthreads {
            handles.push(s.spawn(move || {
                let start = bounds_ref[t];
                let end = bounds_ref[t + 1];
                let mut edges: Vec<(u32, u32)> = Vec::new();
                for line in data_ref[start..end].split(|&c| c == b'\n') {
                    let mut lp = 0usize;
                    let a = match scan_u64(line, &mut lp) {
                        Some(v) => v,
                        None => continue,
                    };
                    let b = match scan_u64(line, &mut lp) {
                        Some(v) => v,
                        None => continue,
                    };
                    if a == 0 || b == 0 || a > nodes as u64 || b > nodes as u64 || a == b {
                        continue;
                    }
                    let a = (a - 1) as u32;
                    let b = (b - 1) as u32;
                    deg_ref[a as usize].fetch_add(1, Ordering::Relaxed);
                    deg_ref[b as usize].fetch_add(1, Ordering::Relaxed);
                    edges.push((a, b));
                }
                edges
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Prefix sum -> row_ptr, and per-node write cursors.
    let mut row_ptr = vec![0u64; nodes + 1];
    for i in 0..nodes {
        row_ptr[i + 1] = row_ptr[i] + deg[i].load(Ordering::Relaxed) as u64;
    }
    let total = row_ptr[nodes] as usize;
    let mut col_idx = vec![0u64; total];

    let cursor: Vec<AtomicU64> = (0..nodes).map(|i| AtomicU64::new(row_ptr[i])).collect();
    let cursor_ref = &cursor;
    let col_ptr = SendMutPtr(col_idx.as_mut_ptr());
    let per_thread_ref = &per_thread;

    // Phase 2: scatter both directions into col_idx (disjoint slots via atomic cursors).
    std::thread::scope(|s| {
        for t in 0..nthreads {
            let col_ptr = &col_ptr;
            s.spawn(move || {
                for &(a, b) in &per_thread_ref[t] {
                    let pa = cursor_ref[a as usize].fetch_add(1, Ordering::Relaxed) as usize;
                    unsafe { *col_ptr.0.add(pa) = b as u64 };
                    let pb = cursor_ref[b as usize].fetch_add(1, Ordering::Relaxed) as usize;
                    unsafe { *col_ptr.0.add(pb) = a as u64 };
                }
            });
        }
    });

    Ok(Csr { row_ptr, col_idx })
}

pub fn bfs_sequential(csr: &Csr, source: u64) -> Vec<u64> {
    let nodes = csr.row_ptr.len() - 1;
    let mut dist = vec![UNVISITED; nodes];
    let mut frontier = vec![source];
    dist[source as usize] = 0;
    let mut level = 0u64;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for node in frontier {
            let start = csr.row_ptr[node as usize] as usize;
            let end = csr.row_ptr[node as usize + 1] as usize;
            for e in start..end {
                let nb = csr.col_idx[e];
                if dist[nb as usize] == UNVISITED {
                    dist[nb as usize] = level + 1;
                    next.push(nb);
                }
            }
        }
        frontier = next;
        level += 1;
    }
    dist
}

/// Host-local parallel BFS without cxl
pub fn bfs_host_parallel(csr: &Csr, source: u64, threads: usize) -> Vec<u64> {
    let nodes = csr.row_ptr.len() - 1;
    let dist: Vec<AtomicU64> = (0..nodes).map(|_| AtomicU64::new(UNVISITED)).collect();
    dist[source as usize].store(0, Ordering::Relaxed);

    let mut cur = vec![source];
    while !cur.is_empty() {
        let cursor = AtomicUsize::new(0);
        let next: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let cur_ref = &cur;
        let dist_ref = &dist;
        let csr_ref = csr;
        let next_ref = &next;
        let cursor_ref = &cursor;

        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(move || {
                    let dist_ptr = dist_ref.as_ptr() as *const u64;
                    let col_ptr = csr_ref.col_idx.as_ptr();
                    let mut local: Vec<u64> = Vec::new();
                    let mut cand: Vec<u64> = Vec::new();
                    loop {
                        let start = cursor_ref.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= cur_ref.len() {
                            break;
                        }
                        let end = (start + CHUNK).min(cur_ref.len());
                        for idx in start..end {
                            let node = cur_ref[idx] as usize;
                            let nd = dist_ref[node].load(Ordering::Relaxed) + 1;
                            let rs = csr_ref.row_ptr[node] as usize;
                            let re = csr_ref.row_ptr[node + 1] as usize;
                            cand.clear();
                            filter_unvisited(
                                dist_ptr,
                                unsafe { col_ptr.add(rs) },
                                re - rs,
                                &mut cand,
                            );

                            // todo
                            for nb in &cand {
                                if dist_ref[*nb as usize]
                                    .compare_exchange(
                                        UNVISITED,
                                        nd,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                                {
                                    local.push(*nb);
                                }
                            }
                        }
                    }
                    next_ref.lock().unwrap().extend(local);
                });
            }
        });

        cur = next.into_inner().unwrap();
    }

    dist.into_iter().map(|a| a.into_inner()).collect()
}

#[inline]
fn deg_of(csr: &Csr, v: usize) -> usize {
    (csr.row_ptr[v + 1] - csr.row_ptr[v]) as usize
}

pub fn pagerank_sequential(csr: &Csr, iters: usize, damping: f64) -> Vec<f64> {
    let n = csr.row_ptr.len() - 1;
    let nf = n as f64;
    let base = (1.0 - damping) / nf;
    let mut rank = vec![1.0 / nf; n];
    let mut contrib = vec![0.0f64; n];
    for _ in 0..iters {
        for v in 0..n {
            let d = deg_of(csr, v);
            contrib[v] = if d > 0 { rank[v] / d as f64 } else { 0.0 };
        }
        for v in 0..n {
            let mut s = 0.0f64;
            for e in csr.row_ptr[v] as usize..csr.row_ptr[v + 1] as usize {
                s += contrib[csr.col_idx[e] as usize];
            }
            rank[v] = base + damping * s;
        }
    }
    rank
}

pub fn pagerank_host_parallel(csr: &Csr, iters: usize, damping: f64, threads: usize) -> Vec<f64> {
    let n = csr.row_ptr.len() - 1;
    let nf = n as f64;
    let base = (1.0 - damping) / nf;
    let nt = threads.max(1);

    let mut rank = vec![1.0 / nf; n];
    let mut rank2 = vec![0.0f64; n];
    let mut contrib = vec![0.0f64; n];

    for _ in 0..iters {
        // phase 1: contrib[v] = rank[v] / deg(v)
        {
            let c_addr = contrib.as_mut_ptr() as usize;
            let r_addr = rank.as_ptr() as usize;
            let csr_ref = &*csr;
            std::thread::scope(|s| {
                for t in 0..nt {
                    let a = n * t / nt;
                    let b = n * (t + 1) / nt;
                    s.spawn(move || {
                        let c = c_addr as *mut f64;
                        let r = r_addr as *const f64;
                        for v in a..b {
                            let d = deg_of(csr_ref, v);
                            let rv = unsafe { *r.add(v) };
                            let val = if d > 0 { rv / d as f64 } else { 0.0 };
                            unsafe { *c.add(v) = val };
                        }
                    });
                }
            });
        }
        // phase 2: rank2[v] = base + d * sum contrib[neighbors]
        {
            let out_addr = rank2.as_mut_ptr() as usize;
            let c_addr = contrib.as_ptr() as usize;
            let csr_ref = &*csr;
            std::thread::scope(|s| {
                for t in 0..nt {
                    let a = n * t / nt;
                    let b = n * (t + 1) / nt;
                    s.spawn(move || {
                        let out = out_addr as *mut f64;
                        let c = c_addr as *const f64;
                        let col = csr_ref.col_idx.as_ptr();
                        for v in a..b {
                            let rs = csr_ref.row_ptr[v] as usize;
                            let re = csr_ref.row_ptr[v + 1] as usize;
                            let s = gather_sum_f64(c, unsafe { col.add(rs) }, re - rs);
                            unsafe { *out.add(v) = base + damping * s };
                        }
                    });
                }
            });
        }
        std::mem::swap(&mut rank, &mut rank2);
    }
    rank
}

#[inline]
pub fn edge_weight(u: u64, v: u64) -> u64 {
    let (a, b) = if u < v { (u, v) } else { (v, u) };
    let mut h = a
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(b.wrapping_mul(0xC2B2AE3D27D4EB4F));
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 32;
    1 + (h % 16)
}

pub fn sssp_sequential(csr: &Csr, source: u64) -> Vec<u64> {
    let n = csr.row_ptr.len() - 1;
    let mut dist = vec![UNVISITED; n];
    dist[source as usize] = 0;
    let mut heap: BinaryHeap<Reverse<(u64, u64)>> = BinaryHeap::new();
    heap.push(Reverse((0, source)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u as usize] {
            continue;
        }
        let rs = csr.row_ptr[u as usize] as usize;
        let re = csr.row_ptr[u as usize + 1] as usize;
        for e in rs..re {
            let v = csr.col_idx[e];
            let nd = d + edge_weight(u, v);
            if nd < dist[v as usize] {
                dist[v as usize] = nd;
                heap.push(Reverse((nd, v)));
            }
        }
    }
    dist
}

pub fn sssp_host_parallel(csr: &Csr, source: u64, threads: usize) -> Vec<u64> {
    let n = csr.row_ptr.len() - 1;
    let dist: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(UNVISITED)).collect();
    dist[source as usize].store(0, Ordering::Relaxed);

    let mut cur = vec![source];
    while !cur.is_empty() {
        let cursor = AtomicUsize::new(0);
        let next: Mutex<Vec<u64>> = Mutex::new(Vec::new());
        let cur_ref = &cur;
        let dist_ref = &dist;
        let csr_ref = csr;
        let next_ref = &next;
        let cursor_ref = &cursor;

        std::thread::scope(|s| {
            for _ in 0..threads.max(1) {
                s.spawn(move || {
                    let mut local: Vec<u64> = Vec::new();
                    loop {
                        let start = cursor_ref.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= cur_ref.len() {
                            break;
                        }
                        let end = (start + CHUNK).min(cur_ref.len());
                        for idx in start..end {
                            let u = cur_ref[idx];
                            let du = dist_ref[u as usize].load(Ordering::Relaxed);
                            let rs = csr_ref.row_ptr[u as usize] as usize;
                            let re = csr_ref.row_ptr[u as usize + 1] as usize;
                            for e in rs..re {
                                let v = csr_ref.col_idx[e];
                                let cand = du + edge_weight(u, v);
                                let prev = dist_ref[v as usize].fetch_min(cand, Ordering::Relaxed);
                                if cand < prev {
                                    local.push(v);
                                }
                            }
                        }
                    }
                    next_ref.lock().unwrap().extend(local);
                });
            }
        });

        let mut nxt = next.into_inner().unwrap();
        nxt.sort_unstable();
        nxt.dedup();
        cur = nxt;
    }

    dist.into_iter().map(|a| a.into_inner()).collect()
}

pub fn sorted_adjacency(csr: &Csr) -> Vec<Vec<u64>> {
    let n = csr.row_ptr.len() - 1;
    let mut adj = Vec::with_capacity(n);
    for v in 0..n {
        let rs = csr.row_ptr[v] as usize;
        let re = csr.row_ptr[v + 1] as usize;
        let mut row: Vec<u64> = csr.col_idx[rs..re].to_vec();
        row.sort_unstable();
        row.dedup();
        adj.push(row);
    }
    adj
}

pub fn intersect_count_gt(a: &[u64], b: &[u64], bound: u64) -> u64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut c = 0u64;
    while i < a.len() && j < b.len() {
        let av = a[i];
        let bv = b[j];
        if av == bv {
            if av > bound {
                c += 1;
            }
            i += 1;
            j += 1;
        } else if av < bv {
            i += 1;
        } else {
            j += 1;
        }
    }
    c
}

pub fn triangle_sequential(csr: &Csr) -> u64 {
    let adj = sorted_adjacency(csr);
    let n = adj.len();
    let mut total = 0u64;
    for u in 0..n {
        for &v in &adj[u] {
            if v as usize > u {
                total += intersect_count_gt(&adj[u], &adj[v as usize], v);
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Distributed sort (sample sort) — keys are generated deterministically per
// global index so every rank and the sequential baseline agree.
// ---------------------------------------------------------------------------

#[inline]
pub fn sort_key_at(i: u64, seed: u64) -> u64 {
    let mut z = i.wrapping_add(seed).wrapping_mul(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Generate `count` keys for global indices `start..start+count`.
pub fn gen_keys_range(seed: u64, start: usize, count: usize) -> Vec<u64> {
    (0..count)
        .map(|k| sort_key_at((start + k) as u64, seed))
        .collect()
}

pub fn sort_sequential(n: usize, seed: u64) -> Vec<u64> {
    let mut v = gen_keys_range(seed, 0, n);
    v.sort_unstable();
    v
}

fn merge_two(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Host-local parallel sort: split into `threads` chunks, sort each on its own
/// thread, then merge the sorted runs pairwise in a parallel tree (each level's
/// merges run concurrently), avoiding a serial k-way merge bottleneck.
pub fn par_sort(mut v: Vec<u64>, threads: usize) -> Vec<u64> {
    let nt = threads.max(1);
    let len = v.len();
    if nt == 1 || len < 200_000 {
        v.sort_unstable();
        return v;
    }
    let chunk = len.div_ceil(nt);
    std::thread::scope(|s| {
        for c in v.chunks_mut(chunk) {
            s.spawn(move || c.sort_unstable());
        }
    });

    // Split the in-place-sorted buffer into owned runs, then tree-merge.
    let mut runs: Vec<Vec<u64>> = Vec::new();
    let mut rest = v;
    while !rest.is_empty() {
        let take = chunk.min(rest.len());
        let tail = rest.split_off(take);
        runs.push(rest);
        rest = tail;
    }

    while runs.len() > 1 {
        runs = std::thread::scope(|s| {
            let mut handles = Vec::new();
            let mut it = runs.into_iter();
            loop {
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => handles.push(s.spawn(move || merge_two(&a, &b))),
                    (Some(a), None) => handles.push(s.spawn(move || a)),
                    (None, _) => break,
                }
            }
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
    }
    runs.pop().unwrap_or_default()
}

fn kway_merge(runs: Vec<Vec<u64>>) -> Vec<u64> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let total: usize = runs.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0usize; runs.len()];
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for (r, run) in runs.iter().enumerate() {
        if !run.is_empty() {
            heap.push(Reverse((run[0], r)));
        }
    }
    while let Some(Reverse((v, r))) = heap.pop() {
        out.push(v);
        idx[r] += 1;
        if idx[r] < runs[r].len() {
            heap.push(Reverse((runs[r][idx[r]], r)));
        }
    }
    out
}

pub fn sort_host_parallel(n: usize, seed: u64, threads: usize) -> Vec<u64> {
    let nt = threads.max(1);
    let runs: Vec<Vec<u64>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..nt)
            .map(|t| {
                let lo = n * t / nt;
                let hi = n * (t + 1) / nt;
                s.spawn(move || {
                    let mut v = gen_keys_range(seed, lo, hi - lo);
                    v.sort_unstable();
                    v
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    kway_merge(runs)
}

pub fn triangle_host_parallel(csr: &Csr, threads: usize) -> u64 {
    let adj = sorted_adjacency(csr);
    let n = adj.len();
    let total = AtomicU64::new(0);
    let cursor = AtomicUsize::new(0);
    let adj_ref = &adj;
    let total_ref = &total;
    let cursor_ref = &cursor;

    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(move || {
                let mut local = 0u64;
                loop {
                    let start = cursor_ref.fetch_add(CHUNK, Ordering::Relaxed);
                    if start >= n {
                        break;
                    }
                    let end = (start + CHUNK).min(n);
                    for u in start..end {
                        for &v in &adj_ref[u] {
                            if v as usize > u {
                                local += intersect_count_gt(&adj_ref[u], &adj_ref[v as usize], v);
                            }
                        }
                    }
                }
                total_ref.fetch_add(local, Ordering::Relaxed);
            });
        }
    });

    total.into_inner()
}
