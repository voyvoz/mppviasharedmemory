pub fn idx(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

pub fn checksum(c: &[f64]) -> f64 {
    c.iter().sum()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RowPartition {
    pub row_begin: usize,
    pub row_end: usize,
    pub local_rows: usize,
}

impl RowPartition {
    pub(crate) fn new(total_rows: usize, part_idx: usize, workers: usize) -> Self {
        assert!(workers > 0);
        assert!(part_idx < workers);

        let row_begin = part_idx * total_rows / workers;
        let row_end = (part_idx + 1) * total_rows / workers;
        let local_rows = row_end - row_begin;

        RowPartition {
            row_begin,
            row_end,
            local_rows,
        }
    }
}

/// A: m x k, B: k x n and C: m x n
pub fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);

    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

pub fn make_matrix(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|idx| {
            let i = idx / cols;
            let j = idx % cols;
            (i + 2 * j) as f64 * 0.5
        })
        .collect()
}

/// Each worker computes rows [row_begin, row_end) of C.
/// a_block only contains those rows of A (row_end - row_begin rows, k cols each).
pub(crate) fn matmul_block(a_block: &[f64], b: &[f64], c_block: &mut [f64], k: usize, n: usize) {
    let rows = c_block.len() / n;
    for i in 0..rows {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_block[i * k + p] * b[p * n + j];
            }
            c_block[i * n + j] = sum;
        }
    }
}

/// Tile width for [`matmul_block_tiled`]: chosen so a `TILE x TILE` block of
/// `f64` (64*64*8 = 32 KiB) comfortably fits in a core's L1/L2 cache and can
/// be reused across every row of `a_block` before being evicted.
const MATMUL_TILE: usize = 64;

/// Same contract as [`matmul_block`], but blocks the multiply over (p, j)
/// tiles of `b` so each tile is pulled into cache once and reused across
/// every row of `a_block`/`c_block`, instead of re-streaming the full width
/// of `b` from cache/memory once per output row. That re-streaming is what
/// makes the untiled kernel fall off a cache-conflict cliff whenever the
/// row stride lines up with the cache geometry (e.g. `n` a power of two).
pub(crate) fn matmul_block_tiled(
    a_block: &[f64],
    b: &[f64],
    c_block: &mut [f64],
    k: usize,
    n: usize,
) {
    let rows = c_block.len() / n;
    c_block.fill(0.0);

    for p0 in (0..k).step_by(MATMUL_TILE) {
        let p_end = (p0 + MATMUL_TILE).min(k);
        for j0 in (0..n).step_by(MATMUL_TILE) {
            let j_end = (j0 + MATMUL_TILE).min(n);
            for i in 0..rows {
                let c_row = &mut c_block[i * n + j0..i * n + j_end];
                for p in p0..p_end {
                    let a_val = a_block[i * k + p];
                    let b_row = &b[p * n + j0..p * n + j_end];
                    for (c_val, &b_val) in c_row.iter_mut().zip(b_row) {
                        *c_val += a_val * b_val;
                    }
                }
            }
        }
    }
}

pub fn matmul_threads(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    num_threads: usize,
) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);

    let mut c = vec![0.0; m * n];
    let workers = num_threads.max(1).min(m.max(1));
    let block_rows = m.div_ceil(workers.max(1));

    std::thread::scope(|s| {
        for (t, c_block) in c.chunks_mut(block_rows * n).enumerate() {
            let row_begin = t * block_rows;
            let rows_this = c_block.len() / n;
            let a_block = &a[row_begin * k..(row_begin + rows_this) * k];

            s.spawn(move || {
                matmul_block(a_block, b, c_block, k, n);
            });
        }
    });

    c
}
