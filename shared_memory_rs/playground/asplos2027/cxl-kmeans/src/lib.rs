pub mod cxl_ipc;
pub mod cxl_mpi;
mod cxl_shared;
pub mod mpi_dram;

use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub use cxl_ipc::{run_kmeans_cxl_ipc, run_synthetic_kmeans_cxl_ipc};
pub use cxl_mpi::{run_kmeans_cxl, run_synthetic_kmeans_cxl};
pub use mpi_dram::{run_kmeans_mpi, run_synthetic_kmeans_mpi};

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iters: usize,
    pub seed: u64,
    pub tol: f64,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        KMeansConfig {
            k: 2,
            max_iters: 100,
            seed: 0xC0FFEE,
            tol: 1e-4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KMeansResult {
    pub centers: Vec<f64>,
    pub labels: Vec<usize>,
    pub counts: Vec<usize>,
    pub inertia: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct DistributedKMeansResult {
    pub centers: Vec<f64>,
    pub counts: Vec<usize>,
    pub inertia: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Time spent in the iteration loop alone, excluding CXL attach, seeding
    /// and result collection.
    pub kernel_elapsed: std::time::Duration,
}

#[derive(Debug, Clone)]
pub(crate) struct LocalAssignmentStats {
    pub sums: Vec<f64>,
    pub counts: Vec<u64>,
    pub changed: u64,
    pub inertia: f64,
}

#[derive(Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub(crate) fn gen_range_usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        (self.next_u64() as usize) % upper
    }

    fn gen_range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

pub fn synthetic_blobs(
    n_points: usize,
    dim: usize,
    blobs: usize,
    spread: f64,
    seed: u64,
) -> Vec<f64> {
    assert!(n_points > 0, "n_points must be > 0");
    assert!(dim > 0, "dim must be > 0");
    assert!(blobs > 0, "blobs must be > 0");
    assert!(
        spread >= 0.0 && spread.is_finite(),
        "spread must be finite and >= 0"
    );

    let mut rng = SplitMix64::new(seed);
    let mut blob_centers = vec![0.0; blobs * dim];
    for c in 0..blobs {
        for d in 0..dim {
            blob_centers[c * dim + d] = rng.gen_range_f64(-20.0, 20.0);
        }
    }

    let mut points = vec![0.0; n_points * dim];
    for i in 0..n_points {
        let cluster = i % blobs;
        for d in 0..dim {
            let base = blob_centers[cluster * dim + d];
            let noise = rng.gen_range_f64(-spread, spread);
            points[i * dim + d] = base + noise;
        }
    }

    points
}

pub fn kmeans(points: &[f64], dim: usize, cfg: &KMeansConfig) -> KMeansResult {
    validate_input(points, dim, cfg);

    let n_points = points.len() / dim;
    let mut rng = SplitMix64::new(cfg.seed);
    let mut centers = init_random_centers(points, n_points, dim, cfg.k, &mut rng);
    let mut labels = vec![usize::MAX; n_points];
    let mut iterations = 0usize;
    let mut converged = false;
    let tol_sq = cfg.tol * cfg.tol;

    for iter in 0..cfg.max_iters {
        let local = assign_accumulate(points, dim, &centers, cfg.k, &mut labels);
        let max_shift_sq = update_centers(
            &mut centers,
            points,
            &local.sums,
            &local.counts,
            dim,
            cfg.k,
            &mut rng,
        );

        iterations = iter + 1;
        if local.changed == 0 || max_shift_sq <= tol_sq {
            converged = true;
            break;
        }
    }

    let (labels, counts, inertia) = assign_labels(points, dim, &centers, cfg.k);

    KMeansResult {
        centers,
        labels,
        counts,
        inertia,
        iterations,
        converged,
    }
}

pub(crate) fn validate_input(points: &[f64], dim: usize, cfg: &KMeansConfig) {
    assert!(dim > 0, "dim must be > 0");
    assert!(!points.is_empty(), "points must not be empty");
    assert!(
        points.len().is_multiple_of(dim),
        "points len must be divisible by dim"
    );
    assert!(cfg.k > 0, "k must be > 0");
    assert!(cfg.max_iters > 0, "max_iters must be > 0");
    assert!(
        cfg.tol >= 0.0 && cfg.tol.is_finite(),
        "tol must be finite and >= 0"
    );

    let n_points = points.len() / dim;
    assert!(cfg.k <= n_points, "k must be <= number of points");
}

pub(crate) fn partition_range(total: usize, rank: usize, size: usize) -> (usize, usize) {
    let base = total / size;
    let rem = total % size;
    let start = rank * base + rank.min(rem);
    let len = base + usize::from(rank < rem);
    (start, start + len)
}

pub(crate) fn init_random_centers(
    points: &[f64],
    n_points: usize,
    dim: usize,
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<f64> {
    let mut centers = vec![0.0; k * dim];
    let mut chosen = vec![false; n_points];

    for cluster in 0..k {
        let idx = loop {
            let candidate = rng.gen_range_usize(n_points);
            if !chosen[candidate] {
                chosen[candidate] = true;
                break candidate;
            }
        };
        centers[cluster * dim..(cluster + 1) * dim].copy_from_slice(point_at(points, dim, idx));
    }

    centers
}

pub(crate) fn assign_accumulate(
    points: &[f64],
    dim: usize,
    centers: &[f64],
    k: usize,
    labels: &mut [usize],
) -> LocalAssignmentStats {
    let n_points = points.len() / dim;
    let mut sums = vec![0.0; k * dim];
    let mut counts = vec![0u64; k];
    let mut changed = 0u64;
    let mut inertia = 0.0f64;

    for (i, label) in labels.iter_mut().enumerate().take(n_points) {
        let point = point_at(points, dim, i);
        let (cluster, dist_sq) = nearest_center_with_distance(point, centers, dim, k);
        if *label != cluster {
            *label = cluster;
            changed += 1;
        }
        counts[cluster] += 1;
        inertia += dist_sq;

        let offset = cluster * dim;
        for d in 0..dim {
            sums[offset + d] += point[d];
        }
    }

    LocalAssignmentStats {
        sums,
        counts,
        changed,
        inertia,
    }
}

/// One centers update, shared by the single-host path and every distributed
/// leader. An empty cluster is reseeded with a random point; every leader
/// holds the full point set and the same seeded RNG, so all variants walk the
/// identical trajectory instead of the distributed ones silently dropping
/// dead clusters (which left census1990 at a 5.5x worse inertia than the
/// single-host run).
pub(crate) fn update_centers(
    centers: &mut [f64],
    points: &[f64],
    sums: &[f64],
    counts: &[u64],
    dim: usize,
    k: usize,
    rng: &mut SplitMix64,
) -> f64 {
    let n_points = points.len() / dim;
    let mut max_shift_sq = 0.0f64;

    for (cluster, &count) in counts.iter().enumerate().take(k) {
        let offset = cluster * dim;
        let mut shift_sq = 0.0f64;

        if count == 0 {
            let idx = rng.gen_range_usize(n_points);
            let point = point_at(points, dim, idx);
            for d in 0..dim {
                let old = centers[offset + d];
                let new = point[d];
                let diff = old - new;
                shift_sq += diff * diff;
                centers[offset + d] = new;
            }
        } else {
            let inv = 1.0 / count as f64;
            for d in 0..dim {
                let old = centers[offset + d];
                let new = sums[offset + d] * inv;
                let diff = old - new;
                shift_sq += diff * diff;
                centers[offset + d] = new;
            }
        }

        if shift_sq > max_shift_sq {
            max_shift_sq = shift_sq;
        }
    }

    max_shift_sq
}

pub(crate) fn assign_labels(
    points: &[f64],
    dim: usize,
    centers: &[f64],
    k: usize,
) -> (Vec<usize>, Vec<usize>, f64) {
    let n_points = points.len() / dim;
    let mut labels = vec![0usize; n_points];
    let mut counts = vec![0usize; k];
    let mut inertia = 0.0f64;

    for (i, label) in labels.iter_mut().enumerate().take(n_points) {
        let point = point_at(points, dim, i);
        let (best_cluster, best_dist) = nearest_center_with_distance(point, centers, dim, k);
        *label = best_cluster;
        counts[best_cluster] += 1;
        inertia += best_dist;
    }

    (labels, counts, inertia)
}

pub(crate) fn nearest_center_with_distance(
    point: &[f64],
    centers: &[f64],
    dim: usize,
    k: usize,
) -> (usize, f64) {
    let mut best_cluster = 0usize;
    let mut best_dist = f64::INFINITY;
    for cluster in 0..k {
        let center = &centers[cluster * dim..(cluster + 1) * dim];
        let d2 = squared_distance(point, center);
        if d2 < best_dist {
            best_dist = d2;
            best_cluster = cluster;
        }
    }
    (best_cluster, best_dist)
}

pub(crate) fn point_at(points: &[f64], dim: usize, idx: usize) -> &[f64] {
    let start = idx * dim;
    &points[start..start + dim]
}

pub(crate) fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

/// Loads a dense point matrix from a delimited text file (CSV or whitespace
/// separated, one point per line), as shipped by the UCI datasets.
///
/// `skip_rows` drops leading lines (a CSV header), `skip_cols` drops that many
/// leading fields per line (HIGGS puts its label first), `take_cols` keeps that
/// many of the remaining ones (0 = all, which is what covtype wants once its
/// trailing class column is excluded via `take_cols`), and `max_points`
/// truncates the input (0 = no limit). Every host reads the same file and
/// derives the same point set, so the CXL and IPC engines partition an
/// identical dataset.
///
/// Returns the flattened row-major points and the dimensionality read from the
/// first data line.
pub fn load_points(
    path: &str,
    skip_rows: usize,
    skip_cols: usize,
    take_cols: usize,
    max_points: usize,
) -> io::Result<(Vec<f64>, usize)> {
    let reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut points: Vec<f64> = Vec::new();
    let mut dim = 0usize;
    let mut n_points = 0usize;

    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if lineno < skip_rows || line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }

        let before = points.len();
        for field in line
            .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
            .filter(|f| !f.is_empty())
            .skip(skip_cols)
            .take(if take_cols == 0 {
                usize::MAX
            } else {
                take_cols
            })
        {
            let value: f64 = field.parse().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{path}:{}: not a number: {field:?}", lineno + 1),
                )
            })?;
            points.push(value);
        }

        let got = points.len() - before;
        if got == 0 {
            points.truncate(before);
            continue;
        }
        if dim == 0 {
            dim = got;
        } else if got != dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{path}:{}: got {got} columns, expected {dim}", lineno + 1),
            ));
        }

        n_points += 1;
        if max_points != 0 && n_points >= max_points {
            break;
        }
    }

    if dim == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: no data rows"),
        ));
    }

    Ok((points, dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_clusters() {
        let points = vec![
            1.0, 1.0, //
            1.0, 2.0, //
            2.0, 1.0, //
            8.0, 8.0, //
            9.0, 8.0, //
            8.0, 9.0, //
        ];

        let cfg = KMeansConfig {
            k: 2,
            max_iters: 50,
            seed: 7,
            tol: 1e-8,
        };
        let result = kmeans(&points, 2, &cfg);

        assert!(result.converged);
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.counts.iter().sum::<usize>(), 6);
        assert!(result.counts.iter().all(|&c| c == 3));
    }

    #[test]
    fn synthetic_points_shape_is_correct() {
        let points = synthetic_blobs(20, 3, 2, 0.2, 11);
        assert_eq!(points.len(), 20 * 3);
    }
}
