use std::time::Instant;

use mpi::datatype::{Partition, PartitionMut};
use mpi::{Count, traits::*};

use shm_cxl::utils::timestamps::{
    IpCxlMetricsGuard, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

use crate::DistRunResult;
use crate::matmul_core::{RowPartition, make_matrix, matmul_block};

fn counts_displs(total_rows: usize, cols: usize, nranks: usize) -> (Vec<Count>, Vec<Count>) {
    let counts: Vec<Count> = (0..nranks)
        .map(|rank| {
            let part = RowPartition::new(total_rows, rank, nranks);
            (part.local_rows * cols) as Count
        })
        .collect();

    let mut displs = Vec::with_capacity(nranks);
    let mut acc = 0;
    for &count in &counts {
        displs.push(acc);
        acc += count;
    }

    (counts, displs)
}

fn scatter_a_rows(
    root: &mpi::topology::Process<'_>,
    rank: i32,
    a: Option<&[f64]>,
    local_a: &mut [f64],
    m: usize,
    k: usize,
    nranks: usize,
) {
    let (counts, displs) = counts_displs(m, k, nranks);
    if rank == 0 {
        let a = a.expect("root rank must provide matrix A");
        let part = Partition::new(a, counts, displs);
        root.scatter_varcount_into_root(&part, local_a);
    } else {
        root.scatter_varcount_into(local_a);
    }
}

fn gather_c_rows(
    root: &mpi::topology::Process<'_>,
    rank: i32,
    local_c: &[f64],
    m: usize,
    n: usize,
    nranks: usize,
) -> Option<Vec<f64>> {
    let (counts, displs) = counts_displs(m, n, nranks);
    if rank == 0 {
        let mut c = vec![0.0; m * n];
        let mut part = PartitionMut::new(&mut c[..], counts, displs);
        root.gather_varcount_into_root(local_c, &mut part);
        Some(c)
    } else {
        root.gather_varcount_into(local_c);
        None
    }
}

pub fn run_mpi_dist(
    world: &mpi::topology::SimpleCommunicator,
    m: usize,
    k: usize,
    n: usize,
    expected_ranks: u16,
) -> DistRunResult {
    run_mpi_dist_timed(world, m, k, n, expected_ranks)
}

pub fn run_mpi_dist_timed(
    world: &mpi::topology::SimpleCommunicator,
    m: usize,
    k: usize,
    n: usize,
    expected_ranks: u16,
) -> DistRunResult {
    let init_start = Instant::now();

    let rank = world.rank();
    let size = world.size();

    assert_eq!(
        size as u16, expected_ranks,
        "mpi-dist expected {expected_ranks} ranks, got {size}"
    );

    let nranks = size as usize;
    let part = RowPartition::new(m, rank as usize, nranks);
    let root = world.process_at_rank(0);

    // Unlike the kernels whose MPI traffic happens inside the timed kernel,
    // this mode moves A/B in init and C in collect, so the IP-over-CXL bridge
    // counters must span the whole distributed phase to capture it.
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    reset_bench_metrics();

    let a = (rank == 0).then(|| make_matrix(m, k));
    let mut b = if rank == 0 {
        make_matrix(k, n)
    } else {
        vec![0.0; k * n]
    };
    let mut local_a = vec![0.0; part.local_rows * k];

    scatter_a_rows(&root, rank, a.as_deref(), &mut local_a, m, k, nranks);
    root.broadcast_into(&mut b[..]);

    world.barrier();
    let init_elapsed = init_start.elapsed();

    let compute_start = Instant::now();
    let mut local_c = vec![0.0; part.local_rows * n];
    matmul_block(&local_a, &b, &mut local_c, k, n);
    world.barrier();
    let compute_elapsed = compute_start.elapsed();

    let collect_start = Instant::now();
    let result = gather_c_rows(&root, rank, &local_c, m, n, nranks);
    let collect_elapsed = collect_start.elapsed();

    let kernel_metrics = get_bench_metrics();
    match ip_cxl_metrics.finish() {
        Some(metrics) => print_bench_metrics_value(metrics),
        None => print_bench_metrics_value(kernel_metrics),
    }

    DistRunResult {
        init_elapsed,
        compute_elapsed,
        collect_elapsed,
        result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_partition_tiles_rows() {
        let parts: Vec<_> = (0..3).map(|rank| RowPartition::new(10, rank, 3)).collect();

        assert_eq!(parts[0].row_begin, 0);
        assert_eq!(parts[0].row_end, 3);
        assert_eq!(parts[1].row_begin, 3);
        assert_eq!(parts[1].row_end, 6);
        assert_eq!(parts[2].row_begin, 6);
        assert_eq!(parts[2].row_end, 10);
    }

    #[test]
    fn counts_follow_row_partition() {
        let (counts, displs) = counts_displs(5, 4, 3);

        assert_eq!(counts, vec![4, 8, 8]);
        assert_eq!(displs, vec![0, 4, 12]);
    }
}
