use std::time::Instant;

use mpi::collective::SystemOperation;
use mpi::traits::*;

use shm_cxl::utils::timestamps::{
    IpCxlMetricsGuard, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

use crate::{
    DistributedKMeansResult, KMeansConfig, SplitMix64, assign_accumulate, init_random_centers,
    partition_range, synthetic_blobs, update_centers, validate_input,
};

pub fn run_synthetic_kmeans_mpi(
    n_points: usize,
    dim: usize,
    blobs: usize,
    spread: f64,
    cfg: &KMeansConfig,
    expect_ranks: u16,
) -> (DistributedKMeansResult, i32) {
    let points = synthetic_blobs(n_points, dim, blobs, spread, cfg.seed);
    run_kmeans_mpi(&points, dim, cfg, expect_ranks)
}

pub fn run_kmeans_mpi(
    points: &[f64],
    dim: usize,
    cfg: &KMeansConfig,
    expect_ranks: u16,
) -> (DistributedKMeansResult, i32) {
    validate_input(points, dim, cfg);

    let n_points = points.len() / dim;
    let tol_sq = cfg.tol * cfg.tol;

    let universe = mpi::initialize().expect("MPI failed to initialize");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    if expect_ranks != 0 {
        assert_eq!(
            size as u16, expect_ranks,
            "mpi_dram: expected {expect_ranks} ranks, got {size} -- check mpirun -np/-host"
        );
    }

    let (start, end) = partition_range(n_points, rank as usize, size as usize);
    let local_points = &points[start * dim..end * dim];
    let mut local_labels = vec![usize::MAX; end - start];

    let root_process = world.process_at_rank(0);
    let mut local_centers = vec![0.0; cfg.k * dim];
    // Rank 0's RNG must outlive seeding: empty-cluster reseeds in the update
    // step continue the same stream, keeping every variant on the single-host
    // trajectory.
    let mut rng = SplitMix64::new(cfg.seed);
    if rank == 0 {
        local_centers = init_random_centers(points, n_points, dim, cfg.k, &mut rng);
    }
    root_process.broadcast_into(&mut local_centers);

    world.barrier();
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    reset_bench_metrics();
    let kernel_start = Instant::now();

    let packed_len = cfg.k * dim + cfg.k + 2;
    let counts_offset = cfg.k * dim;
    let changed_offset = counts_offset + cfg.k;
    let inertia_offset = changed_offset + 1;

    let mut global_counts = vec![0u64; cfg.k];
    let mut global_inertia = 0.0f64;
    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..cfg.max_iters {
        let local = assign_accumulate(local_points, dim, &local_centers, cfg.k, &mut local_labels);

        let mut send_buf = vec![0.0f64; packed_len];
        send_buf[..counts_offset].copy_from_slice(&local.sums);
        for (i, &c) in local.counts.iter().enumerate() {
            send_buf[counts_offset + i] = c as f64;
        }
        send_buf[changed_offset] = local.changed as f64;
        send_buf[inertia_offset] = local.inertia;

        let mut bcast_buf = vec![0.0f64; cfg.k * dim + 1];

        if rank == 0 {
            let mut recv_buf = vec![0.0f64; packed_len];
            root_process.reduce_into_root(&send_buf, &mut recv_buf, SystemOperation::sum());

            global_counts = recv_buf[counts_offset..changed_offset]
                .iter()
                .map(|&c| c as u64)
                .collect();
            let global_changed = recv_buf[changed_offset] as u64;
            global_inertia = recv_buf[inertia_offset];

            let max_shift_sq = update_centers(
                &mut local_centers,
                points,
                &recv_buf[..counts_offset],
                &global_counts,
                dim,
                cfg.k,
                &mut rng,
            );
            converged = global_changed == 0 || max_shift_sq <= tol_sq;

            bcast_buf[..cfg.k * dim].copy_from_slice(&local_centers);
            bcast_buf[cfg.k * dim] = if converged { 1.0 } else { 0.0 };
        } else {
            root_process.reduce_into(&send_buf, SystemOperation::sum());
        }

        root_process.broadcast_into(&mut bcast_buf);
        local_centers.copy_from_slice(&bcast_buf[..cfg.k * dim]);
        converged = bcast_buf[cfg.k * dim] != 0.0;

        iterations = iter + 1;
        if converged {
            break;
        }
    }

    let kernel_elapsed = kernel_start.elapsed();
    let kernel_metrics = get_bench_metrics();
    // Over the IP-over-CXL bridge the TUN device owns the CXL traffic, so its
    // snapshot replaces this process' (empty) direct-CXL counters.
    match ip_cxl_metrics.finish() {
        Some(metrics) => print_bench_metrics_value(metrics),
        None => print_bench_metrics_value(kernel_metrics),
    }

    let mut final_buf = vec![0.0f64; cfg.k + 1];
    if rank == 0 {
        for (i, &c) in global_counts.iter().enumerate() {
            final_buf[i] = c as f64;
        }
        final_buf[cfg.k] = global_inertia;
    }
    root_process.broadcast_into(&mut final_buf);

    (
        DistributedKMeansResult {
            centers: local_centers,
            counts: final_buf[..cfg.k].iter().map(|&c| c as usize).collect(),
            inertia: final_buf[cfg.k],
            iterations,
            converged,
            kernel_elapsed,
        },
        rank,
    )
}
