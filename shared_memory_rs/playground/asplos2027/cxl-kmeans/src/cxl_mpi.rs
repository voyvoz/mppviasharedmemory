use crate::cxl_shared::SharedKMeansState;
use crate::{
    DistributedKMeansResult, KMeansConfig, SplitMix64, assign_accumulate, init_random_centers,
    partition_range, synthetic_blobs, update_centers, validate_input,
};
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{get_machine_cnt, get_machine_id};
use shm_cxl::cxl::cxl_init;
use shm_cxl::utils::timestamps::{
    get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};
use std::time::Instant;

pub fn run_synthetic_kmeans_cxl(
    n_points: usize,
    dim: usize,
    blobs: usize,
    spread: f64,
    cfg: &KMeansConfig,
    expected_hosts: u16,
) -> DistributedKMeansResult {
    let points = synthetic_blobs(n_points, dim, blobs, spread, cfg.seed);
    run_kmeans_cxl(&points, dim, cfg, expected_hosts)
}

pub fn run_kmeans_cxl(
    points: &[f64],
    dim: usize,
    cfg: &KMeansConfig,
    expected_hosts: u16,
) -> DistributedKMeansResult {
    validate_input(points, dim, cfg);
    assert!(expected_hosts >= 2, "CXL mode requires at least 2 hosts");

    let n_points = points.len() / dim;
    let tol_sq = cfg.tol * cfg.tol;
    let mut persist = cxl_init(expected_hosts);

    let nhosts = get_machine_cnt() as usize;
    let machine_id = get_machine_id() as usize;
    let host_idx = machine_id - 1;
    let (start, end) = partition_range(n_points, host_idx, nhosts);
    let local_points = &points[start * dim..end * dim];
    let mut local_labels = vec![usize::MAX; end - start];

    // The leader's RNG must outlive seeding: empty-cluster reseeds in the
    // update step continue the same stream, keeping every variant on the
    // single-host trajectory.
    let mut rng = SplitMix64::new(cfg.seed);
    let shared = if machine_id == 1 {
        let shared = SharedKMeansState::create(persist.alloc.as_ref(), cfg.k, dim, nhosts);
        let centers = init_random_centers(points, n_points, dim, cfg.k, &mut rng);
        shared.initialize(&centers);

        let offset = shared.offset();
        for peer in 2..=expected_hosts {
            persist
                .ipc
                .try_send_to(peer, &Cacheline::from_b0(offset))
                .unwrap();
        }
        shared
    } else {
        let (_, msg) = persist.ipc.try_pop_from(1, None).unwrap();
        SharedKMeansState::attach(msg.d[0], cfg.k, dim, nhosts)
    };

    barrier_wait_all();
    // Clock starts once every host has attached, and stops when the iteration
    // loop ends -- CXL attach, seeding and result collection are excluded.
    reset_bench_metrics();
    let kernel_start = Instant::now();

    let mut local_centers = vec![0.0; cfg.k * dim];
    let mut global_sums = if machine_id == 1 {
        vec![0.0; cfg.k * dim]
    } else {
        Vec::new()
    };
    let mut global_counts = if machine_id == 1 {
        vec![0u64; cfg.k]
    } else {
        Vec::new()
    };

    for iter in 0..cfg.max_iters {
        shared.load_centers(&mut local_centers);

        let local = assign_accumulate(local_points, dim, &local_centers, cfg.k, &mut local_labels);
        shared.write_slot(
            host_idx,
            &local.sums,
            &local.counts,
            local.changed,
            local.inertia,
        );

        barrier_wait_all();

        if machine_id == 1 {
            let (global_changed, global_inertia) =
                shared.accumulate_partials(&mut global_sums, &mut global_counts);
            let max_shift_sq = update_centers(
                &mut local_centers,
                points,
                &global_sums,
                &global_counts,
                dim,
                cfg.k,
                &mut rng,
            );
            let converged = global_changed == 0 || max_shift_sq <= tol_sq;

            shared.store_centers(&local_centers);
            shared.store_global_counts(&global_counts);
            shared.store_global_inertia(global_inertia);
            shared.store_control((iter + 1) as u64, converged);
        }

        barrier_wait_all();

        let (iterations, converged) = shared.load_control();
        if converged || iterations >= cfg.max_iters {
            break;
        }
    }

    let kernel_elapsed = kernel_start.elapsed();
    let kernel_metrics = get_bench_metrics();
    print_bench_metrics_value(kernel_metrics);

    let mut centers = vec![0.0; cfg.k * dim];
    let mut counts_u64 = vec![0u64; cfg.k];
    shared.load_centers(&mut centers);
    shared.load_global_counts(&mut counts_u64);
    let inertia = shared.load_global_inertia();
    let (iterations, converged) = shared.load_control();

    DistributedKMeansResult {
        centers,
        counts: counts_u64.into_iter().map(|c| c as usize).collect(),
        inertia,
        iterations,
        converged,
        kernel_elapsed,
    }
}
