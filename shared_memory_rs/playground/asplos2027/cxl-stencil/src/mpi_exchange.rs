use std::time::{Duration, Instant};

use mpi::datatype::PartitionMut;
use mpi::traits::*;

use shm_cxl::utils::timestamps::{
    IpCxlMetricsGuard, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

use crate::stencil_core::{LocalGrid, RowPartition, init_local_grids, update_local_block};

const TAG_UPWARD: i32 = 1;
const TAG_DOWNWARD: i32 = 2;

fn exchange_halos(world: &mpi::topology::SimpleCommunicator, part: &RowPartition, old: &mut LocalGrid) {
    let (first_row, last_row, top_halo, bottom_halo) = old.split_for_exchange();

    // All four operations touch disjoint memory and independent peers, so post
    // them as one non-blocking batch instead of two sequential send+recv phases
    // (each of which used to wait() before the other could even start).
    mpi::request::scope(|scope| {
        let recv_top = part.up_peer.map(|peer| {
            world
                .process_at_rank(peer as i32)
                .immediate_receive_into_with_tag(scope, top_halo, TAG_DOWNWARD)
        });
        let recv_bottom = part.down_peer.map(|peer| {
            world
                .process_at_rank(peer as i32)
                .immediate_receive_into_with_tag(scope, bottom_halo, TAG_UPWARD)
        });
        let send_up = part.up_peer.map(|peer| {
            world
                .process_at_rank(peer as i32)
                .immediate_send_with_tag(scope, first_row, TAG_UPWARD)
        });
        let send_down = part.down_peer.map(|peer| {
            world
                .process_at_rank(peer as i32)
                .immediate_send_with_tag(scope, last_row, TAG_DOWNWARD)
        });

        if let Some(req) = recv_top {
            req.wait();
        }
        if let Some(req) = recv_bottom {
            req.wait();
        }
        if let Some(req) = send_up {
            req.wait();
        }
        if let Some(req) = send_down {
            req.wait();
        }
    });
}

fn gather_result(
    world: &mpi::topology::SimpleCommunicator,
    rows: usize,
    cols: usize,
    part: &RowPartition,
    old: &LocalGrid,
) -> Option<Vec<f64>> {
    let rank = world.rank();
    let size = world.size();
    let root = world.process_at_rank(0);
    let sendbuf = old.owned_rows();

    if rank == 0 {
        let mut grid = vec![0.0; rows * cols];
        let counts: Vec<mpi::Count> = (0..size)
            .map(|peer| {
                let peer_part = RowPartition::new(rows, peer as usize, size as usize);
                (peer_part.local_rows * cols) as mpi::Count
            })
            .collect();
        let displs: Vec<mpi::Count> = (0..size)
            .map(|peer| {
                let peer_part = RowPartition::new(rows, peer as usize, size as usize);
                (peer_part.row_begin * cols) as mpi::Count
            })
            .collect();

        let mut partition = PartitionMut::new(&mut grid[..], counts, &displs[..]);
        root.gather_varcount_into_root(sendbuf, &mut partition);
        Some(grid)
    } else {
        let _ = part;
        root.gather_varcount_into(sendbuf);
        None
    }

}


pub fn run_mpi_exchange(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_ranks: u16,
) -> Option<Vec<f64>> {
    run_mpi_exchange_timed(rows, cols, steps, expected_ranks).0
}


pub fn run_mpi_exchange_timed(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_ranks: u16,
) -> (Option<Vec<f64>>, Duration) {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    assert_eq!(
        size as u16, expected_ranks,
        "mpi-exchange expected {expected_ranks} ranks, got {size}"
    );

    let part = RowPartition::new(rows, rank as usize, size as usize);
    let (mut old, mut new) = init_local_grids(rows, cols, &part);

    world.barrier();
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    reset_bench_metrics();
    let t = Instant::now();

    for _ in 0..steps {
        if part.local_rows > 0 {
            exchange_halos(&world, &part, &mut old);
            update_local_block(&old, &mut new);
            std::mem::swap(&mut old, &mut new);
        }

        // Keep the baseline in lockstep to match the current CXL exchange mode.
        world.barrier();
    }

    world.barrier();
    let kernel = t.elapsed();
    let kernel_metrics = get_bench_metrics();
    // Over the IP-over-CXL bridge the TUN device owns the CXL traffic, so its
    // snapshot replaces this process' (empty) direct-CXL counters.
    match ip_cxl_metrics.finish() {
        Some(metrics) => print_bench_metrics_value(metrics),
        None => print_bench_metrics_value(kernel_metrics),
    }

    (gather_result(&world, rows, cols, &part, &old), kernel)

}
