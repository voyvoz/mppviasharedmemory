use std::time::Instant;

use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};
use shm_cxl::cxl::cxl_init;
use shm_cxl::utils::timestamps::{
    SpinCounter, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

use crate::DistRunResult;
use crate::matmul_core::{RowPartition, make_matrix, matmul_block_tiled};

/// Splits `data` into 8-element chunks and sends each as one Cacheline
/// message to `peer`, retrying while the target ring is full. Same
/// technique as cxl-stencil's `send_row`, just not limited to one row --
/// `data` can be several rows flattened together (an A-shard, all of B, or
/// a C-shard).
fn send_data(ipc: &mut CXLIpc, peer: PeerMachineID, data: &[f64]) {
    let mut wait_spins = SpinCounter::new();
    let nchunks = data.len().div_ceil(8);

    for chunk_idx in 0..nchunks {
        let mut cl = Cacheline::empty();
        for lane in 0..8 {
            let i = chunk_idx * 8 + lane;
            if i < data.len() {
                cl.d[lane] = data[i].to_bits();
            }
        }

        loop {
            if ipc.try_send_to(peer, &cl).is_ok() {
                break;
            }
            wait_spins.wait();
        }
    }
}

/// Reverse of `send_data`: blocks until enough Cachelines have arrived from
/// `peer` to fill `out`.
fn recv_data(ipc: &mut CXLIpc, peer: PeerMachineID, out: &mut [f64]) {
    let nchunks = out.len().div_ceil(8);

    for chunk_idx in 0..nchunks {
        let (_, cl) = ipc.try_pop_from(peer, None).unwrap();
        for lane in 0..8 {
            let i = chunk_idx * 8 + lane;
            if i < out.len() {
                out[i] = f64::from_bits(cl.d[lane]);
            }
        }
    }
}

pub fn run_ipc_dist(m: usize, k: usize, n: usize, expected_hosts: u16) -> DistRunResult {
    run_ipc_dist_timed(m, k, n, expected_hosts)
}

pub fn run_ipc_dist_timed(m: usize, k: usize, n: usize, expected_hosts: u16) -> DistRunResult {
    let init_start = Instant::now();

    let mut persist = cxl_init(expected_hosts);
    let myid = get_machine_id();
    let nhosts = get_machine_cnt();
    let part = RowPartition::new(m, myid.saturating_sub(1) as usize, nhosts.max(1) as usize);

    let (local_a, b) = if myid == 1 {
        let a = make_matrix(m, k);
        let b = make_matrix(k, n);

        // Scatter each peer's row-shard of A, then broadcast all of B.
        for peer in 2..=expected_hosts {
            let peer_part = RowPartition::new(m, (peer - 1) as usize, expected_hosts as usize);
            send_data(
                &mut persist.ipc,
                peer as PeerMachineID,
                &a[peer_part.row_begin * k..peer_part.row_end * k],
            );
            send_data(&mut persist.ipc, peer as PeerMachineID, &b);
        }

        let own_a = a[part.row_begin * k..part.row_end * k].to_vec();
        (own_a, b)
    } else {
        let mut local_a = vec![0.0; part.local_rows * k];
        let mut b = vec![0.0; k * n];
        recv_data(&mut persist.ipc, 1, &mut local_a);
        recv_data(&mut persist.ipc, 1, &mut b);
        (local_a, b)
    };

    barrier_wait_all();
    let init_elapsed = init_start.elapsed();

    // Kernel-scoped counters like every other engine. The scatter above and
    // the gather below are init/collect, so an (accurate) all-zero snapshot
    // states that this mode's kernel itself does no CXL traffic.
    reset_bench_metrics();
    let compute_start = Instant::now();
    let mut local_c = vec![0.0; part.local_rows * n];
    matmul_block_tiled(&local_a, &b, &mut local_c, k, n);
    barrier_wait_all();
    let compute_elapsed = compute_start.elapsed();
    print_bench_metrics_value(get_bench_metrics());

    let collect_start = Instant::now();
    let result = if myid == 1 {
        let mut c = vec![0.0; m * n];
        c[part.row_begin * n..part.row_end * n].copy_from_slice(&local_c);

        for peer in 2..=expected_hosts {
            let peer_part = RowPartition::new(m, (peer - 1) as usize, expected_hosts as usize);
            recv_data(
                &mut persist.ipc,
                peer as PeerMachineID,
                &mut c[peer_part.row_begin * n..peer_part.row_end * n],
            );
        }
        Some(c)
    } else {
        send_data(&mut persist.ipc, 1, &local_c);
        None
    };
    let collect_elapsed = collect_start.elapsed();

    DistRunResult {
        init_elapsed,
        compute_elapsed,
        collect_elapsed,
        result,
    }
}
