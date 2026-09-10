//! Halo-exchange stencil: the grid stays host-local and only the boundary rows
//! travel, as messages over the CXL IPC rings. Contrast with the `dist` engine in
//! `lib.rs`, which places one grid in CXL memory and shares it.

use std::time::{Duration, Instant};

use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};
use shm_cxl::cxl::cxl_init;
use shm_cxl::utils::timestamps::{
    SpinCounter, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

use crate::stencil_core::{LocalGrid, RowPartition, idx, init_local_grids, update_local_block};

fn row_chunk_count(cols: usize) -> usize {
    cols.div_ceil(8)
}

// send_row works like this:
//
// Split one row into chunks of 8 elements.
// Pack each chunk into one Cacheline.
// f64 -> to_bits() -> u64
// Call ipc.try_send_to(peer, &cl).
// If the ring is full, spin and retry.
fn send_row(ipc: &mut CXLIpc, peer: PeerMachineID, row: &[f64]) {
    let mut wait_spins = SpinCounter::new();
    // row.len() == cols
    let nchunks = row_chunk_count(row.len());

    for chunk_idx in 0..nchunks {
        let mut cl = Cacheline::empty();

        for lane in 0..8 {
            let col = chunk_idx * 8 + lane;
            if col < row.len() {
                cl.d[lane] = row[col].to_bits();
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

// recv_row does the reverse:
//
// Compute how many chunks this row has.
// Receive that many Cachelines in sequence.
// u64 -> from_bits() -> f64
// Restore the values into the row.
fn recv_row(ipc: &mut CXLIpc, peer: PeerMachineID, row: &mut [f64]) {
    let nchunks = row.len().div_ceil(8);

    for chunk_idx in 0..nchunks {
        let (_, cl) = ipc.try_pop_from(peer, None).unwrap();

        for lane in 0..8 {
            let col = chunk_idx * 8 + lane;
            if col < row.len() {
                row[col] = f64::from_bits(cl.d[lane]);
            }
        }
    }
}

// Use the same two-phase order:
// Phase A
//
// If up_peer exists, send first_owned_row() to the upper neighbor.
// If down_peer exists, receive bottom_halo_mut() from the lower neighbor.
//
// Phase B
//
// If down_peer exists, send last_owned_row() to the lower neighbor.
// If up_peer exists, receive top_halo_mut() from the upper neighbor.
fn exchange_halos(ipc: &mut CXLIpc, part: &RowPartition, old: &mut LocalGrid) {
    // Phase A:
    // send first owned row upward
    // receive bottom halo from below
    if let Some(peer) = part.up_peer {
        let peer = peer as PeerMachineID + 1;
        send_row(ipc, peer, old.first_owned_row());
    }

    if let Some(peer) = part.down_peer {
        let peer = peer as PeerMachineID + 1;
        recv_row(ipc, peer, old.bottom_halo_mut());
    }
    // Phase B:
    // send last owned row downward
    // receive top halo from above
    if let Some(peer) = part.down_peer {
        let peer = peer as PeerMachineID + 1;
        send_row(ipc, peer, old.last_owned_row());
    }

    if let Some(peer) = part.up_peer {
        let peer = peer as PeerMachineID + 1;
        recv_row(ipc, peer, old.top_halo_mut());
    }
}

// fn init_local_grids(rows: usize, cols: usize, part: &RowPartition) -> (LocalGrid, LocalGrid) {
//     let mut old = LocalGrid::new(part.local_rows, cols);
//     let mut new = LocalGrid::new(part.local_rows, cols);

//     let center_row = rows / 2;
//     let center_col = cols / 2;

//     if part.owns_global_row(center_row) {
//         let local_i = part.global_to_local(center_row);
//         old.row_mut(local_i)[center_col] = 100.0;
//         new.row_mut(local_i)[center_col] = 100.0;
//     }

//     (old, new)
// }

// fn update_local_block(old: &LocalGrid, new: &mut LocalGrid) {
//     for local_i in 1..=old.local_rows() {
//         let top = old.row(local_i - 1);
//         let mid = old.row(local_i);
//         let bottom = old.row(local_i + 1);

//         let dst = new.row_mut(local_i);

//         // Copy the whole row first, so boundary columns naturally stay unchanged.
//         dst.copy_from_slice(mid);

//         for j in 1..old.cols().saturating_sub(1) {
//             dst[j] = 0.2 * (mid[j] + top[j] + bottom[j] + mid[j - 1] + mid[j + 1]);
//         }
//     }
// }

fn gather_result(
    ipc: &mut CXLIpc,
    rows: usize,
    cols: usize,
    myid: u16,
    nhosts: u16,
    part: &RowPartition,
    old: &LocalGrid,
) -> Option<Vec<f64>> {
    if myid == 1 {
        let mut grid = vec![0.0; rows * cols];

        // Store this host's owned rows first.
        for global_row in part.row_begin..part.row_end {
            let local_i = part.global_to_local(global_row);
            let start = idx(global_row, 0, cols);
            let end = start + cols;
            grid[start..end].copy_from_slice(old.row(local_i));
        }

        // Then receive owned rows from the other hosts.
        for peer_idx in 1..nhosts as usize {
            let peer = peer_idx as PeerMachineID + 1;
            let peer_part = RowPartition::new(rows, peer_idx, nhosts as usize);

            for global_row in peer_part.row_begin..peer_part.row_end {
                let start = idx(global_row, 0, cols);
                let end = start + cols;
                recv_row(ipc, peer, &mut grid[start..end]);
            }
        }

        Some(grid)
    } else {
        // Non-host-1 workers send their owned rows to host 1.
        for global_row in part.row_begin..part.row_end {
            let local_i = part.global_to_local(global_row);
            send_row(ipc, 1, old.row(local_i));
        }

        None
    }
}

pub fn run_dist_exchange(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_hosts: u16,
) -> Option<Vec<f64>> {
    run_dist_exchange_timed(rows, cols, steps, expected_hosts).0
}

/// See [`run_dist_exchange`]; the duration covers only the stepping loop
/// (including its halo exchanges and per-step barriers, which are the algorithm).
/// CXL attach, local grid setup and the final gather are excluded. The clock
/// starts on the barrier that already follows initialisation -- do not add
/// another one, an extra barrier desynchronises the generation counters.
pub fn run_dist_exchange_timed(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_hosts: u16,
) -> (Option<Vec<f64>>, Duration) {
    let mut persist = cxl_init(expected_hosts);

    let myid = get_machine_id();
    let nhosts = get_machine_cnt();

    let part = RowPartition::new(rows, (myid - 1) as usize, nhosts as usize);

    let (mut old, mut new) = init_local_grids(rows, cols, &part);
    // Start only after all hosts have completed local initialization.
    barrier_wait_all();

    reset_bench_metrics();
    let t = Instant::now();

    for _ in 0..steps {
        if part.local_rows > 0 {
            exchange_halos(&mut persist.ipc, &part, &mut old);
            update_local_block(&old, &mut new);
            std::mem::swap(&mut old, &mut new);
        }

        // The first version keeps a barrier here so all hosts advance in lockstep.
        barrier_wait_all();
    }

    // Let all hosts finish the final iteration before collecting results.
    barrier_wait_all();

    let kernel = t.elapsed();
    let kernel_metrics = get_bench_metrics();
    print_bench_metrics_value(kernel_metrics);

    (
        gather_result(&mut persist.ipc, rows, cols, myid, nhosts, &part, &old),
        kernel,
    )
}
