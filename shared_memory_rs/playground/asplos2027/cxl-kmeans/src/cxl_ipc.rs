use crate::cxl_shared::SharedKMeansState;
use crate::{
    DistributedKMeansResult, KMeansConfig, SplitMix64, assign_accumulate, init_random_centers,
    partition_range, synthetic_blobs, update_centers, validate_input,
};
use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::Cacheline;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};
use shm_cxl::cxl::cxl_init;
use shm_cxl::utils::timestamps::{
    get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};
use std::time::Instant;

const ROOT_MACHINE_ID: PeerMachineID = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
enum IpcMsgKind {
    Bootstrap = 1,
    PartialReady = 2,
    IterDone = 3,
}

impl IpcMsgKind {
    fn from_u64(v: u64) -> Option<Self> {
        match v {
            1 => Some(IpcMsgKind::Bootstrap),
            2 => Some(IpcMsgKind::PartialReady),
            3 => Some(IpcMsgKind::IterDone),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IpcMsg {
    kind: IpcMsgKind,
    arg0: u64,
    arg1: u64,
}

impl IpcMsg {
    /// Builds a Bootstrap message.
    ///
    /// Semantics:
    /// - Sent by the root to a worker.
    /// - Carries the offset of `SharedKMeansState` in CXL shared memory.
    ///
    /// Field mapping:
    /// - `kind = Bootstrap`
    /// - `arg0 = shared_offset`
    /// - `arg1 = 0` and is currently unused
    fn bootstrap(shared_offset: u64) -> Self {
        IpcMsg {
            kind: IpcMsgKind::Bootstrap,
            arg0: shared_offset,
            arg1: 0,
        }
    }

    /// Builds a PartialReady message.
    ///
    /// Semantics:
    /// - Sent by a worker to the root.
    /// - Indicates that the worker has finished local computation for
    ///   iteration `iter` and has already written its partial result
    ///   into the shared state.
    ///
    /// Field mapping:
    /// - `kind = PartialReady`
    /// - `arg0 = iter`
    /// - `arg1 = sender machine id
    fn partial_ready(iter: usize, sender: PeerMachineID) -> Self {
        IpcMsg {
            kind: IpcMsgKind::PartialReady,
            arg0: iter as u64,
            arg1: sender as u64,
        }
    }
    // Builds an IterDone message.
    //
    // Semantics:
    // - Broadcast by the root to all workers.
    // - Indicates that global aggregation for iteration `iter` is complete
    //   and that the updated centers and control state have been written
    //   back to shared memory.
    //
    // Field mapping:
    // - `kind = IterDone`
    // - `arg0 = iter`
    // - `arg1 = converged`, encoded as `1` for true and `0` for false
    fn iter_done(iter: usize, converged: bool) -> Self {
        IpcMsg {
            kind: IpcMsgKind::IterDone,
            arg0: iter as u64,
            arg1: u64::from(converged),
        }
    }

    fn encode(self) -> Cacheline {
        Cacheline {
            d: [self.kind as u64, self.arg0, self.arg1, 0, 0, 0, 0, 0],
        }
    }

    fn decode(cl: Cacheline) -> Option<Self> {
        Some(IpcMsg {
            kind: IpcMsgKind::from_u64(cl.d[0])?,
            arg0: cl.d[1],
            arg1: cl.d[2],
        })
    }
}

fn send_msg(ipc: &mut CXLIpc, target: PeerMachineID, msg: IpcMsg) {
    ipc.try_send_to(target, &msg.encode()).unwrap();
}

fn recv_bootstrap(ipc: &mut CXLIpc) -> u64 {
    let (peer, raw) = ipc.try_pop_from(ROOT_MACHINE_ID, None).unwrap();
    assert_eq!(peer.machine_id, ROOT_MACHINE_ID);
    let msg = IpcMsg::decode(raw).expect("bootstrap message must be decodable");
    assert_eq!(msg.kind, IpcMsgKind::Bootstrap);
    msg.arg0
}

fn wait_for_worker_partials(ipc: &mut CXLIpc, nhosts: usize, iter: usize) {
    let mut seen = vec![false; nhosts];
    let mut remaining = nhosts.saturating_sub(1);

    while remaining > 0 {
        let (_peer, raw) = ipc.await_any_msg(None).unwrap();
        let msg = IpcMsg::decode(raw).expect("worker message must be decodable");
        assert_eq!(msg.kind, IpcMsgKind::PartialReady);
        assert_eq!(
            msg.arg0 as usize, iter,
            "received stale or future iteration"
        );

        let sender = msg.arg1 as PeerMachineID;
        assert_ne!(sender, ROOT_MACHINE_ID, "root must not report PartialReady");
        assert!(
            sender >= 1 && sender as usize <= nhosts,
            "sender {} out of range for nhosts={}",
            sender,
            nhosts
        );

        let sender_idx = sender as usize - 1;
        assert!(
            !seen[sender_idx],
            "duplicate partial-ready from peer {}",
            sender
        );

        seen[sender_idx] = true;
        remaining -= 1;
    }
}

fn wait_for_iter_done(ipc: &mut CXLIpc, iter: usize) -> bool {
    let (peer, raw) = ipc.try_pop_from(ROOT_MACHINE_ID, None).unwrap();
    assert_eq!(peer.machine_id, ROOT_MACHINE_ID);
    let msg = IpcMsg::decode(raw).expect("iter-done message must be decodable");
    assert_eq!(msg.kind, IpcMsgKind::IterDone);
    assert_eq!(
        msg.arg0 as usize, iter,
        "received stale or future iteration"
    );
    msg.arg1 != 0
}

pub fn run_synthetic_kmeans_cxl_ipc(
    n_points: usize,
    dim: usize,
    blobs: usize,
    spread: f64,
    cfg: &KMeansConfig,
    expected_hosts: u16,
) -> DistributedKMeansResult {
    let points = synthetic_blobs(n_points, dim, blobs, spread, cfg.seed);
    run_kmeans_cxl_ipc(&points, dim, cfg, expected_hosts)
}

pub fn run_kmeans_cxl_ipc(
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

    // The root's RNG must outlive seeding: empty-cluster reseeds in the
    // update step continue the same stream, keeping every variant on the
    // single-host trajectory.
    let mut rng = SplitMix64::new(cfg.seed);
    // root
    let shared = if machine_id == ROOT_MACHINE_ID as usize {
        let shared = SharedKMeansState::create(&persist.alloc.as_ref(), cfg.k, dim, nhosts);
        let centers = init_random_centers(points, n_points, dim, cfg.k, &mut rng);
        shared.initialize(&centers);

        let offset = shared.offset();
        let bootstrap = IpcMsg::bootstrap(offset);

        for peer in 2..=expected_hosts {
            send_msg(&mut persist.ipc, peer, bootstrap);
        }
        shared
    } else {
        // worker
        let offset = recv_bootstrap(&mut persist.ipc);
        SharedKMeansState::attach(offset, cfg.k, dim, nhosts)
    };

    let mut local_centers = vec![0.0; cfg.k * dim];
    let mut global_sums = if machine_id == ROOT_MACHINE_ID as usize {
        vec![0.0; cfg.k * dim]
    } else {
        Vec::new()
    };

    let mut global_counts = if machine_id == ROOT_MACHINE_ID as usize {
        vec![0u64; cfg.k]
    } else {
        Vec::new()
    };

    barrier_wait_all();
    // Clock covers only the iteration loop -- CXL attach, the bootstrap
    // handshake, seeding and result collection are excluded.
    reset_bench_metrics();
    let kernel_start = Instant::now();

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

        if machine_id == ROOT_MACHINE_ID as usize {
            wait_for_worker_partials(&mut persist.ipc, nhosts, iter);

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

            let done = IpcMsg::iter_done(iter, converged);

            for peer in 2..=expected_hosts {
                send_msg(&mut persist.ipc, peer, done);
            }

            if converged {
                break;
            }
        } else {
            send_msg(
                &mut persist.ipc,
                ROOT_MACHINE_ID,
                IpcMsg::partial_ready(iter, machine_id as PeerMachineID),
            );
            if wait_for_iter_done(&mut persist.ipc, iter) {
                break;
            }
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

#[cfg(test)]
mod test {
    use super::{IpcMsg, IpcMsgKind};

    #[test]
    fn ipc_msg_roundtrip() {
        let msg = IpcMsg::iter_done(17, true);
        let decoded = IpcMsg::decode(msg.encode()).unwrap();
        assert_eq!(decoded.kind, IpcMsgKind::IterDone);
        assert_eq!(decoded.arg0, 17);
        assert_eq!(decoded.arg1, 1);
    }
}
