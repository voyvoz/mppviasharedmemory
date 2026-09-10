use crate::bfs::SharedBfs;
use crate::cxl_bfs::CxlBfs;
use crate::utils::{
    Cli, Csr, bfs_host_parallel, bfs_sequential, gen_csr, load_mtx, resolve_threads,
};
use clap::Parser;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_id};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use shm_cxl::utils::timestamps::{
    get_bench_metrics, ip_cxl_metrics_requested, print_bench_metrics, print_bench_metrics_value,
    reset_bench_metrics,
};
#[cfg(not(feature = "mpi"))]
use std::process::exit;
use std::time::{Duration, Instant};

fn validate(label: &str, got: &[u64], expected: &[u64], nodes: usize) {
    let mut mismatches = 0;
    for index in 0..nodes {
        if got[index] != expected[index] {
            if mismatches < 10 {
                println!(
                    "mismatch node {index}: got={} expected={}",
                    got[index], expected[index]
                );
            }
            mismatches += 1;
        }
    }
    let reached = got.iter().filter(|distance| **distance != u64::MAX).count();
    println!("[{label}] reached {reached}/{nodes}");
    if mismatches == 0 {
        println!("<<- {label}: matches sequential baseline ->>");
    } else {
        println!("<<- {label} FAIL: {mismatches} mismatches ->>");
    }
}

#[cfg(feature = "mpi")]
fn run_mpi(csr: &Csr, args: &Cli, nodes: usize) {
    // Preserve zero as "automatic" so the MPI engine can divide the host CPU
    // budget among colocated ranks. Explicit --threads remains per-rank.
    let result = crate::mpi_pregel::pregel_bfs_mpi(csr, args.source, args.threads);
    println!(
        "[pregel_mpi] rank {}/{} finished in {:?}",
        result.rank, result.ranks, result.kernel
    );
    println!("BENCH_KERNEL_NS={}", result.kernel.as_nanos());
    if let Some(distances) = result.distances {
        validate(
            "pregel_mpi",
            &distances,
            &bfs_sequential(csr, args.source),
            nodes,
        );
    }
}

#[cfg(not(feature = "mpi"))]
fn run_mpi(_csr: &Csr, _args: &Cli, _nodes: usize) {
    eprintln!("--mpi requires cxl-bfs-mpi (build cxl-bfs with --features mpi)");
    exit(1);
}

pub fn run() {
    reset_bench_metrics();
    let args = Cli::parse();
    println!("Args: {args:?}");
    let threads = resolve_threads(args.threads);

    let load_start = Instant::now();
    let csr = match &args.mtx {
        Some(path) => match load_mtx(path, threads) {
            Ok(csr) => {
                let load = load_start.elapsed();
                println!("Loaded {path} in {load:?} ({threads} threads)");
                println!("BENCH_LOAD_NS={}", load.as_nanos());
                csr
            }
            Err(error) => {
                eprintln!("Failed to load {path}: {error}");
                std::process::exit(1);
            }
        },
        None => gen_csr(args.nodes, args.degree, args.seed),
    };
    let nodes = csr.row_ptr.len() - 1;
    println!("Graph: nodes={} edges={}", nodes, csr.col_idx.len());
    if args.source as usize >= nodes {
        eprintln!("source {} out of range", args.source);
        std::process::exit(1);
    }

    if args.mpi {
        run_mpi(&csr, &args, nodes);
        if !ip_cxl_metrics_requested() {
            print_bench_metrics();
        }
        return;
    }

    if args.num_hosts == 0 {
        let start = Instant::now();
        let distances = bfs_sequential(&csr, args.source);
        let reached = distances
            .iter()
            .filter(|distance| **distance != u64::MAX)
            .count();
        let kernel = start.elapsed();
        println!("[baseline] {kernel:?} reached {reached}/{nodes}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        print_bench_metrics();
        return;
    }

    if args.num_hosts == 1 {
        let start = Instant::now();
        let distances = bfs_host_parallel(&csr, args.source, threads);
        let kernel = start.elapsed();
        println!("[host_parallel] {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        validate(
            "host_parallel",
            &distances,
            &bfs_sequential(&csr, args.source),
            nodes,
        );
        print_bench_metrics();
        return;
    }

    let mut persist: PersistantDataStructs = cxl_init(args.num_hosts as u16);
    let myid = get_machine_id();
    let send_offsets = |persist: &mut PersistantDataStructs, offsets| {
        for host in 2..=args.num_hosts {
            persist
                .ipc
                .try_send_to(host as PeerMachineID, &offsets)
                .unwrap();
        }
    };
    let recv_offsets = |persist: &mut PersistantDataStructs| {
        persist
            .ipc
            .try_pop_from(1 as PeerMachineID, Some(Duration::from_secs(600)))
            .unwrap()
            .1
    };

    if args.partitioned {
        let mut bfs = if myid == 1 {
            let (bfs, offsets) = CxlBfs::create(persist.alloc.as_ref(), &csr, args.source);
            send_offsets(&mut persist, offsets);
            bfs
        } else {
            CxlBfs::attach(&recv_offsets(&mut persist), &csr)
        };
        println!("-- partitioned bfs init (host {myid}, {threads} threads) --");
        barrier_wait_all();
        reset_bench_metrics();
        let start = Instant::now();
        bfs.run(threads);
        let kernel = start.elapsed();
        let kernel_metrics = get_bench_metrics();
        println!("[cxl_part_bfs] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        bfs.publish_distances();
        if myid == 1 {
            validate(
                "cxl_part_bfs",
                &bfs.collect_distances(),
                &bfs_sequential(&csr, args.source),
                nodes,
            );
        }
        print_bench_metrics_value(kernel_metrics);
    } else {
        let bfs = if myid == 1 {
            let (bfs, offsets) = SharedBfs::create(persist.alloc.as_ref(), &csr, args.source);
            send_offsets(&mut persist, offsets);
            bfs
        } else {
            SharedBfs::attach(&recv_offsets(&mut persist), &csr)
        };
        println!("-- shared bfs init (host {myid}, {threads} threads) --");
        barrier_wait_all();
        reset_bench_metrics();
        let start = Instant::now();
        bfs.run(threads);
        let kernel = start.elapsed();
        let kernel_metrics = get_bench_metrics();
        println!("[cxl_bfs] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        barrier_wait_all();
        if myid == 1 {
            validate(
                "cxl_bfs",
                &bfs.collect_distances(),
                &bfs_sequential(&csr, args.source),
                nodes,
            );
        }
        print_bench_metrics_value(kernel_metrics);
    }
}
