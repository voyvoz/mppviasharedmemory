use clap::Parser;
use cxl_pagerank::net::{NetComm, infer_rank, parse_peers};
use cxl_pagerank::net_pagerank::net_pagerank;
use cxl_pagerank::pagerank::SharedPageRank;
use cxl_pagerank::pregel_pagerank::PregelPageRank;
use cxl_pagerank::utils::{
    Cli, gen_csr, load_mtx, pagerank_host_parallel, pagerank_sequential, resolve_threads,
};
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_id};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use std::process::exit;
use std::time::{self, Duration};

fn validate_f64(label: &str, got: &[f64], expected: &[f64], nodes: usize) {
    let tol = 1e-6;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for i in 0..nodes {
        let a = (got[i] - expected[i]).abs();
        if a > max_abs {
            max_abs = a;
        }
        let r = a / expected[i].abs().max(1e-30);
        if r > max_rel {
            max_rel = r;
        }
    }
    println!("[{label}] max_abs_err={max_abs:.3e} max_rel_err={max_rel:.3e}");
    if max_abs < tol {
        println!("<<- {label}: matches sequential baseline (tol {tol:.0e}) ->>");
    } else {
        println!("<<- {label} FAIL: max_abs_err {max_abs:.3e} > {tol:.0e} ->>");
    }
}

fn main() {
    let args = Cli::parse();
    println!("Args: {args:?}");

    let threads = resolve_threads(args.threads);

    let csr = match &args.mtx {
        Some(path) => {
            let start = time::SystemTime::now();
            match load_mtx(path, threads) {
                Ok(csr) => {
                    println!(
                        "Loaded {path} in {:?} ({threads} threads)",
                        start.elapsed().unwrap()
                    );
                    csr
                }
                Err(e) => {
                    eprintln!("Failed to load {path}: {e}");
                    exit(1);
                }
            }
        }
        None => gen_csr(args.nodes, args.degree, args.seed),
    };
    let nodes = csr.row_ptr.len() - 1;
    println!("Graph: nodes={} edges={}", nodes, csr.col_idx.len());

    // ------------------------------------------------------------------ standard network (TCP / MPI-style)
    if args.net {
        let peers = parse_peers(&args.net_peers.clone().expect("--net requires --net-peers"));
        let rank = args
            .net_rank
            .or_else(|| infer_rank(&peers))
            .expect("could not determine rank (set --net-rank or match hostname to --net-peers)");
        println!(
            "-- net pagerank init (rank {rank}/{} peers) --",
            peers.len()
        );
        let mut comm = NetComm::connect(&peers, rank);
        let start = time::SystemTime::now();
        let got = net_pagerank(&mut comm, &csr, args.iters, args.damping);
        let kernel = start.elapsed().unwrap();
        println!("[net_pagerank] rank {rank} finished in {:?}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        if let Some(got) = got {
            let expected = pagerank_sequential(&csr, args.iters, args.damping);
            validate_f64("net_pagerank", &got, &expected, nodes);
        }
        exit(0);
    }

    // ------------------------------------------------------------------ single host
    if args.num_hosts == 0 {
        let start = time::SystemTime::now();
        let _ = pagerank_sequential(&csr, args.iters, args.damping);
        let kernel = start.elapsed().unwrap();
        println!("[pr_baseline] {} iters in {:?}", args.iters, kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        exit(0);
    }

    if args.num_hosts == 1 {
        println!(
            "Host-local parallel PageRank ({threads} threads, {} iters)",
            args.iters
        );
        let start = time::SystemTime::now();
        let got = pagerank_host_parallel(&csr, args.iters, args.damping, threads);
        let kernel = start.elapsed().unwrap();
        println!("[pr_host_parallel] {:?}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        let expected = pagerank_sequential(&csr, args.iters, args.damping);
        validate_f64("pr_host_parallel", &got, &expected, nodes);
        exit(0);
    }

    // ------------------------------------------------------------------ multi host (CXL)
    let mut persist: PersistantDataStructs = cxl_init(args.num_hosts as u16);
    let myid = get_machine_id();

    let send_offsets = |persist: &mut PersistantDataStructs, offsets| {
        for h in 2..=args.num_hosts {
            persist
                .ipc
                .try_send_to(h as PeerMachineID, &offsets)
                .unwrap();
        }
    };
    // Host 1 copies the whole input into CXL memory before publishing the
    // offsets, which takes tens of seconds on the larger inputs -- peers have
    // to wait that out. The benchmark harness bounds the run from outside.
    let recv_offsets = |persist: &mut PersistantDataStructs| {
        persist
            .ipc
            .try_pop_from(1 as PeerMachineID, Some(Duration::from_secs(600)))
            .unwrap()
            .1
    };

    if args.pregel {
        // ---- PageRank, Pregel (message passing over IPC)
        let pr = if myid == 1 {
            let (p, o) = PregelPageRank::create(persist.alloc.as_ref(), &csr);
            send_offsets(&mut persist, o);
            p
        } else {
            PregelPageRank::attach(&recv_offsets(&mut persist))
        };
        println!("-- pregel pagerank init (host {myid}) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        pr.run(&mut persist.ipc, args.iters, args.damping);
        let kernel = start.elapsed().unwrap();
        println!("[pregel_pagerank] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        if myid == 1 {
            let got = pr.collect();
            let expected = pagerank_sequential(&csr, args.iters, args.damping);
            validate_f64("pregel_pagerank", &got, &expected, nodes);
        }
    } else {
        // ---- PageRank, shared memory
        let pr = if myid == 1 {
            let (p, o) = SharedPageRank::create(persist.alloc.as_ref(), &csr);
            send_offsets(&mut persist, o);
            p
        } else {
            SharedPageRank::attach(&recv_offsets(&mut persist))
        };
        println!("-- shared pagerank init (host {myid}, {threads} threads) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        pr.run(args.iters, args.damping, threads);
        let kernel = start.elapsed().unwrap();
        println!("[cxl_pagerank] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        barrier_wait_all();
        if myid == 1 {
            let got = pr.collect();
            let expected = pagerank_sequential(&csr, args.iters, args.damping);
            validate_f64("cxl_pagerank", &got, &expected, nodes);
        }
    }
    exit(0);
}
