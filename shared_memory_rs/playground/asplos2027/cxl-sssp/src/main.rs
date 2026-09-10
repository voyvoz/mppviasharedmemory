use clap::Parser;
use cxl_sssp::pregel_sssp::PregelSssp;
use cxl_sssp::sssp::SharedSssp;
use cxl_sssp::utils::{
    Cli, gen_csr, load_mtx, resolve_threads, sssp_host_parallel, sssp_sequential,
};
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_id};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use std::process::exit;
use std::time::{self, Duration};

fn validate(label: &str, got: &[u64], expected: &[u64], nodes: usize) {
    let mut mismatches = 0;
    for i in 0..nodes {
        if got[i] != expected[i] {
            if mismatches < 10 {
                println!("mismatch node {i}: got={} expected={}", got[i], expected[i]);
            }
            mismatches += 1;
        }
    }
    let reached = got.iter().filter(|d| **d != u64::MAX).count();
    println!("[{label}] reached {reached}/{nodes}");
    if mismatches == 0 {
        println!("<<- {label}: matches sequential baseline ->>");
    } else {
        println!("<<- {label} FAIL: {mismatches} mismatches ->>");
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

    if args.source as usize >= nodes {
        eprintln!("source {} out of range", args.source);
        exit(1);
    }

    // ------------------------------------------------------------------ single host
    if args.num_hosts == 0 {
        let start = time::SystemTime::now();
        let dist = sssp_sequential(&csr, args.source);
        let reached = dist.iter().filter(|d| **d != u64::MAX).count();
        let kernel = start.elapsed().unwrap();
        println!("[sssp_baseline] {:?} reached {reached}/{nodes}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        exit(0);
    }

    if args.num_hosts == 1 {
        println!("Host-local parallel SSSP with {threads} threads");
        let start = time::SystemTime::now();
        let got = sssp_host_parallel(&csr, args.source, threads);
        let kernel = start.elapsed().unwrap();
        println!("[sssp_host_parallel] {:?}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        let expected = sssp_sequential(&csr, args.source);
        validate("sssp_host_parallel", &got, &expected, nodes);
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
        // ---- SSSP, Pregel (message passing over IPC)
        let pg = if myid == 1 {
            let (p, o) = PregelSssp::create(persist.alloc.as_ref(), &csr, args.source);
            send_offsets(&mut persist, o);
            p
        } else {
            PregelSssp::attach(&recv_offsets(&mut persist))
        };
        println!("-- pregel sssp init (host {myid}) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        pg.run(&mut persist.ipc);
        let kernel = start.elapsed().unwrap();
        println!("[pregel_sssp] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        if myid == 1 {
            let got = pg.collect();
            let expected = sssp_sequential(&csr, args.source);
            validate("pregel_sssp", &got, &expected, nodes);
        }
    } else {
        // ---- SSSP, shared memory
        let sp = if myid == 1 {
            let (s, o) = SharedSssp::create(persist.alloc.as_ref(), &csr, args.source);
            send_offsets(&mut persist, o);
            s
        } else {
            SharedSssp::attach(&recv_offsets(&mut persist))
        };
        println!("-- shared sssp init (host {myid}, {threads} threads) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        sp.run(threads);
        let kernel = start.elapsed().unwrap();
        println!("[cxl_sssp] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        barrier_wait_all();
        if myid == 1 {
            let got = sp.collect_distances();
            let expected = sssp_sequential(&csr, args.source);
            validate("cxl_sssp", &got, &expected, nodes);
        }
    }
    exit(0);
}
