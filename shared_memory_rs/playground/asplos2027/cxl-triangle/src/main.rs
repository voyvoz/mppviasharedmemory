use clap::Parser;
use cxl_triangle::pregel_triangle::PregelTriangle;
use cxl_triangle::triangle::SharedTriangle;
use cxl_triangle::utils::{
    Cli, gen_csr, load_mtx, resolve_threads, triangle_host_parallel, triangle_sequential,
};
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_id};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use std::process::exit;
use std::time::{self, Duration};

fn validate_scalar(label: &str, got: u64, expected: u64) {
    println!("[{label}] got={got} expected={expected}");
    if got == expected {
        println!("<<- {label}: matches sequential baseline ->>");
    } else {
        println!("<<- {label} FAIL: got {got} != expected {expected} ->>");
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

    // ------------------------------------------------------------------ single host
    if args.num_hosts == 0 {
        let start = time::SystemTime::now();
        let t = triangle_sequential(&csr);
        let kernel = start.elapsed().unwrap();
        println!("[triangle_baseline] {:?} triangles={t}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        exit(0);
    }

    if args.num_hosts == 1 {
        println!("Host-local parallel Triangle Count with {threads} threads");
        let start = time::SystemTime::now();
        let got = triangle_host_parallel(&csr, threads);
        let kernel = start.elapsed().unwrap();
        println!("[triangle_host_parallel] {:?}", kernel);
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        let expected = triangle_sequential(&csr);
        validate_scalar("triangle_host_parallel", got, expected);
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
        // ---- Triangle Count, Pregel (message passing over IPC)
        let tc = if myid == 1 {
            let (t, o) = PregelTriangle::create(persist.alloc.as_ref(), &csr);
            send_offsets(&mut persist, o);
            t
        } else {
            PregelTriangle::attach(&recv_offsets(&mut persist))
        };
        println!("-- pregel triangle init (host {myid}) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        tc.run(&mut persist.ipc);
        let kernel = start.elapsed().unwrap();
        println!("[pregel_triangle] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        if myid == 1 {
            let got = tc.collect();
            let expected = triangle_sequential(&csr);
            validate_scalar("pregel_triangle", got, expected);
        }
    } else {
        // ---- Triangle Count, shared memory
        let tc = if myid == 1 {
            let (t, o) = SharedTriangle::create(persist.alloc.as_ref(), &csr);
            send_offsets(&mut persist, o);
            t
        } else {
            SharedTriangle::attach(&recv_offsets(&mut persist))
        };
        println!("-- shared triangle init (host {myid}, {threads} threads) --");
        barrier_wait_all();
        let start = time::SystemTime::now();
        tc.run(threads);
        let kernel = start.elapsed().unwrap();
        println!("[cxl_triangle] host {myid} finished in {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        barrier_wait_all();
        if myid == 1 {
            let got = tc.collect();
            let expected = triangle_sequential(&csr);
            validate_scalar("cxl_triangle", got, expected);
        }
    }
    exit(0);
}
