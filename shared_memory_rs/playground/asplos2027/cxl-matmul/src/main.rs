use std::env;
use std::time::Instant;

use cxl_matmul::{
    checksum, run_dist, run_dist_opt, run_ipc_dist, run_mpi_dist, run_serial_timed,
    run_threads_timed,
};
use mpi::traits::*;
use shm_cxl::core::host::get_machine_id;
use shm_cxl::utils::timestamps::{print_bench_metrics, reset_bench_metrics};

// single host
// cargo build -p matmul
// ./target/debug/matmul serial 256 256 256

// cxl
// cargo build -p matmul
// export CXL_MAPOFFSET=0x600000000
// ./target/debug/matmul dist 256 256 256 2 or  ./target/debug/matmul dist 256 256 256 3

// mpi
// cargo build -p cxl-matmul
// mpirun -n 4 ./target/debug/cxl-matmul mpi-dist 256 256 256 4

// dist-opt: same as dist, but the compute stages A's row-shard and all of B
// into local memory once and runs a cache-blocked kernel there, instead of
// the untiled kernel operating straight against CXL.
// ./target/debug/cxl-matmul dist-opt 256 256 256 3

fn main() {
    reset_bench_metrics();
    let args: Vec<String> = env::args().collect();
    let should_print = args.iter().any(|a| a == "--print");
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|a| a.as_str() != "--print")
        .map(|a| a.as_str())
        .collect();

    let (mode, base) = match positional.first().copied() {
        Some("serial") => ("serial", 1),
        Some("threads") => ("threads", 1),
        Some("dist") => ("dist", 1),
        Some("dist-opt") => ("dist-opt", 1),
        Some("ipc-dist") => ("ipc-dist", 1),
        Some("mpi-dist") => ("mpi-dist", 1),
        _ => ("threads", 0),
    };

    let m = positional
        .get(base)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let k = positional
        .get(base + 1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let n = positional
        .get(base + 2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    match mode {
        "serial" => {
            let t = Instant::now();
            let (c, kernel) = run_serial_timed(m, k, n);
            println!("mode=serial m={m} k={k} n={n}");
            println!("elapsed: {:.3?}", t.elapsed());
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
            println!("checksum: {:.6}", checksum(&c));
            if should_print {
                print_matrix(&c, m, n);
            }
        }
        "dist" => {
            let expected_hosts = positional
                .get(base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);

            let result = run_dist(m, k, n, expected_hosts);
            let myid = get_machine_id();

            println!("mode=dist host={myid} m={m} k={k} n={n} hosts={expected_hosts}");
            println!("init:    {:.3?}", result.init_elapsed);
            println!("compute: {:.3?}", result.compute_elapsed);
            println!("collect: {:.3?}", result.collect_elapsed);
            println!("BENCH_KERNEL_NS={}", result.compute_elapsed.as_nanos());

            if let Some(c) = result.result {
                println!("checksum: {:.6}", checksum(&c));
                if should_print {
                    print_matrix(&c, m, n);
                }
            }
        }
        "dist-opt" => {
            let expected_hosts = positional
                .get(base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);

            let result = run_dist_opt(m, k, n, expected_hosts);
            let myid = get_machine_id();

            println!("mode=dist-opt host={myid} m={m} k={k} n={n} hosts={expected_hosts}");
            println!("init:    {:.3?}", result.init_elapsed);
            println!("compute: {:.3?}", result.compute_elapsed);
            println!("collect: {:.3?}", result.collect_elapsed);
            println!("BENCH_KERNEL_NS={}", result.compute_elapsed.as_nanos());

            if let Some(c) = result.result {
                println!("checksum: {:.6}", checksum(&c));
                if should_print {
                    print_matrix(&c, m, n);
                }
            }
        }
        "ipc-dist" => {
            let expected_hosts = positional
                .get(base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);

            let result = run_ipc_dist(m, k, n, expected_hosts);
            let myid = get_machine_id();

            println!("mode=ipc-dist host={myid} m={m} k={k} n={n} hosts={expected_hosts}");
            println!("init:    {:.3?}", result.init_elapsed);
            println!("compute: {:.3?}", result.compute_elapsed);
            println!("collect: {:.3?}", result.collect_elapsed);
            println!("BENCH_KERNEL_NS={}", result.compute_elapsed.as_nanos());

            if let Some(c) = result.result {
                println!("checksum: {:.6}", checksum(&c));
                if should_print {
                    print_matrix(&c, m, n);
                }
            }
        }
        "mpi-dist" => {
            let expected_ranks = positional
                .get(base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);

            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            let rank = world.rank();
            let size = world.size();
            let result = run_mpi_dist(&world, m, k, n, expected_ranks);

            println!("mode=mpi-dist rank={rank} m={m} k={k} n={n} ranks={size}");
            println!("init:    {:.3?}", result.init_elapsed);
            println!("compute: {:.3?}", result.compute_elapsed);
            println!("collect: {:.3?}", result.collect_elapsed);
            println!("BENCH_KERNEL_NS={}", result.compute_elapsed.as_nanos());

            if let Some(c) = result.result {
                println!("checksum: {:.6}", checksum(&c));
                if should_print {
                    print_matrix(&c, m, n);
                }
            }
        }
        _ => {
            let threads = positional
                .get(base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);

            let t = Instant::now();
            let (c, kernel) = run_threads_timed(m, k, n, threads);
            println!("mode=threads m={m} k={k} n={n} threads={threads}");
            println!("elapsed: {:.3?}", t.elapsed());
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
            println!("checksum: {:.6}", checksum(&c));
            if should_print {
                print_matrix(&c, m, n);
            }
        }
    }
    // The distributed engines print their own kernel-scoped metrics snapshot
    // (mpi-dist may substitute the IP-over-CXL bridge counters); printing
    // again here would double-count in the harness' summing parser.
    if !matches!(mode, "dist" | "dist-opt" | "ipc-dist" | "mpi-dist") {
        print_bench_metrics();
    }
}

fn print_matrix(c: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            print!("{:8.2}", c[i * cols + j]);
        }
        println!();
    }
}
