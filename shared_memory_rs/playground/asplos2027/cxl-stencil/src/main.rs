use std::env;
use std::time::Instant;

use cxl_stencil::dist_exchange::run_dist_exchange_timed;
use cxl_stencil::mpi_exchange::run_mpi_exchange_timed;
use cxl_stencil::{
    checksum, idx, run_dist_timed, run_dist_timed_opt, run_dist_timed_opt_opt, run_serial_timed,
    run_threads_timed,
};
use shm_cxl::core::host::get_machine_id;
use shm_cxl::utils::timestamps::{print_bench_metrics, reset_bench_metrics};

// single host
// ./target/release/cxl-stencil serial 1024 1024 100
// ./target/release/cxl-stencil threads 1024 1024 100 8

// cxl
// cargo build -p cxl-stencil
// export CXL_MAPOFFSET=0x600000000
// `dist` shares one grid in CXL memory; `dist-exchange` keeps the grid host-local
// and exchanges halo rows over the CXL IPC rings.
// ./target/release/cxl-stencil dist 16 16 10 2 --print
// ./target/release/cxl-stencil dist-exchange 16 16 10 2 --print
// mpirun -np 2 ./target/release/cxl-stencil mpi-exchange 1024 1024 100 2

fn print_grid(grid: &[f64], rows: usize, cols: usize) {
    for i in 0..rows {
        for j in 0..cols {
            print!("{:8.3}", grid[idx(i, j, cols)]);
        }
        println!();
    }
}

fn main() {
    reset_bench_metrics();
    let args: Vec<String> = env::args().collect();
    let should_print = args.iter().any(|arg| arg == "--print");
    let positional: Vec<&str> = args
        .iter()
        .skip(1)
        .filter(|arg| arg.as_str() != "--print")
        .map(|arg| arg.as_str())
        .collect();

    let (mode, arg_base) = match positional.first().copied() {
        Some("serial") => ("serial", 1),
        Some("threads") => ("threads", 1),
        Some("dist") => ("dist", 1),
        Some("dist-opt") => ("dist-opt", 1),
        Some("dist-opt-opt") => ("dist-opt-opt", 1),
        Some("dist-exchange") => ("dist-exchange", 1),
        Some("mpi-exchange") => ("mpi-exchange", 1),
        _ => ("threads", 0),
    };

    let rows = positional
        .get(arg_base)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let cols = positional
        .get(arg_base + 1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let steps = positional
        .get(arg_base + 2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let timer = Instant::now();
    match mode {
        "serial" => {
            let (grid, kernel) = run_serial_timed(rows, cols, steps);
            let elapsed = timer.elapsed();

            println!("mode=serial rows={rows} cols={cols} steps={steps}");
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
            println!("checksum: {:.6}", checksum(&grid));
            println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

            if should_print {
                print_grid(&grid, rows, cols);
            }
        }
        "dist" => {
            let expected_hosts = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let (result, kernel) = run_dist_timed(rows, cols, steps, expected_hosts);
            let elapsed = timer.elapsed();
            let myid = get_machine_id();

            println!(
                "mode=dist host={myid} rows={rows} cols={cols} steps={steps} hosts={expected_hosts}"
            );
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());

            if let Some(grid) = result {
                let sum = checksum(&grid);
                if !sum.is_finite() || grid.iter().any(|value| !value.is_finite()) {
                    eprintln!("FAIL: distributed stencil produced non-finite values");
                }
                println!("checksum: {:.6}", sum);
                println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

                if should_print {
                    print_grid(&grid, rows, cols);
                }
            }
        }
        "dist-opt" => {
            let expected_hosts = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let (result, kernel) = run_dist_timed_opt(rows, cols, steps, expected_hosts);
            let elapsed = timer.elapsed();
            let myid = get_machine_id();

            println!(
                "mode=dist host={myid} rows={rows} cols={cols} steps={steps} hosts={expected_hosts}"
            );
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());

            if let Some(grid) = result {
                let sum = checksum(&grid);
                if !sum.is_finite() || grid.iter().any(|value| !value.is_finite()) {
                    eprintln!("FAIL: optimized distributed stencil produced non-finite values");
                }
                println!("checksum: {:.6}", sum);
                println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

                if should_print {
                    print_grid(&grid, rows, cols);
                }
            }
        }
        "dist-opt-opt" => {
            let expected_hosts = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let (result, kernel) = run_dist_timed_opt_opt(rows, cols, steps, expected_hosts);
            let elapsed = timer.elapsed();
            let myid = get_machine_id();

            println!(
                "mode=dist host={myid} rows={rows} cols={cols} steps={steps} hosts={expected_hosts}"
            );
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());

            if let Some(grid) = result {
                let sum = checksum(&grid);
                if !sum.is_finite() || grid.iter().any(|value| !value.is_finite()) {
                    eprintln!("FAIL: optimized distributed stencil produced non-finite values");
                }
                println!("checksum: {:.6}", sum);
                println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

                if should_print {
                    print_grid(&grid, rows, cols);
                }
            }
        }
        "dist-exchange" => {
            let expected_hosts = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let (result, kernel) = run_dist_exchange_timed(rows, cols, steps, expected_hosts);
            let elapsed = timer.elapsed();
            let myid = get_machine_id();

            println!(
                "mode=dist-exchange host={myid} rows={rows} cols={cols} steps={steps} hosts={expected_hosts}"
            );
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());

            if let Some(grid) = result {
                println!("checksum: {:.6}", checksum(&grid));
                println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

                if should_print {
                    print_grid(&grid, rows, cols);
                }
            }
        }
        "mpi-exchange" => {
            let expected_ranks = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            let (result, kernel) = run_mpi_exchange_timed(rows, cols, steps, expected_ranks);
            let elapsed = timer.elapsed();

            if let Some(grid) = result {
                println!(
                    "mode=mpi-exchange rows={rows} cols={cols} steps={steps} ranks={expected_ranks}"
                );
                println!("elapsed: {:.3?}", elapsed);
                println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
                println!("checksum: {:.6}", checksum(&grid));
                println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

                if should_print {
                    print_grid(&grid, rows, cols);
                }
            }
        }
        _ => {
            let threads = positional
                .get(arg_base + 3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(4);
            let (grid, kernel) = run_threads_timed(rows, cols, steps, threads);
            let elapsed = timer.elapsed();

            println!("mode=threads rows={rows} cols={cols} steps={steps} threads={threads}");
            println!("elapsed: {:.3?}", elapsed);
            println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
            println!("checksum: {:.6}", checksum(&grid));
            println!("center: {:.6}", grid[idx(rows / 2, cols / 2, cols)]);

            if should_print {
                print_grid(&grid, rows, cols);
            }
        }
    }
    // The distributed engines print their own metrics snapshot (mpi-exchange
    // may substitute the IP-over-CXL bridge counters), so printing again here
    // would double-count in the harness' summing parser.
    if !matches!(
        mode,
        "dist" | "dist-opt" | "dist-opt-opt" | "dist-exchange" | "mpi-exchange"
    ) {
        print_bench_metrics();
    }
}
