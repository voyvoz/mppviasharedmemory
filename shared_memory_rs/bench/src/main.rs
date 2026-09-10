//! CXL IPC Benchmark Harness
//!
//! Usage:
//!   Plan A (1 machine, 2 threads):  cargo run --release -p bench
//!   Plan B (2 machines, init):      cargo run --release -p bench -- --plan=b --role=init
//!   Plan B (2 machines, follower):  cargo run --release -p bench -- --plan=b --role=follower

use bench::harness::calibrate_tsc;
use shm_cxl::utils::open_test_cxl_dev;

fn print_help() {
    println!("CXL IPC Benchmark Harness");
    println!();
    println!("Usage:");
    println!("  Plan A (1 machine, 2 threads):     cargo run --release -p bench");
    println!("  Plan B (2 machines, init side):    cargo run --release -p bench -- --plan=b --role=init");
    println!("  Plan B (2 machines, follower):     cargo run --release -p bench -- --plan=b --role=follower");
    println!();
    println!("Plan B startup order:");
    println!("  1. Start on machine 1 first:  cargo run --release -p bench -- --plan=b --role=follower");
    println!("  2. Then start on machine 0:   cargo run --release -p bench -- --plan=b --role=init");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut plan = "a".to_string();
    let mut role = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--plan" | "-p" => { i += 1; plan = args[i].clone(); }
            a if a.starts_with("--plan=") => { plan = a[7..].to_string(); }
            "--role" | "-r" => { i += 1; role = args[i].clone(); }
            a if a.starts_with("--role=") => { role = a[7..].to_string(); }
            "--help" | "-h" => { print_help(); return; }
            _ => { eprintln!("unknown argument: {}", args[i]); return; }
        }
        i += 1;
    }

    let freq_hz = calibrate_tsc();

    match (plan.as_str(), role.as_str()) {
        // ── Plan B ──
        ("b", "init") => {
            let (base, _) = open_test_cxl_dev();
            println!("[main] CXL mapping: base={base:p}");
            bench::planb::run_init(freq_hz, base);
        }
        ("b", "follower") => {
            let (base, _) = open_test_cxl_dev();
            println!("[main] CXL mapping: base={base:p}");
            bench::planb::run_follower(freq_hz, base);
        }
        ("b", _) => {
            eprintln!("Plan B requires --role=init or --role=follower");
            std::process::exit(1);
        }
        // ── Plan A (default) ──
        _ => {
            let (base, _) = open_test_cxl_dev();
            println!("[main] CXL mapping: base={base:p}");
            bench::plana::run(freq_hz, base);
        }
    }
}