use crate::cxl_radix::CxlRadixSort;
use crate::radix::radix_sort_parallel;
use crate::utils::{
    Cli, fingerprint_generated, fingerprint_slice, gen_keys_range, is_sorted, resolve_threads,
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

/// Check a result against the generated input without sorting a reference copy.
///
/// Non-decreasing order plus an identical multiset fingerprint means the result
/// is the sequentially sorted input.
fn validate_sorted(label: &str, got: &[u64], n: usize, seed: u64, threads: usize) {
    let sorted = is_sorted(got, threads);
    let expected = fingerprint_generated(n, seed, threads);
    let actual = fingerprint_slice(got, threads);
    if sorted && got.len() == n && actual == expected {
        println!(
            "<<- {label}: matches sequential baseline ({} keys sorted, ordered and fingerprint-identical to the input) ->>",
            got.len()
        );
    } else {
        println!(
            "<<- {label} FAIL: sorted={sorted}, len={} expected {n}, fingerprint {actual:?} expected {expected:?} ->>",
            got.len()
        );
    }
}

#[cfg(feature = "mpi")]
fn run_mpi(args: &Cli, threads: usize) {
    let result = crate::mpi_radix::run_mpi_radix(args.nodes, args.seed, threads);
    println!(
        "[mpi_radix] rank {}/{} finished in {:?}",
        result.rank, result.ranks, result.kernel
    );
    println!("BENCH_KERNEL_NS={}", result.kernel.as_nanos());
    if let Some(valid) = result.validation {
        if valid {
            println!(
                "<<- mpi_radix: matches sequential baseline ({} keys sorted) ->>",
                result.total_keys
            );
        } else {
            println!("<<- mpi_radix FAIL: distributed validation failed ->>");
        }
    }
}

#[cfg(not(feature = "mpi"))]
fn run_mpi(_args: &Cli, _threads: usize) {
    eprintln!("--mpi requires cxl-sort-mpi (build cxl-sort with --features mpi)");
    exit(1);
}

fn run_local(args: &Cli, threads: usize) {
    let start = Instant::now();
    let sorted = radix_sort_parallel(gen_keys_range(args.seed, 0, args.nodes), threads);
    let kernel = start.elapsed();
    println!(
        "[local_radix] {} keys with {threads} threads in {kernel:?}",
        args.nodes
    );
    println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
    validate_sorted("local_radix", &sorted, args.nodes, args.seed, threads);
}

fn run_cxl(args: &Cli, threads: usize) {
    let mut persist: PersistantDataStructs = cxl_init(args.num_hosts as u16);
    let myid = get_machine_id();
    let sort = if myid == 1 {
        let (sort, descriptor) =
            CxlRadixSort::create(persist.alloc.as_ref(), args.nodes, args.seed);
        for peer in 2..=args.num_hosts {
            persist
                .ipc
                .try_send_to(peer as PeerMachineID, &descriptor)
                .unwrap();
        }
        sort
    } else {
        let descriptor = persist
            .ipc
            .try_pop_from(1 as PeerMachineID, Some(Duration::from_secs(600)))
            .unwrap()
            .1;
        CxlRadixSort::attach(&descriptor)
    };

    barrier_wait_all();
    reset_bench_metrics();
    let start = Instant::now();
    let local = sort.run(threads);
    let kernel = start.elapsed();
    let kernel_metrics = get_bench_metrics();
    println!(
        "[cxl_radix] host {myid}/{} finished in {kernel:?}",
        args.num_hosts
    );
    println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
    // Gathering the distributed result into one readable shared array belongs
    // to validation, not to the sort, so it stays outside BENCH_KERNEL_NS.
    let publish_started = Instant::now();
    sort.publish(&local, threads);
    println!(
        "[cxl_radix_phase] host {myid}/{} publish={:?}",
        args.num_hosts,
        publish_started.elapsed()
    );
    barrier_wait_all();
    if myid == 1 {
        validate_sorted("cxl_radix", &sort.collect(), args.nodes, args.seed, threads);
    }
    print_bench_metrics_value(kernel_metrics);
}

pub fn run() {
    reset_bench_metrics();
    let args = Cli::parse();
    let threads = resolve_threads(args.threads);
    println!("Args: {args:?}");

    if args.mpi {
        run_mpi(&args, threads);
        if !ip_cxl_metrics_requested() {
            print_bench_metrics();
        }
    } else if args.num_hosts >= 2 {
        run_cxl(&args, threads);
    } else {
        run_local(&args, threads);
        print_bench_metrics();
    }
}
