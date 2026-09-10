use clap::{Parser, ValueEnum};
use cxl_kmeans::{
    KMeansConfig, kmeans, load_points, run_kmeans_cxl, run_kmeans_cxl_ipc, run_kmeans_mpi,
    run_synthetic_kmeans_cxl, run_synthetic_kmeans_cxl_ipc, run_synthetic_kmeans_mpi,
    synthetic_blobs,
};
use shm_cxl::core::host::get_machine_id;
use shm_cxl::utils::timestamps::{print_bench_metrics, reset_bench_metrics};
use std::process::exit;
use std::time;

// single host
//  cargo build -p kmeans
// ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7

// cxl
// cargo build -p kmeans
// export CXL_MAPOFFSET=0x600000000
// export CXL_MAPSIZE=32G
// ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7 --cxl-hosts 2
// ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7 --cxl-hosts 3 --cxl-mode shared
// ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7 --cxl-hosts 3 --cxl-mode ipc

// ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7 --cxl-hosts 3

// mpi
// cargo build -p cxl-kmeans
// mpirun -np 3 ./target/debug/kmeans --points 256 --dim 2 --k 3 --seed 7 --mpi

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum CxlMode {
    #[default]
    Shared,
    Ipc,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, default_value_t = 256)]
    points: usize,

    #[arg(long, default_value_t = 2)]
    dim: usize,

    #[arg(short, long, default_value_t = 3)]
    k: usize,

    #[arg(long, default_value_t = 0)]
    blobs: usize,

    #[arg(long, default_value_t = 0.75)]
    spread: f64,

    #[arg(long, default_value_t = 100)]
    max_iters: usize,

    #[arg(long, default_value_t = 1e-4)]
    tol: f64,

    #[arg(long, default_value_t = 0xC0FFEE)]
    seed: u64,

    #[arg(long, default_value_t = 0)]
    cxl_hosts: u16,

    #[arg(long, value_enum, default_value_t = CxlMode::Shared)]
    cxl_mode: CxlMode,

    /// Run the real-MPI backend instead of CXL-shared/CXL-IPC. Launched via
    /// `mpirun -np N`; rank count comes from the launcher, not from
    /// `--cxl-hosts` (mutually exclusive with it).
    #[arg(long)]
    mpi: bool,

    /// Sanity-check the actual MPI rank count against this value (0 = skip).
    /// Does not control how many ranks run -- `mpirun -np` does that.
    #[arg(long, default_value_t = 0)]
    expect_ranks: u16,

    /// Delimited text file (CSV / whitespace) with one point per line. When set,
    /// `--points`/`--dim`/`--blobs`/`--spread` are ignored and every host reads
    /// this same file, so all engines cluster an identical dataset.
    #[arg(long)]
    input: Option<String>,

    /// Leading lines to drop (a CSV header).
    #[arg(long, default_value_t = 0)]
    skip_rows: usize,

    /// Leading columns to drop per line (e.g. HIGGS' label column).
    #[arg(long, default_value_t = 0)]
    skip_cols: usize,

    /// Columns to keep after `--skip-cols` (0 = all remaining).
    #[arg(long, default_value_t = 0)]
    input_cols: usize,

    /// Truncate the input file to this many points (0 = all).
    #[arg(long, default_value_t = 0)]
    max_points: usize,
} 

fn main() {
    reset_bench_metrics();
    let args = Cli::parse();
    let blobs = if args.blobs == 0 { args.k } else { args.blobs };

    let config = KMeansConfig {
        k: args.k,
        max_iters: args.max_iters,
        seed: args.seed,
        tol: args.tol,
    };

    // A file input is read up front on every host so the shared and IPC engines
    // partition byte-identical data; the synthetic paths keep generating inside
    // the engines as before.
    let loaded = args.input.as_deref().map(|path| {
        let start = time::SystemTime::now();
        match load_points(
            path,
            args.skip_rows,
            args.skip_cols,
            args.input_cols,
            args.max_points,
        ) {
            Ok((points, dim)) => {
                let n = points.len() / dim;
                let load = start.elapsed().unwrap();
                println!("Loaded {path} in {load:?} ({n} points, dim {dim})");
                println!("BENCH_LOAD_NS={}", load.as_nanos());
                (points, dim)
            }
            Err(e) => {
                eprintln!("Failed to load {path}: {e}");
                exit(1);
            }
        }
    });

    if args.mpi && args.cxl_hosts >= 2 {
        eprintln!("--mpi and --cxl-hosts are mutually exclusive");
        exit(1);
    }

    if args.mpi {
        let start = time::SystemTime::now();
        let dim = loaded.as_ref().map_or(args.dim, |(_, d)| *d);
        let (result, rank) = match &loaded {
            Some((points, _)) => run_kmeans_mpi(points, dim, &config, args.expect_ranks),
            None => run_synthetic_kmeans_mpi(
                args.points,
                args.dim,
                blobs,
                args.spread,
                &config,
                args.expect_ranks,
            ),
        };

        if rank == 0 {
            let (n_points, source) = match &loaded {
                Some((points, dim)) => (points.len() / dim, args.input.clone().unwrap()),
                None => (args.points, "<synthetic>".to_string()),
            };
            println!("Args: {args:?}");
            println!("Dataset: points={n_points} dim={dim} source={source} mode=<mpi>");
            println!("[kmeans-mpi] {:?}", start.elapsed().unwrap());
            println!("BENCH_KERNEL_NS={}", result.kernel_elapsed.as_nanos());
            println!(
                "[kmeans-mpi] converged={} iterations={} inertia={:.6}",
                result.converged, result.iterations, result.inertia
            );

            for (cluster, center) in result.centers.chunks_exact(dim).enumerate() {
                println!(
                    "cluster {cluster}: count={} center={center:?}",
                    result.counts[cluster]
                );
            }
        }
        return;
    }

    if args.cxl_hosts >= 2 {
        let start = time::SystemTime::now();
        let (result, dim) = match (&loaded, args.cxl_mode) {
            (Some((points, dim)), CxlMode::Shared) => {
                (run_kmeans_cxl(points, *dim, &config, args.cxl_hosts), *dim)
            }
            (Some((points, dim)), CxlMode::Ipc) => (
                run_kmeans_cxl_ipc(points, *dim, &config, args.cxl_hosts),
                *dim,
            ),
            (None, CxlMode::Shared) => (
                run_synthetic_kmeans_cxl(
                    args.points,
                    args.dim,
                    blobs,
                    args.spread,
                    &config,
                    args.cxl_hosts,
                ),
                args.dim,
            ),
            (None, CxlMode::Ipc) => (
                run_synthetic_kmeans_cxl_ipc(
                    args.points,
                    args.dim,
                    blobs,
                    args.spread,
                    &config,
                    args.cxl_hosts,
                ),
                args.dim,
            ),
        };

        if get_machine_id() == 1 {
            let (n_points, source) = match &loaded {
                Some((points, dim)) => (points.len() / dim, args.input.clone().unwrap()),
                None => (args.points, "<synthetic>".to_string()),
            };
            println!("Args: {args:?}");
            println!(
                "Dataset: points={n_points} dim={dim} source={source} mode=<cxl> hosts={}",
                args.cxl_hosts
            );
            println!("[kmeans-cxl] {:?}", start.elapsed().unwrap());
            println!("BENCH_KERNEL_NS={}", result.kernel_elapsed.as_nanos());
            println!(
                "[kmeans-cxl] converged={} iterations={} inertia={:.6}",
                result.converged, result.iterations, result.inertia
            );

            for (cluster, center) in result.centers.chunks_exact(dim).enumerate() {
                println!(
                    "cluster {cluster}: count={} center={center:?}",
                    result.counts[cluster]
                );
            }
        }
    } else {
        let (points, dim, source) = match loaded {
            Some((points, dim)) => (points, dim, args.input.clone().unwrap()),
            None => (
                synthetic_blobs(args.points, args.dim, blobs, args.spread, args.seed),
                args.dim,
                "<synthetic>".to_string(),
            ),
        };
        println!("Args: {args:?}");
        println!(
            "Dataset: points={} dim={dim} source={source} mode=<single>",
            points.len() / dim
        );
        let start = time::SystemTime::now();
        let result = kmeans(&points, dim, &config);
        let kernel = start.elapsed().unwrap();
        println!("[kmeans] {kernel:?}");
        println!("BENCH_KERNEL_NS={}", kernel.as_nanos());
        println!(
            "[kmeans] converged={} iterations={} inertia={:.6}",
            result.converged, result.iterations, result.inertia
        );

        for (cluster, center) in result.centers.chunks_exact(dim).enumerate() {
            println!(
                "cluster {cluster}: count={} center={center:?}",
                result.counts[cluster]
            );
        }
    }
    // The MPI engine prints its own snapshot (possibly the IP-over-CXL bridge
    // counters), so printing again would double-count in the summing parser.
    if args.cxl_hosts < 2 && !args.mpi {
        print_bench_metrics();
    }
}
