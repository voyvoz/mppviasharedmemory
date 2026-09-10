#[macro_use]
extern crate log;
use crate::formula::Lit;
use crate::sat::minisat::budget::Budget;
use crate::sat::*;
use crate::util::{all_reduce, create_cxl_transport, create_mpi_transport, on_fabric_id};
use shm_cxl::utils::timestamps::{
    ip_cxl_metrics_requested, print_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
    IpCxlMetricsGuard,
};
use std::{fs, io, path};
use time;

pub mod sat;
pub(crate) mod util;

#[derive(Debug)]
pub enum SolverOptions {
    Core(minisat::CoreSettings),
    Simp(minisat::SimpSettings),
}


pub struct MainOptions {
    pub strict: bool,
    pub pre: bool,
    pub solve: bool,
    pub in_path: path::PathBuf,
    pub out_path: Option<path::PathBuf>,
    pub dimacs_path: Option<path::PathBuf>,
    pub fabric: String,
}

pub fn solve(main_opts: MainOptions, solver_opts: SolverOptions) -> io::Result<()> {
    reset_bench_metrics();

    // --- Fabric Init
    match main_opts.fabric.as_str() {
        "MPI" => create_mpi_transport(),
        "CXL" => create_cxl_transport(),
        _ => {}
    }

    // -- Solve
    let local_result = match solver_opts {
        SolverOptions::Core(mut opts) => {
            // -- Solve prep
            // make it a portfolio solver
            opts.heur.random_seed = opts.heur.random_seed + on_fabric_id() as f64;
            let solver = minisat::CoreSolver::new(opts);

            // -- Solve
            solve_with(solver, main_opts, &[])
        }
        SolverOptions::Simp(opts) => {
            let mut solver = minisat::SimpSolver::new(opts);
            if !main_opts.pre {
                solver.preprocess(&Budget::new());
            }
            solve_with(solver, main_opts, &[])
        }
    }?;

    // -- Gather result
    all_reduce(local_result);

    if !ip_cxl_metrics_requested() {
        print_bench_metrics();
    }
    Ok(())
}

pub fn solve_with<S: Solver>(
    mut solver: S,
    options: MainOptions,
    assumptions: &[Lit],
) -> io::Result<LocalResult> {
    //info!("============================[ Problem Statistics ]=============================");
    //info!("|                                                                             |");

    let initial_time = time::precise_time_s();
    let load_start = std::time::Instant::now();
    let backward_subst = dimacs::parse_file(&options.in_path, &mut solver, options.strict)?;
    println!("BENCH_LOAD_NS={}", load_start.elapsed().as_nanos());
    //let parse_end_time = time::precise_time_s();

    //info!(
    //    "|  Number of variables:  {:12}                                         |",
    //    solver.n_vars()
    //);
    //info!(
    //    "|  Number of clauses:    {:12}                                         |",
    //    solver.n_clauses()
    //);

    //let parse_time = parse_end_time - initial_time;
    //info!(
    //    "|  Parse time:           {:12.2} s                                       |",
    //    parse_time
    //);

    let mut budget = Budget::new();
    budget.off();

    // Lets no do that yet as we might loose our preset variables?
    // But is that even a problem?
    let elim_res = true; //solver.preprocess(&budget);

    //let simplify_time = time::precise_time_s() - parse_end_time;
    //info!(
    //    "|  Simplification time:  {:12.2} s                                       |",
    //    simplify_time
    //);
    //info!("|                                                                             |");

    // Clock covers the search alone -- MPI init, DIMACS parsing and result
    // reporting are excluded.
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    let kernel_start = std::time::Instant::now();
    let result = if !elim_res {
        info!("===============================================================================");
        info!("Solved by simplification");
        SolveRes::UnSAT(Stats::default())
    } else {
        let result = if options.solve {
            solver.solve_limited(&budget, assumptions)
        } else {
            info!(
                "==============================================================================="
            );
            SolveRes::Interrupted(0.0, solver)
        };

        //            if let TotalResult::Interrupted = result {
        //                if let Some(path) = options.dimacs_path {
        //                    let mut out = try!(fs::File::create(path));
        //                    try!(dimacs::write(&mut out, &solver));
        //                }
        //            }

        result
    };
    let kernel = kernel_start.elapsed();
    if let Some(metrics) = ip_cxl_metrics.finish() {
        print_bench_metrics_value(metrics);
    }
    println!("BENCH_KERNEL_NS={}", kernel.as_nanos());

    let cpu_time = time::precise_time_s() - initial_time;
    let mem_used = util::mem_used_peak();
    let mut ret_val = LocalResult {
        sat: -1,
        cpu_time,
        mem_used,
        stats: Stats::default(),
    };
    match result {
        SolveRes::Interrupted(_, ref s) => {
            ret_val.stats = s.stats();
        }

        SolveRes::UnSAT(ref stats) => {
            ret_val.stats = *stats;
            ret_val.sat = 0;
        }
        SolveRes::SAT(ref model, ref stats) => {
            assert!(
                dimacs::validate_model_file(&options.in_path, &backward_subst, &model)?,
                "SELF-CHECK FAILED"
            );
            ret_val.stats = *stats;
            ret_val.sat = 1;
        }
    }

    if let Some(path) = options.out_path {
        dimacs::write_result(fs::File::create(path)?, result, &backward_subst)?;
    }

    Ok(ret_val)
}

fn print_stats(stats: &Stats, cpu_time: f64, mem_used: Option<usize>) {
    info!("restarts              : {:<12}", stats.restarts);

    {
        let confl_per_s = (stats.conflicts as f64) / cpu_time;
        info!(
            "conflicts             : {:<12}   ({:.0} /sec)",
            stats.conflicts, confl_per_s
        );
    }

    {
        let rnd_percent = (stats.rnd_decisions as f64) * 100.0 / (stats.decisions as f64);
        let decisions_per_s = (stats.decisions as f64) / cpu_time;
        info!(
            "decisions             : {:<12}   ({:4.2} % random) ({:.0} /sec)",
            stats.decisions, rnd_percent, decisions_per_s
        );
    }

    {
        let props_per_s = (stats.propagations as f64) / cpu_time;
        info!(
            "propagations          : {:<12}   ({:.0} /sec)",
            stats.propagations, props_per_s
        );
    }

    {
        let del_percent = (stats.del_literals as f64) * 100.0
            / ((stats.del_literals + stats.tot_literals) as f64);
        info!(
            "conflict literals     : {:<12}   ({:4.2} % deleted)",
            stats.tot_literals, del_percent
        );
    }

    if let Some(mem_used) = mem_used {
        info!(
            "Memory used           : {:.2} MB",
            mem_used as f32 / 1024f32
        );
    }
    info!("CPU time              : {} s", cpu_time);
    info!("");
}
