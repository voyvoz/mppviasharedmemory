mod parser;
mod util;

use std::time::{Duration, Instant};

use crate::util::{barrier, is_root, on_fabric_hosts, on_fabric_id, reset, termination_check};
use crate::{
    parser::parse_file,
    util::{all_reduce_max, create_cxl_transport, create_mpi_transport},
};
use clap::Parser;
use log::info;
use shm_cxl::utils::timestamps::{
    IpCxlMetricsGuard, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

#[derive(Debug, Clone)]
pub struct Item {
    pub id: u64,
    pub profit: u64,
    pub weight: u64,
    pub in_optimal: bool,
}

/// Timing breakdown of one instance, aggregated over the whole input by `main`.
///
/// `search` + `reduction` + `termination` add up to `kernel`, which splits the
/// local branch-and-bound work from the two global operations it synchronizes
/// on. Only `kernel` feeds `BENCH_KERNEL_NS`; the rest is reported for analysis.
#[derive(Default, Clone, Copy, Debug)]
pub struct SolveStats {
    pub best: u64,
    pub kernel: Duration,
    pub reduction: Duration,
    pub termination: Duration,
    pub reductions: u64,
    pub termination_checks: u64,
}

impl SolveStats {
    fn search(&self) -> Duration {
        self.kernel
            .saturating_sub(self.reduction)
            .saturating_sub(self.termination)
    }

    fn accumulate(&mut self, other: &SolveStats) {
        self.kernel += other.kernel;
        self.reduction += other.reduction;
        self.termination += other.termination;
        self.reductions += other.reductions;
        self.termination_checks += other.termination_checks;
    }
}

struct Node {
    idx: usize, // next item to decide
    weight: u64,
    profit: u64,
    bound: u64, // LP-relaxation upper bound (fractional greedy)
}

fn bound(node: &Node, items: &[Item], capacity: u64) -> u64 {
    // standard fractional-relaxation bound.
    // greedily fill remaining capacity, allow fractional last item
    let mut w = node.weight;
    let mut b = node.profit as f64;
    for i in node.idx..items.len() {
        if w + items[i].weight <= capacity {
            w += items[i].weight;
            b += items[i].profit as f64;
        } else {
            let remain = capacity - w;
            b += remain as f64 * (items[i].profit as f64 / items[i].weight as f64);
            break;
        }
    }
    b as u64
}

fn build_initial_stack(items: &[Item], capacity: u64, num_hosts: usize, my_id: usize) -> Vec<Node> {
    assert!(num_hosts > 0);
    let k = (usize::BITS - (num_hosts.max(1) - 1).leading_zeros()) as usize; // ceil(log2(num_hosts)), 0 if num_hosts==1
    assert!(
        k <= items.len(),
        "more hosts ({num_hosts}) than usable partition bits for {} items",
        items.len()
    );
    let num_prefixes = 1usize << k;

    let mut stack = Vec::new();
    for prefix in 0..num_prefixes {
        if prefix % num_hosts != my_id {
            continue;
        } // round-robin ownership, covers every prefix exactly once (extras wrap)

        let mut weight = 0u64;
        let mut profit = 0u64;
        let mut feasible = true;
        for bit in 0..k {
            if (prefix >> bit) & 1 == 1 {
                let item = &items[bit];
                if weight + item.weight > capacity {
                    feasible = false;
                    break;
                } // this prefix is infeasible, skip whole subtree
                weight += item.weight;
                profit += item.profit;
            }
        }
        if !feasible {
            continue;
        }

        let node = Node {
            idx: k,
            weight,
            profit,
            bound: 0,
        };
        let b = bound(&node, items, capacity);
        stack.push(Node { bound: b, ..node });
    }
    stack
}

fn solver(mut items: Vec<Item>, capacity: u64, args: &Cli) -> SolveStats {
    // The clock starts here: item ordering and the initial stack split are part
    // of the algorithm. Parsing, fabric attach and the per-instance `reset()`
    // happen in `main` and stay outside.
    let kernel_start = Instant::now();
    let mut stats = SolveStats::default();

    // Sort by profit/weight ratio
    items.sort_by(|a, b| {
        let ra = a.profit as f64 / a.weight as f64;
        let rb = b.profit as f64 / b.weight as f64;
        rb.partial_cmp(&ra).unwrap()
    });

    let mut local_best: u64 = 0;
    let mut stack =
        build_initial_stack(&items, capacity, on_fabric_hosts(), on_fabric_id() as usize);
    let mut reduction_counter: u64 = 0;

    loop {
        let mut steps_this_round: u64 = 0;
        while steps_this_round < args.steps_until_global_reduction {
            steps_this_round += 1;
            let Some(node) = stack.pop() else { break };

            if node.bound <= local_best {
                continue;
            }
            if node.idx == items.len() {
                if node.profit > local_best {
                    local_best = node.profit;
                }
                continue;
            }
            let item = &items[node.idx];
            if node.weight + item.weight <= capacity {
                let incl = Node {
                    idx: node.idx + 1,
                    weight: node.weight + item.weight,
                    profit: node.profit + item.profit,
                    bound: 0,
                };
                let b = bound(&incl, &items, capacity);
                stack.push(Node { bound: b, ..incl });
            }
            let excl = Node {
                idx: node.idx + 1,
                weight: node.weight,
                profit: node.profit,
                bound: 0,
            };
            let b = bound(&excl, &items, capacity);
            stack.push(Node { bound: b, ..excl });
        }

        // Bound check
        let n = Instant::now();
        local_best = all_reduce_max(local_best);
        stats.reduction += n.elapsed();

        //info!("[{}] {local_best}", on_fabric_id());
        reduction_counter += 1;

        // Termination check
        if reduction_counter % args.reductions_until_termination_check == 0 {
            let n = Instant::now();
            let done = termination_check(stack.is_empty());
            stats.termination_checks += 1;
            if done {
                // Needed here to ensure that all CXL hosts correctly reset their state
                barrier();

                // The latency hiding variant could stop on any state that is
                // not a real read. Two full invalidate->prefetch->read cycles
                // guarantee that the second read observes a line that was both
                // invalidated and prefetched after the barrier above, so every
                // peer's final bound is visible regardless of the entry state.
                for _ in 0..6 {
                    local_best = all_reduce_max(local_best);
                }

                // The drain above is part of stopping, not of the search, so it
                // is billed to the termination check like the check itself.
                stats.termination += n.elapsed();

                // nobody has work left, safe to stop together
                break;
            }
            stats.termination += n.elapsed();
        }
    }

    stats.best = local_best;
    stats.reductions = reduction_counter;
    stats.kernel = kernel_start.elapsed();

    info!(
        "[{}] Reductions: {reduction_counter} in {:?} (steps until reduction {}) Termination checks: {} in {:?} Search: {:?} Result: {local_best}",
        on_fabric_id(),
        stats.reduction,
        args.steps_until_global_reduction,
        stats.termination_checks,
        stats.termination,
        stats.search(),
    );
    stats
}

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    /// Which Fabric to use
    #[arg(long, default_value = "")]
    pub fabric: String,

    /// Path to input file
    #[arg(long, default_value = "")]
    pub input_file: String,

    #[arg(long, default_value_t = 512)]
    pub steps_until_global_reduction: u64,

    /// How many times a reduction happens until we do a termination check
    #[arg(long, default_value_t = 3)]
    pub reductions_until_termination_check: u64,
}

/// Mirrors the clap defaults so `Cli::default()` (used by the tests) describes
/// the same configuration a real run gets. Deriving `Default` instead would set
/// `reductions_until_termination_check` to 0 and divide by zero in `solver`.
impl Default for Cli {
    fn default() -> Self {
        Cli::parse_from(["cxl-bnb"])
    }
}

fn main() {
    let _ = env_logger::try_init();
    reset_bench_metrics();

    let args = Cli::parse();

    match args.fabric.as_str() {
        "MPI" => create_mpi_transport(),
        "CXL" => create_cxl_transport(0),
        "CXLv2" => create_cxl_transport(1),
        _ => {
            info!("Running in local mode")
        }
    }

    if is_root() {
        info!("Args: {args:?}");
    }

    let load_start = Instant::now();
    let instances = parse_file(args.input_file.clone()).unwrap();
    let load = load_start.elapsed();
    println!("BENCH_LOAD_NS={}", load.as_nanos());
    info!(
        "[{}] Loaded {} ({} instances) in {load:?}",
        on_fabric_id(),
        args.input_file,
        instances.len()
    );

    // An input file holds many instances and the run solves all of them, so the
    // reported kernel is their sum. Per-instance times are logged individually.
    // Fabric attach, parsing and the per-instance `reset()` stay outside.
    let ip_cxl_metrics = IpCxlMetricsGuard::start();
    reset_bench_metrics();
    let mut total = SolveStats::default();
    let mut solved = 0u64;

    for i in instances {
        reset();
        // Line the hosts up after `reset()` so the timer measures the search
        // rather than leftover skew, and so no host can zero the sharing
        // cachelines after a faster peer has already published its first bound.
        barrier();

        let (capacity, items) = (i.capacity, i.items);

        let stats = solver(items, capacity, &args);
        total.accumulate(&stats);
        solved += 1;

        let global_best = all_reduce_max(stats.best);
        assert!(global_best == i.optimal, "{global_best} != {}", i.optimal);
        if is_root() {
            info!("{} Result: {global_best} in {:?}", i.name, stats.kernel);
        }

        //break;
    }

    let kernel_metrics = get_bench_metrics();
    // Every instance was asserted against its known optimum above, so reaching
    // here is the verdict the harness looks for in each algorithm's log.
    if is_root() {
        println!(
            "<<- bnb: matches sequential baseline ({solved} instances match their known optimum) ->>"
        );
    }
    println!(
        "[bnb_{}] host {}/{} finished in {:?} ({solved} instances, search {:?}, {} reductions {:?}, {} termination checks {:?})",
        if args.fabric.is_empty() {
            "local"
        } else {
            args.fabric.as_str()
        },
        on_fabric_id(),
        on_fabric_hosts(),
        total.kernel,
        total.search(),
        total.reductions,
        total.reduction,
        total.termination_checks,
        total.termination,
    );
    println!("BENCH_KERNEL_NS={}", total.kernel.as_nanos());
    match ip_cxl_metrics.finish() {
        // MPI over the IP-over-CXL bridge: the bridge owns the CXL traffic, so
        // its snapshot replaces this process' (empty) direct-CXL counters.
        Some(metrics) => print_bench_metrics_value(metrics),
        None => print_bench_metrics_value(kernel_metrics),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn load_basic_problem() -> (u64, Vec<Item>) {
        (
            50,
            vec![
                Item {
                    id: 0,
                    weight: 10,
                    profit: 60,
                    in_optimal: false,
                },
                Item {
                    id: 1,
                    weight: 20,
                    profit: 100,
                    in_optimal: true,
                },
                Item {
                    id: 2,
                    weight: 30,
                    profit: 120,
                    in_optimal: true,
                },
            ],
        )
    }

    #[test]
    fn base() {
        let mut args = Cli::default();
        let (capacity, items) = load_basic_problem();
        args.steps_until_global_reduction = 1024;
        let stats = solver(items, capacity, &args);
        let global_best = all_reduce_max(stats.best);
        assert!(global_best == 220);
    }
}
