use clap::Parser;
use mpi::{
    self,
    traits::{Communicator, CommunicatorCollectives},
};
use shm_cxl::{
    com::barrier::{CXLBarrier, CXLBarrierOptions},
    core::host::{LEADER_ID, get_machine_cnt, get_machine_id},
    utils::{init_n_host_ids, open_test_cxl_dev},
};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    #[arg(short, long, default_value_t = 0)]
    pub num_hosts: usize,

    #[arg(long)]
    pub variant: String,

    #[arg(long, default_value_t = 1000000)]
    pub target: u32,
}

fn run_mpi(args: &Cli) -> (u16, Duration, Duration) {
    let start = std::time::Instant::now();
    // -- Init

    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let init_end = start.elapsed();

    let loop_start = std::time::Instant::now();
    // -- Loop

    let mut counter = 0;
    loop {
        world.barrier();

        counter += 1;
        if counter > args.target {
            break;
        }
    }

    (world.rank() as u16, init_end, loop_start.elapsed())
}

fn run_cxl(args: &Cli) -> (u16, Duration, Duration) {
    let mut num_hosts = args.num_hosts;
    let start = std::time::Instant::now();
    // -- Init

    let (dev, _size) = open_test_cxl_dev();
    let myid = init_n_host_ids(dev, num_hosts as u16, None); // wait for 2 seconds
    num_hosts = get_machine_cnt() as usize;
    let b_otsp = CXLBarrierOptions::from_hosts(num_hosts);
    let b = if myid == LEADER_ID as u16 {
        CXLBarrier::init(dev, b_otsp)
    } else {
        thread::sleep(Duration::from_millis(10));
        CXLBarrier::from_cl(dev, b_otsp).unwrap()
    };

    let init_end = start.elapsed();

    b.wait_for_all(); //sync before we start the measurement

    let loop_start = std::time::Instant::now();
    // -- Loop

    let mut counter = 0;
    loop {
        b.wait_for_all();

        counter += 1;
        if counter > args.target {
            break;
        }
    }

    (get_machine_id(), init_end, loop_start.elapsed())
}

fn main() {
    let _ = env_logger::try_init();

    // Init CXL
    let args = Cli::parse();
    println!("Args: {args:?}");

    let res = match args.variant.as_str() {
        "mpi" => run_mpi(&args),
        "cxl" => run_cxl(&args),
        _ => panic!("required --variant mpi|cxl"),
    };

    println!(
        "[{}] {:?} {:?} {:?} {:?}",
        res.0,
        res.1,
        res.2,
        args.target,
        res.2 / args.target
    );
}
