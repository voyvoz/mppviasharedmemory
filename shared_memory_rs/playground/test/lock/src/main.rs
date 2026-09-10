use std::thread;
use std::time::Duration;

use clap::Parser;
use shm_cxl::com::barrier::CXLBarrier;
use shm_cxl::com::barrier::CXLBarrierOptions;
use shm_cxl::com::barrier::barrier_wait_all;
use shm_cxl::com::barrier::set_global_barrier;
use shm_cxl::core::host::LEADER_ID;
use shm_cxl::core::host::get_machine_id;
use shm_cxl::cxl::CXLSharedState;
use shm_cxl::mutex::common::CXLMutex;
use shm_cxl::mutex::common::TreesLock;
use shm_cxl::mutex::n_host::{BakeryLock, TournamentPetersonLock};
use shm_cxl::mutex::two_host::PetersonLock;
use shm_cxl::{
    core::{Cacheline, cxl_read_cacheline_sync},
    cxl_write_sync,
    utils::{init_n_host_ids, now_nsec, open_test_cxl_dev},
};

fn run_peterson(cxl_mem: *mut Cacheline, args: &Cli) {
    let nhosts = args.num_hosts as usize;
    let n = args.n as usize;
    let myid = get_machine_id();

    if args.num_hosts != 2 {
        panic!("PetersonLock requires num_hosts==2");
    }

    let l = PetersonLock::init(cxl_mem).unwrap();
    let counter_cl = unsafe { cxl_mem.add(3) };
    cxl_write_sync!((counter_cl, 0));

    let stepcnt = if myid == 0 {
        n - (nhosts - 1) * (n / nhosts)
    } else {
        n / nhosts
    };
    let mut steps = vec![0; stepcnt];

    // Bench
    barrier_wait_all();
    let ts = now_nsec();
    for i in 0..stepcnt {
        l.lock();

        steps[i] = cxl_read_cacheline_sync(counter_cl).d[0];
        cxl_write_sync!((counter_cl, steps[i] + 1));

        l.unlock();
    }
    let te = (now_nsec() - ts) / (n as u128);
    // End of Bench
    //
    println!("Finished in {te}ns/step");

    barrier_wait_all();
    let final_step = cxl_read_cacheline_sync(counter_cl).d[0];

    if final_step != n as u64 {
        println!("Error {final_step} != {n}");
        for i in steps {
            print!("{:?} ", i);
        }
    }
    println!("");
}

fn run_tournament(cxl_mem: *mut Cacheline, args: &Cli) {
    let nhosts = args.num_hosts as usize;
    let n = args.n as usize;
    let myid = get_machine_id();

    let l = TournamentPetersonLock::init(cxl_mem, nhosts);
    let counter_cl = unsafe { cxl_mem.add(l.required_cxl_shared_state_cachelines()) };
    cxl_write_sync!((counter_cl, 0));

    let stepcnt = n / nhosts;
    let mut steps = vec![0; stepcnt];

    // Bench
    barrier_wait_all();
    let ts = now_nsec();
    for i in 0..stepcnt {
        l.lock();

        steps[i] = cxl_read_cacheline_sync(counter_cl).d[0];
        cxl_write_sync!((counter_cl, steps[i] + 1));

        l.unlock();
    }
    let te = (now_nsec() - ts) / (n as u128);
    // End of Bench
    //
    println!("Finished in {te}ns/step");

    barrier_wait_all();
    let final_step = cxl_read_cacheline_sync(counter_cl).d[0];

    if final_step != n as u64 {
        println!("Error {final_step} != {n}");
        for i in steps {
            print!("{:?} ", i);
        }
    }
    println!("");
}

fn run_bakery(cxl_mem: *mut Cacheline, args: &Cli) {
    let nhosts = args.num_hosts as usize;
    let n = args.n as usize;
    let myid = get_machine_id();

    let l = BakeryLock::init(cxl_mem, nhosts);

    let counter_cl = unsafe { cxl_mem.add(l.required_cxl_shared_state_cachelines()) };
    cxl_write_sync!((counter_cl, 0));

    let stepcnt: usize;
    if myid == 0 {
        stepcnt = n - (nhosts - 1) * (n / nhosts);
    } else {
        stepcnt = n / nhosts;
    }
    let mut steps = vec![0; stepcnt];

    // Bench
    barrier_wait_all();
    let ts = now_nsec();
    for i in 0..stepcnt {
        l.lock();

        steps[i] = cxl_read_cacheline_sync(counter_cl).d[0];
        cxl_write_sync!((counter_cl, steps[i] + 1));

        l.unlock();
    }
    let te = (now_nsec() - ts) / (n as u128);
    // End of Bench
    //
    println!("Finished in {te}ns/step");

    barrier_wait_all();
    let final_step = cxl_read_cacheline_sync(counter_cl).d[0];

    if final_step != n as u64 {
        println!("Error {final_step} != {n}");
        for i in steps {
            print!("{:?} ", i);
        }
    }
    println!("");
}

#[derive(Clone, Debug, Parser)]
pub struct Cli {
    #[arg(short, long, default_value_t = 2)]
    pub num_hosts: u16,

    #[arg(long)]
    pub locktype: String,

    #[arg(long, default_value_t = 100)]
    pub n: u32,
}

///
/// This must be run on two hosts at the same time!
///
fn main() {
    let args = Cli::parse();
    println!("Args: {args:?}");

    assert!(args.n as usize % args.num_hosts as usize == 0);

    let (mut dev, _size) = open_test_cxl_dev();
    init_n_host_ids(dev, args.num_hosts, Some(5000));

    let b_opts = CXLBarrierOptions::from_hosts(args.num_hosts as usize);
    let b = if get_machine_id() == LEADER_ID as u16 {
        CXLBarrier::init(dev, b_opts)
    } else {
        thread::sleep(Duration::from_millis(100));
        CXLBarrier::from_cl(dev, b_opts).unwrap()
    };
    dev = unsafe { dev.add(b.required_cxl_shared_state_cachelines()) };
    set_global_barrier(b);

    match args.locktype.as_str() {
        "peterson" => run_peterson(dev, &args),
        "pt" => run_tournament(dev, &args),
        "bakery" => run_bakery(dev, &args),
        _ => panic!("Invalid locktype"),
    };
}
