use cidr2027::utils::Cli;
use clap::Parser;
use shm_cxl::core::Cacheline;
use shm_cxl::com::barrier::barrier_wait_all;
use shm_cxl::com::ipc::CXLIpc;
use shm_cxl::core::host::set_machine_id;
use shm_cxl::core::host::{PeerMachineID, get_machine_id};
use shm_cxl::cxl::PersistantDataStructs;
use shm_cxl::utils::init_n_host_ids;
use shm_cxl::utils::open_test_cxl_dev;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use std::{thread, time};

#[derive(Default)]
struct Runtimedata {
    msg_counter: u64,
    spin_counter: u64,
    lock: Mutex<u64>,
}

#[derive(Default)]
struct RuntimedataConsumer {
    msg_counters: Vec<u64>,
    spin_empty_counter: u64,
    spin_err_counter: u64,
    lock: Mutex<u64>,
}

// Jeder Host schickt jedem anderen Host X Werte
fn dist_buffers_producer(tid: usize, args: &Cli, ipc: &mut CXLIpc, rt: Arc<RwLock<Runtimedata>>) {
    println!("Producer {tid} using ipc at {:p}", ipc.get_base_cl());

    let start = time::SystemTime::now();
    for i in 0..args.msgs {
        // Host IDs start at 1
        for h in 1..=args.num_hosts {
            // Skip send to self
            if h as u16 == get_machine_id() {
                continue;
            }
            let target_peer = h as PeerMachineID;
            let msg = Cacheline::from_b0(i as u64);
            loop {
                let res = ipc.try_send_to(target_peer, &msg);

                if res.is_ok() {
                    rt.write().unwrap().msg_counter += 1;
                    break;
                }
                rt.write().unwrap().spin_counter += 1;
            }
        }
    }
    println!(
        "[{:?}] Producer [{tid}] prodcued {} with {} spins",
        start.elapsed().unwrap(),
        rt.read().unwrap().msg_counter,
        rt.read().unwrap().spin_counter
    );
}
fn dist_buffer_consumer(
    tid: usize,
    args: &Cli,
    ipc: &mut CXLIpc,
    rt: Arc<RwLock<RuntimedataConsumer>>,
) {
    println!("Consumer {tid} using ipc at {:p}", ipc.get_base_cl());
    let mut local_msg_counter = 0;
    let msg_target = args.msgs * (args.num_hosts - 1);

    let start = time::SystemTime::now();
    loop {
        let res = ipc.try_pop_any();
        if res.is_ok() {
            let inner = res.unwrap();
            if inner.is_some() {
                let msg = inner.unwrap();
                local_msg_counter += 1;

                let host_id = msg.0.machine_id as usize;

                rt.write().unwrap().msg_counters[host_id] += 1;

                let msg_cnt = rt
                    .read()
                    .unwrap()
                    .msg_counters
                    .iter()
                    .fold(0, |acc, e| acc + e) as usize;
                if msg_cnt >= msg_target {
                    println!(
                        "[{:?}] Consumer [{tid}] consumed {} with {} spins",
                        start.elapsed().unwrap(),
                        local_msg_counter,
                        rt.read().unwrap().spin_empty_counter
                    );
                    return;
                }
            }
            rt.write().unwrap().spin_empty_counter += 1;
        } else {
            println!("Consumer try_pop_any error: {:?}", res);
            rt.write().unwrap().spin_err_counter += 1;
        }
    }
}

fn main() {
    let args = Cli::parse();
    println!("Args: {args:?}");

    if args.num_hosts < 2 {
        panic!("Must use at least 2 hosts");
    }

    let (dev, size) = open_test_cxl_dev();

    init_n_host_ids(dev, args.num_hosts as u16, Some(5000));

    let mut persist: PersistantDataStructs;
    let this_machine_id = get_machine_id();
    if this_machine_id == 1 {
        persist = PersistantDataStructs::init(dev, size);
    } else {
        thread::sleep(Duration::from_millis(50));
        persist = PersistantDataStructs::from_cl_sync(dev);
    }
    let base_cl = persist.ipc.get_base_cl() as u64;

    barrier_wait_all();

    println!(
        "Running... target={} msgs per host init={:?}",
        args.msgs, persist.header.init_timestamp
    );

    let rt_prod = Arc::new(RwLock::new(Runtimedata::default()));
    let d = RuntimedataConsumer {
        msg_counters: vec![0u64; args.num_hosts],
        spin_empty_counter: 0,
        spin_err_counter: 0,
        lock: Mutex::new(0),
    };
    let rt_consumer = Arc::new(RwLock::new(d));

    let base_cl = persist.ipc.get_base_cl() as u64;

    // Observer thread
    let timer = timer::Timer::new();
    let rt_prod_clone = rt_prod.clone();
    let rt_consumer_clone = rt_consumer.clone();
    let start = time::SystemTime::now();
    let _g = timer.schedule_repeating(chrono::Duration::seconds(1), move || {
        let (prod_msg, prod_spins) = (
            rt_prod_clone.read().unwrap().msg_counter,
            rt_prod_clone.read().unwrap().spin_counter,
        );

        let cons_spins = rt_consumer_clone.read().unwrap().spin_empty_counter;
        let cons_msg_sum = rt_consumer_clone
            .read()
            .unwrap()
            .msg_counters
            .iter()
            .fold(0, |acc, x| acc + x);

        let t = time::SystemTime::now()
            .duration_since(start)
            .unwrap();
        println!(
            "[{:?}] P:{} msgs/s | C:{} msgs/s | prod={} spins/msg={} | consumed={:?} spins/msg={} errors_spins={}",
            t.as_micros(),
            prod_msg.div_ceil(t.as_secs()),
            cons_msg_sum.div_ceil(t.as_secs()),
            prod_msg,
            (prod_spins as f64 / prod_msg as f64),
            rt_consumer_clone.read().unwrap().msg_counters,
            (cons_spins as f64 / cons_msg_sum as f64),
            rt_consumer_clone.read().unwrap().spin_err_counter,
        );
    });

    let mut threads = vec![];

    // Consumer threads
    for i in 0..args.consumer_threads {
        let args2 = args.clone();
        let rt_consumer_clone = rt_consumer.clone();
        threads.push(thread::spawn(move || {
            set_machine_id(this_machine_id);
            let mut ipc =
                CXLIpc::from_cl_sync(base_cl as *mut Cacheline, persist.options.com.ipc_opts)
                    .unwrap();
            dist_buffer_consumer(i, &args2, &mut ipc, rt_consumer_clone);
        }));
    }

    // Producer threads
    for i in 0..args.producer_threads {
        let args2 = args.clone();
        let rt_prod_clone = rt_prod.clone();

        threads.push(thread::spawn(move || {
            set_machine_id(this_machine_id);
            let mut ipc =
                CXLIpc::from_cl_sync(base_cl as *mut Cacheline, persist.options.com.ipc_opts)
                    .unwrap();
            dist_buffers_producer(i, &args2, &mut ipc, rt_prod_clone);
        }));
    }

    //
    for join_handle in threads {
        join_handle.join().unwrap();
    }
}
