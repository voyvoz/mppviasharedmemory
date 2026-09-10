use shm_cxl::core::host::get_machine_id;
use shm_cxl::cxl::PersistantDataStructs;
use shm_cxl::core::Cacheline;
use shm_cxl::com::barrier::barrier_wait_all;
use shm_cxl::utils::{init_n_host_ids, open_test_cxl_dev};
use std::{env, thread, time};

fn main() {
    let mut force_reset = false;
    let mut channels = Vec::new();

    let mut it = env::args().skip(1);
    let mut num_hosts = 0;
    while let Some(a) = it.next() {
        if let Some(r) = a.strip_prefix("id=") {
            let id = r.to_string().parse::<u64>().unwrap();
            channels.push(id);
            println!("got id {id}");
        } else if let Some(r) = a.strip_prefix("num_hosts=") {
            num_hosts = r.to_string().parse::<u16>().unwrap();
            println!("using num_hosts={num_hosts:?}");
        } else if a == "reset" {
            force_reset = true;
        }
    }
    println!("Reset? {force_reset}");

    let (dev, size) = open_test_cxl_dev();
    init_n_host_ids(dev, num_hosts, Some(5000));
    let myid = get_machine_id();
    if myid > 1 {
        thread::sleep(time::Duration::from_millis(50)); // "force" 0 to be first
    }

    if myid == 0 {
        PersistantDataStructs::init(dev, size);
    } else {
        PersistantDataStructs::from_cl_sync(dev);
    }

    println!("cxl init done");
    barrier_wait_all();

    thread::sleep(time::Duration::from_millis(1000 + 5 * (myid as u64)));
    println!("done sleeping");
    barrier_wait_all();

    let dev_ptr = dev as u64;
    let mut threads = vec![];
    for c in channels {
        //spawn thread + ping pong
        threads.push(thread::spawn(move || {
            let base_cl = dev_ptr as *mut Cacheline;
            let persist = PersistantDataStructs::from_cl_sync(base_cl);
            // CASE A (static shm_cxl::Communication::get_or_create....
            //let mut myc = shm_cxl::Communication::get_or_create_pipe(c).unwrap()

            //Case B
            let mut myc = persist.named_pipes.get_or_create_pipe(c).unwrap();

            for n in 0..64 {
                myc.try_push(&Cacheline::from_b0(c + n));
                let r = myc.try_pop().unwrap();
                assert_eq!(c + n, r.d[0]);
            }
            for n in 0..64 {
                loop {
                    if myc.try_push(&Cacheline::from_b0(c + n)).is_ok() {
                        break;
                    }
                }
                let r = loop {
                    match myc.try_pop() {
                        Ok(cl) => break cl,
                        Err(_) => continue,
                    }
                };
                assert_eq!(c + n, r.d[0]);
            }
        }));
        //thread::sleep(time::Duration::from_millis(100));
    }

    for i in threads {
        i.join().unwrap();
    }
}
