use shm_cxl::{
    core::{
        Cacheline, cxl_read_cacheline_sync, cxl_write_cl_flush, cxl_write_cl_sync,
        host::{LEADER_ID},
        invalidate, prefetch_into_l0,
    },
    utils::{init_n_host_ids, open_test_cxl_dev},
};
use std::{
    time::{Duration, Instant},
};

/// Busy-wait for approximately `ns` nanoseconds. Portable (no intrinsics),
/// works on both your dev machine and the real CXL boxes.
#[inline]
fn spin_for_ns(ns: u64) {
    if ns == 0 {
        return;
    }
    let target = Duration::from_nanos(ns);
    let start = Instant::now();
    while start.elapsed() < target {
        std::hint::black_box(()); // prevent the loop from being optimized away
    }
}

fn the_other_side(base_cl: *mut Cacheline, rounds: u64) {
    let mut wait_loops = 0;
    let mut cl = Cacheline::empty();
    for i in 1..=rounds {
        cl.d[0] = i;
        cxl_write_cl_sync(base_cl, &cl);
        //println!("Wrote {i}");
        loop {
            let confirm = cxl_read_cacheline_sync(unsafe { base_cl.add(1) });
            if confirm.d[0] == i {
                break;
            }
            wait_loops += 1;
            //println!("didnt see {i}");
            //thread::sleep(Duration::from_secs(1));
        }
    }
    println!("wait_loops={wait_loops}");
}

fn bench(base_cl: *mut Cacheline, work_ns_1: u64, work_ns_2: u64, rounds: u64) -> (u64, u64) {
    let mut state = 0u8;
    let mut expected_next_value = 1;
    let mut full_state_iterations = 0;
    let mut missed_value_counter = 0;
    loop {
        //thread::sleep(Duration::from_secs(1));
        match state {
            0 => {
                invalidate(base_cl);
                // Do some work so that the invalidate is done
                // if the invalidate is not done the prefetch could just retrieve the old
                // value again maybe?
                spin_for_ns(work_ns_1);
            }
            1 => {
                prefetch_into_l0(base_cl);
                // Do some work to ensure the prefetch retrieved the data already
                spin_for_ns(work_ns_2);
            }
            _ => {
                let v = unsafe { *base_cl };
                if v.d[0] == expected_next_value {
                    expected_next_value += 1;
                    cxl_write_cl_flush(unsafe { base_cl.add(1) }, &v);
                } else {
                    missed_value_counter += 1;
                }
                full_state_iterations += 1;
                //println!("Read: {} {expected_next_value} {rounds}", v.d[0]);
                if expected_next_value > rounds {
                    break;
                }
            }
        }
        state = (state + 1) % 3;
    }
    (full_state_iterations, missed_value_counter)
}

fn main() {
    let rounds: u64 = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let work_min: u64 = std::env::var("WORK_NS_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let work_max: u64 = std::env::var("WORK_NS_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let work_step: u64 = std::env::var("WORK_NS_STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let (dev, _size) = open_test_cxl_dev();
    let myid = init_n_host_ids(dev, 2, Some(5000));
    if myid == LEADER_ID as u16 {
        println!("work_ns_1,work_ns_2,ns_per_round,full_state_iterations,missed_value_counter");
        // w2 is time between prefetch and read
        let mut w2 = work_min;
        while w2 <= work_max {
            // w1 is time between invalidate and prefetch
            let mut w = work_min;
            while w <= work_max {
                // warm up branch predictor / caches before timing each point
                let _ = bench(dev, w, w2, rounds);

                let bench_start = std::time::Instant::now();
                let a = bench(dev, w, w2, rounds);
                println!(
                    "{w},{w2},{:?},{},{}",
                    bench_start.elapsed().as_nanos() / rounds as u128,
                    a.0,
                    a.1
                );

                w += work_step;
            }
            w2 += work_step;
        }
    } else {
        loop {
            the_other_side(dev, rounds);
        }
    }
}
