//! Plan B: 2 physical machines, bootstrap coordination
//!
//! Machine 0 (init, hostname dbcxla → machine_id=1): creates the ring, drives phases, collects samples
//! Machine 1 (follower, hostname dbcxlb → machine_id=2): waits for commands, responds passively
//!
//! Coordination: CXL shared memory bootstrap region (offset 1024 cachelines).
//!   Signal values are globally monotonically increasing to prevent reuse of stale values.

use crate::harness::*;
use shm_cxl::cxl_write;
use shm_cxl::l0_api::cxl_primitives::{Cacheline, cxl_read_cacheline};
use shm_cxl::l0_api::cxl_ring_buffer::RING_CAPACITY;
use shm_cxl::l0_api::host;
use shm_cxl::l1_api::ipc::IpcEndpoint;
use std::time::{Duration, Instant};

// ─── message helpers ───
fn cl_tag(tag: u64) -> Cacheline { Cacheline::from_b0(tag) }
fn cl_poison() -> Cacheline { Cacheline::from_b0(POISON) }
fn is_poison(cl: &Cacheline) -> bool { cl.d[0] == POISON }
fn tag_of(cl: &Cacheline) -> u64 { cl.d[0] }

// ═══════════════════════════════════════════════════════════════
// Bootstrap coordination (CXL shared mem, offset 1024 cachelines)
// ═══════════════════════════════════════════════════════════════

const BOOT_OFFSET: isize = 1024;

fn boot(base: *mut Cacheline, s: isize) -> *mut Cacheline {
    unsafe { base.offset(BOOT_OFFSET + s) }
}
fn boot_read(base: *mut Cacheline, s: isize) -> u64 {
    cxl_read_cacheline(boot(base, s)).d[0]
}
fn boot_write(base: *mut Cacheline, s: isize, v: u64) {
    cxl_write!((boot(base, s), v));
}

mod slot {
    pub const INIT_READY: isize = 0;
    pub const FOLLOWER_READY: isize = 1;
    pub const PHASE: isize = 2;
    pub const PARAM: isize = 3;
    pub const START: isize = 4;
    pub const FILL_DONE: isize = 5;
}

const BARRIER_TO: Duration = Duration::from_secs(120);

fn wait_signal(base: *mut Cacheline, s: isize, want: u64, label: &str) {
    let t0 = Instant::now();
    loop {
        let v = boot_read(base, s);
        if v == want {
            println!("  [sync] {label} ✓ ({:.0}ms)", t0.elapsed().as_secs_f64() * 1000.);
            return;
        }
        if t0.elapsed() > BARRIER_TO {
            panic!("[sync] {label} timed out! expected {want:x}, actual {v:x}");
        }
        std::hint::spin_loop();
    }
}

fn clear_bootstrap(base: *mut Cacheline) {
    for s in 0..16 { boot_write(base, s, 0); }
}

/// Handshake: init writes parameters → wait for follower → signal start. Signal values increase monotonically.
fn handshake(base: *mut Cacheline, sig: &mut u64, phase: u64, param: u64) -> u64 {
    *sig += 1;
    let rv = *sig;
    boot_write(base, slot::PHASE, phase);
    boot_write(base, slot::PARAM, param);
    boot_write(base, slot::INIT_READY, rv);
    wait_signal(base, slot::FOLLOWER_READY, rv,
        &format!("follower ready (sig={rv}, phase={phase})"));
    boot_write(base, slot::START, rv);
    rv
}

// ═══════════════════════════════════════════════════════════════
// Phase 0: Idle Pingpong RTT
// ═══════════════════════════════════════════════════════════════

fn phase0(ep: &mut IpcEndpoint, base: *mut Cacheline, sig: &mut u64, freq: f64, count: u64) {
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Plan B / Phase 0: Idle Pingpong ({count} iters)          ║");
    println!("╚══════════════════════════════════════════════╝");

    handshake(base, sig, 0, count);

    for _ in 0..1000 {
        ep.try_send_to(2, &cl_tag(0)).unwrap();
        ep.try_pop_from(2, None).unwrap();
    }
    let mut samples = Vec::with_capacity(count as usize);
    let wall = Instant::now();

    for i in 0..count {
        let t0 = rdtsc();
        ep.try_send_to(2, &cl_tag(i)).unwrap();
        let (_, echo) = ep.try_pop_from(2, None).unwrap();
        assert_eq!(tag_of(&echo), i, "echo tag mismatch");
        samples.push(rdtsc() - t0);
        if i > 0 && i % 20000 == 0 {
            println!("  progress: {i}/{count}  ({:.0} msg/s)",
                i as f64 / wall.elapsed().as_secs_f64());
        }
    }
    ep.try_send_to(2, &cl_poison()).unwrap();
    ep.try_pop_from(2, None).unwrap();

    let elapsed = wall.elapsed().as_secs_f64();
    let st = compute_stats(&samples, freq);
    print_stats(&st, &format!("Plan B Pingpong RTT ({count} iters)"));
    print_histogram(&samples, freq);
    println!("  total time: {:.2}s  one-way estimate (RTT/2): {:.1} ns\n", elapsed, st.mean_ns / 2.0);
}

// ═══════════════════════════════════════════════════════════════
// Phase 1: Latency time series
// ═══════════════════════════════════════════════════════════════

fn phase1(ep: &mut IpcEndpoint, base: *mut Cacheline, sig: &mut u64, freq: f64, secs: f64) {
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Plan B / Phase 1: Latency time series ({secs} s)         ║");
    println!("╚══════════════════════════════════════════════╝");

    handshake(base, sig, 1, (secs * 1e9) as u64);

    let tsc_base = rdtsc();
    let deadline = tsc_base + (freq * secs) as u64;
    let mut csv_rows = Vec::new();
    let mut samples = Vec::new();
    let mut seq: u64 = 0;

    while rdtsc() < deadline {
        let t0 = rdtsc();
        ep.try_send_to(2, &cl_tag(seq)).unwrap();
        let (_, echo) = ep.try_pop_from(2, None).unwrap();
        assert_eq!(tag_of(&echo), seq, "echo tag mismatch");
        let rtt = rdtsc() - t0;
        csv_rows.push((t0.wrapping_sub(tsc_base), rtt, seq));
        samples.push(rtt);
        seq += 1;
    }
    ep.try_send_to(2, &cl_poison()).unwrap();
    ep.try_pop_from(2, None).unwrap();

    let mut csv = CsvWriter::create("timeseries_planB.csv",
        "timestamp_ns,latency_rtt_ns,seq");
    for (ts, rtt, s) in &csv_rows {
        csv.row(&format!("{},{},{}",
            tsc_to_ns(*ts, freq) as u64, tsc_to_ns(*rtt, freq) as u64, s));
    }

    let st = compute_stats(&samples, freq);
    print_stats(&st, &format!("Plan B {secs}-second time series ({seq} iters)"));
    print_histogram(&samples, freq);
    println!("  data written: timeseries_planB.csv");
    let peaks = detect_peaks(&samples, freq, st.p999_ns, 3.0);
    report_peaks("Plan B time series", &peaks, st.p999_ns, 3.0);
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Phase 2: Burst injection latency (RTT version, includes queuing)
// ═══════════════════════════════════════════════════════════════

fn phase2(ep: &mut IpcEndpoint, base: *mut Cacheline, sig: &mut u64, freq: f64) {
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Plan B / Phase 2: Burst injection latency (RTT)     ║");
    println!("╚══════════════════════════════════════════════╝");

    let burst_sizes = [1u64, 2, 4, 8, 16, 32, 64, 128];
    let rounds = 200;

    let mut csv = CsvWriter::create("burst_planB.csv",
        "burst_size,round,position,injection_rtt_ns,send_spins");

    for &burst in &burst_sizes {
        let mut all_rtt = Vec::new();
        println!("\n── burst_size={burst} ({rounds} rounds) ──");

        for round in 0..rounds {
            handshake(base, sig, 2, burst);

            while ep.try_pop_any().is_ok_and(|r| r.is_some()) {}

            let mut send_tsc = vec![0u64; burst as usize];
            let mut send_spins = vec![0u64; burst as usize];
            let mut recv_tsc = vec![0u64; burst as usize];
            let mut recv_done: u64 = 0;

            for pos in 0..burst {
                let t0 = rdtsc();
                let mut spins = 0u64;
                loop {
                    match ep.try_send_to(2, &cl_tag(pos)) {
                        Ok(()) => break,
                        Err(_) => {
                            spins += 1;
                            while let Ok(Some((_, echo))) = ep.try_pop_any() {
                                let tag = tag_of(&echo);
                                if tag < burst { recv_tsc[tag as usize] = rdtsc(); recv_done += 1; }
                            }
                            std::hint::spin_loop();
                        }
                    }
                }
                send_tsc[pos as usize] = t0;
                send_spins[pos as usize] = spins;
                while let Ok(Some((_, echo))) = ep.try_pop_any() {
                    let tag = tag_of(&echo);
                    if tag < burst { recv_tsc[tag as usize] = rdtsc(); recv_done += 1; }
                }
            }
            while recv_done < burst {
                let (_, echo) = ep.try_pop_from(2, Some(Duration::from_secs(10))).unwrap();
                let tag = tag_of(&echo);
                if tag < burst { recv_tsc[tag as usize] = rdtsc(); recv_done += 1; }
            }
            ep.try_send_to(2, &cl_poison()).unwrap();
            ep.try_pop_from(2, None).unwrap();

            for pos in 0..burst {
                let rtt = recv_tsc[pos as usize].wrapping_sub(send_tsc[pos as usize]);
                csv.row(&format!("{},{},{},{:.1},{}",
                    burst, round, pos, tsc_to_ns(rtt, freq), send_spins[pos as usize]));
                all_rtt.push(rtt);
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        let st = compute_stats(&all_rtt, freq);
        print_stats(&st, &format!("Plan B Burst RTT size={burst} (with queuing)"));
        println!("    ring capacity={RING_CAPACITY}, burst exceeds capacity={}", burst > RING_CAPACITY as u64);
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Phase 3: Fixed queue depth RTT (fill → probe → release consumer)
// ═══════════════════════════════════════════════════════════════

fn phase3(ep: &mut IpcEndpoint, base: *mut Cacheline, sig: &mut u64, freq: f64, probe_count: u64) {
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Plan B / Phase 3: Fixed queue depth RTT (backprsr)  ║");
    println!("╚══════════════════════════════════════════════╝");

    let fill_levels = [1u64, 2, 3]; // RING_CAPACITY=4, max fill=3

    for &fill in &fill_levels {
        let mut probes = Vec::with_capacity(probe_count as usize);
        println!("\n── ring pre-fill {fill}/{RING_CAPACITY}, {probe_count} probes ──");

        for _ in 0..probe_count {
            boot_write(base, slot::FILL_DONE, 0); // clear before follower enters wait
            handshake(base, sig, 3, fill);

            while ep.try_pop_any().is_ok_and(|r| r.is_some()) {}

            // 1. Fill `fill` items
            for tag in 0..fill {
                loop {
                    if ep.try_send_to(2, &cl_tag(tag)).is_ok() { break; }
                    std::hint::spin_loop();
                }
            }
            // 2. Insert probe (queued after the fill items)
            let t0 = rdtsc();
            loop {
                if ep.try_send_to(2, &cl_tag(0xDEAD)).is_ok() { break; }
                std::hint::spin_loop();
            }
            // 3. Release consumer (probe already enqueued)
            boot_write(base, slot::FILL_DONE, 1);
            // 4. Wait for echo
            let (_, echo) = ep.try_pop_from(2, None).unwrap();
            assert_eq!(tag_of(&echo), 0xBEEF, "probe echo mismatch");
            probes.push(rdtsc() - t0);
        }
        let st = compute_stats(&probes, freq);
        print_stats(&st, &format!("Plan B fixed queue depth RTT (ring has {fill} items, {probe_count} probes)"));
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════
// Plan B — init main entry
// ═══════════════════════════════════════════════════════════════

pub fn run_init(freq: f64, base: *mut Cacheline) {
    println!("════════ Plan B: 2 machines — init side (host 1) ════════\n");
    host::set_machine_id(1);

    clear_bootstrap(base);
    let mut ep = IpcEndpoint::init(base, 2);
    let mut sig: u64 = 0;
    println!("[setup] IpcEndpoint::init(n=2) done");
    println!("[setup] (please start the follower on machine 1: cargo run --release -p bench -- --plan=b --role=follower)\n");

    phase0(&mut ep, base, &mut sig, freq, 50_000);
    phase1(&mut ep, base, &mut sig, freq, 5.0);
    phase2(&mut ep, base, &mut sig, freq);
    phase3(&mut ep, base, &mut sig, freq, 3000);

    // Exit signal
    sig += 1;
    boot_write(base, slot::PHASE, 255);
    boot_write(base, slot::INIT_READY, sig);
    println!("════════ Plan B init done ════════");
}

// ═══════════════════════════════════════════════════════════════
// Plan B — follower side
// ═══════════════════════════════════════════════════════════════

fn follower_echo(ep: &mut IpcEndpoint) {
    loop {
        let (_, cl) = ep.try_pop_from(1, None).unwrap();
        if is_poison(&cl) {
            ep.try_send_to(1, &cl_poison()).unwrap();
            break;
        }
        ep.try_send_to(1, &cl).unwrap();
    }
}

fn follower_loaded(ep: &mut IpcEndpoint, base: *mut Cacheline) {
    let fill = boot_read(base, slot::PARAM);
    wait_signal(base, slot::FILL_DONE, 1, "fill done");
    for _ in 0..fill {
        let (_, cl) = ep.try_pop_from(1, None).unwrap();
        assert_ne!(cl.d[0], 0xDEAD, "probe before fill drained — FILL_DONE protocol broken");
    }
    let (_, _) = ep.try_pop_from(1, None).unwrap(); // probe
    ep.try_send_to(1, &cl_tag(0xBEEF)).unwrap();
}

pub fn run_follower(_freq: f64, base: *mut Cacheline) {
    println!("════════ Plan B: 2 machines — follower side (host 2) ════════\n");

    host::set_machine_id(2);

    // Wait for any non-zero INIT_READY (do not hardcode the value 1)
    let first_sig: u64 = loop {
        let v = boot_read(base, slot::INIT_READY);
        if v != 0 { break v; }
        std::hint::spin_loop();
    };
    let phase = boot_read(base, slot::PHASE);
    if phase != 0 {
        panic!("[follower] at startup INIT_READY={first_sig} phase={phase}, not a cold start (phase should be 0)");
    }
    let mut ep = IpcEndpoint::from_cl(base);
    println!("[setup] IpcEndpoint::from_cl done\n");

    // Immediately handle the current phase (the phase = 0 corresponding to first_sig)
    {
        let rv = first_sig;
        boot_write(base, slot::FOLLOWER_READY, rv);
        wait_signal(base, slot::START, rv, "start phase 0");
        follower_echo(&mut ep);
        println!("  [follower] Phase 0 done (sig={rv})");
    }
    let mut last_sig: u64 = first_sig;

    loop {
        // Only wait for new values greater than last_sig
        let rv = {
            loop {
                let v = boot_read(base, slot::INIT_READY);
                if v > last_sig { break v; }
                std::hint::spin_loop();
            }
        };
        last_sig = rv;
        let phase = boot_read(base, slot::PHASE);
        if phase == 255 {
            println!("[follower] received exit signal");
            return;
        }
        boot_write(base, slot::FOLLOWER_READY, rv);
        wait_signal(base, slot::START, rv, &format!("start phase {phase}"));

        match phase {
            0 => follower_echo(&mut ep),
            1 => follower_echo(&mut ep),
            2 => follower_echo(&mut ep),
            3 => follower_loaded(&mut ep, base),
            _ => panic!("[follower] unknown phase {phase}"),
        }
        println!("  [follower] Phase {phase} done (sig={rv})");
    }
}
