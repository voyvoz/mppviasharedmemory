//! Plan A: single physical machine, 2 concurrent threads, endpoint kept alive
//!
//! Two worker threads each call IpcEndpoint::from_cl() once after set_machine_id(),
//! then keep the endpoint alive and reuse it across phases. The main thread sends
//! commands / receives results via mpsc channels.
//! This avoids the issue where rebuilding via from_cl() resets the local counters
//! and corrupts shared ring state.

use crate::harness::*;
use shm_cxl::l0_api::cxl_primitives::Cacheline;
use shm_cxl::l0_api::cxl_ring_buffer::RING_CAPACITY;
use shm_cxl::l0_api::host;
use shm_cxl::l1_api::ipc::IpcEndpoint;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

// ─── Cacheline constructors ───
fn cl_tag(tag: u64) -> Cacheline { Cacheline::from_b0(tag) }
fn cl_poison() -> Cacheline { Cacheline::from_b0(POISON) }
fn is_poison(cl: &Cacheline) -> bool { cl.d[0] == POISON }
fn tag_of(cl: &Cacheline) -> u64 { cl.d[0] }

// ─── Commands from main thread → worker ───
enum Cmd {
    /// Run N pingpongs (including warmup), return RTT samples
    Phase0 { count: u64, warmup: u64 },
    /// Run timed pingpong, return (samples, seq, base_tsc)
    Phase1 { deadline_cycles: u64 },
    /// Run burst_size × rounds rounds, return (send_tsc, send_spins)
    Phase2Send { burst: u64, rounds: u64 },
    /// Consume burst rounds, return recv_tsc
    Phase2Recv { rounds: u64 },
    /// Fill `fill` items then insert a probe, return RTT
    Phase3Probe { fill: u64 },
    Exit,
}

// ─── Results from worker → main thread ───
enum Res {
    Pingpong(Vec<u64>),
    Timeseries(Vec<u64>, u64),
    BurstSend(Vec<(u64, u64)>, Vec<u64>), // (tag, send_tsc), spins
    BurstRecv(Vec<(u64, u64)>),            // (tag, recv_tsc)
    ProbeRtt(u64),
    Ack,
}

// ═══════════════════════════════════════════════════════════════
// Worker 0: host=1, responsible for sending + timing
// ═══════════════════════════════════════════════════════════════

fn worker0(base_addr: usize, rx: mpsc::Receiver<Cmd>, tx: mpsc::Sender<Res>, fd: Arc<std::sync::atomic::AtomicBool>, freq: f64) {
    let base = base_addr as *mut Cacheline;
    host::set_machine_id(1);
    let mut ep = IpcEndpoint::from_cl(base);
    pin_to_core(0);

    loop {
        match rx.recv().unwrap() {
            Cmd::Phase0 { count, warmup } => {
                for _ in 0..warmup {
                    ep.try_send_to(2, &cl_tag(0)).unwrap();
                    ep.try_pop_from(2, None).unwrap();
                }
                let mut samples = Vec::with_capacity(count as usize);
                for i in 0..count {
                    let t0 = rdtsc();
                    ep.try_send_to(2, &cl_tag(i)).unwrap();
                    ep.try_pop_from(2, None).unwrap();
                    samples.push(rdtsc() - t0);
                }
                ep.try_send_to(2, &cl_poison()).unwrap();
                ep.try_pop_from(2, None).unwrap();
                tx.send(Res::Pingpong(samples)).unwrap();
            }

            Cmd::Phase1 { deadline_cycles } => {
                let tsc_base = rdtsc();
                // Buffer in memory first, then flush to CSV at the end to avoid I/O polluting the measurement
                let mut csv_rows = Vec::new();
                let mut samples = Vec::new();
                let mut seq: u64 = 0;
                while rdtsc().wrapping_sub(tsc_base) < deadline_cycles {
                    let t0 = rdtsc();
                    ep.try_send_to(2, &cl_tag(seq)).unwrap();
                    ep.try_pop_from(2, None).unwrap();
                    let rtt = rdtsc() - t0;
                    csv_rows.push((t0.wrapping_sub(tsc_base), rtt, seq));
                    samples.push(rtt);
                    seq += 1;
                }
                ep.try_send_to(2, &cl_poison()).unwrap();
                ep.try_pop_from(2, None).unwrap();
                // Batch write CSV
                let mut csv = CsvWriter::create("timeseries_planA.csv",
                    "timestamp_ns,latency_rtt_ns,seq");
                for (ts, rtt, s) in &csv_rows {
                    csv.row(&format!("{},{},{}",
                        tsc_to_ns(*ts, freq) as u64,
                        tsc_to_ns(*rtt, freq) as u64, s));
                }
                tx.send(Res::Timeseries(samples, seq)).unwrap();
            }

            Cmd::Phase2Send { burst, rounds } => {
                let mut all_send = Vec::new();
                let mut all_spins = Vec::new();
                for _round in 0..rounds {
                    std::thread::sleep(Duration::from_micros(100));
                    for pos in 0..burst {
                        let t0 = rdtsc();
                        let mut spins = 0u64;
                        loop {
                            match ep.try_send_to(2, &cl_tag(pos)) {
                                Ok(()) => break,
                                Err(_) => { spins += 1; std::hint::spin_loop(); }
                            }
                        }
                        all_send.push((pos, t0));
                        all_spins.push(spins);
                    }
                    loop {
                        if ep.try_send_to(2, &cl_poison()).is_ok() { break; }
                        std::hint::spin_loop();
                    }
                    ep.try_pop_from(2, None).unwrap();
                }
                tx.send(Res::BurstSend(all_send, all_spins)).unwrap();
            }

            Cmd::Phase3Probe { fill } => {
                for tag in 0..fill {
                    loop {
                        if ep.try_send_to(2, &cl_tag(tag)).is_ok() { break; }
                        std::hint::spin_loop();
                    }
                }
                let t0 = rdtsc();
                loop {
                    if ep.try_send_to(2, &cl_tag(0xDEAD)).is_ok() { break; }
                    std::hint::spin_loop();
                }
                fd.store(true, Ordering::Release);
                ep.try_pop_from(2, None).unwrap();
                tx.send(Res::ProbeRtt(rdtsc() - t0)).unwrap();
            }

            Cmd::Exit => break,
            _ => {} // worker0 ignores other cmds
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Worker 1: host=2, responsible for receiving + echoing back
// ═══════════════════════════════════════════════════════════════

fn worker1(base_addr: usize, rx: mpsc::Receiver<Cmd>, tx: mpsc::Sender<Res>, fd: Arc<std::sync::atomic::AtomicBool>, _freq: f64) {
    let base = base_addr as *mut Cacheline;
    host::set_machine_id(2);
    let mut ep = IpcEndpoint::from_cl(base);
    pin_to_core(1);

    loop {
        match rx.recv().unwrap() {
            Cmd::Phase0 { warmup, .. } => {
                for _ in 0..warmup {
                    ep.try_pop_from(1, None).unwrap();
                    ep.try_send_to(1, &cl_tag(0)).unwrap();
                }
                loop {
                    let (_, cl) = ep.try_pop_from(1, None).unwrap();
                    if is_poison(&cl) {
                        ep.try_send_to(1, &cl_poison()).unwrap();
                        break;
                    }
                    ep.try_send_to(1, &cl).unwrap();
                }
                tx.send(Res::Ack).unwrap();
            }

            Cmd::Phase1 { .. } => {
                loop {
                    let (_, cl) = ep.try_pop_from(1, None).unwrap();
                    if is_poison(&cl) {
                        ep.try_send_to(1, &cl_poison()).unwrap();
                        break;
                    }
                    ep.try_send_to(1, &cl).unwrap();
                }
                tx.send(Res::Ack).unwrap();
            }

            Cmd::Phase2Recv { rounds } => {
                let mut all_recv = Vec::new();
                for _round in 0..rounds {
                    loop {
                        let (_, cl) = ep.try_pop_from(1, None).unwrap();
                        let t1 = rdtsc();
                        if is_poison(&cl) {
                            ep.try_send_to(1, &cl_poison()).unwrap();
                            break;
                        }
                        all_recv.push((tag_of(&cl), t1));
                    }
                }
                tx.send(Res::BurstRecv(all_recv)).unwrap();
            }

            Cmd::Phase3Probe { .. } => {
                // Wait for worker0 to finish filling and insert the probe
                while !fd.load(Ordering::Acquire) { std::hint::spin_loop(); }
                // Drain: non-0xDEAD items are fill items (discard), 0xDEAD is the probe (echo back)
                loop {
                    let (_, cl) = ep.try_pop_from(1, None).unwrap();
                    if tag_of(&cl) == 0xDEAD {
                        ep.try_send_to(1, &cl_tag(0xBEEF)).unwrap();
                        break;
                    }
                }
                tx.send(Res::Ack).unwrap();
            }

            Cmd::Exit => break,
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Main thread: dispatch commands, collect results, print stats
// ═══════════════════════════════════════════════════════════════

pub fn run(freq: f64, base: *mut Cacheline) {
    println!("════════ Plan A: 1 machine, 2 threads, persistent endpoint ════════");
    println!("[setup] ring capacity = {RING_CAPACITY}");

    IpcEndpoint::init(base, 2);
    println!("[setup] IpcEndpoint::init(n=2) done\n");

    let base_addr = base as usize;

    let fill_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fd0 = fill_done.clone();
    let fd1 = fill_done.clone();

    // Create two command channels
    let (cmd0_tx, cmd0_rx) = mpsc::channel();
    let (cmd1_tx, cmd1_rx) = mpsc::channel();
    let (res0_tx, res0_rx) = mpsc::channel();
    let (res1_tx, res1_rx) = mpsc::channel();

    let h0 = std::thread::spawn(move || worker0(base_addr, cmd0_rx, res0_tx, fd0, freq));
    let h1 = std::thread::spawn(move || worker1(base_addr, cmd1_rx, res1_tx, fd1, freq));

    // ── Phase 0 ──
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Phase 0: Idle Pingpong 2 threads (50000 iters)      ║");
    println!("╚══════════════════════════════════════════════╝");
    cmd0_tx.send(Cmd::Phase0 { count: 50_000, warmup: 1000 }).unwrap();
    cmd1_tx.send(Cmd::Phase0 { count: 50_000, warmup: 1000 }).unwrap();
    let samples = match res0_rx.recv().unwrap() { Res::Pingpong(s) => s, _ => panic!() };
    let _ = res1_rx.recv().unwrap();
    let st = compute_stats(&samples, freq);
    print_stats(&st, "Idle Pingpong RTT (50000 iters)");
    print_histogram(&samples, freq);
    println!("  one-way estimate (RTT/2): {:.1} ns\n", st.mean_ns / 2.0);

    // ── Phase 1 ──
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Phase 1: Latency time series (5 seconds)            ║");
    println!("╚══════════════════════════════════════════════╝");
    let deadline = (freq * 5.0) as u64;
    cmd0_tx.send(Cmd::Phase1 { deadline_cycles: deadline }).unwrap();
    cmd1_tx.send(Cmd::Phase1 { deadline_cycles: deadline }).unwrap();
    let (samples, seq) = match res0_rx.recv().unwrap() { Res::Timeseries(s, q) => (s, q), _ => panic!() };
    let _ = res1_rx.recv().unwrap();
    let st = compute_stats(&samples, freq);
    print_stats(&st, &format!("5-second time series ({seq} round-trips)"));
    print_histogram(&samples, freq);
    println!("  data written: timeseries_planA.csv\n");
    let peaks = detect_peaks(&samples, freq, st.p999_ns, 3.0);
    report_peaks("Time series", &peaks, st.p999_ns, 3.0);

    // ── Phase 2 ──
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Phase 2: Burst injection latency (with queuing)     ║");
    println!("╚══════════════════════════════════════════════╝");
    let burst_sizes = [1u64, 2, 4, 8, 16, 32, 64, 128];
    let rounds = 500;
    let mut csv = CsvWriter::create("burst_planA.csv",
        "burst_size,round,position,injection_ns,send_spins");

    for &burst in &burst_sizes {
        let mut all_samples = Vec::new();
        println!("\n── burst_size={burst} ({rounds} rounds) ──");

        for round in 0..rounds {
            // Dispatch commands to both sides
            cmd0_tx.send(Cmd::Phase2Send { burst, rounds: 1 }).unwrap();
            cmd1_tx.send(Cmd::Phase2Recv { rounds: 1 }).unwrap();

            let (send_data, send_spins) = match res0_rx.recv().unwrap() {
                Res::BurstSend(v, s) => (v, s), _ => panic!()
            };
            let recv_data = match res1_rx.recv().unwrap() {
                Res::BurstRecv(v) => v, _ => panic!()
            };

            let mut send_map = vec![None; burst as usize];
            for (tag, tsc) in &send_data { send_map[*tag as usize] = Some(*tsc); }
            let mut recv_map = vec![None; burst as usize];
            for (tag, tsc) in &recv_data { recv_map[*tag as usize] = Some(*tsc); }

            for pos in 0..burst {
                if let (Some(send), Some(recv)) = (send_map[pos as usize], recv_map[pos as usize]) {
                    let delta = recv.wrapping_sub(send);
                    let ns = tsc_to_ns(delta, freq);
                    let spins = send_spins.get(pos as usize).copied().unwrap_or(0);
                    csv.row(&format!("{},{},{},{:.1},{}", burst, round, pos, ns, spins));
                    all_samples.push(delta);
                }
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        let st = compute_stats(&all_samples, freq);
        print_stats(&st, &format!("Burst size={burst} injection latency (with queuing)"));
        println!("    ring capacity={RING_CAPACITY}, burst exceeds capacity={}", burst > RING_CAPACITY as u64);
    }

    // ── Phase 3 ──
    println!("\n╔══════════════════════════════════════════════╗");
    println!("║  Phase 3: Backpressure latency (Ring backpressure)   ║");
    println!("╚══════════════════════════════════════════════╝");
    let fill_levels = [1u64, 2, 3, RING_CAPACITY as u64 - 1];
    let probe_count = 5000;

    for &fill in &fill_levels {
        let mut probes = Vec::with_capacity(probe_count as usize);
        println!("\n── ring pre-fill {fill}/{RING_CAPACITY}, {probe_count} probes ──");

        for _ in 0..probe_count {
            fill_done.store(false, Ordering::Release);
            cmd1_tx.send(Cmd::Phase3Probe { fill }).unwrap();
            cmd0_tx.send(Cmd::Phase3Probe { fill }).unwrap();

            let rtt = match res0_rx.recv().unwrap() {
                Res::ProbeRtt(v) => v, _ => panic!()
            };
            let _ = res1_rx.recv().unwrap();
            probes.push(rtt);
            std::thread::sleep(Duration::from_micros(50));
        }
        let st = compute_stats(&probes, freq);
        print_stats(&st, &format!("Backpressure latency (ring has {fill} items, {probe_count} probe RTTs)"));
    }

    // ── Cleanup ──
    cmd0_tx.send(Cmd::Exit).unwrap();
    cmd1_tx.send(Cmd::Exit).unwrap();
    h0.join().unwrap();
    h1.join().unwrap();

    println!("\n════════ Plan A done ════════");
}
