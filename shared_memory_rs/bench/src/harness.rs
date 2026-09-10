//! Benchmark harness — shared infrastructure
//!
//! Independent of the IPC API. Provides: TSC clock, latency stats, histogram, CSV, spike detection, StartGate.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════
// TSC
// ═══════════════════════════════════════════════════════════════


#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn rdtsc() -> u64 {
    let lo: u32; let hi: u32;
    unsafe { std::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack)); }
    ((hi as u64) << 32) | (lo as u64)
}
#[cfg(not(target_arch = "x86_64"))]
pub fn rdtsc()-> u64{
    0
}

pub fn calibrate_tsc() -> f64 {
    std::thread::sleep(Duration::from_millis(10));
    let a = rdtsc();
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_millis(100));
    let b = rdtsc();
    let f = (b - a) as f64 / t0.elapsed().as_nanos() as f64 * 1e9;
    println!("[TSC] frequency = {:.3} GHz", f / 1e9);
    f
}

#[inline(always)]
pub fn tsc_to_ns(c: u64, f: f64) -> f64 {
    c as f64 / f * 1e9
}

// ═══════════════════════════════════════════════════════════════
// Statistics
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub count: usize,
    pub min_ns: f64,
    pub max_ns: f64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub p50_ns: f64,
    pub p99_ns: f64,
    pub p999_ns: f64,
    pub p9999_ns: f64,
}

pub fn compute_stats(samples: &[u64], freq_hz: f64) -> LatencyStats {
    let n = samples.len();
    if n == 0 {
        return LatencyStats {
            count: 0, min_ns: 0., max_ns: 0., mean_ns: 0., stddev_ns: 0.,
            p50_ns: 0., p99_ns: 0., p999_ns: 0., p9999_ns: 0.,
        };
    }
    let mut s = samples.to_vec();
    s.sort_unstable();
    let sum: u64 = s.iter().sum();
    let mean = sum as f64 / n as f64;
    let var = s.iter().map(|&x| { let d = x as f64 - mean; d * d }).sum::<f64>() / n as f64;
    LatencyStats {
        count: n,
        min_ns: tsc_to_ns(s[0], freq_hz),
        max_ns: tsc_to_ns(s[n - 1], freq_hz),
        mean_ns: tsc_to_ns(mean as u64, freq_hz),
        stddev_ns: tsc_to_ns(var.sqrt() as u64, freq_hz),
        p50_ns: tsc_to_ns(s[n * 50 / 100], freq_hz),
        p99_ns: tsc_to_ns(s[n * 99 / 100], freq_hz),
        p999_ns: tsc_to_ns(s[n * 999 / 1000], freq_hz),
        p9999_ns: tsc_to_ns(s[n * 9999 / 10000], freq_hz),
    }
}

pub fn print_stats(st: &LatencyStats, title: &str) {
    println!("\n═══ {title} ═══");
    println!("  count    : {}", st.count);
    println!("  mean     : {:>10.1} ns", st.mean_ns);
    println!("  stddev   : {:>10.1} ns", st.stddev_ns);
    println!("  min      : {:>10.1} ns", st.min_ns);
    println!("  max      : {:>10.1} ns", st.max_ns);
    println!("  p50      : {:>10.1} ns", st.p50_ns);
    println!("  p99      : {:>10.1} ns", st.p99_ns);
    println!("  p99.9    : {:>10.1} ns", st.p999_ns);
    println!("  p99.99   : {:>10.1} ns", st.p9999_ns);
}

pub fn print_histogram(samples: &[u64], freq_hz: f64) {
    let bounds: [u64; 11] =
        [50, 100, 200, 300, 500, 750, 1000, 2000, 5000, 10_000, 50_000];
    let mut buckets = vec![0usize; bounds.len() + 1];
    for &s in samples {
        let ns = tsc_to_ns(s, freq_hz) as u64;
        let i = bounds.iter().position(|&b| ns <= b).unwrap_or(bounds.len());
        buckets[i] += 1;
    }
    let total = samples.len() as f64;
    println!("\n═══ Latency Histogram ═══");
    for (i, &b) in bounds.iter().enumerate() {
        let pct = buckets[i] as f64 / total * 100.;
        let bar = "█".repeat((pct * 2.) as usize);
        println!("  ≤{:>5}ns : {} {:.1}%", b, bar, pct);
    }
    if *buckets.last().unwrap() > 0 {
        println!("  >50μs   : {} ({:.3}%)",
            buckets.last().unwrap(), *buckets.last().unwrap() as f64 / total * 100.);
    }
}

// ═══════════════════════════════════════════════════════════════
// Spike detection
// ═══════════════════════════════════════════════════════════════

pub fn detect_peaks(samples: &[u64], freq_hz: f64, p999_ns: f64, mult: f64) -> Vec<(u64, f64)> {
    let threshold = p999_ns * mult;
    samples.iter().enumerate()
        .filter_map(|(i, &s)| {
            let ns = tsc_to_ns(s, freq_hz);
            (ns > threshold).then_some((i as u64, ns))
        })
        .collect()
}

pub fn report_peaks(title: &str, peaks: &[(u64, f64)], p999_ns: f64, mult: f64) {
    let threshold = p999_ns * mult;
    match peaks.len() {
        0 => println!("  ✓ {title}: no spikes (> {threshold:.0}ns)"),
        1 => println!("  ⚠ {title}: 1 isolated spike (> {threshold:.0}ns): seq={} {:.1}ns",
                 peaks[0].0, peaks[0].1),
        _ => {
            println!("  ⚠ {title}: {} spikes (> {threshold:.0}ns):", peaks.len());
            let mut gaps: Vec<u64> = peaks.windows(2).map(|w| w[1].0 - w[0].0).collect();
            if !gaps.is_empty() {
                gaps.sort_unstable();
                println!("    gaps (messages): p50={} p99={}",
                    gaps[gaps.len() / 2], gaps[gaps.len() * 99 / 100]);
            }
            for (s, ns) in peaks.iter().take(10) {
                println!("      seq={s:>6}  {ns:.1} ns");
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// CSV
// ═══════════════════════════════════════════════════════════════

pub struct CsvWriter { file: std::fs::File }

impl CsvWriter {
    pub fn create(path: &str, header: &str) -> Self {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "{header}").unwrap();
        CsvWriter { file: f }
    }
    pub fn row(&mut self, line: &str) { writeln!(self.file, "{line}").unwrap(); }
}

// ═══════════════════════════════════════════════════════════════
// StartGate — a real barrier
// ═══════════════════════════════════════════════════════════════
//
// Usage:
//   let gate = Arc::new(StartGate::new(N));
//   // spawn N threads, each receives gate.clone()
//   // inside thread: gate.ready();  // mark ready, wait for release
//   // main thread:   gate.wait_all_ready(N); gate.go();  // start all together after everyone is ready

pub struct StartGate {
    ready: AtomicUsize,
    go: AtomicBool,
}

impl StartGate {
    pub fn new(total: usize) -> Self {
        let _ = total;
        StartGate { ready: AtomicUsize::new(0), go: AtomicBool::new(false) }
    }

    /// Called by each worker thread: notify the main thread that we are ready, then wait for the release signal
    pub fn ready(&self) {
        self.ready.fetch_add(1, Ordering::AcqRel);
        while !self.go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }

    /// Called by the main thread: wait until all N threads are ready
    pub fn wait_all_ready(&self, total: usize) {
        while self.ready.load(Ordering::Acquire) < total {
            std::hint::spin_loop();
        }
    }

    /// Called by the main thread: release all threads
    pub fn go(&self) {
        self.go.store(true, Ordering::Release);
    }
}

// ═══════════════════════════════════════════════════════════════
// CPU affinity (pin to core)
// ═══════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
pub fn pin_to_core(core: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}
#[cfg(not(target_arch = "x86_64"))]
pub fn pin_to_core(_core: usize) {}

/// Poison pill: Cacheline.d[0] == POISON marks the end of the current round
pub const POISON: u64 = u64::MAX;
