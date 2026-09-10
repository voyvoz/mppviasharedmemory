use std::cell::Cell;
use std::hint;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Global storage
static TIMESTAMPS: Mutex<Vec<(String, Instant)>> = Mutex::new(Vec::new());

// Process-wide benchmark counters. Durations are accumulated across worker
// threads; the benchmark harness takes the maximum value across participating
// hosts, matching its critical-path treatment of kernel and wall time.
static CXL_LOAD_NS: AtomicU64 = AtomicU64::new(0);
static CXL_STORE_NS: AtomicU64 = AtomicU64::new(0);
static WAIT_SPINS: AtomicU64 = AtomicU64::new(0);
// Direct-CXL applications collect metrics for their whole process. Long-lived
// services such as the TUN bridge disable collection while idle and enable it
// only between explicit kernel-boundary timestamp points.
static BENCH_METRICS_ENABLED: AtomicBool = AtomicBool::new(true);

thread_local! {
    // Higher-level bulk operations may call instrumented cacheline primitives.
    // Only the outer timer contributes, preventing nested double counting.
    static LOAD_TIMER_DEPTH: Cell<u32> = const { Cell::new(0) };
    static STORE_TIMER_DEPTH: Cell<u32> = const { Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BenchMetrics {
    pub cxl_load_ns: u64,
    pub cxl_store_ns: u64,
    pub wait_spins: u64,
}

#[derive(Clone, Copy)]
enum MetricKind {
    Load,
    Store,
}

/// Scoped timer for a CXL load or store region.
///
/// Timers of the same kind may be nested. Only the outermost timer on each
/// thread is accumulated, which lets applications time a bulk phase while the
/// primitives inside it remain independently instrumented elsewhere.
#[must_use]
pub struct CxlMetricTimer {
    kind: MetricKind,
    start: Option<Instant>,
}

impl CxlMetricTimer {
    pub fn load() -> Self {
        Self::new(MetricKind::Load)
    }

    pub fn store() -> Self {
        Self::new(MetricKind::Store)
    }

    fn new(kind: MetricKind) -> Self {
        let outer = match kind {
            MetricKind::Load => LOAD_TIMER_DEPTH.with(|depth| {
                let current = depth.get();
                depth.set(current + 1);
                current == 0
            }),
            MetricKind::Store => STORE_TIMER_DEPTH.with(|depth| {
                let current = depth.get();
                depth.set(current + 1);
                current == 0
            }),
        };
        CxlMetricTimer {
            kind,
            start: (outer && bench_metrics_enabled()).then(Instant::now),
        }
    }
}

impl Drop for CxlMetricTimer {
    fn drop(&mut self) {
        let elapsed = self.start.map(|start| start.elapsed());
        match self.kind {
            MetricKind::Load => {
                LOAD_TIMER_DEPTH.with(|depth| depth.set(depth.get() - 1));
                if let Some(elapsed) = elapsed {
                    record_cxl_load(elapsed);
                }
            }
            MetricKind::Store => {
                STORE_TIMER_DEPTH.with(|depth| depth.set(depth.get() - 1));
                if let Some(elapsed) = elapsed {
                    record_cxl_store(elapsed);
                }
            }
        }
    }
}

/// Counts spins locally and publishes them once when dropped.
/// This keeps instrumentation out of the hottest polling instruction.
#[derive(Default)]
pub struct SpinCounter {
    spins: u64,
}

impl SpinCounter {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn wait(&mut self) {
        self.spins = self.spins.saturating_add(1);
        hint::spin_loop();
    }

    pub fn spins(&self) -> u64 {
        self.spins
    }
}

impl Drop for SpinCounter {
    fn drop(&mut self) {
        record_wait_spins(self.spins);
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

pub fn record_cxl_load(duration: Duration) {
    if bench_metrics_enabled() {
        CXL_LOAD_NS.fetch_add(duration_ns(duration), Ordering::Relaxed);
    }
}

pub fn record_cxl_store(duration: Duration) {
    if bench_metrics_enabled() {
        CXL_STORE_NS.fetch_add(duration_ns(duration), Ordering::Relaxed);
    }
}

pub fn record_wait_spins(spins: u64) {
    if bench_metrics_enabled() {
        WAIT_SPINS.fetch_add(spins, Ordering::Relaxed);
    }
}

pub fn set_bench_metrics_enabled(enabled: bool) {
    BENCH_METRICS_ENABLED.store(enabled, Ordering::SeqCst);
}

pub fn bench_metrics_enabled() -> bool {
    BENCH_METRICS_ENABLED.load(Ordering::Relaxed)
}

pub fn reset_bench_metrics() {
    CXL_LOAD_NS.store(0, Ordering::Relaxed);
    CXL_STORE_NS.store(0, Ordering::Relaxed);
    WAIT_SPINS.store(0, Ordering::Relaxed);
}

pub fn get_bench_metrics() -> BenchMetrics {
    BenchMetrics {
        cxl_load_ns: CXL_LOAD_NS.load(Ordering::Relaxed),
        cxl_store_ns: CXL_STORE_NS.load(Ordering::Relaxed),
        wait_spins: WAIT_SPINS.load(Ordering::Relaxed),
    }
}

/// Machine-readable output consumed by `scripts/paper_bench.sh`.
pub fn print_bench_metrics() {
    print_bench_metrics_value(get_bench_metrics());
}

pub fn print_bench_metrics_value(metrics: BenchMetrics) {
    println!("BENCH_CXL_LOAD_NS={}", metrics.cxl_load_ns);
    println!("BENCH_CXL_STORE_NS={}", metrics.cxl_store_ns);
    println!("BENCH_WAIT_SPINS={}", metrics.wait_spins);
}

/// Synchronous kernel-boundary control for a local IP-over-CXL TUN bridge.
///
/// When `BENCH_IP_CXL=1`, `start` resets and enables the bridge's timestamp
/// counters before returning. `finish` disables them and returns the exact
/// local-host snapshot, which the MPI rank prints through the same BENCH fields
/// as a direct-CXL algorithm. With the environment variable absent this is a
/// zero-cost inactive guard for ordinary MPI runs.
pub struct IpCxlMetricsGuard {
    active: bool,
}

impl IpCxlMetricsGuard {
    pub fn start() -> Self {
        let active = ip_cxl_metrics_requested();
        if active {
            let response = ip_cxl_control("START");
            assert_eq!(response, "OK", "TUN rejected metrics START");
        }
        Self { active }
    }

    pub fn finish(mut self) -> Option<BenchMetrics> {
        if !self.active {
            return None;
        }
        self.active = false;
        Some(parse_ip_cxl_metrics(&ip_cxl_control("STOP")))
    }
}

pub fn ip_cxl_metrics_requested() -> bool {
    std::env::var("BENCH_IP_CXL").as_deref() == Ok("1")
}

impl Drop for IpCxlMetricsGuard {
    fn drop(&mut self) {
        if self.active {
            let _ = try_ip_cxl_control("STOP");
        }
    }
}

fn ip_cxl_control(command: &str) -> String {
    try_ip_cxl_control(command)
        .unwrap_or_else(|error| panic!("IP-over-CXL metrics {command} failed: {error}"))
}

fn try_ip_cxl_control(command: &str) -> std::io::Result<String> {
    let socket =
        std::env::var("IP_CXL_METRICS_SOCKET").unwrap_or_else(|_| "tun.metrics.sock".to_string());
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    writeln!(stream, "{command}")?;
    let mut response = String::new();
    BufReader::new(stream).read_line(&mut response)?;
    Ok(response.trim().to_string())
}

fn parse_ip_cxl_metrics(response: &str) -> BenchMetrics {
    let mut fields = response.split_ascii_whitespace();
    let metrics = BenchMetrics {
        cxl_load_ns: fields
            .next()
            .expect("TUN STOP response missing load time")
            .parse()
            .expect("invalid TUN load time"),
        cxl_store_ns: fields
            .next()
            .expect("TUN STOP response missing store time")
            .parse()
            .expect("invalid TUN store time"),
        wait_spins: fields
            .next()
            .expect("TUN STOP response missing wait spins")
            .parse()
            .expect("invalid TUN wait spins"),
    };
    assert!(fields.next().is_none(), "unexpected TUN STOP fields");
    metrics
}

/// Creates a new timestamp with the given tag and appends it to the global vector.
pub fn create_timestamp(tag: &str) {
    let mut timestamps = TIMESTAMPS.lock().unwrap();
    timestamps.push((tag.to_string(), Instant::now()));
}

/// Returns a clone of the entire timestamp vector for custom processing.
pub fn get_timestamps() -> Vec<(String, Instant)> {
    let timestamps = TIMESTAMPS.lock().unwrap();
    timestamps.clone()
}

/// Prints all timestamps in order with their deltas to the previous timestamp.
pub fn print_timestamps() {
    let timestamps = TIMESTAMPS.lock().unwrap();

    if timestamps.is_empty() {
        println!("No timestamps recorded.");
        return;
    }

    println!("{}", "=".repeat(56));
    println!("{:<30} {:>12} {:>12}", "tag", "elapsed", "delta");
    println!("{}", "-".repeat(56));

    let mut prev = timestamps[0].1;
    for (i, (tag, ts)) in timestamps.iter().enumerate() {
        if i == 0 {
            println!(
                "{:<30} {:>12.7} {:>12.7}",
                tag,
                ts.duration_since(timestamps[0].1).as_secs_f64(),
                0.0
            );
        } else {
            let delta = ts.duration_since(prev).as_secs_f64();
            println!(
                "{:<30} {:>12.7} {:>12.7}",
                tag,
                ts.duration_since(timestamps[0].1).as_secs_f64(),
                delta
            );
        }
        prev = *ts;
    }
    println!("{}", "=".repeat(56));
}

/// Clears all recorded timestamps.
pub fn clear_timestamps() {
    let mut timestamps = TIMESTAMPS.lock().unwrap();
    timestamps.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn records_durations_and_spins() {
        let _guard = TEST_LOCK.lock().unwrap();
        set_bench_metrics_enabled(true);
        reset_bench_metrics();
        record_cxl_load(Duration::from_nanos(11));
        record_cxl_store(Duration::from_nanos(17));
        record_wait_spins(23);
        assert_eq!(
            get_bench_metrics(),
            BenchMetrics {
                cxl_load_ns: 11,
                cxl_store_ns: 17,
                wait_spins: 23,
            }
        );
    }

    #[test]
    fn spin_counter_publishes_on_drop() {
        let _guard = TEST_LOCK.lock().unwrap();
        set_bench_metrics_enabled(true);
        reset_bench_metrics();
        {
            let mut spins = SpinCounter::new();
            spins.wait();
            spins.wait();
            assert_eq!(spins.spins(), 2);
        }
        assert_eq!(get_bench_metrics().wait_spins, 2);
    }

    #[test]
    fn disabled_collection_ignores_all_metric_kinds() {
        let _guard = TEST_LOCK.lock().unwrap();
        reset_bench_metrics();
        set_bench_metrics_enabled(false);
        record_cxl_load(Duration::from_nanos(11));
        record_cxl_store(Duration::from_nanos(17));
        record_wait_spins(23);
        assert_eq!(get_bench_metrics(), BenchMetrics::default());
        set_bench_metrics_enabled(true);
    }
}
