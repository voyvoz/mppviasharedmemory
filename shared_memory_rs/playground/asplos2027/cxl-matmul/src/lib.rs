pub mod ipc_dist;
mod matmul_core;
pub mod mpi_dist;

use self::matmul_core::{RowPartition, matmul_block_tiled};
pub use self::matmul_core::{checksum, idx, make_matrix, matmul, matmul_threads};
pub use ipc_dist::{run_ipc_dist, run_ipc_dist_timed};
pub use mpi_dist::{run_mpi_dist, run_mpi_dist_timed};

use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};
use shm_cxl::core::{Cacheline, range_flush_sync, range_invalidate_sync};
use shm_cxl::core::{cxl_read_cacheline_sync, cxl_write_cl_sync};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use shm_cxl::utils::timestamps::{
    CxlMetricTimer, SpinCounter, get_bench_metrics, print_bench_metrics_value,
    reset_bench_metrics,
};
use std::alloc::Layout;
use std::time::{Duration, Instant};

const MATMUL_MAGIC: u64 = 0x4d41_544d_554c_3031; // "MATMUL01"

fn alloc_cacheline<A: Allocator>(alloc: &A, cls: usize) -> *mut Cacheline {
    let mem = alloc
        .allocate(Layout::from_size_align(cls.max(1) * 64, 64).unwrap())
        .unwrap();
    mem.as_ptr() as *mut Cacheline
}

#[derive(Copy, Clone)]
struct DistMatrix {
    base_cl: *mut Cacheline,
    rows: usize,
    cols: usize,
    row_cls: usize,
    stride: usize,
    total_cls: usize,
}

unsafe impl Send for DistMatrix {}
unsafe impl Sync for DistMatrix {}

impl DistMatrix {
    /// Rows are padded to an odd number of cachelines so consecutive rows
    /// shift by 64 B modulo every power-of-two cache-set period. Without the
    /// pad, a power-of-two row width (n=2048 -> 16 KiB stride) makes the
    /// column-wise reads of B in the untiled `dist` kernel alias onto a
    /// handful of cache sets, which cost that variant ~7x at n=2048 versus
    /// the *larger* n=2304. `attach` derives the same layout from
    /// (rows, cols), so creator and attacher always agree.
    fn row_cls_for(cols: usize) -> usize {
        let row_cls = cols.div_ceil(8).max(1);
        row_cls | 1
    }

    fn create<A: Allocator>(alloc: &A, rows: usize, cols: usize) -> Self {
        let row_cls = Self::row_cls_for(cols);
        let stride = row_cls * 8;
        let total_cls = rows.max(1) * row_cls;
        let base_cl = alloc_cacheline(alloc, total_cls);

        DistMatrix {
            base_cl,
            rows,
            cols,
            row_cls,
            stride,
            total_cls,
        }
    }

    fn attach(offset: u64, rows: usize, cols: usize) -> Self {
        let row_cls = Self::row_cls_for(cols);
        let stride = row_cls * 8;
        DistMatrix {
            base_cl: Cacheline::ptr_from_offset(offset),
            rows,
            cols,
            row_cls,
            stride,
            total_cls: rows.max(1) * row_cls,
        }
    }

    fn offset(&self) -> u64 {
        Cacheline::ptr_to_offset(self.base_cl)
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.stride + col
    }

    fn as_f64_ptr(&self) -> *mut f64 {
        self.base_cl as *mut f64
    }

    fn load_local(&self, row: usize, col: usize) -> f64 {
        unsafe { *self.as_f64_ptr().add(self.idx(row, col)) }
    }

    fn store_local(&self, row: usize, col: usize, value: f64) {
        unsafe {
            *self.as_f64_ptr().add(self.idx(row, col)) = value;
        }
    }

    fn zero_fill_flush(&self) {
        let _store_timer = CxlMetricTimer::store();
        unsafe {
            std::ptr::write_bytes(self.base_cl as *mut u8, 0, self.total_cls * 64);
        }
        self.flush_all();
    }

    fn fill_from_dense(&self, dense: &[f64]) {
        let _store_timer = CxlMetricTimer::store();
        debug_assert_eq!(dense.len(), self.rows * self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.store_local(i, j, dense[i * self.cols + j]);
            }
        }
    }

    fn row_cl(&self, row: usize) -> *mut Cacheline {
        unsafe { self.base_cl.add(row * self.row_cls) }
    }

    fn flush_all(&self) {
        range_flush_sync(self.base_cl, self.total_cls);
    }

    fn flush_rows(&self, row_begin: usize, row_end: usize) {
        if row_begin >= row_end || self.rows == 0 {
            return;
        }
        range_flush_sync(self.row_cl(row_begin), (row_end - row_begin) * self.row_cls);
    }

    fn invalidate_rows(&self, row_begin: usize, row_end: usize) {
        if row_begin >= row_end || self.rows == 0 {
            return;
        }
        range_invalidate_sync(self.row_cl(row_begin), (row_end - row_begin) * self.row_cls);
    }

    fn snapshot_dense(&self) -> Vec<f64> {
        let _load_timer = CxlMetricTimer::load();
        self.invalidate_rows(0, self.rows);
        let mut dense = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                dense[i * self.cols + j] = self.load_local(i, j);
            }
        }
        dense
    }
}

fn make_descriptor(a: &DistMatrix, b: &DistMatrix, c: &DistMatrix, nhosts: u16) -> Cacheline {
    Cacheline {
        d: [
            MATMUL_MAGIC,
            a.offset(),
            b.offset(),
            c.offset(),
            nhosts as u64,
            0,
            0,
            0,
        ],
    }
}

fn matrices_from_descriptor(
    desc: &Cacheline,
    m: usize,
    k: usize,
    n: usize,
) -> (DistMatrix, DistMatrix, DistMatrix) {
    assert_eq!(desc.d[0], MATMUL_MAGIC, "invalid matmul descriptor");

    let a = DistMatrix::attach(desc.d[1], m, k);
    let b = DistMatrix::attach(desc.d[2], k, n);
    let c = DistMatrix::attach(desc.d[3], m, n);
    (a, b, c)
}

fn init_or_attach_dist_state(
    m: usize,
    k: usize,
    n: usize,
    expected_hosts: u16,
) -> (PersistantDataStructs, DistMatrix, DistMatrix, DistMatrix) {
    let mut persist = cxl_init(expected_hosts);
    let myid = get_machine_id();

    let (a, b, c) = if myid == 1 {
        // Create three shared memory segments.
        let a = DistMatrix::create(persist.alloc.as_ref(), m, k);
        let b = DistMatrix::create(persist.alloc.as_ref(), k, n);
        let c = DistMatrix::create(persist.alloc.as_ref(), m, n);

        // Populate A and B using `make_matrix`, zero out C, and flush everything out.
        a.fill_from_dense(&make_matrix(m, k));
        a.flush_all();
        b.fill_from_dense(&make_matrix(k, n));
        b.flush_all();
        c.zero_fill_flush();

        // Pack the offsets of the three matrices into a descriptor
        // and write it to shared memory.
        let desc_ptr = alloc_cacheline(persist.alloc.as_ref(), 1);
        let desc = make_descriptor(&a, &b, &c, expected_hosts);
        cxl_write_cl_sync(desc_ptr, &desc);

        // Broadcast descriptor to all other hosts.
        let msg = Cacheline::from_b0(Cacheline::ptr_to_offset(desc_ptr));
        let mut wait_spins = SpinCounter::new();
        for peer in 2..=expected_hosts {
            while persist
                .ipc
                .try_send_to(peer as PeerMachineID, &msg)
                .is_err()
            {
                wait_spins.wait();
            }
        }

        (a, b, c)
    } else {
        // Waiting host1
        let (_, msg) = persist
            .ipc
            .try_pop_from(1 as PeerMachineID, Some(Duration::from_secs(10)))
            .unwrap();
        let desc = cxl_read_cacheline_sync(Cacheline::ptr_from_offset(msg.d[0]));
        matrices_from_descriptor(&desc, m, k, n)
    };

    barrier_wait_all();
    (persist, a, b, c)
}

pub struct DistRunResult {
    pub init_elapsed: Duration,
    pub compute_elapsed: Duration,
    pub collect_elapsed: Duration,
    pub result: Option<Vec<f64>>,
}

pub fn run_dist(m: usize, k: usize, n: usize, expected_hosts: u16) -> DistRunResult {
    let init_start = Instant::now();
    let (_persist, a, b, c) = init_or_attach_dist_state(m, k, n, expected_hosts);
    let init_elapsed = init_start.elapsed();

    let myid = get_machine_id();
    let nhosts = get_machine_cnt();
    let part = RowPartition::new(m, myid.saturating_sub(1) as usize, nhosts.max(1) as usize);

    reset_bench_metrics();
    let compute_start = Instant::now();

    // Must invalidate before reading: invalidate only the specific part of A to be read,
    //whereas all of B must be invalidated.
    a.invalidate_rows(part.row_begin, part.row_end);
    b.invalidate_rows(0, k);

    {
        let _load_timer = CxlMetricTimer::load();
        for i in part.row_begin..part.row_end {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a.load_local(i, p) * b.load_local(p, j);
                }
                c.store_local(i, j, sum);
            }
        }
    }

    // Push out the part of C that was written.
    c.flush_rows(part.row_begin, part.row_end);
    barrier_wait_all();

    let compute_elapsed = compute_start.elapsed();
    print_bench_metrics_value(get_bench_metrics());

    // Only host1 collects the results.
    let collect_start = Instant::now();
    let result = if myid == 1 {
        Some(c.snapshot_dense())
    } else {
        None
    };
    let collect_elapsed = collect_start.elapsed();

    DistRunResult {
        init_elapsed,
        compute_elapsed,
        collect_elapsed,
        result,
    }
}

/// Same as [`run_dist`], but stages this host's slice of A and all of B out
/// of CXL memory into plain local (DRAM) buffers in one sequential pass
/// each, then computes with the cache-blocked [`matmul_block_tiled`] kernel
/// entirely against those local copies before writing the result back to
/// CXL once. `run_dist` instead calls `load_local`/`store_local` straight
/// against CXL inside the (i, j, p) loop, which re-reads the whole of B
/// from CXL once per row of C -- that's the difference this mode isolates.
pub fn run_dist_opt(m: usize, k: usize, n: usize, expected_hosts: u16) -> DistRunResult {
    let init_start = Instant::now();
    let (_persist, a, b, c) = init_or_attach_dist_state(m, k, n, expected_hosts);
    let init_elapsed = init_start.elapsed();

    let myid = get_machine_id();
    let nhosts = get_machine_cnt();
    let part = RowPartition::new(m, myid.saturating_sub(1) as usize, nhosts.max(1) as usize);

    reset_bench_metrics();
    let compute_start = Instant::now();

    // Must invalidate before reading: invalidate only the specific part of A to be read,
    //whereas all of B must be invalidated.
    a.invalidate_rows(part.row_begin, part.row_end);
    b.invalidate_rows(0, k);

    // Stage this host's slice of A and all of B out of CXL memory into plain
    // local (DRAM) buffers in one sequential pass each, instead of calling
    // `load_local` straight out of CXL inside the (i, j, p) loop below --
    // that re-reads the whole of B from CXL once per row of C. Every
    // element of A/B is read from CXL exactly once here, sequentially.
    let mut a_local = vec![0.0; part.local_rows * k];
    let mut b_local = vec![0.0; k * n];
    {
        let _load_timer = CxlMetricTimer::load();
        for (i, row) in (part.row_begin..part.row_end).enumerate() {
            for p in 0..k {
                a_local[i * k + p] = a.load_local(row, p);
            }
        }
        for p in 0..k {
            for j in 0..n {
                b_local[p * n + j] = b.load_local(p, j);
            }
        }
    }

    // Compute entirely against the local copies with a cache-blocked kernel --
    // no more CXL accesses from here until the result is written back below.
    let mut c_local = vec![0.0; part.local_rows * n];
    matmul_block_tiled(&a_local, &b_local, &mut c_local, k, n);

    // Push the computed rows back into CXL in one sequential pass, then flush.
    {
        let _store_timer = CxlMetricTimer::store();
        for (i, row) in (part.row_begin..part.row_end).enumerate() {
            for j in 0..n {
                c.store_local(row, j, c_local[i * n + j]);
            }
        }
    }
    c.flush_rows(part.row_begin, part.row_end);
    barrier_wait_all();

    let compute_elapsed = compute_start.elapsed();
    print_bench_metrics_value(get_bench_metrics());

    // Only host1 collects the results.
    let collect_start = Instant::now();
    let result = if myid == 1 {
        Some(c.snapshot_dense())
    } else {
        None
    };
    let collect_elapsed = collect_start.elapsed();

    DistRunResult {
        init_elapsed,
        compute_elapsed,
        collect_elapsed,
        result,
    }
}

pub fn run_serial(m: usize, k: usize, n: usize) -> Vec<f64> {
    run_serial_timed(m, k, n).0
}

pub fn run_threads(m: usize, k: usize, n: usize, num_threads: usize) -> Vec<f64> {
    run_threads_timed(m, k, n, num_threads).0
}

/// Same as [`run_serial`], but the returned duration covers only the multiply --
/// operand construction is excluded so it is comparable with the distributed
/// engine's `compute_elapsed`.
pub fn run_serial_timed(m: usize, k: usize, n: usize) -> (Vec<f64>, Duration) {
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);
    let t = Instant::now();
    let c = matmul(&a, &b, m, k, n);
    (c, t.elapsed())
}

/// See [`run_serial_timed`]; the duration covers only the multiply.
pub fn run_threads_timed(m: usize, k: usize, n: usize, num_threads: usize) -> (Vec<f64>, Duration) {
    let a = make_matrix(m, k);
    let b = make_matrix(k, n);
    let t = Instant::now();
    let c = matmul_threads(&a, &b, m, k, n, num_threads);
    (c, t.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_result() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3

        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let c = matmul(&a, &b, 2, 3, 2);
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn tiled_block_matches_reference() {
        // Include sizes below, at, and straddling the 64-wide tile boundary
        // used by matmul_block_tiled, so remainder-tile handling is covered.
        let cases = [
            (1, 1, 1),
            (5, 3, 4),
            (16, 8, 16),
            (7, 6, 3),
            (64, 64, 64),
            (65, 65, 65),
            (130, 70, 200),
        ];
        for (rows, k, n) in cases {
            let a = make_matrix(rows, k);
            let b = make_matrix(k, n);
            let expected = matmul(&a, &b, rows, k, n);

            let mut actual = vec![0.0; rows * n];
            matmul_block_tiled(&a, &b, &mut actual, k, n);

            assert_eq!(actual, expected, "mismatch rows={rows} k={k} n={n}");
        }
    }

    #[test]
    fn threads_matches_reference() {
        let cases = [(1, 1, 1), (5, 3, 4), (16, 8, 16), (7, 6, 3)];
        let thread_counts = [1, 2, 3, 8];
        for (m, k, n) in cases {
            let a = make_matrix(m, k);
            let b = make_matrix(k, n);
            let expected = matmul(&a, &b, m, k, n);
            for &threads in &thread_counts {
                let actual = matmul_threads(&a, &b, m, k, n, threads);
                assert_eq!(
                    actual, expected,
                    "mismatch m={m} k={k} n={n} threads={threads}"
                );
            }
        }
    }
}
