pub mod dist_exchange;
pub mod mpi_exchange;
pub mod stencil_core;

use std::alloc::Layout;
use std::time::{Duration, Instant};

use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::com::mpi::barrier::barrier_wait_all;
use shm_cxl::core::host::{PeerMachineID, get_machine_cnt, get_machine_id};
use shm_cxl::core::{
    Cacheline, cxl_read_cacheline_sync, cxl_write_cl_sync, range_flush_sync, range_invalidate_sync,
};
use shm_cxl::cxl::{PersistantDataStructs, cxl_init};
use shm_cxl::utils::timestamps::{
    CxlMetricTimer, SpinCounter, get_bench_metrics, print_bench_metrics_value, reset_bench_metrics,
};

const DIST_MAGIC: u64 = 0x5354_454e_4349_4c31;

pub fn idx(i: usize, j: usize, cols: usize) -> usize {
    i * cols + j
}

fn init_hotspot(grid: &mut [f64], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    grid[idx(rows / 2, cols / 2, cols)] = 100.0;
}

pub fn init_grid(rows: usize, cols: usize) -> Vec<f64> {
    let mut grid = vec![0.0; rows * cols];
    init_hotspot(&mut grid, rows, cols);
    grid
}

pub fn step_jacobi(old: &[f64], new: &mut [f64], rows: usize, cols: usize) {
    new.copy_from_slice(old);

    for i in 1..rows.saturating_sub(1) {
        for j in 1..cols.saturating_sub(1) {
            let c = idx(i, j, cols);
            let up = idx(i - 1, j, cols);
            let down = idx(i + 1, j, cols);
            let left = idx(i, j - 1, cols);
            let right = idx(i, j + 1, cols);

            new[c] = 0.2 * (old[c] + old[up] + old[down] + old[left] + old[right]);
        }
    }
}

pub fn step_jacobi_block(
    old: &[f64],
    new_block: &mut [f64],
    cols: usize,
    row_begin: usize,
    row_end: usize,
) {
    for i in row_begin..row_end {
        let local_i = i - row_begin;

        for j in 1..cols.saturating_sub(1) {
            let c = idx(i, j, cols);
            let up = idx(i - 1, j, cols);
            let down = idx(i + 1, j, cols);
            let left = idx(i, j - 1, cols);
            let right = idx(i, j + 1, cols);

            let local_c = local_i * cols + j;

            new_block[local_c] = 0.2 * (old[c] + old[up] + old[down] + old[left] + old[right]);
        }
    }
}

pub fn run_jacobi(rows: usize, cols: usize, steps: usize) -> Vec<f64> {
    let mut old = init_grid(rows, cols);
    let mut new = old.clone();

    for _ in 0..steps {
        step_jacobi(&old, &mut new, rows, cols);
        std::mem::swap(&mut old, &mut new);
    }

    old
}

pub fn run_jacobi_parallel(rows: usize, cols: usize, steps: usize, num_threads: usize) -> Vec<f64> {
    let mut old = init_grid(rows, cols);
    let mut new = old.clone();

    if rows < 3 || cols < 3 {
        return old;
    }

    let interior_rows = rows - 2;
    let workers = num_threads.max(1).min(interior_rows);
    let block_rows = interior_rows.div_ceil(workers);

    for _ in 0..steps {
        new.copy_from_slice(&old); // ?????

        let old_ref: &[f64] = &old;
        let interior = &mut new[cols..(rows - 1) * cols];

        std::thread::scope(|s| {
            for (t, new_block) in interior.chunks_mut(block_rows * cols).enumerate() {
                let row_begin = 1 + t * block_rows;
                let rows_this = new_block.len() / cols;
                let row_end = row_begin + rows_this;

                s.spawn(move || {
                    step_jacobi_block(old_ref, new_block, cols, row_begin, row_end);
                });
            }
        });

        std::mem::swap(&mut old, &mut new);
    }

    old
}

pub fn run_serial(rows: usize, cols: usize, steps: usize) -> Vec<f64> {
    run_jacobi(rows, cols, steps)
}

pub fn run_threads(rows: usize, cols: usize, steps: usize, num_threads: usize) -> Vec<f64> {
    run_jacobi_parallel(rows, cols, steps, num_threads)
}

/// Same as [`run_serial`], but the duration covers only the sweep loop -- grid
/// allocation and initialisation are excluded so it is comparable with the
/// distributed engine, which times only its stepping loop.
pub fn run_serial_timed(rows: usize, cols: usize, steps: usize) -> (Vec<f64>, Duration) {
    let mut old = init_grid(rows, cols);
    let mut new = old.clone();

    let t = Instant::now();
    for _ in 0..steps {
        step_jacobi(&old, &mut new, rows, cols);
        std::mem::swap(&mut old, &mut new);
    }
    (old, t.elapsed())
}

/// See [`run_serial_timed`]; the duration covers only the sweep loop.
pub fn run_threads_timed(
    rows: usize,
    cols: usize,
    steps: usize,
    num_threads: usize,
) -> (Vec<f64>, Duration) {
    let mut old = init_grid(rows, cols);
    let mut new = old.clone();

    if rows < 3 || cols < 3 {
        return (old, Duration::ZERO);
    }

    let interior_rows = rows - 2;
    let workers = num_threads.max(1).min(interior_rows);
    let block_rows = interior_rows.div_ceil(workers);

    let t = Instant::now();
    for _ in 0..steps {
        new.copy_from_slice(&old);

        let old_ref: &[f64] = &old;
        let interior = &mut new[cols..(rows - 1) * cols];

        std::thread::scope(|s| {
            for (t, new_block) in interior.chunks_mut(block_rows * cols).enumerate() {
                let row_begin = 1 + t * block_rows;
                let rows_this = new_block.len() / cols;
                let row_end = row_begin + rows_this;

                s.spawn(move || {
                    step_jacobi_block(old_ref, new_block, cols, row_begin, row_end);
                });
            }
        });

        std::mem::swap(&mut old, &mut new);
    }
    (old, t.elapsed())
}

fn alloc_cachelines<A: Allocator>(alloc: &A, cls: usize) -> *mut Cacheline {
    let mem = alloc
        .allocate(Layout::from_size_align(cls.max(1) * 64, 64).unwrap())
        .unwrap();
    mem.as_ptr() as *mut Cacheline
}

#[derive(Copy, Clone)]
struct DistGrid {
    base_cl: *mut Cacheline,
    rows: usize,
    cols: usize,
    row_cls: usize,
    stride: usize,
    total_cls: usize,
}

unsafe impl Send for DistGrid {}
unsafe impl Sync for DistGrid {}

impl DistGrid {
    fn create<A: Allocator>(alloc: &A, rows: usize, cols: usize) -> Self {
        let row_cls = cols.div_ceil(8).max(1);
        let stride = row_cls * 8;
        let total_cls = rows.max(1) * row_cls;
        let base_cl = alloc_cachelines(alloc, total_cls);

        DistGrid {
            base_cl,
            rows,
            cols,
            row_cls,
            stride,
            total_cls,
        }
    }

    fn attach(offset: u64, rows: usize, cols: usize, row_cls: usize, stride: usize) -> Self {
        DistGrid {
            base_cl: Cacheline::ptr_from_offset(offset),
            rows,
            cols,
            row_cls,
            stride,
            total_cls: rows.max(1) * row_cls.max(1),
        }
    }

    fn offset(&self) -> u64 {
        Cacheline::ptr_to_offset(self.base_cl)
    }

    #[inline(always)]
    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.stride + col
    }

    #[inline(always)]
    fn as_f64_ptr(&self) -> *mut f64 {
        self.base_cl as *mut f64
    }

    #[inline]
    fn row_cl(&self, row: usize) -> *mut Cacheline {
        unsafe { self.base_cl.add(row * self.row_cls) }
    }

    #[inline]
    fn load_local(&self, row: usize, col: usize) -> f64 {
        unsafe { *self.as_f64_ptr().add(self.idx(row, col)) }
    }

    #[inline]
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

    #[inline]
    fn invalidate_row(&self, row: usize) {
        if row >= self.rows || self.rows == 0 {
            return;
        }
        range_invalidate_sync(self.row_cl(row), self.row_cls);
    }

    #[inline]
    fn flush_row(&self, row: usize) {
        if row >= self.rows || self.rows == 0 {
            return;
        }
        range_flush_sync(self.row_cl(row), self.row_cls);
    }

    fn snapshot_dense(&self) -> Vec<f64> {
        let _load_timer = CxlMetricTimer::load();
        self.invalidate_rows(0, self.rows);

        let mut dense = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                dense[idx(i, j, self.cols)] = self.load_local(i, j);
            }
        }
        dense
    }
}

fn init_dist_grids(old: &DistGrid, new: &DistGrid) {
    old.zero_fill_flush();
    new.zero_fill_flush();

    if old.rows == 0 || old.cols == 0 {
        return;
    }

    let center_row = old.rows / 2;
    let center_col = old.cols / 2;
    old.store_local(center_row, center_col, 100.0);
    new.store_local(center_row, center_col, 100.0);
    old.flush_rows(center_row, center_row + 1);
    new.flush_rows(center_row, center_row + 1);
}

fn make_descriptor(old: &DistGrid, new: &DistGrid, nhosts: u16) -> Cacheline {
    Cacheline {
        d: [
            DIST_MAGIC,
            old.rows as u64,
            old.cols as u64,
            old.row_cls as u64,
            old.stride as u64,
            old.offset(),
            new.offset(),
            nhosts as u64,
        ],
    }
}

fn grids_from_descriptor(desc: &Cacheline) -> (DistGrid, DistGrid) {
    assert_eq!(desc.d[0], DIST_MAGIC, "invalid stencil descriptor");

    let rows = desc.d[1] as usize;
    let cols = desc.d[2] as usize;
    let row_cls = desc.d[3] as usize;
    let stride = desc.d[4] as usize;
    let old = DistGrid::attach(desc.d[5], rows, cols, row_cls, stride);
    let new = DistGrid::attach(desc.d[6], rows, cols, row_cls, stride);

    (old, new)
}

fn partition_rows(rows: usize, machine_id: u16, nhosts: u16) -> (usize, usize) {
    if rows < 2 {
        return (0, 0);
    }

    let interior_rows = rows.saturating_sub(2);
    let part = (machine_id.saturating_sub(1)) as usize;
    let workers = nhosts.max(1) as usize;

    let row_begin = 1 + part * interior_rows / workers;
    let row_end = 1 + (part + 1) * interior_rows / workers;

    (row_begin.min(rows - 1), row_end.min(rows - 1))
}

fn update_block_dist(old: &DistGrid, new: &DistGrid, row_begin: usize, row_end: usize) {
    for i in row_begin..row_end {
        for j in 1..old.cols.saturating_sub(1) {
            let value = 0.2
                * (old.load_local(i, j)
                    + old.load_local(i - 1, j)
                    + old.load_local(i + 1, j)
                    + old.load_local(i, j - 1)
                    + old.load_local(i, j + 1));
            new.store_local(i, j, value);
        }
    }
}

fn init_or_attach_dist_state(
    rows: usize,
    cols: usize,
    expected_hosts: u16,
) -> (PersistantDataStructs, DistGrid, DistGrid) {
    let mut persist = cxl_init(expected_hosts);
    let myid = get_machine_id();

    let (old, new) = if myid == 1 {
        let old = DistGrid::create(persist.alloc.as_ref(), rows, cols);
        let new = DistGrid::create(persist.alloc.as_ref(), rows, cols);
        init_dist_grids(&old, &new);

        let desc_ptr = alloc_cachelines(persist.alloc.as_ref(), 1);
        let desc = make_descriptor(&old, &new, expected_hosts);
        cxl_write_cl_sync(desc_ptr, &desc);

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

        (old, new)
    } else {
        let (_, msg) = persist
            .ipc
            .try_pop_from(1 as PeerMachineID, Some(Duration::from_secs(10)))
            .unwrap();
        let desc = cxl_read_cacheline_sync(Cacheline::ptr_from_offset(msg.d[0]));
        grids_from_descriptor(&desc)
    };

    barrier_wait_all();
    (persist, old, new)
}

pub fn run_dist(rows: usize, cols: usize, steps: usize, expected_hosts: u16) -> Option<Vec<f64>> {
    run_dist_timed(rows, cols, steps, expected_hosts).0
}

/// See [`run_dist`]; the duration covers only the stepping loop (including its
/// per-step barriers, which are part of the algorithm). CXL attach and the final
/// snapshot are excluded. `init_or_attach_dist_state` already ends with a
/// barrier, so every host starts the clock from the same point -- do not add
/// another one here, an extra barrier desynchronises the generation counters and
/// corrupts the result.
pub fn run_dist_timed(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_hosts: u16,
) -> (Option<Vec<f64>>, Duration) {
    let (_persist, mut old, mut new) = init_or_attach_dist_state(rows, cols, expected_hosts);
    let myid = get_machine_id();
    let nhosts = get_machine_cnt();

    reset_bench_metrics();
    let t = Instant::now();

    if rows >= 3 && cols >= 3 {
        let (row_begin, row_end) = partition_rows(rows, myid, nhosts);

        for _ in 0..steps {
            let read_begin = row_begin.saturating_sub(1);
            let read_end = (row_end + 1).min(rows);

            old.invalidate_rows(read_begin, read_end);
            update_block_dist(&old, &new, row_begin, row_end);
            new.flush_rows(row_begin, row_end);
            barrier_wait_all();
            std::mem::swap(&mut old, &mut new);
        }
    }

    let kernel = t.elapsed();
    let kernel_metrics = get_bench_metrics();
    print_bench_metrics_value(kernel_metrics);

    if myid == 1 {
        (Some(old.snapshot_dense()), kernel)
    } else {
        (None, kernel)
    }
}

pub fn run_dist_timed_opt(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_hosts: u16,
) -> (Option<Vec<f64>>, Duration) {
    let (_persist, mut old, mut new) = init_or_attach_dist_state(rows, cols, expected_hosts);
    let myid = get_machine_id();
    let nhosts = get_machine_cnt();

    reset_bench_metrics();
    let t = Instant::now();

    if rows >= 3 && cols >= 3 {
        println!("rows {}", rows);
        let (row_begin, row_end) = partition_rows(rows, myid, nhosts);

        let read_begin = row_begin.saturating_sub(1);
        let read_end = (row_end + 1).min(rows);

        // A CXL window is deliberately reused by later benchmark cases.  Host
        // 1 initializes both buffers, but the other hosts can still have cache
        // lines from the previous user of this window.  Seed their local view
        // once before relying on the optimized per-iteration halo traffic.
        // Both buffers matter: after the first swap, the initially-new buffer
        // becomes the input and its fixed boundary columns are read as well.
        old.invalidate_rows(read_begin, read_end);
        new.invalidate_rows(read_begin, read_end);

        for _ in 0..steps {
            if read_begin != 0 {
                old.invalidate_rows(read_begin, read_begin + 1);
            }
            if read_end != rows {
                old.invalidate_rows(row_end, row_end + 1);
            }
            update_block_dist(&old, &new, row_begin, row_end);
            if read_begin != 0 {
                new.flush_rows(row_begin, row_begin + 1);
            }
            if read_end != rows {
                new.flush_rows(row_end - 1, row_end);
            }

            barrier_wait_all();
            std::mem::swap(&mut old, &mut new);
        }
        old.flush_rows(row_begin, row_end);
        // CXL 2.0 is not coherent: rank 1 must not snapshot the final grid
        // until every host has published its owned rows.
        barrier_wait_all();
    }

    let kernel = t.elapsed();
    let kernel_metrics = get_bench_metrics();
    print_bench_metrics_value(kernel_metrics);

    if myid == 1 {
        (Some(old.snapshot_dense()), kernel)
    } else {
        (None, kernel)
    }
}

pub fn run_dist_timed_opt_opt(
    rows: usize,
    cols: usize,
    steps: usize,
    expected_hosts: u16,
) -> (Option<Vec<f64>>, Duration) {
    let (_persist, mut old, mut new) = init_or_attach_dist_state(rows, cols, expected_hosts);
    let myid = get_machine_id();
    let nhosts = get_machine_cnt();

    reset_bench_metrics();
    let t = Instant::now();

    if rows >= 3 && cols >= 3 {
        println!("rows {}", rows);
        let (row_begin, row_end) = partition_rows(rows, myid, nhosts);

        let read_begin = row_begin.saturating_sub(1);
        let read_end = (row_end + 1).min(rows);

        // A CXL window is deliberately reused by later benchmark cases.  Host
        // 1 initializes both buffers, but the other hosts can still have cache
        // lines from the previous user of this window.  Seed their local view
        // once before relying on the optimized per-iteration halo traffic.
        // New will be overwritten so only the untuched borders need invalidation
        if myid != 1 {
            old.invalidate_rows(read_begin, read_end);
            new.invalidate_row(read_begin);
            new.invalidate_row(read_end);
        }

        for _ in 0..steps {
            if read_begin != 0 {
                old.invalidate_row(read_begin);
            }
            if read_end != rows {
                old.invalidate_row(row_end);
            }
            update_block_dist(&old, &new, row_begin + 1, row_end - 1); //cached part first
            update_block_dist(&old, &new, row_begin, row_begin + 1);
            update_block_dist(&old, &new, row_end - 1, row_end);
            if read_begin != 0 {
                new.flush_row(row_begin);
            }
            if read_end != rows {
                new.flush_row(row_end - 1);
            }

            barrier_wait_all();
            std::mem::swap(&mut old, &mut new);
        }
        if myid != 1 {
            old.flush_rows(row_begin, row_end);
            barrier_wait_all();
        } else {
            barrier_wait_all();
            old.invalidate_rows(row_end, rows); // invalidate others; no need to invalidate parts we already have in
            // cache
        }
    }

    let kernel = t.elapsed();
    let kernel_metrics = get_bench_metrics();
    print_bench_metrics_value(kernel_metrics);

    if myid == 1 {
        (Some(old.snapshot_dense()), kernel)
    } else {
        (None, kernel)
    }
}

pub fn checksum(grid: &[f64]) -> f64 {
    grid.iter().sum()
}
