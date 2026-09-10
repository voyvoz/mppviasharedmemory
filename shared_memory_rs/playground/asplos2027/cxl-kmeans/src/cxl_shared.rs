use shm_cxl::allocator_api2::alloc::Allocator;
use shm_cxl::core::Cacheline;
use shm_cxl::core::structs::shm_array::{
    ShmF64Array, ShmU64Array, alloc_cachelines, alloc_u64_region,
};
use shm_cxl::utils::timestamps::CxlMetricTimer;

const CONTROL_WORDS: usize = 2;
const CONTROL_ITERATIONS: usize = 0;
const CONTROL_CONVERGED: usize = 1;

fn align_words(words: usize) -> usize {
    words.div_ceil(8) * 8
}

pub struct SharedKMeansState {
    base: *mut Cacheline,
    k: usize,
    nhosts: usize,
    centers_len: usize,
    centers_words: usize,
    global_counts_words: usize,
    control_words: usize,
    partial_sums_stride: usize,
    partial_counts_stride: usize,
    partial_changed_stride: usize,
    partial_inertia_stride: usize,
    centers: ShmF64Array,
    global_counts: ShmU64Array,
    global_inertia: ShmF64Array,
    control: ShmU64Array,
    partial_sums: ShmF64Array,
    partial_counts: ShmU64Array,
    partial_changed: ShmU64Array,
    partial_inertia: ShmF64Array,
}

unsafe impl Send for SharedKMeansState {}
unsafe impl Sync for SharedKMeansState {}

impl SharedKMeansState {
    pub fn create<A: Allocator>(alloc: &A, k: usize, dim: usize, nhosts: usize) -> Self {
        let total_words = Self::total_words(k, dim, nhosts);
        let base = alloc_u64_region(alloc, total_words);
        let state = Self::from_base(base, k, dim, nhosts);
        state.zero_initialize();
        state
    }

    pub fn attach(offset: u64, k: usize, dim: usize, nhosts: usize) -> Self {
        Self::from_base(Cacheline::ptr_from_offset(offset), k, dim, nhosts)
    }

    pub fn offset(&self) -> u64 {
        Cacheline::ptr_to_offset(self.base)
    }

    pub fn initialize(&self, centers: &[f64]) {
        assert_eq!(centers.len(), self.centers_len);
        self.centers.store_slice_flush_at(0, centers);
        self.global_counts.fill_flush(self.k, 0);
        self.global_inertia.fill_flush(1, 0.0);
        self.control.store_slice_flush_at(0, &[0, 0]);
    }

    pub fn load_centers(&self, out: &mut [f64]) {
        assert_eq!(out.len(), self.centers_len);
        self.centers.load_slice_sync_at(0, out);
    }

    pub fn store_centers(&self, centers: &[f64]) {
        assert_eq!(centers.len(), self.centers_len);
        self.centers.store_slice_flush_at(0, centers);
    }

    pub fn store_global_counts(&self, counts: &[u64]) {
        assert_eq!(counts.len(), self.k);
        self.global_counts.store_slice_flush_at(0, counts);
    }

    pub fn load_global_counts(&self, out: &mut [u64]) {
        assert_eq!(out.len(), self.k);
        self.global_counts.load_slice_sync_at(0, out);
    }

    pub fn store_global_inertia(&self, inertia: f64) {
        self.global_inertia.set(0, inertia);
    }

    pub fn load_global_inertia(&self) -> f64 {
        self.global_inertia.get(0)
    }

    pub fn store_control(&self, iterations: u64, converged: bool) {
        self.control
            .store_slice_flush_at(0, &[iterations, u64::from(converged)]);
    }

    pub fn load_control(&self) -> (usize, bool) {
        let iterations = self.control.get(CONTROL_ITERATIONS) as usize;
        let converged = self.control.get(CONTROL_CONVERGED) != 0;
        (iterations, converged)
    }

    pub fn write_slot(
        &self,
        host_idx: usize,
        sums: &[f64],
        counts: &[u64],
        changed: u64,
        inertia: f64,
    ) {
        assert!(host_idx < self.nhosts);
        assert_eq!(sums.len(), self.centers_len);
        assert_eq!(counts.len(), self.k);

        let sum_offset = host_idx * self.partial_sums_stride;
        let count_offset = host_idx * self.partial_counts_stride;
        let changed_offset = host_idx * self.partial_changed_stride;
        let inertia_offset = host_idx * self.partial_inertia_stride;
        self.partial_sums.store_slice_flush_at(sum_offset, sums);
        self.partial_counts
            .store_slice_flush_at(count_offset, counts);
        self.partial_changed.set(changed_offset, changed);
        self.partial_inertia.set(inertia_offset, inertia);
    }

    pub fn accumulate_partials(
        &self,
        global_sums: &mut [f64],
        global_counts: &mut [u64],
    ) -> (u64, f64) {
        let _load_timer = CxlMetricTimer::load();
        assert_eq!(global_sums.len(), self.centers_len);
        assert_eq!(global_counts.len(), self.k);

        global_sums.fill(0.0);
        global_counts.fill(0);

        self.partial_sums
            .invalidate_range(self.nhosts * self.partial_sums_stride);
        self.partial_counts
            .invalidate_range(self.nhosts * self.partial_counts_stride);
        self.partial_changed
            .invalidate_range(self.nhosts * self.partial_changed_stride);
        self.partial_inertia
            .invalidate_range(self.nhosts * self.partial_inertia_stride);

        let mut global_changed = 0u64;
        let mut global_inertia = 0.0f64;

        for host in 0..self.nhosts {
            let sum_offset = host * self.partial_sums_stride;
            for (i, slot) in global_sums.iter_mut().enumerate() {
                *slot += self.partial_sums.load_local(sum_offset + i);
            }

            let count_offset = host * self.partial_counts_stride;
            for (i, slot) in global_counts.iter_mut().enumerate() {
                *slot += self.partial_counts.load_local(count_offset + i);
            }

            global_changed += self
                .partial_changed
                .load_local(host * self.partial_changed_stride);
            global_inertia += self
                .partial_inertia
                .load_local(host * self.partial_inertia_stride);
        }

        (global_changed, global_inertia)
    }

    fn zero_initialize(&self) {
        self.centers.fill_flush(self.centers_words, 0.0);
        self.global_counts.fill_flush(self.global_counts_words, 0);
        self.global_inertia.fill_flush(1, 0.0);
        self.control.fill_flush(self.control_words, 0);
        self.partial_sums
            .fill_flush(self.nhosts * self.partial_sums_stride, 0.0);
        self.partial_counts
            .fill_flush(self.nhosts * self.partial_counts_stride, 0);
        self.partial_changed
            .fill_flush(self.nhosts * self.partial_changed_stride, 0);
        self.partial_inertia
            .fill_flush(self.nhosts * self.partial_inertia_stride, 0.0);
    }

    fn from_base(base: *mut Cacheline, k: usize, dim: usize, nhosts: usize) -> Self {
        let centers_len = k * dim;
        let centers_words = align_words(centers_len);
        let global_counts_words = align_words(k);
        let global_inertia_words = align_words(1);
        let control_words = align_words(CONTROL_WORDS);
        let partial_sums_stride = align_words(centers_len);
        let partial_counts_stride = align_words(k);
        let partial_changed_stride = 8;
        let partial_inertia_stride = 8;
        let partial_sums_words = nhosts * partial_sums_stride;
        let partial_counts_words = nhosts * partial_counts_stride;
        let partial_changed_words = nhosts * partial_changed_stride;
        let partial_inertia_words = nhosts * partial_inertia_stride;

        let mut cursor = 0usize;
        let base_u64 = base as *mut u64;

        let centers = ShmF64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += centers_words;

        let global_counts = ShmU64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += global_counts_words;

        let global_inertia = ShmF64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += global_inertia_words;

        let control = ShmU64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += control_words;

        let partial_sums = ShmF64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += partial_sums_words;

        let partial_counts = ShmU64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += partial_counts_words;

        let partial_changed = ShmU64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += partial_changed_words;

        let partial_inertia = ShmF64Array::from_ptr(unsafe { base_u64.add(cursor) });
        cursor += partial_inertia_words;

        debug_assert_eq!(cursor, Self::total_words(k, dim, nhosts));

        SharedKMeansState {
            base,
            k,
            nhosts,
            centers_len,
            centers_words,
            global_counts_words,
            control_words,
            partial_sums_stride,
            partial_counts_stride,
            partial_changed_stride,
            partial_inertia_stride,
            centers,
            global_counts,
            global_inertia,
            control,
            partial_sums,
            partial_counts,
            partial_changed,
            partial_inertia,
        }
    }

    fn total_words(k: usize, dim: usize, nhosts: usize) -> usize {
        let centers_len = k * dim;
        let centers_words = align_words(centers_len);
        let global_counts_words = align_words(k);
        let global_inertia_words = align_words(1);
        let control_words = align_words(CONTROL_WORDS);
        let partial_sums_words = nhosts * align_words(centers_len);
        let partial_counts_words = nhosts * align_words(k);
        let partial_changed_words = nhosts * 8;
        let partial_inertia_words = nhosts * 8;

        centers_words
            + global_counts_words
            + global_inertia_words
            + control_words
            + partial_sums_words
            + partial_counts_words
            + partial_changed_words
            + partial_inertia_words
    }
}
