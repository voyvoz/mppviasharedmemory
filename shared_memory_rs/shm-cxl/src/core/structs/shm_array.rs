//! Flat and segmented array views over CXL shared memory.
//!
//! Every distributed benchmark needs the same thing: a region of shared memory
//! addressed as an array of `u64` (or `f64`), with explicit cache maintenance at
//! the points where another host has to see the data. CXL 2.0 gives no hardware
//! coherence between hosts, so the caller decides where the flush and
//! invalidate boundaries are; these types only make that decision easy to
//! express, they never insert one on their own.
//!
//! Two naming conventions run through the API:
//!
//! * `load_local` / `store_local` / [`ShmU64Array::ptr_at`] touch memory with no
//!   cache maintenance at all. Use them inside a host-private phase, or in bulk
//!   ranges that are flushed or invalidated once at the end.
//! * `get` / `set` / `set_rmw` are the single-element synchronising forms: they
//!   invalidate before reading and flush after writing, and they are counted
//!   against the process' CXL load/store timers.
//!
//! [`ShmSegmentedU64Array`] exists because the bitmap allocator caps a single
//! allocation; it presents several allocations as one logical array and
//! publishes a small segment table so peers can attach to it.

use crate::allocator::CXL_ALLOC_MIN_SIZE;
use crate::core::Cacheline;
use crate::core::{
    cxl_read_cacheline_sync, flush_sync, invalidate_sync, range_flush_sync, range_invalidate_sync,
};
use crate::utils::timestamps::CxlMetricTimer;
use allocator_api2::alloc::Allocator;
use std::alloc::Layout;
use std::sync::atomic::AtomicU64;

/// Allocate `cls` cacheline-aligned cachelines of CXL memory.
pub fn alloc_cachelines<A: Allocator>(alloc: &A, cls: usize) -> *mut Cacheline {
    let mem = alloc
        .allocate(Layout::from_size_align(cls.max(1) * 64, 64).unwrap())
        .unwrap();
    mem.as_ptr() as *mut Cacheline
}

/// Allocate room for `len_u64` `u64` values, rounded up to whole cachelines.
pub fn alloc_u64_region<A: Allocator>(alloc: &A, len_u64: usize) -> *mut Cacheline {
    alloc_cachelines(alloc, len_u64.div_ceil(8).max(1))
}

/// A `u64` view over one contiguous CXL region.
///
/// The region is untyped shared memory, so the view carries no length: callers
/// index it with whatever bounds the surrounding data structure defines.
pub struct ShmU64Array {
    base: *mut u64,
}

unsafe impl Send for ShmU64Array {}
unsafe impl Sync for ShmU64Array {}

impl ShmU64Array {
    pub fn from_cl(base_cl: *mut Cacheline) -> Self {
        ShmU64Array {
            base: base_cl as *mut u64,
        }
    }

    /// View a region that is already addressed as `u64`.
    pub fn from_ptr(base: *mut u64) -> Self {
        ShmU64Array { base }
    }

    pub fn from_offset(offset: u64) -> Self {
        ShmU64Array::from_cl(Cacheline::ptr_from_offset(offset))
    }

    /// Offset of this region within the mapping, for publishing to peers.
    pub fn offset(&self) -> u64 {
        Cacheline::ptr_to_offset(self.base as *mut Cacheline)
    }

    pub fn as_ptr(&self) -> *const u64 {
        self.base
    }

    #[inline]
    pub fn ptr_at(&self, i: usize) -> *mut u64 {
        unsafe { self.base.add(i) }
    }

    /// Read without invalidating first. Correct only when this host owns the
    /// value, or when the covering range was invalidated in bulk.
    #[inline]
    pub fn load_local(&self, i: usize) -> u64 {
        unsafe { *self.base.add(i) }
    }

    /// Write without flushing. The caller publishes the range afterwards.
    #[inline]
    pub fn store_local(&self, i: usize, v: u64) {
        unsafe { *self.base.add(i) = v }
    }

    /// Read a value another host may have written.
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        let _timer = CxlMetricTimer::load();
        unsafe { cxl_read_cacheline_sync(self.cl_of(i)).d[i % 8] }
    }

    /// Write a value another host has to see.
    #[inline]
    pub fn set(&self, i: usize, v: u64) {
        let _timer = CxlMetricTimer::store();
        unsafe {
            *self.base.add(i) = v;
            flush_sync(self.cl_of(i));
        }
    }

    /// Write a value into a cacheline whose other lanes must survive, which
    /// needs the line invalidated before the read-modify-write.
    #[inline]
    pub fn set_rmw(&self, i: usize, v: u64) {
        let _timer = CxlMetricTimer::store();
        unsafe {
            invalidate_sync(self.cl_of(i));
            *self.base.add(i) = v;
            flush_sync(self.cl_of(i));
        }
    }

    /// Reinterpret element `i` as an atomic. Atomics are coherent within a
    /// host only; across hosts the caller still needs flush/invalidate.
    pub fn atomic(&self, i: usize) -> &AtomicU64 {
        unsafe { &*(self.base.add(i) as *const AtomicU64) }
    }

    #[inline]
    pub fn load_f64_local(&self, i: usize) -> f64 {
        f64::from_bits(self.load_local(i))
    }

    #[inline]
    pub fn store_f64_local(&self, i: usize, x: f64) {
        self.store_local(i, x.to_bits());
    }

    pub fn fill_flush(&self, len: usize, v: u64) {
        let _timer = CxlMetricTimer::store();
        for i in 0..len {
            self.store_local(i, v);
        }
        self.flush_range(len);
    }

    pub fn store_slice_flush(&self, data: &[u64]) {
        self.store_slice_flush_at(0, data);
    }

    pub fn store_slice_flush_at(&self, start: usize, data: &[u64]) {
        let _timer = CxlMetricTimer::store();
        for (i, &v) in data.iter().enumerate() {
            self.store_local(start + i, v);
        }
        self.flush_slice(start, data.len());
    }

    pub fn load_slice_sync_at(&self, start: usize, out: &mut [u64]) {
        let _timer = CxlMetricTimer::load();
        self.invalidate_slice(start, out.len());
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.load_local(start + i);
        }
    }

    pub fn flush_range(&self, len: usize) {
        range_flush_sync(self.base as *const Cacheline, len.div_ceil(8));
    }

    pub fn invalidate_range(&self, len: usize) {
        range_invalidate_sync(self.base as *const Cacheline, len.div_ceil(8));
    }

    /// Flush only the cachelines covering `[start, start+len)`.
    pub fn flush_slice(&self, start: usize, len: usize) {
        let (first, cls) = Self::cacheline_span(start, len);
        if cls == 0 {
            return;
        }
        range_flush_sync(unsafe { (self.base as *const Cacheline).add(first) }, cls);
    }

    /// Invalidate only the cachelines covering `[start, start+len)`.
    pub fn invalidate_slice(&self, start: usize, len: usize) {
        let (first, cls) = Self::cacheline_span(start, len);
        if cls == 0 {
            return;
        }
        range_invalidate_sync(unsafe { (self.base as *const Cacheline).add(first) }, cls);
    }

    #[inline]
    fn cl_of(&self, i: usize) -> *mut Cacheline {
        unsafe { (self.base as *mut Cacheline).add(i / 8) }
    }

    /// First cacheline index and count covering `[start, start+len)`.
    #[inline]
    fn cacheline_span(start: usize, len: usize) -> (usize, usize) {
        if len == 0 {
            return (0, 0);
        }
        let first = start / 8;
        let last = (start + len - 1) / 8;
        (first, last - first + 1)
    }
}

/// An `f64` view over one contiguous CXL region, stored as raw `u64` bits so it
/// shares the cacheline arithmetic with [`ShmU64Array`].
pub struct ShmF64Array {
    inner: ShmU64Array,
}

unsafe impl Send for ShmF64Array {}
unsafe impl Sync for ShmF64Array {}

impl ShmF64Array {
    pub fn from_cl(base_cl: *mut Cacheline) -> Self {
        Self {
            inner: ShmU64Array::from_cl(base_cl),
        }
    }

    pub fn from_ptr(base: *mut u64) -> Self {
        Self {
            inner: ShmU64Array::from_ptr(base),
        }
    }

    pub fn from_offset(offset: u64) -> Self {
        Self {
            inner: ShmU64Array::from_offset(offset),
        }
    }

    pub fn offset(&self) -> u64 {
        self.inner.offset()
    }

    #[inline]
    pub fn load_local(&self, i: usize) -> f64 {
        self.inner.load_f64_local(i)
    }

    #[inline]
    pub fn store_local(&self, i: usize, v: f64) {
        self.inner.store_f64_local(i, v);
    }

    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        f64::from_bits(self.inner.get(i))
    }

    #[inline]
    pub fn set(&self, i: usize, v: f64) {
        self.inner.set(i, v.to_bits());
    }

    pub fn fill_flush(&self, len: usize, v: f64) {
        self.inner.fill_flush(len, v.to_bits());
    }

    pub fn store_slice_flush_at(&self, start: usize, data: &[f64]) {
        let _timer = CxlMetricTimer::store();
        for (i, &v) in data.iter().enumerate() {
            self.store_local(start + i, v);
        }
        self.inner.flush_slice(start, data.len());
    }

    pub fn load_slice_sync_at(&self, start: usize, out: &mut [f64]) {
        let _timer = CxlMetricTimer::load();
        self.inner.invalidate_slice(start, out.len());
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = self.load_local(start + i);
        }
    }

    pub fn flush_range(&self, len: usize) {
        self.inner.flush_range(len);
    }

    pub fn invalidate_range(&self, len: usize) {
        self.inner.invalidate_range(len);
    }

    pub fn flush_slice(&self, start: usize, len: usize) {
        self.inner.flush_slice(start, len);
    }

    pub fn invalidate_slice(&self, start: usize, len: usize) {
        self.inner.invalidate_slice(start, len);
    }
}

/// Default segment length for [`ShmSegmentedU64Array`], in `u64` values.
///
/// One bitmap word of the allocator covers 64 of its blocks, so a segment of
/// exactly that size stays on the allocator's fast path and within its
/// single-allocation limit.
pub const SEGMENT_U64S_DEFAULT: usize = 64 * CXL_ALLOC_MIN_SIZE / 8;

/// A logical `u64` array split across several CXL allocations.
///
/// The bitmap allocator caps how much one `allocate` call can return, so large
/// arrays are built from a series of segments. A small shared table records the
/// length, the segment length and each segment's offset, so a peer attaches
/// with nothing but that table's offset.
///
/// The segment length is a power of two, which lets element addressing use a
/// shift and a mask instead of a 64-bit division -- worth having, because these
/// arrays are indexed in the innermost loops.
pub struct ShmSegmentedU64Array {
    segments: Vec<ShmU64Array>,
    len: usize,
    segment_len: usize,
    segment_shift: u32,
}

unsafe impl Send for ShmSegmentedU64Array {}
unsafe impl Sync for ShmSegmentedU64Array {}

impl ShmSegmentedU64Array {
    /// Allocate an uninitialised logical array of `len` values, using
    /// [`SEGMENT_U64S_DEFAULT`] segments. Returns the segment table's offset.
    pub fn create_uninit<A: Allocator>(alloc: &A, len: usize) -> (Self, u64) {
        Self::create_uninit_with(alloc, len, SEGMENT_U64S_DEFAULT)
    }

    /// As [`create_uninit`](Self::create_uninit), with an explicit segment
    /// length. `segment_len` must be a power of two.
    pub fn create_uninit_with<A: Allocator>(
        alloc: &A,
        len: usize,
        segment_len: usize,
    ) -> (Self, u64) {
        assert!(
            segment_len.is_power_of_two(),
            "segment length must be a power of two"
        );
        let segment_count = len.max(1).div_ceil(segment_len);
        let table = ShmU64Array::from_cl(alloc_u64_region(alloc, 3 + segment_count));
        let mut segments = Vec::with_capacity(segment_count);
        for segment_index in 0..segment_count {
            let begin = segment_index * segment_len;
            let this_len = (len.saturating_sub(begin)).min(segment_len).max(1);
            let segment = ShmU64Array::from_cl(alloc_u64_region(alloc, this_len));
            table.store_local(3 + segment_index, segment.offset());
            segments.push(segment);
        }
        Self::publish_table(table, len, segment_len, segment_count, segments).with_table_offset()
    }

    /// Allocate a logical array initialised from `values` and publish it.
    pub fn create_from<A: Allocator>(alloc: &A, values: &[u64]) -> (Self, u64) {
        Self::create_from_with(alloc, values, SEGMENT_U64S_DEFAULT)
    }

    /// As [`create_from`](Self::create_from), with an explicit segment length.
    pub fn create_from_with<A: Allocator>(
        alloc: &A,
        values: &[u64],
        segment_len: usize,
    ) -> (Self, u64) {
        assert!(
            segment_len.is_power_of_two(),
            "segment length must be a power of two"
        );
        let len = values.len();
        let segment_count = len.max(1).div_ceil(segment_len);
        let table = ShmU64Array::from_cl(alloc_u64_region(alloc, 3 + segment_count));
        let mut segments = Vec::with_capacity(segment_count);
        for segment_index in 0..segment_count {
            let begin = segment_index * segment_len;
            let end = (begin + segment_len).min(len);
            let segment = ShmU64Array::from_cl(alloc_u64_region(alloc, (end - begin).max(1)));
            if begin < end {
                segment.store_slice_flush(&values[begin..end]);
            }
            table.store_local(3 + segment_index, segment.offset());
            segments.push(segment);
        }
        Self::publish_table(table, len, segment_len, segment_count, segments).with_table_offset()
    }

    /// Attach to an array published by another host.
    pub fn attach(table_offset: u64) -> Self {
        let table = ShmU64Array::from_offset(table_offset);
        table.invalidate_range(3);
        let len = table.load_local(0) as usize;
        let segment_len = table.load_local(1) as usize;
        let segment_count = table.load_local(2) as usize;
        assert!(
            segment_len.is_power_of_two(),
            "segment length must be a power of two"
        );
        table.invalidate_range(3 + segment_count);
        let segments = (0..segment_count)
            .map(|index| ShmU64Array::from_offset(table.load_local(3 + index)))
            .collect();
        Self {
            segments,
            len,
            segment_len,
            segment_shift: segment_len.trailing_zeros(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn load_local(&self, index: usize) -> u64 {
        debug_assert!(index < self.len);
        self.segments[index >> self.segment_shift].load_local(index & (self.segment_len - 1))
    }

    #[inline]
    pub fn store_local(&self, index: usize, value: u64) {
        debug_assert!(index < self.len);
        self.segments[index >> self.segment_shift]
            .store_local(index & (self.segment_len - 1), value);
    }

    /// Split `[start, start+len)` into the maximal ranges that are contiguous
    /// in one allocation, yielding a base pointer and element count each. Bulk
    /// copy and cache-maintenance phases iterate these instead of resolving a
    /// segment per element.
    pub fn contiguous_pieces(
        &self,
        start: usize,
        len: usize,
    ) -> impl Iterator<Item = (*mut u64, usize)> + '_ {
        debug_assert!(start.saturating_add(len) <= self.len);
        let mut cursor = start;
        let end = start + len;
        std::iter::from_fn(move || {
            if cursor >= end {
                return None;
            }
            let segment_index = cursor >> self.segment_shift;
            let segment_offset = cursor & (self.segment_len - 1);
            let chunk = (end - cursor).min(self.segment_len - segment_offset);
            cursor += chunk;
            Some((self.segments[segment_index].ptr_at(segment_offset), chunk))
        })
    }

    pub fn flush_slice(&self, mut start: usize, mut len: usize) {
        debug_assert!(start.saturating_add(len) <= self.len);
        while len != 0 {
            let segment_index = start >> self.segment_shift;
            let segment_offset = start & (self.segment_len - 1);
            let chunk = len.min(self.segment_len - segment_offset);
            self.segments[segment_index].flush_slice(segment_offset, chunk);
            start += chunk;
            len -= chunk;
        }
    }

    pub fn invalidate_slice(&self, mut start: usize, mut len: usize) {
        debug_assert!(start.saturating_add(len) <= self.len);
        while len != 0 {
            let segment_index = start >> self.segment_shift;
            let segment_offset = start & (self.segment_len - 1);
            let chunk = len.min(self.segment_len - segment_offset);
            self.segments[segment_index].invalidate_slice(segment_offset, chunk);
            start += chunk;
            len -= chunk;
        }
    }

    pub fn invalidate_all(&self) {
        self.invalidate_slice(0, self.len);
    }

    fn publish_table(
        table: ShmU64Array,
        len: usize,
        segment_len: usize,
        segment_count: usize,
        segments: Vec<ShmU64Array>,
    ) -> TableAndArray {
        table.store_local(0, len as u64);
        table.store_local(1, segment_len as u64);
        table.store_local(2, segment_count as u64);
        table.flush_range(3 + segment_count);
        TableAndArray {
            table_offset: table.offset(),
            array: Self {
                segments,
                len,
                segment_len,
                segment_shift: segment_len.trailing_zeros(),
            },
        }
    }
}

/// Carries the published table offset out of the constructors.
struct TableAndArray {
    table_offset: u64,
    array: ShmSegmentedU64Array,
}

impl TableAndArray {
    fn with_table_offset(self) -> (ShmSegmentedU64Array, u64) {
        (self.array, self.table_offset)
    }
}
