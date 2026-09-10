use crate::primitives::{ShmSegmentedU64Array, ShmU64Array};

const N_IDX: usize = 0;
const META_BASE: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct FrontierSegment {
    pub base: usize,
    pub len: usize,
}

/// Double-buffered shared frontier with one fixed lane per host.
///
/// Each host writes only its own lane and its own per-parity meta cacheline,
/// so no shared cursor, cross-host lock, or level word is needed: the level
/// is a local counter on every host and the per-level barrier is the only
/// synchronisation. Lane bases are derived from host id, so the meta word
/// only carries each lane's published length.
pub struct SharedFrontier {
    ctrl: ShmU64Array,
    buffers: [ShmSegmentedU64Array; 2],
    lane_stride: usize,
    nhosts: usize,
}

unsafe impl Send for SharedFrontier {}
unsafe impl Sync for SharedFrontier {}

impl SharedFrontier {
    pub fn ctrl_len(nhosts: usize) -> usize {
        META_BASE + 2 * nhosts * 8
    }

    pub fn new(
        ctrl: ShmU64Array,
        a: ShmSegmentedU64Array,
        b: ShmSegmentedU64Array,
        lane_stride: usize,
        nhosts: usize,
    ) -> Self {
        Self {
            ctrl,
            buffers: [a, b],
            lane_stride,
            nhosts,
        }
    }

    fn meta_index(&self, parity: usize, host: usize) -> usize {
        META_BASE + (parity * self.nhosts + host) * 8
    }

    pub fn lane_base(&self, host: usize) -> usize {
        host * self.lane_stride
    }

    pub fn seed(&self, source: u64, n: usize) {
        self.buffers[0].store_local(0, source);
        self.buffers[0].flush_slice(0, 1);
        self.ctrl.fill_flush(Self::ctrl_len(self.nhosts), 0);
        self.ctrl.set(N_IDX, n as u64);
        self.ctrl.set(self.meta_index(0, 0), 1);
    }

    pub fn n_from_ctrl(ctrl: &ShmU64Array) -> usize {
        ctrl.get(N_IDX) as usize
    }

    /// Snapshot every host's published segment for this parity. Valid only
    /// after the level barrier that follows the publishes.
    pub fn snapshot(&self, parity: usize) -> (Vec<FrontierSegment>, usize) {
        // Only this parity's meta block is read, so only it is invalidated:
        // on a 6000-level road graph every extra invalidated cacheline here
        // is a per-level CXL round-trip.
        self.ctrl
            .invalidate_slice(META_BASE + parity * self.nhosts * 8, self.nhosts * 8);
        let mut total = 0usize;
        let mut segments = Vec::with_capacity(self.nhosts);
        for host in 0..self.nhosts {
            let segment = FrontierSegment {
                base: self.lane_base(host),
                len: self.ctrl.load_local(self.meta_index(parity, host)) as usize,
            };
            total += segment.len;
            segments.push(segment);
        }
        (segments, total)
    }

    pub fn buffer(&self, parity: usize) -> &ShmSegmentedU64Array {
        &self.buffers[parity]
    }

    pub fn publish(&self, parity: usize, host: usize, len: usize) {
        // This cacheline is written only by `host` and read after the barrier.
        self.ctrl.set(self.meta_index(parity, host), len as u64);
    }
}
