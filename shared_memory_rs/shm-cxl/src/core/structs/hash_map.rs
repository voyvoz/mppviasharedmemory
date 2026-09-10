use crate::core::{
    Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync, cxl_write_cl_sync, invalidate,
    load_and_store_fence, range_invalidate, range_invalidate_sync,
};
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync, cxl_typed_new_sync};
use crate::cxl_write_sync;
use crate::mutex::CXLMutexImpl;
use allocator_api2::alloc::Allocator;
use log::{debug, trace};
use std::alloc::Layout;
use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::sync::Arc;

// How many offsets we store within a Cacheline
const OFFSETS_PER_CL: usize = 8;

/// CXL State data:
///----------------------------------
/// | MetaData (Buckets)            |
/// ---------------------------------
/// | Offset to bucket 0 | Offset   |
/// ---------------------------------

#[derive(Debug)]
pub struct CXLHashMap<A>
where
    A: Allocator,
{
    buckets: u64,
    bucket_default_capacity: usize,

    lock_size_in_cls: usize,

    alloc: Arc<A>,

    // Remote state
    // ------------
    // Metadata
    base_cl: *mut Cacheline,
    // Cachlines that contain the offsets
    cl_bucket_offsets: *mut Cacheline,
    // Per Bucket lock
    cl_bucket_locks: *mut Cacheline,

    read_cache: Vec<Option<CXLHashMapBucket<A>>>,
}

const HM_BUCKET_COUNTER_IDX: usize = 6;
const HM_BUCKET_DEFAULT_CAPACITY_IDX: usize = 7;

/*
 * Idea: Put all the locks inside the same lock?
 *
 */

impl<A> CXLHashMap<A>
where
    A: Allocator,
{
    #[inline]
    pub fn get_base(&self) -> *mut Cacheline {
        self.base_cl
    }

    /// `offset_base` is used to calculate all offsets.
    /// `buckets` determines how many different buckets will be used. Rehashing is not supported. Must be a multiple of {OFFSETS_PER_CL}.
    /// `bucket_default_capacity` determines how much memory should be preallocated for a bucket.
    /// `hosts` is required to reflect how many hosts are expected to use this data structure at the same time.
    /// `alloc` is the memory allocator to be used for all internal allocations.
    pub fn try_new(
        buckets: usize,
        bucket_default_capacity: usize,
        hosts: usize,
        alloc: Arc<A>,
    ) -> anyhow::Result<Self> {
        assert!(buckets >= 1, "buckets should be more than 1");
        assert!(
            buckets % 8 == 0,
            "buckets must be multiple of {OFFSETS_PER_CL}"
        );

        // alloc the space and place this struct into the allocation
        let mem = alloc
            .allocate(Layout::from_size_align(1024, 64).unwrap())
            .unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;

        // Metadata
        cxl_typed_new_sync::<Self>(base_cl);
        cxl_write_sync!( (base_cl => [(HM_BUCKET_COUNTER_IDX, buckets), (HM_BUCKET_DEFAULT_CAPACITY_IDX, bucket_default_capacity)]) );
        let cl_bucket_offsets = unsafe { base_cl.add(1) };

        // Determine required cachelines for amount of buckets
        let required_cls_for_buckets = buckets / OFFSETS_PER_CL;
        // Reset each cacheline
        for i in 0..required_cls_for_buckets {
            let bucket_cl = unsafe { cl_bucket_offsets.add(i) };
            cl_set_zero_sync(bucket_cl);
        }

        // Init all the locks
        let cl_bucket_locks = unsafe { cl_bucket_offsets.add(required_cls_for_buckets) };
        let mut next_cl = cl_bucket_locks;
        let mut lock_size = 0;
        for _ in 0..buckets {
            let l = CXLMutexImpl::init(next_cl, hosts);
            lock_size = CXLMutexImpl::required_cxl_shared_state_cachelines_n(l.hosts);
            next_cl = unsafe { next_cl.add(lock_size) };
        }

        debug!(
            "[{:#}] CXLHashMap::try_new(buckets={buckets}, cap={bucket_default_capacity}, hosts={hosts}) -> buckets:{} locks:{}",
            Cacheline::ptr_to_offset(base_cl),
            Cacheline::ptr_to_offset(cl_bucket_offsets),
            Cacheline::ptr_to_offset(cl_bucket_locks),
        );

        let mut v = Vec::with_capacity(buckets);
        for _ in 0..buckets {
            v.push(None);
        }
        Ok(CXLHashMap {
            buckets: buckets as u64,
            bucket_default_capacity,
            alloc,
            base_cl,
            cl_bucket_offsets,
            cl_bucket_locks,
            lock_size_in_cls: lock_size,
            read_cache: v,
        })
    }

    /// Tries to create Self from the given `base_cl`.
    pub fn from_cl_sync(base_cl: *mut Cacheline, alloc: Arc<A>) -> Option<Self> {
        cxl_typed_from_cl_sync::<Self>(base_cl, |meta| {
            let buckets = meta.d[HM_BUCKET_COUNTER_IDX] as usize;
            let bucket_default_capacity = meta.d[HM_BUCKET_DEFAULT_CAPACITY_IDX] as usize;

            let cl_bucket_offsets = unsafe { base_cl.add(1) };
            let required_cl_for_buckets = buckets / OFFSETS_PER_CL;
            let cl_bucket_locks = unsafe { cl_bucket_offsets.add(required_cl_for_buckets) };

            let next_cl = cl_bucket_locks;
            let lock_at_idx_1 = CXLMutexImpl::from_cl_sync(next_cl).unwrap();
            let lock_size_in_cls =
                CXLMutexImpl::required_cxl_shared_state_cachelines_n(lock_at_idx_1.hosts);

            debug!(
                "[{:#}] CXLHashMap::from_cl_sync(base_cl={base_cl:p}) -> buckets:{} locks:{}",
                Cacheline::ptr_to_offset(base_cl),
                Cacheline::ptr_to_offset(cl_bucket_offsets),
                Cacheline::ptr_to_offset(cl_bucket_locks),
            );

            let mut v = Vec::with_capacity(buckets);
            for _ in 0..buckets {
                v.push(None);
            }
            CXLHashMap {
                buckets: buckets as u64,
                bucket_default_capacity,
                alloc,
                base_cl,
                cl_bucket_offsets,
                cl_bucket_locks,
                lock_size_in_cls,
                read_cache: v,
            }
        })
    }

    fn calc_bucket_idx(&self, key: u64) -> usize {
        let mut hasher = crate::utils::FnvHasher::default();
        key.hash(&mut hasher);
        let idx = (hasher.finish() % self.buckets) as usize;
        trace!("calc_bucket_idx(key={key}) -> {idx}");
        idx
    }

    fn get_lock_of_bucket_sync(&self, bucket_idx: usize) -> CXLMutexImpl {
        let lock_cl = unsafe { self.cl_bucket_locks.add(self.lock_size_in_cls * bucket_idx) };
        CXLMutexImpl::from_cl_sync(lock_cl).unwrap()
    }

    fn get_bucket_offset_sync(&self, bucket_idx: usize) -> u64 {
        let cl_idx = bucket_idx / OFFSETS_PER_CL;
        let bucket_cl = unsafe { self.cl_bucket_offsets.add(cl_idx) };
        let cl = cxl_read_cacheline_sync(bucket_cl);
        let u64_idx = bucket_idx % OFFSETS_PER_CL;

        let offset = cl.d[u64_idx];
        trace!(
            "get_bucket_offset_sync(bucket_idx={bucket_idx}) -> start:{:p} bucket_cl_offset:{:} u64_idx:{u64_idx} {cl:?} -> offset_is:{offset}",
            self.cl_bucket_offsets,
            Cacheline::ptr_to_offset(bucket_cl),
        );
        offset
    }

    fn set_bucket_offset_sync(&self, bucket_idx: usize, offset: u64) {
        let cl_idx = bucket_idx / OFFSETS_PER_CL;
        let bucket_cl = unsafe { self.cl_bucket_offsets.add(cl_idx) };
        let mut cl = cxl_read_cacheline_sync(bucket_cl);
        let u64_idx = bucket_idx % OFFSETS_PER_CL;

        cl.d[u64_idx] = offset;
        cxl_write_cl_sync(bucket_cl, &cl);
        trace!(
            "set_bucket_offset(bucket_idx={bucket_idx}, offset={offset}) -> start:{:p} bucket_cl_offset:{:} u64_idx:{u64_idx} {cl:?}",
            self.cl_bucket_offsets,
            Cacheline::ptr_to_offset(bucket_cl),
        );
    }

    /// This insert ensures that each update to the data structure is presisted into CXL
    /// and directly purged from local caches to ensure next updates read from remote memory first.
    pub fn insert_sync(&self, key: u64, value: u64) {
        trace!("insert(key={key}, value={value})");

        let bucket_idx = self.calc_bucket_idx(key);

        let lock = self.get_lock_of_bucket_sync(bucket_idx);
        lock.lock();

        let bucket_offset = self.get_bucket_offset_sync(bucket_idx);

        match bucket_offset == 0 {
            // Need to alloc new bucket and set offset to allocated bucket
            true => {
                let bucket =
                    CXLHashMapBucket::new(self.bucket_default_capacity, self.alloc.clone());
                self.set_bucket_offset_sync(
                    bucket_idx,
                    Cacheline::ptr_to_offset(bucket.get_base()),
                );
                bucket
            }
            // Append to existing bucket
            false => CXLHashMapBucket::from_cl_sync(
                Cacheline::ptr_from_offset(bucket_offset),
                self.alloc.clone(),
            )
            .unwrap(),
        }
        .insert_sync(key, value);

        lock.unlock();
    }

    ///
    pub fn remove_sync(&self, key: u64) {
        let bucket_idx = self.calc_bucket_idx(key);

        let lock = self.get_lock_of_bucket_sync(bucket_idx);
        lock.lock();

        let bucket_offset = self.get_bucket_offset_sync(bucket_idx);

        match bucket_offset == 0 {
            true => {}
            false => CXLHashMapBucket::from_cl_sync(
                Cacheline::ptr_from_offset(bucket_offset),
                self.alloc.clone(),
            )
            .unwrap()
            .remove(key),
        };

        lock.unlock();
    }

    /// In anticipation of an upcoming probe phase we invalidate all of our local cache entries
    pub fn prep_for_probe(&self) {
        let mut encountered_empty_bucket = false;
        // invalidate local buckets
        for i in 0..self.buckets {
            let bucket_offset = self.get_bucket_offset(i as usize);
            match bucket_offset == 0 {
                true => {
                    encountered_empty_bucket = true;
                }
                false => CXLHashMapBucket::from_cl_sync(
                    Cacheline::ptr_from_offset(bucket_offset),
                    self.alloc.clone(),
                )
                .unwrap()
                .invalidate(),
            };
            // If we checked the last
            if i % 8 == 7 && encountered_empty_bucket {
                invalidate(Cacheline::ptr_from_offset(bucket_offset));
                // Reset for next bucket cacheline
                encountered_empty_bucket = false;
            }
        }
        load_and_store_fence();
    }

    /// Prints the local state of the HashMap buckets
    pub fn print(&self) {
        for i in 0..self.buckets {
            let bucket_offset = self.get_bucket_offset_sync(i as usize);
            match bucket_offset == 0 {
                true => {}
                false => {
                    let meta = unsafe { *Cacheline::ptr_from_offset(bucket_offset) };
                    if meta.d[0] != 0xb00b132 || meta.d[BUCKET_COUNTER_IDX] > 16 {
                        print!("[{bucket_offset}] invalid state ");
                    } else {
                        let elem_counter = meta.d[BUCKET_COUNTER_IDX];
                        let bucket_data = Cacheline::ptr_from_offset(bucket_offset + 1);
                        print!("[{bucket_offset}] elems: {} ", elem_counter);
                        let elem_ptr = bucket_data as *const u64;
                        for i in (0..(elem_counter * 2)).step_by(2) {
                            let key = unsafe { *elem_ptr.add(i as usize) };
                            let value = unsafe { *elem_ptr.add((i + 1) as usize) };
                            print!(" ({:?}, {:?})", key, value);
                        }
                        println!("");
                    }
                }
            };
        }
    }

    /// A variant that potentially uses locally cached data for bucket offsets.
    /// NOTE must only be called after prep_for_probe() !
    fn get_bucket_offset(&self, bucket_idx: usize) -> u64 {
        let cl_idx = bucket_idx / OFFSETS_PER_CL;
        let bucket_cl = unsafe { self.cl_bucket_offsets.add(cl_idx) };
        let cl = unsafe { *bucket_cl };
        let u64_idx = bucket_idx % OFFSETS_PER_CL;

        let offset = cl.d[u64_idx];
        trace!(
            "get_bucket_offset(bucket_idx:{bucket_idx}) -> start:{:p} bucket_cl_offset:{:} u64_idx:{u64_idx} {cl:?} -> offset_is:{offset}",
            self.cl_bucket_offsets,
            Cacheline::ptr_to_offset(bucket_cl),
        );
        offset
    }

    /// This function assumes that the backing memory does not change anymore
    /// allowing caching and potential prefetching.
    pub fn get_values(&mut self, key: u64) -> Vec<(u64, u64)> {
        let bucket_idx = self.calc_bucket_idx(key);

        if self.read_cache[bucket_idx].is_some() {
            return self.read_cache[bucket_idx]
                .as_ref()
                .unwrap()
                .get_values(key);
        }

        let bucket_offset = self.get_bucket_offset(bucket_idx);
        trace!("get_values(key:{key}) -> bucket_idx:{bucket_idx} bucket_offset:{bucket_offset}");
        let result = match bucket_offset == 0 {
            true => vec![],
            false => {
                let b = CXLHashMapBucket::from_cl_sync(
                    Cacheline::ptr_from_offset(bucket_offset),
                    self.alloc.clone(),
                )
                .unwrap();
                self.read_cache[bucket_idx] = Some(b);
                self.get_values(key)
            }
        };

        result
    }
}

impl<A> Drop for CXLHashMap<A>
where
    A: Allocator,
{
    fn drop(&mut self) {
        // Go to each bucket and drop it
        // Then drop map itself
    }
}

impl<A> CXLSharedState for CXLHashMap<A>
where
    A: Allocator,
{
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        //(self.free_memory_start as usize - self.bit_map_start as usize) / 64
        unimplemented!()
        //1 + Buckets / CL + Buckets * Locks
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        todo!()
    }
    fn get_cxl_type_id() -> u64 {
        0xdeadb00b
    }
}

#[derive(Debug, Clone)]
pub struct CXLHashMapBucket<A>
where
    A: Allocator,
{
    // Shared state
    base_cl: *mut Cacheline,

    // local
    alloc_size_bytes: u64, // Size of the backing allocation in bytes
    elem_counter: u64,

    // Local
    alloc: Arc<A>,
}

const BUCKET_COUNTER_IDX: usize = 6;
const BUCKET_CAPACITY_IDX: usize = 7;

impl<A> CXLHashMapBucket<A>
where
    A: Allocator,
{
    /// Used to calculate the offset to this struct in shared memory
    pub fn get_base(&self) -> *mut Cacheline {
        self.base_cl
    }
    pub fn get_data_start(&self) -> *mut Cacheline {
        unsafe { self.get_base().add(1) }
    }

    /// `bucket_slot_size` size of a single value slot in bytes
    /// `bucket_default_capacity` space that is allocated by default for this bucket
    pub fn new(bucket_default_capacity: usize, alloc: Arc<A>) -> Self {
        // Bytes occupied by a single entry consisting of: sizeof(k) + sizeof(v)
        let kv_pair_bytes = 8 * 2;
        let mem = alloc
            .allocate(Layout::from_size_align(bucket_default_capacity * kv_pair_bytes, 64).unwrap())
            .unwrap();
        let base_cl = mem.as_ptr() as *mut Cacheline;
        let alloc_size = mem.len();

        cxl_typed_new_sync::<Self>(base_cl);
        cxl_write_sync!((base_cl => [(BUCKET_COUNTER_IDX, 0), (BUCKET_CAPACITY_IDX, alloc_size)]));
        cl_set_zero_sync(unsafe { base_cl.add(1) });
        cl_set_zero_sync(unsafe { base_cl.add(2) });
        CXLHashMapBucket {
            base_cl,
            alloc_size_bytes: alloc_size as u64,
            elem_counter: 0,
            alloc,
        }
    }

    pub fn from_cl_sync(base_cl: *mut Cacheline, alloc: Arc<A>) -> Option<Self> {
        cxl_typed_from_cl_sync::<Self>(base_cl, |meta| CXLHashMapBucket {
            base_cl,
            alloc_size_bytes: meta.d[BUCKET_CAPACITY_IDX],
            elem_counter: meta.d[BUCKET_COUNTER_IDX],
            alloc,
        })
    }

    ///Inserts the given Elment into CXL shared memory
    pub fn insert_sync(&mut self, key: u64, value: u64) {
        let meta = cxl_read_cacheline_sync(self.base_cl);
        self.elem_counter = meta.d[BUCKET_COUNTER_IDX];
        // This can hold 4 pairs of key+value per Cacheline.
        let free_space_cl = unsafe { self.get_data_start().add((self.elem_counter / 4) as usize) };
        // Read the one that contains the next empty slot
        let mut cl = cxl_read_cacheline_sync(free_space_cl);
        // Calc the index of the next empty slot
        let kv_pair = (self.elem_counter % 4) as usize * 2;
        // Update the target values
        cl.d[kv_pair] = key;
        cl.d[kv_pair + 1] = value;
        // Write back
        cxl_write_cl_sync(free_space_cl, &cl);
        // Update elem counter
        self.elem_counter = self.elem_counter.add(1);

        /*
        print!(
            "[{}] bucket::insert(k:{key} v:{value}) -> n:{} d:",
            Cacheline::ptr_to_offset(self.base_cl),
            self.elem_counter,
        );
        let elem_ptr = self.get_data_start() as *const u64;
        for i in (0..(self.elem_counter * 2)).step_by(2) {
            let key = unsafe { *elem_ptr.add(i as usize) };
            let value = unsafe { *elem_ptr.add((i + 1) as usize) };
            print!(" ({:?}, {:?})", key, value);
        }
        println!("");
        */

        // Update shared state
        cxl_write_sync!((self.base_cl, BUCKET_COUNTER_IDX, self.elem_counter));
    }

    pub fn invalidate(&self) {
        range_invalidate(
            self.get_data_start(),
            (self.elem_counter.div_ceil(4)) as usize,
        );
    }

    /// This function caches read values from CXL shared memory.
    pub fn get_values(&self, key: u64) -> Vec<(u64, u64)> {
        /*
        print!(
            "[{}] bucket::get_values(key:{key}) -> n:{} d:",
            Cacheline::ptr_to_offset(self.base_cl),
            self.elem_counter
        );
        let elem_ptr = self.get_data_start() as *const u64;
        for i in (0..(self.elem_counter * 2)).step_by(2) {
            let key = unsafe { *elem_ptr.add(i as usize) };
            let value = unsafe { *elem_ptr.add((i + 1) as usize) };
            print!(" ({:?}, {:?})", key, value);
        }
        println!("");
        */

        let data_ptr = self.get_data_start();
        let mut result_vec = vec![];
        // Iterate keys
        let mut kv_pairs_visited = 0;
        for cl_counter in 0..self.elem_counter.div_ceil(4) {
            let cl: Cacheline = unsafe { *(data_ptr.add(cl_counter as usize)) };
            trace!("bucket::get_values(key:{key}) -> {cl:?}");
            for kv_pair_idx in (0..8).step_by(2) {
                if cl.d[kv_pair_idx] == key {
                    result_vec.push((key, cl.d[kv_pair_idx + 1]));
                }
                kv_pairs_visited = kv_pairs_visited + 1;
                if kv_pairs_visited == self.elem_counter {
                    trace!(
                        "bucket::get_values(key:{key}) -> produced: {}",
                        self.elem_counter,
                    );
                    return result_vec;
                }
            }
        }
        trace!(
            "bucket::get_values(k:{key}) -> produced: {}",
            self.elem_counter,
        );
        result_vec
    }

    pub fn remove(&self, key: u64) {
        unimplemented!()
    }
}

impl<A> CXLSharedState for CXLHashMapBucket<A>
where
    A: Allocator,
{
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        unimplemented!()
    }
    fn get_cxl_type_id() -> u64 {
        0xb00b132
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        allocator::CXL_ALLOC_MIN_SIZE,
        allocator::bitmap::CXLBitMapAlloc,
        core::host::{MAX_HOSTS, set_machine_id},
        utils::open_test_cxl_dev,
    };
    use log::*;
    use std::alloc::Layout;

    #[test]
    fn base_sys_alloc() {
        let _ = env_logger::try_init();

        let block_size = CXL_ALLOC_MIN_SIZE;
        let mem_size = block_size * 512 * 3;
        let mem = unsafe { std::alloc::alloc(Layout::from_size_align(mem_size, 64).unwrap()) };
        info!("{mem:p} mem start");

        let buckets = 32;
        let bucket_default_capacity = 1024;
        let mut hm = CXLHashMap::try_new(
            buckets,
            bucket_default_capacity,
            MAX_HOSTS,
            Arc::new(std::alloc::System),
        )
        .unwrap();
        info!("{hm:?}");

        hm.insert_sync(1, 10);
        hm.insert_sync(1, 11);
        hm.insert_sync(2, 12);
        hm.insert_sync(3, 13);
        hm.insert_sync(1, 14);
        hm.insert_sync(1, 15);
        hm.insert_sync(1, 16);

        let k = hm.get_values(2);
        assert_eq!(k.len(), 1);

        let k = hm.get_values(3);
        assert_eq!(k.len(), 1);

        let keys = hm.get_values(1);
        assert_eq!(keys.len(), 5);
        debug!("{keys:?}");
    }

    #[test]
    fn base_cxl_alloc() {
        set_machine_id(1);
        let (mem_base, size) = open_test_cxl_dev();
        let cxl_alloc =
            CXLBitMapAlloc::try_new(mem_base, CXL_ALLOC_MIN_SIZE, size, MAX_HOSTS).unwrap();

        let buckets = 32;
        let bucket_default_capacity = 256;
        let mut hm = CXLHashMap::try_new(
            buckets,
            bucket_default_capacity,
            MAX_HOSTS,
            Arc::new(cxl_alloc),
        )
        .unwrap();
        info!("{hm:?}");

        hm.insert_sync(1, 10);
        hm.insert_sync(1, 11);
        hm.insert_sync(2, 12);
        hm.insert_sync(3, 13);
        hm.insert_sync(1, 14);
        hm.insert_sync(1, 15);
        hm.insert_sync(1, 16);

        let k = hm.get_values(2);
        assert_eq!(k.len(), 1);

        let k = hm.get_values(3);
        assert_eq!(k.len(), 1);

        let keys = hm.get_values(1);
        debug!("{keys:?}");
        assert_eq!(keys.len(), 5);
    }
}
