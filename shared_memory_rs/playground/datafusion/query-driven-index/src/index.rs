use std::collections::HashMap;
use std::path::Path;

use shm_cxl::{
    cxl_write,
    l0_api::cxl_mutex::{CXLMutex, PetersonLock},
    l0_api::cxl_primitives::{Cacheline, cxl_read_cacheline},
    utils::{open_test_cxl_dev, two_host_sync_barrier},
};

// ────────────────────────────────────────────────────────────────────
// DataFusion Index
// ────────────────────────────────────────────────────────────────────

///
#[derive(Debug)]
pub struct Index {
    cxl: CXLSharedMemoryDataStruct,
    pub entries: HashMap<u64, u64>,
}

impl Index {
    pub fn init(path: impl AsRef<Path>, len: usize) -> Self {
        // Try to connec to a shared index
        let cxl_mem = open_test_cxl_dev();

        let id = two_host_sync_barrier(cxl_mem.0);

        Index {
            cxl: CXLSharedMemoryDataStruct::init(cxl_mem.0),
            entries: HashMap::new(),
        }
    }

    pub fn lookup(&mut self, value: u64) -> Option<Vec<u64>> {
        match cxl_flat_hash_map_get(&mut self.cxl, value) {
            None => None,
            Some(v) => Some(vec![v]),
        }
        //self.entries.get(value)
    }

    pub fn insert(&mut self, k: u64, v: u64) {
        cxl_flat_hash_map_insert(&mut self.cxl, k, v);
    }
}

// ────────────────────────────────────────────────────────────────────
// Shared Memory Data structure
// ────────────────────────────────────────────────────────────────────

/// In memory representation of where shared structures reside
#[derive(Debug)]
pub struct CXLSharedMemoryDataStruct {
    /// We need a lock to coordinate reads/writes
    lock: PetersonLock,
    /// The flat hash map data structure
    map: *mut Cacheline,
}

impl CXLSharedMemoryDataStruct {
    pub fn init(base: *mut Cacheline) -> Self {
        // First three cachelines are used by the lock
        let lock_ptr = base;
        let l = PetersonLock::init(lock_ptr).unwrap();

        // Rest is free to grab by the flat HashMap
        let flat_hash_map_ptr = unsafe { lock_ptr.offset(3) }; // TODO make sure this is correct!

        let mut ret = CXLSharedMemoryDataStruct {
            lock: l,
            map: flat_hash_map_ptr,
        };

        cxl_flat_hash_map_init(&mut ret, 20);

        ret
    }

    //
    pub fn lock(&mut self) {
        self.lock.lock();
    }
    pub fn unlock(&mut self) {
        self.lock.unlock();
    }
}
unsafe impl Send for CXLSharedMemoryDataStruct {}
unsafe impl Sync for CXLSharedMemoryDataStruct {}

// ────────────────────────────────────────────────────────────────────
// Flat HashMap that can be placed in CXL shared memory
// ────────────────────────────────────────────────────────────────────

const EMPTY: u64 = u64::MAX;
const TOMBSTONE: u64 = u64::MAX - 1;

/// Header
#[repr(C, align(64))]
pub struct CXLFlatHashMapHeader {
    capacity: u64,
    _pad: [u64; 7],
}

/// A single slot in the hash table fits in one cacheline.
#[repr(C, align(64))]
pub struct CXLFlatHashMapSlot {
    key: u64,
    value: u64,
    _pad: [u64; 6],
}

pub fn cxl_flat_hash_map_get_capactiy(ptr: &CXLSharedMemoryDataStruct) -> u64 {
    let m = ptr.map as *mut CXLFlatHashMapHeader;
    unsafe { (*m).capacity }
}
pub fn cxl_flat_hash_map_get_slot_ptr(ptr: &CXLSharedMemoryDataStruct) -> *mut Cacheline {
    unsafe { ptr.map.offset(1) }
}

/// Initialize the map in already-mmap'd memory.
pub fn cxl_flat_hash_map_init(ptr: &mut CXLSharedMemoryDataStruct, capacity: usize) {
    //
    ptr.lock();

    cxl_write!((ptr.map, capacity as u64));
    for i in 0..capacity {
        cxl_write!((ptr.map.offset(i as isize + 1), EMPTY));
    }

    ptr.unlock();
}

/// Insert or update. Returns false if table is full.
pub fn cxl_flat_hash_map_insert(ptr: &mut CXLSharedMemoryDataStruct, key: u64, value: u64) -> bool {
    assert!(key != EMPTY && key != TOMBSTONE, "key is reserved");
    let cap = cxl_flat_hash_map_get_capactiy(ptr);
    let idx = hash(key) & (cap - 1);

    let slot = cxl_flat_hash_map_get_slot_ptr(ptr);

    let slot_addr = unsafe { slot.offset(idx as isize) };

    ptr.lock();
    let v = cxl_read_cacheline(slot_addr).d[0];
    let ret = if v == EMPTY {
        cxl_write!((slot_addr => [(0, key), (1, value)]));
        true
    } else {
        // we ignore colisions for now!
        // TODO handle this situation
        false
    };
    ptr.unlock();
    ret
}

/// Returns the value if found.
pub fn cxl_flat_hash_map_get(ptr: &mut CXLSharedMemoryDataStruct, key: u64) -> Option<u64> {
    let cap = cxl_flat_hash_map_get_capactiy(ptr);
    let idx = hash(key) & (cap - 1);

    let slot = cxl_flat_hash_map_get_slot_ptr(ptr);
    let slot_addr = unsafe { slot.offset(idx as isize) };

    ptr.lock();
    let v = cxl_read_cacheline(slot_addr);
    let ret = if v.d[0] != key { None } else { Some(v.d[1]) };
    ptr.unlock();
    ret
}

/// Removes a key. Uses tombstone so probing chains stay intact.
pub fn remove(d: &CXLSharedMemoryDataStruct, key: u64) -> bool {
    let cap = cxl_flat_hash_map_get_capactiy(d);
    let _idx = hash(key) & (cap - 1);

    // TODO

    false
}

fn hash(key: u64) -> u64 {
    // Fibonacci hashing
    key.wrapping_mul(0x9e3779b97f4a7c15)
}
