use crate::core::{Cacheline, cl_set_zero_sync, cxl_read_cacheline_sync, cxl_write_cl_sync};
use crate::cxl::{CXLSharedState, cxl_typed_from_cl_sync, cxl_typed_new_sync};
use crate::cxl_write_sync;
use crate::mutex::CXLMutexImpl;
use allocator_api2::alloc::Allocator;
use log::trace;
use std::{
    ops::{BitAnd, BitOr},
    ptr::NonNull,
};

const BLOCK_SIZE_IDX: usize = 7;
const MEM_SIZE_IDX: usize = 6;
const ALLOC_COUNTER_IDX: usize = 5;

/// A CXL global Block allocator
/// Bitmap of free blocks.
/// Small improvment: keep track of local free blocks and use that to serve the next request.
#[derive(Debug)]
pub struct CXLBitMapAlloc {
    pub base_cl: *mut Cacheline,

    // Local constants
    // ---------------
    // Params
    pub block_size: usize,
    pub memory_size: usize,

    // Number of cachelines available for this alloc
    bit_map_cls: usize,
    // Index of the cacheline where we stopped for our last alloc
    last_bit_map_cl_idx: usize,

    // Remote state
    // ------------
    lock: CXLMutexImpl,

    //
    bit_map_start: *mut Cacheline,

    //
    free_memory_start: *mut Cacheline,
}

//TODO maybe implement copy trait?

impl CXLBitMapAlloc {
    ///
    pub fn try_new(
        base_cl: *mut Cacheline,
        block_size: usize,
        memory_size: usize,
        hosts: usize,
    ) -> Result<Self, allocator_api2::alloc::AllocError> {
        // Meta
        cxl_typed_new_sync::<Self>(base_cl);
        cxl_write_sync!( (base_cl=>[(ALLOC_COUNTER_IDX, 0), (MEM_SIZE_IDX, memory_size), (BLOCK_SIZE_IDX, block_size)]) );
        let mut next_cl = unsafe { base_cl.add(1) };
        // Locks
        let lock = CXLMutexImpl::init(next_cl, hosts);
        next_cl = unsafe { next_cl.add(lock.required_cxl_shared_state_cachelines()) };

        let bit_map_start_cl = next_cl;
        // How many blocks we could alloc
        let bit_map_entries = memory_size.div_ceil(block_size);
        // How many cachelines we need to track allocs.
        // We need require that each cacheline needs to be full so we
        // drop the last if its non empty.
        let bit_map_cls = bit_map_entries.div_ceil((64 * 8) - 1);
        if bit_map_cls == 0 {
            return Err(allocator_api2::alloc::AllocError);
        }
        // Clear the bitmap
        for i in 0..bit_map_cls {
            cl_set_zero_sync(unsafe { bit_map_start_cl.add(i) });
        }
        next_cl = unsafe { next_cl.add(bit_map_cls) };

        // Free Mem
        Ok(CXLBitMapAlloc {
            base_cl,
            block_size,
            memory_size,
            bit_map_cls,
            lock,
            last_bit_map_cl_idx: 0,
            bit_map_start: bit_map_start_cl,
            free_memory_start: next_cl,
        })
    }

    pub fn from_cl(base_cl: *mut Cacheline) -> Option<Self> {
        cxl_typed_from_cl_sync(base_cl, |meta| {
            let block_size = meta.d[BLOCK_SIZE_IDX] as usize;
            let memory_size = meta.d[MEM_SIZE_IDX] as usize;

            let mut next_cl = unsafe { base_cl.add(1) };
            let lock = CXLMutexImpl::from_cl_sync(next_cl).unwrap();

            let bit_map_cls = memory_size.div_ceil(block_size).div_ceil((64 * 8) - 1);
            next_cl = unsafe { next_cl.add(lock.required_cxl_shared_state_cachelines()) };

            let free_memory_start = unsafe { next_cl.add(bit_map_cls) };

            CXLBitMapAlloc {
                base_cl,
                block_size,
                memory_size,
                bit_map_cls,
                last_bit_map_cl_idx: 0,
                lock,
                bit_map_start: next_cl,
                free_memory_start,
            }
        })
    }

    fn create_alloc_mask(start_bit_idx: usize, bit_count: usize) -> u64 {
        match bit_count >= 64 {
            true => u64::MAX,
            false => ((1 << bit_count) - 1) << start_bit_idx,
        }
    }

    //
    fn reserve_blocks(
        &self,
        cl: *mut Cacheline,
        cl_v: &mut Cacheline,
        cl_idx: usize,
        u64_idx: usize,
        u64_bit_offset: usize,
        requested_blocks: usize,
    ) -> *mut Cacheline {
        let global_start_bit_idx = (cl_idx * 512) + (u64_idx * 64) + u64_bit_offset;
        // Create Mask & combine with current cl then store that change
        let mask = CXLBitMapAlloc::create_alloc_mask(u64_bit_offset, requested_blocks);
        trace!("reserve_blocks: mask => {mask:#b}");
        cl_v.d[u64_idx] = cl_v.d[u64_idx].bitor(mask);
        cxl_write_cl_sync(cl, cl_v);

        let reserved_blocks = mask.count_ones() as usize;

        // handle case where more bits are needed
        let remainig_blocks = requested_blocks - reserved_blocks;
        if remainig_blocks > 0 {
            // We need to alloc on the subsequent cachelines as well
            unimplemented!()
        }

        // Calc cachline offset and return addr of allocation
        let addr_byte_offset = global_start_bit_idx * self.block_size;
        let res = unsafe { self.free_memory_start.add(addr_byte_offset / 64) };

        trace!(
            "reserve_blocks: glob_start_bit_idx:{global_start_bit_idx} cl_idx:{cl_idx}[u64_idx:{u64_idx}][bit_offset:{u64_bit_offset}] -> {res:p} {requested_blocks}"
        );

        let meta = cxl_read_cacheline_sync(self.base_cl);
        cxl_write_sync!((
            self.base_cl,
            ALLOC_COUNTER_IDX,
            meta.d[ALLOC_COUNTER_IDX] + 1
        ));

        res
    }

    /// Tries to Allocate at least `bytes` amount of consecutive memory.
    /// Overfits allocation based on the configured block size.
    pub fn try_alloc(&self, bytes: usize) -> Result<*mut [u8], allocator_api2::alloc::AllocError> {
        self.lock.lock();

        if bytes > 64 * self.block_size {
            panic!("current version cant alloc more than 64*self.block_size at once");
        }

        // This is equal to the number of consecutive bits we need for the alloc
        let requested_blocks = bytes.div_ceil(self.block_size);

        let mut consecutive_bits = 0;
        let cacheline_idx = self.last_bit_map_cl_idx;

        trace!(
            "try_alloc(bytes:{bytes}) -> bit_map_start:{:p} requested_blocks:{requested_blocks}",
            self.bit_map_start
        );

        // Iterate through cachelines
        for cl_idx in cacheline_idx..self.bit_map_cls {
            let cl_addr = unsafe { self.bit_map_start.add(cl_idx) };
            let mut cl_v = cxl_read_cacheline_sync(cl_addr);
            trace!("try_alloc cl[{cl_idx}]:{cl_addr:p} -> {cl_v:?}");
            for u64_idx in 0..8 {
                let v = cl_v.d[u64_idx];
                match v {
                    // All used
                    0xFFFFFFFFFFFFFFFF => {
                        trace!("cl[{u64_idx}] == u64::MAX");
                        consecutive_bits = 0;
                        continue;
                    }
                    // All bits are free
                    0 => {
                        trace!("cl[{u64_idx}] == 0");
                        consecutive_bits = consecutive_bits + 64;
                        if consecutive_bits >= requested_blocks {
                            let mem_addr = self.reserve_blocks(
                                cl_addr,
                                &mut cl_v,
                                cl_idx,
                                u64_idx,
                                0,
                                requested_blocks,
                            );
                            //self.last_bit_map_cl_idx = cl_idx;
                            self.lock.unlock();
                            return Ok(std::ptr::slice_from_raw_parts_mut(
                                mem_addr as *mut u8,
                                requested_blocks * self.block_size,
                            ));
                        }
                    }
                    // Some
                    _ => {
                        trace!("cl[{u64_idx}] -> is:{v:#b} could have space");
                        if (v.count_zeros() as usize) >= requested_blocks {
                            let mask = CXLBitMapAlloc::create_alloc_mask(0, requested_blocks);
                            for i in 0..(64 - requested_blocks) {
                                if v & (mask << i) == 0 {
                                    let mem_addr = self.reserve_blocks(
                                        cl_addr,
                                        &mut cl_v,
                                        cl_idx,
                                        u64_idx,
                                        i,
                                        requested_blocks,
                                    );
                                    //self.last_bit_map_cl_idx = start_idx as usize+requested_blocks;
                                    self.lock.unlock();
                                    return Ok(std::ptr::slice_from_raw_parts_mut(
                                        mem_addr as *mut u8,
                                        requested_blocks * self.block_size,
                                    ));
                                }
                            }
                        }
                        trace!("cl[{u64_idx}] failed to find enough consecutive bits ");
                        consecutive_bits = 0;
                        continue;
                    }
                }
            }
        }

        self.lock.unlock();
        Err(allocator_api2::alloc::AllocError)
    }

    fn free_blocks(&self, ptr: *mut [u8]) {
        self.lock.lock();

        let offset = unsafe {
            (ptr.as_ref().unwrap().as_ptr() as usize) - (self.free_memory_start as usize)
        };
        let cl_bitmap_idx = offset / (512 * self.block_size);
        let bitmap_cl_u64_idx = cl_bitmap_idx % 512 / 64;
        let bitmap_cl_u64_startbit_idx = offset / self.block_size;
        let consecutive_bits = ptr.len() / self.block_size;

        let mask = CXLBitMapAlloc::create_alloc_mask(bitmap_cl_u64_startbit_idx, consecutive_bits);
        let mask_flipped = mask ^ u64::MAX;

        trace!(
            "free({ptr:p}) {offset} -> {cl_bitmap_idx}[{bitmap_cl_u64_idx}][{bitmap_cl_u64_startbit_idx}..{}] mask:{mask:#b} block_size:{}",
            bitmap_cl_u64_startbit_idx + consecutive_bits,
            self.block_size
        );

        let cl_addr = unsafe { self.bit_map_start.add(cl_bitmap_idx) };
        let mut cl = cxl_read_cacheline_sync(cl_addr);
        for i in bitmap_cl_u64_idx..8 {
            cl.d[i] = cl.d[i].bitand(mask_flipped);
        }
        cxl_write_cl_sync(cl_addr, &cl);

        self.lock.unlock();
    }

    /// returns how many bytes are currently allocated.
    pub fn calc_used_space(&self) -> usize {
        self.lock.lock();

        let mut res = 0;

        for i in 0..self.bit_map_cls {
            let cl_addr = unsafe { self.bit_map_start.add(i) };
            let v = cxl_read_cacheline_sync(cl_addr);
            for j in 0..8 {
                res = res + (v.d[j].count_ones() as usize * self.block_size);
            }
        }

        self.lock.unlock();

        res
    }

    pub fn get_owned_memory_size(&self) -> usize {
        self.bit_map_cls * 512 * self.block_size
    }
}

impl CXLSharedState for CXLBitMapAlloc {
    fn required_cxl_shared_state_cachelines_static() -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines_n(_n: usize) -> usize {
        unimplemented!()
    }
    fn required_cxl_shared_state_cachelines(&self) -> usize {
        (self.free_memory_start as usize - self.bit_map_start as usize) / 64
    }
    fn get_cxl_type_id() -> u64 {
        0x987654321
    }
}

unsafe impl Allocator for CXLBitMapAlloc {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, allocator_api2::alloc::AllocError> {
        match self.try_alloc(layout.size()) {
            Err(e) => Err(e),
            Ok(v) => Ok(NonNull::new(v).unwrap()),
        }
    }
    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        self.free_blocks(std::ptr::slice_from_raw_parts_mut::<u8>(
            ptr.as_ptr(),
            layout.size(),
        ));
    }
}

//##########################
//  Tests
//#########################

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::CXL_ALLOC_MIN_SIZE;
    use log::*;
    use std::{alloc::Layout, collections::HashSet};

    #[test]
    fn masks() {
        let mut m = CXLBitMapAlloc::create_alloc_mask(0, 64);
        assert_eq!(m, u64::MAX);

        m = CXLBitMapAlloc::create_alloc_mask(31, 32);
        assert_eq!(m, ((1u64 << 32) - 1));

        m = CXLBitMapAlloc::create_alloc_mask(1, 2);
        assert_eq!(m, 0b0110);
    }

    #[test]
    fn base() {
        let _ = env_logger::try_init();

        let block_size = CXL_ALLOC_MIN_SIZE;
        let mem_size = block_size * 512 * 3;
        let mem = unsafe { std::alloc::alloc(Layout::from_size_align(mem_size, 64).unwrap()) };
        info!("{mem:?}");

        let alloc =
            CXLBitMapAlloc::try_new(mem as *mut Cacheline, block_size, mem_size, 4).unwrap();
        info!("{alloc:?}");

        let first_8_bytes = alloc.try_alloc(8).unwrap();
        info!("first_8_bytes:{first_8_bytes:?}");
        let slice_ref: &mut [u8] = unsafe { first_8_bytes.as_mut().unwrap() };
        assert_eq!(block_size, first_8_bytes.len());
        for i in slice_ref {
            *i = 1;
        }

        let next_32_bytes = alloc.try_alloc(32).unwrap();
        info!("next_32_bytes:{next_32_bytes:?}");
        assert_ne!(first_8_bytes, next_32_bytes);
        let slice_ref: &mut [u8] = unsafe { next_32_bytes.as_mut().unwrap() };
        assert_eq!(block_size, next_32_bytes.len());
        for i in slice_ref {
            *i = 1;
        }
        assert_ne!(first_8_bytes, next_32_bytes);

        alloc.free_blocks(next_32_bytes);

        alloc.free_blocks(first_8_bytes);

        assert_eq!(alloc.calc_used_space(), 0);

        let first_8_bytes_again = alloc.try_alloc(8).unwrap();
        info!("{first_8_bytes_again:?}");
        let slice_ref: &mut [u8] = unsafe { first_8_bytes_again.as_mut().unwrap() };
        assert_eq!(block_size, first_8_bytes_again.len());
        for i in slice_ref {
            *i = 1;
        }
        assert_ne!(alloc.calc_used_space(), 0);

        alloc.free_blocks(first_8_bytes_again);

        assert_eq!(alloc.calc_used_space(), 0);

        info!("---");
        // Ensure that consecutive alloc get different addresses when no free has been called
        let mut allocs: HashSet<usize> = HashSet::new();
        for _ in 0..10 {
            let res = alloc.try_alloc(1).unwrap();
            assert!(!allocs.contains(&res.addr()));
            allocs.insert(res.addr());
        }
    }
}
