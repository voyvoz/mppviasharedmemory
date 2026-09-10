pub const CACHELINE_SIZE: usize = 64;

/// Should be created from a pointer created from mmap.
///
/// Once a cacheline is loaded even when just the first byte was accessed the whole line should be present in the cache.
/// So its safe to access any byte when any load has fetched only parts of the line.
#[repr(C, align(64))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Cacheline {
    pub d: [u64; 8],
}

impl Cacheline {
    pub fn from_b0(b0: u64) -> Self {
        Cacheline {
            d: [b0, 0, 0, 0, 0, 0, 0, 0],
        }
    }
    pub fn empty() -> Self {
        Cacheline {
            d: [0, 0, 0, 0, 0, 0, 0, 0],
        }
    }

    pub fn print_hex(c: *mut Cacheline) {
        print!("{c:p}: ");
        let v = unsafe { c.as_ref().unwrap() };
        for i in 0..8 {
            print!("{:#010x} ", v.d[i]);
        }
        println!("");
    }

    pub fn ptr_to_offset(c: *mut Cacheline) -> ShmOffset {
        cl_to_offset(c)
    }
    pub fn ptr_from_offset(o: ShmOffset) -> *mut Cacheline {
        offset_to_cl(o)
    }
    /// Calculates the containing cacheline of the given pointer
    pub fn to_containing_cl(ptr: *mut u8) -> *mut Cacheline {
        (ptr as usize & !(CACHELINE_SIZE - 1)) as *mut Cacheline
    }
}

// ---------------------------------------------------------------------
// Global offset is used to determine correct pointers from CXL offsets
// ---------------------------------------------------------------------

use std::sync::OnceLock;

pub type ShmOffset = u64;

static GLOBAL_OFFSET_BASE: OnceLock<ShmOffset> = OnceLock::new();
pub fn set_global_offset(offset_base: ShmOffset) {
    log::trace!("set_global_offset(offset_base={offset_base:x})");
    GLOBAL_OFFSET_BASE
        .set(offset_base)
        .expect("GLOBAL_OFFSET_BASE can be set only once");
}
fn cl_to_offset(cl: *mut Cacheline) -> ShmOffset {
    (cl as u64 - GLOBAL_OFFSET_BASE.get().unwrap()) / 64
}
fn offset_to_cl(offset: ShmOffset) -> *mut Cacheline {
    unsafe { (*GLOBAL_OFFSET_BASE.get().unwrap() as *mut Cacheline).add(offset as usize) }
}
