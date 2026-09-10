use crate::core::Cacheline;

pub const CXL_ALLOC_MIN_SIZE: usize = 2 << 20;

pub trait CXLAlloc {
    // Returns the base pointer
    fn get_base(&self) -> *mut Cacheline;
}
