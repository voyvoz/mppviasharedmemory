pub use llfree::TreesLock;

use crate::core::Cacheline;

/// Mutex is a lock that guarantees mutual exclusion when used on CXL shared memory.
/// lock/unlock are blocking.
pub trait CXLMutex: TreesLock {
    /// By default all CXL shared memory locks will use consective cacheline starting from the given base_ptr.
    fn init(base_ptr: *mut Cacheline) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
}

/// Allows creating a guard object that locks on creation
/// and unlocks on destroy.
pub struct CXLMutexGuard<'a, L: TreesLock> {
    lock: &'a L,
}
impl<'a, L: TreesLock> CXLMutexGuard<'a, L> {
    pub fn new(lock: &'a L) -> Self {
        lock.lock();
        CXLMutexGuard { lock }
    }
}
impl<'a, L: TreesLock> Drop for CXLMutexGuard<'a, L> {
    fn drop(&mut self) {
        self.lock.unlock();
    }
}

/// A convenience Feature
pub trait CXLMutexGuarded: TreesLock + Sized {
    fn lock_guard(&self) -> CXLMutexGuard<'_, Self> {
        CXLMutexGuard::new(self)
    }
}
impl<L: TreesLock> CXLMutexGuarded for L {}
