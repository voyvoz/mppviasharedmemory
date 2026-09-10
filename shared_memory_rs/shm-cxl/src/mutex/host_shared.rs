//! A mutex that excludes both the threads of one host and the hosts themselves.
//!
//! [`CXLMutexImpl`] serialises hosts, but it is not reentrant and knows nothing
//! about the threads inside a host: if two threads of the same host enter it
//! concurrently, the second corrupts the first's claim. Pairing it with a
//! process-local mutex funnels every host through a single contender, which is
//! also much cheaper than having every thread contend over CXL.

use crate::core::Cacheline;
use crate::cxl::CXLSharedState;
use crate::mutex::CXLMutexImpl;
use crate::mutex::common::TreesLock;
use std::sync::Mutex;

pub struct HostSharedLock {
    shared: CXLMutexImpl,
    local: Mutex<()>,
}

impl HostSharedLock {
    /// Cachelines needed for the shared lock state (as required by the library
    /// mutex for `nhosts` participants).
    pub fn required_cachelines(nhosts: usize) -> usize {
        <CXLMutexImpl as CXLSharedState>::required_cxl_shared_state_cachelines_n(nhosts)
    }

    pub fn init_shared(base: *mut Cacheline, nhosts: usize) -> Self {
        HostSharedLock {
            shared: CXLMutexImpl::init(base, nhosts),
            local: Mutex::new(()),
        }
    }

    pub fn attach(base: *mut Cacheline, _nhosts: usize) -> Self {
        HostSharedLock {
            shared: CXLMutexImpl::from_cl_sync(base).unwrap(),
            local: Mutex::new(()),
        }
    }

    pub fn lock(&self) -> HostSharedGuard<'_> {
        let g = self.local.lock().unwrap();
        self.shared.lock();
        HostSharedGuard {
            owner: self,
            _local: g,
        }
    }
}

pub struct HostSharedGuard<'a> {
    owner: &'a HostSharedLock,
    _local: std::sync::MutexGuard<'a, ()>,
}

impl<'a> Drop for HostSharedGuard<'a> {
    fn drop(&mut self) {
        self.owner.shared.unlock();
    }
}
