use crate::core::host::MAX_HOSTS;
use crate::cxl::CXLSharedState;
use crate::cxl_v2::CXLSizedObject;
use crate::mutex::n_host::TournamentPetersonLock;
use crate::mutex::opts::{CXLMutexVariant::Bakery, CXLMutexVariant::TournametPeterson};

// ------------------------------------------------------
// MutexOptions
// ------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum CXLMutexVariant {
    TournametPeterson,
    Bakery,
}

#[derive(Debug, Clone, Copy)]
pub struct MutexOptions {
    // How many hosts should be considered when creating the lock
    hosts: usize,
    // Which variant to use
    variant: CXLMutexVariant,
}
impl MutexOptions {
    pub fn default() -> Self {
        MutexOptions {
            hosts: MAX_HOSTS,
            variant: CXLMutexVariant::Bakery,
        }
    }
    pub fn variant(variant: CXLMutexVariant) -> Self {
        MutexOptions {
            hosts: MAX_HOSTS,
            variant,
        }
    }
}

impl CXLSizedObject for MutexOptions {
    fn required_cachelines(&self) -> usize {
        match self.variant {
            TournametPeterson => {
                TournamentPetersonLock::required_cxl_shared_state_cachelines_n(self.hosts)
            }
            Bakery => 0,
        }
    }
}
