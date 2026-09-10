pub mod common;
pub mod host_shared;
pub mod n_host;
pub mod opts;
pub mod two_host;

///
pub type CXLMutexImpl = n_host::BakeryLock;

pub use common::CXLMutex;
pub use common::CXLMutexGuard;
pub use common::CXLMutexGuarded;
pub use host_shared::{HostSharedGuard, HostSharedLock};
