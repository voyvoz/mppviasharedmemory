pub mod ipc;
pub mod mpi;
pub mod opts;
pub mod pipe;

pub use opts::ComOptions;

// Implements MPI primitives
pub use mpi::*;
