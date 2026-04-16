//! cartan-gpu: portable GPU compute primitives for the cartan ecosystem.
//!
//! See the crate-level docs for the architecture overview. This crate is
//! pre-release; the public API will stabilize once cartan-em v0.1 ships.

pub mod error;
pub mod device;

pub use device::Device;
pub use error::GpuError;
