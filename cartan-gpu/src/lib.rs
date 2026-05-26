//! cartan-gpu: portable wgpu-based GPU compute primitives for the cartan ecosystem.
//!
//! v0.6 split: cartan-gpu provides Device + GpuBuffer + Kernel (wgpu compute);
//! FFT lives in the separate `gpufft` crate. With the `fft` feature, gpufft
//! is re-exported here for convenience.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod buffer;
pub mod device;
pub mod error;
pub mod kernel;

pub use buffer::GpuBuffer;
pub use device::Device;
pub use error::GpuError;
pub use kernel::Kernel;

pub use wgpu;

#[cfg(feature = "fft")]
#[cfg_attr(docsrs, doc(cfg(feature = "fft")))]
pub use gpufft;
