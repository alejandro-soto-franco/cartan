//! cartan-gpu: portable wgpu-based GPU compute primitives for the cartan ecosystem.
//!
//! # Retired
//!
//! Frozen at 0.9.0 and no longer a workspace member. WGSL has no `f64`, so this
//! crate cannot express numerics the rest of cartan agrees at 1e-13 on, and no
//! crate in the workspace depends on it. The double-precision device path is
//! `cartan-cuda`.
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
