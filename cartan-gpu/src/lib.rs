//! cartan-gpu: portable GPU compute primitives for the cartan ecosystem.
//!
//! See the crate-level docs for the architecture overview. This crate is
//! pre-release; the public API will stabilize once cartan-em v0.1 ships.

pub mod buffer;
pub mod device;
pub mod error;
pub mod fft;
pub mod kernel;
#[cfg(feature = "vkfft")]
pub mod hal_vulkan;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cufft")]
pub mod cuda_buffer;
#[cfg(feature = "cufft")]
pub mod cufft;

pub use buffer::GpuBuffer;
pub use device::Device;
pub use error::GpuError;
pub use fft::{Fft, FftDirection};
pub use kernel::Kernel;
#[cfg(feature = "vkfft")]
pub use fft::VkFftBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaDevice;
#[cfg(feature = "cufft")]
pub use cuda_buffer::CudaBuffer;
#[cfg(feature = "cufft")]
pub use cufft::CuFftBackend;
