//! CUDA backend plumbing — minimal `CudaDevice` analog of the wgpu-backed
//! `Device`. Driver API only; no kernels, no FFT yet. The intent is to have a
//! seam where future CUDA-native work (cuFFT, cuBLAS, custom kernels) can land
//! without further infrastructure churn.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::GpuError;

/// Owned CUDA context bound to one device.
///
/// Mirrors the role of [`crate::Device`] for the Vulkan/wgpu path. Held for
/// the lifetime of a CUDA-side cartan-gpu session; clones share the same
/// underlying CUDA context via `Arc`.
#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaDevice {
    /// Open ordinal 0 (the first CUDA device the driver reports).
    ///
    /// Returns [`GpuError::NoAdapter`] when no CUDA devices are visible to
    /// the driver, and [`GpuError::CudaError`] for any other driver-level
    /// failure (load failure, init failure, no permission, etc.).
    pub fn new() -> Result<Self, GpuError> {
        Self::with_ordinal(0)
    }

    /// Open the device at the given ordinal. Useful when the host has more
    /// than one CUDA device.
    pub fn with_ordinal(ordinal: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(ordinal).map_err(map_cuda_err)?;
        Ok(Self { ctx })
    }

    /// Borrow the underlying `cudarc::driver::CudaContext`.
    ///
    /// Exposed so downstream code can build streams, modules, and buffers
    /// against the same context without re-initialising the driver.
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Human-readable device name (e.g. `"NVIDIA GeForce RTX 5060"`).
    pub fn name(&self) -> Result<String, GpuError> {
        self.ctx.name().map_err(map_cuda_err)
    }

    /// `(major, minor)` SM compute capability.
    pub fn compute_capability(&self) -> Result<(i32, i32), GpuError> {
        self.ctx.compute_capability().map_err(map_cuda_err)
    }

    /// Total device memory in bytes.
    pub fn total_memory_bytes(&self) -> Result<usize, GpuError> {
        self.ctx.total_mem().map_err(map_cuda_err)
    }
}

fn map_cuda_err<E: std::fmt::Display>(e: E) -> GpuError {
    GpuError::CudaError(e.to_string())
}
