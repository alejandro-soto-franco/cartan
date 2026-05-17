//! Device-resident Complex32 buffer for the CUDA path.
//!
//! Wraps `cudarc::driver::CudaSlice<cufft::sys::float2>`. The public API is
//! expressed in `num_complex::Complex32` for parity with [`crate::GpuBuffer`]
//! on the Vulkan side.
//!
//! ## Layout note
//!
//! `cudarc::cufft::sys::float2` is `#[repr(C, align(8))]` because CUDA
//! requires 8-byte alignment for vectorised loads. `num_complex::Complex32`
//! is `#[repr(C)]` with 4-byte alignment. Sizes match (8 bytes), alignments
//! do not, so we cannot slice-cast between them safely. Conversions happen
//! element-wise. For the FFT scales we ship today (≤32768 elements per
//! call) the per-element cost is dominated by host↔device transfer; the
//! copy is essentially free.
//!
//! This type is FFT-specific by design — generic Complex32 device storage
//! is the only buffer shape cartan-gpu currently needs on the CUDA side.

#![cfg(feature = "cufft")]

use std::sync::Arc;

use cudarc::cufft::sys::float2;
use cudarc::driver::{CudaSlice, CudaStream};
use num_complex::Complex32;

use crate::{CudaDevice, GpuError};

// Sizes must match for the element-wise conversion to round-trip without
// data loss. Alignment intentionally not checked: see module docs.
const _: () = assert!(
    core::mem::size_of::<Complex32>() == core::mem::size_of::<float2>(),
    "Complex32 and cufft::float2 must be the same size"
);

/// Device-resident `Complex32` storage on a CUDA stream.
pub struct CudaBuffer {
    pub(crate) slice: CudaSlice<float2>,
    stream: Arc<CudaStream>,
    len: usize,
}

impl CudaBuffer {
    /// Allocate `len` zero-initialized Complex32 elements on the device.
    pub fn zeroed(dev: &CudaDevice, len: usize) -> Result<Self, GpuError> {
        let stream = dev.cuda_context().default_stream();
        let slice = stream
            .alloc_zeros::<float2>(len)
            .map_err(map_cuda_err)?;
        Ok(Self { slice, stream, len })
    }

    /// Upload a Complex32 host slice to a fresh device buffer.
    pub fn from_slice(dev: &CudaDevice, host: &[Complex32]) -> Result<Self, GpuError> {
        let stream = dev.cuda_context().default_stream();
        let host_f2: Vec<float2> =
            host.iter().map(|c| float2 { x: c.re, y: c.im }).collect();
        let slice = stream.clone_htod(&host_f2).map_err(map_cuda_err)?;
        Ok(Self { slice, stream, len: host.len() })
    }

    /// Download the buffer's contents back to the host as `Vec<Complex32>`.
    pub fn to_vec(&self) -> Result<Vec<Complex32>, GpuError> {
        let host_f2: Vec<float2> =
            self.stream.clone_dtoh(&self.slice).map_err(map_cuda_err)?;
        Ok(host_f2.iter().map(|f| Complex32::new(f.x, f.y)).collect())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

fn map_cuda_err<E: core::fmt::Display>(e: E) -> GpuError {
    GpuError::CudaError(e.to_string())
}
