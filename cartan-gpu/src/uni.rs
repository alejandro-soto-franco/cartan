//! Runtime backend selection for the FFT path.
//!
//! [`UniFftBackend`] and [`UniBuffer`] are enums that hold either the
//! Vulkan or the CUDA backend, depending on which features are enabled
//! and which backend the caller constructs at session start. Both
//! implement (or pair with) the [`crate::Fft`] trait so user code is
//! identical regardless of backend.
//!
//! Memory is currently backend-local: a `UniBuffer::Vulkan` is wgpu-side
//! storage, a `UniBuffer::Cuda` is cudarc-side storage. True
//! zero-copy memory sharing across both backends (via
//! `VK_KHR_external_memory_fd` + `cudarc::CudaContext::import_external_memory`)
//! is feasible — wgpu-hal 29 enables the extension — and tracked as
//! follow-up work, but not implemented here.

#![cfg(any(feature = "vkfft", feature = "cufft"))]

use num_complex::Complex32;

use crate::{Fft, FftDirection, GpuError};

#[cfg(feature = "vkfft")]
use crate::{Device, GpuBuffer, VkFftBackend};

#[cfg(feature = "cufft")]
use crate::{CuFftBackend, CudaBuffer, CudaDevice};

/// Runtime-selectable FFT backend.
///
/// Construct with [`UniFftBackend::vulkan`] or [`UniFftBackend::cuda`]
/// once at session start, then write call sites against the unified
/// [`crate::Fft`] trait.
///
/// The Vulkan variant is substantially larger than the CUDA one
/// (VkFftBackend holds many ash handles + a HashMap of plans, each plan
/// dragging a VkFFTApplication struct). The size delta is not perf-
/// relevant in practice: the enum is constructed once per session and
/// held by reference thereafter.
#[allow(clippy::large_enum_variant)]
pub enum UniFftBackend {
    #[cfg(feature = "vkfft")]
    Vulkan(VkFftBackend),
    #[cfg(feature = "cufft")]
    Cuda(CuFftBackend),
}

impl UniFftBackend {
    /// Build a Vulkan-backed engine from an existing `Device`.
    #[cfg(feature = "vkfft")]
    pub fn vulkan(dev: &Device) -> Result<Self, GpuError> {
        Ok(Self::Vulkan(VkFftBackend::new(dev)?))
    }

    /// Build a CUDA-backed engine from an existing `CudaDevice`.
    #[cfg(feature = "cufft")]
    pub fn cuda(dev: &CudaDevice) -> Result<Self, GpuError> {
        Ok(Self::Cuda(CuFftBackend::new(dev)?))
    }
}

/// Runtime-selectable buffer matching a [`UniFftBackend`].
///
/// Always construct via [`UniBuffer::from_slice`] passing the same
/// backend you intend to run the FFT on — the variant is locked to the
/// backend at allocation time, and mismatched calls return
/// [`GpuError::BackendMismatch`] at runtime.
#[allow(clippy::large_enum_variant)]
pub enum UniBuffer {
    #[cfg(feature = "vkfft")]
    Vulkan(GpuBuffer<Complex32>),
    #[cfg(feature = "cufft")]
    Cuda(CudaBuffer),
}

impl UniBuffer {
    /// Allocate a backend-matched buffer and upload `host`.
    pub fn from_slice(engine: &UniFftBackend, host: &[Complex32]) -> Result<Self, GpuError> {
        match engine {
            #[cfg(feature = "vkfft")]
            UniFftBackend::Vulkan(b) => Ok(Self::Vulkan(GpuBuffer::<Complex32>::from_slice(
                b.device(),
                host,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )?)),
            #[cfg(feature = "cufft")]
            UniFftBackend::Cuda(b) => Ok(Self::Cuda(CudaBuffer::from_slice(b.device(), host)?)),
        }
    }

    /// Download to host. Routes to the underlying backend's `to_vec`.
    pub fn to_vec(&self, engine: &UniFftBackend) -> Result<Vec<Complex32>, GpuError> {
        match (self, engine) {
            #[cfg(feature = "vkfft")]
            (Self::Vulkan(b), UniFftBackend::Vulkan(e)) => b.to_vec(e.device()),
            #[cfg(feature = "cufft")]
            (Self::Cuda(b), _) => b.to_vec(),
            #[allow(unreachable_patterns)]
            _ => Err(GpuError::BackendMismatch),
        }
    }

    /// Element count (Complex32 elements).
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "vkfft")]
            Self::Vulkan(b) => b.len(),
            #[cfg(feature = "cufft")]
            Self::Cuda(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Fft for UniFftBackend {
    type Buffer = UniBuffer;

    fn fft_1d(
        &mut self,
        buf: &mut Self::Buffer,
        n: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        match (self, buf) {
            #[cfg(feature = "vkfft")]
            (Self::Vulkan(e), UniBuffer::Vulkan(b)) => e.fft_1d(b, n, batch, direction),
            #[cfg(feature = "cufft")]
            (Self::Cuda(e), UniBuffer::Cuda(b)) => e.fft_1d(b, n, batch, direction),
            #[allow(unreachable_patterns)]
            _ => Err(GpuError::BackendMismatch),
        }
    }

    fn fft_2d(
        &mut self,
        buf: &mut Self::Buffer,
        nx: u32,
        ny: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        match (self, buf) {
            #[cfg(feature = "vkfft")]
            (Self::Vulkan(e), UniBuffer::Vulkan(b)) => e.fft_2d(b, nx, ny, batch, direction),
            #[cfg(feature = "cufft")]
            (Self::Cuda(e), UniBuffer::Cuda(b)) => e.fft_2d(b, nx, ny, batch, direction),
            #[allow(unreachable_patterns)]
            _ => Err(GpuError::BackendMismatch),
        }
    }

    fn fft_3d(
        &mut self,
        buf: &mut Self::Buffer,
        nx: u32,
        ny: u32,
        nz: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        match (self, buf) {
            #[cfg(feature = "vkfft")]
            (Self::Vulkan(e), UniBuffer::Vulkan(b)) => e.fft_3d(b, nx, ny, nz, direction),
            #[cfg(feature = "cufft")]
            (Self::Cuda(e), UniBuffer::Cuda(b)) => e.fft_3d(b, nx, ny, nz, direction),
            #[allow(unreachable_patterns)]
            _ => Err(GpuError::BackendMismatch),
        }
    }
}
