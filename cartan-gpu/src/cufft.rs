//! cuFFT-backed FFT implementation.
//!
//! Parallel to [`crate::fft::VkFftBackend`] but operating on CUDA-resident
//! [`crate::CudaBuffer`]s instead of wgpu's [`crate::GpuBuffer`].
//!
//! ## Trait split, not unification
//!
//! The Vulkan and CUDA paths cannot share a single `Fft` trait because
//! their buffer types (`GpuBuffer<Complex32>` vs `CudaBuffer`) sit in
//! disjoint memory spaces. Mirroring the API surface keeps callers
//! recognisable without forcing a generic trait that would require
//! external-memory interop just to type-check.
//!
//! ## Normalization
//!
//! cuFFT does not normalize inverse transforms. Calling
//! `fft.fft_1d(&mut buf, n, 1, Inverse)` after a Forward leaves `buf`
//! holding `N * input`; the caller scales by `1/N` if they want identity
//! semantics. This matches NVIDIA's native convention.

#![cfg(feature = "cufft")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cufft::sys::{cufftType, float2};
use cudarc::cufft::{CudaFft, FftDirection as CuFftDirection};
use cudarc::driver::{CudaSlice, CudaStream};

use crate::{CudaBuffer, CudaDevice, FftDirection, GpuError};

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct PlanKey {
    nx: i32,
    ny: i32,
    nz: i32,
    batch: i32,
    dim: u8,
}

/// cuFFT-backed FFT engine bound to a single CUDA stream.
///
/// Plans are cached by shape, mirroring [`crate::fft::VkFftBackend`].
pub struct CuFftBackend {
    stream: Arc<CudaStream>,
    plans: HashMap<PlanKey, CudaFft>,
}

impl CuFftBackend {
    /// Construct a backend on the device's default stream.
    pub fn new(dev: &CudaDevice) -> Result<Self, GpuError> {
        let stream = dev.cuda_context().default_stream();
        Ok(Self {
            stream,
            plans: HashMap::new(),
        })
    }

    fn get_or_create_plan(&mut self, key: PlanKey) -> Result<&CudaFft, GpuError> {
        if !self.plans.contains_key(&key) {
            let plan = match key.dim {
                1 => CudaFft::plan_1d(
                    key.nx,
                    cufftType::CUFFT_C2C,
                    key.batch,
                    self.stream.clone(),
                ),
                2 => CudaFft::plan_2d(
                    key.nx,
                    key.ny,
                    cufftType::CUFFT_C2C,
                    self.stream.clone(),
                ),
                3 => CudaFft::plan_3d(
                    key.nx,
                    key.ny,
                    key.nz,
                    cufftType::CUFFT_C2C,
                    self.stream.clone(),
                ),
                _ => unreachable!(),
            }
            .map_err(|e| GpuError::CudaError(format!("cuFFT plan: {e:?}")))?;
            self.plans.insert(key, plan);
        }
        Ok(self.plans.get(&key).unwrap())
    }

    fn launch(
        &mut self,
        buf: &mut CudaBuffer,
        key: PlanKey,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        let _ = self.get_or_create_plan(key)?;
        let plan = self.plans.get(&key).unwrap();

        // cuFFT's exec_c2c needs two distinct DevicePtrMut. Allocate a
        // scratch output, exec, then swap so the caller's buffer ends up
        // holding the result while the previous contents drop on `scratch`.
        let mut scratch: CudaSlice<float2> = self
            .stream
            .alloc_zeros::<float2>(buf.len())
            .map_err(|e| GpuError::CudaError(format!("cuFFT scratch alloc: {e:?}")))?;

        let cu_dir = match direction {
            FftDirection::Forward => CuFftDirection::Forward,
            FftDirection::Inverse => CuFftDirection::Inverse,
        };
        plan.exec_c2c(&mut buf.slice, &mut scratch, cu_dir)
            .map_err(|e| GpuError::CudaError(format!("cuFFT exec_c2c: {e:?}")))?;

        // Synchronize so the FFT result is visible before the caller does
        // anything else on the host (e.g. immediate to_vec).
        self.stream
            .synchronize()
            .map_err(|e| GpuError::CudaError(format!("cuFFT sync: {e:?}")))?;

        core::mem::swap(&mut buf.slice, &mut scratch);
        Ok(())
    }
}

/// FFT operations against a CUDA-resident buffer.
///
/// Parallels [`crate::fft::Fft`] for the Vulkan/wgpu side. Kept as a
/// separate trait because the buffer type cannot be unified across
/// disjoint memory spaces without external-memory interop.
pub trait CuFft {
    fn fft_1d(
        &mut self,
        buf: &mut CudaBuffer,
        n: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_2d(
        &mut self,
        buf: &mut CudaBuffer,
        nx: u32,
        ny: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_3d(
        &mut self,
        buf: &mut CudaBuffer,
        nx: u32,
        ny: u32,
        nz: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;
}

impl CuFft for CuFftBackend {
    fn fft_1d(
        &mut self,
        buf: &mut CudaBuffer,
        n: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        assert_eq!(buf.len() as u32, n * batch);
        self.launch(
            buf,
            PlanKey {
                nx: n as i32,
                ny: 1,
                nz: 1,
                batch: batch as i32,
                dim: 1,
            },
            direction,
        )
    }

    fn fft_2d(
        &mut self,
        buf: &mut CudaBuffer,
        nx: u32,
        ny: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        assert_eq!(buf.len() as u32, nx * ny * batch);
        // cuFFT's plan_2d does not take a batch; for batched 2D we'd need
        // plan_many. For v0.1 we accept batch == 1 only on this path.
        assert_eq!(batch, 1, "2D batched FFT requires plan_many; not yet wired");
        self.launch(
            buf,
            PlanKey {
                nx: nx as i32,
                ny: ny as i32,
                nz: 1,
                batch: 1,
                dim: 2,
            },
            direction,
        )
    }

    fn fft_3d(
        &mut self,
        buf: &mut CudaBuffer,
        nx: u32,
        ny: u32,
        nz: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        assert_eq!(buf.len() as u32, nx * ny * nz);
        self.launch(
            buf,
            PlanKey {
                nx: nx as i32,
                ny: ny as i32,
                nz: nz as i32,
                batch: 1,
                dim: 3,
            },
            direction,
        )
    }
}
