//! cuFFT-backed FFT implementation.
//!
//! Implements the same backend-agnostic [`crate::Fft`] trait as
//! [`crate::fft::VkFftBackend`] but with `CudaBuffer` as the associated
//! [`crate::Fft::Buffer`] type. Forward then inverse is identity on both
//! backends; the cuFFT inverse is post-scaled by `1/N` with
//! `cublasSscal_v2` since cuFFT does not auto-normalise.

#![cfg(feature = "cufft")]

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cublas::{self, CudaBlas};
use cudarc::cufft::sys::{cufftType, float2};
use cudarc::cufft::{CudaFft, FftDirection as CuFftDirection};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};

use crate::{CudaBuffer, CudaDevice, Fft, FftDirection, GpuError};

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct PlanKey {
    nx: i32,
    ny: i32,
    nz: i32,
    batch: i32,
    dim: u8,
}

impl PlanKey {
    /// Per-transform size (the divisor for inverse normalisation).
    fn transform_size(&self) -> u64 {
        match self.dim {
            1 => self.nx as u64,
            2 => self.nx as u64 * self.ny as u64,
            3 => self.nx as u64 * self.ny as u64 * self.nz as u64,
            _ => unreachable!(),
        }
    }
}

/// cuFFT-backed FFT engine bound to a single CUDA stream.
pub struct CuFftBackend {
    stream: Arc<CudaStream>,
    plans: HashMap<PlanKey, CudaFft>,
    blas: CudaBlas,
}

impl CuFftBackend {
    /// Construct a backend on the device's default stream.
    pub fn new(dev: &CudaDevice) -> Result<Self, GpuError> {
        let stream = dev.cuda_context().default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| GpuError::CudaError(format!("CudaBlas::new: {e:?}")))?;
        Ok(Self {
            stream,
            plans: HashMap::new(),
            blas,
        })
    }

    fn get_or_create_plan(&mut self, key: PlanKey) -> Result<&CudaFft, GpuError> {
        if !self.plans.contains_key(&key) {
            let plan = match (key.dim, key.batch) {
                // Simple plans cover the unbatched cases.
                (1, _) => CudaFft::plan_1d(
                    key.nx,
                    cufftType::CUFFT_C2C,
                    key.batch,
                    self.stream.clone(),
                ),
                (2, 1) => CudaFft::plan_2d(
                    key.nx,
                    key.ny,
                    cufftType::CUFFT_C2C,
                    self.stream.clone(),
                ),
                (3, 1) => CudaFft::plan_3d(
                    key.nx,
                    key.ny,
                    key.nz,
                    cufftType::CUFFT_C2C,
                    self.stream.clone(),
                ),
                // Batched 2D goes through plan_many; cuFFT has no plan_2d
                // overload taking batch. Layout is contiguous: each transform
                // occupies nx*ny consecutive complex elements.
                (2, _) => CudaFft::plan_many(
                    &[key.nx, key.ny],
                    None,
                    1,
                    key.nx * key.ny,
                    None,
                    1,
                    key.nx * key.ny,
                    cufftType::CUFFT_C2C,
                    key.batch,
                    self.stream.clone(),
                ),
                // Batched 3D also goes through plan_many.
                (3, _) => CudaFft::plan_many(
                    &[key.nx, key.ny, key.nz],
                    None,
                    1,
                    key.nx * key.ny * key.nz,
                    None,
                    1,
                    key.nx * key.ny * key.nz,
                    cufftType::CUFFT_C2C,
                    key.batch,
                    self.stream.clone(),
                ),
                _ => unreachable!(),
            }
            .map_err(|e| GpuError::CudaError(format!("cuFFT plan: {e:?}")))?;
            self.plans.insert(key, plan);
        }
        Ok(self.plans.get(&key).unwrap())
    }

    /// In-place scale of a `CudaSlice<float2>` by a real scalar via
    /// `cublasSscal_v2`. Reinterprets the buffer as 2N f32s — cuBLAS
    /// applies the same scalar to both real and imaginary parts, which
    /// is equivalent to a real-valued complex scale.
    fn scale_inplace(
        &self,
        slice: &mut CudaSlice<float2>,
        scale: f32,
    ) -> Result<(), GpuError> {
        let count_f32 = (slice.len() * 2) as i32;
        let (ptr, _record) = slice.device_ptr_mut(&self.stream);
        let status = unsafe {
            cublas::sys::cublasSscal_v2(
                *self.blas.handle(),
                count_f32,
                &scale as *const f32,
                ptr as *mut f32,
                1,
            )
        };
        if status != cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(GpuError::CudaError(format!(
                "cublasSscal_v2: {status:?}"
            )));
        }
        Ok(())
    }

    fn launch(
        &mut self,
        buf: &mut CudaBuffer,
        key: PlanKey,
        direction: FftDirection,
    ) -> Result<(), GpuError> {
        let _ = self.get_or_create_plan(key)?;
        let plan = self.plans.get(&key).unwrap();

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

        // Normalise inverse on-device so the trait's forward∘inverse=identity
        // semantics hold regardless of backend. Done before swap so we operate
        // on the FFT output (currently in `scratch`).
        if matches!(direction, FftDirection::Inverse) {
            let n = key.transform_size() as f32;
            self.scale_inplace(&mut scratch, 1.0 / n)?;
        }

        // Synchronise so any host-side reads of buf see the FFT result.
        self.stream
            .synchronize()
            .map_err(|e| GpuError::CudaError(format!("cuFFT sync: {e:?}")))?;

        core::mem::swap(&mut buf.slice, &mut scratch);
        Ok(())
    }
}

impl Fft for CuFftBackend {
    type Buffer = CudaBuffer;

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
        self.launch(
            buf,
            PlanKey {
                nx: nx as i32,
                ny: ny as i32,
                nz: 1,
                batch: batch as i32,
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
