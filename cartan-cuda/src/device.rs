//! `Device` is cartan-cuda's entry point. It owns a CUDA context, its default
//! stream, and the module compiled from [`crate::kernels`], and exposes the
//! batched operations as ordinary slice-in, `Vec`-out calls.
//!
//! Every batch is row-major with stride `dim`, which is the layout the kernels
//! index and the layout `nalgebra`'s `as_slice` already hands back for a column
//! vector.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};

use crate::error::CudaError;
use crate::kernels;

/// An open CUDA context with the cartan kernels loaded.
///
/// Construction is the expensive part, so hold one for as long as there is
/// work for it rather than building one per batch.
pub struct Device {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
}

impl Device {
    /// Open device `ordinal` and load the compiled module onto it.
    ///
    /// Fails with [`CudaError::Driver`] carrying error 803 when the loaded
    /// kernel module and the userspace driver are different versions, which is
    /// what a driver upgrade without a reboot leaves behind.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Self {
            ctx,
            stream,
            module,
        })
    }

    /// The underlying context, for callers that need to allocate against it
    /// directly.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The stream every operation on this device runs on.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// `Exp_p(v)` on the unit sphere in `dim` dimensions, for a whole batch.
    ///
    /// `p` and `v` hold `n * dim` doubles each. The result holds `n * dim`,
    /// point `i` occupying `[i * dim, (i + 1) * dim)`.
    ///
    /// The tangency of `v` at `p` is the caller's to establish. The kernel does
    /// not project, and the CPU path does not either.
    pub fn sphere_exp(&self, p: &[f64], v: &[f64], dim: usize) -> Result<Vec<f64>, CudaError> {
        let n = batch_len(p, dim, "point batch")?;
        expect_len(v, p.len(), "tangent batch")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        let p_dev = DeviceBuffer::from_host(&self.stream, p)?;
        let v_dev = DeviceBuffer::from_host(&self.stream, v)?;
        let mut theta = DeviceBuffer::<f64>::zeroed(&self.stream, n)?;

        // SAFETY: `theta` holds one element per point and the launch is sized
        // to the same `n`, so every thread writes inside the allocation. `v`
        // holds `n * dim` doubles, the extent the kernel reads under the `dim`
        // passed alongside it.
        unsafe {
            self.module.sphere_tangent_norm(
                &self.stream,
                LaunchConfig::for_num_elems(n as u32),
                &v_dev,
                dim as u32,
                &mut theta,
            )?;
        }

        let mut out = DeviceBuffer::<f64>::zeroed(&self.stream, p.len())?;
        // SAFETY: the launch is sized to `p.len()`, which is the length of
        // every buffer the kernel indexes per element, and `theta` is indexed
        // at `j / dim < n`.
        unsafe {
            self.module.sphere_exp_apply(
                &self.stream,
                LaunchConfig::for_num_elems(p.len() as u32),
                &p_dev,
                &v_dev,
                &theta,
                dim as u32,
                &mut out,
            )?;
        }

        Ok(out.to_host_vec(&self.stream)?)
    }

    /// `Log_p(q)` on the unit sphere in `dim` dimensions, for a whole batch.
    ///
    /// Shapes match [`Device::sphere_exp`]. Pairs separated by close to `pi`
    /// sit at the cut locus, where the logarithm stops being unique; the kernel
    /// returns the value the formula gives rather than reporting that.
    pub fn sphere_log(&self, p: &[f64], q: &[f64], dim: usize) -> Result<Vec<f64>, CudaError> {
        let n = batch_len(p, dim, "point batch")?;
        expect_len(q, p.len(), "target batch")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        let p_dev = DeviceBuffer::from_host(&self.stream, p)?;
        let q_dev = DeviceBuffer::from_host(&self.stream, q)?;
        let mut cos_angle = DeviceBuffer::<f64>::zeroed(&self.stream, n)?;

        // SAFETY: as in `sphere_exp`, the reduction writes one element per
        // point and the launch is sized to that count.
        unsafe {
            self.module.sphere_cos_angle(
                &self.stream,
                LaunchConfig::for_num_elems(n as u32),
                &p_dev,
                &q_dev,
                dim as u32,
                &mut cos_angle,
            )?;
        }

        let mut out = DeviceBuffer::<f64>::zeroed(&self.stream, p.len())?;
        // SAFETY: the launch is sized to `p.len()`, the length of both point
        // buffers and of the output, and `cos_angle` is indexed at
        // `j / dim < n`.
        unsafe {
            self.module.sphere_log_apply(
                &self.stream,
                LaunchConfig::for_num_elems(p.len() as u32),
                &p_dev,
                &q_dev,
                &cos_angle,
                dim as u32,
                &mut out,
            )?;
        }

        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Affine-invariant distance on SPD(3), for a whole batch of pairs.
    ///
    /// `p` and `q` hold `n * 9` doubles each, row-major 3x3 matrices. The
    /// result holds `n`.
    ///
    /// Definiteness is the caller's to establish. The Cholesky takes a square
    /// root of the leading entry and the eigenvalues are floored at 1e-14
    /// before the logarithm, so an indefinite input degrades rather than
    /// reporting.
    pub fn spd3_dist(&self, p: &[f64], q: &[f64]) -> Result<Vec<f64>, CudaError> {
        let n = batch_len(p, 9, "SPD(3) batch")?;
        expect_len(q, p.len(), "second SPD(3) batch")?;
        if n == 0 {
            return Ok(Vec::new());
        }

        let p_dev = DeviceBuffer::from_host(&self.stream, p)?;
        let q_dev = DeviceBuffer::from_host(&self.stream, q)?;
        let mut out = DeviceBuffer::<f64>::zeroed(&self.stream, n)?;

        // SAFETY: both inputs hold `9 * n` doubles, the extent the kernel reads
        // at pair `i`, and the output holds `n`, one per thread in a launch
        // sized to `n`.
        unsafe {
            self.module.spd3_dist(
                &self.stream,
                LaunchConfig::for_num_elems(n as u32),
                &p_dev,
                &q_dev,
                &mut out,
            )?;
        }

        Ok(out.to_host_vec(&self.stream)?)
    }
}

/// Number of items in a row-major batch of stride `stride`.
fn batch_len(data: &[f64], stride: usize, what: &'static str) -> Result<usize, CudaError> {
    if stride == 0 {
        return Err(CudaError::ZeroDim);
    }
    if !data.len().is_multiple_of(stride) {
        return Err(CudaError::Shape {
            what,
            expected: data.len().next_multiple_of(stride),
            got: data.len(),
        });
    }
    Ok(data.len() / stride)
}

fn expect_len(data: &[f64], expected: usize, what: &'static str) -> Result<(), CudaError> {
    if data.len() != expected {
        return Err(CudaError::Shape {
            what,
            expected,
            got: data.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shape checking runs on the host and needs no device, which is the reason
    // it is factored out of the launch paths.

    #[test]
    fn batch_len_counts_points() {
        assert_eq!(batch_len(&[0.0; 12], 3, "x").unwrap(), 4);
        assert_eq!(batch_len(&[], 3, "x").unwrap(), 0);
    }

    #[test]
    fn ragged_batch_is_rejected() {
        let e = batch_len(&[0.0; 11], 3, "point batch").unwrap_err();
        assert!(matches!(
            e,
            CudaError::Shape {
                expected: 12,
                got: 11,
                ..
            }
        ));
    }

    #[test]
    fn zero_stride_is_rejected() {
        assert!(matches!(
            batch_len(&[0.0; 4], 0, "x").unwrap_err(),
            CudaError::ZeroDim
        ));
    }

    #[test]
    fn mismatched_pair_is_rejected() {
        let e = expect_len(&[0.0; 9], 12, "tangent batch").unwrap_err();
        assert!(matches!(
            e,
            CudaError::Shape {
                expected: 12,
                got: 9,
                ..
            }
        ));
    }
}
