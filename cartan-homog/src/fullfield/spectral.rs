//! Moulinec-Suquet spectral homogenisation through gpufft.
//!
//! Alternative cell-problem solver to the FEM path in
//! [`crate::fullfield::FullField`]. For periodic media this is the classical
//! Lippmann-Schwinger fixed-point iteration: at each step the polarisation
//! field is FFTd, multiplied by a Green's operator in frequency space, and
//! inverse-FFTd; iteration converges to the periodic gradient correctors.
//!
//! Only Order2 (scalar conductivity) is implemented here. The same scheme
//! extends to Order4 (elasticity) with a 4th-order Green's operator and
//! six fields per direction. That extension is follow-up work.
//!
//! Numerics:
//! - Reference medium is the arithmetic mean of κ over voxels.
//! - Convergence: relative RMS change of the ε field, default tolerance 1e-5.
//! - DC (ξ=0) coefficient pins the macroscopic mean to `e_i` exactly.
//!
//! Validation against the FEM path is in `tests/spectral_vs_fem.rs`.

#![cfg(feature = "gpu-fft")]

use alloc::format;
use alloc::vec::Vec;

use gpufft::{
    vulkan::{DeviceOptions, VulkanBackend, VulkanDevice},
    BufferOps, C2cPlanOps, Device, Direction, PlanDesc, Shape,
};
use nalgebra::Matrix3;
use num_complex::Complex32;

use crate::error::HomogError;
use crate::fullfield::voxelize::{VoxelGrid, NO_PHASE};
use crate::schemes::Effective;
use crate::tensor::Order2;

/// Spectral full-field homogenisation engine.
pub struct SpectralFullField {
    /// Cubic-grid resolution. cuFFT and VkFFT both accept arbitrary sizes;
    /// performance peaks at powers of two and smooth composites.
    pub resolution: usize,
    pub max_iters: usize,
    pub tol: f32,
}

impl SpectralFullField {
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            max_iters: 500,
            tol: 1e-5,
        }
    }

    /// Homogenise a voxelised unit cube with the given per-phase scalar
    /// conductivities. Returns the 3×3 effective conductivity together with
    /// the iteration count and final residual summed across the three
    /// directional sub-problems.
    pub fn homogenize_voxel(
        &self,
        voxels: &VoxelGrid,
        conductivities: &[f32],
    ) -> Result<Effective<Order2>, HomogError> {
        let n = self.resolution;
        let nvox = n * n * n;

        // Materialise κ(x) once. Layout: kappa[i + j*N + k*N²], matching
        // VkFFT's 3D buffer convention (x fastest, then y, then z).
        let mut kappa: Vec<f32> = Vec::with_capacity(nvox);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let pid = voxels.get(i, j, k);
                    let idx = if pid == NO_PHASE { 0 } else { pid as usize };
                    if idx >= conductivities.len() {
                        return Err(HomogError::Mesh(format!(
                            "voxel phase id {idx} out of range for {} conductivities",
                            conductivities.len()
                        )));
                    }
                    kappa.push(conductivities[idx]);
                }
            }
        }

        let kappa0 = kappa.iter().copied().sum::<f32>() / (nvox as f32);

        let dev = VulkanBackend::new_device(DeviceOptions::default())
            .map_err(|e| HomogError::Mesh(format!("VulkanBackend: {e}")))?;

        let mut k_eff = Matrix3::<f64>::zeros();
        let mut total_iters = 0usize;
        let mut worst_res = 0.0f64;
        for i_dir in 0..3 {
            let (col, iters, res) =
                self.run_direction(&dev, &kappa, kappa0, i_dir, n)?;
            for r in 0..3 {
                k_eff[(r, i_dir)] = col[r] as f64;
            }
            total_iters += iters;
            worst_res = worst_res.max(res as f64);
        }

        Ok(Effective {
            tensor: k_eff,
            concentration: None,
            iterations: Some(total_iters),
            residual: Some(worst_res),
        })
    }

    /// Run Moulinec-Suquet for a single direction `dir ∈ {0, 1, 2}` and
    /// return the column of `κ_eff` corresponding to applied macroscopic
    /// gradient `e_dir`, along with iteration count and final residual.
    fn run_direction(
        &self,
        dev: &VulkanDevice,
        kappa: &[f32],
        kappa0: f32,
        dir: usize,
        n: usize,
    ) -> Result<([f32; 3], usize, f32), HomogError> {
        let nvox = n * n * n;
        let plan_desc = PlanDesc {
            shape: Shape::D3([n as u32, n as u32, n as u32]),
            batch: 1,
            normalize: true,
        };

        // One plan, reused for all forward and inverse FFTs in this direction.
        let mut plan = dev
            .plan_c2c::<Complex32>(&plan_desc)
            .map_err(|e| HomogError::Mesh(format!("plan_c2c: {e}")))?;

        // ε_α(x) at iteration 0: constant field e_dir.
        let mut eps: [Vec<Complex32>; 3] =
            core::array::from_fn(|_| vec![Complex32::new(0.0, 0.0); nvox]);
        for x in eps[dir].iter_mut() {
            *x = Complex32::new(1.0, 0.0);
        }

        // Precompute signed integer frequencies along each axis.
        let freqs: Vec<i32> = (0..n)
            .map(|k| if k <= n / 2 { k as i32 } else { k as i32 - n as i32 })
            .collect();

        let mut last_res = f32::INFINITY;
        for iter in 0..self.max_iters {
            // ---- (1) σ_α(x) = κ(x) ε_α(x) on the host. ----
            let mut sigma: [Vec<Complex32>; 3] =
                core::array::from_fn(|_| vec![Complex32::new(0.0, 0.0); nvox]);
            for x_idx in 0..nvox {
                let k = kappa[x_idx];
                for a in 0..3 {
                    let e = eps[a][x_idx];
                    sigma[a][x_idx] = Complex32::new(k * e.re, k * e.im);
                }
            }

            // ---- (2) Forward FFT each σ_α on the GPU. ----
            let mut sigma_hat: [Vec<Complex32>; 3] =
                core::array::from_fn(|_| vec![Complex32::default(); nvox]);
            for a in 0..3 {
                let mut buf = dev
                    .alloc::<Complex32>(nvox)
                    .map_err(|e| HomogError::Mesh(format!("σ alloc: {e}")))?;
                buf.write(&sigma[a])
                    .map_err(|e| HomogError::Mesh(format!("σ write: {e}")))?;
                plan.execute(&mut buf, Direction::Forward)
                    .map_err(|e| HomogError::Mesh(format!("σ fft: {e}")))?;
                buf.read(&mut sigma_hat[a])
                    .map_err(|e| HomogError::Mesh(format!("σ read: {e}")))?;
            }

            // ---- (3) Apply Green's operator in frequency space on the host. ----
            //
            //   ε̂_α(ξ) = -ξ_α (Σ_β ξ_β σ̂_β(ξ)) / (κ⁰ |ξ|²)        for ξ ≠ 0
            //   ε̂_α(0) = N³ · e_{dir,α}                            for ξ = 0
            //
            // The DC value compensates for our unnormalized forward FFT so
            // that after the normalized inverse the constant component of ε
            // is exactly e_dir.
            let mut eps_hat: [Vec<Complex32>; 3] =
                core::array::from_fn(|_| vec![Complex32::new(0.0, 0.0); nvox]);
            let n_total = nvox as f32;
            for kz in 0..n {
                for ky in 0..n {
                    for kx in 0..n {
                        let idx = kx + ky * n + kz * n * n;
                        let xi = [
                            freqs[kx] as f32,
                            freqs[ky] as f32,
                            freqs[kz] as f32,
                        ];
                        let xi_sq = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2];
                        if xi_sq < 1e-12 {
                            for (a, hat) in eps_hat.iter_mut().enumerate() {
                                hat[idx] = if a == dir {
                                    Complex32::new(n_total, 0.0)
                                } else {
                                    Complex32::new(0.0, 0.0)
                                };
                            }
                        } else {
                            let dot_re = xi[0] * sigma_hat[0][idx].re
                                + xi[1] * sigma_hat[1][idx].re
                                + xi[2] * sigma_hat[2][idx].re;
                            let dot_im = xi[0] * sigma_hat[0][idx].im
                                + xi[1] * sigma_hat[1][idx].im
                                + xi[2] * sigma_hat[2][idx].im;
                            let denom = kappa0 * xi_sq;
                            for a in 0..3 {
                                eps_hat[a][idx] = Complex32::new(
                                    -xi[a] * dot_re / denom,
                                    -xi[a] * dot_im / denom,
                                );
                            }
                        }
                    }
                }
            }

            // ---- (4) Inverse FFT each ε̂_α on the GPU. ----
            let mut new_eps: [Vec<Complex32>; 3] =
                core::array::from_fn(|_| vec![Complex32::default(); nvox]);
            for a in 0..3 {
                let mut buf = dev
                    .alloc::<Complex32>(nvox)
                    .map_err(|e| HomogError::Mesh(format!("ε̂ alloc: {e}")))?;
                buf.write(&eps_hat[a])
                    .map_err(|e| HomogError::Mesh(format!("ε̂ write: {e}")))?;
                plan.execute(&mut buf, Direction::Inverse)
                    .map_err(|e| HomogError::Mesh(format!("ε̂ ifft: {e}")))?;
                buf.read(&mut new_eps[a])
                    .map_err(|e| HomogError::Mesh(format!("ε̂ read: {e}")))?;
            }

            // ---- (5) Convergence: relative RMS change in ε. ----
            let mut delta_sq = 0.0f64;
            let mut norm_sq = 0.0f64;
            for x_idx in 0..nvox {
                for a in 0..3 {
                    let d = (new_eps[a][x_idx].re - eps[a][x_idx].re) as f64;
                    delta_sq += d * d;
                    norm_sq += (new_eps[a][x_idx].re as f64).powi(2);
                }
            }
            let res = ((delta_sq / norm_sq.max(1e-30)).sqrt()) as f32;
            last_res = res;

            eps = new_eps;

            if res < self.tol {
                return Ok((volume_average_column(kappa, &eps, nvox), iter + 1, res));
            }
        }

        Ok((
            volume_average_column(kappa, &eps, nvox),
            self.max_iters,
            last_res,
        ))
    }
}

fn volume_average_column(kappa: &[f32], eps: &[Vec<Complex32>; 3], nvox: usize) -> [f32; 3] {
    let mut col = [0.0f64; 3];
    for x_idx in 0..nvox {
        let k = kappa[x_idx] as f64;
        for (a, c) in col.iter_mut().enumerate() {
            *c += k * (eps[a][x_idx].re as f64);
        }
    }
    let inv = 1.0 / (nvox as f64);
    [
        (col[0] * inv) as f32,
        (col[1] * inv) as f32,
        (col[2] * inv) as f32,
    ]
}
