//! Batched manifold operations on CUDA, in double precision.
//!
//! The point of this crate is precision. WGSL, which `cartan-gpu` targets, has
//! no `f64`, so anything routed through it is single precision while the rest
//! of cartan is double. CUDA has `f64` natively, so these kernels are held to
//! the same tolerance the CPU code is: agreement is checked at 1e-13, not at
//! the 1e-6 an f32 path would force.
//!
//! Kernels are written in ordinary Rust and compiled to PTX by
//! `rustc-codegen-cuda`. The entry point is [`Device`]:
//!
//! ```no_run
//! use cartan_cuda::Device;
//!
//! // Two points on S^2 and a tangent at the first, row-major with stride 3.
//! let p = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let v = [0.0, 0.7, 0.0, 0.7, 0.0, 0.0];
//! let q = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
//!
//! let dev = Device::new(0)?;
//! let exp = dev.sphere_exp(&p, &v, 3)?;
//! let log = dev.sphere_log(&p, &q, 3)?;
//!
//! // One SPD(3) pair, row-major, nine doubles each.
//! let a = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
//! let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 4.0];
//! let dist = dev.spd3_dist(&a, &b)?;
//! # Ok::<(), cartan_cuda::CudaError>(())
//! ```
//!
//! The binary in this package runs the same operations against the CPU
//! implementation and reports the worst disagreement:
//!
//! ```text
//! cargo oxide run cartan-cuda
//! ```
//!
//! ## Why each sphere operation is two kernels
//!
//! `DisjointSlice` gives each thread exactly one output element, which is what
//! makes the writes provably non-overlapping. A batched `exp` produces `dim`
//! elements per point, so it does not fit that shape directly.
//!
//! Splitting it does. The first kernel runs one thread per point and performs
//! the O(dim) reduction, writing a single scalar. The second runs one thread
//! per output element and consumes that scalar. The reduction happens once per
//! point rather than once per element, which is the whole reason not to fold
//! them together.

use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

mod device;
mod error;
mod mass;

pub use device::Device;
pub use error::CudaError;
pub use mass::DeviceHodgeMass;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `y = M x` for a Galerkin Hodge mass kept in element form.
    ///
    /// One thread owns one degree of freedom and gathers every element-matrix
    /// row that feeds it, so the write is single and no atomic is involved. The
    /// scatter form, looping cells and accumulating into shared degrees of
    /// freedom, would need an atomic floating-point add, and that reorders the
    /// summation between runs, which is not something the host path can be
    /// compared against at 1e-13.
    ///
    /// `offsets` has `ndofs + 1` entries. `entries` packs `cell * nlocal +
    /// local_row`. `dofs` gives the degree of freedom of every local face,
    /// cell-major, with `u32::MAX` marking a constrained face, which is skipped
    /// in the column loop exactly as the host path skips it.
    #[kernel]
    pub fn hodge_mass_apply(
        offsets: &[u32],
        entries: &[u32],
        dofs: &[u32],
        elmats: &[f64],
        x: &[f64],
        nlocal: u32,
        ndofs: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= ndofs as usize {
            return;
        }

        let n = nlocal as usize;
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;

        let mut sum = 0.0f64;
        for e in start..end {
            let slot = entries[e] as usize;
            let cell = slot / n;
            let local_row = slot % n;
            let row_base = (cell * n + local_row) * n;
            let dof_base = cell * n;
            for j in 0..n {
                let col = dofs[dof_base + j];
                if col != u32::MAX {
                    sum += elmats[row_base + j] * x[col as usize];
                }
            }
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = sum;
        }
    }

    /// Geodesic length of each tangent vector: one thread per point.
    ///
    /// This is the O(dim) reduction, kept out of the per-element kernel so it
    /// runs once per point instead of once per component.
    #[kernel]
    pub fn sphere_tangent_norm(v: &[f64], dim: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let d = dim as usize;

        let mut sum_sq = 0.0f64;
        for k in 0..d {
            let x = v[i * d + k];
            sum_sq += x * x;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = sum_sq.sqrt();
        }
    }

    /// `Exp_p(v) = cos(t) p + sin(t)/t v`, one thread per output component.
    ///
    /// `t` comes in precomputed. Below the small-angle cutoff the series
    /// `sin(t)/t -> 1` is used, matching what the CPU path does rather than
    /// dividing by something near zero.
    #[kernel]
    pub fn sphere_exp_apply(
        p: &[f64],
        v: &[f64],
        theta: &[f64],
        dim: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let d = dim as usize;
        let t = theta[j / d];

        let (a, b) = if t < 1e-7 {
            (1.0 - 0.5 * t * t, 1.0)
        } else {
            (t.cos(), t.sin() / t)
        };

        if let Some(slot) = out.get_mut(idx) {
            *slot = a * p[j] + b * v[j];
        }
    }

    /// Inner product of each pair: one thread per point.
    ///
    /// Clamped into [-1, 1] here rather than at the point of use, so the
    /// per-element kernel never hands `acos` an out-of-range argument.
    #[kernel]
    pub fn sphere_cos_angle(p: &[f64], q: &[f64], dim: u32, mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let d = dim as usize;

        let mut c = 0.0f64;
        for k in 0..d {
            c += p[i * d + k] * q[i * d + k];
        }
        c = c.clamp(-1.0, 1.0);

        if let Some(slot) = out.get_mut(idx) {
            *slot = c;
        }
    }

    /// `Log_p(q) = t (q - cos(t) p) / sin(t)`, one thread per output component.
    ///
    /// `t = acos(c)` is recomputed per element rather than stored: it is a
    /// couple of scalar operations against a second global array and a second
    /// launch.
    #[kernel]
    pub fn sphere_log_apply(
        p: &[f64],
        q: &[f64],
        cos_angle: &[f64],
        dim: u32,
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let j = idx.get();
        let d = dim as usize;

        let c = cos_angle[j / d];
        let t = c.acos();

        // At t -> 0 the geodesic degenerates and the projection q - c p is
        // already the answer to first order, so the scale factor is 1.
        let k = if t < 1e-7 { 1.0 } else { t / t.sin() };

        if let Some(slot) = out.get_mut(idx) {
            *slot = (q[j] - c * p[j]) * k;
        }
    }

    /// Affine-invariant distance on SPD(3), one thread per pair.
    ///
    /// ```text
    /// d(P, Q)^2 = sum_i log^2(lambda_i),  lambda_i = eig(P^-1 Q)
    /// ```
    ///
    /// The same route the CPU takes: Cholesky `P = L L^T`, then the spectrum of
    /// `L^-1 Q L^-T`, which is symmetric and similar to `P^-1 Q`. The matrix
    /// logarithm is never formed.
    ///
    /// One scalar out per pair, so this fits the one-element-per-thread shape
    /// directly and needs no second kernel.
    #[kernel]
    pub fn spd3_dist(p: &[f64], q: &[f64], mut out: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let b = i * 9;

        // Cholesky of P, lower triangular, unrolled for 3x3.
        let l00 = p[b].sqrt();
        let l10 = p[b + 3] / l00;
        let l11 = (p[b + 4] - l10 * l10).sqrt();
        let l20 = p[b + 6] / l00;
        let l21 = (p[b + 7] - l20 * l10) / l11;
        let l22 = (p[b + 8] - l20 * l20 - l21 * l21).sqrt();

        // A = L^-1 Q by forward substitution, one column at a time.
        let mut a = [0.0f64; 9];
        for c in 0..3 {
            let q0 = q[b + c];
            let q1 = q[b + 3 + c];
            let q2 = q[b + 6 + c];
            let a0 = q0 / l00;
            let a1 = (q1 - l10 * a0) / l11;
            let a2 = (q2 - l20 * a0 - l21 * a1) / l22;
            a[c] = a0;
            a[3 + c] = a1;
            a[6 + c] = a2;
        }

        // M = L^-1 A^T, giving L^-1 Q L^-T. Symmetric in exact arithmetic.
        let mut m = [0.0f64; 9];
        for c in 0..3 {
            let t0 = a[c * 3];
            let t1 = a[c * 3 + 1];
            let t2 = a[c * 3 + 2];
            let m0 = t0 / l00;
            let m1 = (t1 - l10 * m0) / l11;
            let m2 = (t2 - l20 * m0 - l21 * m1) / l22;
            m[c] = m0;
            m[3 + c] = m1;
            m[6 + c] = m2;
        }

        // Symmetrise away the rounding asymmetry before the eigensolver, which
        // assumes it.
        let mut s = [0.0f64; 9];
        for r in 0..3 {
            for c in 0..3 {
                s[r * 3 + c] = 0.5 * (m[r * 3 + c] + m[c * 3 + r]);
            }
        }

        // Cyclic Jacobi, eigenvalues only: the eigenvectors never reach the
        // answer, so the rotations are applied to the matrix alone.
        for _sweep in 0..12 {
            let mut off = s[1] * s[1] + s[2] * s[2] + s[5] * s[5];
            if off < 1e-300 {
                off = 0.0;
            }
            if off == 0.0 {
                break;
            }

            for pp in 0..3 {
                for qq in (pp + 1)..3 {
                    let apq = s[pp * 3 + qq];
                    if apq != 0.0 {
                        let app = s[pp * 3 + pp];
                        let aqq = s[qq * 3 + qq];
                        let theta = (aqq - app) / (2.0 * apq);
                        let tt = if theta >= 0.0 {
                            1.0 / (theta + (theta * theta + 1.0).sqrt())
                        } else {
                            -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                        };
                        let c = 1.0 / (tt * tt + 1.0).sqrt();
                        let sn = tt * c;

                        s[pp * 3 + pp] = app - tt * apq;
                        s[qq * 3 + qq] = aqq + tt * apq;
                        s[pp * 3 + qq] = 0.0;
                        s[qq * 3 + pp] = 0.0;

                        for k in 0..3 {
                            if k != pp && k != qq {
                                let akp = s[k * 3 + pp];
                                let akq = s[k * 3 + qq];
                                let np = c * akp - sn * akq;
                                let nq = sn * akp + c * akq;
                                s[k * 3 + pp] = np;
                                s[pp * 3 + k] = np;
                                s[k * 3 + qq] = nq;
                                s[qq * 3 + k] = nq;
                            }
                        }
                    }
                }
            }
        }

        // Floored at 1e-14 before the logarithm, matching the CPU path, so a
        // point that has lost definiteness to rounding degrades the same way.
        let mut total = 0.0f64;
        for k in 0..3 {
            let mut lam = s[k * 3 + k];
            if lam < 1e-14 {
                lam = 1e-14;
            }
            let ln = lam.ln();
            total += ln * ln;
        }

        if let Some(slot) = out.get_mut(idx) {
            *slot = total.sqrt();
        }
    }
}
