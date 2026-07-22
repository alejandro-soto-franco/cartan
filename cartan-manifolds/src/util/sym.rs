// ~/cartan/cartan-manifolds/src/util/sym.rs

//! Utilities for symmetric positive definite matrices.
//!
//! These are internal helpers used by `Corr<N>` and future SPD-family manifolds.
//!
//! ## Contents
//!
//! - `sym_min_eigenvalue`: minimum eigenvalue of a symmetric matrix.
//!
//! - `nearest_corr_matrix`: nearest correlation matrix to an arbitrary symmetric
//!   matrix in the Frobenius norm. Uses Higham's (2002) alternating projections
//!   algorithm: alternately project onto the PD cone and the unit-diagonal affine
//!   subspace until convergence.
//!
//! - `sym_symmetrize`: (A + A^T) / 2. Enforces exact symmetry after drift.
//!
//! - `sym_sqrt`: symmetric matrix square root P^{1/2} via eigendecomposition.
//!
//! - `sym_sqrt_inv`: symmetric matrix inverse square root P^{-1/2}.
//!
//! - `sym_inv`: symmetric matrix inverse P^{-1} via eigendecomposition.
//!
//! - `sym_log`: matrix logarithm of a symmetric PD matrix. Input must be PD.
//!
//! - `sym_exp`: matrix exponential of a symmetric matrix. Always valid.
//!
//! ## Note on eigendecomposition
//!
//! nalgebra's `symmetric_eigen()` requires `Const<N>: ToTypenum`, which is
//! only satisfied for specific concrete sizes. For generic `const N: usize`
//! we work via `DMatrix` and convert back to `SMatrix`.
//!
//! ## References
//!
//! - Higham, N. J. (2002). "Computing the Nearest Correlation Matrix — a Problem
//!   from Finance." *IMA Journal of Numerical Analysis*, 22(3), 329–343.
//! - Pennec, X., Fillard, P., Ayache, N. (2006). "A Riemannian Framework for
//!   Tensor Computing." IJCV 66(1), 41–66. (SPD exp/log/sqrt formulas.)

use cartan_core::Real;
use nalgebra::{DMatrix, DVector, SMatrix, SVector};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum eigenvalue of a symmetric N×N matrix.
///
/// Converts to `DMatrix` for eigendecomposition, which handles generic const N.
/// The input is assumed symmetric; only the lower triangle is meaningful.
pub(crate) fn sym_min_eigenvalue<const N: usize>(m: &SMatrix<Real, N, N>) -> Real {
    let dm = DMatrix::from_column_slice(N, N, m.as_slice());
    let eigen = dm.symmetric_eigen();
    eigen
        .eigenvalues
        .iter()
        .cloned()
        .fold(Real::INFINITY, Real::min)
}

/// (A + A^T) / 2: enforce exact symmetry after floating-point drift.
pub(crate) fn sym_symmetrize<const N: usize>(a: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    (a + a.transpose()) * 0.5
}

/// Nearest correlation matrix to `a` in the Frobenius norm.
///
/// A correlation matrix is symmetric positive definite with unit diagonal.
/// Uses Higham's (2002) alternating projections with Dykstra's correction,
/// which converges for any symmetric input.
///
/// ## Algorithm
///
/// Starting from `sym(a)`, alternately apply:
/// 1. Project onto the PD cone: eigendecomposition, clamp eigenvalues to `MIN_EIG`.
/// 2. Project onto the unit-diagonal affine subspace: set diag entries to 1.0.
///
/// The Dykstra correction term accumulates the overshoot of the PD projection
/// so that the diagonal-1 projection does not undo prior PD progress.
///
/// ## References
///
/// - Higham, N. J. (2002). *IMA Journal of Numerical Analysis*, 22(3), 329–343.
pub(crate) fn nearest_corr_matrix<const N: usize>(
    a: &SMatrix<Real, N, N>,
    max_iter: usize,
    tol: Real,
) -> SMatrix<Real, N, N> {
    const MIN_EIG: Real = 1e-8;

    let mut x = sym_symmetrize(a);
    // Dykstra correction (SMatrix, zero-initialized).
    let mut delta_s: SMatrix<Real, N, N> = SMatrix::zeros();

    for _ in 0..max_iter {
        let x_prev = x;

        // Step 1: project onto PD cone with Dykstra correction.
        let y = x + delta_s;
        let ydm = DMatrix::from_column_slice(N, N, y.as_slice());
        let eigen = ydm.symmetric_eigen();
        let clamped: DVector<Real> = eigen
            .eigenvalues
            .map(|v| if v < MIN_EIG { MIN_EIG } else { v });
        let x_pd_dm =
            &eigen.eigenvectors * DMatrix::from_diagonal(&clamped) * eigen.eigenvectors.transpose();
        let x_pd = SMatrix::from_column_slice(x_pd_dm.as_slice());
        delta_s = y - x_pd;

        // Step 2: project onto unit-diagonal affine subspace.
        x = x_pd;
        for i in 0..N {
            x[(i, i)] = 1.0;
        }

        // Convergence check on Frobenius norm of the iterate change.
        if (x - x_prev).norm() < tol {
            break;
        }
    }

    x
}

/// Eigendecomposition of a symmetric matrix: returns (V, d) where M = V diag(d) V^T.
///
/// Internal helper shared by all sym_* functions. Avoids repeating the
/// DMatrix conversion and eigen call at every call site.
#[inline]
fn sym_eigen<const N: usize>(m: &SMatrix<Real, N, N>) -> (DMatrix<Real>, DVector<Real>) {
    let dm = DMatrix::from_column_slice(N, N, m.as_slice());
    let eigen = dm.symmetric_eigen();
    (eigen.eigenvectors, eigen.eigenvalues)
}

/// Symmetric eigendecomposition, on the stack for the sizes that benefit.
///
/// `N` is a compile-time constant, so the branch folds away entirely.
#[inline]
fn sym_eigen_s<const N: usize>(m: &SMatrix<Real, N, N>) -> (SMatrix<Real, N, N>, SVector<Real, N>) {
    if N <= JACOBI_MAX_N {
        jacobi_eigen(m)
    } else {
        let (v, d) = sym_eigen(m);
        (
            SMatrix::from_column_slice(v.as_slice()),
            SVector::from_column_slice(d.as_slice()),
        )
    }
}

/// Eigenvalues of a symmetric matrix, without the eigenvectors.
///
/// `symmetric_eigenvalues` skips the eigenvector accumulation that
/// `symmetric_eigen` performs, which is the bulk of the work. Used where only
/// the spectrum reaches the answer, as in the affine-invariant distance.
///
/// Goes through `DMatrix` for the same reason `sym_eigen` does: the const
/// generic `SMatrix<Real, N, N>` does not satisfy nalgebra's `DimSub<U1>`
/// bound for the symmetric eigensolver.
#[inline]
pub(crate) fn sym_eigenvalues<const N: usize>(m: &SMatrix<Real, N, N>) -> DVector<Real> {
    if N <= JACOBI_MAX_N {
        DVector::from_column_slice(jacobi_eigenvalues(m).as_slice())
    } else {
        DMatrix::from_column_slice(N, N, m.as_slice()).symmetric_eigenvalues()
    }
}

/// Ambient dimension at or below which the stack-based Jacobi solver is used.
///
/// Measured, not chosen. Cost of `Spd::exp`, Jacobi against the `DMatrix`
/// path:
///
/// ```text
///   N        Jacobi     DMatrix
///   3        428 ns      676 ns     Jacobi 1.58x
///   4       1182 ns     1037 ns
///   5       2656 ns     1661 ns
///   6       4083 ns     2336 ns
///  10      18495 ns     6496 ns     DMatrix 2.85x
/// ```
///
/// Jacobi is O(N^3) per sweep and needs several sweeps, so it loses to a
/// tridiagonalise-then-QR solver as soon as the matrix is large enough for
/// that solver's setup to amortise. It wins at N = 3, which is the size most
/// SPD work uses: covariance matrices, diffusion tensors, and the Order2
/// Kelvin-Mandel representation in `cartan-homog`.
const JACOBI_MAX_N: usize = 3;

/// Sweeps before giving up. Cyclic Jacobi converges quadratically and needs
/// six to ten sweeps in practice; this is a backstop, not an operating point.
const JACOBI_MAX_SWEEPS: usize = 30;

/// Symmetric eigendecomposition by cyclic Jacobi rotations, entirely on the
/// stack.
///
/// Returns `(V, d)` with `M = V diag(d) V^T`. Neither the order of the
/// eigenvalues nor the sign of the eigenvectors is canonicalised, because
/// every consumer here recomposes `V diag(f(d)) V^T`, which is invariant to
/// both.
///
/// This exists because the `DMatrix` path allocates on the heap for every
/// call and runs a general tridiagonalise-then-QR solver, which is the wrong
/// trade for the 3x3 to 10x10 matrices the SPD manifold actually uses. Jacobi
/// is also backward stable and tends to give better relative accuracy on small
/// eigenvalues than QR, which matters here since the results feed a logarithm.
fn jacobi_eigen<const N: usize>(m: &SMatrix<Real, N, N>) -> (SMatrix<Real, N, N>, SVector<Real, N>) {
    let mut a = *m;
    let mut v = SMatrix::<Real, N, N>::identity();

    for _ in 0..JACOBI_MAX_SWEEPS {
        // Sum of squared off-diagonals: the quantity Jacobi drives to zero.
        let mut off = 0.0;
        for i in 0..N {
            for j in (i + 1)..N {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off <= Real::EPSILON * Real::EPSILON {
            break;
        }

        for pp in 0..N {
            for qq in (pp + 1)..N {
                let apq = a[(pp, qq)];
                if apq == 0.0 {
                    continue;
                }

                // t is the smaller root of t^2 + 2 theta t - 1 = 0, which is
                // the rotation that keeps the transformation well conditioned.
                let theta = (a[(qq, qq)] - a[(pp, pp)]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let app = a[(pp, pp)];
                let aqq = a[(qq, qq)];
                a[(pp, pp)] = app - t * apq;
                a[(qq, qq)] = aqq + t * apq;
                a[(pp, qq)] = 0.0;
                a[(qq, pp)] = 0.0;

                for k in 0..N {
                    if k != pp && k != qq {
                        let akp = a[(k, pp)];
                        let akq = a[(k, qq)];
                        let np = c * akp - s * akq;
                        let nq = s * akp + c * akq;
                        a[(k, pp)] = np;
                        a[(pp, k)] = np;
                        a[(k, qq)] = nq;
                        a[(qq, k)] = nq;
                    }
                    let vkp = v[(k, pp)];
                    let vkq = v[(k, qq)];
                    v[(k, pp)] = c * vkp - s * vkq;
                    v[(k, qq)] = s * vkp + c * vkq;
                }
            }
        }
    }

    let d = SVector::<Real, N>::from_fn(|i, _| a[(i, i)]);
    (v, d)
}

/// Eigenvalues by cyclic Jacobi, without accumulating the eigenvectors.
///
/// Same rotations as [`jacobi_eigen`], minus the rotation of `V`. That update
/// touches `2N` entries per rotation, the same order as the update to `A`
/// itself, so dropping it is worth roughly a third of the work. The
/// eigenvalues-only consumers, the affine-invariant distance among them, were
/// paying for eigenvectors they discarded.
fn jacobi_eigenvalues<const N: usize>(m: &SMatrix<Real, N, N>) -> SVector<Real, N> {
    let mut a = *m;

    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0;
        for i in 0..N {
            for j in (i + 1)..N {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off <= Real::EPSILON * Real::EPSILON {
            break;
        }

        for pp in 0..N {
            for qq in (pp + 1)..N {
                let apq = a[(pp, qq)];
                if apq == 0.0 {
                    continue;
                }

                let theta = (a[(qq, qq)] - a[(pp, pp)]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                let app = a[(pp, pp)];
                let aqq = a[(qq, qq)];
                a[(pp, pp)] = app - t * apq;
                a[(qq, qq)] = aqq + t * apq;
                a[(pp, qq)] = 0.0;
                a[(qq, pp)] = 0.0;

                for k in 0..N {
                    if k != pp && k != qq {
                        let akp = a[(k, pp)];
                        let akq = a[(k, qq)];
                        let np = c * akp - s * akq;
                        let nq = s * akp + c * akq;
                        a[(k, pp)] = np;
                        a[(pp, k)] = np;
                        a[(k, qq)] = nq;
                        a[(qq, k)] = nq;
                    }
                }
            }
        }
    }

    SVector::<Real, N>::from_fn(|i, _| a[(i, i)])
}

/// Rebuild `V diag(fd) V^T` from an eigendecomposition.
///
/// Column i of `V diag(fd)` is column i of `V` scaled by `fd[i]`, which is
/// O(N^2). Materialising the diagonal matrix and multiplying by it instead
/// costs a second O(N^3) product for the same answer.
#[inline]
fn recompose<const N: usize>(
    v: &SMatrix<Real, N, N>,
    fd: &SVector<Real, N>,
) -> SMatrix<Real, N, N> {
    let mut scaled = *v;
    for i in 0..N {
        let f = fd[i];
        for j in 0..N {
            scaled[(j, i)] *= f;
        }
    }
    scaled * v.transpose()
}

/// Apply a scalar function f to the eigenvalues of a symmetric matrix.
///
/// Returns V * diag(f(d_i)) * V^T as an SMatrix.
#[inline]
fn sym_apply<const N: usize, F: Fn(Real) -> Real>(
    m: &SMatrix<Real, N, N>,
    f: F,
) -> SMatrix<Real, N, N> {
    let (v, d) = sym_eigen_s(m);
    recompose::<N>(&v, &d.map(f))
}

/// Both `P^{1/2}` and `P^{-1/2}` from a single eigendecomposition.
///
/// `sym_sqrt` and `sym_sqrt_inv` decompose the same matrix and differ only in
/// the scalar map applied to its eigenvalues, so calling both, as the SPD
/// exponential and logarithm did, paid for the decomposition twice.
///
/// The clamping matches the two functions exactly: negative eigenvalues floor
/// to zero for the root, and eigenvalues at or below 1e-14 give 1e7 for the
/// inverse root, avoiding a division by zero on a near-singular input.
pub(crate) fn sym_sqrt_pair<const N: usize>(
    m: &SMatrix<Real, N, N>,
) -> (SMatrix<Real, N, N>, SMatrix<Real, N, N>) {
    let (v, d) = sym_eigen_s(m);
    let root = d.map(|x| x.max(0.0).sqrt());
    let inv_root = d.map(|x| if x > 1e-14 { 1.0 / x.sqrt() } else { 1.0 / 1e-7 });
    (recompose::<N>(&v, &root), recompose::<N>(&v, &inv_root))
}

/// Symmetric matrix square root: P^{1/2}.
///
/// Requires all eigenvalues of P to be non-negative. Negative eigenvalues
/// are clamped to zero (they arise only from floating-point drift in nearly
/// PSD inputs).
///
/// Uses eigendecomposition: P = V D V^T, then P^{1/2} = V sqrt(D) V^T.
pub(crate) fn sym_sqrt<const N: usize>(m: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    sym_apply(m, |v| v.max(0.0).sqrt())
}

/// Symmetric matrix inverse square root: P^{-1/2}.
///
/// Requires all eigenvalues of P to be strictly positive. Eigenvalues
/// smaller than 1e-14 are treated as 1e-14 (numerical floor) to avoid
/// division by zero in near-singular inputs.
pub(crate) fn sym_sqrt_inv<const N: usize>(m: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    sym_apply(m, |v| {
        if v > 1e-14 {
            1.0 / v.sqrt()
        } else {
            1.0 / 1e-7
        }
    })
}

/// Symmetric matrix inverse: P^{-1}.
///
/// Computed via eigendecomposition. Eigenvalues smaller than 1e-14 are
/// floored to avoid numerical blow-up on nearly-singular inputs.
pub(crate) fn sym_inv<const N: usize>(m: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    sym_apply(m, |v| if v > 1e-14 { 1.0 / v } else { 1.0 / 1e-14 })
}

/// Matrix logarithm of a symmetric positive definite matrix: log(P).
///
/// P must be positive definite. Eigenvalues are clamped to [1e-14, ∞)
/// before taking the log to handle near-zero eigenvalues.
///
/// Uses eigendecomposition: P = V D V^T, log(P) = V log(D) V^T.
pub(crate) fn sym_log<const N: usize>(m: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    sym_apply(m, |v| v.max(1e-14).ln())
}

/// Matrix exponential of a symmetric matrix: exp(S).
///
/// Valid for any symmetric input (not just PD). The result is always PD
/// since exp maps eigenvalues v -> e^v > 0.
///
/// Uses eigendecomposition: S = V D V^T, exp(S) = V exp(D) V^T.
pub(crate) fn sym_exp<const N: usize>(m: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    sym_apply(m, |v| v.exp())
}

/// Apply a scalar function to the eigenvalues of a symmetric matrix (pub(crate) alias).
///
/// This exposes `sym_apply` under a stable name for use in `spd.rs::project_point`
/// and any other manifold that needs arbitrary spectral maps.
pub(crate) fn sym_apply_pub<const N: usize>(
    m: &SMatrix<Real, N, N>,
    f: impl Fn(Real) -> Real,
) -> SMatrix<Real, N, N> {
    sym_apply(m, f)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The Jacobi path is only reached at N = 3, so a regression there would
    /// be invisible to the larger-N tests. This pins the decomposition itself:
    /// V must be orthogonal and V diag(d) V^T must reproduce the input.
    #[test]
    fn test_jacobi_eigen_reconstructs_and_is_orthogonal() {
        let cases = [
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 1.0, 2.0, 3.0, 0.5, 1.0, 0.5, 2.0]),
            // Degenerate spectrum: repeated eigenvalues are where a rotation
            // based solver can stall if the pivot choice is wrong.
            SMatrix::<Real, 3, 3>::identity() * 2.5,
            // Nearly singular, which is what the eigenvalue floors exist for.
            SMatrix::<Real, 3, 3>::from_row_slice(&[
                1.0, 0.0, 0.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1.0,
            ]),
            // Already diagonal: zero rotations needed.
            SMatrix::<Real, 3, 3>::from_row_slice(&[3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        for m in cases {
            let (v, d) = jacobi_eigen(&m);

            let recon = recompose::<3>(&v, &d);
            assert!(
                (recon - m).norm() < 1e-12,
                "V diag(d) V^T does not reproduce the input: {:.3e}",
                (recon - m).norm()
            );

            let vtv = v.transpose() * v;
            assert!(
                (vtv - SMatrix::<Real, 3, 3>::identity()).norm() < 1e-12,
                "eigenvectors are not orthonormal"
            );
        }
    }

    /// The eigenvalues-only Jacobi is a separate loop from the full one, so a
    /// divergence between them would be silent: the distance would drift while
    /// every decomposition-based result stayed correct.
    #[test]
    fn test_jacobi_eigenvalues_matches_full_decomposition() {
        let cases = [
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 1.0, 2.0, 3.0, 0.5, 1.0, 0.5, 2.0]),
            SMatrix::<Real, 3, 3>::identity() * 2.5,
            SMatrix::<Real, 3, 3>::from_row_slice(&[
                1.0, 0.0, 0.0, 0.0, 1e-13, 0.0, 0.0, 0.0, 1.0,
            ]),
            SMatrix::<Real, 3, 3>::from_row_slice(&[3.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 1.0]),
        ];

        for m in cases {
            let (_, full) = jacobi_eigen(&m);
            let only = jacobi_eigenvalues(&m);

            // Same rotations in the same order, so these should agree exactly;
            // the tolerance is there only to survive a future reordering.
            let mut a: Vec<Real> = full.iter().copied().collect();
            let mut b: Vec<Real> = only.iter().copied().collect();
            a.sort_by(|x, y| x.partial_cmp(y).unwrap());
            b.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-14, "eigenvalues diverge: {x} vs {y}");
            }
        }
    }

    #[test]
    fn test_sym_min_eigenvalue_identity() {
        let m = SMatrix::<Real, 3, 3>::identity();
        assert_relative_eq!(sym_min_eigenvalue(&m), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sym_min_eigenvalue_known() {
        let m = SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::SVector::from([1.0_f64, 2.0, 3.0]));
        assert_relative_eq!(sym_min_eigenvalue(&m), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_corr_identity() {
        let m = SMatrix::<Real, 3, 3>::identity();
        let c = nearest_corr_matrix(&m, 100, 1e-10);
        for i in 0..3 {
            assert_relative_eq!(c[(i, i)], 1.0, epsilon = 1e-8);
        }
        assert_relative_eq!(c[(0, 1)], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_nearest_corr_unit_diagonal_preserved() {
        let mut m = SMatrix::<Real, 4, 4>::identity() * 2.0;
        m[(0, 1)] = 0.3;
        m[(1, 0)] = 0.3;
        m[(2, 3)] = 0.1;
        m[(3, 2)] = 0.1;
        let c = nearest_corr_matrix(&m, 200, 1e-10);
        for i in 0..4 {
            assert_relative_eq!(c[(i, i)], 1.0, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_nearest_corr_pd() {
        let m =
            SMatrix::<Real, 3, 3>::from_row_slice(&[1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0]);
        let c = nearest_corr_matrix(&m, 200, 1e-10);
        let min_ev = sym_min_eigenvalue(&c);
        assert!(
            min_ev > 0.0,
            "result must be PD, min_eigenvalue = {}",
            min_ev
        );
    }

    #[test]
    fn test_sym_sqrt_identity() {
        let m = SMatrix::<Real, 3, 3>::identity();
        let s = sym_sqrt(&m);
        let diff = (s - SMatrix::<Real, 3, 3>::identity()).norm();
        assert!(diff < 1e-13, "sqrt(I) != I: {:.2e}", diff);
    }

    #[test]
    fn test_sym_sqrt_roundtrip() {
        let m =
            SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::SVector::from([4.0_f64, 9.0, 16.0]));
        let s = sym_sqrt(&m);
        let expected =
            SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::SVector::from([2.0_f64, 3.0, 4.0]));
        let diff = (s - expected).norm();
        assert!(diff < 1e-12, "sym_sqrt diagonal case failed: {:.2e}", diff);
    }

    #[test]
    fn test_sym_sqrt_sq_is_original() {
        let m =
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let s = sym_sqrt(&m);
        let m2 = s * s;
        let diff = (m2 - m).norm();
        assert!(diff < 1e-12, "sym_sqrt^2 != m: {:.2e}", diff);
    }

    #[test]
    fn test_sym_sqrt_inv_roundtrip() {
        let m =
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let s = sym_sqrt(&m);
        let s_inv = sym_sqrt_inv(&m);
        let prod = s * s_inv;
        let diff = (prod - SMatrix::<Real, 3, 3>::identity()).norm();
        assert!(diff < 1e-12, "sqrt * sqrt_inv != I: {:.2e}", diff);
    }

    #[test]
    fn test_sym_exp_log_roundtrip() {
        let m =
            SMatrix::<Real, 3, 3>::from_row_slice(&[2.0, 0.5, 0.0, 0.5, 1.5, 0.3, 0.0, 0.3, 1.0]);
        let log_m = sym_log(&m);
        let m2 = sym_exp(&log_m);
        let diff = (m2 - m).norm();
        assert!(diff < 1e-12, "exp(log(m)) != m: {:.2e}", diff);
    }

    #[test]
    fn test_sym_inv_identity() {
        let m = SMatrix::<Real, 3, 3>::identity();
        let inv = sym_inv(&m);
        let diff = (inv - m).norm();
        assert!(diff < 1e-13, "inv(I) != I: {:.2e}", diff);
    }

    #[test]
    fn test_sym_inv_product_is_identity() {
        let m =
            SMatrix::<Real, 3, 3>::from_row_slice(&[4.0, 2.0, 0.0, 2.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let inv = sym_inv(&m);
        let prod = m * inv;
        let diff = (prod - SMatrix::<Real, 3, 3>::identity()).norm();
        assert!(diff < 1e-12, "m * inv(m) != I: {:.2e}", diff);
    }
}
