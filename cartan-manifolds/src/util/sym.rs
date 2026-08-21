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
use nalgebra::SMatrix;

pub(crate) use super::eig::sym_eigenvalues;
use super::eig::{recompose, sym_eigen_s};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Minimum eigenvalue of a symmetric N×N matrix.
///
/// The input is assumed symmetric; only the lower triangle is meaningful.
///
/// Goes through [`sym_eigenvalues`], which skips the eigenvector accumulation
/// that makes up most of the work in a symmetric eigensolver and takes the
/// stack-based Jacobi path at small N. The call here asked for the full
/// decomposition and then read one number off the spectrum.
pub(crate) fn sym_min_eigenvalue<const N: usize>(m: &SMatrix<Real, N, N>) -> Real {
    sym_eigenvalues(m)
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
        //
        // `recompose` scales the columns of V by the clamped eigenvalues and
        // multiplies by V^T. Materialising diag(clamped) and multiplying by it,
        // as this did, costs a second O(N^3) product for the same answer, and
        // the `DMatrix` round trip allocated four times per iteration against
        // a loop that runs up to `max_iter` times.
        let y = x + delta_s;
        let (v, d) = sym_eigen_s(&y);
        let clamped = d.map(|e| if e < MIN_EIG { MIN_EIG } else { e });
        let x_pd = recompose::<N>(&v, &clamped);
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
    let inv_root = d.map(|x| {
        if x > 1e-14 {
            1.0 / x.sqrt()
        } else {
            1.0 / 1e-7
        }
    });
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
