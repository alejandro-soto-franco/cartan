// ~/cartan/cartan-manifolds/src/util/sym.rs

//! Utilities for symmetric positive definite matrices.
//!
//! These are internal helpers used by [`Corr<N>`] and future SPD-family manifolds.
//!
//! ## Contents
//!
//! - [`sym_min_eigenvalue`]: minimum eigenvalue of a symmetric matrix.
//!
//! - [`nearest_corr_matrix`]: nearest correlation matrix to an arbitrary symmetric
//!   matrix in the Frobenius norm. Uses Higham's (2002) alternating projections
//!   algorithm: alternately project onto the PD cone and the unit-diagonal affine
//!   subspace until convergence.
//!
//! - [`sym_symmetrize`]: (A + A^T) / 2. Enforces exact symmetry after drift.
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

use cartan_core::Real;
use nalgebra::{DMatrix, DVector, SMatrix};

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
pub(crate) fn sym_symmetrize<const N: usize>(
    a: &SMatrix<Real, N, N>,
) -> SMatrix<Real, N, N> {
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
        let clamped: DVector<Real> =
            eigen.eigenvalues.map(|v| if v < MIN_EIG { MIN_EIG } else { v });
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
        let m = SMatrix::<Real, 3, 3>::from_diagonal(
            &nalgebra::SVector::from([1.0_f64, 2.0, 3.0]),
        );
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
        let m = SMatrix::<Real, 3, 3>::from_row_slice(&[
            1.0, 0.9, 0.9, 0.9, 1.0, 0.9, 0.9, 0.9, 1.0,
        ]);
        let c = nearest_corr_matrix(&m, 200, 1e-10);
        let min_ev = sym_min_eigenvalue(&c);
        assert!(min_ev > 0.0, "result must be PD, min_eigenvalue = {}", min_ev);
    }
}
