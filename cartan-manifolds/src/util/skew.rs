// ~/cartan/cartan-manifolds/src/util/skew.rs

//! Skew-symmetrization utilities for matrix manifolds.
//!
//! A matrix Ω is **skew-symmetric** (also called antisymmetric) if Ω^T = -Ω.
//! The space of N×N skew-symmetric matrices is the Lie algebra so(N) — the
//! tangent space to SO(N) at the identity. Understanding and projecting onto
//! this subspace is fundamental for SO(N) and SE(N) manifold operations.
//!
//! ## Mathematical background
//!
//! For any square matrix A, its unique decomposition into symmetric and
//! skew-symmetric parts is:
//!
//! ```text
//! A = sym(A) + skew(A)
//! sym(A)  = (A + A^T) / 2        (symmetric part)
//! skew(A) = (A - A^T) / 2        (skew-symmetric part)
//! ```
//!
//! This is the orthogonal projection onto so(N) under the Frobenius inner
//! product `<X, Y> = tr(X^T Y)`.
//!
//! ## Key properties
//!
//! - `skew(skew(A)) = skew(A)` (idempotent projection)
//! - `(skew(A))^T = -skew(A)` (result is genuinely skew-symmetric)
//! - All diagonal entries of a skew-symmetric matrix are zero
//! - For N=3: so(3) is 3-dimensional, identified with R^3 via the hat map:
//!   `[0, -z, y; z, 0, -x; -y, x, 0]` ↔ `(x, y, z)` ∈ R^3

use cartan_core::Real;
use nalgebra::SMatrix;

/// Project a square matrix onto the skew-symmetric subspace so(N).
///
/// Returns `(A - A^T) / 2`, which is the orthogonal projection of A onto
/// the Lie algebra so(N) ⊂ gl(N) under the Frobenius inner product.
///
/// # Mathematical guarantee
///
/// The output satisfies `skew(A)^T = -skew(A)` exactly (up to floating-point
/// rounding on individual entries). Diagonal entries are identically 0.0 by
/// construction (since a_{ii} - a_{ii} = 0).
///
/// # Arguments
///
/// * `a` — any N×N matrix; need not be square-matrix with any special structure.
///
/// # Returns
///
/// The skew-symmetric part `(A - A^T) / 2` as an owned matrix.
///
/// # Example
///
/// ```text
/// skew([[1,2],[3,4]]) = ([[1,2],[3,4]] - [[1,3],[2,4]]) / 2
///                     = [[0,-1],[1,0]] / 2 * 2 = [[0,-0.5],[0.5,0]]
/// ```
///
/// # References
///
/// - do Carmo (1992), §3.1: Lie algebra of a Lie group.
/// - Hall (2015), §2.2: Matrix Lie algebras.
pub fn skew<const N: usize>(a: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    // Compute A - A^T element-wise, then halve.
    // The transpose() call on a stack-allocated SMatrix is O(N^2) and produces
    // a new owned matrix; subtracting and dividing are also O(N^2).
    (a - a.transpose()) * 0.5
}

/// Test whether a matrix is skew-symmetric to within tolerance `tol`.
///
/// A matrix A is skew-symmetric iff A + A^T = 0, i.e. a_{ij} = -a_{ji}
/// for all i, j. This function checks whether the Frobenius norm of `A + A^T`
/// is less than `tol`.
///
/// # Why Frobenius norm?
///
/// The Frobenius norm `||A + A^T||_F = sqrt(sum_{ij} (a_{ij} + a_{ji})^2)`
/// is zero iff every entry satisfies the skew-symmetry condition, making it
/// a natural measure of violation. It is also fast to compute (no eigenvalues).
///
/// # Arguments
///
/// * `a`   — the matrix to test.
/// * `tol` — tolerance threshold; typical values are `1e-10` (strict) or
///   `1e-6` (loose / after numerical operations).
///
/// # Returns
///
/// `true` if `||A + A^T||_F < tol`.
///
/// # Note on diagonal entries
///
/// Diagonal entries satisfy a_{ii} + a_{ii} = 2 * a_{ii}, so any nonzero
/// diagonal contributes `2 * |a_{ii}|` to the norm. If A is exactly
/// skew-symmetric all diagonals are 0.
pub fn is_skew<const N: usize>(a: &SMatrix<Real, N, N>, tol: Real) -> bool {
    // Sum = A + A^T; measure how far from zero.
    // We use the Frobenius norm: nalgebra's .norm() on matrices is the F-norm.
    let sum = a + a.transpose(); // A + A^T
    sum.norm() < tol             // ||A + A^T||_F < tol ↔ A is (approximately) skew
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;

    // Tolerance used for "exact" floating-point comparisons.
    // We expect only roundoff errors (≈ machine epsilon ≈ 2.2e-16 for f64).
    const TOL: Real = 1e-14;

    /// Test that `skew` correctly extracts the skew-symmetric part of a general
    /// 3×3 matrix with distinct entries.
    ///
    /// For A = [[a,b,c],[d,e,f],[g,h,i]], the skew part is:
    ///   skew(A) = [[0, (b-d)/2, (c-g)/2],
    ///              [(d-b)/2, 0, (f-h)/2],
    ///              [(g-c)/2, (h-f)/2, 0]]
    #[test]
    fn test_skew_projection() {
        // Use a matrix with clearly distinct entries so we can verify by hand.
        #[rustfmt::skip]
        let a = SMatrix::<Real, 3, 3>::from_row_slice(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);

        let s = skew(&a);

        // --- Check: s^T = -s (skew-symmetry definition) ---
        // The Frobenius norm of (s + s^T) must be below TOL.
        assert!(
            is_skew(&s, TOL),
            "skew(A) is not skew-symmetric: ||s + s^T||_F = {}",
            (s.clone() + s.transpose()).norm()
        );

        // --- Check diagonal is exactly zero ---
        // For any skew-symmetric matrix, a_{ii} = -a_{ii} implies a_{ii} = 0.
        for i in 0..3 {
            assert!(
                s[(i, i)].abs() < TOL,
                "diagonal entry ({},{}) = {} should be 0",
                i, i, s[(i, i)]
            );
        }

        // --- Check specific off-diagonal values by hand ---
        // s[0,1] = (a[0,1] - a[1,0]) / 2 = (2 - 4) / 2 = -1
        assert!(
            (s[(0, 1)] - (-1.0_f64)).abs() < TOL,
            "s[0,1] expected -1.0, got {}",
            s[(0, 1)]
        );
        // s[1,0] = (a[1,0] - a[0,1]) / 2 = (4 - 2) / 2 = 1
        assert!(
            (s[(1, 0)] - 1.0_f64).abs() < TOL,
            "s[1,0] expected 1.0, got {}",
            s[(1, 0)]
        );
        // s[0,2] = (3 - 7) / 2 = -2
        assert!(
            (s[(0, 2)] - (-2.0_f64)).abs() < TOL,
            "s[0,2] expected -2.0, got {}",
            s[(0, 2)]
        );
        // s[1,2] = (6 - 8) / 2 = -1
        assert!(
            (s[(1, 2)] - (-1.0_f64)).abs() < TOL,
            "s[1,2] expected -1.0, got {}",
            s[(1, 2)]
        );

        // --- Check that the output has the correct is_skew result ---
        assert!(is_skew(&s, TOL), "is_skew should return true for skew(A)");
    }

    /// Test that `skew` is idempotent: skew(skew(A)) = skew(A).
    ///
    /// This is the defining property of a projection. If Ω is already
    /// skew-symmetric, then (Ω - Ω^T)/2 = (Ω - (-Ω))/2 = Ω.
    #[test]
    fn test_skew_idempotent() {
        // Build an already-skew-symmetric 3×3 matrix (the so(3) hat map of (1,2,3)).
        // hat([x,y,z]) = [[0,-z,y],[z,0,-x],[-y,x,0]]
        #[rustfmt::skip]
        let omega = SMatrix::<Real, 3, 3>::from_row_slice(&[
             0.0, -3.0,  2.0,
             3.0,  0.0, -1.0,
            -2.0,  1.0,  0.0,
        ]);

        // Verify that omega is indeed skew-symmetric.
        assert!(
            is_skew(&omega, TOL),
            "Test setup error: omega is not skew-symmetric"
        );

        // Apply skew projection: result should equal omega.
        let projected = skew(&omega);

        // Check elementwise: ||skew(omega) - omega||_F < TOL
        let diff = projected - omega;
        assert!(
            diff.norm() < TOL,
            "skew is not idempotent: ||skew(skew_matrix) - skew_matrix||_F = {}",
            diff.norm()
        );
    }

    /// Test `is_skew` correctly rejects a non-skew matrix.
    ///
    /// A symmetric matrix (A^T = A) satisfies A + A^T = 2A ≠ 0 unless A = 0,
    /// so `is_skew` must return false for any nonzero symmetric matrix.
    #[test]
    fn test_is_skew_rejects_symmetric() {
        // Identity matrix: symmetric but not skew (I + I^T = 2I ≠ 0).
        let id = SMatrix::<Real, 3, 3>::identity();
        assert!(
            !is_skew(&id, TOL),
            "is_skew should return false for the identity matrix"
        );
    }

    /// Test `is_skew` accepts the zero matrix (trivially skew-symmetric).
    #[test]
    fn test_is_skew_accepts_zero() {
        let zero = SMatrix::<Real, 3, 3>::zeros();
        assert!(
            is_skew(&zero, TOL),
            "is_skew should return true for the zero matrix"
        );
    }
}
