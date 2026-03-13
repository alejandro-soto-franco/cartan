// ~/cartan/cartan/tests/common/approx.rs
#![allow(dead_code)]

//! Approximate equality helpers for manifold testing.
//!
//! These provide better error messages than raw float comparison,
//! showing the actual vs expected values and the tolerance.

use nalgebra::SVector;
use cartan_core::Real;

/// Assert two scalars are approximately equal.
///
/// Panics with a detailed message showing the expected value, actual value,
/// the absolute difference, and the tolerance, so failures are easy to diagnose.
pub fn assert_real_eq(actual: Real, expected: Real, tol: Real, context: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff < tol,
        "{}: expected {:.2e}, got {:.2e}, diff {:.2e} > tol {:.2e}",
        context, expected, actual, diff, tol
    );
}

/// Assert two vectors are approximately equal (element-wise, measured by L2 norm).
///
/// Computes the L2 norm of the difference vector and compares it to `tol`.
/// Displays the element arrays on failure for easy debugging.
pub fn assert_vec_eq<const N: usize>(
    actual: &SVector<Real, N>,
    expected: &SVector<Real, N>,
    tol: Real,
    context: &str,
) {
    let diff = (actual - expected).norm();
    assert!(
        diff < tol,
        "{}: vector diff norm {:.2e} > tol {:.2e}\n  actual:   {:?}\n  expected: {:?}",
        context, diff, tol, actual.as_slice(), expected.as_slice()
    );
}

/// Assert a scalar is approximately zero.
///
/// Useful for checking that a quantity that should vanish (e.g. curvature,
/// distance to self, inner product of orthogonal vectors) is numerically zero.
pub fn assert_near_zero(actual: Real, tol: Real, context: &str) {
    assert!(
        actual.abs() < tol,
        "{}: expected ~0, got {:.2e} (tol {:.2e})",
        context, actual, tol
    );
}

/// Assert a scalar is non-negative (up to a small numerical tolerance).
///
/// Used to check positive semi-definiteness of the metric tensor:
/// <v, v>_p >= 0 for all v.
pub fn assert_nonneg(actual: Real, context: &str) {
    assert!(
        actual >= -1e-15,
        "{}: expected non-negative, got {:.2e}",
        context, actual
    );
}
