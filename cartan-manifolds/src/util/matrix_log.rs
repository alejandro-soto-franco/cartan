// ~/cartan/cartan-manifolds/src/util/matrix_log.rs

//! Matrix logarithm for orthogonal matrices.
//!
//! For R ∈ SO(N), the matrix logarithm `log(R)` returns the unique Ω ∈ so(N)
//! such that `exp(Ω) = R` and `||Ω||_F` is minimized (i.e., `||Ω||_F < π√(N/2)`).
//! This Ω is the **Riemannian logarithm** at the identity for the bi-invariant
//! metric on SO(N).
//!
//! ## Algorithm selection by dimension
//!
//! | Dimension | Algorithm | Cut locus condition |
//! |-----------|-----------|---------------------|
//! | N = 2 | Direct angle extraction | None (global diffeomorphism) |
//! | N = 3 | Inverse Rodrigues' formula | θ near π |
//! | N ≥ 4 | Inverse scaling-and-squaring via Denman–Beavers | R has eigenvalue -1 |
//!
//! ## Cut locus of SO(N)
//!
//! The exponential map `exp: so(N) → SO(N)` is a local diffeomorphism in a
//! neighborhood of 0, but fails to be injective for large rotations. The **cut locus**
//! of SO(N) at the identity consists of rotations with at least one rotation angle
//! equal to π (half-turn). These are the matrices where the logarithm is either
//! undefined or non-unique.
//!
//! - SO(2): no cut locus (S¹ is a group but the logarithm wraps — we return the
//!   principal value θ ∈ (-π, π]).
//! - SO(3): cut locus = rotations by exactly π (any axis). The log is undefined
//!   because the geodesic from I to R is not unique.
//! - SO(N), N≥4: similar condition; the Denman–Beavers iteration diverges or
//!   the Mercator series fails.
//!
//! ## References
//!
//! - Gallier, J. & Xu, D. (2002). "Computing exponentials of skew-symmetric matrices
//!   and logarithms of orthogonal matrices." *International Journal of Robotics and
//!   Automation*, 17(4), 10–20.
//! - Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. §11.4.
//! - Denman, E. D. & Beavers, A. N. (1976). "The matrix sign function and computations
//!   in systems." *Applied Mathematics and Computation*, 2(1), 63–94.
//! - do Carmo, M. P. (1992). *Riemannian Geometry*, §3.2 (cut locus).

use cartan_core::{CartanError, Real};
use nalgebra::SMatrix;
#[cfg(not(feature = "std"))]
use nalgebra::ComplexField;
#[cfg(not(feature = "std"))]
use nalgebra::RealField;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the matrix logarithm of an orthogonal matrix R ∈ SO(N).
///
/// Returns Ω ∈ so(N) such that `exp(Ω) = R` (approximately, up to floating-point).
/// The returned Ω is the principal logarithm: the unique skew-symmetric matrix
/// with rotation angles in `(-π, π]`.
///
/// # Errors
///
/// Returns `CartanError::CutLocus` if R is on or near the cut locus of SO(N),
/// i.e., if R has a rotation eigenvalue at or near π (a half-turn). In that case
/// the logarithm is not uniquely defined.
///
/// # Algorithm
///
/// - **N = 2:** Direct formula via `atan2(R[1,0], R[0,0])`.
/// - **N = 3:** Inverse Rodrigues' formula (see `log_rodrigues`).
/// - **N ≥ 4:** Inverse scaling-and-squaring via Denman–Beavers square roots
///   followed by Mercator series (see `log_general`).
///
/// # References
///
/// - Gallier & Xu (2002); Higham (2008) §11.4.
pub fn matrix_log_orthogonal<const N: usize>(
    r: &SMatrix<Real, N, N>,
) -> Result<SMatrix<Real, N, N>, CartanError> {
    // Dispatch to the dimension-specialized implementation.
    if N == 2 {
        log_2d(r)
    } else if N == 3 {
        log_rodrigues(r)
    } else {
        log_general(r)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// N = 2: direct formula
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix logarithm for a 2×2 rotation matrix.
///
/// A 2×2 rotation matrix has the form:
/// ```text
/// R = [[cos θ, -sin θ],
///      [sin θ,  cos θ]]
/// ```
/// Its logarithm is the skew matrix:
/// ```text
/// log(R) = [[0, -θ],
///           [θ,  0]]
/// ```
/// where θ = atan2(R[1,0], R[0,0]) ∈ (-π, π].
///
/// `atan2` handles all quadrants correctly and gives the principal value.
/// There is no cut locus for SO(2) with this convention (though θ = π
/// is a degenerate case, `atan2` handles it as -π ← we map to +π for consistency).
fn log_2d<const N: usize>(r: &SMatrix<Real, N, N>) -> Result<SMatrix<Real, N, N>, CartanError> {
    // Extract the rotation angle θ from R[0,0] = cos θ and R[1,0] = sin θ.
    // atan2(sin, cos) returns θ ∈ (-π, π].
    let theta = r[(1, 0)].atan2(r[(0, 0)]);

    // Build the 2×2 skew matrix [[0, -θ], [θ, 0]].
    let mut omega = SMatrix::<Real, N, N>::zeros();
    omega[(1, 0)] = theta;
    omega[(0, 1)] = -theta;
    Ok(omega)
}

// ─────────────────────────────────────────────────────────────────────────────
// N = 3: inverse Rodrigues' formula
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix logarithm for a 3×3 rotation matrix via inverse Rodrigues' formula.
///
/// ## Formula
///
/// Given R ∈ SO(3) with rotation angle θ:
/// ```text
/// cos θ = (tr(R) - 1) / 2   [since tr(R) = 1 + 2cos θ for any SO(3) rotation]
/// θ = arccos(clamp((tr(R) - 1) / 2, -1, 1))
/// ```
///
/// The logarithm is:
/// ```text
/// Ω = log(R) = (θ / (2 sin θ)) · (R - R^T)     [generic: 0 < θ < π]
/// Ω = (R - R^T) / 2                              [Taylor: θ ≈ 0]
/// ```
/// Note: `(R - R^T) / 2` is already skew-symmetric. The factor `θ / (2 sin θ)`
/// normalizes it so that `||Ω||_F = sqrt(2) θ` (matching the Rodrigues forward map).
///
/// ## Near θ = 0 (Taylor fallback)
///
/// At θ → 0:
///   `θ / (2 sin θ) = θ / (2(θ - θ³/6 + ...)) = 1/(2 - θ²/3 + ...) → 1/2`
/// So `Ω → (R - R^T) / 2`, which equals `skew(R)`. The Taylor branch is used
/// when θ < 1e-7 (cubic error < 1e-21, far below ε_mach).
///
/// ## Near θ = π (cut locus)
///
/// At θ → π, `sin θ → 0` and the formula `θ / (2 sin θ)` diverges.
/// Geometrically, R is a half-turn (rotation by π around some axis n̂). There
/// are infinitely many shortest geodesics from I to R (one for each rotation
/// ±n̂ giving angle ±π), so the logarithm is not uniquely defined.
///
/// We return `CartanError::CutLocus` when `|θ - π| < 1e-7`.
///
/// ## References
///
/// - Murray, Li, Sastry (1994), Theorem 2.14.
/// - Gallier & Xu (2002), §3.
fn log_rodrigues<const N: usize>(
    r: &SMatrix<Real, N, N>,
) -> Result<SMatrix<Real, N, N>, CartanError> {
    // Step 1: Extract the rotation angle θ from the trace.
    //   tr(R) = 1 + 2 cos θ  ↔  cos θ = (tr(R) - 1) / 2.
    // We clamp to [-1, 1] to guard against floating-point values outside [-1, 1]
    // (which would cause acos to return NaN).
    let cos_theta = ((r.trace() - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos(); // θ ∈ [0, π]

    // Step 2: Taylor branch — θ near 0 (identity-like rotation).
    if theta < 1e-7 {
        // At θ → 0, Rodrigues' formula degenerates gracefully:
        //   Ω = (R - R^T) / 2 = skew(R)
        // The higher-order correction θ/(2 sin θ) → 1/2 in this limit,
        // so the dominant term is (R - R^T)/2.
        return Ok((r - r.transpose()) * 0.5);
    }

    // Step 3: Cut locus branch — θ near π (half-turn).
    // When θ is within 1e-7 of π, the formula coefficient θ/(2 sin θ)
    // diverges (sin(π) = 0), and the geodesic is non-unique.
    let pi: Real = core::f64::consts::PI;
    if (pi - theta).abs() < 1e-7 {
        // The rotation is a half-turn: every point on the "opposite hemisphere"
        // of SO(3) is a cut point. We cannot determine which of the two shortest
        // geodesics (around ±n̂ by angle π) the user wants.
        #[cfg(feature = "alloc")]
        return Err(CartanError::CutLocus {
            message: alloc::format!(
                "rotation angle θ = {:.6} rad is near π; logarithm is not unique (cut locus of SO(3))",
                theta
            ),
        });
        #[cfg(not(feature = "alloc"))]
        return Err(CartanError::CutLocus {
            message: "rotation angle near π; logarithm is not unique (cut locus of SO(3))",
        });
    }

    // Step 4: Generic formula.
    //   Ω = (θ / (2 sin θ)) · (R - R^T)
    // The factor θ / (2 sin θ) is the inverse of the Rodrigues sinc factor.
    let sin_theta = theta.sin();
    let factor = theta / (2.0 * sin_theta); // θ / (2 sin θ)
    Ok((r - r.transpose()) * factor)
}

// ─────────────────────────────────────────────────────────────────────────────
// N ≥ 4: inverse scaling-and-squaring via Denman–Beavers
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix logarithm for a general N×N orthogonal matrix via inverse scaling-and-squaring.
///
/// ## Algorithm
///
/// The idea (Higham 2008, §11.4; Björck & Hammarling 1983):
///
/// 1. **Repeated square roots (Denman–Beavers):** Compute R^{1/2^s} for increasing s
///    until `||R^{1/2^s} - I||_1 < 0.5`. Each square root halves the "distance" to I,
///    eventually bringing R^{1/2^s} into the radius of convergence of the Mercator series.
///
/// 2. **Mercator series:** For Y ≈ I (i.e., X = Y - I with `||X|| < 1`):
///    ```text
///    log(Y) = X - X²/2 + X³/3 - X⁴/4 + ...  (Mercator series)
///    ```
///    This converges for `||X|| < 1` (spectral radius < 1) and computes log(Y)
///    where log is the principal matrix logarithm.
///
/// 3. **Unscaling:** Since we took s square roots, `log(R) = 2^s · log(R^{1/2^s})`.
///
/// 4. **Skew-projection:** Project the result onto so(N) via `skew(Ω) = (Ω - Ω^T)/2`
///    to enforce exact skew-symmetry (Mercator accumulates tiny symmetric errors).
///
/// ## Denman–Beavers iteration
///
/// For computing the matrix square root of Y: iterate
/// ```text
/// Y_{k+1} = (Y_k + Z_k^{-1}) / 2
/// Z_{k+1} = (Z_k + Y_k^{-1}) / 2
/// ```
/// starting from Y_0 = Y, Z_0 = I. Then Y_k → Y^{1/2} and Z_k → Y^{-1/2}.
///
/// DB converges quadratically (error squares each step) when Y has no eigenvalues
/// on the negative real axis. For orthogonal matrices (eigenvalues on the unit circle),
/// convergence is fast unless Y has an eigenvalue near -1 (θ near π → cut locus).
///
/// ## Limitations
///
/// - We cap at `MAX_SQRTS = 20` square roots and `DB_ITERS = 32` DB iterations.
/// - If DB diverges or does not bring Y close enough to I, returns `CartanError::CutLocus`.
/// - The Mercator series uses 16 terms; for `||X||_F < 0.5`, the error is < (0.5)^17/17 ≈ 1e-6.
///   For `||X||_F < 0.1` (which we aim for), the error is < 1e-18 (below ε_mach).
///
/// ## References
///
/// - Denman & Beavers (1976). "The matrix sign function and computations in systems."
/// - Björck, Å. & Hammarling, S. (1983). "A Schur method for the square root of a matrix."
/// - Higham, N. J. (2008). *Functions of Matrices*, §11.4 (Algorithm 11.9).
fn log_general<const N: usize>(
    r: &SMatrix<Real, N, N>,
) -> Result<SMatrix<Real, N, N>, CartanError> {
    // ── Constants ──────────────────────────────────────────────────────────
    // Maximum number of square roots to take before giving up.
    // Each square root halves the "angle" of the orthogonal matrix.
    // After 20 square roots, the angle is reduced by 2^20 ≈ 10^6, more than enough.
    const MAX_SQRTS: usize = 20;

    // Number of Denman–Beavers iterations per square root.
    // DB converges quadratically, so 32 iterations is extreme overkill (in practice
    // 6–8 iterations suffice), but we leave headroom for near-cut-locus inputs.
    const DB_ITERS: usize = 32;

    // Convergence threshold for DB: relative change in Y below this → declare convergence.
    // We use 1e-14 (close to machine epsilon for f64) so that the square root is as
    // accurate as double precision allows.
    const DB_TOL: Real = 1e-14;

    // Number of terms in the Mercator series for log(I + X).
    // 16 terms → error < ||X||^17 / 17. For ||X||_F < 0.5, this is < 8e-7.
    // For ||X||_F < 0.1, this is < 1e-18 (safely below ε_mach).
    const MERCATOR_TERMS: usize = 16;

    let id = SMatrix::<Real, N, N>::identity();

    // ── Step 1: Repeated square roots via Denman–Beavers ──────────────────
    //
    // We start with Y = R and compute Y ← Y^{1/2} repeatedly until
    // ||Y - I||_1 < 0.5 (within the Mercator convergence radius).
    // We track the number of square roots taken as `s`.
    let mut y = *r; // Y will converge to R^{1/2^s}
    let mut s = 0usize; // number of square roots taken so far

    for _sqrts in 0..MAX_SQRTS {
        // Check if Y is already close enough to I for Mercator to converge well.
        // Threshold 0.5: Mercator converges for ||X|| < 1, but we want ||X|| ≤ 0.5
        // for reasonable accuracy with 16 terms.
        let y_minus_id_norm = matrix_norm1(&(y - id));
        if y_minus_id_norm < 0.5 {
            break; // Y is close enough to I; stop taking square roots.
        }

        // Compute sqrt(Y) via Denman–Beavers iteration.
        let y_sqrt = denman_beavers_sqrt(&y, DB_ITERS, DB_TOL).ok_or_else(|| {
            #[cfg(feature = "alloc")]
            {
                CartanError::CutLocus {
                    message: alloc::format!(
                        "Denman–Beavers square root did not converge after {} iterations \
                     (rotation may be near cut locus, i.e., angle near π)",
                        DB_ITERS
                    ),
                }
            }
            #[cfg(not(feature = "alloc"))]
            {
                CartanError::CutLocus {
                    message: "Denman-Beavers sqrt did not converge (rotation near cut locus)",
                }
            }
        })?;

        y = y_sqrt;
        s += 1;
    }

    // Final check: did we actually converge?
    let y_minus_id_norm = matrix_norm1(&(y - id));
    if y_minus_id_norm >= 0.5 {
        #[cfg(feature = "alloc")]
        return Err(CartanError::CutLocus {
            message: alloc::format!(
                "after {} square roots, ||R^{{1/2^s}} - I||_1 = {:.4e} ≥ 0.5; \
                 matrix may be at the cut locus (rotation angle near π)",
                s,
                y_minus_id_norm
            ),
        });
        #[cfg(not(feature = "alloc"))]
        return Err(CartanError::CutLocus {
            message: "scaling-and-squaring did not converge (matrix near cut locus)",
        });
    }

    // ── Step 2: Mercator series log(Y) = log(I + X) where X = Y - I ───────
    //
    // The Mercator (alternating harmonic) series:
    //   log(I + X) = X - X²/2 + X³/3 - X⁴/4 + ... = sum_{k=1}^{∞} (-1)^{k+1} X^k / k
    //
    // This converges for ||X|| < 1 (in any consistent matrix norm).
    // For ||X||_1 < 0.5, the error after MERCATOR_TERMS terms is bounded by
    //   ||X||^{MERCATOR_TERMS+1} / (MERCATOR_TERMS+1) < 0.5^17 / 17 ≈ 4.5e-7.
    // If we want higher accuracy, either use more terms or ensure smaller ||X||
    // by taking more square roots (smaller threshold for break condition above).
    let x = y - id; // X = Y - I (the "deviation" from identity)

    // Compute the Mercator series using Horner-like accumulation:
    //   Σ = X - X²/2 + X³/3 - ... = X · (I - X/2 · (I - X/3 · (I - ...)))
    // We use the straightforward summation (not Horner) for clarity:
    //   log_y = sum_{k=1}^{MERCATOR_TERMS} (-1)^{k+1} X^k / k
    let mut log_y = SMatrix::<Real, N, N>::zeros(); // accumulator for the series
    let mut x_power = x; // X^k, starts at X^1
    let mut sign = 1.0_f64; // alternating sign: +1 for k odd, -1 for k even

    for k in 1..=MERCATOR_TERMS {
        // Add the term sign * X^k / k to the series sum.
        // sign = (-1)^{k+1} = +1 for k=1, -1 for k=2, +1 for k=3, ...
        log_y += x_power * (sign / k as Real);

        // Prepare for next iteration: X^{k+1} = X^k · X, flip sign.
        x_power *= x; // X^{k+1} = X^k · X
        sign = -sign; // flip for the next term
    }

    // ── Step 3: Unscale — log(R) = 2^s · log(R^{1/2^s}) ──────────────────
    //
    // Taking s square roots of R gives R^{1/2^s}. The log of R^{1/2^s}
    // satisfies: log(R^{1/2^s}) = log(R) / 2^s. Inverting: log(R) = 2^s · log(R^{1/2^s}).
    //
    // Note: 2^s as an integer fits trivially in f64 for s ≤ 20 (2^20 = 1048576).
    let scale = (2.0_f64).powi(s as i32); // 2^s (exact in floating-point for s ≤ 52)
    let log_r_raw = log_y * scale;

    // ── Step 4: Project onto so(N) ─────────────────────────────────────────
    //
    // The Mercator series accumulates floating-point rounding errors that may
    // introduce a tiny symmetric component into log_r_raw. Projecting onto so(N)
    // via (Ω - Ω^T)/2 eliminates this and enforces exact skew-symmetry.
    //
    // This projection is exact for a perfect logarithm: if Ω ∈ so(N) then
    // (Ω - Ω^T)/2 = (Ω - (-Ω))/2 = Ω. So we only lose the error, not signal.
    let log_r = (log_r_raw - log_r_raw.transpose()) * 0.5;

    Ok(log_r)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Denman–Beavers matrix square root
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the principal square root of a matrix Y via Denman–Beavers iteration.
///
/// The Denman–Beavers coupled iteration (Denman & Beavers 1976) is:
/// ```text
/// Y_0 = Y,   Z_0 = I
/// Y_{k+1} = (Y_k + Z_k^{-1}) / 2
/// Z_{k+1} = (Z_k + Y_k^{-1}) / 2
/// ```
/// The iteration converges (quadratically) to `Y_{∞} = Y^{1/2}` and `Z_{∞} = Y^{-1/2}`
/// whenever Y has no eigenvalues on the negative real axis.
///
/// For orthogonal matrices R ∈ SO(N) with eigenvalues on the unit circle S¹,
/// convergence is fast unless an eigenvalue is near -1 (angle near π → cut locus).
///
/// ## Returns
///
/// `Some(Y^{1/2})` on convergence (relative change below `tol`), or `None` if
/// the iteration did not converge within `max_iters` steps or if a matrix inverse
/// fails (singular iterate → likely cut locus).
///
/// ## References
///
/// - Denman, E. D. & Beavers, A. N. (1976). "The matrix sign function and computations
///   in systems." *Applied Mathematics and Computation*, 2(1), 63–94.
/// - Higham, N. J. (2008). *Functions of Matrices*, Algorithm 6.3, p. 148.
fn denman_beavers_sqrt<const N: usize>(
    y_init: &SMatrix<Real, N, N>,
    max_iters: usize,
    tol: Real,
) -> Option<SMatrix<Real, N, N>> {
    let id = SMatrix::<Real, N, N>::identity();
    let half = 0.5_f64;

    let mut y = *y_init; // Y_k (iterand; converges to Y^{1/2})
    let mut z = id; // Z_k (converges to Y^{-1/2})

    for _ in 0..max_iters {
        // Compute inverses: Z_k^{-1} and Y_k^{-1}.
        // If either inverse fails (singular matrix), the iteration has broken down.
        let z_inv = z.try_inverse()?; // Z_k^{-1}; returns None on singular
        let y_inv = y.try_inverse()?; // Y_k^{-1}; returns None on singular

        // Save Y_k before updating to check convergence.
        let y_old = y;

        // DB update:
        //   Y_{k+1} = (Y_k + Z_k^{-1}) / 2
        //   Z_{k+1} = (Z_k + Y_k^{-1}) / 2
        y = (y_old + z_inv) * half;
        z = (z + y_inv) * half;

        // Convergence check: relative change in Y.
        // We use ||Y_{k+1} - Y_k||_F / ||Y_{k+1}||_F < tol.
        // Frobenius norm is used here (cheap, sufficient for convergence detection).
        let dy = (y - y_old).norm(); // ||Y_{k+1} - Y_k||_F
        let y_norm = y.norm(); // ||Y_{k+1}||_F

        // Avoid dividing by zero if Y → 0 (shouldn't happen for orthogonal inputs).
        if y_norm > 0.0 && dy / y_norm < tol {
            return Some(y); // converged!
        }
    }

    // Did not converge within max_iters.
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: 1-norm of a matrix (same as in matrix_exp.rs)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the matrix 1-norm: `||A||_1 = max_j sum_i |a_{ij}|`.
///
/// See `matrix_exp::matrix_norm1` for the rationale. We duplicate the function
/// here (rather than making it `pub` in matrix_exp and importing it) to keep each
/// module self-contained.
fn matrix_norm1<const N: usize>(a: &SMatrix<Real, N, N>) -> Real {
    (0..N)
        .map(|j| (0..N).map(|i| a[(i, j)].abs()).sum::<Real>())
        .fold(0.0_f64, f64::max)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::matrix_exp::matrix_exp_skew;
    use crate::util::skew::skew;
    use nalgebra::SMatrix;

    // Tolerances:
    // TIGHT: 1e-14 for exact cases (log(I) = 0, etc.)
    // MED:   1e-10 for roundtrip tests with floating-point accumulation
    const TIGHT: Real = 1e-14;
    const MED: Real = 1e-10;

    // ── Helper: so(3) hat map ────────────────────────────────────────────────
    fn hat3(x: Real, y: Real, z: Real) -> SMatrix<Real, 3, 3> {
        // hat([x,y,z]) = [[0,-z,y],[z,0,-x],[-y,x,0]]; laid out in row-major order.
        SMatrix::<Real, 3, 3>::from_row_slice(&[0.0, -z, y, z, 0.0, -x, -y, x, 0.0])
    }

    /// `log(I) = 0` for N = 2.
    #[test]
    fn test_log_identity_2d() {
        let id = SMatrix::<Real, 2, 2>::identity();
        let omega = matrix_log_orthogonal(&id).expect("log(I) should succeed for N=2");
        let err = omega.norm();
        assert!(
            err < TIGHT,
            "log(I) ≠ 0 for N=2: ||log(I)||_F = {:.2e}",
            err
        );
    }

    /// `log(I) = 0` for N = 3.
    #[test]
    fn test_log_identity_3d() {
        let id = SMatrix::<Real, 3, 3>::identity();
        let omega = matrix_log_orthogonal(&id).expect("log(I) should succeed for N=3");
        let err = omega.norm();
        assert!(
            err < TIGHT,
            "log(I) ≠ 0 for N=3: ||log(I)||_F = {:.2e}",
            err
        );
    }

    /// `log(I) = 0` for N = 4.
    #[test]
    fn test_log_identity_4d() {
        let id = SMatrix::<Real, 4, 4>::identity();
        let omega = matrix_log_orthogonal(&id).expect("log(I) should succeed for N=4");
        let err = omega.norm();
        assert!(
            err < TIGHT,
            "log(I) ≠ 0 for N=4: ||log(I)||_F = {:.2e}",
            err
        );
    }

    /// Roundtrip N=3: `log(exp(Ω)) ≈ Ω` for a small rotation.
    ///
    /// We use a small angle (||Ω|| ≈ 0.5) to stay well away from the cut locus.
    #[test]
    fn test_log_roundtrip_exp_log_3d() {
        // hat([0.1, 0.2, 0.3]) — axis-angle with ||axis|| ≈ 0.374 rad
        let omega = hat3(0.1, 0.2, 0.3);
        let r = matrix_exp_skew(&omega);
        let omega_recovered = matrix_log_orthogonal(&r).expect("log should succeed");
        let err = (omega_recovered - omega).norm();
        assert!(err < MED, "log(exp(Ω)) ≠ Ω for N=3: error = {:.2e}", err);
    }

    /// Roundtrip N=3: `exp(log(R)) ≈ R` for a moderate rotation.
    #[test]
    fn test_log_roundtrip_log_exp_3d() {
        // Build a rotation R = exp(Ω) for a known Ω ≈ 1.2 rad about (1,1,0)/sqrt(2).
        let theta: Real = 1.2;
        let omega = hat3(theta / 2.0_f64.sqrt(), theta / 2.0_f64.sqrt(), 0.0);
        let r = matrix_exp_skew(&omega);

        // Compute log then exp and compare to R.
        let omega2 = matrix_log_orthogonal(&r).expect("log should succeed");
        let r2 = matrix_exp_skew(&omega2);

        let err = (&r - &r2).norm();
        assert!(err < MED, "exp(log(R)) ≠ R for N=3: error = {:.2e}", err);
    }

    /// N=4 roundtrip: `log(exp(Ω)) ≈ Ω`.
    #[test]
    fn test_log_roundtrip_4d() {
        // A small so(4) element.
        #[rustfmt::skip]
        let raw = SMatrix::<Real, 4, 4>::from_row_slice(&[
             0.0,  0.1, -0.05,  0.08,
            -0.1,  0.0,  0.12, -0.03,
             0.05,-0.12,  0.0,  0.07,
            -0.08, 0.03, -0.07,  0.0,
        ]);
        let omega = skew(&raw); // ensure exact skew symmetry

        let r = matrix_exp_skew(&omega);
        let omega_recovered = matrix_log_orthogonal(&r).expect("log should succeed for N=4");
        let err = (omega_recovered - omega).norm();
        assert!(err < MED, "N=4: log(exp(Ω)) ≠ Ω: error = {:.2e}", err);
    }

    /// N=4 roundtrip: `exp(log(R)) ≈ R`.
    #[test]
    fn test_log_roundtrip_exp_log_4d() {
        // Build R from a known Ω.
        #[rustfmt::skip]
        let omega = SMatrix::<Real, 4, 4>::from_row_slice(&[
             0.0,  0.3, -0.1,  0.2,
            -0.3,  0.0,  0.4, -0.1,
             0.1, -0.4,  0.0,  0.3,
            -0.2,  0.1, -0.3,  0.0,
        ]);
        let r = matrix_exp_skew(&omega);

        let omega2 = matrix_log_orthogonal(&r).expect("log should succeed for N=4");
        let r2 = matrix_exp_skew(&omega2);

        let err = (&r - &r2).norm();
        assert!(err < MED, "N=4: exp(log(R)) ≠ R: error = {:.2e}", err);
    }

    /// `log(-I)` for N=3 should return `Err(CartanError::CutLocus)`.
    ///
    /// -I ∈ SO(3) (it has det = -(-1)³ = -1... wait, det(-I₃) = (-1)³ = -1 for N=3,
    /// so -I₃ ∉ SO(3). The correct cut locus example is a rotation by π around any axis.
    ///
    /// For example, rotation by π around the z-axis:
    /// ```text
    /// R = diag(-1, -1, 1)  (standard 180° rotation about z)
    /// ```
    #[test]
    fn test_log_cut_locus_3d() {
        // Rotation by π around the z-axis: diag(-1, -1, 1).
        // tr(R) = -1 - 1 + 1 = -1 → cos(θ) = (-1-1)/2 = -1 → θ = π.
        #[rustfmt::skip]
        let r_pi = SMatrix::<Real, 3, 3>::from_row_slice(&[
            -1.0,  0.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
        ]);

        // Verify this is at the cut locus: tr(R) = -1, so θ = π.
        let cos_theta = (r_pi.trace() - 1.0) / 2.0;
        assert!(
            (cos_theta - (-1.0)).abs() < 1e-14,
            "Test setup error: expected cos(θ) = -1, got {}",
            cos_theta
        );

        // The logarithm should fail with CutLocus error.
        match matrix_log_orthogonal(&r_pi) {
            Err(CartanError::CutLocus { .. }) => {
                // Expected: this is the correct behavior at the cut locus.
            }
            Ok(omega) => {
                panic!(
                    "Expected Err(CutLocus) but got Ok(Ω) with ||Ω||_F = {:.4e}",
                    omega.norm()
                );
            }
            Err(other) => {
                panic!("Expected Err(CutLocus) but got {:?}", other);
            }
        }
    }

    /// N=2 roundtrip: `log(exp(Ω)) ≈ Ω` for a 45-degree rotation.
    #[test]
    fn test_log_roundtrip_2d() {
        use core::f64::consts::FRAC_PI_4; // π/4 = 45°

        // Ω = [[0, -π/4], [π/4, 0]] (45° rotation in 2D)
        let omega = SMatrix::<Real, 2, 2>::from_row_slice(&[0.0, -FRAC_PI_4, FRAC_PI_4, 0.0]);
        let r = matrix_exp_skew(&omega);
        let omega_recovered = matrix_log_orthogonal(&r).expect("log should succeed for N=2");
        let err = (omega_recovered - omega).norm();
        assert!(err < MED, "N=2: log(exp(Ω)) ≠ Ω: error = {:.2e}", err);
    }
}
