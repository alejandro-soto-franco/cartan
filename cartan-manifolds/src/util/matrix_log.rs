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
#[cfg(not(feature = "std"))]
use nalgebra::ComplexField;
#[cfg(not(feature = "std"))]
use nalgebra::RealField;
use nalgebra::SMatrix;

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
        log_normal(r)
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
/// where θ = atan2(`R[1,0]`, `R[0,0]`) ∈ (-π, π].
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
// N ≥ 4: the normal-matrix formula
// ─────────────────────────────────────────────────────────────────────────────

/// Matrix logarithm of a general N×N rotation, from one symmetric
/// eigendecomposition.
///
/// ## Formula
///
/// Split `R` into its symmetric and skew parts:
///
/// ```text
/// S = (R + R^T) / 2,   A = (R - R^T) / 2
/// ```
///
/// With `R = exp(Ω)` and `Ω ∈ so(N)` these are the matrix cosine and sine of
/// `Ω`, so `S = cos(Ω)` and `A = sin(Ω)`. A rotation is normal, `R R^T = I`,
/// which makes `S` and `A` commute:
///
/// ```text
/// S A = (R + R^T)(R - R^T)/4 = (R² - R^T²)/4 = (R - R^T)(R + R^T)/4 = A S
/// ```
///
/// In the basis where `Ω` is block diagonal with blocks `[[0, -θ], [θ, 0]]`,
/// `S` is `cos θ` on that block and `A` is the same block scaled by `sin θ`.
/// Multiplying `A` by `θ / sin θ` therefore recovers `Ω` exactly:
///
/// ```text
/// Ω = F A,   F = V diag(θ_k / sin θ_k) V^T,   θ_k = arccos(λ_k(S))
/// ```
///
/// where `S = V diag(λ) V^T`. `F` is a function of `S`, so it commutes with
/// `A`, and a symmetric matrix times a commuting skew matrix is skew: the
/// result lands in `so(N)` by construction.
///
/// At `N = 3` every non-zero `θ_k` is the single rotation angle, `F` collapses
/// to `(θ / sin θ) I`, and the formula is exactly the inverse Rodrigues
/// formula that [`log_rodrigues`] applies directly.
///
/// ## Cost
///
/// One symmetric eigendecomposition and three N×N products. The inverse
/// scaling-and-squaring scheme this replaces took repeated Denman–Beavers
/// square roots, each of which ran up to 32 coupled iterations with two matrix
/// inverses apiece, then summed a 16-term Mercator series. On SO(10) that was
/// two orders of magnitude more arithmetic for a less accurate answer: the
/// Mercator truncation caps the old path at roughly 1e-7 relative error, while
/// this one is limited only by the eigensolver.
///
/// ## Near θ = 0
///
/// `θ / sin θ → 1`, and the ratio is evaluated from its Taylor series below
/// `1e-4` to avoid `0/0`.
///
/// ## Near θ = π (cut locus)
///
/// `sin θ → 0` and the factor diverges. Geometrically `R` has a half-turn in
/// some invariant 2-plane and the shortest geodesic from `I` is not unique, so
/// this returns [`CartanError::CutLocus`], matching the `N = 3` branch.
///
/// ## References
///
/// - Gallier & Xu (2002), §4 (logarithm of a rotation via its invariant planes).
/// - Higham (2008), *Functions of Matrices*, §1.2 (functions of a normal matrix).
fn log_normal<const N: usize>(r: &SMatrix<Real, N, N>) -> Result<SMatrix<Real, N, N>, CartanError> {
    let pi: Real = core::f64::consts::PI;

    // Symmetric part: cos(Ω). Skew part: sin(Ω).
    let s = (r + r.transpose()) * 0.5;
    let a = (r - r.transpose()) * 0.5;

    let (v, lambda) = crate::util::eig::sym_eigen_s(&s);

    // θ_k = arccos(λ_k). Clamping absorbs the roundoff that puts an eigenvalue
    // of a numerically orthogonal matrix a few ulps outside [-1, 1].
    let mut factors = lambda;
    for k in 0..N {
        let theta = factors[k].clamp(-1.0, 1.0).acos();

        if pi - theta < CUT_LOCUS_TOL {
            #[cfg(feature = "alloc")]
            return Err(CartanError::CutLocus {
                message: alloc::format!(
                    "invariant plane with rotation angle θ = {:.6} rad is near π; \
                     logarithm is not unique (cut locus of SO(N))",
                    theta
                ),
            });
            #[cfg(not(feature = "alloc"))]
            return Err(CartanError::CutLocus {
                message: "invariant plane with rotation angle near π; logarithm is not unique",
            });
        }

        factors[k] = theta_over_sin(theta);
    }

    // Ω = F A with F = V diag(θ/sin θ) V^T. `recompose` builds V diag(f) V^T
    // without materialising the diagonal matrix.
    let f = crate::util::eig::recompose::<N>(&v, &factors);
    let omega = f * a;

    // F and A commute, so the product is already skew. The projection removes
    // the asymmetry that roundoff in the eigenvectors leaves behind.
    Ok((omega - omega.transpose()) * 0.5)
}

/// Rotation angle at which the logarithm is declared to sit on the cut locus.
///
/// The `N = 3` branch uses the same figure against the angle read off the
/// trace, so both dimensions report a half-turn at the same distance from π.
const CUT_LOCUS_TOL: Real = 1e-7;

/// `θ / sin θ`, evaluated by series where the quotient is `0/0`.
///
/// The switch is at `θ = 1e-4`, where the series truncated after the `θ⁴` term
/// is accurate to `θ⁶ ≈ 1e-24`, below the resolution of the surrounding
/// arithmetic, and the direct quotient still has `sin θ ≈ 1e-4` in the
/// denominator against a `θ` of the same size, so the cancellation is the
/// whole value.
#[inline]
fn theta_over_sin(theta: Real) -> Real {
    if theta < 1e-4 {
        // θ/sin θ = 1 + θ²/6 + 7θ⁴/360 + O(θ⁶)
        let t2 = theta * theta;
        1.0 + t2 * (1.0 / 6.0 + t2 * (7.0 / 360.0))
    } else {
        theta / theta.sin()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::matrix_exp::matrix_exp_skew;
    use crate::util::skew::skew;
    use nalgebra::{SMatrix, SVector};

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

        let err = (r - r2).norm();
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

        let err = (r - r2).norm();
        assert!(err < MED, "N=4: exp(log(R)) ≠ R: error = {:.2e}", err);
    }

    /// Two Householder reflections, giving a matrix orthogonal to machine
    /// precision with determinant +1.
    ///
    /// Conjugating by an exactly orthogonal `Q` lets a test state the answer
    /// in closed form, so the accuracy of the logarithm is measured on its
    /// own rather than through `matrix_exp_skew`.
    fn householder_pair<const N: usize>(seed: usize) -> SMatrix<Real, N, N> {
        let reflect = |off: usize| -> SMatrix<Real, N, N> {
            let mut v = SVector::<Real, N>::zeros();
            for i in 0..N {
                v[i] = (((i * 31 + off * 17 + seed * 7) % 23) as Real / 23.0) - 0.5 + 0.3;
            }
            let vtv = v.dot(&v);
            SMatrix::<Real, N, N>::identity() - (v * v.transpose()) * (2.0 / vtv)
        };
        reflect(1) * reflect(2)
    }

    /// Block-diagonal rotation and its logarithm, for prescribed angles.
    ///
    /// Angles are laid on the invariant 2-planes in order; a trailing
    /// coordinate on an odd `N` is fixed.
    fn blocks<const N: usize>(angles: &[Real]) -> (SMatrix<Real, N, N>, SMatrix<Real, N, N>) {
        let mut r = SMatrix::<Real, N, N>::identity();
        let mut omega = SMatrix::<Real, N, N>::zeros();
        for (b, &theta) in angles.iter().enumerate().take(N / 2) {
            let i = 2 * b;
            r[(i, i)] = theta.cos();
            r[(i, i + 1)] = -theta.sin();
            r[(i + 1, i)] = theta.sin();
            r[(i + 1, i + 1)] = theta.cos();
            omega[(i, i + 1)] = -theta;
            omega[(i + 1, i)] = theta;
        }
        (r, omega)
    }

    /// The logarithm against a closed-form answer, at N = 4, 6, 8 and 10, over
    /// angles from `1e-8` to within a tenth of a radian of a half-turn.
    ///
    /// The tolerance is 1e-12. Measured against the inverse scaling-and-squaring
    /// path this replaces, on these same cases, the new formula is between
    /// three and four orders more accurate away from the cut locus (4e-16
    /// against 4e-12 at N = 4 with two equal angles of 0.5 rad) and about
    /// three times less accurate within 0.1 rad of it (2.6e-13 against
    /// 9.7e-14), where `arccos` is ill conditioned. Both stay far inside this
    /// tolerance.
    #[test]
    fn test_log_matches_closed_form_across_dimensions_and_angles() {
        fn check<const N: usize>(angles: &[Real]) {
            let q = householder_pair::<N>(3);
            let (r_blocks, omega_blocks) = blocks::<N>(angles);
            let r = q * r_blocks * q.transpose();
            let expected = q * omega_blocks * q.transpose();

            let got =
                matrix_log_orthogonal(&r).expect("log should succeed away from the cut locus");
            let err = (got - expected).norm();
            assert!(
                err < 1e-12,
                "N={N}, angles={angles:?}: log(R) ≠ Ω, error = {err:.3e}"
            );
        }

        // Mixed magnitudes, a near-zero angle, and one close to the cut locus.
        check::<4>(&[1e-8, 3.0]);
        check::<4>(&[0.5, 0.5]);
        check::<6>(&[0.01, 1.2, 3.04]);
        check::<8>(&[1e-6, 0.7, 2.0, 3.04]);
        check::<10>(&[0.3, 0.3, 1.5, 2.9, 1e-7]);
        check::<10>(&[2.5, 2.5, 2.5, 2.5, 2.5]);
    }

    /// Round trip through `exp` then `log` at N = 4, 6, 8 and 10.
    ///
    /// The tolerance bounds the two maps together, `matrix_exp_skew` included.
    /// It stood at 1e-9 while that function scaled its Pade approximant to the
    /// wrong threshold; see `PADE6_THETA` in `matrix_exp`.
    #[test]
    fn test_log_roundtrip_general_dimensions_and_angles() {
        fn check<const N: usize>(scale: Real) {
            // A deterministic skew matrix whose entries vary in sign and size,
            // scaled to sweep the angle range.
            let mut raw = SMatrix::<Real, N, N>::zeros();
            for i in 0..N {
                for j in (i + 1)..N {
                    let e = ((i * 7 + j * 13) % 11) as Real / 11.0 - 0.5;
                    raw[(i, j)] = e * scale;
                    raw[(j, i)] = -e * scale;
                }
            }
            let omega = skew(&raw);

            let r = matrix_exp_skew(&omega);
            let recovered =
                matrix_log_orthogonal(&r).expect("log should succeed away from the cut locus");
            let err = (recovered - omega).norm();
            assert!(
                err < 1e-12,
                "N={N}, scale={scale}: log(exp(Ω)) ≠ Ω, error = {err:.3e}"
            );
        }

        for scale in [1e-6, 0.01, 0.3, 1.0, 1.6] {
            check::<4>(scale);
            check::<6>(scale);
            check::<8>(scale);
            check::<10>(scale);
        }
    }

    /// A rotation acting in one plane only leaves N-2 eigenvalues of the
    /// symmetric part at exactly 1, where `arccos` is at its worst
    /// conditioned. The `θ/sin θ` factor still has to come out as 1 there.
    #[test]
    fn test_log_single_plane_rotation_with_fixed_axes() {
        const N: usize = 6;
        let theta: Real = 1.1;
        let mut r = SMatrix::<Real, N, N>::identity();
        r[(0, 0)] = theta.cos();
        r[(0, 1)] = -theta.sin();
        r[(1, 0)] = theta.sin();
        r[(1, 1)] = theta.cos();

        let omega = matrix_log_orthogonal(&r).expect("log should succeed");

        let mut expected = SMatrix::<Real, N, N>::zeros();
        expected[(0, 1)] = -theta;
        expected[(1, 0)] = theta;

        let err = (omega - expected).norm();
        assert!(err < 1e-13, "single-plane rotation: error = {err:.3e}");
    }

    /// Repeated rotation angles make the eigenvalues of the symmetric part
    /// degenerate, so the eigenvectors within each eigenspace are arbitrary.
    /// The answer must not depend on which basis the solver picks.
    #[test]
    fn test_log_repeated_angles() {
        const N: usize = 8;
        let theta: Real = 0.7;
        let mut r = SMatrix::<Real, N, N>::zeros();
        for b in 0..(N / 2) {
            let i = 2 * b;
            r[(i, i)] = theta.cos();
            r[(i, i + 1)] = -theta.sin();
            r[(i + 1, i)] = theta.sin();
            r[(i + 1, i + 1)] = theta.cos();
        }

        let omega = matrix_log_orthogonal(&r).expect("log should succeed");

        let mut expected = SMatrix::<Real, N, N>::zeros();
        for b in 0..(N / 2) {
            let i = 2 * b;
            expected[(i, i + 1)] = -theta;
            expected[(i + 1, i)] = theta;
        }

        let err = (omega - expected).norm();
        assert!(err < 1e-13, "four equal angles: error = {err:.3e}");
    }

    /// A half-turn in one invariant plane puts `R` on the cut locus, whatever
    /// the other planes do.
    #[test]
    fn test_log_cut_locus_general_dimension() {
        const N: usize = 6;
        let mut r = SMatrix::<Real, N, N>::identity();
        // Rotation by exactly π in the (0,1) plane.
        r[(0, 0)] = -1.0;
        r[(1, 1)] = -1.0;
        // A benign rotation elsewhere, so the failure comes from the half-turn.
        let phi: Real = 0.4;
        r[(2, 2)] = phi.cos();
        r[(2, 3)] = -phi.sin();
        r[(3, 2)] = phi.sin();
        r[(3, 3)] = phi.cos();

        assert!(
            matches!(matrix_log_orthogonal(&r), Err(CartanError::CutLocus { .. })),
            "a half-turn in one plane must report the cut locus"
        );
    }

    /// The result must be exactly skew-symmetric, not merely close to it:
    /// `SO(N)::log` hands it straight to `check_tangent`.
    #[test]
    fn test_log_result_is_skew() {
        const N: usize = 7;
        let mut raw = SMatrix::<Real, N, N>::zeros();
        for i in 0..N {
            for j in (i + 1)..N {
                let e = ((i * 5 + j * 3) % 7) as Real / 7.0 - 0.5;
                raw[(i, j)] = e;
                raw[(j, i)] = -e;
            }
        }
        let r = matrix_exp_skew(&skew(&raw));
        let omega = matrix_log_orthogonal(&r).expect("log should succeed");
        let asym = (omega + omega.transpose()).norm();
        assert!(asym < 1e-15, "log is not skew: ||Ω + Ω^T|| = {asym:.3e}");
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
