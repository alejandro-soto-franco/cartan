// ~/cartan/cartan-manifolds/src/spd.rs

//! Symmetric Positive Definite manifold SPD(N) with the affine-invariant metric.
//!
//! SPD(N) is the open cone of N×N real symmetric positive definite matrices:
//!
//! ```text
//! SPD(N) = { P ∈ Sym(N) : P > 0 }
//! ```
//!
//! It has dimension N(N+1)/2 (one degree of freedom per upper-triangular entry).
//!
//! ## Affine-invariant metric
//!
//! The standard Riemannian metric on SPD(N) is the *affine-invariant* (or
//! *Fisher-Rao*) metric, defined at P by:
//!
//! ```text
//! <U, V>_P = tr(P^{-1} U P^{-1} V)
//! ```
//!
//! This metric is invariant under congruence transformations P -> A P A^T
//! for any invertible A, which makes it natural for covariance matrices
//! (invariant to linear re-parameterisation of the underlying Gaussian).
//!
//! ## Key formulas
//!
//! Let P, Q ∈ SPD(N) and V ∈ T_P SPD(N) = Sym(N).
//!
//! ```text
//! Exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
//! Log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
//! d(P,Q)^2 = ||log(P^{-1/2} Q P^{-1/2})||_F^2 = sum_i log(lambda_i)^2
//! ```
//!
//! where lambda_i are the eigenvalues of P^{-1/2} Q P^{-1/2} (= gen. eigenvalues of Q, P).
//!
//! ## Geometry: Cartan-Hadamard manifold
//!
//! SPD(N) with the affine-invariant metric is a *Cartan-Hadamard manifold*:
//!
//! - Non-positive sectional curvature: K ≤ 0 everywhere.
//! - Geodesically complete: the exponential map is a global diffeomorphism.
//! - Injectivity radius: ∞ (no cut locus; Log_P is defined globally).
//! - Unique geodesic between any two points.
//!
//! The curvature at the identity for U, V ∈ Sym(N) is:
//!
//! ```text
//! R_I(U, V)W = -1/4 [[U, V], W]
//! ```
//!
//! where [A, B] = AB - BA is the matrix commutator.
//!
//! The sectional curvature of the 2-plane spanned by U, V ∈ T_I SPD at I is:
//!
//! ```text
//! K(U, V) = -1/4 ||[U, V]||_F^2 / (||U||^2 ||V||^2 - <U,V>^2)   ≤  0
//! ```
//!
//! ## Parallel transport
//!
//! Parallel transport of U ∈ T_P SPD along the geodesic from P to Q:
//!
//! ```text
//! Gamma_{P->Q}(U) = E U E^T,   E = P^{1/2} (P^{-1/2} Q P^{-1/2})^{1/2} P^{-1/2}
//! ```
//!
//! This is an isometry: ||Gamma(U)||_Q = ||U||_P.
//!
//! ## Connection and Hessian
//!
//! The Levi-Civita connection has Christoffel symbols:
//!
//! ```text
//! Gamma_P(U, V) = -1/2 (U P^{-1} V + V P^{-1} U)
//! ```
//!
//! The Riemannian Hessian of f at P applied to V is:
//!
//! ```text
//! Hess f(P)[V] = P hess_E[V] P + 1/2 (V P^{-1} G + G P^{-1} V)
//! ```
//!
//! where G = grad_R f(P) is the Riemannian gradient and hess_E\[V\] is the
//! Euclidean Hessian-vector product.
//!
//! ## References
//!
//! - Pennec, X., Fillard, P., Ayache, N. (2006). A Riemannian Framework for
//!   Tensor Computing. IJCV 66(1), 41-66.
//! - Bhatia, R. (2007). Positive Definite Matrices. Princeton. Chapter 6.
//! - Absil, P.-A., Mahony, R., Sepulchre, R. (2008). Optimization Algorithms
//!   on Matrix Manifolds. Princeton. Chapter 7.
//! - Bridson, M., Haefliger, A. (1999). Metric Spaces of Non-Positive Curvature.
//!   Springer. (Symmetric spaces of non-compact type.)

use nalgebra::SMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation,
    Manifold, ParallelTransport, Real, Retraction,
};

use crate::util::sym::{
    sym_exp, sym_inv, sym_log, sym_min_eigenvalue, sym_sqrt, sym_sqrt_inv, sym_symmetrize,
};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Tolerance for point/tangent validation.
const VALIDATION_TOL: Real = 1e-9;

/// Minimum eigenvalue clamp when projecting onto SPD(N).
const MIN_EIG: Real = 1e-8;

// ─────────────────────────────────────────────────────────────────────────────
// Struct
// ─────────────────────────────────────────────────────────────────────────────

/// The manifold of N×N symmetric positive definite matrices with the
/// affine-invariant metric.
///
/// SPD(N) is a Cartan-Hadamard manifold: non-positive sectional curvature,
/// globally defined exponential map, no cut locus. The geodesic between
/// any two points is unique.
///
/// # Type parameter
///
/// `N` is the matrix size. SPD(1) = (0, ∞) (positive reals with the
/// hyperbolic metric). SPD(2) has dimension 3. SPD(3) has dimension 6.
///
/// # Examples
///
/// ```rust,ignore
/// use cartan::manifolds::Spd;
/// use cartan_core::Manifold;
///
/// let m = Spd::<3>;
/// let p = m.random_point(&mut rng);
/// let v = m.random_tangent(&p, &mut rng);
/// let q = m.exp(&p, &v);
/// assert!(m.check_point(&q).is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Spd<const N: usize>;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the "transport operator" E = P^{1/2} M^{1/2} P^{-1/2} where
/// M = P^{-1/2} Q P^{-1/2}. Used for both parallel transport and geodesic.
#[inline]
fn transport_op<const N: usize>(
    p: &SMatrix<Real, N, N>,
    q: &SMatrix<Real, N, N>,
) -> SMatrix<Real, N, N> {
    let sqrt_p = sym_sqrt(p);
    let sqrt_p_inv = sym_sqrt_inv(p);
    let m = sqrt_p_inv * q * sqrt_p_inv;    // P^{-1/2} Q P^{-1/2}, symmetric PD
    let m_sqrt = sym_sqrt(&m);              // M^{1/2}
    sqrt_p * m_sqrt * sqrt_p_inv           // E = P^{1/2} M^{1/2} P^{-1/2}
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifold
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Manifold for Spd<N> {
    /// An SPD matrix: SMatrix<Real, N, N>.
    type Point = SMatrix<Real, N, N>;
    /// A tangent vector: any symmetric N×N matrix.
    type Tangent = SMatrix<Real, N, N>;

    /// Intrinsic dimension: N(N+1)/2.
    fn dim(&self) -> usize {
        N * (N + 1) / 2
    }

    /// Ambient dimension: N^2.
    fn ambient_dim(&self) -> usize {
        N * N
    }

    /// Injectivity radius: ∞ (Cartan-Hadamard manifold, no cut locus).
    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        Real::INFINITY
    }

    /// Affine-invariant inner product: tr(P^{-1} U P^{-1} V).
    ///
    /// Equivalent to the Frobenius inner product of the "whitened" tangent
    /// vectors P^{-1/2} U P^{-1/2} and P^{-1/2} V P^{-1/2}.
    fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        let p_inv = sym_inv(p);
        (p_inv * u * p_inv * v).trace()
    }

    /// Exponential map (affine-invariant geodesic):
    ///
    /// ```text
    /// Exp_P(V) = P^{1/2} exp(P^{-1/2} V P^{-1/2}) P^{1/2}
    /// ```
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        let sqrt_p = sym_sqrt(p);
        let sqrt_p_inv = sym_sqrt_inv(p);
        let s = sqrt_p_inv * v * sqrt_p_inv;   // P^{-1/2} V P^{-1/2}, symmetric
        sqrt_p * sym_exp(&s) * sqrt_p           // P^{1/2} exp(S) P^{1/2}
    }

    /// Logarithmic map (affine-invariant geodesic):
    ///
    /// ```text
    /// Log_P(Q) = P^{1/2} log(P^{-1/2} Q P^{-1/2}) P^{1/2}
    /// ```
    ///
    /// Globally defined: SPD(N) is a Cartan-Hadamard manifold (no cut locus).
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        let sqrt_p = sym_sqrt(p);
        let sqrt_p_inv = sym_sqrt_inv(p);
        let m = sqrt_p_inv * q * sqrt_p_inv;   // P^{-1/2} Q P^{-1/2}, symmetric PD
        Ok(sqrt_p * sym_log(&m) * sqrt_p)       // P^{1/2} log(M) P^{1/2}
    }

    /// Project ambient matrix onto T_P SPD(N): symmetrize.
    ///
    /// T_P SPD(N) = Sym(N) (all symmetric matrices). Projection is just
    /// (V + V^T) / 2.
    fn project_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        sym_symmetrize(v)
    }

    /// Project ambient matrix onto SPD(N): clamp negative eigenvalues to MIN_EIG.
    ///
    /// Symmetrizes, then clamps eigenvalues to [MIN_EIG, ∞). This gives the
    /// nearest SPD matrix in the Frobenius norm (up to a small floor value).
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        let sym = sym_symmetrize(p);
        crate::util::sym::sym_apply_pub::<N>(&sym, |v| v.max(MIN_EIG))
    }

    /// Zero tangent vector: the N×N zero matrix.
    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SMatrix::zeros()
    }

    /// Validate that P is symmetric and positive definite.
    ///
    /// Checks:
    /// 1. P is symmetric: ||P - P^T||_F < tol.
    /// 2. P is positive definite: λ_min(P) > 0.
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        let sym_violation = (p - p.transpose()).norm();
        if sym_violation > VALIDATION_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: format!("P = P^T (SPD({}))", N),
                violation: sym_violation,
            });
        }
        let min_ev = sym_min_eigenvalue(p);
        if min_ev <= 0.0 {
            return Err(CartanError::NotOnManifold {
                constraint: format!("P > 0 (SPD({})), min eigenvalue", N),
                violation: -min_ev,
            });
        }
        Ok(())
    }

    /// Validate that V ∈ T_P SPD(N): check V is symmetric.
    fn check_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        let sym_violation = (v - v.transpose()).norm();
        if sym_violation > VALIDATION_TOL {
            return Err(CartanError::NotInTangentSpace {
                constraint: format!("V = V^T (T_P SPD({}))", N),
                violation: sym_violation,
            });
        }
        Ok(())
    }

    /// Random SPD matrix: G G^T + epsilon * I for random Gaussian G.
    ///
    /// Adding epsilon * I ensures strict positive definiteness even for
    /// small N where G G^T might be borderline singular.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample::<Real, _>(StandardNormal));
        let m = g * g.transpose();
        m + SMatrix::identity() * (N as Real * 0.1)
    }

    /// Random tangent vector at P: random symmetric matrix.
    ///
    /// Samples a random N×N Gaussian matrix and symmetrizes it.
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent {
        let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample::<Real, _>(StandardNormal));
        self.project_tangent(p, &g)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Retraction for Spd<N> {
    /// Retraction: Exp_P(V) — the affine-invariant geodesic exponential.
    ///
    /// Using exp as the retraction makes `retract`/`inverse_retract` a
    /// consistent inverse pair (retract = exp, inverse_retract = log),
    /// which is required for harness tests of the roundtrip identity.
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        self.exp(p, v)
    }

    /// Inverse retraction: Log_P(Q).
    ///
    /// On a Cartan-Hadamard manifold, Log is globally defined, so this
    /// always succeeds.
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        self.log(p, q)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelTransport
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> ParallelTransport for Spd<N> {
    /// Parallel transport of U ∈ T_P SPD along the geodesic from P to Q.
    ///
    /// ```text
    /// Gamma_{P->Q}(U) = E U E^T
    ///
    /// E = P^{1/2} (P^{-1/2} Q P^{-1/2})^{1/2} P^{-1/2}
    /// ```
    ///
    /// This is an isometry of the inner product: ||Gamma(U)||_Q = ||U||_P.
    ///
    /// Ref: Pennec et al. (2006), equation (A.6).
    fn transport(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let e = transport_op(p, q);
        Ok(sym_symmetrize(&(e * u * e.transpose())))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Connection for Spd<N> {
    /// Riemannian Hessian-vector product for the affine-invariant metric.
    ///
    /// Derived from the Christoffel symbols Gamma_P(U, V) = -1/2(U P^{-1} V + V P^{-1} U):
    ///
    /// ```text
    /// Hess f(P)[V] = P hess_E[V] P + 1/2 (V P^{-1} G + G P^{-1} V)
    /// ```
    ///
    /// where G = grad_f is the Riemannian gradient and hess_E\[V\] = hess_ambient(V).
    ///
    /// Ref: Absil-Mahony-Sepulchre Chapter 5; connection formula for SPD.
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let p_inv = sym_inv(p);
        let hvp = hess_ambient(v);
        let result = p * hvp * p + (v * p_inv * grad_f + grad_f * p_inv * v) * 0.5;
        Ok(sym_symmetrize(&result))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Curvature for Spd<N> {
    /// Riemann curvature tensor at P: R(U, V)W.
    ///
    /// Transport to the identity via congruence, apply the identity formula,
    /// then transport back:
    ///
    /// ```text
    /// R_P(U, V)W = P^{1/2} R_I(A, B, C) P^{1/2}
    ///
    /// where A = P^{-1/2} U P^{-1/2},  B = P^{-1/2} V P^{-1/2},
    ///       C = P^{-1/2} W P^{-1/2}
    ///
    /// R_I(A, B)C = -1/4 [[A, B], C]
    ///            = -1/4 ((AB - BA)C - C(AB - BA))
    /// ```
    ///
    /// Ref: Bhatia (2007), Theorem 6.1.12.
    fn riemann_curvature(
        &self,
        p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
        w: &Self::Tangent,
    ) -> Self::Tangent {
        let sqrt_p = sym_sqrt(p);
        let sqrt_p_inv = sym_sqrt_inv(p);

        // Transport to identity.
        let a = sqrt_p_inv * u * sqrt_p_inv;
        let b = sqrt_p_inv * v * sqrt_p_inv;
        let c = sqrt_p_inv * w * sqrt_p_inv;

        // Commutator [A, B] = AB - BA.
        let comm_ab = a * b - b * a;

        // R_I(A, B)C = -1/4 [[A,B], C] = -1/4 (comm_ab * C - C * comm_ab).
        let r_i = (comm_ab * c - c * comm_ab) * (-0.25);

        // Transport back to P: P^{1/2} R_I P^{1/2}.
        sym_symmetrize(&(sqrt_p * r_i * sqrt_p))
    }

    /// Ricci curvature: Ric(U, V) = -(N+1)/4 * <U, V>_P.
    ///
    /// SPD(N) with the affine-invariant metric is an Einstein manifold
    /// (Ricci tensor proportional to the metric):
    ///
    /// ```text
    /// Ric(U, V) = -((N+1)/4) <U, V>_P
    /// ```
    ///
    /// Ref: Bridson-Haefliger (1999), symmetric space structure of SPD(N).
    fn ricci_curvature(
        &self,
        p: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
    ) -> Real {
        -(N as Real + 1.0) / 4.0 * self.inner(p, u, v)
    }

    /// Scalar curvature: s = -(N+1)/4 * dim = -N(N+1)^2/8.
    ///
    /// Obtained by tracing the Ricci tensor over an orthonormal basis of
    /// T_P SPD(N):
    ///
    /// ```text
    /// s = sum_i Ric(e_i, e_i) = -(N+1)/4 * N(N+1)/2 = -N(N+1)^2/8
    /// ```
    fn scalar_curvature(&self, _p: &Self::Point) -> Real {
        let n = N as Real;
        -n * (n + 1.0) * (n + 1.0) / 8.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeodesicInterpolation
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> GeodesicInterpolation for Spd<N> {
    /// Geodesic interpolation at parameter t:
    ///
    /// ```text
    /// gamma(P, Q, t) = P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}
    ///                = Exp_P(t * Log_P(Q))
    /// ```
    ///
    /// For t = 1/2 this gives the (unique) geometric mean of P and Q.
    /// For any t ∈ R the result is a valid SPD matrix.
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        let sqrt_p = sym_sqrt(p);
        let sqrt_p_inv = sym_sqrt_inv(p);
        let m = sqrt_p_inv * q * sqrt_p_inv;   // P^{-1/2} Q P^{-1/2}, symmetric PD
        // M^t = exp(t * log(M))
        let log_m = sym_log(&m);
        let mt = sym_exp(&(log_m * t));
        Ok(sym_symmetrize(&(sqrt_p * mt * sqrt_p)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample_spd_3() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            4.0, 2.0, 1.0,
            2.0, 3.0, 0.5,
            1.0, 0.5, 2.0,
        ])
    }

    fn sample_spd_3b() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            3.0, 1.0, 0.5,
            1.0, 4.0, 1.5,
            0.5, 1.5, 2.5,
        ])
    }

    // ── Basic geometry ──────────────────────────────────────────────────────

    #[test]
    fn test_dim() {
        assert_eq!(Spd::<2>.dim(), 3);
        assert_eq!(Spd::<3>.dim(), 6);
        assert_eq!(Spd::<4>.dim(), 10);
    }

    #[test]
    fn test_check_point_valid() {
        assert!(Spd::<3>.check_point(&sample_spd_3()).is_ok());
    }

    #[test]
    fn test_check_point_identity() {
        assert!(Spd::<3>.check_point(&SMatrix::<Real, 3, 3>::identity()).is_ok());
    }

    #[test]
    fn test_check_point_not_pd_fails() {
        let mut m = sample_spd_3();
        m[(0, 0)] = -1.0;   // break PD
        assert!(Spd::<3>.check_point(&m).is_err());
    }

    #[test]
    fn test_check_tangent_symmetric() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let mut v = SMatrix::<Real, 3, 3>::zeros();
        v[(0, 1)] = 1.0;
        v[(1, 0)] = 1.0;
        assert!(m.check_tangent(&p, &v).is_ok());
    }

    #[test]
    fn test_check_tangent_asymmetric_fails() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let mut v = SMatrix::<Real, 3, 3>::zeros();
        v[(0, 1)] = 1.0;  // asymmetric
        assert!(m.check_tangent(&p, &v).is_err());
    }

    // ── Exp / Log roundtrip ─────────────────────────────────────────────────

    #[test]
    fn test_exp_stays_in_spd() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        use rand::SeedableRng;
        let v = m.random_tangent(&p, &mut rand::rngs::SmallRng::seed_from_u64(1));
        let q = m.exp(&p, &v);
        assert!(m.check_point(&q).is_ok(), "exp result not in SPD");
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let v = sample_symmetric_tangent_3();
        let q = m.exp(&p, &v);
        let v2 = m.log(&p, &q).unwrap();
        let diff = (v - v2).norm();
        assert!(diff < 1e-12, "exp-log roundtrip error: {:.2e}", diff);
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let v = m.log(&p, &q).unwrap();
        let q2 = m.exp(&p, &v);
        let diff = (q - q2).norm();
        assert!(diff < 1e-12, "log-exp roundtrip error: {:.2e}", diff);
    }

    // ── Distance / metric ───────────────────────────────────────────────────

    #[test]
    fn test_dist_self_is_zero() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let d = m.dist(&p, &p).unwrap();
        assert!(d < 1e-12, "dist(P, P) = {:.2e}", d);
    }

    #[test]
    fn test_dist_symmetry() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let dpq = m.dist(&p, &q).unwrap();
        let dqp = m.dist(&q, &p).unwrap();
        assert!((dpq - dqp).abs() < 1e-12, "dist not symmetric: {:.2e}", (dpq - dqp).abs());
    }

    #[test]
    fn test_inner_positive_definite() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let v = sample_symmetric_tangent_3();
        let inner = m.inner(&p, &v, &v);
        assert!(inner > 0.0, "<V,V>_P = {:.2e} must be positive", inner);
    }

    // ── Injectivity radius ───────────────────────────────────────────────────

    #[test]
    fn test_injectivity_radius_infinity() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        assert_eq!(m.injectivity_radius(&p), Real::INFINITY);
    }

    // ── Parallel transport ───────────────────────────────────────────────────

    #[test]
    fn test_parallel_transport_norm_preserving() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let u = sample_symmetric_tangent_3();

        let u_norm = m.norm(&p, &u);
        let u_transported = m.transport(&p, &q, &u).unwrap();
        let u_t_norm = m.norm(&q, &u_transported);

        let diff = (u_norm - u_t_norm).abs();
        assert!(diff < 1e-10, "transport not norm-preserving: {:.2e}", diff);
    }

    #[test]
    fn test_parallel_transport_result_symmetric() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let u = sample_symmetric_tangent_3();

        let u_t = m.transport(&p, &q, &u).unwrap();
        let sym_err = (u_t - u_t.transpose()).norm();
        assert!(sym_err < 1e-12, "transported vector not symmetric: {:.2e}", sym_err);
    }

    // ── Curvature: K ≤ 0 ────────────────────────────────────────────────────

    #[test]
    fn test_sectional_curvature_nonpositive() {
        use cartan_core::Curvature;
        let m = Spd::<3>;
        let p = sample_spd_3();
        let u = sample_symmetric_tangent_3();
        let v = sample_symmetric_tangent_3b();
        let k = m.sectional_curvature(&p, &u, &v);
        assert!(k <= 1e-10, "sectional curvature positive: K = {:.4e}", k);
    }

    #[test]
    fn test_sectional_curvature_zero_for_commuting() {
        use cartan_core::Curvature;
        let m = Spd::<3>;
        // At identity, K(U,V) = -||[U,V]||^2 / (4 * area^2).
        // For diagonal (commuting) U, V: [U,V] = 0, so K = 0.
        let p = SMatrix::<Real, 3, 3>::identity();
        let u = SMatrix::<Real, 3, 3>::from_diagonal(
            &nalgebra::SVector::from([1.0_f64, 2.0, 3.0]),
        );
        let v = SMatrix::<Real, 3, 3>::from_diagonal(
            &nalgebra::SVector::from([3.0_f64, 1.0, 2.0]),
        );
        let k = m.sectional_curvature(&p, &u, &v);
        assert!(k.abs() < 1e-12, "K for commuting matrices: {:.2e}", k);
    }

    #[test]
    fn test_ricci_sign() {
        use cartan_core::Curvature;
        let m = Spd::<3>;
        let p = sample_spd_3();
        let v = sample_symmetric_tangent_3();
        let ric = m.ricci_curvature(&p, &v, &v);
        assert!(ric <= 0.0, "Ricci(V,V) should be <= 0: {:.4e}", ric);
    }

    #[test]
    fn test_scalar_curvature_sign() {
        use cartan_core::Curvature;
        let m = Spd::<3>;
        let p = sample_spd_3();
        let s = m.scalar_curvature(&p);
        assert!(s < 0.0, "scalar curvature should be negative: {:.4e}", s);
    }

    // ── GeodesicInterpolation ────────────────────────────────────────────────

    #[test]
    fn test_geodesic_t0_is_p() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let g0 = m.geodesic(&p, &q, 0.0).unwrap();
        let diff = (g0 - p).norm();
        assert!(diff < 1e-12, "geodesic(t=0) != P: {:.2e}", diff);
    }

    #[test]
    fn test_geodesic_t1_is_q() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let g1 = m.geodesic(&p, &q, 1.0).unwrap();
        let diff = (g1 - q).norm();
        assert!(diff < 1e-12, "geodesic(t=1) != Q: {:.2e}", diff);
    }

    #[test]
    fn test_geodesic_midpoint_stays_in_spd() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let mid = m.geodesic(&p, &q, 0.5).unwrap();
        assert!(m.check_point(&mid).is_ok(), "midpoint not in SPD");
    }

    #[test]
    fn test_geodesic_midpoint_equidistant() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let mid = m.geodesic(&p, &q, 0.5).unwrap();

        let d_total = m.dist(&p, &q).unwrap();
        let d_p_mid = m.dist(&p, &mid).unwrap();
        let d_mid_q = m.dist(&mid, &q).unwrap();

        assert_relative_eq!(d_p_mid, d_total / 2.0, epsilon = 1e-10);
        assert_relative_eq!(d_mid_q, d_total / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_geodesic_consistent_with_exp_log() {
        let m = Spd::<3>;
        let p = sample_spd_3();
        let q = sample_spd_3b();
        let t = 0.3;

        // geodesic(p, q, t) should equal exp(p, t * log(p, q)).
        let via_geo = m.geodesic(&p, &q, t).unwrap();
        let via_exp = m.exp(&p, &(m.log(&p, &q).unwrap() * t));
        let diff = (via_geo - via_exp).norm();
        assert!(diff < 1e-12, "geodesic vs exp-log: {:.2e}", diff);
    }

    // ── Random point / tangent ───────────────────────────────────────────────

    #[test]
    fn test_random_point_is_valid() {
        use rand::SeedableRng;
        let m = Spd::<4>;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        for _ in 0..10 {
            let p = m.random_point(&mut rng);
            assert!(m.check_point(&p).is_ok(), "random_point not SPD");
        }
    }

    #[test]
    fn test_random_tangent_is_valid() {
        use rand::SeedableRng;
        let m = Spd::<4>;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(13);
        let p = m.random_point(&mut rng);
        for _ in 0..10 {
            let v = m.random_tangent(&p, &mut rng);
            assert!(m.check_tangent(&p, &v).is_ok(), "random_tangent not symmetric");
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    fn sample_symmetric_tangent_3() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            0.2,  0.1, -0.05,
            0.1,  0.3,  0.08,
           -0.05, 0.08, 0.15,
        ])
    }

    fn sample_symmetric_tangent_3b() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            0.1, -0.2, 0.0,
           -0.2,  0.4, 0.1,
            0.0,  0.1, 0.2,
        ])
    }
}
