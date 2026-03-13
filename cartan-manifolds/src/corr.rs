// ~/cartan/cartan-manifolds/src/corr.rs

//! Correlation matrix manifold Corr(N) with the Frobenius metric.
//!
//! A correlation matrix is a symmetric positive definite N×N matrix with
//! unit diagonal entries:
//!
//! ```text
//! Corr(N) = { C ∈ Sym(N) : C > 0,  C_{ii} = 1 for all i }
//! ```
//!
//! ## Geometry: flat submanifold
//!
//! With the Frobenius inner product `<U, V> = tr(U^T V)`, Corr(N) is an
//! **open subset of an affine subspace** of the Euclidean space Sym(N).
//! The unit-diagonal constraint `{C_{ii} = 1}` is an affine hyperplane
//! in Sym(N); intersecting with the (open) PD cone gives Corr(N) as an
//! open subset of that affine space. Consequently:
//!
//! - **Sectional curvature: identically zero.** Corr(N) is flat.
//! - **Geodesics: straight lines** in Sym(N) that stay inside Corr(N).
//! - **Exp/Log: exact, no truncation.**
//!
//! ## Key formulas
//!
//! Let `C ∈ Corr(N)` and `V ∈ T_C Corr(N) = {V ∈ Sym(N) : V_{ii} = 0}`.
//!
//! ```text
//! Inner product:  <U, V>_C  = tr(U V)               (independent of C)
//! Exp_C(V)     =  C + V                             (flat geodesic)
//! Log_C(Q)     =  Q - C                             (flat geodesic)
//! Transport    =  identity                           (flat connection)
//! Curvature    =  0                                  (flat)
//! ```
//!
//! The injectivity radius at `C` is the smallest eigenvalue of `C`: the
//! straight-line geodesic remains PD (and hence in Corr(N)) as long as the
//! step size is less than `λ_min(C)` (by Weyl's perturbation bound).
//!
//! ## Tangent space
//!
//! ```text
//! T_C Corr(N) = { V ∈ R^{N×N} : V = V^T,  V_{ii} = 0 }
//! ```
//!
//! Dimension: N(N-1)/2. The zero-diagonal constraint is the linearization
//! of the unit-diagonal constraint `C_{ii} = 1`.
//!
//! ## Projection
//!
//! - `project_tangent(C, V)`: symmetrize V and zero out its diagonal.
//! - `project_point(C)`: nearest correlation matrix via Higham (2002)
//!   alternating projections with Dykstra's correction.
//!
//! ## References
//!
//! - Higham, N. J. (2002). Computing the Nearest Correlation Matrix — a
//!   Problem from Finance. IMA Journal of Numerical Analysis, 22(3), 329-343.
//! - Absil, Mahony, Sepulchre. Optimization Algorithms on Matrix Manifolds.
//!   Princeton, 2008. (flat manifold geometry, Chapter 3.)
//! - Bonnabel, Sepulchre. Riemannian Metric and Geometric Mean for a Class
//!   of Symmetric Positive Semi-definite Matrices. SIAM J. Matrix Anal., 2009.

use nalgebra::SMatrix;
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation,
    Manifold, ParallelTransport, Real, Retraction,
};

use crate::util::sym::{nearest_corr_matrix, sym_min_eigenvalue, sym_symmetrize};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Tolerance for point/tangent constraint validation.
const VALIDATION_TOL: Real = 1e-9;

/// Max iterations for nearest_corr_matrix in project_point.
const PROJ_MAX_ITER: usize = 500;

/// Tolerance for nearest_corr_matrix convergence in project_point.
const PROJ_TOL: Real = 1e-12;

// ─────────────────────────────────────────────────────────────────────────────
// Struct
// ─────────────────────────────────────────────────────────────────────────────

/// The manifold of N×N correlation matrices with the Frobenius metric.
///
/// A correlation matrix is symmetric positive definite with unit diagonal.
/// Corr(N) has intrinsic dimension N(N-1)/2 and is flat: it is an open
/// subset of the affine subspace {C ∈ Sym(N) : C_{ii} = 1} of the
/// Euclidean space Sym(N).
///
/// The zero-sized type `Corr<N>` carries no data; the geometry is fully
/// determined by the dimension parameter N.
///
/// # Type parameter
///
/// `N` is the matrix size. Corr<2> is the set {[[1,r],[r,1]] : |r| < 1},
/// a 1-dimensional flat manifold isometric to the open interval (-1, 1).
///
/// # Examples
///
/// ```rust,ignore
/// use cartan::manifolds::Corr;
/// use cartan_core::Manifold;
///
/// let m = Corr::<3>;
/// let c = m.random_point(&mut rng);
/// let v = m.random_tangent(&c, &mut rng);
/// let q = m.exp(&c, &v);
/// assert!(m.check_point(&q).is_ok());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Corr<const N: usize>;

// ─────────────────────────────────────────────────────────────────────────────
// Manifold
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Manifold for Corr<N> {
    /// A correlation matrix: SMatrix<Real, N, N>.
    type Point = SMatrix<Real, N, N>;
    /// A tangent vector: symmetric matrix with zero diagonal.
    type Tangent = SMatrix<Real, N, N>;

    /// Intrinsic dimension: N(N-1)/2.
    ///
    /// The number of independent off-diagonal entries of a symmetric N×N matrix.
    fn dim(&self) -> usize {
        N * (N - 1) / 2
    }

    /// Ambient dimension: N^2 (entries of an N×N matrix).
    ///
    /// Points are stored as full N×N matrices even though they live in the
    /// N(N+1)/2-dimensional subspace Sym(N).
    fn ambient_dim(&self) -> usize {
        N * N
    }

    /// Injectivity radius at C: the minimum eigenvalue of C.
    ///
    /// By Weyl's perturbation bound, a step V ∈ T_C Corr(N) with ||V||_F < λ_min(C)
    /// keeps C + V positive definite, so the geodesic stays inside Corr(N).
    fn injectivity_radius(&self, p: &Self::Point) -> Real {
        sym_min_eigenvalue(p)
    }

    /// Frobenius inner product: tr(U V) = sum_{i,j} U_{ij} V_{ij}.
    ///
    /// Independent of the base point C. This is the flat Euclidean metric
    /// on Sym(N) restricted to the affine subspace {C_{ii} = 1}.
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        (u * v).trace()
    }

    /// Exponential map (flat geodesic): Exp_C(V) = C + V.
    ///
    /// Since Corr(N) is flat, geodesics are straight lines in Sym(N).
    /// The result is a correlation matrix whenever V ∈ T_C Corr(N) and
    /// ||V||_F < λ_min(C).
    ///
    /// Note: this does not clamp to the PD cone. Use project_point if
    /// the step may exceed the injectivity radius.
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        p + v
    }

    /// Logarithmic map (flat geodesic): Log_C(Q) = Q - C.
    ///
    /// Since geodesics are straight lines, the log map is globally defined
    /// and never fails. It returns Err only as a formality; Corr(N) has no
    /// cut locus in its interior (the "cut locus" is the boundary where
    /// positive-definiteness is lost, which is not part of the manifold).
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        Ok(q - p)
    }

    /// Project an ambient matrix onto T_C Corr(N).
    ///
    /// Symmetrizes V and zeros its diagonal:
    ///   1. V <- (V + V^T) / 2   (enforce symmetry)
    ///   2. V_{ii} <- 0           (enforce zero diagonal)
    fn project_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        let mut sym = sym_symmetrize(v);
        for i in 0..N {
            sym[(i, i)] = 0.0;
        }
        sym
    }

    /// Project an ambient matrix onto Corr(N).
    ///
    /// Computes the nearest correlation matrix in the Frobenius norm using
    /// Higham's (2002) alternating projections algorithm with Dykstra's
    /// correction. The algorithm alternately projects onto the PD cone
    /// and the unit-diagonal affine subspace until convergence.
    ///
    /// Ref: Higham (2002), IMA Journal of Numerical Analysis 22(3), 329-343.
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        nearest_corr_matrix(p, PROJ_MAX_ITER, PROJ_TOL)
    }

    /// Zero tangent vector: the N×N zero matrix.
    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SMatrix::zeros()
    }

    /// Validate that C is a correlation matrix.
    ///
    /// Checks:
    /// 1. All diagonal entries equal 1 (within VALIDATION_TOL).
    /// 2. C is symmetric (within VALIDATION_TOL in Frobenius norm).
    /// 3. C is positive definite: λ_min(C) > 0.
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        // Check unit diagonal.
        for i in 0..N {
            let diag_violation = (p[(i, i)] - 1.0).abs();
            if diag_violation > VALIDATION_TOL {
                return Err(CartanError::NotOnManifold {
                    constraint: format!("C_{{{}{}}} = 1 (Corr({}))", i, i, N),
                    violation: diag_violation,
                });
            }
        }

        // Check symmetry.
        let sym_violation = (p - p.transpose()).norm();
        if sym_violation > VALIDATION_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: format!("C = C^T (Corr({}))", N),
                violation: sym_violation,
            });
        }

        // Check positive definiteness.
        let min_ev = sym_min_eigenvalue(p);
        if min_ev <= 0.0 {
            return Err(CartanError::NotOnManifold {
                constraint: format!("C > 0 (Corr({})), min eigenvalue", N),
                violation: -min_ev,
            });
        }

        Ok(())
    }

    /// Validate that V ∈ T_C Corr(N).
    ///
    /// Checks:
    /// 1. V is symmetric (within VALIDATION_TOL in Frobenius norm).
    /// 2. All diagonal entries of V are zero (within VALIDATION_TOL).
    fn check_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        // Check symmetry.
        let sym_violation = (v - v.transpose()).norm();
        if sym_violation > VALIDATION_TOL {
            return Err(CartanError::NotInTangentSpace {
                constraint: format!("V = V^T (T_C Corr({}))", N),
                violation: sym_violation,
            });
        }

        // Check zero diagonal.
        for i in 0..N {
            let diag_violation = v[(i, i)].abs();
            if diag_violation > VALIDATION_TOL {
                return Err(CartanError::NotInTangentSpace {
                    constraint: format!("V_{{{}{}}} = 0 (T_C Corr({}))", i, i, N),
                    violation: diag_violation,
                });
            }
        }

        Ok(())
    }

    /// Random correlation matrix.
    ///
    /// Algorithm:
    ///   1. Sample a random N×N Gaussian matrix G.
    ///   2. Form M = G G^T (symmetric PD with probability 1).
    ///   3. Normalize to correlation matrix: C_{ij} = M_{ij} / sqrt(M_{ii} M_{jj}).
    ///
    /// This gives a uniformly "spread" distribution over Corr(N) and
    /// always produces a valid (strictly) positive definite correlation matrix.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample::<Real, _>(StandardNormal));
        let m = g * g.transpose();

        // Normalize to unit diagonal.
        let mut c = m;
        for i in 0..N {
            for j in 0..N {
                let dij = (m[(i, i)] * m[(j, j)]).sqrt();
                c[(i, j)] = if dij > 1e-15 { m[(i, j)] / dij } else { 0.0 };
            }
        }
        // Enforce exact unit diagonal (may drift by rounding above).
        for i in 0..N {
            c[(i, i)] = 1.0;
        }
        c
    }

    /// Random tangent vector at C: symmetric matrix with zero diagonal.
    ///
    /// Samples a random symmetric zero-diagonal matrix with independent
    /// standard Gaussian off-diagonal entries.
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent {
        let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample::<Real, _>(StandardNormal));
        let v = self.project_tangent(p, &g);
        // Scale to stay within a safe fraction of the injectivity radius λ_min(C),
        // so that exp(C, v) = C + v remains inside the PD cone.
        let inj = self.injectivity_radius(p);
        let v_norm = v.norm();
        if v_norm > 1e-15 {
            v * ((inj * 0.3) / v_norm).min(1.0)
        } else {
            v
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Retraction for Corr<N> {
    /// Retraction: project C + V onto Corr(N).
    ///
    /// For steps well within the injectivity radius, this coincides with exp.
    /// For larger steps, it snaps the result back to the nearest correlation
    /// matrix. Cheaper than a full Higham solve when the step is small because
    /// nearest_corr_matrix converges immediately if C + V is already PD
    /// with unit diagonal.
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        self.project_point(&(p + v))
    }

    /// Inverse retraction: Q - C.
    ///
    /// The inverse of the exp/projection retraction on the flat manifold.
    /// The result is already in T_C Corr(N) (symmetric with zero diagonal)
    /// when C and Q are both correlation matrices, since Q_{ii} - C_{ii} = 0.
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        Ok(q - p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ParallelTransport
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> ParallelTransport for Corr<N> {
    /// Parallel transport from C to Q: the identity map.
    ///
    /// On a flat manifold, parallel transport is the identity: a tangent vector
    /// U at C is transported to the same matrix U at Q. The transported vector
    /// lives in T_Q Corr(N) because U is symmetric with zero diagonal regardless
    /// of the base point.
    fn transport(
        &self,
        _p: &Self::Point,
        _q: &Self::Point,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        Ok(*u)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Connection for Corr<N> {
    /// Riemannian Hessian-vector product for the flat manifold.
    ///
    /// On a flat submanifold of Euclidean space, the shape operator vanishes
    /// and the Riemannian Hessian is the projection of the Euclidean Hessian
    /// onto the tangent space:
    ///
    ///   Hess f(C)[V] = proj_T(D^2 f(C)[V])
    ///
    /// where proj_T symmetrizes and zeros the diagonal.
    ///
    /// Ref: Absil-Mahony-Sepulchre, Proposition 5.3.2 (flat case).
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        _grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let ambient_hvp = hess_ambient(v);
        Ok(self.project_tangent(p, &ambient_hvp))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Curvature for Corr<N> {
    /// Riemann curvature tensor: identically zero.
    ///
    /// Corr(N) is flat (open subset of an affine Euclidean subspace),
    /// so R(U, V)W = 0 for all tangent vectors U, V, W.
    fn riemann_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
        _w: &Self::Tangent,
    ) -> Self::Tangent {
        SMatrix::zeros()
    }

    /// Ricci curvature: identically zero (flat manifold).
    fn ricci_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
    ) -> Real {
        0.0
    }

    /// Scalar curvature: identically zero (flat manifold).
    fn scalar_curvature(&self, _p: &Self::Point) -> Real {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GeodesicInterpolation
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> GeodesicInterpolation for Corr<N> {
    /// Geodesic interpolation: (1 - t) C + t Q.
    ///
    /// Since geodesics are straight lines, the interpolation is linear.
    /// For t ∈ (0, 1) and C, Q ∈ Corr(N), the result is a convex combination
    /// of two correlation matrices: it is symmetric with unit diagonal and,
    /// by convexity of the PD cone, positive definite. So it is always a valid
    /// correlation matrix and this never fails.
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        Ok(p * (1.0 - t) + q * t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn identity_corr<const N: usize>() -> SMatrix<Real, N, N> {
        SMatrix::identity()
    }

    /// Random tangent vector with fixed entries for reproducibility.
    fn small_tangent_3() -> SMatrix<Real, 3, 3> {
        let mut v = SMatrix::<Real, 3, 3>::zeros();
        v[(0, 1)] = 0.1;
        v[(1, 0)] = 0.1;
        v[(0, 2)] = 0.05;
        v[(2, 0)] = 0.05;
        v[(1, 2)] = -0.08;
        v[(2, 1)] = -0.08;
        v
    }

    /// A 3×3 correlation matrix with off-diagonal entries in (-1, 1).
    fn sample_corr_3() -> SMatrix<Real, 3, 3> {
        SMatrix::<Real, 3, 3>::from_row_slice(&[
            1.0, 0.5, 0.3,
            0.5, 1.0, 0.2,
            0.3, 0.2, 1.0,
        ])
    }

    // ── Manifold: basic geometry ────────────────────────────────────────────

    #[test]
    fn test_dim() {
        assert_eq!(Corr::<2>.dim(), 1);
        assert_eq!(Corr::<3>.dim(), 3);
        assert_eq!(Corr::<4>.dim(), 6);
    }

    #[test]
    fn test_check_point_identity() {
        let m = Corr::<3>;
        let i3 = identity_corr::<3>();
        assert!(m.check_point(&i3).is_ok());
    }

    #[test]
    fn test_check_point_sample() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        assert!(m.check_point(&c).is_ok());
    }

    #[test]
    fn test_check_point_non_unit_diagonal_fails() {
        let m = Corr::<3>;
        let mut c = sample_corr_3();
        c[(1, 1)] = 1.5;
        assert!(m.check_point(&c).is_err());
    }

    #[test]
    fn test_check_tangent_zero() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = m.zero_tangent(&c);
        assert!(m.check_tangent(&c, &v).is_ok());
    }

    #[test]
    fn test_check_tangent_valid() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        assert!(m.check_tangent(&c, &v).is_ok());
    }

    #[test]
    fn test_check_tangent_nonzero_diagonal_fails() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let mut v = small_tangent_3();
        v[(0, 0)] = 0.1;
        assert!(m.check_tangent(&c, &v).is_err());
    }

    // ── Exp / Log roundtrip ─────────────────────────────────────────────────

    #[test]
    fn test_exp_log_roundtrip() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();

        let q = m.exp(&c, &v);
        let v2 = m.log(&c, &q).unwrap();

        let diff = (v - v2).norm();
        assert!(diff < 1e-14, "exp-log roundtrip diff = {:.2e}", diff);
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let q = m.exp(&c, &v);

        let v_recovered = m.log(&c, &q).unwrap();
        let q2 = m.exp(&c, &v_recovered);

        let diff = (q - q2).norm();
        assert!(diff < 1e-14, "log-exp roundtrip diff = {:.2e}", diff);
    }

    // ── project_tangent ─────────────────────────────────────────────────────

    #[test]
    fn test_project_tangent_idempotent() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();

        let pv = m.project_tangent(&c, &v);
        let ppv = m.project_tangent(&c, &pv);
        let diff = (pv - ppv).norm();
        assert!(diff < 1e-14, "project_tangent not idempotent: diff = {:.2e}", diff);
    }

    #[test]
    fn test_project_tangent_zeros_diagonal() {
        let m = Corr::<3>;
        let c = sample_corr_3();

        // Start with a full ambient matrix.
        let g = SMatrix::<Real, 3, 3>::from_element(1.0);
        let v = m.project_tangent(&c, &g);

        for i in 0..3 {
            assert_relative_eq!(v[(i, i)], 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_project_tangent_symmetrizes() {
        let m = Corr::<3>;
        let c = sample_corr_3();

        let mut g = SMatrix::<Real, 3, 3>::zeros();
        g[(0, 1)] = 1.0;
        g[(1, 0)] = 0.0; // asymmetric
        let v = m.project_tangent(&c, &g);

        let sym_violation = (v - v.transpose()).norm();
        assert!(sym_violation < 1e-14, "project_tangent result not symmetric");
    }

    // ── project_point ───────────────────────────────────────────────────────

    #[test]
    fn test_project_point_identity() {
        let m = Corr::<3>;
        let i3 = identity_corr::<3>();
        let c = m.project_point(&i3);
        assert!(m.check_point(&c).is_ok());
    }

    #[test]
    fn test_project_point_already_corr() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let c2 = m.project_point(&c);
        let diff = (c - c2).norm();
        assert!(diff < 1e-8, "project_point moved a valid correlation matrix: diff = {:.2e}", diff);
    }

    // ── Inner product / norm ─────────────────────────────────────────────────

    #[test]
    fn test_inner_nonneg() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let norm_sq = m.inner(&c, &v, &v);
        assert!(norm_sq >= 0.0, "inner product of v with itself is negative");
    }

    #[test]
    fn test_inner_zero_tangent() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let z = m.zero_tangent(&c);
        assert_relative_eq!(m.inner(&c, &z, &z), 0.0, epsilon = 1e-14);
    }

    // ── Injectivity radius ───────────────────────────────────────────────────

    #[test]
    fn test_injectivity_radius_identity() {
        let m = Corr::<3>;
        let i3 = identity_corr::<3>();
        // Identity has all eigenvalues = 1.
        assert_relative_eq!(m.injectivity_radius(&i3), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_injectivity_radius_positive() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        assert!(m.injectivity_radius(&c) > 0.0);
    }

    // ── Parallel transport (identity) ────────────────────────────────────────

    #[test]
    fn test_parallel_transport_is_identity() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let q = m.exp(&c, &v);

        let u = small_tangent_3();
        let transported = m.transport(&c, &q, &u).unwrap();
        let diff = (u - transported).norm();
        assert!(diff < 1e-14, "parallel transport is not identity: diff = {:.2e}", diff);
    }

    // ── Curvature (identically zero) ─────────────────────────────────────────

    #[test]
    fn test_riemann_curvature_zero() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let u = small_tangent_3();
        let mut w = small_tangent_3();
        w[(0, 1)] *= -1.0;
        w[(1, 0)] *= -1.0;

        let r = m.riemann_curvature(&c, &u, &u, &w);
        assert!(r.norm() < 1e-14, "curvature tensor not zero");
    }

    #[test]
    fn test_sectional_curvature_zero() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let u = small_tangent_3();

        // Build a second independent tangent vector.
        let mut v = SMatrix::<Real, 3, 3>::zeros();
        v[(0, 1)] = 0.2;
        v[(1, 0)] = 0.2;
        v[(1, 2)] = 0.1;
        v[(2, 1)] = 0.1;

        assert_relative_eq!(m.sectional_curvature(&c, &u, &v), 0.0, epsilon = 1e-14);
    }

    // ── GeodesicInterpolation ────────────────────────────────────────────────

    #[test]
    fn test_geodesic_t0_is_p() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let q = m.exp(&c, &v);

        let gamma0 = m.geodesic(&c, &q, 0.0).unwrap();
        let diff = (gamma0 - c).norm();
        assert!(diff < 1e-14, "geodesic(t=0) != p: diff = {:.2e}", diff);
    }

    #[test]
    fn test_geodesic_t1_is_q() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let q = m.exp(&c, &v);

        let gamma1 = m.geodesic(&c, &q, 1.0).unwrap();
        let diff = (gamma1 - q).norm();
        assert!(diff < 1e-14, "geodesic(t=1) != q: diff = {:.2e}", diff);
    }

    #[test]
    fn test_geodesic_midpoint() {
        let m = Corr::<3>;
        let c = sample_corr_3();
        let v = small_tangent_3();
        let q = m.exp(&c, &v);

        let mid = m.geodesic(&c, &q, 0.5).unwrap();
        let expected = (c + q) * 0.5;
        let diff = (mid - expected).norm();
        assert!(diff < 1e-14, "geodesic midpoint incorrect: diff = {:.2e}", diff);
    }

    // ── Random point / tangent ───────────────────────────────────────────────

    #[test]
    fn test_random_point_is_valid() {
        use rand::SeedableRng;
        let m = Corr::<4>;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        for _ in 0..10 {
            let c = m.random_point(&mut rng);
            assert!(
                m.check_point(&c).is_ok(),
                "random_point not a valid correlation matrix: {:?}",
                m.check_point(&c)
            );
        }
    }

    #[test]
    fn test_random_tangent_is_valid() {
        use rand::SeedableRng;
        let m = Corr::<4>;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(99);
        let c = m.random_point(&mut rng);
        for _ in 0..10 {
            let v = m.random_tangent(&c, &mut rng);
            assert!(
                m.check_tangent(&c, &v).is_ok(),
                "random_tangent not in tangent space: {:?}",
                m.check_tangent(&c, &v)
            );
        }
    }
}
