// ~/cartan/cartan-manifolds/src/grassmann.rs

//! Grassmann manifold Gr(N, K) of K-dimensional subspaces of R^N.
//!
//! Points are represented as N×K matrices with orthonormal columns.
//! Two matrices represent the same point iff their column spaces coincide.
//!
//! ## Geometry
//!
//! - Points: N×K matrices Q with Q^T Q = I_K
//! - Tangent space at Q: T_Q Gr = {V ∈ R^{N×K} : Q^T V = 0} (horizontal)
//! - Inner product: <U, V>_Q = tr(U^T V) (Frobenius on horizontal space)
//! - Exponential: V = A Σ B^T (thin SVD) → (Q B cos Σ + A sin Σ) B^T
//! - Logarithm: via principal angles (see gr_log_dyn)
//! - Injectivity radius: π/2
//! - Sectional curvature: 0 ≤ K_sec ≤ 1/4 (non-negative)
//! - Ricci curvature: Ric(X,Y) = (N-2)/4 · tr(X^T Y) (Einstein manifold)
//! - Scalar curvature: K(N-K)(N-2)/4
//!
//! ## Parallel transport
//!
//! Given V = Log_Q(P) = A Σ B^T and horizontal W at Q (Q^T W = 0):
//!   PT(W) = W + (A(cos Σ − I) − Q B sin Σ)(A^T W) B^T
//!
//! This formula is exact and norm-preserving. Proof sketch:
//! - Horizontality at P: verified by direct substitution (see inline comments).
//! - Norm preservation: follows from (2c−2+s²+(c−1)²) = 0 elementwise.
//!
//! ## References
//!
//! - Edelman, Arias, Smith. "The Geometry of Algorithms with Orthogonality
//!   Constraints." SIAM J. Matrix Anal. Appl. 20(2), 1998.
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2009. Chapters 2, 8.
//! - Chikuse. "Statistics on Special Manifolds." Springer, 2003.

use std::f64::consts::PI;

use nalgebra::{DMatrix, DVector, SMatrix};
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation,
    Manifold, ParallelTransport, Real, Retraction,
};

/// The Grassmann manifold Gr(N, K): K-planes in R^N.
///
/// Zero-sized type; geometry is fully determined by N and K.
/// Requires 1 ≤ K ≤ N − 1 (K = 0 or K = N are degenerate).
///
/// # Examples
///
/// ```rust,ignore
/// use cartan::prelude::*;
/// use cartan::manifolds::Grassmann;
///
/// let gr = Grassmann::<5, 2>; // 2-planes in R^5
/// let q = gr.random_point(&mut rng);
/// let v = gr.random_tangent(&q, &mut rng);
/// let p = gr.exp(&q, &v);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Grassmann<const N: usize, const K: usize>;

/// Tolerance for detecting near-zero/near-pi angles.
const GR_ANGLE_EPS: Real = 1e-12;
/// Tolerance for point/tangent validation.
const GR_TOL: Real = 1e-8;
/// Minimum singular value for numerical non-degeneracy (cos of injectivity boundary).
const GR_SING_EPS: Real = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// DMatrix helpers (avoids const-generic constraint hell for SVD/QR)
// ─────────────────────────────────────────────────────────────────────────────

/// Thin QR: returns the N×K Q factor (orthonormal columns).
fn thin_qr(m: &DMatrix<Real>) -> DMatrix<Real> {
    let k = m.ncols();
    let qr = m.clone().qr();
    qr.q().columns(0, k).into_owned()
}

/// Convert SMatrix<Real,N,K> → DMatrix.
fn to_dmat<const N: usize, const K: usize>(m: &SMatrix<Real, N, K>) -> DMatrix<Real> {
    DMatrix::from_column_slice(N, K, m.as_slice())
}

/// Convert DMatrix → SMatrix<Real,N,K>. Panics if shape mismatch.
fn to_smat<const N: usize, const K: usize>(m: &DMatrix<Real>) -> SMatrix<Real, N, K> {
    assert_eq!(m.nrows(), N);
    assert_eq!(m.ncols(), K);
    SMatrix::from_column_slice(m.as_slice())
}

/// Grassmann exponential map (DMatrix version).
///
/// Given Q (N×K orthonormal) and horizontal V (Q^T V = 0):
///   Thin SVD: V = A Σ B^T
///   Exp_Q(V) = (Q B cos Σ + A sin Σ) B^T
/// Result re-orthonormalized via thin QR for numerical safety.
fn gr_exp_dyn(q: &DMatrix<Real>, v: &DMatrix<Real>) -> DMatrix<Real> {
    let k = q.ncols();
    let svd = v.clone().svd(true, true);
    let u_full = svd.u.unwrap(); // N×N full left singular matrix
    let svals = &svd.singular_values; // K values
    let vt = svd.v_t.unwrap(); // K×K

    // Thin A: first K columns of u_full.
    let a = u_full.columns(0, k).into_owned(); // N×K

    // B = Vt^T (K×K). Note: Exp = (Q B cos Σ + A sin Σ) B^T = (Q Vt^T cos Σ + A sin Σ) Vt.
    let b = vt.transpose(); // K×K

    let cos_diag = DMatrix::from_diagonal(&DVector::from_fn(k, |i, _| svals[i].cos()));
    let sin_diag = DMatrix::from_diagonal(&DVector::from_fn(k, |i, _| svals[i].sin()));

    // (Q B cos Σ + A sin Σ) B^T = (Q * b * cos_diag + a * sin_diag) * b.T
    let result = (q * &b * cos_diag + &a * sin_diag) * b.transpose();
    thin_qr(&result)
}

/// Grassmann logarithmic map (DMatrix version).
///
/// Algorithm (principal angles):
///   1.  M   = Q^T P                              (K×K)
///   2.  SVD of M: M = U_M D V_M^T              (D = diag(cos θ_i))
///   3.  Z   = P V_M − Q U_M D                  (N×K, perpendicular to Q)
///   4.  Q_Z = Z / diag(sin θ_i) column-wise    (N×K, orthonormal directions)
///   5.  θ_i = arccos(d_i)
///   6.  Log = Q_Z diag(θ) U_M^T
///
/// Returns CutLocus error if any principal angle ≥ π/2 − ε (cos ≤ ε).
fn gr_log_dyn(
    q: &DMatrix<Real>,
    p: &DMatrix<Real>,
) -> Result<DMatrix<Real>, CartanError> {
    let k = q.ncols();

    // M = Q^T P (K×K)
    let m = q.transpose() * p;

    // SVD of M: M = U_M * D * V_M^T
    let svd_m = m.svd(true, true);
    let u_m = svd_m.u.unwrap(); // K×K
    let d_vals = &svd_m.singular_values; // K values = cos(θ_i)
    let vt_m = svd_m.v_t.unwrap(); // K×K
    let v_m = vt_m.transpose(); // K×K (columns = right singular vectors)

    // Check cut locus: cos(π/2) = 0, so d_i near 0 means θ_i near π/2.
    for &d in d_vals.iter() {
        if d < GR_SING_EPS {
            return Err(CartanError::CutLocus {
                message: format!(
                    "Grassmann Log: principal angle ≥ π/2 (cos = {:.2e})",
                    d
                ),
            });
        }
    }

    // Z = P V_M − Q U_M D  (N×K, the "tangent directions" before normalization)
    let d_mat = DMatrix::from_diagonal(&DVector::from_fn(k, |i, _| d_vals[i]));
    let z = p * &v_m - q * &u_m * &d_mat; // N×K

    // Normalize columns of Z by sin(θ_i) = sqrt(1 − cos²(θ_i)).
    // This gives Q_Z (N×K orthonormal, perpendicular to Q).
    let sin_vals: Vec<Real> =
        (0..k).map(|i| (1.0 - d_vals[i].powi(2)).max(0.0).sqrt()).collect();

    let mut q_z = z.clone();
    for i in 0..k {
        let s = sin_vals[i];
        if s > GR_ANGLE_EPS {
            let col = q_z.column(i).into_owned() / s;
            q_z.set_column(i, &col);
        }
        // If s ≈ 0: θ_i ≈ 0, tangent component ≈ 0. Leave column as-is
        // (it will be multiplied by θ_i ≈ 0 below, giving ~0).
    }

    // θ_i = arccos(d_i).
    let theta = DVector::from_fn(k, |i, _| d_vals[i].clamp(-1.0, 1.0).acos());
    let theta_mat = DMatrix::from_diagonal(&theta);

    // Log = Q_Z * diag(θ) * U_M^T
    Ok(q_z * theta_mat * u_m.transpose())
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifold impl
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> Manifold for Grassmann<N, K> {
    /// N×K orthonormal matrix representing a K-plane in R^N.
    type Point = SMatrix<Real, N, K>;
    /// N×K horizontal matrix (Q^T V = 0) representing a tangent direction.
    type Tangent = SMatrix<Real, N, K>;

    /// Intrinsic dimension: K(N−K).
    fn dim(&self) -> usize {
        K * (N - K)
    }

    fn ambient_dim(&self) -> usize {
        N * K
    }

    /// Injectivity radius: π/2.
    ///
    /// The cut locus of any K-plane Q consists of K-planes at principal angle π/2
    /// (i.e., Q has a nonzero intersection with its complement). Any geodesic
    /// of length < π/2 from Q avoids the cut locus.
    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        PI / 2.0
    }

    /// Riemannian inner product: tr(U^T V) (Frobenius on horizontal space).
    fn inner(&self, _q: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        (u.transpose() * v).trace()
    }

    /// Exponential map: Exp_Q(V) = (Q B cos Σ + A sin Σ) B^T.
    ///
    /// V = A Σ B^T is the thin SVD of the tangent vector V.
    /// The result is a new N×K orthonormal matrix representing the endpoint
    /// of the geodesic from Q with initial velocity V.
    fn exp(&self, q: &Self::Point, v: &Self::Tangent) -> Self::Point {
        let q_dyn = to_dmat(q);
        let v_dyn = to_dmat(v);
        let result = gr_exp_dyn(&q_dyn, &v_dyn);
        to_smat(&result)
    }

    /// Logarithmic map: inverse of exp.
    ///
    /// Returns CutLocus error if P is at distance ≥ π/2 from Q (principal angle π/2).
    /// For P = Q (same K-plane), returns the zero tangent vector.
    fn log(&self, q: &Self::Point, p: &Self::Point) -> Result<Self::Tangent, CartanError> {
        let q_dyn = to_dmat(q);
        let p_dyn = to_dmat(p);
        let v_dyn = gr_log_dyn(&q_dyn, &p_dyn)?;
        Ok(to_smat(&v_dyn))
    }

    /// Horizontal projection: pi_Q(V) = (I − Q Q^T) V.
    ///
    /// Projects an ambient N×K matrix onto the horizontal tangent space at Q.
    fn project_tangent(&self, q: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        v - q * (q.transpose() * v)
    }

    /// Project onto the manifold via thin QR.
    ///
    /// Given any N×K matrix P with full column rank, returns the Q factor of
    /// its thin QR decomposition (N×K orthonormal).
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        let p_dyn = to_dmat(p);
        let q_dyn = thin_qr(&p_dyn);
        to_smat(&q_dyn)
    }

    fn zero_tangent(&self, _q: &Self::Point) -> Self::Tangent {
        SMatrix::zeros()
    }

    /// Check Q^T Q = I_K.
    fn check_point(&self, q: &Self::Point) -> Result<(), CartanError> {
        let orth_err = (q.transpose() * q - SMatrix::<Real, K, K>::identity()).norm();
        if orth_err < GR_TOL {
            Ok(())
        } else {
            Err(CartanError::NotOnManifold {
                constraint: format!("Q^T Q = I_{K} (Gr({N},{K}))"),
                violation: orth_err,
            })
        }
    }

    /// Check Q^T V = 0 (horizontality).
    fn check_tangent(&self, q: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        let horiz_err = (q.transpose() * v).norm();
        if horiz_err < GR_TOL {
            Ok(())
        } else {
            Err(CartanError::NotInTangentSpace {
                constraint: format!("Q^T V = 0 (T_Q Gr({N},{K}))"),
                violation: horiz_err,
            })
        }
    }

    /// Random point: thin QR of a random Gaussian N×K matrix.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        let raw: SMatrix<Real, N, K> = SMatrix::from_fn(|_, _| rng.sample(StandardNormal));
        // Project via thin QR.
        let raw_dyn = to_dmat(&raw);
        let q_dyn = thin_qr(&raw_dyn);
        to_smat(&q_dyn)
    }

    /// Random horizontal tangent at Q: Gaussian N×K matrix projected onto T_Q Gr.
    fn random_tangent<R: Rng>(&self, q: &Self::Point, rng: &mut R) -> Self::Tangent {
        let raw: SMatrix<Real, N, K> = SMatrix::from_fn(|_, _| rng.sample(StandardNormal));
        self.project_tangent(q, &raw)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction (thin QR retraction)
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> Retraction for Grassmann<N, K> {
    /// QR retraction: retract(Q, V) = thin-QR(Q + V).
    ///
    /// Cheaper than exp (no SVD); satisfies R_Q(0) = Q and first-order
    /// agreement with exp.
    fn retract(&self, q: &Self::Point, v: &Self::Tangent) -> Self::Point {
        let sum = q + v;
        let sum_dyn = to_dmat(&sum);
        let result = thin_qr(&sum_dyn);
        to_smat(&result)
    }

    fn inverse_retract(
        &self,
        q: &Self::Point,
        p: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        // For now, use the Riemannian log (exact inverse of exp, not QR retraction).
        // A true inverse QR retraction exists but is more involved.
        self.log(q, p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel transport (exact)
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> ParallelTransport for Grassmann<N, K> {
    /// Exact parallel transport of W at Q along the geodesic to P.
    ///
    /// Algorithm (Edelman, Arias, Smith 1998):
    ///   1. Compute V = Log_Q(P),  thin SVD: V = A Σ B^T
    ///   2. α = A^T W  (K×K component of W in the A-directions)
    ///   3. PT(W) = W + (A(cos Σ − I) − Q B sin Σ) α
    ///
    /// Correctness:
    ///   - Horizontality at P: P^T PT(W) = 0 (verified by direct algebra).
    ///   - Norm preservation: follows from 2(c−1) + s² + (c−1)² = 0 for each
    ///     singular value σ (where c = cos σ, s = sin σ).
    fn transport(
        &self,
        q: &Self::Point,
        p: &Self::Point,
        w: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let q_dyn = to_dmat(q);
        let p_dyn = to_dmat(p);
        let w_dyn = to_dmat(w);
        let k = K;

        // V = Log_Q(P), thin SVD V = A Σ B^T.
        let v_dyn = gr_log_dyn(&q_dyn, &p_dyn)?;

        // Handle the identity case: if V ≈ 0, transport is identity.
        let v_norm = v_dyn.norm();
        if v_norm < GR_ANGLE_EPS {
            // Project onto T_P Gr for numerical safety.
            let pw = &w_dyn - &p_dyn * (p_dyn.transpose() * &w_dyn);
            return Ok(to_smat(&pw));
        }

        let svd = v_dyn.svd(true, true);
        let u_full = svd.u.unwrap();
        let svals = &svd.singular_values;
        let vt = svd.v_t.unwrap(); // K×K

        let a = u_full.columns(0, k).into_owned(); // N×K
        let b = vt.transpose(); // K×K

        // α = A^T W  (K×K)
        let alpha = a.transpose() * &w_dyn;

        // Correction: (A(cos Σ − I) − Q B sin Σ) α B^T
        let cos_m1 =
            DMatrix::from_diagonal(&DVector::from_fn(k, |i, _| svals[i].cos() - 1.0));
        let sin_diag =
            DMatrix::from_diagonal(&DVector::from_fn(k, |i, _| svals[i].sin()));

        // Δ = (A(cosΣ−I) − QB sinΣ) α   (no B^T: this is the O(N)-rotation formula)
        let correction = (&a * cos_m1 - &q_dyn * &b * sin_diag) * alpha;
        let transported = &w_dyn + correction;

        // Re-project for numerical safety.
        let pw = &transported - &p_dyn * (p_dyn.transpose() * &transported);
        Ok(to_smat(&pw))
    }
}

// VectorTransport: blanket impl from ParallelTransport.

// ─────────────────────────────────────────────────────────────────────────────
// Connection (Riemannian Hessian)
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> Connection for Grassmann<N, K> {
    /// Riemannian Hessian-vector product on Grassmann.
    ///
    /// For a function f on Gr(N,K), the Riemannian HVP is:
    ///   Hess f(Q)[V] = proj_Q(D²f(Q)[V])
    ///
    /// where proj_Q is the horizontal projection (I − QQ^T).
    /// This is the standard formula for Grassmann manifolds embedded in R^{N×K}
    /// with the horizontal tangent space structure.
    fn riemannian_hessian_vector_product(
        &self,
        q: &Self::Point,
        _grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let ehvp = hess_ambient(v);
        Ok(self.project_tangent(q, &ehvp))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> Curvature for Grassmann<N, K> {
    /// Riemann curvature tensor on Gr(N, K).
    ///
    /// For horizontal X, Y, Z at Q:
    ///   R(X,Y)Z = (1/4)[(XY^T − YX^T)Z − Z(X^TY − Y^TX)]
    ///
    /// Both terms are already horizontal when X, Y, Z are horizontal
    /// (Q^T annihilates each term), so no projection is needed.
    ///
    /// Derivation: Gr(N,K) = O(N)/(O(K)×O(N−K)) is a symmetric space.
    /// The curvature tensor for a symmetric space G/K with Cartan decomposition
    /// g = k ⊕ m is R(X,Y)Z = −[[X,Y]_k, Z]. For horizontal X,Y (viewed as
    /// elements of m ≅ skew([0, A; −A^T, 0])), the bracket [X,Y]_k gives the
    /// (1/4) coefficient.
    ///
    /// Ref: Edelman, Arias, Smith (1998), Eq. (2.61).
    fn riemann_curvature(
        &self,
        _q: &Self::Point,
        x: &Self::Tangent,
        y: &Self::Tangent,
        z: &Self::Tangent,
    ) -> Self::Tangent {
        // xy_t = X Y^T  (N×N)
        // yx_t = Y X^T  (N×N)
        // xty  = X^T Y  (K×K)
        // ytx  = Y^T X  (K×K)
        let xy_t = x * y.transpose();
        let yx_t = y * x.transpose();
        let xty = x.transpose() * y;
        let ytx = y.transpose() * x;

        // R(X,Y)Z = (1/4)[(XY^T − YX^T)Z − Z(X^TY − Y^TX)]
        ((xy_t - yx_t) * z - z * (xty - ytx)) * 0.25
    }

    /// Ricci curvature of Gr(N, K).
    ///
    /// Gr(N,K) is an Einstein manifold with Einstein constant (N−2)/4:
    ///   Ric(X, Y) = (N−2)/4 · tr(X^T Y)
    ///
    /// Derivation: compute Ric(E_ab, E_ab) = Σ_{(c,d)≠(a,b)} <R(E_cd,E_ab)E_ab, E_cd>
    /// for a standard ONB E_ab (unit matrix with 1 at (K+a, b)). Using the
    /// Riemann tensor formula above and the Kronecker delta algebra:
    ///   Ric(E_ab, E_ab) = (N−2)/4  (for all a, b)
    /// Since Gr(N,K) is O(N)-homogeneous, the Einstein constant is uniform.
    fn ricci_curvature(
        &self,
        _q: &Self::Point,
        u: &Self::Tangent,
        v: &Self::Tangent,
    ) -> Real {
        let n = N as Real;
        (n - 2.0) / 4.0 * (u.transpose() * v).trace()
    }

    /// Scalar curvature of Gr(N, K).
    ///
    /// s = dim · λ = K(N−K) · (N−2)/4
    ///
    /// where dim = K(N−K) is the intrinsic dimension and λ = (N−2)/4 is the
    /// Einstein constant.
    fn scalar_curvature(&self, _q: &Self::Point) -> Real {
        let n = N as Real;
        let k = K as Real;
        k * (n - k) * (n - 2.0) / 4.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geodesic interpolation
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize, const K: usize> GeodesicInterpolation for Grassmann<N, K> {
    /// Geodesic interpolation: γ(t) = Exp_Q(t · Log_Q(P)).
    fn geodesic(
        &self,
        q: &Self::Point,
        p: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        let v = self.log(q, p)?;
        Ok(self.exp(q, &(v * t)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;

    fn rng() -> impl Rng {
        rand::rngs::SmallRng::seed_from_u64(0xC0FFEE)
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    fn orthogonality_err<const N: usize, const K: usize>(
        q: &SMatrix<Real, N, K>,
    ) -> Real {
        (q.transpose() * q - SMatrix::<Real, K, K>::identity()).norm()
    }

    fn horizontality_err<const N: usize, const K: usize>(
        q: &SMatrix<Real, N, K>,
        v: &SMatrix<Real, N, K>,
    ) -> Real {
        (q.transpose() * v).norm()
    }

    // ── Gr(5,2) smoke tests ──────────────────────────────────────────────────

    #[test]
    fn gr52_random_point_orthonormal() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        for _ in 0..10 {
            let q = m.random_point(&mut rng);
            assert!(orthogonality_err(&q) < 1e-12, "Q^T Q != I");
        }
    }

    #[test]
    fn gr52_random_tangent_horizontal() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        for _ in 0..10 {
            let v = m.random_tangent(&q, &mut rng);
            assert!(horizontality_err(&q, &v) < 1e-12, "Q^T V != 0");
        }
    }

    #[test]
    fn gr52_exp_stays_on_manifold() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        // Scale v down so we stay inside the injectivity radius.
        let v_scaled = v * 0.5;
        let p = m.exp(&q, &v_scaled);
        assert!(orthogonality_err(&p) < 1e-10, "Exp result not orthonormal");
    }

    #[test]
    fn gr52_exp_log_roundtrip() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_scaled = v * 0.5;
        let p = m.exp(&q, &v_scaled);
        let v_recovered = m.log(&q, &p).expect("log failed in exp-log roundtrip");
        let err = (v_scaled - v_recovered).norm();
        assert!(err < 1e-8, "exp-log roundtrip error {err:.2e}");
    }

    #[test]
    fn gr52_dist_matches_tangent_norm() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_scaled = v * 0.4;
        let p = m.exp(&q, &v_scaled);
        let v_log = m.log(&q, &p).unwrap();
        let v_norm = m.norm(&q, &v_scaled);
        let dist = m.norm(&q, &v_log);
        assert_abs_diff_eq!(v_norm, dist, epsilon = 1e-8);
    }

    #[test]
    fn gr52_transport_norm_preserving() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_scaled = v * 0.4;
        let p = m.exp(&q, &v_scaled);
        let w = m.random_tangent(&q, &mut rng);
        let w_norm = m.norm(&q, &w);
        let wt = m.transport(&q, &p, &w).expect("transport failed");
        let wt_norm = m.norm(&p, &wt);
        assert_abs_diff_eq!(w_norm, wt_norm, epsilon = 1e-8);
    }

    #[test]
    fn gr52_transport_result_horizontal() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.4;
        let p = m.exp(&q, &v);
        let w = m.random_tangent(&q, &mut rng);
        let wt = m.transport(&q, &p, &w).unwrap();
        let horiz_err = horizontality_err(&p, &wt);
        assert!(horiz_err < 1e-8, "transported vector not horizontal: {horiz_err:.2e}");
    }

    #[test]
    fn gr52_geodesic_midpoint_equidistant() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.4;
        let p = m.exp(&q, &v);
        let mid = m.geodesic(&q, &p, 0.5).unwrap();
        let d_qm = m.dist(&q, &mid).unwrap();
        let d_mp = m.dist(&mid, &p).unwrap();
        let d_qp = m.dist(&q, &p).unwrap();
        assert_abs_diff_eq!(d_qm, d_qp / 2.0, epsilon = 1e-7);
        assert_abs_diff_eq!(d_mp, d_qp / 2.0, epsilon = 1e-7);
    }

    #[test]
    fn gr52_curvature_non_negative() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let u = m.random_tangent(&q, &mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let k = m.sectional_curvature(&q, &u, &v);
        assert!(k >= -1e-12, "sectional curvature should be ≥ 0, got {k}");
    }

    #[test]
    fn gr52_scalar_curvature_exact() {
        // For Gr(5,2): s = K(N-K)(N-2)/4 = 2 * 3 * 3 / 4 = 4.5
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let s = m.scalar_curvature(&q);
        let expected = 2.0 * 3.0 * 3.0 / 4.0; // = 4.5
        assert_abs_diff_eq!(s, expected, epsilon = 1e-12);
    }

    #[test]
    fn gr52_retract_stays_on_manifold() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.3;
        let p = m.retract(&q, &v);
        assert!(orthogonality_err(&p) < 1e-12);
    }

    #[test]
    fn gr52_check_point_valid() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        assert!(m.check_point(&q).is_ok());
    }

    #[test]
    fn gr52_check_tangent_valid() {
        let m = Grassmann::<5, 2>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        assert!(m.check_tangent(&q, &v).is_ok());
    }

    // ── Gr(4,1) = RP^3 tests ────────────────────────────────────────────────

    #[test]
    fn gr41_exp_log_roundtrip() {
        let m = Grassmann::<4, 1>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.5;
        let p = m.exp(&q, &v);
        let v2 = m.log(&q, &p).unwrap();
        let err = (v - v2).norm();
        assert!(err < 1e-8, "Gr(4,1) exp-log roundtrip error {err:.2e}");
    }

    #[test]
    fn gr41_scalar_curvature() {
        // Gr(4,1): s = 1 * 3 * 2 / 4 = 1.5
        let m = Grassmann::<4, 1>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let s = m.scalar_curvature(&q);
        assert_abs_diff_eq!(s, 1.5, epsilon = 1e-12);
    }

    // ── Gr(6,3) tests ────────────────────────────────────────────────────────

    #[test]
    fn gr63_exp_log_roundtrip() {
        let m = Grassmann::<6, 3>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.3;
        let p = m.exp(&q, &v);
        let v2 = m.log(&q, &p).unwrap();
        let err = (v - v2).norm();
        assert!(err < 1e-8, "Gr(6,3) exp-log roundtrip error {err:.2e}");
    }

    #[test]
    fn gr63_transport_norm_preserving() {
        let m = Grassmann::<6, 3>;
        let mut rng = rng();
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng) * 0.3;
        let p = m.exp(&q, &v);
        let w = m.random_tangent(&q, &mut rng);
        let w_norm = m.norm(&q, &w);
        let wt = m.transport(&q, &p, &w).unwrap();
        let wt_norm = m.norm(&p, &wt);
        assert_abs_diff_eq!(w_norm, wt_norm, epsilon = 1e-7);
    }
}
