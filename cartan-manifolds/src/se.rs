// ~/cartan/cartan-manifolds/src/se.rs

//! Special Euclidean group SE(N) — rigid motions in N dimensions.
//!
//! SE(N) = SO(N) ⋉ R^N is the semidirect product of the rotation group SO(N)
//! and the translation group R^N. Elements of SE(N) represent rigid-body
//! transformations (rotation + translation) in N-dimensional Euclidean space:
//!
//! ```text
//! SE(N) = { (R, t) : R ∈ SO(N), t ∈ R^N }
//! ```
//!
//! The group operation is:
//! ```text
//! (R₁, t₁) · (R₂, t₂) = (R₁ R₂, R₁ t₂ + t₁)
//! ```
//!
//! ## Representation
//!
//! Classically, SE(N) elements are represented as (N+1)×(N+1) homogeneous matrices:
//! ```text
//! T = [[R, t], [0, 1]]  ∈ R^{(N+1)×(N+1)}
//! ```
//! However, this requires `{N+1}` in const-generic position, which needs the unstable
//! `generic_const_exprs` feature in Rust. Instead, we store the rotation and translation
//! components separately via the [`SEPoint`] and [`SETangent`] wrapper types.
//!
//! ## Geometry
//!
//! SE(N) is equipped with a weighted product metric:
//! ```text
//! <(V_rot, v_trans), (W_rot, w_trans)>_{(R,t)} =
//!     (1/2) tr(Ω_V^T Ω_W) + weight · v_body^T w_body
//! ```
//! where `Ω_V = R^T V_rot ∈ so(N)` and `v_body = R^T v_trans ∈ R^N`.
//!
//! The `weight` parameter controls the relative importance of rotation vs. translation
//! in the metric. Common choices:
//! - `weight = 1.0`: equal weighting (default).
//! - `weight >> 1`: translation-dominant (e.g., for large-scale SLAM).
//! - `weight << 1`: rotation-dominant (e.g., for attitude estimation).
//!
//! ## Lie algebra se(N)
//!
//! The Lie algebra of SE(N) consists of "twists":
//! ```text
//! se(N) = { (Ω, v) : Ω ∈ so(N), v ∈ R^N }
//! ```
//! with dimension N(N-1)/2 + N = N(N+1)/2.
//!
//! ## Key formulas
//!
//! The exponential and logarithmic maps on SE(N) couple the rotation and translation
//! components via the **left Jacobian** J(Ω) of SO(N):
//!
//! - **Exp**: `(R, t) · exp(Ω, v) = (R · exp(Ω), t + R · J(Ω) · v)`
//! - **Log**: `log_{(R₁,t₁)}(R₂,t₂) = (R₁ · log(R₁^T R₂), R₁ · J(Ω)^{-1} · R₁^T(t₂ - t₁))`
//!
//! See [`left_jacobian`] and [`left_jacobian_inverse`] in the matrix_exp module.
//!
//! ## Injectivity radius
//!
//! The injectivity radius is π, limited by the SO(N) factor. The R^N factor
//! has infinite injectivity radius (flat), so the bottleneck is the rotation.
//!
//! ## References
//!
//! - Lynch, K. M. & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning,
//!   and Control.* Cambridge University Press. Chapter 3 (rigid-body motions).
//! - Chirikjian, G. S. (2012). *Stochastic Models, Information Theory, and Lie
//!   Groups.* Vol 2, Sections 10.2–10.3 (SE(3) geometry and Jacobians).
//! - Barfoot, T. D. (2017). *State Estimation for Robotics.* Cambridge.
//!   §7.1 (SE(3) exponential coordinates).
//! - Murray, Li, Sastry (1994). *A Mathematical Introduction to Robotic
//!   Manipulation.* Chapter 2 (rigid-body motions and SE(3)).
//! - Sola, J. et al. (2018). "A micro Lie theory for state estimation in robotics."
//!   arXiv:1812.01537. (Concise reference for SE(3) operations.)

use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg};

use nalgebra::{SMatrix, SVector};
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, GeodesicInterpolation, Manifold, ParallelTransport, Real, Retraction,
};

use crate::util::matrix_exp::{left_jacobian, left_jacobian_inverse, matrix_exp_skew};
use crate::util::matrix_log::matrix_log_orthogonal;
use crate::util::skew::{is_skew, skew};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Tolerance for point validation (orthogonality of rotation, det = +1).
///
/// Same as SO(N): 1e-8 is lenient enough for accumulated matrix rounding errors
/// while strict enough to catch grossly invalid inputs.
const VALIDATION_TOL: Real = 1e-8;

/// Tolerance for tangent space check: is R^T V_rot skew-symmetric?
const TANGENT_TOL: Real = 1e-8;

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper types: SEPoint and SETangent
// ─────────────────────────────────────────────────────────────────────────────

/// A point on SE(N): a rigid-body transformation (rotation, translation).
///
/// Stores the rotation matrix R ∈ SO(N) and translation vector t ∈ R^N separately,
/// avoiding the need for (N+1)×(N+1) homogeneous matrices (which would require
/// unstable const-generic arithmetic `{N+1}`).
///
/// ## Invariants
///
/// - `rotation` must be an orthogonal matrix with det = +1 (i.e., in SO(N)).
/// - `translation` is unconstrained (any vector in R^N).
///
/// ## Group operation
///
/// ```text
/// (R₁, t₁) · (R₂, t₂) = (R₁ R₂, R₁ t₂ + t₁)
/// (R, t)^{-1} = (R^T, -R^T t)
/// ```
#[derive(Debug, Clone)]
pub struct SEPoint<const N: usize> {
    /// Rotation component R ∈ SO(N): orthogonal N×N matrix with det(R) = +1.
    pub rotation: SMatrix<Real, N, N>,
    /// Translation component t ∈ R^N: unconstrained vector.
    pub translation: SVector<Real, N>,
}

/// A tangent vector in T_{(R,t)} SE(N).
///
/// Consists of a rotational component V_rot (an N×N matrix in the ambient tangent
/// space of SO(N) at R, i.e., R^T V_rot ∈ so(N)) and a translational component
/// v_trans ∈ R^N.
///
/// ## Ambient representation
///
/// Following the cartan convention, tangent vectors are stored in ambient (extrinsic)
/// coordinates:
/// - `rotation`: V_rot ∈ R^{N×N} such that R^T V_rot is skew-symmetric.
/// - `translation`: v_trans ∈ R^N (unconstrained).
///
/// The corresponding Lie algebra element (body-frame twist) is:
/// - `Ω = R^T V_rot ∈ so(N)` (the right-trivialized rotation velocity)
/// - `v_body = R^T v_trans ∈ R^N` (the body-frame translational velocity)
///
/// ## Arithmetic
///
/// `Add`, `Mul<Real>`, and `Neg` are implemented componentwise for use in generic
/// optimization algorithms (conjugate gradient, Frechet mean, etc.).
#[derive(Debug, Clone)]
pub struct SETangent<const N: usize> {
    /// Rotational tangent: V_rot ∈ R^{N×N} with R^T V_rot ∈ so(N).
    ///
    /// This is the ambient representation of the rotational velocity at R.
    /// Multiply on the left by R^T to get the body-frame skew-symmetric matrix Ω.
    pub rotation: SMatrix<Real, N, N>,
    /// Translational tangent: v_trans ∈ R^N.
    ///
    /// This is the spatial-frame translational velocity. Multiply by R^T to get
    /// the body-frame velocity v_body.
    pub translation: SVector<Real, N>,
}

// ─────────────────────────────────────────────────────────────────────────────
// SETangent arithmetic: Add, Mul<Real>, Neg (required by the Manifold trait)
// ─────────────────────────────────────────────────────────────────────────────

/// Componentwise addition of two SE(N) tangent vectors.
///
/// (V₁_rot, v₁_trans) + (V₂_rot, v₂_trans) = (V₁_rot + V₂_rot, v₁_trans + v₂_trans)
impl<const N: usize> Add for SETangent<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        SETangent {
            rotation: self.rotation + rhs.rotation,
            translation: self.translation + rhs.translation,
        }
    }
}

/// Scalar multiplication of an SE(N) tangent vector.
///
/// (V_rot, v_trans) * s = (s · V_rot, s · v_trans)
impl<const N: usize> Mul<Real> for SETangent<N> {
    type Output = Self;

    fn mul(self, scalar: Real) -> Self {
        SETangent {
            rotation: self.rotation * scalar,
            translation: self.translation * scalar,
        }
    }
}

/// Negation of an SE(N) tangent vector.
///
/// -(V_rot, v_trans) = (-V_rot, -v_trans)
impl<const N: usize> Neg for SETangent<N> {
    type Output = Self;

    fn neg(self) -> Self {
        SETangent {
            rotation: -self.rotation,
            translation: -self.translation,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Struct definition
// ─────────────────────────────────────────────────────────────────────────────

/// The special Euclidean group SE(N): rigid motions in N dimensions.
///
/// SE(N) = SO(N) ⋉ R^N is a Lie group of dimension N(N+1)/2. Elements represent
/// rigid-body transformations (rotation + translation).
///
/// ## Parameters
///
/// - `N`: spatial dimension (e.g., N=2 for planar motions, N=3 for 3D rigid bodies).
/// - `weight`: the weight of the translational component in the Riemannian metric.
///   Controls the relative importance of rotation vs. translation in distance
///   computations, geodesics, and optimization.
///
/// ## Examples
///
/// ```rust,ignore
/// use cartan::manifolds::SpecialEuclidean;
///
/// // SE(3) with default weight
/// let se3 = SpecialEuclidean::<3> { weight: 1.0 };
/// let mut rng = rand::thread_rng();
/// let p = se3.random_point(&mut rng);
/// let v = se3.random_tangent(&p, &mut rng);
/// let q = se3.exp(&p, &v);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SpecialEuclidean<const N: usize> {
    /// Weight of the translational component in the metric.
    ///
    /// The Riemannian inner product is:
    ///   `<u, v> = (1/2) tr(Ω_u^T Ω_v) + weight · v_u^T v_v`
    ///
    /// where Ω_i = R^T V_rot_i and v_i = R^T v_trans_i (body-frame components).
    ///
    /// - `weight = 1.0`: equal weighting (isotropic metric on SE(N))
    /// - `weight > 1`: translation-dominant
    /// - `weight < 1`: rotation-dominant
    pub weight: Real,
}

// ─────────────────────────────────────────────────────────────────────────────
// Manifold implementation
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Manifold for SpecialEuclidean<N> {
    /// Points on SE(N) are (rotation, translation) pairs.
    ///
    /// Stored via the [`SEPoint`] wrapper to avoid (N+1)×(N+1) homogeneous matrices.
    type Point = SEPoint<N>;

    /// Tangent vectors at (R, t) are (V_rot, v_trans) pairs.
    ///
    /// Stored via the [`SETangent`] wrapper. The `Add`, `Mul<Real>`, and `Neg` traits
    /// are implemented componentwise (see above).
    type Tangent = SETangent<N>;

    /// Intrinsic dimension of SE(N) = dim SO(N) + dim R^N = N(N-1)/2 + N = N(N+1)/2.
    ///
    /// Examples:
    /// - SE(2): 2·3/2 = 3 (one rotation angle + two translations)
    /// - SE(3): 3·4/2 = 6 (three rotation angles + three translations)
    fn dim(&self) -> usize {
        N * (N + 1) / 2
    }

    /// Ambient space dimension: N² (rotation matrix) + N (translation vector) = N² + N.
    ///
    /// The rotation is stored as an N×N matrix (N² entries) and the translation as an
    /// N-vector (N entries), giving N² + N total ambient coordinates.
    fn ambient_dim(&self) -> usize {
        N * N + N
    }

    /// Injectivity radius: π (limited by the SO(N) factor).
    ///
    /// The R^N factor has infinite injectivity radius (flat space), so the bottleneck
    /// is the rotation component. The SO(N) cut locus is at rotation angle π.
    ///
    /// This is conservative: the true injectivity radius of SE(N) with the product
    /// metric is at least π (could be larger depending on the weight, but we use
    /// the safe lower bound).
    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        PI
    }

    /// Weighted product Riemannian inner product on T_{(R,t)} SE(N).
    ///
    /// ```text
    /// <u, v>_{(R,t)} = (1/2) tr(Ω_u^T Ω_v) + weight · v_body_u^T v_body_v
    /// ```
    ///
    /// where:
    /// - Ω_u = R^T u_rot, Ω_v = R^T v_rot (body-frame rotation velocities in so(N))
    /// - v_body_u = R^T u_trans, v_body_v = R^T v_trans (body-frame translational velocities)
    ///
    /// ## Why body-frame?
    ///
    /// The inner product is expressed in the body frame (right-trivialized) to ensure
    /// left-invariance: `<u, v>_{g·p} = <u, v>_p` for all g ∈ SE(N). This is a standard
    /// construction for Lie group metrics.
    ///
    /// ## Note on the rotation term
    ///
    /// The factor `(1/2) tr(Ω_u^T Ω_v)` is exactly the bi-invariant metric on SO(N),
    /// matching the convention in `so.rs`. The rotation inner product is independent of R
    /// (bi-invariance), so `(u_rot^T v_rot).trace() * 0.5` gives the same result.
    ///
    /// Ref: Chirikjian (2012), Vol 2, §10.2 (left-invariant metrics on SE(3)).
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        // Rotation inner product: (1/2) tr(U_rot^T V_rot) = (1/2) tr(Ω_u^T Ω_v)
        // Since U_rot = R Ω_u, we have U_rot^T V_rot = Ω_u^T R^T R Ω_v = Ω_u^T Ω_v
        // (R^T R = I for R ∈ SO(N)). So tr(U_rot^T V_rot) = tr(Ω_u^T Ω_v).
        let rotation_inner = (u.rotation.transpose() * v.rotation).trace() * 0.5;

        // Translation inner product: weight · v_body_u^T v_body_v
        // v_body_u = R^T u_trans, v_body_v = R^T v_trans
        // v_body_u^T v_body_v = u_trans^T R R^T v_trans = u_trans^T v_trans
        // (again, R R^T = I), so the body-frame dot product equals the spatial dot product.
        //
        // This means the translation inner product is actually independent of R as well,
        // which is a nice property of the product metric on SE(N).
        let translation_inner = u.translation.dot(&v.translation);

        rotation_inner + self.weight * translation_inner
    }

    /// Exponential map on SE(N): Exp_{(R,t)}(V_rot, v_trans).
    ///
    /// The SE(N) exponential map couples the rotation and translation via the
    /// left Jacobian J(Ω) of SO(N):
    ///
    /// ```text
    /// Ω = R^T V_rot                        (right-trivialize rotation)
    /// v_body = R^T v_trans                  (right-trivialize translation)
    /// R_new = R · exp(Ω)                    (SO(N) exponential)
    /// t_new = t + R · J(Ω) · v_body        (coupled translational update)
    /// ```
    ///
    /// ## Why the left Jacobian?
    ///
    /// In a semidirect product, the translation does not evolve independently of the
    /// rotation along a geodesic. The left Jacobian `J(Ω) = ∫₀¹ exp(sΩ) ds` captures
    /// how the rotation "sweeps" the translational velocity as it integrates from 0 to 1.
    ///
    /// For Ω = 0 (pure translation): J(0) = I, so `t_new = t + R · v_body = t + v_trans`.
    /// This reduces to the flat R^N exponential map, as expected.
    ///
    /// ## References
    ///
    /// - Lynch & Park (2017), eq. (3.88) (SE(3) exponential map).
    /// - Chirikjian (2012), Vol 2, eq. (10.93).
    /// - Barfoot (2017), eq. (7.7).
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        // Step 1: Right-trivialize (pull back to Lie algebra).
        //   Ω = R^T V_rot ∈ so(N)
        //   v_body = R^T v_trans ∈ R^N
        let omega = p.rotation.transpose() * v.rotation;
        let v_body = p.rotation.transpose() * v.translation;

        // Step 2: SO(N) exponential for the rotation component.
        //   exp(Ω) ∈ SO(N), uses Rodrigues (N=3) or Padé (N≥4).
        let exp_omega = matrix_exp_skew(&omega);

        // Step 3: Left Jacobian for the translation coupling.
        //   J(Ω) maps body-frame velocity to the integrated displacement.
        let j_omega = left_jacobian(&omega);

        // Step 4: Assemble the new point.
        //   R_new = R · exp(Ω)
        //   t_new = t + R · J(Ω) · v_body
        SEPoint {
            rotation: p.rotation * exp_omega,
            translation: p.translation + p.rotation * (j_omega * v_body),
        }
    }

    /// Logarithmic map on SE(N): Log_{(R₁,t₁)}(R₂, t₂).
    ///
    /// Inverts the exponential map: finds the tangent vector V at (R₁, t₁) whose
    /// exponential reaches (R₂, t₂).
    ///
    /// ```text
    /// Ω = log(R₁^T R₂)                     (relative rotation in so(N))
    /// Δt = R₁^T (t₂ - t₁)                  (relative translation in body frame)
    /// v_body = J(Ω)^{-1} · Δt              (invert the left Jacobian coupling)
    /// V_rot = R₁ · Ω                        (left-translate to T_{R₁} SO(N))
    /// v_trans = R₁ · v_body                 (left-translate to spatial frame)
    /// ```
    ///
    /// ## Failure modes
    ///
    /// - `CartanError::CutLocus`: rotation component R₁^T R₂ has angle near π.
    /// - `CartanError::NumericalFailure`: J(Ω)^{-1} fails (should not happen within
    ///   the injectivity radius, but can occur for degenerate inputs).
    ///
    /// ## References
    ///
    /// - Lynch & Park (2017), eq. (3.89) (SE(3) logarithm).
    /// - Chirikjian (2012), Vol 2, eq. (10.94).
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        // Step 1: Relative rotation M = R₁^T R₂ ∈ SO(N).
        let m = p.rotation.transpose() * q.rotation;

        // Step 2: Matrix logarithm of M → Ω ∈ so(N).
        //   Fails at the cut locus (rotation angle near π).
        let omega = matrix_log_orthogonal(&m)?;

        // Step 3: Relative translation in body frame.
        //   Δt = R₁^T (t₂ - t₁)
        let delta_t = p.rotation.transpose() * (q.translation - p.translation);

        // Step 4: Invert the left Jacobian coupling.
        //   v_body = J(Ω)^{-1} · Δt
        let j_inv = left_jacobian_inverse(&omega).ok_or_else(|| CartanError::NumericalFailure {
            operation: "log(SE(N))".to_string(),
            message: "left Jacobian J(Ω) is singular — rotation may have angle at 2kπ \
                      where k ≥ 1, causing the Jacobian to degenerate."
                .to_string(),
        })?;
        let v_body = j_inv * delta_t;

        // Step 5: Left-translate back to ambient coordinates at (R₁, t₁).
        //   V_rot = R₁ · Ω
        //   v_trans = R₁ · v_body
        Ok(SETangent {
            rotation: p.rotation * omega,
            translation: p.rotation * v_body,
        })
    }

    /// Project an ambient tangent vector onto T_{(R,t)} SE(N).
    ///
    /// The tangent space at (R, t) ∈ SE(N) is:
    /// ```text
    /// T_{(R,t)} SE(N) = { (R Ω, v) : Ω ∈ so(N), v ∈ R^N }
    /// ```
    ///
    /// The rotation component must satisfy R^T V_rot ∈ so(N). To project an
    /// arbitrary matrix V_rot, we extract the skew-symmetric part of R^T V_rot:
    /// ```text
    /// V_rot_proj = R · skew(R^T V_rot)
    /// ```
    ///
    /// The translation component is unconstrained, so it passes through unchanged.
    fn project_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        // Project the rotation component onto T_R SO(N).
        let a = p.rotation.transpose() * v.rotation;
        let omega = skew(&a);
        SETangent {
            rotation: p.rotation * omega,
            // Translation is unconstrained — no projection needed.
            translation: v.translation,
        }
    }

    /// Project an ambient (R, t) pair onto SE(N).
    ///
    /// Projects the rotation component onto SO(N) via the Newton polar iteration
    /// (same algorithm as SO(N)), and keeps the translation unchanged.
    ///
    /// ## Algorithm
    ///
    /// 1. Apply Newton polar iteration to R to find the nearest orthogonal matrix Q.
    /// 2. If det(Q) = -1, flip the first column to get det = +1 (project O(N) → SO(N)).
    /// 3. Return (Q, t).
    ///
    /// See `so.rs::project_point` for detailed documentation of the Newton polar iteration.
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        // Project rotation onto SO(N) using Newton polar iteration.
        // This reuses the same algorithm as SpecialOrthogonal<N>::project_point,
        // but we inline it here to avoid circular dependencies.
        let q = project_to_so_n(&p.rotation);

        SEPoint {
            rotation: q,
            // Translation is unconstrained — no projection needed.
            translation: p.translation,
        }
    }

    /// Zero tangent vector at (R, t): both components are zero.
    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        SETangent {
            rotation: SMatrix::zeros(),
            translation: SVector::zeros(),
        }
    }

    /// Validate that (R, t) is a point on SE(N).
    ///
    /// Checks that R ∈ SO(N):
    /// 1. ||R^T R - I||_F < VALIDATION_TOL (orthogonality)
    /// 2. det(R) ≈ +1 (positive orientation)
    ///
    /// The translation t is unconstrained, so it always passes.
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError> {
        // Check orthogonality: ||R^T R - I||_F
        let id = SMatrix::<Real, N, N>::identity();
        let rtr = p.rotation.transpose() * p.rotation;
        let ortho_violation = (rtr - id).norm();

        if ortho_violation >= VALIDATION_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: format!("R^T R = I (rotation component of SE({}))", N),
                violation: ortho_violation,
            });
        }

        // Check determinant sign using Gaussian elimination.
        // We use the same gauss_det_sign helper as SO(N) to avoid nalgebra's
        // Const<N>: ToTypenum requirement.
        let det_sign = gauss_det_sign(&p.rotation);
        let det_violation = (det_sign - 1.0).abs();

        if det_violation >= VALIDATION_TOL {
            return Err(CartanError::NotOnManifold {
                constraint: format!("det(R) = +1 (rotation component of SE({}))", N),
                violation: det_violation,
            });
        }

        Ok(())
    }

    /// Validate that (V_rot, v_trans) is a tangent vector at (R, t) ∈ SE(N).
    ///
    /// Checks that R^T V_rot is skew-symmetric (i.e., V_rot ∈ T_R SO(N)).
    /// The translation component v_trans is unconstrained.
    fn check_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError> {
        // Check that R^T V_rot is skew-symmetric.
        let omega_approx = p.rotation.transpose() * v.rotation;

        if !is_skew(&omega_approx, TANGENT_TOL) {
            let violation = (omega_approx + omega_approx.transpose()).norm();
            return Err(CartanError::NotInTangentSpace {
                constraint: format!("R^T V_rot is skew-symmetric (T_{{(R,t)}} SE({}))", N),
                violation,
            });
        }

        Ok(())
    }

    /// Random point on SE(N): Haar-uniform rotation + standard Gaussian translation.
    ///
    /// The rotation is sampled from the Haar measure on SO(N) using the Householder QR
    /// method (Mezzadri 2006), identical to `SpecialOrthogonal<N>::random_point`.
    /// The translation is sampled from N(0, I_N) (standard Gaussian in each coordinate).
    ///
    /// For the translation, we use standard normal because SE(N) is non-compact — there
    /// is no "uniform" distribution on R^N. Standard Gaussian is the natural default.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        // Sample Haar-uniform rotation via Householder QR.
        let rotation = random_so_n(rng);

        // Sample standard Gaussian translation.
        let translation = SVector::from_fn(|_, _| rng.sample(StandardNormal));

        SEPoint {
            rotation,
            translation,
        }
    }

    /// Random tangent vector at (R, t): random skew rotation + Gaussian translation.
    ///
    /// The rotation component is a random element of T_R SO(N): sample a Gaussian
    /// on so(N) and left-translate to T_R SO(N).
    /// The translation component is standard Gaussian in R^N.
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent {
        // Random rotation tangent: R · skew(Gaussian) ∈ T_R SO(N)
        let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample(StandardNormal));
        let omega = skew(&g);
        let v_rot = p.rotation * omega;

        // Random translation tangent: standard Gaussian in R^N
        let v_trans = SVector::from_fn(|_, _| rng.sample(StandardNormal));

        SETangent {
            rotation: v_rot,
            translation: v_trans,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction — Cayley × Euclidean product retraction
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Retraction for SpecialEuclidean<N> {
    /// Cayley × Euclidean product retraction on SE(N).
    ///
    /// The retraction is a product of:
    /// - **Rotation:** Cayley retraction on SO(N): `R_new = R · (I - Ω/2)^{-1} (I + Ω/2)`
    /// - **Translation:** Euclidean retraction (addition): `t_new = t + v_trans`
    ///
    /// This product retraction satisfies the retraction axioms:
    /// 1. `retract(p, 0) = p` ✓ (Cayley(0) = I, translation += 0)
    /// 2. `d/ds retract(p, s·v)|_{s=0} = v` ✓ (first-order match for both components)
    ///
    /// ## Advantages over exp
    ///
    /// - Avoids matrix exponential and left Jacobian computation.
    /// - Only requires a matrix inverse (Cayley) + vector addition.
    /// - Sufficient for first-order optimization (gradient descent, CG).
    ///
    /// ## Note: decoupled retraction
    ///
    /// Unlike the exact exp (which couples rotation and translation via J(Ω)),
    /// this retraction treats them independently. This is a valid retraction
    /// because the coupling only affects second-order behavior, and retractions
    /// are required to match exp only to first order.
    ///
    /// Ref: Absil-Mahony-Sepulchre (2008), §4.1.1 (product retractions).
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        // Rotation: Cayley retraction on SO(N).
        //   Ω = R^T V_rot ∈ so(N)
        //   R_new = R · (I - Ω/2)^{-1} (I + Ω/2)
        let omega = p.rotation.transpose() * v.rotation;
        let id = SMatrix::<Real, N, N>::identity();
        let half_omega = &omega * 0.5;
        let lhs = id - half_omega;
        let rhs = id + half_omega;

        // (I - Ω/2) is always invertible for skew Ω (eigenvalues have positive real part).
        let lhs_inv = lhs.try_inverse().expect(
            "SE(N) Cayley retraction: (I - Ω/2) is singular — should not occur for skew Ω",
        );
        let cayley = lhs_inv * rhs;

        SEPoint {
            rotation: p.rotation * cayley,
            // Translation: simple Euclidean retraction (addition).
            translation: p.translation + v.translation,
        }
    }

    /// Inverse of the Cayley × Euclidean product retraction.
    ///
    /// Given (R₁, t₁) and (R₂, t₂), find V such that retract((R₁,t₁), V) = (R₂, t₂).
    ///
    /// - **Rotation:** Inverse Cayley on SO(N):
    ///   `Ω = 2 (M + I)^{-1} (M - I)` where `M = R₁^T R₂`, then `V_rot = R₁ Ω`.
    /// - **Translation:** `v_trans = t₂ - t₁`.
    ///
    /// ## Failure modes
    ///
    /// `(M + I)` is singular when R₁^T R₂ has eigenvalue -1 (rotation by π).
    /// Returns `CartanError::NumericalFailure` in this case.
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        // Rotation: inverse Cayley on SO(N).
        let m = p.rotation.transpose() * q.rotation;
        let id = SMatrix::<Real, N, N>::identity();
        let m_plus_i = m + id;
        let m_minus_i = m - id;

        let m_plus_i_inv =
            m_plus_i
                .try_inverse()
                .ok_or_else(|| CartanError::NumericalFailure {
                    operation: "inverse_retract(SE(N))".to_string(),
                    message: "matrix (M + I) is singular — R₁^T R₂ has eigenvalue -1 \
                          (Cayley cut locus). Consider using log instead."
                        .to_string(),
                })?;
        let half_omega = m_plus_i_inv * m_minus_i;
        let omega = half_omega * 2.0;

        Ok(SETangent {
            rotation: p.rotation * omega,
            // Translation: simple subtraction.
            translation: q.translation - p.translation,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel transport — approximate product transport
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> ParallelTransport for SpecialEuclidean<N> {
    /// Approximate parallel transport on SE(N).
    ///
    /// Transports a tangent vector u from T_p SE(N) to T_q SE(N) using a product
    /// approximation:
    ///
    /// - **Rotation component:** Left-translation transport from SO(N):
    ///   `u_rot_transported = Q R^T u_rot` (exact for the bi-invariant metric on SO(N)).
    /// - **Translation component:** Identity transport (R^N is flat):
    ///   `u_trans_transported = u_trans`.
    ///
    /// ## Approximation quality
    ///
    /// This is the exact parallel transport for a product metric SO(N) × R^N (since
    /// each factor's transport is exact). For the semidirect product metric on SE(N),
    /// this is an approximation — the coupling between rotation and translation
    /// introduces O(||Ω||²) corrections that we neglect.
    ///
    /// For first-order optimization algorithms (CG, LBFGS), this approximation is
    /// sufficient because vector transport only needs to be an approximate isometry.
    ///
    /// Ref: Absil-Mahony-Sepulchre (2008), §8.1 (vector transport on product manifolds).
    fn transport(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // Rotation transport: Q R^T u_rot (left-translation, exact for SO(N) bi-invariant metric).
        let transported_rot = q.rotation * (p.rotation.transpose() * u.rotation);

        // Translation transport: identity (R^N is flat — parallel transport is trivial).
        Ok(SETangent {
            rotation: transported_rot,
            translation: u.translation,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection — Riemannian Hessian via tangent projection
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> Connection for SpecialEuclidean<N> {
    /// Riemannian Hessian-vector product on SE(N).
    ///
    /// Uses the projection formula: `Hess f [v] ≈ project_tangent(p, ehvp)`
    /// where ehvp is the ambient (Euclidean) Hessian-vector product.
    ///
    /// This is the dominant term of the Riemannian HVP. For many practical cost
    /// functions, the Weingarten correction is either zero or absorbed into the
    /// ambient computation.
    ///
    /// See `so.rs::Connection` for detailed discussion of this approximation.
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        _grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        let ehvp = hess_ambient(v);
        Ok(self.project_tangent(p, &ehvp))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geodesic interpolation
// ─────────────────────────────────────────────────────────────────────────────

impl<const N: usize> GeodesicInterpolation for SpecialEuclidean<N> {
    /// Interpolate along the geodesic from p to q at parameter t.
    ///
    /// ```text
    /// γ(t) = Exp_p(t · Log_p(q))
    /// ```
    ///
    /// - t = 0: returns p
    /// - t = 1: returns q
    /// - t = 0.5: geodesic midpoint
    ///
    /// Fails if Log_p(q) fails (rotation component at the cut locus).
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        let v = self.log(p, q)?;
        Ok(self.exp(p, &(v * t)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Project an N×N matrix onto SO(N) via Newton polar iteration.
///
/// This is the same algorithm used in `so.rs::project_point`, duplicated here
/// to avoid coupling SE(N) to the SO(N) struct. Both use the Newton iteration
/// `Q_{k+1} = (Q_k + Q_k^{-T}) / 2` which converges quadratically to the
/// orthogonal polar factor.
///
/// If det(Q) = -1 after convergence, the first column is flipped to get SO(N).
///
/// Ref: Higham (2008), Algorithm 8.20.
fn project_to_so_n<const N: usize>(p: &SMatrix<Real, N, N>) -> SMatrix<Real, N, N> {
    // Degenerate case: near-zero matrix → return identity.
    if p.norm() < 1e-15 {
        return SMatrix::<Real, N, N>::identity();
    }

    // Newton polar iteration.
    const MAX_ITERS: usize = 20;
    let mut q = *p;

    for _ in 0..MAX_ITERS {
        let q_inv = match q.try_inverse() {
            Some(inv) => inv,
            None => break,
        };
        let q_new = (q + q_inv.transpose()) * 0.5;
        let change = (q_new - q).norm();
        let scale = q_new.norm().max(1e-15);
        q = q_new;
        if change / scale < 1e-14 {
            break;
        }
    }

    // Fix determinant sign: flip first column if det = -1.
    let det_sign = gauss_det_sign(&q);
    if det_sign < 0.0 {
        for i in 0..N {
            q[(i, 0)] = -q[(i, 0)];
        }
    }

    q
}

/// Compute the SIGN of the determinant via Gaussian elimination with partial pivoting.
///
/// Returns +1.0 if det > 0, -1.0 if det < 0, 0.0 if singular.
/// See `so.rs::gauss_det_sign` for detailed documentation.
#[allow(clippy::needless_range_loop)]
fn gauss_det_sign<const N: usize>(a: &SMatrix<Real, N, N>) -> Real {
    let mut mat = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            mat[i][j] = a[(i, j)];
        }
    }

    let mut det_sign: Real = 1.0;

    for col in 0..N {
        // Partial pivoting: find row with largest absolute value in this column.
        let mut pivot_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..N {
            if mat[row][col].abs() > max_val {
                max_val = mat[row][col].abs();
                pivot_row = row;
            }
        }

        if max_val < 1e-15 {
            return 0.0; // singular
        }

        // Swap rows if needed.
        if pivot_row != col {
            for j in 0..N {
                let tmp = mat[col][j];
                mat[col][j] = mat[pivot_row][j];
                mat[pivot_row][j] = tmp;
            }
            det_sign = -det_sign;
        }

        if mat[col][col] < 0.0 {
            det_sign = -det_sign;
        }

        // Eliminate below pivot.
        for row in (col + 1)..N {
            if mat[col][col].abs() < 1e-300 {
                return 0.0;
            }
            let factor = mat[row][col] / mat[col][col];
            for j in col..N {
                mat[row][j] -= factor * mat[col][j];
            }
        }
    }

    det_sign
}

/// Sample a Haar-uniform random rotation matrix in SO(N).
///
/// Uses the Householder QR method of Mezzadri (2006):
/// 1. Sample G ~ N(0,1)^{N×N}
/// 2. Compute QR decomposition via Householder reflectors
/// 3. Apply sign corrections for Haar uniformity
/// 4. Flip a column if det = -1 to land in SO(N)
///
/// This is the same algorithm as `so.rs::random_point`, duplicated here
/// for self-containment of the SE(N) module.
///
/// Ref: Mezzadri (2006), "How to Generate Random Matrices from the Classical
/// Compact Groups." Notices AMS 54(5), §2.
fn random_so_n<const N: usize, R: Rng>(rng: &mut R) -> SMatrix<Real, N, N> {
    // Step 1: Sample G ~ N(0,1)^{N×N}.
    let g: SMatrix<Real, N, N> = SMatrix::from_fn(|_, _| rng.sample(StandardNormal));

    // Step 2: Householder QR decomposition.
    let (q_out, r_diag_signs) = householder_qr(&g);

    // Step 3: Apply sign corrections for Haar measure.
    let mut q = q_out;
    for j in 0..N {
        if r_diag_signs[j] < 0.0 {
            for i in 0..N {
                q[(i, j)] *= -1.0;
            }
        }
    }

    // Step 4: Correct for det = -1.
    let det_sign = gauss_det_sign(&q);
    if det_sign < 0.0 {
        for i in 0..N {
            q[(i, 0)] = -q[(i, 0)];
        }
    }

    q
}

/// Householder QR decomposition of an N×N matrix.
///
/// Returns (Q, r_diag_signs) where Q is orthogonal and r_diag_signs[j] is the
/// sign of R[j,j] in the upper-triangular factor R.
///
/// This is a copy of the Householder QR implementation in `so.rs`, included here
/// for self-containment. See `so.rs::householder_qr` for detailed documentation.
///
/// Ref: Golub & Van Loan (2013), §5.2.1.
#[allow(clippy::needless_range_loop)]
fn householder_qr<const N: usize>(
    g: &SMatrix<Real, N, N>,
) -> (SMatrix<Real, N, N>, [Real; N]) {
    let mut a = [[0.0f64; N]; N];
    for i in 0..N {
        for j in 0..N {
            a[i][j] = g[(i, j)];
        }
    }

    let mut q = [[0.0f64; N]; N];
    for i in 0..N {
        q[i][i] = 1.0;
    }

    let mut r_diag_signs = [1.0f64; N];

    for k in 0..N {
        // Extract subcolumn and compute norm.
        let mut x_norm_sq = 0.0f64;
        for i in k..N {
            x_norm_sq += a[i][k] * a[i][k];
        }
        let x_norm = x_norm_sq.sqrt();

        let sign_xk = if a[k][k] >= 0.0 { 1.0 } else { -1.0 };
        let sigma = sign_xk * x_norm;

        r_diag_signs[k] = if sigma.abs() < 1e-15 {
            1.0
        } else {
            -sign_xk
        };

        // Householder vector v.
        let mut v = [0.0f64; N];
        for i in k..N {
            v[i] = a[i][k];
        }
        v[k] += sigma;

        let v_norm_sq: f64 = v[k..N].iter().map(|&vi| vi * vi).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }
        let v_norm_sq_inv = 1.0 / v_norm_sq;

        // Apply H_k to A.
        for j in k..N {
            let mut beta: f64 = 0.0;
            for i in k..N {
                beta += v[i] * a[i][j];
            }
            beta *= 2.0 * v_norm_sq_inv;
            for i in k..N {
                a[i][j] -= beta * v[i];
            }
        }

        // Apply H_k to Q.
        for i in 0..N {
            let mut beta: f64 = 0.0;
            for l in k..N {
                beta += q[i][l] * v[l];
            }
            beta *= 2.0 * v_norm_sq_inv;
            for l in k..N {
                q[i][l] -= beta * v[l];
            }
        }
    }

    let mut q_out = SMatrix::<Real, N, N>::zeros();
    for i in 0..N {
        for j in 0..N {
            q_out[(i, j)] = q[i][j];
        }
    }

    (q_out, r_diag_signs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SMatrix, SVector};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    // Tolerances for different test categories.
    // TIGHT: for exact/identity cases.
    // MED: for roundtrip tests with floating-point accumulation.
    const TIGHT: Real = 1e-12;
    const MED: Real = 1e-8;

    /// Deterministic RNG for reproducible tests.
    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    /// Construct an SE(3) instance with default weight.
    fn se3() -> SpecialEuclidean<3> {
        SpecialEuclidean::<3> { weight: 1.0 }
    }

    /// Construct an SE(2) instance with default weight.
    fn se2() -> SpecialEuclidean<2> {
        SpecialEuclidean::<2> { weight: 1.0 }
    }

    /// The identity element of SE(N): (I, 0).
    fn identity<const N: usize>() -> SEPoint<N> {
        SEPoint {
            rotation: SMatrix::<Real, N, N>::identity(),
            translation: SVector::<Real, N>::zeros(),
        }
    }

    // ── Point validation ─────────────────────────────────────────────────────

    /// check_point accepts the identity.
    #[test]
    fn test_check_point_identity_3d() {
        let m = se3();
        assert!(m.check_point(&identity::<3>()).is_ok());
    }

    /// check_point accepts a random valid point.
    #[test]
    fn test_check_point_random_3d() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        assert!(
            m.check_point(&p).is_ok(),
            "random point should be valid: {:?}",
            m.check_point(&p)
        );
    }

    /// check_point rejects a point with non-orthogonal rotation.
    #[test]
    fn test_check_point_rejects_bad_rotation() {
        let m = se3();
        let p = SEPoint {
            rotation: SMatrix::<Real, 3, 3>::from_row_slice(&[
                1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            translation: SVector::<Real, 3>::zeros(),
        };
        assert!(m.check_point(&p).is_err());
    }

    // ── Tangent validation ───────────────────────────────────────────────────

    /// check_tangent accepts the zero tangent.
    #[test]
    fn test_check_tangent_zero() {
        let m = se3();
        let p = identity::<3>();
        let v = m.zero_tangent(&p);
        assert!(m.check_tangent(&p, &v).is_ok());
    }

    /// check_tangent accepts a properly constructed tangent.
    #[test]
    fn test_check_tangent_random() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = m.random_tangent(&p, &mut r);
        assert!(
            m.check_tangent(&p, &v).is_ok(),
            "random tangent should be valid"
        );
    }

    // ── Dimension checks ─────────────────────────────────────────────────────

    /// SE(3) has dim = 6, ambient_dim = 12.
    #[test]
    fn test_dim_se3() {
        let m = se3();
        assert_eq!(m.dim(), 6, "SE(3) should have dim = 6");
        assert_eq!(m.ambient_dim(), 12, "SE(3) should have ambient_dim = 12");
    }

    /// SE(2) has dim = 3, ambient_dim = 6.
    #[test]
    fn test_dim_se2() {
        let m = se2();
        assert_eq!(m.dim(), 3, "SE(2) should have dim = 3");
        assert_eq!(m.ambient_dim(), 6, "SE(2) should have ambient_dim = 6");
    }

    // ── Exp/Log roundtrip ────────────────────────────────────────────────────

    /// Exp then Log roundtrip: log(p, exp(p, v)) ≈ v for small v.
    #[test]
    fn test_exp_log_roundtrip_3d() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);

        // Create a small tangent vector (well within injectivity radius).
        let v_full = m.random_tangent(&p, &mut r);
        let v = SETangent {
            rotation: v_full.rotation * 0.3,
            translation: v_full.translation * 0.3,
        };

        // Exp then Log.
        let q = m.exp(&p, &v);
        let v_recovered = m.log(&p, &q).expect("log should succeed for small tangent");

        // Compare rotation and translation components.
        let rot_err = (&v_recovered.rotation - &v.rotation).norm();
        let trans_err = (&v_recovered.translation - &v.translation).norm();
        assert!(
            rot_err < MED,
            "rotation roundtrip error: {:.2e}",
            rot_err
        );
        assert!(
            trans_err < MED,
            "translation roundtrip error: {:.2e}",
            trans_err
        );
    }

    /// Log then Exp roundtrip: exp(p, log(p, q)) ≈ q for nearby points.
    #[test]
    fn test_log_exp_roundtrip_3d() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);

        // Create q close to p via a small exponential step.
        let v_small = SETangent {
            rotation: m.random_tangent(&p, &mut r).rotation * 0.2,
            translation: m.random_tangent(&p, &mut r).translation * 0.2,
        };
        let q = m.exp(&p, &v_small);

        // Log then Exp.
        let v = m.log(&p, &q).expect("log should succeed");
        let q_recovered = m.exp(&p, &v);

        let rot_err = (&q_recovered.rotation - &q.rotation).norm();
        let trans_err = (&q_recovered.translation - &q.translation).norm();
        assert!(rot_err < MED, "rotation roundtrip error: {:.2e}", rot_err);
        assert!(
            trans_err < MED,
            "translation roundtrip error: {:.2e}",
            trans_err
        );
    }

    /// Exp/Log roundtrip for SE(2).
    #[test]
    fn test_exp_log_roundtrip_2d() {
        let m = se2();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v_full = m.random_tangent(&p, &mut r);
        let v = SETangent {
            rotation: v_full.rotation * 0.3,
            translation: v_full.translation * 0.3,
        };

        let q = m.exp(&p, &v);
        let v_recovered = m.log(&p, &q).expect("log should succeed for SE(2)");

        let rot_err = (&v_recovered.rotation - &v.rotation).norm();
        let trans_err = (&v_recovered.translation - &v.translation).norm();
        assert!(rot_err < MED, "SE(2) rotation roundtrip error: {:.2e}", rot_err);
        assert!(
            trans_err < MED,
            "SE(2) translation roundtrip error: {:.2e}",
            trans_err
        );
    }

    // ── Exp at zero tangent ──────────────────────────────────────────────────

    /// exp(p, 0) = p.
    #[test]
    fn test_exp_zero_tangent() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let zero = m.zero_tangent(&p);
        let q = m.exp(&p, &zero);

        let rot_err = (&q.rotation - &p.rotation).norm();
        let trans_err = (&q.translation - &p.translation).norm();
        assert!(rot_err < TIGHT, "exp(p, 0) rotation ≠ p: {:.2e}", rot_err);
        assert!(
            trans_err < TIGHT,
            "exp(p, 0) translation ≠ p: {:.2e}",
            trans_err
        );
    }

    // ── Inner product ────────────────────────────────────────────────────────

    /// Inner product is positive for nonzero tangent.
    #[test]
    fn test_inner_product_positive() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = m.random_tangent(&p, &mut r);
        let ip = m.inner(&p, &v, &v);
        assert!(ip > 0.0, "inner product should be positive: got {}", ip);
    }

    /// Inner product is symmetric: <u,v> = <v,u>.
    #[test]
    fn test_inner_product_symmetric() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let u = m.random_tangent(&p, &mut r);
        let v = m.random_tangent(&p, &mut r);
        let uv = m.inner(&p, &u, &v);
        let vu = m.inner(&p, &v, &u);
        assert!(
            (uv - vu).abs() < TIGHT,
            "<u,v> ≠ <v,u>: diff = {:.2e}",
            (uv - vu).abs()
        );
    }

    /// Zero tangent has zero norm.
    #[test]
    fn test_zero_tangent_norm() {
        let m = se3();
        let p = identity::<3>();
        let z = m.zero_tangent(&p);
        let n = m.norm(&p, &z);
        assert!(n < TIGHT, "||zero_tangent|| = {:.2e} (expected 0)", n);
    }

    // ── Retraction ───────────────────────────────────────────────────────────

    /// retract(p, 0) = p.
    #[test]
    fn test_retract_zero() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let zero = m.zero_tangent(&p);
        let q = m.retract(&p, &zero);

        let rot_err = (&q.rotation - &p.rotation).norm();
        let trans_err = (&q.translation - &p.translation).norm();
        assert!(rot_err < TIGHT, "retract(p, 0) rotation ≠ p");
        assert!(trans_err < TIGHT, "retract(p, 0) translation ≠ p");
    }

    /// Retraction result lies on SE(N).
    #[test]
    fn test_retract_on_manifold() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = m.random_tangent(&p, &mut r);
        let q = m.retract(&p, &(v * 0.1));
        assert!(
            m.check_point(&q).is_ok(),
            "retracted point should be on manifold"
        );
    }

    /// Inverse retraction roundtrip: inverse_retract(p, retract(p, v)) ≈ v.
    #[test]
    fn test_inverse_retract_roundtrip() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v_full = m.random_tangent(&p, &mut r);
        let v = SETangent {
            rotation: v_full.rotation * 0.2,
            translation: v_full.translation * 0.2,
        };

        let q = m.retract(&p, &v);
        let v_recovered = m
            .inverse_retract(&p, &q)
            .expect("inverse_retract should succeed");

        let rot_err = (&v_recovered.rotation - &v.rotation).norm();
        let trans_err = (&v_recovered.translation - &v.translation).norm();
        assert!(rot_err < MED, "inverse_retract rotation error: {:.2e}", rot_err);
        assert!(
            trans_err < MED,
            "inverse_retract translation error: {:.2e}",
            trans_err
        );
    }

    // ── Parallel transport ───────────────────────────────────────────────────

    /// Parallel transport preserves norm (approximately).
    #[test]
    fn test_transport_preserves_norm() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = SETangent {
            rotation: m.random_tangent(&p, &mut r).rotation * 0.3,
            translation: m.random_tangent(&p, &mut r).translation * 0.3,
        };
        let q = m.exp(&p, &v);

        let u = m.random_tangent(&p, &mut r);
        let u_transported = m.transport(&p, &q, &u).expect("transport should succeed");

        let norm_before = m.norm(&p, &u);
        let norm_after = m.norm(&q, &u_transported);

        // For the product transport approximation, norm is preserved exactly for each factor.
        // The rotation norm is preserved (left-translation is isometric for bi-invariant metric).
        // The translation norm is preserved (identity transport).
        let rel_err = (norm_after - norm_before).abs() / norm_before.max(1e-15);
        assert!(
            rel_err < MED,
            "transport changed norm: before = {:.6}, after = {:.6}, rel_err = {:.2e}",
            norm_before,
            norm_after,
            rel_err
        );
    }

    // ── Geodesic interpolation ───────────────────────────────────────────────

    /// geodesic(p, q, 0) = p.
    #[test]
    fn test_geodesic_at_zero() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = SETangent {
            rotation: m.random_tangent(&p, &mut r).rotation * 0.3,
            translation: m.random_tangent(&p, &mut r).translation * 0.3,
        };
        let q = m.exp(&p, &v);

        let g0 = m.geodesic(&p, &q, 0.0).expect("geodesic(0) should succeed");
        let rot_err = (&g0.rotation - &p.rotation).norm();
        let trans_err = (&g0.translation - &p.translation).norm();
        assert!(rot_err < MED, "geodesic(p,q,0) rotation ≠ p");
        assert!(trans_err < MED, "geodesic(p,q,0) translation ≠ p");
    }

    /// geodesic(p, q, 1) = q.
    #[test]
    fn test_geodesic_at_one() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = SETangent {
            rotation: m.random_tangent(&p, &mut r).rotation * 0.3,
            translation: m.random_tangent(&p, &mut r).translation * 0.3,
        };
        let q = m.exp(&p, &v);

        let g1 = m.geodesic(&p, &q, 1.0).expect("geodesic(1) should succeed");
        let rot_err = (&g1.rotation - &q.rotation).norm();
        let trans_err = (&g1.translation - &q.translation).norm();
        assert!(rot_err < MED, "geodesic(p,q,1) rotation ≠ q");
        assert!(trans_err < MED, "geodesic(p,q,1) translation ≠ q");
    }

    // ── Project tangent ──────────────────────────────────────────────────────

    /// project_tangent is idempotent.
    #[test]
    fn test_project_tangent_idempotent() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = m.random_tangent(&p, &mut r);

        let proj1 = m.project_tangent(&p, &v);
        let proj2 = m.project_tangent(&p, &proj1);

        let rot_err = (&proj2.rotation - &proj1.rotation).norm();
        let trans_err = (&proj2.translation - &proj1.translation).norm();
        assert!(rot_err < TIGHT, "project_tangent not idempotent (rotation)");
        assert!(
            trans_err < TIGHT,
            "project_tangent not idempotent (translation)"
        );
    }

    // ── Exp output lies on manifold ──────────────────────────────────────────

    /// exp result is a valid SE(N) point.
    #[test]
    fn test_exp_on_manifold() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = m.random_tangent(&p, &mut r);
        let q = m.exp(&p, &v);
        assert!(
            m.check_point(&q).is_ok(),
            "exp result should be on manifold: {:?}",
            m.check_point(&q)
        );
    }

    // ── Distance ─────────────────────────────────────────────────────────────

    /// dist(p, p) = 0.
    #[test]
    fn test_dist_self_zero() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let d = m.dist(&p, &p).expect("dist(p, p) should succeed");
        assert!(d < TIGHT, "dist(p, p) = {:.2e} (expected 0)", d);
    }

    /// dist is symmetric: dist(p, q) = dist(q, p).
    #[test]
    fn test_dist_symmetric() {
        let m = se3();
        let mut r = rng();
        let p = m.random_point(&mut r);
        let v = SETangent {
            rotation: m.random_tangent(&p, &mut r).rotation * 0.3,
            translation: m.random_tangent(&p, &mut r).translation * 0.3,
        };
        let q = m.exp(&p, &v);

        let d_pq = m.dist(&p, &q).expect("dist(p,q) should succeed");
        let d_qp = m.dist(&q, &p).expect("dist(q,p) should succeed");
        let rel_err = (d_pq - d_qp).abs() / d_pq.max(1e-15);
        assert!(
            rel_err < MED,
            "dist not symmetric: d(p,q) = {:.6}, d(q,p) = {:.6}",
            d_pq,
            d_qp
        );
    }

    // ── Weight parameter ─────────────────────────────────────────────────────

    /// Higher weight makes translation contribute more to distance.
    #[test]
    fn test_weight_affects_distance() {
        let m_low = SpecialEuclidean::<3> { weight: 0.1 };
        let m_high = SpecialEuclidean::<3> { weight: 10.0 };

        let p = identity::<3>();
        // Pure translation step.
        let v = SETangent {
            rotation: SMatrix::<Real, 3, 3>::zeros(),
            translation: SVector::<Real, 3>::new(1.0, 0.0, 0.0),
        };

        let norm_low = m_low.norm(&p, &v);
        let norm_high = m_high.norm(&p, &v);

        // With weight = 0.1: norm = sqrt(0 + 0.1 * 1) = sqrt(0.1)
        // With weight = 10:  norm = sqrt(0 + 10 * 1) = sqrt(10)
        assert!(
            norm_high > norm_low,
            "higher weight should give larger norm: low = {:.4}, high = {:.4}",
            norm_low,
            norm_high
        );
        let expected_ratio = (10.0_f64 / 0.1).sqrt();
        let actual_ratio = norm_high / norm_low;
        let ratio_err = (actual_ratio - expected_ratio).abs() / expected_ratio;
        assert!(
            ratio_err < TIGHT,
            "norm ratio mismatch: expected {:.4}, got {:.4}",
            expected_ratio,
            actual_ratio
        );
    }
}
