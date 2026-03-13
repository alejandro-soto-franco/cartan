// ~/cartan/cartan-core/src/manifold.rs

//! The fundamental `Manifold` trait.
//!
//! Every Riemannian manifold in cartan implements this trait, which provides:
//! - Exponential and logarithmic maps (geodesic-based point transport)
//! - Riemannian inner product (the metric)
//! - Projections (ambient -> manifold, ambient -> tangent space)
//! - Validation (check_point, check_tangent)
//! - Random sampling
//!
//! ## Design decisions
//!
//! **Extrinsic representation:** Tangent vectors are stored in ambient coordinates.
//! For S^{N-1} in R^N, both Point and Tangent are [Real; N], even though the
//! tangent space is (N-1)-dimensional. The constraint (e.g., p^T v = 0) is
//! enforced by project_tangent and check_tangent, not by the type system.
//! This matches the convention used by Pymanopt, Geomstats, and Manopt.
//!
//! **exp is total, log returns Result:** On a complete Riemannian manifold
//! (all v0.1 manifolds are complete by Hopf-Rinow), the exponential map
//! is defined on all of T_pM. The logarithmic map fails at the cut locus.
//!
//! **Retraction:** Cheap retractions live on the `Retraction` trait, not the base
//! `Manifold` trait. Callers that don't need cheap retraction use `exp()` directly.
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2008. Chapter 3 (Manifold structure).
//! - do Carmo. "Riemannian Geometry." Birkhauser, 1992. Chapter 3 (exp/log maps).

use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

use rand::Rng;

use crate::{CartanError, Real};

/// A smooth Riemannian manifold.
///
/// This is the foundational trait in cartan. It defines the geometric
/// structure needed for optimization, geodesic computation, and statistics
/// on manifolds. Every concrete manifold (Sphere, SO(n), SPD(n), etc.)
/// implements this trait.
///
/// # Type parameters
///
/// The associated types `Point` and `Tangent` use extrinsic (ambient)
/// coordinates. For example, on S^2 in R^3, both are `SVector<Real, 3>`.
///
/// `Tangent` requires arithmetic operations (Add, `Mul<Real>`, Neg) because
/// generic algorithms like Frechet mean and conjugate gradient need to
/// add and scale tangent vectors.
pub trait Manifold {
    /// A point on the manifold, stored in ambient (extrinsic) coordinates.
    type Point: Clone + Debug;

    /// A tangent vector at a point, stored in ambient (extrinsic) coordinates.
    ///
    /// The tangent space constraint is semantic (enforced by project_tangent
    /// and check_tangent), not by the type system. This means any ambient
    /// vector can be stored as a Tangent and then projected.
    ///
    /// The Neg bound is an addition beyond the original spec (which had only
    /// Add + `Mul<Real>`). It is needed for conjugate gradient (d = -grad) and
    /// other algorithms that negate tangent vectors. Without it, callers would
    /// need `v * (-1.0)` which is less ergonomic.
    type Tangent: Clone
        + Debug
        + Add<Output = Self::Tangent>
        + Sub<Output = Self::Tangent>
        + Mul<Real, Output = Self::Tangent>
        + Neg<Output = Self::Tangent>;

    /// Intrinsic dimension of the manifold.
    ///
    /// For S^{N-1}, this is N-1. For SO(N), this is N(N-1)/2.
    /// For Grassmann(N,K), this is K(N-K).
    fn dim(&self) -> usize;

    /// Ambient space dimension.
    ///
    /// For S^{N-1} embedded in R^N, this is N.
    /// For SO(N) embedded in R^{N x N}, this is N*N.
    fn ambient_dim(&self) -> usize;

    /// Injectivity radius at a point.
    ///
    /// The largest radius r such that exp_p is a diffeomorphism on
    /// the ball B(0, r) in T_pM. Log/exp roundtrips are valid for
    /// tangent vectors with norm < injectivity_radius.
    ///
    /// Returns f64::INFINITY for Cartan-Hadamard manifolds (SPD, Hyperbolic)
    /// where exp is a global diffeomorphism.
    ///
    /// - Sphere S^{N-1}: pi
    /// - SO(N): pi
    /// - Grassmann(N,K): pi/2
    /// - SPD(N): infinity
    /// - Hyperbolic H^N: infinity
    fn injectivity_radius(&self, p: &Self::Point) -> Real;

    /// Riemannian inner product <u, v>_p.
    ///
    /// A smoothly varying family of inner products, one per point p.
    /// The inner product at p may depend on p (e.g., SPD affine-invariant
    /// metric: <U, V>_P = tr(P^{-1} U P^{-1} V)).
    fn inner(&self, p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real;

    /// Induced norm: ||v||_p = sqrt(<v, v>_p).
    fn norm(&self, p: &Self::Point, v: &Self::Tangent) -> Real {
        self.inner(p, v, v).sqrt()
    }

    /// Geodesic distance: d(p, q) = ||Log_p(q)||_p.
    ///
    /// Fails if log fails (cut locus).
    fn dist(&self, p: &Self::Point, q: &Self::Point) -> Result<Real, CartanError> {
        let v = self.log(p, q)?;
        Ok(self.norm(p, &v))
    }

    /// Exponential map: Exp_p(v).
    ///
    /// Follows the unique geodesic gamma with gamma(0) = p, gamma'(0) = v
    /// for unit time. Total on complete manifolds by the Hopf-Rinow theorem.
    /// All v0.1 manifolds are complete, so this always succeeds.
    ///
    /// Ref: do Carmo, "Riemannian Geometry", Chapter 3, Proposition 2.4.
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point;

    /// Logarithmic map: Log_p(q).
    ///
    /// Returns the initial velocity of the unique minimizing geodesic from p to q.
    /// Inverse of exp: log(p, exp(p, v)) = v for ||v|| < injectivity_radius.
    ///
    /// Fails at the cut locus where the minimizing geodesic is non-unique:
    /// - Sphere: antipodal points (p^T q ~ -1)
    /// - SO(N): rotations with angle near pi
    /// - Grassmann: principal angles near pi/2
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError>;

    /// Orthogonal projection from ambient space onto tangent space T_p M.
    ///
    /// Since Tangent uses ambient coordinates, any ambient vector can be
    /// passed as a Tangent value and projected. This also serves as the
    /// Euclidean-to-Riemannian gradient conversion:
    ///   riem_grad = project_tangent(p, eucl_grad)
    ///
    /// Idempotent: project_tangent(p, project_tangent(p, v)) == project_tangent(p, v).
    fn project_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Tangent;

    /// Project an ambient point onto the manifold (nearest point).
    ///
    /// Idempotent: project_point(project_point(x)) == project_point(x).
    fn project_point(&self, p: &Self::Point) -> Self::Point;

    /// The zero tangent vector at p.
    ///
    /// Satisfies: norm(p, zero_tangent(p)) == 0 and
    /// check_tangent(p, zero_tangent(p)) is Ok.
    fn zero_tangent(&self, p: &Self::Point) -> Self::Tangent;

    /// Validate that a point lies on the manifold.
    ///
    /// Checks the defining constraint (e.g., ||p|| = 1 for sphere,
    /// R^T R = I for SO(N)) to within numerical tolerance.
    /// Returns Ok(()) if valid, Err(NotOnManifold) with the violation magnitude.
    fn check_point(&self, p: &Self::Point) -> Result<(), CartanError>;

    /// Validate that a tangent vector lies in T_p M.
    ///
    /// Checks the tangent space constraint (e.g., p^T v = 0 for sphere)
    /// to within numerical tolerance. Assumes p is a valid manifold point.
    fn check_tangent(&self, p: &Self::Point, v: &Self::Tangent) -> Result<(), CartanError>;

    /// Random point on the manifold.
    ///
    /// Uniform distribution where meaningful (e.g., Haar measure on SO(N),
    /// uniform on sphere). For non-compact manifolds (SPD, Euclidean),
    /// samples from a reasonable default distribution.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point;

    /// Random tangent vector at p.
    ///
    /// Standard Gaussian in tangent space coordinates, then projected.
    fn random_tangent<R: Rng>(&self, p: &Self::Point, rng: &mut R) -> Self::Tangent;
}
