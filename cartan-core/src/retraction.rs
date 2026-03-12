// ~/cartan/cartan-core/src/retraction.rs

//! The `Retraction` trait: cheap approximations to the exponential map.
//!
//! A retraction is a smooth map R: TM -> M that approximates the exponential
//! map to first order. Specifically, for any manifold point p and tangent
//! vector v, a retraction R_p: T_pM -> M must satisfy:
//!
//!   1. R_p(0) = p               (centered at p)
//!   2. DR_p(0) = id_{T_pM}      (first-order consistency with exp)
//!
//! Retractions are cheaper than exact exp in many cases:
//! - Stiefel manifold: QR retraction is O(nk^2) vs exp at O((n+k)^3)
//! - Sphere: normalization retraction (p + v) / ||p + v|| vs sin/cos
//! - Grassmann: QR retraction
//!
//! The Manifold trait's default retract() calls exp(). Manifolds that have
//! a cheaper retraction implement this trait and override it. Generic
//! algorithms that only need first-order convergence (SGD, momentum methods)
//! should prefer retract() over exp() for speed.
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2008. Definition 4.1.1 (Retraction).
//! - Absil and Malick. "Projection-like Retractions on Matrix Manifolds."
//!   SIAM Journal on Optimization, 2012.

use crate::{CartanError, Manifold};

/// A manifold equipped with a cheap retraction.
///
/// Implement this trait to provide a faster alternative to exp() for
/// first-order optimization methods. The retract() method here overrides
/// the default retract() from Manifold.
///
/// # Supertraiting Manifold
///
/// Retraction requires Self: Manifold so that implementors must provide
/// the full manifold structure. This ensures that a type claiming to have
/// a retraction also has exp/log/inner/etc.
pub trait Retraction: Manifold {
    /// Apply the retraction at p in direction v.
    ///
    /// Returns a point on the manifold. Must satisfy:
    /// - retract(p, 0) == p
    /// - d/dt retract(p, t*v)|_{t=0} == v  (first-order agreement with exp)
    ///
    /// The result may differ from exp(p, v) but must lie on the manifold.
    /// For optimization convergence proofs, only the above two properties
    /// are required.
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point;

    /// Inverse of the retraction (approximate log).
    ///
    /// Returns a tangent vector v such that retract(p, v) approximates q.
    /// Not all retractions have a tractable inverse; this returns an error
    /// if the inverse is undefined or numerically infeasible.
    ///
    /// For the QR retraction, the inverse is given by the polar decomposition.
    /// For the normalization retraction on the sphere, the inverse is
    /// proportional to q - (p^T q) p followed by appropriate scaling.
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError>;
}
