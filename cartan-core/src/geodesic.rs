// ~/cartan/cartan-core/src/geodesic.rs

//! The `GeodesicInterpolation` trait: sampling points along geodesics.
//!
//! Geodesic interpolation provides a way to move continuously between
//! two points on a manifold along the shortest path (minimizing geodesic).
//! This is the manifold analogue of linear interpolation in Euclidean space.
//!
//! ## Definition
//!
//! Given points p, q in M and parameter t in [0, 1], the geodesic
//! interpolation gamma(p, q, t) is defined as:
//!
//!   gamma(p, q, t) = Exp_p(t * Log_p(q))
//!
//! Properties:
//!   gamma(p, q, 0) = p
//!   gamma(p, q, 1) = q
//!   d/dt gamma(p, q, t)|_{t=0} = Log_p(q)  (initial velocity)
//!
//! ## Applications
//!
//! - Frechet mean computation (iterative: barycenter in T_pM)
//! - Geodesic grid generation for visualization
//! - Midpoint computation (t = 0.5) for Riemannian bisection
//! - Slerp generalization (spherical: reduces to standard slerp)
//!
//! ## Failure modes
//!
//! Like Log_p(q), geodesic interpolation fails at the cut locus. The
//! method returns an error rather than panicking, allowing callers to
//! detect and handle this case (e.g., by using a retraction-based fallback).
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 3 (geodesics).
//! - Pennec, Fillard, Ayache. "A Riemannian Framework for Tensor Computing."
//!   IJCV, 2006. Section 3.2 (geodesic interpolation for SPD matrices).

use crate::{CartanError, Manifold};

/// A manifold with geodesic interpolation between points.
///
/// The geodesic interpolation gamma(p, q, t) = Exp_p(t * Log_p(q))
/// parameterizes the unique minimizing geodesic from p to q (when it exists).
///
/// # Supertraiting Manifold
///
/// GeodesicInterpolation requires Self: Manifold because it is defined
/// via exp and log. A default implementation could be provided here using
/// self.exp(p, self.log(p, q)? * t), but we leave it as an abstract method
/// to allow manifolds to provide more numerically stable implementations
/// (e.g., direct formula on the sphere using sin/cos).
pub trait GeodesicInterpolation: Manifold {
    /// Interpolate along the geodesic from p to q at parameter t.
    ///
    /// For t = 0: returns p.
    /// For t = 1: returns q.
    /// For t = 0.5: returns the geodesic midpoint.
    /// For t outside [0,1]: extrapolates along the geodesic (valid if
    ///   t * dist(p, q) < injectivity_radius(p)).
    ///
    /// Fails if Log_p(q) fails (p and q are at or beyond the cut locus).
    ///
    /// # Arguments
    ///
    /// - `p`: starting point
    /// - `q`: ending point
    /// - `t`: interpolation parameter (0 = start, 1 = end, 0.5 = midpoint)
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: crate::Real,
    ) -> Result<Self::Point, CartanError>;
}
