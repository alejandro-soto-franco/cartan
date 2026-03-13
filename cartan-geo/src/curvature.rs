// ~/cartan/cartan-geo/src/curvature.rs

//! Curvature queries at a manifold point.
//!
//! `CurvatureQuery` wraps a manifold and a point, providing ergonomic
//! access to the curvature quantities defined in `cartan_core::Curvature`:
//! sectional, Ricci, and scalar curvature.
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 4 (Riemann curvature tensor).
//! - Petersen. "Riemannian Geometry." Chapter 3 (sectional curvature).

use cartan_core::{Curvature, Real};

/// Curvature queries at a fixed point on a manifold.
///
/// Holds a reference to the manifold and a clone of the query point.
/// All curvature computations are delegated to the `Curvature` trait.
pub struct CurvatureQuery<'a, M: Curvature> {
    manifold: &'a M,
    point: M::Point,
}

impl<'a, M: Curvature> CurvatureQuery<'a, M> {
    /// Create a new curvature query at point `p`.
    pub fn new(manifold: &'a M, point: M::Point) -> Self {
        Self { manifold, point }
    }

    /// Riemann curvature tensor: R(u, v)w at this point.
    ///
    /// Anti-symmetric in u, v: R(u,v)w = -R(v,u)w.
    pub fn riemann(
        &self,
        u: &M::Tangent,
        v: &M::Tangent,
        w: &M::Tangent,
    ) -> M::Tangent {
        self.manifold.riemann_curvature(&self.point, u, v, w)
    }

    /// Sectional curvature of the 2-plane spanned by u and v.
    ///
    /// K(u,v) = <R(u,v)v, u> / (||u||²||v||² - <u,v>²).
    /// Returns 0.0 if u and v are (nearly) parallel.
    pub fn sectional(&self, u: &M::Tangent, v: &M::Tangent) -> Real {
        self.manifold.sectional_curvature(&self.point, u, v)
    }

    /// Ricci curvature Ric(u, v) at this point.
    pub fn ricci(&self, u: &M::Tangent, v: &M::Tangent) -> Real {
        self.manifold.ricci_curvature(&self.point, u, v)
    }

    /// Scalar curvature at this point.
    pub fn scalar(&self) -> Real {
        self.manifold.scalar_curvature(&self.point)
    }
}

/// Compute the sectional curvature at `p` for the 2-plane spanned by `u` and `v`.
///
/// Convenience function that does not require constructing a `CurvatureQuery`.
pub fn sectional_at<M: Curvature>(
    manifold: &M,
    p: &M::Point,
    u: &M::Tangent,
    v: &M::Tangent,
) -> Real {
    manifold.sectional_curvature(p, u, v)
}

/// Compute the scalar curvature at `p`.
///
/// Convenience function that does not require constructing a `CurvatureQuery`.
pub fn scalar_at<M: Curvature>(manifold: &M, p: &M::Point) -> Real {
    manifold.scalar_curvature(p)
}
