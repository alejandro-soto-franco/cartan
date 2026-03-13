// ~/cartan/cartan-manifolds/src/euclidean.rs

//! Euclidean space R^N.
//!
//! The trivial Riemannian manifold: flat, with the standard dot product
//! as the metric. All geometric operations reduce to linear algebra.
//!
//! This manifold serves two purposes:
//! 1. Baseline for testing: every identity should hold exactly (no curvature effects).
//! 2. Component for product manifolds: ProductManifold<Euclidean<3>, Sphere<4>>.
//!
//! ## Geometry
//!
//! - Exp map: p + v (geodesics are straight lines)
//! - Log map: q - p (always defined, no cut locus)
//! - Inner product: standard dot product u^T v (independent of base point)
//! - Parallel transport: identity (tangent spaces are all the same R^N)
//! - Curvature: identically zero (flat)
//! - Injectivity radius: infinity

use nalgebra::SVector;
use rand::Rng;
use rand_distr::StandardNormal;

use cartan_core::{
    CartanError, Connection, Curvature, GeodesicInterpolation,
    Manifold, ParallelTransport, Real, Retraction,
};
// Note: VectorTransport is NOT imported here because Euclidean<N> gets its
// VectorTransport implementation automatically via the blanket impl in
// cartan-core: `impl<M: ParallelTransport> VectorTransport for M`.

/// Euclidean space R^N with the standard dot product metric.
///
/// Zero-sized type: carries no runtime data since the geometry is
/// completely determined by the dimension N.
#[derive(Debug, Clone, Copy)]
pub struct Euclidean<const N: usize>;

impl<const N: usize> Manifold for Euclidean<N> {
    type Point = SVector<Real, N>;
    type Tangent = SVector<Real, N>;

    fn dim(&self) -> usize {
        // Euclidean R^N has intrinsic dimension equal to its ambient dimension.
        N
    }

    fn ambient_dim(&self) -> usize {
        // R^N is embedded in R^N: ambient == intrinsic.
        N
    }

    fn injectivity_radius(&self, _p: &Self::Point) -> Real {
        // No cut locus: the exponential map (translation) is a global diffeomorphism.
        // Every geodesic can be extended to any length without losing injectivity.
        Real::INFINITY
    }

    /// Standard dot product, independent of the base point.
    ///
    /// On a flat manifold all tangent spaces are canonically isomorphic to R^N,
    /// so the metric tensor g_p(u, v) = u^T v is the same at every point p.
    fn inner(&self, _p: &Self::Point, u: &Self::Tangent, v: &Self::Tangent) -> Real {
        u.dot(v)
    }

    /// Geodesics are straight lines: Exp_p(v) = p + v.
    ///
    /// On R^N the Christoffel symbols vanish, so the geodesic equation
    /// gamma'' = 0 gives gamma(t) = p + t*v, hence Exp_p(v) = gamma(1) = p + v.
    fn exp(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        p + v
    }

    /// Inverse of exp: Log_p(q) = q - p. Always defined (no cut locus).
    ///
    /// The unique geodesic from p to q has initial velocity q - p,
    /// so Log_p(q) = q - p exactly. This is defined for all p, q in R^N.
    fn log(&self, p: &Self::Point, q: &Self::Point) -> Result<Self::Tangent, CartanError> {
        Ok(q - p)
    }

    /// Identity projection: every vector is tangent to R^N.
    ///
    /// The tangent space T_pR^N = R^N at every point, so every ambient
    /// vector is already a valid tangent vector. No projection needed.
    fn project_tangent(&self, _p: &Self::Point, v: &Self::Tangent) -> Self::Tangent {
        *v
    }

    /// Identity projection: every point is in R^N.
    ///
    /// R^N has no constraint defining which points are "on the manifold,"
    /// so every point already satisfies the (vacuous) constraint.
    fn project_point(&self, p: &Self::Point) -> Self::Point {
        *p
    }

    fn zero_tangent(&self, _p: &Self::Point) -> Self::Tangent {
        // The additive identity in R^N is the zero vector.
        SVector::zeros()
    }

    /// Always valid: any vector in R^N is a point in R^N.
    ///
    /// There is no non-trivial constraint to violate, so every point passes.
    fn check_point(&self, _p: &Self::Point) -> Result<(), CartanError> {
        Ok(())
    }

    /// Always valid: any vector is tangent to R^N.
    ///
    /// There is no tangent space constraint (T_pR^N = R^N), so every
    /// ambient vector is already a valid tangent vector at every point.
    fn check_tangent(&self, _p: &Self::Point, _v: &Self::Tangent) -> Result<(), CartanError> {
        Ok(())
    }

    /// Sample a random point from N(0, I_N): standard Gaussian in R^N.
    ///
    /// For non-compact manifolds like R^N there is no uniform (Lebesgue-flat)
    /// probability measure, so we use the standard Gaussian as a reasonable default.
    fn random_point<R: Rng>(&self, rng: &mut R) -> Self::Point {
        SVector::from_fn(|_, _| rng.sample(StandardNormal))
    }

    /// Sample a random tangent vector from N(0, I_N).
    ///
    /// Since T_pR^N = R^N, a random tangent vector is just a random ambient vector.
    fn random_tangent<R: Rng>(&self, _p: &Self::Point, rng: &mut R) -> Self::Tangent {
        SVector::from_fn(|_, _| rng.sample(StandardNormal))
    }
}

impl<const N: usize> Retraction for Euclidean<N> {
    /// Same as exp for Euclidean space: retract(p, v) = p + v.
    ///
    /// For flat R^N the retraction and the exponential map coincide exactly.
    /// Both are just vector addition.
    fn retract(&self, p: &Self::Point, v: &Self::Tangent) -> Self::Point {
        p + v
    }

    /// Inverse retraction: q - p. Same as log for flat R^N.
    fn inverse_retract(
        &self,
        p: &Self::Point,
        q: &Self::Point,
    ) -> Result<Self::Tangent, CartanError> {
        // On R^N, inverse_retract = log = q - p.
        Ok(q - p)
    }
}

impl<const N: usize> ParallelTransport for Euclidean<N> {
    /// Identity: all tangent spaces are the same R^N.
    ///
    /// Parallel transport on R^N is trivial: since all tangent spaces are
    /// canonically identified with R^N via the global chart, transporting a
    /// vector from p to q leaves it unchanged. This corresponds to
    /// Gamma^k_{ij} = 0 (all Christoffel symbols vanish).
    fn transport(
        &self,
        _p: &Self::Point,
        _q: &Self::Point,
        v: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // Return the vector unchanged: parallel transport on R^N is the identity.
        Ok(*v)
    }
}

// Note: VectorTransport is automatically implemented for Euclidean<N> via the
// blanket impl `impl<M: ParallelTransport> VectorTransport for M` in cartan-core.
// The blanket impl computes q = exp(p, direction) and delegates to transport(p, q, v).

impl<const N: usize> Connection for Euclidean<N> {
    /// Flat connection: Riemannian Hessian-vector product = projected ambient HVP.
    ///
    /// On R^N all Christoffel symbols vanish, so the Riemannian Hessian equals
    /// the Euclidean Hessian (the shape operator correction is zero).
    /// Since every ambient vector is tangent, the projection is also the identity,
    /// giving: Hess f(p)\[v\] = D^2 f(p)\[v\].
    ///
    /// The `hess_ambient` callback computes the Euclidean HVP D^2 f(p)\[v\],
    /// which we return directly (no correction needed for flat space).
    fn riemannian_hessian_vector_product(
        &self,
        _p: &Self::Point,
        _grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // On flat R^N: Riemannian HVP = Euclidean HVP (no curvature correction).
        Ok(hess_ambient(v))
    }
}

impl<const N: usize> Curvature for Euclidean<N> {
    /// Zero curvature: R(u, v)w = 0 for all u, v, w.
    ///
    /// Euclidean space is flat: the Riemann curvature tensor vanishes identically.
    /// This follows directly from vanishing Christoffel symbols.
    fn riemann_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
        _w: &Self::Tangent,
    ) -> Self::Tangent {
        // R(u,v)w = 0 on flat space.
        SVector::zeros()
    }

    /// Ricci curvature is identically zero on R^N.
    ///
    /// Ric(u, v) = trace of Z -> R(Z, u)v = trace of 0 = 0.
    fn ricci_curvature(
        &self,
        _p: &Self::Point,
        _u: &Self::Tangent,
        _v: &Self::Tangent,
    ) -> Real {
        0.0
    }

    /// Scalar curvature is identically zero on R^N.
    ///
    /// s = trace(Ric) = 0 since Ric = 0.
    fn scalar_curvature(&self, _p: &Self::Point) -> Real {
        0.0
    }
}

impl<const N: usize> GeodesicInterpolation for Euclidean<N> {
    /// Linear interpolation: gamma(t) = (1 - t) * p + t * q.
    ///
    /// On R^N the unique unit-speed geodesic from p to q is the straight line
    /// gamma(t) = p + t * (q - p) = (1-t)*p + t*q. This is well-defined for
    /// all t in R and all p, q in R^N (no cut locus).
    fn geodesic(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        t: Real,
    ) -> Result<Self::Point, CartanError> {
        // Linear combination: (1-t)*p + t*q.
        Ok(p * (1.0 - t) + q * t)
    }
}
