// ~/cartan/cartan-geo/src/geodesic.rs

//! Parameterized geodesics: gamma(t) = Exp_p(t * v).
//!
//! A `Geodesic` wraps a base point and initial velocity vector and provides:
//! - Point evaluation at any parameter t
//! - Uniform sampling of n points on [0, 1]
//! - Construction from two endpoints via Log
//!
//! The geodesic is parameterized proportionally to arc length when
//! `velocity` is the unit-speed vector: ||v|| = 1 means t = arc length.
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 3 (geodesics and exp map).
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Section 2.1 (geodesics on matrix manifolds).

use cartan_core::{CartanError, Manifold, Real};

/// A parameterized geodesic on a Riemannian manifold.
///
/// Represents the curve gamma: R -> M defined by
///   gamma(t) = Exp_{base}(t * velocity)
///
/// At t = 0: gamma(0) = base.
/// At t = 1: gamma(1) = Exp_{base}(velocity) (the "endpoint").
/// The arc-length traversed from t=0 to t=1 equals ||velocity||.
///
/// The manifold is stored by reference to avoid requiring Clone on M.
pub struct Geodesic<'a, M: Manifold> {
    pub manifold: &'a M,
    /// Base point: gamma(0).
    pub base: M::Point,
    /// Initial velocity: gamma'(0). Arc length = ||velocity||.
    pub velocity: M::Tangent,
}

impl<'a, M: Manifold> Geodesic<'a, M> {
    /// Construct a geodesic from a base point and initial velocity.
    ///
    /// The velocity is NOT normalized: the geodesic travels at constant speed
    /// ||velocity|| from t=0. Set velocity = Log_p(q) to get the unit-interval
    /// geodesic from p to q.
    pub fn new(manifold: &'a M, base: M::Point, velocity: M::Tangent) -> Self {
        Self {
            manifold,
            base,
            velocity,
        }
    }

    /// Construct a geodesic from base point `p` to `q`.
    ///
    /// The velocity is Log_p(q), so gamma(0) = p and gamma(1) = q.
    /// Fails if p and q are at the cut locus.
    pub fn from_two_points(
        manifold: &'a M,
        p: M::Point,
        q: &M::Point,
    ) -> Result<Self, CartanError> {
        let v = manifold.log(&p, q)?;
        Ok(Self::new(manifold, p, v))
    }

    /// Evaluate the geodesic at parameter t: Exp_{base}(t * velocity).
    ///
    /// t = 0 returns base, t = 1 returns the endpoint.
    /// The exponential map on complete manifolds is total, so this always succeeds.
    pub fn eval(&self, t: Real) -> M::Point {
        self.manifold.exp(&self.base, &(self.velocity.clone() * t))
    }

    /// Arc length of the geodesic on [0, 1]: ||velocity||_{base}.
    pub fn length(&self) -> Real {
        self.manifold.norm(&self.base, &self.velocity)
    }

    /// Sample `n` evenly-spaced points along the geodesic on [0, 1].
    ///
    /// For n = 1: returns [gamma(0)].
    /// For n = 2: returns [gamma(0), gamma(1)].
    /// For n >= 2: returns points at t = 0, 1/(n-1), 2/(n-1), ..., 1.
    ///
    /// Panics if n == 0.
    #[cfg(feature = "alloc")]
    pub fn sample(&self, n: usize) -> alloc::vec::Vec<M::Point> {
        assert!(n > 0, "Geodesic::sample: n must be at least 1");
        if n == 1 {
            return alloc::vec![self.eval(0.0)];
        }
        let step = 1.0 / (n - 1) as Real;
        (0..n).map(|i| self.eval(i as Real * step)).collect()
    }

    /// Sample up to N evenly-spaced points (no_alloc version).
    ///
    /// Returns an array of `Option<M::Point>` of length N, filled for indices
    /// 0..count, plus the actual count of filled entries.
    /// Use this instead of `sample` when the `alloc` feature is unavailable.
    ///
    /// Panics if n == 0.
    #[cfg(not(feature = "alloc"))]
    pub fn sample_fixed<const N: usize>(&self, n: usize) -> ([Option<M::Point>; N], usize)
    where
        M::Point: Copy,
    {
        assert!(n > 0, "Geodesic::sample_fixed: n must be at least 1");
        let count = n.min(N);
        let mut out = [None; N];
        if count == 1 {
            out[0] = Some(self.eval(0.0));
            return (out, 1);
        }
        let step = 1.0 / (count - 1) as Real;
        for i in 0..count {
            out[i] = Some(self.eval(i as Real * step));
        }
        (out, count)
    }

    /// Midpoint of the geodesic: gamma(0.5) = Exp_{base}(0.5 * velocity).
    pub fn midpoint(&self) -> M::Point {
        self.eval(0.5)
    }
}
