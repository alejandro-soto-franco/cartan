// ~/cartan/cartan-core/src/connection.rs

//! The `Connection` trait: Riemannian Hessian-vector products.
//!
//! The Levi-Civita connection on a Riemannian manifold provides a notion
//! of "derivative of a vector field along a curve." For optimization, the
//! key application is computing the Riemannian Hessian of a cost function,
//! or more precisely, Hessian-vector products (HVPs) needed by second-order
//! methods like Riemannian trust region and Riemannian Newton.
//!
//! ## Hessian-vector products
//!
//! Given a smooth cost function f: M -> R, the Riemannian Hessian at p
//! is a symmetric bilinear form on T_pM. For trust region methods, we need
//! H[v] = Hess f(p)[v], the action of the Hessian on a tangent vector v.
//!
//! For a Riemannian manifold embedded in Euclidean space, the Riemannian HVP
//! can be computed from the Euclidean Hessian as:
//!
//!   Hess f(p)[v] = proj_T( D^2 f(p)[v] - II(grad f, v) )
//!
//! where II is the second fundamental form (shape operator), and proj_T is
//! the projection onto the tangent space.
//!
//! Alternatively, for cost functions defined via retraction, one uses the
//! differentiated retraction formula from Absil-Mahony-Sepulchre Ch. 5.
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2008. Chapter 5 (Riemannian Hessian and trust region).
//! - Boumal. "An Introduction to Optimization on Smooth Manifolds."
//!   Cambridge, 2023. Chapter 6 (connections and Hessian).
//! - do Carmo. "Riemannian Geometry." Chapter 2 (Levi-Civita connection).

use crate::{CartanError, Manifold};

/// A manifold with a compatible Levi-Civita connection.
///
/// Implement this trait to enable second-order optimization methods
/// (trust region, Newton) that require Riemannian Hessian-vector products.
///
/// # Supertraiting Manifold
///
/// Connection requires Self: Manifold to ensure that the manifold has
/// the full geometric structure (exp, log, inner, project) before adding
/// the connection. The Levi-Civita connection depends on the metric, so
/// this dependency is semantically required, not just a convenience.
pub trait Connection: Manifold {
    /// Riemannian Hessian-vector product: Hess f(p)[v].
    ///
    /// Given:
    /// - p: a point on the manifold
    /// - grad_f: the Riemannian gradient of f at p (a tangent vector in T_pM)
    /// - v: a tangent vector in T_pM (the direction for the HVP)
    /// - hess_ambient: a callback that computes the Euclidean Hessian-vector
    ///   product H_eucl[v] = D^2 f(p)[v] in ambient coordinates
    ///
    /// Returns Hess f(p)[v] in T_pM.
    ///
    /// The `hess_ambient` callback takes a tangent vector in ambient coordinates
    /// and returns the ambient Euclidean HVP. This avoids the need to explicitly
    /// construct the full Hessian matrix (which is expensive for large N).
    ///
    /// Ref: Absil-Mahony-Sepulchre, Chapter 5, Proposition 5.3.2.
    fn riemannian_hessian_vector_product(
        &self,
        p: &Self::Point,
        grad_f: &Self::Tangent,
        v: &Self::Tangent,
        hess_ambient: &dyn Fn(&Self::Tangent) -> Self::Tangent,
    ) -> Result<Self::Tangent, CartanError>;
}
