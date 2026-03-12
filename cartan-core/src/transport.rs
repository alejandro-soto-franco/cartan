// ~/cartan/cartan-core/src/transport.rs

//! Parallel transport and vector transport traits.
//!
//! Parallel transport moves a tangent vector along a curve while preserving
//! the Riemannian connection: a tangent vector at p is moved to a tangent
//! vector at q along the geodesic, without changing its length or angle
//! relative to the geodesic.
//!
//! ## ParallelTransport vs VectorTransport
//!
//! - `ParallelTransport`: geodesically exact. Solves the ODE
//!   D/dt V(t) = 0 along gamma. Expensive: typically O(n^3) per step
//!   for matrix manifolds.
//!
//! - `VectorTransport`: approximation satisfying only the isometry condition
//!   approximately. Cheaper alternatives: Schild's ladder, pole ladder,
//!   differentiated retraction.
//!
//! The blanket impl `impl<M: ParallelTransport> VectorTransport for M` means
//! that any manifold with exact parallel transport automatically satisfies
//! the VectorTransport interface. Algorithms that only need approximate
//! transport (most conjugate gradient methods) can use VectorTransport
//! without requiring the more expensive exact version.
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 4 (parallel transport).
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Chapter 8 (vector transport).
//! - Zhu. "A Riemannian Conjugate Gradient Method for Optimization on the
//!   Stiefel Manifold." Optimization, 2017.

use crate::{CartanError, Manifold};

/// Exact geodesic parallel transport.
///
/// Parallel transport of a tangent vector u at p along the geodesic to q
/// gives a tangent vector at q that has the same length (||P_p^q u||_q = ||u||_p)
/// and preserves inner products with all other parallelly transported vectors.
///
/// This is the most expensive transport operation. It is needed for:
/// - Conjugate gradient with exact beta computation
/// - Hessian approximations on curved manifolds
/// - Geodesic deviation (Jacobi fields)
///
/// # Supertraiting Manifold
///
/// ParallelTransport requires Self: Manifold to ensure that implementing
/// types have the full geometric structure (exp, log, inner).
pub trait ParallelTransport: Manifold {
    /// Parallel transport u from p to q along the geodesic.
    ///
    /// Returns the transported vector in T_q M, or an error if the geodesic
    /// between p and q passes through or near the cut locus.
    ///
    /// The transported vector satisfies:
    ///   ||transport(p, q, u)||_q == ||u||_p  (norm preserving)
    ///
    /// Ref: do Carmo, Chapter 4, Definition 2.3.
    fn transport(
        &self,
        p: &Self::Point,
        q: &Self::Point,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError>;
}

/// Approximate vector transport (cheaper than exact parallel transport).
///
/// A vector transport on a manifold M is a smooth map
///   T: TM x_M TM -> TM
/// satisfying:
///   1. T_{p,0}(u) = u  (trivial transport by zero vector)
///   2. T_{p,v}(u) in T_{R_p(v)} M  (lands in the right tangent space)
///   3. <T_{p,v}(u), T_{p,v}(w)>_{R_p(v)} = <u, w>_p  (isometry, approx OK)
///
/// Cheaper alternatives to exact parallel transport include:
/// - Differentiated retraction: T_{p,v}(u) = d/ds R_p(v + s*u)|_{s=0}
/// - Pole ladder: two successive Schild's ladder steps
/// - Schild's ladder: O(n^2) vs O(n^3) for matrix exponential transport
///
/// Most first-order Riemannian conjugate gradient convergence proofs only
/// require approximate isometry (condition 3 up to constants), so VectorTransport
/// is sufficient for CG implementations.
///
/// # Blanket impl
///
/// Any type implementing `ParallelTransport` automatically implements
/// `VectorTransport` via the blanket impl below. This means manifolds
/// with exact PT don't need to implement VT separately.
pub trait VectorTransport: Manifold {
    /// Apply the vector transport: move u at p to T_{p,v}(u) at retract(p,v).
    ///
    /// The `direction` v is the displacement (like the step in an optimizer).
    /// The result lives in T_{retract(p,v)} M.
    fn vector_transport(
        &self,
        p: &Self::Point,
        direction: &Self::Tangent,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError>;
}

/// Blanket implementation: exact parallel transport implies vector transport.
///
/// Any manifold implementing `ParallelTransport` automatically satisfies
/// `VectorTransport` by computing q = exp(p, direction) and calling transport.
/// This avoids code duplication for manifolds with exact PT.
///
/// The blanket impl uses exp() (from the Manifold supertrait) to compute q,
/// then delegates to the exact transport. Manifolds that need to use their
/// retraction instead of exp for the destination point can override VectorTransport
/// directly rather than relying on this blanket impl.
impl<M: ParallelTransport> VectorTransport for M {
    fn vector_transport(
        &self,
        p: &Self::Point,
        direction: &Self::Tangent,
        u: &Self::Tangent,
    ) -> Result<Self::Tangent, CartanError> {
        // Compute destination point q = Exp_p(direction).
        // For all complete manifolds, exp is total (never fails).
        let q = self.exp(p, direction);
        // Delegate to exact parallel transport along the geodesic from p to q.
        self.transport(p, &q, u)
    }
}
