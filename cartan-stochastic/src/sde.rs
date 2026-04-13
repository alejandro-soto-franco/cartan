//! Stratonovich stochastic differential equations on Riemannian manifolds.
//!
//! A single Stratonovich step on the orthonormal frame bundle advances `(p, r)`
//! by `dt` under the horizontal vector fields:
//!
//! ```text
//! d(p, r) = H_i(p, r) ∘ dW^i      (Stratonovich)
//! ```
//!
//! where `H_i` is the horizontal lift of the `i`-th frame vector. In the
//! minimal retraction-based discretisation used here, a step is
//!
//! 1. Form horizontal velocity `u = Σ dW_i · e_i ∈ T_p M`.
//! 2. Advance the base point: `p' = retract_p(u · √dt)`.
//! 3. Parallel-transport the frame from `p` to `p'` along the step, then
//!    Gram-Schmidt re-orthonormalise to absorb discretisation drift.
//!
//! This is first-order accurate in `dt` for Stratonovich SDEs (the
//! Stratonovich correction is absent because the drift is zero in frame
//! coordinates — the discretisation bias appears only as `O(dt)` in the
//! base-point projection).

use cartan_core::{Manifold, Real, Retraction, VectorTransport};

use crate::error::StochasticError;
use crate::frame::OrthonormalFrame;
use crate::horizontal::horizontal_velocity;

/// Marker trait for manifolds on which Stratonovich stochastic development
/// is available via this crate's default retraction-based integrator.
///
/// Automatically implemented for any manifold that already supplies
/// retraction and vector transport. (Vector transport is auto-implemented
/// for manifolds with exact parallel transport via the blanket impl in
/// `cartan-core`, so in practice this trait is available wherever
/// `Manifold + Retraction + ParallelTransport` is.)
///
/// Downstream crates can implement this trait explicitly to override the
/// default with a manifold-specific higher-order scheme (for example, the
/// exact Lie group development on SO(n) via the matrix exponential).
pub trait StratonovichDevelopment: Manifold + Retraction + VectorTransport {}
impl<M: Manifold + Retraction + VectorTransport> StratonovichDevelopment for M {}

/// One Stratonovich-Euler step on `O(M)`.
///
/// Returns the new base point and the transported-and-reorthonormalised
/// frame. Fails if `dw.len() != frame.len()` or if parallel transport lands
/// on a frame that Gram-Schmidt cannot orthonormalise (rank-deficient,
/// indicating a numerical collapse, typically at a cut locus).
///
/// `tol` is the Gram-Schmidt rank-deficiency threshold (1e-10 is a good
/// default for f64 on well-conditioned manifolds; loosen to 1e-6 near
/// near-singular points).
pub fn stratonovich_step<M: StratonovichDevelopment>(
    manifold: &M,
    p: &M::Point,
    frame: &OrthonormalFrame<M>,
    dw: &[Real],
    dt: Real,
    tol: Real,
) -> Result<(M::Point, OrthonormalFrame<M>), StochasticError> {
    // Horizontal velocity u = Σ dW_i e_i, then scale by √dt for the Brownian
    // increment convention. Callers who already baked √dt into dw can pass
    // dt = 1.0.
    let u = horizontal_velocity(frame, dw)?;
    let step = u * dt.sqrt();

    // Advance base point via retraction. Exact exp would be first-order
    // identical; retraction is chosen for cost and for manifolds (e.g.
    // Grassmann, Stiefel) where exp is expensive.
    let p_next = manifold.retract(p, &step);

    // Transport the frame along the step. The vector-transport trait reports
    // Result<Tangent, CartanError> per basis vector; collect them, bailing on
    // the first cut-locus failure, then Gram-Schmidt re-orthonormalise to
    // absorb discretisation drift and any residual isometry error.
    let mut transported: Vec<M::Tangent> = Vec::with_capacity(frame.basis.len());
    for e in &frame.basis {
        transported.push(manifold.vector_transport(p, &step, e)?);
    }
    let mut next_frame = OrthonormalFrame::from_orthonormal(transported);
    next_frame.reorthonormalize(manifold, &p_next, tol)?;

    Ok((p_next, next_frame))
}
