//! Horizontal lift from `R^n` to the tangent bundle via an orthonormal frame.
//!
//! Given a frame `r = (e_1, …, e_n)` at `p ∈ M` and a vector `ξ ∈ R^n`, the
//! horizontal lift of `ξ` at `(p, r)` has ground component `Σ ξ^i e_i ∈ T_p M`.
//! This is the fundamental vector field `H_i` (when `ξ = ê_i`) of the
//! principal `O(n)`-connection on the orthonormal frame bundle.
//!
//! Anti-development (the reverse direction) is the inverse operation: given
//! a tangent vector `u ∈ T_p M` and a frame `r`, the components of `u` in the
//! basis `r` are `ξ^i = <u, e_i>_p`. That operation is `Manifold::inner`
//! applied against each basis vector and does not require a dedicated helper.

use cartan_core::{Manifold, Real};

use crate::error::StochasticError;
use crate::frame::OrthonormalFrame;

/// Horizontal velocity of the frame bundle curve driven by `dW`.
///
/// Computes `u = Σ dW_i · e_i ∈ T_p M` where `e_i` are the frame basis vectors.
/// The result is the tangent-space velocity of the base-point component of a
/// curve in `O(M)` whose driving noise is `dW` in frame coordinates.
///
/// `dW.len()` must equal `frame.len()` (= intrinsic dim of `M`).
pub fn horizontal_velocity<M: Manifold>(
    frame: &OrthonormalFrame<M>,
    dw: &[Real],
) -> Result<M::Tangent, StochasticError> {
    if dw.len() != frame.len() {
        return Err(StochasticError::NoiseDimMismatch {
            frame_dim: frame.len(),
            noise_dim: dw.len(),
        });
    }
    // The manifold does not expose a `zero_tangent`-like combinator without
    // a base point, so we build the sum by scaling the first basis vector
    // and accumulating. This works because Tangent: Add + Mul<Real>.
    let mut iter = frame.basis.iter().zip(dw.iter());
    let (e0, w0) = iter.next().expect("frame must be non-empty for horizontal lift");
    let mut acc = e0.clone() * *w0;
    for (e, w) in iter {
        acc = acc + e.clone() * *w;
    }
    Ok(acc)
}
