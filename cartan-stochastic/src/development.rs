//! Stochastic development (Eells-Elworthy-Malliavin).
//!
//! The rolling-without-slipping construction: drive `O(M)` along the
//! horizontal lift of Euclidean Brownian motion `W_t ∈ R^n`. The base-point
//! projection is a diffusion on `M` with generator `½ Δ_M` (Laplace-Beltrami).
//!
//! Anti-development is the inverse: given a curve on `M`, read off its
//! representation in any chosen initial frame via repeated parallel-transport
//! and inner products. That is not implemented here because it is an
//! application rather than a primitive — call `Manifold::log` to get
//! consecutive tangent vectors, then `Manifold::inner` against each frame
//! basis vector.

use cartan_core::Real;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::StochasticError;
use crate::frame::OrthonormalFrame;
use crate::sde::{stratonovich_step, StratonovichDevelopment};

/// A stored stochastic-development trajectory on `M`.
///
/// Contains the base-point path and the final frame. Intermediate frames
/// are not stored by default because most downstream consumers (BEL weight
/// computation, Bismut integration-by-parts) only need the base-point path
/// and a terminal frame. Callers needing the full frame history can call
/// `stratonovich_step` directly in a loop.
#[derive(Debug, Clone)]
pub struct DevelopmentPath<M: cartan_core::Manifold> {
    /// The base-point path, length `n_steps + 1`, starting at the caller's
    /// initial point.
    pub path: Vec<M::Point>,
    /// The orthonormal frame at the terminal point, parallel-transported and
    /// re-orthonormalised at each step.
    pub final_frame: OrthonormalFrame<M>,
}

/// Integrate a Brownian motion on `M` via stochastic development.
///
/// Draws `n_steps × dim(M)` i.i.d. standard normals and steps the Stratonovich
/// SDE on the orthonormal frame bundle. Returns the full base-point path.
///
/// `dt` is the per-step time increment; standard practice is `T / n_steps`
/// for total horizon `T`. `tol` is the Gram-Schmidt rank-deficiency threshold
/// used to re-orthonormalise the frame after each step.
pub fn stochastic_development<M: StratonovichDevelopment, R: Rng + ?Sized>(
    manifold: &M,
    p0: &M::Point,
    frame0: OrthonormalFrame<M>,
    n_steps: usize,
    dt: Real,
    rng: &mut R,
    tol: Real,
) -> Result<DevelopmentPath<M>, StochasticError>
where
    StandardNormal: Distribution<Real>,
{
    let n = manifold.dim();
    let mut path: Vec<M::Point> = Vec::with_capacity(n_steps + 1);
    path.push(p0.clone());
    let mut p = p0.clone();
    let mut frame = frame0;
    let mut dw = vec![0.0_f64; n];
    for _ in 0..n_steps {
        for w in dw.iter_mut() {
            *w = StandardNormal.sample(rng);
        }
        let (p_next, frame_next) = stratonovich_step(manifold, &p, &frame, &dw, dt, tol)?;
        path.push(p_next.clone());
        p = p_next;
        frame = frame_next;
    }
    Ok(DevelopmentPath { path, final_frame: frame })
}
