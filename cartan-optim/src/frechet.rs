// ~/cartan/cartan-optim/src/frechet.rs

//! Fréchet (Karcher) mean on Riemannian manifolds.
//!
//! ## Problem
//!
//! Given N points x_1,...,x_N on a manifold M, the Fréchet mean minimizes:
//!
//!   mu* = argmin_{p ∈ M} (1/N) Σ_i d(p, x_i)²
//!
//! This is the Riemannian analogue of the Euclidean mean.
//!
//! ## Algorithm (Karcher flow / gradient descent on the variance)
//!
//! The gradient of f(p) = (1/2N) Σ_i d(p, x_i)² is:
//!   grad f(p) = -(1/N) Σ_i Log_p(x_i)
//!
//! So the iteration is:
//!   p_{k+1} = Exp_{p_k}(t · (1/N) Σ_i Log_{p_k}(x_i))
//!
//! With t = 1 (unit step), this is the exact gradient flow.
//! We iterate until the mean tangent velocity is small.
//!
//! ## Convergence
//!
//! For points within a geodesically convex ball of radius < injectivity_radius/2,
//! the Fréchet mean exists and is unique, and Karcher flow converges at a linear
//! rate from any starting point inside the ball.
//!
//! ## References
//!
//! - Karcher. "Riemannian Center of Mass and Mollifier Smoothing." Comm. Pure
//!   Appl. Math. 1977.
//! - Afsari. "Riemannian L^p center of mass: Existence, uniqueness, and
//!   convexity." Proc. AMS. 2011.
//! - Pennec. "Intrinsic Statistics on Riemannian Manifolds." J. Math. Imaging
//!   and Vision. 2006.

use cartan_core::{Manifold, Real};

use crate::result::OptResult;

/// Configuration for the Fréchet mean (Karcher flow).
#[derive(Debug, Clone)]
pub struct FrechetConfig {
    /// Maximum number of Karcher iterations.
    pub max_iters: usize,
    /// Stop when ||mean tangent velocity|| < tol.
    pub tol: Real,
    /// Step size for each Karcher step (1.0 = exact gradient flow).
    pub step_size: Real,
}

impl Default for FrechetConfig {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-8,
            step_size: 1.0,
        }
    }
}

/// Compute the Fréchet mean of a set of points.
///
/// # Arguments
///
/// - `manifold`: The manifold.
/// - `points`: Slice of points x_1, ..., x_N.
/// - `init`: Initial estimate of the mean. If `None`, uses `points[0]`.
/// - `config`: Iteration parameters.
///
/// # Returns
///
/// [`OptResult`] where `value` is the final variance (1/N Σ d(mean, x_i)²)
/// and `grad_norm` is ||mean velocity|| at convergence.
///
/// # Panics
///
/// Panics if `points` is empty.
pub fn frechet_mean<M>(
    manifold: &M,
    points: &[M::Point],
    init: Option<M::Point>,
    config: &FrechetConfig,
) -> OptResult<M::Point>
where
    M: Manifold,
{
    assert!(!points.is_empty(), "frechet_mean: points must be non-empty");

    let n = points.len();
    let mut mu = init.unwrap_or_else(|| points[0].clone());

    let variance = |p: &M::Point| -> Real {
        points
            .iter()
            .filter_map(|xi| manifold.dist(p, xi).ok())
            .map(|d| d * d)
            .sum::<Real>()
            / n as Real
    };

    for iter in 0..config.max_iters {
        // Compute mean tangent velocity: v = (1/N) Σ_i Log_{mu}(x_i).
        // Skip any x_i for which Log fails (at cut locus).
        let mut velocity = manifold.zero_tangent(&mu);
        let mut count = 0usize;
        for xi in points {
            if let Ok(log_i) = manifold.log(&mu, xi) {
                velocity = velocity + log_i;
                count += 1;
            }
        }
        if count == 0 {
            // All points at cut locus — return current estimate.
            let v = variance(&mu);
            return OptResult {
                point: mu,
                value: v,
                grad_norm: Real::NAN,
                iterations: iter,
                converged: false,
            };
        }
        velocity = velocity * (1.0 / count as Real);

        let vel_norm = manifold.norm(&mu, &velocity);

        if vel_norm < config.tol {
            let v = variance(&mu);
            return OptResult {
                point: mu,
                value: v,
                grad_norm: vel_norm,
                iterations: iter,
                converged: true,
            };
        }

        // Karcher step: mu ← Exp_{mu}(step_size · velocity).
        mu = manifold.exp(&mu, &(velocity * config.step_size));
    }

    let v = variance(&mu);
    let g_sq = points
        .iter()
        .filter_map(|xi| manifold.log(&mu, xi).ok())
        .fold(manifold.zero_tangent(&mu), |acc, l| acc + l);
    let g_sq_norm = manifold.norm(&mu, &(g_sq * (1.0 / n as Real)));

    OptResult {
        point: mu,
        value: v,
        grad_norm: g_sq_norm,
        iterations: config.max_iters,
        converged: false,
    }
}
