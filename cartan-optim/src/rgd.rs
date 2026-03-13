// ~/cartan/cartan-optim/src/rgd.rs

//! Riemannian Gradient Descent with Armijo line search.
//!
//! ## Algorithm
//!
//! Given iterate x_k, Riemannian gradient g_k = grad f(x_k):
//!   1. d_k = -g_k  (steepest descent)
//!   2. Find step t_k via Armijo backtracking:
//!      f(Retract(x_k, t·d_k)) ≤ f(x_k) + c · t · <g_k, d_k>_x_k
//!   3. x_{k+1} = Retract(x_k, t_k · d_k)
//!
//! ## Convergence
//!
//! For smooth f on a complete Riemannian manifold with Armijo line search,
//! the iterates satisfy lim inf ||grad f(x_k)|| = 0. For geodesically convex f,
//! the method converges to the global minimum.
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Chapter 4 (line search methods).
//! - Boumal. "An Introduction to Optimization on Smooth Manifolds."
//!   Chapter 4 (gradient descent).

use cartan_core::{Manifold, Real, Retraction};

use crate::result::OptResult;

/// Configuration for Riemannian Gradient Descent.
#[derive(Debug, Clone)]
pub struct RGDConfig {
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Stop when ||grad f(x)|| < grad_tol.
    pub grad_tol: Real,
    /// Initial step size for Armijo backtracking.
    pub init_step: Real,
    /// Armijo sufficient decrease constant (typically 1e-4 to 0.5).
    pub armijo_c: Real,
    /// Backtracking factor (< 1, typically 0.5).
    pub armijo_beta: Real,
    /// Maximum number of backtracking steps per iteration.
    pub max_ls_iters: usize,
}

impl Default for RGDConfig {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            grad_tol: 1e-6,
            init_step: 1.0,
            armijo_c: 1e-4,
            armijo_beta: 0.5,
            max_ls_iters: 50,
        }
    }
}

/// Run Riemannian Gradient Descent.
///
/// # Arguments
///
/// - `manifold`: The manifold (must implement `Retraction`).
/// - `cost`: Cost function f: M → R.
/// - `rgrad`: Riemannian gradient of f at x (already projected onto T_x M).
///   Typically: `|x| manifold.project_tangent(x, &euclidean_grad(x))`.
/// - `x0`: Initial point.
/// - `config`: Solver parameters.
///
/// # Returns
///
/// [`OptResult`] with the final point, value, gradient norm, and convergence flag.
pub fn minimize_rgd<M, F, G>(
    manifold: &M,
    cost: F,
    rgrad: G,
    x0: M::Point,
    config: &RGDConfig,
) -> OptResult<M::Point>
where
    M: Manifold + Retraction,
    F: Fn(&M::Point) -> Real,
    G: Fn(&M::Point) -> M::Tangent,
{
    let mut x = x0;
    let mut f_x = cost(&x);
    let mut g = rgrad(&x);
    let mut g_norm = manifold.norm(&x, &g);

    for iter in 0..config.max_iters {
        if g_norm < config.grad_tol {
            return OptResult {
                point: x,
                value: f_x,
                grad_norm: g_norm,
                iterations: iter,
                converged: true,
            };
        }

        // Steepest descent direction: d = -g.
        let d = -g.clone();

        // Armijo backtracking line search.
        // Sufficient decrease: f(Retract(x, t·d)) ≤ f(x) + c·t·<g, d>_x
        // Since d = -g: <g, d> = -||g||² < 0 (descent direction). ✓
        let slope = manifold.inner(&x, &g, &d); // = -||g||²
        let mut t = config.init_step;
        let f_threshold_base = f_x + config.armijo_c * slope; // at t=1

        for _ in 0..config.max_ls_iters {
            let x_trial = manifold.retract(&x, &(d.clone() * t));
            let f_trial = cost(&x_trial);
            if f_trial <= f_x + config.armijo_c * t * slope {
                x = x_trial;
                f_x = f_trial;
                break;
            }
            t *= config.armijo_beta;
        }

        // Suppress unused variable warning: f_threshold_base is for documentation.
        let _ = f_threshold_base;

        g = rgrad(&x);
        g_norm = manifold.norm(&x, &g);
    }

    OptResult {
        point: x,
        value: f_x,
        grad_norm: g_norm,
        iterations: config.max_iters,
        converged: false,
    }
}
