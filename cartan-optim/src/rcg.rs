// ~/cartan/cartan-optim/src/rcg.rs

//! Riemannian Conjugate Gradient (Fletcher-Reeves and Polak-Ribière).
//!
//! ## Algorithm
//!
//! At iterate x_k with gradient g_k and conjugate direction p_k:
//!   1. Armijo line search: find t_k
//!   2. x_{k+1} = Retract(x_k, t_k · p_k)
//!   3. g_{k+1} = rgrad(x_{k+1})
//!   4. Transport p_k and g_k from x_k to x_{k+1} via parallel transport.
//!   5. Compute β:
//!      FR:  β = ||g_{k+1}||² / ||g_k||²
//!      PR+: β = max(0, <g_{k+1}, g_{k+1} − PT(g_k)>_{x_{k+1}} / ||g_k||²)
//!   6. p_{k+1} = −g_{k+1} + β · PT(p_k)
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Chapter 8 (Riemannian CG).
//! - Sato. "Riemannian Conjugate Gradient Methods." SIAM J. Optim. 2022.

use cartan_core::{Manifold, ParallelTransport, Real, Retraction};

use crate::result::OptResult;

/// Which β formula to use for the conjugate direction update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CgVariant {
    /// Fletcher-Reeves: β = ||g_{k+1}||² / ||g_k||²
    FletcherReeves,
    /// Polak-Ribière+ (clamped): β = max(0, <g_{k+1}, g_{k+1}−PT(g_k)> / ||g_k||²)
    #[default]
    PolakRibiere,
}

/// Configuration for Riemannian Conjugate Gradient.
#[derive(Debug, Clone)]
pub struct RCGConfig {
    /// Maximum number of iterations.
    pub max_iters: usize,
    /// Stop when ||grad f(x)|| < grad_tol.
    pub grad_tol: Real,
    /// Initial step size for Armijo backtracking.
    pub init_step: Real,
    /// Armijo sufficient decrease constant.
    pub armijo_c: Real,
    /// Backtracking factor (< 1).
    pub armijo_beta: Real,
    /// Maximum Armijo backtracking steps per iteration.
    pub max_ls_iters: usize,
    /// Fletcher-Reeves or Polak-Ribière+.
    pub variant: CgVariant,
    /// Restart to steepest descent every N iterations (0 = never force restart).
    ///
    /// Automatic restart still occurs when the conjugate direction is not
    /// a descent direction.
    pub restart_every: usize,
}

impl Default for RCGConfig {
    fn default() -> Self {
        Self {
            max_iters: 1000,
            grad_tol: 1e-6,
            init_step: 1.0,
            armijo_c: 1e-4,
            armijo_beta: 0.5,
            max_ls_iters: 50,
            variant: CgVariant::PolakRibiere,
            restart_every: 0,
        }
    }
}

/// Run Riemannian Conjugate Gradient.
///
/// # Arguments
///
/// - `manifold`: Must implement `Retraction` and `ParallelTransport`.
/// - `cost`: Cost function f: M → R.
/// - `rgrad`: Riemannian gradient (already projected onto T_x M).
/// - `x0`: Initial point.
/// - `config`: Solver parameters.
pub fn minimize_rcg<M, F, G>(
    manifold: &M,
    cost: F,
    rgrad: G,
    x0: M::Point,
    config: &RCGConfig,
) -> OptResult<M::Point>
where
    M: Manifold + Retraction + ParallelTransport,
    F: Fn(&M::Point) -> Real,
    G: Fn(&M::Point) -> M::Tangent,
{
    let mut x = x0;
    let mut f_x = cost(&x);
    let mut g = rgrad(&x);
    let mut g_sq = manifold.inner(&x, &g, &g);
    let mut g_norm = {
        #[cfg(feature = "std")]
        {
            g_sq.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sqrt(g_sq)
        }
    };

    // Initial direction: steepest descent.
    let mut p = -g.clone();

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

        // Ensure p is a descent direction; if not, restart.
        if manifold.inner(&x, &g, &p) >= 0.0 {
            p = -g.clone();
        }
        let slope = manifold.inner(&x, &g, &p);

        // Armijo backtracking line search.
        let mut t = config.init_step;
        let mut x_new = manifold.retract(&x, &(p.clone() * t));
        let mut f_new = cost(&x_new);
        for _ in 0..config.max_ls_iters {
            if f_new <= f_x + config.armijo_c * t * slope {
                break;
            }
            t *= config.armijo_beta;
            x_new = manifold.retract(&x, &(p.clone() * t));
            f_new = cost(&x_new);
        }

        // Capture state before stepping.
        let x_prev = x.clone();
        let g_prev = g.clone();
        let g_sq_prev = g_sq;
        let p_prev = p.clone();

        // Accept step.
        x = x_new;
        f_x = f_new;
        g = rgrad(&x);
        g_sq = manifold.inner(&x, &g, &g);
        g_norm = {
            #[cfg(feature = "std")]
            {
                g_sq.sqrt()
            }
            #[cfg(not(feature = "std"))]
            {
                libm::sqrt(g_sq)
            }
        };

        // Forced restart check.
        let force_restart = config.restart_every > 0 && (iter + 1) % config.restart_every == 0;

        let beta = if force_restart || g_sq_prev < 1e-30 {
            0.0
        } else {
            match config.variant {
                CgVariant::FletcherReeves => g_sq / g_sq_prev,
                CgVariant::PolakRibiere => {
                    // Transport g_prev from x_prev to x, compute PR+ β.
                    let g_pt = manifold
                        .transport(&x_prev, &x, &g_prev)
                        .unwrap_or_else(|_| g.clone());
                    let diff = g.clone() - g_pt; // g_{k+1} - PT(g_k)
                    let num = manifold.inner(&x, &g, &diff);
                    (num / g_sq_prev).max(0.0)
                }
            }
        };

        // Transport p_prev from x_prev to x and form new direction.
        let p_pt = if beta.abs() < 1e-30 {
            manifold.zero_tangent(&x)
        } else {
            manifold
                .transport(&x_prev, &x, &p_prev)
                .unwrap_or_else(|_| manifold.zero_tangent(&x))
        };

        p = -g.clone() + p_pt * beta;
    }

    OptResult {
        point: x,
        value: f_x,
        grad_norm: g_norm,
        iterations: config.max_iters,
        converged: false,
    }
}
