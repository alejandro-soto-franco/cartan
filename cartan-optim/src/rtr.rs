// ~/cartan/cartan-optim/src/rtr.rs

//! Riemannian Trust Region (RTR) with Steihaug-Toint truncated CG subproblem.
//!
//! ## Algorithm (Absil-Baker-Gallivan 2007)
//!
//! At iterate x_k with gradient g_k = grad f(x_k) and trust radius Δ_k:
//!
//!   1. **Subproblem**: solve (approximately) for η_k ∈ T_{x_k}M:
//!      min m_k(η) = f(x_k) + <g_k, η> + ½ <Hess f(x_k)[η], η>
//!      subject to ||η||_{x_k} ≤ Δ_k
//!      via Steihaug-Toint truncated CG.
//!
//!   2. **Ratio**: ρ_k = (f(x_k) − f(Retract(x_k, η_k))) / (m_k(0) − m_k(η_k))
//!
//!   3. **Accept/reject**:
//!      if ρ_k > ρ_min: x_{k+1} = Retract(x_k, η_k)
//!      else:           x_{k+1} = x_k
//!
//!   4. **Update Δ**:
//!      ρ < 0.25 → Δ_{k+1} = Δ_k / 4
//!      ρ > 0.75 and ||η_k|| ≈ Δ_k → Δ_{k+1} = min(2Δ_k, Δ_max)
//!      else → Δ_{k+1} = Δ_k
//!
//! ## References
//!
//! - Absil, Baker, Gallivan. "Trust-Region Methods on Riemannian Manifolds."
//!   Found. Comput. Math. 2007.
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Chapter 7 (trust region).

use cartan_core::{Connection, Manifold, Real, Retraction};

use crate::result::OptResult;

/// Configuration for Riemannian Trust Region.
#[derive(Debug, Clone)]
pub struct RTRConfig {
    /// Maximum number of outer iterations.
    pub max_iters: usize,
    /// Stop when ||grad f(x)|| < grad_tol.
    pub grad_tol: Real,
    /// Initial trust radius.
    pub delta_init: Real,
    /// Maximum trust radius.
    pub delta_max: Real,
    /// Minimum acceptable ratio ρ for step acceptance.
    pub rho_min: Real,
    /// Maximum number of inner CG iterations per outer step.
    pub max_cg_iters: usize,
    /// CG convergence tolerance (relative to ||g||).
    pub cg_tol: Real,
}

impl Default for RTRConfig {
    fn default() -> Self {
        Self {
            max_iters: 500,
            grad_tol: 1e-6,
            delta_init: 1.0,
            delta_max: 8.0,
            rho_min: 0.1,
            max_cg_iters: 50,
            cg_tol: 0.1,
        }
    }
}

/// Solve the trust-region subproblem via Steihaug-Toint truncated CG.
///
/// Minimize m(η) = <g, η> + ½ <H[η], η>  s.t. ||η||_M ≤ Δ
///
/// Returns the step η and whether the boundary was hit.
fn solve_trs<M>(
    manifold: &M,
    x: &M::Point,
    g: &M::Tangent,
    hess: &dyn Fn(&M::Tangent) -> M::Tangent,
    delta: Real,
    max_cg: usize,
    cg_tol: Real,
) -> M::Tangent
where
    M: Manifold,
{
    let g_norm = manifold.norm(x, g);
    let tol = cg_tol * g_norm;

    // η_0 = 0, r_0 = g, p_0 = -r_0 = -g
    let mut eta = manifold.zero_tangent(x);
    let mut r = g.clone();
    let mut p = -g.clone();

    for _ in 0..max_cg {
        // κ = <p, H[p]>
        let hp = hess(&p);
        let kappa = manifold.inner(x, &p, &hp);

        // Negative curvature: move to boundary in direction p.
        if kappa <= 0.0 {
            return boundary_step(manifold, x, &eta, &p, delta);
        }

        let r_sq = manifold.inner(x, &r, &r);
        let alpha = r_sq / kappa;

        // Proposed step: η_new = η + α p
        let eta_new = eta.clone() + p.clone() * alpha;

        // Boundary hit: step exceeds trust radius.
        if manifold.norm(x, &eta_new) >= delta {
            return boundary_step(manifold, x, &eta, &p, delta);
        }

        eta = eta_new;
        let r_new = r.clone() + hp * alpha;

        if manifold.norm(x, &r_new) < tol {
            return eta;
        }

        let r_sq_new = manifold.inner(x, &r_new, &r_new);
        let beta = r_sq_new / r_sq;
        p = -r_new.clone() + p * beta;
        r = r_new;
    }

    eta
}

/// Find τ ≥ 0 such that ||η + τ p|| = Δ, then return η + τ p.
///
/// This is the boundary intercept: solve ||η||² + 2τ<η,p> + τ²||p||² = Δ².
fn boundary_step<M>(
    manifold: &M,
    x: &M::Point,
    eta: &M::Tangent,
    p: &M::Tangent,
    delta: Real,
) -> M::Tangent
where
    M: Manifold,
{
    let eta_sq = manifold.inner(x, eta, eta);
    let ep = manifold.inner(x, eta, p);
    let p_sq = manifold.inner(x, p, p);

    if p_sq < 1e-30 {
        return eta.clone(); // p ≈ 0, can't move to boundary
    }

    // Solve: p_sq τ² + 2 ep τ + (eta_sq − Δ²) = 0
    let discriminant = ep * ep - p_sq * (eta_sq - delta * delta);
    if discriminant < 0.0 {
        return eta.clone();
    }
    let tau = (-ep + discriminant.sqrt()) / p_sq;
    eta.clone() + p.clone() * tau
}

/// Run Riemannian Trust Region.
///
/// # Arguments
///
/// - `manifold`: Must implement `Retraction` and `Connection`.
/// - `cost`: Cost function.
/// - `rgrad`: Riemannian gradient.
/// - `ehvp`: Euclidean Hessian-vector product. Given a tangent direction `v`,
///   returns the ambient Euclidean HVP `D²f(x)[v]` at the current `x`.
///   The `Connection` impl converts this to the Riemannian HVP.
/// - `x0`: Initial point.
/// - `config`: Solver parameters.
pub fn minimize_rtr<M, F, G, H>(
    manifold: &M,
    cost: F,
    rgrad: G,
    ehvp: H,
    x0: M::Point,
    config: &RTRConfig,
) -> OptResult<M::Point>
where
    M: Manifold + Retraction + Connection,
    F: Fn(&M::Point) -> Real,
    G: Fn(&M::Point) -> M::Tangent,
    H: Fn(&M::Point, &M::Tangent) -> M::Tangent,
{
    let mut x = x0;
    let mut f_x = cost(&x);
    let mut g = rgrad(&x);
    let mut g_norm = manifold.norm(&x, &g);
    let mut delta = config.delta_init;

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

        // Build Riemannian HVP: H_riem[v] = Connection::riemannian_hessian_vector_product(...)
        let hess_riem = |v: &M::Tangent| -> M::Tangent {
            manifold
                .riemannian_hessian_vector_product(&x, &g, v, &|w| ehvp(&x, w))
                .unwrap_or_else(|_| manifold.zero_tangent(&x))
        };

        // Solve the trust-region subproblem.
        let eta = solve_trs(
            manifold,
            &x,
            &g,
            &hess_riem,
            delta,
            config.max_cg_iters,
            config.cg_tol,
        );

        // Model decrease: m(0) - m(eta) = -<g, eta> - ½<H[eta], eta>
        let h_eta = hess_riem(&eta);
        let model_decrease =
            -manifold.inner(&x, &g, &eta) - 0.5 * manifold.inner(&x, &h_eta, &eta);

        // Actual decrease: f(x) - f(Retract(x, eta))
        let x_new = manifold.retract(&x, &eta);
        let f_new = cost(&x_new);
        let actual_decrease = f_x - f_new;

        // Ratio ρ = actual / model.
        let rho = if model_decrease.abs() < 1e-30 {
            1.0 // model predicts ~0, accept any step
        } else {
            actual_decrease / model_decrease
        };

        // Accept or reject.
        if rho > config.rho_min {
            x = x_new;
            f_x = f_new;
            g = rgrad(&x);
            g_norm = manifold.norm(&x, &g);
        }

        // Update trust radius.
        let eta_norm = manifold.norm(&x, &eta);
        if rho < 0.25 {
            delta *= 0.25;
        } else if rho > 0.75 && (delta - eta_norm).abs() < 1e-10 * delta {
            delta = (2.0 * delta).min(config.delta_max);
        }
    }

    OptResult {
        point: x,
        value: f_x,
        grad_norm: g_norm,
        iterations: config.max_iters,
        converged: false,
    }
}
