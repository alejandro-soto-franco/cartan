// ~/cartan/cartan-geo/src/jacobi.rs

//! Jacobi field integration along a geodesic.
//!
//! A Jacobi field J(t) is a vector field along a geodesic gamma(t) satisfying
//! the Jacobi equation (geodesic deviation equation):
//!
//!   D²J/dt² + R(J, γ') γ' = 0
//!
//! where D/dt is the covariant derivative along gamma and R is the Riemann
//! curvature tensor. Jacobi fields encode the linearized behavior of nearby
//! geodesics and are the key tool for understanding how geodesics spread
//! (positive curvature → converging, negative curvature → diverging).
//!
//! ## Algorithm
//!
//! We integrate the Jacobi ODE using a 4th-order Runge-Kutta scheme on the
//! *tangent* bundle. The state is (J, J') in T_{gamma(t)} M × T_{gamma(t)} M.
//!
//! Because we work in ambient (extrinsic) coordinates and the Curvature trait
//! gives us R(u,v)w directly, we can integrate without explicitly constructing
//! the Levi-Civita connection. The covariant derivative is approximated by the
//! ambient derivative followed by tangent projection — exact for submanifolds
//! of Euclidean space (sphere, Grassmann, SPD, SO(N)).
//!
//! ## Limitations
//!
//! - This implementation integrates in the ambient space and projects back to
//!   the tangent bundle at each step. This is correct for all manifolds embedded
//!   isometrically in Euclidean space (which is all v0.1 manifolds).
//! - For abstract manifolds without an ambient Euclidean space, the covariant
//!   derivative requires explicit connection coefficients (not yet in cartan-core).
//! - Step count controls accuracy: use more steps for highly curved manifolds.
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Chapter 5 (Jacobi fields).
//! - Milnor. "Morse Theory." Chapter 2 (index theorem via Jacobi fields).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use cartan_core::{Curvature, Manifold, ParallelTransport, Real};

use crate::geodesic::Geodesic;

/// Result of a Jacobi field integration.
#[cfg(feature = "alloc")]
pub struct JacobiResult<T> {
    /// Parameter values t_0, t_1, ..., t_n where samples were taken.
    pub params: alloc::vec::Vec<Real>,
    /// J(t_i): the Jacobi field values at each sample parameter.
    pub field: alloc::vec::Vec<T>,
    /// J'(t_i): the covariant derivative (velocity) of J at each sample.
    pub velocity: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
/// Integrate a Jacobi field along a geodesic using 4th-order Runge-Kutta.
///
/// # Arguments
///
/// - `geodesic`: The base geodesic gamma(t) = Exp_p(t * v).
/// - `j0`: Initial value J(0) — a tangent vector at gamma(0).
/// - `j0_dot`: Initial covariant velocity J'(0) — a tangent vector at gamma(0).
/// - `n_steps`: Number of integration steps on [0, 1].
///
/// # Returns
///
/// A `JacobiResult` with field values and velocities at each step.
///
/// # Type bounds
///
/// Requires `M: Curvature + ParallelTransport` because:
/// - `Curvature`: to evaluate R(J, γ') γ' at each point.
/// - `ParallelTransport`: to move (J, J') along gamma to the next step's
///   tangent space before applying the RK4 update.
pub fn integrate_jacobi<M>(
    geodesic: &Geodesic<'_, M>,
    j0: M::Tangent,
    j0_dot: M::Tangent,
    n_steps: usize,
) -> JacobiResult<M::Tangent>
where
    M: Manifold + Curvature + ParallelTransport,
{
    assert!(n_steps >= 1, "integrate_jacobi: n_steps must be >= 1");

    let m = geodesic.manifold;
    let dt = 1.0 / n_steps as Real;
    let gamma_dot = geodesic.velocity.clone(); // γ'(0) = constant speed vector

    let mut params = Vec::with_capacity(n_steps + 1);
    let mut field = Vec::with_capacity(n_steps + 1);
    let mut velocity = Vec::with_capacity(n_steps + 1);

    params.push(0.0);
    field.push(j0.clone());
    velocity.push(j0_dot.clone());

    let mut j = j0;
    let mut j_dot = j0_dot;

    for k in 0..n_steps {
        let t = k as Real * dt;
        let p = geodesic.eval(t);
        let p_next = geodesic.eval(t + dt);

        // RK4 on the system: d/dt (J, J') = (J', -R(J, γ')γ')
        // Since we're in ambient coordinates projected onto T_pM, the curvature
        // term is exact for isometrically embedded manifolds.
        //
        // All evaluations happen at the base point p (frozen coefficients within
        // one step), which is an explicit RK4 with fixed t. This is valid for
        // small dt and is O(dt^4) accurate.

        let rhs = |j_val: &M::Tangent, j_vel: &M::Tangent| -> (M::Tangent, M::Tangent) {
            // Acceleration: -R(J, γ') γ'
            let r = m.riemann_curvature(&p, j_val, &gamma_dot, &gamma_dot);
            let acc = -r;
            (j_vel.clone(), acc)
        };

        let (k1_j, k1_v) = rhs(&j, &j_dot);
        let j_mid1 = j.clone() + k1_j.clone() * (dt * 0.5);
        let v_mid1 = j_dot.clone() + k1_v.clone() * (dt * 0.5);

        let (k2_j, k2_v) = rhs(&j_mid1, &v_mid1);
        let j_mid2 = j.clone() + k2_j.clone() * (dt * 0.5);
        let v_mid2 = j_dot.clone() + k2_v.clone() * (dt * 0.5);

        let (k3_j, k3_v) = rhs(&j_mid2, &v_mid2);
        let j_end = j.clone() + k3_j.clone() * dt;
        let v_end = j_dot.clone() + k3_v.clone() * dt;

        let (k4_j, k4_v) = rhs(&j_end, &v_end);

        // RK4 update in ambient space
        let j_new_ambient = j.clone()
            + (k1_j.clone() + k2_j.clone() * 2.0 + k3_j.clone() * 2.0 + k4_j) * (dt / 6.0);
        let v_new_ambient = j_dot.clone() + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);

        // Project back onto tangent space at p_next (corrects ambient drift).
        let j_new = m.project_tangent(&p_next, &j_new_ambient);
        let v_new = m.project_tangent(&p_next, &v_new_ambient);

        // Transport for next iteration: move j, j_dot from p to p_next so that
        // the RK4 rhs is evaluated in the correct tangent space.
        // Use parallel transport to preserve the geometric meaning of J.
        j = m
            .transport(&p, &p_next, &j_new)
            .unwrap_or_else(|_| j_new.clone());
        j_dot = m
            .transport(&p, &p_next, &v_new)
            .unwrap_or_else(|_| v_new.clone());

        params.push(t + dt);
        field.push(j.clone());
        velocity.push(j_dot.clone());
    }

    JacobiResult {
        params,
        field,
        velocity,
    }
}

#[cfg(feature = "alloc")]
/// Integrate a Jacobi field along an **arbitrary base curve** using RK4.
///
/// The Jacobi equation `D²J/dt² + R(J, γ') γ' = 0` holds for any smooth
/// curve `γ`, not only geodesics. This lets Jacobi-field integration ride
/// on an SDE path from `cartan-stochastic::stochastic_development` or any
/// piecewise-linear trajectory on `M`.
///
/// # Arguments
///
/// - `manifold`: the ambient manifold.
/// - `path`: the base curve sampled at times `t_i = i · dt`. Length
///   `n_points = n_steps + 1`.
/// - `dt`: uniform time increment between consecutive path samples.
/// - `j0`: initial Jacobi value `J(t_0)` as a tangent at `path[0]`.
/// - `j0_dot`: initial covariant velocity `J'(t_0)` as a tangent at `path[0]`.
///
/// # Velocity reconstruction
///
/// At each step the instantaneous velocity `γ'(t_i)` is approximated by
/// `log_{path[i]}(path[i+1]) / dt`. This is exact to leading order in `dt`
/// for any `C^2` curve and matches the discretisation level of the RK4
/// Jacobi integrator itself. For the final step we reuse the previous
/// velocity as a fallback (the RK4 update at `t_{n-1}` only needs one
/// velocity evaluation and the integrator never reads `velocity[n]`).
///
/// # Errors
///
/// Returns `CartanError` when `log` fails (typically at the cut locus).
/// Pass a finer `path` discretisation to avoid.
pub fn integrate_jacobi_along_path<M>(
    manifold: &M,
    path: &[M::Point],
    dt: Real,
    j0: M::Tangent,
    j0_dot: M::Tangent,
) -> Result<JacobiResult<M::Tangent>, cartan_core::CartanError>
where
    M: Manifold + Curvature + ParallelTransport,
{
    assert!(
        path.len() >= 2,
        "integrate_jacobi_along_path: need at least 2 path points (n_steps >= 1)"
    );
    let n_steps = path.len() - 1;
    let inv_dt = 1.0 / dt;

    let mut params = Vec::with_capacity(n_steps + 1);
    let mut field = Vec::with_capacity(n_steps + 1);
    let mut velocity = Vec::with_capacity(n_steps + 1);

    params.push(0.0);
    field.push(j0.clone());
    velocity.push(j0_dot.clone());

    let mut j = j0;
    let mut j_dot = j0_dot;

    for k in 0..n_steps {
        let p = &path[k];
        let p_next = &path[k + 1];

        // Approximate γ'(t_k) by finite-difference log. This is exact to
        // leading order in dt for C^2 curves. A future higher-order scheme
        // could use three-point stencils but RK4 accuracy on the Jacobi ODE
        // is itself O(dt^4), so upgrading just the velocity estimate alone
        // does not improve the overall order.
        let gamma_dot = manifold.log(p, p_next)? * inv_dt;

        // Same RK4 update as the geodesic case, with frozen γ' and base
        // point p within one step.
        let rhs = |j_val: &M::Tangent, j_vel: &M::Tangent| -> (M::Tangent, M::Tangent) {
            let r = manifold.riemann_curvature(p, j_val, &gamma_dot, &gamma_dot);
            (j_vel.clone(), -r)
        };

        let (k1_j, k1_v) = rhs(&j, &j_dot);
        let j_mid1 = j.clone() + k1_j.clone() * (dt * 0.5);
        let v_mid1 = j_dot.clone() + k1_v.clone() * (dt * 0.5);

        let (k2_j, k2_v) = rhs(&j_mid1, &v_mid1);
        let j_mid2 = j.clone() + k2_j.clone() * (dt * 0.5);
        let v_mid2 = j_dot.clone() + k2_v.clone() * (dt * 0.5);

        let (k3_j, k3_v) = rhs(&j_mid2, &v_mid2);
        let j_end = j.clone() + k3_j.clone() * dt;
        let v_end = j_dot.clone() + k3_v.clone() * dt;

        let (k4_j, k4_v) = rhs(&j_end, &v_end);

        let j_new_ambient = j.clone()
            + (k1_j.clone() + k2_j.clone() * 2.0 + k3_j.clone() * 2.0 + k4_j) * (dt / 6.0);
        let v_new_ambient = j_dot.clone() + (k1_v + k2_v * 2.0 + k3_v * 2.0 + k4_v) * (dt / 6.0);

        let j_new = manifold.project_tangent(p_next, &j_new_ambient);
        let v_new = manifold.project_tangent(p_next, &v_new_ambient);

        // Parallel-transport to the next tangent space so the next RK4
        // evaluation operates in the correct coordinates.
        j = manifold
            .transport(p, p_next, &j_new)
            .unwrap_or_else(|_| j_new.clone());
        j_dot = manifold
            .transport(p, p_next, &v_new)
            .unwrap_or_else(|_| v_new.clone());

        params.push((k + 1) as Real * dt);
        field.push(j.clone());
        velocity.push(j_dot.clone());
    }

    Ok(JacobiResult {
        params,
        field,
        velocity,
    })
}
