//! Gradient flow of the p-atic functional.
//!
//! The state moves in `so(3)` and in the amplitudes. Rotor updates are a
//! left multiplication by `exp(-dt xi)`, which is a unit rotor by
//! construction, so the admissible set `|R| = 1` is preserved to machine
//! precision rather than by renormalising.
//!
//! Two integrators. The explicit one is cheap and conditionally stable, with
//! its threshold measured rather than assumed. The discrete-gradient one
//! satisfies `E^{n+1} - E^n = -dt |g_bar|^2` at any step size it solves, and
//! takes a fixed-point solve per step.
//!
//! ## The identity is a statement about the free energy alone
//!
//! Both integrators here move the state down the gradient of `F`, and the
//! identity is about that motion. It says nothing about a coupled run.
//!
//! The active stress is not a gradient of anything: it injects energy, which
//! is what makes the system active. In [`crate::simulation`] the free energy
//! is therefore no longer a Lyapunov function, and the governing statement
//! becomes a balance rather than a decrease: at a steady state the power the
//! active force does on the flow equals the viscous dissipation, which is what
//! `power_input_equals_dissipation` measures. Reading the identity below as a
//! guarantee about an active run is the mistake to avoid.

use cartan_core::rotor::Rotor3;

use crate::energy::{Energy, State};
use crate::error::PaticError;
use crate::spin::Incidence;

/// The rotor `exp(theta * axis)` for a rotation of angle `theta`.
fn exp_so3(v: [f64; 3]) -> Rotor3 {
    let theta = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if theta < 1e-300 {
        return Rotor3::IDENTITY;
    }
    let (s, c) = (theta / 2.0).sin_cos();
    let k = s / theta;
    Rotor3 {
        w: c,
        x: k * v[0],
        y: k * v[1],
        z: k * v[2],
    }
}

/// Move `state` along `-step * direction`, in place.
fn advance(state: &mut State, direction: &[f64], step: f64, n_amp: usize) {
    let block = 3 + n_amp;
    for v in 0..state.n_vertices() {
        let xi = [
            -step * direction[v * block],
            -step * direction[v * block + 1],
            -step * direction[v * block + 2],
        ];
        state.rotors[v] = exp_so3(xi).compose(&state.rotors[v]);
        for i in 0..n_amp {
            state.amplitudes[v * n_amp + i] -= step * direction[v * block + 3 + i];
        }
    }
}

/// Explicit gradient descent with a retraction on `S^3`.
pub struct ExplicitFlow {
    dt: f64,
}

impl ExplicitFlow {
    /// A flow with the given step.
    #[must_use]
    pub fn new(dt: f64) -> Self {
        Self { dt }
    }

    /// One step. Returns the energy after it.
    pub fn step(&self, energy: &Energy, inc: &Incidence, state: &mut State) -> f64 {
        let n = energy.basis().n_amplitudes();
        let g = energy.gradient(inc, state);
        advance(state, &g, self.dt, n);
        energy.total(inc, state)
    }
}

/// Gonzalez discrete-gradient flow.
///
/// The discrete gradient `g_bar` is the midpoint gradient corrected along the
/// step so that `<g_bar, delta> = E^{n+1} - E^n` exactly. The step
/// `delta = -dt g_bar` then gives `E^{n+1} - E^n = -dt |g_bar|^2` at any step
/// size the fixed point reaches, rather than an estimate under a step
/// restriction.
///
/// This is a statement about the gradient flow of `F`. Activity is not a
/// gradient and injects energy, so a coupled run obeys a balance instead; see
/// the module documentation.
pub struct DiscreteGradientFlow {
    dt: f64,
    tol: f64,
    max_iter: usize,
}

impl DiscreteGradientFlow {
    /// A flow with the given step, fixed-point tolerance and iteration cap.
    #[must_use]
    pub fn new(dt: f64, tol: f64, max_iter: usize) -> Self {
        Self { dt, tol, max_iter }
    }

    /// One step. Returns the energy after it, or reports non-convergence.
    ///
    /// The fixed point is on the step direction: propose one, retract by it,
    /// form the corrected discrete gradient, repeat. Plain Picard iteration
    /// contracts only for small `dt`, so the update is damped and the damping
    /// halves on a retry, which is what makes the large-step cases converge.
    pub fn step(
        &self,
        energy: &Energy,
        inc: &Incidence,
        state: &mut State,
        step_index: usize,
    ) -> Result<f64, PaticError> {
        let n = energy.basis().n_amplitudes();
        let e0 = energy.total(inc, state);
        let g0 = energy.gradient(inc, state);

        let mut omega = 1.0_f64;
        let mut residual = f64::INFINITY;
        let mut dir = g0.clone();

        for _attempt in 0..8 {
            dir.clone_from(&g0);
            residual = f64::INFINITY;
            let mut diverged = false;

            for _ in 0..self.max_iter {
                let mut trial = state.clone();
                advance(&mut trial, &dir, self.dt, n);
                let mut mid = state.clone();
                advance(&mut mid, &dir, self.dt * 0.5, n);

                let e1 = energy.total(inc, &trial);
                if !e1.is_finite() {
                    diverged = true;
                    break;
                }
                let gm = energy.gradient(inc, &mid);

                let delta: Vec<f64> = dir.iter().map(|d| -self.dt * d).collect();
                let dd: f64 = delta.iter().map(|d| d * d).sum();
                let gdotd: f64 = gm.iter().zip(&delta).map(|(a, b)| a * b).sum();
                let alpha = if dd > 1e-300 {
                    (e1 - e0 - gdotd) / dd
                } else {
                    0.0
                };

                let mut worst = 0.0_f64;
                for i in 0..dir.len() {
                    let target = gm[i] + alpha * delta[i];
                    let next = (1.0 - omega) * dir[i] + omega * target;
                    worst = worst.max((next - dir[i]).abs());
                    dir[i] = next;
                }
                if !dir.iter().all(|v| v.is_finite()) {
                    diverged = true;
                    break;
                }
                residual = worst;
                if residual < self.tol {
                    break;
                }
            }

            if !diverged && residual < self.tol {
                advance(state, &dir, self.dt, n);
                let e = energy.total(inc, state);
                return Ok(e);
            }
            omega *= 0.5;
        }

        Err(PaticError::NewtonDiverged {
            step: step_index,
            residual,
        })
    }

    /// The step size.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.dt
    }
}
