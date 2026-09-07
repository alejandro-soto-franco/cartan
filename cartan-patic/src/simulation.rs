//! The coupled active p-atic loop.
//!
//! One step is: assemble the active force from the current order parameter,
//! solve Stokes for the velocity, transport the order parameter along the flow
//! and co-rotate it by the local vorticity, then take one gradient-flow step
//! on the free energy. Transport and co-rotation together are the material
//! derivative.
//!
//! ## Co-rotation
//!
//! A material element in a flow of vorticity `omega` turns at angular velocity
//! `omega / 2`, so the rotor updates by `exp(dt omega / 2)` on the left. That
//! factor of one half is the whole coupling at leading order and it has an
//! exact test: a rigid rotation must turn the director at half its vorticity.
//!
//! Transport is semi-Lagrangian, in [`crate::advect`]. It is exact on affine
//! data, so a field linear in space translates with no numerical diffusion.

use nalgebra::DVector;

use cartan_core::rotor::Rotor3;

use crate::active::active_force_general;
use crate::advect::advect;
use crate::complex3::Complex3;
use crate::energy::{Energy, State};
use crate::error::PaticError;
use crate::geometry::Geometry3;
use crate::group::SymmetryGroup;
use crate::spin::Incidence;
use crate::stokes::Stokes;

/// Diagnostics from one coupled step.
#[derive(Clone, Copy, Debug)]
pub struct StepReport {
    /// Free energy after the step.
    pub energy: f64,
    /// `L2` norm of the velocity.
    pub speed: f64,
    /// Viscous dissipation.
    pub dissipation: f64,
    /// Worst deviation of any rotor from the unit sphere.
    pub norm_defect: f64,
}

/// The coupled system: order parameter, flow, and their interaction.
pub struct Simulation<'a> {
    complex: &'a Complex3,
    geometry: &'a Geometry3,
    /// Triangle incidence for the order-parameter energy.
    incidence: &'a Incidence,
    energy: &'a Energy,
    stokes: Stokes,
    no_slip_edges: Vec<usize>,
    zeta: f64,
    dt: f64,
}

impl<'a> Simulation<'a> {
    /// Assemble the coupled system.
    #[must_use]
    pub fn new(
        complex: &'a Complex3,
        geometry: &'a Geometry3,
        incidence: &'a Incidence,
        energy: &'a Energy,
        eta: f64,
        zeta: f64,
        dt: f64,
    ) -> Self {
        Self {
            complex,
            geometry,
            incidence,
            energy,
            stokes: Stokes::assemble(complex, geometry, eta),
            no_slip_edges: Vec::new(),
            zeta,
            dt,
        }
    }

    /// Constrain the given edges to zero velocity.
    #[must_use]
    pub fn with_no_slip(mut self, edges: &[usize]) -> Self {
        self.no_slip_edges = edges.to_vec();
        self
    }

    /// The activity.
    #[must_use]
    pub fn zeta(&self) -> f64 {
        self.zeta
    }

    /// Solve for the velocity driven by the current order parameter.
    pub fn velocity(&self, state: &State) -> Result<DVector<f64>, PaticError> {
        let f = active_force_general(self.complex, self.geometry, self.energy, state, self.zeta)?;
        let (u, _) = if self.no_slip_edges.is_empty() {
            self.stokes.solve(&f)
        } else {
            self.stokes.solve_no_slip(&f, &self.no_slip_edges)
        };
        Ok(u)
    }

    /// Vorticity as a vector at each vertex, from the Whitney reconstruction
    /// of `curl u` averaged over the incident tetrahedra.
    #[must_use]
    pub fn vorticity(&self, u: &DVector<f64>) -> Vec<[f64; 3]> {
        let mut acc = vec![[0.0_f64; 3]; self.complex.n_vertices()];
        let mut weight = vec![0.0_f64; self.complex.n_vertices()];
        const LE: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        for tet in self.complex.tets() {
            let d = self.geometry.tet_data(tet);
            // curl of the Whitney interpolant is constant on the cell:
            // curl w_ij = 2 grad l_i x grad l_j.
            let mut w = [0.0_f64; 3];
            for &[a, b] in LE.iter() {
                let e = self.complex.edge_of(&[tet[a], tet[b]]);
                let ga = d.grads[a];
                let gb = d.grads[b];
                let cr = [
                    ga[1] * gb[2] - ga[2] * gb[1],
                    ga[2] * gb[0] - ga[0] * gb[2],
                    ga[0] * gb[1] - ga[1] * gb[0],
                ];
                for (k, wk) in w.iter_mut().enumerate() {
                    *wk += 2.0 * u[e] * cr[k];
                }
            }
            for &v in tet.iter() {
                weight[v] += d.volume;
                for (k, ak) in acc[v].iter_mut().enumerate() {
                    *ak += w[k] * d.volume;
                }
            }
        }
        for (v, a) in acc.iter_mut().enumerate() {
            let ww = weight[v].max(1e-300);
            for ak in a.iter_mut() {
                *ak /= ww;
            }
        }
        acc
    }

    /// Rotate every rotor by the local vorticity for one step.
    ///
    /// A material element turns at half the vorticity, so the increment is
    /// `exp(dt omega / 2)`.
    pub fn corotate(&self, state: &mut State, omega: &[[f64; 3]], dt: f64) {
        for (v, w) in omega.iter().enumerate() {
            let x = [0.5 * dt * w[0], 0.5 * dt * w[1], 0.5 * dt * w[2]];
            let theta = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            if theta < 1e-300 {
                continue;
            }
            let (s, c) = (theta / 2.0).sin_cos();
            let k = s / theta;
            let r = Rotor3 {
                w: c,
                x: k * x[0],
                y: k * x[1],
                z: k * x[2],
            };
            state.rotors[v] = r.compose(&state.rotors[v]);
        }
    }

    /// One coupled step.
    ///
    /// Generic over the symmetry because transport has to interpolate a coset:
    /// the four rotors of a tetrahedron are aligned through the defect group
    /// before they are averaged.
    pub fn step<H: SymmetryGroup>(&self, state: &mut State) -> Result<StepReport, PaticError> {
        let u = self.velocity(state)?;
        let omega = self.vorticity(&u);
        *state = advect::<H>(self.complex, self.geometry, state, u.as_slice(), self.dt);
        self.corotate(state, &omega, self.dt);

        // One explicit gradient-flow step on the free energy.
        let n = self.energy.basis().n_amplitudes();
        let block = 3 + n;
        let g = self.energy.gradient(self.incidence, state);
        for v in 0..state.n_vertices() {
            let x = [
                -self.dt * g[v * block],
                -self.dt * g[v * block + 1],
                -self.dt * g[v * block + 2],
            ];
            let theta = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
            if theta > 1e-300 {
                let (s, c) = (theta / 2.0).sin_cos();
                let k = s / theta;
                let r = Rotor3 {
                    w: c,
                    x: k * x[0],
                    y: k * x[1],
                    z: k * x[2],
                };
                state.rotors[v] = r.compose(&state.rotors[v]);
            }
            for i in 0..n {
                state.amplitudes[v * n + i] -= self.dt * g[v * block + 3 + i];
            }
        }

        Ok(StepReport {
            energy: self.energy.total(self.incidence, state),
            speed: self.stokes.velocity_norm(&u),
            dissipation: self.stokes.dissipation(&u),
            norm_defect: state.worst_norm_defect(),
        })
    }
}
