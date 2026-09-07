//! Selection over a control base.
//!
//! The programme notes organise selection over a base of activity, curvature
//! and topology by an attractor structure. This module supplies the sweep and
//! the recurrent-set analysis over that base; the sheaf-theoretic machinery
//! above it is not here.
//!
//! What a sweep gives is a curve of observables against the control, and the
//! Morse decomposition of the observable's own drift picks out the attracting
//! branches and the transitions between them.

use crate::complex3::Complex3;
use crate::energy::{Energy, State};
use crate::error::PaticError;
use crate::geometry::Geometry3;
use crate::group::SymmetryGroup;
use crate::simulation::Simulation;
use crate::spin::Incidence;

/// One point of a control sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepPoint {
    /// Activity.
    pub zeta: f64,
    /// Free energy at the end of the run.
    pub energy: f64,
    /// Flow speed at the end of the run.
    pub speed: f64,
    /// Viscous dissipation at the end of the run.
    pub dissipation: f64,
    /// Worst rotor norm defect over the run, as a health check.
    pub worst_norm_defect: f64,
}

/// The fixed part of a sweep: the domain and the functional.
pub struct SweepDomain<'a> {
    /// Tetrahedral complex.
    pub complex: &'a Complex3,
    /// Vertex positions.
    pub geometry: &'a Geometry3,
    /// Triangle incidence for the order-parameter energy.
    pub incidence: &'a Incidence,
    /// The free energy.
    pub energy: &'a Energy,
    /// Edges constrained to zero velocity.
    pub no_slip: &'a [usize],
}

/// The run parameters held fixed across a sweep.
#[derive(Clone, Copy, Debug)]
pub struct SweepRun {
    /// Viscosity.
    pub eta: f64,
    /// Time step.
    pub dt: f64,
    /// Steps per control value.
    pub steps: usize,
}

/// Run the coupled loop across a range of activities.
///
/// Each point starts from the same initial state, so the sweep measures the
/// response to activity rather than a continuation along it.
pub fn sweep<H: SymmetryGroup>(
    d: &SweepDomain<'_>,
    initial: &State,
    zetas: &[f64],
    run: SweepRun,
) -> Result<Vec<SweepPoint>, PaticError> {
    let (c, g, inc, e, no_slip) = (d.complex, d.geometry, d.incidence, d.energy, d.no_slip);
    let (eta, dt, steps) = (run.eta, run.dt, run.steps);
    let mut out = Vec::with_capacity(zetas.len());
    for &zeta in zetas {
        let sim = Simulation::new(c, g, inc, e, eta, zeta, dt).with_no_slip(no_slip);
        let mut state = initial.clone();
        let mut worst = 0.0_f64;
        let mut last = None;
        for _ in 0..steps {
            let r = sim.step::<H>(&mut state)?;
            worst = worst.max(r.norm_defect);
            last = Some(r);
        }
        let r = last.expect("at least one step");
        out.push(SweepPoint {
            zeta,
            energy: r.energy,
            speed: r.speed,
            dissipation: r.dissipation,
            worst_norm_defect: worst,
        });
    }
    Ok(out)
}

/// Indices where an observable changes by more than `threshold` between
/// adjacent control values, relative to its own range.
///
/// A transition in the attracting branch shows up as a jump; a smooth response
/// does not.
#[must_use]
pub fn transitions(
    points: &[SweepPoint],
    observable: fn(&SweepPoint) -> f64,
    threshold: f64,
) -> Vec<usize> {
    if points.len() < 2 {
        return Vec::new();
    }
    let vals: Vec<f64> = points.iter().map(observable).collect();
    let lo = vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (hi - lo).abs().max(1e-300);
    (1..vals.len())
        .filter(|&i| (vals[i] - vals[i - 1]).abs() / range > threshold)
        .collect()
}
