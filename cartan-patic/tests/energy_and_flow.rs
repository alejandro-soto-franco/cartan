//! Well-posedness of the functional and stability of the flow.

use cartan_core::rotor::Rotor3;
use cartan_patic::energy::{Energy, State};
use cartan_patic::flow::{DiscreteGradientFlow, ExplicitFlow};
use cartan_patic::group::{AxialApolar, Dicyclic, SymmetryGroup};
use cartan_patic::spin::Incidence;
use nalgebra::DMatrix;

/// A small triangulated patch: two rings of four vertices.
fn patch() -> Incidence {
    let mut t = Vec::new();
    for i in 0..4 {
        let j = (i + 1) % 4;
        t.push([i, j, 4 + i]);
        t.push([j, 4 + j, 4 + i]);
    }
    Incidence::from_triangles(8, &t)
}

fn energy_for<H: SymmetryGroup>(a_diag: f64, elastic: f64) -> Energy {
    let n = H::N_AMPLITUDES;
    let a = DMatrix::from_diagonal_element(n, n, a_diag);
    let c = vec![0.0; n * n * n];
    let b = DMatrix::identity(n, n);
    Energy::new::<H>(a, c, b, elastic).expect("coercive")
}

fn scattered_state(nv: usize, n_amp: usize) -> State {
    let mut s = State::uniform(nv, Rotor3::IDENTITY, &vec![0.6; n_amp]);
    for v in 0..nv {
        let t = 0.3 + v as f64 * 0.41;
        let (sn, cs) = (t / 2.0).sin_cos();
        s.rotors[v] = Rotor3 {
            w: cs,
            x: sn * 0.6,
            y: sn * 0.8,
            z: 0.0,
        };
        for i in 0..n_amp {
            s.amplitudes[v * n_amp + i] = 0.4 + 0.2 * ((v + i) as f64).sin();
        }
    }
    s
}

/// Move along `-step * dir`, matching the flow's own retraction.
fn nudge(state: &mut State, dir: &[f64], step: f64, n_amp: usize) {
    let block = 3 + n_amp;
    for v in 0..state.n_vertices() {
        let x = [
            -step * dir[v * block],
            -step * dir[v * block + 1],
            -step * dir[v * block + 2],
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
        for i in 0..n_amp {
            state.amplitudes[v * n_amp + i] -= step * dir[v * block + 3 + i];
        }
    }
}

#[test]
fn an_indefinite_quartic_form_is_rejected() {
    let n = <AxialApolar as SymmetryGroup>::N_AMPLITUDES;
    let a = DMatrix::from_diagonal_element(n, n, -1.0);
    let c = vec![0.0; n * n * n];
    let b = DMatrix::from_diagonal_element(n, n, -1.0);
    assert!(
        Energy::new::<AxialApolar>(a, c, b, 1.0).is_err(),
        "a negative-definite quartic form is unbounded below and must be rejected"
    );
}

#[test]
fn the_gradient_matches_finite_differences() {
    for &elastic in &[0.0_f64, 1.0] {
        let e = energy_for::<AxialApolar>(-1.0, elastic);
        let inc = patch();
        let n = e.basis().n_amplitudes();
        let state = scattered_state(inc.n_vertices(), n);
        let g = e.gradient(&inc, &state);
        let h = 1e-6;
        let block = 3 + n;
        for v in [0usize, 3, 7] {
            for axis in 0..3 {
                let mut dir = vec![0.0; state.n_vertices() * block];
                dir[v * block + axis] = 1.0;
                let mut plus = state.clone();
                let mut minus = state.clone();
                nudge(&mut plus, &dir, -h, n);
                nudge(&mut minus, &dir, h, n);
                let fd = (e.total(&inc, &plus) - e.total(&inc, &minus)) / (2.0 * h);
                let an = g[v * block + axis];
                assert!(
                    (fd - an).abs() < 1e-5 * (1.0 + an.abs()),
                    "elastic {elastic}, vertex {v} axis {axis}: fd {fd:e} analytic {an:e}"
                );
            }
            for i in 0..n {
                let mut plus = state.clone();
                let mut minus = state.clone();
                plus.amplitudes[v * n + i] += h;
                minus.amplitudes[v * n + i] -= h;
                let fd = (e.total(&inc, &plus) - e.total(&inc, &minus)) / (2.0 * h);
                let an = g[v * block + 3 + i];
                assert!(
                    (fd - an).abs() < 1e-5 * (1.0 + an.abs()),
                    "elastic {elastic}, vertex {v} amp {i}: fd {fd:e} analytic {an:e}"
                );
            }
        }
    }
}

/// The explicit flow dissipates below its threshold, measured between
/// `dt = 0.3` and `dt = 1` on this patch.
#[test]
fn the_explicit_flow_dissipates_and_stays_on_the_sphere() {
    let e = energy_for::<AxialApolar>(-1.0, 0.5);
    let inc = patch();
    let n = e.basis().n_amplitudes();
    let mut state = scattered_state(inc.n_vertices(), n);
    let flow = ExplicitFlow::new(1e-3);
    let mut last = e.total(&inc, &state);
    for k in 0..2000 {
        let now = flow.step(&e, &inc, &mut state);
        assert!(
            now <= last + 1e-9,
            "step {k}: energy rose from {last:e} to {now:e}"
        );
        last = now;
    }
    assert!(
        state.worst_norm_defect() < 1e-14,
        "rotor left the unit sphere by {:e}",
        state.worst_norm_defect()
    );
}

/// The energy identity holds wherever the implicit solve converges.
///
/// Measured 2026-09-07 on this patch: the damped fixed point converges up to
/// `dt = 3` and fails by `dt = 5`, while the explicit flow stops dissipating
/// between `dt = 0.3` and `dt = 1`. So the discrete gradient buys about a
/// factor of ten in step size, and across its whole range the energy never
/// rose by any amount at all.
#[test]
fn the_discrete_gradient_dissipates_across_its_solver_range() {
    let inc = patch();
    for &dt in &[1e-3_f64, 1e-2, 1e-1, 0.5, 1.0, 2.0, 3.0] {
        let e = energy_for::<AxialApolar>(-1.0, 0.5);
        let n = e.basis().n_amplitudes();
        let mut state = scattered_state(inc.n_vertices(), n);
        let flow = DiscreteGradientFlow::new(dt, 1e-12, 300);
        let mut before = e.total(&inc, &state);
        for k in 0..20 {
            let after = match flow.step(&e, &inc, &mut state, k) {
                Ok(v) => v,
                Err(err) => panic!("dt {dt}: {err}"),
            };
            assert!(
                after <= before + 1e-9 * before.abs().max(1.0),
                "dt {dt} step {k}: energy rose from {before:e} to {after:e}"
            );
            before = after;
        }
        assert!(
            state.worst_norm_defect() < 1e-13,
            "dt {dt}: left the sphere"
        );
    }
}

#[test]
fn biaxial_runs_with_two_amplitudes() {
    let e = energy_for::<Dicyclic<2>>(-1.0, 0.5);
    assert_eq!(e.basis().n_amplitudes(), 2);
    let inc = patch();
    let mut state = scattered_state(inc.n_vertices(), 2);
    let flow = ExplicitFlow::new(1e-3);
    let mut last = e.total(&inc, &state);
    for k in 0..500 {
        let now = flow.step(&e, &inc, &mut state);
        assert!(now <= last + 1e-9, "biaxial step {k}: energy rose");
        last = now;
    }
}
