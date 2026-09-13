//! Stability of the 3D Maxwell evolver on a simplicial background.

use cartan_maxwell::{
    FlrwDriver, MaxwellEvolver, MaxwellState, MetricDriver, cfl_dt, coboundary_matrix,
};
use derham::cochain::Cochain;
use simplicial::r#gen::cartesian::CartesianGrid;

struct Run {
    /// (max U - min U) / U_0 over the whole history.
    excursion: f64,
    /// (mean of the last tenth - mean of the first tenth) / U_0. Secular drift.
    drift: f64,
    worst_gauss: f64,
    finite: bool,
}

fn run(dim: usize, subdiv: usize, steps: usize, dt_factor: f64) -> Run {
    let (complex, coords) = CartesianGrid::new_unit(dim, subdiv).triangulate();
    let base = coords.to_edge_lengths_sq(&complex);
    let driver = FlrwDriver::static_metric(complex.clone(), base);
    let dt = dt_factor * cfl_dt(&driver.lengths_sq_at(0.0));
    let mut evolver = MaxwellEvolver::new(&driver, dt);

    // B = d1(seed) is divergence free by d2 d1 = 0, so d2 B = 0 exactly.
    let d1 = coboundary_matrix(&complex, 1);
    let ne = complex.nsimplices(1);
    let seed = nalgebra::DVector::from_iterator(ne, (0..ne).map(|i| ((i % 7) as f64 - 3.0) / 3.0));
    let b = Cochain::new(2, &d1 * &seed);
    let e = Cochain::new(1, nalgebra::DVector::zeros(ne));
    let mut state = MaxwellState::new(e, b);

    let mut hist = Vec::with_capacity(steps);
    let mut worst_gauss = 0.0f64;
    for _ in 0..steps {
        let u = evolver.step_with_energy(&mut state, None);
        if !u.is_finite() {
            return Run {
                excursion: f64::INFINITY,
                drift: f64::INFINITY,
                worst_gauss,
                finite: false,
            };
        }
        hist.push(u);
        worst_gauss = worst_gauss.max(evolver.magnetic_gauss_residual(&state));
    }
    let u0 = hist[0].abs();
    let lo = hist.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = hist.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w = (steps / 10).max(1);
    let head: f64 = hist[..w].iter().sum::<f64>() / w as f64;
    let tail: f64 = hist[steps - w..].iter().sum::<f64>() / w as f64;
    Run {
        excursion: (hi - lo) / u0,
        drift: (tail - head) / u0,
        worst_gauss,
        finite: true,
    }
}

#[test]
fn gauss_constraint_is_exact_in_3d() {
    for subdiv in [2usize, 3] {
        let r = run(3, subdiv, 200, 1.0);
        assert!(
            r.worst_gauss < 1e-12,
            "3D subdiv {subdiv}: magnetic Gauss residual {:e} is not at round-off",
            r.worst_gauss
        );
    }
}

/// A symplectic integrator keeps the energy oscillating inside a band that does
/// not widen with the run length. The excursion is allowed to be a few percent;
/// what must not happen is secular growth.
#[test]
fn energy_excursion_does_not_widen_with_run_length_in_3d() {
    let short = run(3, 3, 500, 1.0);
    let long = run(3, 3, 8000, 1.0);
    assert!(
        short.finite && long.finite,
        "3D run went non-finite at the CFL step"
    );
    assert!(
        long.excursion < 1.5 * short.excursion.max(1e-12),
        "3D energy band widened with run length: {:e} over 500 steps, {:e} over 8000",
        short.excursion,
        long.excursion
    );
    assert!(
        long.drift.abs() < 0.1 * long.excursion.max(1e-12),
        "3D energy drifts secularly: drift {:e} against band {:e}",
        long.drift,
        long.excursion
    );
}

/// The built-in `cfl_dt` (0.1 * min edge length) must sit strictly inside the
/// stable region, in 3D as well as 2D.
///
/// Measured 2026-09-07 on the Kuhn triangulation of the unit cube: the scheme
/// first goes unstable at 3.2, 3.0, 3.0, 2.8 and 2.8 times the built-in step
/// for subdivisions 2 to 6, against 4.5 in 2D. The margin narrows with
/// refinement and settles near 2.6, so the factor-2 assertion below has room
/// but the 2D figure must not be assumed in 3D.
#[test]
fn builtin_cfl_step_is_inside_the_stable_region_in_3d() {
    for subdiv in [2usize, 3, 5] {
        let r = run(3, subdiv, 2000, 1.0);
        assert!(
            r.finite,
            "3D subdiv {subdiv}: the built-in CFL step is unstable"
        );
        assert!(
            r.excursion < 0.1,
            "3D subdiv {subdiv}: energy excursion {:e} at the built-in step",
            r.excursion
        );
        let margin = run(3, subdiv, 2000, 2.0);
        assert!(
            margin.finite,
            "3D subdiv {subdiv}: no factor-2 margin above the built-in CFL step"
        );
    }
}
