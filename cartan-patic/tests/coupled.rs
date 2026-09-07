//! The coupled loop and the control sweep, end to end.

use cartan_core::rotor::Rotor3;
use cartan_patic::boundary::Boundary;
use cartan_patic::complex3::Complex3;
use cartan_patic::defect::DefectField;
use cartan_patic::energy::{Energy, State};
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::{AxialApolar, SymmetryGroup};
use cartan_patic::selection::{SweepDomain, SweepRun, sweep, transitions};
use cartan_patic::simulation::Simulation;
use cartan_patic::spin::Incidence;
use nalgebra::{DMatrix, DVector};

fn energy_obj() -> Energy {
    let n = <AxialApolar as SymmetryGroup>::N_AMPLITUDES;
    Energy::new::<AxialApolar>(
        DMatrix::from_diagonal_element(n, n, -1.0),
        vec![0.0; n * n * n],
        DMatrix::identity(n, n),
        0.5,
    )
    .expect("coercive")
}

/// Triangle incidence over the same vertices, for the order-parameter energy.
fn incidence_of(c: &Complex3) -> Incidence {
    let mut tris = Vec::new();
    for t in 0..c.n_tets() {
        for f in c.tet_triangles(t) {
            let tri = c.triangle(f);
            if !tris.contains(&tri) {
                tris.push(tri);
            }
        }
    }
    Incidence::from_triangles(c.n_vertices(), &tris)
}

fn twisted(c: &Complex3, g: &Geometry3) -> State {
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
    for (v, p) in g.positions().iter().enumerate() {
        let theta = 2.0 * p[1];
        let (sn, cs) = (theta / 2.0).sin_cos();
        s.rotors[v] = Rotor3 {
            w: cs,
            x: sn,
            y: 0.0,
            z: 0.0,
        };
    }
    s
}

/// A rigid rotation turns the director at half its vorticity. This is the
/// exact statement of the co-rotational coupling and it fixes the factor.
#[test]
fn a_rigid_rotation_turns_the_director_at_half_the_vorticity() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let inc = incidence_of(&c);
    let e = energy_obj();
    let sim = Simulation::new(&c, &g, &inc, &e, 1.0, 0.0, 1.0);

    // Rigid rotation about z at rate 1: u = (-y, x, 0), vorticity (0, 0, 2).
    let p = g.positions();
    let u = DVector::from_iterator(
        c.n_edges(),
        c.edges().iter().map(|ed| {
            let f = |q: &[f64; 3]| [-(q[1] - 0.5), q[0] - 0.5, 0.0];
            let a = f(&p[ed[0]]);
            let b = f(&p[ed[1]]);
            let d = [
                p[ed[1]][0] - p[ed[0]][0],
                p[ed[1]][1] - p[ed[0]][1],
                p[ed[1]][2] - p[ed[0]][2],
            ];
            let m = [
                (a[0] + b[0]) / 2.0,
                (a[1] + b[1]) / 2.0,
                (a[2] + b[2]) / 2.0,
            ];
            m[0] * d[0] + m[1] * d[1] + m[2] * d[2]
        }),
    );
    let w = sim.vorticity(&u);
    for (v, wv) in w.iter().enumerate() {
        assert!(
            (wv[2] - 2.0).abs() < 1e-9 && wv[0].abs() < 1e-9 && wv[1].abs() < 1e-9,
            "vertex {v}: vorticity {wv:?}, expected (0, 0, 2)"
        );
    }

    // Co-rotating for time dt turns the frame by dt * omega / 2 = dt.
    let dt = 0.1;
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
    sim.corotate(&mut s, &w, dt);
    let got = s.rotors[0];
    let (sn, cs) = (dt / 2.0).sin_cos();
    let want = Rotor3 {
        w: cs,
        x: 0.0,
        y: 0.0,
        z: sn,
    };
    for (a, b) in [
        (got.w, want.w),
        (got.x, want.x),
        (got.y, want.y),
        (got.z, want.z),
    ] {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

/// With no activity the loop is pure gradient flow, so the energy falls.
#[test]
fn zero_activity_reduces_to_gradient_flow() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let inc = incidence_of(&c);
    let e = energy_obj();
    let sim = Simulation::new(&c, &g, &inc, &e, 1.0, 0.0, 1e-3);
    let mut s = twisted(&c, &g);
    let mut last = e.total(&inc, &s);
    for k in 0..300 {
        let r = sim.step(&mut s).expect("degree 2 runs");
        assert!(
            r.speed < 1e-12,
            "step {k}: zero activity produced flow {}",
            r.speed
        );
        assert!(r.energy <= last + 1e-9, "step {k}: energy rose");
        last = r.energy;
    }
    assert!(s.worst_norm_defect() < 1e-13);
}

/// The active force is linear in the activity, so the velocity is too and the
/// dissipation scales as the square. Both signs give the same dissipation.
#[test]
fn dissipation_scales_as_the_square_of_the_activity() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let inc = incidence_of(&c);
    let e = energy_obj();
    let s = twisted(&c, &g);

    let d = |zeta: f64| {
        let sim = Simulation::new(&c, &g, &inc, &e, 1.0, zeta, 1e-4);
        let u = sim.velocity(&s).expect("degree 2 runs");
        let st = cartan_patic::stokes::Stokes::assemble(&c, &g, 1.0);
        st.dissipation(&u)
    };
    let d1 = d(1.0);
    let d2 = d(2.0);
    let dm = d(-1.0);
    assert!(d1 > 1e-12, "activity must dissipate");
    assert!(
        (d2 / d1 - 4.0).abs() < 1e-8,
        "doubling activity gave a factor {}",
        d2 / d1
    );
    assert!(
        (dm / d1 - 1.0).abs() < 1e-9,
        "sign of activity changed dissipation"
    );
}

/// The whole stack in one run: complex, geometry, boundary, order parameter,
/// active force, flow, and defect detection.
#[test]
fn the_full_pipeline_runs_with_no_slip() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let b = Boundary::extract(&c, &g);
    let inc = incidence_of(&c);
    let e = energy_obj();
    let sim = Simulation::new(&c, &g, &inc, &e, 1.0, 2.0, 1e-3).with_no_slip(b.edges());
    let mut s = twisted(&c, &g);

    for k in 0..100 {
        let r = sim.step(&mut s).expect("degree 2 runs");
        assert!(r.energy.is_finite(), "step {k}: energy went non-finite");
        assert!(r.norm_defect < 1e-12, "step {k}: rotor left the sphere");
    }
    let u = sim.velocity(&s).expect("degree 2 runs");
    for &edge in b.edges() {
        assert!(u[edge].abs() < 1e-12, "no-slip violated on edge {edge}");
    }
    let d = DefectField::detect::<AxialApolar>(&c, &s.rotors, 1e-8);
    assert_eq!(
        d.worst_parity_violation(),
        0,
        "the evolved field must still conserve defect lines"
    );
}

/// A sweep over activity, with the observables it produces.
#[test]
fn a_control_sweep_responds_to_activity() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let inc = incidence_of(&c);
    let e = energy_obj();
    let s = twisted(&c, &g);
    let zetas = [0.0, 0.5, 1.0, 2.0, 4.0];
    let domain = SweepDomain {
        complex: &c,
        geometry: &g,
        incidence: &inc,
        energy: &e,
        no_slip: &[],
    };
    let run = SweepRun {
        eta: 1.0,
        dt: 1e-3,
        steps: 40,
    };
    let pts = sweep(&domain, &s, &zetas, run).expect("sweep runs");

    assert_eq!(pts.len(), zetas.len());
    assert!(pts[0].speed < 1e-12, "zero activity must not flow");
    for w in pts.windows(2) {
        assert!(
            w[1].dissipation >= w[0].dissipation - 1e-9,
            "dissipation fell as activity rose: {:e} -> {:e}",
            w[0].dissipation,
            w[1].dissipation
        );
    }
    for p in &pts {
        assert!(
            p.worst_norm_defect < 1e-12,
            "zeta {}: left the sphere",
            p.zeta
        );
    }
    // A smooth response has no jump larger than a third of the range.
    let jumps = transitions(&pts, |p| p.speed, 0.9);
    assert!(jumps.is_empty(), "unexpected jump at {jumps:?}");
}
