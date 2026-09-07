//! The active force at any harmonic degree, and the recovery under it.

use cartan_core::rotor::Rotor3;
use cartan_patic::active::{active_force, active_force_general};
use cartan_patic::complex3::Complex3;
use cartan_patic::energy::{Energy, State};
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::{
    AxialApolar, BinaryIcosahedral, BinaryOctahedral, BinaryTetrahedral, SymmetryGroup,
};
use cartan_patic::recovery::recover_gradient;
use nalgebra::DMatrix;

fn energy_for<H: SymmetryGroup>() -> Energy {
    let n = H::N_AMPLITUDES;
    Energy::new::<H>(
        DMatrix::from_diagonal_element(n, n, -1.0),
        vec![0.0; n * n * n],
        DMatrix::identity(n, n),
        0.5,
    )
    .expect("coercive")
}

fn rig(n: usize) -> (Complex3, Geometry3) {
    (Complex3::cube_grid(n), Geometry3::cube_grid(n))
}

// --- recovery --------------------------------------------------------------

/// A constant field has zero gradient everywhere, boundary vertices included.
#[test]
fn recovery_of_a_constant_is_exactly_zero() {
    let (c, g) = rig(3);
    let f = vec![2.7_f64; c.n_vertices()];
    for (v, grad) in recover_gradient(&c, &g, &f).iter().enumerate() {
        for (k, gk) in grad.iter().enumerate() {
            assert!(gk.abs() < 1e-13, "vertex {v} component {k}: {gk}");
        }
    }
}

/// An affine field has a constant gradient, and every cell reproduces it, so
/// the volume-weighted average is exact at every vertex.
#[test]
fn recovery_of_an_affine_field_is_exact() {
    let (c, g) = rig(3);
    let a = [0.4_f64, -0.9, 0.25];
    let f: Vec<f64> = g
        .positions()
        .iter()
        .map(|p| 1.3 + a[0] * p[0] + a[1] * p[1] + a[2] * p[2])
        .collect();
    for (v, grad) in recover_gradient(&c, &g, &f).iter().enumerate() {
        for (k, gk) in grad.iter().enumerate() {
            assert!(
                (gk - a[k]).abs() < 1e-12,
                "vertex {v} component {k}: {gk} against {}",
                a[k]
            );
        }
    }
}

// --- the general force -----------------------------------------------------

fn varying(c: &Complex3, g: &Geometry3, n_amp: usize) -> State {
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &vec![0.8; n_amp]);
    for (v, p) in g.positions().iter().enumerate() {
        let theta = 1.9 * p[0] + 0.7 * p[2];
        let (sn, cs) = (theta / 2.0).sin_cos();
        s.rotors[v] = Rotor3 {
            w: cs,
            x: 0.0,
            y: sn,
            z: 0.0,
        };
    }
    s
}

/// At degree 2 the general path must reproduce the direct assembly, which
/// needs no recovery at all. This is what pins the tensor conversion, the
/// index ordering and the contraction order.
#[test]
fn the_general_path_reproduces_the_direct_one_at_degree_two() {
    let (c, g) = rig(3);
    let e = energy_for::<AxialApolar>();
    let s = varying(&c, &g, 1);
    let direct = active_force(&c, &g, &e, &s, 1.7).expect("degree 2 runs");
    let general = active_force_general(&c, &g, &e, &s, 1.7).expect("degree 2 runs");
    let scale = direct.amax().max(1e-30);
    assert!(direct.amax() > 1e-6, "the fixture must drive a force");
    assert!(
        (&general - &direct).amax() < 1e-9 * scale,
        "general and direct differ by {:e} against a scale of {scale:e}",
        (&general - &direct).amax()
    );
}

/// A uniform field has no derivatives, so the force vanishes at every degree.
/// A term without its full derivative count would survive this.
#[test]
fn a_uniform_field_drives_no_force_at_any_degree() {
    let (c, g) = rig(2);
    macro_rules! check {
        ($h:ty, $name:literal) => {{
            let e = energy_for::<$h>();
            let n = <$h as SymmetryGroup>::N_AMPLITUDES;
            let s = State::uniform(
                c.n_vertices(),
                Rotor3 {
                    w: 0.6,
                    x: 0.8,
                    y: 0.0,
                    z: 0.0,
                },
                &vec![0.7; n],
            );
            let f = active_force_general(&c, &g, &e, &s, 1.3).expect("runs");
            assert!(
                f.amax() < 1e-11,
                "{} (degree {}): uniform field drove {:e}",
                $name,
                e.basis().degree(),
                f.amax()
            );
        }};
    }
    check!(AxialApolar, "uniaxial");
    check!(BinaryTetrahedral, "tetrahedral");
    check!(BinaryOctahedral, "cubatic");
    check!(BinaryIcosahedral, "icosahedral");
}

/// Degree 6 runs and drives a force, which is the case the crate previously
/// refused.
#[test]
fn the_icosahedral_phase_is_driven() {
    let (c, g) = rig(3);
    let e = energy_for::<BinaryIcosahedral>();
    assert_eq!(e.basis().degree(), 6);
    let s = varying(&c, &g, <BinaryIcosahedral as SymmetryGroup>::N_AMPLITUDES);
    let f = active_force_general(&c, &g, &e, &s, 1.0).expect("degree 6 runs");
    assert!(
        f.amax() > 1e-12,
        "a varying icosahedral texture must drive a force"
    );
    assert!(f.iter().all(|v| v.is_finite()));
}

#[test]
fn the_general_force_is_linear_in_the_activity() {
    let (c, g) = rig(2);
    let e = energy_for::<BinaryOctahedral>();
    let s = varying(&c, &g, <BinaryOctahedral as SymmetryGroup>::N_AMPLITUDES);
    let f1 = active_force_general(&c, &g, &e, &s, 1.0).expect("runs");
    let f2 = active_force_general(&c, &g, &e, &s, -2.5).expect("runs");
    assert!(f1.amax() > 1e-12);
    assert!(
        (&f2 + &f1 * 2.5).amax() < 1e-12 * f1.amax(),
        "the force is not linear in the activity"
    );
}

/// The stress is driven by the whole tensor, so an order parameter of zero
/// amplitude drives nothing whatever the frame does.
#[test]
fn a_melted_order_parameter_drives_no_force() {
    let (c, g) = rig(2);
    let e = energy_for::<BinaryTetrahedral>();
    let mut s = varying(&c, &g, <BinaryTetrahedral as SymmetryGroup>::N_AMPLITUDES);
    for a in s.amplitudes.iter_mut() {
        *a = 0.0;
    }
    let f = active_force_general(&c, &g, &e, &s, 1.0).expect("runs");
    assert!(f.amax() < 1e-14, "melted order drove {:e}", f.amax());
}

// --- end to end at high degree, and the attracting structure ----------------

mod coupled {
    use super::*;
    use cartan_patic::selection::{SweepDomain, SweepRun, basin_count, state_distance};
    use cartan_patic::simulation::Simulation;
    use cartan_patic::spin::Incidence;

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

    /// The whole coupled loop at degree 6, the case that previously errored.
    #[test]
    fn the_coupled_loop_runs_at_degree_six() {
        let (c, g) = rig(2);
        let inc = incidence_of(&c);
        let e = energy_for::<BinaryIcosahedral>();
        assert_eq!(e.basis().degree(), 6);
        let sim = Simulation::new(&c, &g, &inc, &e, 1.0, 1.5, 1e-3);
        let mut s = varying(&c, &g, <BinaryIcosahedral as SymmetryGroup>::N_AMPLITUDES);
        for k in 0..25 {
            let r = sim
                .step::<BinaryIcosahedral>(&mut s)
                .expect("degree 6 runs");
            assert!(r.energy.is_finite(), "step {k}: energy went non-finite");
            assert!(r.norm_defect < 1e-12, "step {k}: rotor left the sphere");
        }
    }

    /// Identical states are one basin; clearly different ones are two.
    #[test]
    fn basins_are_counted_by_a_gauge_invariant_distance() {
        let (c, g) = rig(2);
        let inc = incidence_of(&c);
        let e = energy_for::<AxialApolar>();
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
            steps: 10,
        };

        let a = varying(&c, &g, 1);
        let same = vec![a.clone(), a.clone(), a.clone()];
        assert_eq!(
            basin_count::<AxialApolar>(&domain, &same, 0.0, run, 1e-6).expect("runs"),
            1
        );

        let mut b = a.clone();
        for amp in b.amplitudes.iter_mut() {
            *amp = 0.2;
        }
        assert!(
            state_distance(&e, &a, &b) > 1e-3,
            "the fixture states must differ"
        );
        let two = vec![a, b];
        assert_eq!(
            basin_count::<AxialApolar>(&domain, &two, 0.0, run, 1e-6).expect("runs"),
            2
        );
    }

    /// A rotor and the same rotor moved by a group element are the same state,
    /// so the distance between them is zero.
    #[test]
    fn the_state_distance_ignores_the_gauge() {
        let (c, g) = rig(2);
        let e = energy_for::<AxialApolar>();
        let a = varying(&c, &g, 1);
        let mut b = a.clone();
        // The flip through the axis leaves the uniaxial tensor alone.
        let flip = Rotor3 {
            w: 0.0,
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        for r in b.rotors.iter_mut() {
            *r = r.compose(&flip);
        }
        assert!(
            state_distance(&e, &a, &b) < 1e-12,
            "gauge motion changed the distance: {}",
            state_distance(&e, &a, &b)
        );
    }
}
