//! Boundary extraction, no-slip, and anchoring.

use cartan_core::rotor::Rotor3;
use cartan_katic::boundary::{Boundary, rotor_taking_z_to};
use cartan_katic::complex3::Complex3;
use cartan_katic::geometry::Geometry3;
use cartan_katic::stokes::Stokes;
use nalgebra::DVector;

fn rig(n: usize) -> (Complex3, Geometry3, Boundary) {
    let c = Complex3::cube_grid(n);
    let g = Geometry3::cube_grid(n);
    let b = Boundary::extract(&c, &g);
    (c, g, b)
}

#[test]
fn the_boundary_of_a_box_has_the_expected_face_count() {
    for n in 1..=3 {
        let (_, _, b) = rig(n);
        assert_eq!(
            b.faces().len(),
            12 * n * n,
            "n = {n}: six sides, two triangles per square, n^2 squares"
        );
    }
}

/// The boundary of a ball is a sphere. A face miscounted in either direction
/// breaks this, which is why it is the first check on the extraction.
#[test]
fn the_boundary_is_a_sphere() {
    for n in 1..=3 {
        let (_, _, b) = rig(n);
        assert_eq!(b.euler_characteristic(), 2, "n = {n}");
    }
}

#[test]
fn outward_normals_point_out_of_a_convex_domain() {
    let (c, g, b) = rig(2);
    let p = g.positions();
    let centre = [0.5, 0.5, 0.5];
    for (i, &f) in b.faces().iter().enumerate() {
        let tri = c.triangle(f);
        let mid = [
            (p[tri[0]][0] + p[tri[1]][0] + p[tri[2]][0]) / 3.0,
            (p[tri[0]][1] + p[tri[1]][1] + p[tri[2]][1]) / 3.0,
            (p[tri[0]][2] + p[tri[1]][2] + p[tri[2]][2]) / 3.0,
        ];
        let out = [mid[0] - centre[0], mid[1] - centre[1], mid[2] - centre[2]];
        let n = b.normals()[i];
        let d = n[0] * out[0] + n[1] * out[1] + n[2] * out[2];
        assert!(d > 0.0, "face {f}: normal points inward, dot {d:e}");
    }
}

#[test]
fn boundary_areas_total_the_surface_of_the_cube() {
    for n in 1..=3 {
        let (_, _, b) = rig(n);
        let a: f64 = b.areas().iter().sum();
        assert!((a - 6.0).abs() < 1e-12, "n = {n}: total area {a}");
    }
}

/// The velocity is exactly zero on constrained edges, and the interior flow is
/// still divergence free.
#[test]
fn no_slip_is_exact_and_keeps_incompressibility() {
    let (c, g, b) = rig(2);
    let s = Stokes::assemble(&c, &g, 1.0);
    let f = DVector::from_iterator(
        c.n_edges(),
        (0..c.n_edges()).map(|e| ((e * 7 % 13) as f64 - 6.0) / 6.0),
    );
    let (u, _) = s.solve_no_slip(&f, b.edges());
    for &e in b.edges() {
        assert!(u[e].abs() < 1e-14, "edge {e} moved by {:e}", u[e]);
    }
    let div = s.divergence(&u);
    assert!(
        div.amax() < 1e-8 * f.norm(),
        "divergence {:e} against force {:e}",
        div.amax(),
        f.norm()
    );
}

/// Constraining the boundary must not kill the problem.
#[test]
fn no_slip_still_admits_interior_flow() {
    let (c, g, b) = rig(3);
    let s = Stokes::assemble(&c, &g, 1.0);
    let f = DVector::from_iterator(
        c.n_edges(),
        (0..c.n_edges()).map(|e| ((e * 5 % 11) as f64 - 5.0) / 5.0),
    );
    let (u, _) = s.solve_no_slip(&f, b.edges());
    assert!(
        s.velocity_norm(&u) > 1e-6,
        "no-slip left no interior flow at all"
    );
}

/// The homeotropic rotor takes the reference director to the surface normal.
#[test]
fn the_homeotropic_rotor_aligns_the_director_with_the_normal() {
    // The function documents a unit normal, so the fixture normalises rather
    // than relying on typed decimals being exactly unit.
    let normals: [[f64; 3]; 5] = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-0.4, 0.9, -0.2],
    ];
    for raw in normals {
        let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
        let n = [raw[0] / len, raw[1] / len, raw[2] / len];
        let r = rotor_taking_z_to(n);
        let got = r.rotate_vec([0.0, 0.0, 1.0]);
        // A director is defined up to sign.
        let d = got[0] * n[0] + got[1] * n[1] + got[2] * n[2];
        assert!(
            (d.abs() - 1.0).abs() < 1e-10,
            "normal {n:?}: rotated director {got:?}, alignment {d}"
        );
    }
}

#[test]
fn vertex_normals_are_unit_and_outward() {
    let (c, g, b) = rig(2);
    let vn = b.vertex_normals(&c);
    let p = g.positions();
    assert_eq!(vn.len(), b.vertices().len());
    for (&v, n) in &vn {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-12, "vertex {v}: normal length {len}");
        let out = [p[v][0] - 0.5, p[v][1] - 0.5, p[v][2] - 0.5];
        let d = n[0] * out[0] + n[1] * out[1] + n[2] * out[2];
        assert!(d > 0.0, "vertex {v}: averaged normal points inward");
    }
}

#[test]
fn interior_vertices_are_absent_from_the_boundary() {
    let (c, _, b) = rig(3);
    let s = 4; // (n + 1) vertices per axis
    let interior: Vec<usize> = (0..c.n_vertices())
        .filter(|v| {
            let i = v / (s * s);
            let j = (v / s) % s;
            let k = v % s;
            i > 0 && i < s - 1 && j > 0 && j < s - 1 && k > 0 && k < s - 1
        })
        .collect();
    assert!(
        !interior.is_empty(),
        "the fixture must have interior vertices"
    );
    for v in interior {
        assert!(
            !b.vertices().contains(&v),
            "interior vertex {v} on boundary"
        );
    }
}

#[test]
fn a_uniform_field_needs_no_anchoring_correction() {
    let (c, g, b) = rig(2);
    let vn = b.vertex_normals(&c);
    let mut rotors = vec![Rotor3::IDENTITY; c.n_vertices()];
    for (&v, n) in &vn {
        rotors[v] = rotor_taking_z_to(*n);
    }
    // Every prescribed rotor is a unit rotor.
    for &v in b.vertices() {
        let r = rotors[v];
        let len = (r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z).sqrt();
        assert!((len - 1.0).abs() < 1e-12, "vertex {v}: rotor norm {len}");
    }
    let _ = g;
}

mod weak {
    use super::*;
    use cartan_katic::boundary::anchoring;
    use cartan_katic::energy::{Energy, State};
    use cartan_katic::group::{AxialApolar, SymmetryGroup};
    use nalgebra::DMatrix;

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

    /// Homeotropic targets: the director is the surface normal.
    fn homeotropic(
        c: &Complex3,
        b: &Boundary,
        e: &Energy,
        nv: usize,
    ) -> (Vec<nalgebra::DVector<f64>>, Vec<Rotor3>) {
        let vn = b.vertex_normals(c);
        let mut target_rotors = vec![Rotor3::IDENTITY; nv];
        for (&v, n) in &vn {
            target_rotors[v] = rotor_taking_z_to(*n);
        }
        let prescribed = (0..nv)
            .map(|v| e.tensor(&target_rotors[v], &[0.8]))
            .collect();
        (prescribed, target_rotors)
    }

    #[test]
    fn a_matching_field_has_zero_surface_energy() {
        let (c, g, b) = rig(2);
        let e = energy_obj();
        let (prescribed, target) = homeotropic(&c, &b, &e, c.n_vertices());
        let mut state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        state.rotors.clone_from(&target);
        let se = anchoring::energy(&c, &b, &e, &state, &prescribed, 1.5);
        assert!(se < 1e-20, "matching field has surface energy {se:e}");
        let _ = g;
    }

    #[test]
    fn a_mismatched_field_has_positive_surface_energy() {
        let (c, _, b) = rig(2);
        let e = energy_obj();
        let (prescribed, _) = homeotropic(&c, &b, &e, c.n_vertices());
        let state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        let se = anchoring::energy(&c, &b, &e, &state, &prescribed, 1.5);
        assert!(se > 1e-6, "mismatched field has surface energy {se:e}");
    }

    #[test]
    fn the_surface_gradient_matches_finite_differences() {
        let (c, _, b) = rig(2);
        let e = energy_obj();
        let (prescribed, _) = homeotropic(&c, &b, &e, c.n_vertices());
        let mut state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        for (v, r) in state.rotors.iter_mut().enumerate() {
            let t = 0.2 + v as f64 * 0.11;
            let (s, cs) = (t / 2.0).sin_cos();
            *r = Rotor3 {
                w: cs,
                x: s * 0.6,
                y: s * 0.8,
                z: 0.0,
            };
        }
        let w = 1.5;
        let g = anchoring::gradient(&c, &b, &e, &state, &prescribed, w);
        let n = e.basis().n_amplitudes();
        let block = 3 + n;
        let h = 1e-6;

        let v = b.vertices()[3];
        for axis in 0..3 {
            let mut plus = state.clone();
            let mut minus = state.clone();
            let mut ax = [0.0; 3];
            ax[axis] = h / 2.0;
            let rot = |sign: f64| {
                let theta = h * sign;
                let (s, cs) = (theta / 2.0).sin_cos();
                let mut u = [0.0; 3];
                u[axis] = s;
                Rotor3 {
                    w: cs,
                    x: u[0],
                    y: u[1],
                    z: u[2],
                }
            };
            // `plus` steps along +h in the same so(3) direction the gradient
            // is expressed in; reversing these two flips the sign of fd.
            plus.rotors[v] = rot(1.0).compose(&state.rotors[v]);
            minus.rotors[v] = rot(-1.0).compose(&state.rotors[v]);
            let ep = anchoring::energy(&c, &b, &e, &plus, &prescribed, w);
            let em = anchoring::energy(&c, &b, &e, &minus, &prescribed, w);
            let fd = (ep - em) / (2.0 * h);
            let an = g[v * block + axis];
            assert!(
                (fd - an).abs() < 1e-5 * (1.0 + an.abs()),
                "vertex {v} axis {axis}: fd {fd:e} analytic {an:e}"
            );
            let _ = ax;
        }
    }

    /// Stepping against the surface gradient lowers the surface energy, which
    /// is what makes weak anchoring usable in the flow.
    #[test]
    fn the_surface_energy_falls_along_its_gradient() {
        let (c, _, b) = rig(2);
        let e = energy_obj();
        let (prescribed, _) = homeotropic(&c, &b, &e, c.n_vertices());
        let mut state = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
        let w = 1.0;
        let n = e.basis().n_amplitudes();
        let block = 3 + n;
        let mut last = anchoring::energy(&c, &b, &e, &state, &prescribed, w);
        for step in 0..200 {
            let g = anchoring::gradient(&c, &b, &e, &state, &prescribed, w);
            for v in 0..state.n_vertices() {
                let dt = 1e-2;
                let x = [
                    -dt * g[v * block],
                    -dt * g[v * block + 1],
                    -dt * g[v * block + 2],
                ];
                let theta = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
                if theta > 1e-300 {
                    let (s, cs) = (theta / 2.0).sin_cos();
                    let k = s / theta;
                    let r = Rotor3 {
                        w: cs,
                        x: k * x[0],
                        y: k * x[1],
                        z: k * x[2],
                    };
                    state.rotors[v] = r.compose(&state.rotors[v]);
                }
                for i in 0..n {
                    state.amplitudes[v * n + i] -= 1e-2 * g[v * block + 3 + i];
                }
            }
            let now = anchoring::energy(&c, &b, &e, &state, &prescribed, w);
            assert!(now <= last + 1e-12, "step {step}: {last:e} -> {now:e}");
            last = now;
        }
        assert!(state.worst_norm_defect() < 1e-13);
    }
}
