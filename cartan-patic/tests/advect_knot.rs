//! Semi-Lagrangian advection and knot invariants.

use cartan_core::rotor::Rotor3;
use cartan_patic::advect::{advect, vertex_velocity};
use cartan_patic::complex3::Complex3;
use cartan_patic::energy::State;
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::AxialApolar;
use cartan_patic::knot::{linking_number, writhe};

// --- advection -------------------------------------------------------------

fn uniform_flow(c: &Complex3, g: &Geometry3, v: [f64; 3]) -> Vec<f64> {
    let p = g.positions();
    c.edges()
        .iter()
        .map(|e| {
            let d = [
                p[e[1]][0] - p[e[0]][0],
                p[e[1]][1] - p[e[0]][1],
                p[e[1]][2] - p[e[0]][2],
            ];
            v[0] * d[0] + v[1] * d[1] + v[2] * d[2]
        })
        .collect()
}

#[test]
fn a_uniform_cochain_reconstructs_the_velocity() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let v = [0.3, -0.7, 0.5];
    let u = uniform_flow(&c, &g, v);
    for (i, w) in vertex_velocity(&c, &g, &u).iter().enumerate() {
        for k in 0..3 {
            assert!(
                (w[k] - v[k]).abs() < 1e-10,
                "vertex {i} component {k}: {} against {}",
                w[k],
                v[k]
            );
        }
    }
}

#[test]
fn zero_velocity_leaves_the_state_alone() {
    let c = Complex3::cube_grid(2);
    let g = Geometry3::cube_grid(2);
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.7]);
    for (v, p) in g.positions().iter().enumerate() {
        let t = 1.3 * p[0];
        let (sn, cs) = (t / 2.0).sin_cos();
        s.rotors[v] = Rotor3 {
            w: cs,
            x: sn,
            y: 0.0,
            z: 0.0,
        };
        s.amplitudes[v] = 0.5 + 0.3 * p[1];
    }
    let u = vec![0.0; c.n_edges()];
    let out = advect::<AxialApolar>(&c, &g, &s, &u, 0.1);
    for v in 0..c.n_vertices() {
        assert!((out.amplitudes[v] - s.amplitudes[v]).abs() < 1e-12);
        let (a, b) = (out.rotors[v], s.rotors[v]);
        let d =
            (a.w - b.w).powi(2) + (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2);
        assert!(d < 1e-24, "vertex {v} moved under zero flow");
    }
}

/// Barycentric interpolation is exact on affine data, so advecting a field
/// linear in space by a uniform velocity reproduces it shifted, with no
/// numerical diffusion at all. Any error in the departure point, the point
/// location or the weights breaks this.
#[test]
fn a_linear_field_advects_exactly_under_uniform_flow() {
    let c = Complex3::cube_grid(4);
    let g = Geometry3::cube_grid(4);
    let p = g.positions();
    let field = |q: &[f64; 3]| 0.4 + 0.25 * q[0] - 0.15 * q[1] + 0.1 * q[2];

    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.0]);
    for (v, q) in p.iter().enumerate() {
        s.amplitudes[v] = field(q);
    }
    let vel = [0.2, 0.1, -0.15];
    let dt = 0.05;
    let u = uniform_flow(&c, &g, vel);
    let out = advect::<AxialApolar>(&c, &g, &s, &u, dt);

    let mut checked = 0;
    for (v, q) in p.iter().enumerate() {
        let x = [q[0] - dt * vel[0], q[1] - dt * vel[1], q[2] - dt * vel[2]];
        // Only interior departure points are inside the domain.
        if x.iter().any(|&q| q < 1e-9 || q > 1.0 - 1e-9) {
            continue;
        }
        checked += 1;
        assert!(
            (out.amplitudes[v] - field(&x)).abs() < 1e-11,
            "vertex {v}: got {}, exact {}",
            out.amplitudes[v],
            field(&x)
        );
    }
    assert!(checked > 20, "the fixture must exercise interior vertices");
}

#[test]
fn advection_keeps_rotors_on_the_sphere() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let mut s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.7]);
    for (v, p) in g.positions().iter().enumerate() {
        let t = 2.0 * p[2];
        let (sn, cs) = (t / 2.0).sin_cos();
        s.rotors[v] = Rotor3 {
            w: cs,
            x: 0.0,
            y: sn,
            z: 0.0,
        };
    }
    let u = uniform_flow(&c, &g, [0.1, 0.05, 0.0]);
    let mut cur = s;
    for _ in 0..20 {
        cur = advect::<AxialApolar>(&c, &g, &cur, &u, 0.01);
    }
    assert!(
        cur.worst_norm_defect() < 1e-13,
        "rotor norm drifted by {:e}",
        cur.worst_norm_defect()
    );
}

// --- knots -----------------------------------------------------------------

fn circle(centre: [f64; 3], radius: f64, normal: usize, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let t = 2.0 * core::f64::consts::PI * i as f64 / n as f64;
            let (s, c) = t.sin_cos();
            let mut p = centre;
            let (a, b) = match normal {
                0 => (1, 2),
                1 => (2, 0),
                _ => (0, 1),
            };
            p[a] += radius * c;
            p[b] += radius * s;
            p
        })
        .collect()
}

#[test]
fn two_distant_circles_are_unlinked() {
    let a = circle([0.0, 0.0, 0.0], 1.0, 2, 64);
    let b = circle([10.0, 0.0, 0.0], 1.0, 2, 64);
    let lk = linking_number(&a, &b);
    assert!(lk.abs() < 1e-6, "linking number {lk}");
}

/// The Hopf link: two circles in perpendicular planes, each through the
/// other's disc. Linking number is one.
#[test]
fn the_hopf_link_has_linking_number_one() {
    let a = circle([0.0, 0.0, 0.0], 1.0, 2, 128);
    let b = circle([1.0, 0.0, 0.0], 1.0, 1, 128);
    let lk = linking_number(&a, &b);
    assert!(
        (lk.abs() - 1.0).abs() < 1e-6,
        "linking number {lk}, expected +/-1"
    );
}

#[test]
fn linking_is_symmetric_and_flips_with_orientation() {
    let a = circle([0.0, 0.0, 0.0], 1.0, 2, 96);
    let b = circle([1.0, 0.0, 0.0], 1.0, 1, 96);
    let ab = linking_number(&a, &b);
    let ba = linking_number(&b, &a);
    assert!((ab - ba).abs() < 1e-9, "asymmetric: {ab} against {ba}");
    let mut rev = b.clone();
    rev.reverse();
    let flipped = linking_number(&a, &rev);
    assert!(
        (flipped + ab).abs() < 1e-9,
        "reversal gave {flipped} against {ab}"
    );
}

#[test]
fn a_planar_circle_has_no_writhe() {
    let a = circle([0.0, 0.0, 0.0], 1.0, 2, 96);
    let w = writhe(&a);
    assert!(w.abs() < 1e-6, "planar writhe {w}");
}

/// Doubling the linking: a circle threaded twice by a longer loop is not
/// tested here, but scaling both curves must leave the invariant alone.
#[test]
fn linking_is_scale_invariant() {
    let a = circle([0.0, 0.0, 0.0], 1.0, 2, 96);
    let b = circle([1.0, 0.0, 0.0], 1.0, 1, 96);
    let base = linking_number(&a, &b);
    let sa: Vec<[f64; 3]> = a
        .iter()
        .map(|p| [3.0 * p[0], 3.0 * p[1], 3.0 * p[2]])
        .collect();
    let sb: Vec<[f64; 3]> = b
        .iter()
        .map(|p| [3.0 * p[0], 3.0 * p[1], 3.0 * p[2]])
        .collect();
    let scaled = linking_number(&sa, &sb);
    assert!((scaled - base).abs() < 1e-9, "{scaled} against {base}");
}
