//! Disclination detection, charge, and the conservation law.

use cartan_core::rotor::Rotor3;
use cartan_patic::complex3::Complex3;
use cartan_patic::defect::{DefectField, edge_transition, is_identity};
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::{AxialApolar, AxialPolar, Dicyclic, SymmetryGroup};

const TOL: f64 = 1e-8;

/// A straight `+1/2` disclination along the line `x = y = 0.5`.
///
/// The director is `n(phi) = (cos(phi/2), sin(phi/2), 0)`. The reference
/// director of the invariant construction is `e_z`, so the rotor has to carry
/// `e_z` to `n`: a quarter turn about the axis perpendicular to both, which is
/// `(-sin theta, cos theta, 0)` with `theta = phi/2`.
///
/// A rotor about `z` would leave the director at `e_z` everywhere, which is a
/// uniform field with no defect at all. That was the first version of this
/// fixture and the detector was right to find nothing in it.
fn half_disclination(g: &Geometry3) -> Vec<Rotor3> {
    let r = core::f64::consts::FRAC_1_SQRT_2;
    g.positions()
        .iter()
        .map(|p| {
            let phi = (p[1] - 0.5).atan2(p[0] - 0.5);
            let theta = phi / 2.0;
            let (st, ct) = theta.sin_cos();
            // Half-angle pi/4 about the axis (-sin theta, cos theta, 0).
            Rotor3 {
                w: r,
                x: -r * st,
                y: r * ct,
                z: 0.0,
            }
        })
        .collect()
}

fn uniform(n: usize) -> Vec<Rotor3> {
    vec![
        Rotor3 {
            w: 0.6,
            x: 0.8,
            y: 0.0,
            z: 0.0
        };
        n
    ]
}

#[test]
fn a_uniform_field_has_no_defects() {
    let c = Complex3::cube_grid(3);
    let d = DefectField::detect::<AxialApolar>(&c, &uniform(c.n_vertices()), TOL);
    assert!(
        d.pierced().is_empty(),
        "{} faces pierced",
        d.pierced().len()
    );
}

/// A line has no endpoints, so a line entering a tetrahedron leaves it.
#[test]
fn every_tetrahedron_has_an_even_number_of_pierced_faces() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let d = DefectField::detect::<AxialApolar>(&c, &half_disclination(&g), TOL);
    assert!(!d.pierced().is_empty(), "the seeded line must be detected");
    assert_eq!(
        d.worst_parity_violation(),
        0,
        "{} tetrahedra have an odd number of pierced faces",
        d.worst_parity_violation()
    );
}

#[test]
fn the_seeded_line_is_one_connected_component() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let d = DefectField::detect::<AxialApolar>(&c, &half_disclination(&g), TOL);
    let lines = d.lines(&c);
    assert_eq!(lines.len(), 1, "expected one line, found {}", lines.len());
    assert_eq!(
        lines[0].len(),
        d.pierced().len(),
        "the component must contain every pierced face"
    );
}

/// Rotating the whole field conjugates every holonomy uniformly, so the set of
/// pierced faces is unchanged.
#[test]
fn a_global_rotation_leaves_the_detection_unchanged() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let base = half_disclination(&g);
    let before = DefectField::detect::<AxialApolar>(&c, &base, TOL);

    let q = Rotor3 {
        w: 0.5,
        x: 0.5,
        y: 0.5,
        z: 0.5,
    };
    let rotated: Vec<Rotor3> = base.iter().map(|r| q.compose(r)).collect();
    let after = DefectField::detect::<AxialApolar>(&c, &rotated, TOL);

    let mut a: Vec<usize> = before.pierced().iter().map(|p| p.triangle).collect();
    let mut b: Vec<usize> = after.pierced().iter().map(|p| p.triangle).collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b, "a global rotation changed which faces are pierced");
}

/// The same texture at a different overall scale gives the same detection,
/// which no order-parameter threshold can promise.
#[test]
fn detection_is_threshold_free() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let base = half_disclination(&g);
    let a = DefectField::detect::<AxialApolar>(&c, &base, 1e-8);
    let b = DefectField::detect::<AxialApolar>(&c, &base, 1e-3);
    assert_eq!(a.pierced().len(), b.pierced().len());
}

/// `pi_1(S^2) = 0`, so a polar phase admits no line defects whatever the
/// texture.
#[test]
fn a_polar_phase_has_no_line_defects() {
    let c = Complex3::cube_grid(3);
    let g = Geometry3::cube_grid(3);
    let d = DefectField::detect::<AxialPolar>(&c, &half_disclination(&g), TOL);
    assert!(d.pierced().is_empty(), "S^2 has trivial pi_1");
}

/// `Dicyclic<2>` is `Q_8`. Its commutators are `-1`, which is a genuine
/// non-trivial class there, so charges cannot be numbers and fusion cannot be
/// addition. Identifying `-1` with `1`, as the uniaxial case requires, would
/// hide exactly this.
#[test]
fn the_biaxial_charge_group_is_non_abelian() {
    let modulo_sign = <Dicyclic<2> as SymmetryGroup>::CHARGE_MODULO_SIGN;
    assert!(!modulo_sign, "-1 is a genuine class in Q_8");
    let g: Vec<Rotor3> = <Dicyclic<2> as SymmetryGroup>::defect_group().collect();
    assert_eq!(g.len(), 8, "Q_8 has eight elements");
    let mut found = false;
    for a in &g {
        for b in &g {
            let comm = a.compose(b).reverse().compose(&b.compose(a));
            if !is_identity(&comm, 1e-9) {
                found = true;
            }
        }
    }
    assert!(found, "Q_8 must contain a non-commuting pair");
}

/// The uniaxial nematic has two charges, not four: `-1` is contractible in
/// `RP^2`.
#[test]
fn the_uniaxial_charge_group_identifies_the_sign() {
    let uniaxial = <AxialApolar as SymmetryGroup>::CHARGE_MODULO_SIGN;
    let dicyclic = <Dicyclic<3> as SymmetryGroup>::CHARGE_MODULO_SIGN;
    assert!(uniaxial, "RP^2 contracts the 2 pi rotation");
    assert!(!dicyclic, "a finite lift keeps -1 as a class");
}

/// The edge transition is the group element that best aligns the two ends, so
/// applying it makes the residual no larger than the identity would.
#[test]
fn the_edge_transition_minimises_the_coset_distance() {
    let ru = Rotor3 {
        w: 0.6,
        x: 0.8,
        y: 0.0,
        z: 0.0,
    };
    let rv = Rotor3 {
        w: 0.0,
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    let h = edge_transition::<Dicyclic<2>>(&ru, &rv);
    let with = {
        let x = rv.compose(&h);
        (ru.w - x.w).powi(2) + (ru.x - x.x).powi(2) + (ru.y - x.y).powi(2) + (ru.z - x.z).powi(2)
    };
    for g in <Dicyclic<2> as SymmetryGroup>::defect_group() {
        let x = rv.compose(&g);
        let d = (ru.w - x.w).powi(2)
            + (ru.x - x.x).powi(2)
            + (ru.y - x.y).powi(2)
            + (ru.z - x.z).powi(2);
        assert!(
            with <= d + 1e-12,
            "a better transition exists: {with:e} vs {d:e}"
        );
    }
}
