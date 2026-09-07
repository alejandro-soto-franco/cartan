//! The winding a disclination cross-section reports.

use cartan_core::rotor::Rotor3;
use cartan_patic::complex3::Complex3;
use cartan_patic::energy::State;
use cartan_patic::geometry::Geometry3;
use cartan_patic::group::AxialApolar;
use cartan_patic::profile::{Segment, profile_at};

/// A straight wedge disclination of charge `q` along z through `(0.5, 0.5)`.
///
/// The director is `n(phi) = (cos(q phi), sin(q phi), 0)`, which is the
/// standard planar profile: `q = +1/2` and `q = -1/2` are the two wedges.
fn wedge(g: &Geometry3, q: f64) -> State {
    let mut s = State::uniform(g.positions().len(), Rotor3::IDENTITY, &[0.8]);
    for (v, p) in g.positions().iter().enumerate() {
        let phi = (p[1] - 0.5).atan2(p[0] - 0.5);
        let theta = q * phi;
        let n = [theta.cos(), theta.sin(), 0.0];
        s.rotors[v] = cartan_patic::boundary::rotor_taking_z_to(n);
    }
    s
}

fn rig() -> (Complex3, Geometry3) {
    (Complex3::cube_grid(12), Geometry3::cube_grid(12))
}

/// The whole point of the module: a half-integer winding is only visible if
/// the angle is unwrapped modulo pi. Reading `+0.5` here is the check.
#[test]
fn a_plus_half_wedge_winds_by_a_half() {
    let (c, g) = rig();
    let s = wedge(&g, 0.5);
    let p = profile_at::<AxialApolar>(&c, &g, &s, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 0.2, 64)
        .expect("the circle is inside the domain");
    assert!(
        (p.winding - 0.5).abs() < 0.02,
        "winding {} should be +1/2",
        p.winding
    );
    assert!(p.twist < 0.05, "a wedge has no twist, got {}", p.twist);
    assert_eq!(p.classify(0.3), Segment::PlusHalf);
}

#[test]
fn a_minus_half_wedge_winds_the_other_way() {
    let (c, g) = rig();
    let s = wedge(&g, -0.5);
    let p = profile_at::<AxialApolar>(&c, &g, &s, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 0.2, 64)
        .expect("inside");
    assert!(
        (p.winding + 0.5).abs() < 0.02,
        "winding {} should be -1/2",
        p.winding
    );
    assert_eq!(p.classify(0.3), Segment::MinusHalf);
}

/// A uniform field has no winding at all, so a section anywhere reads zero.
#[test]
fn a_uniform_field_has_no_winding() {
    let (c, g) = rig();
    let s = State::uniform(
        c.n_vertices(),
        Rotor3 {
            w: 0.6,
            x: 0.8,
            y: 0.0,
            z: 0.0,
        },
        &[0.8],
    );
    let p = profile_at::<AxialApolar>(&c, &g, &s, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 0.2, 64)
        .expect("inside");
    assert!(
        p.winding.abs() < 1e-6,
        "uniform field wound by {}",
        p.winding
    );
}

/// A director everywhere along the line is a pure twist: no winding in the
/// section plane, and `|n . t| = 1`.
#[test]
fn a_field_along_the_line_reads_as_twist() {
    let (c, g) = rig();
    let s = State::uniform(c.n_vertices(), Rotor3::IDENTITY, &[0.8]);
    let p = profile_at::<AxialApolar>(&c, &g, &s, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], 0.2, 64)
        .expect("inside");
    assert!(
        (p.twist - 1.0).abs() < 1e-9,
        "twist {} should be 1",
        p.twist
    );
    assert_eq!(p.classify(0.3), Segment::Twist);
}

/// The winding is a topological quantity, so it does not depend on how far out
/// the circle is drawn.
#[test]
fn the_winding_is_independent_of_the_section_radius() {
    let (c, g) = rig();
    let s = wedge(&g, 0.5);
    for r in [0.12_f64, 0.2, 0.3] {
        let p = profile_at::<AxialApolar>(&c, &g, &s, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0], r, 64)
            .expect("inside");
        assert!(
            (p.winding - 0.5).abs() < 0.03,
            "radius {r}: winding {}",
            p.winding
        );
    }
}

/// Tags match the paper's colouring: yellow +1, purple -1, green 0.
#[test]
fn the_segment_tags_match_the_paper_colouring() {
    assert_eq!(Segment::PlusHalf.tag(), 1.0);
    assert_eq!(Segment::MinusHalf.tag(), -1.0);
    assert_eq!(Segment::Twist.tag(), 0.0);
}
