//! The local defect profile along a disclination line.
//!
//! A disclination is characterised by what the director does on a small circle
//! around it. Taking that circle in the plane perpendicular to the local
//! tangent and following the director round it gives the winding, and the
//! director's component along the tangent separates a wedge from a twist.
//!
//! ## Half-integer unwrapping
//!
//! A director is defined up to sign, so the angle it makes in the section
//! plane lives modulo `pi`, not `2 pi`. Following it once around a `+1/2`
//! disclination turns it by `pi`, which is the half-integer charge. Unwrapping
//! with period `2 pi` instead would read every half-integer defect as zero,
//! which is the one mistake this module exists to avoid.

use cartan_core::rotor::Rotor3;

use crate::advect::sample_state;
use crate::complex3::Complex3;
use crate::energy::State;
use crate::geometry::Geometry3;
use crate::group::SymmetryGroup;

/// What the director does on a section through a disclination.
#[derive(Clone, Copy, Debug)]
pub struct SegmentProfile {
    /// Winding of the director in the section plane, in turns. A wedge
    /// disclination gives `+0.5` or `-0.5`.
    pub winding: f64,
    /// Mean `|n . t|` on the circle. Zero for a wedge, where the director
    /// stays in the section plane, and one for a pure twist, where it lies
    /// along the line.
    pub twist: f64,
    /// Scalar order averaged on the circle, which drops toward the core.
    pub order: f64,
}

impl SegmentProfile {
    /// The paper's three-way classification: `+1/2` wedge, `-1/2` wedge, or
    /// twist between them.
    #[must_use]
    pub fn classify(&self, twist_cut: f64) -> Segment {
        if self.twist > twist_cut {
            Segment::Twist
        } else if self.winding > 0.0 {
            Segment::PlusHalf
        } else {
            Segment::MinusHalf
        }
    }
}

/// The three profile kinds a disclination segment takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `+1/2` wedge.
    PlusHalf,
    /// `-1/2` wedge.
    MinusHalf,
    /// Twist, where the director tips along the line.
    Twist,
}

impl Segment {
    /// A numeric tag for export, matching the paper's colouring: `+1` yellow,
    /// `-1` purple, `0` green.
    #[must_use]
    pub fn tag(self) -> f64 {
        match self {
            Segment::PlusHalf => 1.0,
            Segment::MinusHalf => -1.0,
            Segment::Twist => 0.0,
        }
    }
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v).max(1e-300);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// An orthonormal pair spanning the plane perpendicular to `t`.
fn frame_perp(t: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if t[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = unit(cross(t, seed));
    let e2 = unit(cross(t, e1));
    (e1, e2)
}

/// Profile the director on a circle of radius `radius` about one point of a
/// disclination curve.
///
/// Returns `None` when any sample falls outside the domain.
#[must_use]
pub fn profile_at<H: SymmetryGroup>(
    c: &Complex3,
    g: &Geometry3,
    state: &State,
    centre: [f64; 3],
    tangent: [f64; 3],
    radius: f64,
    samples: usize,
) -> Option<SegmentProfile> {
    let t = unit(tangent);
    let (e1, e2) = frame_perp(t);
    let mut angles = Vec::with_capacity(samples);
    let mut tw = 0.0;
    let mut ord = 0.0;

    for k in 0..samples {
        let th = 2.0 * core::f64::consts::PI * (k as f64) / (samples as f64);
        let (s, cth) = th.sin_cos();
        let p = [
            centre[0] + radius * (cth * e1[0] + s * e2[0]),
            centre[1] + radius * (cth * e1[1] + s * e2[1]),
            centre[2] + radius * (cth * e1[2] + s * e2[2]),
        ];
        let (r, amps) = sample_state::<H>(c, g, state, p)?;
        let n = r.rotate_vec([0.0, 0.0, 1.0]);
        angles.push(dot(n, e2).atan2(dot(n, e1)));
        tw += dot(n, t).abs();
        ord += amps.iter().map(|a| a * a).sum::<f64>().sqrt();
    }

    // Unwrap modulo pi, since the director has no sign.
    let pi = core::f64::consts::PI;
    let mut total = 0.0;
    for k in 0..samples {
        let a = angles[k];
        let b = angles[(k + 1) % samples];
        let mut d = b - a;
        while d > pi / 2.0 {
            d -= pi;
        }
        while d < -pi / 2.0 {
            d += pi;
        }
        total += d;
    }

    Some(SegmentProfile {
        winding: total / (2.0 * pi),
        twist: tw / samples as f64,
        order: ord / samples as f64,
    })
}

/// Profile every point of a disclination curve.
///
/// The tangent is a central difference along the curve, wrapping at the ends
/// so a closed loop is handled without a special case.
#[must_use]
pub fn profile_along<H: SymmetryGroup>(
    c: &Complex3,
    g: &Geometry3,
    state: &State,
    curve: &[[f64; 3]],
    radius: f64,
    samples: usize,
) -> Vec<Option<SegmentProfile>> {
    let n = curve.len();
    (0..n)
        .map(|i| {
            let a = curve[(i + n - 1) % n];
            let b = curve[(i + 1) % n];
            let t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            if norm(t) < 1e-12 {
                return None;
            }
            profile_at::<H>(c, g, state, curve[i], t, radius, samples)
        })
        .collect()
}

/// One sample of a section disk.
#[derive(Clone, Copy, Debug)]
pub struct DiskSample {
    /// In-plane coordinates, in units of the disk radius.
    pub u: f64,
    /// In-plane coordinate.
    pub v: f64,
    /// Director projected into the section plane, first component.
    pub nu: f64,
    /// Director projected into the section plane, second component.
    pub nv: f64,
    /// Director component along the line: 1 at a pure twist.
    pub nt: f64,
    /// Scalar order, which drops toward the core.
    pub order: f64,
}

/// Sample the director on a square grid inside the section disk.
///
/// This is the disk the paper draws around each point of the loop: the
/// director in the plane perpendicular to the line, which is where a `+1/2`
/// profile looks different from a `-1/2` one by eye.
#[must_use]
pub fn section_disk<H: SymmetryGroup>(
    c: &Complex3,
    g: &Geometry3,
    state: &State,
    centre: [f64; 3],
    tangent: [f64; 3],
    radius: f64,
    grid: usize,
) -> Vec<DiskSample> {
    let t = unit(tangent);
    let (e1, e2) = frame_perp(t);
    let mut out = Vec::with_capacity(grid * grid);
    for i in 0..grid {
        for j in 0..grid {
            let u = -1.0 + 2.0 * (i as f64 + 0.5) / grid as f64;
            let v = -1.0 + 2.0 * (j as f64 + 0.5) / grid as f64;
            if u * u + v * v > 1.0 {
                continue;
            }
            let p = [
                centre[0] + radius * (u * e1[0] + v * e2[0]),
                centre[1] + radius * (u * e1[1] + v * e2[1]),
                centre[2] + radius * (u * e1[2] + v * e2[2]),
            ];
            let Some((r, amps)) = sample_state::<H>(c, g, state, p) else {
                continue;
            };
            let n = r.rotate_vec([0.0, 0.0, 1.0]);
            out.push(DiskSample {
                u,
                v,
                nu: dot(n, e1),
                nv: dot(n, e2),
                nt: dot(n, t),
                order: amps.iter().map(|a| a * a).sum::<f64>().sqrt(),
            });
        }
    }
    out
}

/// The rotor whose director makes angle `theta` in the plane spanned by `e1`
/// and `e2`. Used to build exact disclination fixtures.
#[must_use]
pub fn director_rotor(n: [f64; 3]) -> Rotor3 {
    crate::boundary::rotor_taking_z_to(unit(n))
}
