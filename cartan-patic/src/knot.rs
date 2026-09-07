//! Linking number and writhe of disclination lines.
//!
//! Lines come out of detection as a set of pierced faces. Ordering them into a
//! curve is a walk on the dual graph, where two pierced faces are adjacent
//! when a tetrahedron contains both.
//!
//! The Gauss integral is evaluated exactly for polygonal curves by the
//! solid-angle formula, so a linking number computed here is an integer up to
//! round-off rather than up to a quadrature tolerance.

use crate::complex3::Complex3;
use crate::defect::DefectField;
use crate::geometry::Geometry3;

type V3 = [f64; 3];

fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: &V3, b: &V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: &V3, b: &V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn unit(a: V3) -> V3 {
    let n = dot(&a, &a).sqrt();
    if n < 1e-300 {
        [0.0, 0.0, 0.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}

/// The signed solid angle of the quadrilateral spanned by two segments.
///
/// Klenin and Langowski's method 1a: the Gauss double integral over a pair of
/// straight segments is a sum of four arcsines, signed by the orientation of
/// the pair.
fn segment_pair(r1: &V3, r2: &V3, r3: &V3, r4: &V3) -> f64 {
    let r13 = sub(r3, r1);
    let r14 = sub(r4, r1);
    let r23 = sub(r3, r2);
    let r24 = sub(r4, r2);
    let n1 = unit(cross(&r13, &r14));
    let n2 = unit(cross(&r14, &r24));
    let n3 = unit(cross(&r24, &r23));
    let n4 = unit(cross(&r23, &r13));
    let a = |x: f64| x.clamp(-1.0, 1.0).asin();
    let omega = a(dot(&n1, &n2)) + a(dot(&n2, &n3)) + a(dot(&n3, &n4)) + a(dot(&n4, &n1));
    let r34 = sub(r4, r3);
    let r12 = sub(r2, r1);
    let s = dot(&cross(&r34, &r12), &r13);
    omega * s.signum()
}

/// Gauss linking number of two closed polygonal curves.
///
/// Integer for disjoint closed curves, up to round-off.
#[must_use]
pub fn linking_number(a: &[V3], b: &[V3]) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let (p1, p2) = (a[i], a[(i + 1) % a.len()]);
        for j in 0..b.len() {
            let (p3, p4) = (b[j], b[(j + 1) % b.len()]);
            s += segment_pair(&p1, &p2, &p3, &p4);
        }
    }
    s / (4.0 * core::f64::consts::PI)
}

/// Writhe of one closed polygonal curve: the Gauss integral of the curve with
/// itself, skipping adjacent segments where the integrand is singular.
#[must_use]
pub fn writhe(a: &[V3]) -> f64 {
    let n = a.len();
    let mut s = 0.0;
    for i in 0..n {
        let (p1, p2) = (a[i], a[(i + 1) % n]);
        for j in 0..n {
            if i == j || (i + 1) % n == j || (j + 1) % n == i {
                continue;
            }
            let (p3, p4) = (a[j], a[(j + 1) % n]);
            s += segment_pair(&p1, &p2, &p3, &p4);
        }
    }
    s / (4.0 * core::f64::consts::PI)
}

/// Order a component's pierced faces into a curve of face centroids.
///
/// Two pierced faces are adjacent when a tetrahedron contains both, which is
/// the dual edge the line crosses. The walk starts at an endpoint when the
/// component is an open arc and anywhere when it is a cycle.
#[must_use]
pub fn line_curve(c: &Complex3, g: &Geometry3, component: &[usize]) -> Vec<V3> {
    let member: std::collections::HashSet<usize> = component.iter().copied().collect();
    let mut adj: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for t in 0..c.n_tets() {
        let here: Vec<usize> = c
            .tet_triangles(t)
            .into_iter()
            .filter(|f| member.contains(f))
            .collect();
        for i in 0..here.len() {
            for j in 0..here.len() {
                if i != j {
                    adj.entry(here[i]).or_default().push(here[j]);
                }
            }
        }
    }
    let start = component
        .iter()
        .copied()
        .find(|f| adj.get(f).map(Vec::len).unwrap_or(0) < 2)
        .unwrap_or(component[0]);

    let mut order = vec![start];
    let mut seen: std::collections::HashSet<usize> = [start].into_iter().collect();
    loop {
        let cur = *order.last().expect("non-empty");
        let next = adj
            .get(&cur)
            .and_then(|ns| ns.iter().copied().find(|n| !seen.contains(n)));
        match next {
            Some(n) => {
                seen.insert(n);
                order.push(n);
            }
            None => break,
        }
    }

    let p = g.positions();
    order
        .into_iter()
        .map(|f| {
            let t = c.triangle(f);
            [
                (p[t[0]][0] + p[t[1]][0] + p[t[2]][0]) / 3.0,
                (p[t[0]][1] + p[t[1]][1] + p[t[2]][1]) / 3.0,
                (p[t[0]][2] + p[t[1]][2] + p[t[2]][2]) / 3.0,
            ]
        })
        .collect()
}

/// Every detected line as an ordered curve.
#[must_use]
pub fn curves(c: &Complex3, g: &Geometry3, d: &DefectField) -> Vec<Vec<V3>> {
    d.lines(c).iter().map(|l| line_curve(c, g, l)).collect()
}
