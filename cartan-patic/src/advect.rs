//! Semi-Lagrangian advection of the order parameter.
//!
//! Each vertex traces back along the flow to `x - u dt`, and the state there
//! is interpolated barycentrically from the tetrahedron containing it.
//!
//! ## Interpolating a coset
//!
//! Rotors cannot be averaged componentwise: the field is defined only up to
//! `H^` at each vertex, so four rotors of one tetrahedron may sit in four
//! different fundamental domains and their mean would be meaningless. Each is
//! therefore aligned to the nearest vertex's representative through the same
//! defect-group transition the detector uses, and only then averaged and
//! renormalised.
//!
//! Interpolating the invariant tensor instead would be linear and safe, and
//! would discard the lift, which is the one thing the rotor representation
//! exists to keep.

use cartan_core::rotor::Rotor3;

use crate::complex3::Complex3;
use crate::defect::edge_transition;
use crate::energy::State;
use crate::geometry::Geometry3;
use crate::group::SymmetryGroup;

/// Barycentric coordinates of a point in a tetrahedron.
fn barycentric(g: &Geometry3, tet: &[usize; 4], x: [f64; 3]) -> [f64; 4] {
    let d = g.tet_data(tet);
    let p = g.positions();
    let mut out = [0.0_f64; 4];
    for i in 0..4 {
        // lambda_i is affine with the stored gradient and value 1 at vertex i.
        let v = p[tet[i]];
        let dx = [x[0] - v[0], x[1] - v[1], x[2] - v[2]];
        out[i] = 1.0 + d.grads[i][0] * dx[0] + d.grads[i][1] * dx[1] + d.grads[i][2] * dx[2];
    }
    out
}

/// Whether every barycentric coordinate is non-negative to `tol`.
fn inside(l: &[f64; 4], tol: f64) -> bool {
    l.iter().all(|&v| v >= -tol)
}

/// Velocity at each vertex, from the Whitney reconstruction of the one-cochain
/// averaged over the incident tetrahedra.
#[must_use]
pub fn vertex_velocity(c: &Complex3, g: &Geometry3, u: &[f64]) -> Vec<[f64; 3]> {
    const LE: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    let mut acc = vec![[0.0_f64; 3]; c.n_vertices()];
    let mut wt = vec![0.0_f64; c.n_vertices()];
    for tet in c.tets() {
        let d = g.tet_data(tet);
        // The Whitney interpolant at the centroid, where every lambda is 1/4.
        let mut v = [0.0_f64; 3];
        for &[a, b] in LE.iter() {
            let e = c.edge_of(&[tet[a], tet[b]]);
            for (k, vk) in v.iter_mut().enumerate() {
                *vk += u[e] * (d.grads[b][k] - d.grads[a][k]) / 4.0;
            }
        }
        for &vtx in tet.iter() {
            wt[vtx] += d.volume;
            for (k, ak) in acc[vtx].iter_mut().enumerate() {
                *ak += v[k] * d.volume;
            }
        }
    }
    for (i, a) in acc.iter_mut().enumerate() {
        let w = wt[i].max(1e-300);
        for ak in a.iter_mut() {
            *ak /= w;
        }
    }
    acc
}

/// Advect the state by the flow for one step.
///
/// Departure points outside the domain fall back to the vertex itself, which
/// is the right behaviour under no-slip, where the boundary velocity vanishes
/// and no departure point should leave.
pub fn advect<H: SymmetryGroup>(
    c: &Complex3,
    g: &Geometry3,
    state: &State,
    u: &[f64],
    dt: f64,
) -> State {
    let vel = vertex_velocity(c, g, u);
    let p = g.positions();
    let n_amp = state.amps(0).len();

    // Tetrahedra incident to each vertex, tried first since a departure point
    // is within `|u| dt` of its vertex.
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); c.n_vertices()];
    for (t, tet) in c.tets().iter().enumerate() {
        for &v in tet.iter() {
            incident[v].push(t);
        }
    }

    let mut out = state.clone();
    for v in 0..c.n_vertices() {
        let x = [
            p[v][0] - dt * vel[v][0],
            p[v][1] - dt * vel[v][1],
            p[v][2] - dt * vel[v][2],
        ];
        let found = incident[v]
            .iter()
            .copied()
            .chain(0..c.n_tets())
            .find_map(|t| {
                let tet = c.tets()[t];
                let l = barycentric(g, &tet, x);
                inside(&l, 1e-9).then_some((tet, l))
            });
        let Some((tet, lam)) = found else {
            continue;
        };

        // Align the four rotors to the one with the largest weight, then
        // average in R^4 and renormalise.
        let anchor = (0..4)
            .max_by(|a, b| {
                lam[*a]
                    .partial_cmp(&lam[*b])
                    .unwrap_or(core::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let r_ref = state.rotors[tet[anchor]];
        let mut acc = Rotor3 {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let mut amps = vec![0.0_f64; n_amp];
        for i in 0..4 {
            let r = state.rotors[tet[i]];
            let h = edge_transition::<H>(&r_ref, &r);
            let a = r.compose(&h);
            acc.w += lam[i] * a.w;
            acc.x += lam[i] * a.x;
            acc.y += lam[i] * a.y;
            acc.z += lam[i] * a.z;
            let src = state.amps(tet[i]);
            for (k, a) in amps.iter_mut().enumerate() {
                *a += lam[i] * src[k];
            }
        }
        let norm = (acc.w * acc.w + acc.x * acc.x + acc.y * acc.y + acc.z * acc.z).sqrt();
        if norm > 1e-12 {
            out.rotors[v] = Rotor3 {
                w: acc.w / norm,
                x: acc.x / norm,
                y: acc.y / norm,
                z: acc.z / norm,
            };
        }
        for (k, a) in amps.iter().enumerate() {
            out.amplitudes[v * n_amp + k] = *a;
        }
    }
    out
}
