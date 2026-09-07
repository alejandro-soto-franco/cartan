//! Gradient recovery, which is what lets the active stress reach any harmonic
//! degree.
//!
//! Piecewise-linear data has a constant gradient on each tetrahedron and none
//! at a vertex. Averaging those constants back to the vertices, weighted by
//! volume, returns a vertex field again, so the operation composes: applying
//! it `r` times gives `r` derivatives of a field that started with one.
//!
//! This is the standard volume-weighted patch recovery. It is first-order
//! accurate in the interior and drops an order at the boundary, and every
//! property the crate tests on it is exact rather than asymptotic: a field
//! that is affine recovers its exact gradient, and a constant field recovers
//! exactly zero at every vertex including boundary ones.

use crate::complex3::Complex3;
use crate::geometry::Geometry3;

/// Recover a vertex gradient from a piecewise-linear vertex field.
#[must_use]
pub fn recover_gradient(c: &Complex3, g: &Geometry3, f: &[f64]) -> Vec<[f64; 3]> {
    let mut acc = vec![[0.0_f64; 3]; c.n_vertices()];
    let mut wt = vec![0.0_f64; c.n_vertices()];
    for tet in c.tets() {
        let d = g.tet_data(tet);
        let mut grad = [0.0_f64; 3];
        for (a, &v) in tet.iter().enumerate() {
            for (k, gk) in grad.iter_mut().enumerate() {
                *gk += f[v] * d.grads[a][k];
            }
        }
        for &v in tet.iter() {
            wt[v] += d.volume;
            for (k, ak) in acc[v].iter_mut().enumerate() {
                *ak += grad[k] * d.volume;
            }
        }
    }
    for (v, a) in acc.iter_mut().enumerate() {
        let w = wt[v].max(1e-300);
        for ak in a.iter_mut() {
            *ak /= w;
        }
    }
    acc
}

/// Contract one index of a rank-`r` vertex tensor field with a derivative.
///
/// `field` holds `3^r` components per vertex, flattened in base 3 with the
/// contracted index last. The result has `3^(r-1)` components per vertex and
/// equals `d_k S_{... k}`.
#[must_use]
pub fn divergence_last_index(
    c: &Complex3,
    g: &Geometry3,
    field: &[Vec<f64>],
    rank: usize,
) -> Vec<Vec<f64>> {
    let free = 3usize.pow(rank as u32 - 1);
    let mut out = vec![vec![0.0_f64; free]; c.n_vertices()];
    for head in 0..free {
        for k in 0..3 {
            let comp = head + free * k;
            let scalar: Vec<f64> = (0..c.n_vertices()).map(|v| field[v][comp]).collect();
            let grad = recover_gradient(c, g, &scalar);
            for (v, row) in out.iter_mut().enumerate() {
                row[head] += grad[v][k];
            }
        }
    }
    out
}
