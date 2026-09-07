//! Geometry and Whitney mass matrices on a tetrahedral complex.
//!
//! Whitney forms give the Hodge masses on a general tetrahedral mesh, with no
//! well-centredness requirement. Every integral below is exact, from the
//! barycentric identity
//!
//! ```text
//! integral over T of lambda_a lambda_b = V (1 + delta_ab) / 20
//! ```
//!
//! so nothing here is quadrature and nothing carries a quadrature error.

use nalgebra::{Matrix3, Vector3};
use nalgebra_sparse::{CooMatrix, CsrMatrix};

use crate::complex3::Complex3;

/// Vertex positions attached to a complex.
#[derive(Clone, Debug)]
pub struct Geometry3 {
    positions: Vec<[f64; 3]>,
}

/// Per-tetrahedron geometric data: signed volume and barycentric gradients.
#[derive(Clone, Copy, Debug)]
pub struct TetData {
    /// Unsigned volume.
    pub volume: f64,
    /// Gradient of each barycentric coordinate, constant on the tetrahedron.
    pub grads: [[f64; 3]; 4],
}

impl Geometry3 {
    /// Attach positions. One per vertex.
    #[must_use]
    pub fn new(positions: Vec<[f64; 3]>) -> Self {
        Self { positions }
    }

    /// Unit-cube grid positions matching [`Complex3::cube_grid`].
    #[must_use]
    pub fn cube_grid(n: usize) -> Self {
        let s = n + 1;
        let h = 1.0 / n as f64;
        let mut p = Vec::with_capacity(s * s * s);
        for i in 0..s {
            for j in 0..s {
                for k in 0..s {
                    p.push([i as f64 * h, j as f64 * h, k as f64 * h]);
                }
            }
        }
        Self::new(p)
    }

    /// The positions.
    #[must_use]
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Volume and barycentric gradients of one tetrahedron.
    ///
    /// The gradients solve `grad lambda_i . (v_j - v_0) = delta_ij - delta_0j`,
    /// which is the inverse of the edge matrix.
    #[must_use]
    pub fn tet_data(&self, tet: &[usize; 4]) -> TetData {
        let p: Vec<Vector3<f64>> = tet
            .iter()
            .map(|&v| Vector3::from_column_slice(&self.positions[v]))
            .collect();
        let e = Matrix3::from_columns(&[p[1] - p[0], p[2] - p[0], p[3] - p[0]]);
        let det = e.determinant();
        let volume = det.abs() / 6.0;
        let inv = e.try_inverse().unwrap_or_else(Matrix3::zeros);
        // Rows of inv are grad lambda_1, grad lambda_2, grad lambda_3.
        let mut grads = [[0.0_f64; 3]; 4];
        for i in 0..3 {
            for (k, gk) in grads[i + 1].iter_mut().enumerate() {
                *gk = inv[(i, k)];
            }
        }
        let (head, tail) = grads.split_at_mut(1);
        for (k, g0) in head[0].iter_mut().enumerate() {
            *g0 = -(tail[0][k] + tail[1][k] + tail[2][k]);
        }
        TetData { volume, grads }
    }

    /// Total volume.
    #[must_use]
    pub fn total_volume(&self, c: &Complex3) -> f64 {
        c.tets().iter().map(|t| self.tet_data(t).volume).sum()
    }
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `integral lambda_a lambda_b = V (1 + delta_ab) / 20`.
fn lam2(v: f64, a: usize, b: usize) -> f64 {
    v * if a == b { 2.0 } else { 1.0 } / 20.0
}

/// The Whitney mass matrix on 0-forms, `M0[a][b] = integral lambda_a lambda_b`.
#[must_use]
pub fn mass0(c: &Complex3, g: &Geometry3) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(c.n_vertices(), c.n_vertices());
    for t in c.tets() {
        let d = g.tet_data(t);
        for a in 0..4 {
            for b in 0..4 {
                coo.push(t[a], t[b], lam2(d.volume, a, b));
            }
        }
    }
    CsrMatrix::from(&coo)
}

/// The Whitney mass matrix on 1-forms.
///
/// The Whitney edge form is `w_ij = lambda_i grad lambda_j - lambda_j grad
/// lambda_i`, so the element entry expands into four barycentric integrals.
#[must_use]
pub fn mass1(c: &Complex3, g: &Geometry3) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(c.n_edges(), c.n_edges());
    // Local edges of a tetrahedron, ascending, matching the global convention.
    const LE: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
    for t in c.tets() {
        let d = g.tet_data(t);
        let gl = |i: usize| d.grads[i];
        for (ea, &[i, j]) in LE.iter().enumerate() {
            for (eb, &[k, l]) in LE.iter().enumerate() {
                let _ = (ea, eb);
                let val = lam2(d.volume, i, k) * dot(&gl(j), &gl(l))
                    - lam2(d.volume, i, l) * dot(&gl(j), &gl(k))
                    - lam2(d.volume, j, k) * dot(&gl(i), &gl(l))
                    + lam2(d.volume, j, l) * dot(&gl(i), &gl(k));
                let ga = c.edge_of(&[t[i], t[j]]);
                let gb = c.edge_of(&[t[k], t[l]]);
                coo.push(ga, gb, val);
            }
        }
    }
    CsrMatrix::from(&coo)
}

/// The Whitney mass matrix on 2-forms.
///
/// In three dimensions the Whitney face form has the vector proxy
/// `w_ijk = 2 (lambda_i gj x gk + lambda_j gk x gi + lambda_k gi x gj)`.
#[must_use]
pub fn mass2(c: &Complex3, g: &Geometry3) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(c.n_triangles(), c.n_triangles());
    const LF: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    for t in c.tets() {
        let d = g.tet_data(t);
        let gl = |i: usize| d.grads[i];
        for &fa in LF.iter() {
            for &fb in LF.iter() {
                // Each proxy is a sum of three terms lambda_a * (cross of the
                // other two gradients); the product integrates term by term.
                let terms = |f: [usize; 3]| {
                    [
                        (f[0], cross(&gl(f[1]), &gl(f[2]))),
                        (f[1], cross(&gl(f[2]), &gl(f[0]))),
                        (f[2], cross(&gl(f[0]), &gl(f[1]))),
                    ]
                };
                let ta = terms(fa);
                let tb = terms(fb);
                let mut val = 0.0;
                for (a, va) in &ta {
                    for (b, vb) in &tb {
                        val += 4.0 * lam2(d.volume, *a, *b) * dot(va, vb);
                    }
                }
                let ga = c.triangle_of(&[t[fa[0]], t[fa[1]], t[fa[2]]]);
                let gb = c.triangle_of(&[t[fb[0]], t[fb[1]], t[fb[2]]]);
                coo.push(ga, gb, val);
            }
        }
    }
    CsrMatrix::from(&coo)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> (Complex3, Geometry3) {
        (Complex3::cube_grid(n), Geometry3::cube_grid(n))
    }

    fn sum(m: &CsrMatrix<f64>) -> f64 {
        m.values().iter().sum()
    }

    fn is_symmetric(m: &CsrMatrix<f64>) -> f64 {
        let t = m.transpose();
        let d = m - &t;
        d.values().iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    }

    #[test]
    fn the_kuhn_grid_fills_the_unit_cube() {
        for n in 1..=3 {
            let (c, g) = grid(n);
            let v = g.total_volume(&c);
            assert!((v - 1.0).abs() < 1e-13, "n = {n}: volume {v}");
        }
    }

    #[test]
    fn barycentric_gradients_are_a_partition_of_unity() {
        let (c, g) = grid(2);
        for t in c.tets() {
            let d = g.tet_data(t);
            for k in 0..3 {
                let s: f64 = (0..4).map(|i| d.grads[i][k]).sum();
                assert!(s.abs() < 1e-10, "gradients must sum to zero, got {s}");
            }
        }
    }

    #[test]
    fn the_zero_form_mass_totals_the_volume() {
        let (c, g) = grid(2);
        let m0 = mass0(&c, &g);
        assert!(is_symmetric(&m0) < 1e-14);
        assert!((sum(&m0) - 1.0).abs() < 1e-12, "sum M0 = {}", sum(&m0));
    }

    /// Whitney one-forms reproduce a constant vector field exactly, so the
    /// discrete energy of its cochain equals `|U|^2` times the volume. This is
    /// the sharpest available check on `mass1`: any error in the barycentric
    /// integral, the gradient, or the edge orientation breaks it.
    #[test]
    fn the_one_form_mass_reproduces_a_constant_field() {
        let (c, g) = grid(2);
        let m1 = mass1(&c, &g);
        assert!(
            is_symmetric(&m1) < 1e-12,
            "M1 asymmetry {}",
            is_symmetric(&m1)
        );
        let p = g.positions();
        for u in [[1.0, 0.0, 0.0], [0.3, -0.7, 0.5]] {
            let coch: Vec<f64> = c
                .edges()
                .iter()
                .map(|e| {
                    let d = [
                        p[e[1]][0] - p[e[0]][0],
                        p[e[1]][1] - p[e[0]][1],
                        p[e[1]][2] - p[e[0]][2],
                    ];
                    dot(&u, &d)
                })
                .collect();
            let mut energy = 0.0;
            for (r, row) in m1.row_iter().enumerate() {
                for (&col, &v) in row.col_indices().iter().zip(row.values()) {
                    energy += coch[r] * v * coch[col];
                }
            }
            let expect = dot(&u, &u);
            assert!(
                (energy - expect).abs() < 1e-11,
                "constant field {u:?}: discrete energy {energy:e}, expected {expect:e}"
            );
        }
    }

    /// The same check one degree up: a constant two-form's cochain is the flux
    /// through each face, and its discrete energy is `|B|^2` times the volume.
    #[test]
    fn the_two_form_mass_reproduces_a_constant_flux() {
        let (c, g) = grid(2);
        let m2 = mass2(&c, &g);
        assert!(
            is_symmetric(&m2) < 1e-10,
            "M2 asymmetry {}",
            is_symmetric(&m2)
        );
        let p = g.positions();
        for b in [[1.0, 0.0, 0.0], [0.2, 0.4, -0.9]] {
            let coch: Vec<f64> = (0..c.n_triangles())
                .map(|i| {
                    let t = c.triangle(i);
                    let e1 = [
                        p[t[1]][0] - p[t[0]][0],
                        p[t[1]][1] - p[t[0]][1],
                        p[t[1]][2] - p[t[0]][2],
                    ];
                    let e2 = [
                        p[t[2]][0] - p[t[0]][0],
                        p[t[2]][1] - p[t[0]][1],
                        p[t[2]][2] - p[t[0]][2],
                    ];
                    let n = cross(&e1, &e2);
                    0.5 * dot(&b, &n)
                })
                .collect();
            let mut energy = 0.0;
            for (r, row) in m2.row_iter().enumerate() {
                for (&col, &v) in row.col_indices().iter().zip(row.values()) {
                    energy += coch[r] * v * coch[col];
                }
            }
            let expect = dot(&b, &b);
            assert!(
                (energy - expect).abs() < 1e-10,
                "constant flux {b:?}: discrete energy {energy:e}, expected {expect:e}"
            );
        }
    }
}
