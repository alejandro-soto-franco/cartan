//! Tetrahedral simplicial complexes and their coboundary chain.
//!
//! Sub-project 1's `Incidence` stops at triangles because the order parameter
//! needed no volumes. Flow needs the full chain: vertices, edges, triangles,
//! tetrahedra, and the signed coboundaries `d0`, `d1`, `d2` between them.
//!
//! Every simplex is stored with ascending vertex indices, so the face obtained
//! by deleting the `i`-th vertex enters with sign `(-1)^i` and `d d = 0` holds
//! by the alternating sum rather than by construction of the mesh.

use std::collections::HashMap;

use nalgebra_sparse::{CooMatrix, CsrMatrix};

/// A tetrahedral complex with its full face lattice.
#[derive(Clone, Debug)]
pub struct Complex3 {
    n_vertices: usize,
    edges: Vec<[usize; 2]>,
    triangles: Vec<[usize; 3]>,
    tets: Vec<[usize; 4]>,
}

fn key2(v: [usize; 2]) -> (usize, usize) {
    (v[0], v[1])
}
fn key3(v: [usize; 3]) -> (usize, usize, usize) {
    (v[0], v[1], v[2])
}

impl Complex3 {
    /// Build from tetrahedra given as vertex quadruples. Faces of every
    /// dimension are derived and deduplicated, each stored ascending.
    #[must_use]
    pub fn from_tets(n_vertices: usize, tets: &[[usize; 4]]) -> Self {
        let mut tet_list = Vec::with_capacity(tets.len());
        let mut tri_map: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut triangles = Vec::new();
        let mut edges = Vec::new();

        for t in tets {
            let mut v = *t;
            v.sort_unstable();
            tet_list.push(v);
            for i in 0..4 {
                let f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| v[k]).collect();
                let tri = [f[0], f[1], f[2]];
                tri_map.entry(key3(tri)).or_insert_with(|| {
                    triangles.push(tri);
                    triangles.len() - 1
                });
                for j in 0..3 {
                    let e: Vec<usize> = (0..3).filter(|&k| k != j).map(|k| tri[k]).collect();
                    let ed = [e[0], e[1]];
                    edge_map.entry(key2(ed)).or_insert_with(|| {
                        edges.push(ed);
                        edges.len() - 1
                    });
                }
            }
        }
        Self {
            n_vertices,
            edges,
            triangles,
            tets: tet_list,
        }
    }

    /// The Kuhn triangulation of an `n x n x n` grid of unit cubes: six
    /// tetrahedra per cube, one per permutation of the three axes.
    #[must_use]
    pub fn cube_grid(n: usize) -> Self {
        let s = n + 1;
        let idx = |i: usize, j: usize, k: usize| (i * s + j) * s + k;
        const PERMS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut tets = Vec::with_capacity(6 * n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for p in PERMS {
                        let mut c = [0usize; 3];
                        let mut quad = [0usize; 4];
                        quad[0] = idx(i, j, k);
                        for (step, &axis) in p.iter().enumerate() {
                            c[axis] += 1;
                            quad[step + 1] = idx(i + c[0], j + c[1], k + c[2]);
                        }
                        tets.push(quad);
                    }
                }
            }
        }
        Self::from_tets(s * s * s, &tets)
    }

    /// Number of vertices.
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }
    /// Number of edges.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
    /// Number of triangles.
    #[must_use]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }
    /// Number of tetrahedra.
    #[must_use]
    pub fn n_tets(&self) -> usize {
        self.tets.len()
    }
    /// The edges.
    #[must_use]
    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }
    /// The tetrahedra.
    #[must_use]
    pub fn tets(&self) -> &[[usize; 4]] {
        &self.tets
    }

    /// Euler characteristic `V - E + F - T`.
    #[must_use]
    pub fn euler_characteristic(&self) -> i64 {
        self.n_vertices as i64 - self.n_edges() as i64 + self.n_triangles() as i64
            - self.n_tets() as i64
    }

    fn edge_index(&self, e: [usize; 2]) -> usize {
        self.edges
            .iter()
            .position(|x| *x == e)
            .expect("edge not in complex")
    }
    fn triangle_index(&self, t: [usize; 3]) -> usize {
        self.triangles
            .iter()
            .position(|x| *x == t)
            .expect("triangle not in complex")
    }

    /// `d0`: vertices to edges. Row per edge.
    #[must_use]
    pub fn d0(&self) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(self.n_edges(), self.n_vertices);
        for (r, e) in self.edges.iter().enumerate() {
            // Deleting vertex 0 leaves e[1] with sign +1; deleting vertex 1
            // leaves e[0] with sign -1.
            coo.push(r, e[1], 1.0);
            coo.push(r, e[0], -1.0);
        }
        CsrMatrix::from(&coo)
    }

    /// `d1`: edges to triangles. Row per triangle.
    #[must_use]
    pub fn d1(&self) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(self.n_triangles(), self.n_edges());
        for (r, t) in self.triangles.iter().enumerate() {
            for i in 0..3 {
                let f: Vec<usize> = (0..3).filter(|&k| k != i).map(|k| t[k]).collect();
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                coo.push(r, self.edge_index([f[0], f[1]]), sign);
            }
        }
        CsrMatrix::from(&coo)
    }

    /// `d2`: triangles to tetrahedra. Row per tetrahedron.
    #[must_use]
    pub fn d2(&self) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(self.n_tets(), self.n_triangles());
        for (r, t) in self.tets.iter().enumerate() {
            for i in 0..4 {
                let f: Vec<usize> = (0..4).filter(|&k| k != i).map(|k| t[k]).collect();
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                coo.push(r, self.triangle_index([f[0], f[1], f[2]]), sign);
            }
        }
        CsrMatrix::from(&coo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(m: &CsrMatrix<f64>) -> f64 {
        m.values().iter().fold(0.0_f64, |a, &v| a.max(v.abs()))
    }

    #[test]
    fn single_tet_has_the_right_face_counts() {
        let c = Complex3::from_tets(4, &[[0, 1, 2, 3]]);
        assert_eq!(
            (c.n_vertices(), c.n_edges(), c.n_triangles(), c.n_tets()),
            (4, 6, 4, 1)
        );
        assert_eq!(c.euler_characteristic(), 1, "a tetrahedron is contractible");
    }

    #[test]
    fn the_cube_grid_is_a_ball() {
        for n in 1..=3 {
            let c = Complex3::cube_grid(n);
            assert_eq!(c.n_tets(), 6 * n * n * n);
            assert_eq!(
                c.euler_characteristic(),
                1,
                "n = {n}: a solid box is contractible"
            );
        }
    }

    /// The defining property of the chain. A sign error anywhere in the face
    /// enumeration shows up here and nowhere else.
    #[test]
    fn the_coboundary_squares_to_zero() {
        for c in [
            Complex3::from_tets(4, &[[0, 1, 2, 3]]),
            Complex3::cube_grid(2),
        ] {
            let d0 = c.d0();
            let d1 = c.d1();
            let d2 = c.d2();
            let dd1 = &d1 * &d0;
            let dd2 = &d2 * &d1;
            assert!(max_abs(&dd1) < 1e-14, "d1 d0 = {:e}", max_abs(&dd1));
            assert!(max_abs(&dd2) < 1e-14, "d2 d1 = {:e}", max_abs(&dd2));
        }
    }

    #[test]
    fn coboundary_shapes_match_the_face_counts() {
        let c = Complex3::cube_grid(2);
        assert_eq!(
            (c.d0().nrows(), c.d0().ncols()),
            (c.n_edges(), c.n_vertices())
        );
        assert_eq!(
            (c.d1().nrows(), c.d1().ncols()),
            (c.n_triangles(), c.n_edges())
        );
        assert_eq!(
            (c.d2().nrows(), c.d2().ncols()),
            (c.n_tets(), c.n_triangles())
        );
    }
}
