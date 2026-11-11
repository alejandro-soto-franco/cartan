//! The complex and its incidence operators.
//!
//! `B1` maps edge values to vertices: the column of edge `(i, j)` has `-1` at
//! `i` and `+1` at `j`, so `B1^T phi` is the difference `phi_j - phi_i`. `B2`
//! maps triangle values to edges: the column of triangle `(a, b, c)` has `+1`
//! on `(a, b)`, `+1` on `(b, c)` and `-1` on `(a, c)`. With these signs
//! `B1 B2 = 0` exactly.

use crate::ComplexError;

/// A sparse matrix in compressed rows with entries `+1` or `-1`.
#[derive(Debug, Clone, Default)]
pub(crate) struct SignedCsr {
    pub(crate) n_cols: usize,
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) cols: Vec<u32>,
    pub(crate) signs: Vec<f64>,
}

impl SignedCsr {
    /// Builds from `(row, col, sign)` triples. Rows need not be sorted.
    fn from_triples(n_rows: usize, n_cols: usize, triples: &[(u32, u32, f64)]) -> Self {
        let mut counts = vec![0usize; n_rows + 1];
        for &(r, _, _) in triples {
            counts[r as usize + 1] += 1;
        }
        for i in 0..n_rows {
            counts[i + 1] += counts[i];
        }
        let row_ptr = counts.clone();
        let mut next = counts;
        let mut cols = vec![0u32; triples.len()];
        let mut signs = vec![0.0; triples.len()];
        for &(r, c, s) in triples {
            let k = next[r as usize];
            cols[k] = c;
            signs[k] = s;
            next[r as usize] += 1;
        }
        Self {
            n_cols,
            row_ptr,
            cols,
            signs,
        }
    }

    pub(crate) fn n_rows(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// `out = A x`.
    pub(crate) fn apply(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols, "input length");
        assert_eq!(out.len(), self.n_rows(), "output length");
        for (r, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in self.row_ptr[r]..self.row_ptr[r + 1] {
                acc += self.signs[k] * x[self.cols[k] as usize];
            }
            *o = acc;
        }
    }
}

/// An abstract simplicial complex of dimension at most two.
#[derive(Debug, Clone)]
pub struct Complex {
    n_vertices: usize,
    edges: Vec<[u32; 2]>,
    triangles: Vec<[u32; 3]>,
    /// `B1`, vertices × edges.
    pub(crate) b1: SignedCsr,
    /// `B1^T`, edges × vertices.
    pub(crate) b1t: SignedCsr,
    /// `B2`, edges × triangles.
    pub(crate) b2: SignedCsr,
    /// `B2^T`, triangles × edges.
    pub(crate) b2t: SignedCsr,
}

impl Complex {
    /// A complex with `n_vertices` vertices and the given edges, each `[i, j]`
    /// with `i < j`. The edges are sorted lexicographically.
    pub fn new(n_vertices: usize, mut edges: Vec<[u32; 2]>) -> Result<Self, ComplexError> {
        for &e in &edges {
            if e[0] >= e[1] {
                return Err(ComplexError::EdgeNotOrdered { edge: e });
            }
            if e[1] as usize >= n_vertices {
                return Err(ComplexError::VertexOutOfRange {
                    vertex: e[1],
                    n_vertices,
                });
            }
        }
        edges.sort_unstable();
        if let Some(w) = edges.windows(2).find(|w| w[0] == w[1]) {
            return Err(ComplexError::DuplicateEdge { edge: w[0] });
        }
        let m = edges.len();
        let mut t1 = Vec::with_capacity(2 * m);
        let mut t1t = Vec::with_capacity(2 * m);
        for (k, e) in edges.iter().enumerate() {
            let k = k as u32;
            t1.push((e[0], k, -1.0));
            t1.push((e[1], k, 1.0));
            t1t.push((k, e[0], -1.0));
            t1t.push((k, e[1], 1.0));
        }
        Ok(Self {
            n_vertices,
            b1: SignedCsr::from_triples(n_vertices, m, &t1),
            b1t: SignedCsr::from_triples(m, n_vertices, &t1t),
            b2: SignedCsr::from_triples(m, 0, &[]),
            b2t: SignedCsr::from_triples(0, m, &[]),
            edges,
            triangles: Vec::new(),
        })
    }

    /// Adds the triangles, each `[a, b, c]` with `a < b < c`, listed in
    /// lexicographic order, every one of whose edges is in the complex.
    pub fn with_triangles(mut self, triangles: Vec<[u32; 3]>) -> Result<Self, ComplexError> {
        let m = self.edges.len();
        let t = triangles.len();
        let mut tr = Vec::with_capacity(3 * t);
        let mut trt = Vec::with_capacity(3 * t);
        for (k, &tri) in triangles.iter().enumerate() {
            let [a, b, c] = tri;
            let ordered = a < b && b < c && (k == 0 || triangles[k - 1] < tri);
            if !ordered {
                return Err(ComplexError::TriangleNotOrdered { triangle: tri });
            }
            if c as usize >= self.n_vertices {
                return Err(ComplexError::VertexOutOfRange {
                    vertex: c,
                    n_vertices: self.n_vertices,
                });
            }
            let k = k as u32;
            for (edge, sign) in [([a, b], 1.0), ([b, c], 1.0), ([a, c], -1.0)] {
                let e = self
                    .edges
                    .binary_search(&edge)
                    .map_err(|_| ComplexError::MissingEdge {
                        triangle: tri,
                        edge,
                    })? as u32;
                tr.push((e, k, sign));
                trt.push((k, e, sign));
            }
        }
        self.b2 = SignedCsr::from_triples(m, t, &tr);
        self.b2t = SignedCsr::from_triples(t, m, &trt);
        self.triangles = triangles;
        Ok(self)
    }

    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// The edges, sorted lexicographically.
    pub fn edges(&self) -> &[[u32; 2]] {
        &self.edges
    }

    /// The triangles, in lexicographic order.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// `out = B1 f`, from edge values to vertex values.
    pub fn b1(&self, f: &[f64], out: &mut [f64]) {
        self.b1.apply(f, out);
    }

    /// `out = B1^T phi`, from vertex values to edge values.
    pub fn b1t(&self, phi: &[f64], out: &mut [f64]) {
        self.b1t.apply(phi, out);
    }

    /// `out = B2 y`, from triangle values to edge values.
    pub fn b2(&self, y: &[f64], out: &mut [f64]) {
        self.b2.apply(y, out);
    }

    /// `out = B2^T r`, from edge values to triangle values.
    pub fn b2t(&self, r: &[f64], out: &mut [f64]) {
        self.b2t.apply(r, out);
    }

    /// A component label per vertex, numbered `0, 1, ...` in order of each
    /// component's lowest vertex. An isolated vertex is its own component.
    pub fn components(&self) -> Vec<u32> {
        let n = self.n_vertices;
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for e in &self.edges {
            let (a, b) = (
                find(&mut parent, e[0] as usize),
                find(&mut parent, e[1] as usize),
            );
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut label = vec![u32::MAX; n];
        let mut next = 0u32;
        (0..n)
            .map(|v| {
                let r = find(&mut parent, v);
                if label[r] == u32::MAX {
                    label[r] = next;
                    next += 1;
                }
                label[r]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ComplexError;

    fn k4() -> Complex {
        Complex::new(4, vec![[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
            .unwrap()
            .with_triangles(vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
            .unwrap()
    }

    #[test]
    fn b1_of_one_edge() {
        let c = Complex::new(2, vec![[0, 1]]).unwrap();
        let mut v = [0.0; 2];
        c.b1(&[2.0], &mut v);
        assert_eq!(v, [-2.0, 2.0]);
        let mut e = [0.0; 1];
        c.b1t(&[1.0, 4.0], &mut e);
        assert_eq!(e, [3.0]);
    }

    #[test]
    fn boundary_of_boundary_vanishes() {
        let c = k4();
        let mut e = [0.0; 6];
        c.b2(&[1.0, -2.0, 0.5, 3.0], &mut e);
        let mut v = [1.0; 4];
        c.b1(&e, &mut v);
        assert_eq!(v, [0.0; 4]);
    }

    #[test]
    fn b2_signs_on_one_triangle() {
        let c = Complex::new(3, vec![[0, 1], [0, 2], [1, 2]])
            .unwrap()
            .with_triangles(vec![[0, 1, 2]])
            .unwrap();
        let mut e = [0.0; 3];
        c.b2(&[1.0], &mut e);
        assert_eq!(e, [1.0, -1.0, 1.0]);
        let mut t = [0.0; 1];
        c.b2t(&[1.0, -1.0, 1.0], &mut t);
        assert_eq!(t, [3.0]);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            Complex::new(2, vec![[1, 0]]),
            Err(ComplexError::EdgeNotOrdered { .. })
        ));
        assert!(matches!(
            Complex::new(2, vec![[0, 1], [0, 1]]),
            Err(ComplexError::DuplicateEdge { .. })
        ));
        assert!(matches!(
            Complex::new(2, vec![[0, 2]]),
            Err(ComplexError::VertexOutOfRange { .. })
        ));
        let r = Complex::new(3, vec![[0, 1], [1, 2]])
            .unwrap()
            .with_triangles(vec![[0, 1, 2]]);
        assert!(matches!(
            r,
            Err(ComplexError::MissingEdge { edge: [0, 2], .. })
        ));
        let r = Complex::new(3, vec![[0, 1], [0, 2], [1, 2]])
            .unwrap()
            .with_triangles(vec![[0, 2, 1]]);
        assert!(matches!(r, Err(ComplexError::TriangleNotOrdered { .. })));
    }

    #[test]
    fn components_of_two_pieces() {
        let c = Complex::new(5, vec![[0, 1], [2, 3]]).unwrap();
        assert_eq!(c.components(), vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn edges_are_sorted_on_construction() {
        let c = Complex::new(3, vec![[1, 2], [0, 2], [0, 1]]).unwrap();
        assert_eq!(c.edges(), &[[0, 1], [0, 2], [1, 2]]);
        assert_eq!((c.n_vertices(), c.n_edges(), c.n_triangles()), (3, 3, 0));
    }
}
