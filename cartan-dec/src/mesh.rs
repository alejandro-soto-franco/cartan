// ~/cartan/cartan-dec/src/mesh.rs

//! Simplicial complex: vertices, (oriented) edges, and (oriented) triangles.
//!
//! ## Orientation conventions
//!
//! **Edges** `[i, j]` are stored with `i < j`. The signed boundary operator
//! assigns +1 to vertex j and -1 to vertex i for the canonical orientation.
//!
//! **Triangles** `[i, j, k]` are stored in counter-clockwise orientation as
//! seen from outside the manifold. The signed boundary operator assigns +1 to
//! edge (j,k), -1 to edge (i,k), and +1 to edge (i,j).
//!
//! ## Building a mesh
//!
//! Provide vertex positions (2D) and a triangle connectivity list. The edge
//! table is built automatically using `Mesh::from_triangles`. Edges are
//! globally deduplicated and consistently oriented (lower index first).
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341, 2005.
//!   Section 3 (simplicial complexes and orientation).
//! - Hirani. "Discrete Exterior Calculus." PhD thesis, Caltech, 2003.
//!   Chapter 2 (primal/dual mesh).

use nalgebra::Vector2;
use std::collections::HashMap;

use crate::error::DecError;

/// A 2D simplicial complex with vertices, edges, and triangles.
///
/// Vertex coordinates are in R². The complex lives in the plane (flat metric).
/// For curved manifolds, replace vertex positions with manifold points and
/// use geodesic distances for all metric computations.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex positions: n_vertices × 2 matrix (rows = vertices).
    pub vertices: Vec<[f64; 2]>,

    /// Edges: each is `[i, j]` with `i < j` (canonical orientation).
    /// Built automatically from `triangles` by `from_triangles`.
    pub edges: Vec<[usize; 2]>,

    /// Triangles: each is `[i, j, k]` in CCW order.
    pub triangles: Vec<[usize; 3]>,

    /// For each triangle, the indices of its three edges in `self.edges`.
    /// `tri_edges[t] = [e_ij, e_jk, e_ik]` (edge (i,j), (j,k), (i,k)).
    pub tri_edges: Vec<[usize; 3]>,

    /// Sign of each edge relative to its canonical (low→high) orientation.
    /// `tri_edge_signs[t][e] = +1 if the triangle uses the edge in its
    /// canonical direction, -1 otherwise.`
    pub tri_edge_signs: Vec<[f64; 3]>,
}

impl Mesh {
    /// Construct a mesh from vertex positions and a triangle list.
    ///
    /// Edges are deduplicated and canonically oriented (lower-index vertex first).
    /// Triangle orientation (CCW) is assumed as given by the caller.
    ///
    /// # Panics
    ///
    /// Panics if any triangle vertex index is out of bounds.
    pub fn from_triangles(vertices: Vec<[f64; 2]>, triangles: Vec<[usize; 3]>) -> Self {
        let n_v = vertices.len();
        // Build edge table: map canonical (i,j) pair → edge index.
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();

        let mut tri_edges = Vec::with_capacity(triangles.len());
        let mut tri_edge_signs = Vec::with_capacity(triangles.len());

        for &[i, j, k] in &triangles {
            assert!(i < n_v && j < n_v && k < n_v, "triangle index out of bounds");

            // The three edges of triangle (i,j,k) = [v₀,v₁,v₂]:
            //   raw_edges[0] = (v₀,v₁) = (i,j)  boundary coeff = +1
            //   raw_edges[1] = (v₁,v₂) = (j,k)  boundary coeff = +1
            //   raw_edges[2] = (v₀,v₂) = (i,k)  boundary coeff = -1
            //
            // d₁[t, e] = boundary_coeff × direction_sign
            // direction_sign = +1 if the edge is traversed low→high in the triangle,
            //                  -1 if traversed high→low.
            let raw_edges = [(i, j), (j, k), (i, k)];
            let boundary_coeffs = [1.0f64, 1.0, -1.0];
            let mut local_edges = [0usize; 3];
            let mut local_signs = [0.0f64; 3];

            for (slot, (&(a, b), &coeff)) in raw_edges.iter().zip(boundary_coeffs.iter()).enumerate() {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let direction_sign = if a < b { 1.0 } else { -1.0 };

                let idx = *edge_map.entry((lo, hi)).or_insert_with(|| {
                    let e = edges.len();
                    edges.push([lo, hi]);
                    e
                });

                local_edges[slot] = idx;
                local_signs[slot] = coeff * direction_sign;
            }

            tri_edges.push(local_edges);
            tri_edge_signs.push(local_signs);
        }

        Self {
            vertices,
            edges,
            triangles,
            tri_edges,
            tri_edge_signs,
        }
    }

    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn n_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Vertex position as a nalgebra Vector2.
    pub fn vertex(&self, i: usize) -> Vector2<f64> {
        Vector2::new(self.vertices[i][0], self.vertices[i][1])
    }

    /// Edge midpoint.
    pub fn edge_midpoint(&self, e: usize) -> Vector2<f64> {
        let [i, j] = self.edges[e];
        (self.vertex(i) + self.vertex(j)) * 0.5
    }

    /// Edge length (primal 1-simplex volume).
    pub fn edge_length(&self, e: usize) -> f64 {
        let [i, j] = self.edges[e];
        (self.vertex(j) - self.vertex(i)).norm()
    }

    /// Triangle area (primal 2-simplex volume), signed positive for CCW.
    pub fn triangle_area(&self, t: usize) -> f64 {
        let [i, j, k] = self.triangles[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);
        // Signed area via cross product: 0.5 * ((b-a) × (c-a)).
        let ab = b - a;
        let ac = c - a;
        0.5 * (ab.x * ac.y - ab.y * ac.x)
    }

    /// Triangle circumcenter.
    ///
    /// For a CCW triangle (A, B, C) with sides a, b, c:
    /// circumcenter = A + s*(B-A) + t*(C-A) where s, t solve the
    /// circumcircle equations.
    pub fn circumcenter(&self, t: usize) -> Vector2<f64> {
        let [i, j, k] = self.triangles[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);

        let ab = b - a;
        let ac = c - a;

        let d = 2.0 * (ab.x * ac.y - ab.y * ac.x);
        if d.abs() < 1e-30 {
            return (a + b + c) / 3.0; // degenerate: fall back to centroid
        }

        let ux = (ac.y * (ab.x * ab.x + ab.y * ab.y) - ab.y * (ac.x * ac.x + ac.y * ac.y)) / d;
        let uy = (ab.x * (ac.x * ac.x + ac.y * ac.y) - ac.x * (ab.x * ab.x + ab.y * ab.y)) / d;

        a + Vector2::new(ux, uy)
    }

    /// Check whether this mesh is well-centered.
    ///
    /// A mesh is well-centered if every triangle's circumcenter lies strictly
    /// inside the triangle. This guarantees positive Hodge weights.
    ///
    /// Returns `Ok(())` if well-centered, or `Err(DecError::NotWellCentered)`
    /// identifying the first offending triangle.
    pub fn check_well_centered(&self) -> Result<(), DecError> {
        for (t, &[i, j, k]) in self.triangles.iter().enumerate() {
            let a = self.vertex(i);
            let b = self.vertex(j);
            let c = self.vertex(k);
            let cc = self.circumcenter(t);

            // Check: circumcenter is inside iff all barycentric coordinates > 0.
            let area = self.triangle_area(t).abs();
            if area < 1e-30 {
                continue; // degenerate, skip
            }
            let la = (0.5 * ((b - cc).x * (c - cc).y - (b - cc).y * (c - cc).x)).abs() / area;
            let lb = (0.5 * ((a - cc).x * (c - cc).y - (a - cc).y * (c - cc).x)).abs() / area;
            let lc = (0.5 * ((a - cc).x * (b - cc).y - (a - cc).y * (b - cc).x)).abs() / area;

            if la < 0.0 || lb < 0.0 || lc < 0.0 || (la + lb + lc - 1.0).abs() > 1e-10 {
                return Err(DecError::NotWellCentered {
                    simplex: t,
                    volume: la.min(lb).min(lc),
                });
            }
        }
        Ok(())
    }

    /// Build a simple uniform triangulated grid on [0,1]² with `n` divisions per side.
    ///
    /// Produces 2*n² right triangles. Well-centered for all n.
    /// Useful for tests and prototyping.
    pub fn unit_square_grid(n: usize) -> Self {
        assert!(n >= 1, "grid must have at least 1 division");
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        // Vertices at i/(n), j/(n) for i,j in 0..=n.
        for j in 0..=n {
            for i in 0..=n {
                vertices.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }

        let idx = |i: usize, j: usize| j * (n + 1) + i;

        // Each square (i,j)..(i+1,j+1) is split into two CCW triangles.
        for j in 0..n {
            for i in 0..n {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                // Lower-left triangle: (v00, v10, v01)
                triangles.push([v00, v10, v01]);
                // Upper-right triangle: (v10, v11, v01)
                triangles.push([v10, v11, v01]);
            }
        }

        Self::from_triangles(vertices, triangles)
    }

    /// Assembles the full mesh topology into a `DMatrix` adjacency view.
    ///
    /// Returns a summary: (n_vertices, n_edges, n_triangles).
    pub fn euler_characteristic(&self) -> i32 {
        self.n_vertices() as i32 - self.n_edges() as i32 + self.n_triangles() as i32
    }
}
