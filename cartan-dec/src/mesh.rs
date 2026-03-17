// ~/cartan/cartan-dec/src/mesh.rs

//! Simplicial complex: vertices, (oriented) boundary faces, and simplices.
//!
//! ## Orientation conventions
//!
//! **Boundary faces (edges for K=3)** are stored with canonical orientation
//! (lower-index vertex first for K=3). The signed boundary operator assigns
//! +1 to the head vertex and -1 to the tail vertex.
//!
//! **Simplices (triangles for K=3)** are stored in counter-clockwise order.
//!
//! ## Generic mesh
//!
//! `Mesh<M, const K, const B>` is the general type. `K` = vertices per simplex
//! (K=3 for triangles) and `B = K-1` = vertices per boundary face. The
//! `FlatMesh` type alias provides full backward compatibility with the old
//! flat `Mesh` type.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341, 2005.
//! - Hirani. "Discrete Exterior Calculus." PhD thesis, Caltech, 2003.

use core::marker::PhantomData;
use std::collections::HashMap;

use nalgebra::{SVector, Vector2};

use cartan_core::{Manifold, Real};
use cartan_manifolds::euclidean::Euclidean;

use crate::error::DecError;

/// A simplicial complex with vertices on a Riemannian manifold `M`.
///
/// # Type parameters
///
/// - `M`: Manifold type. Vertex positions are `M::Point`.
/// - `K`: Vertices per simplex (K=3 for triangles).
/// - `B`: Vertices per boundary face; must equal K-1. Separate const because
///   Rust does not allow `K-1` as an array size in const generics.
#[derive(Debug, Clone)]
pub struct Mesh<M: Manifold, const K: usize = 3, const B: usize = 2> {
    /// Vertex positions (one `M::Point` per vertex).
    pub vertices: Vec<M::Point>,

    /// Simplices: each entry is K vertex indices.
    pub simplices: Vec<[usize; K]>,

    /// Boundary faces: each entry is B vertex indices (B = K-1, edges for K=3).
    /// Globally deduplicated, canonically oriented (low index first for K=3).
    pub boundaries: Vec<[usize; B]>,

    /// For each simplex, the K indices into `self.boundaries` of its boundary faces.
    pub simplex_boundary_ids: Vec<[usize; K]>,

    /// For each simplex, the K signed contributions of its boundary faces (+/-1.0).
    pub boundary_signs: Vec<[f64; K]>,

    _phantom: PhantomData<M>,
}

/// Backward-compatible type alias: flat 2D triangular mesh.
///
/// Identical API to the old `Mesh` type. All existing code works via this alias.
pub type FlatMesh = Mesh<Euclidean<2>, 3, 2>;

// ─────────────────────────────────────────────────────────────────────────────
// Generic impl — topology only (no metric)
// ─────────────────────────────────────────────────────────────────────────────

impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B> {
    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of boundary faces (edges for K=3).
    pub fn n_boundaries(&self) -> usize {
        self.boundaries.len()
    }

    /// Number of simplices (triangles for K=3).
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Euler characteristic: V - E + F (for 2D: n_vertices - n_edges + n_triangles).
    pub fn euler_characteristic(&self) -> i32 {
        self.n_vertices() as i32 - self.n_boundaries() as i32 + self.n_simplices() as i32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle mesh constructor (K=3, B=2) — works for any manifold M
// ─────────────────────────────────────────────────────────────────────────────

impl<M: Manifold> Mesh<M, 3, 2> {
    /// Construct a triangle mesh from vertex positions and a triangle list.
    ///
    /// Edges are deduplicated and canonically oriented (lower-index vertex first).
    /// Triangle orientation (CCW) is assumed as given by the caller.
    /// The `_manifold` parameter fixes the type but is not used during construction.
    ///
    /// # Panics
    ///
    /// Panics if any triangle vertex index is out of bounds.
    pub fn from_simplices(
        _manifold: &M,
        vertices: Vec<M::Point>,
        triangles: Vec<[usize; 3]>,
    ) -> Self {
        let n_v = vertices.len();
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut simplex_boundary_ids = Vec::with_capacity(triangles.len());
        let mut boundary_signs = Vec::with_capacity(triangles.len());

        for &[i, j, k] in &triangles {
            assert!(i < n_v && j < n_v && k < n_v, "triangle index out of bounds");

            // The three edges of triangle [i, j, k]:
            //   (i,j): boundary coeff = +1
            //   (j,k): boundary coeff = +1
            //   (i,k): boundary coeff = -1 (opposite orientation)
            let raw_edges = [(i, j), (j, k), (i, k)];
            let boundary_coeffs = [1.0f64, 1.0, -1.0];
            let mut local_edge_ids = [0usize; 3];
            let mut local_signs = [0.0f64; 3];

            for (slot, (&(a, b), &coeff)) in
                raw_edges.iter().zip(boundary_coeffs.iter()).enumerate()
            {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let direction_sign = if a < b { 1.0 } else { -1.0 };
                let idx = *edge_map.entry((lo, hi)).or_insert_with(|| {
                    let e = edges.len();
                    edges.push([lo, hi]);
                    e
                });
                local_edge_ids[slot] = idx;
                local_signs[slot] = coeff * direction_sign;
            }

            simplex_boundary_ids.push(local_edge_ids);
            boundary_signs.push(local_signs);
        }

        Self {
            vertices,
            simplices: triangles,
            boundaries: edges,
            simplex_boundary_ids,
            boundary_signs,
            _phantom: PhantomData,
        }
    }

    /// Geodesic edge length: `manifold.dist(v_i, v_j)`.
    pub fn edge_length(&self, manifold: &M, e: usize) -> Real {
        let [i, j] = self.boundaries[e];
        manifold
            .dist(&self.vertices[i], &self.vertices[j])
            .unwrap_or(0.0)
    }

    /// Triangle area via tangent-space Gram determinant.
    ///
    /// Maps v1 and v2 into `T_{v0}M` via `log`, then computes
    /// `0.5 * sqrt(||u||^2 * ||v||^2 - <u,v>^2)`.
    pub fn triangle_area(&self, manifold: &M, t: usize) -> Real {
        let [i, j, k] = self.simplices[t];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let v2 = &self.vertices[k];
        let u = manifold
            .log(v0, v1)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        let v = manifold
            .log(v0, v2)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        let uu = manifold.inner(v0, &u, &u);
        let vv = manifold.inner(v0, &v, &v);
        let uv = manifold.inner(v0, &u, &v);
        0.5 * (uu * vv - uv * uv).abs().sqrt()
    }

    /// Circumcenter via tangent-space 2x2 equidistance system at v0.
    ///
    /// For flat meshes this is exact; for curved manifolds this is a
    /// first-order tangent-space approximation. Falls back to v0 for
    /// degenerate (near-zero-area) triangles.
    pub fn circumcenter(&self, manifold: &M, t: usize) -> M::Point {
        let [i, j, k] = self.simplices[t];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let v2 = &self.vertices[k];
        let u = manifold
            .log(v0, v1)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        let v = manifold
            .log(v0, v2)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        let uu = manifold.inner(v0, &u, &u);
        let vv = manifold.inner(v0, &v, &v);
        let uv = manifold.inner(v0, &u, &v);
        let det = uu * vv - uv * uv;
        if det.abs() < 1e-30 {
            return v0.clone();
        }
        let s = (vv * 0.5 * uu - uv * 0.5 * vv) / det;
        let tc = (uu * 0.5 * vv - uv * 0.5 * uu) / det;
        let tangent = u * s + v * tc;
        manifold.exp(v0, &tangent)
    }

    /// Geodesic midpoint of boundary face (edge) e.
    pub fn boundary_midpoint(&self, manifold: &M, e: usize) -> M::Point {
        let [i, j] = self.boundaries[e];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let half_log = manifold
            .log(v0, v1)
            .map(|u| u * 0.5)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        manifold.exp(v0, &half_log)
    }

    /// Check whether this mesh is well-centered.
    ///
    /// Returns `Ok(())` if all circumcenters lie strictly inside their triangles,
    /// or `Err(DecError::NotWellCentered)` identifying the first offending simplex.
    pub fn check_well_centered(&self, manifold: &M) -> Result<(), DecError> {
        for t in 0..self.n_simplices() {
            let [i, j, k] = self.simplices[t];
            let v0 = &self.vertices[i];
            let v1 = &self.vertices[j];
            let v2 = &self.vertices[k];
            let u = manifold
                .log(v0, v1)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            let v = manifold
                .log(v0, v2)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            let uu = manifold.inner(v0, &u, &u);
            let vv = manifold.inner(v0, &v, &v);
            let uv = manifold.inner(v0, &u, &v);
            let det = uu * vv - uv * uv;
            if det.abs() < 1e-30 {
                continue;
            }
            let s = (vv * 0.5 * uu - uv * 0.5 * vv) / det;
            let tc = (uu * 0.5 * vv - uv * 0.5 * uu) / det;
            if s <= 0.0 || tc <= 0.0 || s + tc >= 1.0 {
                return Err(DecError::NotWellCentered {
                    simplex: t,
                    volume: s.min(tc).min(1.0 - s - tc),
                });
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlatMesh-specific impl — backward-compatible flat builders and metric helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Mesh<Euclidean<2>, 3, 2> {
    /// Construct a flat mesh from raw `[f64; 2]` vertex positions.
    ///
    /// Backward-compatible entry point. Converts positions to `SVector<f64, 2>`.
    pub fn from_triangles(vertices: Vec<[f64; 2]>, triangles: Vec<[usize; 3]>) -> Self {
        let manifold = Euclidean::<2>;
        let mv: Vec<SVector<f64, 2>> =
            vertices.iter().map(|&[x, y]| SVector::from([x, y])).collect();
        Self::from_simplices(&manifold, mv, triangles)
    }

    /// Vertex position as a nalgebra Vector2.
    pub fn vertex(&self, i: usize) -> Vector2<f64> {
        Vector2::new(self.vertices[i].x, self.vertices[i].y)
    }

    /// Flat edge midpoint (arithmetic mean of endpoints).
    pub fn edge_midpoint(&self, e: usize) -> Vector2<f64> {
        let [i, j] = self.boundaries[e];
        (self.vertex(i) + self.vertex(j)) * 0.5
    }

    /// Flat edge length (Euclidean norm). Faster than the generic geodesic method.
    pub fn edge_length_flat(&self, e: usize) -> f64 {
        let [i, j] = self.boundaries[e];
        (self.vertex(j) - self.vertex(i)).norm()
    }

    /// Flat triangle area (signed, CCW positive). Faster than the generic Gram method.
    pub fn triangle_area_flat(&self, t: usize) -> f64 {
        let [i, j, k] = self.simplices[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);
        let ab = b - a;
        let ac = c - a;
        0.5 * (ab.x * ac.y - ab.y * ac.x)
    }

    /// Flat circumcenter (exact formula). Faster than the generic tangent-space method.
    pub fn circumcenter_flat(&self, t: usize) -> Vector2<f64> {
        let [i, j, k] = self.simplices[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);
        let ab = b - a;
        let ac = c - a;
        let d = 2.0 * (ab.x * ac.y - ab.y * ac.x);
        if d.abs() < 1e-30 {
            return (a + b + c) / 3.0;
        }
        let ux = (ac.y * (ab.x * ab.x + ab.y * ab.y)
            - ab.y * (ac.x * ac.x + ac.y * ac.y))
            / d;
        let uy = (ab.x * (ac.x * ac.x + ac.y * ac.y)
            - ac.x * (ab.x * ab.x + ab.y * ab.y))
            / d;
        a + Vector2::new(ux, uy)
    }

    /// Build a uniform triangulated grid on [0,1]^2 with `n` divisions per side.
    ///
    /// Produces 2n^2 right triangles. Well-centered for all n.
    pub fn unit_square_grid(n: usize) -> Self {
        assert!(n >= 1, "grid must have at least 1 division");
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        let idx = |i: usize, j: usize| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                triangles.push([v00, v10, v01]);
                triangles.push([v10, v11, v01]);
            }
        }
        Self::from_triangles(vertices, triangles)
    }

    // Deprecated stubs for migration: removed after internal consumers are updated.

    /// Deprecated: use `n_boundaries()` instead.
    #[allow(dead_code)]
    pub fn n_edges(&self) -> usize {
        self.n_boundaries()
    }

    /// Deprecated: use `n_simplices()` instead.
    #[allow(dead_code)]
    pub fn n_triangles(&self) -> usize {
        self.n_simplices()
    }
}
