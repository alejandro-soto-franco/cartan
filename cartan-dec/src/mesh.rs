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

    /// Vertex -> incident boundary-face indices. `vertex_boundaries[v]` lists all
    /// boundary faces that contain vertex v.
    pub vertex_boundaries: Vec<Vec<usize>>,

    /// Vertex -> incident simplex indices. `vertex_simplices[v]` lists all
    /// simplices that contain vertex v.
    pub vertex_simplices: Vec<Vec<usize>>,

    /// Boundary-face -> adjacent simplex indices. `boundary_simplices[b]` lists all
    /// simplices that have boundary face b in their boundary.
    pub boundary_simplices: Vec<Vec<usize>>,

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

    /// Euler characteristic: V - E + F (for 2D: n_vertices - n_boundaries + n_simplices).
    pub fn euler_characteristic(&self) -> i32 {
        self.n_vertices() as i32 - self.n_boundaries() as i32 + self.n_simplices() as i32
    }

    /// K-generic constructor. For K=3, prefer `from_simplices` on `Mesh<M, 3, 2>`.
    ///
    /// Boundary faces are B-tuples of vertices, deduplicated and canonically
    /// oriented (sorted vertex indices). The boundary operator signs are computed
    /// from the relative orientation of each boundary face within its parent simplex.
    ///
    /// # Panics
    ///
    /// Panics if any simplex vertex index is out of bounds, or if B != K-1.
    pub fn from_simplices_generic(
        _manifold: &M,
        vertices: Vec<M::Point>,
        simplices: Vec<[usize; K]>,
    ) -> Self {
        assert_eq!(B, K - 1, "B must equal K-1");
        let n_v = vertices.len();

        let mut boundary_map: HashMap<[usize; B], usize> = HashMap::new();
        let mut boundaries: Vec<[usize; B]> = Vec::new();
        let mut simplex_boundary_ids = Vec::with_capacity(simplices.len());
        let mut boundary_signs_vec = Vec::with_capacity(simplices.len());

        for simplex in &simplices {
            for &v in simplex {
                assert!(
                    v < n_v,
                    "simplex vertex index {v} out of bounds (n_v={n_v})"
                );
            }

            let mut local_boundary_ids = [0usize; K];
            let mut local_signs = [0.0f64; K];

            // The k-th boundary face of simplex [v0, v1, ..., v_{K-1}] is obtained
            // by omitting vertex k. The sign is (-1)^k (from the boundary operator).
            for omit in 0..K {
                let sign = if omit % 2 == 0 { 1.0 } else { -1.0 };

                // Build the boundary face by collecting all vertices except the omitted one.
                let mut face = [0usize; B];
                let mut idx = 0;
                for (pos, &v) in simplex.iter().enumerate() {
                    if pos != omit {
                        face[idx] = v;
                        idx += 1;
                    }
                }

                // Canonical orientation: sort the face vertices.
                let mut sorted_face = face;
                sorted_face.sort();

                // Determine the parity of the permutation from face to sorted_face.
                let parity = permutation_sign(&face, &sorted_face);
                let effective_sign = sign * parity;

                let boundary_idx = *boundary_map.entry(sorted_face).or_insert_with(|| {
                    let b = boundaries.len();
                    boundaries.push(sorted_face);
                    b
                });

                local_boundary_ids[omit] = boundary_idx;
                local_signs[omit] = effective_sign;
            }

            simplex_boundary_ids.push(local_boundary_ids);
            boundary_signs_vec.push(local_signs);
        }

        let mut mesh = Self {
            vertices,
            simplices,
            boundaries,
            simplex_boundary_ids,
            boundary_signs: boundary_signs_vec,
            vertex_boundaries: Vec::new(),
            vertex_simplices: Vec::new(),
            boundary_simplices: Vec::new(),
            _phantom: PhantomData,
        };
        mesh.rebuild_adjacency();
        mesh
    }

    /// Volume of simplex s via the Gram determinant in the tangent space at vertex 0.
    ///
    /// For K=3 (triangle): area. For K=4 (tet): volume.
    /// The formula is: vol = (1 / (K-1)!) * sqrt(|det(G)|) where G is the
    /// (K-1) x (K-1) Gram matrix G_{ij} = <log(v0, v_i), log(v0, v_j)>.
    pub fn simplex_volume(&self, manifold: &M, s: usize) -> f64 {
        let simplex = &self.simplices[s];
        let v0 = &self.vertices[simplex[0]];

        let n = K - 1;
        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for &vi_idx in &simplex[1..] {
            let vi = &self.vertices[vi_idx];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let det = dense_determinant(&gram, n);
        let factorial = (1..K).product::<usize>() as f64;
        det.abs().sqrt() / factorial
    }

    /// Volume of boundary face b via the Gram determinant in the tangent space at vertex 0.
    ///
    /// For K=3 (edge): length. For K=4 (face): area of the triangular face.
    pub fn boundary_volume(&self, manifold: &M, b: usize) -> f64 {
        let boundary = &self.boundaries[b];
        let v0 = &self.vertices[boundary[0]];

        let n = B - 1;
        if n == 0 {
            // B=1 means boundary faces are single vertices, volume = 1.
            return 1.0;
        }

        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for &vi_idx in &boundary[1..] {
            let vi = &self.vertices[vi_idx];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let det = dense_determinant(&gram, n);
        let factorial = (1..B).product::<usize>().max(1) as f64;
        det.abs().sqrt() / factorial
    }

    /// Circumcenter of simplex s via the equidistance system in tangent space.
    ///
    /// Solves the system: for each edge vector e_i from v0, find barycentric
    /// coordinates t such that |c - v_i|^2 = |c - v0|^2 for all i, where
    /// c = v0 + sum_i t_i * e_i. This reduces to G * t = 0.5 * diag(G)
    /// where G is the Gram matrix. Falls back to the centroid for degenerate simplices.
    pub fn simplex_circumcenter(&self, manifold: &M, s: usize) -> M::Point {
        let simplex = &self.simplices[s];
        let v0 = &self.vertices[simplex[0]];

        let n = K - 1;
        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for &vi_idx in &simplex[1..] {
            let vi = &self.vertices[vi_idx];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = 0.5 * gram[i * n + i];
        }

        let t = match dense_solve(&gram, &rhs, n) {
            Some(sol) => sol,
            None => {
                // Degenerate: return centroid.
                let mut tangent = manifold.zero_tangent(v0);
                for log in &logs {
                    tangent = tangent + log.clone() * (1.0 / K as f64);
                }
                return manifold.exp(v0, &tangent);
            }
        };

        let mut tangent = manifold.zero_tangent(v0);
        for (i, ti) in t.iter().enumerate() {
            tangent = tangent + logs[i].clone() * *ti;
        }
        manifold.exp(v0, &tangent)
    }

    /// Circumcenter of boundary face b.
    ///
    /// For B=2 (edges): geodesic midpoint. For B=3 (triangular faces): circumcenter
    /// via tangent-space equidistance system.
    pub fn boundary_circumcenter(&self, manifold: &M, b: usize) -> M::Point {
        let boundary = &self.boundaries[b];
        let v0 = &self.vertices[boundary[0]];

        let n = B - 1;
        if n == 0 {
            return v0.clone();
        }
        if n == 1 {
            // Edge midpoint via geodesic.
            let v1 = &self.vertices[boundary[1]];
            let half_log = manifold
                .log(v0, v1)
                .map(|u| u * 0.5)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            return manifold.exp(v0, &half_log);
        }

        let mut logs: Vec<M::Tangent> = Vec::with_capacity(n);
        for &vi_idx in &boundary[1..] {
            let vi = &self.vertices[vi_idx];
            let u = manifold
                .log(v0, vi)
                .unwrap_or_else(|_| manifold.zero_tangent(v0));
            logs.push(u);
        }

        let mut gram = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                gram[i * n + j] = manifold.inner(v0, &logs[i], &logs[j]);
            }
        }

        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            rhs[i] = 0.5 * gram[i * n + i];
        }

        let t_coeff = match dense_solve(&gram, &rhs, n) {
            Some(sol) => sol,
            None => {
                let mut tangent = manifold.zero_tangent(v0);
                for log in &logs {
                    tangent = tangent + log.clone() * (1.0 / B as f64);
                }
                return manifold.exp(v0, &tangent);
            }
        };

        let mut tangent = manifold.zero_tangent(v0);
        for (i, ti) in t_coeff.iter().enumerate() {
            tangent = tangent + logs[i].clone() * *ti;
        }
        manifold.exp(v0, &tangent)
    }

    /// Rebuild all derived mesh data (boundaries, signs, adjacency) from the
    /// current `vertices` and `simplices` arrays.
    ///
    /// Call after mutations that change the simplex list (split, collapse, flip).
    /// This is equivalent to constructing a new mesh from the same vertices
    /// and simplices, but modifies in place.
    ///
    /// Only implemented for K=3, B=2 (triangle meshes).
    pub fn rebuild_topology(&mut self)
    where
        M: Manifold,
    {
        // Delegate to from_simplices logic: rebuild edges, signs, adjacency.
        // We take ownership of vertices and simplices, rebuild, and put them back.
        let vertices = std::mem::take(&mut self.vertices);
        let simplices = std::mem::take(&mut self.simplices);

        // Rebuild boundaries and signs from simplices (K=3, B=2 specialization).
        let mut edge_map: HashMap<[usize; B], usize> = HashMap::new();
        let mut boundaries: Vec<[usize; B]> = Vec::new();
        let mut simplex_boundary_ids = Vec::with_capacity(simplices.len());
        let mut boundary_signs_vec = Vec::with_capacity(simplices.len());

        for simplex in &simplices {
            let mut local_boundary_ids = [0usize; K];
            let mut local_signs = [0.0f64; K];

            for omit in 0..K {
                let sign = if omit % 2 == 0 { 1.0 } else { -1.0 };

                let mut face = [0usize; B];
                let mut idx = 0;
                for (pos, &v) in simplex.iter().enumerate() {
                    if pos != omit {
                        face[idx] = v;
                        idx += 1;
                    }
                }

                let mut sorted_face = face;
                sorted_face.sort();

                let parity = permutation_sign(&face, &sorted_face);
                let effective_sign = sign * parity;

                let boundary_idx = *edge_map.entry(sorted_face).or_insert_with(|| {
                    let b = boundaries.len();
                    boundaries.push(sorted_face);
                    b
                });

                local_boundary_ids[omit] = boundary_idx;
                local_signs[omit] = effective_sign;
            }

            simplex_boundary_ids.push(local_boundary_ids);
            boundary_signs_vec.push(local_signs);
        }

        self.vertices = vertices;
        self.simplices = simplices;
        self.boundaries = boundaries;
        self.simplex_boundary_ids = simplex_boundary_ids;
        self.boundary_signs = boundary_signs_vec;
        self.rebuild_adjacency();
    }

    /// Recompute all adjacency maps from the current `simplices`, `boundaries`,
    /// and `simplex_boundary_ids` arrays.
    ///
    /// Call after any mutation that changes the mesh topology (edge split,
    /// collapse, flip, etc.). This rebuilds `vertex_boundaries`,
    /// `vertex_simplices`, and `boundary_simplices` from scratch.
    pub fn rebuild_adjacency(&mut self) {
        let nv = self.vertices.len();
        let nb = self.boundaries.len();

        // vertex_boundaries: for each vertex, which boundaries contain it
        let mut vb: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (b, boundary) in self.boundaries.iter().enumerate() {
            for &v in boundary {
                vb[v].push(b);
            }
        }

        // vertex_simplices: for each vertex, which simplices contain it
        let mut vs: Vec<Vec<usize>> = vec![Vec::new(); nv];
        for (s, simplex) in self.simplices.iter().enumerate() {
            for &v in simplex {
                vs[v].push(s);
            }
        }

        // boundary_simplices: for each boundary face, which simplices are adjacent
        let mut bs: Vec<Vec<usize>> = vec![Vec::new(); nb];
        for (s, sbi) in self.simplex_boundary_ids.iter().enumerate() {
            for &b in sbi {
                bs[b].push(s);
            }
        }

        self.vertex_boundaries = vb;
        self.vertex_simplices = vs;
        self.boundary_simplices = bs;
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
            assert!(
                i < n_v && j < n_v && k < n_v,
                "triangle index out of bounds"
            );

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

        let mut mesh = Self {
            vertices,
            simplices: triangles,
            boundaries: edges,
            simplex_boundary_ids,
            boundary_signs,
            vertex_boundaries: Vec::new(),
            vertex_simplices: Vec::new(),
            boundary_simplices: Vec::new(),
            _phantom: PhantomData,
        };
        mesh.rebuild_adjacency();
        mesh
    }

    /// For triangle meshes, return the adjacent faces of edge `e`.
    ///
    /// An interior edge has exactly 2 co-faces: returns `(face_a, Some(face_b))`.
    /// A boundary edge has exactly 1 co-face: returns `(face_a, None)`.
    ///
    /// # Panics
    ///
    /// Panics if the edge is non-manifold (more than 2 co-faces) or has 0 co-faces.
    pub fn edge_faces(&self, e: usize) -> (usize, Option<usize>) {
        let cofaces = &self.boundary_simplices[e];
        match cofaces.len() {
            1 => (cofaces[0], None),
            2 => (cofaces[0], Some(cofaces[1])),
            n => panic!("non-manifold edge {e}: has {n} co-faces (expected 1 or 2)"),
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
        let mv: Vec<SVector<f64, 2>> = vertices
            .iter()
            .map(|&[x, y]| SVector::from([x, y]))
            .collect();
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
        let ux = (ac.y * (ab.x * ab.x + ab.y * ab.y) - ab.y * (ac.x * ac.x + ac.y * ac.y)) / d;
        let uy = (ab.x * (ac.x * ac.x + ac.y * ac.y) - ac.x * (ab.x * ab.x + ab.y * ab.y)) / d;
        a + Vector2::new(ux, uy)
    }

    /// Build a uniform triangulated grid on \[0,1\]^2 with `n` divisions per side.
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

    #[deprecated(since = "0.1.1", note = "use n_boundaries() instead")]
    pub fn n_edges(&self) -> usize {
        self.n_boundaries()
    }

    #[deprecated(since = "0.1.1", note = "use n_simplices() instead")]
    pub fn n_triangles(&self) -> usize {
        self.n_simplices()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense linear algebra helpers (module-private)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the determinant of an n x n matrix stored row-major in a flat slice.
///
/// Uses LU decomposition with partial pivoting. For n <= 3, uses direct formulas.
fn dense_determinant(a: &[f64], n: usize) -> f64 {
    match n {
        0 => 1.0,
        1 => a[0],
        2 => a[0] * a[3] - a[1] * a[2],
        3 => {
            a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6])
                + a[2] * (a[3] * a[7] - a[4] * a[6])
        }
        _ => {
            let mut lu: Vec<f64> = a.to_vec();
            let mut sign = 1.0f64;
            for col in 0..n {
                let mut max_val = lu[col * n + col].abs();
                let mut max_row = col;
                for row in (col + 1)..n {
                    let v = lu[row * n + col].abs();
                    if v > max_val {
                        max_val = v;
                        max_row = row;
                    }
                }
                if max_val < 1e-30 {
                    return 0.0;
                }
                if max_row != col {
                    for k in 0..n {
                        lu.swap(col * n + k, max_row * n + k);
                    }
                    sign = -sign;
                }
                let pivot = lu[col * n + col];
                for row in (col + 1)..n {
                    let factor = lu[row * n + col] / pivot;
                    lu[row * n + col] = factor;
                    for k in (col + 1)..n {
                        lu[row * n + k] -= factor * lu[col * n + k];
                    }
                }
            }
            let mut det = sign;
            for i in 0..n {
                det *= lu[i * n + i];
            }
            det
        }
    }
}

/// Solve A * x = b for an n x n system via Gaussian elimination with partial pivoting.
///
/// Returns `None` if the matrix is singular (pivot < 1e-30).
fn dense_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0f64; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        if max_row != col {
            for k in 0..(n + 1) {
                aug.swap(col * (n + 1) + k, max_row * (n + 1) + k);
            }
        }
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for k in col..(n + 1) {
                aug[row * (n + 1) + k] -= factor * aug[col * (n + 1) + k];
            }
        }
    }

    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }
    Some(x)
}

/// Compute the sign of the permutation that maps `from` to `to`.
///
/// Both slices must contain the same elements. Returns +1.0 for even
/// permutations and -1.0 for odd permutations.
fn permutation_sign<const N: usize>(from: &[usize; N], to: &[usize; N]) -> f64 {
    let mut perm = [0usize; N];
    for (i, &val) in from.iter().enumerate() {
        for (j, &tval) in to.iter().enumerate() {
            if val == tval {
                perm[i] = j;
                break;
            }
        }
    }
    let mut inversions = 0usize;
    for i in 0..N {
        for j in (i + 1)..N {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 { 1.0 } else { -1.0 }
}
