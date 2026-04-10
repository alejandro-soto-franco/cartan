// ~/cartan/cartan-dec/src/hodge.rs

//! Discrete Hodge star operators.
//!
//! The Hodge star encodes the metric. For a well-centered mesh, all Hodge
//! stars are diagonal with entries equal to dual/primal simplex volume ratios.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 5.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis. Chapter 4.

use nalgebra::DVector;

use cartan_core::Manifold;
use cartan_manifolds::euclidean::Euclidean;

use crate::error::DecError;
use crate::mesh::{FlatMesh, Mesh};

/// Diagonal Hodge star operators for a simplicial mesh.
///
/// `star[k]` is the diagonal of the Hodge star for k-forms. For a triangle
/// mesh (n=2): star[0] (vertices), star[1] (edges), star[2] (faces).
pub struct HodgeStar {
    /// star[k] contains the diagonal entries of the k-form Hodge star.
    pub star: Vec<DVector<f64>>,
}

impl HodgeStar {
    /// Compute the Hodge star from a flat 2D mesh (fast path).
    ///
    /// Uses flat metric methods (`triangle_area_flat`, `edge_length_flat`,
    /// `circumcenter_flat`, `edge_midpoint`).
    pub fn from_mesh(mesh: &FlatMesh, _manifold: &Euclidean<2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star2: 1 / triangle area
        let mut s2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            s2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // star0: barycentric dual cell area = (1/3) * sum_{t containing v} area(t)
        let mut s0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            for &v in &mesh.simplices[t] {
                s0[v] += area / 3.0;
            }
        }

        // star1: |dual edge| / |primal edge|
        let mut s1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.edge_length_flat(e);
            if primal_len < 1e-30 {
                s1[e] = 0.0;
                continue;
            }
            let cofaces = &mesh.boundary_simplices[e];
            let dual_len = match cofaces.len() {
                2 => {
                    let c1 = mesh.circumcenter_flat(cofaces[0]);
                    let c2 = mesh.circumcenter_flat(cofaces[1]);
                    (c2 - c1).norm()
                }
                1 => {
                    let c = mesh.circumcenter_flat(cofaces[0]);
                    let mid = mesh.edge_midpoint(e);
                    (c - mid).norm()
                }
                _ => 0.0,
            };
            s1[e] = dual_len / primal_len;
        }

        Self {
            star: vec![s0, s1, s2],
        }
    }

    /// K-generic Hodge star via circumcentric duality.
    ///
    /// For each k-simplex sigma, star_k[sigma] = vol(dual(sigma)) / vol(sigma).
    ///
    /// Currently implemented for K=3 (triangle meshes on any manifold).
    /// Uses barycentric dual for star0, circumcentric dual edge length ratio
    /// for star1, and reciprocal area for star2.
    pub fn from_mesh_generic<M: Manifold, const K: usize, const B: usize>(
        mesh: &Mesh<M, K, B>,
        manifold: &M,
    ) -> Result<Self, DecError> {
        assert_eq!(K, 3, "from_mesh_generic currently supports K=3 only");
        assert_eq!(B, 2, "from_mesh_generic currently supports B=2 only");

        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star2: 1 / simplex area
        let mut s2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.simplex_volume(manifold, t);
            s2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // star0: barycentric dual cell area = (1/3) * sum_{t containing v} area(t)
        let mut s0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.simplex_volume(manifold, t);
            for &v in &mesh.simplices[t] {
                s0[v] += area / 3.0;
            }
        }

        // star1: |dual edge| / |primal edge|
        let mut s1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.boundary_volume(manifold, e);
            if primal_len < 1e-30 {
                s1[e] = 0.0;
                continue;
            }
            let cofaces = &mesh.boundary_simplices[e];
            let dual_len = match cofaces.len() {
                2 => {
                    let c1 = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let c2 = mesh.simplex_circumcenter(manifold, cofaces[1]);
                    manifold.dist(&c1, &c2).unwrap_or(0.0)
                }
                1 => {
                    let c = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let mid = mesh.boundary_circumcenter(manifold, e);
                    manifold.dist(&c, &mid).unwrap_or(0.0)
                }
                _ => 0.0,
            };
            s1[e] = dual_len / primal_len;
        }

        Ok(Self {
            star: vec![s0, s1, s2],
        })
    }

    /// Circumcentric Hodge star (requires a well-centered mesh).
    ///
    /// star0 uses circumcentric dual cell areas (sum of sub-triangle areas
    /// formed by circumcenters and edge midpoints). This preserves
    /// primal-dual orthogonality, which the barycentric dual does not.
    ///
    /// star1 and star2 are identical to `from_mesh_generic`.
    ///
    /// Returns `Err` if the mesh is not well-centered.
    pub fn from_mesh_circumcentric<M: Manifold>(
        mesh: &Mesh<M, 3, 2>,
        manifold: &M,
    ) -> Result<Self, DecError> {

        mesh.check_well_centered(manifold)?;

        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star0: circumcentric dual cell area.
        // For each triangle, the circumcenter C and edge midpoints form sub-triangles.
        // Each vertex v gets the area of the sub-quadrilateral (midpoint_prev, C, midpoint_next, v).
        let mut s0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let cc = mesh.simplex_circumcenter(manifold, t);
            let simplex = &mesh.simplices[t];

            for local in 0..3 {
                let v = simplex[local];
                let v_prev = simplex[(local + 2) % 3];
                let v_next = simplex[(local + 1) % 3];

                let mid_prev = geodesic_midpoint(manifold, &mesh.vertices[v], &mesh.vertices[v_prev]);
                let mid_next = geodesic_midpoint(manifold, &mesh.vertices[v], &mesh.vertices[v_next]);

                // Sub-quad area = area(v, mid_prev, cc) + area(v, cc, mid_next).
                let area1 = tangent_triangle_area(manifold, &mesh.vertices[v], &mid_prev, &cc);
                let area2 = tangent_triangle_area(manifold, &mesh.vertices[v], &cc, &mid_next);
                s0[v] += area1 + area2;
            }
        }

        // star1 and star2: same logic as from_mesh_generic.
        let mut s2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.simplex_volume(manifold, t);
            s2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        let mut s1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.boundary_volume(manifold, e);
            if primal_len < 1e-30 {
                s1[e] = 0.0;
                continue;
            }
            let cofaces = &mesh.boundary_simplices[e];
            let dual_len = match cofaces.len() {
                2 => {
                    let c1 = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let c2 = mesh.simplex_circumcenter(manifold, cofaces[1]);
                    manifold.dist(&c1, &c2).unwrap_or(0.0)
                }
                1 => {
                    let c = mesh.simplex_circumcenter(manifold, cofaces[0]);
                    let mid = mesh.boundary_circumcenter(manifold, e);
                    manifold.dist(&c, &mid).unwrap_or(0.0)
                }
                _ => 0.0,
            };
            s1[e] = dual_len / primal_len;
        }

        Ok(Self {
            star: vec![s0, s1, s2],
        })
    }

    /// Access the k-form Hodge star diagonal.
    pub fn star_k(&self, k: usize) -> &DVector<f64> {
        &self.star[k]
    }

    /// Inverse Hodge star for k-forms (element-wise reciprocal).
    pub fn star_k_inv(&self, k: usize) -> DVector<f64> {
        self.star[k].map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Backward-compatible accessor: star0 (0-form Hodge star diagonal).
    pub fn star0(&self) -> &DVector<f64> {
        &self.star[0]
    }

    /// Backward-compatible accessor: star1 (1-form Hodge star diagonal).
    pub fn star1(&self) -> &DVector<f64> {
        &self.star[1]
    }

    /// Backward-compatible accessor: star2 (2-form Hodge star diagonal).
    pub fn star2(&self) -> &DVector<f64> {
        &self.star[2]
    }

    /// Backward-compatible inverse: star0_inv.
    pub fn star0_inv(&self) -> DVector<f64> {
        self.star_k_inv(0)
    }

    /// Backward-compatible inverse: star1_inv.
    pub fn star1_inv(&self) -> DVector<f64> {
        self.star_k_inv(1)
    }

    /// Backward-compatible inverse: star2_inv.
    pub fn star2_inv(&self) -> DVector<f64> {
        self.star_k_inv(2)
    }
}

/// Geodesic midpoint of two points on a manifold.
fn geodesic_midpoint<M: Manifold>(manifold: &M, p: &M::Point, q: &M::Point) -> M::Point {
    let log = manifold
        .log(p, q)
        .unwrap_or_else(|_| manifold.zero_tangent(p));
    manifold.exp(p, &(log * 0.5))
}

/// Area of a triangle (a, b, c) computed via the Gram determinant in the tangent space at a.
///
/// area = 0.5 * sqrt(|ab|^2 |ac|^2 - <ab, ac>^2)
fn tangent_triangle_area<M: Manifold>(
    manifold: &M,
    a: &M::Point,
    b: &M::Point,
    c: &M::Point,
) -> f64 {
    let ab = manifold
        .log(a, b)
        .unwrap_or_else(|_| manifold.zero_tangent(a));
    let ac = manifold
        .log(a, c)
        .unwrap_or_else(|_| manifold.zero_tangent(a));

    let ab2 = manifold.inner(a, &ab, &ab);
    let ac2 = manifold.inner(a, &ac, &ac);
    let abac = manifold.inner(a, &ab, &ac);
    let det = (ab2 * ac2 - abac * abac).max(0.0);
    0.5 * det.sqrt()
}
