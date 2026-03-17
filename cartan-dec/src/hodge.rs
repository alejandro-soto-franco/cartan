// ~/cartan/cartan-dec/src/hodge.rs

//! Discrete Hodge star operators star0, star1, star2.
//!
//! The Hodge star encodes the metric. For a well-centered 2D mesh, all three
//! Hodge stars are diagonal with entries equal to dual/primal simplex volume ratios.
//!
//! ## Formulas (flat 2D mesh)
//!
//! - **star0[v]** = dual cell area = (1/3) * sum_{t containing v} area(t) (barycentric dual).
//! - **star1[e]** = |dual edge| / |primal edge|.
//! - **star2[t]** = 1 / area(t).
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 5.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis. Chapter 4.

use nalgebra::DVector;

use cartan_manifolds::euclidean::Euclidean;

use crate::mesh::FlatMesh;

/// Diagonal Hodge star operators for a 2D simplicial mesh.
pub struct HodgeStar {
    /// star0: diagonal entries for the n_vertices x n_vertices Hodge star on 0-forms.
    pub star0: DVector<f64>,
    /// star1: diagonal entries for the n_boundaries Hodge star on 1-forms.
    pub star1: DVector<f64>,
    /// star2: diagonal entries for the n_simplices Hodge star on 2-forms.
    pub star2: DVector<f64>,
}

impl HodgeStar {
    /// Compute the Hodge star diagonals from a flat 2D mesh.
    ///
    /// Uses flat metric methods (`triangle_area_flat`, `edge_length_flat`,
    /// `circumcenter_flat`, `edge_midpoint`) on the `FlatMesh` specialization.
    /// The `_manifold` parameter is accepted for API uniformity but not used
    /// (flat metric methods are called directly on the mesh).
    pub fn from_mesh(mesh: &FlatMesh, _manifold: &Euclidean<2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // star2: 1 / triangle area
        let mut star2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            star2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // star0: barycentric dual cell area = (1/3) * sum_{t containing v} area(t)
        let mut star0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            for &v in &mesh.simplices[t] {
                star0[v] += area / 3.0;
            }
        }

        // star1: |dual edge| / |primal edge|
        // Dual edge of interior primal edge e connects circumcenters of adjacent triangles.
        // For boundary edges, the dual endpoint is the edge midpoint.
        let mut edge_tris: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (t, local_e) in mesh.simplex_boundary_ids.iter().enumerate() {
            for &e in local_e {
                edge_tris[e].push(t);
            }
        }

        let mut star1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.edge_length_flat(e);
            if primal_len < 1e-30 {
                star1[e] = 0.0;
                continue;
            }
            let dual_len = match edge_tris[e].as_slice() {
                [t1, t2] => {
                    let c1 = mesh.circumcenter_flat(*t1);
                    let c2 = mesh.circumcenter_flat(*t2);
                    (c2 - c1).norm()
                }
                [t1] => {
                    let c = mesh.circumcenter_flat(*t1);
                    let mid = mesh.edge_midpoint(e);
                    (c - mid).norm()
                }
                _ => 0.0,
            };
            star1[e] = dual_len / primal_len;
        }

        Self { star0, star1, star2 }
    }

    /// Inverse Hodge star star0_inv (entries are 1/star0[v]).
    pub fn star0_inv(&self) -> DVector<f64> {
        self.star0.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Inverse Hodge star star1_inv (entries are 1/star1[b] for each boundary b).
    pub fn star1_inv(&self) -> DVector<f64> {
        self.star1.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Inverse Hodge star star2_inv (entries are 1/star2[t] = simplex volume).
    pub fn star2_inv(&self) -> DVector<f64> {
        self.star2.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }
}
