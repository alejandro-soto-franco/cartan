// ~/cartan/cartan-dec/src/hodge.rs

//! Discrete Hodge star operators ⋆₀, ⋆₁, ⋆₂.
//!
//! The Hodge star encodes the *metric* of the manifold: it converts primal
//! k-forms to dual (n-k)-forms. For a well-centered 2D mesh, all three Hodge
//! stars are diagonal, with entries equal to the ratio of dual to primal
//! simplex volumes.
//!
//! ## Formulas (flat 2D mesh)
//!
//! For a well-centered Delaunay mesh with circumcentric dual:
//!
//! - **⋆₀[v]** = (dual cell area at v) / 1
//!   The dual cell of a vertex v is the union of circumcenter-midpoint triangles
//!   surrounding v. Its area = (1/3) * Σ_{t ∋ v} area(t) (barycentric dual).
//!   For circumcentric dual: area = Σ contributions from adjacent triangles.
//!
//! - **⋆₁[e]** = (dual edge length) / (primal edge length)
//!   The dual edge of primal edge e is the segment connecting the circumcenters
//!   of the two triangles sharing e (or the circumcenter to the boundary midpoint
//!   for boundary edges).
//!
//! - **⋆₂[t]** = 1 / area(t)
//!   The dual of a 2-simplex is a 0-simplex (point, the circumcenter), so
//!   the ratio is 1/area(t).
//!
//! ## Barycentric vs circumcentric dual
//!
//! This implementation uses the **barycentric dual** (also called the Whitney
//! complex), where the dual cell area of vertex v is (1/3) * sum of adjacent
//! triangle areas. The dual edge length for edge (i,j) shared by triangles
//! t₁, t₂ is approximated by the distance between their circumcenters.
//!
//! The barycentric dual is always well-defined (positive weights for any
//! non-degenerate mesh), whereas the circumcentric dual requires well-centered
//! meshes for positivity. We use circumcentric for ⋆₁ (the only case where
//! sign matters for the Laplacian) and barycentric for ⋆₀.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 5.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis. Chapter 4.

use nalgebra::DVector;

use crate::mesh::Mesh;

/// Diagonal Hodge star operators for a 2D simplicial mesh.
///
/// All three operators are stored as `DVector<f64>` (the diagonal entries).
/// The full diagonal matrix can be constructed as `DMatrix::from_diagonal(star)`.
pub struct HodgeStar {
    /// ⋆₀: diagonal entries for the n_vertices × n_vertices Hodge star on 0-forms.
    pub star0: DVector<f64>,
    /// ⋆₁: diagonal entries for the n_edges × n_edges Hodge star on 1-forms.
    pub star1: DVector<f64>,
    /// ⋆₂: diagonal entries for the n_triangles × n_triangles Hodge star on 2-forms.
    pub star2: DVector<f64>,
}

impl HodgeStar {
    /// Compute the Hodge star diagonals from a 2D flat mesh.
    ///
    /// Uses barycentric dual for ⋆₀ and circumcentric dual for ⋆₁.
    /// ⋆₂[t] = 1 / area(t).
    ///
    /// All entries are guaranteed positive for non-degenerate meshes
    /// (no zero-area triangles and no zero-length edges).
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_edges();
        let nt = mesh.n_triangles();

        // ── ⋆₂: 1 / triangle area ──────────────────────────────────────────
        let mut star2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.triangle_area(t).abs();
            star2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // ── ⋆₀: barycentric dual cell area = (1/3) Σ_{t ∋ v} area(t) ───────
        let mut star0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.triangle_area(t).abs();
            for &v in &mesh.triangles[t] {
                star0[v] += area / 3.0;
            }
        }

        // ── ⋆₁: |dual edge| / |primal edge| ──────────────────────────────
        // Dual edge of primal edge e connects circumcenters of adjacent triangles.
        // For boundary edges, the dual edge endpoint is the edge midpoint.
        //
        // Build: for each edge, collect adjacent triangles.
        let mut edge_tris: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (t, local_e) in mesh.tri_edges.iter().enumerate() {
            for &e in local_e {
                edge_tris[e].push(t);
            }
        }

        let mut star1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.edge_length(e);
            if primal_len < 1e-30 {
                star1[e] = 0.0;
                continue;
            }

            let dual_len = match edge_tris[e].as_slice() {
                [t1, t2] => {
                    // Interior edge: dual connects circumcenters of t1 and t2.
                    let c1 = mesh.circumcenter(*t1);
                    let c2 = mesh.circumcenter(*t2);
                    (c2 - c1).norm()
                }
                [t1] => {
                    // Boundary edge: dual connects circumcenter to edge midpoint.
                    let c = mesh.circumcenter(*t1);
                    let mid = mesh.edge_midpoint(e);
                    (c - mid).norm()
                }
                _ => 0.0, // degenerate: no adjacent triangles
            };

            star1[e] = dual_len / primal_len;
        }

        Self {
            star0,
            star1,
            star2,
        }
    }

    /// Inverse Hodge star ⋆₀⁻¹ (entries are 1/star0[v]).
    pub fn star0_inv(&self) -> DVector<f64> {
        self.star0.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Inverse Hodge star ⋆₁⁻¹ (entries are 1/star1[e]).
    pub fn star1_inv(&self) -> DVector<f64> {
        self.star1.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }

    /// Inverse Hodge star ⋆₂⁻¹ (entries are 1/star2[t] = area(t)).
    pub fn star2_inv(&self) -> DVector<f64> {
        self.star2.map(|x| if x.abs() > 1e-30 { 1.0 / x } else { 0.0 })
    }
}
