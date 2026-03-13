// ~/cartan/cartan-dec/src/exterior.rs

//! Discrete exterior derivative operators d₀ and d₁.
//!
//! The exterior derivative is a purely *combinatorial* operator: it encodes
//! the boundary map of the simplicial complex and is independent of the metric.
//! On a k-form (a real-valued function on k-simplices), d_k maps to a (k+1)-form.
//!
//! ## Matrices
//!
//! - **d₀**: n_edges × n_vertices. d₀[e, v] = +1 if v is the head of edge e,
//!   -1 if v is the tail. Acts on 0-forms (vertex scalars).
//!
//! - **d₁**: n_triangles × n_edges. d₁[t, e] = +1 if edge e is positively
//!   oriented relative to triangle t, -1 if negatively oriented.
//!   Acts on 1-forms (edge scalars).
//!
//! ## Exactness: d₁ ∘ d₀ = 0
//!
//! This is the fundamental identity of DEC: the boundary of a boundary is empty.
//! Numerically, `d1 * d0` should be exactly the zero matrix.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 4.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003. Chapter 3.

use nalgebra::DMatrix;

use crate::mesh::Mesh;

/// The discrete exterior derivative operators for a simplicial complex.
///
/// Constructed once from a `Mesh` and reused for all operator assemblies.
pub struct ExteriorDerivative {
    /// d₀: n_edges × n_vertices. Maps 0-forms → 1-forms.
    pub d0: DMatrix<f64>,
    /// d₁: n_triangles × n_edges. Maps 1-forms → 2-forms.
    pub d1: DMatrix<f64>,
}

impl ExteriorDerivative {
    /// Build d₀ and d₁ from a mesh.
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_edges();
        let nt = mesh.n_triangles();

        // d₀: n_edges × n_vertices
        // For edge e = [i, j] with i < j: d₀[e, j] = +1, d₀[e, i] = -1.
        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.edges.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

        // d₁: n_triangles × n_edges
        // For triangle t with edge list [e0, e1, e2] and signs [s0, s1, s2]:
        // d₁[t, e_k] = s_k.
        let mut d1 = DMatrix::<f64>::zeros(nt, ne);
        for (t, (local_e, local_s)) in mesh.tri_edges.iter().zip(mesh.tri_edge_signs.iter()).enumerate() {
            for k in 0..3 {
                d1[(t, local_e[k])] = local_s[k];
            }
        }

        Self { d0, d1 }
    }

    /// Verify that d₁ ∘ d₀ = 0 (the fundamental identity of exterior calculus).
    ///
    /// Returns the maximum absolute entry of d₁ * d₀.
    pub fn check_exactness(&self) -> f64 {
        let prod = &self.d1 * &self.d0;
        prod.abs().max()
    }
}
