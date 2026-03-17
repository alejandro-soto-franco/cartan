// ~/cartan/cartan-dec/src/exterior.rs

//! Discrete exterior derivative operators d0 and d1.
//!
//! The exterior derivative is a purely combinatorial operator: it encodes
//! the boundary map of the simplicial complex and is independent of the metric.
//!
//! ## Matrices
//!
//! - **d0**: n_edges x n_vertices. d0[e, v] = +1 if v is the head of edge e,
//!   -1 if v is the tail.
//!
//! - **d1**: n_triangles x n_edges. d1[t, e] = +1 if edge e is positively
//!   oriented relative to triangle t, -1 if negatively oriented.
//!
//! ## Exactness: d1 * d0 = 0
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 4.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003. Chapter 3.

use nalgebra::DMatrix;

use cartan_core::Manifold;

use crate::mesh::{FlatMesh, Mesh};

/// The discrete exterior derivative operators for a simplicial complex.
pub struct ExteriorDerivative {
    /// d0: n_edges x n_vertices. Maps 0-forms to 1-forms.
    pub d0: DMatrix<f64>,
    /// d1: n_triangles x n_edges. Maps 1-forms to 2-forms.
    pub d1: DMatrix<f64>,
}

impl ExteriorDerivative {
    /// Build d0 and d1 from a flat mesh. Delegates to `from_mesh_generic`.
    pub fn from_mesh(mesh: &FlatMesh) -> Self {
        Self::from_mesh_generic(mesh)
    }

    /// Generic version: accepts any `Mesh<M, 3, 2>`. Purely topological.
    pub fn from_mesh_generic<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // d0: n_edges x n_vertices
        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

        // d1: n_triangles x n_edges
        let mut d1 = DMatrix::<f64>::zeros(nt, ne);
        for (t, (local_e, local_s)) in mesh
            .simplex_boundary_ids
            .iter()
            .zip(mesh.boundary_signs.iter())
            .enumerate()
        {
            for k in 0..3 {
                d1[(t, local_e[k])] = local_s[k];
            }
        }

        Self { d0, d1 }
    }

    /// Verify that d1 * d0 = 0 (the fundamental identity of exterior calculus).
    ///
    /// Returns the maximum absolute entry of d1 * d0.
    pub fn check_exactness(&self) -> f64 {
        let prod = &self.d1 * &self.d0;
        prod.abs().max()
    }
}
