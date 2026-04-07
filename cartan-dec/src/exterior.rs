// ~/cartan/cartan-dec/src/exterior.rs

//! Discrete exterior derivative operators.
//!
//! The exterior derivative is a purely combinatorial operator: it encodes
//! the boundary map of the simplicial complex and is independent of the metric.
//!
//! For a K-simplex mesh, the chain of exterior derivatives is:
//!   d[0]: n_boundaries x n_vertices   (0-forms -> 1-forms)
//!   d[1]: n_simplices x n_boundaries  (1-forms -> 2-forms)
//! For K=4 (tets), there would be d[2]: n_tets x n_faces as well.
//!
//! ## Exactness: d[k+1] * d[k] = 0
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341. Section 4.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003. Chapter 3.

use nalgebra::DMatrix;
use sprs::{CsMat, TriMat};

use cartan_core::Manifold;

use crate::mesh::{FlatMesh, Mesh};

/// The discrete exterior derivative operators for a simplicial complex.
///
/// Stores a chain of sparse incidence matrices `d[k]` for k = 0..(K-2).
/// For a triangle mesh (K=3): d[0] (edges x vertices) and d[1] (faces x edges).
pub struct ExteriorDerivative {
    /// d[k] maps k-cochains to (k+1)-cochains. Stored as CSC sparse matrices.
    pub d: Vec<CsMat<f64>>,
}

impl ExteriorDerivative {
    /// Build sparse exterior derivative operators from a flat mesh.
    /// Delegates to `from_mesh_sparse_generic`.
    pub fn from_mesh(mesh: &FlatMesh) -> Self {
        Self::from_mesh_sparse_generic(mesh)
    }

    /// Build sparse exterior derivative from any triangle mesh.
    pub fn from_mesh_sparse<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Self {
        Self::from_mesh_sparse_generic(mesh)
    }

    /// K-generic sparse construction. Builds d[0] and d[1] from the mesh topology.
    ///
    /// d[0]: n_boundaries x n_vertices. d0[b, v] = +1 if v is the head of
    /// boundary b, -1 if v is the tail.
    ///
    /// d[1]: n_simplices x n_boundaries. d1[t, b] = boundary_signs[t][k] for
    /// the k-th boundary face of simplex t.
    fn from_mesh_sparse_generic<M: Manifold, const K: usize, const B: usize>(
        mesh: &Mesh<M, K, B>,
    ) -> Self {
        let nv = mesh.n_vertices();
        let nb = mesh.n_boundaries();
        let ns = mesh.n_simplices();

        // d[0]: nb x nv
        // For each boundary face [v0, v1, ..., v_{B-1}], the boundary operator
        // assigns alternating signs: d0[b, v_k] = (-1)^k.
        let mut tri0 = TriMat::new((nb, nv));
        for (b, boundary) in mesh.boundaries.iter().enumerate() {
            for (k, &v) in boundary.iter().enumerate() {
                let sign = if k % 2 == 0 { -1.0 } else { 1.0 };
                tri0.add_triplet(b, v, sign);
            }
        }
        let d0 = tri0.to_csc();

        // d[1]: ns x nb
        // Uses the precomputed simplex_boundary_ids and boundary_signs.
        let mut tri1 = TriMat::new((ns, nb));
        for (s, (local_b, local_s)) in mesh
            .simplex_boundary_ids
            .iter()
            .zip(mesh.boundary_signs.iter())
            .enumerate()
        {
            for k in 0..K {
                tri1.add_triplet(s, local_b[k], local_s[k]);
            }
        }
        let d1 = tri1.to_csc();

        Self { d: vec![d0, d1] }
    }

    /// Build dense d0 and d1 from a triangle mesh.
    ///
    /// Retained for testing and backward compatibility. Prefer `from_mesh_sparse`.
    #[deprecated(since = "0.2.0", note = "use from_mesh_sparse or from_mesh instead")]
    pub fn from_mesh_generic_dense<M: Manifold>(
        mesh: &Mesh<M, 3, 2>,
    ) -> (DMatrix<f64>, DMatrix<f64>) {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

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

        (d0, d1)
    }

    /// Backward-compatible accessor: d0 (0-forms to 1-forms).
    pub fn d0(&self) -> &CsMat<f64> {
        &self.d[0]
    }

    /// Backward-compatible accessor: d1 (1-forms to 2-forms).
    pub fn d1(&self) -> &CsMat<f64> {
        &self.d[1]
    }

    /// Number of exterior derivative operators in the chain.
    pub fn degree(&self) -> usize {
        self.d.len()
    }

    /// Verify the exactness property d[k+1] * d[k] = 0 for all k.
    ///
    /// Returns the maximum absolute entry across all products.
    pub fn check_exactness(&self) -> f64 {
        let mut max_err = 0.0f64;
        for k in 0..self.d.len().saturating_sub(1) {
            let prod = &self.d[k + 1] * &self.d[k];
            for &val in prod.data() {
                max_err = max_err.max(val.abs());
            }
        }
        max_err
    }
}
