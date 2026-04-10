//! Extrinsic operators on embedded manifolds via tangent-plane projection.
//!
//! Implements the discretisation from Zhu, Saintillan, Chern (2025):
//! "Stokes flow of an evolving fluid film with arbitrary shape and topology."
//! arXiv:2407.14025, JFM 1003, R1.
//!
//! All operators bypass the intrinsic connection by working in the ambient
//! R^D and projecting to the tangent plane per simplex. This generalises
//! to k-manifolds embedded in R^D without discretising connection Laplacians
//! on tensor bundles.
//!
//! Currently implemented for triangle meshes (K=3) in R^3 (D=3).

use nalgebra::{Matrix3, SMatrix, SVector};
use sprs::{CsMat, TriMat};

use crate::mesh::Mesh;
use cartan_core::Manifold;

/// Ambient dimension for all extrinsic operators in this module.
const D: usize = 3;

// ─────────────────────────────────────────────────────────────────────────────
// Per-face geometric data
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed per-face geometric quantities for the extrinsic discretisation.
#[derive(Debug, Clone)]
pub struct FaceData {
    /// Face normals (unit, outward-pointing).
    pub normals: Vec<SVector<f64, D>>,
    /// Face areas.
    pub areas: Vec<f64>,
    /// Tangent projectors P = I - N*N^T, one per face.
    pub projectors: Vec<Matrix3<f64>>,
    /// FEM gradients of hat functions: `fem_grads[f][local_v]` is the gradient
    /// of the hat function at local vertex `local_v` of face `f`, projected
    /// to the tangent plane.
    pub fem_grads: Vec<[SVector<f64, D>; 3]>,
}

impl FaceData {
    /// Compute all per-face data from a triangle mesh embedded in R^3.
    ///
    /// Requires that the manifold's Point type is `SVector<f64, 3>`.
    pub fn from_mesh<M: Manifold<Point = SVector<f64, D>>>(mesh: &Mesh<M, 3, 2>) -> Self {
        let nf = mesh.n_simplices();
        let mut normals = Vec::with_capacity(nf);
        let mut areas = Vec::with_capacity(nf);
        let mut projectors = Vec::with_capacity(nf);
        let mut fem_grads = Vec::with_capacity(nf);

        for f in 0..nf {
            let [i0, i1, i2] = mesh.simplices[f];
            let v0 = mesh.vertices[i0];
            let v1 = mesh.vertices[i1];
            let v2 = mesh.vertices[i2];

            let e01 = v1 - v0;
            let e02 = v2 - v0;
            let cross = e01.cross(&e02);
            let area = 0.5 * cross.norm();
            let n = if cross.norm() > 1e-30 {
                cross / cross.norm()
            } else {
                SVector::<f64, D>::zeros()
            };

            // Tangent projector: P = I - n*n^T.
            let proj = Matrix3::identity() - n * n.transpose();

            // FEM gradient of hat function at vertex i in a triangle:
            // grad(phi_i) = (1/2A) * (N x e_opposite)
            // where e_opposite is the edge opposite to vertex i.
            let e12 = v2 - v1; // opposite to v0
            let e20 = v0 - v2; // opposite to v1
            let e01_opp = v1 - v0; // opposite to v2

            let inv_2a = if area > 1e-30 {
                1.0 / (2.0 * area)
            } else {
                0.0
            };

            let grad0 = inv_2a * n.cross(&e12);
            let grad1 = inv_2a * n.cross(&e20);
            let grad2 = inv_2a * n.cross(&e01_opp);

            normals.push(n);
            areas.push(area);
            projectors.push(proj);
            fem_grads.push([grad0, grad1, grad2]);
        }

        Self {
            normals,
            areas,
            projectors,
            fem_grads,
        }
    }
}

// No vertex_3d helper needed: mesh.vertices[i] is already SVector<f64, 3>.

// ─────────────────────────────────────────────────────────────────────────────
// Extrinsic operators as sparse matrices
// ─────────────────────────────────────────────────────────────────────────────

/// Assembled extrinsic operators for a triangle mesh in R^3.
///
/// Velocity is R^3-valued per vertex (3*nv DOFs).
/// Strain rate is symmetric 3x3 per face (stored as 6-component Voigt: xx, yy, zz, xy, xz, yz).
/// Pressure is scalar per vertex (nv DOFs).
#[derive(Debug, Clone)]
pub struct ExtrinsicOperators {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of faces.
    pub n_faces: usize,
    /// Divergence: maps 3*nv velocity to nv scalar (area change rate).
    pub div: CsMat<f64>,
    /// Gradient: maps nv scalar (pressure) to 3*nv velocity. GRAD = -DIV^T.
    pub grad: CsMat<f64>,
    /// Viscosity Laplacian: 3*nv -> 3*nv, symmetric negative semi-definite.
    /// L = K^T A^{-1} K where K is the Killing operator.
    pub viscosity_lap: CsMat<f64>,
    /// Face areas (for weighting).
    pub face_areas: Vec<f64>,
}

impl ExtrinsicOperators {
    /// Assemble all extrinsic operators from a triangle mesh embedded in R^3.
    pub fn from_mesh<M: Manifold<Point = SVector<f64, D>>>(mesh: &Mesh<M, 3, 2>) -> Self {
        let face_data = FaceData::from_mesh(mesh);
        let nv = mesh.n_vertices();
        let nf = mesh.n_simplices();

        // Build DIV as a sparse matrix (nv x 3*nv).
        // DIV(U)_v = sum_{f containing v} (1/2) * P_f * grad(phi_v) . U_f_avg
        // More precisely from paper 2 Eq. 3: DIV = tr(K).
        //
        // For each face f with vertices (v0, v1, v2), the contribution to
        // DIV at vertex vi from velocity component j at vertex vl is:
        // DIV[vi, 3*vl + j] += (face contribution from Killing trace).
        //
        // The Killing operator K maps velocity U to strain rate E.
        // K_{f,kj,l,i} = (1/2)(P^{ik} nabla_l^j + P^{ij} nabla_l^k)
        // DIV = tr_over_{kj}(K) means summing k=j:
        // DIV_{f,l,i} = sum_k K_{f,kk,l,i} = sum_k (1/2)(P^{ik} nabla_l^k + P^{ik} nabla_l^k)
        //             = sum_k P^{ik} nabla_l^k = (P * nabla_l)^i
        //
        // So DIV contribution at vertex v from velocity at vertex vl, component i:
        // = sum_{f containing both v and vl} (P_f * grad_f(phi_vl))^i * (something with dual area)
        //
        // Actually, paper 2's DIV maps velocity to scalar per vertex.
        // Let's build it directly from the definition.

        // Build the Killing operator K as a dense per-face contribution,
        // then assemble DIV and viscosity Laplacian.

        // Killing stiffness: for each face f, K_f maps [U_{v0}, U_{v1}, U_{v2}]
        // (9 DOFs) to the symmetric strain rate E_f (6 DOFs in Voigt).
        // We'll directly build the viscosity Laplacian L = sum_f (1/A_f) K_f^T K_f
        // and the divergence DIV = sum_f tr(K_f).

        let mut div_triplets = TriMat::new((nv, 3 * nv));
        let mut lap_triplets = TriMat::new((3 * nv, 3 * nv));

        for f in 0..nf {
            let simplex = &mesh.simplices[f];
            let proj = &face_data.projectors[f];
            let grads = &face_data.fem_grads[f];
            let area = face_data.areas[f];

            if area < 1e-30 {
                continue;
            }

            // Build the 6x9 Killing matrix K_f for this face.
            // Input: [U_{v0}^x, U_{v0}^y, U_{v0}^z, U_{v1}^x, ..., U_{v2}^z]
            // Output: [E^xx, E^yy, E^zz, E^xy, E^xz, E^yz] (Voigt notation)
            //
            // K_{kj,li} = (1/2)(P^{ik} nabla_l^j + P^{ij} nabla_l^k)
            // where l is local vertex index, i is spatial component of velocity.
            let mut k_f = SMatrix::<f64, 6, 9>::zeros();

            for (local_v, grad) in grads.iter().enumerate().take(3) {
                let col_offset = local_v * 3;

                for i in 0..3 {
                    // Voigt indices: xx=0, yy=1, zz=2, xy=3, xz=4, yz=5
                    // Diagonal entries: E^{kk} for k=0,1,2
                    for k in 0..3 {
                        // K_{kk,l,i} = (1/2)(P^{ik} grad_l^k + P^{ik} grad_l^k)
                        //            = P^{ik} grad_l^k
                        k_f[(k, col_offset + i)] += proj[(i, k)] * grad[k];
                    }

                    // Off-diagonal entries (with sqrt(2) Voigt scaling for proper inner product):
                    // E^{01} (xy), voigt index 3
                    // K_{01,l,i} = (1/2)(P^{i0} grad_l^1 + P^{i1} grad_l^0)
                    k_f[(3, col_offset + i)] +=
                        0.5 * (proj[(i, 0)] * grad[1] + proj[(i, 1)] * grad[0]);
                    // E^{02} (xz), voigt index 4
                    k_f[(4, col_offset + i)] +=
                        0.5 * (proj[(i, 0)] * grad[2] + proj[(i, 2)] * grad[0]);
                    // E^{12} (yz), voigt index 5
                    k_f[(5, col_offset + i)] +=
                        0.5 * (proj[(i, 1)] * grad[2] + proj[(i, 2)] * grad[1]);
                }
            }

            // Viscosity Laplacian contribution: L += (1/A_f) K_f^T K_f
            let ktk = k_f.transpose() * k_f * (1.0 / area);

            // Map local DOFs to global DOFs.
            for local_a in 0..3 {
                let va = simplex[local_a];
                for local_b in 0..3 {
                    let vb = simplex[local_b];
                    for ia in 0..3 {
                        for ib in 0..3 {
                            let val = ktk[(local_a * 3 + ia, local_b * 3 + ib)];
                            if val.abs() > 1e-30 {
                                lap_triplets.add_triplet(va * 3 + ia, vb * 3 + ib, val);
                            }
                        }
                    }
                }
            }

            // Divergence contribution: DIV_{v, 3*vl + i} += (P_f * grad_f(phi_vl))^i
            // But DIV maps to per-vertex scalars, so we need the trace of K_f.
            // tr(E) = E^xx + E^yy + E^zz = K_f[0,:] + K_f[1,:] + K_f[2,:]
            // This gives us DIV_f as a 1x9 row vector.
            // We distribute to vertices weighted by 1/3 of face area (barycentric).
            for &v in simplex.iter().take(3) {
                for (local_l, &vl) in simplex.iter().enumerate().take(3) {
                    for i in 0..3 {
                        let col = local_l * 3 + i;
                        let trace_val = k_f[(0, col)] + k_f[(1, col)] + k_f[(2, col)];
                        if trace_val.abs() > 1e-30 {
                            // Weight: area / 3 per vertex (dual cell contribution).
                            div_triplets.add_triplet(v, vl * 3 + i, trace_val * area / 3.0);
                        }
                    }
                }
            }
        }

        let div = div_triplets.to_csc();
        let viscosity_lap = lap_triplets.to_csc();

        // GRAD = -DIV^T
        let grad = div.transpose_view().to_csc().map(|&v| -v);

        Self {
            n_vertices: nv,
            n_faces: nf,
            div,
            grad,
            viscosity_lap,
            face_areas: face_data.areas,
        }
    }

    /// Apply the divergence operator to a velocity field.
    ///
    /// Input: R^3-valued velocity per vertex (length 3*nv).
    /// Output: scalar per vertex (length nv).
    pub fn apply_div(&self, velocity: &[f64]) -> Vec<f64> {
        assert_eq!(velocity.len(), 3 * self.n_vertices);
        sparse_matvec(&self.div, velocity)
    }

    /// Apply the gradient operator to a pressure field.
    ///
    /// Input: scalar per vertex (length nv).
    /// Output: R^3-valued per vertex (length 3*nv).
    pub fn apply_grad(&self, pressure: &[f64]) -> Vec<f64> {
        assert_eq!(pressure.len(), self.n_vertices);
        sparse_matvec(&self.grad, pressure)
    }

    /// Apply the viscosity Laplacian to a velocity field.
    ///
    /// Input: R^3-valued velocity per vertex (length 3*nv).
    /// Output: R^3-valued per vertex (length 3*nv).
    pub fn apply_viscosity_lap(&self, velocity: &[f64]) -> Vec<f64> {
        assert_eq!(velocity.len(), 3 * self.n_vertices);
        sparse_matvec(&self.viscosity_lap, velocity)
    }
}

/// Sparse matrix-vector multiply.
fn sparse_matvec(mat: &CsMat<f64>, x: &[f64]) -> Vec<f64> {
    let nrows = mat.rows();
    let mut y = vec![0.0; nrows];
    // CSC: iterate over columns.
    for (col, col_view) in mat.outer_iterator().enumerate() {
        let xc = x[col];
        if xc.abs() < 1e-30 {
            continue;
        }
        for (row, &val) in col_view.iter() {
            y[row] += val * xc;
        }
    }
    y
}
