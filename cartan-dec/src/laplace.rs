// ~/cartan/cartan-dec/src/laplace.rs

//! Discrete Laplace-Beltrami, Bochner, and Lichnerowicz operators.
//!
//! ## Laplace-Beltrami (scalar)
//!
//! The scalar Laplacian on 0-forms is:
//!
//!   Δ₀ = ⋆₀⁻¹ d₀ᵀ diag(⋆₁) d₀
//!
//! This is the standard cotangent-weight Laplacian. For a uniform Cartesian
//! grid it reduces to the standard 5-point finite difference stencil.
//!
//! ## Bochner Laplacian (connection Laplacian on vector fields)
//!
//! The Bochner (or rough/connection) Laplacian on vector fields u is:
//!
//!   ∇*∇ u = -tr(∇²u)
//!
//! For a flat 2D domain, this reduces to the scalar Laplacian applied
//! component-wise: (Δ u)_i = Δ (u_i).
//!
//! For curved manifolds, the curvature correction R(u)_ij = Ric^j_k u^k
//! is added, giving the Lichnerowicz Laplacian (see below).
//!
//! ## Lichnerowicz Laplacian (on symmetric 2-tensors)
//!
//! For a symmetric 2-tensor Q_ij (e.g., the Q-tensor in nematodynamics):
//!
//!   ΔL Q = ∇*∇ Q + 2 R(Q)
//!
//! where R(Q)_ij = R_{ikjl} Q^{kl} + R_{jkil} Q^{kl} (curvature correction).
//!
//! On a flat domain R² (curvature = 0):
//!   ΔL Q_ij = Δ Q_ij   (scalar Laplacian applied entry-wise)
//!
//! The curvature correction requires the Riemann tensor from cartan-core.
//! We expose it as an optional `curvature_correction` callback.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003.
//! - Lichnerowicz. "Propagateurs et Commutateurs en Relativité Générale." 1961.

use core::marker::PhantomData;

use nalgebra::DVector;
use sprs::{CsMat, TriMat};

use cartan_core::Manifold;
use cartan_manifolds::euclidean::Euclidean;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::FlatMesh;

/// Assembled discrete differential operators for a mesh.
///
/// Generic over the manifold type `M`. All `apply_*` methods work for any M.
/// `from_mesh` is currently only implemented for `Euclidean<2>` (flat meshes).
pub struct Operators<M: Manifold = Euclidean<2>> {
    /// Scalar Laplace-Beltrami: n_vertices x n_vertices (sparse).
    pub laplace_beltrami: CsMat<f64>,
    /// Diagonal entries of star0 (dual cell areas, for mass matrix).
    pub mass0: DVector<f64>,
    /// Diagonal entries of star1 (for 1-form computations).
    pub mass1: DVector<f64>,
    /// Exterior derivative chain (kept for advection/divergence).
    pub ext: ExteriorDerivative,
    /// Hodge star diagonals (kept for user access).
    pub hodge: HodgeStar,
    _phantom: PhantomData<M>,
}

/// Assemble the scalar Laplace-Beltrami: star0_inv * d0^T * diag(star1) * d0.
///
/// All operations are sparse. The result is a sparse CSC matrix.
fn assemble_scalar_laplacian(ext: &ExteriorDerivative, hodge: &HodgeStar) -> CsMat<f64> {
    let d0 = ext.d0();
    let ne = hodge.star1.len();

    // Build diag(star1) as a sparse diagonal matrix.
    let mut star1_tri = TriMat::new((ne, ne));
    for e in 0..ne {
        let w = hodge.star1[e];
        if w.abs() > 1e-30 {
            star1_tri.add_triplet(e, e, w);
        }
    }
    let star1_diag = star1_tri.to_csc();

    // star1 * d0
    let star1_d0 = &star1_diag * d0;

    // d0^T * (star1 * d0)
    let d0t = d0.transpose_view();
    let d0t_star1_d0 = &d0t * &star1_d0;

    // star0_inv * (d0^T * star1 * d0)
    let nv = hodge.star0.len();
    let star0_inv = hodge.star0_inv();
    let mut star0_inv_tri = TriMat::new((nv, nv));
    for v in 0..nv {
        let w = star0_inv[v];
        if w.abs() > 1e-30 {
            star0_inv_tri.add_triplet(v, v, w);
        }
    }
    let star0_inv_diag = star0_inv_tri.to_csc();

    &star0_inv_diag * &d0t_star1_d0
}

impl Operators<Euclidean<2>> {
    /// Assemble all discrete operators from a flat mesh.
    pub fn from_mesh(mesh: &FlatMesh, manifold: &Euclidean<2>) -> Self {
        let ext = ExteriorDerivative::from_mesh(mesh);
        let hodge = HodgeStar::from_mesh(mesh, manifold);

        let laplace_beltrami = assemble_scalar_laplacian(&ext, &hodge);

        Self {
            laplace_beltrami,
            mass0: hodge.star0.clone(),
            mass1: hodge.star1.clone(),
            ext,
            hodge,
            _phantom: PhantomData,
        }
    }
}

impl<M: Manifold> Operators<M> {
    /// Apply the scalar Laplace-Beltrami operator to a 0-form (vertex field).
    ///
    /// Uses sparse matrix-vector product. The Laplacian is stored in CSC format,
    /// so `outer_iterator()` iterates over columns: y += A[:,j] * x[j].
    pub fn apply_laplace_beltrami(&self, f: &DVector<f64>) -> DVector<f64> {
        let n = f.len();
        let mut result = DVector::<f64>::zeros(n);
        // CSC outer_iterator: (col_idx, column_slice)
        for (col_idx, col) in self.laplace_beltrami.outer_iterator().enumerate() {
            let x_j = f[col_idx];
            for (row_idx, &val) in col.iter() {
                result[row_idx] += val * x_j;
            }
        }
        result
    }

    /// Apply the Bochner (connection) Laplacian to a vector field.
    ///
    /// Input `u` is a 2*n_v vector with [u_x[0..n_v], u_y[0..n_v]] layout
    /// (structure-of-arrays: x-components first, then y-components).
    ///
    /// `ricci_correction`: optional per-vertex Ricci tensor callback.
    /// Returns the 2×2 Ricci tensor at vertex v as `[[r00, r01], [r10, r11]]`.
    /// For flat R²: `None`. For Einstein manifolds (Ric = κ·g):
    /// `Some(&|_| [[κ, 0.0], [0.0, κ]])`.
    pub fn apply_bochner_laplacian(
        &self,
        u: &DVector<f64>,
        ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(u.len(), 2 * nv, "Bochner: u must have 2*n_v entries");

        let ux = u.rows(0, nv).into_owned();
        let uy = u.rows(nv, nv).into_owned();

        let mut lux = self.apply_laplace_beltrami(&ux);
        let mut luy = self.apply_laplace_beltrami(&uy);

        // Weitzenboeck correction: (nabla*nabla + Ric)(u)_v = Delta*u_v + Ric_v * u_v
        if let Some(ric) = ricci_correction {
            for v in 0..nv {
                let r = ric(v); // [[r00, r01], [r10, r11]]
                let ux_v = ux[v];
                let uy_v = uy[v];
                lux[v] += r[0][0] * ux_v + r[0][1] * uy_v;
                luy[v] += r[1][0] * ux_v + r[1][1] * uy_v;
            }
        }

        let mut result = DVector::<f64>::zeros(2 * nv);
        result.rows_mut(0, nv).copy_from(&lux);
        result.rows_mut(nv, nv).copy_from(&luy);
        result
    }

    /// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field Q.
    ///
    /// Input `q` is a 3*n_v vector with [Q_xx, Q_xy, Q_yy] layout
    /// (three independent components of a symmetric 2×2 tensor per vertex).
    ///
    /// `curvature_correction`: optional per-vertex curvature endomorphism callback.
    /// Returns a 3×3 matrix acting on the [Qxx, Qxy, Qyy] components at vertex v.
    /// For flat R²: `None`. For a space form with sectional curvature K=κ:
    /// `Some(&|_| [[2.*κ,0.,0.],[0.,2.*κ,0.],[0.,0.,2.*κ]])`.
    pub fn apply_lichnerowicz_laplacian(
        &self,
        q: &DVector<f64>,
        curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.rows();
        assert_eq!(q.len(), 3 * nv, "Lichnerowicz: q must have 3*n_v entries");

        let qxx = q.rows(0, nv).into_owned();
        let qxy = q.rows(nv, nv).into_owned();
        let qyy = q.rows(2 * nv, nv).into_owned();

        let mut lxx = self.apply_laplace_beltrami(&qxx);
        let mut lxy = self.apply_laplace_beltrami(&qxy);
        let mut lyy = self.apply_laplace_beltrami(&qyy);

        if let Some(curv) = curvature_correction {
            for v in 0..nv {
                let c = curv(v); // 3x3 acting on [qxx, qxy, qyy]
                let qx = qxx[v];
                let qm = qxy[v];
                let qy = qyy[v];
                lxx[v] += c[0][0] * qx + c[0][1] * qm + c[0][2] * qy;
                lxy[v] += c[1][0] * qx + c[1][1] * qm + c[1][2] * qy;
                lyy[v] += c[2][0] * qx + c[2][1] * qm + c[2][2] * qy;
            }
        }

        let mut result = DVector::<f64>::zeros(3 * nv);
        result.rows_mut(0, nv).copy_from(&lxx);
        result.rows_mut(nv, nv).copy_from(&lxy);
        result.rows_mut(2 * nv, nv).copy_from(&lyy);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_manifolds::euclidean::Euclidean;

    #[test]
    fn test_bochner_tensor_ricci_zero() {
        // Zero callback must match None
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let u = DVector::from_element(2 * nv, 0.5);

        let result_none = ops.apply_bochner_laplacian(&u, None);
        let result_zero = ops.apply_bochner_laplacian(&u, Some(&|_| [[0.0, 0.0], [0.0, 0.0]]));
        let diff = (&result_none - &result_zero).norm();
        assert!(
            diff < 1e-12,
            "zero callback should equal None: diff = {diff}"
        );
    }

    #[test]
    fn test_bochner_tensor_ricci_einstein() {
        // Einstein manifold Ric = kappa*I: tensor result must match manual scalar computation
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let kappa = 2.5;
        let u = DVector::from_fn(2 * nv, |i, _| i as f64 * 0.01);

        let result_tensor =
            ops.apply_bochner_laplacian(&u, Some(&|_| [[kappa, 0.0], [0.0, kappa]]));

        // Manual: Delta*u_x + kappa*u_x, Delta*u_y + kappa*u_y
        let ux = u.rows(0, nv).into_owned();
        let uy = u.rows(nv, nv).into_owned();
        let lux = ops.apply_laplace_beltrami(&ux) + &ux * kappa;
        let luy = ops.apply_laplace_beltrami(&uy) + &uy * kappa;
        let mut expected = DVector::zeros(2 * nv);
        expected.rows_mut(0, nv).copy_from(&lux);
        expected.rows_mut(nv, nv).copy_from(&luy);

        let diff = (&result_tensor - &expected).norm();
        assert!(
            diff < 1e-10,
            "Einstein tensor != scalar path: diff = {diff}"
        );
    }

    #[test]
    fn test_lichnerowicz_tensor_callback_zero() {
        // Zero callback must match None
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let q = DVector::from_element(3 * nv, 0.3);

        let result_none = ops.apply_lichnerowicz_laplacian(&q, None);
        let result_zero = ops.apply_lichnerowicz_laplacian(&q, Some(&|_| [[0.0_f64; 3]; 3]));
        let diff = (&result_none - &result_zero).norm();
        assert!(
            diff < 1e-12,
            "zero callback should match None: diff = {diff}"
        );
    }

    #[test]
    fn test_lichnerowicz_tensor_callback_diagonal() {
        // Diagonal callback with kappa matches manual computation
        let mesh = FlatMesh::unit_square_grid(4);
        let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
        let nv = mesh.n_vertices();
        let kappa = 1.0;
        let q = DVector::from_fn(3 * nv, |i, _| (i + 1) as f64 * 0.01);

        let c = [
            [2.0 * kappa, 0., 0.],
            [0., 2.0 * kappa, 0.],
            [0., 0., 2.0 * kappa],
        ];
        let result_tensor = ops.apply_lichnerowicz_laplacian(&q, Some(&|_| c));

        // Manual: Delta*q_xx + 2*kappa*q_xx, etc.
        let qxx = q.rows(0, nv).into_owned();
        let qxy = q.rows(nv, nv).into_owned();
        let qyy = q.rows(2 * nv, nv).into_owned();
        let lxx = ops.apply_laplace_beltrami(&qxx) + &qxx * (2.0 * kappa);
        let lxy = ops.apply_laplace_beltrami(&qxy) + &qxy * (2.0 * kappa);
        let lyy = ops.apply_laplace_beltrami(&qyy) + &qyy * (2.0 * kappa);
        let mut expected = DVector::zeros(3 * nv);
        expected.rows_mut(0, nv).copy_from(&lxx);
        expected.rows_mut(nv, nv).copy_from(&lxy);
        expected.rows_mut(2 * nv, nv).copy_from(&lyy);

        let diff = (&result_tensor - &expected).norm();
        assert!(
            diff < 1e-10,
            "diagonal tensor != scalar path: diff = {diff}"
        );
    }
}
