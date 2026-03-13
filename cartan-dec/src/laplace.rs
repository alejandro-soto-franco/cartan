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
//! The assembled matrix L is:
//!   L[i, i] = Σ_j w_ij       (sum of cotangent weights, positive)
//!   L[i, j] = -w_ij           (negative weight, off-diagonal)
//!
//! where w_ij = ⋆₁\[e_ij\] / primal_len^... — actually the full formula comes
//! from the matrix product above.
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

use nalgebra::{DMatrix, DVector};

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::Mesh;

/// Assembled discrete differential operators for a mesh.
///
/// The operators are assembled once from the mesh and reused for all
/// field evaluations. Store the `Operators` and call `apply_*` methods.
pub struct Operators {
    /// Scalar Laplace-Beltrami: n_vertices × n_vertices.
    pub laplace_beltrami: DMatrix<f64>,
    /// Diagonal entries of ⋆₀ (dual cell areas, for mass matrix).
    pub mass0: DVector<f64>,
    /// Diagonal entries of ⋆₁ (for 1-form computations).
    pub mass1: DVector<f64>,
    /// Exterior derivative d₀ and d₁ (kept for advection/divergence).
    pub ext: ExteriorDerivative,
    /// Hodge star diagonals (kept for user access).
    pub hodge: HodgeStar,
}

impl Operators {
    /// Assemble all discrete operators from a mesh.
    ///
    /// Cost: O(n_v² + n_e² + n_t²) for the matrix products.
    /// Worthwhile since operators are reused across many time steps.
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let ext = ExteriorDerivative::from_mesh(mesh);
        let hodge = HodgeStar::from_mesh(mesh);

        // L = ⋆₀⁻¹ * d₀ᵀ * diag(⋆₁) * d₀
        // = diag(1/star0) * d0^T * diag(star1) * d0
        let star1_d0 = DMatrix::from_diagonal(&hodge.star1) * &ext.d0;
        let d0t_star1_d0 = ext.d0.transpose() * star1_d0;
        let star0_inv = hodge.star0_inv();
        let laplace_beltrami = DMatrix::from_diagonal(&star0_inv) * d0t_star1_d0;

        Self {
            laplace_beltrami,
            mass0: hodge.star0.clone(),
            mass1: hodge.star1.clone(),
            ext,
            hodge,
        }
    }

    /// Apply the scalar Laplace-Beltrami operator to a 0-form (vertex field).
    ///
    /// Returns Δf at each vertex. For an interior vertex, this is:
    ///   (Δf)\[v\] = (1/A_v) Σ_{e ∋ v} w_e (f\[w\] - f\[v\])
    /// where A_v is the dual cell area and w_e is the Hodge weight of edge e.
    pub fn apply_laplace_beltrami(&self, f: &DVector<f64>) -> DVector<f64> {
        &self.laplace_beltrami * f
    }

    /// Apply the Bochner (connection) Laplacian to a vector field.
    ///
    /// For a flat domain, this is the scalar Laplacian applied component-wise.
    /// The input `u` is a 2*n_v vector with [u_x[0..n_v], u_y[0..n_v]] layout
    /// (structure-of-arrays: x-components first, then y-components).
    ///
    /// For curved manifolds with a Ricci curvature correction, pass
    /// `ricci_correction` as Some(f64 * identity) for Einstein manifolds.
    pub fn apply_bochner_laplacian(
        &self,
        u: &DVector<f64>,
        ricci_correction: Option<f64>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.nrows();
        assert_eq!(u.len(), 2 * nv, "Bochner: u must have 2*n_v entries");

        let ux = u.rows(0, nv);
        let uy = u.rows(nv, nv);

        let mut lux = &self.laplace_beltrami * ux;
        let mut luy = &self.laplace_beltrami * uy;

        // On an Einstein manifold with Ric = κ * g, the Bochner Laplacian
        // gains a correction: (∇*∇ + Ric)(u) = Δu + κ * u.
        // This is the Weitzenböck identity: ΔH = ∇*∇ + Ric.
        if let Some(kappa) = ricci_correction {
            lux += ux * kappa;
            luy += uy * kappa;
        }

        let mut result = DVector::<f64>::zeros(2 * nv);
        result.rows_mut(0, nv).copy_from(&lux);
        result.rows_mut(nv, nv).copy_from(&luy);
        result
    }

    /// Apply the Lichnerowicz Laplacian to a symmetric 2-tensor field Q.
    ///
    /// The input `q` is a 3*n_v vector with [Q_xx, Q_xy, Q_yy] layout
    /// (three independent components of a symmetric 2×2 tensor per vertex).
    ///
    /// For flat domains: ΔL Q_ij = Δ Q_ij (component-wise scalar Laplacian).
    ///
    /// `curvature_correction`: optional constant κ for the curvature term
    /// R_{ikjl} Q^{kl} = κ * Q_ij (valid on space forms with constant sectional K=κ).
    /// For R², κ = 0.
    pub fn apply_lichnerowicz_laplacian(
        &self,
        q: &DVector<f64>,
        curvature_correction: Option<f64>,
    ) -> DVector<f64> {
        let nv = self.laplace_beltrami.nrows();
        assert_eq!(q.len(), 3 * nv, "Lichnerowicz: q must have 3*n_v entries");

        let qxx = q.rows(0, nv);
        let qxy = q.rows(nv, nv);
        let qyy = q.rows(2 * nv, nv);

        let mut lxx = &self.laplace_beltrami * qxx;
        let mut lxy = &self.laplace_beltrami * qxy;
        let mut lyy = &self.laplace_beltrami * qyy;

        // Curvature correction: +2 * κ * Q for symmetric space form.
        // For flat R²: κ = 0. For S²: κ = 1.
        if let Some(kappa) = curvature_correction {
            lxx += qxx * (2.0 * kappa);
            lxy += qxy * (2.0 * kappa);
            lyy += qyy * (2.0 * kappa);
        }

        let mut result = DVector::<f64>::zeros(3 * nv);
        result.rows_mut(0, nv).copy_from(&lxx);
        result.rows_mut(nv, nv).copy_from(&lxy);
        result.rows_mut(2 * nv, nv).copy_from(&lyy);
        result
    }
}
