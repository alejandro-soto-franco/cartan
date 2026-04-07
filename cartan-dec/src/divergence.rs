// ~/cartan/cartan-dec/src/divergence.rs

//! Discrete covariant divergence of a vector field.
//!
//! The divergence of a vector field u on a 2D domain is computed via DEC as:
//!
//!   div(u) = ⋆₀⁻¹ d₀ᵀ ⋆₁ û
//!
//! where û is the 1-form dual to u (obtained by applying the metric to u),
//! and the discrete codifferential δ₁ = ⋆₀⁻¹ d₀ᵀ ⋆₁ is the adjoint of d₀.
//!
//! ## Discrete 1-form from a vector field
//!
//! Given a vertex-based vector field u, we build the 1-form û by
//! integrating u along each edge:
//!
//!   û\[e\] = (u\[i\] + u\[j\]) / 2 · (v_j - v_i)
//!
//! (trapezoidal approximation of the line integral of u · dl along edge e = \[i,j\]).
//!
//! ## Codifferential
//!
//! The codifferential δ = ⋆ d ⋆ is the formal adjoint of d with respect to
//! the L² inner product weighted by the Hodge star. For 1-forms on surfaces:
//!
//!   δ₁ = -⋆₀⁻¹ d₀ᵀ ⋆₁
//!
//! The sign depends on convention; we use the convention where
//! Δ = dδ + δd (positive semi-definite Laplacian).
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003.

use nalgebra::DVector;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::FlatMesh;

/// Compute the discrete divergence of a vertex-based vector field.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh.
/// - `ext`: precomputed exterior derivative operators.
/// - `hodge`: precomputed Hodge star operators.
/// - `u`: vector field at vertices, stored as [u_x[0..n_v], u_y[0..n_v]] (2*n_v vector).
///
/// # Returns
///
/// div(u) at each vertex as an n_v vector.
pub fn apply_divergence(
    mesh: &FlatMesh,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    u: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    let ne = mesh.n_boundaries();
    assert_eq!(u.len(), 2 * nv, "divergence: u must have 2*n_v entries");

    // Step 1: Build the 1-form û from u.
    // û[e] = avg(u[i], u[j]) · (v_j - v_i)  for edge e = [i, j].
    let mut u1form = DVector::<f64>::zeros(ne);
    for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
        let vi = mesh.vertex(i);
        let vj = mesh.vertex(j);
        let edge_vec = vj - vi;

        // Average velocity at the edge midpoint.
        let ux_avg = 0.5 * (u[i] + u[j]);
        let uy_avg = 0.5 * (u[nv + i] + u[nv + j]);

        u1form[e] = ux_avg * edge_vec.x + uy_avg * edge_vec.y;
    }

    // Step 2: Apply ⋆₁ to the 1-form.
    let star1_u1form = u1form.component_mul(&hodge.star1);

    // Step 3: Apply d₀ᵀ: sparse transpose-multiply.
    let d0t = ext.d0().transpose_view();
    let mut d0t_star1_u = DVector::<f64>::zeros(nv);
    for (row_idx, row) in d0t.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row.iter() {
            sum += val * star1_u1form[col_idx];
        }
        d0t_star1_u[row_idx] = sum;
    }

    // Step 4: Apply ⋆₀⁻¹.
    let star0_inv = hodge.star0_inv();
    d0t_star1_u.component_mul(&star0_inv)
}

/// Compute the discrete divergence of a symmetric 2-tensor field.
///
/// For a symmetric 2-tensor field T with components [T_xx, T_xy, T_yy] at
/// each vertex, the divergence is a vector field:
///
///   (div T)_x = ∂T_xx/∂x + ∂T_xy/∂y
///   (div T)_y = ∂T_xy/∂x + ∂T_yy/∂y
///
/// We compute this by treating each column of T as a separate vector field
/// and taking the divergence of each.
///
/// # Arguments
///
/// - `t`: symmetric 2-tensor at vertices, stored as [T_xx, T_xy, T_yy] (3*n_v vector).
///
/// # Returns
///
/// div(T) at each vertex as a 2*n_v vector [div_x, div_y].
pub fn apply_tensor_divergence(
    mesh: &FlatMesh,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    t: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(
        t.len(),
        3 * nv,
        "tensor_divergence: t must have 3*n_v entries"
    );

    let txx = t.rows(0, nv).into_owned();
    let txy = t.rows(nv, nv).into_owned();
    let tyy = t.rows(2 * nv, nv).into_owned();

    // First column of T: (T_xx, T_xy) — treated as a vector field.
    let mut col1 = DVector::<f64>::zeros(2 * nv);
    col1.rows_mut(0, nv).copy_from(&txx);
    col1.rows_mut(nv, nv).copy_from(&txy);

    // Second column of T: (T_xy, T_yy).
    let mut col2 = DVector::<f64>::zeros(2 * nv);
    col2.rows_mut(0, nv).copy_from(&txy);
    col2.rows_mut(nv, nv).copy_from(&tyy);

    // div_x = div of first column, div_y = div of second column.
    let div_x = apply_divergence(mesh, ext, hodge, &col1);
    let div_y = apply_divergence(mesh, ext, hodge, &col2);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&div_x);
    result.rows_mut(nv, nv).copy_from(&div_y);
    result
}
