// ~/cartan/cartan-dec/src/divergence.rs

//! Discrete covariant divergence of a vector field.
//!
//! The DEC divergence formula is: div(u) = star_0_inv * d0^T * star_1 * u_1form
//!
//! This formula is K-agnostic: it only uses the 0-form and 1-form operators
//! (d0, star0, star1). The velocity-to-1-form conversion uses trapezoidal
//! integration along boundary faces (edges for K=3).
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003.

use nalgebra::DVector;

use cartan_core::Manifold;

use crate::exterior::ExteriorDerivative;
use crate::hodge::HodgeStar;
use crate::mesh::{FlatMesh, Mesh};

/// K-generic discrete divergence of a vertex-based vector field.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh.
/// - `manifold`: the Riemannian manifold.
/// - `ext`: precomputed exterior derivative operators.
/// - `hodge`: precomputed Hodge star operators.
/// - `u`: velocity field as one tangent vector per vertex.
///
/// # Returns
///
/// div(u) at each vertex as an n_v vector.
pub fn apply_divergence_generic<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    u: &[M::Tangent],
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    let nb = mesh.n_boundaries();
    assert_eq!(u.len(), nv, "divergence: u must have n_v tangent vectors");

    // Step 1: Build the 1-form from the vector field.
    // For each boundary face (edge for K=3) with B=2 vertices [i, j]:
    //   u_1form[b] = 0.5 * (u[i] + u[j]) . edge_vector
    let mut u1form = DVector::<f64>::zeros(nb);
    for (b, boundary) in mesh.boundaries.iter().enumerate() {
        if B == 2 {
            let i = boundary[0];
            let j = boundary[1];
            let pi = &mesh.vertices[i];
            let pj = &mesh.vertices[j];
            let edge_vec = match manifold.log(pi, pj) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let ui_dot = manifold.inner(pi, &u[i], &edge_vec);
            let uj_dot = manifold.inner(pi, &u[j], &edge_vec);
            u1form[b] = 0.5 * (ui_dot + uj_dot);
        }
    }

    // Step 2: Apply star1 to the 1-form.
    let star1_u1form = u1form.component_mul(hodge.star1());

    // Step 3: Apply d0^T (sparse transpose multiply).
    // d0 is CSC, so transpose_view gives CSR. For CSR, outer_iterator
    // iterates over rows of d0^T.
    let d0t = ext.d0().transpose_view();
    let mut d0t_star1_u = DVector::<f64>::zeros(nv);
    for (row_idx, row) in d0t.outer_iterator().enumerate() {
        let mut sum = 0.0;
        for (col_idx, &val) in row.iter() {
            sum += val * star1_u1form[col_idx];
        }
        d0t_star1_u[row_idx] = sum;
    }

    // Step 4: Apply star0_inv.
    let star0_inv = hodge.star0_inv();
    d0t_star1_u.component_mul(&star0_inv)
}

/// Backward-compatible discrete divergence (flat mesh, DVector velocity layout).
pub fn apply_divergence(
    mesh: &FlatMesh,
    ext: &ExteriorDerivative,
    hodge: &HodgeStar,
    u: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(u.len(), 2 * nv, "divergence: u must have 2*n_v entries");

    let u_tangent: Vec<nalgebra::SVector<f64, 2>> = (0..nv)
        .map(|v| nalgebra::SVector::<f64, 2>::new(u[v], u[nv + v]))
        .collect();

    let manifold = cartan_manifolds::euclidean::Euclidean::<2>;
    apply_divergence_generic(mesh, &manifold, ext, hodge, &u_tangent)
}

/// Compute the discrete divergence of a symmetric 2-tensor field (K=3 only).
///
/// For a symmetric 2-tensor field T with components [T_xx, T_xy, T_yy] at
/// each vertex, the divergence is a vector field:
///   (div T)_x = div of first column  (T_xx, T_xy)
///   (div T)_y = div of second column (T_xy, T_yy)
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

    // First column of T: (T_xx, T_xy).
    let mut col1 = DVector::<f64>::zeros(2 * nv);
    col1.rows_mut(0, nv).copy_from(&txx);
    col1.rows_mut(nv, nv).copy_from(&txy);

    // Second column of T: (T_xy, T_yy).
    let mut col2 = DVector::<f64>::zeros(2 * nv);
    col2.rows_mut(0, nv).copy_from(&txy);
    col2.rows_mut(nv, nv).copy_from(&tyy);

    let div_x = apply_divergence(mesh, ext, hodge, &col1);
    let div_y = apply_divergence(mesh, ext, hodge, &col2);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&div_x);
    result.rows_mut(nv, nv).copy_from(&div_y);
    result
}
