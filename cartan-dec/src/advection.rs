// ~/cartan/cartan-dec/src/advection.rs

//! Discrete covariant advection operator for scalar and tensor-valued fields.
//!
//! The covariant advection of a scalar field f by a velocity field u is:
//!
//!   (u · ∇) f
//!
//! Discretized via upwind finite differences on the primal mesh:
//! for each vertex v, the advective flux from neighbor w is:
//!
//!   flux(v, w) = max(u_{vw}, 0) * f\[w\] - max(-u_{vw}, 0) * f\[v\]
//!
//! where u_{vw} = u · (w - v) / |w - v| is the velocity projected onto
//! edge (v, w). The upwind scheme is first-order but unconditionally stable.
//!
//! ## Covariant advection of tensor fields
//!
//! For a vector field q transported by velocity u, the covariant advection is:
//!
//!   (u · ∇) q + (connection term)
//!
//! On a flat domain, the connection term vanishes and this reduces to
//! component-wise scalar advection.
//!
//! ## References
//!
//! - LeVeque. "Finite Volume Methods for Hyperbolic Problems." Cambridge, 2002.
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.

use nalgebra::DVector;

use crate::mesh::FlatMesh;

/// Apply the upwind covariant advection operator to a scalar 0-form.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh.
/// - `f`: scalar field at vertices (n_v vector).
/// - `u`: velocity field at vertices, stored as [u_x[0..n_v], u_y[0..n_v]] (2*n_v vector).
///
/// # Returns
///
/// `(u · ∇) f` at each vertex as an n_v vector.
pub fn apply_scalar_advection(mesh: &FlatMesh, f: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(f.len(), nv, "advection: f must have n_v entries");
    assert_eq!(u.len(), 2 * nv, "advection: u must have 2*n_v entries");

    let mut result = DVector::<f64>::zeros(nv);

    for v in 0..nv {
        let pv = mesh.vertex(v);
        let uv = nalgebra::Vector2::new(u[v], u[nv + v]);

        // Accumulate advective flux from all edges containing v.
        // We iterate over edges and contribute to both endpoints.
        for &[i, j] in &mesh.boundaries {
            if i != v && j != v {
                continue;
            }
            let other = if i == v { j } else { i };
            let po = mesh.vertex(other);

            // Unit vector from v to other.
            let diff = po - pv;
            let len = diff.norm();
            if len < 1e-30 {
                continue;
            }
            let edge_dir = diff / len;

            // Normal velocity component along edge: u_v · (other - v)/|other - v|.
            let u_proj = uv.dot(&edge_dir);

            // Upwind flux: if u_proj > 0, advect from other to v.
            result[v] += if u_proj > 0.0 {
                u_proj * (f[other] - f[v]) / len
            } else {
                u_proj * (f[v] - f[other]) / len
            };
        }
    }

    result
}

/// Apply the upwind covariant advection operator to a vector-valued 0-form.
///
/// For a flat domain, applies scalar advection component-wise.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh.
/// - `q`: field at vertices, stored as [q_x[0..n_v], q_y[0..n_v]] (2*n_v vector).
/// - `u`: velocity field at vertices, stored as [u_x[0..n_v], u_y[0..n_v]] (2*n_v vector).
///
/// # Returns
///
/// `(u · ∇) q` at each vertex as a 2*n_v vector.
pub fn apply_vector_advection(
    mesh: &FlatMesh,
    q: &DVector<f64>,
    u: &DVector<f64>,
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(q.len(), 2 * nv, "vector_advection: q must have 2*n_v entries");
    assert_eq!(u.len(), 2 * nv, "vector_advection: u must have 2*n_v entries");

    let qx = q.rows(0, nv).into_owned();
    let qy = q.rows(nv, nv).into_owned();

    let lqx = apply_scalar_advection(mesh, &qx, u);
    let lqy = apply_scalar_advection(mesh, &qy, u);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&lqx);
    result.rows_mut(nv, nv).copy_from(&lqy);
    result
}
