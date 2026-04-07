// ~/cartan/cartan-dec/src/advection.rs

//! Discrete covariant advection operator for scalar and tensor-valued fields.
//!
//! The upwind scheme computes (u . nabla) f at each vertex by iterating over
//! incident boundary faces (edges for K=3) via the adjacency maps. This gives
//! O(V * avg_degree) complexity regardless of K.
//!
//! ## References
//!
//! - LeVeque. "Finite Volume Methods for Hyperbolic Problems." Cambridge, 2002.
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341.

use nalgebra::DVector;

use cartan_core::Manifold;

use crate::mesh::{FlatMesh, Mesh};

/// K-generic upwind covariant advection of a scalar 0-form.
///
/// # Arguments
///
/// - `mesh`: the simplicial mesh (must have adjacency maps built).
/// - `manifold`: the Riemannian manifold for metric operations.
/// - `f`: scalar field at vertices (n_v vector).
/// - `u`: velocity field as one tangent vector per vertex.
///
/// # Returns
///
/// `(u . nabla) f` at each vertex as an n_v vector.
pub fn apply_scalar_advection_generic<M: Manifold, const K: usize, const B: usize>(
    mesh: &Mesh<M, K, B>,
    manifold: &M,
    f: &DVector<f64>,
    u: &[M::Tangent],
) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(f.len(), nv, "advection: f must have n_v entries");
    assert_eq!(u.len(), nv, "advection: u must have n_v tangent vectors");

    let mut result = DVector::<f64>::zeros(nv);

    for v in 0..nv {
        let pv = &mesh.vertices[v];
        let uv = &u[v];

        // Iterate over incident boundary faces of vertex v.
        for &b in &mesh.vertex_boundaries[v] {
            let boundary = &mesh.boundaries[b];

            // Find the other vertices of this boundary face.
            for &other in boundary {
                if other == v {
                    continue;
                }

                let po = &mesh.vertices[other];

                // Direction from v to other in tangent space.
                let edge_tangent = match manifold.log(pv, po) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let len = manifold.norm(pv, &edge_tangent);
                if len < 1e-30 {
                    continue;
                }

                // Project velocity onto edge direction.
                let u_proj = manifold.inner(pv, uv, &edge_tangent) / len;

                // Upwind flux.
                result[v] += if u_proj > 0.0 {
                    u_proj * (f[other] - f[v]) / len
                } else {
                    u_proj * (f[v] - f[other]) / len
                };
            }
        }
    }

    result
}

/// Apply the upwind covariant advection operator to a scalar 0-form (flat mesh, old API).
///
/// Backward-compatible wrapper. Velocity is stored as [u_x[0..n_v], u_y[0..n_v]].
pub fn apply_scalar_advection(mesh: &FlatMesh, f: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(f.len(), nv, "advection: f must have n_v entries");
    assert_eq!(u.len(), 2 * nv, "advection: u must have 2*n_v entries");

    let u_tangent: Vec<nalgebra::SVector<f64, 2>> = (0..nv)
        .map(|v| nalgebra::SVector::<f64, 2>::new(u[v], u[nv + v]))
        .collect();

    let manifold = cartan_manifolds::euclidean::Euclidean::<2>;
    apply_scalar_advection_generic(mesh, &manifold, f, &u_tangent)
}

/// Apply the upwind covariant advection operator to a vector-valued 0-form (flat mesh, old API).
///
/// For a flat domain, applies scalar advection component-wise.
pub fn apply_vector_advection(mesh: &FlatMesh, q: &DVector<f64>, u: &DVector<f64>) -> DVector<f64> {
    let nv = mesh.n_vertices();
    assert_eq!(
        q.len(),
        2 * nv,
        "vector_advection: q must have 2*n_v entries"
    );
    assert_eq!(
        u.len(),
        2 * nv,
        "vector_advection: u must have 2*n_v entries"
    );

    let qx = q.rows(0, nv).into_owned();
    let qy = q.rows(nv, nv).into_owned();

    let lqx = apply_scalar_advection(mesh, &qx, u);
    let lqy = apply_scalar_advection(mesh, &qy, u);

    let mut result = DVector::<f64>::zeros(2 * nv);
    result.rows_mut(0, nv).copy_from(&lqx);
    result.rows_mut(nv, nv).copy_from(&lqy);
    result
}
