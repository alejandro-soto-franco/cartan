//! Whitney "sharp" reconstruction: discrete 1-form (one value per edge) to a
//! per-vertex ambient vector field in R^3.
//!
//! A discrete 1-form stores, per edge `e=(i,j)`, the integral of a vector field
//! along that edge, approximately `omega_e = u . (x_j - x_i)`. The Whitney sharp
//! reconstructs a per-vertex vector by least-squares fitting, per triangle, the
//! constant vector `u_f` whose three edge integrals match the stored edge
//! values, then area-averaging `u_f` to the incident vertices.
//!
//! The triangle system is rank-2 (a surface 1-form carries no out-of-plane
//! component), so we solve the normal equations with a tiny normal-direction
//! regularizer that pins the out-of-plane component to zero.

use cartan_core::Manifold;
use cartan_dec::mesh::Mesh;
use nalgebra::{Matrix3, SVector};

/// Reconstruct a per-vertex R^3 vector field from a discrete 1-form on edges.
///
/// `omega` has length `mesh.n_boundaries()` (one value per edge), oriented along
/// the canonical (low-index-first) edge direction `boundaries[e] = [tail, head]`
/// so that `omega[e] = u . (x_head - x_tail)`.
///
/// Returns a `Vec<f64>` of length `3 * n_vertices`, interleaved `[vx,vy,vz, ...]`.
pub fn sharp_1form_to_vertex_vectors<M>(mesh: &Mesh<M, 3, 2>, omega: &[f64]) -> Vec<f64>
where
    M: Manifold<Point = SVector<f64, 3>>,
{
    let nv = mesh.n_vertices();
    let mut accum = vec![SVector::<f64, 3>::zeros(); nv];
    let mut weight = vec![0.0_f64; nv];

    for (t, tri) in mesh.simplices.iter().enumerate() {
        let p = [
            mesh.vertices[tri[0]],
            mesh.vertices[tri[1]],
            mesh.vertices[tri[2]],
        ];
        let normal = (p[1] - p[0]).cross(&(p[2] - p[0]));
        let area = 0.5 * normal.norm();
        if area <= 0.0 {
            continue;
        }

        // Assemble the least-squares system E u_f = b, where each row of E is a
        // triangle edge vector and the matching b entry is that edge's 1-form
        // value. Both the row and the stored value use the SAME canonical edge
        // direction (boundaries[eid] = [tail, head]), so the per-triangle
        // orientation sign cancels: row = x_head - x_tail and val = omega[eid].
        let mut ete = Matrix3::<f64>::zeros();
        let mut etb = SVector::<f64, 3>::zeros();
        for k in 0..3 {
            let eid = mesh.simplex_boundary_ids[t][k];
            let b = &mesh.boundaries[eid];
            let row = mesh.vertices[b[1]] - mesh.vertices[b[0]];
            let val = omega[eid];
            ete += row * row.transpose();
            etb += row * val;
        }
        // Tiny normal-direction regularizer so the rank-2 in-plane system is
        // invertible; it pins the out-of-plane component of u_f to zero.
        let n = normal.normalize();
        ete += 1e-12 * (n * n.transpose());
        let u_f = ete
            .try_inverse()
            .map(|inv| inv * etb)
            .unwrap_or_else(SVector::zeros);

        for &vi in tri {
            accum[vi] += area * u_f;
            weight[vi] += area;
        }
    }

    let mut out = vec![0.0; nv * 3];
    for v in 0..nv {
        let u = if weight[v] > 0.0 {
            accum[v] / weight[v]
        } else {
            SVector::zeros()
        };
        out[v * 3] = u[0];
        out[v * 3 + 1] = u[1];
        out[v * 3 + 2] = u[2];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_dec::mesh::Mesh;
    use cartan_manifolds::euclidean::Euclidean;
    use nalgebra::SVector;

    #[test]
    fn recovers_constant_field_on_flat_triangle() {
        let verts = vec![
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 0.0, 0.0),
            SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        ];
        let mesh =
            Mesh::<Euclidean<3>, 3, 2>::from_simplices(&Euclidean::<3>, verts, vec![[0, 1, 2]]);
        let u = SVector::<f64, 3>::new(0.7, -0.3, 0.0);
        // Build the edge 1-form: omega_e = u . (x_head - x_tail), with the
        // canonical (low-index-first) orientation cartan uses for boundaries.
        let mut omega = vec![0.0; mesh.n_boundaries()];
        for (e, b) in mesh.boundaries.iter().enumerate() {
            let (i, j) = (b[0], b[1]);
            let edge = mesh.vertices[j] - mesh.vertices[i];
            omega[e] = u.dot(&edge);
        }
        let out = sharp_1form_to_vertex_vectors(&mesh, &omega);
        for v in 0..mesh.n_vertices() {
            assert!((out[v * 3] - 0.7).abs() < 1e-9, "vx at {v}");
            assert!((out[v * 3 + 1] + 0.3).abs() < 1e-9, "vy at {v}");
            assert!(out[v * 3 + 2].abs() < 1e-9, "vz at {v}");
        }
    }
}
