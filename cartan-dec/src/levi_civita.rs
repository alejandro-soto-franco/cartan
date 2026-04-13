//! Construct discrete Levi-Civita connections from mesh geometry.
//!
//! Wraps [`ConnectionAngles`] into [`EdgeTransport2D`] (SO(2) rotation matrices)
//! for use with the generic [`CovLaplacian`] from cartan-core.
//!
//! For 2-manifolds, the connection angle alpha at each edge gives the SO(2)
//! rotation matrix `[[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]]`.
//! The `CovLaplacian` then uses `Fiber::transport_by` to derive the
//! representation-specific transport (e.g., spin-2 phase rotation for nematics).

use cartan_core::bundle::EdgeTransport2D;
use cartan_core::Manifold;

use crate::line_bundle::ConnectionAngles;
use crate::mesh::Mesh;

/// Build an [`EdgeTransport2D`] (SO(2) per edge) from a triangle mesh.
///
/// Uses [`ConnectionAngles::from_mesh`] to compute the Levi-Civita connection,
/// then converts each scalar angle to a 2x2 rotation matrix.
///
/// The resulting transport, combined with [`CovLaplacian`], recovers the same
/// covariant Laplacian as [`BochnerLaplacian`] when applied to `U1Spin2` sections.
pub fn levi_civita_2d<M: Manifold>(
    mesh: &Mesh<M, 3, 2>,
    manifold: &M,
) -> EdgeTransport2D {
    let conn = ConnectionAngles::from_mesh(mesh, manifold);
    let ne = mesh.n_boundaries();

    let mut transports = Vec::with_capacity(ne);
    let mut edges = Vec::with_capacity(ne);

    for e in 0..ne {
        let alpha = conn.primal[e];
        let ca = alpha.cos();
        let sa = alpha.sin();
        // SO(2) rotation matrix, row-major: [cos, -sin, sin, cos]
        transports.push([ca, -sa, sa, ca]);
        edges.push(mesh.boundaries[e]);
    }

    EdgeTransport2D { edges, transports }
}

/// Build an [`EdgeTransport2D`] from pre-computed [`ConnectionAngles`].
///
/// Useful when you already have connection angles (e.g., from the line bundle
/// module) and want to reuse them with the fiber bundle framework.
pub fn edge_transport_from_angles(
    angles: &ConnectionAngles,
    boundaries: &[[usize; 2]],
) -> EdgeTransport2D {
    let ne = angles.primal.len();
    let mut transports = Vec::with_capacity(ne);
    let mut edges = Vec::with_capacity(ne);

    for (e, &alpha) in angles.primal.iter().enumerate().take(ne) {
        let ca = alpha.cos();
        let sa = alpha.sin();
        transports.push([ca, -sa, sa, ca]);
        edges.push(boundaries[e]);
    }

    EdgeTransport2D { edges, transports }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_core::bundle::CovLaplacian;
    use cartan_core::fiber::{Section, U1Spin2, VecSection};
    use cartan_manifolds::sphere::Sphere;

    use crate::hodge::HodgeStar;
    use crate::line_bundle::BochnerLaplacian;
    use crate::mesh_gen;

    fn test_sphere_mesh() -> Mesh<Sphere<3>, 3, 2> {
        mesh_gen::icosphere(&Sphere::<3>, 2, false)
    }

    #[test]
    fn levi_civita_2d_constructs_on_sphere() {
        let mesh = test_sphere_mesh();
        let transport = levi_civita_2d(&mesh, &Sphere::<3>);
        assert_eq!(transport.edges.len(), mesh.n_boundaries());
        assert_eq!(transport.transports.len(), mesh.n_boundaries());
    }

    #[test]
    fn levi_civita_2d_orthogonal_matrices() {
        let mesh = test_sphere_mesh();
        let transport = levi_civita_2d(&mesh, &Sphere::<3>);
        for t in &transport.transports {
            // R^T R = I for SO(2): [cos, sin; -sin, cos] * [cos, -sin; sin, cos]
            let det = t[0] * t[3] - t[1] * t[2];
            assert!((det - 1.0).abs() < 1e-12, "det = {det}");
            let rtr_00 = t[0] * t[0] + t[2] * t[2];
            assert!((rtr_00 - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cov_laplacian_matches_bochner_on_sphere() {
        // The CovLaplacian<U1Spin2> with EdgeTransport2D should produce the
        // same result as the existing BochnerLaplacian<2> on a sphere.
        let mesh = test_sphere_mesh();
        let manifold = Sphere::<3>;
        let hodge = HodgeStar::from_mesh_generic(&mesh, &manifold).unwrap();
        let conn_angles = ConnectionAngles::from_mesh(&mesh, &manifold);
        let transport = edge_transport_from_angles(&conn_angles, &mesh.boundaries);

        let nv = mesh.n_vertices();
        let star0: Vec<f64> = (0..nv).map(|i| hodge.star0()[i]).collect();
        let star1: Vec<f64> = (0..mesh.n_boundaries()).map(|i| hodge.star1()[i]).collect();

        // Build generic CovLaplacian.
        let cov_lap = CovLaplacian::new(nv, &transport.edges, &star1, &star0);

        // Build existing BochnerLaplacian<2>.
        let bochner = BochnerLaplacian::<2>::from_mesh_data(&mesh, &hodge, &conn_angles);

        // Deterministic test field: use vertex index to seed values.
        let data: Vec<[f64; 2]> = (0..nv)
            .map(|i| {
                let x = ((i * 7 + 3) % 100) as f64 / 100.0 - 0.5;
                let y = ((i * 13 + 7) % 100) as f64 / 100.0 - 0.5;
                [x, y]
            })
            .collect();
        let section = VecSection::<U1Spin2>::from_vec(data.clone());

        // Apply generic CovLaplacian.
        let result_cov = cov_lap.apply::<U1Spin2, 2, _>(&section, &transport);

        // Apply BochnerLaplacian via complex Section<2>.
        use num_complex::Complex;
        let complex_section = crate::line_bundle::Section::<2> {
            values: data
                .iter()
                .map(|[r, i]| Complex::new(*r, *i))
                .collect(),
        };
        let result_bochner = bochner.apply(&complex_section);

        // Compare. The BochnerLaplacian is NEGATIVE-semidefinite (from the code:
        // inv_star0[i] is negated: -1.0/star0[i]). The CovLaplacian is POSITIVE.
        // So result_cov = -result_bochner.
        let mut max_diff = 0.0_f64;
        for v in 0..nv {
            let cov_q1 = result_cov.at(v)[0];
            let cov_q2 = result_cov.at(v)[1];
            let boch_q1 = -result_bochner.values[v].re; // negate for sign convention
            let boch_q2 = -result_bochner.values[v].im;
            max_diff = max_diff.max((cov_q1 - boch_q1).abs());
            max_diff = max_diff.max((cov_q2 - boch_q2).abs());
        }
        assert!(
            max_diff < 1e-10,
            "CovLaplacian should match -BochnerLaplacian, max_diff = {max_diff}"
        );
    }
}
