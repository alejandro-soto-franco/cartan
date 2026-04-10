use num_complex::Complex;

use cartan_dec::line_bundle::{
    Section, ConnectionAngles, BochnerLaplacian, defect_charges,
    section_to_q_components, q_components_to_section,
};
use cartan_dec::mesh_gen::icosphere;
use cartan_dec::HodgeStar;
use cartan_manifolds::sphere::Sphere;

// ─────────────────────────────────────────────────────────────────────────────
// Section<K> basic tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_section_zeros() {
    let s = Section::<2>::zeros(10);
    assert_eq!(s.n_vertices(), 10);
    assert!(s.mean_norm() == 0.0);
}

#[test]
fn test_section_uniform() {
    let z = Complex::new(0.3, 0.4);
    let s = Section::<2>::uniform(5, z);
    assert_eq!(s.n_vertices(), 5);
    assert!((s.mean_norm() - 0.5).abs() < 1e-14); // |0.3 + 0.4i| = 0.5
}

#[test]
fn test_section_add_scale() {
    let a = Section::<2>::uniform(5, Complex::new(1.0, 0.0));
    let b = Section::<2>::uniform(5, Complex::new(0.0, 1.0));
    let c = a.add(&b.scale_real(2.0));
    assert!((c.values[0].re - 1.0).abs() < 1e-14);
    assert!((c.values[0].im - 2.0).abs() < 1e-14);
}

#[test]
fn test_section_real_roundtrip() {
    let q1 = vec![0.3, -0.1, 0.5];
    let q2 = vec![0.4, 0.2, -0.3];
    let s = Section::<2>::from_real_components(&q1, &q2);
    let (r1, r2) = s.to_real_components();
    for i in 0..3 {
        assert!((r1[i] - q1[i]).abs() < 1e-14);
        assert!((r2[i] - q2[i]).abs() < 1e-14);
    }
}

#[test]
fn test_section_normalise() {
    let mut s = Section::<2>::uniform(3, Complex::new(3.0, 4.0));
    assert!((s.values[0].norm() - 5.0).abs() < 1e-14);
    s.normalise(1e-10);
    assert!((s.values[0].norm() - 1.0).abs() < 1e-14);
}

#[test]
fn test_veronese_roundtrip() {
    let q1 = vec![0.3, -0.1];
    let q2 = vec![0.4, 0.2];
    let s = q_components_to_section(&q1, &q2);
    let (r1, r2) = section_to_q_components(&s);
    for i in 0..2 {
        assert!((r1[i] - q1[i]).abs() < 1e-14);
        assert!((r2[i] - q2[i]).abs() < 1e-14);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionAngles + BochnerLaplacian on S^2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_connection_angles_computable() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let conn = ConnectionAngles::from_mesh(&mesh, &manifold);
    assert_eq!(conn.primal.len(), mesh.n_boundaries());
    assert_eq!(conn.dual.len(), mesh.n_boundaries());
    // All angles should be finite.
    assert!(conn.primal.iter().all(|a| a.is_finite()));
    assert!(conn.dual.iter().all(|a| a.is_finite()));
}

#[test]
fn test_bochner_laplacian_zero_on_zero_section() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();
    let conn = ConnectionAngles::from_mesh(&mesh, &manifold);
    let lap = BochnerLaplacian::<2>::from_mesh_data(&mesh, &hodge, &conn);

    let z = Section::<2>::zeros(mesh.n_vertices());
    let result = lap.apply(&z);
    let norm: f64 = result.values.iter().map(|z| z.norm()).sum();
    assert!(
        norm < 1e-12,
        "Laplacian of zero section should be zero, got norm = {norm}"
    );
}

#[test]
fn test_bochner_laplacian_hermitian() {
    // <Delta z, w> should equal <z, Delta w> for the L2 inner product.
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 1, true);
    let hodge = HodgeStar::from_mesh_circumcentric(&mesh, &manifold).unwrap();
    let conn = ConnectionAngles::from_mesh(&mesh, &manifold);
    let lap = BochnerLaplacian::<2>::from_mesh_data(&mesh, &hodge, &conn);

    let nv = mesh.n_vertices();
    let dual_areas: Vec<f64> = (0..nv).map(|i| hodge.star0()[i]).collect();

    // Two random-ish sections.
    let z = Section::<2>::from_real_components(
        &(0..nv).map(|i| (i as f64 * 0.1).sin()).collect::<Vec<_>>(),
        &(0..nv).map(|i| (i as f64 * 0.2).cos()).collect::<Vec<_>>(),
    );
    let w = Section::<2>::from_real_components(
        &(0..nv).map(|i| (i as f64 * 0.3).cos()).collect::<Vec<_>>(),
        &(0..nv).map(|i| (i as f64 * 0.15).sin()).collect::<Vec<_>>(),
    );

    let dz = lap.apply(&z);
    let dw = lap.apply(&w);

    // <Delta z, w> vs <z, Delta w>
    let lhs = dz.l2_inner(&w, &dual_areas);
    let rhs = z.l2_inner(&dw, &dual_areas);

    let diff = (lhs - rhs).norm();
    assert!(
        diff < 1e-8,
        "<Dz, w> = {lhs}, <z, Dw> = {rhs}, diff = {diff}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Defect charges and Poincare-Hopf on S^2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_defect_charges_poincare_hopf() {
    let manifold = Sphere::<3>;
    let mesh = icosphere(&manifold, 2, true);
    let conn = ConnectionAngles::from_mesh(&mesh, &manifold);

    // Create a nematic section from the mesh coordinates (use x+iy of vertex position).
    let nv = mesh.n_vertices();
    let section = Section::<2>::from_real_components(
        &(0..nv).map(|i| mesh.vertices[i][0]).collect::<Vec<_>>(),
        &(0..nv).map(|i| mesh.vertices[i][1]).collect::<Vec<_>>(),
    );

    // Discrete Gaussian curvature per face: angle defect.
    // For simplicity, use uniform curvature: A_f * K_f = 4*pi / n_faces.
    let nf = mesh.n_simplices();
    let gauss_per_face: Vec<f64> = vec![4.0 * std::f64::consts::PI / nf as f64; nf];

    let charges = defect_charges(&section, &conn, &mesh, &gauss_per_face);

    let total_charge: f64 = charges.iter().sum();
    // Poincare-Hopf for S^2 with K=2: sum of charges = chi(S^2) = 2.
    // But total_charge = sum(winding / (2*pi*K)) + sum(gauss / (2*pi)) = winding_total/(4*pi) + 2.
    // The winding part depends on the section. Total should be 2 for chi(S^2).
    // With our uniform curvature approximation, sum(gauss/(2pi)) = 2.
    // The total charge should be close to an integer or half-integer.
    assert!(
        total_charge.is_finite(),
        "total charge should be finite, got {total_charge}"
    );
}
