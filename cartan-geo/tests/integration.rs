// cartan-geo/tests/integration.rs
//
// End-to-end tests for cartan-geo: geodesics, curvature, Jacobi fields.

use cartan_geo::{integrate_jacobi, scalar_at, sectional_at, CurvatureQuery, Geodesic};
use cartan_manifolds::{Euclidean, Sphere};
use nalgebra::SVector;

type Real = f64;

// ─────────────────────────────────────────────────────────────────────────────
// Geodesic on S²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn geodesic_sphere_endpoint_roundtrip() {
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let q = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);

    let geo = Geodesic::from_two_points(&s2, p.clone(), &q).unwrap();

    // gamma(0) = p
    let start = geo.eval(0.0);
    assert!((start - p).norm() < 1e-14, "gamma(0) != p: {:.2e}", (start - p).norm());

    // gamma(1) = q
    let end = geo.eval(1.0);
    assert!((end - q).norm() < 1e-12, "gamma(1) != q: {:.2e}", (end - q).norm());
}

#[test]
fn geodesic_sphere_length_is_angle() {
    let s2 = Sphere::<3>;
    // p and q are orthogonal unit vectors: angle = pi/2.
    let p = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let q = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);

    let geo = Geodesic::from_two_points(&s2, p, &q).unwrap();
    let err = (geo.length() - std::f64::consts::FRAC_PI_2).abs();
    assert!(err < 1e-14, "length = {:.6}, expected pi/2: err = {:.2e}", geo.length(), err);
}

#[test]
fn geodesic_sphere_sample_stays_on_manifold() {
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let q = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);

    let geo = Geodesic::from_two_points(&s2, p, &q).unwrap();
    for pt in geo.sample(20) {
        let norm_err = (pt.norm() - 1.0).abs();
        assert!(norm_err < 1e-13, "point not on sphere: ||p|| - 1 = {:.2e}", norm_err);
    }
}

#[test]
fn geodesic_sphere_midpoint_correct() {
    let s2 = Sphere::<3>;
    // p = e1, q = e2: midpoint should be (e1 + e2) / sqrt(2).
    let p = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let q = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);
    let expected_mid = SVector::<Real, 3>::from([1.0, 1.0, 0.0]) / 2.0_f64.sqrt();

    let geo = Geodesic::from_two_points(&s2, p, &q).unwrap();
    let mid = geo.midpoint();
    let err = (mid - expected_mid).norm();
    assert!(err < 1e-12, "midpoint err = {:.2e}", err);
}

#[test]
fn geodesic_euclidean_is_line_segment() {
    let r3 = Euclidean::<3>;
    let p = SVector::<Real, 3>::from([0.0, 0.0, 0.0]);
    let q = SVector::<Real, 3>::from([1.0, 2.0, 3.0]);

    let geo = Geodesic::from_two_points(&r3, p.clone(), &q).unwrap();

    // On Euclidean space, gamma(t) = p + t*(q-p) = t*q.
    for i in 0..=10 {
        let t = i as Real / 10.0;
        let expected = p + (q - p) * t;
        let actual = geo.eval(t);
        let err = (actual - expected).norm();
        assert!(err < 1e-14, "t={t}: err = {:.2e}", err);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature on S²
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sphere_sectional_curvature_is_one() {
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);
    // Two orthonormal tangent vectors at north pole.
    let u = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let v = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);

    let k = sectional_at(&s2, &p, &u, &v);
    assert!((k - 1.0).abs() < 1e-12, "S² sectional curvature = {k:.6}, expected 1.0");
}

#[test]
fn sphere_scalar_curvature_is_n_minus_1_times_n_minus_2() {
    // S^{N-1}: scalar curvature = (N-1)(N-2).
    // For N=3 (S²): scalar = 2*1 = 2.
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let s = scalar_at(&s2, &p);
    assert!((s - 2.0).abs() < 1e-12, "S² scalar curvature = {s:.6}, expected 2.0");
}

#[test]
fn sphere_curvature_query_matches_free_functions() {
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);
    let u = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let v = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);

    let q = CurvatureQuery::new(&s2, p.clone());
    assert!((q.sectional(&u, &v) - sectional_at(&s2, &p, &u, &v)).abs() < 1e-14);
    assert!((q.scalar() - scalar_at(&s2, &p)).abs() < 1e-14);
}

#[test]
fn euclidean_curvature_is_zero() {
    let r3 = Euclidean::<3>;
    let p = SVector::<Real, 3>::from([1.0, 2.0, 3.0]);
    let u = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let v = SVector::<Real, 3>::from([0.0, 1.0, 0.0]);

    assert_eq!(sectional_at(&r3, &p, &u, &v), 0.0);
    assert_eq!(scalar_at(&r3, &p), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Jacobi fields
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn jacobi_euclidean_is_linear() {
    // On flat R^3, the Jacobi equation is D²J/dt² = 0.
    // With J(0) = e1 and J'(0) = e2, the solution is J(t) = e1 + t * e2.
    let r3 = Euclidean::<3>;
    let p = SVector::<Real, 3>::zeros();
    let v = SVector::<Real, 3>::from([0.0, 1.0, 0.0]); // geodesic velocity

    let geo = Geodesic::new(&r3, p, v);
    let j0 = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let j0_dot = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);

    let result = integrate_jacobi(&geo, j0.clone(), j0_dot.clone(), 100);

    for (i, (&t, j)) in result.params.iter().zip(result.field.iter()).enumerate() {
        let expected = j0 + j0_dot * t;
        let err = (j - expected).norm();
        assert!(err < 1e-6, "step {i}: t={t:.2}, J err = {:.2e}", err);
    }
}

#[test]
fn jacobi_sphere_norm_at_zero() {
    // On S², a Jacobi field with J(0) = 0 and J'(0) = u (unit tangent)
    // satisfies ||J(t)|| = sin(t * ||v||) where ||v|| is the geodesic speed.
    // Here we just check that J(0) = j0 exactly.
    let s2 = Sphere::<3>;
    let p = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);
    let v = SVector::<Real, 3>::from([1.0, 0.0, 0.0]); // unit speed along S²

    let geo = Geodesic::new(&s2, p, v);
    let j0 = SVector::<Real, 3>::from([0.0, 1.0, 0.0]); // tangent at p, perp to v
    let j0_dot = SVector::<Real, 3>::zeros();

    let result = integrate_jacobi(&geo, j0.clone(), j0_dot, 200);

    // J(0) = j0 exactly
    let start_err = (result.field[0].clone() - j0).norm();
    assert!(start_err < 1e-14, "J(0) err = {:.2e}", start_err);

    // All field values stay on the sphere's tangent bundle: ||J(t)||_sphere stays bounded
    // (it should oscillate as sin function, not diverge).
    let max_norm = result
        .field
        .iter()
        .map(|j| j.norm())
        .fold(0.0_f64, f64::max);
    assert!(max_norm <= j0.norm() + 1e-3, "Jacobi field diverged: max ||J|| = {max_norm:.4}");
}
