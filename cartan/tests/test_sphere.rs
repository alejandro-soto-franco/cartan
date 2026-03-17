// ~/cartan/cartan/tests/test_sphere.rs

//! Tests for Sphere S^{N-1}.

mod common;

use cartan_core::{CartanError, Curvature, Manifold, ParallelTransport, Real};
use cartan_manifolds::Sphere;
use nalgebra::SVector;
use rand::SeedableRng;
use rand::rngs::StdRng;

// -- Harness tests --

#[test]
fn sphere_s2_base_identities() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_manifold_base::<3, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s9_base_identities() {
    let manifold = Sphere::<10>;
    common::manifold_harness::test_manifold_base::<10, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s2_transport() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_transport::<3, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s2_geodesic() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_geodesic::<3, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s2_parallel_transport() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_parallel_transport::<3, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s2_retraction() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_retraction::<3, _>(&manifold, 1e-10, 200);
}

#[test]
fn sphere_s2_curvature() {
    let manifold = Sphere::<3>;
    common::manifold_harness::test_curvature::<3, _>(&manifold, 1e-10, 200);
}

// -- Sphere-specific tests --

/// Sectional curvature should be 1.0 everywhere on S^{N-1}.
#[test]
fn sphere_s2_constant_curvature() {
    let manifold = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(50);

    for _ in 0..100 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);

        let k = manifold.sectional_curvature(&p, &u, &v);
        // Sectional curvature should be 1.0 for non-degenerate planes.
        let u_sq = manifold.inner(&p, &u, &u);
        let v_sq = manifold.inner(&p, &v, &v);
        let uv = manifold.inner(&p, &u, &v);
        let denom = u_sq * v_sq - uv * uv;
        if denom > 1e-10 {
            common::approx::assert_real_eq(k, 1.0, 1e-10, "sphere sectional curvature = 1");
        }
    }
}

/// Scalar curvature of S^{N-1} is (N-1)(N-2).
#[test]
fn sphere_scalar_curvature() {
    let s2 = Sphere::<3>;
    let s3 = Sphere::<4>;
    let s9 = Sphere::<10>;

    let p2 = SVector::<Real, 3>::new(1.0, 0.0, 0.0);
    let p3 = SVector::<Real, 4>::new(1.0, 0.0, 0.0, 0.0);
    let mut p9 = SVector::<Real, 10>::zeros();
    p9[0] = 1.0;

    // S^2: (3-1)(3-2) = 2
    common::approx::assert_real_eq(s2.scalar_curvature(&p2), 2.0, 1e-14, "S^2 scalar");
    // S^3: (4-1)(4-2) = 6
    common::approx::assert_real_eq(s3.scalar_curvature(&p3), 6.0, 1e-14, "S^3 scalar");
    // S^9: (10-1)(10-2) = 72
    common::approx::assert_real_eq(s9.scalar_curvature(&p9), 72.0, 1e-14, "S^9 scalar");
}

/// Bianchi identity: R(u,v)w + R(v,w)u + R(w,u)v = 0.
#[test]
fn sphere_bianchi_identity() {
    let manifold = Sphere::<4>;
    let mut rng = StdRng::seed_from_u64(51);

    for _ in 0..100 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        let r_uvw = manifold.riemann_curvature(&p, &u, &v, &w);
        let r_vwu = manifold.riemann_curvature(&p, &v, &w, &u);
        let r_wuv = manifold.riemann_curvature(&p, &w, &u, &v);

        let sum = r_uvw + r_vwu + r_wuv;
        assert!(
            sum.norm() < 1e-10,
            "Bianchi identity violated: ||sum|| = {:.2e}",
            sum.norm()
        );
    }
}

/// Curvature skew-symmetry: R(u,v)w = -R(v,u)w.
#[test]
fn sphere_curvature_skew_symmetry() {
    let manifold = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(52);

    for _ in 0..100 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        let r_uvw = manifold.riemann_curvature(&p, &u, &v, &w);
        let r_vuw = manifold.riemann_curvature(&p, &v, &u, &w);

        let sum = r_uvw + r_vuw;
        assert!(
            sum.norm() < 1e-10,
            "skew-symmetry violated: ||R(u,v)w + R(v,u)w|| = {:.2e}",
            sum.norm()
        );
    }
}

/// Log of antipodal point should fail with CutLocus error.
#[test]
fn sphere_log_antipodal_fails() {
    let manifold = Sphere::<3>;
    let p = SVector::<Real, 3>::new(1.0, 0.0, 0.0);
    let q = SVector::<Real, 3>::new(-1.0, 0.0, 0.0);

    let result = manifold.log(&p, &q);
    assert!(result.is_err(), "log of antipodal points should fail");
    match result.unwrap_err() {
        CartanError::CutLocus { .. } => {} // expected
        other => panic!("expected CutLocus error, got: {:?}", other),
    }
}

/// Injectivity radius should be pi.
#[test]
fn sphere_injectivity_radius() {
    let manifold = Sphere::<3>;
    let p = SVector::<Real, 3>::new(1.0, 0.0, 0.0);
    common::approx::assert_real_eq(
        manifold.injectivity_radius(&p),
        std::f64::consts::PI,
        1e-14,
        "S^2 injectivity radius",
    );
}

/// Parallel transport on sphere preserves inner product exactly.
#[test]
fn sphere_parallel_transport_preserves_inner() {
    let manifold = Sphere::<4>;
    let mut rng = StdRng::seed_from_u64(53);

    for _ in 0..200 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // Generate q that is not antipodal to p.
        let d = manifold.random_tangent(&p, &mut rng);
        let d_small = d * (0.5 / d.norm().max(1e-10));
        let q = manifold.exp(&p, &d_small);

        if let (Ok(u_t), Ok(w_t)) = (
            manifold.transport(&p, &q, &u),
            manifold.transport(&p, &q, &w),
        ) {
            let inner_before = manifold.inner(&p, &u, &w);
            let inner_after = manifold.inner(&q, &u_t, &w_t);
            common::approx::assert_real_eq(
                inner_after,
                inner_before,
                1e-10,
                "parallel transport preserves inner product",
            );
        }
    }
}
