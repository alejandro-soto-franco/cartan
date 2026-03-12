// ~/cartan/cartan/tests/test_euclidean.rs

//! Tests for Euclidean R^N manifold.
//!
//! Since Euclidean space is flat and trivial, these tests serve as a
//! baseline: every manifold identity should hold with very tight tolerance.
//! If these fail, there's a bug in the test harness itself.

mod common;

use cartan_manifolds::Euclidean;

/// Run the full base manifold test suite for R^3.
#[test]
fn euclidean_r3_base_identities() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_manifold_base::<3, _>(&manifold, 1e-14, 100);
}

/// Run the full base manifold test suite for R^10.
#[test]
fn euclidean_r10_base_identities() {
    let manifold = Euclidean::<10>;
    common::manifold_harness::test_manifold_base::<10, _>(&manifold, 1e-14, 100);
}

/// Vector transport is identity in Euclidean space.
///
/// For R^N, parallel transport (and hence the blanket VectorTransport)
/// is the identity, so transporting any vector along any direction leaves
/// it unchanged.
#[test]
fn euclidean_r3_transport() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_transport::<3, _>(&manifold, 1e-14, 100);
}

/// Geodesic interpolation is linear interpolation.
///
/// For R^N, geodesic(p, q, t) = (1-t)*p + t*q. The boundary and constant-speed
/// tests should all pass exactly.
#[test]
fn euclidean_r3_geodesic() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_geodesic::<3, _>(&manifold, 1e-14, 100);
}

/// Parallel transport is identity in Euclidean space (exact preservation).
///
/// transport(p, q, v) = v for all p, q, v in R^N.
#[test]
fn euclidean_r3_parallel_transport() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_parallel_transport::<3, _>(&manifold, 1e-14, 100);
}

/// Retraction roundtrip (retraction = exp for Euclidean).
///
/// For R^N, retract(p, v) = p + v and inverse_retract(p, q) = q - p,
/// so the roundtrip is exact.
#[test]
fn euclidean_r3_retraction() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_retraction::<3, _>(&manifold, 1e-14, 100);
}

/// Curvature harness (skew-symmetry, Bianchi identity for zero curvature).
///
/// All curvature identities hold trivially since R = 0.
#[test]
fn euclidean_r3_curvature_harness() {
    let manifold = Euclidean::<3>;
    common::manifold_harness::test_curvature::<3, _>(&manifold, 1e-14, 100);
}

/// Curvature is identically zero.
///
/// Checks Riemann tensor, sectional curvature, Ricci curvature, and scalar
/// curvature all vanish on R^N.
#[test]
fn euclidean_r3_zero_curvature() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use cartan_core::{Manifold, Curvature};

    let manifold = Euclidean::<3>;
    let mut rng = StdRng::seed_from_u64(45);

    for _ in 0..50 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // Riemann curvature tensor vanishes.
        let r = manifold.riemann_curvature(&p, &u, &v, &w);
        assert!(r.norm() < 1e-14, "Euclidean curvature should be zero, got norm {:.2e}", r.norm());

        // Sectional curvature vanishes.
        let sec = manifold.sectional_curvature(&p, &u, &v);
        // Note: sectional_curvature = <R(u,v)v, u> / (||u||^2 ||v||^2 - <u,v>^2).
        // For Euclidean: numerator = 0, so the result is 0/denom.
        // If u and v are parallel, denominator = 0 and result is NaN; skip those.
        if sec.is_finite() {
            assert!(sec.abs() < 1e-14, "Euclidean sectional curvature should be zero, got {:.2e}", sec);
        }

        // Ricci curvature vanishes.
        let ric = manifold.ricci_curvature(&p, &u, &v);
        assert!(ric.abs() < 1e-14, "Euclidean Ricci should be zero, got {:.2e}", ric);

        // Scalar curvature vanishes.
        let scal = manifold.scalar_curvature(&p);
        assert!(scal.abs() < 1e-14, "Euclidean scalar curvature should be zero, got {:.2e}", scal);
    }
}

/// exp(p, v) = p + v exactly.
///
/// Explicit test that the exp map is exactly vector addition, not just
/// approximately, since this is the defining property of Euclidean space.
#[test]
fn euclidean_exp_is_addition() {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use cartan_core::Manifold;

    let manifold = Euclidean::<5>;
    let mut rng = StdRng::seed_from_u64(46);

    for _ in 0..50 {
        let p = manifold.random_point(&mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let result = manifold.exp(&p, &v);
        let expected = p + v;
        common::approx::assert_vec_eq(&result, &expected, 1e-15, "exp = p + v");
    }
}

/// Injectivity radius is infinity for Euclidean space.
///
/// On R^N the exponential map (translation) is a global diffeomorphism,
/// so there is no cut locus and the injectivity radius is infinite.
#[test]
fn euclidean_injectivity_radius() {
    use cartan_core::Manifold;
    use nalgebra::SVector;

    let manifold = Euclidean::<3>;
    let p = SVector::<f64, 3>::zeros();
    assert!(manifold.injectivity_radius(&p).is_infinite());
}
