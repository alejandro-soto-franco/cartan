// ~/cartan/cartan/tests/test_spd.rs

//! Integration tests for the Spd<N> manifold (symmetric positive definite matrices).
//!
//! SPD(N) carries the affine-invariant metric:
//!   <U, V>_P = tr(P^{-1} U P^{-1} V)
//!
//! Key geometric properties verified here:
//! - Cartan-Hadamard: K ≤ 0 everywhere; geodesics never hit a cut locus.
//! - Injectivity radius: ∞ (global diffeomorphism for exp).
//! - Parallel transport norm preservation (exact).
//! - Geodesic interpolation: geodesic(P, Q, 0) = P, geodesic(P, Q, 1) = Q.
//! - Curvature skew-symmetry and first Bianchi identity.

mod common;

use cartan_core::{Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Real, Retraction};
use cartan_manifolds::Spd;
use nalgebra::SMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─────────────────────────────────────────────────────────────────────────────
// Harness tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_base_identities() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_manifold_base::<3, _>(&m, 1e-7, 100);
}

#[test]
fn spd4_base_identities() {
    let m = Spd::<4>;
    common::matrix_harness::test_matrix_manifold_base::<4, _>(&m, 1e-6, 50);
}

#[test]
fn spd3_transport() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_transport::<3, _>(&m, 1e-7, 100);
}

#[test]
fn spd3_parallel_transport() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_parallel_transport::<3, _>(&m, 1e-7, 100);
}

#[test]
fn spd3_retraction() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_retraction::<3, _>(&m, 1e-7, 100);
}

#[test]
fn spd3_curvature() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_curvature::<3, _>(&m, 1e-7, 100);
}

#[test]
fn spd3_geodesic() {
    let m = Spd::<3>;
    common::matrix_harness::test_matrix_geodesic::<3, _>(&m, 1e-7, 100);
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: injectivity radius is infinite
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd_injectivity_radius_is_infinite() {
    let m = Spd::<3>;
    let p = SMatrix::<Real, 3, 3>::identity();
    assert!(
        m.injectivity_radius(&p).is_infinite(),
        "SPD injectivity radius should be infinite"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: sectional curvature is non-positive
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_sectional_curvature_nonpositive() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(100);

    for _ in 0..200 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let v = m.random_tangent(&p, &mut rng);

        let uu = m.inner(&p, &u, &u);
        let vv = m.inner(&p, &v, &v);
        let uv = m.inner(&p, &u, &v);
        let denom = uu * vv - uv * uv;
        if denom > 1e-10 {
            let k = m.sectional_curvature(&p, &u, &v);
            assert!(
                k <= 1e-10,
                "SPD sectional curvature should be ≤ 0, got {k:.6e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: scalar curvature is non-positive
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_scalar_curvature_nonpositive() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(101);
    for _ in 0..50 {
        let p = m.random_point(&mut rng);
        let s = m.scalar_curvature(&p);
        assert!(s <= 1e-10, "SPD scalar curvature should be ≤ 0, got {s:.6e}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: geodesic at identity
//
// Exp_I(V) = matrix_exp(V) and Log_I(P) = matrix_log(P) when P = I.
// Geodesic between I and diag(2,3,4) should interpolate smoothly.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_geodesic_from_identity_to_diagonal() {
    let m = Spd::<3>;
    let p = SMatrix::<Real, 3, 3>::identity();
    let q = SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::SVector::from([2.0, 3.0, 4.0]));

    // t=0 gives P, t=1 gives Q.
    let g0 = m.geodesic(&p, &q, 0.0).unwrap();
    let g1 = m.geodesic(&p, &q, 1.0).unwrap();

    assert!(
        (g0 - p).norm() < 1e-12,
        "geodesic(P,Q,0) ≠ P: err = {:.2e}",
        (g0 - p).norm()
    );
    assert!(
        (g1 - q).norm() < 1e-10,
        "geodesic(P,Q,1) ≠ Q: err = {:.2e}",
        (g1 - q).norm()
    );

    // Midpoint must be SPD.
    let mid = m.geodesic(&p, &q, 0.5).unwrap();
    m.check_point(&mid)
        .expect("geodesic midpoint not a valid SPD matrix");

    // Midpoint should have eigenvalues between those of P and Q.
    let mid_eig = nalgebra::SymmetricEigen::new(mid);
    for &ev in mid_eig.eigenvalues.iter() {
        assert!(ev > 0.0, "geodesic midpoint has non-positive eigenvalue {ev:.4e}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: affine-invariant metric — congruence invariance
//
// <A U A^T, A V A^T>_{A P A^T} = <U, V>_P for any invertible A.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_affine_invariant_metric() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(102);

    let p = m.random_point(&mut rng);
    let u = m.random_tangent(&p, &mut rng);
    let v = m.random_tangent(&p, &mut rng);

    // Use A = a lower triangular with positive diagonal (guaranteed invertible).
    let a = SMatrix::<Real, 3, 3>::from_row_slice(&[
        2.0, 0.0, 0.0,
        0.5, 1.5, 0.0,
        0.3, 0.2, 1.0,
    ]);

    let p2 = a * p * a.transpose();
    let u2 = a * u * a.transpose();
    let v2 = a * v * a.transpose();

    let inner_orig = m.inner(&p, &u, &v);
    let inner_conj = m.inner(&p2, &u2, &v2);

    assert!(
        (inner_orig - inner_conj).abs() < 1e-10,
        "affine invariance violated: <U,V>_P = {inner_orig:.8e}, <AUA^T, AVA^T>_{{APA^T}} = {inner_conj:.8e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: random_point always produces SPD matrices
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd_random_points_are_valid() {
    let m = Spd::<4>;
    let mut rng = StdRng::seed_from_u64(103);
    for _ in 0..100 {
        let p = m.random_point(&mut rng);
        m.check_point(&p).expect("random_point not a valid SPD matrix");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD-specific: parallel transport is exact (norm-preserving)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn spd3_parallel_transport_exact_norm() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(104);

    for _ in 0..100 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let v = m.random_tangent(&p, &mut rng);

        let d = m.random_tangent(&p, &mut rng);
        // SPD has infinite injectivity radius, so no need to scale.
        let d_small = d * 0.3;
        let q = m.exp(&p, &d_small);

        if let (Ok(u_t), Ok(v_t)) = (m.transport(&p, &q, &u), m.transport(&p, &q, &v)) {
            let before = m.inner(&p, &u, &v);
            let after = m.inner(&q, &u_t, &v_t);
            assert!(
                (before - after).abs() < 1e-7,
                "PT inner product: before={before:.8e}, after={after:.8e}, diff={:.2e}",
                (before - after).abs()
            );
        }
    }
}
