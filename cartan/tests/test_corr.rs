// ~/cartan/cartan/tests/test_corr.rs

//! Integration tests for the Corr<N> manifold (correlation matrices).
//!
//! Corr(N) = { C ∈ Sym(N) : C > 0, C_{ii} = 1 } with the Frobenius metric.
//! It is flat: an open subset of the affine subspace {C_{ii} = 1} of Sym(N).
//!
//! Key properties verified:
//! - Geodesics are straight lines: Exp_C(V) = C + V, Log_C(Q) = Q - C.
//! - Parallel transport is the identity (flat connection).
//! - Sectional and scalar curvature are identically zero.
//! - project_point (nearest correlation matrix) converges to a valid Corr(N) point.
//! - Geodesic interpolation is linear: gamma(t) = (1-t)C + tQ.

mod common;

use cartan_core::{
    Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Real, Retraction,
};
use cartan_manifolds::Corr;
use nalgebra::SMatrix;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// Harness tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_base_identities() {
    let m = Corr::<3>;
    // Tolerance 1e-10: Corr(N) uses iterative Higham projection (tol=1e-12), so
    // idempotency is guaranteed to ~1e-11; 1e-10 avoids floating-point edge cases.
    common::matrix_harness::test_matrix_manifold_base::<3, _>(&m, 1e-10, 200);
}

#[test]
fn corr4_base_identities() {
    let m = Corr::<4>;
    common::matrix_harness::test_matrix_manifold_base::<4, _>(&m, 1e-10, 100);
}

#[test]
fn corr3_transport() {
    let m = Corr::<3>;
    common::matrix_harness::test_matrix_transport::<3, _>(&m, 1e-12, 200);
}

#[test]
fn corr3_parallel_transport() {
    let m = Corr::<3>;
    common::matrix_harness::test_matrix_parallel_transport::<3, _>(&m, 1e-12, 200);
}

#[test]
fn corr3_retraction() {
    let m = Corr::<3>;
    common::matrix_harness::test_matrix_retraction::<3, _>(&m, 1e-8, 200);
}

#[test]
fn corr3_curvature() {
    let m = Corr::<3>;
    common::matrix_harness::test_matrix_curvature::<3, _>(&m, 1e-14, 200);
}

#[test]
fn corr3_geodesic() {
    let m = Corr::<3>;
    common::matrix_harness::test_matrix_geodesic::<3, _>(&m, 1e-12, 200);
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat geometry: curvature is exactly zero
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_curvature_exactly_zero() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(200);

    for _ in 0..200 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let v = m.random_tangent(&p, &mut rng);
        let w = m.random_tangent(&p, &mut rng);

        let r = m.riemann_curvature(&p, &u, &v, &w);
        assert!(
            r.norm() < 1e-15,
            "R(u,v)w should be exactly 0, got ||R|| = {:.2e}",
            r.norm()
        );

        let k = m.sectional_curvature(&p, &u, &v);
        assert_eq!(k, 0.0, "sectional curvature should be exactly 0");

        let s = m.scalar_curvature(&p);
        assert_eq!(s, 0.0, "scalar curvature should be exactly 0");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat geometry: parallel transport is the identity
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_parallel_transport_is_identity() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(201);

    for _ in 0..200 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let v = m.random_tangent(&p, &mut rng);

        let d = m.random_tangent(&p, &mut rng);
        let d_small = d * 0.05;
        let q = m.exp(&p, &d_small);

        let u_t = m.transport(&p, &q, &u).unwrap();
        let v_t = m.transport(&p, &q, &v).unwrap();

        // Transport is the identity: transported vector = original.
        assert!(
            (u_t - u).norm() < 1e-14,
            "PT should be identity: ||u_t - u|| = {:.2e}",
            (u_t - u).norm()
        );
        assert!(
            (v_t - v).norm() < 1e-14,
            "PT should be identity: ||v_t - v|| = {:.2e}",
            (v_t - v).norm()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Flat geometry: geodesic is linear interpolation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_geodesic_is_linear() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(202);

    for _ in 0..100 {
        let p = m.random_point(&mut rng);
        let v = m.random_tangent(&p, &mut rng);
        let d_small = v * 0.1;
        let q = m.exp(&p, &d_small);

        for k in 0..=10 {
            let t = k as Real / 10.0;
            let gamma_t = m.geodesic(&p, &q, t).unwrap();
            let linear_t = p * (1.0 - t) + q * t;
            let err = (gamma_t - linear_t).norm();
            assert!(
                err < 1e-13,
                "geodesic is not linear at t={t}: err = {err:.2e}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// project_point: Higham alternating projections converges to valid Corr matrix
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_project_point_finds_valid_corr() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(203);

    for _ in 0..50 {
        // Construct a random symmetric matrix with off-diagonal entries in (-0.8, 0.8).
        let a: SMatrix<Real, 3, 3> = SMatrix::from_fn(|i, j| {
            if i == j {
                1.0
            } else {
                let mut rng2 = rng.clone();
                rng2.random::<Real>() * 1.6 - 0.8
            }
        });
        let sym = (a + a.transpose()) * 0.5;
        // Set diagonal to 1.
        let mut input = sym;
        input[(0, 0)] = 1.0;
        input[(1, 1)] = 1.0;
        input[(2, 2)] = 1.0;

        let c = m.project_point(&input);
        m.check_point(&c)
            .expect("project_point result not a valid correlation matrix");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// project_point: already-valid Corr matrix is unchanged
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_project_point_fixpoint() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(204);

    for _ in 0..100 {
        let c = m.random_point(&mut rng);
        let c2 = m.project_point(&c);
        let err = (c - c2).norm();
        assert!(
            err < 1e-8,
            "project_point moved a valid correlation matrix: err = {err:.2e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Corr(2): analytical check — Corr(2) is parameterized by r ∈ (-1, 1)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr2_is_interval() {
    let m = Corr::<2>;
    // Corr(2) = { [[1, r], [r, 1]] : -1 < r < 1 }.
    for k in [-9i32, -7, -5, -3, 0, 3, 5, 7, 9] {
        let r = k as Real / 10.0;
        let c = SMatrix::<Real, 2, 2>::from_row_slice(&[1.0, r, r, 1.0]);
        m.check_point(&c)
            .unwrap_or_else(|e| panic!("Corr(2) point r={r} invalid: {e}"));
    }
    // r = ±1 must fail (boundary of PD cone).
    for &r in &[-1.0_f64, 1.0_f64] {
        let c = SMatrix::<Real, 2, 2>::from_row_slice(&[1.0, r, r, 1.0]);
        assert!(
            m.check_point(&c).is_err(),
            "Corr(2) r={r} should fail (singular)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Injectivity radius: minimum eigenvalue of C
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr3_injectivity_radius_is_min_eigenvalue() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(205);

    for _ in 0..50 {
        let c = m.random_point(&mut rng);
        let r = m.injectivity_radius(&c);
        assert!(r > 0.0, "injectivity radius should be positive, got {r}");
        assert!(
            r <= 1.0,
            "injectivity radius of a correlation matrix cannot exceed 1, got {r}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dim() formula: N(N-1)/2
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn corr_dim() {
    assert_eq!(Corr::<2>.dim(), 1);
    assert_eq!(Corr::<3>.dim(), 3);
    assert_eq!(Corr::<4>.dim(), 6);
    assert_eq!(Corr::<5>.dim(), 10);
}
