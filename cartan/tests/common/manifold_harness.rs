// ~/cartan/cartan/tests/common/manifold_harness.rs
#![allow(dead_code)]

//! Generic test harnesses for manifold identity verification.
//!
//! These functions are parameterized over any M: Manifold and run
//! the mathematical identities that all Riemannian manifolds must satisfy.
//! Each manifold's test module calls the applicable subset.
//!
//! The harnesses are tiered by trait bounds:
//! - test_manifold_base: requires only Manifold
//! - test_retraction: requires Manifold + Retraction
//! - test_transport: requires Manifold + VectorTransport
//! - test_parallel_transport: requires Manifold + ParallelTransport
//! - test_curvature: requires Manifold + Curvature
//! - test_geodesic: requires Manifold + GeodesicInterpolation
//!
//! ## API note
//!
//! The cartan-core `VectorTransport::vector_transport` signature is:
//!   vector_transport(p, direction, u) -> Result<Tangent, CartanError>
//! where `direction` is the displacement tangent vector (not the destination point).
//! The destination is computed internally as exp(p, direction).
//!
//! The cartan-core `ParallelTransport::transport` signature is:
//!   transport(p, q, u) -> Result<Tangent, CartanError>
//! where `q` is the destination point (already computed).

use nalgebra::SVector;
use rand::rngs::StdRng;
use rand::Rng; // required for rng.sample() calls
use rand::SeedableRng;
use rand_distr;

use cartan_core::*;

use super::approx::*;

/// Run base manifold identity tests.
///
/// Tests: exp/log roundtrip (both directions), metric consistency (distance
/// symmetry, triangle inequality, inner product properties), projection
/// idempotence, validation, zero_tangent, and check_point/check_tangent on
/// projected arbitrary vectors.
///
/// These properties must hold for every correctly implemented Riemannian manifold.
/// If these fail on a curved manifold, the implementation has a bug. If they fail
/// here (on Euclidean R^N), then the test harness itself has a bug.
pub fn test_manifold_base<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>>,
{
    let mut rng = StdRng::seed_from_u64(42);
    // Query injectivity radius once; used to scale tangent vectors so that
    // the exp/log roundtrip test stays within the injectivity ball.
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let q = manifold.random_point(&mut rng);

        // -- Validation: random_point must pass check_point --
        manifold.check_point(&p).unwrap_or_else(|e| {
            panic!("sample {}: random_point failed check_point: {}", i, e)
        });

        // -- Zero tangent --
        // The zero vector at p must be a valid tangent and have zero norm.
        let zero = manifold.zero_tangent(&p);
        manifold.check_tangent(&p, &zero).unwrap_or_else(|e| {
            panic!("sample {}: zero_tangent failed check_tangent: {}", i, e)
        });
        assert_near_zero(
            manifold.norm(&p, &zero),
            tol,
            &format!("sample {}: zero_tangent norm", i),
        );

        // -- Random tangent validation --
        // random_tangent must return a vector that passes check_tangent.
        let v = manifold.random_tangent(&p, &mut rng);
        manifold.check_tangent(&p, &v).unwrap_or_else(|e| {
            panic!("sample {}: random_tangent failed check_tangent: {}", i, e)
        });

        // -- Exp lands on manifold --
        // exp(p, v) must return a point that passes check_point.
        let exp_pv = manifold.exp(&p, &v);
        manifold.check_point(&exp_pv).unwrap_or_else(|e| {
            panic!("sample {}: exp(p, v) failed check_point: {}", i, e)
        });

        // -- Exp/log roundtrip: log(p, exp(p, v_small)) ~ v_small --
        // Scale v so ||v|| < injectivity_radius * 0.5 to stay well inside the
        // injectivity ball and avoid numerical issues near the cut locus.
        let v_norm = manifold.norm(&p, &v);
        let scale = if v_norm > 1e-10 && inj_radius.is_finite() {
            // For compact manifolds: cap at half the injectivity radius.
            (inj_radius * 0.5).min(v_norm) / v_norm
        } else if v_norm > 1e-10 {
            // For non-compact manifolds (R^N, SPD): cap at unit norm.
            1.0_f64.min(1.0 / v_norm)
        } else {
            // Very small vector: no scaling needed.
            1.0
        };
        let v_small = v * scale;
        let q_exp = manifold.exp(&p, &v_small);
        if let Ok(v_recovered) = manifold.log(&p, &q_exp) {
            assert_vec_eq(
                &v_recovered,
                &v_small,
                tol * 100.0, // slightly looser for roundtrip to handle numerical errors
                &format!("sample {}: log(p, exp(p, v)) ~ v", i),
            );
        }

        // -- Distance symmetry: dist(p, q) = dist(q, p) --
        if let (Ok(d_pq), Ok(d_qp)) = (manifold.dist(&p, &q), manifold.dist(&q, &p)) {
            assert_real_eq(
                d_pq,
                d_qp,
                tol,
                &format!("sample {}: dist symmetry", i),
            );
        }

        // -- dist(p, p) = 0 --
        if let Ok(d_pp) = manifold.dist(&p, &p) {
            assert_near_zero(d_pp, tol, &format!("sample {}: dist(p, p)", i));
        }

        // -- Inner product positive definiteness: <v, v>_p >= 0 --
        let v_inner = manifold.inner(&p, &v, &v);
        assert_nonneg(v_inner, &format!("sample {}: inner(p, v, v) >= 0", i));

        // -- Inner product symmetry: <v, w>_p = <w, v>_p --
        let w = manifold.random_tangent(&p, &mut rng);
        let inner_vw = manifold.inner(&p, &v, &w);
        let inner_wv = manifold.inner(&p, &w, &v);
        assert_real_eq(
            inner_vw,
            inner_wv,
            tol,
            &format!("sample {}: inner symmetry", i),
        );

        // -- Projection idempotence: project_tangent(p, project_tangent(p, v)) = project_tangent(p, v) --
        let proj_v = manifold.project_tangent(&p, &v);
        let proj_proj_v = manifold.project_tangent(&p, &proj_v);
        assert_vec_eq(
            &proj_proj_v,
            &proj_v,
            tol,
            &format!("sample {}: project_tangent idempotent", i),
        );

        // -- project_point idempotence: project_point(project_point(p)) = project_point(p) --
        let proj_p = manifold.project_point(&p);
        let proj_proj_p = manifold.project_point(&proj_p);
        assert_vec_eq(
            &proj_proj_p,
            &proj_p,
            tol,
            &format!("sample {}: project_point idempotent", i),
        );

        // -- exp(p, log(p, q)) ~ q (the other roundtrip direction) --
        // Only test if log succeeds (i.e., q is not on the cut locus of p).
        if let Ok(v_pq) = manifold.log(&p, &q) {
            let q_recovered = manifold.exp(&p, &v_pq);
            assert_vec_eq(
                &q_recovered,
                &q,
                tol * 100.0,
                &format!("sample {}: exp(p, log(p, q)) ~ q", i),
            );
        }

        // -- Inner product linearity: <a*v + b*w, w2>_p = a*<v, w2>_p + b*<w, w2>_p --
        // Verifies bilinearity of the metric tensor in the first argument.
        let w2 = manifold.random_tangent(&p, &mut rng);
        let a: Real = 2.3;
        let b: Real = -0.7;
        let lhs = manifold.inner(&p, &(v * a + w * b), &w2);
        let rhs = a * manifold.inner(&p, &v, &w2) + b * manifold.inner(&p, &w, &w2);
        assert_real_eq(
            lhs,
            rhs,
            tol * 10.0,
            &format!("sample {}: inner product linearity", i),
        );

        // -- check_point(project_point(arbitrary)) --
        // An arbitrary ambient vector projected onto the manifold must pass check_point.
        let arb: SVector<Real, N> = SVector::from_fn(|_, _| rng.sample::<f64, _>(rand_distr::StandardNormal));
        let projected = manifold.project_point(&arb);
        manifold.check_point(&projected).unwrap_or_else(|e| {
            panic!("sample {}: project_point result failed check_point: {}", i, e)
        });

        // -- check_tangent(p, project_tangent(p, arbitrary)) --
        // An arbitrary ambient vector projected onto T_pM must pass check_tangent.
        let arb_t: SVector<Real, N> = SVector::from_fn(|_, _| rng.sample::<f64, _>(rand_distr::StandardNormal));
        let projected_t = manifold.project_tangent(&p, &arb_t);
        manifold.check_tangent(&p, &projected_t).unwrap_or_else(|e| {
            panic!("sample {}: project_tangent result failed check_tangent: {}", i, e)
        });
    }

    // -- Triangle inequality: d(p, r) <= d(p, q) + d(q, r) --
    // Tested in a separate loop to get three distinct random points.
    let mut rng2 = StdRng::seed_from_u64(420);
    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng2);
        let q = manifold.random_point(&mut rng2);
        let r = manifold.random_point(&mut rng2);

        if let (Ok(d_pq), Ok(d_qr), Ok(d_pr)) = (
            manifold.dist(&p, &q),
            manifold.dist(&q, &r),
            manifold.dist(&p, &r),
        ) {
            assert!(
                d_pr <= d_pq + d_qr + tol * 100.0,
                "sample {}: triangle inequality violated: d(p,r) = {:.2e} > d(p,q) + d(q,r) = {:.2e}",
                i, d_pr, d_pq + d_qr
            );
        }
    }
}

/// Test vector transport preserves inner products (approximately).
///
/// VectorTransport in cartan-core has signature:
///   vector_transport(p, direction, u) -> Result<Tangent, CartanError>
/// where `direction` is the displacement in T_pM, and the destination
/// q = exp(p, direction) is computed by the blanket impl internally.
///
/// Also tests the transport identity: vector_transport(p, 0, v) ~ v.
pub fn test_transport<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>> + VectorTransport,
{
    let mut rng = StdRng::seed_from_u64(43);
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // Build a displacement direction d_small that stays within the injectivity ball,
        // so the transport is along a well-defined geodesic.
        let d = manifold.random_tangent(&p, &mut rng);
        let d_norm = manifold.norm(&p, &d);
        let scale = if d_norm > 1e-10 && inj_radius.is_finite() {
            // For compact manifolds: use 30% of the injectivity radius.
            (inj_radius * 0.3) / d_norm
        } else if d_norm > 1e-10 {
            // For non-compact manifolds: use a small but non-trivial displacement.
            0.5 / d_norm
        } else {
            1.0
        };
        let d_small = d * scale;

        // Transport both u and w along the direction d_small.
        // vector_transport signature: (p, direction, tangent_to_transport)
        if let (Ok(u_transported), Ok(w_transported)) = (
            manifold.vector_transport(&p, &d_small, &u),
            manifold.vector_transport(&p, &d_small, &w),
        ) {
            // The destination is q = exp(p, d_small).
            let q = manifold.exp(&p, &d_small);

            // Inner product preservation (approximate for vector transport,
            // exact up to tolerance for parallel transport).
            let inner_before = manifold.inner(&p, &u, &w);
            let inner_after = manifold.inner(&q, &u_transported, &w_transported);
            // Use a looser tolerance for vector transport (may not be isometric).
            assert_real_eq(
                inner_after,
                inner_before,
                tol * 1000.0,
                &format!("sample {}: transport preserves inner product", i),
            );
        }

        // -- Transport identity: vector_transport(p, 0, v) ~ v --
        // Transporting along the zero direction should be the identity.
        let zero_dir = manifold.zero_tangent(&p);
        if let Ok(v_self) = manifold.vector_transport(&p, &zero_dir, &u) {
            assert_vec_eq(
                &v_self,
                &u,
                tol,
                &format!("sample {}: vector_transport(p, 0, v) = v", i),
            );
        }
    }
}

/// Test parallel transport preserves inner products exactly (stricter than vector transport).
///
/// ParallelTransport in cartan-core has signature:
///   transport(p, q, u) -> Result<Tangent, CartanError>
/// where `q` is the destination point.
///
/// Parallel transport is isometric: ||P_p^q u||_q = ||u||_p exactly.
pub fn test_parallel_transport<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>> + ParallelTransport,
{
    let mut rng = StdRng::seed_from_u64(45);
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // Compute a destination q within the injectivity ball.
        let d = manifold.random_tangent(&p, &mut rng);
        let d_norm = manifold.norm(&p, &d);
        let scale = if d_norm > 1e-10 && inj_radius.is_finite() {
            (inj_radius * 0.3) / d_norm
        } else if d_norm > 1e-10 {
            0.5 / d_norm
        } else {
            1.0
        };
        let d_small = d * scale;
        // q is the endpoint of the geodesic from p in direction d_small.
        let q = manifold.exp(&p, &d_small);

        // parallel transport signature: transport(p, q, u)
        if let (Ok(u_t), Ok(w_t)) = (
            manifold.transport(&p, &q, &u),
            manifold.transport(&p, &q, &w),
        ) {
            // Exact inner product preservation for parallel transport.
            // Unlike vector transport, this must hold with tight tolerance.
            let inner_before = manifold.inner(&p, &u, &w);
            let inner_after = manifold.inner(&q, &u_t, &w_t);
            assert_real_eq(
                inner_after,
                inner_before,
                tol,
                &format!("sample {}: parallel transport preserves inner product exactly", i),
            );

            // Transported vector is in tangent space at q.
            manifold.check_tangent(&q, &u_t).unwrap_or_else(|e| {
                panic!("sample {}: transported vector not in T_qM: {}", i, e)
            });
        }
    }
}

/// Test retraction roundtrip: inverse_retract(p, retract(p, v)) ~ v.
///
/// Also tests:
/// - retract(p, 0) = p (centering condition)
/// - retract(p, v) lies on the manifold
pub fn test_retraction<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>> + Retraction,
{
    let mut rng = StdRng::seed_from_u64(46);
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let v = manifold.random_tangent(&p, &mut rng);

        // Scale tangent vector to stay within the injectivity radius.
        let v_norm = manifold.norm(&p, &v);
        let scale = if v_norm > 1e-10 && inj_radius.is_finite() {
            (inj_radius * 0.3).min(v_norm) / v_norm
        } else if v_norm > 1e-10 {
            0.5_f64.min(1.0 / v_norm)
        } else {
            1.0
        };
        let v_small = v * scale;

        // retract(p, 0) = p: centering condition.
        // Disambiguate between Manifold::retract (default exp) and Retraction::retract
        // by calling through the Retraction trait explicitly.
        let r_zero = Retraction::retract(manifold, &p, &manifold.zero_tangent(&p));
        assert_vec_eq(&r_zero, &p, tol, &format!("sample {}: retract(p, 0) = p", i));

        // retract(p, v) must land on the manifold.
        let q = Retraction::retract(manifold, &p, &v_small);
        manifold.check_point(&q).unwrap_or_else(|e| {
            panic!("sample {}: retract(p, v) failed check_point: {}", i, e)
        });

        // Roundtrip: inverse_retract(p, retract(p, v)) ~ v.
        if let Ok(v_recovered) = manifold.inverse_retract(&p, &q) {
            assert_vec_eq(
                &v_recovered,
                &v_small,
                tol * 100.0,
                &format!("sample {}: inverse_retract(p, retract(p, v)) ~ v", i),
            );
        }
    }
}

/// Test curvature identities: skew-symmetry and the first Bianchi identity.
///
/// These are algebraic identities that hold for the Riemann curvature tensor
/// of any Riemannian manifold:
/// - Skew-symmetry: R(u,v)w = -R(v,u)w
/// - First Bianchi: R(u,v)w + R(v,w)u + R(w,u)v = 0
pub fn test_curvature<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>> + Curvature,
{
    let mut rng = StdRng::seed_from_u64(47);

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // -- Skew-symmetry: R(u,v)w = -R(v,u)w, equivalently R(u,v)w + R(v,u)w = 0 --
        let r_uvw = manifold.riemann_curvature(&p, &u, &v, &w);
        let r_vuw = manifold.riemann_curvature(&p, &v, &u, &w);
        let sum = r_uvw.clone() + r_vuw;
        assert!(
            sum.norm() < tol * 10.0,
            "sample {}: skew-symmetry violated: ||R(u,v)w + R(v,u)w|| = {:.2e}",
            i, sum.norm()
        );

        // -- First Bianchi identity: R(u,v)w + R(v,w)u + R(w,u)v = 0 --
        let r_vwu = manifold.riemann_curvature(&p, &v, &w, &u);
        let r_wuv = manifold.riemann_curvature(&p, &w, &u, &v);
        let bianchi = r_uvw + r_vwu + r_wuv;
        assert!(
            bianchi.norm() < tol * 10.0,
            "sample {}: Bianchi identity violated: ||sum|| = {:.2e}",
            i, bianchi.norm()
        );
    }
}

/// Test geodesic interpolation boundary values and constant speed.
///
/// Tests:
/// - geodesic(p, q, 0) = p
/// - geodesic(p, q, 1) = q
/// - dist(p, geodesic(p, q, 0.5)) = 0.5 * dist(p, q)  (constant speed)
pub fn test_geodesic<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SVector<Real, N>, Tangent = SVector<Real, N>> + GeodesicInterpolation,
{
    let mut rng = StdRng::seed_from_u64(44);

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let q = manifold.random_point(&mut rng);

        // Skip if p and q are on each other's cut locus (no well-defined geodesic).
        if manifold.log(&p, &q).is_err() {
            continue;
        }

        // -- Boundary values: geodesic(p, q, 0) = p --
        if let Ok(g0) = manifold.geodesic(&p, &q, 0.0) {
            assert_vec_eq(&g0, &p, tol, &format!("sample {}: geodesic(p,q,0) = p", i));
        }
        // -- Boundary values: geodesic(p, q, 1) = q --
        if let Ok(g1) = manifold.geodesic(&p, &q, 1.0) {
            assert_vec_eq(&g1, &q, tol, &format!("sample {}: geodesic(p,q,1) = q", i));
        }

        // -- Constant speed: dist(p, gamma(0.5)) = 0.5 * dist(p, q) --
        if let (Ok(d_pq), Ok(g_half)) = (manifold.dist(&p, &q), manifold.geodesic(&p, &q, 0.5)) {
            if let Ok(d_half) = manifold.dist(&p, &g_half) {
                assert_real_eq(
                    d_half,
                    0.5 * d_pq,
                    tol * 100.0,
                    &format!("sample {}: constant speed at t=0.5", i),
                );
            }
        }
    }
}
