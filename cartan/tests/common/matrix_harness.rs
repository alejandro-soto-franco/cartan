// ~/cartan/cartan/tests/common/matrix_harness.rs
#![allow(dead_code)]

//! Generic test harnesses for matrix manifold identity verification.
//!
//! These functions are parameterized over any M: Manifold where Point and Tangent
//! are SMatrix<Real, N, N> (N×N real matrices). They run the same mathematical
//! identities as the vector harness (`manifold_harness.rs`), but adapted for
//! matrix-valued points and tangent vectors.
//!
//! These harnesses are designed for manifolds embedded in the space of N×N matrices,
//! such as SO(N), SE(N), GL(N), SPD(N), etc.
//!
//! The harnesses are tiered by trait bounds (same as the vector harness):
//! - test_matrix_manifold_base: requires only Manifold
//! - test_matrix_retraction: requires Manifold + Retraction
//! - test_matrix_transport: requires Manifold + VectorTransport
//! - test_matrix_parallel_transport: requires Manifold + ParallelTransport
//! - test_matrix_curvature: requires Manifold + Curvature
//! - test_matrix_geodesic: requires Manifold + GeodesicInterpolation
//!
//! ## Key differences from vector harness
//!
//! - Use `SMatrix<Real, N, N>` instead of `SVector<Real, N>`.
//! - Random ambient matrices: `SMatrix::from_fn(|_, _| rng.sample(StandardNormal))`.
//! - Norm/equality checks use the Frobenius norm `(a - b).norm()`.
//! - The `v * scale` scalar multiplication is replaced by `v * scale` using SMatrix's
//!   scalar-mul impl (which is defined for nalgebra matrices).
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

use nalgebra::SMatrix;
use rand::rngs::StdRng;
use rand::Rng; // required for rng.sample() calls
use rand::SeedableRng;
use rand_distr::StandardNormal;

use cartan_core::*;

use super::approx::*;

// ─────────────────────────────────────────────────────────────────────────────
// Internal assertion helper — matrix version
// ─────────────────────────────────────────────────────────────────────────────

/// Assert two N×N matrices are approximately equal, measured by Frobenius norm.
///
/// Panics with a detailed message showing the Frobenius distance and tolerance
/// if the two matrices differ by more than `tol`.
///
/// This is the matrix analogue of `assert_vec_eq` in approx.rs.
fn assert_mat_eq<const N: usize>(
    actual: &SMatrix<Real, N, N>,
    expected: &SMatrix<Real, N, N>,
    tol: Real,
    context: &str,
) {
    // Frobenius norm of the difference: ||A - B||_F = sqrt(sum_{ij} (A_ij - B_ij)^2)
    let err = (actual - expected).norm();
    assert!(
        err < tol,
        "{}: ||actual - expected||_F = {:.2e} >= tol {:.2e}",
        context, err, tol
    );
}

/// Assert a matrix has Frobenius norm approximately zero.
///
/// Used to check that a matrix which should vanish (e.g., a curvature sum,
/// a zero tangent, a skew-symmetry residual) is numerically zero.
fn assert_mat_near_zero<const N: usize>(
    actual: &SMatrix<Real, N, N>,
    tol: Real,
    context: &str,
) {
    let norm = actual.norm();
    assert!(
        norm < tol,
        "{}: expected ~0 matrix, got ||A||_F = {:.2e} (tol {:.2e})",
        context, norm, tol
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Base manifold harness
// ─────────────────────────────────────────────────────────────────────────────

/// Run base manifold identity tests for a manifold with N×N matrix Point/Tangent types.
///
/// Tests:
/// - `check_point` on random_point outputs
/// - `zero_tangent` has zero norm and passes `check_tangent`
/// - `random_tangent` passes `check_tangent`
/// - `exp(p, v)` lands on the manifold
/// - Exp/log roundtrip: `log(p, exp(p, v_small)) ≈ v_small`
/// - The other roundtrip direction: `exp(p, log(p, q)) ≈ q`
/// - Distance symmetry: `d(p,q) = d(q,p)`
/// - Self-distance: `d(p,p) = 0`
/// - Inner product positive definiteness: `<v,v>_p ≥ 0`
/// - Inner product symmetry: `<v,w>_p = <w,v>_p`
/// - Inner product linearity (bilinearity in first argument)
/// - `project_tangent` idempotence
/// - `project_point` idempotence
/// - `check_point(project_point(ambient_matrix))`
/// - `check_tangent(p, project_tangent(p, ambient_matrix))`
/// - Triangle inequality: `d(p,r) ≤ d(p,q) + d(q,r)`
///
/// All these properties must hold for every correctly implemented Riemannian manifold.
pub fn test_matrix_manifold_base<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>>,
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
        // The zero matrix at p must be a valid tangent and have zero norm under the metric.
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
        // random_tangent must return a matrix that passes check_tangent.
        let v = manifold.random_tangent(&p, &mut rng);
        manifold.check_tangent(&p, &v).unwrap_or_else(|e| {
            panic!("sample {}: random_tangent failed check_tangent: {}", i, e)
        });

        // -- Exp lands on manifold --
        // exp(p, v) must return a matrix that passes check_point.
        let exp_pv = manifold.exp(&p, &v);
        manifold.check_point(&exp_pv).unwrap_or_else(|e| {
            panic!("sample {}: exp(p, v) failed check_point: {}", i, e)
        });

        // -- Exp/log roundtrip: log(p, exp(p, v_small)) ≈ v_small --
        // Scale v so ||v||_metric < injectivity_radius * 0.5 to stay well inside
        // the injectivity ball. This avoids numerical issues near the cut locus.
        //
        // We use manifold.norm() (the metric norm) not Frobenius norm, since the
        // metric norm is what determines proximity to the cut locus.
        let v_norm = manifold.norm(&p, &v);
        let scale = if v_norm > 1e-10 && inj_radius.is_finite() {
            // For compact manifolds (SO(N)): cap at half the injectivity radius.
            (inj_radius * 0.5).min(v_norm) / v_norm
        } else if v_norm > 1e-10 {
            // For non-compact manifolds: cap at unit metric norm.
            1.0_f64.min(1.0 / v_norm)
        } else {
            // Very small vector: no scaling needed.
            1.0
        };
        let v_small = v * scale; // SMatrix supports scalar multiplication
        let q_exp = manifold.exp(&p, &v_small);
        if let Ok(v_recovered) = manifold.log(&p, &q_exp) {
            assert_mat_eq(
                &v_recovered,
                &v_small,
                tol * 100.0, // slightly looser for roundtrip to handle numerical errors
                &format!("sample {}: log(p, exp(p, v)) ≈ v", i),
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

        // -- Inner product positive definiteness: <v, v>_p ≥ 0 --
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
        // The projection onto the tangent space at p must be idempotent (a projector).
        let proj_v = manifold.project_tangent(&p, &v);
        let proj_proj_v = manifold.project_tangent(&p, &proj_v);
        assert_mat_eq(
            &proj_proj_v,
            &proj_v,
            tol,
            &format!("sample {}: project_tangent idempotent", i),
        );

        // -- project_point idempotence: project_point(project_point(p)) = project_point(p) --
        // The projection onto the manifold must be idempotent.
        let proj_p = manifold.project_point(&p);
        let proj_proj_p = manifold.project_point(&proj_p);
        assert_mat_eq(
            &proj_proj_p,
            &proj_p,
            tol,
            &format!("sample {}: project_point idempotent", i),
        );

        // -- exp(p, log(p, q)) ≈ q (the other roundtrip direction) --
        // Only test if log succeeds (i.e., q is not on the cut locus of p).
        if let Ok(v_pq) = manifold.log(&p, &q) {
            let q_recovered = manifold.exp(&p, &v_pq);
            assert_mat_eq(
                &q_recovered,
                &q,
                tol * 100.0,
                &format!("sample {}: exp(p, log(p, q)) ≈ q", i),
            );
        }

        // -- Inner product linearity: <a*v + b*w, w2>_p = a*<v,w2>_p + b*<w,w2>_p --
        // Verifies bilinearity of the metric tensor in the first argument.
        let w2 = manifold.random_tangent(&p, &mut rng);
        let a: Real = 2.3;
        let b: Real = -0.7;
        // v * a + w * b: scalar multiplication and addition for SMatrix
        let lhs = manifold.inner(&p, &(v * a + w * b), &w2);
        let rhs = a * manifold.inner(&p, &v, &w2) + b * manifold.inner(&p, &w, &w2);
        assert_real_eq(
            lhs,
            rhs,
            tol * 10.0,
            &format!("sample {}: inner product linearity", i),
        );

        // -- check_point(project_point(arbitrary_ambient_matrix)) --
        // An arbitrary ambient N×N matrix projected onto the manifold must pass check_point.
        let arb: SMatrix<Real, N, N> =
            SMatrix::from_fn(|_, _| rng.sample::<f64, _>(StandardNormal));
        let projected = manifold.project_point(&arb);
        manifold.check_point(&projected).unwrap_or_else(|e| {
            panic!("sample {}: project_point result failed check_point: {}", i, e)
        });

        // -- check_tangent(p, project_tangent(p, arbitrary_ambient_matrix)) --
        // An arbitrary ambient N×N matrix projected onto T_pM must pass check_tangent.
        let arb_t: SMatrix<Real, N, N> =
            SMatrix::from_fn(|_, _| rng.sample::<f64, _>(StandardNormal));
        let projected_t = manifold.project_tangent(&p, &arb_t);
        manifold.check_tangent(&p, &projected_t).unwrap_or_else(|e| {
            panic!(
                "sample {}: project_tangent result failed check_tangent: {}",
                i, e
            )
        });
    }

    // -- Triangle inequality: d(p, r) ≤ d(p, q) + d(q, r) --
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
                i,
                d_pr,
                d_pq + d_qr
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector transport harness
// ─────────────────────────────────────────────────────────────────────────────

/// Test vector transport preserves inner products (approximately) and is
/// the identity when transporting along the zero direction.
///
/// VectorTransport in cartan-core has signature:
///   vector_transport(p, direction, u) -> Result<Tangent, CartanError>
/// where `direction` is the displacement in T_pM, and the destination
/// q = exp(p, direction) is computed by the blanket impl internally.
///
/// Note: Vector transport (as opposed to parallel transport) only needs to
/// approximately preserve inner products. For the zero direction, the transport
/// should be the identity.
pub fn test_matrix_transport<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>> + VectorTransport,
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
            // For compact manifolds (SO(N)): use 30% of the injectivity radius.
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

            // Inner product preservation (approximate for vector transport):
            // <T(u), T(w)>_q ≈ <u, w>_p
            //
            // Use a looser tolerance for vector transport (it may not be exactly isometric;
            // it only needs to be a "retraction-compatible transport").
            let inner_before = manifold.inner(&p, &u, &w);
            let inner_after = manifold.inner(&q, &u_transported, &w_transported);
            assert_real_eq(
                inner_after,
                inner_before,
                tol * 1000.0,
                &format!("sample {}: transport preserves inner product (approx)", i),
            );
        }

        // -- Transport identity: vector_transport(p, 0, v) ≈ v --
        // Transporting along the zero direction should be the identity.
        let zero_dir = manifold.zero_tangent(&p);
        if let Ok(v_self) = manifold.vector_transport(&p, &zero_dir, &u) {
            assert_mat_eq(
                &v_self,
                &u,
                tol,
                &format!("sample {}: vector_transport(p, 0, v) = v", i),
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel transport harness
// ─────────────────────────────────────────────────────────────────────────────

/// Test parallel transport preserves inner products exactly and keeps vectors
/// in the tangent space.
///
/// ParallelTransport in cartan-core has signature:
///   transport(p, q, u) -> Result<Tangent, CartanError>
/// where `q` is the destination point.
///
/// Parallel transport is isometric: ||P_{p→q} u||_q = ||u||_p exactly.
/// This is stricter than vector transport, which is only approximately isometric.
pub fn test_matrix_parallel_transport<const N: usize, M>(
    manifold: &M,
    tol: Real,
    n_samples: usize,
) where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>> + ParallelTransport,
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
            // -- Exact inner product preservation --
            // <P(u), P(w)>_q = <u, w>_p exactly (not just approximately).
            let inner_before = manifold.inner(&p, &u, &w);
            let inner_after = manifold.inner(&q, &u_t, &w_t);
            assert_real_eq(
                inner_after,
                inner_before,
                tol,
                &format!("sample {}: parallel transport preserves inner product exactly", i),
            );

            // -- Transported vector is in tangent space at q --
            // check_tangent(q, P(u)) should succeed.
            manifold.check_tangent(&q, &u_t).unwrap_or_else(|e| {
                panic!("sample {}: transported vector not in T_qM: {}", i, e)
            });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Retraction harness
// ─────────────────────────────────────────────────────────────────────────────

/// Test retraction roundtrip, centering condition, and manifold membership.
///
/// Tests:
/// - `retract(p, 0) = p` (centering condition)
/// - `retract(p, v)` lies on the manifold
/// - `inverse_retract(p, retract(p, v)) ≈ v` (roundtrip)
pub fn test_matrix_retraction<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>> + Retraction,
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

        // -- Centering condition: retract(p, 0) = p --
        // Disambiguate between Manifold::retract (default exp) and Retraction::retract
        // by calling through the Retraction trait explicitly.
        let r_zero = Retraction::retract(manifold, &p, &manifold.zero_tangent(&p));
        assert_mat_eq(
            &r_zero,
            &p,
            tol,
            &format!("sample {}: retract(p, 0) = p", i),
        );

        // -- retract(p, v) must land on the manifold --
        let q = Retraction::retract(manifold, &p, &v_small);
        manifold.check_point(&q).unwrap_or_else(|e| {
            panic!("sample {}: retract(p, v) failed check_point: {}", i, e)
        });

        // -- Roundtrip: inverse_retract(p, retract(p, v)) ≈ v --
        if let Ok(v_recovered) = manifold.inverse_retract(&p, &q) {
            assert_mat_eq(
                &v_recovered,
                &v_small,
                tol * 100.0,
                &format!("sample {}: inverse_retract(p, retract(p, v)) ≈ v", i),
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature harness
// ─────────────────────────────────────────────────────────────────────────────

/// Test curvature identities: skew-symmetry and the first Bianchi identity.
///
/// These are algebraic identities that hold for the Riemann curvature tensor
/// of any Riemannian manifold:
///
/// - **Skew-symmetry:** R(u,v)w = -R(v,u)w  (equivalently R(u,v)w + R(v,u)w = 0)
/// - **First Bianchi identity:** R(u,v)w + R(v,w)u + R(w,u)v = 0
///
/// Both are checked at each of the n_samples random base points.
pub fn test_matrix_curvature<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>> + Curvature,
{
    let mut rng = StdRng::seed_from_u64(47);

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);
        let w = manifold.random_tangent(&p, &mut rng);

        // -- Skew-symmetry: R(u,v)w + R(v,u)w = 0 --
        // R(u,v)w = -R(v,u)w must hold for all tangent vectors u, v, w.
        let r_uvw = manifold.riemann_curvature(&p, &u, &v, &w);
        let r_vuw = manifold.riemann_curvature(&p, &v, &u, &w);
        let skew_sum = r_uvw.clone() + r_vuw;
        assert_mat_near_zero(
            &skew_sum,
            tol * 10.0,
            &format!("sample {}: skew-symmetry R(u,v)w + R(v,u)w = 0", i),
        );

        // -- First Bianchi identity: R(u,v)w + R(v,w)u + R(w,u)v = 0 --
        // This is the algebraic Bianchi identity, which holds for any Levi-Civita connection.
        let r_vwu = manifold.riemann_curvature(&p, &v, &w, &u);
        let r_wuv = manifold.riemann_curvature(&p, &w, &u, &v);
        let bianchi = r_uvw + r_vwu + r_wuv;
        assert_mat_near_zero(
            &bianchi,
            tol * 10.0,
            &format!("sample {}: first Bianchi identity R(u,v)w + R(v,w)u + R(w,u)v = 0", i),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geodesic interpolation harness
// ─────────────────────────────────────────────────────────────────────────────

/// Test geodesic interpolation boundary values and constant speed.
///
/// Tests:
/// - `geodesic(p, q, 0) = p` (initial endpoint)
/// - `geodesic(p, q, 1) = q` (terminal endpoint)
/// - `dist(p, geodesic(p, q, 0.5)) = 0.5 * dist(p, q)` (constant speed)
///
/// The constant speed test verifies that the interpolation moves at constant velocity,
/// which is the defining property of a geodesic.
pub fn test_matrix_geodesic<const N: usize, M>(manifold: &M, tol: Real, n_samples: usize)
where
    M: Manifold<Point = SMatrix<Real, N, N>, Tangent = SMatrix<Real, N, N>>
        + GeodesicInterpolation,
{
    let mut rng = StdRng::seed_from_u64(44);

    for i in 0..n_samples {
        let p = manifold.random_point(&mut rng);
        let q = manifold.random_point(&mut rng);

        // Skip if p and q are on each other's cut locus (no well-defined geodesic segment).
        if manifold.log(&p, &q).is_err() {
            continue;
        }

        // -- Boundary value at t=0: geodesic(p, q, 0) = p --
        if let Ok(g0) = manifold.geodesic(&p, &q, 0.0) {
            assert_mat_eq(
                &g0,
                &p,
                tol,
                &format!("sample {}: geodesic(p,q,0) = p", i),
            );
        }

        // -- Boundary value at t=1: geodesic(p, q, 1) = q --
        if let Ok(g1) = manifold.geodesic(&p, &q, 1.0) {
            assert_mat_eq(
                &g1,
                &q,
                tol,
                &format!("sample {}: geodesic(p,q,1) = q", i),
            );
        }

        // -- Constant speed: dist(p, gamma(0.5)) = 0.5 * dist(p, q) --
        // The geodesic γ(t) from p to q satisfies ||γ'(t)||_γ(t) = const = dist(p,q),
        // so the arc length from p to γ(0.5) is 0.5 * dist(p, q).
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
