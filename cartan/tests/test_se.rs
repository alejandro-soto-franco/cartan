// ~/cartan/cartan/tests/test_se.rs

//! Integration tests for the SpecialEuclidean<N> manifold.
//!
//! SE(N) = SO(N) ⋉ R^N is the special Euclidean group: the group of rigid-body
//! transformations (rotation + translation) in N-dimensional space. Because SE(N)
//! uses custom point and tangent wrapper types (`SEPoint<N>` and `SETangent<N>`
//! rather than `SMatrix`), the generic `matrix_harness` from the SO(N) tests
//! cannot be reused here. All tests are written directly against the SE(N) API.
//!
//! ## Organization
//!
//! 1. **SE(3) core identity tests** — exp/log roundtrip, metric properties,
//!    projection idempotence, check_point/check_tangent, zero tangent.
//! 2. **SE(3) trait tests** — Retraction, ParallelTransport, GeodesicInterpolation.
//! 3. **SE-specific structure tests** — pure rotation, pure translation, weight
//!    effect on distance, SE(2) roundtrip.
//!
//! ## Tolerance conventions
//!
//! - Core identity tests (exp/log, metric): 1e-8 (matrix exponential + Jacobian)
//! - Parallel transport norm preservation: 1e-6 (approximate product transport)
//! - Geodesic boundary/speed: 1e-7 (exp ∘ log composition)
//! - Pure rotation/translation structural tests: 1e-12 to 1e-14 (closed-form)
//! - Inner product symmetry: 1e-14 (single arithmetic operation)
//! - Distance symmetry: 1e-12
//!
//! ## Why different tolerances?
//!
//! The SE(N) exponential map requires:
//!   (a) SO(N) matrix exponential (Rodrigues / Padé, ~1e-15 error for SO(3))
//!   (b) Left Jacobian J(Ω) (series expansion, ~1e-14 for ||Ω|| < 0.5)
//!   (c) Left Jacobian inverse J(Ω)^{-1} (another series expansion)
//!   (d) Matrix multiplications and vector additions
//! Each step compounds floating-point error. The 1e-8 roundtrip tolerance
//! matches the SO(N) convention and accounts for O(N²) accumulation.

mod common;

use cartan_manifolds::{SpecialEuclidean, se::{SEPoint, SETangent}};
use cartan_core::{GeodesicInterpolation, Manifold, ParallelTransport, Real, Retraction};
use nalgebra::{SMatrix, SVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities for comparing SEPoint and SETangent
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a scalar "distance" between two SEPoints for assertion purposes.
///
/// We measure:
///   ||A.rotation - B.rotation||_F + ||A.translation - B.translation||_2
///
/// This is NOT the Riemannian distance (which requires log and norm), but a
/// simple ambient Frobenius + L2 sum that is zero iff both components agree.
/// For points near each other (as in roundtrip tests), this is a reasonable
/// proxy for closeness.
///
/// Note: We clone because nalgebra's SMatrix subtraction consumes inputs when
/// not using references, but the trait bound requires Clone. Explicit clone
/// keeps the code clear.
fn se_point_diff<const N: usize>(a: &SEPoint<N>, b: &SEPoint<N>) -> f64 {
    // Frobenius norm of rotation matrix difference
    let rot_diff = (a.rotation.clone() - b.rotation.clone()).norm();
    // L2 norm of translation vector difference
    let trans_diff = (a.translation.clone() - b.translation.clone()).norm();
    rot_diff + trans_diff
}

/// Compute a scalar "distance" between two SETangent vectors for assertion purposes.
///
/// We measure:
///   ||A.rotation - B.rotation||_F + ||A.translation - B.translation||_2
///
/// Same rationale as `se_point_diff` above: a simple ambient norm that is zero iff
/// both components agree exactly. For tangent-space roundtrip tests, this is
/// appropriate because the tangent error should be < tol in both components.
fn se_tangent_diff<const N: usize>(a: &SETangent<N>, b: &SETangent<N>) -> f64 {
    // Frobenius norm of rotation-component difference
    let rot_diff = (a.rotation.clone() - b.rotation.clone()).norm();
    // L2 norm of translation-component difference
    let trans_diff = (a.translation.clone() - b.translation.clone()).norm();
    rot_diff + trans_diff
}

/// Scale a SETangent vector by a real scalar.
///
/// Returns a new SETangent with both rotation and translation components scaled.
/// This is needed because SETangent only implements `Mul<Real>` (consuming), and
/// we sometimes want to scale without consuming the original.
fn scale_tangent<const N: usize>(v: &SETangent<N>, s: f64) -> SETangent<N> {
    SETangent {
        rotation: v.rotation.clone() * s,
        translation: v.translation.clone() * s,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SE(3) core identity tests
// ─────────────────────────────────────────────────────────────────────────────

/// Exp/log roundtrip (forward): log(p, exp(p, v)) ≈ v for small v.
///
/// Mathematical basis: For any Riemannian manifold M and any point p ∈ M,
/// the logarithmic map is the inverse of the exponential map within the
/// injectivity ball:
///
///   log_p(exp_p(v)) = v  for all v ∈ T_p M with ||v|| < inj_radius(p).
///
/// For SE(3), the injectivity radius is π (limited by the SO(3) factor).
/// We scale tangent vectors to have ||v|| < 0.5 (well inside the ball) to
/// avoid numerical issues near the cut locus.
///
/// Tolerance 1e-8: the exp/log cycle involves matrix_exp_skew → left_jacobian →
/// matrix_log_orthogonal → left_jacobian_inverse, each contributing ~1e-14 per
/// step; with 4 steps and N=3, accumulated error is ~1e-8 to 1e-12.
#[test]
fn se3_exp_log_roundtrip_forward() {
    // weight = 1.0: equal weighting of rotation and translation in the metric.
    // This is the standard choice and the one most likely to expose bugs in the
    // coupling between rotation and translation in the SE(N) exponential.
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(100);

    // Tolerance: 1e-8 matches SO(3) exp/log roundtrip precision.
    let tol = 1e-8;
    // Target tangent norm: stay within 0.5 << π ≈ 3.14 (injectivity radius).
    let target_norm = 0.5;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let v = se3.random_tangent(&p, &mut rng);

        // Compute the Riemannian norm of v to determine the scaling factor.
        // We want ||v_scaled||_p = target_norm, so scale = target_norm / ||v||_p.
        let v_norm = se3.norm(&p, &v);
        let scale = if v_norm > 1e-10 { target_norm / v_norm } else { 1.0 };
        let v_small = scale_tangent(&v, scale);

        // Compute exp(p, v_small) and then log(p, result).
        let q = se3.exp(&p, &v_small);
        let v_recovered = se3
            .log(&p, &q)
            .unwrap_or_else(|e| panic!("sample {}: log(p, exp(p, v)) failed: {}", i, e));

        // The recovered tangent should equal v_small within tolerance.
        let diff = se_tangent_diff(&v_recovered, &v_small);
        assert!(
            diff < tol,
            "sample {}: log(p, exp(p, v)) ≠ v: tangent diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// Exp/log roundtrip (reverse): exp(p, log(p, q)) ≈ q.
///
/// Mathematical basis: For any Riemannian manifold M and p, q ∈ M with q in the
/// injectivity ball of p:
///
///   exp_p(log_p(q)) = q.
///
/// This is the other direction of the exp/log inverse relationship. We test it
/// separately because numerical errors can accumulate differently in this direction:
/// log is applied first, which can amplify errors if p and q are close to the
/// cut locus.
///
/// We use random_point for both p and q. Since random translations are Gaussian,
/// ||t_q - t_p|| can be large, which can cause ||v|| > π and log to fail at the
/// SO(3) cut locus. If log fails, we skip that sample (noted below).
#[test]
fn se3_exp_log_roundtrip_reverse() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(101);

    let tol = 1e-8;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        // To ensure log(p, q) succeeds (rotation component within π),
        // construct q = exp(p, v_small) rather than fully random.
        // This guarantees q is within the injectivity ball.
        let v = se3.random_tangent(&p, &mut rng);
        let v_norm = se3.norm(&p, &v);
        // Scale to 0.4 < π: well within the injectivity ball of SE(3).
        let scale = if v_norm > 1e-10 { 0.4 / v_norm } else { 1.0 };
        let v_small = scale_tangent(&v, scale);
        let q = se3.exp(&p, &v_small);

        // log(p, q) must succeed because q = exp(p, v_small) with ||v_small|| < π.
        let v_pq = se3
            .log(&p, &q)
            .unwrap_or_else(|e| panic!("sample {}: log(p, q) failed: {}", i, e));

        // exp(p, log(p, q)) should recover q.
        let q_recovered = se3.exp(&p, &v_pq);
        let diff = se_point_diff(&q_recovered, &q);
        assert!(
            diff < tol,
            "sample {}: exp(p, log(p, q)) ≠ q: point diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// check_point passes for random SE(3) points.
///
/// Any point produced by `random_point` must satisfy the SE(N) manifold constraints:
///   1. R^T R = I (orthogonality of rotation component)
///   2. det(R) = +1 (positive orientation)
///
/// The translation component is unconstrained, so it always passes.
///
/// This test verifies that `random_point` produces valid SE(3) elements.
#[test]
fn se3_check_point_on_random_points() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(102);

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        se3.check_point(&p).unwrap_or_else(|e| {
            panic!("sample {}: random_point failed check_point: {}", i, e)
        });
    }
}

/// check_tangent passes for random SE(3) tangent vectors.
///
/// Any tangent vector produced by `random_tangent` must satisfy:
///   R^T V_rot ∈ so(3)  (the rotation component is in the Lie algebra)
///
/// This is checked by verifying that R^T V_rot is skew-symmetric.
/// The translation component is unconstrained, so it always passes.
#[test]
fn se3_check_tangent_on_random_tangents() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(103);

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let v = se3.random_tangent(&p, &mut rng);
        se3.check_tangent(&p, &v).unwrap_or_else(|e| {
            panic!("sample {}: random_tangent failed check_tangent: {}", i, e)
        });
    }
}

/// Inner product symmetry: <u, v>_p = <v, u>_p.
///
/// Mathematical basis: The Riemannian metric tensor is a symmetric bilinear form.
/// For SE(N) with weight w, the inner product is:
///
///   <u, v>_{(R,t)} = (1/2) tr(Ω_u^T Ω_v) + w · v_body_u^T v_body_v
///
/// The rotation term (1/2) tr(Ω_u^T Ω_v) = (1/2) tr(Ω_v^T Ω_u) (trace symmetry).
/// The translation term v_body_u^T v_body_v = v_body_v^T v_body_u (dot product symmetry).
/// Therefore, <u, v> = <v, u> exactly, up to floating-point precision.
///
/// We use tol = 1e-14 (near machine epsilon) since this involves only a few
/// floating-point operations and should be nearly exact.
#[test]
fn se3_inner_product_symmetry() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(104);

    // Tolerance: 1e-14 is achievable because symmetry holds analytically for
    // the trace formula with a single matrix multiplication per term.
    let tol = 1e-14;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let u = se3.random_tangent(&p, &mut rng);
        let v = se3.random_tangent(&p, &mut rng);

        let inner_uv = se3.inner(&p, &u, &v);
        let inner_vu = se3.inner(&p, &v, &u);

        let diff = (inner_uv - inner_vu).abs();
        assert!(
            diff < tol,
            "sample {}: |<u,v> - <v,u>| = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// Inner product linearity: <a·u + b·w, v>_p = a·<u,v>_p + b·<w,v>_p.
///
/// Mathematical basis: The metric tensor is bilinear, so for any scalar a, b and
/// tangent vectors u, v, w ∈ T_p M:
///
///   <a·u + b·w, v>_p = a·<u, v>_p + b·<w, v>_p
///
/// This follows from the linearity of trace and dot product.
///
/// Tolerance 1e-10: linearity involves O(N²) multiplications in the trace term,
/// slightly more error accumulation than symmetry alone.
#[test]
fn se3_inner_product_linearity() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(105);

    let tol = 1e-10;
    // Fixed scalars a, b for the linear combination test.
    // Chosen to be non-trivial (not 1 or 0) to exercise both terms.
    let a: Real = 2.3;
    let b: Real = -0.7;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let u = se3.random_tangent(&p, &mut rng);
        let v = se3.random_tangent(&p, &mut rng);
        let w = se3.random_tangent(&p, &mut rng);

        // Compute a·u + b·w using SETangent arithmetic (componentwise).
        // SETangent implements Mul<Real> and Add, so this is a·u + b·w.
        let lin_combo = scale_tangent(&u, a) + scale_tangent(&w, b);

        // LHS: <a·u + b·w, v>
        let lhs = se3.inner(&p, &lin_combo, &v);

        // RHS: a·<u,v> + b·<w,v>
        let rhs = a * se3.inner(&p, &u, &v) + b * se3.inner(&p, &w, &v);

        let diff = (lhs - rhs).abs();
        assert!(
            diff < tol,
            "sample {}: |<au+bw, v> - a<u,v> - b<w,v>| = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// Distance symmetry: d(p, q) = d(q, p).
///
/// Mathematical basis: The Riemannian distance function is a metric, hence symmetric.
/// For SE(N):
///   d(p, q)² = ||log_p(q)||_p² = <log_p(q), log_p(q)>_p
///
/// Symmetry holds because the geodesic from p to q and from q to p have equal length.
///
/// Additionally, d(p, p) = 0 because log_p(p) = 0.
///
/// Tolerance for symmetry: 1e-12 (involves log + norm, error is small but not machine eps).
/// Tolerance for d(p,p) = 0: 1e-14 (log(p,p) = 0 exactly, norm(0) = 0).
#[test]
fn se3_distance_symmetry_and_self() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(106);

    let tol_sym = 1e-12;
    let tol_self = 1e-14;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        // Construct q within the injectivity ball to ensure log succeeds.
        let v = se3.random_tangent(&p, &mut rng);
        let v_norm = se3.norm(&p, &v);
        let scale = if v_norm > 1e-10 { 0.4 / v_norm } else { 1.0 };
        let q = se3.exp(&p, &scale_tangent(&v, scale));

        // Distance symmetry: dist(p, q) = dist(q, p).
        // dist() returns Result because log can fail at the cut locus.
        if let (Ok(d_pq), Ok(d_qp)) = (se3.dist(&p, &q), se3.dist(&q, &p)) {
            let sym_err = (d_pq - d_qp).abs();
            assert!(
                sym_err < tol_sym,
                "sample {}: |d(p,q) - d(q,p)| = {:.2e} > tol = {:.2e}",
                i, sym_err, tol_sym
            );
        }

        // d(p, p) = 0: log_p(p) = 0, so ||0||_p = 0.
        if let Ok(d_pp) = se3.dist(&p, &p) {
            assert!(
                d_pp < tol_self,
                "sample {}: d(p,p) = {:.2e} > tol = {:.2e} (should be 0)",
                i, d_pp, tol_self
            );
        }
    }
}

/// project_tangent is idempotent: project(p, project(p, v)) = project(p, v).
///
/// Mathematical basis: The tangent space T_p SE(N) is a closed subspace of the
/// ambient space. Projection is a linear map that satisfies π² = π (idempotent).
///
/// For SE(N), the rotation component is projected via:
///   Ω = skew(R^T V_rot)  (symmetrize to get a skew-symmetric matrix)
///   V_rot_proj = R · Ω
///
/// The translation component passes through unchanged (it is already unconstrained).
///
/// A second projection of V_rot_proj is: R^T (R Ω) = Ω, and skew(Ω) = Ω (since Ω
/// is already skew-symmetric), so V_rot_proj is already in T_R SO(3). Hence
/// project(p, project(p, v)) = project(p, v).
///
/// Tolerance 1e-8: the projection involves a matrix multiply and skew extraction,
/// matching the standard matrix operation tolerance.
#[test]
fn se3_project_tangent_idempotent() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(107);

    let tol = 1e-8;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let v = se3.random_tangent(&p, &mut rng);

        // First projection.
        let proj_v = se3.project_tangent(&p, &v);
        // Second projection.
        let proj_proj_v = se3.project_tangent(&p, &proj_v);

        // The two projections should be identical.
        let diff = se_tangent_diff(&proj_proj_v, &proj_v);
        assert!(
            diff < tol,
            "sample {}: project_tangent not idempotent: diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// project_point is idempotent on the rotation component.
///
/// Mathematical basis: project_point(p) projects the rotation component onto SO(N)
/// via Newton polar iteration. Applying it twice should give the same result,
/// because the projected R is already orthogonal with det = +1.
///
/// The translation component passes through unchanged in both applications.
///
/// We verify idempotence by checking that the rotation difference is small
/// (same criterion as se_point_diff).
///
/// Tolerance 1e-8: polar iteration converges quadratically; the result is already
/// very close to SO(N), so the second projection changes almost nothing.
#[test]
fn se3_project_point_idempotent() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(108);

    let tol = 1e-8;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);

        // First projection.
        let proj_p = se3.project_point(&p);
        // Second projection.
        let proj_proj_p = se3.project_point(&proj_p);

        // Both projections should agree.
        let diff = se_point_diff(&proj_proj_p, &proj_p);
        assert!(
            diff < tol,
            "sample {}: project_point not idempotent: diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );

        // Additionally verify that the projected rotation component is in SO(3).
        // This checks that project_point produces valid SE(3) elements.
        se3.check_point(&proj_p).unwrap_or_else(|e| {
            panic!("sample {}: project_point result failed check_point: {}", i, e)
        });
    }
}

/// Zero tangent has zero Riemannian norm.
///
/// Mathematical basis: The zero tangent vector 0 ∈ T_p M must satisfy ||0||_p = 0.
/// For SE(N): zero_tangent returns (0_{N×N}, 0_N), and
///   ||0||² = (1/2) tr(0^T 0) + w · 0^T 0 = 0 + 0 = 0.
///
/// Tolerance 1e-14: the norm of the zero vector is computed exactly as 0 (no floating-
/// point cancellation), so this should hold to machine precision.
#[test]
fn se3_zero_tangent_has_zero_norm() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(109);

    let tol = 1e-14;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let zero = se3.zero_tangent(&p);

        let norm = se3.norm(&p, &zero);
        assert!(
            norm < tol,
            "sample {}: ||zero_tangent|| = {:.2e} > tol = {:.2e} (should be 0)",
            i, norm, tol
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SE(3) trait tests
// ─────────────────────────────────────────────────────────────────────────────

/// Retraction: retract(p, v) is a valid SE(3) point for random small v.
///
/// Mathematical basis: The Cayley × Euclidean product retraction on SE(N) satisfies
/// the manifold-landing property: retract(p, v) ∈ SE(N) for all (p, v).
///
/// Specifically, the Cayley map for the rotation component is:
///   R_new = R · (I - Ω/2)^{-1} (I + Ω/2)
/// which is exactly orthogonal (not just approximately), so R_new ∈ SO(3) exactly.
///
/// The translation component is just t + v_trans, which is always in R^N.
///
/// Therefore, check_point should succeed for all retracted points.
#[test]
fn se3_retraction_lands_on_manifold() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(200);

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let v = se3.random_tangent(&p, &mut rng);

        // Scale to be small (0.4 < π), consistent with being within the retraction
        // domain. The Cayley map is defined for all skew Ω, but inverse_retract
        // can fail near eigenvalue -1.
        let v_norm = se3.norm(&p, &v);
        let scale = if v_norm > 1e-10 { 0.4 / v_norm } else { 1.0 };
        let v_small = scale_tangent(&v, scale);

        // Retract and verify membership.
        let q = Retraction::retract(&se3, &p, &v_small);
        se3.check_point(&q).unwrap_or_else(|e| {
            panic!("sample {}: retract(p, v) failed check_point: {}", i, e)
        });
    }
}

/// Parallel transport approximately preserves Riemannian norm.
///
/// Mathematical basis: Parallel transport is an isometry of tangent spaces.
/// For SE(N) with the product metric, the implemented transport is:
///
///   transported_rot = Q R^T u_rot  (left-translation, exact for SO(N))
///   transported_trans = u_trans    (identity, exact for R^N)
///
/// This is the exact parallel transport for the SO(N) × R^N product metric.
/// For the SE(N) semidirect product metric, the coupling introduces O(||Ω||²)
/// corrections that we neglect, making the transport approximate.
///
/// We test norm preservation: ||transported_u||_q ≈ ||u||_p.
///
/// Tolerance 1e-6: tighter than for VectorTransport (which tolerates 1e-3)
/// because the SO(N) transport is exact. The residual is only from the coupling.
#[test]
fn se3_transport_preserves_norm() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(201);

    let tol = 1e-6;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let u = se3.random_tangent(&p, &mut rng);

        // Destination q = exp(p, d_small): close to p to minimize coupling error.
        let d = se3.random_tangent(&p, &mut rng);
        let d_norm = se3.norm(&p, &d);
        let scale = if d_norm > 1e-10 { 0.3 / d_norm } else { 1.0 };
        let d_small = scale_tangent(&d, scale);
        let q = se3.exp(&p, &d_small);

        // Transport u from T_p SE(3) to T_q SE(3).
        let u_transported = se3
            .transport(&p, &q, &u)
            .unwrap_or_else(|e| panic!("sample {}: transport failed: {}", i, e));

        // Check that the transported norm matches the original norm.
        let norm_before = se3.norm(&p, &u);
        let norm_after = se3.norm(&q, &u_transported);
        let norm_diff = (norm_before - norm_after).abs();

        assert!(
            norm_diff < tol,
            "sample {}: |||P u||_q - ||u||_p| = {:.2e} > tol = {:.2e}",
            i, norm_diff, tol
        );
    }
}

/// Geodesic boundary conditions: geodesic(p, q, 0) = p and geodesic(p, q, 1) = q.
///
/// Mathematical basis: The geodesic γ: [0,1] → SE(3) defined by
///   γ(t) = exp_p(t · log_p(q))
/// satisfies the boundary conditions:
///   γ(0) = exp_p(0) = p
///   γ(1) = exp_p(log_p(q)) = q
///
/// We test that these hold numerically within tol = 1e-8.
///
/// Note: We use exp(p, v_small) for q to ensure log(p, q) succeeds.
#[test]
fn se3_geodesic_boundary_conditions() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(202);

    let tol = 1e-8;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        // Construct q within the injectivity ball.
        let v = se3.random_tangent(&p, &mut rng);
        let v_norm = se3.norm(&p, &v);
        let scale = if v_norm > 1e-10 { 0.4 / v_norm } else { 1.0 };
        let q = se3.exp(&p, &scale_tangent(&v, scale));

        // -- geodesic(p, q, 0) = p --
        let g0 = se3
            .geodesic(&p, &q, 0.0)
            .unwrap_or_else(|e| panic!("sample {}: geodesic(p,q,0) failed: {}", i, e));
        let diff0 = se_point_diff(&g0, &p);
        assert!(
            diff0 < tol,
            "sample {}: geodesic(p,q,0) ≠ p: diff = {:.2e} > tol = {:.2e}",
            i, diff0, tol
        );

        // -- geodesic(p, q, 1) = q --
        let g1 = se3
            .geodesic(&p, &q, 1.0)
            .unwrap_or_else(|e| panic!("sample {}: geodesic(p,q,1) failed: {}", i, e));
        let diff1 = se_point_diff(&g1, &q);
        assert!(
            diff1 < tol,
            "sample {}: geodesic(p,q,1) ≠ q: diff = {:.2e} > tol = {:.2e}",
            i, diff1, tol
        );
    }
}

/// Geodesic constant speed: d(p, γ(0.5)) = d(p, q) / 2.
///
/// Mathematical basis: A geodesic parameterized by arc length satisfies
///   d(p, γ(t)) = t · d(p, q)
/// for all t ∈ [0, 1]. In particular, at t = 0.5, the midpoint is equidistant
/// from both endpoints:
///   d(p, γ(0.5)) = d(p, q) / 2.
///
/// Tolerance 1e-7: the geodesic midpoint is computed as exp(p, 0.5 · log(p,q)).
/// The exp and log each contribute ~1e-12 error, but the distance computation
/// requires another log, totaling ~3 applications of exp/log → ~1e-7 tolerance.
#[test]
fn se3_geodesic_constant_speed() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(203);

    let tol = 1e-7;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        // Construct q within the injectivity ball, with nontrivial distance
        // from p (otherwise the speed test is trivially true at d = 0).
        let v = se3.random_tangent(&p, &mut rng);
        let v_norm = se3.norm(&p, &v);
        // Use a tangent of norm 0.3 to get a non-negligible geodesic length.
        let scale = if v_norm > 1e-10 { 0.3 / v_norm } else { 1.0 };
        let q = se3.exp(&p, &scale_tangent(&v, scale));

        // Compute d(p, q) and the midpoint γ(0.5).
        let d_pq = se3.dist(&p, &q).unwrap_or_else(|e| {
            panic!("sample {}: dist(p,q) failed: {}", i, e)
        });
        let g_half = se3
            .geodesic(&p, &q, 0.5)
            .unwrap_or_else(|e| panic!("sample {}: geodesic(p,q,0.5) failed: {}", i, e));

        // d(p, γ(0.5)) should equal d(p, q) / 2.
        let d_half = se3.dist(&p, &g_half).unwrap_or_else(|e| {
            panic!("sample {}: dist(p, g(0.5)) failed: {}", i, e)
        });

        let speed_err = (d_half - 0.5 * d_pq).abs();
        assert!(
            speed_err < tol,
            "sample {}: |d(p, g(0.5)) - d(p,q)/2| = {:.2e} > tol = {:.2e}",
            i, speed_err, tol
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SE-specific structural tests
// ─────────────────────────────────────────────────────────────────────────────

/// Pure rotation: exp at identity with zero translation leaves translation unchanged.
///
/// Mathematical basis: For p = (I, 0) and v = (V_rot, 0) (pure rotation, no translation),
/// the SE(N) exponential map gives:
///
///   Ω = I^T V_rot = V_rot ∈ so(3)   (body-frame = world-frame at identity)
///   v_body = I^T · 0 = 0
///   R_new = I · exp(Ω) = exp(Ω) ∈ SO(3)
///   t_new = 0 + I · J(Ω) · 0 = 0
///
/// Therefore, pure rotation preserves the zero translation.
///
/// Tolerance 1e-12: J(Ω) · 0 = 0 exactly (matrix times zero vector = zero),
/// so there is no accumulated error in the translation component.
#[test]
fn se3_pure_rotation_preserves_zero_translation() {
    // weight = 1.0: the pure rotation test is weight-independent (translation = 0).
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };

    // Base point: identity rotation + zero translation = the group identity of SE(3).
    let p_identity = SEPoint::<3> {
        rotation: SMatrix::<Real, 3, 3>::identity(),
        translation: SVector::<Real, 3>::zeros(),
    };

    // Tangent vector: pure rotation around z-axis by 45° (π/4), zero translation.
    // The Lie algebra element is Ω = (π/4) · e₃ (45° rotation in the xy-plane).
    let angle = std::f64::consts::FRAC_PI_4; // π/4 ≈ 0.785 rad
    let mut omega = SMatrix::<Real, 3, 3>::zeros();
    omega[(0, 1)] = -angle; // Ω[0,1] = -θ (skew-symmetric: upper-right)
    omega[(1, 0)] = angle;  // Ω[1,0] = +θ (skew-symmetric: lower-left)
    // Ω[2,*] = 0: pure rotation in the xy-plane, no z component.

    // The tangent at the identity is V_rot = I · Ω = Ω.
    let v_pure_rot = SETangent::<3> {
        rotation: omega, // V_rot = Ω at the identity (I · Ω = Ω)
        translation: SVector::<Real, 3>::zeros(), // zero translational velocity
    };

    // Apply the exponential map.
    let q = se3.exp(&p_identity, &v_pure_rot);

    // Verify that the translation component is still zero.
    let trans_norm = q.translation.norm();
    assert!(
        trans_norm < 1e-12,
        "pure rotation at identity moved translation: ||t_new|| = {:.2e} (expected 0)",
        trans_norm
    );

    // Also verify that the rotation is valid (in SO(3)).
    se3.check_point(&q).expect("pure rotation result should be a valid SE(3) point");
}

/// Pure translation: exp at identity with zero rotation gives R = I and t = v_trans.
///
/// Mathematical basis: For p = (I, 0) and v = (0, v_trans) (pure translation, no rotation),
/// the SE(N) exponential map gives:
///
///   Ω = I^T · 0 = 0                 (zero rotation velocity → Ω = 0)
///   v_body = I^T · v_trans = v_trans (body-frame = world-frame at identity)
///   R_new = I · exp(0) = I · I = I   (rotation unchanged)
///   t_new = 0 + I · J(0) · v_trans  = J(0) · v_trans
///
/// For Ω = 0: J(0) = I (the left Jacobian at zero rotation is the identity matrix),
/// so t_new = v_trans.
///
/// Therefore, exp((I, 0), (0, [1,2,3])) = (I, [1,2,3]).
///
/// Tolerance 1e-14: J(0) = I exactly (no rotation, so the Jacobian is the
/// identity without any approximation), hence t_new = v_trans to machine precision.
#[test]
fn se3_pure_translation_gives_correct_result() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };

    // Base point: identity rotation + zero translation.
    let p_identity = SEPoint::<3> {
        rotation: SMatrix::<Real, 3, 3>::identity(),
        translation: SVector::<Real, 3>::zeros(),
    };

    // Tangent vector: pure translation [1, 2, 3], zero rotation.
    // At the identity, the body-frame velocity equals the spatial-frame velocity.
    let t_desired = SVector::<Real, 3>::new(1.0, 2.0, 3.0);
    let v_pure_trans = SETangent::<3> {
        rotation: SMatrix::<Real, 3, 3>::zeros(), // zero rotational velocity
        translation: t_desired.clone(),           // spatial translational velocity
    };

    // Apply the exponential map.
    let q = se3.exp(&p_identity, &v_pure_trans);

    // Verify that the rotation is still the identity.
    let id = SMatrix::<Real, 3, 3>::identity();
    let rot_err = (q.rotation.clone() - id).norm();
    assert!(
        rot_err < 1e-14,
        "pure translation changed rotation: ||R_new - I||_F = {:.2e} (expected 0)",
        rot_err
    );

    // Verify that the translation is exactly [1, 2, 3].
    // J(0) = I, so t_new = 0 + I · I · [1,2,3] = [1,2,3].
    let trans_err = (q.translation.clone() - t_desired).norm();
    assert!(
        trans_err < 1e-14,
        "pure translation: ||t_new - [1,2,3]|| = {:.2e} (expected 0)",
        trans_err
    );
}

/// exp(p, zero_tangent(p)) = p.
///
/// Mathematical basis: The exponential map at the zero tangent vector returns the
/// base point itself. This is the defining property of the exponential map at p:
///   exp_p(0) = p  for all p ∈ M.
///
/// For SE(N): exp((R, t), (0, 0)) = (R · exp(0), t + R · J(0) · 0) = (R · I, t + 0) = (R, t).
///
/// Tolerance 1e-14: exp(0) = I exactly (Rodrigues formula for zero axis-angle),
/// and J(0) = I exactly, so no error should accumulate.
#[test]
fn se3_exp_zero_tangent_is_identity() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(300);

    let tol = 1e-14;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);
        let zero = se3.zero_tangent(&p);

        // exp(p, 0) should return p.
        let q = se3.exp(&p, &zero);
        let diff = se_point_diff(&q, &p);
        assert!(
            diff < tol,
            "sample {}: exp(p, 0) ≠ p: diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}

/// log(p, p) = zero tangent (norm < 1e-14).
///
/// Mathematical basis: For any Riemannian manifold and any point p ∈ M:
///   log_p(p) = 0 ∈ T_p M.
///
/// This is because the geodesic from p to p has zero length, so the logarithm
/// (which gives the initial velocity of the geodesic) must be the zero vector.
///
/// For SE(N): log((R,t), (R,t)) requires:
///   Ω = matrix_log(R^T R) = matrix_log(I) = 0
///   Δt = R^T (t - t) = 0
///   v_body = J(0)^{-1} · 0 = 0
///   V_rot = R · 0 = 0
///   v_trans = R · 0 = 0
///
/// Tolerance 1e-14: matrix_log(I) = 0 exactly (identity has all eigenvalues 1,
/// log(1) = 0), and the rest follows.
#[test]
fn se3_log_self_is_zero() {
    let se3 = SpecialEuclidean::<3> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(301);

    let tol = 1e-14;

    for i in 0..10 {
        let p = se3.random_point(&mut rng);

        let v = se3
            .log(&p, &p)
            .unwrap_or_else(|e| panic!("sample {}: log(p, p) failed: {}", i, e));

        // The norm of log(p, p) must be zero.
        let v_norm = se3.norm(&p, &v);
        assert!(
            v_norm < tol,
            "sample {}: ||log(p, p)||_p = {:.2e} > tol = {:.2e} (expected 0)",
            i, v_norm, tol
        );
    }
}

/// Weight parameter affects distance: higher weight → larger distance for same
/// non-zero translation component.
///
/// Mathematical basis: The Riemannian metric on SE(N) is:
///   <u, v> = (1/2) tr(Ω_u^T Ω_v) + weight · v_body_u^T v_body_v
///
/// For two points with the same rotation component (R₁ = R₂ = I) but different
/// translations (t₁ ≠ t₂), the distance is determined by the translational part.
/// Specifically:
///
///   d(p_low, q_low) = sqrt(weight_low) · ||Δt||
///   d(p_high, q_high) = sqrt(weight_high) · ||Δt||
///
/// So d_high / d_low = sqrt(weight_high / weight_low) > 1 when weight_high > weight_low.
///
/// We verify: d_{w=10.0}(p, q) > d_{w=0.1}(p, q) for points with nonzero translation.
#[test]
fn se3_weight_affects_translation_distance() {
    // Two manifolds with very different weights.
    let se3_low = SpecialEuclidean::<3> { weight: 0.1 };  // rotation-dominant
    let se3_high = SpecialEuclidean::<3> { weight: 10.0 }; // translation-dominant

    // Create two SE(3) points with the SAME rotation (identity) but different translations.
    // This isolates the effect of weight on the translational distance.
    let id = SMatrix::<Real, 3, 3>::identity();
    let t1 = SVector::<Real, 3>::zeros();          // translation of point 1
    let t2 = SVector::<Real, 3>::new(1.0, 0.0, 0.0); // translation of point 2 (1 unit along x)

    let p = SEPoint::<3> { rotation: id.clone(), translation: t1 };
    let q = SEPoint::<3> { rotation: id.clone(), translation: t2 };

    // For (I, 0) → (I, [1,0,0]):
    //   log = (0, [1,0,0])  (pure translation, Ω = 0, v_body = [1,0,0])
    //   d_low² = (1/2) tr(0) + 0.1 · 1 = 0.1  → d_low = sqrt(0.1)
    //   d_high² = (1/2) tr(0) + 10.0 · 1 = 10 → d_high = sqrt(10)
    //   d_high / d_low = sqrt(10/0.1) = sqrt(100) = 10

    let d_low = se3_low.dist(&p, &q).expect("dist with low weight should succeed");
    let d_high = se3_high.dist(&p, &q).expect("dist with high weight should succeed");

    assert!(
        d_high > d_low,
        "weight=10 distance ({:.4}) should be greater than weight=0.1 distance ({:.4})",
        d_high, d_low
    );

    // Quantitative check: d_high / d_low should be ~10 (sqrt(10/0.1) = sqrt(100) = 10).
    // We verify this is at least 5 (loose check) to avoid false positives.
    let ratio = d_high / d_low;
    assert!(
        ratio > 5.0,
        "weight ratio should produce d_high/d_low ≈ 10, got {:.4}",
        ratio
    );

    // Also verify the exact values match theory:
    //   d_low = sqrt(weight_low) · ||Δt|| = sqrt(0.1) · 1 ≈ 0.3162
    //   d_high = sqrt(weight_high) · ||Δt|| = sqrt(10) · 1 ≈ 3.1623
    let expected_low = (0.1_f64).sqrt();
    let expected_high = (10.0_f64).sqrt();
    assert!(
        (d_low - expected_low).abs() < 1e-8,
        "d_low = {:.6}, expected {:.6}",
        d_low, expected_low
    );
    assert!(
        (d_high - expected_high).abs() < 1e-8,
        "d_high = {:.6}, expected {:.6}",
        d_high, expected_high
    );
}

/// SE(2) exp/log roundtrip test.
///
/// Mathematical basis: SE(2) = SO(2) ⋉ R² is the group of planar rigid motions.
/// Its Lie algebra se(2) has dimension 3 (one rotation angle + two translations).
/// The same exp/log formulas apply, but with N=2.
///
/// For SE(2) with the product metric (weight = 1.0), we verify:
///   log(p, exp(p, v)) = v  for ||v|| < 0.5
///
/// This serves as a dimensionality check: the implementation should work
/// for arbitrary N, not just N=3.
///
/// Tolerance 1e-8: same as SE(3), since the key operations (matrix_exp, J, J^{-1})
/// are equally well-conditioned for SO(2) (1D rotation, exact formula available).
#[test]
fn se2_exp_log_roundtrip() {
    // SE(2): planar rigid motions (1 rotation + 2 translations = 3 DOF).
    let se2 = SpecialEuclidean::<2> { weight: 1.0 };
    let mut rng = StdRng::seed_from_u64(400);

    let tol = 1e-8;
    let target_norm = 0.5;

    for i in 0..10 {
        let p = se2.random_point(&mut rng);
        let v = se2.random_tangent(&p, &mut rng);

        // Scale to target norm to stay well within the injectivity ball.
        let v_norm = se2.norm(&p, &v);
        let scale = if v_norm > 1e-10 { target_norm / v_norm } else { 1.0 };
        let v_small = scale_tangent(&v, scale);

        // Forward roundtrip: log(p, exp(p, v)) = v.
        let q = se2.exp(&p, &v_small);
        let v_recovered = se2
            .log(&p, &q)
            .unwrap_or_else(|e| panic!("SE(2) sample {}: log(p, exp(p, v)) failed: {}", i, e));

        let diff = se_tangent_diff(&v_recovered, &v_small);
        assert!(
            diff < tol,
            "SE(2) sample {}: log(p, exp(p, v)) ≠ v: diff = {:.2e} > tol = {:.2e}",
            i, diff, tol
        );
    }
}
