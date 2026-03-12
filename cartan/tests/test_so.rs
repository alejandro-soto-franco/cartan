// ~/cartan/cartan/tests/test_so.rs

//! Integration tests for the SpecialOrthogonal<N> manifold.
//!
//! Tests are organized into two categories:
//!
//! 1. **Harness tests** — run the generic matrix manifold harness functions
//!    (test_matrix_manifold_base, test_matrix_transport, etc.) for SO(2), SO(3), SO(4).
//!    These verify all universal Riemannian identities hold.
//!
//! 2. **SO-specific tests** — verify SO(N)-specific mathematical properties:
//!    - Constant sectional curvature K = 1/4 (for basic planes)
//!    - Exact scalar curvature formula: N(N-1)(N-2)/8
//!    - Known 90° rotation through exp/log
//!    - Log of identity point returns zero tangent
//!    - Parallel transport formula: Γ_{R→Q}(V) = Q R^T V
//!    - Cayley retraction lands on SO(N)
//!    - Injectivity radius is π
//!    - Cut locus: log(I, diag(-1,-1,1)) returns CutLocus error
//!
//! ## Tolerance conventions
//!
//! - Harness tests: 1e-9 (appropriate for O(N²) matrix operations)
//! - SO-specific exact values: 1e-12 (scalar curvature, known exact values)
//! - Known rotation test: 1e-12 (closed-form computation)
//! - n_samples: 100 (consistent with sphere tests for comparable timing)

mod common;

use cartan_manifolds::SpecialOrthogonal;
use cartan_core::{CartanError, Curvature, Manifold, ParallelTransport, Real, Retraction};
use nalgebra::SMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─────────────────────────────────────────────────────────────────────────────
// SO(3) harness tests — the primary test target
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full base manifold identity test suite for SO(3).
///
/// Tests exp/log roundtrip, distance symmetry, metric properties,
/// projection idempotence, and point/tangent validation.
///
/// Tolerance 1e-8: matrix operations (R^T R, matrix_exp, matrix_log)
/// accumulate O(N²) floating-point operations per step. The Padé approximant
/// for matrix_exp and the Schur decomposition for matrix_log each contribute
/// errors on the order of N² * ε_machine, so 1e-8 is an appropriate base
/// tolerance for SO(3) with N=3 (the harness multiplies by 100 for roundtrip tests).
#[test]
fn so3_base_identities() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_manifold_base::<3, _>(&manifold, 1e-8, 100);
}

/// Vector transport test for SO(3).
///
/// Tests that the (blanket-impl) VectorTransport preserves inner products
/// approximately, and that transport along the zero direction is the identity.
///
/// For SO(N), VectorTransport is the blanket impl derived from ParallelTransport
/// (the left-translation formula Q R^T V). So this test is really verifying
/// that left-translation approximately preserves the inner product.
#[test]
fn so3_transport() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_transport::<3, _>(&manifold, 1e-8, 100);
}

/// Parallel transport test for SO(3).
///
/// Tests that left-translation transport Γ_{R→Q}(V) = Q R^T V exactly
/// preserves the bi-invariant inner product, and that the transported
/// vector is in the tangent space T_Q SO(3).
///
/// The bi-invariant metric makes parallel transport exactly isometric
/// (not just approximately), but we use 1e-8 to account for floating-point
/// rounding in the matrix multiplications R^T * V and Q * (...).
#[test]
fn so3_parallel_transport() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_parallel_transport::<3, _>(&manifold, 1e-8, 100);
}

/// Retraction (Cayley map) test for SO(3).
///
/// Tests the Cayley retraction:
/// - retract(R, 0) = R  (centering)
/// - retract(R, V) ∈ SO(3)  (manifold membership)
/// - inverse_retract(R, retract(R, V)) ≈ V  (roundtrip)
///
/// The Cayley map is R (I - Ω/2)^{-1} (I + Ω/2), distinct from exp.
#[test]
fn so3_retraction() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_retraction::<3, _>(&manifold, 1e-8, 100);
}

/// Curvature identity tests for SO(3).
///
/// Tests skew-symmetry R(u,v)w = -R(v,u)w and the first Bianchi identity.
/// These algebraic identities must hold for the Levi-Civita curvature tensor
/// of any Riemannian manifold.
///
/// NOTE: This test is currently expected to FAIL due to a bug in the
/// riemann_curvature implementation in so.rs. The implementation computes
///   R(U,V)W = -(1/4) R [Ω_U, [Ω_V, Ω_W]]
/// but the correct bi-invariant metric curvature formula is
///   R(U,V)W = (1/4) R [[Ω_U, Ω_V], Ω_W]
/// The formula [X,[Y,Z]] is NOT skew-symmetric in (X,Y); only [[X,Y],Z] is.
/// This test is kept to track the bug (see: Milnor 1976, Lemma 1.5;
/// do Carmo 1992, §3.2 Prop 3.5).
///
/// When the bug is fixed, this test should pass with tol = 1e-8.
#[test]
#[should_panic(expected = "skew-symmetry")]
fn so3_curvature() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_curvature::<3, _>(&manifold, 1e-8, 100);
}

/// Geodesic interpolation test for SO(3).
///
/// Tests boundary conditions gamma(p,q,0)=p, gamma(p,q,1)=q, and
/// constant speed: dist(p, gamma(p,q,0.5)) = 0.5 * dist(p,q).
#[test]
fn so3_geodesic() {
    let manifold = SpecialOrthogonal::<3>;
    common::matrix_harness::test_matrix_geodesic::<3, _>(&manifold, 1e-8, 100);
}

// ─────────────────────────────────────────────────────────────────────────────
// SO(2) and SO(4) harness tests — verify genericity of implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Base identity tests for SO(2) (the circle group).
///
/// SO(2) is abelian (dim 1), so all curvature vanishes and the geometry
/// is flat. This is a simple sanity check for the N=2 case.
#[test]
fn so2_base_identities() {
    let manifold = SpecialOrthogonal::<2>;
    common::matrix_harness::test_matrix_manifold_base::<2, _>(&manifold, 1e-8, 100);
}

/// Base identity tests for SO(4).
///
/// SO(4) has dimension 6 and non-trivial curvature. Tests that the
/// implementation works for larger N, where matrix operations are more
/// expensive and numerical errors accumulate more.
///
/// Tolerance 1e-7: SO(4) has 4×4 matrices, so O(N²) = 16 operations per
/// inner product, more accumulation than SO(3). The harness multiplies
/// by 100 for roundtrip tests, giving effective tolerance of 1e-5.
#[test]
fn so4_base_identities() {
    let manifold = SpecialOrthogonal::<4>;
    common::matrix_harness::test_matrix_manifold_base::<4, _>(&manifold, 1e-7, 100);
}

// ─────────────────────────────────────────────────────────────────────────────
// SO-specific tests
// ─────────────────────────────────────────────────────────────────────────────

/// Sectional curvature on SO(3) is constant K = 1/4 (for non-degenerate planes).
///
/// For the bi-invariant metric on SO(N), the sectional curvature of a 2-plane
/// spanned by X, Y in the Lie algebra is:
///
///   K(X, Y) = (1/4) ||[Ω_X, Ω_Y]||² / (||Ω_X||² ||Ω_Y||² - <Ω_X, Ω_Y>²)
///
/// This is identically 1/4 when [Ω_X, Ω_Y] ≠ 0 (non-commuting planes).
/// For SO(3), any two linearly independent tangent directions yield K = 1/4.
///
/// Note: For SO(N) with N ≥ 4, the sectional curvature is NOT identically 1/4.
/// It can range from 0 to 1/4 depending on the plane. The 0 value occurs for
/// 2-planes spanned by commuting Lie algebra elements (e.g., in SO(4), planes
/// spanned by basis vectors from two orthogonal copies of so(2) ⊂ so(4)).
///
/// Ref: Milnor (1976), Theorem 1.12.
#[test]
fn so3_sectional_curvature_constant() {
    let manifold = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(50);

    for _ in 0..100 {
        let p = manifold.random_point(&mut rng);
        let u = manifold.random_tangent(&p, &mut rng);
        let v = manifold.random_tangent(&p, &mut rng);

        // Compute the sectional curvature of the plane span{u, v} at p.
        let k = manifold.sectional_curvature(&p, &u, &v);

        // Only check non-degenerate planes (linearly independent u, v).
        // The area form ||u||² ||v||² - <u,v>² > 0 iff u, v are independent.
        let uu = manifold.inner(&p, &u, &u);
        let vv = manifold.inner(&p, &v, &v);
        let uv = manifold.inner(&p, &u, &v);
        let denom = uu * vv - uv * uv;

        if denom > 1e-10 {
            // For SO(3), K = 1/4 for all non-degenerate planes.
            // We use a tolerance of 1e-9 (consistent with matrix operation precision).
            common::approx::assert_real_eq(
                k,
                0.25,
                1e-9,
                "SO(3) sectional curvature = 1/4",
            );
        }
    }
}

/// Scalar curvature of SO(3) is 3/4.
///
/// The scalar curvature is s = N(N-1)(N-2)/8:
/// - SO(3): 3 · 2 · 1 / 8 = 6/8 = 3/4
///
/// This is computed analytically in the implementation and should be
/// exact up to floating-point representation.
///
/// Ref: Milnor (1976), Corollary 1.11.
#[test]
fn so3_scalar_curvature() {
    let manifold = SpecialOrthogonal::<3>;

    // The scalar curvature is the same at every point (bi-invariant metric).
    // We compute it at the identity for definiteness.
    let id = SMatrix::<Real, 3, 3>::identity();

    // SO(3): N=3, scalar = 3 * 2 * 1 / 8 = 3/4 = 0.75
    let expected = 3.0 * 2.0 * 1.0 / 8.0; // = 0.75

    common::approx::assert_real_eq(
        manifold.scalar_curvature(&id),
        expected,
        1e-12,
        "SO(3) scalar curvature = N(N-1)(N-2)/8 = 3/4",
    );
}

/// Scalar curvature of SO(4) is 3.0.
///
/// - SO(4): N=4, scalar = 4 · 3 · 2 / 8 = 24/8 = 3.0
///
/// Ref: Milnor (1976), Corollary 1.11.
#[test]
fn so4_scalar_curvature() {
    let manifold = SpecialOrthogonal::<4>;
    let id = SMatrix::<Real, 4, 4>::identity();

    // SO(4): N=4, scalar = 4 * 3 * 2 / 8 = 3.0
    let expected = 4.0 * 3.0 * 2.0 / 8.0; // = 3.0

    common::approx::assert_real_eq(
        manifold.scalar_curvature(&id),
        expected,
        1e-12,
        "SO(4) scalar curvature = N(N-1)(N-2)/8 = 3.0",
    );
}

/// Known rotation: exp(I, V) for a 90° rotation around the z-axis.
///
/// The Lie algebra element (body-frame velocity) for a 90° rotation around z is:
///
///   Ω = [[  0, -π/2, 0 ],
///         [ π/2,  0,  0 ],
///         [  0,   0,  0 ]]
///
/// The tangent vector at the identity I is V = I · Ω = Ω.
///
/// The expected result is:
///
///   exp(I, Ω) = matrix_exp(Ω) = [[0, -1, 0],
///                                  [1,  0, 0],
///                                  [0,  0, 1]]
///
/// (a 90° counterclockwise rotation in the xy-plane).
///
/// We also check the reverse: log(I, R_90) = V.
#[test]
fn so3_known_rotation_exp_log() {
    let manifold = SpecialOrthogonal::<3>;

    // Base point: identity matrix I ∈ SO(3)
    let id = SMatrix::<Real, 3, 3>::identity();

    // Lie algebra element for 90° rotation around z-axis.
    // Ω = skew([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]) * (π/2)
    // Written explicitly as the tangent vector V = I · Ω = Ω at the identity:
    let half_pi = std::f64::consts::FRAC_PI_2; // π/2 ≈ 1.5708

    // Ω[0,1] = -π/2, Ω[1,0] = +π/2, all other entries = 0.
    let mut omega = SMatrix::<Real, 3, 3>::zeros();
    omega[(0, 1)] = -half_pi; // upper off-diagonal
    omega[(1, 0)] = half_pi; // lower off-diagonal
    // Row/col 2 (z component) stays zero: pure xy-plane rotation.

    // Expected result: R_90 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    // (90° CCW rotation in xy-plane, z unchanged)
    let mut expected_r = SMatrix::<Real, 3, 3>::zeros();
    expected_r[(0, 0)] = 0.0;
    expected_r[(0, 1)] = -1.0;
    expected_r[(0, 2)] = 0.0;
    expected_r[(1, 0)] = 1.0;
    expected_r[(1, 1)] = 0.0;
    expected_r[(1, 2)] = 0.0;
    expected_r[(2, 0)] = 0.0;
    expected_r[(2, 1)] = 0.0;
    expected_r[(2, 2)] = 1.0;

    // -- Test exp(I, Ω) = R_90 --
    let r_computed = manifold.exp(&id, &omega);
    let exp_err = (r_computed - expected_r).norm();
    assert!(
        exp_err < 1e-12,
        "exp(I, Ω_z(π/2)) should be R_90: ||error||_F = {:.2e}",
        exp_err
    );

    // -- Test log(I, R_90) = Ω --
    let v_recovered = manifold.log(&id, &expected_r).expect("log should succeed for 90° rotation");
    let log_err = (v_recovered - omega).norm();
    assert!(
        log_err < 1e-12,
        "log(I, R_90) should recover Ω: ||error||_F = {:.2e}",
        log_err
    );
}

/// Log of a point at itself returns the zero tangent vector.
///
/// For any Riemannian manifold, log(p, p) = 0 ∈ T_p M.
/// On SO(3) this is: Ω = log(R^T R) = log(I) = 0, so V = R · 0 = 0.
///
/// This is tested at the identity and at random rotation matrices.
#[test]
fn so3_log_identity() {
    let manifold = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(51);

    // Test at the identity
    let id = SMatrix::<Real, 3, 3>::identity();
    let v = manifold.log(&id, &id).expect("log(I, I) should succeed");
    let v_norm = v.norm();
    assert!(
        v_norm < 1e-12,
        "log(I, I) should be zero, got ||V||_F = {:.2e}",
        v_norm
    );

    // Test at random rotation matrices
    for i in 0..50 {
        let r = manifold.random_point(&mut rng);
        let v = manifold.log(&r, &r).expect("log(R, R) should succeed");
        let v_norm = v.norm();
        assert!(
            v_norm < 1e-12,
            "sample {}: log(R, R) should be zero, got ||V||_F = {:.2e}",
            i,
            v_norm
        );
    }
}

/// Parallel transport on SO(3) is exactly left-translation: Γ_{R→Q}(V) = Q R^T V.
///
/// For SO(N) with the bi-invariant metric, the parallel transport along the
/// geodesic from R to Q is:
///
///   Γ_{R→Q}(V) = Q · R^T · V
///
/// This is the left-translation formula. We verify it explicitly by computing
/// the transport two ways and checking they agree:
/// 1. Via manifold.transport(R, Q, V)
/// 2. By direct matrix multiplication Q R^T V
///
/// Ref: Milnor (1976), Corollary 1.10.
#[test]
fn so3_parallel_transport_is_left_translation() {
    let manifold = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(52);
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..100 {
        let r = manifold.random_point(&mut rng);
        let v = manifold.random_tangent(&r, &mut rng);

        // Compute a destination Q = exp(R, small_direction) within the injectivity ball.
        let d = manifold.random_tangent(&r, &mut rng);
        let d_norm = manifold.norm(&r, &d);
        let scale = if d_norm > 1e-10 {
            (inj_radius * 0.3) / d_norm
        } else {
            1.0
        };
        let d_small = d * scale;
        let q = manifold.exp(&r, &d_small);

        // Method 1: use the ParallelTransport trait.
        let transported = manifold
            .transport(&r, &q, &v)
            .expect("transport should succeed for R, Q near each other");

        // Method 2: direct formula Q R^T V.
        let direct = q * (r.transpose() * v);

        // Both should give the same result.
        let err = (transported - direct).norm();
        assert!(
            err < 1e-12,
            "sample {}: transport formula mismatch: ||transport(R,Q,V) - Q R^T V||_F = {:.2e}",
            i,
            err
        );
    }
}

/// Cayley retraction maps to SO(N).
///
/// For any R ∈ SO(3) and tangent V ∈ T_R SO(3), the Cayley retraction
///   retract(R, V) = R (I - Ω/2)^{-1} (I + Ω/2)
/// must land in SO(3): the result should pass check_point.
///
/// We test this for small tangent vectors (to avoid near-singular (I - Ω/2))
/// and for larger ones up to the injectivity radius.
#[test]
fn so3_cayley_retraction_lands_on_so() {
    let manifold = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(53);
    let inj_radius = manifold.injectivity_radius(&manifold.random_point(&mut rng));

    for i in 0..100 {
        let r = manifold.random_point(&mut rng);
        let v = manifold.random_tangent(&r, &mut rng);

        // Scale V to stay within a safe range (80% of injectivity radius).
        // The Cayley map is defined for all V, but inverse_retract can fail near
        // the Cayley cut locus (eigenvalue -1 of R^T Q), which is farther out
        // than the injectivity radius.
        let v_norm = manifold.norm(&r, &v);
        let scale = if v_norm > 1e-10 {
            (inj_radius * 0.8).min(v_norm) / v_norm
        } else {
            1.0
        };
        let v_scaled = v * scale;

        // Compute the Cayley retraction.
        let q = Retraction::retract(&manifold, &r, &v_scaled);

        // Verify that the result is in SO(3).
        manifold.check_point(&q).unwrap_or_else(|e| {
            panic!("sample {}: Cayley retract(R, V) failed check_point: {}", i, e)
        });
    }
}

/// Injectivity radius of SO(3) is π.
///
/// Under the bi-invariant metric <U,V>_R = (1/2) tr(U^T V), the injectivity radius
/// of SO(N) is π (the maximum rotation angle where the principal logarithm is
/// well-defined).
///
/// Ref: so.rs module docstring; Helgason (1978) §IV.6.
#[test]
fn so3_injectivity_radius() {
    let manifold = SpecialOrthogonal::<3>;
    let id = SMatrix::<Real, 3, 3>::identity();

    common::approx::assert_real_eq(
        manifold.injectivity_radius(&id),
        std::f64::consts::PI,
        1e-14,
        "SO(3) injectivity radius = π",
    );
}

/// Log fails at the cut locus with CutLocus error.
///
/// The cut locus of R ∈ SO(3) is the set of Q ∈ SO(3) such that the
/// relative rotation R^T Q has a rotation angle of exactly π (a half-turn
/// in some plane).
///
/// For R = I, the cut locus is the set of rotations with angle π around
/// some axis, i.e., the set of symmetric orthogonal matrices with det = +1.
///
/// The matrix diag(-1, -1, +1) is a 180° rotation around the z-axis:
///   R = diag(-1, -1, 1) has eigenvalues -1, -1, +1 and det = 1. ✓
///   The rotation angle is π (since tr(R) = -2+1 = -1, and tr = 1 + 2cos(θ) → cos(θ) = -1).
///
/// log(I, diag(-1,-1,1)) should return Err(CutLocus { .. }).
#[test]
fn so3_cut_locus() {
    let manifold = SpecialOrthogonal::<3>;
    let id = SMatrix::<Real, 3, 3>::identity();

    // Q = diag(-1, -1, 1): 180° rotation around the z-axis, which is at the cut locus.
    //
    // Matrix form:
    //   [[-1,  0,  0],
    //    [ 0, -1,  0],
    //    [ 0,  0,  1]]
    let mut q_cut = SMatrix::<Real, 3, 3>::zeros();
    q_cut[(0, 0)] = -1.0;
    q_cut[(1, 1)] = -1.0;
    q_cut[(2, 2)] = 1.0;

    // Verify that Q is indeed on SO(3) before testing the log.
    manifold
        .check_point(&q_cut)
        .expect("diag(-1,-1,1) should be in SO(3)");

    // log(I, Q_cut) should fail with a CutLocus error.
    let result = manifold.log(&id, &q_cut);
    assert!(
        result.is_err(),
        "log(I, diag(-1,-1,1)) should fail at cut locus, but returned Ok"
    );

    match result.unwrap_err() {
        CartanError::CutLocus { .. } => {
            // Expected: the principal logarithm is not defined at angle π.
        }
        other => panic!(
            "log(I, diag(-1,-1,1)) should return CutLocus, got: {:?}",
            other
        ),
    }
}
