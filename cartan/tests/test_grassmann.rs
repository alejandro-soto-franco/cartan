// ~/cartan/cartan/tests/test_grassmann.rs

//! Integration tests for the Grassmann<N, K> manifold.
//!
//! Gr(N, K) is the manifold of K-dimensional subspaces of R^N, represented
//! as N×K orthonormal frames (Q^T Q = I_K). Points are equivalent iff their
//! column spans coincide.
//!
//! Key properties verified:
//! - Points have orthonormal columns: Q^T Q = I_K.
//! - Tangent vectors are Q-horizontal: Q^T V = 0.
//! - Exp/Log roundtrip (within injectivity radius π/2).
//! - Parallel transport is exact (inner product preserving).
//! - Sectional curvature lies in [0, 2].
//! - Curvature skew-symmetry and first Bianchi identity.
//! - Geodesic interpolation boundary conditions.
//! - Dim = K(N-K).

use cartan_core::{Curvature, GeodesicInterpolation, Manifold, ParallelTransport, Real, Retraction};
use cartan_manifolds::Grassmann;
use nalgebra::SMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn rng() -> StdRng {
    StdRng::seed_from_u64(0xDEAD_BEEF)
}

/// Assert Frobenius distance.
fn assert_frob<const R: usize, const C: usize>(
    a: &SMatrix<Real, R, C>,
    b: &SMatrix<Real, R, C>,
    tol: Real,
    ctx: &str,
) {
    let err = (a - b).norm();
    assert!(err < tol, "{ctx}: ||A-B||_F = {err:.2e} >= tol {tol:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Gr(5, 2): primary test target
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gr52_dim() {
    assert_eq!(Grassmann::<5, 2>.dim(), 6); // K(N-K) = 2*3
}

#[test]
fn gr52_random_points_are_orthonormal() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    for i in 0..200 {
        let q = m.random_point(&mut rng);
        let gram = q.transpose() * q;
        let err = (gram - SMatrix::<Real, 2, 2>::identity()).norm();
        assert!(err < 1e-12, "sample {i}: Q^T Q != I_2: err = {err:.2e}");
    }
}

#[test]
fn gr52_random_tangents_are_horizontal() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    for i in 0..200 {
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        // Horizontality: Q^T V = 0 (2×2 zero matrix).
        let err = (q.transpose() * v).norm();
        assert!(err < 1e-12, "sample {i}: Q^T V != 0: err = {err:.2e}");
    }
}

#[test]
fn gr52_exp_log_roundtrip() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    let inj = std::f64::consts::FRAC_PI_2;

    for i in 0..100 {
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_norm = m.norm(&q, &v);
        let v_small = if v_norm > 1e-10 { v * (inj * 0.4 / v_norm) } else { v };

        let q2 = m.exp(&q, &v_small);
        let v_rec = m.log(&q, &q2).expect("Log should succeed within injectivity ball");
        assert_frob(&v_rec, &v_small, 1e-10, &format!("sample {i}: log(exp) roundtrip"));
    }
}

#[test]
fn gr52_log_exp_roundtrip() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();

    for i in 0..100 {
        let p = m.random_point(&mut rng);
        let d = m.random_tangent(&p, &mut rng);
        let d_small = d * (0.4 * std::f64::consts::FRAC_PI_2 / m.norm(&p, &d).max(1e-10));
        let q = m.exp(&p, &d_small);
        let q_rec = m.exp(&p, &m.log(&p, &q).unwrap());
        // Subspace equality: Q_rec and Q should span the same subspace.
        // ||P_{Q_rec} - P_Q||_F = 0, i.e. Q_rec Q_rec^T = Q Q^T.
        let proj_rec = &q_rec * q_rec.transpose();
        let proj_q = &q * q.transpose();
        assert_frob(&proj_rec, &proj_q, 1e-10, &format!("sample {i}: exp(log) roundtrip (subspace)"));
    }
}

#[test]
fn gr52_exp_lands_on_manifold() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    for i in 0..200 {
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_small = v * 0.3;
        let q2 = m.exp(&q, &v_small);
        m.check_point(&q2).unwrap_or_else(|e| panic!("sample {i}: exp result not on Gr(5,2): {e}"));
    }
}

#[test]
fn gr52_dist_symmetry() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    for i in 0..100 {
        let p = m.random_point(&mut rng);
        let q = m.random_point(&mut rng);
        if let (Ok(dpq), Ok(dqp)) = (m.dist(&p, &q), m.dist(&q, &p)) {
            assert!(
                (dpq - dqp).abs() < 1e-12,
                "sample {i}: dist asymmetry: d(p,q)={dpq:.6e}, d(q,p)={dqp:.6e}"
            );
        }
    }
}

#[test]
fn gr52_parallel_transport_preserves_inner() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    let inj = std::f64::consts::FRAC_PI_2;

    for i in 0..100 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let w = m.random_tangent(&p, &mut rng);
        let d = m.random_tangent(&p, &mut rng);
        let d_small = d * (inj * 0.3 / m.norm(&p, &d).max(1e-10));
        let q = m.exp(&p, &d_small);

        if let (Ok(u_t), Ok(w_t)) = (m.transport(&p, &q, &u), m.transport(&p, &q, &w)) {
            let before = m.inner(&p, &u, &w);
            let after = m.inner(&q, &u_t, &w_t);
            assert!(
                (before - after).abs() < 1e-10,
                "sample {i}: PT inner product: before={before:.8e}, after={after:.8e}"
            );
        }
    }
}

#[test]
fn gr52_curvature_skew_symmetry_and_bianchi() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();

    for i in 0..100 {
        let p = m.random_point(&mut rng);
        let u = m.random_tangent(&p, &mut rng);
        let v = m.random_tangent(&p, &mut rng);
        let w = m.random_tangent(&p, &mut rng);

        // Skew-symmetry: R(u,v)w + R(v,u)w = 0.
        let r_uvw = m.riemann_curvature(&p, &u, &v, &w);
        let r_vuw = m.riemann_curvature(&p, &v, &u, &w);
        assert!(
            (r_uvw.clone() + r_vuw).norm() < 1e-12,
            "sample {i}: skew-symmetry violated"
        );

        // First Bianchi: R(u,v)w + R(v,w)u + R(w,u)v = 0.
        let r_vwu = m.riemann_curvature(&p, &v, &w, &u);
        let r_wuv = m.riemann_curvature(&p, &w, &u, &v);
        let bianchi = r_uvw + r_vwu + r_wuv;
        assert!(
            bianchi.norm() < 1e-12,
            "sample {i}: first Bianchi violated: ||sum|| = {:.2e}",
            bianchi.norm()
        );
    }
}

#[test]
fn gr52_sectional_curvature_in_range() {
    // Gr(N,K) has sectional curvature in [0, 2].
    let m = Grassmann::<5, 2>;
    let mut rng = rng();

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
            assert!(k >= -1e-10, "sectional curvature below 0: K = {k:.6e}");
            assert!(k <= 2.0 + 1e-10, "sectional curvature above 2: K = {k:.6e}");
        }
    }
}

#[test]
fn gr52_scalar_curvature_exact() {
    // Gr(N,K) scalar curvature = K(N-K)(N-2)/4.
    // Gr(5,2): 2*3*(5-2)/4 = 6*3/4 = 4.5.
    let m = Grassmann::<5, 2>;
    let p = m.random_point(&mut rng());
    let s = m.scalar_curvature(&p);
    let expected = 4.5;
    assert!(
        (s - expected).abs() < 1e-10,
        "Gr(5,2) scalar curvature = {s:.8e}, expected {expected}"
    );
}

#[test]
fn gr52_geodesic_boundary_conditions() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();
    let inj = std::f64::consts::FRAC_PI_2;

    for i in 0..50 {
        let p = m.random_point(&mut rng);
        let d = m.random_tangent(&p, &mut rng);
        let d_small = d * (inj * 0.35 / m.norm(&p, &d).max(1e-10));
        let q = m.exp(&p, &d_small);

        let g0 = m.geodesic(&p, &q, 0.0).unwrap();
        let g1 = m.geodesic(&p, &q, 1.0).unwrap();

        // Subspace equality: compare projection matrices.
        let proj_g0_p = &g0 * g0.transpose() - &p * p.transpose();
        let proj_g1_q = &g1 * g1.transpose() - &q * q.transpose();
        assert!(
            proj_g0_p.norm() < 1e-10,
            "sample {i}: geodesic(0) not equal to p as subspace: err = {:.2e}",
            proj_g0_p.norm()
        );
        assert!(
            proj_g1_q.norm() < 1e-10,
            "sample {i}: geodesic(1) not equal to q as subspace: err = {:.2e}",
            proj_g1_q.norm()
        );
    }
}

#[test]
fn gr52_retraction() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();

    for i in 0..100 {
        let p = m.random_point(&mut rng);
        let v = m.random_tangent(&p, &mut rng);
        let v_small = v * 0.3;

        // retract(p, v) must land on Gr(5,2).
        let q = Retraction::retract(&m, &p, &v_small);
        m.check_point(&q).unwrap_or_else(|e| panic!("sample {i}: retract not on Gr(5,2): {e}"));

        // retract(p, 0) = p (as a subspace).
        let r0 = Retraction::retract(&m, &p, &m.zero_tangent(&p));
        let proj_diff = (r0 * r0.transpose() - p * p.transpose()).norm();
        assert!(
            proj_diff < 1e-12,
            "sample {i}: retract(p,0) subspace error = {proj_diff:.2e}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gr(4, 1) = RP^3 (real projective space): additional coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gr41_dim() {
    assert_eq!(Grassmann::<4, 1>.dim(), 3); // K(N-K) = 1*3
}

#[test]
fn gr41_scalar_curvature() {
    // Gr(4,1) = RP^3: scalar curvature = K(N-K)(N-2)/4 = 1*3*2/4 = 1.5.
    let m = Grassmann::<4, 1>;
    let p = m.random_point(&mut rng());
    let s = m.scalar_curvature(&p);
    assert!((s - 1.5).abs() < 1e-10, "Gr(4,1) scalar curvature = {s:.8e}, expected 1.5");
}

#[test]
fn gr41_base_identities() {
    let m = Grassmann::<4, 1>;
    let mut rng = rng();
    let inj = std::f64::consts::FRAC_PI_2;

    // Manual base test for the rectangular matrix manifold.
    for i in 0..100 {
        let q = m.random_point(&mut rng);
        m.check_point(&q).unwrap_or_else(|e| panic!("sample {i}: random_point invalid: {e}"));

        let v = m.random_tangent(&q, &mut rng);
        m.check_tangent(&q, &v).unwrap_or_else(|e| panic!("sample {i}: random_tangent invalid: {e}"));

        let v_small = v * (inj * 0.4 / m.norm(&q, &v).max(1e-10));
        let q2 = m.exp(&q, &v_small);
        m.check_point(&q2).unwrap_or_else(|e| panic!("sample {i}: exp invalid: {e}"));

        // Inner product symmetry.
        let w = m.random_tangent(&q, &mut rng);
        let vw = m.inner(&q, &v, &w);
        let wv = m.inner(&q, &w, &v);
        assert!((vw - wv).abs() < 1e-14, "sample {i}: inner product not symmetric");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gr(6, 3): square case (K = N/2)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn gr63_dim() {
    assert_eq!(Grassmann::<6, 3>.dim(), 9); // K(N-K) = 3*3
}

#[test]
fn gr63_scalar_curvature() {
    // Gr(6,3): K(N-K)(N-2)/4 = 3*3*4/4 = 9.0.
    let m = Grassmann::<6, 3>;
    let p = m.random_point(&mut rng());
    let s = m.scalar_curvature(&p);
    assert!((s - 9.0).abs() < 1e-10, "Gr(6,3) scalar curvature = {s:.8e}, expected 9.0");
}

#[test]
fn gr63_exp_log_roundtrip() {
    let m = Grassmann::<6, 3>;
    let mut rng = rng();
    let inj = std::f64::consts::FRAC_PI_2;

    for i in 0..50 {
        let q = m.random_point(&mut rng);
        let v = m.random_tangent(&q, &mut rng);
        let v_small = v * (inj * 0.25 / m.norm(&q, &v).max(1e-10));

        let q2 = m.exp(&q, &v_small);
        let v_rec = m.log(&q, &q2).expect("Log failed within injectivity ball");
        assert!(
            (v_rec - v_small).norm() < 1e-8,
            "sample {i}: Gr(6,3) exp-log roundtrip err = {:.2e}",
            (v_rec - v_small).norm()
        );
    }
}
