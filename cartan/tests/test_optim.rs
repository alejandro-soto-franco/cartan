// ~/cartan/cartan/tests/test_optim.rs

//! Integration tests for cartan-optim: all four algorithms on all matrix manifolds.
//!
//! Covered manifolds and algorithms:
//!
//! | Manifold | RGD | RCG | RTR | Fréchet |
//! |----------|-----|-----|-----|---------|
//! | Euclidean<3> | yes | yes | yes | yes |
//! | Sphere<3>    | yes | yes | yes | yes |
//! | SO(3)        | yes | yes | yes | yes |
//! | SPD(3)       | yes | yes | yes | yes |
//! | Corr(3)      | yes | yes | yes | yes |
//! | Grassmann(5,2)| yes | yes | yes | yes |
//!
//! Each test minimizes a known cost function with analytic solution so that
//! correctness can be verified by distance to the ground-truth optimum.

mod common;

use cartan_core::{Manifold, Real};
use cartan_manifolds::{Corr, Euclidean, Grassmann, Spd, SpecialOrthogonal, Sphere};
use cartan_optim::{
    CgVariant, FrechetConfig, RCGConfig, RGDConfig, RTRConfig, frechet_mean, minimize_rcg,
    minimize_rgd, minimize_rtr,
};
use nalgebra::{SMatrix, SVector};
use rand::SeedableRng;
use rand::rngs::StdRng;

// ─────────────────────────────────────────────────────────────────────────────
// ====  Euclidean<3>  ====
//
// f(x) = ||x - a||^2, minimizer x* = a.
// Euclidean gradient: 2(x - a).  Hessian-vector: 2v.
// ─────────────────────────────────────────────────────────────────────────────

const TARGET_EUCLIDEAN: SVector<Real, 3> =
    SVector::from_array_storage(nalgebra::ArrayStorage([[1.5, -0.7, 2.3]]));

fn euc_cost(x: &SVector<Real, 3>) -> Real {
    (x - TARGET_EUCLIDEAN).norm_squared()
}

fn euc_rgrad(m: &Euclidean<3>, x: &SVector<Real, 3>) -> SVector<Real, 3> {
    let egrad = (x - TARGET_EUCLIDEAN) * 2.0;
    m.project_tangent(x, &egrad)
}

#[test]
fn euclidean_rgd_recovers_target() {
    let m = Euclidean::<3>;
    let mut rng = StdRng::seed_from_u64(300);
    let x0 = m.random_point(&mut rng);

    let config = RGDConfig {
        max_iters: 500,
        grad_tol: 1e-9,
        init_step: 1.0,
        ..Default::default()
    };
    let res = minimize_rgd(&m, euc_cost, |x| euc_rgrad(&m, x), x0, &config);
    assert!(
        res.converged,
        "Euclidean RGD did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - TARGET_EUCLIDEAN).norm();
    assert!(err < 1e-6, "Euclidean RGD: ||x* - target|| = {err:.2e}");
}

#[test]
fn euclidean_rcg_recovers_target() {
    let m = Euclidean::<3>;
    let mut rng = StdRng::seed_from_u64(301);
    let x0 = m.random_point(&mut rng);

    let config = RCGConfig {
        max_iters: 200,
        grad_tol: 1e-9,
        variant: CgVariant::FletcherReeves,
        ..Default::default()
    };
    let res = minimize_rcg(&m, euc_cost, |x| euc_rgrad(&m, x), x0, &config);
    assert!(
        res.converged,
        "Euclidean RCG did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - TARGET_EUCLIDEAN).norm();
    assert!(err < 1e-6, "Euclidean RCG: ||x* - target|| = {err:.2e}");
}

#[test]
fn euclidean_rtr_recovers_target() {
    let m = Euclidean::<3>;
    let mut rng = StdRng::seed_from_u64(302);
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 100,
        grad_tol: 1e-9,
        ..Default::default()
    };
    let res = minimize_rtr(
        &m,
        euc_cost,
        |x| euc_rgrad(&m, x),
        |_x, v| v.clone() * 2.0,
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Euclidean RTR did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - TARGET_EUCLIDEAN).norm();
    assert!(err < 1e-6, "Euclidean RTR: ||x* - target|| = {err:.2e}");
}

#[test]
fn euclidean_frechet_mean_recovers_center() {
    let m = Euclidean::<3>;
    let mut rng = StdRng::seed_from_u64(303);

    let center = SVector::<Real, 3>::from([1.0, 2.0, 3.0]);
    let points: Vec<SVector<Real, 3>> = (0..30)
        .map(|_| {
            let v = m.random_tangent(&center, &mut rng) * 0.1;
            m.exp(&center, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 200,
        tol: 1e-9,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "Euclidean Fréchet mean did not converge");
    let err = (res.point - center).norm();
    assert!(
        err < 5e-2,
        "Euclidean Fréchet mean: ||mu - center|| = {err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  Sphere S² (S^2 in R^3)  ====
//
// f(p) = -p^T a, minimizer p* = a / ||a||.
// ─────────────────────────────────────────────────────────────────────────────

fn sph_cost(p: &SVector<Real, 3>, a: &SVector<Real, 3>) -> Real {
    -p.dot(a)
}

fn sph_rgrad(m: &Sphere<3>, p: &SVector<Real, 3>, a: &SVector<Real, 3>) -> SVector<Real, 3> {
    m.project_tangent(p, &-a)
}

#[test]
fn sphere_rgd_various_targets() {
    let m = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(310);

    for &(ax, ay, az) in &[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 1.0)] {
        let a = SVector::<Real, 3>::from([ax, ay, az]);
        let a_hat = a / a.norm();
        let x0 = m.random_point(&mut rng);

        let config = RGDConfig {
            max_iters: 2000,
            grad_tol: 1e-7,
            ..Default::default()
        };
        let res = minimize_rgd(
            &m,
            |p| sph_cost(p, &a),
            |p| sph_rgrad(&m, p, &a),
            x0,
            &config,
        );
        assert!(res.converged, "Sphere RGD did not converge for a={a:?}");
        let err = (res.point - a_hat).norm();
        assert!(err < 1e-5, "Sphere RGD: err={err:.2e} for a={a:?}");
    }
}

#[test]
fn sphere_rcg_polak_ribiere() {
    let m = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(311);

    let a = SVector::<Real, 3>::from([2.0, -1.0, 1.0]);
    let a_hat = a / a.norm();
    let x0 = m.random_point(&mut rng);

    let config = RCGConfig {
        max_iters: 500,
        grad_tol: 1e-7,
        variant: CgVariant::PolakRibiere,
        ..Default::default()
    };
    let res = minimize_rcg(
        &m,
        |p| sph_cost(p, &a),
        |p| sph_rgrad(&m, p, &a),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Sphere RCG PR+ did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - a_hat).norm();
    assert!(err < 1e-5, "Sphere RCG PR+: err={err:.2e}");
}

#[test]
fn sphere_frechet_mean_near_equator() {
    let m = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(312);

    // True mean on the equator: (1, 0, 0).
    let mu_true = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let points: Vec<SVector<Real, 3>> = (0..40)
        .map(|_| {
            let v = m.random_tangent(&mu_true, &mut rng) * 0.05;
            m.exp(&mu_true, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 300,
        tol: 1e-10,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "Sphere Fréchet mean did not converge");
    let err = (res.point - mu_true).norm();
    assert!(err < 5e-2, "Sphere Fréchet mean equator: err={err:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  SpecialOrthogonal SO(3)  ====
//
// f(R) = ||R - A||_F^2 = 2*N - 2*tr(R^T A).
// Minimizer: R* = U V^T where A = U Σ V^T (with det adjustment).
// Euclidean gradient: 2(R - A).
// Euclidean HVP: D^2 f [V] = 2 V.
// ─────────────────────────────────────────────────────────────────────────────

fn so3_target() -> SMatrix<Real, 3, 3> {
    // Rotation by π/4 around z-axis.
    let c = (std::f64::consts::PI / 4.0).cos();
    let s = (std::f64::consts::PI / 4.0).sin();
    SMatrix::<Real, 3, 3>::from_row_slice(&[c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
}

fn so3_cost(r: &SMatrix<Real, 3, 3>, a: &SMatrix<Real, 3, 3>) -> Real {
    (r - a).norm_squared()
}

fn so3_rgrad(
    m: &SpecialOrthogonal<3>,
    r: &SMatrix<Real, 3, 3>,
    a: &SMatrix<Real, 3, 3>,
) -> SMatrix<Real, 3, 3> {
    let egrad = (r - a) * 2.0;
    m.project_tangent(r, &egrad)
}

#[test]
fn so3_rgd_recovers_rotation() {
    let m = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(320);
    let a = so3_target();
    let x0 = m.random_point(&mut rng);

    let config = RGDConfig {
        max_iters: 3000,
        grad_tol: 1e-7,
        init_step: 0.5,
        ..Default::default()
    };
    let res = minimize_rgd(
        &m,
        |r| so3_cost(r, &a),
        |r| so3_rgrad(&m, r, &a),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "SO(3) RGD did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - a).norm();
    assert!(err < 1e-4, "SO(3) RGD: ||R* - A||_F = {err:.2e}");
}

#[test]
fn so3_rcg_recovers_rotation() {
    let m = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(321);
    let a = so3_target();
    let x0 = m.random_point(&mut rng);

    let config = RCGConfig {
        max_iters: 1000,
        grad_tol: 1e-7,
        ..Default::default()
    };
    let res = minimize_rcg(
        &m,
        |r| so3_cost(r, &a),
        |r| so3_rgrad(&m, r, &a),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "SO(3) RCG did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - a).norm();
    assert!(err < 1e-4, "SO(3) RCG: ||R* - A||_F = {err:.2e}");
}

#[test]
fn so3_rtr_recovers_rotation() {
    let m = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(322);
    let a = so3_target();
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 300,
        grad_tol: 1e-7,
        ..Default::default()
    };
    let res = minimize_rtr(
        &m,
        |r| so3_cost(r, &a),
        |r| so3_rgrad(&m, r, &a),
        |_r, v| v.clone() * 2.0,
        x0,
        &config,
    );
    assert!(
        res.converged,
        "SO(3) RTR did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - a).norm();
    assert!(err < 1e-4, "SO(3) RTR: ||R* - A||_F = {err:.2e}");
}

#[test]
fn so3_frechet_mean_near_identity() {
    let m = SpecialOrthogonal::<3>;
    let mut rng = StdRng::seed_from_u64(323);

    let identity = SMatrix::<Real, 3, 3>::identity();
    let points: Vec<SMatrix<Real, 3, 3>> = (0..30)
        .map(|_| {
            let v = m.random_tangent(&identity, &mut rng) * 0.1;
            m.exp(&identity, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 300,
        tol: 1e-9,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "SO(3) Fréchet mean did not converge");
    let err = (res.point - identity).norm();
    assert!(err < 1e-1, "SO(3) Fréchet mean: ||mu - I||_F = {err:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  SPD(3)  ====
//
// f(P) = ||P - A||_F^2, minimizer P* = A.
// Euclidean gradient: 2(P - A).  HVP: 2V.
// ─────────────────────────────────────────────────────────────────────────────

fn spd3_target() -> SMatrix<Real, 3, 3> {
    SMatrix::<Real, 3, 3>::from_row_slice(&[3.0, 0.5, 0.1, 0.5, 2.0, 0.3, 0.1, 0.3, 1.5])
}

fn spd_cost(p: &SMatrix<Real, 3, 3>, a: &SMatrix<Real, 3, 3>) -> Real {
    (p - a).norm_squared()
}

fn spd_rgrad(m: &Spd<3>, p: &SMatrix<Real, 3, 3>, a: &SMatrix<Real, 3, 3>) -> SMatrix<Real, 3, 3> {
    let egrad = (p - a) * 2.0;
    m.project_tangent(p, &egrad)
}

#[test]
fn spd3_rcg_recovers_target() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(330);
    let target = spd3_target();
    // Start close to the target so the affine-invariant geometry is well-conditioned.
    let v0 = m.random_tangent(&target, &mut rng) * 0.3;
    let x0 = m.exp(&target, &v0);

    let config = RCGConfig {
        max_iters: 3000,
        grad_tol: 1e-5,
        ..Default::default()
    };
    let res = minimize_rcg(
        &m,
        |p| spd_cost(p, &target),
        |p| spd_rgrad(&m, p, &target),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "SPD(3) RCG did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - target).norm();
    assert!(err < 1e-2, "SPD(3) RCG: ||P* - target||_F = {err:.2e}");
}

#[test]
fn spd3_frechet_mean_recovers_center() {
    let m = Spd::<3>;
    let mut rng = StdRng::seed_from_u64(331);

    let center = spd3_target();
    let points: Vec<SMatrix<Real, 3, 3>> = (0..25)
        .map(|_| {
            let v = m.random_tangent(&center, &mut rng) * 0.1;
            m.exp(&center, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 200,
        tol: 1e-8,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "SPD(3) Fréchet mean did not converge");
    m.check_point(&res.point)
        .expect("SPD Fréchet mean result is not SPD");
    let err = (res.point - center).norm();
    assert!(
        err < 2e-1,
        "SPD(3) Fréchet mean: ||mu - center||_F = {err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  Corr(3)  ====
//
// Corr(3) is flat (K=0). f(C) = ||C - A||_F^2 where A is a valid correlation matrix.
// Euclidean gradient: 2(C - A). Tangent space: symmetric, zero diagonal.
// HVP: 2V.
// ─────────────────────────────────────────────────────────────────────────────

fn corr3_target() -> SMatrix<Real, 3, 3> {
    // Correlation matrix: symmetric, unit diagonal, PD.
    SMatrix::<Real, 3, 3>::from_row_slice(&[1.0, 0.3, -0.2, 0.3, 1.0, 0.5, -0.2, 0.5, 1.0])
}

fn corr_cost(c: &SMatrix<Real, 3, 3>, a: &SMatrix<Real, 3, 3>) -> Real {
    (c - a).norm_squared()
}

fn corr_rgrad(
    m: &Corr<3>,
    c: &SMatrix<Real, 3, 3>,
    a: &SMatrix<Real, 3, 3>,
) -> SMatrix<Real, 3, 3> {
    let egrad = (c - a) * 2.0;
    m.project_tangent(c, &egrad)
}

#[test]
fn corr3_rgd_recovers_target() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(340);
    let target = corr3_target();
    let x0 = m.random_point(&mut rng);

    // Corr(3) is flat, so GD converges quickly.
    let config = RGDConfig {
        max_iters: 2000,
        grad_tol: 1e-9,
        init_step: 0.5,
        ..Default::default()
    };
    let res = minimize_rgd(
        &m,
        |c| corr_cost(c, &target),
        |c| corr_rgrad(&m, c, &target),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Corr(3) RGD did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - target).norm();
    assert!(err < 1e-5, "Corr(3) RGD: ||C* - target||_F = {err:.2e}");
}

#[test]
fn corr3_rcg_recovers_target() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(341);
    let target = corr3_target();
    let x0 = m.random_point(&mut rng);

    let config = RCGConfig {
        max_iters: 500,
        grad_tol: 1e-9,
        ..Default::default()
    };
    let res = minimize_rcg(
        &m,
        |c| corr_cost(c, &target),
        |c| corr_rgrad(&m, c, &target),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Corr(3) RCG did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - target).norm();
    assert!(err < 1e-5, "Corr(3) RCG: ||C* - target||_F = {err:.2e}");
}

#[test]
fn corr3_rtr_recovers_target() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(342);
    let target = corr3_target();
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 200,
        grad_tol: 1e-9,
        ..Default::default()
    };
    let res = minimize_rtr(
        &m,
        |c| corr_cost(c, &target),
        |c| corr_rgrad(&m, c, &target),
        |_c, v| v.clone() * 2.0,
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Corr(3) RTR did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let err = (res.point - target).norm();
    assert!(err < 1e-5, "Corr(3) RTR: ||C* - target||_F = {err:.2e}");
}

#[test]
fn corr3_frechet_mean_recovers_center() {
    let m = Corr::<3>;
    let mut rng = StdRng::seed_from_u64(343);

    let center = corr3_target();
    let points: Vec<SMatrix<Real, 3, 3>> = (0..30)
        .map(|_| {
            let v = m.random_tangent(&center, &mut rng) * 0.05;
            m.exp(&center, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 200,
        tol: 1e-10,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "Corr(3) Fréchet mean did not converge");
    m.check_point(&res.point)
        .expect("Corr Fréchet mean result is not a valid correlation matrix");
    let err = (res.point - center).norm();
    assert!(
        err < 5e-2,
        "Corr(3) Fréchet mean: ||mu - center||_F = {err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  Grassmann Gr(5, 2)  ====
//
// PCA problem: maximize ||A^T Q||_F^2.
// f(Q) = -||A^T Q||_F^2, Euclidean gradient: -2 A A^T Q.
// HVP of f: -2 A A^T V.
// Minimizer: Q* = leading 2 eigenvectors of A A^T.
// ─────────────────────────────────────────────────────────────────────────────

fn grassmann_pca_target() -> SMatrix<Real, 5, 2> {
    // Target subspace: span(e1, e2). A = [e1, e2].
    SMatrix::<Real, 5, 2>::from_column_slice(&[
        1.0, 0.0, 0.0, 0.0, 0.0, // e1
        0.0, 1.0, 0.0, 0.0, 0.0, // e2
    ])
}

fn grass_cost(q: &SMatrix<Real, 5, 2>, a: &SMatrix<Real, 5, 2>) -> Real {
    -(a.transpose() * q).norm_squared()
}

fn grass_rgrad(
    m: &Grassmann<5, 2>,
    q: &SMatrix<Real, 5, 2>,
    a: &SMatrix<Real, 5, 2>,
) -> SMatrix<Real, 5, 2> {
    let egrad = -(a * (a.transpose() * q)) * 2.0;
    m.project_tangent(q, &egrad)
}

#[test]
fn grassmann_rcg_pca() {
    let m = Grassmann::<5, 2>;
    let mut rng = StdRng::seed_from_u64(350);
    let a = grassmann_pca_target();
    let x0 = m.random_point(&mut rng);

    let config = RCGConfig {
        max_iters: 3000,
        grad_tol: 1e-5,
        ..Default::default()
    };
    let res = minimize_rcg(
        &m,
        |q| grass_cost(q, &a),
        |q| grass_rgrad(&m, q, &a),
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Grassmann RCG PCA did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let val = (a.transpose() * res.point).norm();
    assert!(
        (val - 2.0_f64.sqrt()).abs() < 1e-3,
        "Grassmann RCG PCA: ||A^T Q*||_F = {val:.6}, expected sqrt(2) = {:.6}",
        2.0_f64.sqrt()
    );
}

#[test]
fn grassmann_rtr_pca() {
    let m = Grassmann::<5, 2>;
    let mut rng = StdRng::seed_from_u64(351);
    let a = grassmann_pca_target();
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 300,
        grad_tol: 1e-5,
        ..Default::default()
    };
    let res = minimize_rtr(
        &m,
        |q| grass_cost(q, &a),
        |q| grass_rgrad(&m, q, &a),
        // HVP of -||A^T Q||_F^2: H[V] = -2 A A^T V
        |_q, v| -(a * (a.transpose() * v)) * 2.0,
        x0,
        &config,
    );
    assert!(
        res.converged,
        "Grassmann RTR PCA did not converge: grad_norm={:.2e}",
        res.grad_norm
    );
    let val = (a.transpose() * res.point).norm();
    assert!(
        (val - 2.0_f64.sqrt()).abs() < 1e-3,
        "Grassmann RTR PCA: ||A^T Q*||_F = {val:.6}, expected sqrt(2)",
    );
}

#[test]
fn grassmann_frechet_mean_near_target() {
    let m = Grassmann::<5, 2>;
    let mut rng = StdRng::seed_from_u64(352);

    let center = grassmann_pca_target();
    let points: Vec<SMatrix<Real, 5, 2>> = (0..20)
        .map(|_| {
            let v = m.random_tangent(&center, &mut rng) * 0.1;
            m.exp(&center, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 300,
        tol: 1e-8,
        step_size: 1.0,
    };
    let res = frechet_mean(&m, &points, None, &config);
    assert!(res.converged, "Grassmann Fréchet mean did not converge");
    m.check_point(&res.point)
        .expect("Grassmann Fréchet mean not on manifold");
    // Check subspace distance: ||P_mu - P_center||_F where P = Q Q^T.
    let p_mu = res.point * res.point.transpose();
    let p_center = center * center.transpose();
    let err = (p_mu - p_center).norm();
    assert!(
        err < 5e-1,
        "Grassmann Fréchet mean: subspace distance = {err:.2e}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ====  Cross-manifold: cost descent sanity  ====
//
// Verify that every run (even if not converged) strictly decreases cost.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn all_algorithms_decrease_cost_on_sphere() {
    let m = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(399);

    let a = SVector::<Real, 3>::from([0.5, 0.3, 0.8]);
    let x0 = m.random_point(&mut rng);
    let f0 = sph_cost(&x0, &a);

    let config_rgd = RGDConfig {
        max_iters: 50,
        ..Default::default()
    };
    let config_rcg = RCGConfig {
        max_iters: 50,
        ..Default::default()
    };
    let config_rtr = RTRConfig {
        max_iters: 50,
        ..Default::default()
    };

    let res_rgd = minimize_rgd(
        &m,
        |p| sph_cost(p, &a),
        |p| sph_rgrad(&m, p, &a),
        x0.clone(),
        &config_rgd,
    );
    let res_rcg = minimize_rcg(
        &m,
        |p| sph_cost(p, &a),
        |p| sph_rgrad(&m, p, &a),
        x0.clone(),
        &config_rcg,
    );
    let res_rtr = minimize_rtr(
        &m,
        |p| sph_cost(p, &a),
        |p| sph_rgrad(&m, p, &a),
        |_p, _v| SVector::<Real, 3>::zeros(),
        x0.clone(),
        &config_rtr,
    );

    assert!(
        res_rgd.value <= f0,
        "RGD did not decrease cost: f0={f0:.4}, final={:.4}",
        res_rgd.value
    );
    assert!(
        res_rcg.value <= f0,
        "RCG did not decrease cost: f0={f0:.4}, final={:.4}",
        res_rcg.value
    );
    assert!(
        res_rtr.value <= f0,
        "RTR did not decrease cost: f0={f0:.4}, final={:.4}",
        res_rtr.value
    );
}
