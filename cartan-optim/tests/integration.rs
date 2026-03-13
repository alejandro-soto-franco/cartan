// cartan-optim/tests/integration.rs
//
// End-to-end tests: minimize known functions on manifolds and check recovery.

use cartan_core::Manifold;
use cartan_manifolds::{Grassmann, Sphere, Spd};
use cartan_optim::{
    frechet_mean, minimize_rcg, minimize_rgd, minimize_rtr,
    FrechetConfig, RCGConfig, RGDConfig, RTRConfig,
};
use nalgebra::{SMatrix, SVector};
use rand::SeedableRng;

type Real = f64;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn rng() -> impl rand::Rng {
    rand::rngs::SmallRng::seed_from_u64(0xCAFE_BABE)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere S²: minimize f(p) = -p^T a  (answer: p* = a/||a||)
// ─────────────────────────────────────────────────────────────────────────────

fn sphere_cost(p: &SVector<Real, 3>, a: &SVector<Real, 3>) -> Real {
    -p.dot(a)
}

fn sphere_rgrad(
    m: &Sphere<3>,
    p: &SVector<Real, 3>,
    a: &SVector<Real, 3>,
) -> SVector<Real, 3> {
    // Euclidean gradient of -p^T a is -a; project onto tangent space.
    let egrad = -a;
    m.project_tangent(p, &egrad)
}

#[test]
fn rgd_sphere_recovers_north_pole() {
    let m = Sphere::<3>;
    let mut rng = rng();

    // Target: unit vector in direction (1, 2, 3).
    let a = SVector::<Real, 3>::from([1.0, 2.0, 3.0]);
    let a_hat = a / a.norm();

    let x0 = m.random_point(&mut rng);
    let config = RGDConfig {
        max_iters: 2000,
        grad_tol: 1e-7,
        ..Default::default()
    };

    let res = minimize_rgd(
        &m,
        |p| sphere_cost(p, &a),
        |p| sphere_rgrad(&m, p, &a),
        x0,
        &config,
    );

    assert!(res.converged, "RGD did not converge: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - a_hat).norm();
    assert!(err < 1e-5, "RGD sphere: ||p* - a_hat|| = {err:.2e}");
}

#[test]
fn rcg_sphere_recovers_north_pole() {
    let m = Sphere::<3>;
    let mut rng = rng();

    let a = SVector::<Real, 3>::from([1.0, 1.0, 1.0]);
    let a_hat = a / a.norm();

    let x0 = m.random_point(&mut rng);
    let config = RCGConfig {
        max_iters: 500,
        grad_tol: 1e-7,
        ..Default::default()
    };

    let res = minimize_rcg(
        &m,
        |p| sphere_cost(p, &a),
        |p| sphere_rgrad(&m, p, &a),
        x0,
        &config,
    );

    assert!(res.converged, "RCG did not converge: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - a_hat).norm();
    assert!(err < 1e-5, "RCG sphere: ||p* - a_hat|| = {err:.2e}");
}

#[test]
fn rtr_sphere_recovers_north_pole() {
    let m = Sphere::<3>;
    let mut rng = rng();

    let a = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);
    let a_hat = a; // already unit

    let x0 = m.random_point(&mut rng);
    let config = RTRConfig {
        max_iters: 200,
        grad_tol: 1e-7,
        ..Default::default()
    };

    let res = minimize_rtr(
        &m,
        |p| sphere_cost(p, &a),
        |p| sphere_rgrad(&m, p, &a),
        // Euclidean HVP: D²(-p^T a)[v] = 0 (linear → zero Hessian)
        |_p, _v| SVector::<Real, 3>::zeros(),
        x0,
        &config,
    );

    assert!(res.converged, "RTR did not converge: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - a_hat).norm();
    assert!(err < 1e-5, "RTR sphere: ||p* - a_hat|| = {err:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// SPD(3): minimize f(P) = ||P - A||_F^2  (answer: P* = A)
//
// Euclidean gradient: 2(P - A)
// Euclidean Hessian-vector product: D²f[V] = 2V
// ─────────────────────────────────────────────────────────────────────────────

fn make_spd3() -> SMatrix<Real, 3, 3> {
    // A simple SPD matrix: I + small perturbation
    let a: SMatrix<Real, 3, 3> = SMatrix::from_row_slice(&[
        2.0, 0.5, 0.1,
        0.5, 3.0, 0.2,
        0.1, 0.2, 1.5,
    ]);
    a
}

#[test]
fn rgd_spd_recovers_target() {
    let m = Spd::<3>;
    let mut rng = rng();

    let target = make_spd3();
    let x0 = m.random_point(&mut rng);

    let config = RGDConfig {
        max_iters: 3000,
        grad_tol: 1e-5,
        init_step: 0.1,
        ..Default::default()
    };

    let res = minimize_rgd(
        &m,
        |p| (p - target).norm().powi(2),
        |p| {
            let egrad = (p - target) * 2.0;
            m.project_tangent(p, &egrad)
        },
        x0,
        &config,
    );

    assert!(res.converged, "RGD SPD did not converge: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - target).norm();
    assert!(err < 1e-3, "RGD SPD: ||P* - target||_F = {err:.2e}");
}

#[test]
fn rtr_spd_recovers_target() {
    let m = Spd::<3>;
    let mut rng = rng();

    let target = make_spd3();
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 200,
        grad_tol: 1e-6,
        delta_init: 0.5,
        ..Default::default()
    };

    let res = minimize_rtr(
        &m,
        |p| (p - target).norm().powi(2),
        |p| {
            let egrad = (p - target) * 2.0;
            m.project_tangent(p, &egrad)
        },
        // EHessian-vector: D²(||P-A||²)[V] = 2V
        |_p, v| v.clone() * 2.0,
        x0,
        &config,
    );

    assert!(res.converged, "RTR SPD did not converge: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - target).norm();
    assert!(err < 1e-3, "RTR SPD: ||P* - target||_F = {err:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Fréchet mean on S²: cluster of points near a known mean
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn frechet_mean_sphere_recovers_center() {
    let m = Sphere::<3>;
    let mut rng = rng();

    // True mean: north pole.
    let mu_true = SVector::<Real, 3>::from([0.0, 0.0, 1.0]);

    // Generate 50 points with small perturbations so sample mean ≈ true mean.
    let points: Vec<SVector<Real, 3>> = (0..50)
        .map(|_| {
            let v = m.random_tangent(&mu_true, &mut rng) * 0.05;
            m.exp(&mu_true, &v)
        })
        .collect();

    let config = FrechetConfig {
        max_iters: 300,
        tol: 1e-9,
        step_size: 1.0,
    };

    let res = frechet_mean(&m, &points, None, &config);

    // The algorithm should converge to the sample mean.
    assert!(res.converged, "Fréchet mean did not converge: grad_norm={:.2e}", res.grad_norm);
    // With small noise (0.05) and 50 samples, sample mean is within ~1e-2 of true mean.
    let err = (res.point - mu_true).norm();
    assert!(err < 5e-2, "Fréchet mean sphere: ||mu - mu_true|| = {err:.2e}");
}

// ─────────────────────────────────────────────────────────────────────────────
// Grassmann Gr(5,2): minimize f(Q) = -||A^T Q||_F^2 where A is N×K
// (answer: Q* = leading K eigenvectors of A A^T)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rgd_grassmann_principal_subspace() {
    let m = Grassmann::<5, 2>;
    let mut rng = rng();

    // A is 5×2 with columns forming an orthonormal frame.
    // We'll use the first two standard basis vectors.
    let a: SMatrix<Real, 5, 2> = SMatrix::from_column_slice(&[
        1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
    ]);

    // f(Q) = -||A^T Q||_F^2  →  max ||A^T Q||_F^2  →  Q* = span(e1, e2)
    // Euclidean gradient of -||A^T Q||_F^2 = -2 A A^T Q
    let x0 = m.random_point(&mut rng);
    let config = RGDConfig {
        max_iters: 3000,
        grad_tol: 1e-5,
        init_step: 0.5, // smaller step for Grassmann PCA (non-strongly-convex)
        ..Default::default()
    };

    let res = minimize_rgd(
        &m,
        |q| -(a.transpose() * q).norm().powi(2),
        |q| {
            let egrad = -a * (a.transpose() * q) * 2.0;
            m.project_tangent(q, &egrad)
        },
        x0,
        &config,
    );

    assert!(res.converged, "RGD Grassmann did not converge: grad_norm={:.2e}", res.grad_norm);

    // Check that the optimal subspace aligns with span(e1, e2):
    // ||A^T Q*||_F should be ≈ √2 (Q* ≈ A up to rotation)
    let val = (a.transpose() * res.point).norm();
    assert!(
        (val - 2.0_f64.sqrt()).abs() < 1e-4,
        "Grassmann PCA: ||A^T Q*||_F = {val:.6}, expected √2 ≈ {:.6}",
        2.0_f64.sqrt()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// RTR convergence: check model decrease is accurate for quadratic functions
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rtr_converges_quadratic_sphere() {
    let m = Sphere::<3>;
    let mut rng = rng();

    // f(p) = ||p - target||^2 (Euclidean) restricted to sphere.
    // Minimum on sphere is p* = target/||target|| if target ≠ 0.
    let target = SVector::<Real, 3>::from([1.0, 0.0, 0.0]);
    let x0 = m.random_point(&mut rng);

    let config = RTRConfig {
        max_iters: 200,
        grad_tol: 1e-7,
        ..Default::default()
    };

    let res = minimize_rtr(
        &m,
        |p| (p - target).norm_squared(),
        |p| {
            let egrad = (p - target) * 2.0;
            m.project_tangent(p, &egrad)
        },
        |_p, v| v.clone() * 2.0,
        x0,
        &config,
    );

    assert!(res.converged, "RTR quadratic sphere: grad_norm={:.2e}", res.grad_norm);
    let err = (res.point - target).norm();
    assert!(err < 1e-5, "RTR quadratic sphere: ||p* - target|| = {err:.2e}");
}
