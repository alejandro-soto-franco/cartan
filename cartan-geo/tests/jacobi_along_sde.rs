//! End-to-end integration test:
//! `cartan-manifolds` Sphere → `cartan-stochastic` BM → `cartan-geo` Jacobi.
//!
//! Builds a Brownian path on `S^2` via stochastic development, then rides a
//! Jacobi field along it. Expected behaviours:
//!
//! 1. Each sampled Jacobi value `J(t_i)` must be tangent to the sphere at
//!    `path[i]` (i.e., `<path[i], J(t_i)>` = 0 to numerical tolerance).
//! 2. For a degenerate initial condition (`J(0) = 0`, `J'(0) = 0`) the
//!    Jacobi field must stay identically zero along the entire path.
//! 3. For a short-horizon generic initial condition the Jacobi norm must
//!    remain bounded (no blow-up) — the unit sphere has positive curvature
//!    K = 1, which causes focusing rather than divergence.

use cartan_core::{Manifold, Real};
use cartan_geo::integrate_jacobi_along_path;
use cartan_manifolds::Sphere;
use cartan_stochastic::{random_frame_at, stochastic_development};
use nalgebra::SVector;
use rand::SeedableRng;
use rand::rngs::StdRng;

type Vec3 = SVector<Real, 3>;

fn sphere_path(seed: u64, n_steps: usize, dt: Real) -> (Sphere<3>, Vec<Vec3>) {
    let s: Sphere<3> = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(seed);
    let p0 = Vec3::new(0.0, 0.0, 1.0);
    let frame = random_frame_at(&s, &p0, &mut rng).expect("frame");
    let result =
        stochastic_development(&s, &p0, frame, n_steps, dt, &mut rng, 1e-10).expect("dev");
    (s, result.path)
}

#[test]
fn jacobi_stays_tangent_along_sde() {
    let (s, path) = sphere_path(0x5EED, 60, 0.002);
    let p0 = path[0];

    // Initial J at p0: a random tangent vector, mild scale.
    let mut rng = StdRng::seed_from_u64(0xABCDEF);
    let j0 = s.random_tangent(&p0, &mut rng) * 0.1;
    let j0_dot = s.random_tangent(&p0, &mut rng) * 0.05;

    let dt = 0.002;
    let result = integrate_jacobi_along_path(&s, &path, dt, j0, j0_dot).expect("jacobi");

    for (i, (p, j)) in path.iter().zip(result.field.iter()).enumerate() {
        let normal_component = p.dot(j).abs();
        assert!(
            normal_component < 1e-6,
            "step {i}: J not tangent to S^2 (|<p,J>| = {normal_component})"
        );
    }
}

#[test]
fn jacobi_zero_initial_stays_zero() {
    let (s, path) = sphere_path(0xDEAD, 40, 0.003);
    let p0 = path[0];
    let j0 = s.zero_tangent(&p0);
    let j0_dot = s.zero_tangent(&p0);
    let dt = 0.003;
    let result = integrate_jacobi_along_path(&s, &path, dt, j0, j0_dot).expect("jacobi");
    for (i, j) in result.field.iter().enumerate() {
        let n = j.norm();
        assert!(n < 1e-8, "step {i}: expected 0 Jacobi, got norm {n}");
    }
}

#[test]
fn jacobi_norm_bounded_on_short_horizon() {
    // Positive curvature K = 1 on the unit sphere: Jacobi fields along any
    // curve satisfy a damped-oscillator-like equation and remain bounded
    // on short horizons. Specifically, on a geodesic with unit speed, J
    // is dominated by cos(t)·J(0) + sin(t)·J'(0); for an SDE path the
    // same boundedness holds with a slightly larger amplification factor
    // driven by the varying velocity. We check a loose 5× bound — a blow-up
    // would produce 10^5+ growth within 50 RK4 steps on any bug.
    let (s, path) = sphere_path(0xCAFE, 50, 0.002);
    let p0 = path[0];
    let mut rng = StdRng::seed_from_u64(0x1234);
    let j0 = s.random_tangent(&p0, &mut rng) * 0.1;
    let j0_dot = s.random_tangent(&p0, &mut rng) * 0.05;
    let initial_norm = j0.norm() + j0_dot.norm();
    let result = integrate_jacobi_along_path(&s, &path, 0.002, j0, j0_dot).expect("jacobi");
    let max_norm = result.field.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    assert!(
        max_norm < initial_norm * 5.0,
        "jacobi blew up: max {max_norm} vs initial {initial_norm}"
    );
}
