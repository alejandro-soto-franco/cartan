//! Stochastic development on S^2: path points must stay on the manifold.
//!
//! Equatorial BM on the sphere under the Stratonovich frame-bundle integrator
//! should preserve the constraint ||p|| = 1 to float tolerance. We also check
//! (very loosely) that the endpoint distribution is not biased toward either
//! pole: the mean of z-coordinate over N paths, starting from the equator,
//! should be near 0 for small T where the diffusion has not yet equilibrated
//! to the uniform spherical distribution.

use cartan_core::{Manifold, Real};
use cartan_manifolds::sphere::Sphere;
use cartan_stochastic::{random_frame_at, stochastic_development};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn sphere_bm_stays_on_sphere() {
    let s: Sphere<3> = Sphere::<3>;
    let mut rng = StdRng::seed_from_u64(0xCAFEBABE);

    // Start at the north pole.
    let p0 = nalgebra::SVector::<Real, 3>::new(0.0, 0.0, 1.0);
    let frame = random_frame_at(&s, &p0, &mut rng).expect("frame");

    let n_steps = 200;
    let dt = 0.001;
    let result = stochastic_development(&s, &p0, frame, n_steps, dt, &mut rng, 1e-10).expect("dev");

    assert_eq!(result.path.len(), n_steps + 1);
    for (i, p) in result.path.iter().enumerate() {
        let norm_sq: Real = p.dot(p);
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "step {i}: ||p||^2 = {norm_sq}, drifted off S^2"
        );
        // The manifold's own check_point should also pass.
        s.check_point(p).expect("check_point");
    }
}

#[test]
fn sphere_bm_endpoint_mean_unbiased() {
    // Start at north pole, short time horizon: the z-coordinate of the
    // endpoint should average to something strictly less than 1 (diffusion
    // pushed off the pole) but clearly positive (hasn't reached uniform
    // equilibrium yet). Any systematic z-drift would indicate a bug in the
    // Stratonovich correction.
    let s: Sphere<3> = Sphere::<3>;
    let n_paths = 400;
    let n_steps = 100;
    let dt = 0.005; // T = 0.5
    let mut z_sum = 0.0;
    for seed in 0..n_paths {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let p0 = nalgebra::SVector::<Real, 3>::new(0.0, 0.0, 1.0);
        let frame = random_frame_at(&s, &p0, &mut rng).expect("frame");
        let result =
            stochastic_development(&s, &p0, frame, n_steps, dt, &mut rng, 1e-10).expect("dev");
        z_sum += result.path.last().unwrap().z;
    }
    let z_mean = z_sum / n_paths as f64;
    // For BM on S^2 started at the north pole, E[z_T] = exp(-T) (known exactly
    // from the spherical heat kernel on the zonal harmonic). With T = 0.5,
    // E[z_T] ≈ 0.607. Allow ±0.1 for Monte Carlo noise at N=400.
    assert!(
        (z_mean - 0.607).abs() < 0.1,
        "endpoint z-mean {z_mean} outside expected band [0.507, 0.707]"
    );
}
