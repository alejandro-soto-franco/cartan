//! Integration test: stochastic development on Bures-Wasserstein SPD.
//!
//! Exercises the L0↔L1 boundary: a manifold from `cartan-manifolds`
//! (Bures-Wasserstein SPD, uses a non-exact vector transport) plugged into
//! `stochastic_development` from `cartan-stochastic`. Every path point must
//! remain SPD (symmetric, all eigenvalues positive).

use cartan_core::{Manifold, Real};
use cartan_manifolds::SpdBuresWasserstein;
use cartan_stochastic::{random_frame_at, stochastic_development};
use nalgebra::SMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn bw_spd_bm_stays_on_cone() {
    const N: usize = 3;
    let m: SpdBuresWasserstein<N> = SpdBuresWasserstein;
    let mut rng = StdRng::seed_from_u64(0xBADBADBE);

    // Start at identity.
    let p0 = SMatrix::<Real, N, N>::identity();
    let frame = random_frame_at(&m, &p0, &mut rng).expect("frame at I");

    // Short horizon, small steps: BW retraction is a polynomial and the
    // vector transport is first-order, so we stay conservative on dt to
    // avoid the Gram-Schmidt re-orthonormalisation straining at large steps.
    let n_steps = 80;
    let dt = 0.002;
    let result = stochastic_development(&m, &p0, frame, n_steps, dt, &mut rng, 1e-8)
        .expect("development on BW-SPD");

    assert_eq!(result.path.len(), n_steps + 1);
    for (i, p) in result.path.iter().enumerate() {
        m.check_point(p).unwrap_or_else(|e| {
            panic!("step {i}: check_point failed: {e:?}, P = {p:?}");
        });
    }

    // Sanity: the trace should drift (Wishart-like expansion) away from the
    // initial trace of N. Very loose bound: trace stays in (0.5·N, 10·N)
    // over the short horizon. This catches gross numerical blowup.
    let final_tr = result.path.last().unwrap().trace();
    let n_real = N as Real;
    assert!(
        final_tr > 0.5 * n_real && final_tr < 10.0 * n_real,
        "final trace {final_tr} out of sanity band for N={N}"
    );
}

#[test]
fn bw_spd_bm_from_nontrivial_point() {
    // Starting away from the identity: the optimal-transport log and the
    // Lyapunov-based vector transport both exercise non-diagonal arithmetic.
    const N: usize = 2;
    let m: SpdBuresWasserstein<N> = SpdBuresWasserstein;
    let mut rng = StdRng::seed_from_u64(0xF00DF00D);
    let p0 = SMatrix::<Real, N, N>::new(2.0, 0.4, 0.4, 1.3);
    m.check_point(&p0).unwrap();
    let frame = random_frame_at(&m, &p0, &mut rng).expect("frame");
    let result = stochastic_development(&m, &p0, frame, 50, 0.001, &mut rng, 1e-8)
        .expect("development");
    for (i, p) in result.path.iter().enumerate() {
        m.check_point(p).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
    }
}
