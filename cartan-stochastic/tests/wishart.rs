//! Wishart BM: starting from I with shape n=N, E[X_T] = (1 + n·T) · I.
//!
//! Derivation: taking the expectation of the Itô SDE `dX = √X dB + dB^T √X + n I dt`
//! and using E[√X dB] = 0 (Itô isometry for martingale increments) gives
//! `dE[X]/dt = n I`, so E[X_T] = X_0 + n T I. Starting from I with n=N,
//! E[X_T] = (1 + N·T) · I.

use cartan_core::Real;
use cartan_stochastic::wishart_step;
use nalgebra::SMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn wishart_mean_matches_closed_form() {
    const N: usize = 3;
    let shape: Real = N as Real;
    let t_total: Real = 0.5;
    let n_steps: usize = 200;
    let dt = t_total / n_steps as Real;
    let n_paths: usize = 400;

    let x0 = SMatrix::<Real, N, N>::identity();
    let mut sum = SMatrix::<Real, N, N>::zeros();
    for seed in 0..n_paths {
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut x = x0;
        for _ in 0..n_steps {
            x = wishart_step(&x, shape, dt, &mut rng);
        }
        sum += x;
    }
    let mean = sum * (1.0 / n_paths as Real);

    // Expected: (1 + n·T) · I = (1 + 1.5) · I = 2.5 · I.
    let expected = shape * t_total + 1.0;
    // Diagonal Monte Carlo SE at N=400 is ~0.1-0.2 for this variance.
    for i in 0..N {
        let diag = mean[(i, i)];
        assert!(
            (diag - expected).abs() < 0.2,
            "Wishart mean diag[{i}] = {diag}, expected {expected}"
        );
    }
    // Off-diagonals should average to ~0 (no symmetry breaking).
    for i in 0..N {
        for j in 0..N {
            if i != j {
                let off = mean[(i, j)];
                assert!(off.abs() < 0.2, "off-diag [{i},{j}] = {off}");
            }
        }
    }
}

#[test]
fn wishart_stays_symmetric() {
    const N: usize = 2;
    let mut rng = StdRng::seed_from_u64(7);
    let mut x = SMatrix::<Real, N, N>::identity();
    for _ in 0..50 {
        x = wishart_step(&x, 2.0, 0.01, &mut rng);
        let asym = (x - x.transpose()).norm();
        assert!(asym < 1e-10, "symmetry drift {asym}");
    }
}
