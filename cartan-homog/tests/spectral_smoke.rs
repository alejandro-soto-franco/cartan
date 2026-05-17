//! Smoke test for the Moulinec-Suquet spectral solver.
//!
//! Validation: homogeneous medium round-trips to `κ * I` exactly (the
//! Green's operator zeroes everything but DC on a constant polarisation
//! field, so iteration converges in a few steps to numerical zero
//! residual).
//!
//! Both scenarios (single phase + NO_PHASE-fallback-to-phase-0) run inside
//! one test because creating two `VkFftBackend` instances in the same
//! process trips on wgpu instance lifecycle / VkFFT static C state.
//! Tests across other Vulkan-using crates run in separate processes via
//! cargo test's per-test binary isolation.

#![cfg(feature = "gpu-fft")]

use cartan_homog::fullfield::voxelize::{VoxelGrid, NO_PHASE};
use cartan_homog::fullfield::SpectralFullField;

#[test]
fn spectral_homogeneous_medium_converges_to_kappa_identity() {
    // Skip cleanly if no Vulkan adapter.
    match cartan_gpu::Device::new() {
        Ok(_) => {}
        Err(cartan_gpu::GpuError::NoAdapter) => {
            eprintln!("spectral test skipped: no Vulkan adapter");
            return;
        }
        Err(e) => panic!("Vulkan device init failed: {e}"),
    }

    let mut solver = SpectralFullField::new(8);
    solver.max_iters = 50;
    solver.tol = 1e-6;

    // ----- Scenario A: single phase, all voxels tagged 0 -----
    let n = 8usize;
    let mut grid = VoxelGrid::new(n);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                grid.set(i, j, k, 0);
            }
        }
    }
    let kappa_a = [2.5_f32];
    let eff_a = solver
        .homogenize_voxel(&grid, &kappa_a)
        .expect("spectral homogenisation");

    eprintln!("κ_eff (κ=2.5, all-tagged) =\n{:.6}", eff_a.tensor);
    eprintln!(
        "  iterations: {:?}, residual: {:?}",
        eff_a.iterations, eff_a.residual
    );

    let identity = nalgebra::Matrix3::<f64>::identity();
    let frob_a = (eff_a.tensor - identity * 2.5).norm();
    assert!(
        frob_a < 1e-3,
        "homogeneous case κ_eff deviated from κ·I by {frob_a}"
    );

    // ----- Scenario B: NO_PHASE everywhere should fall back to index 0 -----
    let grid_b = VoxelGrid::new(n);
    assert_eq!(grid_b.get(0, 0, 0), NO_PHASE);
    let kappa_b = [1.5_f32];
    let eff_b = solver
        .homogenize_voxel(&grid_b, &kappa_b)
        .expect("NO_PHASE fallback");
    let frob_b = (eff_b.tensor - identity * 1.5).norm();
    assert!(
        frob_b < 1e-3,
        "NO_PHASE-fallback case κ_eff deviated from κ·I by {frob_b}"
    );
}
