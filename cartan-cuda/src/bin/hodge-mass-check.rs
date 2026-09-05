//! The device Hodge mass against the host path, on a refinement sequence.
//!
//! Reports the worst relative disagreement per mesh and exits non-zero if any
//! of them exceeds the tolerance the rest of cartan is measured at.

use cartan_cuda::{Device, DeviceHodgeMass};
use cartan_matfree::{GatherMap, HostMass, Interior, MassBackend};
use simplicial::r#gen::cartesian::CartesianGrid;
use std::time::Instant;

const TOLERANCE: f64 = 1e-13;

fn probe(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let mut worst_overall = 0.0f64;

    println!(
        "{:>4} {:>9} {:>9} {:>12} {:>12} {:>9} {:>11} {:>11}",
        "ref", "cells", "dofs", "cpu us/app", "gpu us/app", "speedup", "norm rel", "worst comp"
    );

    for refinement in [4, 8, 12, 16, 20, 24, 28, 32] {
        let (topology, coords) = CartesianGrid::new_unit(3, refinement).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let interior = Interior::boundary_constrained(&topology, 1);
        let mass = HostMass::restricted(&topology, &geometry, &interior);
        let gather = GatherMap::new(&mass);
        let n = mass.ndofs();
        let ncells = mass.ncells();

        let x = probe(n, 31);

        // Host reference, in the gather ordering the kernel uses.
        let mut expected = vec![0.0; n];
        let reps = 20;
        let t = Instant::now();
        for _ in 0..reps {
            gather.apply_slice(&mass, &x, &mut expected);
        }
        let cpu_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        let dev_mass = DeviceHodgeMass::new(
            &device,
            gather.offsets(),
            gather.entries(),
            mass.dof_map(),
            mass.elmats(),
            gather.nlocal(),
        )?;

        // Warm the kernel, then time it clear of the transfer.
        let got = dev_mass.apply(&device, &x)?;
        let t = Instant::now();
        dev_mass.apply_repeated(&device, &x, reps)?;
        let gpu_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

        // Componentwise relative error divides by an entry that can sit near
        // zero through cancellation, which reports a large number for a tiny
        // absolute disagreement. The norm-relative error is the one that says
        // whether the two paths computed the same vector; the componentwise
        // figure is reported alongside it, restricted to entries large enough
        // for the division to mean anything.
        let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let abs_err = got
            .iter()
            .zip(&expected)
            .fold(0.0f64, |m, (g, e)| m.max((g - e).abs()));
        let norm_rel = abs_err / scale;

        let floor = 1e-3 * scale;
        let worst_comp = got
            .iter()
            .zip(&expected)
            .filter(|(_, e)| e.abs() > floor)
            .fold(0.0f64, |m, (g, e)| m.max((g - e).abs() / e.abs()));

        worst_overall = worst_overall.max(norm_rel);

        println!(
            "{refinement:>4} {ncells:>9} {n:>9} {cpu_us:>12.1} {gpu_us:>12.1} {:>8.2}x {norm_rel:>11.2e} {worst_comp:>11.2e}",
            cpu_us / gpu_us
        );
    }

    if worst_overall > TOLERANCE {
        eprintln!("worst norm-relative disagreement {worst_overall:.3e} exceeds {TOLERANCE:.0e}");
        std::process::exit(1);
    }
    println!("\nagreement within {TOLERANCE:.0e} norm-relative on every mesh");
    Ok(())
}
