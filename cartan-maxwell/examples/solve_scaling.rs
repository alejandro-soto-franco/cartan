//! Cost of one Ampere solve: the dense factorisation against the matrix-free
//! iteration, over a refinement sequence.

use cartan_matfree::{HostMass, Interior, MassBackend, pcg};
use formoniq::whitney_complex::{RelativeWhitneyComplex, WhitneyComplex};
use nalgebra::{DMatrix, DVector};
use simplicial::r#gen::cartesian::CartesianGrid;
use std::time::Instant;

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

fn main() {
    // Above this the dense interior mass stops fitting in a sensible budget.
    let dense_dof_ceiling: usize = 12000;

    println!(
        "{:>4} {:>8} {:>9} {:>12} {:>12} {:>9} {:>7}",
        "ref", "cells", "int dofs", "dense ms", "matfree ms", "speedup", "iters"
    );

    for refinement in [3, 4, 5, 6, 8, 10, 12, 14] {
        let (topology, coords) = CartesianGrid::new_unit(3, refinement).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let interior = Interior::boundary_constrained(&topology, 1);
        let ndofs = interior.ndofs();
        let ncells = topology.cells().len();
        let rhs = probe(ndofs, 7);

        // The route being replaced: assemble the relative mass, densify it,
        // factorise, solve. All of it is redone every step, since the metric
        // moves on an evolving background.
        let dense_ms = if ndofs <= dense_dof_ceiling {
            let t = Instant::now();
            let wc = WhitneyComplex::new(&topology, &geometry);
            let rel = RelativeWhitneyComplex::new(wc);
            let dense = DMatrix::from(&rel.mass(1));
            let _sol = dense
                .cholesky()
                .expect("interior mass is SPD")
                .solve(&DVector::from_vec(rhs.clone()));
            Some(t.elapsed().as_secs_f64() * 1e3)
        } else {
            None
        };

        // The replacement: element matrices once, then a Krylov iteration.
        let t = Instant::now();
        let mass = HostMass::restricted(&topology, &geometry, &interior);
        let mut sol = vec![0.0; mass.ndofs()];
        let report = pcg(&mass, &rhs, &mut sol, 1e-12, 500);
        let matfree_ms = t.elapsed().as_secs_f64() * 1e3;
        assert!(report.converged);

        match dense_ms {
            Some(d) => println!(
                "{refinement:>4} {ncells:>8} {ndofs:>9} {d:>12.1} {matfree_ms:>12.1} {:>8.1}x {:>7}",
                d / matfree_ms,
                report.iterations
            ),
            None => println!(
                "{refinement:>4} {ncells:>8} {ndofs:>9} {:>12} {matfree_ms:>12.1} {:>9} {:>7}",
                "over cap", "-", report.iterations
            ),
        }
    }
}
