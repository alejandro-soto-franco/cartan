//! Iteration count of Jacobi-preconditioned CG on the interior grade-1 mass,
//! across a refinement sequence.

use cartan_matfree::{HostMass, Interior, MassBackend, pcg};
use simplicial::r#gen::cartesian::CartesianGrid;

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
    println!(
        "{:>4} {:>10} {:>8} {:>12}",
        "ref", "ndofs", "iters", "residual"
    );
    for refinement in 2..=8 {
        let (topology, coords) = CartesianGrid::new_unit(3, refinement).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let interior = Interior::boundary_constrained(&topology, 1);
        let mass = HostMass::restricted(&topology, &geometry, &interior);
        let n = mass.ndofs();
        let rhs = probe(n, 7);
        let mut x = vec![0.0; n];
        let r = pcg(&mass, &rhs, &mut x, 1e-10, 2000);
        println!(
            "{refinement:>4} {n:>10} {:>8} {:>12.3e}",
            r.iterations, r.residual
        );
    }
}
