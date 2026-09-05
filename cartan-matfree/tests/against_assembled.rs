//! The element-by-element operator must agree with the assembled Galerkin
//! matrix, and the CG solve with the dense factorisation it replaces.

use cartan_matfree::{pcg, HostMass, Interior, MassBackend};
use formoniq::whitney_complex::{RelativeWhitneyComplex, WhitneyComplex};
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use simplicial::r#gen::cartesian::CartesianGrid;

/// A reproducible pseudo-random vector. The exact values do not matter, only
/// that the same ones reach both paths.
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

#[test]
fn element_apply_matches_assembled_mass_every_grade() {
    for dim in 2..=3 {
        let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let wc = WhitneyComplex::new(&topology, &geometry);

        for grade in 0..=dim {
            let assembled = CsrMatrix::from(&wc.mass(grade));
            let matfree = HostMass::new(&topology, &geometry, grade);
            let n = matfree.ndofs();
            assert_eq!(n, assembled.nrows(), "dim {dim} grade {grade}");

            let x = probe(n, 17 + grade as u64);
            let reference = &assembled * DVector::from_vec(x.clone());

            let mut y = vec![0.0; n];
            matfree.apply_slice(&x, &mut y);

            for i in 0..n {
                approx::assert_relative_eq!(y[i], reference[i], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn restricted_apply_matches_the_relative_mass() {
    for dim in 2..=3 {
        let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let wc = WhitneyComplex::new(&topology, &geometry);
        let rel = RelativeWhitneyComplex::new(wc);

        for grade in 0..=dim {
            let assembled = CsrMatrix::from(&rel.mass(grade));
            let interior = Interior::boundary_constrained(&topology, grade);
            let matfree = HostMass::restricted(&topology, &geometry, &interior);
            let n = matfree.ndofs();
            assert_eq!(n, assembled.nrows(), "dim {dim} grade {grade}");
            if n == 0 {
                continue;
            }

            let x = probe(n, 91 + grade as u64);
            let reference = &assembled * DVector::from_vec(x.clone());

            let mut y = vec![0.0; n];
            matfree.apply_slice(&x, &mut y);

            for i in 0..n {
                approx::assert_relative_eq!(y[i], reference[i], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn cg_reproduces_the_dense_cholesky_solution() {
    let dim = 3;
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);
    let wc = WhitneyComplex::new(&topology, &geometry);
    let rel = RelativeWhitneyComplex::new(wc);

    let grade = 1;
    let interior = Interior::boundary_constrained(&topology, grade);
    let matfree = HostMass::restricted(&topology, &geometry, &interior);
    let n = matfree.ndofs();
    assert!(n > 0);

    let rhs = probe(n, 404);

    // The route this replaces: densify the interior mass and factorise it.
    let dense = DMatrix::from(&rel.mass(grade));
    let expected = dense
        .cholesky()
        .expect("interior mass is SPD")
        .solve(&DVector::from_vec(rhs.clone()));

    let mut got = vec![0.0; n];
    let report = pcg(&matfree, &rhs, &mut got, 1e-13, 500);
    assert!(report.converged, "CG stalled at {}", report.residual);

    for i in 0..n {
        approx::assert_relative_eq!(got[i], expected[i], epsilon = 1e-9);
    }
}

#[test]
fn cg_iteration_count_does_not_grow_with_the_mesh() {
    // Small meshes sit in a pre-asymptotic regime where the count still climbs
    // (13, 19, 22 at refinements 2, 3, 4). Mesh-independence is a statement
    // about the asymptotic regime, so the comparison starts past it.
    let mut counts = Vec::new();
    for refinement in [5, 8] {
        let (topology, coords) = CartesianGrid::new_unit(3, refinement).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);
        let interior = Interior::boundary_constrained(&topology, 1);
        let matfree = HostMass::restricted(&topology, &geometry, &interior);
        let n = matfree.ndofs();

        let rhs = probe(n, 7);
        let mut x = vec![0.0; n];
        let report = pcg(&matfree, &rhs, &mut x, 1e-10, 500);
        assert!(report.converged);
        counts.push((n, report.iterations));
    }

    let (n_first, it_first) = counts[0];
    let (n_last, it_last) = counts[1];
    assert!(n_last > 4 * n_first, "meshes were too close: {counts:?}");
    assert!(
        it_last <= it_first + 2,
        "iteration count grew with the mesh: {counts:?}"
    );
}

#[test]
fn gather_form_matches_the_scatter_form() {
    use cartan_matfree::GatherMap;

    for dim in 2..=3 {
        let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
        let geometry = coords.to_edge_lengths_sq(&topology);

        for grade in 0..=dim {
            let interior = Interior::boundary_constrained(&topology, grade);
            let mass = HostMass::restricted(&topology, &geometry, &interior);
            let n = mass.ndofs();
            if n == 0 {
                continue;
            }
            let gather = GatherMap::new(&mass);
            assert_eq!(gather.ndofs(), n);

            let x = probe(n, 55 + grade as u64);
            let mut scattered = vec![0.0; n];
            let mut gathered = vec![0.0; n];
            mass.apply_slice(&x, &mut scattered);
            gather.apply_slice(&mass, &x, &mut gathered);

            for i in 0..n {
                approx::assert_relative_eq!(gathered[i], scattered[i], epsilon = 1e-13);
            }
        }
    }
}
