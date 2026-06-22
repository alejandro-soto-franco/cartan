// Ported from luiswirth/formoniq (used with permission), adapted for cartan.
//! Dense generalized symmetric eigenvalue solver for small assembled systems.
//! No PETSc/faer dependency; densifies the sparse operators (test meshes are small).

use nalgebra::{DMatrix, SymmetricEigen};
use sprs::CsMat;

fn to_dense(a: &CsMat<f64>) -> DMatrix<f64> {
    let mut d = DMatrix::zeros(a.rows(), a.cols());
    for (&v, (r, c)) in a.iter() {
        d[(r, c)] += v;
    }
    d
}

/// Solve `K x = lambda M x` for symmetric K and SPD M. Returns eigenvalues ascending.
pub fn generalized_symmetric_eigenvalues(k: &CsMat<f64>, m: &CsMat<f64>) -> Vec<f64> {
    let kd = to_dense(k);
    let md = to_dense(m);
    // M = L L^T; transform to standard problem A = L^-1 K L^-T.
    let chol = md.cholesky().expect("mass matrix must be SPD");
    let l = chol.l();
    let l_inv = l.clone().try_inverse().expect("Cholesky factor invertible");
    let a = &l_inv * &kd * l_inv.transpose();
    // Symmetrize against round-off, then solve.
    let a_sym = (&a + &a.transpose()) * 0.5;
    let mut vals: Vec<f64> = SymmetricEigen::new(a_sym).eigenvalues.iter().copied().collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    vals
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use crate::assemble::{assemble_galmat, interior_grade_dofs, restrict_galmat};
    use crate::operators::{LaplaceBeltramiElmat, ScalarMassElmat};
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    fn lowest_dirichlet_eigenvalue(n: usize) -> f64 {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, n).compute_coord_complex();
        let geom = coords.to_edge_lengths(&complex);
        let k = assemble_galmat(&complex, &geom, LaplaceBeltramiElmat::new(2));
        let m = assemble_galmat(&complex, &geom, ScalarMassElmat);
        let interior = interior_grade_dofs(&complex, 0);
        let k = restrict_galmat(&k, &interior);
        let m = restrict_galmat(&m, &interior);
        let vals = generalized_symmetric_eigenvalues(&k, &m);
        *vals.first().unwrap()
    }

    #[test]
    fn dirichlet_laplacian_lowest_eigenvalue_converges() {
        let target = 2.0 * std::f64::consts::PI.powi(2); // ~19.7392
        let coarse = lowest_dirichlet_eigenvalue(8);
        let fine = lowest_dirichlet_eigenvalue(16);
        // P1 FEM converges from above; both within range, fine closer than coarse.
        assert!(coarse > target - 1.0, "coarse {coarse} below target {target}");
        assert!((fine - target).abs() < (coarse - target).abs() + 1e-9,
            "no convergence: coarse {coarse}, fine {fine}, target {target}");
        assert!((fine - target).abs() / target < 0.05,
            "fine {fine} not within 5% of {target}");
    }
}
