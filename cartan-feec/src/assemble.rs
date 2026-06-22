// Ported from luiswirth/formoniq (used with permission), adapted for cartan.
// Assembly loop adapted from formoniq/src/assemble.rs; CooMatrix replaced
// by sprs::TriMat, boundary handling generalized to grade-k DOFs.

use crate::operators::{DofIdx, ElMatProvider, ElVecProvider};

use cartan_exterior::ExteriorGrade;
use cartan_simplicial::{
    geometry::metric::mesh::MeshLengths,
    topology::complex::Complex,
};

use nalgebra::DVector;
use rayon::prelude::*;
use sprs::{CsMat, TriMat};
use std::collections::HashSet;

/// Assemble the Galerkin matrix for an element-matrix provider.
/// Returns a CSC sparse matrix of shape (nsimps_row x nsimps_col).
pub fn assemble_galmat(
    topology: &Complex,
    geometry: &MeshLengths,
    elmat: impl ElMatProvider,
) -> CsMat<f64> {
    let row_grade = elmat.row_grade();
    let col_grade = elmat.col_grade();

    let nsimps_row = topology.nsimplices(row_grade);
    let nsimps_col = topology.nsimplices(col_grade);

    let triplets: Vec<(usize, usize, f64)> = topology
        .cells()
        .handle_iter()
        .par_bridge()
        .flat_map(|cell| {
            let geo = geometry.simplex_lengths(cell);
            let local_mat = elmat.eval(&geo);

            let row_subs: Vec<_> = cell.mesh_subsimps(row_grade).collect();
            let col_subs: Vec<_> = cell.mesh_subsimps(col_grade).collect();

            let mut local_triplets = Vec::new();
            for (ilocal, iglobal) in row_subs.iter().enumerate() {
                for (jlocal, jglobal) in col_subs.iter().enumerate() {
                    let val = local_mat[(ilocal, jlocal)];
                    if val != 0.0 {
                        local_triplets.push((iglobal.kidx(), jglobal.kidx(), val));
                    }
                }
            }

            local_triplets
        })
        .collect();

    let mut tri = TriMat::new((nsimps_row, nsimps_col));
    for (r, c, v) in triplets {
        tri.add_triplet(r, c, v);
    }
    tri.to_csc()
}

/// Assemble the Galerkin vector for an element-vector provider.
/// Returns a dense DVector of length nsimps for the given grade.
pub fn assemble_galvec(
    topology: &Complex,
    geometry: &MeshLengths,
    elvec: impl ElVecProvider,
) -> DVector<f64> {
    let grade = elvec.grade();
    let nsimps = topology.nsimplices(grade);

    let entries: Vec<(usize, f64)> = topology
        .cells()
        .handle_iter()
        .par_bridge()
        .flat_map(|cell| {
            let geo = geometry.simplex_lengths(cell);
            let cell_simp = (*cell).clone();
            let local_vec = elvec.eval(&geo, &cell_simp);

            let subs: Vec<_> = cell.mesh_subsimps(grade).collect();
            let mut local_entries = Vec::new();
            for (ilocal, iglobal) in subs.iter().enumerate() {
                if local_vec[ilocal] != 0.0 {
                    local_entries.push((iglobal.kidx(), local_vec[ilocal]));
                }
            }

            local_entries
        })
        .collect();

    let mut galvec = DVector::zeros(nsimps);
    for (irow, val) in entries {
        galvec[irow] += val;
    }
    galvec
}

/// Compute the set of DOF indices (grade-k simplex kidx values) that lie on
/// the boundary. A grade-k simplex is "on the boundary" if it is a
/// sub-simplex of at least one boundary facet.
pub fn boundary_grade_dofs(complex: &Complex, grade: ExteriorGrade) -> Vec<DofIdx> {
    if grade == 0 {
        return complex.boundary_vertices();
    }

    let boundary_facets = complex.boundary_facets();
    let mut dofs: HashSet<DofIdx> = HashSet::new();

    for facet_idx in boundary_facets {
        let facet_handle = facet_idx.handle(complex);
        // Each grade-k subsimplex of this boundary facet is a boundary DOF.
        for sub in facet_handle.mesh_subsimps(grade) {
            dofs.insert(sub.kidx());
        }
    }

    let mut result: Vec<DofIdx> = dofs.into_iter().collect();
    result.sort_unstable();
    result
}

/// Compute the set of DOF indices (grade-k simplex kidx values) that are
/// in the interior (complement of boundary DOFs), sorted ascending.
pub fn interior_grade_dofs(complex: &Complex, grade: ExteriorGrade) -> Vec<DofIdx> {
    let total = complex.nsimplices(grade);
    let boundary: HashSet<DofIdx> = boundary_grade_dofs(complex, grade).into_iter().collect();
    let mut interior: Vec<DofIdx> = (0..total).filter(|i| !boundary.contains(i)).collect();
    interior.sort_unstable();
    interior
}

/// Return the row/col submatrix of `m` on the given retained indices (in order).
pub fn restrict_galmat(m: &CsMat<f64>, retained: &[usize]) -> CsMat<f64> {
    let mut pos = std::collections::HashMap::new();
    for (new, &old) in retained.iter().enumerate() {
        pos.insert(old, new);
    }
    let mut tri = TriMat::new((retained.len(), retained.len()));
    for (&val, (r, c)) in m.iter() {
        if let (Some(&nr), Some(&nc)) = (pos.get(&r), pos.get(&c)) {
            tri.add_triplet(nr, nc, val);
        }
    }
    tri.to_csc()
}

/// Enforce homogeneous Dirichlet boundary conditions on grade-0 DOFs
/// (vertices) by zeroing out boundary rows/cols and placing 1 on the diagonal.
/// The galvec entries at boundary DOFs are set to zero.
pub fn enforce_homogeneous_dirichlet_bc(
    complex: &Complex,
    galmat: &mut CsMat<f64>,
    galvec: &mut DVector<f64>,
) {
    let boundary_dofs: HashSet<DofIdx> =
        boundary_grade_dofs(complex, 0).into_iter().collect();

    let ndofs = galmat.rows();
    assert_eq!(ndofs, galmat.cols());

    // Build a new matrix zeroing boundary rows and cols, identity on boundary diagonal.
    let mut tri = TriMat::new((ndofs, ndofs));
    for (&val, (r, c)) in galmat.iter() {
        if !boundary_dofs.contains(&r) && !boundary_dofs.contains(&c) {
            tri.add_triplet(r, c, val);
        }
    }
    for &i in &boundary_dofs {
        tri.add_triplet(i, i, 1.0);
        galvec[i] = 0.0;
    }
    *galmat = tri.to_csc();
}

/// Returns the interior DOF indices for grade-k boundary restriction.
/// Alias matching the plan's interface block.
pub fn drop_boundary_dofs_galmat(complex: &Complex, grade: ExteriorGrade) -> Vec<DofIdx> {
    interior_grade_dofs(complex, grade)
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use crate::operators::{HodgeMassElmat, ScalarMassElmat};
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    fn assemble_dense(grade: cartan_exterior::ExteriorGrade, n: usize) -> nalgebra::DMatrix<f64> {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, n).compute_coord_complex();
        let geom = coords.to_edge_lengths(&complex);
        let galmat = if grade == 0 {
            assemble_galmat(&complex, &geom, ScalarMassElmat)
        } else {
            assemble_galmat(&complex, &geom, HodgeMassElmat::new(2, grade))
        };
        let nd = galmat.rows();
        let mut dense = nalgebra::DMatrix::zeros(nd, nd);
        for (&v, (r, c)) in galmat.iter() {
            dense[(r, c)] += v;
        }
        dense
    }

    #[test]
    fn assembled_scalar_mass_is_symmetric_spd() {
        let m = assemble_dense(0, 3);
        assert!((&m - &m.transpose()).norm() < 1e-12, "not symmetric");
        assert!(m.clone().cholesky().is_some(), "not SPD");
    }

    #[test]
    fn assembled_grade1_hodge_mass_is_symmetric_spd() {
        let m = assemble_dense(1, 3);
        assert!((&m - &m.transpose()).norm() < 1e-12, "not symmetric");
        assert!(m.clone().cholesky().is_some(), "grade-1 Hodge mass not SPD");
    }
}
