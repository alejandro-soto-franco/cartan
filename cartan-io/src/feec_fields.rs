//! FEEC field reconstruction from cochains to per-cell scalar and vector fields.
//!
//! Uses Whitney forms evaluated at cell barycenters to convert discrete
//! cochains into per-cell values suitable for VTK output.

use cartan_feec::cochain::Cochain;
use cartan_feec::whitney::form::WhitneyForm;
use cartan_simplicial::geometry::coord::mesh::MeshCoords;
use cartan_simplicial::topology::complex::Complex;
use cartan_simplicial::topology::handle::SimplexHandle;
use nalgebra::DVector;

/// Reconstruct a scalar field from a top-dimensional cochain.
///
/// For each top cell, evaluates the Whitney form at the cell barycenter
/// and returns the L2 norm of the ExteriorElement coefficients. This works
/// for any grade: a 2-cochain in 2D has a 1-dimensional grade-2 space, so
/// the norm equals the absolute value of the single coefficient.
///
/// Returns a flat `Vec<f64>` of length `complex.nsimplices(complex.dim())`.
pub fn cell_scalar_from_cochain(
    complex: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
) -> Vec<f64> {
    let form = WhitneyForm::new(cochain.clone(), complex, coords);
    let mut result = Vec::new();
    for cell in complex.cells().handle_iter() {
        let barycenter = cell_barycenter(&cell, coords);
        let elem = form.eval_known_cell(cell, &barycenter);
        let norm: f64 = elem.coeffs().iter().map(|&v| v * v).sum::<f64>().sqrt();
        result.push(norm);
    }
    result
}

/// Reconstruct a vector field from a grade-1 cochain.
///
/// For each top cell, evaluates the Whitney form at the cell barycenter
/// and copies the grade-1 ExteriorElement coefficients into a length-3 slot,
/// padding z=0 for 2D meshes.
///
/// Returns a flat `Vec<f64>` of length `3 * complex.nsimplices(complex.dim())`.
pub fn cell_vectors_from_cochain(
    complex: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
) -> Vec<f64> {
    let form = WhitneyForm::new(cochain.clone(), complex, coords);
    let ambient_dim = coords.dim();
    let mut result = Vec::new();
    for cell in complex.cells().handle_iter() {
        let barycenter = cell_barycenter(&cell, coords);
        let elem = form.eval_known_cell(cell, &barycenter);
        let coeffs = elem.coeffs();
        let x = if !coeffs.is_empty() { coeffs[0] } else { 0.0 };
        let y = if coeffs.len() > 1 { coeffs[1] } else { 0.0 };
        let z = if ambient_dim >= 3 && coeffs.len() > 2 { coeffs[2] } else { 0.0 };
        result.push(x);
        result.push(y);
        result.push(z);
    }
    result
}

/// Compute the barycenter of a cell given mesh coordinates.
fn cell_barycenter(cell: &SimplexHandle<'_>, coords: &MeshCoords) -> DVector<f64> {
    let verts: Vec<usize> = cell.iter().collect();
    let n = verts.len() as f64;
    let dim = coords.dim();
    let mut bary = DVector::zeros(dim);
    for &v in &verts {
        bary += coords.coord(v);
    }
    bary / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartan_feec::cochain::Cochain;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    #[test]
    fn reconstruction_lengths_match_cell_count() {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
        let ncells = complex.nsimplices(2);
        let e = Cochain::new(1, nalgebra::DVector::from_element(complex.nsimplices(1), 1.0));
        let b = Cochain::new(2, nalgebra::DVector::from_element(complex.nsimplices(2), 0.5));
        let evec = cell_vectors_from_cochain(&complex, &coords, &e);
        let bscal = cell_scalar_from_cochain(&complex, &coords, &b);
        assert_eq!(evec.len(), 3 * ncells);
        assert_eq!(bscal.len(), ncells);
        // a unit B 2-cochain reconstructs to a nonzero density somewhere
        assert!(bscal.iter().any(|&v| v.abs() > 0.0));
    }
}
