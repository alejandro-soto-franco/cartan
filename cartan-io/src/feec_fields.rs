//! FEEC field reconstruction from cochains to per-cell scalar and vector fields.
//!
//! Uses Whitney forms evaluated at cell barycentres to convert discrete
//! cochains into per-cell values suitable for VTK output.
//!
//! Values are sampled through [`Sampler`], which expresses the result in the
//! **ambient** frame rather than the cell reference frame. VTK consumers read
//! vector components in world coordinates, so the pullback is required and not
//! cosmetic.
//!
//! [`Sampler`]: derham::section::Sampler

use derham::cochain::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::section::SectionExt;
use simplicial::atlas::MeshPoint;
use simplicial::geometry::coord::mesh::MeshCoords;
use simplicial::topology::complex::Complex;

/// Reconstruct a scalar field from a top-dimensional cochain.
///
/// For each top cell, evaluates the Whitney form at the cell barycentre
/// and returns the L2 norm of the exterior-element coefficients. This works
/// for any grade: a 2-cochain in 2D has a 1-dimensional grade-2 space, so
/// the norm equals the absolute value of the single coefficient.
///
/// Returns a flat `Vec<f64>` of length `complex.nsimplices(complex.dim())`.
pub fn cell_scalar_from_cochain(
    complex: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
) -> Vec<f64> {
    let interpolant = WhitneyInterpolant::new(cochain.clone(), complex);
    let sampler = interpolant.sampled_on(complex, coords);
    complex
        .cells()
        .handle_iter()
        .map(|cell| {
            let value = sampler.at_point(&MeshPoint::barycenter(cell.idx()));
            value.coeffs().iter().map(|&v| v * v).sum::<f64>().sqrt()
        })
        .collect()
}

/// Reconstruct a vector field from a grade-1 cochain.
///
/// For each top cell, evaluates the Whitney form at the cell barycentre
/// and copies the grade-1 coefficients into a length-3 slot, padding z=0
/// for 2D meshes.
///
/// Returns a flat `Vec<f64>` of length `3 * complex.nsimplices(complex.dim())`.
pub fn cell_vectors_from_cochain(
    complex: &Complex,
    coords: &MeshCoords,
    cochain: &Cochain,
) -> Vec<f64> {
    let interpolant = WhitneyInterpolant::new(cochain.clone(), complex);
    let sampler = interpolant.sampled_on(complex, coords);
    let ambient_dim = coords.dim();
    let mut result = Vec::with_capacity(3 * complex.nsimplices(complex.dim()));
    for cell in complex.cells().handle_iter() {
        let value = sampler.at_point(&MeshPoint::barycenter(cell.idx()));
        let coeffs = value.coeffs();
        let x = coeffs.get(0).copied().unwrap_or(0.0);
        let y = coeffs.get(1).copied().unwrap_or(0.0);
        let z = if ambient_dim >= 3 {
            coeffs.get(2).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        result.push(x);
        result.push(y);
        result.push(z);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use simplicial::r#gen::cartesian::CartesianGrid;

    #[test]
    fn reconstruction_lengths_match_cell_count() {
        let (complex, coords) = CartesianGrid::new_unit(2, 2).triangulate();
        let ncells = complex.nsimplices(2);
        let e = Cochain::new(
            1,
            nalgebra::DVector::from_element(complex.nsimplices(1), 1.0),
        );
        let b = Cochain::new(
            2,
            nalgebra::DVector::from_element(complex.nsimplices(2), 0.5),
        );
        let evec = cell_vectors_from_cochain(&complex, &coords, &e);
        let bscal = cell_scalar_from_cochain(&complex, &coords, &b);
        assert_eq!(evec.len(), 3 * ncells);
        assert_eq!(bscal.len(), ncells);
        // a unit B 2-cochain reconstructs to a nonzero density somewhere
        assert!(bscal.iter().any(|&v| v.abs() > 0.0));
    }
}
