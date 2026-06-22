//! The Maxwell field state: E (1-cochain) and B (2-cochain).

use cartan_feec::cochain::Cochain;
use nalgebra::DVector;
use sprs::CsMat;

/// Multiply a sparse CSC/CSR matrix by a dense column vector, returning a dense vector.
fn spmv(m: &CsMat<f64>, v: &DVector<f64>) -> DVector<f64> {
    let mut result = DVector::zeros(m.rows());
    for (&val, (row, col)) in m.iter() {
        result[row] += val * v[col];
    }
    result
}

/// Electromagnetic field state on the simplicial complex.
/// `e` is a 1-cochain (one DOF per edge); `b` is a 2-cochain (one DOF per face).
pub struct MaxwellState {
    pub e: Cochain,
    pub b: Cochain,
}

impl MaxwellState {
    pub fn new(e: Cochain, b: Cochain) -> Self {
        assert_eq!(e.dim(), 1, "E must be a 1-cochain");
        assert_eq!(b.dim(), 2, "B must be a 2-cochain");
        Self { e, b }
    }

    /// Discrete electromagnetic energy U = 1/2 (E^T M1 E + B^T M2 B) at the
    /// metric whose masses are `m1`, `m2`.
    pub fn energy(&self, m1: &CsMat<f64>, m2: &CsMat<f64>) -> f64 {
        let e = self.e.coeffs();
        let b = self.b.coeffs();
        let ue = e.dot(&spmv(m1, e));
        let ub = b.dot(&spmv(m2, b));
        0.5 * (ue + ub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolver::coboundary_matrix;
    use cartan_feec::assemble::assemble_galmat;
    use cartan_feec::operators::HodgeMassElmat;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;

    #[test]
    fn energy_is_nonnegative() {
        let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
        let geom = coords.to_edge_lengths(&complex);
        let m1 = assemble_galmat(&complex, &geom, HodgeMassElmat::new(2, 1));
        let m2 = assemble_galmat(&complex, &geom, HodgeMassElmat::new(2, 2));
        let e = Cochain::new(1, nalgebra::DVector::from_element(complex.nsimplices(1), 0.5));
        let b = Cochain::new(2, nalgebra::DVector::from_element(complex.nsimplices(2), 0.25));
        let state = MaxwellState::new(e, b);
        assert!(state.energy(&m1, &m2) > 0.0);
    }

    #[test]
    fn coboundary_composition_is_zero_in_3d() {
        // d2 . d1 = 0 (the structural identity Faraday relies on).
        let (complex, _coords) = CartesianMeshInfo::new_unit(3, 1).compute_coord_complex();
        let d1 = coboundary_matrix(&complex, 1); // 2-cochain <- 1-cochain
        let d2 = coboundary_matrix(&complex, 2); // 3-cochain <- 2-cochain
        let prod = &d2 * &d1;
        let max = prod.data().iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        assert!(max < 1e-12, "d2 . d1 = {max:e}, expected 0");
    }
}
