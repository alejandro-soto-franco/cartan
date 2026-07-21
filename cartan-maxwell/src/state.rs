//! The Maxwell field state: E (1-cochain) and B (2-cochain).

use derham::cochain::Cochain;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;

/// Electromagnetic field state on the simplicial complex.
/// `e` is a 1-cochain (one DOF per edge); `b` is a 2-cochain (one DOF per face).
pub struct MaxwellState {
    pub e: Cochain,
    pub b: Cochain,
}

impl MaxwellState {
    pub fn new(e: Cochain, b: Cochain) -> Self {
        assert_eq!(e.grade(), 1, "E must be a 1-cochain");
        assert_eq!(b.grade(), 2, "B must be a 2-cochain");
        Self { e, b }
    }

    /// Discrete electromagnetic energy U = 1/2 (E^T M1 E + B^T M2 B) at the
    /// metric whose masses are `m1`, `m2`.
    pub fn energy(&self, m1: &CsrMatrix<f64>, m2: &CsrMatrix<f64>) -> f64 {
        let e = self.e.coeffs();
        let b = self.b.coeffs();
        let ue = e.dot(&(m1 * e));
        let ub = b.dot(&(m2 * b));
        0.5 * (ue + ub)
    }

    /// Synchronized half-step energy: U_sync = 1/2 (E_half^T M1 E_half + B^T M2 B),
    /// where E_half = 0.5 * (E^n + E^{n+1}) is the average of the electric field
    /// at two consecutive integer steps. This evaluates E and B at the same stagger
    /// point as B (the half-integer step), giving a better-conserved diagnostic than
    /// the cross-time energy from `energy()`.
    pub fn synchronized_energy(
        &self,
        e_next: &DVector<f64>,
        m1: &CsrMatrix<f64>,
        m2: &CsrMatrix<f64>,
    ) -> f64 {
        let e_half = 0.5 * (self.e.coeffs() + e_next);
        let b = self.b.coeffs();
        let ue = e_half.dot(&(m1 * &e_half));
        let ub = b.dot(&(m2 * b));
        0.5 * (ue + ub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolver::coboundary_matrix;
    use formoniq::whitney_complex::WhitneyComplex;
    use simplicial::r#gen::cartesian::CartesianGrid;

    #[test]
    fn energy_is_nonnegative() {
        let (complex, coords) = CartesianGrid::new_unit(2, 2).triangulate();
        let geom = coords.to_edge_lengths_sq(&complex);
        let wc = WhitneyComplex::new(&complex, &geom);
        let m1 = CsrMatrix::from(&wc.mass(1));
        let m2 = CsrMatrix::from(&wc.mass(2));
        let e = Cochain::new(1, DVector::from_element(complex.nsimplices(1), 0.5));
        let b = Cochain::new(2, DVector::from_element(complex.nsimplices(2), 0.25));
        let state = MaxwellState::new(e, b);
        assert!(state.energy(&m1, &m2) > 0.0);
    }

    #[test]
    fn coboundary_composition_is_zero_in_3d() {
        // d2 . d1 = 0 (the structural identity Faraday relies on).
        let (complex, _coords) = CartesianGrid::new_unit(3, 1).triangulate();
        let d1 = coboundary_matrix(&complex, 1); // 2-cochain <- 1-cochain
        let d2 = coboundary_matrix(&complex, 2); // 3-cochain <- 2-cochain
        let prod = &d2 * &d1;
        let max = prod.values().iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        assert!(max < 1e-12, "d2 . d1 = {max:e}, expected 0");
    }
}
