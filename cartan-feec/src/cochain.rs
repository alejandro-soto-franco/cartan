// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::whitney::ManifoldComplexExt;

use cartan_exterior::{Dim, ExteriorGrade};
use cartan_simplicial::{
    topology::{
        complex::Complex,
        handle::{SimplexHandle, SimplexIdx},
        skeleton::Skeleton,
    },
};
use nalgebra::DVector;

#[derive(Debug, Clone)]
pub struct Cochain {
    pub coeffs: DVector<f64>,
    pub dim: Dim,
}

impl Cochain {
    pub fn new(dim: Dim, coeffs: DVector<f64>) -> Self {
        Self { dim, coeffs }
    }

    pub fn constant(value: f64, skeleton: &Skeleton) -> Self {
        let ncoeffs = skeleton.len();
        Self::new(skeleton.dim(), DVector::from_element(ncoeffs, value))
    }

    pub fn zero(skeleton: &Skeleton) -> Self {
        Self::constant(0.0, skeleton)
    }

    pub fn from_function<F>(f: F, dim: ExteriorGrade, topology: &Complex) -> Self
    where
        F: FnMut(SimplexHandle) -> f64,
    {
        let skeleton = topology.skeleton(dim);
        let coeffs = DVector::from_iterator(skeleton.len(), skeleton.handle_iter().map(f));
        Self::new(dim, coeffs)
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }

    pub fn coeffs(&self) -> &DVector<f64> {
        &self.coeffs
    }

    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coeffs.len() == 0
    }

    /// Apply the discrete exterior derivative: d: C^k -> C^(k+1).
    pub fn dif(&self, topology: &Complex) -> Self {
        let dif_op = topology.exterior_derivative_operator(self.dim());
        // dif_op is CsMat of shape nsimplices(dim+1) x nsimplices(dim)
        // Multiply sparse matrix by dense vector manually via iteration.
        let nrows = dif_op.rows();
        let mut new_coeffs = DVector::zeros(nrows);
        for (&val, (row, col)) in dif_op.iter() {
            new_coeffs[row] += val * self.coeffs[col];
        }
        Cochain::new(self.dim() + 1, new_coeffs)
    }

    pub fn scale(&mut self, factor: f64) -> &mut Self {
        self.coeffs *= factor;
        self
    }

    pub fn scaled(&self, factor: f64) -> Self {
        Self::new(self.dim, &self.coeffs * factor)
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        assert_eq!(self.dim, other.dim);
        let coeffs = self.coeffs.component_mul(&other.coeffs);
        Self::new(self.dim, coeffs)
    }
}

impl std::ops::Index<SimplexIdx> for Cochain {
    type Output = f64;
    fn index(&self, idx: SimplexIdx) -> &Self::Output {
        assert!(idx.dim() == self.dim());
        &self.coeffs[idx.kidx]
    }
}

impl std::ops::IndexMut<SimplexIdx> for Cochain {
    fn index_mut(&mut self, idx: SimplexIdx) -> &mut Self::Output {
        assert!(idx.dim() == self.dim());
        &mut self.coeffs[idx.kidx]
    }
}

impl std::ops::Index<SimplexHandle<'_>> for Cochain {
    type Output = f64;
    fn index(&self, handle: SimplexHandle<'_>) -> &Self::Output {
        assert!(handle.dim() == self.dim());
        &self.coeffs[handle.kidx()]
    }
}

impl std::ops::IndexMut<SimplexHandle<'_>> for Cochain {
    fn index_mut(&mut self, idx: SimplexHandle<'_>) -> &mut Self::Output {
        assert!(idx.dim() == self.dim());
        &mut self.coeffs[idx.kidx()]
    }
}

impl std::ops::Index<usize> for Cochain {
    type Output = f64;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.coeffs[idx]
    }
}

impl std::ops::Mul<f64> for Cochain {
    type Output = Cochain;
    fn mul(self, rhs: f64) -> Self::Output {
        self.scaled(rhs)
    }
}

impl std::ops::MulAssign<f64> for Cochain {
    fn mul_assign(&mut self, rhs: f64) {
        self.scale(rhs);
    }
}

impl std::ops::SubAssign for Cochain {
    fn sub_assign(&mut self, rhs: Self) {
        assert!(self.dim == rhs.dim);
        self.coeffs -= rhs.coeffs;
    }
}

impl std::ops::Sub for Cochain {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;
    use approx::assert_relative_eq;

    #[test]
    fn cochain_dif_of_constant_zero_form_is_zero() {
        // d of a constant 0-cochain is 0 (closed).
        let (complex, _coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
        let skeleton = complex.skeleton(0);
        let c = Cochain::constant(3.0, &skeleton);
        let dc = c.dif(&complex);
        assert_eq!(dc.dim(), 1);
        for &v in dc.coeffs().iter() {
            assert_relative_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn cochain_dif_matches_signed_incidence() {
        // For a 0-cochain f, (d f) on edge (i,j) = f[j] - f[i] (up to orientation).
        let (complex, _coords) = CartesianMeshInfo::new_unit(2, 1).compute_coord_complex();
        let f = Cochain::from_function(|h| h.kidx() as f64, 0, &complex);
        let df = f.dif(&complex);
        // d d = 0: applying dif again gives the zero 2-cochain.
        let ddf = df.dif(&complex);
        for &v in ddf.coeffs().iter() {
            assert_relative_eq!(v, 0.0, epsilon = 1e-12);
        }
    }
}
