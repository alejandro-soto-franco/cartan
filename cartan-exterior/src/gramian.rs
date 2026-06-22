// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use nalgebra::{DMatrix, DVector, Cholesky};
use crate::Dim;

/// A Gram Matrix represents an inner product expressed in a basis.
#[derive(Debug, Clone)]
pub struct Gramian {
  /// S.P.D. matrix
  matrix: DMatrix<f64>,
}

impl Gramian {
  pub fn new(matrix: DMatrix<f64>) -> Self {
    assert!(Self::is_spd(&matrix), "Matrix must be s.p.d.");
    Self { matrix }
  }

  pub fn new_unchecked(matrix: DMatrix<f64>) -> Self {
    if cfg!(debug_assertions) {
      Self::new(matrix)
    } else {
      Self { matrix }
    }
  }

  pub fn from_euclidean_vectors(vectors: DMatrix<f64>) -> Self {
    assert!(Self::is_full_rank(&vectors, 1e-9), "Matrix must be full rank.");
    let matrix = vectors.transpose() * vectors;
    Self::new_unchecked(matrix)
  }

  /// Orthonormal euclidean metric.
  pub fn standard(dim: Dim) -> Self {
    let matrix = DMatrix::identity(dim, dim);
    Self::new_unchecked(matrix)
  }

  pub fn matrix(&self) -> &DMatrix<f64> {
    &self.matrix
  }

  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }

  pub fn det(&self) -> f64 {
    self.matrix.determinant()
  }

  pub fn det_sqrt(&self) -> f64 {
    self.det().sqrt()
  }

  pub fn inverse(self) -> Self {
    let matrix = self
      .matrix
      .try_inverse()
      .expect("Symmetric Positive Definite is always invertible.");
    Self::new_unchecked(matrix)
  }

  // Helper function to check if a matrix is full rank
  fn is_full_rank(matrix: &DMatrix<f64>, eps: f64) -> bool {
    if matrix.is_empty() {
      return true;
    }
    matrix.rank(eps) == matrix.nrows().min(matrix.ncols())
  }

  // Helper function to check if a matrix is symmetric positive definite
  fn is_spd(matrix: &DMatrix<f64>) -> bool {
    Cholesky::new(matrix.clone()).is_some()
  }
}

/// Inner product functionality directly on the basis.
impl Gramian {
  pub fn basis_inner(&self, i: usize, j: usize) -> f64 {
    self.matrix[(i, j)]
  }

  pub fn basis_norm_sq(&self, i: usize) -> f64 {
    self.basis_inner(i, i)
  }

  pub fn basis_norm(&self, i: usize) -> f64 {
    self.basis_norm_sq(i).sqrt()
  }

  pub fn basis_angle_cos(&self, i: usize, j: usize) -> f64 {
    self.basis_inner(i, j) / self.basis_norm(i) / self.basis_norm(j)
  }

  pub fn basis_angle(&self, i: usize, j: usize) -> f64 {
    self.basis_angle_cos(i, j).acos()
  }
}

impl std::ops::Index<(usize, usize)> for Gramian {
  type Output = f64;
  fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
    &self.matrix[(i, j)]
  }
}

/// Inner product functionality directly on any element.
impl Gramian {
  pub fn inner(&self, v: &DVector<f64>, w: &DVector<f64>) -> f64 {
    (v.transpose() * &self.matrix * w).x
  }

  pub fn inner_mat(&self, v: &DMatrix<f64>, w: &DMatrix<f64>) -> DMatrix<f64> {
    v.transpose() * &self.matrix * w
  }

  pub fn norm_sq(&self, v: &DVector<f64>) -> f64 {
    self.inner(v, v)
  }

  pub fn norm_sq_mat(&self, v: &DMatrix<f64>) -> DMatrix<f64> {
    self.inner_mat(v, v)
  }

  pub fn norm(&self, v: &DVector<f64>) -> f64 {
    self.inner(v, v).sqrt()
  }

  pub fn norm_mat(&self, v: &DMatrix<f64>) -> DMatrix<f64> {
    self.inner_mat(v, v).map(|v| v.sqrt())
  }

  pub fn angle_cos(&self, v: &DVector<f64>, w: &DVector<f64>) -> f64 {
    self.inner(v, w) / self.norm(v) / self.norm(w)
  }

  pub fn angle(&self, v: &DVector<f64>, w: &DVector<f64>) -> f64 {
    self.angle_cos(v, w).acos()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_relative_eq;

  #[test]
  fn standard_metric_is_identity_inner_product() {
    let g = Gramian::standard(3);
    assert_eq!(g.dim(), 3);
    let e0 = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let e1 = DVector::from_vec(vec![0.0, 1.0, 0.0]);
    assert_relative_eq!(g.inner(&e0, &e0), 1.0);
    assert_relative_eq!(g.inner(&e0, &e1), 0.0);
  }

  #[test]
  fn inverse_of_diagonal_metric() {
    let m = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0]));
    let g = Gramian::new(m);
    let gi = g.inverse();
    assert_relative_eq!(gi.basis_inner(0, 0), 0.25);
    assert_relative_eq!(gi.basis_inner(1, 1), 1.0 / 9.0);
  }
}
