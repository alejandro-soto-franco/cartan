// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::{
  mesh::MeshCoords,
  simplex::{barycenter_local, SimplexCoords},
  CoordRef,
};
use crate::{
  Dim,
};

use crate::linalg::{Matrix, Vector};
use crate::topology::complex::Complex;
use crate::topology::handle::SimplexHandle;

/// A quadrature rule defined on the reference simplex.
///
/// Can be used to integrate functions defined on the reference simplex.
pub struct SimplexQuadRule {
  /// Points in local coordinates.
  points: Matrix,
  /// Normalized weights that sum to 1.
  weights: Vector,
}
impl SimplexQuadRule {
  pub fn dim(&self) -> Dim {
    self.points.nrows()
  }
  pub fn npoints(&self) -> usize {
    self.points.ncols()
  }
  /// Uses a local coordinate function `f`.
  pub fn integrate_local<F>(&self, f: &F, vol: f64) -> f64
  where
    F: Fn(CoordRef) -> f64,
  {
    let mut integral = 0.0;
    for i in 0..self.npoints() {
      let col = self.points.column(i);
      integral += self.weights[i] * f(&col);
    }
    vol * integral
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_coord<F>(&self, f: &F, coords: &SimplexCoords) -> f64
  where
    F: Fn(CoordRef) -> f64,
  {
    self.integrate_local(
      &|local_coord| {
        let local_vec = local_coord.as_ref().to_owned().into();
        let global = coords.local2global(&local_vec);
        f(&global.as_view())
      },
      coords.vol(),
    )
  }

  /// Uses a global coordinate function `f`.
  pub fn integrate_mesh<F>(&self, f: &F, complex: &Complex, coords: &MeshCoords) -> f64
  where
    F: Fn(CoordRef, SimplexHandle) -> f64,
  {
    let mut integral = 0.0;
    for cell in complex.cells().handle_iter() {
      let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, coords);
      integral += self.integrate_coord(&|x| f(x, cell), &cell_coords);
    }
    integral
  }
}

impl SimplexQuadRule {
  pub fn dim0() -> Self {
    let points = Matrix::zeros(0, 1);
    let weights = nalgebra::dvector![1.0];
    Self { points, weights }
  }

  /// Integrates 1st order affine linear functions exactly.
  pub fn barycentric(dim: Dim) -> Self {
    let barycenter = barycenter_local(dim);
    let points = Matrix::from_columns(&[barycenter]);
    let weight = 1.0;
    let weights = Vector::from_element(1, weight);
    Self { points, weights }
  }

  pub fn vertices(dim: Dim) -> Self {
    let nvertices = dim + 1;
    let mut points = Matrix::zeros(dim, nvertices);
    for ivertex in 1..nvertices {
      points[(ivertex - 1, ivertex)] = 1.0;
    }
    let weight = (nvertices as f64).recip();
    let weights = Vector::from_element(nvertices, weight);
    Self { points, weights }
  }
}

impl SimplexQuadRule {
  pub fn order3(dim: Dim) -> Self {
    match dim {
      0 => Self::dim0(),
      1 => Self::dim1_order3(),
      2 => Self::dim2_order3(),
      3 => Self::dim3_order3(),
      _ => unimplemented!("No order 3 Quadrature available for dim {dim}."),
    }
  }

  /// Simpsons Rule
  pub fn dim1_order3() -> Self {
    let points = nalgebra::dmatrix![0.0, 0.5, 1.0];
    let weights = nalgebra::dvector![1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0];
    Self { points, weights }
  }
  pub fn dim2_order3() -> Self {
    let points = nalgebra::dmatrix![
      1.0/3.0, 1.0/5.0, 3.0/5.0, 1.0/5.0;
      1.0/3.0, 1.0/5.0, 1.0/5.0, 3.0/5.0;
    ];
    let weights = nalgebra::dvector![-27.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0];
    Self { points, weights }
  }
  pub fn dim3_order3() -> Self {
    let a: f64 = 1.0 / 4.0; // centroid
    let b: f64 = 1.0 / 6.0;
    let c: f64 = 1.0 / 2.0;

    let points = nalgebra::dmatrix![
        a, b, c, b, b;
        a, b, b, c, b;
        a, b, b, b, c;
    ];
    let weights = nalgebra::dvector![-4.0 / 5.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0, 9.0 / 20.0];
    Self { points, weights }
  }
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn barycentric_rule_integrates_constant_to_volume() {
        // Integral of 1 over the reference triangle = its volume = 1/2.
        let qr = SimplexQuadRule::barycentric(2);
        let val = qr.integrate_local(&|_p| 1.0, 0.5);
        assert_relative_eq!(val, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn order3_rule_integrates_a_cubic_on_the_interval() {
        // On [0,1], integral of x^3 = 1/4. order3 in 1D is Simpson-exact for cubics.
        let qr = SimplexQuadRule::order3(1);
        let coords = crate::geometry::coord::simplex::SimplexCoords::new(
            nalgebra::DMatrix::from_row_slice(1, 2, &[0.0, 1.0]),
        );
        let val = qr.integrate_coord(&|p| p[0].powi(3), &coords);
        assert_relative_eq!(val, 0.25, epsilon = 1e-12);
    }
}
