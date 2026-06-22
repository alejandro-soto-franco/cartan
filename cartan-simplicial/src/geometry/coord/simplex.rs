// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::mesh::MeshCoords;
use crate::{
  geometry::{metric::simplex::SimplexLengths, refsimp_vol},
  topology::simplex::Simplex,
  affine::AffineTransform,
  linalg::{Matrix, RowVector, Vector},
  Dim,
};

use cartan_exterior::gramian::Gramian;
use cartan_exterior::combo::Sign;

#[derive(Debug, Clone)]
pub struct SimplexCoords {
  pub vertices: MeshCoords,
}

impl SimplexCoords {
  pub fn new(vertices: Matrix) -> Self {
    let vertices = vertices.into();
    Self { vertices }
  }
  pub fn standard(ndim: Dim) -> Self {
    let nvertices = ndim + 1;
    let mut vertices = Matrix::zeros(ndim, nvertices);
    for i in 0..ndim {
      vertices[(i, i + 1)] = 1.0;
    }
    Self::new(vertices)
  }
  pub fn from_simplex_and_coords(simp: &Simplex, coords: &MeshCoords) -> SimplexCoords {
    let mut vert_coords = Matrix::zeros(coords.dim(), simp.nvertices());
    for (i, v) in simp.iter().enumerate() {
      vert_coords.set_column(i, &coords.coord(v));
    }
    SimplexCoords::new(vert_coords)
  }

  pub fn nvertices(&self) -> usize {
    self.vertices.nvertices()
  }
  pub fn dim_intrinsic(&self) -> Dim {
    self.nvertices() - 1
  }
  pub fn dim_ambient(&self) -> Dim {
    self.vertices.dim()
  }
  pub fn is_same_dim(&self) -> bool {
    self.dim_intrinsic() == self.dim_ambient()
  }

  pub fn coord(&self, ivertex: usize) -> Vector {
    self.vertices.coord(ivertex).into_owned()
  }
  pub fn coord_iter(&self) -> impl Iterator<Item = Vector> + '_ {
    self.vertices.coord_iter().map(|c| c.into_owned())
  }

  pub fn base_vertex(&self) -> Vector {
    self.coord(0)
  }

  pub fn spanning_vector(&self, i: usize) -> Vector {
    assert!(i < self.dim_intrinsic());
    self.coord(i + 1) - self.base_vertex()
  }
  pub fn spanning_vectors(&self) -> Matrix {
    let mut mat = Matrix::zeros(self.dim_ambient(), self.dim_intrinsic());
    let v0 = self.base_vertex();
    for (i, vi) in self.vertices.coord_iter().skip(1).enumerate() {
      let v0i = vi - &v0;
      mat.set_column(i, &v0i);
    }
    mat
  }

  pub fn metric_tensor(&self) -> Gramian {
    Gramian::from_euclidean_vectors(self.spanning_vectors())
  }

  pub fn det(&self) -> f64 {
    let det = if self.is_same_dim() {
      self.spanning_vectors().determinant()
    } else {
      self.metric_tensor().det_sqrt()
    };
    refsimp_vol(self.dim_intrinsic()) * det
  }
  pub fn vol(&self) -> f64 {
    self.det().abs()
  }
  pub fn is_degenerate(&self) -> bool {
    self.vol() <= 1e-12
  }

  pub fn orientation(&self) -> Sign {
    Sign::from_f64(self.det()).unwrap()
  }

  pub fn linear_transform(&self) -> Matrix {
    self.spanning_vectors()
  }
  pub fn inv_linear_transform(&self) -> Matrix {
    if self.dim_intrinsic() == 0 {
      Matrix::zeros(0, 0)
    } else {
      self.linear_transform().pseudo_inverse(1e-12).unwrap()
    }
  }

  pub fn pushforward_vector(&self, local: &Vector) -> Vector {
    self.linear_transform() * local
  }
  pub fn pullback_covector(&self, global: &RowVector) -> RowVector {
    global * &self.linear_transform()
  }

  pub fn affine_transform(&self) -> AffineTransform {
    let translation = self.base_vertex();
    let linear = self.linear_transform();
    AffineTransform::new(translation, linear)
  }
  pub fn local2global(&self, local: &Vector) -> Vector {
    self.affine_transform().apply_forward(local)
  }
  pub fn global2local(&self, global: &Vector) -> Vector {
    self.affine_transform().apply_backward(global)
  }
  pub fn global2bary(&self, global: &Vector) -> Vector {
    local2bary(&self.global2local(global))
  }

  pub fn bary2global(&self, bary: &Vector) -> Vector {
    self
      .vertices
      .coord_iter()
      .zip(bary.iter())
      .map(|(vi, &baryi)| baryi * vi)
      .sum()
  }

  pub fn difbarys(&self) -> Matrix {
    let difs = self.inv_linear_transform();
    let mut difs = difs.insert_row(0, 0.0);
    difs.set_row(0, &-difs.row_sum());
    difs
  }

  pub fn barycenter(&self) -> Vector {
    let mut barycenter = Vector::zeros(self.dim_ambient());
    self.vertices.coord_iter().for_each(|v| barycenter += v);
    barycenter /= self.nvertices() as f64;
    barycenter
  }
  pub fn is_global_inside(&self, global: &Vector) -> bool {
    is_bary_inside(&self.global2bary(global))
  }
}

pub fn bary2local(bary: &Vector) -> Vector {
  bary.view_range(1.., ..).into_owned()
}

pub fn local2bary(local: &Vector) -> Vector {
  let bary0 = 1.0 - local.sum();
  local.clone().insert_row(0, bary0)
}

pub fn is_bary_inside(bary: &Vector) -> bool {
  let sum = bary.sum();
  (sum - 1.0).abs() < 1e-9 && bary.iter().all(|&b| (0.0..=1.0).contains(&b))
}

impl SimplexCoords {
  pub fn subsimps(&self, sub_dim: Dim) -> Vec<SimplexCoords> {
    Simplex::standard(self.dim_intrinsic())
      .subsequences(sub_dim)
      .map(|edge| Self::from_simplex_and_coords(&edge, &self.vertices))
      .collect()
  }
  pub fn edges(&self) -> Vec<SimplexCoords> {
    self.subsimps(1)
  }

  pub fn swap_vertices(&mut self, icol: usize, jcol: usize) {
    self.vertices.swap_coords(icol, jcol)
  }
  pub fn flip_orientation(&mut self) {
    if self.nvertices() >= 2 {
      self.swap_vertices(0, 1)
    }
  }
  pub fn flipped_orientation(mut self) -> Self {
    self.flip_orientation();
    self
  }

  pub fn to_lengths(&self) -> SimplexLengths {
    SimplexLengths::from_metric_tensor(&self.metric_tensor())
  }
}

pub fn barycenter_local(dim: Dim) -> Vector {
  let nvertices = dim + 1;
  let value = 1.0 / nvertices as f64;
  Vector::from_element(dim, value)
}

pub fn barycenter_bary(dim: Dim) -> Vector {
  let nvertices = dim + 1;
  let value = 1.0 / nvertices as f64;
  Vector::from_element(nvertices, value)
}

pub fn ref_bary(ivertex: usize, coord: &Vector) -> f64 {
  let dim = coord.len();
  assert!(ivertex <= dim);
  if ivertex == 0 {
    1.0 - coord.sum()
  } else {
    coord[ivertex - 1]
  }
}

pub fn ref_difbary(dim: Dim, ivertex: usize) -> RowVector {
  assert!(ivertex <= dim);
  if ivertex == 0 {
    RowVector::from_element(dim, -1.0)
  } else {
    let mut v = RowVector::zeros(dim);
    v[ivertex - 1] = 1.0;
    v
  }
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn coord_volume_matches_length_volume() {
        // Right triangle with legs 1 along x and y in R^2.
        let m = nalgebra::DMatrix::from_row_slice(2, 3, &[
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let sc = SimplexCoords::new(m);
        assert_relative_eq!(sc.vol(), 0.5, epsilon = 1e-12);
        let sl = sc.to_lengths();
        assert_relative_eq!(sl.vol(), sc.vol(), epsilon = 1e-12);
    }

    #[test]
    fn metric_tensor_of_orthonormal_legs_is_identity() {
        let m = nalgebra::DMatrix::from_row_slice(2, 3, &[
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
        let sc = SimplexCoords::new(m);
        let g = sc.metric_tensor();
        assert_relative_eq!(g.matrix()[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(g.matrix()[(1, 1)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(g.matrix()[(0, 1)], 0.0, epsilon = 1e-12);
    }
}
