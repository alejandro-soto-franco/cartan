// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::simplex::SimplexCoords;
use crate::{
  geometry::metric::mesh::MeshLengths,
  topology::{complex::Complex, VertexIdx},
  linalg::{Matrix, Vector},
  Dim,
};

/// The coordinates of the vertices of the mesh.
#[derive(Debug, Clone)]
pub struct MeshCoords {
  matrix: Matrix,
}

impl MeshCoords {
  pub fn standard(ndim: Dim) -> Self {
    SimplexCoords::standard(ndim).vertices
  }
  pub fn new(matrix: Matrix) -> Self {
    Self { matrix }
  }

  pub fn matrix(&self) -> &Matrix {
    &self.matrix
  }
  pub fn matrix_mut(&mut self) -> &mut Matrix {
    &mut self.matrix
  }
  pub fn into_matrix(self) -> Matrix {
    self.matrix
  }

  pub fn swap_coords(&mut self, icol: usize, jcol: usize) {
    self.matrix.swap_columns(icol, jcol)
  }
}

impl From<Matrix> for MeshCoords {
  fn from(matrix: Matrix) -> Self {
    Self::new(matrix)
  }
}

impl From<&[Vector]> for MeshCoords {
  fn from(vectors: &[Vector]) -> Self {
    let matrix = Matrix::from_columns(vectors);
    Self::new(matrix)
  }
}

impl MeshCoords {
  pub fn dim(&self) -> Dim {
    self.matrix.nrows()
  }
  pub fn nvertices(&self) -> usize {
    self.matrix.ncols()
  }

  pub fn coord(&self, ivertex: VertexIdx) -> Vector {
    self.matrix.column(ivertex).into_owned()
  }

  pub fn coord_iter(&self) -> impl Iterator<Item = Vector> + '_ {
    self.matrix.column_iter().map(|c| c.into_owned())
  }

  pub fn coord_iter_mut(&mut self) -> impl Iterator<Item = Vector> + '_ {
    // Note: we can't directly return mutable column iterators, so we'll work with the matrix
    (0..self.nvertices()).map(|i| {
      self.matrix.column(i).into_owned()
    })
  }

  pub fn to_edge_lengths(&self, topology: &Complex) -> MeshLengths {
    let edges = topology.edges();
    let mut edge_lengths = Vector::zeros(edges.len());
    for (iedge, edge) in edges.handle_iter().enumerate() {
      let edge_vec: Vec<usize> = edge.iter().collect();
      let [vi, vj] = [edge_vec[0], edge_vec[1]];
      let length = (self.coord(vj) - &self.coord(vi)).norm();
      edge_lengths[iedge] = length;
    }
    // SAFETY: Edge Lengths come from a coordinate realization.
    MeshLengths::new_unchecked(edge_lengths)
  }
}

impl MeshCoords {
  pub fn embed_euclidean(mut self, dim: Dim) -> MeshCoords {
    let old_dim = self.matrix.nrows();
    self.matrix = self.matrix.insert_rows(old_dim, dim - old_dim, 0.0);
    self
  }
}

impl MeshCoords {
  pub fn find_cell_containing(
    &self,
    topology: &Complex,
    coord: &Vector,
  ) -> Option<SimplexCoords> {
    topology
      .cells()
      .handle_iter()
      .find(|cell| {
        let cell_coords = SimplexCoords::from_simplex_and_coords(&*cell, self);
        cell_coords.is_global_inside(coord)
      })
      .map(|cell| SimplexCoords::from_simplex_and_coords(&*cell, self))
  }
}

pub fn standard_coord_complex(dim: Dim) -> (Complex, MeshCoords) {
  let topology = Complex::standard(dim);

  let coords: Vec<Vector> = topology
    .vertices()
    .handle_iter()
    .map(|v| v.kidx())
    .map(|v| {
      let mut vec = Vector::zeros(dim);
      if v > 0 {
        vec[v - 1] = 1.0;
      }
      vec
    })
    .collect();
  let coords = Matrix::from_columns(&coords);
  let coords = MeshCoords::new(coords);

  (topology, coords)
}
