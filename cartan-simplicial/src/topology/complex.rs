// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::handle::{SimplexIdx, SkeletonHandle};
use super::simplex::Simplex;
use super::skeleton::Skeleton;
use cartan_exterior::combo::Sign;
use cartan_exterior::Dim;
use itertools::Itertools;
use sprs::{CsMat, TriMat};

/// A simplicial manifold complex.
#[derive(Default, Debug, Clone)]
pub struct Complex {
  skeletons: Vec<ComplexSkeleton>,
}

/// A skeleton inside of a complex.
#[derive(Default, Debug, Clone)]
pub struct ComplexSkeleton {
  skeleton: Skeleton,
  complex_data: SkeletonComplexData,
}
impl ComplexSkeleton {
  pub fn skeleton(&self) -> &Skeleton {
    &self.skeleton
  }
  pub fn complex_data(&self) -> &[SimplexComplexData] {
    &self.complex_data
  }
}

pub type SkeletonComplexData = Vec<SimplexComplexData>;

#[derive(Default, Debug, Clone)]
pub struct SimplexComplexData {
  pub cocells: Vec<SimplexIdx>,
}

impl Complex {
  pub fn skeletons(&self) -> impl Iterator<Item = SkeletonHandle<'_>> {
    (0..=self.dim()).map(|d| SkeletonHandle::new(self, d))
  }
  pub fn skeleton(&self, dim: Dim) -> SkeletonHandle<'_> {
    SkeletonHandle::new(self, dim)
  }
  pub fn mesh_skeleton_raw(&self, dim: Dim) -> &ComplexSkeleton {
    &self.skeletons[dim]
  }
  pub fn nsimplices(&self, dim: Dim) -> usize {
    self.skeleton(dim).len()
  }
  pub fn vertices(&self) -> SkeletonHandle<'_> {
    self.skeleton(0)
  }
  pub fn edges(&self) -> SkeletonHandle<'_> {
    self.skeleton(1)
  }
  pub fn facets(&self) -> SkeletonHandle<'_> {
    self.skeleton(self.dim() - 1)
  }
  pub fn cells(&self) -> SkeletonHandle<'_> {
    self.skeleton(self.dim())
  }
}

impl Complex {
  pub fn standard(dim: Dim) -> Self {
    Self::from_cells(Skeleton::standard(dim))
  }
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }

  pub fn has_boundary(&self) -> bool {
    !self.boundary_facets().is_empty()
  }

  /// For a d-mesh computes the boundary, which consists of facets ((d-1)-subs).
  ///
  /// The boundary facets are characterized by the fact that they
  /// only have 1 cell as super entity.
  pub fn boundary_facets(&self) -> Vec<SimplexIdx> {
    self
      .facets()
      .handle_iter()
      .filter(|f| f.cocells().count() == 1)
      .map(|f| f.idx())
      .collect()
  }

  pub fn boundary_cells(&self) -> Vec<SimplexIdx> {
    self
      .boundary_facets()
      .into_iter()
      // the boundary has only one parent cell by definition
      .map(|facet| {
        facet
          .handle(self)
          .cocells()
          .next()
          .expect("Boundary facets have exactly one cell.")
          .idx()
      })
      .unique()
      .collect()
  }

  /// The vertices that lie on the boundary of the mesh.
  /// No particular order of vertices.
  pub fn boundary_vertices(&self) -> Vec<usize> {
    self
      .boundary_facets()
      .into_iter()
      .flat_map(|facet| {
        facet
          .handle(self)
          .iter()
          .collect::<Vec<_>>()
      })
      .unique()
      .collect()
  }

  /// Signed boundary operator d_dim mapping `dim`-chains to `(dim-1)`-chains,
  /// as a sparse CSC matrix of shape `nsimplices(dim-1) x nsimplices(dim)`.
  /// Entry `(i, j)` is the incidence sign of the i-th `(dim-1)`-simplex in the
  /// boundary of the j-th `dim`-simplex.
  pub fn boundary_matrix(&self, dim: Dim) -> CsMat<f64> {
    assert!(dim >= 1 && dim <= self.dim());
    let n_lo = self.nsimplices(dim - 1);
    let n_hi = self.nsimplices(dim);
    let lo = self.skeleton(dim - 1);
    let mut tri = TriMat::new((n_lo, n_hi));
    for hi in self.skeleton(dim).handle_iter() {
      let j = hi.kidx();
      for (sign, facet) in hi.boundary_chain() {
        let facet_simp = (*facet).clone();
        let i = lo.kidx_by_simplex(&facet_simp.sorted());
        let sign_val = match sign {
          Sign::Pos => 1.0,
          Sign::Neg => -1.0,
        };
        tri.add_triplet(i, j, sign_val);
      }
    }
    tri.to_csc()
  }

  /// The full chain of signed boundary operators.
  /// `boundary_chain()[k]` is `boundary_matrix(k+1)`: shape
  /// `nsimplices(k) x nsimplices(k+1)`, for k = 0..self.dim().
  pub fn boundary_chain(&self) -> Vec<CsMat<f64>> {
    (1..=self.dim()).map(|d| self.boundary_matrix(d)).collect()
  }
}

impl Complex {
  pub fn from_cells(cells: Skeleton) -> Self {
    let dim = cells.dim();

    let mut skeletons = vec![ComplexSkeleton::default(); dim + 1];
    skeletons[0] = ComplexSkeleton {
      skeleton: Skeleton::new((0..cells.nvertices()).map(Simplex::single).collect()),
      complex_data: (0..cells.nvertices())
        .map(|_| SimplexComplexData::default())
        .collect(),
    };

    for (icell, cell) in cells.iter().enumerate() {
      for (
        dim_skeleton,
        ComplexSkeleton {
          skeleton,
          complex_data: mesh_data,
        },
      ) in skeletons.iter_mut().enumerate()
      {
        for sub in cell.subsequences(dim_skeleton) {
          let (sub_idx, is_new) = skeleton.insert(sub);
          let sub_data = if is_new {
            mesh_data.push(SimplexComplexData::default());
            mesh_data.last_mut().unwrap()
          } else {
            &mut mesh_data[sub_idx]
          };
          sub_data.cocells.push(SimplexIdx::new(dim, icell));
        }
      }
    }

    // Topology checks.
    if dim >= 1 {
      let facet_data = skeletons[dim - 1].complex_data();
      for SimplexComplexData { cocells } in facet_data {
        let nparents = cocells.len();
        let is_manifold = nparents == 2 || nparents == 1;
        assert!(is_manifold, "Topology must be manifold.");
      }
    }

    Self { skeletons }
  }
}

#[cfg(test)]
mod cartan_tests {
  use super::*;
  use crate::topology::simplex::Simplex;
  use crate::topology::skeleton::Skeleton;

  fn single_tetrahedron() -> Complex {
    // One 3-simplex on vertices 0,1,2,3.
    Complex::from_cells(Skeleton::new(vec![Simplex::new(vec![0, 1, 2, 3])]))
  }

  #[test]
  fn boundary_of_boundary_is_zero() {
    let cx = single_tetrahedron();
    let chain = cx.boundary_chain();
    assert_eq!(chain.len(), 3); // d1, d2, d3
    // d_k * d_{k+1} = 0 for each consecutive pair.
    for k in 0..chain.len() - 1 {
      let prod = &chain[k] * &chain[k + 1];
      let max = prod.data().iter().fold(0.0f64, |m, &v| m.max(v.abs()));
      assert!(max < 1e-12, "d{k} * d{} has nonzero entry {max:e}", k + 1);
    }
  }

  #[test]
  fn simplex_counts_of_a_tetrahedron() {
    let cx = single_tetrahedron();
    assert_eq!(cx.dim(), 3);
    assert_eq!(cx.nsimplices(0), 4); // vertices
    assert_eq!(cx.nsimplices(1), 6); // edges
    assert_eq!(cx.nsimplices(2), 4); // faces
    assert_eq!(cx.nsimplices(3), 1); // the cell
  }
}
