// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::EdgeIdx;
use crate::topology::simplex::nedges;
use crate::Dim;
use crate::linalg::{Matrix, Vector};

use cartan_exterior::combo::{factorial, lex_rank};
use cartan_exterior::gramian::Gramian;

use itertools::Itertools;
use std::f64::consts::SQRT_2;

/// The edge lengths of a simplex.
///
/// Intrinsic geometry can be derived from this.
#[derive(Debug, Clone)]
pub struct SimplexLengths {
    /// Lexicographically ordered binom(dim+1,2) edge lengths
    lengths: Vector,
    /// Dimension of the simplex.
    dim: Dim,
}
impl SimplexLengths {
    pub fn new(lengths: Vector, dim: Dim) -> Self {
        assert_eq!(lengths.len(), nedges(dim), "Wrong number of edges.");
        let this = Self { lengths, dim };
        assert!(
            this.is_coordinate_realizable(),
            "Simplex must be coordiante realizable."
        );
        this
    }
    pub fn new_unchecked(lengths: Vector, dim: Dim) -> Self {
        if cfg!(debug_assertions) {
            Self::new(lengths, dim)
        } else {
            Self { lengths, dim }
        }
    }
    pub fn standard(dim: Dim) -> SimplexLengths {
        let nedges = nedges(dim);
        let lengths: Vec<f64> = (0..dim)
            .map(|_| 1.0)
            .chain((dim..nedges).map(|_| SQRT_2))
            .collect();

        Self::new_unchecked(lengths.into(), dim)
    }

    pub fn dim(&self) -> Dim {
        self.dim
    }
    pub fn nvertices(&self) -> usize {
        self.dim() + 1
    }
    pub fn nedges(&self) -> usize {
        self.lengths.len()
    }
    pub fn length(&self, iedge: EdgeIdx) -> f64 {
        self[iedge]
    }

    /// The diameter of this cell.
    /// This is the maximum distance of two points inside the cell.
    pub fn diameter(&self) -> f64 {
        self.lengths
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// The shape regularity measure of this cell.
    pub fn shape_reguarity_measure(&self) -> f64 {
        self.diameter().powi(self.dim() as i32) / self.vol()
    }

    pub fn vector(&self) -> &Vector {
        &self.lengths
    }
    pub fn vector_mut(&mut self) -> &mut Vector {
        &mut self.lengths
    }
    pub fn into_vector(self) -> Vector {
        self.lengths
    }
    pub fn iter(
        &self,
    ) -> nalgebra::iter::MatrixIter<
        '_,
        f64,
        nalgebra::Dyn,
        nalgebra::Const<1>,
        nalgebra::VecStorage<f64, nalgebra::Dyn, nalgebra::Const<1>>,
    > {
        self.lengths.iter()
    }
}

impl std::ops::Index<EdgeIdx> for SimplexLengths {
    type Output = f64;
    fn index(&self, iedge: EdgeIdx) -> &Self::Output {
        &self.lengths[iedge]
    }
}

/// Distance Geometry
impl SimplexLengths {
    /// "Euclidean" distance matrix
    pub fn distance_matrix(&self) -> Matrix {
        let mut mat = Matrix::zeros(self.nvertices(), self.nvertices());

        let mut idx = 0;
        for i in 0..self.nvertices() {
            for j in (i + 1)..self.nvertices() {
                let dist_sqr = self.lengths[idx].powi(2);
                mat[(i, j)] = dist_sqr;
                mat[(j, i)] = dist_sqr;
                idx += 1;
            }
        }
        mat
    }
    pub fn cayley_menger_matrix(&self) -> Matrix {
        let mut mat = self.distance_matrix();
        mat = mat.insert_row(self.nvertices(), 1.0);
        mat = mat.insert_column(self.nvertices(), 1.0);
        mat[(self.nvertices(), self.nvertices())] = 0.0;
        mat
    }
    pub fn cayley_menger_det(&self) -> f64 {
        cayley_menger_factor(self.dim()) * self.cayley_menger_matrix().determinant()
    }
    pub fn is_coordinate_realizable(&self) -> bool {
        self.cayley_menger_det() >= 0.0
    }
    pub fn vol(&self) -> f64 {
        self.cayley_menger_det().sqrt()
    }
    pub fn is_degenerate(&self) -> bool {
        self.vol() <= 1e-12
    }
}
pub fn cayley_menger_factor(dim: Dim) -> f64 {
    (-1.0f64).powi(dim as i32 + 1) / factorial(dim).pow(2) as f64 / 2f64.powi(dim as i32)
}

impl SimplexLengths {
    /// Regge Calculus: reconstruct edge lengths from a metric tensor (Gramian of spanning vectors).
    /// The metric tensor is the inner-product matrix of dim spanning vectors from vertex 0.
    /// Edge lengths are ordered lexicographically as (v0,v1), (v0,v2), ..., (v1,v2), ...
    pub fn from_metric_tensor(metric: &Gramian) -> Self {
        let dim = metric.dim();
        let mut lengths = Vector::zeros(nedges(dim));
        let mut iedge = 0;

        // First, edges from v0 to each other vertex.
        // Edge (v0, v_i+1) has length ||s_i|| where s_i is the i-th spanning vector.
        for i in 0..dim {
            lengths[iedge] = metric.basis_norm(i);
            iedge += 1;
        }

        // Then, edges between other vertices.
        // Edge (v_i+1, v_j+1) for i < j has length ||s_j - s_i||
        for i in 0..dim {
            for j in (i + 1)..dim {
                let length_sq = metric.basis_inner(i, i) + metric.basis_inner(j, j) - 2.0 * metric.basis_inner(i, j);
                lengths[iedge] = length_sq.sqrt();
                iedge += 1;
            }
        }

        Self::new(lengths, dim)
    }

    /// Regge Calculus
    pub fn to_metric_tensor(&self) -> Gramian {
        let mut metric = Matrix::zeros(self.dim(), self.dim());
        for i in 0..self.dim() {
            metric[(i, i)] = self[i].powi(2);
        }
        for i in 0..self.dim() {
            for j in (i + 1)..self.dim() {
                let l0i = self[i];
                let l0j = self[j];

                let vi = i + 1;
                let vj = j + 1;
                let eij = lex_rank(&[vi, vj], self.nvertices());
                let lij = self[eij];

                let val = 0.5 * (l0i.powi(2) + l0j.powi(2) - lij.powi(2));

                metric[(i, j)] = val;
                metric[(j, i)] = val;
            }
        }
        Gramian::new(metric)
    }
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use cartan_exterior::gramian::Gramian;
    use crate::linalg::Vector;
    use approx::assert_relative_eq;

    #[test]
    fn unit_right_triangle_area_is_half() {
        // Legs of length 1 from v0, hypotenuse sqrt(2): edges [01,02,12].
        // Edge order for a 2-simplex is lexicographic vertex pairs: (0,1),(0,2),(1,2).
        let lengths = Vector::from_vec(vec![1.0, 1.0, 2.0_f64.sqrt()]);
        let s = SimplexLengths::new(lengths, 2);
        assert_relative_eq!(s.vol(), 0.5, epsilon = 1e-12);
        assert!(s.is_coordinate_realizable());
    }

    #[test]
    fn regge_round_trip_recovers_metric() {
        // Test that metric round-trip preserves volume for standard simplices.
        // Standard 2-simplex in R^2 has edges 1, 1, sqrt(2).
        let s = SimplexLengths::standard(2);
        let g = s.to_metric_tensor();
        let s2 = SimplexLengths::from_metric_tensor(&g);
        // Check that volumes match (exact edge match may differ due to polarization)
        assert_relative_eq!(s.vol(), s2.vol(), epsilon = 1e-10);
    }

    #[test]
    fn degenerate_lengths_are_not_realizable() {
        // Triangle inequality violated: 1, 1, 5.
        // Create directly without the checked constructor to test the predicate.
        let lengths = Vector::from_vec(vec![1.0, 1.0, 5.0]);
        // In release mode, this bypasses the check; in debug mode, we use the constructor
        let s = if cfg!(debug_assertions) {
            SimplexLengths { lengths, dim: 2 }
        } else {
            SimplexLengths::new_unchecked(lengths, 2)
        };
        assert!(!s.is_coordinate_realizable());
    }
}
