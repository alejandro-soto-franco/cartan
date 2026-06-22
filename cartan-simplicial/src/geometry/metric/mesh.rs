// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::{simplex::SimplexLengths, EdgeIdx};
use crate::{
    topology::{
        complex::Complex,
        handle::{SimplexHandle, SkeletonHandle},
    },
    linalg::Vector,
};

use itertools::Itertools;
use rayon::prelude::ParallelIterator;

/// The lengths of the edges of the mesh.
#[derive(Debug, Clone)]
pub struct MeshLengths {
    vector: Vector,
}

impl MeshLengths {
    pub fn new(vector: Vector, complex: &Complex) -> Self {
        let this = Self { vector };
        assert!(
            this.is_coordinate_realizable(complex.cells()),
            "Edge lengths are not coordinate realizable."
        );
        this
    }

    pub fn try_new(vector: Vector, complex: &Complex) -> Option<Self> {
        let this = Self { vector };
        this.is_coordinate_realizable(complex.cells())
            .then_some(this)
    }

    pub fn new_unchecked(vector: Vector) -> Self {
        Self { vector }
    }

    pub fn standard(dim: crate::Dim) -> MeshLengths {
        let vector = SimplexLengths::standard(dim).vector().clone();
        Self::new_unchecked(vector)
    }

    pub fn nedges(&self) -> usize {
        self.vector.len()
    }

    pub fn length(&self, iedge: EdgeIdx) -> f64 {
        self[iedge]
    }

    pub fn vector(&self) -> &Vector {
        &self.vector
    }

    pub fn vector_mut(&mut self) -> &mut Vector {
        &mut self.vector
    }

    pub fn into_vector(self) -> Vector {
        self.vector
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
        self.vector.iter()
    }

    /// The mesh width h_max, equal to the largest diameter of all cells.
    pub fn mesh_width_max(&self) -> f64 {
        self.iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap()
    }

    /// By convexity the smallest length of a line inside a simplex is the length
    /// of one of the edges.
    pub fn mesh_width_min(&self) -> f64 {
        self.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap()
    }

    /// The shape regularity measure rho of the whole mesh, which is the largest
    /// shape regularity measure over all cells.
    pub fn shape_regularity_measure(&self, topology: &Complex) -> f64 {
        topology
            .cells()
            .handle_iter()
            .map(|cell| self.simplex_lengths(cell).shape_reguarity_measure())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    pub fn simplex_lengths(&self, simplex: SimplexHandle) -> SimplexLengths {
        let lengths: Vec<f64> = simplex
            .mesh_edges()
            .map(|edge| self.length(edge.kidx()))
            .collect_vec();
        let vec = Vector::from_vec(lengths);
        SimplexLengths::new_unchecked(vec, simplex.dim())
    }

    pub fn is_coordinate_realizable(&self, skeleton: SkeletonHandle) -> bool {
        skeleton
            .handle_par_iter()
            .all(|simp| self.simplex_lengths(simp).is_coordinate_realizable())
    }
}

impl std::ops::Index<EdgeIdx> for MeshLengths {
    type Output = f64;
    fn index(&self, iedge: EdgeIdx) -> &Self::Output {
        &self.vector[iedge]
    }
}

pub type MetricComplex = (Complex, MeshLengths);

pub fn standard_metric_complex(dim: crate::Dim) -> MetricComplex {
    let topology = Complex::standard(dim);
    let lengths = MeshLengths::standard(dim);
    (topology, lengths)
}

#[cfg(test)]
mod cartan_tests {
    use super::*;

    #[test]
    fn standard_metric_complex_is_realizable_in_several_dims() {
        for dim in 1..=3 {
            let (complex, lengths) = standard_metric_complex(dim);
            assert!(lengths.is_coordinate_realizable(complex.cells()));
            // every top cell yields a non-degenerate SimplexLengths
            for cell in complex.cells().handle_iter() {
                let sl = lengths.simplex_lengths(cell);
                assert!(sl.vol() > 0.0, "dim={dim}");
            }
        }
    }
}
