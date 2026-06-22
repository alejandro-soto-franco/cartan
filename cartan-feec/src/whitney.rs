// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

pub mod lsf;
pub mod form;

use cartan_exterior::{ExteriorGrade, MultiForm, MultiVector};
use cartan_simplicial::{
    geometry::coord::simplex::SimplexCoords,
    topology::complex::Complex,
};
use sprs::{CsMat, TriMat};

/// Extension trait for Complex: produces the exterior derivative operator
/// as a sparse matrix (coboundary = transpose of boundary).
/// d^k : W Lambda^k -> W Lambda^(k+1)
pub trait ManifoldComplexExt {
    fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> CsMat<f64>;
}

impl ManifoldComplexExt for Complex {
    fn exterior_derivative_operator(&self, grade: ExteriorGrade) -> CsMat<f64> {
        // boundary_matrix(grade+1) has shape nsimplices(grade) x nsimplices(grade+1)
        // Exterior derivative = transpose of boundary operator
        let b = self.boundary_matrix(grade + 1);
        b.transpose_into()
    }
}

/// Extension trait for SimplexCoords: produces difbarys and spanning multivector.
pub trait CoordSimplexExt {
    fn difbarys_ext(&self) -> Vec<MultiForm>;
    fn spanning_multivector(&self) -> MultiVector;
}

impl CoordSimplexExt for SimplexCoords {
    fn spanning_multivector(&self) -> MultiVector {
        let vectors = self.spanning_vectors();
        let vectors = vectors
            .column_iter()
            .map(|v| MultiVector::line(v.into_owned()));
        MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_ambient()))
    }

    fn difbarys_ext(&self) -> Vec<MultiForm> {
        self.difbarys()
            .row_iter()
            .map(|difbary| MultiForm::line(difbary.transpose()))
            .collect()
    }
}
