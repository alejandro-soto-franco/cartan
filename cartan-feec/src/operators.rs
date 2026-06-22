// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::whitney::{lsf::WhitneyLsf, ManifoldComplexExt};

use cartan_exterior::{
    combo::{factorial, Sign},
    field::ExteriorField,
    list::ExteriorElementList,
    multi_gramian::multi_gramian,
    Dim, ExteriorGrade,
};
use cartan_simplicial::{
    geometry::{
        coord::{
            mesh::MeshCoords,
            quadrature::SimplexQuadRule,
            simplex::SimplexCoords,
            CoordRef,
        },
        metric::simplex::SimplexLengths,
    },
    topology::{
        complex::Complex,
        simplex::{standard_subsimps, Simplex},
    },
};

use sprs::CsMat;

pub type DofIdx = usize;
pub type DofCoeff = f64;

pub type ElMat = nalgebra::DMatrix<f64>;
pub type ElVec = nalgebra::DVector<f64>;

pub trait ElMatProvider: Sync {
    fn row_grade(&self) -> ExteriorGrade;
    fn col_grade(&self) -> ExteriorGrade;
    fn eval(&self, geometry: &SimplexLengths) -> ElMat;
}

pub trait ElVecProvider: Sync {
    fn grade(&self) -> ExteriorGrade;
    fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec;
}

/// Convert a sparse CSC matrix to a dense DMatrix.
fn csc_to_dense(m: &CsMat<f64>) -> nalgebra::DMatrix<f64> {
    let mut d = nalgebra::DMatrix::zeros(m.rows(), m.cols());
    for (&val, (r, c)) in m.iter() {
        d[(r, c)] += val;
    }
    d
}

/// Exact Element Matrix Provider for the Laplace-Beltrami operator.
pub struct LaplaceBeltramiElmat {
    dim: Dim,
    ref_difbarys: nalgebra::DMatrix<f64>,
}
impl LaplaceBeltramiElmat {
    pub fn new(dim: Dim) -> Self {
        let ref_difbarys = SimplexCoords::standard(dim).difbarys().transpose();
        Self { dim, ref_difbarys }
    }
}
impl ElMatProvider for LaplaceBeltramiElmat {
    fn row_grade(&self) -> ExteriorGrade {
        0
    }
    fn col_grade(&self) -> ExteriorGrade {
        0
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        assert!(self.dim == geometry.dim());
        geometry.vol()
            * geometry
                .to_metric_tensor()
                .inverse()
                .norm_sq_mat(&self.ref_difbarys)
    }
}

/// Exact Element Matrix Provider for scalar mass bilinear form.
pub struct ScalarMassElmat;
impl ElMatProvider for ScalarMassElmat {
    fn row_grade(&self) -> ExteriorGrade {
        0
    }
    fn col_grade(&self) -> ExteriorGrade {
        0
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        let ndofs = geometry.nvertices();
        let dim = geometry.dim();
        let v = geometry.vol() / ((dim + 1) * (dim + 2)) as f64;
        let mut elmat = nalgebra::DMatrix::from_element(ndofs, ndofs, v);
        elmat.fill_diagonal(2.0 * v);
        elmat
    }
}

/// Approximated Element Matrix Provider for scalar mass bilinear form
/// obtained through trapezoidal quadrature rule.
pub struct ScalarLumpedMassElmat;
impl ElMatProvider for ScalarLumpedMassElmat {
    fn row_grade(&self) -> ExteriorGrade {
        0
    }
    fn col_grade(&self) -> ExteriorGrade {
        0
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        let n = geometry.nvertices();
        let v = geometry.vol() / n as f64;
        nalgebra::DMatrix::from_diagonal_element(n, n, v)
    }
}

/// Element Matrix for the weak Hodge star operator / the mass bilinear form.
pub struct HodgeMassElmat {
    dim: Dim,
    grade: ExteriorGrade,
    simplices: Vec<Simplex>,
    wedge_terms: Vec<ExteriorElementList>,
}
impl HodgeMassElmat {
    pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
        let simplices: Vec<_> = standard_subsimps(dim, grade);
        let wedge_terms: Vec<ExteriorElementList> = simplices
            .iter()
            .cloned()
            .map(|simp| WhitneyLsf::standard(dim, simp).wedge_terms().collect())
            .collect();

        Self {
            dim,
            grade,
            simplices,
            wedge_terms,
        }
    }
}
impl ElMatProvider for HodgeMassElmat {
    fn row_grade(&self) -> ExteriorGrade {
        self.grade
    }
    fn col_grade(&self) -> ExteriorGrade {
        self.grade
    }

    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        assert_eq!(self.dim, geometry.dim());

        let scalar_mass = ScalarMassElmat.eval(geometry);

        let mut elmat = nalgebra::DMatrix::zeros(self.simplices.len(), self.simplices.len());
        for (i, _asimp) in self.simplices.iter().enumerate() {
            for (j, _bsimp) in self.simplices.iter().enumerate() {
                let wedge_terms_a = &self.wedge_terms[i];
                let wedge_terms_b = &self.wedge_terms[j];
                let wedge_inners =
                    multi_gramian(&geometry.to_metric_tensor().inverse(), self.grade)
                        .inner_mat(wedge_terms_a.coeffs(), wedge_terms_b.coeffs());

                let nvertices = self.grade + 1;
                let mut sum = 0.0;
                for avertex in 0..nvertices {
                    for bvertex in 0..nvertices {
                        let sign = Sign::from_parity(avertex + bvertex);
                        let inner = wedge_inners[(avertex, bvertex)];
                        sum += sign.as_f64()
                            * inner
                            * scalar_mass[(self.simplices[i][avertex], self.simplices[j][bvertex])];
                    }
                }

                elmat[(i, j)] = sum;
            }
        }

        factorial(self.grade).pow(2) as f64 * elmat
    }
}

/// Element Matrix Provider for the weak mixed exterior derivative (dif sigma, v).
pub struct DifElmat {
    mass: HodgeMassElmat,
    dif: nalgebra::DMatrix<f64>,
}
impl DifElmat {
    pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
        let mass = HodgeMassElmat::new(dim, grade);
        let dif_sparse = Complex::standard(dim).exterior_derivative_operator(grade - 1);
        let dif = csc_to_dense(&dif_sparse);
        Self { mass, dif }
    }
}
impl ElMatProvider for DifElmat {
    fn row_grade(&self) -> ExteriorGrade {
        self.mass.grade
    }
    fn col_grade(&self) -> ExteriorGrade {
        self.mass.grade - 1
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        let mass = self.mass.eval(geometry);
        mass * &self.dif
    }
}

/// Element Matrix Provider for the weak mixed codifferential (u, dif tau).
pub struct CodifElmat {
    mass: HodgeMassElmat,
    codif: nalgebra::DMatrix<f64>,
}
impl CodifElmat {
    pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
        let mass = HodgeMassElmat::new(dim, grade);
        let dif_sparse = Complex::standard(dim).exterior_derivative_operator(grade - 1);
        let dif = csc_to_dense(&dif_sparse);
        let codif = dif.transpose();
        Self { mass, codif }
    }
}
impl ElMatProvider for CodifElmat {
    fn row_grade(&self) -> ExteriorGrade {
        self.mass.grade - 1
    }
    fn col_grade(&self) -> ExteriorGrade {
        self.mass.grade
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        let mass = self.mass.eval(geometry);
        &self.codif * mass
    }
}

/// Element Matrix Provider for the (dif u, dif v) bilinear form.
pub struct CodifDifElmat {
    mass: HodgeMassElmat,
    dif: nalgebra::DMatrix<f64>,
    codif: nalgebra::DMatrix<f64>,
}
impl CodifDifElmat {
    pub fn new(dim: Dim, grade: ExteriorGrade) -> Self {
        let mass = HodgeMassElmat::new(dim, grade + 1);
        let dif_sparse = Complex::standard(dim).exterior_derivative_operator(grade);
        let dif = csc_to_dense(&dif_sparse);
        let codif = dif.transpose();
        Self { mass, dif, codif }
    }
}
impl ElMatProvider for CodifDifElmat {
    fn row_grade(&self) -> ExteriorGrade {
        self.mass.grade - 1
    }
    fn col_grade(&self) -> ExteriorGrade {
        self.mass.grade - 1
    }
    fn eval(&self, geometry: &SimplexLengths) -> ElMat {
        let mass = self.mass.eval(geometry);
        &self.codif * mass * &self.dif
    }
}

pub struct SourceElVec<'a, F>
where
    F: ExteriorField,
{
    source: &'a F,
    mesh_coords: &'a MeshCoords,
    qr: SimplexQuadRule,
}
impl<'a, F> SourceElVec<'a, F>
where
    F: ExteriorField,
{
    pub fn new(source: &'a F, mesh_coords: &'a MeshCoords, qr: Option<SimplexQuadRule>) -> Self {
        let qr = qr.unwrap_or(SimplexQuadRule::barycentric(source.dim_intrinsic()));
        Self {
            source,
            mesh_coords,
            qr,
        }
    }
}
impl<F> ElVecProvider for SourceElVec<'_, F>
where
    F: Sync + ExteriorField,
{
    fn grade(&self) -> ExteriorGrade {
        self.source.grade()
    }
    fn eval(&self, geometry: &SimplexLengths, topology: &Simplex) -> ElVec {
        let cell_coords = SimplexCoords::from_simplex_and_coords(topology, self.mesh_coords);

        let dim = self.source.dim_intrinsic();
        let grade = self.grade();
        let dof_simps: Vec<_> = standard_subsimps(dim, grade);
        let whitneys: Vec<_> = dof_simps
            .iter()
            .cloned()
            .map(|dof_simp| WhitneyLsf::standard(dim, dof_simp))
            .collect();

        let inner = multi_gramian(&geometry.to_metric_tensor().inverse(), grade);

        let mut elvec = nalgebra::DVector::zeros(whitneys.len());
        for (iwhitney, whitney) in whitneys.iter().enumerate() {
            let inner_pointwise = |local: CoordRef| {
                let local_owned: nalgebra::DVector<f64> = local.clone_owned();
                let global = cell_coords.local2global(&local_owned);
                inner.inner(
                    self.source
                        .at_point(&global)
                        .precompose_form(&cell_coords.linear_transform())
                        .coeffs(),
                    whitney.at_point(local).coeffs(),
                )
            };
            let value = self.qr.integrate_local(&inner_pointwise, geometry.vol());
            elvec[iwhitney] = value;
        }
        elvec
    }
}

#[cfg(test)]
mod cartan_tests {
    use super::*;
    use cartan_simplicial::geometry::metric::simplex::SimplexLengths;
    use approx::assert_relative_eq;

    #[test]
    fn hodge_mass_grade0_equals_scalar_mass() {
        for dim in 1..=3 {
            let geo = SimplexLengths::standard(dim);
            let hodge = HodgeMassElmat::new(dim, 0).eval(&geo);
            let scalar = ScalarMassElmat.eval(&geo);
            assert_relative_eq!(
                (hodge - scalar).norm(), 0.0, epsilon = 1e-12
            );
        }
    }

    #[test]
    fn hodge_mass_dim2_grade1_reference() {
        // Hand-checked reference from formoniq operators.rs.
        let geo = SimplexLengths::standard(2);
        let m = HodgeMassElmat::new(2, 1).eval(&geo);
        let expected = nalgebra::DMatrix::from_row_slice(3, 3, &[
            1.0 / 3.0, 1.0 / 6.0, 0.0,
            1.0 / 6.0, 1.0 / 3.0, 0.0,
            0.0,       0.0,       1.0 / 6.0,
        ]);
        assert_relative_eq!((m - expected).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn codifdif_grade0_equals_laplace_beltrami() {
        for dim in 1..=3 {
            let geo = SimplexLengths::standard(dim);
            let codifdif = CodifDifElmat::new(dim, 0).eval(&geo);
            let laplace = LaplaceBeltramiElmat::new(dim).eval(&geo);
            assert_relative_eq!((codifdif - laplace).norm(), 0.0, epsilon = 1e-10);
        }
    }
}
