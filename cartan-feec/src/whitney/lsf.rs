// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use crate::whitney::CoordSimplexExt;

use cartan_exterior::{
    combo::{factorial, factorialf, Sign},
    field::ExteriorField,
    ExteriorGrade, MultiForm,
};
use cartan_simplicial::{
    geometry::coord::simplex::SimplexCoords,
    topology::simplex::Simplex,
    Dim,
};

#[derive(Debug, Clone)]
pub struct WhitneyLsf {
    cell_coords: SimplexCoords,
    dof_simp: Simplex,
}

impl WhitneyLsf {
    pub fn from_coords(cell_coords: SimplexCoords, dof_simp: Simplex) -> Self {
        Self {
            cell_coords,
            dof_simp,
        }
    }

    pub fn standard(cell_dim: Dim, dof_simp: Simplex) -> Self {
        Self::from_coords(SimplexCoords::standard(cell_dim), dof_simp)
    }

    pub fn grade(&self) -> ExteriorGrade {
        self.dof_simp.dim()
    }

    /// The difbarys of the vertices of the DOF simplex.
    pub fn difbarys(&self) -> impl Iterator<Item = MultiForm> + use<'_> {
        self.cell_coords
            .difbarys_ext()
            .into_iter()
            .enumerate()
            .filter_map(|(ibary, difbary)| self.dof_simp.contains(ibary).then_some(difbary))
    }

    /// d-lambda_i0 wedge ... (omit d-lambda_i_iterm) ... wedge d-lambda_i_dim
    pub fn wedge_term(&self, iterm: usize) -> MultiForm {
        let dim_cell = self.cell_coords.dim_intrinsic();
        let wedge = self
            .difbarys()
            .enumerate()
            .filter_map(|(pos, difbary)| (pos != iterm).then_some(difbary));
        MultiForm::wedge_big(wedge).unwrap_or(MultiForm::one(dim_cell))
    }

    pub fn wedge_terms(&self) -> impl ExactSizeIterator<Item = MultiForm> + use<'_> {
        (0..self.dof_simp.nvertices()).map(move |iwedge| self.wedge_term(iwedge))
    }

    /// The constant exterior derivative of the Whitney LSF.
    pub fn dif(&self) -> MultiForm {
        let dim = self.cell_coords.dim_intrinsic();
        let grade = self.grade();
        if grade == dim {
            return MultiForm::zero(dim, grade + 1);
        }
        factorialf(grade + 1) * MultiForm::wedge_big(self.difbarys()).unwrap()
    }
}

impl ExteriorField for WhitneyLsf {
    fn dim_ambient(&self) -> cartan_exterior::Dim {
        self.cell_coords.dim_ambient()
    }
    fn dim_intrinsic(&self) -> cartan_exterior::Dim {
        self.cell_coords.dim_intrinsic()
    }
    fn grade(&self) -> ExteriorGrade {
        self.grade()
    }
    fn at_point<'a>(&self, coord: impl Into<nalgebra::DVectorView<'a, f64>>) -> MultiForm {
        let coord_view: nalgebra::DVectorView<f64> = coord.into();
        let coord_owned = coord_view.clone_owned();
        let barys = self.cell_coords.global2bary(&coord_owned);

        let dim = self.dim_ambient();
        let grade = self.grade();
        let mut form = MultiForm::zero(dim, grade);
        for (iterm, &vertex) in self.dof_simp.vertices.iter().enumerate() {
            let sign = Sign::from_parity(iterm);
            let wedge = self.wedge_term(iterm);
            let bary = barys[vertex];
            form += sign.as_f64() * bary * wedge;
        }
        (factorial(grade) as f64) * form
    }
}

pub struct WhitneyPushforwardLsf {
    pub cell_coords: SimplexCoords,
    pub ref_lsf: WhitneyLsf,
}

impl WhitneyPushforwardLsf {
    pub fn new(cell_coords: SimplexCoords, dof_simp: Simplex) -> Self {
        let ref_lsf = WhitneyLsf::standard(cell_coords.dim_intrinsic(), dof_simp);
        Self {
            cell_coords,
            ref_lsf,
        }
    }
}

impl ExteriorField for WhitneyPushforwardLsf {
    fn dim_ambient(&self) -> cartan_exterior::Dim {
        self.cell_coords.dim_ambient()
    }
    fn dim_intrinsic(&self) -> cartan_exterior::Dim {
        self.ref_lsf.dim_intrinsic()
    }
    fn grade(&self) -> ExteriorGrade {
        self.ref_lsf.grade()
    }
    fn at_point<'a>(&self, coord_global: impl Into<nalgebra::DVectorView<'a, f64>>) -> MultiForm {
        let coord_global_view: nalgebra::DVectorView<f64> = coord_global.into();
        let coord_global_owned = coord_global_view.clone_owned();
        let coord_ref = self.cell_coords.global2local(&coord_global_owned);
        let value_ref = self.ref_lsf.at_point(coord_ref.as_view());
        value_ref.precompose_form(&self.cell_coords.linear_transform())
    }
}
