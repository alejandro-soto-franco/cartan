// Ported from luiswirth/formoniq (used with permission), adapted for cartan.

use super::lsf::WhitneyLsf;
use crate::cochain::Cochain;

use cartan_exterior::{field::ExteriorField, ExteriorElement, MultiForm};
use cartan_simplicial::{
    geometry::coord::mesh::MeshCoords,
    topology::{complex::Complex, handle::SimplexHandle},
    Dim,
};

pub struct WhitneyForm<'a> {
    cochain: Cochain,
    complex: &'a Complex,
    mesh_coords: &'a MeshCoords,
}

impl<'a> WhitneyForm<'a> {
    pub fn new(cochain: Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
        Self {
            cochain,
            complex,
            mesh_coords,
        }
    }

    pub fn dif(&self) -> Self {
        Self {
            cochain: self.cochain.dif(self.complex),
            complex: self.complex,
            mesh_coords: self.mesh_coords,
        }
    }

    /// Evaluate the Whitney form at a known cell, avoiding the slow cell search.
    pub fn eval_known_cell<'b>(&self, cell: SimplexHandle<'b>, coord: &nalgebra::DVector<f64>) -> ExteriorElement {
        
        use cartan_simplicial::geometry::coord::simplex::SimplexCoords;

        let cell_coords = SimplexCoords::from_simplex_and_coords(&cell, self.mesh_coords);

        let mut value = MultiForm::zero(self.dim_ambient(), self.grade());
        for dof_simp in cell.mesh_subsimps(self.grade()) {
            let local_dof_simp = (*dof_simp).relative_to(&cell);

            let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
            let lsf_value = lsf.at_point(coord.as_view());
            let dof_value = self.cochain[dof_simp.kidx()];
            value += dof_value * lsf_value;
        }
        value
    }
}

impl ExteriorField for WhitneyForm<'_> {
    fn dim_ambient(&self) -> Dim {
        self.mesh_coords.dim()
    }
    fn dim_intrinsic(&self) -> Dim {
        self.complex.dim()
    }
    fn grade(&self) -> cartan_exterior::ExteriorGrade {
        self.cochain.dim()
    }
    /// Evaluate at a global coordinate (slow: searches for the containing cell).
    fn at_point<'a>(&self, coord: impl Into<nalgebra::DVectorView<'a, f64>>) -> ExteriorElement {
        
        use cartan_simplicial::geometry::coord::simplex::SimplexCoords;

        let coord_view: nalgebra::DVectorView<f64> = coord.into();
        let coord_owned = coord_view.clone_owned();

        let cell_coords = self
            .mesh_coords
            .find_cell_containing(self.complex, &coord_owned)
            .unwrap();

        // Find which cell it is
        let cell = self.complex.cells()
            .handle_iter()
            .find(|c| {
                let cc = SimplexCoords::from_simplex_and_coords(c, self.mesh_coords);
                cc.is_global_inside(&coord_owned)
            })
            .unwrap();

        let mut value = MultiForm::zero(self.dim_intrinsic(), self.grade());
        for dof_simp in cell.mesh_subsimps(self.grade()) {
            let local_dof_simp = (*dof_simp).relative_to(&cell);

            let lsf = WhitneyLsf::from_coords(cell_coords.clone(), local_dof_simp);
            let lsf_value = lsf.at_point(coord_owned.as_view());
            let dof_value = self.cochain[dof_simp.kidx()];
            value += dof_value * lsf_value;
        }
        value
    }
}

pub struct DifWhitneyForm<'a> {
    cochain: &'a Cochain,
    complex: &'a Complex,
    mesh_coords: &'a MeshCoords,
}

impl<'a> DifWhitneyForm<'a> {
    pub fn new(cochain: &'a Cochain, complex: &'a Complex, mesh_coords: &'a MeshCoords) -> Self {
        Self {
            cochain,
            complex,
            mesh_coords,
        }
    }
}

impl ExteriorField for DifWhitneyForm<'_> {
    fn dim_ambient(&self) -> Dim {
        self.mesh_coords.dim()
    }
    fn dim_intrinsic(&self) -> Dim {
        self.complex.dim()
    }
    fn grade(&self) -> cartan_exterior::ExteriorGrade {
        self.cochain.dim()
    }
    fn at_point<'a>(&self, _coord: impl Into<nalgebra::DVectorView<'a, f64>>) -> ExteriorElement {
        unimplemented!("DifWhitneyForm::at_point not yet implemented")
    }
}
