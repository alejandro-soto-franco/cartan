//! Full-field DEC homogenisation (β). Order2 only in v1.
//!
//! Pipeline: voxelise the RVE onto a structured tet mesh, tag each tet with its
//! phase's isotropic conductivity, assemble the P1-FEM stiffness and right-hand
//! side for each cell-problem direction, solve with PCG, volume-average to get
//! the effective tensor. Direct FEM assembly (not via cartan-dec::Operators)
//! because per-tet K weighting is cleaner expressed as element stiffness than
//! as a modified Hodge star.

use crate::{error::HomogError, rve::Rve, schemes::Effective, tensor::{Order2, TensorOrder}};

pub mod voxelize;
pub mod mesh;
pub mod cell_problem;
pub mod solver;

pub use mesh::{PeriodicCubeMeshBuilder, PeriodicCubeMeshBuilderOpts, partition_boundary};

use nalgebra::{DVector, Matrix3, Vector3};

#[derive(Clone, Debug)]
pub struct FullField<O: TensorOrder> {
    pub mesh_opts: PeriodicCubeMeshBuilderOpts,
    pub tol: f64,
    pub max_iter: usize,
    _marker: core::marker::PhantomData<O>,
}

impl<O: TensorOrder> Default for FullField<O> {
    fn default() -> Self {
        Self {
            mesh_opts: PeriodicCubeMeshBuilderOpts::default(),
            tol: 1e-8,
            max_iter: 5_000,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<O: TensorOrder> FullField<O> {
    pub fn new_with_resolution(resolution: usize) -> Self {
        Self {
            mesh_opts: PeriodicCubeMeshBuilderOpts { resolution, refine_depth: 0 },
            ..Default::default()
        }
    }
}

impl FullField<Order2> {
    /// Homogenise a 2-phase isotropic RVE (matrix + single inclusion kind) via
    /// a full-field cell-problem solve. For this v1 we accept any Rve<Order2>
    /// whose phase properties are isotropic scalar tensors (k·I) and the first
    /// phase is the matrix.
    ///
    /// The inclusion is rasterised as a centred spherical region of volume
    /// fraction equal to `phase.fraction`. More expressive voxel-mask support
    /// is v1.1.
    pub fn homogenize(&self, rve: &Rve<Order2>) -> Result<Effective<Order2>, HomogError> {
        if rve.phases.len() < 2 {
            // Homogeneous case: K_eff == matrix property, no solve required.
            let c = *rve.matrix_property()?;
            return Ok(Effective { tensor: c, concentration: None, iterations: None, residual: None });
        }

        let builder = PeriodicCubeMeshBuilder::new(&self.mesh_opts);
        let (mesh, barycenters) = builder.build()?;
        let (boundary_verts, _interior) = partition_boundary(&mesh.vertices, 1e-12);

        // Voxelise: for each tet barycentre, assign the phase whose centred-sphere
        // indicator contains the point. The inclusion sphere is centred at (0.5, 0.5, 0.5)
        // with radius set to reproduce the inclusion's volume fraction.
        let inc_fraction: f64 = rve.phases.iter()
            .skip(1)    // matrix first, inclusion second (convention for v1 fullfield)
            .map(|p| p.fraction)
            .sum();
        let r_inclusion = (3.0 * inc_fraction / (4.0 * core::f64::consts::PI)).powf(1.0 / 3.0);
        let centre = Vector3::new(0.5, 0.5, 0.5);

        let extract_k = |m: &nalgebra::Matrix3<f64>| m[(0, 0)];
        let k_matrix = extract_k(&rve.phases[0].property);
        let k_inclusion = if rve.phases.len() >= 2 { extract_k(&rve.phases[1].property) } else { k_matrix };

        let k_per_tet: alloc::vec::Vec<f64> = barycenters.iter().map(|b| {
            if (b - centre).norm() < r_inclusion { k_inclusion } else { k_matrix }
        }).collect();

        let td = cell_problem::build_tet_data(&mesh, k_per_tet)?;

        let nv = mesh.n_vertices();
        let mut chi_cols: [DVector<f64>; 3] = [DVector::zeros(nv), DVector::zeros(nv), DVector::zeros(nv)];
        let mut total_iters = 0;
        let mut worst_residual = 0.0_f64;
        for (dir, chi_slot) in chi_cols.iter_mut().enumerate() {
            let mut a = cell_problem::assemble_stiffness(&mesh, &td);
            let mut b = cell_problem::assemble_rhs(&mesh, &td, dir);
            cell_problem::apply_dirichlet_zero(&mut a, &mut b, &boundary_verts);
            let (chi, iters, res) = solver::pcg_jacobi(&a, &b, self.tol, self.max_iter)?;
            *chi_slot = chi;
            total_iters += iters;
            worst_residual = worst_residual.max(res);
        }

        let k_eff = cell_problem::effective_tensor(
            &mesh, &td,
            [&chi_cols[0], &chi_cols[1], &chi_cols[2]],
        );
        // Symmetrise (the solve is exact up to tol; round-off breaks exact symmetry).
        let k_eff_sym = (k_eff + k_eff.transpose()) * 0.5;

        Ok(Effective {
            tensor: k_eff_sym,
            concentration: None,
            iterations: Some(total_iters),
            residual: Some(worst_residual),
        })
    }
}

/// Affine-invariant reliability indicator: d_AI(C_MF, C_FF) on SPD(KM_DIM).
pub fn reliability_indicator_order2(
    c_mf: &Matrix3<f64>, c_ff: &Matrix3<f64>,
) -> Option<f64> {
    use cartan_core::Manifold;
    let spd = cartan_manifolds::Spd::<3>;
    let sa = (c_mf + c_mf.transpose()) * 0.5;
    let sb = (c_ff + c_ff.transpose()) * 0.5;
    spd.dist(&sa, &sb).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rve::Phase, shapes::Sphere};
    use alloc::sync::Arc;
    use approx::assert_relative_eq;

    #[test]
    fn reliability_zero_for_equal_tensors() {
        let c = Matrix3::<f64>::identity() * 3.0;
        let d = reliability_indicator_order2(&c, &c).unwrap();
        assert_relative_eq!(d, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn homogeneous_rve_gives_k_matrix_exactly() {
        // Matrix-only RVE: the cell problem has chi = 0 solution, K_eff = K_matrix.
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase {
            name: "M".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(2.5), fraction: 1.0,
        });
        rve.set_matrix("M");
        let ff = FullField::<Order2>::new_with_resolution(4);
        let e = ff.homogenize(&rve).unwrap();
        assert_relative_eq!(e.tensor, Matrix3::identity() * 2.5, epsilon = 1e-10);
    }

    #[test]
    fn full_field_tracks_mori_tanaka_for_dilute_sphere() {
        // Dilute spherical inclusion (phi = 0.05). Full-field on a coarse mesh
        // should agree with Mori-Tanaka to within discretisation error.
        use crate::schemes::{MoriTanaka, Scheme, SchemeOpts};
        let k0 = 1.0;
        let k1 = 5.0;
        let phi = 0.05;
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(k0), fraction: 1.0 - phi });
        rve.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(k1), fraction: phi });
        rve.set_matrix("M");

        let ff = FullField::<Order2>::new_with_resolution(8);
        let e_ff = ff.homogenize(&rve).unwrap();

        let e_mf = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();

        let d = reliability_indicator_order2(&e_ff.tensor, &e_mf.tensor).unwrap();
        println!("  FF k_eff diag:  [{}, {}, {}]",
                 e_ff.tensor[(0, 0)], e_ff.tensor[(1, 1)], e_ff.tensor[(2, 2)]);
        println!("  MT k_eff diag:  [{}, {}, {}]",
                 e_mf.tensor[(0, 0)], e_mf.tensor[(1, 1)], e_mf.tensor[(2, 2)]);
        println!("  d_AI(FF, MF) = {d:.3e}");
        // With N=8 voxelisation + Dirichlet BCs, dilute-limit agreement is ~0.1.
        // This test is the v1 infrastructure proof; tighter agreement requires
        // periodic BCs (v1.2) and finer meshes.
        assert!(d < 0.3, "expected d_AI < 0.3 at phi=0.05, got {d}");
    }
}
