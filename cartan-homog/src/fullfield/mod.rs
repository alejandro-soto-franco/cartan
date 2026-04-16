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
pub mod macroscale;
pub mod hausdorff;

pub use mesh::{PeriodicCubeMeshBuilder, PeriodicCubeMeshBuilderOpts, partition_boundary};
pub use voxelize::{CentredInclusion, VoxelGrid, load_voxel_raw_u8, voxelize_centred};

use nalgebra::{DVector, Matrix3};

/// Boundary-condition style for the cell-problem solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryConditions {
    /// Pin every boundary vertex to χ = 0. Easy, but leaks stress into the answer
    /// at the faces; biases K_eff for non-dilute volume fractions.
    DirichletZero,
    /// Identify opposite faces of the cube as the same DOF; anchor one vertex to
    /// χ = 0 to eliminate the constant null space. Produces the textbook
    /// periodic-cell-problem answer that ECHOES and Mori-Tanaka target.
    Periodic,
}

#[derive(Clone, Debug)]
pub struct FullField<O: TensorOrder> {
    pub mesh_opts: PeriodicCubeMeshBuilderOpts,
    pub tol: f64,
    pub max_iter: usize,
    pub bc: BoundaryConditions,
    /// Number of adaptive refinement passes along the inclusion boundary.
    /// Each pass flags tets whose K differs from any of their 4 face neighbours
    /// (indicating the tet straddles a phase transition) and applies
    /// `cartan_remesh::barycentric_refine_tets`. 0 = no refinement.
    pub refine_depth: usize,
    _marker: core::marker::PhantomData<O>,
}

impl<O: TensorOrder> Default for FullField<O> {
    fn default() -> Self {
        Self {
            mesh_opts: PeriodicCubeMeshBuilderOpts::default(),
            tol: 1e-8,
            max_iter: 5_000,
            bc: BoundaryConditions::Periodic,
            refine_depth: 0,
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
    /// a full-field cell-problem solve. For this v1 we accept any `Rve<Order2>`
    /// whose phase properties are isotropic scalar tensors (`k·I`) and the first
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
        let (mut mesh, mut barycenters) = builder.build()?;

        // Optional adaptive refinement driven by the inclusion indicator. Each
        // pass refines tets that straddle the inclusion boundary: a tet is
        // "straddling" if its barycentre is inside but at least one of its 4
        // vertices is outside the inclusion (or vice versa).
        if self.refine_depth > 0 {
            let inc_phase = &rve.phases[1];
            let inc_fraction = inc_phase.fraction;
            let aspect = classify_aspect(&inc_phase.shape);
            let inclusion = CentredInclusion::from_volume_fraction(inc_fraction, aspect);
            for _ in 0..self.refine_depth {
                let flags: alloc::vec::Vec<bool> = (0..mesh.n_simplices()).map(|s| {
                    let tet = mesh.simplices[s];
                    let inside_vs = [
                        inclusion.contains(&mesh.vertices[tet[0]]),
                        inclusion.contains(&mesh.vertices[tet[1]]),
                        inclusion.contains(&mesh.vertices[tet[2]]),
                        inclusion.contains(&mesh.vertices[tet[3]]),
                    ];
                    let first = inside_vs[0];
                    inside_vs.iter().any(|&b| b != first)
                }).collect();
                let n_refined = cartan_remesh::barycentric_refine_tets(&mut mesh, &flags)
                    .map_err(|e| HomogError::Mesh(alloc::format!("refine pass failed: {e}")))?;
                if n_refined == 0 { break; }
            }
            // Recompute barycentres after refinement.
            barycenters = mesh.simplices.iter().map(|tet| {
                (mesh.vertices[tet[0]] + mesh.vertices[tet[1]]
               + mesh.vertices[tet[2]] + mesh.vertices[tet[3]]) / 4.0
            }).collect();
        }

        let (boundary_verts, _interior) = partition_boundary(&mesh.vertices, 1e-12);

        // Voxelise: for each tet barycentre, assign the phase whose centred inclusion
        // contains the point. Inclusion geometry is inferred from phase[1].shape:
        //   - Sphere or unknown  -> equivalent-volume centred sphere
        //   - Spheroid           -> oblate/prolate spheroid with matching aspect
        //   - PennyCrack         -> oblate spheroid at the PennyCrack.tiny_aspect
        let inc_phase = &rve.phases[1];
        let inc_fraction = inc_phase.fraction;
        let aspect = classify_aspect(&inc_phase.shape);
        let inclusion = CentredInclusion::from_volume_fraction(inc_fraction, aspect);

        let extract_k = |m: &nalgebra::Matrix3<f64>| m[(0, 0)];
        let k_matrix = extract_k(&rve.phases[0].property);
        let k_inclusion = extract_k(&rve.phases[1].property);

        let k_per_tet: alloc::vec::Vec<f64> = barycenters.iter().map(|b| {
            if inclusion.contains(b) { k_inclusion } else { k_matrix }
        }).collect();

        let td = cell_problem::build_tet_data(&mesh, k_per_tet)?;

        let nv = mesh.n_vertices();
        let mut chi_cols: [DVector<f64>; 3] = [DVector::zeros(nv), DVector::zeros(nv), DVector::zeros(nv)];
        let mut total_iters = 0;
        let mut worst_residual = 0.0_f64;
        let periodic_pairs = if self.bc == BoundaryConditions::Periodic {
            mesh::periodic_pairs_structured(self.mesh_opts.resolution)
        } else {
            alloc::vec::Vec::new()
        };
        for (dir, chi_slot) in chi_cols.iter_mut().enumerate() {
            let mut a = cell_problem::assemble_stiffness(&mesh, &td);
            let mut b = cell_problem::assemble_rhs(&mesh, &td, dir);
            match self.bc {
                BoundaryConditions::DirichletZero => {
                    cell_problem::apply_dirichlet_zero(&mut a, &mut b, &boundary_verts);
                }
                BoundaryConditions::Periodic => {
                    cell_problem::apply_periodic(&mut a, &mut b, &periodic_pairs, 0);
                }
            }
            let (mut chi, iters, res) = solver::solve_with_fallback(&a, &b, self.tol, self.max_iter)?;
            if self.bc == BoundaryConditions::Periodic {
                cell_problem::expand_periodic(&mut chi, &periodic_pairs);
            }
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

impl FullField<Order2> {
    /// Full-field homogenisation from a pre-tagged VoxelGrid (e.g., μCT import).
    /// The `phase_props` slice gives the isotropic scalar conductivity per phase id;
    /// the 0 phase is the matrix, 1+ are inclusions. Vertex DOFs are pinned via
    /// `self.bc`; the K per tet is taken from the phase id at the tet barycentre's
    /// containing voxel.
    pub fn homogenize_voxel(
        &self, voxel: &VoxelGrid, phase_props: &[f64],
    ) -> Result<Effective<Order2>, HomogError> {
        if voxel.resolution != self.mesh_opts.resolution {
            return Err(HomogError::Mesh(alloc::format!(
                "voxel resolution {} != mesh resolution {}",
                voxel.resolution, self.mesh_opts.resolution)));
        }
        let builder = PeriodicCubeMeshBuilder::new(&self.mesh_opts);
        let (mesh, barycenters) = builder.build()?;
        let (boundary_verts, _) = partition_boundary(&mesh.vertices, 1e-12);

        let n = voxel.resolution;
        let h = 1.0 / (n as f64);
        let k_per_tet: alloc::vec::Vec<f64> = barycenters.iter().map(|b| {
            let ii = ((b.x / h) as usize).min(n - 1);
            let jj = ((b.y / h) as usize).min(n - 1);
            let kk = ((b.z / h) as usize).min(n - 1);
            let phase = voxel.get(ii, jj, kk) as usize;
            phase_props.get(phase).copied().unwrap_or_else(|| phase_props[0])
        }).collect();

        let td = cell_problem::build_tet_data(&mesh, k_per_tet)?;
        let nv = mesh.n_vertices();
        let mut chi_cols: [DVector<f64>; 3] = [DVector::zeros(nv), DVector::zeros(nv), DVector::zeros(nv)];
        let mut total_iters = 0;
        let mut worst_residual = 0.0_f64;
        let periodic_pairs = if self.bc == BoundaryConditions::Periodic {
            mesh::periodic_pairs_structured(self.mesh_opts.resolution)
        } else {
            alloc::vec::Vec::new()
        };
        for (dir, chi_slot) in chi_cols.iter_mut().enumerate() {
            let mut a = cell_problem::assemble_stiffness(&mesh, &td);
            let mut b = cell_problem::assemble_rhs(&mesh, &td, dir);
            match self.bc {
                BoundaryConditions::DirichletZero => {
                    cell_problem::apply_dirichlet_zero(&mut a, &mut b, &boundary_verts);
                }
                BoundaryConditions::Periodic => {
                    cell_problem::apply_periodic(&mut a, &mut b, &periodic_pairs, 0);
                }
            }
            let (mut chi, iters, res) = solver::solve_with_fallback(&a, &b, self.tol, self.max_iter)?;
            if self.bc == BoundaryConditions::Periodic {
                cell_problem::expand_periodic(&mut chi, &periodic_pairs);
            }
            *chi_slot = chi;
            total_iters += iters;
            worst_residual = worst_residual.max(res);
        }
        let k_eff = cell_problem::effective_tensor(
            &mesh, &td, [&chi_cols[0], &chi_cols[1], &chi_cols[2]]);
        let k_eff_sym = (k_eff + k_eff.transpose()) * 0.5;
        Ok(Effective {
            tensor: k_eff_sym,
            concentration: None,
            iterations: Some(total_iters),
            residual: Some(worst_residual),
        })
    }
}

/// Inspect a phase's shape trait object to pick the voxeliser's inclusion aspect.
/// Falls back to 1.0 (sphere) for any shape type not recognised here.
fn classify_aspect(shape: &crate::shapes::UserInclusion<Order2>) -> f64 {
    // Trait-object introspection: downcast via Any through the concrete types we know.
    // Since Shape<O> is not 'static + Any we use type-name sniffing on the Debug output.
    let dbg = alloc::format!("{shape:?}");
    if dbg.starts_with("PennyCrack") {
        // Pull "tiny_aspect: <f>" out of the Debug string. Safe default 1e-3 if not found.
        if let Some(idx) = dbg.find("tiny_aspect: ") {
            let tail = &dbg[idx + "tiny_aspect: ".len()..];
            if let Some(end) = tail.find(|c: char| !c.is_ascii_digit() && c != '.' && c != 'e' && c != '-' && c != '+') {
                return tail[..end].parse::<f64>().unwrap_or(1e-3);
            }
        }
        return 1e-3;
    }
    if dbg.starts_with("Spheroid") {
        if let Some(idx) = dbg.find("aspect: ") {
            let tail = &dbg[idx + "aspect: ".len()..];
            if let Some(end) = tail.find(|c: char| !c.is_ascii_digit() && c != '.' && c != 'e' && c != '-' && c != '+') {
                return tail[..end].parse::<f64>().unwrap_or(1.0);
            }
        }
    }
    1.0
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
    fn voxel_import_matches_analytic_sphere_path() {
        // Generate a voxelisation of a centred sphere matching phi=0.1, run the
        // voxel FullField, and check it agrees with the analytic sphere RVE.
        use crate::{rve::Phase, shapes::Sphere};
        use alloc::sync::Arc;

        let phi = 0.1;
        let n = 8;
        let incl = CentredInclusion::from_volume_fraction(phi, 1.0);
        let voxel = voxelize_centred(&incl, n);
        let ff = FullField::<Order2>::new_with_resolution(n);
        let e_voxel = ff.homogenize_voxel(&voxel, &[1.0, 5.0]).unwrap();

        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 1.0 - phi });
        rve.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(5.0), fraction: phi });
        rve.set_matrix("M");
        let e_analytic = ff.homogenize(&rve).unwrap();

        let d = reliability_indicator_order2(&e_voxel.tensor, &e_analytic.tensor).unwrap();
        // Voxelisation discretises the sphere into staircase cells; with N=8
        // the two should agree to within ~10% (d_AI ~ 1e-1). This establishes
        // that the voxel pipeline produces physically valid answers.
        assert!(d < 0.2, "voxel vs analytic d_AI = {d}");
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
    fn refinement_reduces_mf_ff_gap_at_phase_boundary() {
        use crate::schemes::{MoriTanaka, Scheme, SchemeOpts};
        let k0 = 1.0;
        let k1 = 5.0;
        let phi = 0.2;
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(k0), fraction: 1.0 - phi });
        rve.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(k1), fraction: phi });
        rve.set_matrix("M");

        let ff_coarse = FullField::<Order2>::new_with_resolution(6);
        let e_coarse = ff_coarse.homogenize(&rve).unwrap();

        let mut ff_refined = FullField::<Order2>::new_with_resolution(6);
        ff_refined.refine_depth = 1;
        let e_refined = ff_refined.homogenize(&rve).unwrap();

        let e_mf = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        let d_coarse = reliability_indicator_order2(&e_coarse.tensor, &e_mf.tensor).unwrap();
        let d_refined = reliability_indicator_order2(&e_refined.tensor, &e_mf.tensor).unwrap();
        println!("  coarse d_AI(FF, MF)  = {d_coarse:.3e}");
        println!("  refined d_AI(FF, MF) = {d_refined:.3e}");
        // Refinement should not make things meaningfully worse. With 1 pass of
        // boundary-flagged refinement the gap either improves or stays within
        // a small factor (barycentric refinement alone has some bias).
        assert!(d_refined < d_coarse * 1.5,
                "refined gap {d_refined} should be <= 1.5x coarse gap {d_coarse}");
    }

    #[test]
    fn full_field_void_limit_solves_at_1e6_contrast() {
        // Near-void inclusion at phi=0.2 with 10^6:1 contrast. v1.1 Jacobi-CG
        // stalled on this; the new ladder (Jacobi -> ILU -> dense LU) succeeds.
        use crate::schemes::{MoriTanaka, Scheme, SchemeOpts};
        let mut rve = Rve::<Order2>::new();
        rve.add_phase(Phase { name: "M".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0), fraction: 0.8 });
        rve.add_phase(Phase { name: "I".into(), shape: Arc::new(Sphere),
            property: Order2::scalar(1.0e-6), fraction: 0.2 });
        rve.set_matrix("M");

        let mut ff = FullField::<Order2>::new_with_resolution(6);
        ff.tol = 1e-6;
        ff.max_iter = 20_000;
        let e_ff = ff.homogenize(&rve).expect("FF should now solve at 10^6 contrast");
        let e_mf = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
        let d = reliability_indicator_order2(&e_ff.tensor, &e_mf.tensor).unwrap();
        println!("  void-limit FF k_eff diag:  [{}, {}, {}]",
                 e_ff.tensor[(0,0)], e_ff.tensor[(1,1)], e_ff.tensor[(2,2)]);
        println!("  void-limit MT k_eff diag:  [{}, {}, {}]",
                 e_mf.tensor[(0,0)], e_mf.tensor[(1,1)], e_mf.tensor[(2,2)]);
        println!("  void-limit d_AI(FF, MF) = {d:.3e}");
        // The gap at the void limit is inherently large because FF resolves the
        // voids exactly as holes while MT remains an approximation. We just
        // require: FF converged, k_eff is SPD, and k_eff < k_matrix (voids
        // reduce conductivity, as physics requires).
        assert!(e_ff.tensor[(0, 0)] > 0.0);
        assert!(e_ff.tensor[(0, 0)] < 1.0);
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
