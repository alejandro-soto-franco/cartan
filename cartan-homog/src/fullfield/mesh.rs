//! Periodic cube mesh builder. Emits a tetrahedral cartan-dec Mesh on the
//! unit cube with face-matched periodic pairing.
//!
//! Stub (v1): the builder struct and options surface. Actual mesh generation
//! (Task 35 steps 1a-1c) lands in v1.1 — requires extending cartan-dec::mesh_gen
//! with a PeriodicCubeBuilder or duplicating the split-mirror logic here.

use crate::error::HomogError;

#[derive(Clone, Debug)]
pub struct PeriodicCubeMeshBuilderOpts {
    /// Grid resolution (N per side). Initial mesh has N^3 voxels tetrahedralised.
    pub resolution: usize,
    /// Curvature-CFL refinement depth (cartan-remesh driver input).
    pub refine_depth: usize,
}

impl Default for PeriodicCubeMeshBuilderOpts {
    fn default() -> Self { Self { resolution: 16, refine_depth: 3 } }
}

pub struct PeriodicCubeMeshBuilder {
    pub opts: PeriodicCubeMeshBuilderOpts,
}

impl PeriodicCubeMeshBuilder {
    pub fn new(opts: &PeriodicCubeMeshBuilderOpts) -> Self {
        Self { opts: opts.clone() }
    }

    /// Build the mesh. v1 returns NotImplemented with the v1.1 plan.
    pub fn build(&self) -> Result<(), HomogError> {
        Err(HomogError::Mesh(alloc::string::String::from(
            "PeriodicCubeMeshBuilder::build: v1.1. Needs 1-tet-per-voxel seed + \
             face-mirror pairing + phase-tag propagation. cartan-dec::mesh_gen \
             currently does not guarantee matching face discretisations on \
             opposite cube sides.")))
    }
}
