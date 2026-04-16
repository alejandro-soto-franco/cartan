//! Voxelisation: produces a phase-id array over a regular grid of the unit cube.
//!
//! v1: analytic shape level sets via `Shape`-side indicator functions. v1.1:
//! μCT array import for measured microstructures.

use crate::error::HomogError;
use alloc::vec::Vec;

pub type PhaseId = u16;
pub const NO_PHASE: PhaseId = PhaseId::MAX;

#[derive(Clone, Debug)]
pub struct VoxelGrid {
    pub resolution: usize,
    pub phase_ids: Vec<PhaseId>,
}

impl VoxelGrid {
    pub fn new(resolution: usize) -> Self {
        let n = resolution * resolution * resolution;
        Self { resolution, phase_ids: alloc::vec![NO_PHASE; n] }
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, phase: PhaseId) {
        let n = self.resolution;
        self.phase_ids[i * n * n + j * n + k] = phase;
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> PhaseId {
        let n = self.resolution;
        self.phase_ids[i * n * n + j * n + k]
    }
}

/// Placeholder voxeliser: fills all voxels with the matrix phase (id 0).
/// Task 35 follow-up: use per-shape level-set indicators for actual microstructures.
pub fn voxelize_placeholder(resolution: usize) -> Result<VoxelGrid, HomogError> {
    let mut g = VoxelGrid::new(resolution);
    for id in g.phase_ids.iter_mut() { *id = 0; }
    Ok(g)
}
