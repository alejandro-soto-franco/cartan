//! The cell-to-degree-of-freedom incidence, transposed.

use crate::mass::HostMass;
use crate::restrict::CONSTRAINED;

/// For every degree of freedom, which cell rows contribute to it.
///
/// The element-by-element loop in [`HostMass::apply_slice`] runs over cells and
/// accumulates into degrees of freedom, so two cells sharing a face write the
/// same slot. On one thread that is a plain `+=`. Across threads it needs an
/// atomic, and an atomic floating-point add reorders the summation, so the
/// result varies between runs and cannot be compared against the host path at
/// the tolerance the rest of this stack is measured at.
///
/// Transposing the incidence removes the problem rather than working around it.
/// One thread owns one degree of freedom, gathers every cell row that feeds it,
/// and writes once. The total memory traffic is unchanged: the scatter form
/// reads each element-matrix row once and writes it once, and so does this.
#[derive(Debug, Clone)]
pub struct GatherMap {
    ndofs: usize,
    nlocal: usize,
    /// Start of each degree of freedom's run in [`GatherMap::entries`], with a
    /// trailing total. Length `ndofs + 1`.
    offsets: Vec<u32>,
    /// `cell * nlocal + local_row`, packed so a device kernel reads one `u32`
    /// per incidence rather than two.
    entries: Vec<u32>,
}

impl GatherMap {
    /// Transpose the incidence of `mass`.
    pub fn new(mass: &HostMass) -> Self {
        let ndofs = mass.ndofs_total();
        let nlocal = mass.nlocal();
        let dofs = mass.dof_map();

        let mut counts = vec![0u32; ndofs + 1];
        for &dof in dofs {
            if dof != CONSTRAINED {
                counts[dof as usize + 1] += 1;
            }
        }
        for i in 0..ndofs {
            counts[i + 1] += counts[i];
        }
        let offsets = counts;

        let mut cursor = offsets.clone();
        let mut entries = vec![0u32; offsets[ndofs] as usize];
        for (slot, &dof) in dofs.iter().enumerate() {
            if dof != CONSTRAINED {
                let at = &mut cursor[dof as usize];
                entries[*at as usize] = slot as u32;
                *at += 1;
            }
        }

        Self {
            ndofs,
            nlocal,
            offsets,
            entries,
        }
    }

    /// Degrees of freedom the map covers.
    pub fn ndofs(&self) -> usize {
        self.ndofs
    }

    /// Faces of the operator's grade per cell.
    pub fn nlocal(&self) -> usize {
        self.nlocal
    }

    /// Run boundaries, length `ndofs + 1`.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Packed `cell * nlocal + local_row` incidences.
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    /// The largest number of cell rows any one degree of freedom gathers. A
    /// device launch sizes its per-thread loop bound against this.
    pub fn max_valence(&self) -> usize {
        (0..self.ndofs)
            .map(|i| (self.offsets[i + 1] - self.offsets[i]) as usize)
            .max()
            .unwrap_or(0)
    }

    /// `y <- M x` by gathering, which is what the device kernel computes.
    ///
    /// Present so the ordering the device path uses can be checked on the host,
    /// independently of whether a GPU is present.
    pub fn apply_slice(&self, mass: &HostMass, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.ndofs);
        assert_eq!(y.len(), self.ndofs);
        let n = self.nlocal;
        let dofs = mass.dof_map();
        let elmats = mass.elmats();

        for (i, slot_out) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            for e in self.offsets[i]..self.offsets[i + 1] {
                let slot = self.entries[e as usize] as usize;
                let cell = slot / n;
                let local_row = slot % n;
                let row = &elmats[(cell * n + local_row) * n..(cell * n + local_row + 1) * n];
                let cell_dofs = &dofs[cell * n..(cell + 1) * n];
                for j in 0..n {
                    let col = cell_dofs[j];
                    if col != CONSTRAINED {
                        sum += row[j] * x[col as usize];
                    }
                }
            }
            *slot_out = sum;
        }
    }
}
