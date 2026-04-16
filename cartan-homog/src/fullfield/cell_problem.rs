//! Cell-problem assembly over a cartan-dec Mesh. v1 stub, v1.1 impl.

use crate::error::HomogError;

/// Assemble the sparse LHS + RHS for one direction of the cell problem.
pub fn assemble_direction(_direction: usize) -> Result<(), HomogError> {
    Err(HomogError::Solver(alloc::string::String::from(
        "cell_problem::assemble_direction: v1.1. Needs cartan-dec::Operators \
         (Laplace-Beltrami) with per-tet conductivity weights + periodic BC \
         row/column pairing against the PeriodicCubeMeshBuilder output.")))
}

/// Effective tensor from volume-averaged flux over the solved corrector fields.
pub fn volume_average_effective() -> Result<(), HomogError> {
    Err(HomogError::Solver(alloc::string::String::from(
        "cell_problem::volume_average_effective: v1.1. Integrates C(x)(e_i + \
         grad chi_i) weighted by tet volumes.")))
}
