//! Linear solver for the cell problem. v1 stub; v1.1 impl.

use crate::error::HomogError;

/// Jacobi-preconditioned CG on a sparse SPD system from sprs. v1.1 upgrade path:
/// algebraic multigrid (AMG) with smoothed-aggregation coarsening.
pub fn solve_cg() -> Result<(), HomogError> {
    Err(HomogError::Solver(alloc::string::String::from(
        "solver::solve_cg: v1.1. Plan: Jacobi-CG on sprs::CsMat for the Poisson-\
         like cell problem, loosen to AMG-CG if convergence is poor near \
         phase boundaries.")))
}
