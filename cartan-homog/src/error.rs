use thiserror::Error;
use alloc::string::String;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum HomogError {
    #[error("tensor is not positive definite")]
    NotPositiveDefinite,
    #[error("matrix inversion failed")]
    SingularMatrix,
    #[error("iterative scheme failed to converge in {iters} iterations (residual = {residual})")]
    DidNotConverge { iters: usize, residual: f64 },
    #[error("phase `{0}` not found in RVE")]
    UnknownPhase(String),
    #[error("quadrature degree {0} not supported; pick one of 14, 26, 50, 110, 194")]
    UnsupportedLebedevDegree(usize),
    #[error("percolation threshold reached; effective tensor is not positive definite")]
    PercolationThreshold,
    #[error("geometry error: {0}")]
    Geometry(String),
    #[error("mesh error: {0}")]
    Mesh(String),
    #[error("solver error: {0}")]
    Solver(String),
}
