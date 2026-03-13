// ~/cartan/cartan-dec/src/error.rs

//! Error type for cartan-dec operations.

use thiserror::Error;

/// Errors that can occur in discrete exterior calculus operations.
#[derive(Debug, Error)]
pub enum DecError {
    /// The mesh has no vertices, edges, or triangles.
    #[error("empty mesh")]
    EmptyMesh,

    /// A simplex index is out of bounds.
    #[error("index out of bounds: index {index} in collection of size {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    /// A mesh is not well-centered (a circumcenter lies outside its simplex).
    /// This causes the Hodge star to have negative weights, breaking SPD.
    #[error("mesh is not well-centered: simplex {simplex} has negative dual volume {volume:.6e}")]
    NotWellCentered { simplex: usize, volume: f64 },

    /// The Laplacian matrix is singular (expected for closed manifolds without
    /// boundary conditions; caller should use a pseudoinverse or add a pin constraint).
    #[error("Laplacian is singular; add a Dirichlet pin or use pseudoinverse")]
    SingularLaplacian,

    /// A field has the wrong number of components for this mesh.
    #[error("field length mismatch: expected {expected}, got {got}")]
    FieldLengthMismatch { expected: usize, got: usize },

    /// Linear algebra error (e.g., Cholesky failed on non-SPD matrix).
    #[error("linear algebra error: {0}")]
    LinearAlgebra(String),
}
