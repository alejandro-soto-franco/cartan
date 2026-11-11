//! Errors raised while building a complex or decomposing a flow on it.

use thiserror::Error;

/// A malformed complex, or input whose length does not match it.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ComplexError {
    /// An edge `[i, j]` with `i >= j`. Edges are stored low index first.
    #[error("edge {edge:?} is not ordered low index first")]
    EdgeNotOrdered { edge: [u32; 2] },
    /// The same edge listed twice.
    #[error("edge {edge:?} is listed more than once")]
    DuplicateEdge { edge: [u32; 2] },
    /// A vertex index at or beyond the vertex count.
    #[error("vertex {vertex} is out of range for {n_vertices} vertices")]
    VertexOutOfRange { vertex: u32, n_vertices: usize },
    /// A triangle `[a, b, c]` without `a < b < c`, or out of lexicographic order.
    #[error("triangle {triangle:?} is not ordered a < b < c in sorted position")]
    TriangleNotOrdered { triangle: [u32; 3] },
    /// A triangle whose boundary uses an edge the complex does not contain.
    #[error("triangle {triangle:?} needs edge {edge:?}, which the complex does not contain")]
    MissingEdge { triangle: [u32; 3], edge: [u32; 2] },
    /// An input vector whose length does not match the complex.
    #[error("{what} has length {got}, expected {expected}")]
    LengthMismatch {
        what: &'static str,
        got: usize,
        expected: usize,
    },
}
