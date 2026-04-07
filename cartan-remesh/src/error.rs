// ~/cartan/cartan-remesh/src/error.rs

//! Error types for remeshing operations.

/// Errors that can occur during remeshing.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RemeshError {
    /// Edge collapse would cause a triangle foldover (normal inversion).
    #[error(
        "foldover detected at face {face}: normal rotation {angle_rad:.4} rad exceeds threshold {threshold:.4} rad"
    )]
    Foldover {
        face: usize,
        angle_rad: f64,
        threshold: f64,
    },

    /// The edge is a boundary edge and cannot be flipped.
    #[error("edge {edge} is a boundary edge (only one adjacent face)")]
    BoundaryEdge { edge: usize },

    /// The edge flip would not improve the Delaunay criterion.
    #[error("edge {edge} already satisfies Delaunay criterion (opposite angle sum = {angle_sum:.4} rad)")]
    AlreadyDelaunay { edge: usize, angle_sum: f64 },

    /// A vertex or edge index is out of bounds.
    #[error("index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    /// The edge has fewer than 2 adjacent faces (boundary or degenerate).
    #[error("edge {edge} has {count} adjacent faces, need exactly 2")]
    NotInteriorEdge { edge: usize, count: usize },

    /// A manifold geodesic operation (log/exp) failed.
    #[error("geodesic computation failed: {reason}")]
    GeodesicFailed { reason: String },
}
