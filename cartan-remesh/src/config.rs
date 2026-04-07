// ~/cartan/cartan-remesh/src/config.rs

//! Remesh configuration parameters.

/// Configuration for the adaptive remeshing pipeline.
///
/// The curvature-CFL criterion enforces `h_e < curvature_scale / sqrt(k_max)`
/// where `k_max = |H| + sqrt(H^2 - K)` is the larger principal curvature
/// magnitude. Edges violating this bound are split; edges below
/// `min_edge_length` are collapsed.
#[derive(Debug, Clone)]
pub struct RemeshConfig {
    /// Constant C in the curvature-CFL criterion h < C / sqrt(k_max).
    pub curvature_scale: f64,
    /// Minimum allowed edge length. Edges shorter than this are collapsed.
    pub min_edge_length: f64,
    /// Maximum allowed edge length. Edges longer than this are split.
    pub max_edge_length: f64,
    /// Minimum allowed triangle area. Triangles below this are collapsed.
    pub min_face_area: f64,
    /// Maximum allowed triangle area. Triangles above this trigger splits.
    pub max_face_area: f64,
    /// Foldover rejection threshold in radians. Default: 0.5 (~28.6 degrees).
    pub foldover_threshold: f64,
    /// LCR spring stiffness for conformal regularization. 0.0 to disable.
    pub lcr_spring_stiffness: f64,
    /// Number of tangential Laplacian smoothing iterations per remesh pass.
    pub smoothing_iterations: usize,
}

impl Default for RemeshConfig {
    fn default() -> Self {
        Self {
            curvature_scale: 0.5,
            min_edge_length: 0.01,
            max_edge_length: 1.0,
            min_face_area: 1e-6,
            max_face_area: 1.0,
            foldover_threshold: 0.5,
            lcr_spring_stiffness: 0.0,
            smoothing_iterations: 3,
        }
    }
}
