// ~/cartan/cartan-remesh/src/driver.rs

//! Adaptive remeshing driver and predicate.

use cartan_core::Manifold;
use cartan_dec::Mesh;

use crate::config::RemeshConfig;
use crate::log::RemeshLog;

/// Run the full adaptive remeshing pipeline.
pub fn adaptive_remesh<M: Manifold>(
    _mesh: &mut Mesh<M, 3, 2>,
    _manifold: &M,
    _mean_curvatures: &[f64],
    _gaussian_curvatures: &[f64],
    _config: &RemeshConfig,
) -> RemeshLog {
    todo!("Task 13")
}

/// Check whether the mesh needs remeshing.
pub fn needs_remesh<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _mean_curvatures: &[f64],
    _gaussian_curvatures: &[f64],
    _config: &RemeshConfig,
) -> bool {
    todo!("Task 13")
}
