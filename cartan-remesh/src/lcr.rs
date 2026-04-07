// ~/cartan/cartan-remesh/src/lcr.rs

//! Length-cross-ratio (LCR) conformal regularization.

use cartan_core::Manifold;
use cartan_dec::Mesh;

/// Compute the length-cross-ratio of an interior edge.
pub fn length_cross_ratio<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _edge: usize,
) -> f64 {
    todo!("Task 12")
}

/// Capture reference LCR values for all edges.
pub fn capture_reference_lcrs<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
) -> Vec<f64> {
    todo!("Task 12")
}

/// Total LCR spring energy.
pub fn lcr_spring_energy<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _ref_lcrs: &[f64],
    _kst: f64,
) -> f64 {
    todo!("Task 12")
}

/// Per-vertex gradient of the LCR spring energy.
pub fn lcr_spring_gradient<M: Manifold>(
    _mesh: &Mesh<M, 3, 2>,
    _manifold: &M,
    _ref_lcrs: &[f64],
    _kst: f64,
) -> Vec<M::Tangent> {
    todo!("Task 12")
}
