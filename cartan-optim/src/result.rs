// ~/cartan/cartan-optim/src/result.rs

//! Optimization result type.

use cartan_core::Real;

/// The outcome of a Riemannian optimization run.
///
/// Returned by all optimizers. Contains the final iterate, convergence info,
/// and per-iteration diagnostics.
#[derive(Debug, Clone)]
pub struct OptResult<P> {
    /// The final iterate (best point found).
    pub point: P,
    /// Cost function value at the final iterate.
    pub value: Real,
    /// Riemannian gradient norm at the final iterate.
    pub grad_norm: Real,
    /// Total number of iterations executed.
    pub iterations: usize,
    /// Whether the optimizer reached the gradient tolerance.
    ///
    /// `false` means max iterations were hit without converging.
    pub converged: bool,
}
