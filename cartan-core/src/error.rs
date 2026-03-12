// ~/cartan/cartan-core/src/error.rs

//! Error types for cartan operations.
//!
//! CartanError covers all failure modes in Riemannian geometry computations:
//! logarithmic map failures at cut loci, numerical breakdowns in matrix
//! decompositions, constraint violations, and optimizer convergence failures.

use std::fmt;

/// The unified error type for all cartan operations.
///
/// Each variant captures a specific class of failure with enough context
/// to diagnose the problem. Mathematical operations that can fail (log map,
/// parallel transport across cut locus, Cholesky of near-singular matrix)
/// return Result<T, CartanError>.
#[derive(Debug, Clone)]
pub enum CartanError {
    /// Log map failed: point is on or near the cut locus.
    ///
    /// On the sphere, this means the two points are nearly antipodal.
    /// On SO(n), this means the rotation angle is near pi.
    /// On Cartan-Hadamard manifolds (SPD, Hyperbolic), this should
    /// never occur since the cut locus is empty.
    CutLocus {
        message: String,
    },

    /// A matrix decomposition or numerical computation failed.
    ///
    /// Examples: Cholesky on a matrix that lost positive-definiteness
    /// due to roundoff, SVD that did not converge, matrix logarithm
    /// of a matrix with negative eigenvalues.
    NumericalFailure {
        operation: String,
        message: String,
    },

    /// Point does not satisfy the manifold constraint.
    ///
    /// The `constraint` field describes what was checked (e.g., "||p|| = 1"
    /// for the sphere), and `violation` gives the magnitude of the deviation.
    NotOnManifold {
        constraint: String,
        violation: f64,
    },

    /// Tangent vector is not in the tangent space at the given point.
    ///
    /// The `constraint` field describes the tangent space condition
    /// (e.g., "p^T v = 0" for the sphere).
    NotInTangentSpace {
        constraint: String,
        violation: f64,
    },

    /// Line search failed to find a step size satisfying the Armijo condition.
    LineSearchFailed {
        steps_tried: usize,
    },

    /// Optimizer did not converge within the maximum number of iterations.
    ConvergenceFailure {
        iterations: usize,
        gradient_norm: f64,
    },
}

impl fmt::Display for CartanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CartanError::CutLocus { message } => {
                write!(f, "cut locus: {}", message)
            }
            CartanError::NumericalFailure { operation, message } => {
                write!(f, "numerical failure in {}: {}", operation, message)
            }
            CartanError::NotOnManifold { constraint, violation } => {
                write!(
                    f,
                    "point not on manifold: {} violated by {}",
                    constraint, violation
                )
            }
            CartanError::NotInTangentSpace { constraint, violation } => {
                write!(
                    f,
                    "tangent vector not in tangent space: {} violated by {}",
                    constraint, violation
                )
            }
            CartanError::LineSearchFailed { steps_tried } => {
                write!(f, "line search failed after {} steps", steps_tried)
            }
            CartanError::ConvergenceFailure {
                iterations,
                gradient_norm,
            } => {
                write!(
                    f,
                    "optimizer did not converge after {} iterations (gradient norm: {:.2e})",
                    iterations, gradient_norm
                )
            }
        }
    }
}

impl std::error::Error for CartanError {}
