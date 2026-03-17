// ~/cartan/cartan-core/src/error.rs

//! Error types for cartan operations.
//!
//! CartanError covers all failure modes in Riemannian geometry computations:
//! logarithmic map failures at cut loci, numerical breakdowns in matrix
//! decompositions, constraint violations, and optimizer convergence failures.

use core::fmt;

use crate::Real;

/// The unified error type for all cartan operations.
///
/// Each variant captures a specific class of failure with enough context
/// to diagnose the problem. Mathematical operations that can fail (log map,
/// parallel transport across cut locus, Cholesky of near-singular matrix)
/// return Result<T, CartanError>.
///
/// Under `no_alloc`, message fields are `&'static str` (no heap).
/// Under `alloc` or `std`, message fields are `String` (rich formatting).
#[derive(Debug, Clone)]
pub enum CartanError {
    /// Log map failed: point is on or near the cut locus.
    ///
    /// On the sphere, this means the two points are nearly antipodal.
    /// On SO(n), this means the rotation angle is near pi.
    /// On Cartan-Hadamard manifolds (SPD, Hyperbolic), this should
    /// never occur since the cut locus is empty.
    CutLocus {
        #[cfg(feature = "alloc")]
        message: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        message: &'static str,
    },

    /// A matrix decomposition or numerical computation failed.
    ///
    /// Examples: Cholesky on a matrix that lost positive-definiteness
    /// due to roundoff, SVD that did not converge, matrix logarithm
    /// of a matrix with negative eigenvalues.
    NumericalFailure {
        #[cfg(feature = "alloc")]
        operation: alloc::string::String,
        #[cfg(feature = "alloc")]
        message: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        operation: &'static str,
        #[cfg(not(feature = "alloc"))]
        message: &'static str,
    },

    /// Point does not satisfy the manifold constraint.
    ///
    /// The `constraint` field describes what was checked (e.g., "||p|| = 1"
    /// for the sphere), and `violation` gives the magnitude of the deviation.
    NotOnManifold {
        #[cfg(feature = "alloc")]
        constraint: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        constraint: &'static str,
        violation: Real,
    },

    /// Tangent vector is not in the tangent space at the given point.
    ///
    /// The `constraint` field describes the tangent space condition
    /// (e.g., "p^T v = 0" for the sphere).
    NotInTangentSpace {
        #[cfg(feature = "alloc")]
        constraint: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        constraint: &'static str,
        violation: Real,
    },

    /// Line search failed to find a step size satisfying the Armijo condition.
    LineSearchFailed { steps_tried: usize },

    /// Optimizer did not converge within the maximum number of iterations.
    ConvergenceFailure {
        iterations: usize,
        gradient_norm: Real,
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
            CartanError::NotOnManifold {
                constraint,
                violation,
            } => {
                write!(
                    f,
                    "point not on manifold: {} violated by {}",
                    constraint, violation
                )
            }
            CartanError::NotInTangentSpace {
                constraint,
                violation,
            } => {
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

#[cfg(feature = "std")]
impl std::error::Error for CartanError {}
