//! Error type for k-atic order-parameter operations.

use thiserror::Error;

/// Errors that can occur when building or evolving a k-atic field.
#[derive(Debug, Error)]
pub enum KaticError {
    /// The rotor connection does not lift consistently: the product of the
    /// three edge rotors around a triangle is -1 rather than +1. This is the
    /// discrete appearance of a non-vanishing second Stiefel-Whitney class.
    #[error("inconsistent spin lift: triangle {triangle} has holonomy -1")]
    InconsistentSpinLift { triangle: usize },

    /// The top-degree homogeneous part of the bulk energy is not positive
    /// definite on the invariant cone, so the functional is unbounded below.
    #[error("energy is not coercive: top-degree part of degree {degree} is indefinite")]
    NonCoerciveEnergy { degree: usize },

    /// A rotor left the unit sphere, so the state is off the admissible set.
    #[error("rotor at vertex {vertex} has norm {norm:.6e}, expected 1")]
    NonUnitRotor { vertex: usize, norm: f64 },

    /// The implicit discrete-gradient step failed to converge.
    #[error("discrete-gradient Newton solve diverged at step {step}, residual {residual:.6e}")]
    NewtonDiverged { step: usize, residual: f64 },

    /// No invariant of separating rank was found below the search ceiling.
    #[error("no separating invariant found up to rank {max_rank}")]
    NoSeparatingInvariant { max_rank: usize },
}
