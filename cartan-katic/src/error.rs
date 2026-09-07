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

    /// A symmetry's declared constants disagree with the computed invariant
    /// theory, so the compile-time amplitude count is wrong for it.
    #[error(
        "{group}: declared rank {declared_rank} amplitudes {declared_amplitudes}, \
         computed rank {computed_rank} amplitudes {computed_amplitudes}"
    )]
    ConstMismatch {
        group: &'static str,
        declared_rank: usize,
        declared_amplitudes: usize,
        computed_rank: usize,
        computed_amplitudes: usize,
    },

    /// The active stress at this symmetry order needs more derivatives than
    /// the element space provides.
    #[error(
        "symmetry order {degree} needs {needed} derivatives for the active force, \
         and piecewise-linear elements supply {available}"
    )]
    InsufficientRegularity {
        degree: usize,
        needed: usize,
        available: usize,
    },

    /// No invariant of separating rank was found below the search ceiling.
    #[error("no separating invariant found up to rank {max_rank}")]
    NoSeparatingInvariant { max_rank: usize },
}
