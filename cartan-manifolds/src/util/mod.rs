// ~/cartan/cartan-manifolds/src/util/mod.rs

//! Matrix utility functions shared across manifold implementations.
//!
//! These are **internal** helpers for SO(N), SE(N), and future matrix manifolds.
//! They are `pub(crate)` at the module level — exposed publicly within this crate
//! for use in `euclidean`, `sphere`, and the planned matrix manifolds, but NOT
//! re-exported from the crate root to downstream users.
//!
//! ## Contents
//!
//! - [`skew`]: Skew-symmetrization and predicate (`skew(A) = (A - A^T)/2`, `is_skew`).
//! - [`matrix_exp`]: Matrix exponential specialized for skew-symmetric inputs
//!   (Rodrigues for N=2,3; Padé [6/6] scaling-and-squaring for N≥4).
//! - [`matrix_log`]: Matrix logarithm for orthogonal inputs
//!   (inverse Rodrigues for N=2,3; inverse scaling-and-squaring for N≥4).
//!
//! ## Design rationale
//!
//! We specialize the matrix exponential and logarithm for *skew-symmetric* inputs
//! rather than implementing a general matrix exp/log, for two reasons:
//!
//! 1. **Correctness:** SO(N) elements are exactly the matrix exponentials of
//!    skew-symmetric matrices (the Lie algebra so(N)). Specializing lets us use
//!    Rodrigues' formula (exact, no truncation error) for N=3.
//!
//! 2. **Efficiency:** The skew-symmetry halves the degrees of freedom and lets us
//!    pick better algorithms than a fully general approach would allow.
//!
//! ## References
//!
//! - Higham, N. J. (2005). "The Scaling and Squaring Method for the Matrix Exponential
//!   Revisited." *SIAM Review*, 47(3), 504–514.
//! - do Carmo, M. P. (1992). *Riemannian Geometry*. Chapter 3 (Lie groups).
//! - Rodrigues, O. (1840). Formula for 3D rotations; see Hall, B.C. (2015) §5.3.

pub mod matrix_exp;
pub mod matrix_log;
pub mod skew;
// sym uses symmetric_eigen() from nalgebra; requires std (Jacobi iteration).
#[cfg(feature = "std")]
pub mod sym;
