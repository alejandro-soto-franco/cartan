// ~/cartan/cartan-core/src/lib.rs

//! # cartan-core
//!
//! Core trait definitions for Riemannian geometry.
//!
//! This crate defines the foundational traits that all cartan manifolds,
//! optimizers, and tools depend on. It has minimal dependencies (only `rand`
//! for the Rng trait bound) and can be used standalone by downstream crates
//! that want to implement custom manifolds against the cartan trait system.
//!
//! ## The `Real` type alias
//!
//! All floating-point computation in cartan uses `Real`, currently aliased
//! to `f64`. This alias exists so that a future version can generify over
//! `T: Scalar` (supporting f32 for GPU/ML workloads) with a mechanical
//! find-and-replace refactor.

pub mod error;
pub mod manifold;

/// The floating-point type used throughout cartan.
///
/// Currently f64. Designed so that replacing this alias with a generic
/// type parameter `T: Scalar` is a mechanical refactor when f32 support
/// is needed. All structs and functions use `Real` instead of `f64` directly.
pub type Real = f64;

pub use error::CartanError;
pub use manifold::Manifold;
