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
//! ## Trait hierarchy
//!
//! ```text
//! Manifold (base: exp, log, inner, project, validate)
//!   |
//!   +-- Retraction (cheaper exp approximation)
//!   +-- ParallelTransport -> VectorTransport (blanket impl)
//!   +-- Connection (Riemannian Hessian)
//!   |     |
//!   |     +-- Curvature (Riemann tensor, Ricci, scalar)
//!   +-- GeodesicInterpolation (gamma(t) sampling)
//! ```
//!
//! ## The `Real` type alias
//!
//! All floating-point computation uses `Real`, currently aliased to `f64`.
//! This exists so that a future version can generify over `T: Scalar`
//! with a mechanical find-and-replace refactor.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod connection;
pub mod curvature;
pub mod error;
pub mod geodesic;
pub mod manifold;
pub mod retraction;
pub mod transport;

/// The floating-point type used throughout cartan.
///
/// Currently f64. Designed so that replacing this alias with a generic
/// type parameter `T: Scalar` is a mechanical refactor when f32 support
/// is needed. All structs and functions use `Real` instead of `f64` directly.
pub type Real = f64;

// Re-exports for convenience.
pub use connection::Connection;
pub use curvature::Curvature;
pub use error::CartanError;
pub use geodesic::GeodesicInterpolation;
pub use manifold::Manifold;
pub use retraction::Retraction;
pub use transport::{ParallelTransport, VectorTransport};
