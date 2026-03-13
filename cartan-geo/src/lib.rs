// ~/cartan/cartan-geo/src/lib.rs

//! # cartan-geo
//!
//! Geodesic computation and geometric tools for the cartan library.
//!
//! This crate provides higher-level geometric utilities built on top of
//! the `Manifold` trait from `cartan-core` and the concrete manifolds from
//! `cartan-manifolds`. It focuses on *global* geometry: geodesic curves,
//! curvature queries, and Jacobi field integration.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`geodesic`] | `Geodesic<M>` — parameterized geodesic, sampling, two-point construction |
//! | [`curvature`] | `CurvatureQuery<M>` — sectional, Ricci, scalar curvature at a point |
//! | [`jacobi`] | `integrate_jacobi` — RK4 Jacobi field ODE integration |
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Birkhäuser, 1992. Chapters 3–5.
//! - Petersen. "Riemannian Geometry." Springer, 2016. Chapter 11.

pub mod curvature;
pub mod geodesic;
pub mod jacobi;

pub use curvature::{scalar_at, sectional_at, CurvatureQuery};
pub use geodesic::Geodesic;
pub use jacobi::{integrate_jacobi, JacobiResult};
