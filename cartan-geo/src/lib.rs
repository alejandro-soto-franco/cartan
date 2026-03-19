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
//! | [`geodesic`] | `Geodesic<M>` -- parameterized geodesic, sampling, two-point construction |
//! | [`curvature`] | `CurvatureQuery<M>` -- sectional, Ricci, scalar curvature at a point |
//! | [`jacobi`] | `integrate_jacobi` -- RK4 Jacobi field ODE integration |
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Birkhauser, 1992. Chapters 3-5.
//! - Petersen. "Riemannian Geometry." Springer, 2016. Chapter 11.

#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod curvature;
pub mod geodesic;
#[cfg(feature = "alloc")]
pub mod holonomy;
pub mod jacobi;

pub use curvature::{CurvatureQuery, scalar_at, sectional_at};
pub use geodesic::Geodesic;
#[cfg(feature = "alloc")]
pub use holonomy::{
    Disclination, edge_transition, holonomy_deviation, is_half_disclination,
    loop_holonomy, rotation_angle, scan_disclinations,
};
#[cfg(feature = "alloc")]
pub use jacobi::{JacobiResult, integrate_jacobi};
