//! # cartan-geo
//!
//! Geodesic computation and geometric tools for the cartan library.
//!
//! This crate provides higher-level geometric utilities built on top of
//! the `Manifold` trait from `cartan-core` and the concrete manifolds
//! from `cartan-manifolds`. It focuses on global geometry: geodesic curves,
//! curvature queries, Jacobi field integration, and Voronoi cells.
//!
//! ## Planned functionality (v0.1 roadmap)
//!
//! - `Geodesic` -- parameterized geodesic curve gamma: [0,1] -> M
//! - `GeodesicGrid` -- uniform sampling along geodesics
//! - `CurvatureQuery` -- sectional and Ricci curvature at a point
//! - `JacobiField` -- ODE integration for geodesic deviation
//!
//! ## References
//!
//! - do Carmo. "Riemannian Geometry." Birkhauser, 1992. Chapter 5 (Jacobi fields).
//! - Petersen. "Riemannian Geometry." Springer, 2016. Chapter 11 (curvature).
