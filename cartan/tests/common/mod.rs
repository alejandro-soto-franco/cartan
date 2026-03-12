// ~/cartan/cartan/tests/common/mod.rs

//! Shared test utilities for cartan integration tests.
//!
//! - `approx`: scalar and vector approximate equality assertion helpers
//! - `manifold_harness`: generic identity tests for Riemannian manifolds with SVector points
//! - `matrix_harness`: generic identity tests for Riemannian manifolds with SMatrix points

pub mod approx;
pub mod manifold_harness;
pub mod matrix_harness;
