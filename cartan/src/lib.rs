//! # cartan
//!
//! Riemannian geometry, manifold optimization, and geodesic computation in Rust.
//!
//! This is the top-level facade crate. It re-exports the most commonly used
//! items from the sub-crates so that downstream users only need one dependency:
//! `cartan = "0.1"`.
//!
//! ## Crate structure
//!
//! - `cartan-core` -- abstract trait system (Manifold, Retraction, etc.)
//! - `cartan-nalgebra` -- nalgebra backend (SVector, SMatrix storage types)
//! - `cartan-manifolds` -- concrete manifolds (Sphere, SO(N), SE(N), ...)
//! - `cartan-optim` -- optimization algorithms (RGD, RCG, trust region)
//! - `cartan-geo` -- geodesic and curvature tools
//! - `cartan-dec` -- discrete exterior calculus: simplicial complexes, Hodge
//!   operators, and covariant differential operators for PDE solvers
//!
//! ## Prelude
//!
//! Import `use cartan::prelude::*;` to bring all traits into scope.
//! This lets you call `.exp()`, `.log()`, `.inner()` etc. on any manifold
//! without individually importing each trait.

// Re-export sub-crates under descriptive aliases.
//
// NOTE: The core traits crate is aliased as `traits`, NOT `core`.
// Aliasing as `core` would shadow `std::core` and break macro hygiene
// in downstream crates that use standard library macros.
pub use cartan_core as traits;
pub use cartan_dec as dec;
pub use cartan_geo as geo;
pub use cartan_manifolds as manifolds;
pub use cartan_nalgebra as nalgebra_backend;
pub use cartan_optim as optim;

/// Prelude module: import with `use cartan::prelude::*` to bring all traits into scope.
///
/// This is the recommended way to use cartan. It brings all the abstract
/// geometry traits into scope so you can call their methods on any manifold
/// without needing to name each trait individually.
pub mod prelude {
    // The foundational Manifold trait: exp, log, inner, project, validate, random.
    pub use cartan_core::Manifold;

    // Optional capability traits, in order of computational cost.
    // Retraction: cheaper approximation to exp (e.g., QR on Stiefel).
    pub use cartan_core::Retraction;
    // ParallelTransport: exact parallel transport along geodesics.
    pub use cartan_core::ParallelTransport;
    // VectorTransport: approximation to parallel transport (blanket impl from PT).
    pub use cartan_core::VectorTransport;
    // Connection: Riemannian Hessian-vector products for second-order methods.
    pub use cartan_core::Connection;
    // Curvature: Riemann tensor, sectional curvature, Ricci, scalar curvature.
    pub use cartan_core::Curvature;
    // GeodesicInterpolation: geodesic curve sampling gamma(p, q, t).
    pub use cartan_core::GeodesicInterpolation;

    // The unified error type for all cartan operations.
    pub use cartan_core::CartanError;

    // The Real type alias (f64) used throughout cartan.
    pub use cartan_core::Real;
}
