//! # cartan
//!
//! Riemannian geometry, manifold optimisation, and geodesic computation in Rust.
//!
//! This is the facade crate. It re-exports the family under short aliases so a
//! downstream user needs one dependency:
//!
//! ```toml
//! cartan = "0.8"
//! ```
//!
//! ## Three regimes
//!
//! One trait system spans geometry at points, at fields, and along paths:
//!
//! | regime | crates | what it does |
//! |---|---|---|
//! | points | `manifolds`, `optim`, `geo` | `exp`, `log`, transport, optimisation |
//! | fields | `dec`, `remesh`, `io` | discrete exterior calculus, bundles, export |
//! | paths | `stochastic` | frame bundle, horizontal lift, development |
//!
//! ## Features
//!
//! The default is `std` plus `dec`. Everything else is opt-in, so a user who
//! wants `exp` and `log` does not compile a sparse solver.
//!
//! | feature | brings in | needs |
//! |---|---|---|
//! | `alloc` | core, manifolds, optim, geo | no_std with an allocator |
//! | `std` | the above, with std | std |
//! | `dec` (default) | `cartan-dec` | std |
//! | `remesh` | `cartan-remesh` | `dec` |
//! | `stochastic` | `cartan-stochastic` | std |
//! | `homog` | `cartan-homog` mean-field schemes | alloc |
//! | `full-field` | `cartan-homog` cell-problem solver | `homog`, `remesh`, std |
//! | `io` | `cartan-io` VTK and Blender export | `dec` |
//! | `maxwell` | `cartan-maxwell` | `io` |
//! | `full` | all of the above | std |
//!
//! ## Embedded and no_std
//!
//! ```toml
//! cartan = { version = "0.8", default-features = false, features = ["alloc"] }
//! ```
//!
//! That gives the point-geometry stack and, by adding `homog`, the mean-field
//! homogenisation schemes. CI builds this configuration for
//! `thumbv7em-none-eabihf` on every run.
//!
//! ## Prelude
//!
//! `use cartan::prelude::*;` brings the geometry traits into scope, so `.exp()`,
//! `.log()` and `.inner()` are callable on any manifold without naming each
//! trait.
//!
//! ## Guide
//!
//! See [`guide`] for worked chapters. Every code block there is a doctest.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod guide;

// Re-export sub-crates under descriptive aliases.
//
// NOTE: The core traits crate is aliased as `traits`, NOT `core`.
// Aliasing as `core` would shadow `std::core` and break macro hygiene
// in downstream crates that use standard library macros.
pub use cartan_core as traits;
pub use cartan_geo as geo;
pub use cartan_manifolds as manifolds;
pub use cartan_optim as optim;

#[cfg(feature = "dec")]
#[cfg_attr(docsrs, doc(cfg(feature = "dec")))]
pub use cartan_dec as dec;

#[cfg(feature = "remesh")]
#[cfg_attr(docsrs, doc(cfg(feature = "remesh")))]
pub use cartan_remesh as remesh;

#[cfg(feature = "stochastic")]
#[cfg_attr(docsrs, doc(cfg(feature = "stochastic")))]
pub use cartan_stochastic as stochastic;

#[cfg(feature = "homog")]
#[cfg_attr(docsrs, doc(cfg(feature = "homog")))]
pub use cartan_homog as homog;

#[cfg(feature = "io")]
#[cfg_attr(docsrs, doc(cfg(feature = "io")))]
pub use cartan_io as io;

#[cfg(feature = "maxwell")]
#[cfg_attr(docsrs, doc(cfg(feature = "maxwell")))]
pub use cartan_maxwell as maxwell;

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
