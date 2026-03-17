// ~/cartan/cartan-optim/src/lib.rs

#![cfg_attr(not(feature = "std"), no_std)]

//! # cartan-optim
//!
//! Riemannian optimization algorithms for the cartan library.
//!
//! This crate implements first- and second-order optimization algorithms that
//! operate on any manifold implementing the `Manifold` trait from `cartan-core`.
//!
//! ## Algorithms
//!
//! | Algorithm | Struct/function | Trait requirements |
//! |-----------|----------------|--------------------|
//! | Riemannian Gradient Descent | [`minimize_rgd`] | `Manifold + Retraction` |
//! | Riemannian Conjugate Gradient | [`minimize_rcg`] | `+ ParallelTransport` |
//! | Fréchet Mean (Karcher flow) | [`frechet_mean`] | `Manifold` |
//! | Riemannian Trust Region | [`minimize_rtr`] | `+ Connection` |
//!
//! ## Usage pattern
//!
//! ```rust,ignore
//! use cartan_manifolds::Sphere;
//! use cartan_optim::{minimize_rgd, RGDConfig};
//!
//! let s2 = Sphere::<3>;
//! let config = RGDConfig::default();
//!
//! // Minimize f(p) = -p[0] (find the "north pole") on S²
//! let result = minimize_rgd(
//!     &s2,
//!     |p| -p[0],
//!     |p| s2.project_tangent(p, &SVector::from([1.0, 0.0, 0.0])),
//!     p0,
//!     &config,
//! );
//! ```
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2008.
//! - Boumal. "An Introduction to Optimization on Smooth Manifolds."
//!   Cambridge, 2023.

pub mod frechet;
pub mod rcg;
pub mod result;
pub mod rgd;
pub mod rtr;

pub use frechet::{frechet_mean, FrechetConfig};
pub use rcg::{minimize_rcg, CgVariant, RCGConfig};
pub use result::OptResult;
pub use rgd::{minimize_rgd, RGDConfig};
pub use rtr::{minimize_rtr, RTRConfig};
