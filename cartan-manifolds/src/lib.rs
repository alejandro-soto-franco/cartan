// ~/cartan/cartan-manifolds/src/lib.rs

//! Concrete Riemannian manifold implementations.
//!
//! Each manifold implements the traits from `cartan-core` using
//! `nalgebra` `SVector` and `SMatrix` types for statically-sized storage.
//!
//! ## Available manifolds
//!
//! - [`Euclidean<N>`] -- R^N with the flat metric (trivial baseline)
//! - [`Sphere<N>`] -- S^{N-1} in R^N with round metric
//! - [`SpecialOrthogonal<N>`] -- SO(N) with bi-invariant metric
//! - [`SpecialEuclidean<N>`] -- SE(N) with product metric
//! - [`Corr<N>`] -- Corr(N) correlation matrices with Frobenius metric (flat)
//! - [`Spd<N>`] -- SPD(N) with affine-invariant metric (Cartan-Hadamard)
//! - [`Grassmann<N, K>`] -- Gr(N, K) with canonical metric

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

// corr, grassmann, and spd use DMatrix/DVector (dynamic nalgebra) and Vec; requires alloc.
#[cfg(feature = "alloc")]
pub mod corr;
pub mod euclidean;
#[cfg(feature = "alloc")]
pub mod grassmann;
pub mod se;
pub mod so;
#[cfg(feature = "alloc")]
pub mod spd;
pub mod sphere;
pub mod util;

#[cfg(feature = "alloc")]
pub use corr::Corr;
pub use euclidean::Euclidean;
#[cfg(feature = "alloc")]
pub use grassmann::Grassmann;
pub use se::{SEPoint, SETangent, SpecialEuclidean};
pub use so::SpecialOrthogonal;
#[cfg(feature = "alloc")]
pub use spd::Spd;
pub use sphere::Sphere;
