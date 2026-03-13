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

pub mod corr;
pub mod euclidean;
pub mod grassmann;
pub mod se;
pub mod so;
pub mod spd;
pub mod sphere;
pub mod util;

pub use corr::Corr;
pub use euclidean::Euclidean;
pub use grassmann::Grassmann;
pub use se::{SEPoint, SETangent, SpecialEuclidean};
pub use so::SpecialOrthogonal;
pub use spd::Spd;
pub use sphere::Sphere;
