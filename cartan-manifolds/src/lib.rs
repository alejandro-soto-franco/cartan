// ~/cartan/cartan-manifolds/src/lib.rs

//! Concrete Riemannian manifold implementations.
//!
//! Each manifold implements the traits from cartan-core using
//! nalgebra types from cartan-nalgebra.
//!
//! ## Available manifolds (v0.1 roadmap)
//!
//! - `Euclidean<N>` -- R^N with the flat metric (trivial baseline)
//! - `Sphere<N>` -- S^{N-1} in R^N with round metric
//! - `SpecialOrthogonal<N>` -- SO(N) with bi-invariant metric
//! - `SymmetricPositiveDefinite<N>` -- SPD(N) with affine-invariant metric
//! - `Grassmann<N, K>` -- Gr(N, K) with canonical metric

pub mod euclidean;
pub mod se;
pub mod so;
pub mod sphere;
pub mod util;

pub use euclidean::Euclidean;
pub use se::SpecialEuclidean;
pub use so::SpecialOrthogonal;
pub use sphere::Sphere;
