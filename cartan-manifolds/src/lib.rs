//! # cartan-manifolds
//!
//! Concrete Riemannian manifold implementations for the cartan library.
//!
//! This crate provides ready-to-use manifolds that implement the `Manifold`
//! trait from `cartan-core`. Each manifold includes geodesically-exact
//! exp/log maps, Riemannian inner products, tangent space projections,
//! and validated implementations of all trait methods.
//!
//! ## Available manifolds (v0.1 roadmap)
//!
//! - `Euclidean<N>` -- R^N with the flat metric (trivial baseline)
//! - `Sphere<N>` -- S^{N-1} in R^N with round metric
//! - `SpecialOrthogonal<N>` -- SO(N) with bi-invariant metric
//! - `SymmetricPositiveDefinite<N>` -- SPD(N) with affine-invariant metric
//! - `Grassmann<N, K>` -- Gr(N, K) with canonical metric
