//! # cartan-nalgebra
//!
//! nalgebra backend for the cartan Riemannian geometry library.
//!
//! This crate provides concrete implementations of cartan-core traits
//! using nalgebra's `SVector` and `SMatrix` types for statically-sized
//! linear algebra. It bridges cartan's abstract trait system to nalgebra's
//! efficient BLAS-backed operations.
//!
//! All manifold implementations in `cartan-manifolds` depend on this crate
//! for their concrete point and tangent vector storage types.
