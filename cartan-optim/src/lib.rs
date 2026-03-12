//! # cartan-optim
//!
//! Riemannian optimization algorithms for the cartan library.
//!
//! This crate implements first- and second-order optimization algorithms
//! that operate on any manifold implementing the `Manifold` trait from
//! `cartan-core`. Algorithms use the manifold's exp/retract for stepping,
//! project_tangent for gradient conversion, and inner product for norms.
//!
//! ## Planned algorithms (v0.1 roadmap)
//!
//! - `RiemannianGradientDescent` -- steepest descent with Armijo line search
//! - `RiemannianConjugateGradient` -- Fletcher-Reeves and Polak-Ribiere variants
//! - `RiemannianTrustRegion` -- second-order method using Connection trait
//! - `FrechetMean` -- iterative Riemannian mean (Karcher flow)
//!
//! ## References
//!
//! - Absil, Mahony, Sepulchre. "Optimization Algorithms on Matrix Manifolds."
//!   Princeton, 2008.
//! - Boumal. "An Introduction to Optimization on Smooth Manifolds."
//!   Cambridge, 2023.
