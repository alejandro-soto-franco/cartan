//! # Paths: stochastic development
//!
//! Brownian motion on a manifold cannot be defined by adding Gaussian noise to
//! coordinates, because the result leaves the manifold and depends on the chart.
//! The Eells-Elworthy-Malliavin construction instead runs the noise in a flat
//! tangent space and rolls the manifold along it, carrying an orthonormal frame
//! by parallel transport. The frame is what makes successive increments
//! comparable.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//! use cartan::stochastic::{random_frame_at, stratonovich_step};
//!
//! let s2 = Sphere::<3>;
//! let mut rng = rand::rng();
//!
//! let p0 = s2.random_point(&mut rng);
//! let frame = random_frame_at(&s2, &p0, &mut rng).unwrap();
//!
//! // One Stratonovich step: a 2-dimensional increment, since S^2 is
//! // 2-dimensional, developed onto the sphere.
//! let dw = [0.1, -0.05];
//! let (p1, _frame1) = stratonovich_step(&s2, &p0, &frame, &dw, 0.01, 1e-12).unwrap();
//!
//! // The walker is still on the sphere, which is the point of the construction.
//! assert!(s2.check_point(&p1).is_ok());
//! ```
//!
//! ## Why the frame is carried along
//!
//! Each step returns a new frame as well as a new point. Discarding it and
//! drawing a fresh one would break the martingale property, because successive
//! increments would no longer be expressed in a consistently transported basis.
//! The returned frame is the parallel transport of the old one along the step.
//!
//! This machinery is the foundation for Bismut-Elworthy-Li derivative formulae,
//! where a gradient of an expectation is rewritten as an expectation against a
//! stochastic weight, with no derivative of the payoff required.
