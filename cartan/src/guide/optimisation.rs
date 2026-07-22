//! # Optimisation on manifolds
//!
//! Minimising over a manifold differs from unconstrained minimisation in two
//! places. The gradient must be projected into the tangent space, and the step
//! must move along the manifold rather than off it. `cartan-optim` handles the
//! second; you supply the first via `project_tangent`.
//!
//! ## Riemannian gradient descent
//!
//! Minimising `f(x) = -a . x` over the unit sphere drives `x` to `a / ||a||`.
//! The Euclidean gradient is the constant `-a`; the Riemannian gradient is that
//! projected onto the tangent space at `x`.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//! use cartan::optim::{minimize_rgd, RGDConfig};
//! use nalgebra::SVector;
//!
//! let s2 = Sphere::<3>;
//! let a = SVector::<f64, 3>::new(1.0, 2.0, 2.0); // norm 3
//! let target = a / a.norm();
//!
//! let cost = |x: &SVector<f64, 3>| -a.dot(x);
//! let rgrad = |x: &SVector<f64, 3>| s2.project_tangent(x, &(-a));
//!
//! let x0 = SVector::<f64, 3>::new(0.0, 0.0, 1.0);
//! let result = minimize_rgd(&s2, cost, rgrad, x0, &RGDConfig::default());
//!
//! assert!(result.converged);
//! assert!((result.point - target).norm() < 1e-6);
//! assert!((result.value + 3.0).abs() < 1e-6); // minimum is -||a||
//! ```
//!
//! `minimize_rcg` and `minimize_rtr` take the same shape of arguments.
//! Conjugate gradient needs a vector transport to combine directions from
//! different tangent spaces; the trust-region method additionally needs
//! `Connection` for Hessian-vector products.
//!
//! ## Fréchet mean
//!
//! The mean of points on a curved space is the minimiser of summed squared
//! distance. On a sphere that is not the normalised arithmetic mean, though the
//! two agree when the points are tightly clustered.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//! use cartan::optim::{frechet_mean, FrechetConfig};
//! use nalgebra::SVector;
//!
//! let s2 = Sphere::<3>;
//!
//! // Three points spread around the north pole.
//! let pts = vec![
//!     SVector::<f64, 3>::new(0.0, 0.0, 1.0),
//!     SVector::<f64, 3>::new(0.2, 0.0, 1.0).normalize(),
//!     SVector::<f64, 3>::new(0.0, 0.2, 1.0).normalize(),
//! ];
//!
//! let mean = frechet_mean(&s2, &pts, None, &FrechetConfig::default());
//! assert!(mean.converged);
//!
//! // The mean is itself a point of the manifold, which an arithmetic
//! // average would not be.
//! assert!(s2.check_point(&mean.point).is_ok());
//!
//! // It sits inside the cluster: nearer to every sample than the samples
//! // are to each other at their widest.
//! for p in &pts {
//!     assert!(s2.dist(&mean.point, p).unwrap() < 0.2);
//! }
//! ```
//!
//! ## A note on gradients
//!
//! The commonest mistake is passing a Euclidean gradient unprojected. It will
//! often still converge, because the retraction pulls the iterate back onto the
//! manifold each step, but the step direction is wrong and the convergence rate
//! degrades. `project_tangent` is cheap; use it.
