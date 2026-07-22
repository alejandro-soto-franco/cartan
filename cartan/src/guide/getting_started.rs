//! # Getting started
//!
//! Every manifold implements [`Manifold`](crate::traits::Manifold), which
//! supplies `exp`, `log`, `dist`, `inner` and `project_tangent`. Manifold types
//! are zero-sized: the geometry lives in the trait impls, so `Sphere::<3>` costs
//! nothing to hold or pass around.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//!
//! let s2 = Sphere::<3>; // the 2-sphere embedded in R^3
//!
//! let mut rng = rand::rng();
//! let p = s2.random_point(&mut rng);
//!
//! // Scaled inside the injectivity radius, which is pi on the unit sphere.
//! // `random_tangent` draws a projected standard normal, so its norm is
//! // unbounded; past pi the geodesic runs beyond the antipode and `log`
//! // returns the shorter way back, so the two stop being inverse.
//! let v = s2.random_tangent(&p, &mut rng).normalize() * 1.2;
//!
//! // Walk along the geodesic leaving p with velocity v.
//! let q = s2.exp(&p, &v);
//!
//! // Recover the velocity that carries p to q.
//! let v_back = s2.log(&p, &q).unwrap();
//! assert!((v - v_back).norm() < 1e-10);
//!
//! // Geodesic distance is the length of that velocity.
//! let d = s2.dist(&p, &q).unwrap();
//! assert!((d - s2.norm(&p, &v)).abs() < 1e-10);
//!
//! // Sectional curvature needs two independent tangents to span a plane.
//! // Every such plane on the unit sphere has K = 1.
//! let w = s2.random_tangent(&p, &mut rng);
//! assert!((s2.sectional_curvature(&p, &v, &w) - 1.0).abs() < 1e-9);
//! ```
//!
//! ## exp and log are inverses, within the injectivity radius
//!
//! `log` returns a [`Result`] because it fails past the cut locus, where the
//! minimising geodesic stops being unique. On a sphere that is the antipode:
//! every great circle through `p` reaches `-p` in the same distance, so there is
//! no single answer to return.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//! use nalgebra::SVector;
//!
//! let s2 = Sphere::<3>;
//! let north = SVector::<f64, 3>::new(0.0, 0.0, 1.0);
//! let south = SVector::<f64, 3>::new(0.0, 0.0, -1.0);
//!
//! // Antipodal points sit exactly on the cut locus.
//! assert!(s2.log(&north, &south).is_err());
//!
//! // Anywhere nearer than that, log succeeds.
//! let equator = SVector::<f64, 3>::new(1.0, 0.0, 0.0);
//! assert!(s2.log(&north, &equator).is_ok());
//! ```
//!
//! ## Parallel transport
//!
//! Moving a tangent vector between points requires a connection, since tangent
//! spaces at different points are different vector spaces. Transport preserves
//! inner products, which is the defining property of the Levi-Civita connection.
//!
//! ```
//! use cartan::prelude::*;
//! use cartan::manifolds::Sphere;
//!
//! let s2 = Sphere::<3>;
//! let mut rng = rand::rng();
//!
//! let p = s2.random_point(&mut rng);
//! let q = s2.random_point(&mut rng);
//! let u = s2.random_tangent(&p, &mut rng);
//! let v = s2.random_tangent(&p, &mut rng);
//!
//! let u_q = s2.transport(&p, &q, &u).unwrap();
//! let v_q = s2.transport(&p, &q, &v).unwrap();
//!
//! // Transport is an isometry between tangent spaces.
//! let before = s2.inner(&p, &u, &v);
//! let after = s2.inner(&q, &u_q, &v_q);
//! assert!((before - after).abs() < 1e-9);
//! ```
//!
//! ## Where to go next
//!
//! - [`manifolds`](super::manifolds) for the catalogue and how to choose.
//! - [`optimisation`](super::optimisation) to minimise a function on a manifold.
