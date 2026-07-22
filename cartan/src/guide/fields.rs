//! # Fields: discrete exterior calculus
//!
//! Where [`optimisation`](super::optimisation) works with single points, this
//! layer works with functions defined over a whole mesh. `cartan-dec`
//! discretises the exterior derivative, the Hodge star and the Laplacian on a
//! simplicial complex.
//!
//! The exterior derivative is purely combinatorial: it encodes the boundary map
//! and knows nothing about the metric. The Hodge star carries all the geometry,
//! as a ratio of dual to primal volumes. Splitting them this way is what makes
//! the discretisation work on a curved mesh without extra machinery.
//!
//! ```
//! use cartan::dec::{ExteriorDerivative, HodgeStar, Operators};
//! use cartan::dec::mesh_gen::icosphere;
//! use cartan::manifolds::Sphere;
//!
//! let s2 = Sphere::<3>;
//! let mesh = icosphere(&s2, 2, true);
//!
//! let ext = ExteriorDerivative::from_mesh_sparse(&mesh);
//!
//! // d compose d = 0 holds exactly, to machine precision, because it is a
//! // statement about the topology rather than about the geometry.
//! assert!(ext.check_exactness() < 1e-13);
//! ```
//!
//! ## The Laplacian
//!
//! `Operators` assembles the scalar Laplace-Beltrami operator as
//! `star0^-1 d0^T diag(star1) d0`, the standard cotangent-weight Laplacian.
//! Constants are in its kernel, which is the discrete statement that a constant
//! function has no curvature of its own.
//!
//! ```
//! use cartan::dec::Operators;
//! use cartan::dec::mesh_gen::icosphere;
//! use cartan::manifolds::Sphere;
//! use nalgebra::DVector;
//!
//! let s2 = Sphere::<3>;
//! let mesh = icosphere(&s2, 2, true);
//! let ops = Operators::from_mesh_generic(&mesh, &s2).unwrap();
//!
//! let nv = mesh.n_vertices();
//! let constant = DVector::<f64>::from_element(nv, 3.5);
//! let lap = ops.apply_laplace_beltrami(&constant);
//!
//! assert!(lap.amax() < 1e-10, "constants lie in the kernel");
//! ```
//!
//! ## Sparse storage
//!
//! Operators are `nalgebra_sparse::CscMatrix`. Column-major storage suits the
//! matvec used throughout, which accumulates `y += A[:, j] * x[j]`. The same
//! type is used by `cartan-io` and `cartan-maxwell`, so the layers compose
//! without conversion.
