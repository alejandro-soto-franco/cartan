// ~/cartan/cartan-dec/src/lib.rs

//! # cartan-dec
//!
//! Discrete exterior calculus (DEC) on Riemannian manifolds.
//!
//! Bridges continuous geometry (`cartan-core`) to discrete operators for PDE
//! solvers on simplicial meshes. All metric information flows through the
//! Hodge star; topology is encoded in the metric-free exterior derivative.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`mesh`] | `Mesh` — 2D simplicial complex with vertices, edges, triangles |
//! | [`exterior`] | `ExteriorDerivative` — d₀ (V→E) and d₁ (E→T) incidence matrices |
//! | [`hodge`] | `HodgeStar` — diagonal ⋆₀, ⋆₁, ⋆₂ from primal/dual volumes |
//! | [`laplace`] | `Operators` — Laplace-Beltrami, Bochner, and Lichnerowicz Laplacians |
//! | [`advection`] | Upwind covariant advection for scalar and vector fields |
//! | [`divergence`] | Discrete covariant divergence of vector and tensor fields |
//! | [`error`] | `DecError` — error type for DEC operations |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use cartan_dec::{Mesh, Operators};
//! use nalgebra::DVector;
//!
//! // Build a 4×4 uniform grid on [0,1]².
//! let mesh = Mesh::unit_square_grid(4);
//! let ops = Operators::from_mesh(&mesh);
//!
//! // Apply the scalar Laplacian to a vertex field.
//! let f = DVector::from_element(mesh.n_vertices(), 1.0);
//! let lf = ops.apply_laplace_beltrami(&f);
//! ```
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341, 2005.
//! - Hirani. "Discrete Exterior Calculus." Caltech PhD thesis, 2003.

pub mod advection;
pub mod divergence;
pub mod error;
pub mod exterior;
pub mod hodge;
pub mod laplace;
pub mod mesh;

pub use advection::{apply_scalar_advection, apply_vector_advection};
pub use divergence::{apply_divergence, apply_tensor_divergence};
pub use error::DecError;
pub use exterior::ExteriorDerivative;
pub use hodge::HodgeStar;
pub use laplace::Operators;
pub use mesh::Mesh;
