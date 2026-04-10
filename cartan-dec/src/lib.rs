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
//! | [`mesh`] | `Mesh<M,K,B>` generic simplicial complex; `FlatMesh` = flat 2D triangular mesh |
//! | [`exterior`] | `ExteriorDerivative` — sparse d₀ (0-forms to 1-forms) and d₁ (1-forms to 2-forms) |
//! | [`hodge`] | `HodgeStar` — diagonal ⋆ operators indexed by degree |
//! | [`laplace`] | `Operators` — Laplace-Beltrami, Bochner, and Lichnerowicz Laplacians |
//! | [`advection`] | Upwind covariant advection for scalar and vector fields |
//! | [`divergence`] | Discrete covariant divergence of vector and tensor fields |
//! | [`error`] | `DecError` — error type for DEC operations |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use cartan_dec::{FlatMesh, Operators};
//! use cartan_manifolds::euclidean::Euclidean;
//! use nalgebra::DVector;
//!
//! // Build a 4x4 uniform grid on [0,1]^2.
//! let mesh = FlatMesh::unit_square_grid(4);
//! let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);
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
pub mod extrinsic;
pub mod hodge;
pub mod laplace;
pub mod line_bundle;
pub mod mesh;
pub mod mesh_gen;
pub mod mesh_quality;
pub mod stokes;

pub use advection::{
    apply_scalar_advection, apply_scalar_advection_generic, apply_vector_advection,
};
pub use divergence::{apply_divergence, apply_divergence_generic, apply_tensor_divergence};
pub use error::DecError;
pub use exterior::ExteriorDerivative;
pub use hodge::HodgeStar;
pub use laplace::Operators;
pub use mesh::{FlatMesh, Mesh};
