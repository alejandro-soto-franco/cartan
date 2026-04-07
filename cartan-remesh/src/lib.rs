// ~/cartan/cartan-remesh/src/lib.rs

//! # cartan-remesh
//!
//! Adaptive remeshing primitives for triangle meshes on Riemannian manifolds.
//!
//! All operations are generic over `M: Manifold` and operate on
//! `&mut Mesh<M, 3, 2>`. Every mutation is logged in a [`RemeshLog`] so that
//! downstream solvers can interpolate fields across topology changes.

pub mod config;
pub mod driver;
pub mod error;
pub mod lcr;
pub mod log;
pub mod primitives;

pub use config::RemeshConfig;
pub use driver::adaptive_remesh;
pub use driver::needs_remesh;
pub use error::RemeshError;
pub use lcr::{capture_reference_lcrs, lcr_spring_energy, lcr_spring_gradient, length_cross_ratio};
pub use log::{EdgeCollapse, EdgeFlip, EdgeSplit, RemeshLog, VertexShift};
pub use primitives::{collapse_edge, flip_edge, shift_vertex, split_edge};
