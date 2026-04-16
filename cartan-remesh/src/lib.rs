// ~/cartan/cartan-remesh/src/lib.rs

//! # cartan-remesh
//!
//! Adaptive remeshing primitives for simplicial meshes on Riemannian manifolds.
//!
//! 2D triangle operations (`primitives`, `driver`, `lcr`) are generic over
//! `M: Manifold` and operate on `&mut Mesh<M, 3, 2>`. 3D tet operations
//! (`primitives_3d`) operate on `&mut Mesh<Euclidean<3>, 4, 3>` and are
//! limited to barycentric refinement in v1.2. Every mutation is logged in a
//! [`RemeshLog`] so that downstream solvers can interpolate fields across
//! topology changes.

pub mod config;
pub mod driver;
pub mod error;
pub mod lcr;
pub mod log;
pub mod primitives;
pub mod primitives_3d;

pub use config::RemeshConfig;
pub use driver::adaptive_remesh;
pub use driver::needs_remesh;
pub use error::RemeshError;
pub use lcr::{capture_reference_lcrs, lcr_spring_energy, lcr_spring_gradient, length_cross_ratio};
pub use log::{EdgeCollapse, EdgeFlip, EdgeSplit, RemeshLog, VertexShift};
pub use primitives::{collapse_edge, flip_edge, shift_vertex, split_edge};
pub use primitives_3d::{barycentric_refine_tets, indicator_flags, red_refine_tets_uniform, refine_to_depth};
