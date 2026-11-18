//! Levi-Civita geometry of a sampled submanifold.
//!
//! The input is a point cloud in `R^D` sampling a submanifold of dimension `d`.
//! Each point gets a tangent frame from local principal components. Along an
//! edge of the mutual nearest-neighbour graph, the frame of one point is carried
//! to the tangent plane of the other by the Grassmann geodesic. In frame
//! coordinates that transport is the orthogonal Procrustes rotation
//! `polar(T_i^T T_j)`, an element of `O(d)` and the discrete Levi-Civita
//! connection of the cloud.
//!
//! On this connection the crate computes:
//!
//! - the holonomy of every triangle of the graph, read as sectional curvature
//!   on the triangle's plane ([`plaquette`]);
//! - a symmetric curvature operator per point, with scalar and Ricci curvature
//!   ([`plaquette::curvature_operator`]);
//! - the covariant derivative `A = nabla v` of a vector field, either fitted
//!   point by point or as one section of `End(TM)` regularised by the
//!   connection Laplacian ([`jet`]).
//!
//! Neighbour search is exact. Candidates come from blocked matrix products and
//! are re-ranked on exact distances, ties to the lower index.

pub mod connection;
pub mod frames;
pub mod graph;
pub mod jet;
pub mod knn;
pub mod plaquette;

pub use connection::{Connection, transport};
pub use graph::MutualGraph;
pub use jet::{JetFit, JetOptions, LocalJet, fit_jet, local_jet};
pub use plaquette::{CurvatureOperator, TriangleGeometry, curvature_operator, triangle_geometry};
