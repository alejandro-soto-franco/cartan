//! Abstract simplicial complexes and the Hodge decomposition of edge flows.
//!
//! The complex has vertices, oriented edges and oriented triangles, with no
//! manifold structure assumed. It is the setting of a neighbour graph built on
//! cells, where `cartan-dec`'s manifold meshes do not apply.

pub mod complex;
pub mod error;
pub mod hodge;
mod solve;
pub mod triangles;

pub use complex::Complex;
pub use error::ComplexError;
pub use hodge::{Decomposition, HodgeOptions, decompose};
pub use triangles::triangles_of;
