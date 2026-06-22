//! Runtime dimension-generic simplicial topology and intrinsic Regge geometry.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.
//!
//! Build a mesh in any dimension, extract its intrinsic edge lengths, and
//! confirm the discrete boundary chain satisfies d circ d = 0.
//!
//! ```
//! use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;
//! let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
//! let lengths = coords.to_edge_lengths(&complex);
//! assert!(lengths.is_coordinate_realizable(complex.cells()));
//! let chain = complex.boundary_chain();
//! let prod = &chain[0] * &chain[1];
//! let max = prod.data().iter().fold(0.0f64, |m, &v| m.max(v.abs()));
//! assert!(max < 1e-12);
//! ```

pub use cartan_exterior::Dim;

pub mod linalg;
pub mod affine;
pub mod topology;
pub mod geometry;
pub mod r#gen;
