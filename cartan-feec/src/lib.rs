//! Finite element exterior calculus on simplicial meshes.
//! Ported from luiswirth/formoniq (used with permission), adapted for cartan.
//!
//! Assemble the Galerkin Whitney Hodge mass on 1-forms over a 2D mesh; the
//! same code assembles grade-k mass in any dimension.
//!
//! ```
//! use cartan_feec::assemble::assemble_galmat;
//! use cartan_feec::operators::HodgeMassElmat;
//! use cartan_simplicial::r#gen::cartesian::CartesianMeshInfo;
//!
//! let (complex, coords) = CartesianMeshInfo::new_unit(2, 2).compute_coord_complex();
//! let geom = coords.to_edge_lengths(&complex);
//! let m1 = assemble_galmat(&complex, &geom, HodgeMassElmat::new(2, 1));
//! assert_eq!(m1.rows(), complex.nsimplices(1)); // one DOF per edge
//! assert_eq!(m1.rows(), m1.cols());
//! ```

pub mod cochain;
pub mod operators;
pub mod assemble;
pub mod whitney;
pub mod eigen;
