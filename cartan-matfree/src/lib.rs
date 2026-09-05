//! Matrix-free Galerkin Hodge mass operators and Krylov solvers.
//!
//! The assembled route builds `M_k` as a sparse matrix and factorises it. On an
//! evolving Regge background the metric changes every step, so that route pays a
//! fresh assembly and a fresh factorisation per step, and the factorisation is
//! cubic in the number of interior degrees of freedom.
//!
//! This crate replaces both with an element-by-element application. The element
//! matrices are computed once per metric and applied repeatedly by a conjugate
//! gradient iteration. A Galerkin mass matrix is spectrally equivalent to its
//! diagonal with a mesh-independent constant, so Jacobi-preconditioned CG
//! converges in a number of iterations that does not grow with the mesh, and the
//! element matrices are amortised over those iterations.
//!
//! # Layering
//!
//! [`MassBackend`] states every vector operation the iteration needs, so
//! [`pcg`] runs wherever the vectors live. [`HostMass`] is the reference
//! implementation on host memory, and it is what correctness is measured
//! against. A device backend implements the same trait and keeps its vectors
//! resident, which matters because a Krylov iteration that copies its vectors
//! across PCIe every step spends more time on the copies than on the operator.
//!
//! ```no_run
//! use cartan_matfree::{HostMass, Interior, MassBackend, pcg};
//! # fn demo(topology: &simplicial::topology::complex::Complex,
//! #         geometry: &simplicial::geometry::metric::mesh::MeshLengthsSq,
//! #         rhs: &[f64]) {
//! let interior = Interior::boundary_constrained(topology, 1);
//! let mass = HostMass::restricted(topology, geometry, &interior);
//! let mut x = vec![0.0; mass.ndofs()];
//! let report = pcg(&mass, rhs, &mut x, 1e-12, 1000);
//! assert!(report.converged);
//! # }
//! ```

mod cg;
mod gather;
mod mass;
mod restrict;

pub use cg::{pcg, CgReport, MassBackend};
pub use gather::GatherMap;
pub use mass::HostMass;
pub use restrict::Interior;
