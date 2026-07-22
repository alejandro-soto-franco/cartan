//! # Interop: export and bindings
//!
//! `cartan-io` writes meshes and fields as VTK `UnstructuredGrid` files for
//! ParaView, and as MDD caches for Blender. The writer is dimension-generic, so
//! the same call handles a triangle mesh in R^3 and a tetrahedral mesh in R^4.
//!
//! ```
//! use cartan::io::Field;
//!
//! // Fields carry their own name and arity, so the writer needs no schema.
//! let temperature = Field::Scalar {
//!     name: "temperature".into(),
//!     values: vec![1.0, 2.0, 3.0],
//! };
//!
//! // A director is a vector that has no head or tail. Marking it nematic
//! // tells the renderer to draw it double-ended rather than as an arrow.
//! let director = Field::Vector {
//!     name: "director".into(),
//!     values: vec![1.0, 0.0, 0.0],
//!     nematic: true,
//! };
//!
//! match temperature {
//!     Field::Scalar { ref name, ref values } => {
//!         assert_eq!(name, "temperature");
//!         assert_eq!(values.len(), 3);
//!     }
//!     _ => unreachable!(),
//! }
//! let _ = director;
//! ```
//!
//! ## Frames
//!
//! One thing to know before reading exported vectors back: interpolated field
//! values come out in the **ambient** frame, not in a per-simplex local frame.
//! VTK has no notion of a chart, so a vector must be expressed in the embedding
//! space to be drawn correctly. If you need intrinsic components, convert after
//! reading rather than expecting the file to hold them.
//!
//! ## Recording a run
//!
//! `cartan_io::run` writes a directory of timesteps plus a `.pvd` index, which
//! ParaView opens as a single time series. `cartan-maxwell` uses this for its
//! evolving-background Maxwell runs; see `cartan-maxwell/examples/maxwell_record.rs`.
//!
//! ## Python
//!
//! `cartan-py` exposes the manifolds, optimisers, DEC operators and stochastic
//! primitives through PyO3, with numpy interop on every array boundary. Sparse
//! operators are densified at the boundary, so `ext.d0` hands back a plain
//! `ndarray` rather than a scipy matrix.
//!
//! ```text
//! pip install cartan
//! ```
