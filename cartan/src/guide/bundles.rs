//! # Bundles: fields with an internal symmetry
//!
//! A scalar field assigns a number to each vertex. A section of a bundle
//! assigns an element of some fibre, and comparing values at neighbouring
//! vertices then requires a connection. This is the layer that distinguishes
//! cartan from a flat finite-element library.
//!
//! The motivating case is a k-atic order parameter. A nematic director has no
//! head or tail, so it is defined only up to a rotation by pi. Representing it
//! as a section of a complex line bundle with charge `K` makes that ambiguity
//! structural rather than something the caller must remember.
//!
//! ```
//! use cartan::dec::line_bundle::Section;
//!
//! // A charge-2 section: the nematic case, invariant under rotation by pi.
//! // Built from real components, so no complex type is needed at the boundary.
//! let q1 = vec![1.0, 0.0, 0.6];
//! let q2 = vec![0.0, 1.0, 0.8];
//! let mut s = Section::<2>::from_real_components(&q1, &q2);
//!
//! assert_eq!(s.n_vertices(), 3);
//!
//! // Every sample above is already a unit vector.
//! assert!((s.mean_norm() - 1.0).abs() < 1e-12);
//!
//! // The scalar order parameter follows the nematic convention S = 2|z|,
//! // so perfect alignment reads as 2 rather than 1.
//! assert!((s.mean_scalar_order() - 2.0).abs() < 1e-12);
//!
//! // Normalising projects each fibre element back to the unit circle,
//! // so on already-unit data it is a no-op.
//! s.normalise(1e-12);
//! assert!((s.mean_norm() - 1.0).abs() < 1e-12);
//! ```
//!
//! ## Defects are a topological count
//!
//! A section of a line bundle over a closed surface cannot be nowhere-zero
//! unless the bundle is trivial. The zeros are the defects, and their total
//! charge is fixed by the Euler characteristic rather than by the dynamics.
//! For a sphere, chi = 2, so a nematic must carry total charge 2, usually as
//! four +1/2 defects. `defect_charges` computes the per-face winding, and its
//! sum is a check on the discretisation itself.
//!
//! See `cartan-dec/examples/weitzenbock_sphere_gap.rs` for the spectral
//! counterpart: the Weitzenbock identity relating the Bochner and Hodge
//! Laplacians, whose gap on the sphere is set by the same curvature.
