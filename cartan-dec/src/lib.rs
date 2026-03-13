//! # cartan-dec
//!
//! Discrete exterior calculus (DEC) on Riemannian manifolds.
//!
//! This crate bridges the continuous geometry of `cartan-core` to the discrete
//! operators required by PDE solvers. Given a manifold implementing
//! [`cartan_core::Manifold`], it builds a simplicial complex over a domain,
//! precomputes all static operators, and exposes them for use in time-stepping loops.
//!
//! ## Mathematical structure
//!
//! DEC represents smooth differential operators through their discrete analogues
//! on a simplicial complex. The central factorization is:
//!
//! ```text
//! Laplace-Beltrami = d * star * d * star
//! ```
//!
//! where `d_k` is the exterior derivative (a sparse {0, +1, -1} incidence matrix,
//! metric-free and fixed for a given mesh topology) and `star_k` is the Hodge star
//! (a diagonal matrix for well-centered meshes, encoding the metric geometry).
//!
//! This factorization isolates all metric dependence in diagonal `star` operators,
//! making the Laplacian a sequence of sparse matrix-vector products and diagonal
//! scalings rather than a dense or irregularly sparse stiffness matrix.
//!
//! ## Mesh requirements
//!
//! All operators in this crate assume a **well-centered mesh**: every simplex's
//! circumcenter lies strictly inside that simplex. This property guarantees that
//! all Hodge weights are positive, keeping the Laplacian positive semi-definite
//! and the discrete Hodge star diagonal. Mesh generation utilities enforce this
//! via constrained Delaunay triangulation with circumcentric duals.
//!
//! ## Cache layout
//!
//! Simplices are reordered by Hilbert space-filling curve index so that spatially
//! local simplices are adjacent in memory. Field arrays use structure-of-arrays
//! (SoA) layout: each independent tensor component occupies a contiguous
//! `Vec<f64>`, enabling SIMD vectorization and minimizing cache pressure during
//! stencil evaluation. Graph coloring of the dual mesh enables race-free parallel
//! updates via `rayon`.
//!
//! ## Operators
//!
//! ### Combinatorial (metric-free, precomputed once)
//!
//! - [`ExteriorDerivative`] -- sparse {0, +1, -1} incidence matrix `d_k` for each
//!   form degree k. Encodes the boundary operator on the simplicial complex.
//!
//! ### Metric (diagonal, depends on manifold geometry)
//!
//! - [`HodgeStar`] -- diagonal weight matrix `star_k` for each degree k. Weights
//!   are ratios of primal and dual simplex volumes, computed from the manifold
//!   metric via `cartan_core::Manifold`.
//!
//! ### Composed
//!
//! - [`LaplaceBeltrami`] -- scalar Laplacian `d star d star`, assembled from the
//!   above. Acts on 0-forms (vertex-valued scalar fields).
//!
//! - [`BochnerLaplacian`] -- connection Laplacian on tensor-valued fields.
//!   Generalizes `LaplaceBeltrami` to sections of tensor bundles using the
//!   Levi-Civita connection from `cartan_core::Connection`.
//!
//! - [`LichnerowiczLaplacian`] -- Bochner Laplacian plus a pointwise curvature
//!   correction from `cartan_core::Curvature`. Acts on symmetric 2-tensor fields
//!   and is the correct operator for the Q-tensor equation in nematohydrodynamics.
//!
//! - [`CovariantAdvection`] -- upwind covariant advection operator for
//!   tensor-valued fields transported by a vector field.
//!
//! - [`CovariantDivergence`] -- covariant divergence of a tensor field, used for
//!   body-force and stress divergence computations.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use cartan_dec::Mesh;
//! use cartan_manifolds::Euclidean;
//!
//! // Build a well-centered Delaunay mesh over a 2D domain.
//! // let mesh = Mesh::delaunay_2d(&domain_points);
//!
//! // Precompute all static operators.
//! // let ops = mesh.operators::<Euclidean<2>>();
//!
//! // Apply the Laplace-Beltrami operator to a scalar field.
//! // let lf = ops.laplace_beltrami.apply(&f);
//! ```
//!
//! ## Relation to other cartan crates
//!
//! `cartan-dec` depends on `cartan-core` for manifold traits and `cartan-manifolds`
//! for concrete manifold types. It is a pure computation layer with no I/O and no
//! application-domain knowledge. PDE solvers built on `cartan-dec` bring their own
//! field semantics and equations of motion.
