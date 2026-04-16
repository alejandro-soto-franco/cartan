# Changelog

All notable changes to cartan are documented here.

---

## [Unreleased]

### Added

- **cartan-homog** (new crate): mean-field and full-field homogenisation of
  random media on SPD manifolds, generic over tensor order (Order2 = 3×3
  conductivity, Order4 = 6×6 Kelvin-Mandel stiffness).
  - `TensorOrder` trait with `spd_geodesic_step` delegating to `cartan-manifolds::Spd<N>`.
  - Shape catalog: `Sphere`, `Spheroid`, `PennyCrack`, `Ellipsoid` (Carlson RD),
    `SphereNLayers` (Herve-Zaoui). `UserInclusion = Arc<dyn Shape<O>>`.
  - 10 schemes matching ECHOES's feature set: `VoigtBound`, `ReussBound`,
    `Dilute`, `DiluteStress`, `MoriTanaka`, `SelfConsistent` (SPD-geodesic
    fixed-point), `AsymmetricSc`, `Maxwell`, `PonteCastanedaWillis`, `Differential`.
  - `--features stochastic`: `WishartRveEnsemble` with Karcher-mean aggregation
    via affine-invariant SPD metric.
  - `--features full-field` (β scaffold): `FullField<O>`, `PeriodicCubeMeshBuilder`,
    voxelize/mesh/cell_problem/solver module tree, `reliability_indicator_order2`.
    Full DEC cell-problem assembly is v1.1.

- **cartan-homog-valid** (new crate, unpublished): ECHOES-backed numerical
  validation harness.
  - Fixture loader with `CARTAN_HOMOG_FIXTURES_DIR` env var for external sets
    (default = in-repo `fixtures/basic/`).
  - `assert_spd_close_o2!` / `assert_spd_close_o4!` macros using affine-invariant
    SPD distance with 4-tier tolerance ladder (exact/tight/iterative/qsens).
  - Python generator (`cartan-homog-valid/python/generate_fixtures.py`) drives
    ECHOES via its wheel (Zenodo DOI 10.5281/zenodo.14959866) and emits NPZ + JSON
    meta pairs per test case.
  - 8-case committed basic fixture set (3 fractions × 7 schemes × 2 orders for
    iso-matrix spheres subset), 42-case extended set at
    `/run/media/alejandrosotofranco/ASF-EX2/cartan/homog-fixtures/v1/`.
  - Integration test: all 8 basic cases agree with ECHOES to `d_AI < 2.5e-15`.
  - Capstone fractured-sandstone pipeline test: 7 depths, Mori-Tanaka with
    depth-varying penny-crack density, crack-induced anisotropy verified,
    JSON report emitted.

### Changed

- Workspace members include `cartan-homog` and `cartan-homog-valid`.
- `cartan/Cargo.toml` version pins bumped 0.4 → 0.5 to match workspace.

---

## [0.5.0] - 2026-04-14

### Added

- **cartan-stochastic** (new crate): stochastic analysis on Riemannian manifolds.
  Foundation for the downstream `hsu` / `bismut` / `elworthy` stack and any
  Bismut-Elworthy-Li Greeks work built on cartan.
  - `OrthonormalFrame<M>` and `random_frame_at`: orthonormal frames over any
    `Manifold` with modified Gram-Schmidt re-orthonormalisation against the
    Riemannian metric.
  - `horizontal_velocity`: `R^n → T_pM` lift via a frame.
  - `stratonovich_step` and `stochastic_development`: the Eells-Elworthy-Malliavin
    construction of Brownian motion on a manifold via the orthonormal frame bundle.
    Available for any manifold implementing `Manifold + Retraction + VectorTransport`
    through the blanket `StratonovichDevelopment` marker trait.
  - `wishart_step`: Itô-Euler step of the Wishart SPD diffusion.
  - Validated against the S² heat kernel: `E[z_T] ≈ e^{-T}` recovered to Monte
    Carlo precision at 400 paths.

- **cartan-manifolds**: `SpdBuresWasserstein<N>` — the SPD cone with the
  Bures-Wasserstein (optimal-transport) metric, an alternative to the existing
  affine-invariant `Spd<N>`. Coincides with the L²-Wasserstein metric on centred
  Gaussian measures.
  - Manifold impl with closed-form `inner`, `exp`, `log` via a Lyapunov solver
    in the P-eigenbasis.
  - Retraction delegates to `exp` (Cartan-Hadamard-like global completeness).
  - `VectorTransport` via the Fréchet derivative of the retraction — enough
    structure to plug into `stochastic_development`. Exact `ParallelTransport`
    and `Curvature` impls are deferred until an elworthy/bismut consumer reads them.
  - Standalone `bw_distance_sq()` helper for the closed-form 2-Wasserstein
    squared distance between centred Gaussians.

- **cartan-geo**: `integrate_jacobi_along_path` — RK4 Jacobi-field integration
  along arbitrary `C²` base curves, not just geodesics. Velocity is reconstructed
  per step via `log`. Enables Jacobi-along-SDE-path computation directly on
  trajectories produced by `stochastic_development`.

- **cartan-py**: Python bindings for the L1 stochastic primitives.
  - `cartan.stochastic_bm_on_sphere(intrinsic_dim, p0, n_steps, dt, seed)` for
    sphere dimensions 1..=9.
  - `cartan.stochastic_bm_on_spd(n, p0, n_steps, dt, seed)` for SPD dimensions
    2..=5 (affine-invariant metric).
  - `cartan.wishart_step(x, shape_param, dt, seed)` for SPD dimensions 2..=5.

- **benchmarks**: `cartan-bench` gains stochastic-primitive benchmarks
  (`horizontal_velocity`, `stratonovich_step`, `stochastic_development_32`,
  `wishart_step`) across sphere dims 2..50 and SPD dims 2..5. JSON-line
  output rendered on `cartan.sotofranco.dev/performance/stochastic`.

- **docs**: `STACK.md` at the repo root lays out the cartan → hsu → bismut /
  elworthy → malliavin layer architecture with placement rules. `cartan-docs`
  gains a full `/stochastic/*` section plus `/manifolds/spd-bures-wasserstein`
  and `/performance/stochastic` pages.

### Fixed

- **no_std build**: `cartan-core::bundle` and the `VecSection` impl now gate
  behind `feature = "alloc"`, which they always required but previously
  failed to declare. The bare-`no_std` CI job compiles cleanly.
- **clippy**: cleaned up pre-existing `needless_range_loop` warnings in
  `cartan-core::fiber` and `cartan-dec::levi_civita` so the
  `cargo clippy --workspace -- -D warnings` CI gate passes.

### Changed

- All 7 workspace crates bumped to 0.5.0.
- `cartan-py` bumped to 0.5.0 on PyPI; `__version__` now reports the crate
  version correctly (previously hard-coded to the initial 0.1.7 placeholder).

---

## [0.4.0] - 2026-04-11

### Added

- **cartan-core**: fiber bundle traits for covariant field transport on simplicial meshes.
  - `Fiber` trait: abstract fiber type with SO(d) representation map `transport_by`.
  - `FiberOps` trait: component-wise arithmetic for generic covariant Laplacian.
  - `Section` / `VecSection`: fiber element per mesh vertex.
  - `DiscreteConnection<D>` trait: SO(D) frame transport per edge.
  - `EdgeTransport2D` / `EdgeTransport3D`: concrete SO(2)/SO(3) storage.
  - `CovLaplacian`: generic covariant Laplacian over any `FiberOps + DiscreteConnection`. Positive-semidefinite (DEC convention).
  - `U1Spin2` fiber: nematic on 2-manifolds (spin-2 phase rotation).
  - `TangentFiber<D>` fiber: R^D vector with fundamental SO(D) representation.
  - `NematicFiber3D` fiber: traceless symmetric 3x3, 5 components (Q -> R Q R^T).
- **cartan-dec**: `levi_civita_2d()` builds `EdgeTransport2D` from triangle mesh geometry via `ConnectionAngles`. Verified against `BochnerLaplacian<2>` on icosphere.
- **cartan-dec**: `cartesian_3d_connection()` builds `EdgeTransport3D` + `CovLaplacian` for periodic Cartesian grids with SO(3) identity transport.

---

## [0.3.0] - 2026-04-10

### Added

- **cartan-dec**: `line_bundle` module with `Section<K>` (complex section of L_k), `ConnectionAngles` (discrete Levi-Civita on primal and dual edges), `BochnerLaplacian<K>` (sparse Hermitian Laplacian on L_k), and `defect_charges` (exact discrete Poincare-Hopf topological charge).
- **cartan-dec**: extrinsic operators module: `KillingOperator`, `ExtrinsicDiv`, `ExtrinsicGrad`, and viscosity Laplacian for surface Stokes problems.
- **cartan-dec**: augmented Lagrangian Stokes solver on triangle meshes with Killing vector projection.
- **cartan-dec**: circumcentric Hodge star for well-centered meshes.
- **cartan-dec**: torus mesh generator with well-centered option.

---

## [0.2.0] - 2026-04-08

### Added

- **cartan-remesh** (new crate): adaptive remeshing primitives for triangle meshes on Riemannian manifolds. Five primitive operations (`split_edge`, `collapse_edge`, `flip_edge`, `shift_vertex`), all generic over `M: Manifold`. `collapse_edge` includes a foldover guard via signed-area orientation check. Length-cross-ratio (LCR) conformal regularization with reference snapshot, spring energy, and gradient placeholder. `adaptive_remesh` driver with curvature-CFL split criterion and foldover-guarded collapse pass. `needs_remesh` predicate. `RemeshLog` records all mutations for downstream field interpolation. `RemeshConfig` with curvature scale, edge length bounds, area bounds, and smoothing parameters.
- **cartan-dec**: sparse `ExteriorDerivative` via `sprs::CsMat<f64>` (replaces dense `nalgebra::DMatrix`). K-generic const-generic operators: `ExteriorDerivative`, `HodgeStar`, and `Operators` are now parameterised over `<M: Manifold, const K: usize, const B: usize>`. New `from_mesh_generic` constructors on `HodgeStar` and `Operators` for any manifold (not just flat `Euclidean<2>`).
- **cartan-dec**: K-generic geometric primitives on `Mesh<M, K, B>`: `simplex_volume`, `boundary_volume`, `simplex_circumcenter`, `boundary_circumcenter` via Gram determinant and tangent-space equidistance system. `from_simplices_generic` constructor for arbitrary K. Dense linear algebra helpers (`dense_determinant`, `dense_solve`, `permutation_sign`).
- **cartan-dec**: K-generic `apply_scalar_advection_generic` and `apply_divergence_generic` using tangent-vector velocity fields and adjacency-map traversal (O(V * avg_degree)).
- **cartan-dec**: `rebuild_topology` on `Mesh<M, K, B>` for full boundary/sign/adjacency reconstruction after topology-changing mutations (split, collapse, flip).
- **cartan-dec**: K=4 tetrahedral mesh support via `from_simplices_generic` (adjacency, volume, circumcenter verified; exactness requires full chain complex).

### Changed

- **cartan-dec**: `HodgeStar` storage changed from three separate `pub star0/star1/star2: DVector<f64>` fields to `pub star: Vec<DVector<f64>>` indexed by degree. Backward-compatible accessors `star0()`, `star1()`, `star2()` and `star_k(k)` / `star_k_inv(k)` provided.
- **cartan-dec**: `Operators` struct now has const generics `<M, K, B>` (default `<Euclidean<2>, 3, 2>` for backward compatibility). `mass` field is `Vec<DVector<f64>>` (replaces `mass0`/`mass1`; backward aliases retained).
- **cartan-dec**: `apply_scalar_advection` and `apply_divergence` are now thin wrappers around their generic counterparts. The old DVector-layout APIs are preserved.
- **cartan-dec**: `ExteriorDerivative::from_mesh_sparse_generic` is now `pub` (was private).

### Fixed

- **cartan-dec**: CSC sparse matrix-vector product in `apply_laplace_beltrami` was iterating `outer_iterator()` as rows instead of columns (CSC format). Fixed to correct column-major traversal.
- **cartan-dec**: CSR transpose-view matvec in divergence was swapping row/column indices. Fixed to standard CSR row iteration.

---

## [0.1.7] - 2026-03-29

### Added

- **Python bindings** (`pip install cartan`): full PyO3 bindings exposing every manifold, optimizer, geodesic tool, and DEC operator to Python 3.9+ with zero-copy numpy interop. A single abi3 wheel covers all supported Python versions.
  - Manifolds: `Euclidean(n)`, `Sphere(n)`, `SO(n)`, `SE(n)`, `SPD(n)`, `Grassmann(n,k)`, `Corr(n)`, `QTensor3`, `FrameField3D`
  - Optimization: `RGD`, `RCG`, `RTR`, `FrechetMean`
  - Geodesics: `Geodesic`, `CurvatureQuery`, Jacobi field integration
  - DEC: `Mesh`, `Operators`, advection, divergence
  - Holonomy: disclination scanning, winding number computation
  - Batch operations: `dist_matrix`, `exp_batch`
  - Hypothesis property-based tests for all manifold bindings
- CI: Python bindings test workflow (pytest across Python 3.9-3.13, wheel smoke test)
- CI: PyPI publish workflow via OIDC trusted publishing on `cartan-py-v*` tags
- Published to PyPI as `cartan` (not `cartan-py`): https://pypi.org/project/cartan/

### Fixed

- `Sphere::dist`: replaced `arccos(p . q)` with the numerically stable half-chord formula `2 * asin(||p - q|| / 2)`. The old formula suffered catastrophic cancellation near p == q, returning ~1.5e-8 instead of 0.
- `Sphere::dist` at antipodes: previous test tolerance was too tight for the haversine path; corrected.
- `Grassmann`: subspace equality test used raw matrix comparison instead of principal-angle check.
- `S^1` curvature test: expected value corrected.

---

## [0.1.6] - 2026-03-25

### Added

- **cartan-py** crate: PyO3 scaffold, maturin build system, numpy conversion helpers, dimension dispatch macros, Python exception hierarchy mapping `CartanError`/`DecError`.

### Changed

- `cartan-py` excluded from the Rust workspace (PyO3 cdylib with `extension-module` cannot link as a standalone test binary). Uses its own standalone `[workspace]` table.

### Note

v0.1.6 was partially uploaded to PyPI and superseded by v0.1.7. Use v0.1.7.

---

## [0.1.5] - 2026-03-25

### Added

- `cartan` facade: `full` feature (default) makes `cartan-dec` optional. Disabling it (`default-features = false, features = ["alloc"]`) gives a no_std-compatible facade that excludes the mesh/PDE layer. `full` implies `std`; `std` implies `alloc`.
- `cartan` facade: `std` and `alloc` feature flags that propagate to all sub-crates, so embedded users no longer need to list sub-crates individually.

### Fixed

- README: removed em dashes (style consistency).
- `cartan` facade: `full = ["dep:cartan-dec"]` now also implies `std`, fixing unused-import warnings in `cartan-geo/src/jacobi.rs` that appeared when the facade was built without explicit std propagation.

---

## [0.1.4] - 2026-03-25

### Added

- **no_std + alloc support** for `cartan-manifolds`, `cartan-geo`, and `cartan-optim`. All four sub-crates (cartan-core, cartan-manifolds, cartan-geo, cartan-optim) now compile with `default-features = false, features = ["alloc"]` on any target with a global allocator. Float arithmetic uses `libm` when `std` is absent.
- Available without `std`: `Euclidean<N>`, `Sphere<N>`, `SpecialOrthogonal<N>`, `SpecialEuclidean<N>`, `Grassmann<N,K>`, all cartan-optim algorithms, `Geodesic`, `CurvatureQuery`, `integrate_jacobi`.
- `nalgebra/libm` feature enabled unconditionally in cartan-manifolds and cartan-geo so that `f64: RealField` is satisfied in no_std mode without a separate simba dependency.
- `ComplexField` and `RealField` trait imports in source files gated behind `#[cfg(not(feature = "std"))]` so they do not produce unused-import warnings in std builds.
- CI: `no-std-check` job added, building cartan-core (bare no_std), cartan-manifolds, and cartan-geo under `--no-default-features --features alloc` on each PR, pinned to Rust 1.85.
- README: dedicated **Embedded and no_std Targets** section with feature tier tables, correct `Cargo.toml` snippets for embedded users, and an SO(3) attitude control example.

### Changed

- `cartan-manifolds`: `qtensor`, `frame_field`, `corr`, `spd` modules reclassified from `alloc`-gated to `std`-gated. These modules require `symmetric_eigen()` (Jacobi iteration) which depends on std float behavior. `grassmann` remains `alloc`-gated since it uses `DMatrix`/SVD but not eigendecomposition.
- `cartan-manifolds/util`: `sym` module reclassified from `alloc`-gated to `std`-gated for the same reason.
- `cartan-geo`: `disclination` and `holonomy` modules reclassified from `alloc`-gated to `std`-gated. Both depend on `std::collections` (HashMap, HashSet, VecDeque) and `frame_field::d2_gauge_fix` which is itself std-gated.
- `cartan-manifolds/Cargo.toml`, `cartan-geo/Cargo.toml`: `nalgebra/std` and `nalgebra/alloc` are now threaded through the respective crate-level feature flags rather than being hardcoded on the dependency line.
- CI: `test` job toolchain pinned to `dtolnay/rust-toolchain@1.85` for consistency with the new `no-std-check` job.

### Not changed

- `cartan-dec` has no no_std support and is not expected to gain it. It depends on rayon, thiserror, and serde and operates on heap-allocated mesh structures. Embedded users should not depend on `cartan-dec`.
- The `cartan` facade crate unconditionally re-exports `cartan-dec` and therefore requires std. **Embedded and no_std users must depend on the sub-crates directly** (see README).

---

## [0.1.2] - 2026-03-17

### Added

- `cartan-geo`: no_std/alloc feature flags. `Geodesic::sample` and `JacobiResult`/`integrate_jacobi_field` are now gated behind `#[cfg(feature = "alloc")]`. `Geodesic::sample_fixed<const N>` added for the no_alloc tier.
- `[package.metadata.docs.rs] all-features = true` added to all six crates so docs.rs builds feature-gated items.

### Changed

- `cartan-manifolds`: Weingarten correction in `Connection` impl for `Sphere<N>`, `SpecialOrthogonal<N>`, `Grassmann<N,K>`, and `SpecialEuclidean<N>`. `riemannian_hessian_vector_product` now returns the correct Riemannian Hessian-vector product (not the ambient Euclidean one). `Spd<N>` uses the full affine-invariant Christoffel formula -- no Weingarten correction needed.
- `cartan-dec`: Bochner and Lichnerowicz Laplacian curvature corrections upgraded from scalar `Option<f64>` to vertex-indexed tensor callbacks (`Option<&dyn Fn(usize) -> [[f64; 2]; 2]>` and `Option<&dyn Fn(usize) -> [[f64; 3]; 3]>`). See `Operators::apply_bochner_laplacian` and `apply_lichnerowicz_laplacian`.
- `cartan-dec`: `Mesh` is now generic over any `M: Manifold` as `Mesh<M, const K: usize = 3, const B: usize = 2>`. `FlatMesh = Mesh<Euclidean<2>, 3, 2>` is a permanent public alias. All internal consumers updated to the two-argument `Operators::from_mesh(&mesh, &Euclidean::<2>)` API.
- `cartan-manifolds`, `cartan-optim`: no_std/alloc feature flags (landed in 0.1.1, documented here for completeness).
- Doctests across `cartan-manifolds` changed from `ignore` to `no_run` or runnable where possible. Zero bare `ignore` annotations remain.

### Fixed

- `cartan-manifolds`: Clippy `needless_return` in `Sphere::check_point`, `Sphere::check_tangent`, `Grassmann::check_point`, `Grassmann::check_tangent` (#[cfg] blocks).
- `cartan-manifolds`: Clippy `op_ref` in `SpecialEuclidean::riemannian_hessian_vector_product` (unnecessary `&` on matrix operands).
- `cartan-manifolds`: Clippy `doc_overindented_list_items` in `SpecialEuclidean::riemannian_hessian_vector_product` doc comment.
- `cartan-dec`, `cartan-manifolds`: rustdoc `broken_intra_doc_links` in `hodge.rs`, `mesh.rs`, `grassmann.rs`, `sphere.rs`, `se.rs` (bracket sequences in math formulas misinterpreted as item links; escaped with `\[`, `\]` or wrapped in backticks).
- `cartan-optim`: Remaining `ignore` doctest in `lib.rs` changed to `no_run`.
- Code formatting (`cargo fmt`) applied across the full workspace.

---

## [0.1.1] - 2026-03-10

### Added

- `SymmetricPositiveDefinite<N>` (`Spd<N>`): all 7 traits. Affine-invariant metric, K <= 0 (Cartan-Hadamard), full Riemannian HVP via Christoffel symbols.
- `Grassmann<N,K>`: all 7 traits. Gr(N,K) with principal angles, horizontal representation, Boumal 2022 Weingarten correction.
- `Corr<N>`: all 7 traits. Correlation matrices with flat Frobenius metric, Higham alternating-projections for `project_point`.
- `cartan-dec`: discrete exterior calculus layer -- `Mesh<M,K,B>`, `FlatMesh`, `ExteriorDerivative` (d0, d1), `HodgeStar` (star0, star1, star2), `Operators` (Laplace-Beltrami, Bochner, Lichnerowicz), advection, divergence.
- `cartan-optim`: Riemannian trust region (RTR/Steihaug-Toint) added.
- `cartan-manifolds`, `cartan-optim`: initial no_std/alloc feature flags.

### Changed

- `SpecialEuclidean<N>`: `Connection` impl updated with SO block Weingarten correction and flat translation block.
- `SpecialOrthogonal<N>`: Cayley map registered as formal `Retraction` trait impl (was previously only available as `cayley_retract`).

---

## [0.1.0] - 2026-02-15

Initial release.

- `cartan-core`: all 7 traits (`Manifold`, `Retraction`, `ParallelTransport`, `VectorTransport`, `Connection`, `Curvature`, `GeodesicInterpolation`), `CartanError` (6 variants), `Real = f64`.
- `cartan-manifolds`: `Euclidean<N>`, `Sphere<N>`, `SpecialOrthogonal<N>`, `SpecialEuclidean<N>`.
- `cartan-optim`: RGD, RCG, Frechet mean.
- `cartan-geo`: parameterized geodesics, curvature queries, Jacobi field integration.
- Integration test harness (`manifold_harness`, `matrix_harness`).
