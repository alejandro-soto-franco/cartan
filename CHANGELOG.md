# Changelog

All notable changes to cartan are documented here.

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
