# Changelog

All notable changes to cartan are documented here.

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
