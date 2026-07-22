# Changelog

All notable changes to cartan are documented here.

---

## [0.8.0]

Breaking: six sparse signatures change type, `full` widens to the whole family,
and `cartan-homog`'s `serde` feature is removed. See below.

### Changed

- **The facade re-exports all ten sub-crates**, one capability feature each:
  `dec`, `remesh`, `stochastic`, `homog`, `full-field`, `io`, `maxwell`, plus
  the `alloc` and `std` tiers and `full`. Previously it re-exported five, so
  `remesh`, `stochastic`, `homog`, `io` and `maxwell` were unreachable through
  `use cartan::`. Default is still `std` plus `dec`. Gated re-exports carry
  `doc(cfg)` badges, so docs.rs shows the whole surface and names the flag each
  item needs.
- **`full` now means the whole family**, not "std plus dec". Existing users of
  that feature get a larger build, not a broken one.
- **`sprs` is replaced by `nalgebra-sparse` throughout.** `cartan-io` and
  `cartan-maxwell` already used it; `cartan-dec` and `cartan-homog` have joined
  them, so the workspace no longer carries two sparse types. Six public
  signatures change from `sprs::CsMat` to `nalgebra_sparse::CscMatrix`:
  `exterior::ExteriorDerivative::{d0, d1}`, `line_bundle::BochnerLaplacian::matrix`,
  and `fullfield::solver::{Ilu0::factor, solve_dense_lu, Amg::build}`. CSC, not
  CSR: every assembly here builds CSC and the matvec loops walk columns to
  match. The operators are unchanged and the 84-case numerical cross-check
  passes without modification.
- **`Spd::dist` no longer forms the logarithm.** It inherited the default
  `||Log_P(Q)||_P`, costing four eigendecompositions: three in `log` for
  `P^{1/2}`, `P^{-1/2}` and `log(M)`, a fourth in `inner` via `sym_inv`. Only
  the spectrum of `P^{-1} Q` reaches the answer. Eigenvalues now come off
  `L^{-1} Q L^{-T}` with `P = L L^T` the Cholesky factor: symmetric, so the
  cheaper solver applies, and similar to `P^{-1} Q`, so the spectrum matches.
  SPD(3) 1830 to 405 ns, SPD(6) 5745 to 1426 ns, SPD(10) 14653 to 3884 ns.
  Distance is now cheaper than `log`. Sphere distance is deliberately untouched:
  `2*asin(||p-q||/2)` avoids the cancellation `acos(p.q)` suffers near
  coincident points.
- **In-place `exp_into`, `log_into` and `transport_into`.** The
  value-returning forms must materialise and return a tangent vector, which at
  ambient dimension 50 is 400 bytes moved per call. These write into a
  caller-owned buffer instead. Both traits gain them with default
  implementations that delegate, so no existing implementor breaks and only
  manifolds where it pays need override; `Sphere` does, using `copy_from` and
  `axpy` so no temporary is built either.

  | op | N = 10 | N = 50 |
  |---|---|---|
  | `exp_into` | 1.18x | 1.12x |
  | `log_into` | 1.42x | 1.61x |
  | `transport_into` | 1.05x | 1.29x |

  At N = 3 they are level with the value-returning forms, which stay the
  clearer call there. `log_into` leaves `out` untouched when it fails at the
  cut locus, so a caller reusing a buffer cannot read a stale value as a fresh
  one.

  `transport_into` uses the collapse
  `(q - cp)/(1+c) + p = (p+q)/(1+c)`, so the direction vector is never formed.
  The same collapse loses in the value-returning form, where `(p+q) * k`
  materialises a vector: 91 ns to 149 ns at N = 50.
- **`Sphere::transport` no longer calls `log`.** The two-log formula
  (Absil et al. 8.1.3) evaluated `log` in each direction, each paying an
  inverse trig call, a norm and a division. Written on the dot product alone it
  is `PT(v) = v - (v.w)(w/(1+c) + p)` with `c = p.q` and `w = q - cp`; both trig
  calls cancel because `cos(theta) = c` and `||w|| = sin(theta)` for unit
  vectors. S^2 37 to 7 ns, S^9 97 to 23 ns, S^49 167 to 96 ns. Against
  Manifolds.jl, 0.4x-0.9x becomes 0.9x-4.9x. The result is still re-projected:
  tangency is exact in exact arithmetic, but `w/(1+c)` amplifies rounding near
  the cut locus, where the residual reaches 4e-10 without it.
- **`SelfConsistent` damps linearly at the `alloc` tier**, rather than along SPD
  geodesics, because the geodesic step needs an eigen decomposition that
  requires std. Same fixed point, slower convergence. Under std it is unchanged.
- README rebuilt as an on-ramp; the per-crate inventory moves to
  `CAPABILITIES.md`. MSRV badge corrected from 1.85 to 1.89.

### Added

- **`cartan::guide`**: doctested chapters over the three regimes, points, fields
  and paths. `getting_started`, `manifolds`, `optimisation`, `fields`,
  `bundles`, `stochastic`, `homogenisation`, `interop`. Run by
  `cargo test --doc`, so a chapter that drifts from the API fails CI.
- **Cross-language benchmarks** (`benchmarks/CROSSLANG.md`) against
  Manifolds.jl, geomstats and geoopt. All four read one fixture file, so
  agreement is measured on identical inputs. 51 of 51 comparisons match to
  better than 1e-12, worst 9.4e-14, over exp, log, dist and transport on the
  sphere and SPD cone. Timing rows are gated on the values having matched.
  Against Manifolds.jl the median is 1.9x. Against the Python libraries, two to
  three orders of magnitude, which largely measures interpreter overhead.
- **Criterion suite**: `cargo bench -p cartan-manifolds`, over exp, log, dist
  and transport for `Sphere` and `Spd`.
- **Bare-metal CI**: builds the facade, `cartan-core` bare, and every alloc-tier
  crate for `thumbv7em-none-eabihf`. A feature-matrix job builds each capability
  feature alone and runs the guide doctests.
- **`cartan-homog/examples/rve_to_effective.rs`**: sweeps volume fraction across
  six schemes and asserts each stays inside the Reuss-Voigt bracket.

### Fixed

- **The `cartan` facade never declared `#![no_std]`.** The embedded
  configuration its own documentation recommended compiled on a host, where std
  stays linkable, and failed on any target without one.
- **`cartan-homog`'s `alloc` tier did not build**: 17 errors on
  `thumbv7em-none-eabihf`, 15 std-only float methods and 2 imports of a
  std-gated `Spd`. Transcendentals now route through `libm` behind a `#[cfg]`
  shim, matching `cartan-core`. The SPD geodesic step is gated behind std and
  falls back to the linear damping arm the function already carried.
- Both shipped because no CI job had ever targeted a platform without std.
  `--no-default-features` on x86_64 proves only that the source avoids std.
- **Benchmark timing methodology**, twice. Per-call timing charged
  `Instant::now` overhead to every sample, reporting a 15 ns `exp` as 40 ns.
  Batching fixed that and exposed the next: with loop-invariant inputs the
  optimiser hoisted the call out of the batch, reporting 0 ns. Inputs are now
  black-boxed and the harness is cross-checked against criterion.
- **`cartan-py`** consumed the old `cartan-dec` signatures. It is its own
  workspace, so `cargo build --workspace` never checked it. Its declared MSRV
  was also stale at 1.85.

### Removed

- **Six unused dependencies**: `rayon` and `serde` from `cartan-dec` and
  `cartan-homog`, `thiserror` from `cartan-maxwell`, `rayon` and `indexmap` from
  the workspace. `rayon` was referenced nowhere in the workspace, and
  `cartan-dec` is in the default path, so every user compiled a thread pool for
  nothing.
- **`cartan-homog`'s `serde` feature.** It was in the default set and gated a
  dependency the crate never used. Nothing in the workspace named it.

### Notes

- geoopt's `transp` is a projection onto the target tangent space, not exact
  parallel transport, which is the correct choice for an optimiser. The
  comparison verifies this each run by checking it equals geoopt's own `proju`,
  so a future change is reported rather than excused.
- `cartan-core`, `cartan-dec` and `cartan-maxwell` now inherit the workspace
  version instead of pinning their own.

---

## [0.7.0]

Breaking: the FEEC crates are gone from the published set, and the geometry
primitive changed. See below.

### Changed

- **FEEC layer now comes from upstream `formoniq`.** `cartan-exterior`,
  `cartan-simplicial` and `cartan-feec` are removed from the workspace and
  replaced by the `formoniq`, `simplicial`, `exterior` and `derham` crates,
  pinned at `=0.2.0`. Those three crates were a port of formoniq made before it
  was published; formoniq now ships the same stack on crates.io, so cartan no
  longer carries a copy of it.
- **Geometry is carried as squared edge lengths** (`MeshLengthsSq`), matching
  upstream. This is the Regge primitive: the per-cell metric is linear in it, so
  a prescribed metric evolution enters polynomially rather than through square
  roots, and indefinite signatures stay representable. `FlrwDriver` consequently
  scales its stored data by `a(t)^2` where it previously scaled lengths by
  `a(t)`; the length-level behaviour is unchanged and is covered by a test.
- **`cartan-io`** drops its `sprs` and exterior-algebra dependencies, which it
  did not use. Field reconstruction now goes through `WhitneyInterpolant` and
  `Sampler`, which returns ambient-frame values, the frame VTK consumers expect.
- **`cartan-io`'s MDD writer** is rewritten from the format specification and
  now rejects ragged frames and time/frame count mismatches, neither of which
  the MDD container can represent.
- `cartan-dec`, `cartan-homog` and the rest of the stack are untouched, and
  `sprs` remains the workspace sparse-matrix default.

### Added — v1.3 (full-field enhancements)

- **DifferentialCompliance scheme** (`cartan-homog::schemes`): Norris-Davies
  dual variant of the differential scheme, integrating `dS*/df` on compliance.
  Physically preferred for soft inclusions (dry-pore, crack limit). The primal
  `Differential` and dual `DifferentialCompliance` are both exposed; the 0.008%
  gap vs ECHOES's DIFF is a primal-vs-dual formulation choice (Milton 2002
  Ch. 10.12), not a bug.
- **Lebedev quadrature for anisotropic-reference Hill tensor** (`cartan-homog::shapes::lebedev`):
  degree-14 grid on S² (exact up to degree-3 spherical harmonics). `Sphere::hill`
  now falls through to Lebedev when the reference is not isotropic; previously
  returned an error. Closes Task 28 from the v1 spec.
- **μCT voxel import for FullField** (`cartan-homog::fullfield::voxelize`):
  `load_voxel_raw_u8(path, N)` reads N³ u8 phase-id binary files (Digital
  Porous Media Portal / Imperial College Berea / NIST CBT convention);
  `FullField::homogenize_voxel(voxel, props)` runs the cell problem directly
  on voxel-tagged tets.
- **Two-level aggregation AMG preconditioner** (`cartan-homog::fullfield::solver`):
  strong-connection graph → greedy aggregation → piecewise-constant prolongation
  → Galerkin coarse operator → cached LU factor. V-cycle with damped Jacobi
  pre/post-smoother. `solve_with_fallback` now escalates
  `Jacobi-PCG → ILU(0)-PCG → AMG-PCG → dense LU`.
- **Conforming red (1-to-8) tet refinement** (`cartan-remesh::primitives_3d`):
  `red_refine_tets_uniform` bisects all 6 edges per tet, produces 8 sub-tets
  via the Bey decomposition. Edge midpoints are shared across adjacent tets
  (HashMap cache), so the refined mesh is conforming. Complements the v1.2
  non-conforming barycentric refinement.

### Added — v1.2 (full-field v1.2)

- 3D tet-mesh barycentric refinement in `cartan-remesh::primitives_3d` (non-conforming).
- Periodic BCs for the full-field cell problem (slave→master DOF elimination, gauge-vertex anchor).
- Dense LU fallback (`solve_dense_lu`) + ILU(0) preconditioner for the solver ladder.
- Macroscale slab Darcy solver (`cartan-homog::fullfield::macroscale`).
- Hausdorff gate for adaptive-refinement vs analytic transition-layer sets.

### Added — v1.1 (full-field v1)

- Full-field DEC cell-problem solver for Order2 RVEs (`cartan-homog::fullfield`).
- Kuhn-triangulated periodic cube mesh builder.
- P1-FEM stiffness + RHS assembly; volume-averaged effective tensor.
- Reliability indicator `d_AI(C_MF, C_FF)` via affine-invariant SPD distance.

### Added — v1 (homogenisation foundation)

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
