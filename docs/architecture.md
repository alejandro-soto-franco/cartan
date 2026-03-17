# cartan Architecture

Internal developer reference. Not committed to the repo (see .gitignore).

---

## Crate Dependency Graph

```
cartan  (facade)
  |-- cartan-core       (no cartan deps)
  |-- cartan-nalgebra   (cartan-core)
  |-- cartan-manifolds  (cartan-core, cartan-nalgebra)
  |-- cartan-optim      (cartan-core, cartan-manifolds)
  |-- cartan-geo        (cartan-core, cartan-manifolds)
  |-- cartan-dec        (cartan-core, cartan-manifolds)
```

Downstream crates (volterra, future solvers):
```
volterra
  |-- cartan-core
  |-- cartan-manifolds
  |-- cartan-dec
```

Rules:
- cartan-core has NO cartan dependencies. It only depends on `rand` (for Rng trait bounds) and `thiserror`.
- cartan-nalgebra provides the concrete storage types (SVector, SMatrix) used by all manifold implementations.
- cartan-manifolds depends on cartan-nalgebra. Manifold impls use SVector/SMatrix throughout.
- cartan-dec depends on cartan-core and cartan-manifolds. It bridges continuous geometry to discrete PDE operators.
- cartan-optim and cartan-geo depend only on cartan-core (generic algorithms) and cartan-manifolds (concrete manifold utilities).
- The facade crate `cartan` re-exports all sub-crates. Downstream users add only `cartan` as a dependency.

---

## Trait Hierarchy

```
Manifold
  (base: exp, log, inner, norm, dist, project_tangent, project_point,
         zero_tangent, check_point, check_tangent, random_point,
         random_tangent, dim, ambient_dim, injectivity_radius)
  |
  +-- Retraction
  |     retract(p, v) -> Point          [cheaper than exp; default delegates to exp]
  |     inverse_retract(p, q) -> Tangent
  |
  +-- ParallelTransport
  |     transport(p, q, v) -> Tangent   [exact parallel transport along geodesic]
  |     |
  |     +-- VectorTransport (blanket impl from ParallelTransport)
  |           vector_transport(p, v, w) -> Tangent   [transport v along exp(p,w)]
  |
  +-- Connection
  |     riemannian_hessian_vector_product(p, grad_f, hess_f_eucl, v) -> Tangent
  |     |
  |     +-- Curvature
  |           riemann_curvature(p, u, v, w) -> Tangent
  |           sectional_curvature(p, u, v) -> Real   [default from riemann + inner]
  |           ricci_curvature(p, u, v) -> Real
  |           scalar_curvature(p) -> Real
  |
  +-- GeodesicInterpolation
        geodesic(p, q, t) -> Point      [gamma(t) along minimizing geodesic]
```

Key design decisions:

**Extrinsic coordinates.** Both `Point` and `Tangent` live in ambient space.
For S^{N-1} in R^N, both are `SVector<Real, N>` even though T_pM is (N-1)-dimensional.
The tangent constraint (p^T v = 0) is enforced by `project_tangent` and `check_tangent`,
not the type system. Matches Manopt, Pymanopt, Geomstats convention.

**exp is total, log returns Result.** On complete manifolds (all current ones are complete
by Hopf-Rinow), exp is globally defined. log fails at the cut locus (antipodal on sphere,
etc.) and returns `Err(CartanError::CutLocus)`.

**Retraction defaults to exp.** `Manifold::retract` has a default impl that calls `exp`.
The `Retraction` trait overrides this with a cheaper approximation. Solvers always call
`retract()`, never `exp()` directly, so the cheaper version is automatically used when available.

**The Real alias.** All floats are `pub type Real = f64`. This enables a future mechanical
refactor to `T: Scalar` generics (f32 support) without changing any type signatures in
calling code.

**Neg on Tangent.** The Tangent associated type requires `Add + Mul<Real> + Neg`.
Neg was added beyond the original spec to support conjugate gradient (`d = -grad`)
and other algorithms that negate tangent vectors.

**Weingarten correction in Connection.** All `Connection` impls receiving `grad_f` treat
it as the Euclidean gradient (egrad), matching Pymanopt/Manopt convention. The correction
term `<egrad, normal> * v` is applied manifold-specifically: sphere uses `egrad.dot(p) * v`,
SO(N) uses the `sym(R^T * egrad)` symmetric part, Grassmann uses `V * sym(Q^T * G)`,
SE(N) applies the SO block correction and leaves the translation block flat.

---

## Error Handling

All fallible operations return `Result<T, CartanError>` from cartan-core.

CartanError variants:
- `CutLocus` -- log() called at or near the cut locus (non-unique minimizing geodesic)
- `NumericalFailure` -- numerical computation failed (matrix log diverged, etc.)
- `NotOnManifold { violation }` -- check_point() failed; violation is the constraint residual
- `NotInTangentSpace { violation }` -- check_tangent() failed
- `LineSearchFailed` -- Armijo/Wolfe line search did not converge
- `ConvergenceFailure { iterations }` -- iterative algorithm did not converge

---

## Memory Layout (cartan-manifolds)

All manifold implementations use nalgebra `SVector<Real, N>` for points and tangent vectors.
This means:
- All geometry is stack-allocated for small N (N <= ~32).
- No heap allocation in the critical path.
- nalgebra's SIMD optimizations apply automatically.

For large N, `DVector<Real>` would be used instead. Current manifolds are all const-generic
over N, so this is a compile-time choice.

---

## Testing Strategy

Integration tests live in `cartan/tests/`. Each manifold has its own test file.
Tests use two harnesses from `cartan/tests/common/`:

`manifold_harness.rs` -- property-based tests for any `Manifold` impl:
- exp/log roundtrip: log(p, exp(p, v)) == v for small v
- log/exp roundtrip: exp(p, log(p, q)) == q for nearby q
- project_tangent idempotency
- check_point and check_tangent pass on randomly generated points/tangents
- dist symmetry
- zero tangent has zero norm

`matrix_harness.rs` -- additional tests for matrix manifolds (SO(N), SE(N)):
- exp/log roundtrip at higher tolerance (matrix exp/log numerics)
- parallel transport preserves inner product
- Cayley retraction satisfies retraction conditions

All tests use the `approx` crate for floating-point comparisons with configurable tolerance.

---

## cartan-dec Architecture

cartan-dec is the bridge to PDE solvers. It accepts a user-supplied simplicial mesh,
precomputes metric-free and metric-dependent operators, and exposes them for time-stepping.

### Mesh Layer

The core type is `Mesh<M: Manifold, const K: usize = 3, const B: usize = 2>`:
- `M` is the manifold type for vertex positions.
- `K` is vertices per simplex (K=3 for triangles, the default).
- `B = K-1` is vertices per boundary face (B=2 for edges).

`FlatMesh = Mesh<Euclidean<2>, 3, 2>` is the type alias for flat 2D triangular meshes.

Key `FlatMesh` builders:
- `FlatMesh::from_triangles(vertices, triangles)` -- from raw `[f64; 2]` arrays
- `FlatMesh::unit_square_grid(n)` -- uniform n x n grid on [0,1]^2

Metric methods on generic `Mesh<M, 3, 2>` use geodesic distance and tangent-space pullback:
- `edge_length(&manifold, e)` -- `manifold.dist(vi, vj)`
- `triangle_area(&manifold, t)` -- Gram determinant in `T_{v0}M`
- `circumcenter(&manifold, t)` -- equidistance system in `T_{v0}M`, mapped back via exp
- `check_well_centered(&manifold)` -- verifies all circumcenters lie strictly inside their triangles

Flat-specific fast-path methods on `Mesh<Euclidean<2>, 3, 2>`:
- `edge_length_flat(e)`, `triangle_area_flat(t)`, `circumcenter_flat(t)`, `edge_midpoint(e)`

The mesh topology is fixed at construction. Triangulation generation (Bowyer-Watson Delaunay,
Hilbert curve reordering) is out of scope for the current release; users supply the mesh.

### Operator Layer

**ExteriorDerivative** is metric-free (purely combinatorial):
- `d0`: `n_boundaries x n_vertices` incidence matrix, maps 0-forms to 1-forms.
- `d1`: `n_simplices x n_boundaries` incidence matrix, maps 1-forms to 2-forms.
- Invariant: `d1 * d0 = 0` (exactness of the exterior derivative).
- `from_mesh(&FlatMesh)` and `from_mesh_generic<M>(&Mesh<M, 3, 2>)`.

**HodgeStar** encodes the metric via dual/primal volume ratios:
- `star0`: barycentric dual cell areas (n_vertices diagonal).
- `star1`: `|dual edge| / |primal edge|` (n_boundaries diagonal).
- `star2`: `1 / triangle_area` (n_simplices diagonal).
- Currently implemented for `FlatMesh` only (`from_mesh(&FlatMesh, &Euclidean<2>)`).
- Inverse helpers: `star0_inv()`, `star1_inv()`, `star2_inv()`.

**Operators<M>** assembles the composed discrete operators:
- `laplace_beltrami`: `star0_inv * d0^T * diag(star1) * d0` (n_vertices x n_vertices dense).
- `mass0`, `mass1`: diagonal Hodge star entries, kept for L2 inner products in solvers.
- `ext`: `ExteriorDerivative` (for advection/divergence callers).
- `hodge`: `HodgeStar` (for direct access).
- `apply_laplace_beltrami(&f)` -- scalar Laplace-Beltrami on 0-forms.
- `apply_bochner_laplacian(&u, ricci_correction)` -- Bochner (connection) Laplacian on 2D vector fields.
  Callback signature: `Option<&dyn Fn(usize) -> [[f64; 2]; 2]>` (Ricci tensor at vertex v).
- `apply_lichnerowicz_laplacian(&q, curvature_correction)` -- Lichnerowicz Laplacian on
  symmetric 2-tensors. Callback: `Option<&dyn Fn(usize) -> [[f64; 3]; 3]>` (curvature
  endomorphism on the 3-component space [Qxx, Qxy, Qyy] at vertex v).
- `from_mesh(&FlatMesh, &Euclidean<2>)` assembles all operators. `apply_*` methods are
  generic over M (PhantomData carries the manifold type for future non-flat assembly).

**Advection and divergence** (flat-only in current release):
- `apply_scalar_advection(&FlatMesh, f, u)` -- upwind (u.nabla)f on a scalar 0-form.
- `apply_vector_advection(&FlatMesh, q, u)` -- component-wise scalar advection.
- `apply_divergence(&FlatMesh, ext, hodge, u)` -- codifferential delta_1 applied to û.
- `apply_tensor_divergence(&FlatMesh, ext, hodge, t)` -- divergence of symmetric 2-tensor.

### Field Layout

For multi-component fields, structure-of-arrays (SoA) layout is used:
- `DVector<f64>` of length `2 * n_vertices` with x-components in `[0..n_v]`, y-components in `[n_v..2*n_v]`.
- For symmetric 2-tensors: length `3 * n_vertices` with Qxx, Qxy, Qyy blocks.
- All `apply_*` methods document the expected layout in their doc comments.

### Parallelism (planned)

Rayon-based graph coloring is not yet implemented. The time-stepping loop is sequential.
Future plan: color simplices (typically 4-7 colors in 2D Delaunay meshes) and use
`rayon::par_iter` over each color class without data races.
