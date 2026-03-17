# cartan Generalization Design
**Date:** 2026-03-17
**Status:** Approved

## Overview

Two major features — generic manifold-embedded mesh (`Mesh<M>`) and `no_std` support — delivered as three sequential workstreams. End state: a single clean generic `Mesh<M>` that works across std, no_std+alloc, and no_std+no_alloc targets, with all existing TODOs resolved.

## Goals

1. Make `cartan-dec` generic over any `M: Manifold` (volterra and robotics use cases equally)
2. Add three-tier `no_std` support across all six crates
3. Fix all contained code TODOs: Weingarten corrections, full tensor Ricci, doctest annotations
4. Preserve 100% backward compatibility during migration via type alias

## Non-Goals

- No new manifold implementations
- No change to optimizer algorithms
- No Python bindings or FFI layer
- No GPU acceleration

---

## Simplex Dimension Convention

Throughout this spec, `K` in `Mesh<M, const K: usize>` denotes the **number of vertices per simplex** (not the topological dimension):
- `K=2` → edge (1-simplex)
- `K=3` → triangle (2-simplex, the default)
- `K=4` → tetrahedron (3-simplex)

A K-simplex has exactly K boundary faces, each a (K-1)-simplex with K-1 vertices. So boundary arrays are `[[usize; K-1]; count]`. In Rust const generics, this requires a helper: boundary face type is `[usize; K]` where K is the number of vertices of the boundary simplex (one less than the parent).

For the default triangle case: `Mesh<M, 3>` with `type FlatMesh = Mesh<Euclidean<2>, 3>`.

---

## Three Workstreams

### Workstream A — `no_std` Feature Flags (foundation, must land first)

**Purpose:** Thread `no_std` compatibility through all six crates before any DEC changes so the generic mesh is `no_std`-compatible from day one.

**Three feature tiers:**

```toml
[features]
default = ["std"]
std = ["alloc"]
alloc = []
# (no features active) = no_std, no_alloc — fixed-size only
```

#### A1 — Replace `use std::` with `use core::` throughout

This is non-trivial work, not simple boilerplate. Every `use std::` in `cartan-core` and `cartan-manifolds` must be replaced with the `core::` equivalent:

- `use std::fmt::Debug` → `use core::fmt::Debug`
- `use std::ops::{Add, Mul, Neg, Sub}` → `use core::ops::{Add, Mul, Neg, Sub}`
- `use std::f64::consts::PI` → `use core::f64::consts::PI`
- `format!` macros require `alloc` — all usages must be audited

All six crates get `#![cfg_attr(not(feature = "std"), no_std)]` at the crate root.
Add `#[cfg(feature = "alloc")] extern crate alloc;` where `Vec`, `String`, or `Box` are used.

#### A2 — `CartanError` redesign for no_alloc tier

`CartanError` currently has variants with `String` fields (e.g., `CutLocus { message: String }`). `String` requires `alloc` and cannot exist in the no_alloc tier.

Three-tier strategy:

```rust
// no_alloc tier: &'static str only
#[cfg(not(feature = "alloc"))]
#[derive(Debug, Clone)]
pub enum CartanError {
    CutLocus { message: &'static str },
    NumericalFailure { operation: &'static str, message: &'static str },
    NotOnManifold { constraint: &'static str, violation: Real },
    NotInTangentSpace { constraint: &'static str, violation: Real },
    LineSearchFailed { steps_tried: usize },
    ConvergenceFailure { iterations: usize, gradient_norm: Real },
}

// alloc/std tiers: String for rich messages (current behavior)
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub enum CartanError {
    CutLocus { message: String },
    // ... (unchanged)
}
```

`std::error::Error` impl is gated behind `#[cfg(feature = "std")]`. Under `alloc`-only, `CartanError` implements `Display` but not `std::error::Error`. Under no_alloc, implements `Display` with static strings only.

All `format!` calls producing `CartanError` messages must be gated behind `#[cfg(feature = "alloc")]` with `&'static str` fallbacks for no_alloc.

#### A3 — Dependency feature flags (per-crate, explicit)

Each crate's `Cargo.toml` must explicitly set `default-features = false` for deps that have std-optional features:

**`cartan-core`:**
```toml
[dependencies]
rand = { workspace = true, default-features = false, features = ["getrandom"] }
```
`Rng` is in `rand_core` (no_std). The `getrandom` feature provides entropy on embedded targets.

**`cartan-manifolds`:**
```toml
[dependencies]
nalgebra = { workspace = true, default-features = false }
rand = { workspace = true, default-features = false, features = ["getrandom"] }
rand_distr = { workspace = true, default-features = false }
```
nalgebra's `std` feature is disabled by `default-features = false`; no additional feature flag is required to enable no_std. rand_distr supports no_std similarly.

**`cartan-dec`:**
```toml
[dependencies]
rayon = { workspace = true, optional = true }  # only under std

[features]
default = ["std"]
std = ["alloc", "rayon"]
alloc = ["hashbrown"]
hashbrown = ["dep:hashbrown"]
```
`rayon` requires `std::thread` — it must be feature-gated and all `rayon`-using code wrapped in `#[cfg(feature = "std")]`. A sequential fallback (simple loop) is used otherwise.

#### A4 — Replace `HashMap` in `cartan-dec`

`mesh.rs` uses `HashMap<(usize, usize), usize>` for edge deduplication in `from_triangles`. `std::collections::HashMap` requires `std`.

Replacement strategy:
- **alloc tier**: use `hashbrown::HashMap` (no_std-compatible, same API)
- **no_alloc tier**: sort edge pairs and binary-search for deduplication (O(E log E) but zero-alloc)

```toml
# cartan-dec Cargo.toml
[dependencies]
hashbrown = { version = "0.15", optional = true, default-features = false }
```

```rust
#[cfg(feature = "alloc")]
use hashbrown::HashMap;

#[cfg(not(feature = "alloc"))]
// sorted-array edge dedup used in FixedMesh::from_triangles_fixed()
```

#### A5 — `cartan-geo` fixed-size alternatives

`JacobiResult` holds `Vec<T>` — gated behind `alloc`. Fixed-size alternative for no_alloc:

```rust
#[cfg(feature = "alloc")]
pub struct JacobiResult<T> {
    pub params: Vec<Real>,
    pub field: Vec<T>,
    pub velocity: Vec<T>,
}

#[cfg(not(feature = "alloc"))]
pub struct JacobiResultFixed<T: Copy, const N: usize> {
    pub params: [Real; N],
    pub field: [T; N],
    pub velocity: [T; N],
    pub len: usize,
}
```

`Geodesic::sample` (returns `Vec`) gated behind `alloc`. Core methods `eval`, `length`, `midpoint` available in all tiers.

#### Per-crate summary

| Crate | std→core | String→&'static str | HashMap | rayon | Vec-gating |
|-------|----------|---------------------|---------|-------|------------|
| `cartan-core` | yes | yes (CartanError) | no | no | no |
| `cartan-manifolds` | yes | no | no | no | no |
| `cartan-optim` | yes | no | no | no | no |
| `cartan-geo` | yes | no | no | no | yes (sample, JacobiResult) |
| `cartan-dec` | yes | no | yes → hashbrown | yes → feature gate | yes (Mesh itself) |
| `cartan` facade | yes | no | no | no | no |

---

### Workstream B — Contained TODOs (parallel with A)

#### B1 — Weingarten Correction on 5 `Connection` Impls

**SPD and Corr are excluded from B1:**
- `Spd<N>`: The existing `riemannian_hessian_vector_product` implementation is already correct — it implements the full formula `P·ehvp·P + 0.5*(V·P⁻¹·G + G·P⁻¹·V)` from AMS Ch.7, derived from Christoffel symbols of the affine-invariant metric. This is not a shape-operator correction — it is the full Riemannian HVP. **Do not modify.**
- `Corr<N>`: Uses flat Frobenius metric. The embedding has zero shape operator. Correction is zero. Verify this at implementation time by checking that `project_tangent` for Corr is already the identity restricted to correlation matrices, confirming the flat structure.

**Manifolds receiving Weingarten corrections:**

**`Sphere<N>`** (fixes the existing `TODO(phase6)` comment):
```
Hess f(p)[v] = proj_p(D²f(p)[v]) - <grad_f, p> * v
correction = -inner(grad_f, p) * v
```
where `p` is the unit normal (serving as the shape operator for the sphere embedding in R^N). The `grad_f` argument is the Riemannian gradient already projected onto `T_pS`.

**`SpecialOrthogonal<N>`** (Absil-Mahony-Sepulchre §5.3, Boumal §6):
The correct Riemannian HVP for SO(N) with bi-invariant metric:
```
Hess f(R)[V] = R · skew(R^T · ehvp(V)) - 0.5 * R · sym(R^T · grad_f) · R^T · V
```
where `skew(A) = (A - A^T)/2` projects onto the Lie algebra so(N), and `sym(A) = (A + A^T)/2` is the symmetric part. The second term is the Weingarten correction. `grad_f` and `V` are in `T_R SO(N)` (skew matrices times R).

**`Grassmann<N,K>`** (Absil-Mahony-Sepulchre §5.3):
The Riemannian HVP for Grassmann Gr(N,K) with canonical metric:
```
Hess f(Q)[V] = proj_Q(ehvp(V)) - Q · sym(Q^T · grad_f) · Q^T · ehvp(V) · ...
```
More precisely, following AMS: let `G = grad_f` (in horizontal space, `N×K`), `V` in horizontal space. The correction from the shape operator of the Stiefel embedding is:
```
correction = -(I - QQ^T) · G · (Q^T · V) - Q · sym(Q^T · G) · (Q^T · V)
           = -G · (Q^T · V) + Q · skew(Q^T · G) · (Q^T · V)
```
Note: `(I - QQ^T)·G·(Q^T·V)` gives an N×K matrix (N×N times N×K times K×K). The final result is projected back to the horizontal tangent space via `proj_tangent(Q, ·)`.

**`SpecialEuclidean<N>`** (product structure):
- Rotation block: receives the SO(N) correction with `grad_f` restricted to the rotation component
- Translation block: zero correction (flat Euclidean factor)
- Combined as block-diagonal application on the `SETangent` struct

**`Euclidean<N>`**: zero correction — flat, no shape operator. Already correct (returns `proj_p(ehvp)` which on Euclidean is just `ehvp`). **No change needed.**

#### B2 — Full Tensor Ricci Correction in `cartan-dec`

Replace scalar `Option<f64>` with vertex-indexed callbacks returning full tensor:

```rust
// apply_bochner_laplacian — 2D vector fields
// Callback: vertex index → 2×2 Ricci tensor at that vertex
pub fn apply_bochner_laplacian(
    &self,
    u: &DVector<f64>,
    ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
) -> DVector<f64>

// apply_lichnerowicz_laplacian — 2D symmetric 2-tensors (3 components: xx, xy, yy)
// Callback: vertex index → 3×3 matrix representing the action of the
// curvature contraction R_{ikjl} Q^{kl} on the symmetric 2-tensor Q
// (maps the 3 independent components [Qxx, Qxy, Qyy] to a new 3-vector)
pub fn apply_lichnerowicz_laplacian(
    &self,
    q: &DVector<f64>,
    curvature_correction: Option<&dyn Fn(usize) -> [[f64; 3]; 3]>,
) -> DVector<f64>
```

**On the Lichnerowicz callback dimension:** The Lichnerowicz Laplacian acts on symmetric 2-tensors. In 2D, a symmetric 2-tensor has 3 independent components `[Qxx, Qxy, Qyy]`. The curvature correction `2·R_{ikjl} Q^{kl}` is itself a symmetric 2-tensor (3 components), so the callback returns a `3×3` matrix mapping the 3-component input to a 3-component output. This is consistent with the 2D assumption: the `[[f64; 3]; 3]` is not a 3D tensor, it is the matrix of the curvature endomorphism on the 3-dimensional space of symmetric 2×2 tensors. For the Bochner Laplacian acting on 2D vector fields (2 components), the Ricci tensor is a `2×2` matrix. Both callbacks are therefore correct for the 2D setting.

For Einstein manifolds (Ric = κ·g): Bochner caller passes `|_| [[k,0.],[0.,k]]`, Lichnerowicz caller passes `|_| [[2.*k,0.,0.],[0.,2.*k,0.],[0.,0.,2.*k]]`. `None` means zero correction.

#### B3 — Doctest Annotations in `cartan-manifolds`

Replace all bare `ignore` annotations with one of:
- Run (no annotation): self-contained examples with hardcoded inputs (`exp`, `log`, `inner`, `dist`, `check_point`, `check_tangent`)
- `no_run`: examples using `random_point` / `random_tangent` (non-deterministic sampling)

Zero remaining bare `ignore` annotations after this pass. CI must run `cargo test --doc` and pass.

---

### Workstream C — Generic `Mesh<M>` (builds on A)

**Dependency:** Workstream A must be complete first.

#### Core Type

Using the K = number of vertices per simplex convention:

```rust
// alloc/std tier
// K = vertices per simplex (3 = triangle, 4 = tet)
// B = vertices per boundary face = K-1 (2 = edge, 3 = triangle)
pub struct Mesh<M: Manifold, const K: usize = 3, const B: usize = 2> {
    pub vertices: Vec<M::Point>,
    pub simplices: Vec<[usize; K]>,          // K=3 → triangles, K=4 → tets
    pub boundaries: Vec<[usize; B]>,          // B=K-1 vertices per boundary face; K faces per simplex
    pub boundary_signs: Vec<[f64; K]>,        // K signs per simplex (one per boundary face)
    _phantom: PhantomData<M>,
}

// Backward-compat alias (deleted after migration)
// Triangle mesh in 2D Euclidean space: K=3 vertices/simplex, B=2 vertices/boundary edge
pub type FlatMesh = Mesh<Euclidean<2>, 3, 2>;

// no_alloc tier
// K = vertices per simplex, B = vertices per boundary face (= K-1)
// V = max vertices, S = max simplices
pub struct FixedMesh<M: Manifold, const K: usize, const B: usize, const V: usize, const S: usize>
where M::Point: Copy
{
    pub vertices: [M::Point; V],
    pub simplices: [[usize; K]; S],
    pub boundaries: [[usize; B]; S],          // B = K-1 vertex indices per boundary face
    pub boundary_signs: [[f64; K]; S],
    pub n_vertices: usize,
    pub n_simplices: usize,
    _phantom: PhantomData<M>,
}
```

`B = K-1` is a separate const generic because Rust const generics do not support arithmetic expressions like `K-1` as array sizes. The caller must supply matching values (triangle: `K=3, B=2`; tet: `K=4, B=3`). A debug assertion `assert_eq!(B, K-1)` is added to `from_simplices` to catch mismatches at runtime.

#### Manifold Passed Per-Method

```rust
impl<M: Manifold, const K: usize> Mesh<M, K> {
    pub fn from_simplices(manifold: &M, vertices: Vec<M::Point>, simplices: Vec<[usize; K]>) -> Self
    pub fn edge_length(&self, manifold: &M, e: usize) -> Real
    pub fn triangle_area(&self, manifold: &M, t: usize) -> Real
    pub fn circumcenter(&self, manifold: &M, t: usize) -> M::Point
    pub fn check_well_centered(&self, manifold: &M) -> Result<(), DecError>
    pub fn euler_characteristic(&self) -> i32
}
```

#### Flat-Specific Builders

`unit_square_grid` and the Euclidean-specific circumcenter formula are moved to inherent impls on `Mesh<Euclidean<2>, 3>`:

```rust
impl Mesh<Euclidean<2>, 3> {
    pub fn unit_square_grid(n: usize) -> Self { ... }
    // flat circumcenter formula (faster than geodesic Newton for flat meshes)
    pub fn circumcenter_flat(&self, t: usize) -> <Euclidean<2> as Manifold>::Point { ... }
}
```

The generic `circumcenter` method on `Mesh<M, K>` uses the geodesic Newton method. The flat override on `Mesh<Euclidean<2>, 3>` uses the direct formula from the current implementation.

#### Metric Strategy Per Operation

| Operation | Strategy |
|-----------|-----------|
| Edge length | `manifold.dist(vi, vj)` — geodesic distance |
| Triangle area | `Log_{v0}(v1)` and `Log_{v0}(v2)` into `T_{v0}M`, cross product magnitude in ambient coords |
| Circumcenter | Equidistance system in `T_{v0}M` via 2-step Newton, map back via `exp_{v0}` |
| Well-centered check | Geodesic barycentric coords via sub-triangle areas in tangent space |

#### Assembled Operators Stay Pure Linear Algebra

```rust
pub struct Operators<M: Manifold> {
    pub laplace_beltrami: DMatrix<f64>,
    pub mass0: DVector<f64>,
    pub mass1: DVector<f64>,
    pub ext: ExteriorDerivative,    // pure topology, no M needed
    pub hodge: HodgeStar,           // pure scalars, no M needed
    _phantom: PhantomData<M>,
}

impl<M: Manifold> Operators<M> {
    // Restricted to triangle meshes (K=3, B=2). Operator assembly for tetrahedra
    // (K=4) is out of scope for this spec. A note is placed in code for future extension.
    pub fn from_mesh(mesh: &Mesh<M, 3, 2>, manifold: &M) -> Self { ... }
}
```

**Restriction:** `Operators::from_mesh` accepts only `Mesh<M, 3, 2>` (triangle meshes). Volumetric meshes (`K=4`) can be constructed and used for topology queries, but DEC operator assembly (Hodge stars, Laplacians) is triangle-only in this release. A `// TODO: extend to K=4` comment is placed at the `from_mesh` signature.

Manifold geometry consumed at construction time. Time-stepping uses only `DMatrix<f64>` ops — no manifold calls at runtime. `PhantomData<M>` is needed here since no field holds `M::Point`.

#### Backward Compatibility Path

1. Add `FlatMesh = Mesh<Euclidean<2>, 3>` alias
2. All existing tests pass via alias — CI must stay green at every commit
3. New tests: `Mesh<Sphere<3>, 3>` geodesic edge lengths, areas, circumcenters
4. New tests: `Mesh<SpecialOrthogonal<3>, 3>` topology construction + `d1 * d0 = 0`
5. Exactness test on non-trivial topology (mesh with hole, Euler characteristic ≠ 2)
6. Once all tests green: delete alias in final cleanup commit

---

## Testing Strategy

- All existing tests pass via `FlatMesh` alias throughout migration
- New unit tests: `Mesh<Sphere<3>, 3>` geodesic edge lengths, triangle areas, circumcenters
- New unit tests: `Mesh<SpecialOrthogonal<3>, 3>` topology construction
- Exactness test: `d1 * d0 = 0` on generic mesh (trivial and non-trivial topology)
- Weingarten tests: Riemannian HVP matches finite difference on Sphere, SO(3), Grassmann
- Doctest execution: `cargo test --doc` passes in CI
- `no_std` build tests:
  - `cargo build --no-default-features` (no_alloc tier) — must compile
  - `cargo build --no-default-features --features alloc` (alloc tier) — must compile
  - `cargo build` (std default) — must compile

---

## Sequencing

```
A (no_std flags)  ─────────────────────────────────► merge
B (TODOs)         ──────────────────────────────────► merge   (parallel with A)
C (Mesh<M>)                ────────────────────────► merge   (after A)
```

A and B can be developed in parallel. C starts only after A is merged.
