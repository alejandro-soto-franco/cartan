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

**Per-crate changes:**

| Crate | Changes |
|-------|---------|
| `cartan-core` | `#![cfg_attr(not(feature="std"), no_std)]`; `CartanError` implements `std::error::Error` behind `#[cfg(feature="std")]`, bare `Display` otherwise |
| `cartan-manifolds` | Same header; `nalgebra` and `rand_distr` both already support `no_std` — no math changes |
| `cartan-optim` | Same header; no `std`-only types used — clean pass |
| `cartan-geo` | Same header; `JacobiResult` (holds `Vec<T>`) and `Geodesic::sample` (returns `Vec`) gated behind `#[cfg(feature="alloc")]`; core methods `eval`, `length`, `midpoint` available in all tiers |
| `cartan-dec` | Feature flag structure added now; actual `no_std` content lands with Workstream C |
| `cartan` (facade) | Propagates feature flags to all members |

**`no_alloc` fixed-size alternatives in `cartan-geo`:**

```rust
// alloc tier (unchanged):
pub fn sample(&self, n: usize) -> Vec<M::Point> { ... }

// no_alloc tier:
pub fn sample_fixed<const N: usize>(&self) -> ([M::Point; N], usize) { ... }

// JacobiResult equivalent:
pub struct JacobiResultFixed<T, const N: usize> {
    pub params: [Real; N],
    pub field: [T; N],
    pub velocity: [T; N],
    pub len: usize,
}
```

**Workspace `Cargo.toml`:** All workspace dependencies get `default-features = false` so crate-level feature selection propagates correctly downstream.

---

### Workstream B — Contained TODOs (parallel with A)

#### B1 — Weingarten Correction on All 7 `Connection` Impls

The `Connection::riemannian_hessian_vector_product` signature already passes `grad_f` and `v` — they are currently ignored in all impls. Fix: use them.

Full formula: `Hess f(p)[v] = proj_p(D²f(p)[v]) + Weingarten(grad_f, v)`

Corrections per manifold:

| Manifold | Weingarten correction term |
|----------|-----------------------------|
| `Sphere<N>` | `- inner(grad_f, p) * v` — removes the `TODO(phase6)` comment |
| `SpecialOrthogonal<N>` | `- 0.5 * (grad_f * v^T - v * grad_f^T)` projected onto `T_p SO(N)` |
| `Spd<N>` | `- sym(P^{-1} * grad_f * P^{-1} * v)` (affine-invariant metric) |
| `Grassmann<N,K>` | `- (I - QQ^T) * grad_f^T * v` projected onto Stiefel tangent |
| `Euclidean<N>` | zero — flat, no correction |
| `Corr<N>` | zero — flat embedding, no correction |
| `SpecialEuclidean<N>` | rotation block: SO(N) correction; translation block: zero |

#### B2 — Full Tensor Ricci Correction in `cartan-dec`

Replace scalar `Option<f64>` with a vertex-indexed callback returning the full Ricci tensor:

```rust
// Before:
pub fn apply_bochner_laplacian(
    &self,
    u: &DVector<f64>,
    ricci_correction: Option<f64>,
) -> DVector<f64>

// After:
pub fn apply_bochner_laplacian(
    &self,
    u: &DVector<f64>,
    ricci_correction: Option<&dyn Fn(usize) -> [[f64; 2]; 2]>,
) -> DVector<f64>
```

- Callback takes a vertex index, returns the 2x2 Ricci tensor at that vertex
- For Einstein manifolds: caller passes `|_| [[k, 0.], [0., k]]` — identical result to current scalar path
- `None` still means zero correction — fully backward compatible

Same pattern for `apply_lichnerowicz_laplacian`: callback returns the curvature contraction `R_{ikjl} Q^{kl}` at each vertex as `[[f64; 3]; 3]` (symmetric 2-tensor to symmetric 2-tensor map).

#### B3 — Doctest Annotations in `cartan-manifolds`

Replace all bare `ignore` annotations:
- Self-contained examples with hardcoded inputs (exp, log, inner, dist, check_point): `#[doctest]` — actually executed
- Examples using `random_point` / `random_tangent`: `no_run` — compile-checked, not executed (non-deterministic)
- Zero remaining bare `ignore` annotations after this pass

---

### Workstream C — Generic `Mesh<M>` (builds on A)

**Dependency:** Workstream A must be complete first so the generic mesh is `no_std`-compatible from the start.

#### Core Type

```rust
pub struct Mesh<M: Manifold, const K: usize = 2> {
    pub vertices: Vec<M::Point>,           // was Vec<[f64; 2]>
    pub simplices: Vec<[usize; K]>,        // edges, triangles, or tets
    pub boundaries: Vec<[usize; K]>,       // (K-1)-faces per K-simplex
    pub boundary_signs: Vec<[f64; K]>,
    _phantom: PhantomData<M>,
}

// Backward-compat alias — deleted after all consumers migrated
pub type FlatMesh = Mesh<Euclidean<2>>;
```

`K=2` (triangles) is the default. `K=3` adds tetrahedral meshes for volumetric problems.

#### Manifold Passed Per-Method

The manifold is not stored in `Mesh`. Every geometric method takes `&M`:

```rust
impl<M: Manifold, const K: usize> Mesh<M, K> {
    pub fn edge_length(&self, manifold: &M, e: usize) -> Real { ... }
    pub fn triangle_area(&self, manifold: &M, t: usize) -> Real { ... }
    pub fn circumcenter(&self, manifold: &M, t: usize) -> M::Point { ... }
    pub fn check_well_centered(&self, manifold: &M) -> Result<(), DecError> { ... }
}
```

**Rationale:** All current cartan manifolds are zero-sized types — passing `&M` is a zero-cost no-op. `Operators::from_mesh` takes `&M` once and handles all threading internally. User-facing API stays ergonomic.

#### Metric Strategy Per Operation

| Operation | Strategy | Reason |
|-----------|-----------|--------|
| Edge length | `manifold.dist(vi, vj)` | Geodesic distance — exact |
| Triangle area | Log all vertices into `T_{v0}M`, cross product in ambient | Orientation-aware; exact for small triangles |
| Circumcenter | Equidistance solve in `T_{v0}M` via 2-step Newton, map back via `exp` | Geodesic circumcenter definition |
| Well-centered check | Geodesic barycentric coordinates (signs of sub-triangle areas in tangent space) | Consistent with circumcenter definition |

#### Assembled Operators Stay Pure Linear Algebra

```rust
pub struct Operators<M: Manifold> {
    pub laplace_beltrami: DMatrix<f64>,
    pub mass0: DVector<f64>,
    pub mass1: DVector<f64>,
    pub ext: ExteriorDerivative,    // pure topology, no M
    pub hodge: HodgeStar,           // pure scalars, no M
    _phantom: PhantomData<M>,
}

impl<M: Manifold> Operators<M> {
    pub fn from_mesh(mesh: &Mesh<M>, manifold: &M) -> Self { ... }
}
```

Manifold geometry is consumed at construction time to compute Hodge weights. The assembled operators are pure `DMatrix<f64>` — no manifold calls at time-stepping runtime. This preserves the cache-friendly sparse matrix-vector product structure.

#### `no_alloc` Fixed-Size Mesh

```rust
pub struct FixedMesh<M: Manifold, const K: usize, const V: usize, const E: usize, const S: usize> {
    pub vertices: [M::Point; V],
    pub simplices: [[usize; K]; S],
    pub boundaries: [[usize; K]; S],
    pub boundary_signs: [[f64; K]; S],
    _phantom: PhantomData<M>,
}
```

Capped at `K <= 3` (triangles + tetrahedra) for this tier. Fully stack-allocated.

#### Backward Compatibility Path

1. Add `FlatMesh = Mesh<Euclidean<2>>` alias
2. Existing tests pass unchanged through the alias
3. New tests added for `Mesh<Sphere<3>>` and `Mesh<SpecialOrthogonal<3>>`
4. Once all tests green, alias deleted in a final cleanup commit

---

## Trait Bounds Summary

| Tier | Mesh type | Trait requirement |
|------|-----------|-------------------|
| Full | `Mesh<M, K>` | `M: Manifold` |
| Geometric ops | `mesh.triangle_area(m, t)` | `M: Manifold` (exp + log + dist) |
| Circumcenter | `mesh.circumcenter(m, t)` | `M: Manifold` (exp + log) |
| Operators | `Operators<M>` | `M: Manifold` |
| Fixed-size | `FixedMesh<M, K, V, E, S>` | `M: Manifold`, `M::Point: Copy` |

---

## Testing Strategy

- All existing tests pass via `FlatMesh` alias throughout migration
- New unit tests: `Mesh<Sphere<3>>` geodesic edge lengths, triangle areas, circumcenters
- New unit tests: `Mesh<SpecialOrthogonal<3>>` topology construction
- Exactness test: `d1 * d0 = 0` on generic mesh
- Weingarten correction tests: verify Hessian-vector products match finite difference on sphere, SO(3), SPD
- Doctest execution: CI runs `cargo test --doc` and must pass
- `no_std` build test: `cargo build --no-default-features` and `cargo build --no-default-features --features alloc` must compile clean

---

## Sequencing

```
A (no_std flags) ─────────────────────────────────► merge
B (TODOs)        ──────────────────────────────────► merge   (parallel with A)
C (Mesh<M>)               ─────────────────────────► merge   (after A)
```

A and B can be developed in parallel. C starts after A is merged.
