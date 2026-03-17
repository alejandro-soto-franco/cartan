# Workstream C: Generic `Mesh<M>` — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `cartan-dec::Mesh` from a flat 2D type to `Mesh<M: Manifold, const K: usize = 3, const B: usize = 2>` so that any cartan manifold (Sphere, SO(N), etc.) can serve as the vertex space, while keeping all existing code working via a `FlatMesh = Mesh<Euclidean<2>, 3, 2>` type alias.

**Architecture:** In-place generalization. The current flat `Mesh` becomes `Mesh<Euclidean<2>, 3, 2>` via a rename + type alias. Flat-specific builders (`unit_square_grid`, flat `from_triangles`) move to an inherent impl on the specialized type. Generic metric methods (`edge_length`, `triangle_area`, `circumcenter`) are added to `impl<M: Manifold> Mesh<M, 3, 2>` using geodesic distance and tangent-space pullback. `Operators`, `HodgeStar`, and `ExteriorDerivative` are updated to accept the generic mesh. `advection.rs` and `divergence.rs` are pinned to `FlatMesh`. All existing tests pass via the alias throughout.

**Tech Stack:** Rust 1.85, `cartan-core` (Manifold trait), `cartan-manifolds` (Euclidean<2>, Sphere, SpecialOrthogonal), `nalgebra`. Depends on Workstream A being merged (all crates are `no_std` compatible).

**Spec:** `docs/superpowers/specs/2026-03-17-cartan-generalization-design.md` — Workstream C.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `cartan-dec/Cargo.toml` | Modify | Add `cartan-core` and `cartan-manifolds` workspace deps |
| `cartan-dec/src/mesh.rs` | Modify | Generic `Mesh<M, K, B>` struct, `FlatMesh` alias, `from_simplices`, metric methods |
| `cartan-dec/src/exterior.rs` | Modify | Update `from_mesh` to use renamed fields (`boundaries`, `simplex_boundary_ids`, `boundary_signs`) |
| `cartan-dec/src/hodge.rs` | Modify | Update `from_mesh` to take `&M`, use geodesic metric methods |
| `cartan-dec/src/laplace.rs` | Modify | Add `PhantomData<M>` to `Operators`, update `from_mesh` signature |
| `cartan-dec/src/advection.rs` | Modify | Change `&Mesh` → `&FlatMesh` (flat-only, no generalization needed) |
| `cartan-dec/src/divergence.rs` | Modify | Change `&Mesh` → `&FlatMesh` |
| `cartan-dec/src/lib.rs` | Modify | Export `FlatMesh`, update `Operators` re-export |
| `cartan-dec/tests/integration.rs` | Modify | Add `Mesh<Sphere<3>, 3>` and `Mesh<SpecialOrthogonal<3>, 3>` tests |

---

## Chunk 1: Cargo and Struct Skeleton

### Task 1: Add dependencies to `cartan-dec/Cargo.toml`

**Files:**
- Modify: `cartan-dec/Cargo.toml`

- [ ] **Step 1: Add cartan-core and cartan-manifolds**

Open `cartan-dec/Cargo.toml`. In the `[dependencies]` section, add:

```toml
cartan-core = { workspace = true }
cartan-manifolds = { workspace = true }
```

Verify `workspace = true` deps exist for both in the root `Cargo.toml`. They should already be there from Workstream A.

- [ ] **Step 2: Verify workspace compiles**

```bash
cargo build -p cartan-dec
```
Expected: PASS (no new code yet, just adding deps — may cause unused import warnings, ignore for now).

- [ ] **Step 3: Commit**

```bash
git add cartan-dec/Cargo.toml && git commit -m "feat(dec): add cartan-core and cartan-manifolds deps"
```

---

### Task 2: Generic `Mesh<M, K, B>` struct + `FlatMesh` alias

**Files:**
- Modify: `cartan-dec/src/mesh.rs`

This is the core structural change. Replace the concrete `struct Mesh` with `struct Mesh<M, K, B>` and add the `FlatMesh` alias. All method bodies stay identical for now — they will be updated in subsequent tasks.

- [ ] **Step 1: Write a failing test verifying `FlatMesh` alias works**

Add to `cartan-dec/tests/integration.rs` (or a new `#[cfg(test)]` block at the bottom of `mesh.rs`):

```rust
#[test]
fn test_flat_mesh_alias_compiles() {
    use cartan_dec::mesh::FlatMesh;
    // Build a small flat mesh via the alias — same API as the old Mesh.
    let mesh: FlatMesh = FlatMesh::from_triangles(
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        vec![[0, 1, 2]],
    );
    assert_eq!(mesh.n_vertices(), 3);
    assert_eq!(mesh.n_boundaries(), 3);
    assert_eq!(mesh.n_simplices(), 1);
}
```

- [ ] **Step 2: Run the test — expect compile error (FlatMesh not yet defined)**

```bash
cargo test -p cartan-dec test_flat_mesh_alias_compiles -- --nocapture 2>&1 | head -20
```
Expected: compile error "cannot find type `FlatMesh`".

- [ ] **Step 3: Rewrite `cartan-dec/src/mesh.rs`**

Replace the entire file with the generic version. The new file has three sections:
(A) the generic struct definition and FlatMesh alias,
(B) the generic impl block (topology-only methods, no metric),
(C) the `Mesh<Euclidean<2>, 3, 2>` specialized impl block (flat builders + flat metric methods).

```rust
// ~/cartan/cartan-dec/src/mesh.rs

//! Simplicial complex: vertices, (oriented) edges, and (oriented) triangles.
//!
//! ## Orientation conventions
//!
//! **Edges** `[i, j]` are stored with `i < j`. The signed boundary operator
//! assigns +1 to vertex j and -1 to vertex i for the canonical orientation.
//!
//! **Triangles** `[i, j, k]` are stored in counter-clockwise orientation as
//! seen from outside the manifold. The signed boundary operator assigns +1 to
//! edge (j,k), -1 to edge (i,k), and +1 to edge (i,j).
//!
//! ## Generic mesh
//!
//! `Mesh<M, const K, const B>` replaces the old flat `Mesh` type. `K` is the
//! number of vertices per simplex (K=3 for triangles) and `B = K-1` is the
//! number of vertices per boundary face (B=2 for edges). The `FlatMesh` type
//! alias provides full backward compatibility.
//!
//! ## References
//!
//! - Desbrun et al. "Discrete Exterior Calculus." arXiv:math/0508341, 2005.
//! - Hirani. "Discrete Exterior Calculus." PhD thesis, Caltech, 2003.

use core::marker::PhantomData;
use std::collections::HashMap;

use nalgebra::Vector2;

use cartan_core::{Manifold, Real};
use cartan_manifolds::euclidean::Euclidean;

use crate::error::DecError;

/// A simplicial complex with vertices on a Riemannian manifold.
///
/// # Type parameters
///
/// - `M`: The manifold. Vertex coordinates are `M::Point`.
/// - `K`: Vertices per simplex (K=3 = triangle, K=4 = tetrahedron).
/// - `B`: Vertices per boundary face = K-1 (B=2 = edge, B=3 = triangle face).
///
/// `B = K-1` is a separate const generic because Rust const generics do not
/// support arithmetic expressions like `K-1` as array sizes. A runtime
/// `debug_assert_eq!(B, K-1)` in `from_simplices` catches mismatches.
#[derive(Debug, Clone)]
pub struct Mesh<M: Manifold, const K: usize = 3, const B: usize = 2> {
    /// Vertex positions (one M::Point per vertex).
    pub vertices: Vec<M::Point>,

    /// Simplices: each entry is K vertex indices. K=3 → triangles.
    pub simplices: Vec<[usize; K]>,

    /// Boundary faces: each entry is B vertex indices (B = K-1).
    /// Deduplicated globally; for triangles these are the edges.
    pub boundaries: Vec<[usize; B]>,

    /// For each simplex, the K indices into `self.boundaries` of its boundary faces.
    ///
    /// **Note:** This field is not in the spec's struct definition, which only lists
    /// `vertices`, `simplices`, `boundaries`, and `boundary_signs`. It is added here
    /// because `boundary_signs` records ±1 per boundary slot but without this field
    /// there is no way to know *which* boundary index each sign slot corresponds to.
    /// Without it, `ExteriorDerivative::from_mesh` cannot be built. The spec struct
    /// definition is incomplete as written; this field is the correct addition.
    pub simplex_boundary_ids: Vec<[usize; K]>,

    /// For each simplex, the K signs of its boundary faces (±1.0).
    pub boundary_signs: Vec<[f64; K]>,

    _phantom: PhantomData<M>,
}

/// Backward-compatible type alias: flat 2D triangular mesh.
///
/// All existing code using `Mesh` continues to work via this alias.
/// The alias will be removed after full migration.
pub type FlatMesh = Mesh<Euclidean<2>, 3, 2>;

// ─────────────────────────────────────────────────────────────────────────────
// Generic impl — topology only (no metric)
// ─────────────────────────────────────────────────────────────────────────────

impl<M: Manifold, const K: usize, const B: usize> Mesh<M, K, B> {
    /// Number of vertices.
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of boundary faces (edges for K=3).
    pub fn n_boundaries(&self) -> usize {
        self.boundaries.len()
    }

    /// Number of simplices (triangles for K=3).
    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// Euler characteristic: V - E + F (for 2D: n_vertices - n_edges + n_triangles).
    pub fn euler_characteristic(&self) -> i32 {
        self.n_vertices() as i32 - self.n_boundaries() as i32 + self.n_simplices() as i32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle mesh constructor (K=3, B=2) — works for any M
// ─────────────────────────────────────────────────────────────────────────────

impl<M: Manifold> Mesh<M, 3, 2> {
    /// Construct a triangle mesh from vertex positions and a triangle list.
    ///
    /// Edges are deduplicated and canonically oriented (lower-index vertex first).
    /// Triangle orientation (CCW) is assumed as given by the caller.
    /// `manifold` is not used here but is required to fix the type parameter.
    ///
    /// # Panics
    ///
    /// Panics if any triangle vertex index is out of bounds.
    /// Debug-asserts B == K-1 == 2.
    pub fn from_simplices(_manifold: &M, vertices: Vec<M::Point>, triangles: Vec<[usize; 3]>) -> Self {
        // B = K-1 = 2 for triangles — holds by construction on this monomorphized impl.
        // This assertion is documentary only (it checks literals 2 == 2, always true).
        // A future generic `impl<M, const K, const B>` would enforce B == K-1 at runtime.
        debug_assert_eq!(2, 3_usize - 1, "B must equal K-1 for triangle mesh");
        let n_v = vertices.len();
        let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut simplex_boundary_ids = Vec::with_capacity(triangles.len());
        let mut boundary_signs = Vec::with_capacity(triangles.len());

        for &[i, j, k] in &triangles {
            assert!(i < n_v && j < n_v && k < n_v, "triangle index out of bounds");

            // The three edges of triangle [i, j, k]:
            //   (i,j): boundary coeff = +1
            //   (j,k): boundary coeff = +1
            //   (i,k): boundary coeff = -1 (opposite orientation)
            let raw_edges = [(i, j), (j, k), (i, k)];
            let boundary_coeffs = [1.0f64, 1.0, -1.0];
            let mut local_edge_ids = [0usize; 3];
            let mut local_signs = [0.0f64; 3];

            for (slot, (&(a, b), &coeff)) in raw_edges.iter().zip(boundary_coeffs.iter()).enumerate() {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let direction_sign = if a < b { 1.0 } else { -1.0 };
                let idx = *edge_map.entry((lo, hi)).or_insert_with(|| {
                    let e = edges.len();
                    edges.push([lo, hi]);
                    e
                });
                local_edge_ids[slot] = idx;
                local_signs[slot] = coeff * direction_sign;
            }

            simplex_boundary_ids.push(local_edge_ids);
            boundary_signs.push(local_signs);
        }

        Self {
            vertices,
            simplices: triangles,
            boundaries: edges,
            simplex_boundary_ids,
            boundary_signs,
            _phantom: PhantomData,
        }
    }

    /// Geodesic edge length: `manifold.dist(v_i, v_j)`.
    pub fn edge_length(&self, manifold: &M, e: usize) -> Real {
        let [i, j] = self.boundaries[e];
        manifold.dist(&self.vertices[i], &self.vertices[j])
            .expect("edge_length: log failed (antipodal vertices?)")
    }

    /// Triangle area via tangent-space Gram determinant.
    ///
    /// Maps v1 → u = Log_{v0}(v1) and v2 → v = Log_{v0}(v2) into T_{v0}M,
    /// then computes `0.5 * sqrt(||u||²||v||² - <u,v>²)`.
    pub fn triangle_area(&self, manifold: &M, t: usize) -> Real {
        let [i, j, k] = self.simplices[t];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let v2 = &self.vertices[k];
        let u = manifold.log(v0, v1).unwrap_or_else(|_| manifold.zero_tangent(v0));
        let v = manifold.log(v0, v2).unwrap_or_else(|_| manifold.zero_tangent(v0));
        let uu = manifold.inner(v0, &u, &u);
        let vv = manifold.inner(v0, &v, &v);
        let uv = manifold.inner(v0, &u, &v);
        0.5 * (uu * vv - uv * uv).abs().sqrt()
    }

    /// Circumcenter via tangent-space 2×2 equidistance system at v0.
    ///
    /// Solves for (s, t) such that the point `exp_{v0}(s·u + t·v)` is
    /// equidistant (in the Riemannian metric) from v0, v1, and v2,
    /// where u = Log_{v0}(v1), v = Log_{v0}(v2). Falls back to v0 on
    /// degenerate (near-zero-area) triangles.
    ///
    /// For flat Euclidean meshes, this produces the exact circumcenter.
    /// For curved manifolds, this is a first-order tangent-space approximation.
    pub fn circumcenter(&self, manifold: &M, t: usize) -> M::Point {
        let [i, j, k] = self.simplices[t];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let v2 = &self.vertices[k];
        let u = manifold.log(v0, v1).unwrap_or_else(|_| manifold.zero_tangent(v0));
        let v = manifold.log(v0, v2).unwrap_or_else(|_| manifold.zero_tangent(v0));
        let uu = manifold.inner(v0, &u, &u);
        let vv = manifold.inner(v0, &v, &v);
        let uv = manifold.inner(v0, &u, &v);
        let det = uu * vv - uv * uv;
        if det.abs() < 1e-30 {
            return v0.clone(); // degenerate: fall back to v0
        }
        // Solve [[uu, uv], [uv, vv]] [s; tc] = [0.5*uu; 0.5*vv]
        // Cramer's rule: M^{-1} = (1/det) [[vv, -uv], [-uv, uu]]
        let s = (vv * 0.5 * uu - uv * 0.5 * vv) / det;
        let tc = (uu * 0.5 * vv - uv * 0.5 * uu) / det;
        let tangent = u * s + v * tc;
        manifold.exp(v0, &tangent)
    }

    /// Geodesic midpoint of boundary face (edge) e.
    pub fn boundary_midpoint(&self, manifold: &M, e: usize) -> M::Point {
        let [i, j] = self.boundaries[e];
        let v0 = &self.vertices[i];
        let v1 = &self.vertices[j];
        let half_log = manifold.log(v0, v1)
            .map(|u| u * 0.5)
            .unwrap_or_else(|_| manifold.zero_tangent(v0));
        manifold.exp(v0, &half_log)
    }

    /// Check whether this mesh is well-centered (all circumcenters inside their triangles).
    ///
    /// Uses the tangent-space circumcenter barycentric coordinates (s, tc) computed
    /// from the same system as `circumcenter()`. The circumcenter is inside the
    /// triangle iff s > 0, tc > 0, and s + tc < 1 (all three barycentric coords
    /// are positive). This avoids the Gram-determinant sub-area approach, which
    /// always produces non-negative values via `.sqrt()` and cannot detect
    /// whether the circumcenter is outside the triangle.
    pub fn check_well_centered(&self, manifold: &M) -> Result<(), DecError> {
        for t in 0..self.n_simplices() {
            let [i, j, k] = self.simplices[t];
            let v0 = &self.vertices[i];
            let v1 = &self.vertices[j];
            let v2 = &self.vertices[k];
            let u = manifold.log(v0, v1).unwrap_or_else(|_| manifold.zero_tangent(v0));
            let v = manifold.log(v0, v2).unwrap_or_else(|_| manifold.zero_tangent(v0));
            let uu = manifold.inner(v0, &u, &u);
            let vv = manifold.inner(v0, &v, &v);
            let uv = manifold.inner(v0, &u, &v);
            let det = uu * vv - uv * uv;
            if det.abs() < 1e-30 {
                continue; // degenerate, skip
            }
            // Circumcenter tangent-space barycentric coords at v0:
            // cc_tangent = s*u + tc*v   (same system as circumcenter())
            let s  = (vv * 0.5 * uu - uv * 0.5 * vv) / det;
            let tc = (uu * 0.5 * vv - uv * 0.5 * uu) / det;
            // Well-centered iff circumcenter strictly inside: s>0, tc>0, s+tc<1
            if s <= 0.0 || tc <= 0.0 || s + tc >= 1.0 {
                return Err(DecError::NotWellCentered {
                    simplex: t,
                    volume: s.min(tc).min(1.0 - s - tc),
                });
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlatMesh-specific impl — backward-compatible flat builders and metric helpers
// ─────────────────────────────────────────────────────────────────────────────

impl Mesh<Euclidean<2>, 3, 2> {
    /// Construct a flat mesh from raw `[f64; 2]` vertex positions.
    ///
    /// Backward-compatible entry point. Converts positions to `SVector<f64, 2>`
    /// (which is `Euclidean<2>::Point`) and calls `from_simplices`.
    pub fn from_triangles(vertices: Vec<[f64; 2]>, triangles: Vec<[usize; 3]>) -> Self {
        use nalgebra::SVector;
        let manifold = Euclidean::<2>;
        let mv: Vec<SVector<f64, 2>> = vertices.iter()
            .map(|&[x, y]| SVector::from([x, y]))
            .collect();
        Self::from_simplices(&manifold, mv, triangles)
    }

    /// Vertex position as a nalgebra Vector2.
    ///
    /// Uses `.x`/`.y` accessors (not `[0]`/`[1]`) to avoid ambiguity between
    /// nalgebra's matrix index `[(row, col)]` and slice index `[usize]`.
    pub fn vertex(&self, i: usize) -> Vector2<f64> {
        Vector2::new(self.vertices[i].x, self.vertices[i].y)
    }

    /// Flat edge midpoint (arithmetic mean of endpoints).
    pub fn edge_midpoint(&self, e: usize) -> Vector2<f64> {
        let [i, j] = self.boundaries[e];
        (self.vertex(i) + self.vertex(j)) * 0.5
    }

    /// Flat edge length (Euclidean norm).
    ///
    /// Faster than the generic geodesic method for flat meshes.
    pub fn edge_length_flat(&self, e: usize) -> f64 {
        let [i, j] = self.boundaries[e];
        (self.vertex(j) - self.vertex(i)).norm()
    }

    /// Flat triangle area (signed, CCW positive).
    ///
    /// Faster than the generic Gram determinant method for flat meshes.
    pub fn triangle_area_flat(&self, t: usize) -> f64 {
        let [i, j, k] = self.simplices[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);
        let ab = b - a;
        let ac = c - a;
        0.5 * (ab.x * ac.y - ab.y * ac.x)
    }

    /// Flat circumcenter (exact formula for 2D Euclidean mesh).
    ///
    /// Faster than the generic tangent-space method for flat meshes.
    pub fn circumcenter_flat(&self, t: usize) -> Vector2<f64> {
        let [i, j, k] = self.simplices[t];
        let a = self.vertex(i);
        let b = self.vertex(j);
        let c = self.vertex(k);
        let ab = b - a;
        let ac = c - a;
        let d = 2.0 * (ab.x * ac.y - ab.y * ac.x);
        if d.abs() < 1e-30 {
            return (a + b + c) / 3.0;
        }
        let ux = (ac.y * (ab.x * ab.x + ab.y * ab.y) - ab.y * (ac.x * ac.x + ac.y * ac.y)) / d;
        let uy = (ab.x * (ac.x * ac.x + ac.y * ac.y) - ac.x * (ab.x * ab.x + ab.y * ab.y)) / d;
        a + Vector2::new(ux, uy)
    }

    /// Build a simple uniform triangulated grid on [0,1]² with `n` divisions per side.
    ///
    /// Produces 2n² right triangles. Well-centered for all n.
    pub fn unit_square_grid(n: usize) -> Self {
        assert!(n >= 1, "grid must have at least 1 division");
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                vertices.push([i as f64 / n as f64, j as f64 / n as f64]);
            }
        }
        let idx = |i: usize, j: usize| j * (n + 1) + i;
        for j in 0..n {
            for i in 0..n {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                triangles.push([v00, v10, v01]);
                triangles.push([v10, v11, v01]);
            }
        }
        Self::from_triangles(vertices, triangles)
    }

    // Old names forwarded for backward compat with hodge.rs and exterior.rs
    // (will be removed after those modules are updated in Tasks 3–4).

    /// Deprecated: use `n_boundaries()` instead. Kept for migration.
    pub fn n_edges(&self) -> usize { self.n_boundaries() }

    /// Deprecated: use `n_simplices()` instead. Kept for migration.
    pub fn n_triangles(&self) -> usize { self.n_simplices() }

    // Expose old field names for migration period.
    // hodge.rs and exterior.rs still reference `mesh.edges`, `mesh.triangles`,
    // `mesh.tri_edges`, `mesh.tri_edge_signs` — those are updated in Tasks 3 and 4.
}
```

- [ ] **Step 4: Run the test and confirm it passes**

```bash
cargo test -p cartan-dec test_flat_mesh_alias_compiles -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Run the full test suite (it will fail — callers of old Mesh still broken)**

```bash
cargo test -p cartan-dec 2>&1 | tail -30
```
Expected: compile errors in `exterior.rs`, `hodge.rs`, `laplace.rs`, `advection.rs`, `divergence.rs` because they reference `Mesh` (not `FlatMesh`) and use old field names. Record which files fail — these are fixed in Tasks 3–6.

- [ ] **Step 6: Commit**

```bash
git add cartan-dec/src/mesh.rs && git commit -m "feat(dec): generic Mesh<M, K, B> + FlatMesh alias + per-manifold metric methods"
```

---

## Chunk 2: Updating Internal Consumers

### Task 3: Update `exterior.rs`

**Files:**
- Modify: `cartan-dec/src/exterior.rs`

The exterior derivative is purely topological. Only field name changes needed: `Mesh` → `FlatMesh`, `edges` → `boundaries`, `tri_edges` → `simplex_boundary_ids`, `tri_edge_signs` → `boundary_signs`, `n_edges()` → `n_boundaries()`, `n_triangles()` → `n_simplices()`.

- [ ] **Step 1: No new test needed (existing exactness tests cover this)**

The existing test `d1 * d0 = 0` in `integration.rs` already validates the exterior derivative. It will pass once the field names are updated.

- [ ] **Step 2: Update `exterior.rs`**

Replace the `from_mesh` signature and body:

```rust
use crate::mesh::FlatMesh;

impl ExteriorDerivative {
    pub fn from_mesh(mesh: &FlatMesh) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // d₀: n_edges × n_vertices
        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

        // d₁: n_triangles × n_edges
        let mut d1 = DMatrix::<f64>::zeros(nt, ne);
        for (t, (local_e, local_s)) in mesh.simplex_boundary_ids.iter()
            .zip(mesh.boundary_signs.iter()).enumerate()
        {
            for k in 0..3 {
                d1[(t, local_e[k])] = local_s[k];
            }
        }

        Self { d0, d1 }
    }
    // check_exactness stays unchanged
}
```

- [ ] **Step 3: Build to check**

```bash
cargo build -p cartan-dec 2>&1 | grep "exterior"
```
Expected: no errors from `exterior.rs`.

- [ ] **Step 4: Commit**

```bash
git add cartan-dec/src/exterior.rs && git commit -m "fix(dec): update ExteriorDerivative to use renamed mesh fields"
```

---

### Task 4: Update `hodge.rs`

**Files:**
- Modify: `cartan-dec/src/hodge.rs`

Update `HodgeStar::from_mesh` to take both `mesh: &FlatMesh` and `manifold: &Euclidean<2>`. Use the flat metric methods (`triangle_area_flat`, `edge_length_flat`, `circumcenter_flat`, `edge_midpoint`) on the specialized type. This keeps the existing behavior unchanged while updating the API to accept the manifold parameter.

- [ ] **Step 1: Write a failing test for the updated signature**

```rust
#[test]
fn test_hodge_from_mesh_with_manifold() {
    use cartan_dec::mesh::FlatMesh;
    use cartan_dec::hodge::HodgeStar;
    use cartan_manifolds::euclidean::Euclidean;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let hodge = HodgeStar::from_mesh(&mesh, &manifold);
    // ⋆₀ entries are positive (dual cell areas)
    assert!(hodge.star0.iter().all(|&x| x > 0.0), "all star0 entries must be positive");
    // ⋆₂ entries are positive (1/area)
    assert!(hodge.star2.iter().all(|&x| x > 0.0), "all star2 entries must be positive");
}
```

- [ ] **Step 2: Run and confirm failure (old signature)**

```bash
cargo test -p cartan-dec test_hodge_from_mesh_with_manifold -- --nocapture 2>&1 | head -10
```
Expected: compile error (wrong signature).

- [ ] **Step 3: Update `hodge.rs`**

Update the import and signature. Replace the function header and update all metric calls to use the flat-specific methods:

```rust
use crate::mesh::FlatMesh;
use cartan_manifolds::euclidean::Euclidean;

impl HodgeStar {
    pub fn from_mesh(mesh: &FlatMesh, _manifold: &Euclidean<2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        // ── ⋆₂: 1 / triangle area ──────────────────────────────────────────
        let mut star2 = DVector::<f64>::zeros(nt);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            star2[t] = if area > 1e-30 { 1.0 / area } else { 0.0 };
        }

        // ── ⋆₀: barycentric dual cell area = (1/3) Σ_{t ∋ v} area(t) ───────
        let mut star0 = DVector::<f64>::zeros(nv);
        for t in 0..nt {
            let area = mesh.triangle_area_flat(t).abs();
            for &v in &mesh.simplices[t] {
                star0[v] += area / 3.0;
            }
        }

        // ── ⋆₁: |dual edge| / |primal edge| ──────────────────────────────
        let mut edge_tris: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for (t, local_e) in mesh.simplex_boundary_ids.iter().enumerate() {
            for &e in local_e {
                edge_tris[e].push(t);
            }
        }

        let mut star1 = DVector::<f64>::zeros(ne);
        for e in 0..ne {
            let primal_len = mesh.edge_length_flat(e);
            if primal_len < 1e-30 {
                star1[e] = 0.0;
                continue;
            }
            let dual_len = match edge_tris[e].as_slice() {
                [t1, t2] => {
                    let c1 = mesh.circumcenter_flat(*t1);
                    let c2 = mesh.circumcenter_flat(*t2);
                    (c2 - c1).norm()
                }
                [t1] => {
                    let c = mesh.circumcenter_flat(*t1);
                    let mid = mesh.edge_midpoint(e);
                    (c - mid).norm()
                }
                _ => 0.0,
            };
            star1[e] = dual_len / primal_len;
        }

        Self { star0, star1, star2 }
    }
}
```

- [ ] **Step 4: Run test and confirm pass**

```bash
cargo test -p cartan-dec test_hodge_from_mesh_with_manifold -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Run full test suite**

```bash
cargo test -p cartan-dec
```
Expected: all pass (laplace.rs and advection.rs may still fail — fixed in Tasks 5–6).

- [ ] **Step 6: Commit**

```bash
git add cartan-dec/src/hodge.rs && git commit -m "fix(dec): update HodgeStar::from_mesh to accept manifold parameter"
```

---

### Task 5: Update `laplace.rs`

**Files:**
- Modify: `cartan-dec/src/laplace.rs`

Add `PhantomData<M>` to `Operators`, update `from_mesh` signature to accept `&M` (for future generic support). For the flat case keep using flat-specific methods.

> Note: `apply_bochner_laplacian` and `apply_lichnerowicz_laplacian` signatures are being updated in Workstream B (Task 5). If B is merged first, skip the scalar→callback changes here; if C is merged first, update the scalar signatures in B. The two changes are independent and do not conflict.

- [ ] **Step 1: Write a failing test for the new Operators signature**

```rust
#[test]
fn test_operators_from_mesh_with_manifold() {
    use cartan_dec::{Operators, mesh::FlatMesh};
    use cartan_manifolds::euclidean::Euclidean;

    let mesh = FlatMesh::unit_square_grid(4);
    let manifold = Euclidean::<2>;
    let ops = Operators::from_mesh(&mesh, &manifold);
    let nv = mesh.n_vertices();
    assert_eq!(ops.laplace_beltrami.nrows(), nv);
    assert_eq!(ops.laplace_beltrami.ncols(), nv);
}
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p cartan-dec test_operators_from_mesh_with_manifold -- --nocapture 2>&1 | head -10
```

- [ ] **Step 3: Update `laplace.rs`**

```rust
use core::marker::PhantomData;
use cartan_core::Manifold;
use crate::mesh::FlatMesh;

pub struct Operators<M: Manifold = cartan_manifolds::euclidean::Euclidean<2>> {
    pub laplace_beltrami: DMatrix<f64>,
    pub mass0: DVector<f64>,
    pub mass1: DVector<f64>,
    pub ext: ExteriorDerivative,
    pub hodge: HodgeStar,
    _phantom: PhantomData<M>,
}
```

Update `from_mesh`:
```rust
impl<M: Manifold> Operators<M> {
    /// Assemble all discrete operators from a triangle mesh.
    ///
    /// Restricted to triangle meshes (K=3, B=2). For future extension to
    /// tetrahedral meshes (K=4), see TODO below.
    ///
    /// # TODO
    /// Extend to K=4 volumetric meshes (Hodge stars in 3D).
    pub fn from_mesh(mesh: &FlatMesh, manifold: &M) -> Self
    where
        M: std::ops::Deref<Target = cartan_manifolds::euclidean::Euclidean<2>>,
    {
```

Actually, the constraint `M: Deref<Target = Euclidean<2>>` is too restrictive. The real constraint is that `from_mesh` currently needs a `FlatMesh` and uses flat metric methods. The `&M` parameter is passed through to `HodgeStar::from_mesh`. But `HodgeStar::from_mesh` takes `&Euclidean<2>`.

Simplest correct approach: keep `Operators` generic over M for future use, but `from_mesh` only implemented for `Euclidean<2>`:

```rust
impl Operators<cartan_manifolds::euclidean::Euclidean<2>> {
    pub fn from_mesh(mesh: &FlatMesh, manifold: &cartan_manifolds::euclidean::Euclidean<2>) -> Self {
        let ext = ExteriorDerivative::from_mesh(mesh);
        let hodge = HodgeStar::from_mesh(mesh, manifold);
        // ... rest of assembly unchanged
        Self {
            laplace_beltrami,
            mass0: hodge.star0.clone(),
            mass1: hodge.star1.clone(),
            ext,
            hodge,
            _phantom: PhantomData,
        }
    }
}
```

Full updated `laplace.rs` `from_mesh` body (read the current body first to preserve it, just update the signature and add PhantomData):

Open `cartan-dec/src/laplace.rs`. Find `impl Operators {` and change to:

```rust
impl Operators<cartan_manifolds::euclidean::Euclidean<2>> {
    pub fn from_mesh(mesh: &FlatMesh, manifold: &cartan_manifolds::euclidean::Euclidean<2>) -> Self {
```

Add `_phantom: PhantomData` to the `Self { ... }` construction at the end of `from_mesh`.
Add `_phantom: PhantomData<M>` field to the `Operators` struct.

Change the field `pub struct Operators` to `pub struct Operators<M: Manifold = cartan_manifolds::euclidean::Euclidean<2>>`.

The `apply_*` methods (`apply_laplace_beltrami`, `apply_bochner_laplacian`, `apply_lichnerowicz_laplacian`) use only `self.laplace_beltrami`, `self.mass0`, `self.mass1` — no `M::Point` operations — so they are valid in `impl<M: Manifold> Operators<M>`. Move them to this generic impl block. Only `from_mesh` goes in `impl Operators<Euclidean<2>>`.

- [ ] **Step 4: Run test and confirm pass**

```bash
cargo test -p cartan-dec test_operators_from_mesh_with_manifold -- --nocapture
```

- [ ] **Step 5: Full test suite**

```bash
cargo test -p cartan-dec
```
Expected: laplace-specific tests pass; advection/divergence may still fail.

- [ ] **Step 6: Commit**

```bash
git add cartan-dec/src/laplace.rs && git commit -m "fix(dec): add PhantomData<M> to Operators, update from_mesh signature"
```

---

### Task 6: Update `advection.rs` and `divergence.rs`

**Files:**
- Modify: `cartan-dec/src/advection.rs`
- Modify: `cartan-dec/src/divergence.rs`

These functions operate on flat vertex coordinates and are intentionally kept flat-only. Change their signatures from `mesh: &Mesh` to `mesh: &FlatMesh`. The function bodies stay identical since `FlatMesh::vertex()` returns `Vector2<f64>` as before, and `mesh.edges` → `mesh.boundaries` is the only field change.

- [ ] **Step 1: Update `advection.rs`**

```rust
// Change:
use crate::mesh::Mesh;
// To:
use crate::mesh::FlatMesh;

// Change all function signatures from:
pub fn apply_scalar_advection(mesh: &Mesh, ...) -> ...
pub fn apply_vector_advection(mesh: &Mesh, ...) -> ...
// To:
pub fn apply_scalar_advection(mesh: &FlatMesh, ...) -> ...
pub fn apply_vector_advection(mesh: &FlatMesh, ...) -> ...
```

In the function bodies, change `mesh.edges` → `mesh.boundaries`:
```rust
// Change:
for &[i, j] in &mesh.edges {
// To:
for &[i, j] in &mesh.boundaries {
```

- [ ] **Step 2: Update `divergence.rs`**

Same pattern:
```rust
use crate::mesh::FlatMesh;  // was: Mesh

// Signatures:
pub fn apply_divergence(mesh: &FlatMesh, ...)
pub fn apply_tensor_divergence(mesh: &FlatMesh, ...)

// Field:
for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {  // was: mesh.edges
```

- [ ] **Step 3: Full build**

```bash
cargo build -p cartan-dec
```
Expected: clean compile.

- [ ] **Step 4: Full test suite**

```bash
cargo test -p cartan-dec
```
Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add cartan-dec/src/advection.rs cartan-dec/src/divergence.rs
git commit -m "fix(dec): pin advection and divergence to FlatMesh (flat-only)"
```

---

### Task 7: Update `lib.rs` exports

**Files:**
- Modify: `cartan-dec/src/lib.rs`

Add `FlatMesh` to the public interface and update the `Operators` re-export.

- [ ] **Step 1: Update `lib.rs`**

```rust
pub use mesh::{FlatMesh, Mesh};  // was: pub use mesh::Mesh
```

Also update the quick-start doctest in `lib.rs` — the `Operators::from_mesh` call now requires a second `&manifold` argument. Failure to update this causes `cargo test --doc` to fail. The updated example:

```rust
//! ```rust,no_run
//! use cartan_dec::{FlatMesh, Operators};
//! use cartan_manifolds::euclidean::Euclidean;
//! use nalgebra::DVector;
//!
//! let mesh = FlatMesh::unit_square_grid(4);
//! let manifold = Euclidean::<2>;
//! let ops = Operators::from_mesh(&mesh, &manifold);
//!
//! let f = DVector::from_element(mesh.n_vertices(), 1.0);
//! let lf = ops.apply_laplace_beltrami(&f);
//! ```
```

- [ ] **Step 2: Full test suite**

```bash
cargo test -p cartan-dec
```
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add cartan-dec/src/lib.rs && git commit -m "fix(dec): export FlatMesh alias from cartan-dec crate root"
```

---

## Chunk 3: Generic Mesh Tests

### Task 8: New tests for generic mesh

**Files:**
- Modify: `cartan-dec/tests/integration.rs` (or add `cartan-dec/tests/generic_mesh.rs`)

These tests verify that `Mesh<M, 3, 2>` works for non-flat manifolds. They use the generic `from_simplices` constructor and test topology + metric methods.

- [ ] **Step 1: Write failing tests (generic mesh not yet tested)**

Add to `cartan-dec/tests/integration.rs`:

```rust
#[cfg(test)]
mod generic_mesh_tests {
    use cartan_dec::Mesh;
    use cartan_manifolds::sphere::Sphere;
    use cartan_manifolds::so::SpecialOrthogonal;
    use cartan_core::Manifold;
    use nalgebra::SVector;

    // ── Sphere mesh tests ──────────────────────────────────────────────────

    #[test]
    fn test_sphere_mesh_geodesic_edge_length() {
        // Place 3 orthonormal points on S^2: e1, e2, e3.
        // Geodesic distance between any two: acos(e_i · e_j) = pi/2.
        let s2 = Sphere::<3>;
        let e1: SVector<f64, 3> = SVector::from([1.0, 0.0, 0.0]);
        let e2: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
        let e3: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);
        let mesh = Mesh::from_simplices(&s2, vec![e1, e2, e3], vec![[0, 1, 2]]);

        // Edge 0: between e1 and e2 (or e1 and e3, etc. — depends on edge order)
        for e in 0..mesh.n_boundaries() {
            let len = mesh.edge_length(&s2, e);
            let expected = std::f64::consts::PI / 2.0;  // acos(0) = pi/2
            assert!((len - expected).abs() < 1e-10,
                "Sphere edge {e}: expected {expected}, got {len}");
        }
    }

    #[test]
    fn test_sphere_mesh_triangle_area() {
        // Spherical triangle with three orthonormal vertices spans 1/8 of S^2.
        // Spherical excess: E = pi/2 + pi/2 + pi/2 - pi = pi/2.
        // Area on unit sphere = pi/2 (Gauss-Bonnet: area = angular excess).
        // Tangent-space Gram determinant gives a FIRST-ORDER approximation,
        // not exact spherical area. For the equilateral spherical triangle,
        // the approximation overestimates slightly. Test for rough magnitude.
        let s2 = Sphere::<3>;
        let e1: SVector<f64, 3> = SVector::from([1.0, 0.0, 0.0]);
        let e2: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
        let e3: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);
        let mesh = Mesh::from_simplices(&s2, vec![e1, e2, e3], vec![[0, 1, 2]]);

        let area = mesh.triangle_area(&s2, 0);
        // Exact spherical area = pi/2 ≈ 1.571. Tangent-space approx gives 0.5*sqrt(||u||^2*||v||^2 - <u,v>^2).
        // At v0=e1: u = Log_{e1}(e2) = [0,pi/2,0]^T projected, v = Log_{e1}(e3) = [0,0,pi/2]^T projected.
        // ||u||^2 = (pi/2)^2, ||v||^2 = (pi/2)^2, <u,v> = 0 (orthogonal tangent vecs).
        // area = 0.5 * (pi/2)^2 ≈ 0.5 * 2.467 ≈ 1.234.
        // The tangent-space area is in reasonable range (differs from spherical by ~20%):
        assert!(area > 0.5 && area < 2.0,
            "Sphere triangle area should be in (0.5, 2.0), got {area}");
    }

    #[test]
    fn test_sphere_mesh_euler_characteristic() {
        // Single triangle: V=3, E=3, F=1 → χ = 3-3+1 = 1.
        let s2 = Sphere::<3>;
        let e1: SVector<f64, 3> = SVector::from([1.0, 0.0, 0.0]);
        let e2: SVector<f64, 3> = SVector::from([0.0, 1.0, 0.0]);
        let e3: SVector<f64, 3> = SVector::from([0.0, 0.0, 1.0]);
        let mesh = Mesh::from_simplices(&s2, vec![e1, e2, e3], vec![[0, 1, 2]]);
        assert_eq!(mesh.euler_characteristic(), 1);
    }

    // ── SO(3) mesh topology test ───────────────────────────────────────────

    #[test]
    fn test_so3_mesh_topology() {
        // Construct a topological triangle mesh on SO(3) using 3 identity-adjacent rotations.
        // Only test that from_simplices works and topology is correct (d1*d0=0).
        use cartan_dec::ExteriorDerivative;
        use nalgebra::SMatrix;

        let so3 = SpecialOrthogonal::<3>;

        // Three SO(3) points: I, R_x(0.1), R_y(0.1)
        let id = SMatrix::<f64, 3, 3>::identity();
        let t = 0.1_f64;
        let rx = SMatrix::<f64, 3, 3>::from_row_slice(&[
            1.0, 0.0,    0.0,
            0.0, t.cos(), -t.sin(),
            0.0, t.sin(),  t.cos(),
        ]);
        let ry = SMatrix::<f64, 3, 3>::from_row_slice(&[
             t.cos(), 0.0, t.sin(),
             0.0,     1.0, 0.0,
            -t.sin(), 0.0, t.cos(),
        ]);

        let mesh = Mesh::from_simplices(&so3, vec![id, rx, ry], vec![[0, 1, 2]]);
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_boundaries(), 3);  // 3 edges
        assert_eq!(mesh.n_simplices(), 1);   // 1 triangle

        // Topology: d1 * d0 = 0 (exactness of exterior derivative)
        let ext = ExteriorDerivative::from_mesh_generic(&mesh);
        let exactness = ext.check_exactness();
        assert!(exactness < 1e-14, "d1*d0 != 0 on SO(3) mesh: {exactness}");
    }

    // ── FlatMesh backward compat ───────────────────────────────────────────

    #[test]
    fn test_flat_mesh_backward_compat() {
        use cartan_dec::mesh::FlatMesh;
        // Ensure FlatMesh API is unchanged: unit_square_grid, n_vertices, n_edges.
        // n=4: V=(n+1)^2=25, F=2*n^2=32 triangles.
        // By Euler (disk, χ=1): E = V + F - 1 = 25 + 32 - 1 = 56.
        // Direct count: 20 horizontal + 20 vertical + 16 diagonal = 56.
        let mesh = FlatMesh::unit_square_grid(4);
        assert_eq!(mesh.n_vertices(), 25);
        assert_eq!(mesh.n_boundaries(), 56);  // same as old n_edges()
    }
}
```

> **Note on `ExteriorDerivative::from_mesh_generic`:** The SO(3) test needs a version of `from_mesh` that accepts `Mesh<M, 3, 2>` for any M. This is a topological operation — it is added in Task 8 Step 2 (below). Task 3 handles the FlatMesh-specific `from_mesh`; the generic version is a separate addition done here in Task 8.

- [ ] **Step 2: Add `from_mesh_generic` to `ExteriorDerivative`**

In `exterior.rs`, add a second constructor that works for any M:

```rust
impl ExteriorDerivative {
    /// Generic version: accepts any `Mesh<M, 3, 2>`. Purely topological.
    pub fn from_mesh_generic<M: Manifold>(mesh: &Mesh<M, 3, 2>) -> Self {
        let nv = mesh.n_vertices();
        let ne = mesh.n_boundaries();
        let nt = mesh.n_simplices();

        let mut d0 = DMatrix::<f64>::zeros(ne, nv);
        for (e, &[i, j]) in mesh.boundaries.iter().enumerate() {
            d0[(e, i)] = -1.0;
            d0[(e, j)] = 1.0;
        }

        let mut d1 = DMatrix::<f64>::zeros(nt, ne);
        for (t, (local_e, local_s)) in mesh.simplex_boundary_ids.iter()
            .zip(mesh.boundary_signs.iter()).enumerate()
        {
            for k in 0..3 {
                d1[(t, local_e[k])] = local_s[k];
            }
        }

        Self { d0, d1 }
    }
}
```

Also update `from_mesh` to just call `from_mesh_generic`:
```rust
pub fn from_mesh(mesh: &FlatMesh) -> Self {
    Self::from_mesh_generic(mesh)
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p cartan-dec generic_mesh_tests -- --nocapture
```
Expected: all pass.

- [ ] **Step 4: Run full test suite**

```bash
cargo test -p cartan-dec
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cartan-dec/tests/ cartan-dec/src/exterior.rs
git commit -m "test(dec): generic Mesh<Sphere>, Mesh<SO(3)> topology and metric tests"
```

---

## Chunk 4: Integration and Cleanup

### Task 9: Final integration check

- [ ] **Step 1: Full workspace test**

```bash
cargo test
```
Expected: all tests across all crates pass.

- [ ] **Step 2: Check no bare `Mesh` (non-generic) references remain in `cartan-dec`**

```bash
# run from workspace root
grep -rn '\bMesh\b' cartan-dec/src/ | grep -v 'FlatMesh\|Mesh<\|pub type\|pub struct\|//\|///'
```
Expected: no hits (all `Mesh` references are now `FlatMesh`, `Mesh<M,...>`, or in the struct/type definition itself).

- [ ] **Step 3: Verify deprecated method stubs removed (optional)**

If the deprecated stubs (`n_edges`, `n_triangles`) in `mesh.rs` are no longer referenced anywhere, remove them:

```bash
grep -rn 'n_edges\(\)\|n_triangles\(\)' cartan-dec/src/ cartan-dec/tests/
```
If no hits, delete the deprecated stub methods from the `impl Mesh<Euclidean<2>, 3, 2>` block.

- [ ] **Step 4: Final commit**

```bash
git add -p  # review all changes
cargo test  # confirm clean
git commit -m "refactor(dec): complete generic Mesh<M, K, B> migration"
```

---

## Verification

After all tasks complete:

```bash
# All tests pass
cargo test

# cartan-dec in isolation
cargo test -p cartan-dec

# FlatMesh alias still works
cargo test -p cartan-dec test_flat_mesh_backward_compat

# Generic mesh tests
cargo test -p cartan-dec generic_mesh_tests

# No old Mesh references
grep -rn '\buse crate::mesh::Mesh\b' cartan-dec/src/
# Expected: no output

# Workspace builds clean (no warnings on renamed fields)
cargo build 2>&1 | grep -E "warning|error" | head -20
```
