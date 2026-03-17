# Workstream A: `no_std` Feature Flags — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread three-tier `no_std` support (std / alloc / no_alloc) through all six cartan crates so that cartan compiles on embedded targets without a standard library.

**Architecture:** Feature flags `std` (default), `alloc`, and no-features (no_alloc) are added to every crate. `use std::` is replaced with `use core::` throughout. `CartanError` gets two `cfg`-gated variants — `String` fields under `alloc`, `&'static str` fields under no_alloc. `HashMap` in `cartan-dec` becomes `hashbrown` under `alloc` and a sorted-array approach under no_alloc. `rayon` in `cartan-dec` is gated behind `std`. Alloc-dependent types in `cartan-geo` (`Vec`-returning functions, `JacobiResult`) are gated behind `#[cfg(feature = "alloc")]`.

**Tech Stack:** Rust 1.85, `cartan` workspace, `hashbrown 0.15`, `nalgebra` (already no_std-capable), `rand` / `rand_distr` (no_std-capable with `default-features = false`).

**Spec:** `docs/superpowers/specs/2026-03-17-cartan-generalization-design.md` — Workstream A.

---

## Chunk 1: Workspace and `cartan-core`

### Task 1: Update workspace `Cargo.toml`

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add `hashbrown` to workspace deps and fix `rand`/`rand_distr` defaults**

Open `Cargo.toml`. Replace the `[workspace.dependencies]` section:

```toml
[workspace.dependencies]
nalgebra = { version = "0.33", default-features = false }
rand = { version = "0.9", default-features = false, features = ["getrandom"] }
rand_distr = { version = "0.5", default-features = false }
rayon = { version = "1" }  # rayon has no meaningful no_std subset; optional in cartan-dec
thiserror = { version = "2", default-features = false }
serde = { version = "1", features = ["derive"], default-features = false }
serde_json = { version = "1", default-features = false }
approx = { version = "0.5", default-features = false }
hashbrown = { version = "0.15", default-features = false }
```

Note: `features = ["getrandom"]` on `rand` is required for embedded targets without OS entropy sources. `rayon`'s `default-features` setting is cosmetic — rayon has no feature flags that affect std usage; it is controlled by making it `optional = true` in `cartan-dec`.

- [ ] **Step 2: Verify workspace still builds**

```bash
cargo build 2>&1 | grep -E "^error"
```
Expected: no errors (warnings about unused deps are OK at this stage).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml && git commit -m "chore: set default-features=false on workspace deps; add hashbrown"
```

---

### Task 2: `cartan-core` — feature flags + `use core::` + `CartanError` dual enum

**Files:**
- Modify: `cartan-core/Cargo.toml`
- Modify: `cartan-core/src/lib.rs`
- Modify: `cartan-core/src/error.rs`
- Modify: `cartan-core/src/manifold.rs`
- Modify: `cartan-core/src/connection.rs`
- Modify: `cartan-core/src/curvature.rs`
- Modify: `cartan-core/src/geodesic.rs`
- Modify: `cartan-core/src/retraction.rs`
- Modify: `cartan-core/src/transport.rs`

- [ ] **Step 1: Add feature flags to `cartan-core/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc", "rand/std"]
alloc = ["rand/alloc"]

[dependencies]
# getrandom feature required for entropy on bare-metal embedded targets
# default-features = false required at crate level in case this task runs before Task 1
rand = { workspace = true, default-features = false, features = ["getrandom"] }

[package.metadata.docs.rs]
rustdoc-args = ["--cfg", "docsrs"]
```

- [ ] **Step 2: Add `no_std` header to `cartan-core/src/lib.rs`**

Add at the very top of the file (before all existing content):

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
```

- [ ] **Step 3: Rewrite `cartan-core/src/error.rs`**

Replace the entire file with:

> **Implementation note:** The spec proposes two entirely separate enum definitions gated by `#[cfg]` on the whole item. This plan uses a single enum with `cfg` attributes on individual fields instead. Both approaches produce the same ABI in Rust 1.85 (edition 2024) and the `Display` impl works identically since both `String` and `&'static str` implement `Display`. The field-level `cfg` approach is used here because it avoids duplicating the `Display` and `Debug` impls. If the whole-item form is preferred, replace the single enum with two `#[cfg]`-gated definitions covering the same variants.

```rust
// ~/cartan/cartan-core/src/error.rs

//! Error types for cartan operations.

use core::fmt;

use crate::Real;

/// The unified error type for all cartan operations.
///
/// Under `no_alloc`, message fields are `&'static str` (no heap).
/// Under `alloc` or `std`, message fields are `String` (rich formatting).
#[derive(Debug, Clone)]
pub enum CartanError {
    /// Log map failed: point is on or near the cut locus.
    CutLocus {
        #[cfg(feature = "alloc")]
        message: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        message: &'static str,
    },

    /// A matrix decomposition or numerical computation failed.
    NumericalFailure {
        #[cfg(feature = "alloc")]
        operation: alloc::string::String,
        #[cfg(feature = "alloc")]
        message: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        operation: &'static str,
        #[cfg(not(feature = "alloc"))]
        message: &'static str,
    },

    /// Point does not satisfy the manifold constraint.
    NotOnManifold {
        #[cfg(feature = "alloc")]
        constraint: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        constraint: &'static str,
        violation: Real,
    },

    /// Tangent vector is not in the tangent space at the given point.
    NotInTangentSpace {
        #[cfg(feature = "alloc")]
        constraint: alloc::string::String,
        #[cfg(not(feature = "alloc"))]
        constraint: &'static str,
        violation: Real,
    },

    /// Line search failed to find a step size satisfying the Armijo condition.
    LineSearchFailed {
        steps_tried: usize,
    },

    /// Optimizer did not converge within the maximum number of iterations.
    ConvergenceFailure {
        iterations: usize,
        gradient_norm: Real,
    },
}

impl fmt::Display for CartanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CartanError::CutLocus { message } => {
                write!(f, "cut locus: {}", message)
            }
            CartanError::NumericalFailure { operation, message } => {
                write!(f, "numerical failure in {}: {}", operation, message)
            }
            CartanError::NotOnManifold { constraint, violation } => {
                write!(f, "point not on manifold: {} violated by {}", constraint, violation)
            }
            CartanError::NotInTangentSpace { constraint, violation } => {
                write!(f, "tangent vector not in tangent space: {} violated by {}", constraint, violation)
            }
            CartanError::LineSearchFailed { steps_tried } => {
                write!(f, "line search failed after {} steps", steps_tried)
            }
            CartanError::ConvergenceFailure { iterations, gradient_norm } => {
                write!(
                    f,
                    "optimizer did not converge after {} iterations (gradient norm: {:.2e})",
                    iterations, gradient_norm
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CartanError {}
```

- [ ] **Step 4: Fix `use std::` in `cartan-core/src/manifold.rs`**

Replace:
```rust
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};
```
With:
```rust
use core::fmt::Debug;
use core::ops::{Add, Mul, Neg, Sub};
```

- [ ] **Step 5: Check remaining `cartan-core` files for `std::` imports**

```bash
grep -rn "use std::" cartan-core/src/
```

Expected: no matches (connection, curvature, geodesic, retraction, transport use only `crate::` imports).

- [ ] **Step 6: Build `cartan-core` without std to verify**

```bash
cargo build -p cartan-core --no-default-features 2>&1 | grep -E "^error"
```
Expected: no errors.

```bash
cargo build -p cartan-core --no-default-features --features alloc 2>&1 | grep -E "^error"
```
Expected: no errors.

```bash
cargo build -p cartan-core 2>&1 | grep -E "^error"
```
Expected: no errors.

- [ ] **Step 7: Run `cartan-core` tests**

```bash
cargo test -p cartan-core
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add cartan-core/ && git commit -m "feat(core): no_std support — use core::, cfg-gated CartanError, feature flags"
```

---

### Task 3: `cartan-manifolds` — feature flags + `use core::`

**Files:**
- Modify: `cartan-manifolds/Cargo.toml`
- Modify: `cartan-manifolds/src/lib.rs`
- Modify: `cartan-manifolds/src/sphere.rs`
- Modify: `cartan-manifolds/src/so.rs`
- Modify: `cartan-manifolds/src/spd.rs`
- Modify: `cartan-manifolds/src/se.rs`
- Modify: `cartan-manifolds/src/grassmann.rs`
- Modify: `cartan-manifolds/src/euclidean.rs`
- Modify: `cartan-manifolds/src/corr.rs`
- Modify: `cartan-manifolds/src/util/matrix_exp.rs`
- Modify: `cartan-manifolds/src/util/matrix_log.rs`

- [ ] **Step 1: Add feature flags to `cartan-manifolds/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc", "cartan-core/std", "nalgebra/std", "rand/std", "rand_distr/std"]
alloc = ["cartan-core/alloc", "nalgebra/alloc", "rand/alloc", "rand_distr/alloc"]

[dependencies]
cartan-core = { path = "../cartan-core", version = "0.1", default-features = false }
nalgebra = { workspace = true }
rand = { workspace = true, default-features = false, features = ["getrandom"] }
rand_distr = { workspace = true, default-features = false }

[dev-dependencies]
approx = { workspace = true }
```

- [ ] **Step 2: Add `no_std` header to `cartan-manifolds/src/lib.rs`**

Add at the very top:

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
```

- [ ] **Step 3: Audit all `use std::` in `cartan-manifolds`**

```bash
grep -rn "use std::" cartan-manifolds/src/
```

Replace every occurrence:
- `use std::f64::consts::` → `use core::f64::consts::`
- `use std::fmt` → `use core::fmt`
- `use std::ops::` → `use core::ops::`

Known occurrences (verify with the grep above): `se.rs` line 79 (`std::f64::consts::PI`) and line 80 (`std::ops::{Add, Mul, Neg, Sub}`), `sphere.rs` (`std::f64::consts::PI`). All `use std::ops` imports must become `use core::ops`.

- [ ] **Step 4: Fix `format!` calls in error constructors**

`format!` requires `alloc`. Every `CartanError` variant with a formatted message in `cartan-manifolds` must be gated. The pattern to find:

```bash
grep -rn "format!" cartan-manifolds/src/
```

For each occurrence constructing a `CartanError`, wrap it:

```rust
// Under alloc: rich format
#[cfg(feature = "alloc")]
return Err(CartanError::CutLocus {
    message: alloc::format!("points are nearly antipodal on S^{}: angle = {:.2e}", N-1, theta),
});
// Under no_alloc: static fallback
#[cfg(not(feature = "alloc"))]
return Err(CartanError::CutLocus {
    message: "points are nearly antipodal",
});
```

- [ ] **Step 5: Build `cartan-manifolds` in all three tiers**

```bash
cargo build -p cartan-manifolds --no-default-features 2>&1 | grep -E "^error"
cargo build -p cartan-manifolds --no-default-features --features alloc 2>&1 | grep -E "^error"
cargo build -p cartan-manifolds 2>&1 | grep -E "^error"
```
Expected: no errors in any tier.

- [ ] **Step 6: Run `cartan-manifolds` tests**

```bash
cargo test -p cartan-manifolds
```
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add cartan-manifolds/ && git commit -m "feat(manifolds): no_std support — use core::, cfg-gated format! calls, feature flags"
```

---

### Task 4: `cartan-optim` — feature flags (clean pass)

**Files:**
- Modify: `cartan-optim/Cargo.toml`
- Modify: `cartan-optim/src/lib.rs`

- [ ] **Step 1: Add feature flags to `cartan-optim/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc", "cartan-core/std", "cartan-manifolds/std"]
alloc = ["cartan-core/alloc", "cartan-manifolds/alloc"]

[dependencies]
cartan-core = { path = "../cartan-core", version = "0.1", default-features = false }
cartan-manifolds = { path = "../cartan-manifolds", version = "0.1", default-features = false }

[dev-dependencies]
nalgebra = { workspace = true }
rand = { workspace = true }
approx = { workspace = true }
```

- [ ] **Step 2: Add `no_std` header to `cartan-optim/src/lib.rs`**

Add at the very top:

```rust
#![cfg_attr(not(feature = "std"), no_std)]
```

- [ ] **Step 3: Audit for any `std::` usage**

```bash
grep -rn "use std::" cartan-optim/src/
```
Expected: no matches. If any found, replace with `core::`.

- [ ] **Step 4: Build and test all tiers**

```bash
cargo build -p cartan-optim --no-default-features 2>&1 | grep -E "^error"
cargo build -p cartan-optim --no-default-features --features alloc 2>&1 | grep -E "^error"
cargo test -p cartan-optim
```
Expected: clean build in all tiers, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add cartan-optim/ && git commit -m "feat(optim): no_std support — feature flags, no_std header"
```

---

## Chunk 2: `cartan-geo` and `cartan-dec`

### Task 5: `cartan-geo` — gate `Vec`-returning APIs behind `alloc`

**Files:**
- Modify: `cartan-geo/Cargo.toml`
- Modify: `cartan-geo/src/lib.rs`
- Modify: `cartan-geo/src/geodesic.rs`
- Modify: `cartan-geo/src/jacobi.rs`

- [ ] **Step 1: Add feature flags to `cartan-geo/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc", "cartan-core/std", "cartan-manifolds/std"]
alloc = ["cartan-core/alloc", "cartan-manifolds/alloc"]

[dependencies]
cartan-core = { path = "../cartan-core", version = "0.1", default-features = false }
cartan-manifolds = { path = "../cartan-manifolds", version = "0.1", default-features = false }

[dev-dependencies]
nalgebra = { workspace = true }
rand = { workspace = true }
approx = { workspace = true }
```

- [ ] **Step 2: Add `no_std` header to `cartan-geo/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
```

- [ ] **Step 3: Gate `Geodesic::sample` behind `alloc` in `geodesic.rs`**

The `sample` method returns `Vec<M::Point>`. Wrap its entire `impl` block method:

```rust
#[cfg(feature = "alloc")]
pub fn sample(&self, n: usize) -> alloc::vec::Vec<M::Point> {
    assert!(n > 0, "Geodesic::sample: n must be at least 1");
    if n == 1 {
        return alloc::vec![self.eval(0.0)];
    }
    let step = 1.0 / (n - 1) as Real;
    (0..n).map(|i| self.eval(i as Real * step)).collect()
}
```

Add a no_alloc fixed-size alternative:

```rust
/// Sample up to N evenly-spaced points (no_alloc version).
/// Returns the array and the number of valid entries (min(n, N)).
#[cfg(not(feature = "alloc"))]
pub fn sample_fixed<const N: usize>(&self, n: usize) -> ([M::Point; N], usize)
where
    M::Point: Copy + Default,
{
    let count = n.min(N);
    let mut out = [M::Point::default(); N];
    if count == 0 {
        return (out, 0);
    }
    if count == 1 {
        out[0] = self.eval(0.0);
        return (out, 1);
    }
    let step = 1.0 / (count - 1) as Real;
    for i in 0..count {
        out[i] = self.eval(i as Real * step);
    }
    (out, count)
}
```

- [ ] **Step 4: Gate `JacobiResult` and `integrate_jacobi` behind `alloc` in `jacobi.rs`**

Wrap the entire `JacobiResult` struct and `integrate_jacobi` function:

```rust
#[cfg(feature = "alloc")]
pub struct JacobiResult<T> {
    pub params: alloc::vec::Vec<Real>,
    pub field: alloc::vec::Vec<T>,
    pub velocity: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
pub fn integrate_jacobi<M>(...) -> JacobiResult<M::Tangent>
where M: Manifold + Curvature + ParallelTransport
{ ... }
```

Add fixed-size alternative:

```rust
/// Fixed-size Jacobi result for no_alloc tier.
#[cfg(not(feature = "alloc"))]
pub struct JacobiResultFixed<T: Copy, const N: usize> {
    pub params: [Real; N],
    pub field: [T; N],
    pub velocity: [T; N],
    pub len: usize,
}
```

Update `cartan-geo/src/lib.rs` re-exports to be `cfg`-gated:

```rust
#[cfg(feature = "alloc")]
pub use jacobi::{integrate_jacobi, JacobiResult};
#[cfg(not(feature = "alloc"))]
pub use jacobi::JacobiResultFixed;
pub use geodesic::Geodesic;
pub use curvature::{scalar_at, sectional_at, CurvatureQuery};
```

- [ ] **Step 5: Build and test all tiers**

```bash
cargo build -p cartan-geo --no-default-features 2>&1 | grep -E "^error"
cargo build -p cartan-geo --no-default-features --features alloc 2>&1 | grep -E "^error"
cargo build -p cartan-geo 2>&1 | grep -E "^error"
cargo test -p cartan-geo
```
Expected: clean in all tiers, all tests pass.

- [ ] **Step 6: Commit**

```bash
git add cartan-geo/ && git commit -m "feat(geo): no_std support — gate Vec APIs behind alloc, JacobiResultFixed"
```

---

### Task 6: `cartan-dec` — feature flags, `hashbrown`, `rayon` gating

**Files:**
- Modify: `cartan-dec/Cargo.toml`
- Modify: `cartan-dec/src/lib.rs`
- Modify: `cartan-dec/src/mesh.rs`
- Modify: `cartan-dec/src/exterior.rs`
- Modify: `cartan-dec/src/hodge.rs`
- Modify: `cartan-dec/src/laplace.rs`
- Modify: `cartan-dec/src/advection.rs`
- Modify: `cartan-dec/src/divergence.rs`

- [ ] **Step 1: Update `cartan-dec/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc", "rayon", "cartan-core/std", "cartan-manifolds/std",
       "nalgebra/std", "serde/std"]
alloc = ["cartan-core/alloc", "cartan-manifolds/alloc", "nalgebra/alloc",
         "hashbrown"]

[dependencies]
cartan-core      = { path = "../cartan-core", version = "0.1", default-features = false }
cartan-manifolds = { path = "../cartan-manifolds", version = "0.1", default-features = false }
nalgebra         = { workspace = true }
rayon            = { workspace = true, optional = true }
# thiserror removed — DecError uses a manual Display impl (see Task 6, Step 5)
serde            = { workspace = true, default-features = false }
hashbrown        = { workspace = true, optional = true }
```

Note: `thiserror` requires `std` for its proc-macro. For the no_alloc tier, `DecError` must be derived manually or `thiserror` replaced with a manual `Display` impl gated by `#[cfg(feature = "std")]`.

- [ ] **Step 2: Add `no_std` header to `cartan-dec/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;
```

Gate module declarations — `cartan-dec` as a whole only functions under `alloc` (nalgebra's `DMatrix` requires heap). Add a compile-time message:

```rust
#[cfg(not(feature = "alloc"))]
compile_error!(
    "cartan-dec requires at least the `alloc` feature. \
     Add `cartan-dec = { features = [\"alloc\"] }` to your Cargo.toml."
);
```

This makes the constraint explicit and gives a clear error rather than a cascade of missing-type errors.

- [ ] **Step 3: Replace `HashMap` in `cartan-dec/src/mesh.rs`**

Replace:
```rust
use std::collections::HashMap;
```
With:
```rust
#[cfg(feature = "alloc")]
use hashbrown::HashMap;
```

Also replace `use nalgebra::Vector2;` path if it uses std-only features — check with:
```bash
grep -n "use std::" cartan-dec/src/mesh.rs
```

Replace `Vec<...>` with `alloc::vec::Vec<...>` where needed OR add at top of file:
```rust
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
```

- [ ] **Step 4: Gate `rayon`-using code in `cartan-dec`**

Search for rayon usage:
```bash
grep -rn "rayon\|par_iter\|into_par_iter" cartan-dec/src/
```

Wrap any parallel iterators:
```rust
#[cfg(feature = "std")]
use rayon::prelude::*;
```

Replace parallel iterators with sequential fallbacks under `#[cfg(not(feature = "std"))]`.

- [ ] **Step 5: Fix `thiserror` usage in `cartan-dec/src/error.rs`**

`thiserror` requires `std`. Replace `DecError` with a manual implementation:

```rust
// cartan-dec/src/error.rs

use core::fmt;

/// Errors that can occur in discrete exterior calculus operations.
#[derive(Debug, Clone)]
pub enum DecError {
    EmptyMesh,
    IndexOutOfBounds { index: usize, len: usize },
    NotWellCentered { simplex: usize, volume: f64 },
    SingularLaplacian,
    FieldLengthMismatch { expected: usize, got: usize },
    #[cfg(feature = "alloc")]
    LinearAlgebra(alloc::string::String),
    #[cfg(not(feature = "alloc"))]
    LinearAlgebra(&'static str),
}

impl fmt::Display for DecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecError::EmptyMesh => write!(f, "empty mesh"),
            DecError::IndexOutOfBounds { index, len } => {
                write!(f, "index out of bounds: index {} in collection of size {}", index, len)
            }
            DecError::NotWellCentered { simplex, volume } => {
                write!(f, "mesh is not well-centered: simplex {} has negative dual volume {:.6e}", simplex, volume)
            }
            DecError::SingularLaplacian => {
                write!(f, "Laplacian is singular; add a Dirichlet pin or use pseudoinverse")
            }
            DecError::FieldLengthMismatch { expected, got } => {
                write!(f, "field length mismatch: expected {}, got {}", expected, got)
            }
            DecError::LinearAlgebra(msg) => write!(f, "linear algebra error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecError {}
```

Remove `thiserror` from `cartan-dec/Cargo.toml` entirely (no longer needed).

- [ ] **Step 6: Build `cartan-dec` under `alloc` and `std` tiers**

```bash
cargo build -p cartan-dec --no-default-features --features alloc 2>&1 | grep -E "^error"
cargo build -p cartan-dec 2>&1 | grep -E "^error"
```
Expected: no errors.

- [ ] **Step 7: Run `cartan-dec` tests**

```bash
cargo test -p cartan-dec
```
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add cartan-dec/ && git commit -m "feat(dec): no_std/alloc support — hashbrown, rayon gate, thiserror removal, DecError manual impl"
```

---

### Task 7: `cartan` facade — propagate feature flags

**Files:**
- Modify: `cartan/Cargo.toml`
- Modify: `cartan/src/lib.rs`

- [ ] **Step 1: Update `cartan/Cargo.toml`**

```toml
[features]
default = ["std"]
std = ["alloc",
       "cartan-core/std", "cartan-manifolds/std", "cartan-optim/std",
       "cartan-geo/std", "cartan-dec/std"]
alloc = ["cartan-core/alloc", "cartan-manifolds/alloc", "cartan-optim/alloc",
         "cartan-geo/alloc", "cartan-dec/alloc"]

[dependencies]
cartan-core      = { path = "../cartan-core",      version = "0.1", default-features = false }
cartan-manifolds = { path = "../cartan-manifolds", version = "0.1", default-features = false }
cartan-optim     = { path = "../cartan-optim",     version = "0.1", default-features = false }
cartan-geo       = { path = "../cartan-geo",       version = "0.1", default-features = false }
cartan-dec       = { path = "../cartan-dec",       version = "0.1", default-features = false }
```

- [ ] **Step 2: Add `no_std` header to `cartan/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
```

Gate `alloc`-only re-exports in the prelude if any are added.

- [ ] **Step 3: Full workspace build — all three tiers**

```bash
# alloc tier: build each crate that supports it
cargo build --no-default-features --features alloc -p cartan-core -p cartan-manifolds -p cartan-optim -p cartan-geo 2>&1 | grep -E "^error"
# std tier (default): build everything
cargo build 2>&1 | grep -E "^error"
# full test suite
cargo test 2>&1 | grep -E "FAILED|^error"
```
Expected: no errors, no failures.

Note: the no_alloc tier (`--no-default-features` with no `--features`) is tested per-crate in the individual crate tasks. Workspace-level no_alloc build is done in Task 8 (CI) using `thumbv7m-none-eabi`.

- [ ] **Step 4: Commit**

```bash
git add cartan/ && git commit -m "feat(cartan): propagate no_std/alloc feature flags through facade crate"
```

---

### Task 8: Add `no_std` build checks to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: View current CI config**

```bash
cat .github/workflows/ci.yml
```

- [ ] **Step 2: Add no_std build matrix jobs**

Add after the existing test step:

```yaml
- name: Install bare-metal target
  run: rustup target add thumbv7m-none-eabi

- name: Build (alloc only, no std — host)
  run: cargo build --no-default-features --features alloc -p cartan-core -p cartan-manifolds -p cartan-optim -p cartan-geo

- name: Build (no alloc, no std — bare-metal Cortex-M3 target)
  # thumbv7m-none-eabi has no OS, no allocator by default — this is a true no_std check
  run: cargo build --no-default-features --target thumbv7m-none-eabi -p cartan-core -p cartan-manifolds -p cartan-optim

- name: Build (full std, default)
  run: cargo build

- name: Doc tests
  run: cargo test --doc
```

Note: `cartan-geo` is excluded from the bare-metal build because its alloc-gated items still require `alloc` to be useful. `cartan-dec` is excluded because it requires `alloc` minimum.

- [ ] **Step 3: Commit**

```bash
git add .github/ && git commit -m "ci: add no_std and alloc-only build checks to CI matrix"
```

---

## Verification

After all tasks complete, run the full verification suite:

```bash
# All tiers compile
cargo build --no-default-features -p cartan-core -p cartan-manifolds -p cartan-optim
cargo build --no-default-features --features alloc
cargo build

# All tests pass
cargo test

# No std:: leaks
grep -rn "use std::" cartan-core/src/ cartan-manifolds/src/ cartan-optim/src/ cartan-geo/src/
# Expected: no output

# Doctests pass
cargo test --doc
```
