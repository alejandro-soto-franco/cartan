# cartan-core

Core trait definitions for Riemannian geometry.

[![crates.io](https://img.shields.io/crates/v/cartan-core.svg)](https://crates.io/crates/cartan-core)
[![docs.rs](https://docs.rs/cartan-core/badge.svg)](https://docs.rs/cartan-core)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-core` defines the foundational trait hierarchy that all cartan
manifolds, optimizers, and geometric tools depend on. It has minimal
dependencies (only `rand` for RNG trait bounds) and can be used standalone
by downstream crates that implement custom manifolds against the cartan
trait system.

The trait hierarchy is:

```text
Manifold (exp, log, inner, project, validate)
  |
  +-- Retraction (cheaper exp approximation)
  +-- ParallelTransport -> VectorTransport (blanket impl)
  +-- Connection (Riemannian Hessian)
  |     |
  |     +-- Curvature (Riemann tensor, Ricci, scalar)
  +-- GeodesicInterpolation (gamma(t) sampling)
```

All floating-point computation uses the `Real` type alias (currently `f64`),
so that a future generic refactor is mechanical. The crate also provides
`CartanError` for structured error handling across the workspace.

## Example

```rust,no_run
use cartan_core::{Manifold, Real};

/// Check that a point lies on the manifold and compute a tangent norm.
fn tangent_norm<M: Manifold>(m: &M, p: &M::Point, v: &M::Tangent) -> Real {
    assert!(m.check(p).is_ok());
    m.inner(p, v, v).sqrt()
}
```

## no_std

Disable default features and enable `alloc` for embedded targets:

```toml
cartan-core = { version = "0.1", default-features = false, features = ["alloc"] }
```

## License

[MIT](../LICENSE-MIT)
