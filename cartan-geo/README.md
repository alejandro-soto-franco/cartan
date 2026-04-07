# cartan-geo

Geodesic computation and geometric tools for cartan.

[![crates.io](https://img.shields.io/crates/v/cartan-geo.svg)](https://crates.io/crates/cartan-geo)
[![docs.rs](https://docs.rs/cartan-geo/badge.svg)](https://docs.rs/cartan-geo)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-geo` provides higher-level geometric utilities built on the
`Manifold` trait from `cartan-core` and the concrete manifolds from
`cartan-manifolds`. It focuses on global geometry: parameterized geodesic
curves, curvature queries, and Jacobi field integration.

| Module | Contents |
|--------|----------|
| `geodesic` | `Geodesic<M>`, parameterized geodesic sampling, two-point construction |
| `curvature` | `CurvatureQuery<M>`, sectional, Ricci, and scalar curvature at a point |
| `jacobi` | `integrate_jacobi`, RK4 Jacobi field ODE integration |
| `holonomy` | Loop holonomy, disclination scanning (requires `std`) |
| `disclination` | 3D disclination line tracking and event detection (requires `std`) |

## Example

```rust,no_run
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use cartan_geo::Geodesic;
use nalgebra::SVector;

let s2 = Sphere::<3>;
let p = SVector::from([1.0, 0.0, 0.0]);
let v = s2.project_tangent(&p, &SVector::from([0.0, 1.0, 0.0]));

let geo = Geodesic::new(&s2, p, v);
let midpoint = geo.at(&s2, 0.5);
assert!(s2.check(&midpoint).is_ok());
```

## no_std

Disable default features and enable `alloc` for embedded targets.
`Geodesic`, `CurvatureQuery`, and `integrate_jacobi` are available without
`std`. Holonomy and disclination modules require `std`.

```toml
cartan-geo = { version = "0.1", default-features = false, features = ["alloc"] }
```

## License

[MIT](../LICENSE-MIT)
