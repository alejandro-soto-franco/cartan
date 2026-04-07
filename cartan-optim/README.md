# cartan-optim

Riemannian optimization algorithms for cartan.

[![crates.io](https://img.shields.io/crates/v/cartan-optim.svg)](https://crates.io/crates/cartan-optim)
[![docs.rs](https://docs.rs/cartan-optim/badge.svg)](https://docs.rs/cartan-optim)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-optim` implements first- and second-order optimization algorithms
that operate on any manifold implementing traits from `cartan-core`. Each
algorithm requires progressively richer geometry:

| Algorithm | Function | Trait requirements |
|-----------|----------|--------------------|
| Riemannian Gradient Descent | `minimize_rgd` | `Manifold + Retraction` |
| Riemannian Conjugate Gradient | `minimize_rcg` | `+ ParallelTransport` |
| Riemannian Trust Region | `minimize_rtr` | `+ Connection` |
| Frechet Mean (Karcher flow) | `frechet_mean` | `Manifold` |

All solvers return an `OptResult` containing the final point, objective
value, gradient norm, and iteration count.

## Example

```rust,no_run
use nalgebra::SVector;
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use cartan_optim::{minimize_rgd, RGDConfig};

let s2 = Sphere::<3>;
let config = RGDConfig::default();
let p0 = SVector::<f64, 3>::from([0.0, 1.0, 0.0]);

// Minimize f(p) = -p[0] on S^2, driving p toward [1, 0, 0].
let result = minimize_rgd(
    &s2,
    |p| -p[0],
    |p| s2.project_tangent(p, &SVector::from([1.0, 0.0, 0.0])),
    p0,
    &config,
);
```

## License

[MIT](../LICENSE-MIT)
