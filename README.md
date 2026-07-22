# cartan

Riemannian geometry, manifold optimisation, and geodesic computation in Rust.

[![crates.io](https://img.shields.io/crates/v/cartan.svg)](https://crates.io/crates/cartan)
[![PyPI](https://img.shields.io/pypi/v/cartan.svg)](https://pypi.org/project/cartan/)
[![docs.rs](https://docs.rs/cartan/badge.svg)](https://docs.rs/cartan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Tests](https://github.com/alejandro-soto-franco/cartan/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/cartan/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](Cargo.toml)
[![no_std](https://img.shields.io/badge/no__std-bare--metal%20tested-brightgreen.svg)](#embedded-and-no_std)

**cartan** puts one trait system across three regimes: geometry at **points**,
at **fields**, and along **paths**. Manifolds are const-generic and zero-sized,
so dimension errors are caught at compile time and the abstraction costs nothing
at runtime.

Documentation: [cartan.sotofranco.dev](https://cartan.sotofranco.dev) ·
[docs.rs](https://docs.rs/cartan) ·
[capability inventory](CAPABILITIES.md)

## Install

```toml
[dependencies]
cartan = "0.8"
```

```rust
use cartan::prelude::*;
use cartan::manifolds::Sphere;

let s2 = Sphere::<3>;                       // the 2-sphere in R^3
let mut rng = rand::rng();

let p = s2.random_point(&mut rng);
let v = s2.random_tangent(&p, &mut rng);

let q = s2.exp(&p, &v);                     // walk along the geodesic
let v_back = s2.log(&p, &q).unwrap();       // recover the velocity
let d = s2.dist(&p, &q).unwrap();           // geodesic distance

assert!((v - v_back).norm() < 1e-10);
assert!((d - s2.norm(&p, &v)).abs() < 1e-10);
```

Every example in the [guide](https://docs.rs/cartan/latest/cartan/guide/) is a
doctest, so nothing there can drift from the API.

## Three regimes

| regime | crates | what it gives you |
|---|---|---|
| **points** | `manifolds`, `optim`, `geo` | `exp`, `log`, transport, curvature, optimisation, Fréchet means |
| **fields** | `dec`, `remesh`, `homog`, `io` | discrete exterior calculus, line bundles, homogenisation, VTK export |
| **paths** | `stochastic` | orthonormal frame bundle, horizontal lift, Stratonovich development |

Manopt, Manifolds.jl and geomstats cover points. FEniCS and Firedrake cover
fields, on flat domains. The bundle layer, where a field carries an internal
symmetry so comparing neighbouring values needs a connection, is what cartan
adds.

## Features

The default is `std` plus `dec`. Everything else is opt-in, so computing a
geodesic does not compile a sparse solver.

| feature | brings in | requires |
|---|---|---|
| `alloc` | core, manifolds, optim, geo | no_std with an allocator |
| `std` | the above, with std | std |
| `dec` *(default)* | `cartan-dec` | std |
| `remesh` | `cartan-remesh` | `dec` |
| `stochastic` | `cartan-stochastic` | std |
| `homog` | `cartan-homog` mean-field schemes | alloc |
| `full-field` | `cartan-homog` cell-problem solver | `homog`, `remesh`, std |
| `io` | `cartan-io` VTK and Blender export | `dec` |
| `maxwell` | `cartan-maxwell` | `io` |
| `full` | everything above | std |

docs.rs is built with all features, and each gated item carries a badge naming
the flag it needs, so the whole surface stays visible regardless of your build.

## Crates

| crate | what it is |
|---|---|
| [`cartan`](https://docs.rs/cartan) | facade; depend on this one |
| [`cartan-core`](https://docs.rs/cartan-core) | trait system: `Manifold`, `Fiber`, `DiscreteConnection`, `Curvature` |
| [`cartan-manifolds`](https://docs.rs/cartan-manifolds) | Sphere, SO(N), SE(N), SPD, Grassmann, Corr, Q-tensor |
| [`cartan-optim`](https://docs.rs/cartan-optim) | RGD, RCG, trust region, Fréchet mean |
| [`cartan-geo`](https://docs.rs/cartan-geo) | geodesic curves, Jacobi fields, holonomy, Chern-Simons |
| [`cartan-dec`](https://docs.rs/cartan-dec) | discrete exterior calculus, line bundles, Stokes solver |
| [`cartan-remesh`](https://docs.rs/cartan-remesh) | adaptive remeshing, 2D and 3D |
| [`cartan-stochastic`](https://docs.rs/cartan-stochastic) | frame bundle, horizontal lift, Wishart SDE |
| [`cartan-homog`](https://docs.rs/cartan-homog) | mean-field and full-field homogenisation on SPD |
| [`cartan-io`](https://docs.rs/cartan-io) | VTK, ParaView and Blender export |
| [`cartan-maxwell`](https://docs.rs/cartan-maxwell) | Maxwell evolution on an evolving Regge background |
| [`cartan-py`](https://pypi.org/project/cartan/) | Python bindings via PyO3, with numpy interop |

The FEEC layer comes from the upstream [`formoniq`](https://crates.io/crates/formoniq)
crates rather than being vendored.

## Embedded and no_std

```toml
cartan = { version = "0.8", default-features = false, features = ["alloc"] }
```

That gives the point-geometry stack, and with `homog` the mean-field
homogenisation schemes as well. CI builds both configurations for
`thumbv7em-none-eabihf` on every run, so the claim is tested against a target
with no standard library rather than inferred from a host build.

`cartan-core` also builds bare, with no allocator at all. See
[CAPABILITIES.md](CAPABILITIES.md) for the per-crate tier table and a worked
attitude-control example on a microcontroller.

## Documentation

- [Guide](https://docs.rs/cartan/latest/cartan/guide/), doctested chapters across all three regimes
- [CAPABILITIES.md](CAPABILITIES.md), the full per-crate inventory
- [cartan.sotofranco.dev](https://cartan.sotofranco.dev), long-form articles

## Licence

MIT. See [LICENSE-MIT](LICENSE-MIT).
