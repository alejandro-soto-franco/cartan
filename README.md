# cartan

Riemannian geometry, manifold optimization, and geodesic computation in Rust.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

**cartan** is a general-purpose Rust library for Riemannian geometry. It provides a backend-agnostic trait system with const-generic manifolds, correct numerics, and clean composability. Think of it as "the nalgebra of Riemannian geometry."

Documentation: [cartan.sotofranco.dev](https://cartan.sotofranco.dev)

## Features

- **Generic trait hierarchy**: `Manifold`, `Retraction`, `ParallelTransport`, `VectorTransport`, `Connection`, `Curvature`, `GeodesicInterpolation`
- **Const-generic manifolds**: `Sphere<3>`, `Euclidean<10>` -- dimension checked at compile time
- **Correct numerics**: Taylor expansions near singularities, cut locus detection, structured error handling
- **Zero-cost abstractions**: manifold types are zero-sized, all geometry is in the trait impls

## Quick Start

```rust
use cartan::prelude::*;
use cartan::manifolds::Sphere;

let s2 = Sphere::<3>; // the 2-sphere in R^3

let mut rng = rand::rng();
let p = s2.random_point(&mut rng);
let v = s2.random_tangent(&p, &mut rng);

// Exponential map: walk along the geodesic
let q = s2.exp(&p, &v);

// Logarithmic map: recover the tangent vector
let v_recovered = s2.log(&p, &q).unwrap();

// Geodesic distance
let d = s2.dist(&p, &q).unwrap();

// Parallel transport a vector from p to q
let u = s2.random_tangent(&p, &mut rng);
let u_at_q = s2.transport(&p, &q, &u).unwrap();

// Sectional curvature (K = 1 for the unit sphere)
let k = s2.sectional_curvature(&p, &u, &v);
```

## Manifolds (v0.1)

| Manifold | Type | Traits |
|----------|------|--------|
| Euclidean R^N | `Euclidean<N>` | All (trivial: flat geometry) |
| Sphere S^{N-1} | `Sphere<N>` | All (constant curvature K=1) |

Planned: SO(N), SE(N), SPD(N), Grassmann, Stiefel, Hyperbolic, Simplex, Corr(N), Product manifolds.

## Crate Structure

```
cartan/              # Facade crate (use this)
cartan-core/         # Trait definitions, error types, Real alias
cartan-manifolds/    # Manifold implementations
cartan-nalgebra/     # nalgebra backend (default)
cartan-optim/        # Riemannian optimizers (planned)
cartan-geo/          # Geodesic tools (planned)
```

## License

[MIT License](LICENSE-MIT)
