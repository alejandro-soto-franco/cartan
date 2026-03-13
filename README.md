# cartan

Riemannian geometry, manifold optimization, and geodesic computation in Rust.

[![crates.io](https://img.shields.io/crates/v/cartan.svg)](https://crates.io/crates/cartan)
[![docs.rs](https://docs.rs/cartan/badge.svg)](https://docs.rs/cartan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Tests](https://github.com/alejandro-soto-franco/cartan/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/cartan/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

**cartan** is a general-purpose Rust library for Riemannian geometry. It provides a backend-agnostic trait system with const-generic manifolds, correct numerics, and clean composability. It is also the substrate for covariant PDE solvers via `cartan-dec`, its discrete exterior calculus layer.

Documentation: [cartan.sotofranco.dev](https://cartan.sotofranco.dev)

## Features

- **Generic trait hierarchy**: `Manifold`, `Retraction`, `ParallelTransport`, `VectorTransport`, `Connection`, `Curvature`, `GeodesicInterpolation`
- **Const-generic manifolds**: `Sphere<3>`, `Euclidean<10>`, dimensions checked at compile time
- **Correct numerics**: Taylor expansions near singularities, cut locus detection, structured error handling
- **Zero-cost abstractions**: manifold types are zero-sized, all geometry lives in the trait impls
- **DEC layer**: `cartan-dec` discretizes covariant differential operators on any manifold for use in PDE solvers

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

| Manifold | Type | Status |
|----------|------|--------|
| Euclidean R^N | `Euclidean<N>` | done |
| Sphere S^(N-1) | `Sphere<N>` | done |
| Special orthogonal SO(N) | `SpecialOrthogonal<N>` | done |
| Special Euclidean SE(N) | `SpecialEuclidean<N>` | done |
| Symmetric positive definite SPD(N) | `SymmetricPositiveDefinite<N>` | planned |
| Grassmann Gr(N, K) | `Grassmann<N, K>` | planned |
| Stiefel St(N, K) | `Stiefel<N, K>` | planned |
| Hyperbolic H^N | `Hyperbolic<N>` | planned |
| Simplex | `Simplex<N>` | planned |
| Correlation Corr(N) | `Corr<N>` | planned |
| Product manifolds | `Product<M1, M2>` | planned |

## Crate Structure

```
cartan              facade crate (use this)
cartan-core         trait definitions, error types, Real alias
cartan-manifolds    concrete manifold implementations
cartan-nalgebra     nalgebra backend (SVector, SMatrix storage)
cartan-dec          discrete exterior calculus for PDE solvers
cartan-optim        Riemannian optimization algorithms (planned)
cartan-geo          geodesic curves and curvature tools (planned)
```

## cartan-dec

`cartan-dec` is the bridge between cartan's continuous geometry and discrete PDE solvers. It builds a simplicial complex over any domain, precomputes Hodge operators and covariant derivatives, and exposes them for use in time-stepping loops.

The key design property is that on a well-centered Delaunay mesh, the Hodge star is diagonal. This means the full Laplace-Beltrami operator factors into two sparse {0, +1, -1} matrix-vector products interleaved with diagonal scalings, which is both cache-friendly and SIMD-vectorizable. Simplices are reordered by Hilbert space-filling curve for spatial locality. Fields use structure-of-arrays layout.

Operators provided: `ExteriorDerivative`, `HodgeStar`, `LaplaceBeltrami`, `BochnerLaplacian`, `LichnerowiczLaplacian`, `CovariantAdvection`, `CovariantDivergence`.

## License

[MIT](LICENSE-MIT)
