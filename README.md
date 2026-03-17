# cartan

Riemannian geometry, manifold optimization, and geodesic computation in Rust.

[![crates.io](https://img.shields.io/crates/v/cartan.svg)](https://crates.io/crates/cartan)
[![docs.rs](https://docs.rs/cartan/badge.svg)](https://docs.rs/cartan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Tests](https://github.com/alejandro-soto-franco/cartan/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/cartan/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)

**cartan** is a general-purpose Rust library for Riemannian geometry. It provides a backend-agnostic trait system with const-generic manifolds, correct numerics, and clean composability: from basic exp/log maps through second-order optimization to discrete exterior calculus for covariant PDE solvers.

Documentation: [cartan.sotofranco.dev](https://cartan.sotofranco.dev)

## Features

- **Generic trait hierarchy**: `Manifold`, `Retraction`, `ParallelTransport`, `VectorTransport`, `Connection`, `Curvature`, `GeodesicInterpolation`
- **Const-generic manifolds**: `Sphere<3>`, `Grassmann<5,2>`, dimensions checked at compile time
- **Correct numerics**: Taylor expansions near singularities, cut locus detection, structured error handling
- **Zero-cost abstractions**: manifold types are zero-sized; all geometry lives in the trait impls
- **Optimization**: `cartan-optim` provides RGD, RCG, RTR, and Fréchet mean on any `Manifold`
- **Geodesic tools**: `cartan-geo` provides parameterized geodesics, curvature queries, and Jacobi field integration
- **DEC layer**: `cartan-dec` discretizes covariant differential operators on simplicial meshes for PDE solvers

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

## Manifolds

Every manifold implements all seven traits in the hierarchy. Intrinsic dimensions are checked at compile time via const generics.

| Manifold | Type | Dim | Geometry |
|----------|------|-----|----------|
| Euclidean R^N | `Euclidean<N>` | N | flat, K = 0 |
| Sphere S^(N-1) | `Sphere<N>` | N−1 | K = 1 |
| Special orthogonal SO(N) | `SpecialOrthogonal<N>` | N(N−1)/2 | K ≥ 0 (bi-invariant) |
| Special Euclidean SE(N) | `SpecialEuclidean<N>` | N(N+1)/2 | flat × sphere |
| Symmetric positive definite SPD(N) | `Spd<N>` | N(N+1)/2 | K ≤ 0 (Cartan-Hadamard) |
| Grassmann Gr(N, K) | `Grassmann<N, K>` | K(N−K) | 0 ≤ K ≤ 2 |
| Correlation Corr(N) | `Corr<N>` | N(N−1)/2 | flat, K = 0 |

## Crate Structure

```
cartan              facade crate (re-exports everything)
cartan-core         trait definitions, CartanError, Real alias
cartan-manifolds    concrete manifold implementations (7 manifolds)
cartan-optim        Riemannian optimization: RGD, RCG, RTR, Fréchet mean
cartan-geo          geodesic curves, curvature queries, Jacobi fields
cartan-dec          discrete exterior calculus for PDE solvers
```

All manifolds use `nalgebra` `SVector`/`SMatrix` types directly; no intermediate backend crate is needed.

## cartan-optim

Four algorithms on any `Manifold`:

| Algorithm | Function | Traits required |
|-----------|----------|-----------------|
| Riemannian gradient descent | `minimize_rgd` | `Manifold + Retraction` |
| Riemannian conjugate gradient (FR / PR+) | `minimize_rcg` | `+ ParallelTransport` |
| Riemannian trust region (Steihaug-Toint) | `minimize_rtr` | `+ Connection` |
| Fréchet mean (Karcher flow) | `frechet_mean` | `Manifold` |

```rust
use cartan_optim::{minimize_rgd, RGDConfig};
use cartan_manifolds::Sphere;

let s2 = Sphere::<3>;
let result = minimize_rgd(
    &s2,
    |p| -p[0],                                           // cost
    |p| s2.project_tangent(p, &SVector::from([1.,0.,0.])), // riemannian gradient
    p0,
    &RGDConfig::default(),
);
```

## cartan-geo

```rust
use cartan_geo::{Geodesic, integrate_jacobi};
use cartan_manifolds::Sphere;

let s2 = Sphere::<3>;

// Parameterized geodesic from p to q
let geo = Geodesic::from_two_points(&s2, p, &q).unwrap();
let points = geo.sample(20);           // 20 evenly-spaced points on [0,1]
println!("arc length = {:.4}", geo.length());

// Jacobi field: D²J/dt² + R(J, γ')γ' = 0
let result = integrate_jacobi(&geo, j0, j0_dot, 200);
```

## cartan-dec

`cartan-dec` is the bridge between cartan's continuous geometry and discrete PDE solvers. It builds a 2D simplicial complex, precomputes Hodge operators and covariant derivatives, and exposes them for time-stepping loops.

On a well-centered Delaunay mesh the Hodge star is diagonal, so the full Laplace-Beltrami operator factors into sparse {0, +1, -1} incidence matrix-vector products interleaved with diagonal scalings (cache-friendly and SIMD-vectorizable). Fields use structure-of-arrays layout.

```rust
use cartan_dec::{FlatMesh, Operators};
use cartan_manifolds::euclidean::Euclidean;

let mesh = FlatMesh::unit_square_grid(32);    // 32x32 uniform grid on [0,1]^2
let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);

// Scalar Laplacian, Bochner Laplacian (vector fields),
// Lichnerowicz Laplacian (symmetric 2-tensors / Q-tensor equation)
let lf = ops.apply_laplace_beltrami(&f);
let lu = ops.apply_bochner_laplacian(&u, ricci_correction);
let lq = ops.apply_lichnerowicz_laplacian(&q, curvature_correction);
```

Also provided: `ExteriorDerivative` (d₀, d₁), `HodgeStar` (⋆₀, ⋆₁, ⋆₂), upwind `apply_scalar_advection` / `apply_vector_advection`, and `apply_divergence` / `apply_tensor_divergence`.

## License

[MIT](LICENSE-MIT)
