# cartan

Riemannian geometry, manifold optimization, and geodesic computation in Rust.

[![crates.io](https://img.shields.io/crates/v/cartan.svg)](https://crates.io/crates/cartan)
[![PyPI](https://img.shields.io/pypi/v/cartan.svg)](https://pypi.org/project/cartan/)
[![docs.rs](https://docs.rs/cartan/badge.svg)](https://docs.rs/cartan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)
[![Tests](https://github.com/alejandro-soto-franco/cartan/actions/workflows/ci.yml/badge.svg)](https://github.com/alejandro-soto-franco/cartan/actions)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-blue.svg)](Cargo.toml)
[![no_std](https://img.shields.io/badge/no__std-compatible-brightgreen.svg)](#embedded-and-nostd-targets)

**cartan** is a general-purpose Rust library for Riemannian geometry. It provides a backend-agnostic trait system with const-generic manifolds, correct numerics, and clean composability: from basic exp/log maps through second-order optimization to discrete exterior calculus for covariant PDE solvers.

Documentation: [cartan.sotofranco.dev](https://cartan.sotofranco.dev)

## Features

- **Generic trait hierarchy**: `Manifold`, `Retraction`, `ParallelTransport`, `VectorTransport`, `Connection`, `Curvature`, `GeodesicInterpolation`, `Fiber`, `DiscreteConnection`, `CovLaplacian`
- **Const-generic manifolds**: `Sphere<3>`, `Grassmann<5,2>`, dimensions checked at compile time
- **Correct numerics**: Taylor expansions near singularities, cut locus detection, structured error handling
- **Zero-cost abstractions**: manifold types are zero-sized; all geometry lives in the trait impls
- **Optimization**: `cartan-optim` provides RGD, RCG, RTR, and Fréchet mean on any `Manifold`
- **Geodesic tools**: `cartan-geo` provides parameterized geodesics, curvature queries, and Jacobi field integration
- **Fiber bundles**: `Fiber` trait with frame-first SO(d) transport, `CovLaplacian` generic over any fiber type. Built-in fibers: `U1Spin2` (nematics on surfaces), `NematicFiber3D` (3D Q-tensors), `TangentFiber<D>`
- **DEC layer**: `cartan-dec` discretizes covariant differential operators on simplicial meshes for PDE solvers, with complex line bundle sections for k-atic fields, extrinsic Killing operator, and augmented Lagrangian Stokes solver
- **Adaptive remeshing**: `cartan-remesh` provides split, collapse, flip, shift, and curvature-CFL-driven adaptive refinement
- **Python bindings**: `cartan-py` exposes the full library to Python via PyO3 with numpy interop

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
| Q-tensor Sym_0(R^3) | `QTensor3` | 5 | flat, Frobenius metric |

## Crate Structure

```
cartan              facade crate (re-exports everything)
cartan-core         trait definitions, CartanError, Real alias
cartan-manifolds    concrete manifold implementations (8 manifolds + FrameField3D)
cartan-optim        Riemannian optimization: RGD, RCG, RTR, Frechet mean
cartan-geo          geodesic curves, curvature queries, Jacobi fields
cartan-dec          discrete exterior calculus, line bundles, extrinsic operators, Stokes solver
cartan-remesh       adaptive remeshing: split, collapse, flip, shift, curvature-CFL driver
cartan-py           Python bindings via PyO3 (pip install cartan)
```

All manifolds use `nalgebra` `SVector`/`SMatrix` types directly; no intermediate backend crate is needed.

## Embedded and no_std Targets

cartan is designed to run on embedded and no_std targets. The geometry core (manifolds, geodesics, optimization) compiles without the standard library. The discrete exterior calculus layer (`cartan-dec`) requires std because it depends on rayon for parallelism and operates on heap-allocated mesh structures; it is not designed for embedded use.

### What is available without std

The following manifolds compile on any target with an allocator (`alloc`), with all float arithmetic implemented via `libm`:

| Crate | `--no-default-features --features alloc` |
|-------|------------------------------------------|
| `cartan-core` | All traits, `CartanError`, `Real` |
| `cartan-manifolds` | `Euclidean<N>`, `Sphere<N>`, `SpecialOrthogonal<N>`, `SpecialEuclidean<N>`, `Grassmann<N,K>` |
| `cartan-geo` | `Geodesic`, `CurvatureQuery`, `integrate_jacobi` |
| `cartan-optim` | All four optimizers (RGD, RCG, RTR, Fréchet mean) |

The following require `std` because their algorithms depend on iterative eigendecomposition (`symmetric_eigen`) or `std::collections`:

| Crate | Requires `std` |
|-------|----------------|
| `cartan-manifolds` | `Spd<N>`, `Corr<N>`, `QTensor3`, `FrameField3D` |
| `cartan-geo` | `Disclination`, holonomy scanning |
| `cartan-dec` | entire crate (rayon, mesh allocation) |

For robotics and embedded work the key manifolds (SO(3) for attitude, SE(3) for pose, S² for bearing vectors, Grassmann for subspace tracking) are all available without std.

### How to depend on cartan without std

The `cartan` facade crate supports no_std targets. The default `full` feature includes `cartan-dec`; disable it to drop the std requirement:

```toml
[dependencies]
# no_std with allocator (recommended for most embedded targets, e.g. RTIC, Embassy)
cartan = { version = "0.1", default-features = false, features = ["alloc"] }

# std, but without the cartan-dec mesh/PDE layer
cartan = { version = "0.1", default-features = false, features = ["std"] }
```

Alternatively, depend on sub-crates directly if you want finer control over which geometry modules are included:

```toml
[dependencies]
cartan-manifolds = { version = "0.1", default-features = false, features = ["alloc"] }
cartan-optim     = { version = "0.1", default-features = false, features = ["alloc"] }
```

If you are on a fully bare-metal target with no allocator, depend only on `cartan-core` and implement your own manifold types against its traits.

### Example: attitude control on a microcontroller

```rust
#![no_std]
extern crate alloc;

use cartan_manifolds::SpecialOrthogonal;
use cartan_core::Manifold;

// SO(3): 3x3 rotation matrices with bi-invariant metric
let so3 = SpecialOrthogonal::<3>;

// Riemannian log: tangent vector from R_current to R_target (in so(3))
let error_tangent = so3.log(&r_current, &r_target).unwrap();

// Scale by gain and retract back to SO(3)
let r_next = so3.exp(&r_current, &(error_tangent * gain));

// Geodesic interpolation for trajectory following
let r_interp = so3.geodesic_interpolate(&r_start, &r_end, 0.3).unwrap();
```

SE(3) pose estimation, Riemannian optimization over attitude, and geodesic path planning on S² all follow the same pattern.

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

Also provided: sparse `ExteriorDerivative` (d₀, d₁ via `sprs`), K-generic `HodgeStar`, upwind `apply_scalar_advection_generic` / `apply_divergence_generic` (tangent-vector API for any manifold), and backward-compatible flat-mesh wrappers.

### cartan-remesh

`cartan-remesh` provides adaptive remeshing primitives for triangle meshes on Riemannian manifolds. All operations are generic over `M: Manifold` and record mutations in a `RemeshLog` for downstream field interpolation.

```rust,no_run
use cartan_remesh::{split_edge, collapse_edge, flip_edge, adaptive_remesh, RemeshConfig};

// Split an edge, inserting a vertex at the geodesic midpoint
let log = split_edge(&mut mesh, &manifold, edge_idx);

// Adaptive pipeline: split long edges, collapse short edges
let config = RemeshConfig { max_edge_length: 0.5, min_edge_length: 0.05, ..Default::default() };
let log = adaptive_remesh(&mut mesh, &manifold, &mean_h, &gauss_k, &config);
```

## cartan-py (Python bindings)

`cartan-py` exposes cartan's full geometry stack to Python with zero-copy numpy interop. A single abi3 wheel covers Python 3.9+.

```bash
pip install cartan
```

```python
import cartan
import numpy as np

# Manifolds
s2 = cartan.Sphere(2)
p = s2.random_point(seed=42)
q = s2.random_point(seed=43)
v = s2.log(p, q)
print(s2.dist(p, q))

# Optimization
result = cartan.minimize_rgd(s2, cost_fn, grad_fn, p0)

# Geodesics
geo = cartan.Geodesic(s2, p, q)
points = geo.sample(20)

# Curvature
cq = cartan.CurvatureQuery(s2, p)
print(cq.sectional(u, v))

# DEC
mesh = cartan.Mesh.unit_square_grid(32)
ext = cartan.ExteriorDerivative(mesh)
hodge = cartan.HodgeStar(mesh)
ops = cartan.Operators(mesh)
lf = ops.apply_laplace_beltrami(f)

# Batch operations
D = s2.dist_matrix([s2.random_point(seed=i) for i in range(100)])
```

All 8 manifolds (Euclidean, Sphere, SPD up to 8, SO, SE, Grassmann, Corr, QTensor3), all 4 optimizers, geodesic/Jacobi field integration, holonomy/disclination scanning, and the full DEC layer are exposed.

## License

[MIT](LICENSE-MIT)
