# cartan-geo

Geodesics, curvature, Jacobi fields and holonomy for cartan.

[![crates.io](https://img.shields.io/crates/v/cartan-geo.svg)](https://crates.io/crates/cartan-geo)
[![docs.rs](https://docs.rs/cartan-geo/badge.svg)](https://docs.rs/cartan-geo)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-core` gives geometry at a point: exponential map, logarithm, metric.
`cartan-geo` uses that to answer questions about a whole curve, a whole loop,
or a whole field. Everything here is generic over the `Manifold`, `Curvature`
and `ParallelTransport` traits, so it applies to every manifold in
[`cartan-manifolds`](https://crates.io/crates/cartan-manifolds) and to any
manifold you implement yourself.

| Module | Contents | Needs |
|---|---|---|
| `geodesic` | `Geodesic<M>`: parameterised geodesic, sampling, two-point construction | |
| `curvature` | `CurvatureQuery<M>`: sectional, Ricci and scalar curvature at a point | |
| `jacobi` | `integrate_jacobi`, `integrate_jacobi_along_path`: RK4 geodesic deviation | `alloc` |
| `chern_simons` | `U1Connection`, `Su2Connection`: Chern-Simons 3-form density and box integration | `alloc` |
| `holonomy` | Plaquette holonomy on a frame field, 2D disclination scan | `std` |
| `disclination` | 3D disclination line tracking, Frenet-Serret geometry, event classification | `std` |

## Geodesics

A `Geodesic<M>` holds a base point and an initial velocity, and evaluates
`γ(t) = Exp_p(t v)` at any parameter. Arc length over `[0, 1]` is `||v||`.

```rust
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use cartan_geo::Geodesic;
use nalgebra::SVector;

let s2 = Sphere::<3>;
let p = SVector::from([1.0, 0.0, 0.0]);
let q = SVector::from([0.0, 1.0, 0.0]);

let geo = Geodesic::from_two_points(&s2, p, &q).unwrap();
assert!((geo.length() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

let arc = geo.sample(64);          // 64 points from p to q
let mid = geo.midpoint();          // γ(0.5)
assert!(s2.check_point(&mid).is_ok());
```

`from_two_points` fails at the cut locus, where the minimising geodesic stops
being unique.

## Curvature and Jacobi fields

`CurvatureQuery` fixes a point and reads the curvature quantities off the
`Curvature` trait. `integrate_jacobi` then solves the geodesic deviation
equation

```text
D²J/dt² + R(J, γ') γ' = 0
```

by RK4 on the tangent bundle, projecting back to the tangent space and
parallel-transporting between steps. Positive curvature focuses the field,
negative curvature spreads it.

```rust
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use cartan_geo::{Geodesic, integrate_jacobi, sectional_at};
use nalgebra::SVector;

let s2 = Sphere::<3>;
let p = SVector::from([1.0, 0.0, 0.0]);
let u = SVector::from([0.0, 1.0, 0.0]);
let w = SVector::from([0.0, 0.0, 1.0]);

// The round sphere has constant sectional curvature 1.
assert!((sectional_at(&s2, &p, &u, &w) - 1.0).abs() < 1e-12);

let geo = Geodesic::new(&s2, p, u);
let result = integrate_jacobi(&geo, w, s2.zero_tangent(&p), 64);
assert_eq!(result.field.len(), 65);
```

`integrate_jacobi_along_path` takes the same equation along an arbitrary
sampled curve rather than a geodesic, which lets a Jacobi field ride a
Brownian path produced by
[`cartan-stochastic`](https://crates.io/crates/cartan-stochastic).

## Chern-Simons invariants

`chern_simons` evaluates

```text
CS(A) = Tr( A ∧ dA + (2/3) A ∧ A ∧ A )
```

for a `U(1)` or `su(2)` connection given as three component functions on a
3-parameter chart with their analytic partial derivatives, and integrates it
over a box by tensor-product Gauss-Legendre quadrature. The abelian case drops
the cubic term. On the Hopf bundle over `S³` in Euler coordinates the
integrator reproduces the normalised invariant `-1` to quadrature tolerance.

## Holonomy and disclinations

For a discrete frame field, the holonomy around an oriented plaquette is the
ordered product of edge transition matrices, each gauge-fixed over `D₂` to
resolve the director sign ambiguity. A rotation angle near `π` around a
plaquette means the loop encloses a half-integer disclination. Detection needs
no search for the point where `|Q| = 0`, so it is insensitive to how the defect
core is regularised, and it handles biaxial nematics.

`scan_disclinations` applies this to a 2D grid. The `disclination` module
extends it to 3D: `scan_disclination_lines_3d` tests every grid edge against
its dual loop, `connect_disclination_lines` joins the pierced edges into ordered
lines with tangent, curvature, binormal and torsion, and
`track_disclination_events` classifies creation, annihilation and reconnection
between consecutive frames.

## Performance

Ratios against 0.8.1, measured back to back on the same machine with
`cargo bench -p cartan-geo`, each the smaller of two independent runs.
Absolute figures depend on the machine and on what else is running on it.

| benchmark | 0.8.1 | 0.9.0 |
|---|---|---|
| `integrate_jacobi`, `Sphere<10>`, 32 steps | 1.00x | **1.26x** |
| `integrate_jacobi`, `Spd<6>`, 16 steps | 1.00x | **1.63x** |
| `Geodesic::sample`, `Sphere<10>`, 64 points | 1.00x | 1.05x |
| `Geodesic::sample`, `Spd<6>`, 16 points | 1.00x | 1.00x |

Jacobi integration evaluated the base geodesic twice per step, once for the
step's own base point and once for its endpoint, which is the next step's base
point. Reusing the endpoint halves the exponential maps, and that
alone is the `Sphere<10>` figure. On `Spd<6>` the curvature tensor and the
parallel transport also got cheaper in `cartan-manifolds` 0.9.0, which is the
rest of the 1.63x.

Sampling a geodesic is one exponential map per sample with nothing to share
between them, so it moves only with the underlying manifold.

## no_std

Disable default features and enable `alloc` for embedded targets.

```toml
cartan-geo = { version = "0.9", default-features = false, features = ["alloc"] }
```

`Geodesic` and `CurvatureQuery` are available with no features at all;
`Geodesic::sample_fixed` writes into a fixed-size array in place of the
allocating `sample`. Jacobi integration and the Chern-Simons integrators need
`alloc`. Holonomy and disclination tracking need `std`.

## License

[MIT](LICENSE-MIT)
