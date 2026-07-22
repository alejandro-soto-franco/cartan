# cartan-manifolds

Concrete Riemannian manifold implementations for cartan.

[![crates.io](https://img.shields.io/crates/v/cartan-manifolds.svg)](https://crates.io/crates/cartan-manifolds)
[![docs.rs](https://docs.rs/cartan-manifolds/badge.svg)](https://docs.rs/cartan-manifolds)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-manifolds` provides const-generic manifold types that implement the
trait hierarchy from `cartan-core`. All geometry uses `nalgebra` `SVector`
and `SMatrix` types for statically-sized, stack-allocated storage. Manifold
types are zero-sized; dimensions are checked at compile time.

Available manifolds:

| Type | Space |
|------|-------|
| `Euclidean<N>` | Flat R^N (trivial baseline) |
| `Sphere<N>` | S^{N-1} with round metric |
| `SpecialOrthogonal<N>` | SO(N) with bi-invariant metric |
| `SpecialEuclidean<N>` | SE(N) with product metric |
| `Spd<N>` | SPD(N) with affine-invariant metric |
| `SpdBuresWasserstein<N>` | SPD(N) with the Bures-Wasserstein metric |
| `Grassmann<N, K>` | Gr(N,K) with canonical metric |
| `Corr<N>` | Correlation matrices with Frobenius metric |
| `QTensor3` | Traceless symmetric Q-tensors in 3D |

The crate also provides `FrameField3D` for director/frame-field operations.

## Example

```rust,no_run
use cartan_core::Manifold;
use cartan_manifolds::Sphere;
use nalgebra::SVector;

let s2 = Sphere::<3>;
let north = SVector::from([0.0, 0.0, 1.0]);
let tangent = s2.project_tangent(&north, &SVector::from([1.0, 0.0, 0.0]));
let q = s2.exp(&north, &tangent);
assert!(s2.check(&q).is_ok());
```

## Performance

Median nanoseconds per call on an AMD Ryzen 9 8940HX, from
`cargo bench -p cartan-manifolds`. Ratios are against Manifolds.jl on identical
inputs; see [CROSSLANG.md](../benchmarks/CROSSLANG.md) for the full comparison
and its caveats.

| operation | size | cartan | vs Manifolds.jl |
|---|---|---|---|
| `Spd::exp` | 3 | 252 ns | 13.9x |
| `Spd::log` | 3 | 250 ns | 11.3x |
| `Spd::dist` | 3 | 227 ns | 4.8x |
| `Spd::exp` | 10 | 4176 ns | 4.3x |
| `Sphere::exp` | 10 | 23 ns | 1.6x |
| `Sphere::transport` | 3 | 7 ns | 4.9x |
| `Corr::dist` | 20 | 276 ns | |
| `Grassmann::dist` | Gr(20,5) | 1242 ns | |
| `SpecialOrthogonal::exp` | 3 | 42 ns | |
| `SpecialEuclidean::exp` | 3 | 59 ns | |

Every operation is cross-validated against Manifolds.jl and geomstats on shared
fixtures: 84 comparisons agree to better than 1e-12, worst 8.9e-14.

`exp_into`, `log_into` and `transport_into` write into a caller-owned buffer
and are worth up to 1.6x over the value-returning forms at larger ambient
dimension.

## no_std

Disable default features and enable `alloc` for embedded targets. Some
manifolds (`Spd`, `Corr`, `QTensor3`, `FrameField3D`) require `std`.

```toml
cartan-manifolds = { version = "0.8", default-features = false, features = ["alloc"] }
```

## License

[MIT](LICENSE-MIT)
