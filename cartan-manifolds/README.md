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

## no_std

Disable default features and enable `alloc` for embedded targets. Some
manifolds (`Spd`, `Corr`, `QTensor3`, `FrameField3D`) require `std`.

```toml
cartan-manifolds = { version = "0.1", default-features = false, features = ["alloc"] }
```

## License

[MIT](../LICENSE-MIT)
