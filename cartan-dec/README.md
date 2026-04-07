# cartan-dec

Discrete exterior calculus on Riemannian manifolds.

[![crates.io](https://img.shields.io/crates/v/cartan-dec.svg)](https://crates.io/crates/cartan-dec)
[![docs.rs](https://docs.rs/cartan-dec/badge.svg)](https://docs.rs/cartan-dec)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-dec` bridges continuous Riemannian geometry (`cartan-core`) to
discrete differential operators for PDE solvers on simplicial meshes. All
metric information flows through the Hodge star; topology is encoded in the
metric-free exterior derivative.

The crate provides:

- `Mesh<M, K, B>`, a generic simplicial complex parameterized by manifold
  type `M`, simplex dimension `K`, and embedding dimension `B`.
- `ExteriorDerivative`, sparse boundary operators d0 and d1 (via `sprs`).
- `HodgeStar`, diagonal Hodge star operators indexed by form degree.
- `Operators`, assembled Laplace-Beltrami, Bochner, and Lichnerowicz
  Laplacians.
- Upwind covariant advection and discrete divergence for scalar, vector,
  and tensor fields.

All operators are generic over `M: Manifold` with const generics `K` and
`B`, so the same code works on flat meshes and curved Riemannian surfaces.

## Example

```rust,no_run
use cartan_dec::{FlatMesh, Operators};
use cartan_manifolds::Euclidean;
use nalgebra::DVector;

// Build a 4x4 uniform triangular grid on [0,1]^2.
let mesh = FlatMesh::unit_square_grid(4);
let ops = Operators::from_mesh(&mesh, &Euclidean::<2>);

// Apply the scalar Laplacian to a vertex field.
let f = DVector::from_element(mesh.n_vertices(), 1.0);
let lf = ops.apply_laplace_beltrami(&f);
```

## License

[MIT](../LICENSE-MIT)
