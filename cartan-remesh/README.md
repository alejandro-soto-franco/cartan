# cartan-remesh

Adaptive remeshing primitives for triangle meshes on Riemannian manifolds.

[![crates.io](https://img.shields.io/crates/v/cartan-remesh.svg)](https://crates.io/crates/cartan-remesh)
[![docs.rs](https://docs.rs/cartan-remesh/badge.svg)](https://docs.rs/cartan-remesh)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-remesh` provides local mesh modification operations for
`Mesh<M, 3, 2>` from `cartan-dec`. All operations are generic over
`M: Manifold`, and every mutation is recorded in a `RemeshLog` so that
downstream solvers can interpolate fields across topology changes.

Primitives:

- `split_edge`, bisects an edge and inserts a new vertex.
- `collapse_edge`, removes a short edge by merging its endpoints.
- `flip_edge`, swaps the diagonal of two adjacent triangles.
- `shift_vertex`, repositions a vertex within its one-ring.

Higher-level tools:

- `length_cross_ratio` and LCR conformal regularisation (spring energy
  and gradient) for maintaining mesh quality.
- `adaptive_remesh`, a driver that applies split/collapse/flip/shift
  passes with a curvature-CFL criterion controlled by `RemeshConfig`.

## Example

```rust,no_run
use cartan_dec::FlatMesh;
use cartan_manifolds::Euclidean;
use cartan_remesh::{adaptive_remesh, RemeshConfig};

let mut mesh = FlatMesh::unit_square_grid(8);
let config = RemeshConfig::default();
let log = adaptive_remesh(&mut mesh, &Euclidean::<2>, &config);
println!("remesh ops: {} splits, {} collapses", log.splits.len(), log.collapses.len());
```

## License

[MIT](../LICENSE-MIT)
