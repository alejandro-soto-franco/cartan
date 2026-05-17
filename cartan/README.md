# cartan

Top-level facade for the [cartan](https://github.com/alejandro-soto-franco/cartan)
workspace: Riemannian geometry, manifold optimisation, and geodesic
computation in Rust.

[![crates.io](https://img.shields.io/crates/v/cartan.svg)](https://crates.io/crates/cartan)
[![docs.rs](https://docs.rs/cartan/badge.svg)](https://docs.rs/cartan)

## What this crate does

`cartan` is a re-export façade. Downstream users add `cartan = "0.5"` and
get the most commonly used items from the subcrates under one namespace
— no need to track sibling crate versions individually.

For finer-grained dependency control (e.g. embedded targets that want
`cartan-core` without `cartan-manifolds`, or projects that want the FFT
stack without the optimisation layer), depend on the subcrates directly.

## Workspace layout

| Subcrate | What it does |
|---|---|
| [`cartan-core`](../cartan-core) | Abstract trait system: `Manifold`, `Retraction`, `ParallelTransport`, `Connection`, `Curvature` |
| [`cartan-manifolds`](../cartan-manifolds) | Concrete manifolds: `Sphere<N>`, `Grassmann<N,K>`, `Spd<N>`, `SO(N)`, `SE(N)`, `Stiefel<N,K>` |
| [`cartan-optim`](../cartan-optim) | Optimisation: RGD, RCG, RTR, Fréchet mean |
| [`cartan-geo`](../cartan-geo) | Geodesics, curvature queries, Jacobi field integration |
| [`cartan-dec`](../cartan-dec) | Discrete exterior calculus on simplicial meshes, k-atic line bundles, Stokes solver |
| [`cartan-remesh`](../cartan-remesh) | 2D + 3D adaptive remeshing (split / collapse / flip / red-refinement) |
| [`cartan-stochastic`](../cartan-stochastic) | Frame bundle, horizontal lift, Stratonovich development, Wishart SDE on SPD |
| [`cartan-homog`](../cartan-homog) | Mean-field + full-field homogenisation; spectral solver via `cartan-gpu` |
| [`cartan-homog-valid`](../cartan-homog-valid) | ECHOES cross-validation harness |
| [`cartan-gpu`](../cartan-gpu) | Portable GPU primitives: VkFFT (Vulkan) + cuFFT (CUDA) unified behind one trait, zero-copy interop |
| [`cartan-gpu-sys`](../cartan-gpu-sys) | Raw FFI to vendored VkFFT (internal plumbing for `cartan-gpu`) |
| [`cartan-py`](../cartan-py) | PyO3 / numpy bindings to the full library |

## Quick start

```rust,no_run
use cartan::prelude::*;
use cartan::manifolds::Sphere;

let s2 = Sphere::<3>;
let mut rng = rand::rng();
let p = s2.random_point(&mut rng);
let v = s2.random_tangent(&p, &mut rng);

let q = s2.exp(&p, &v);
let v_back = s2.log(&p, &q).unwrap();
let d = s2.dist(&p, &q).unwrap();
```

See the [top-level README](../README.md) for the full feature catalogue,
no_std support, and the published rendered documentation site at
[cartan.sotofranco.dev](https://cartan.sotofranco.dev).

## License

[MIT](../LICENSE-MIT)
