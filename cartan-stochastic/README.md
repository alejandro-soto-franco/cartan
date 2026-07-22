# cartan-stochastic

Stochastic analysis primitives on Riemannian manifolds.

[![crates.io](https://img.shields.io/crates/v/cartan-stochastic.svg)](https://crates.io/crates/cartan-stochastic)
[![docs.rs](https://docs.rs/cartan-stochastic/badge.svg)](https://docs.rs/cartan-stochastic)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## What this crate does

`cartan-stochastic` provides the foundation downstream stochastic stacks
need to do probability, SDE integration, and pathwise-derivative
computation on manifolds — independent of the manifold type, as long as
it implements `cartan-core`'s `Manifold + ParallelTransport + Retraction`.

The architectural purpose is to **prevent primitive duplication** across
the Hsu / Bismut / Elworthy / Malliavin stack: horizontal lift, the
orthonormal frame bundle `O(M)`, Stratonovich development, and Euler-
Maruyama stochastic development are defined once here.

## Core constructs

- **Orthonormal frame bundle `O(M)`** — the total space of orthonormal
  bases of the tangent spaces of `M`. A point is `(p, r)` where
  `r = (e_1, …, e_n)` is an orthonormal basis of `T_p M`.
- **Horizontal lift** — given `u ∈ T_p M`, a curve in `O(M)` whose
  velocity projects to `u` and whose frame evolves by parallel transport.
  Implemented as a right action of `R^n` on the frame bundle.
- **Stochastic development (Eells-Elworthy-Malliavin)** — solve the SDE
  on `O(M)` driven by Euclidean Brownian motion `W_t` with Stratonovich
  differential `∂_t (p, r) = H_i(p, r) ∘ dW^i_t`. Pushed-down trajectory
  on `M` is Brownian motion in the Riemannian sense.
- **Wishart SPD diffusion** — closed-form SDE on the SPD manifold, used
  by `cartan-homog`'s stochastic ensembles for Wishart-perturbed phase
  property propagation.

## Downstream consumers

- `cartan-homog`'s `WishartRveEnsemble` (stochastic feature): perturb
  one phase's property along a Wishart trajectory, aggregate the
  effective tensors with the Karcher (Frechet) mean on `Spd<N>`.
- Reserved foundation for future Bismut-Elworthy-Li Greeks work.

## Example

```rust,no_run
use cartan_manifolds::Spd;
use cartan_stochastic::WishartSpdDiffusion;
use rand::SeedableRng;

let manifold = Spd::<3>::new();
let mut rng = rand::rngs::StdRng::seed_from_u64(0);

let p0 = manifold.identity();
let diff = WishartSpdDiffusion::new(/* parameters */);
// step the SDE forward in time
```

## License

[MIT](LICENSE-MIT)
