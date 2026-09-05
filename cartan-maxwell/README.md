# cartan-maxwell

Maxwell evolution on a prescribed evolving simplicial-Riemannian background.

[![crates.io](https://img.shields.io/crates/v/cartan-maxwell.svg)](https://crates.io/crates/cartan-maxwell)
[![docs.rs](https://docs.rs/cartan-maxwell/badge.svg)](https://docs.rs/cartan-maxwell)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-maxwell` evolves the electromagnetic field on a background whose
geometry changes with time. The background is prescribed rather than solved
for, so this is Maxwell on a given spacetime, not Einstein-Maxwell.

Geometry is represented by **squared edge lengths** (`MeshLengthsSq`), the Regge
primitive. The per-cell metric is linear in them, so a prescribed metric
evolution enters polynomially rather than through square roots, and indefinite
signatures stay representable.

The FEEC layer comes from the upstream
[`formoniq`](https://crates.io/crates/formoniq) crates.

## Structure

| Module | Role |
|---|---|
| `driver` | prescribes the background: a metric as a function of time |
| `evolver` | leapfrog Ampere and Faraday updates on that background |
| `state` | field state as cochains |

`FlrwDriver` supplies a Friedmann-Lemaitre-Robertson-Walker background, scaling
its stored data by `a(t)^2`, since the data is squared lengths.

## Ampere solve

The metric moves every step, so the grade-1 Hodge mass moves with it. The
evolver applies that mass in element form through
[`cartan-matfree`](https://crates.io/crates/cartan-matfree) and solves with
Jacobi-preconditioned conjugate gradients, rebuilding neither a sparse matrix
nor a factorisation.

A mass matrix is spectrally equivalent to its diagonal with a mesh-independent
constant, so the iteration count stops growing once past the pre-asymptotic
regime: 31 iterations at both 10,836 and 17,486 interior degrees of freedom.

This replaced a dense Cholesky factorisation rebuilt on every step, which was
cubic in the interior degree-of-freedom count and capped the reachable mesh.
One Ampere solve, measured on a unit cube:

| interior dofs | dense | matrix-free | speedup |
|---:|---:|---:|---:|
| 3,032 | 934 ms | 8.9 ms | 105x |
| 6,130 | 18.6 s | 18.0 ms | 1,035x |
| 10,836 | 100.4 s | 31.1 ms | 3,228x |
| 17,486 | over the memory cap | 47.3 ms | n/a |

`with_cg` sets the relative residual and the iteration ceiling. The default is
`1e-12` in at most 500 iterations, tight enough to stay out of the energy drift
and reached in about 31 iterations.

## Example

See `examples/maxwell_record.rs`, which evolves a field on an expanding
background and records it through `cartan-io` for ParaView.

```bash
cargo run -p cartan-maxwell --example maxwell_record
```

## License

[MIT](LICENSE-MIT)
