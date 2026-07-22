# cartan-maxwell

Maxwell evolution on a prescribed evolving simplicial-Riemannian background.

[![crates.io](https://img.shields.io/crates/v/cartan-maxwell.svg)](https://crates.io/crates/cartan-maxwell)
[![docs.rs](https://docs.rs/cartan-maxwell/badge.svg)](https://docs.rs/cartan-maxwell)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-maxwell` evolves the electromagnetic field on a background whose
geometry changes with time. The background is prescribed rather than solved
for, so this is Maxwell on a given spacetime, not Einstein-Maxwell.

Geometry is carried as **squared edge lengths** (`MeshLengthsSq`), the Regge
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
its stored data by `a(t)^2` since it holds squared lengths.

## Example

See `examples/maxwell_record.rs`, which evolves a field on an expanding
background and records it through `cartan-io` for ParaView.

```bash
cargo run -p cartan-maxwell --example maxwell_record
```

## License

[MIT](LICENSE-MIT)
