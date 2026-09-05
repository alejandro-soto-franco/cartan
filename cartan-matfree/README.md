# cartan-matfree

Matrix-free Galerkin Hodge mass operators and Krylov solvers on a Whitney complex.

[![crates.io](https://img.shields.io/crates/v/cartan-matfree.svg)](https://crates.io/crates/cartan-matfree)
[![docs.rs](https://docs.rs/cartan-matfree/badge.svg)](https://docs.rs/cartan-matfree)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Purpose

Assembling `M_k` and factorising it repeats both on every step of an evolving
Regge background, and the factorisation is cubic in the interior
degree-of-freedom count. This crate keeps the element matrices instead and
applies them inside a conjugate gradient iteration.

A Galerkin mass matrix is spectrally equivalent to its diagonal with a
mesh-independent constant, so Jacobi-preconditioned CG converges in an iteration
count that stops growing with the mesh. The element matrices are computed once
per metric and amortised over those iterations.

Measured against the dense route on a unit cube, one solve at grade 1:

| interior dofs | dense | matrix-free | iterations |
|---:|---:|---:|---:|
| 3,032 | 934 ms | 8.9 ms | 30 |
| 10,836 | 100.4 s | 31.1 ms | 31 |
| 17,486 | over the memory cap | 47.3 ms | 31 |

## Layering

`MassBackend` names every vector operation the iteration needs, so `pcg` runs
wherever the vectors live. `HostMass` is the reference implementation on host
memory and the one correctness is measured against; a device backend implements
the same trait and keeps its vectors resident, which matters because a Krylov
iteration copying its vectors across PCIe every step spends longer on the copies
than on the operator.

`Interior` skips constrained faces in both the row and the column loop of the
element product, which is the Galerkin projection `E^T M E` written without a
projection matrix.

`GatherMap` transposes the cell-to-degree-of-freedom incidence. The scatter form
would need an atomic floating-point add across threads, and an atomic add
reorders the summation, so the result varies between runs. Gathering has the
same memory traffic and no atomic. `cartan-cuda` computes this form.

## Usage

```rust,ignore
use cartan_matfree::{pcg, HostMass, Interior, MassBackend};

let interior = Interior::boundary_constrained(&topology, 1);
let mass = HostMass::restricted(&topology, &geometry, &interior);

let mut x = vec![0.0; mass.ndofs()];
let report = pcg(&mass, &rhs, &mut x, 1e-12, 500);
assert!(report.converged);
```

The operator is symmetric positive definite whenever the metric is Riemannian
and every cell has positive volume, which is what CG requires. On a Lorentzian
metric the pairing is indefinite and this iteration does not apply.

## Licence

[MIT](LICENSE-MIT)
