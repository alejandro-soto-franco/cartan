# cartan-homog

Mean-field and full-field homogenisation of random media on SPD manifolds, generic over tensor order.

[![crates.io](https://img.shields.io/crates/v/cartan-homog.svg)](https://crates.io/crates/cartan-homog)
[![docs.rs](https://docs.rs/cartan-homog/badge.svg)](https://docs.rs/cartan-homog)

## What this crate does

`cartan-homog` implements the microscopic-to-macroscopic pipeline described in [From RVE to Mesh: A Pipeline for Heterogeneous Continua](https://sotofranco.dev/blog/posts/rve-to-mesh-pipeline). Given an RVE (representative volume element) described by phase volume fractions, inclusion shapes, and material properties, it computes effective (homogenised) tensors via mean-field or full-field methods.

The crate is generic over tensor order via the `TensorOrder` trait:
- **Order2** (3x3 Kelvin-Mandel): conductivity, permeability, thermal transport
- **Order4** (6x6 Kelvin-Mandel): elasticity, stiffness, compliance

All effective tensors live on the `Spd<N>` manifold from `cartan-manifolds`, with affine-invariant geodesic iteration for self-consistent schemes and Karcher-mean aggregation for stochastic ensembles.

## Schemes

| Scheme | Type | Reference medium |
|---|---|---|
| `VoigtBound` / `ReussBound` | Bounds | n/a |
| `Dilute` / `DiluteStress` | Non-interacting | Matrix |
| `MoriTanaka` | Matrix-inclusion | Matrix |
| `SelfConsistent` | Fully disordered | Effective (SPD-geodesic iteration) |
| `AsymmetricSc` | Matrix-inclusion, SC-like | Matrix + effective |
| `Maxwell` / `PonteCastanedaWillis` | Distribution ellipsoid | Matrix |
| `Differential` | ODE (Roscoe-Brinkman) | Evolving |
| `DifferentialCompliance` | ODE dual (Norris-Davies) | Evolving (compliance) |

## Shapes

`Sphere`, `Spheroid` (oblate/prolate), `PennyCrack` (Budiansky-O'Connell), `Ellipsoid` (Carlson RD), `SphereNLayers` (Herve-Zaoui), `UserInclusion` (trait object).

Lebedev quadrature (degree 14) for Sphere Hill tensor in anisotropic reference media.

## Full-field solver (`--features full-field`)

3D cell-problem solver on periodic tetrahedral meshes:
- Kuhn-triangulated cube mesh builder with periodic face matching
- P1-FEM stiffness assembly with per-tet conductivity
- Four-tier solver ladder: Jacobi-PCG, ILU(0)-PCG, AMG-PCG (two-level aggregation), dense LU fallback
- Adaptive refinement via `cartan-remesh` (barycentric or red 1-to-8)
- Voxel import (`load_voxel_raw_u8`) for micro-CT data
- Macroscale slab Darcy solver with anisotropic per-tet permeability
- Hausdorff gate for refinement-vs-analytic validation

## Stochastic ensembles (`--features stochastic`)

`WishartRveEnsemble`: perturb one phase's property along a Wishart SDE trajectory (via `cartan-stochastic`), sample N realisations through any scheme, aggregate with the Karcher (Frechet) mean on `Spd<N>`.

## Validation

Validated against [ECHOES](https://jfbarthelemy.github.io/echoes/) (Barthelemy 2022, Zenodo DOI 10.5281/zenodo.14959866):
- 84 cases pass at d_AI < 3e-15 (affine-invariant SPD distance)
- 42 cases skip (Spheroid Order4 anisotropic-ref Hill, pending Lebedev 4D quadrature)
- 0 failures on the 126-case extended matrix
- All interaction-corrected schemes respect Hashin-Shtrikman bounds (0 violations on 60 checked cases)
- 200x median speedup over ECHOES per homogenisation call
- ASC formula bug caught and fixed by the HS envelope validator

Real-data benchmark: Berea sandstone (Zimmerman 1991 mineral moduli, Andra et al. 2013 porosity) matches ECHOES to machine precision on all schemes, cross-checked against Hart 1995 laboratory measurements and Arns 2002 FEM.

## Quick start

```rust
use cartan_homog::prelude::*;  // or individual imports
use std::sync::Arc;

let mut rve = Rve::<Order2>::new();
rve.add_phase(Phase {
    name: "MATRIX".into(),
    shape: Arc::new(Sphere),
    property: Order2::scalar(1.0),   // k = 1
    fraction: 0.8,
});
rve.add_phase(Phase {
    name: "INCLUSION".into(),
    shape: Arc::new(Sphere),
    property: Order2::scalar(5.0),   // k = 5
    fraction: 0.2,
});
rve.set_matrix("MATRIX");

let k_eff = MoriTanaka.homogenize(&rve, &SchemeOpts::default()).unwrap();
println!("k_eff = {:?}", k_eff.tensor);
```
