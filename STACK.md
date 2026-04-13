# Cartan → Hsu → Bismut / Elworthy → Malliavin

A single-page map of what each crate is responsible for across the stochastic-analysis and options-Greeks stack. Read this before adding a primitive — it may belong in a different layer.

## Layer responsibilities

| Layer | Crate(s) | Lives in | Provides |
|---|---|---|---|
| **L0: geometry** | `cartan-core`, `cartan-manifolds`, `cartan-geo`, `cartan-dec`, `cartan-optim`, `cartan-remesh` | `~/cartan` | Riemannian manifolds, connections, curvature (Riemann / Ricci / scalar), parallel transport, retraction, geodesics, Jacobi fields, holonomy, discrete exterior calculus, manifold optimisers. |
| **L1: stochastic analysis on manifolds** | `cartan-stochastic` | `~/cartan/cartan-stochastic` | Orthonormal frame bundle, horizontal lift, Stratonovich SDE integrator, stochastic development (Eells-Elworthy-Malliavin construction of Brownian motion on `M`). |
| **L2: probabilistic models** | `hsu` | `~/hsu` | Applications of stochastic development: heat kernel estimates, Feynman-Kac, path-space integration, measure-change formulas on `M`. Consumes L1 primitives; does not reinvent them. |
| **L3: pathwise derivatives** | `bismut`, `elworthy` | `~/bismut`, `~/elworthy` | Bismut formula (heat-semigroup derivative), Bismut-Elworthy-Li weights for Greeks, variation processes along SDE paths, Malliavin matrix. Consumes L0 curvature + L1 frame bundle. |
| **L4: applications** | `malliavin` (backtest + live), `volterra`, `mars-lnp` | `~/malliavin`, `~/volterra`, `~/mars-lnp` | Options Greeks, pricing, strategy sizing, regime detection, PDE solvers on manifolds. Consumes L2 + L3. |

## Placement rules

- If a primitive is generic over `Manifold`, it belongs in L0 or L1. It does **not** belong in an application crate even if that crate is the first consumer.
- Stochastic primitives that only need `Manifold + ParallelTransport + Retraction` belong in `cartan-stochastic` (L1), not `hsu`. `hsu` uses them.
- Curvature quantities (Ricci, sectional, Bochner operators) belong in `cartan-geo` even when the only consumer is a BEL weight in `elworthy`. BEL needs Ricci as input, not implementation.
- Heat-kernel estimates go in `hsu` (L2). The **derivatives** of heat semigroups (Bismut formula, BEL) go in `bismut` / `elworthy` (L3).
- Strategy-specific pathwise derivatives (e.g. Greek of a particular spread) go in `malliavin` (L4). The **framework** for pathwise derivation goes in `elworthy` (L3).

## Data flow for a malliavin Greek

1. `malliavin` receives an option position with parameter `θ` (strike, maturity, vol, …).
2. `elworthy` (L3) supplies the BEL weight as a path integral against a control process.
3. The control process is evaluated along an SDE path produced by `cartan-stochastic::stochastic_development` (L1), seeded by an initial frame from `random_frame_at`.
4. The SDE path lives on whichever manifold `cartan-manifolds` (L0) defines for this strategy's state space: `Sphere<N>`, `Spd<N>`, `Grassmann<N,K>`, … .
5. Ricci and sectional curvature along the path come from the manifold's `Curvature` impl in L0.
6. `malliavin` aggregates the BEL weights × payoffs over an MC batch and reports the Greek.

## Traits to depend on (bottom-up)

- L1 crates depend on `cartan_core::{Manifold, Retraction, VectorTransport}` and **nothing higher**.
- L2 crates (`hsu`) depend on L1 + L0; **never** on L3 or L4.
- L3 crates depend on L0 + L1; may optionally pull L2 heat-kernel machinery.
- L4 crates depend on anything below, but implementations in L4 that look reusable should be raised to L3 before the second consumer appears.

## Current state (2026-04-13)

- L0: published on crates.io as `cartan` v0.4.0 (six crates) + `cartan-py` on PyPI.
- L1: `cartan-stochastic` v0.4.0 scaffold in this tree; provides `OrthonormalFrame`, `horizontal_velocity`, `stratonovich_step`, `stochastic_development`. Tested against the S² heat kernel.
- L2: `hsu` is a 424-LOC data-handling scaffold; no stochastic-analysis content yet. First real build-out consumes L1 exclusively for frame / development primitives.
- L3: `bismut` (26 GB of Rust target, ~200 LOC source — early). `elworthy` scaffolded, nine tests, BEL implementation pending. Both will consume L0 Ricci/Jacobi and L1 development.
- L4: `malliavin` options backtest engine runs end-to-end; BEL Greeks path waits on L3.

## Pending upstream work (for contributors)

- [x] Bures-Wasserstein metric on `Spd<N>` (alternative to affine-invariant). L0. `SpdBuresWasserstein<N>` in `cartan-manifolds::spd_bures`; 6 tests.
- [x] Wishart BM generator. L1. `cartan_stochastic::wishart_step`; 2 tests.
- [x] `VectorTransport` for `SpdBuresWasserstein<N>` via differentiated retraction. Unlocks BW-SPD for `stochastic_development`; 2 end-to-end integration tests.
- [x] Jacobi fields along non-geodesic (SDE) base curves. L0. `integrate_jacobi_along_path` in `cartan-geo::jacobi`; 3 integration tests against sphere-BM paths.
- [x] PyO3 bindings for L1 primitives: `stochastic_bm_on_sphere`, `stochastic_bm_on_spd`, `wishart_step` in `cartan-py::stochastic`; 6 Python tests.
- [x] Frame-bundle benchmarks in `benchmarks/rust/src/main.rs` covering `horizontal_velocity`, `stratonovich_step`, `stochastic_development_32`, `wishart_step`.
- [ ] Exact `ParallelTransport` and `Curvature` impls for `SpdBuresWasserstein<N>`. Deferred until an elworthy/bismut consumer reads BW Ricci explicitly; the differentiated-retraction `VectorTransport` is sufficient for `stochastic_development`.
- [ ] Publish frame-bundle timings to `cartan.sotofranco.dev/performance`. Data collection is now possible via `cargo run -p cartan-bench -- --all`.
