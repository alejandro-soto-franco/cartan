# Cross-language comparison

cartan against Manifolds.jl, geomstats and geoopt. Generated 2026-07-22.

All four libraries read the same fixture file, so agreement is measured
on identical inputs rather than on separately-seeded random draws.
Regenerate with `python/make_fixtures.py` and the three harnesses.

## Agreement

Every comparison agrees to better than `1e-12`. Largest deviation across all 51 comparisons: `9.415e-14`.

### Verified convention difference: geoopt transport

geoopt's `transp` is a projection onto the target tangent space, not
exact parallel transport. That is the correct choice for its purpose,
since a Riemannian optimiser needs a cheap transport rather than an
isometric one, and it is why the values differ by far more than
floating point would explain.

This is checked, not assumed: each row confirms geoopt's transport
equals its own `proju` to 1e-14. If geoopt ever switched to exact
transport, these rows would be reported as real disagreements.

| manifold | dim | gap vs cartan | equals its own projection to |
|---|---|---|---|
| sphere | 3 | 1.627e-01 | 0.0e+00 |
| sphere | 10 | 7.795e-02 | 0.0e+00 |
| sphere | 50 | 5.832e-02 | 0.0e+00 |

## Timing

Median nanoseconds per call. A ratio above 1 means cartan is faster.
Rows whose values did not agree are omitted rather than reported.

| case | op | cartan (ns) | Manifolds.jl | geomstats | geoopt |
|---|---|---|---|---|---|
| spd 3 | dist | 405 | 1405 (3.5x) | 12854 (31.8x) | - |
| spd 3 | exp | 1485 | 3443 (2.3x) | 32912 (22.2x) | - |
| spd 3 | log | 1403 | 3492 (2.5x) | 35717 (25.5x) | - |
| spd 6 | dist | 1426 | 2830 (2.0x) | 15348 (10.8x) | - |
| spd 6 | exp | 4653 | 8399 (1.8x) | 37080 (8.0x) | - |
| spd 6 | log | 4566 | 9060 (2.0x) | 39704 (8.7x) | - |
| spd 10 | dist | 3884 | 6926 (1.8x) | 21039 (5.4x) | - |
| spd 10 | exp | 11322 | 17462 (1.5x) | 45114 (4.0x) | - |
| spd 10 | log | 11649 | 17403 (1.5x) | 47820 (4.1x) | - |
| sphere 3 | dist | 6 | 8 (1.3x) | 7494 (1246.1x) | 6702 (1114.4x) |
| sphere 3 | exp | 17 | 32 (1.9x) | 18194 (1085.8x) | 14196 (847.2x) |
| sphere 3 | log | 16 | 45 (2.8x) | 15238 (954.1x) | 20519 (1284.8x) |
| sphere 3 | transport | 40 | 32 (0.8x) | 25678 (644.0x) | - |
| sphere 10 | dist | 7 | 11 (1.6x) | 7725 (1136.8x) | 6552 (964.2x) |
| sphere 10 | exp | 23 | 37 (1.6x) | 19005 (816.0x) | 14187 (609.1x) |
| sphere 10 | log | 23 | 59 (2.5x) | 15759 (681.0x) | 20308 (877.6x) |
| sphere 10 | transport | 116 | 42 (0.4x) | 26380 (226.9x) | - |
| sphere 50 | dist | 36 | 13 (0.4x) | 7514 (207.9x) | 6392 (176.9x) |
| sphere 50 | exp | 70 | 64 (0.9x) | 18645 (266.6x) | 13796 (197.2x) |
| sphere 50 | log | 57 | 82 (1.4x) | 15459 (271.7x) | 19858 (349.0x) |
| sphere 50 | transport | 173 | 89 (0.5x) | 26009 (150.5x) | - |

### Median speedup

| comparator | median ratio | cases |
|---|---|---|
| manifolds.jl | 1.6x | 21 |
| geomstats | 207.9x | 21 |
| geoopt | 847.2x | 9 |

## Caveats

- Absolute timings are hardware-specific. Ratios are the meaningful figure.
- All three harnesses time in batches, not per call. `Instant::now` and
  `perf_counter_ns` cost tens of nanoseconds, comparable to the fastest
  operations here, so per-call timing reported a 15 ns `exp` as 40 ns.
  Batching also needs the inputs black-boxed: with loop-invariant
  arguments the optimiser hoists the whole call out and reports 0 ns.
- The Rust harness is cross-checked against criterion, which measures
  independently: criterion puts sphere exp at dim 3 at 15.8 ns and this
  harness at 15 ns.
- The Python harness uses a fixed 300-sample median and is the least
  precise of the three, though Python call overhead dwarfs that anyway.
- Julia timings exclude compilation, which BenchmarkTools warms away. A
  cold Julia process pays that cost once and this table does not show it.
- geoopt is measured on float64 CPU tensors. It is built for batched GPU
  autograd, so single-point CPU timings are not what it optimises for.
- Parallel transport is compared on the sphere only: the SPD convention
  differs between libraries, so a mismatch would measure the convention.

## Provenance

| | |
|---|---|
| cpu | AMD Ryzen 9 8940HX with Radeon Graphics |
| os | Linux 7.1.4-200.fc44.x86_64 |
| python | 3.12.13 |
| rustc | rustc 1.96.0 (ac68faa20 2026-05-25) |
| julia | julia version 1.12.6 |
| geomstats | 2.8.0 |
| geoopt | 0.5.1 |
| numpy | 2.5.1 |
| torch | 2.13.0 |
