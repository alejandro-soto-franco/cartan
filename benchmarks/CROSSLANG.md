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
| spd 3 | dist | 1830 | 1405 (0.8x) | 12854 (7.0x) | - |
| spd 3 | exp | 1381 | 3443 (2.5x) | 32912 (23.8x) | - |
| spd 3 | log | 1328 | 3492 (2.6x) | 35717 (26.9x) | - |
| spd 6 | dist | 5745 | 2830 (0.5x) | 15348 (2.7x) | - |
| spd 6 | exp | 4376 | 8399 (1.9x) | 37080 (8.5x) | - |
| spd 6 | log | 4254 | 9060 (2.1x) | 39704 (9.3x) | - |
| spd 10 | dist | 14653 | 6926 (0.5x) | 21039 (1.4x) | - |
| spd 10 | exp | 10903 | 17462 (1.6x) | 45114 (4.1x) | - |
| spd 10 | log | 10978 | 17403 (1.6x) | 47820 (4.4x) | - |
| sphere 3 | dist | 5 | 8 (1.5x) | 7494 (1387.5x) | 6702 (1240.9x) |
| sphere 3 | exp | 15 | 32 (2.2x) | 18194 (1254.5x) | 14196 (978.8x) |
| sphere 3 | log | 18 | 45 (2.4x) | 15238 (825.9x) | 20519 (1112.1x) |
| sphere 3 | transport | 37 | 32 (0.9x) | 25678 (689.4x) | - |
| sphere 10 | dist | 6 | 11 (1.7x) | 7725 (1214.1x) | 6552 (1029.7x) |
| sphere 10 | exp | 21 | 37 (1.7x) | 19005 (901.0x) | 14187 (672.6x) |
| sphere 10 | log | 21 | 59 (2.8x) | 15759 (744.9x) | 20308 (959.9x) |
| sphere 10 | transport | 97 | 42 (0.4x) | 26380 (272.1x) | - |
| sphere 50 | dist | 33 | 13 (0.4x) | 7514 (225.4x) | 6392 (191.7x) |
| sphere 50 | exp | 68 | 64 (0.9x) | 18645 (275.2x) | 13796 (203.6x) |
| sphere 50 | log | 60 | 82 (1.4x) | 15459 (255.7x) | 19858 (328.5x) |
| sphere 50 | transport | 167 | 89 (0.5x) | 26009 (155.8x) | - |

### Median speedup

| comparator | median ratio | cases |
|---|---|---|
| manifolds.jl | 1.6x | 21 |
| geomstats | 225.4x | 21 |
| geoopt | 959.9x | 9 |

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
