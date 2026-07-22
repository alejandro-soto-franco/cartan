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
| spd 3 | dist | 402 | 1405 (3.5x) | 12854 (32.0x) | - |
| spd 3 | exp | 1448 | 3443 (2.4x) | 32912 (22.7x) | - |
| spd 3 | log | 1386 | 3492 (2.5x) | 35717 (25.8x) | - |
| spd 6 | dist | 1427 | 2830 (2.0x) | 15348 (10.8x) | - |
| spd 6 | exp | 4713 | 8399 (1.8x) | 37080 (7.9x) | - |
| spd 6 | log | 4541 | 9060 (2.0x) | 39704 (8.7x) | - |
| spd 10 | dist | 3688 | 6926 (1.9x) | 21039 (5.7x) | - |
| spd 10 | exp | 11339 | 17462 (1.5x) | 45114 (4.0x) | - |
| spd 10 | log | 11343 | 17403 (1.5x) | 47820 (4.2x) | - |
| sphere 3 | dist | 6 | 8 (1.4x) | 7494 (1283.6x) | 6702 (1148.0x) |
| sphere 3 | exp | 16 | 32 (2.0x) | 18194 (1141.5x) | 14196 (890.6x) |
| sphere 3 | log | 15 | 45 (2.9x) | 15238 (988.6x) | 20519 (1331.2x) |
| sphere 3 | transport | 7 | 32 (4.9x) | 25678 (3868.4x) | - |
| sphere 10 | dist | 7 | 11 (1.6x) | 7725 (1140.1x) | 6552 (967.0x) |
| sphere 10 | exp | 23 | 37 (1.6x) | 19005 (835.1x) | 14187 (623.4x) |
| sphere 10 | log | 23 | 59 (2.6x) | 15759 (693.1x) | 20308 (893.2x) |
| sphere 10 | transport | 23 | 42 (1.9x) | 26380 (1167.2x) | - |
| sphere 50 | dist | 35 | 13 (0.4x) | 7514 (213.4x) | 6392 (181.5x) |
| sphere 50 | exp | 66 | 64 (1.0x) | 18645 (283.9x) | 13796 (210.1x) |
| sphere 50 | log | 54 | 82 (1.5x) | 15459 (283.9x) | 19858 (364.7x) |
| sphere 50 | transport | 96 | 89 (0.9x) | 26009 (271.1x) | - |

### Median speedup

| comparator | median ratio | cases |
|---|---|---|
| manifolds.jl | 1.9x | 21 |
| geomstats | 271.1x | 21 |
| geoopt | 890.6x | 9 |

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
