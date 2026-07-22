# Cross-language comparison

cartan against Manifolds.jl, geomstats and geoopt. Generated 2026-07-22.

All four libraries read the same fixture file, so agreement is measured
on identical inputs rather than on separately-seeded random draws.
Regenerate with `python/make_fixtures.py` and the three harnesses.

## Agreement

Every comparison agrees to better than `1e-12`. Largest deviation across all 84 comparisons: `9.415e-14`.

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

| case | op | cartan (ns) | Manifolds.jl | numba | numba buf | geomstats | geoopt |
|---|---|---|---|---|---|---|---|
| spd 3 | dist | 355 | 1405 (4.0x) | 1641 (4.6x) | - | 12854 (36.3x) | - |
| spd 3 | exp | 1334 | 3443 (2.6x) | 3585 (2.7x) | - | 32912 (24.7x) | - |
| spd 3 | log | 1265 | 3492 (2.8x) | 3529 (2.8x) | - | 35717 (28.2x) | - |
| spd 6 | dist | 1283 | 2830 (2.2x) | 3452 (2.7x) | - | 15348 (12.0x) | - |
| spd 6 | exp | 4149 | 8399 (2.0x) | 8572 (2.1x) | - | 37080 (8.9x) | - |
| spd 6 | log | 4078 | 9060 (2.2x) | 8324 (2.0x) | - | 39704 (9.7x) | - |
| spd 10 | dist | 3445 | 6926 (2.0x) | 7402 (2.1x) | - | 21039 (6.1x) | - |
| spd 10 | exp | 10487 | 17462 (1.7x) | 19339 (1.8x) | - | 45114 (4.3x) | - |
| spd 10 | log | 10569 | 17403 (1.6x) | 20280 (1.9x) | - | 47820 (4.5x) | - |
| sphere 3 | dist | 5 | 8 (1.5x) | 42 (7.9x) | 5 (0.9x) | 7494 (1411.5x) | 6702 (1262.3x) |
| sphere 3 | exp | 14 | 32 (2.2x) | 34 (2.4x) | 16 (1.1x) | 18194 (1261.4x) | 14196 (984.2x) |
| sphere 3 | log | 14 | 45 (3.2x) | 56 (4.0x) | 15 (1.1x) | 15238 (1074.4x) | 20519 (1446.7x) |
| sphere 3 | transport | 6 | 32 (5.4x) | 92 (15.4x) | 15 (2.5x) | 25678 (4280.5x) | - |
| sphere 10 | dist | 6 | 11 (1.7x) | 39 (6.3x) | 6 (0.9x) | 7725 (1242.0x) | 6552 (1053.4x) |
| sphere 10 | exp | 21 | 37 (1.8x) | 45 (2.2x) | 20 (1.0x) | 19005 (922.9x) | 14187 (688.9x) |
| sphere 10 | log | 21 | 59 (2.8x) | 72 (3.5x) | 21 (1.0x) | 15759 (760.3x) | 20308 (979.7x) |
| sphere 10 | transport | 21 | 42 (2.0x) | 99 (4.8x) | 21 (1.0x) | 26380 (1280.2x) | - |
| sphere 50 | dist | 32 | 13 (0.4x) | 70 (2.2x) | 11 (0.3x) | 7514 (235.2x) | 6392 (200.1x) |
| sphere 50 | exp | 61 | 64 (1.1x) | 114 (1.9x) | 30 (0.5x) | 18645 (307.2x) | 13796 (227.3x) |
| sphere 50 | log | 54 | 82 (1.5x) | 162 (3.0x) | 39 (0.7x) | 15459 (284.6x) | 19858 (365.6x) |
| sphere 50 | transport | 92 | 89 (1.0x) | 267 (2.9x) | 56 (0.6x) | 26009 (283.7x) | - |

### numba: compiled kernel against the Python call

The `numba` column above is the compiled kernel with the batch loop
inside nopython mode, which is the fair comparison against Rust and
Julia machine code. Calling that kernel from Python adds argument
unboxing and dispatch on every call, and that is what a Python program
actually pays. Both are given because quoting either alone misleads.

| case | op | kernel (ns) | called from Python (ns) | dispatch overhead |
|---|---|---|---|---|
| spd 3 | dist | 1641 | 1847 | 206 |
| spd 3 | exp | 3585 | 3889 | 305 |
| spd 3 | log | 3529 | 4035 | 506 |
| spd 6 | dist | 3452 | 3671 | 218 |
| spd 6 | exp | 8572 | 9337 | 765 |
| spd 6 | log | 8324 | 9096 | 772 |
| spd 10 | dist | 7402 | 7848 | 446 |
| spd 10 | exp | 19339 | 20351 | 1012 |
| spd 10 | log | 20280 | 20460 | 180 |
| sphere 3 | dist | 42 | 157 | 115 |
| sphere 3 | exp | 34 | 384 | 349 |
| sphere 3 | log | 56 | 596 | 540 |
| sphere 3 | transport | 92 | 507 | 415 |
| sphere 10 | dist | 39 | 159 | 120 |
| sphere 10 | exp | 45 | 408 | 362 |
| sphere 10 | log | 72 | 453 | 380 |
| sphere 10 | transport | 99 | 543 | 444 |
| sphere 50 | dist | 70 | 179 | 109 |
| sphere 50 | exp | 114 | 478 | 364 |
| sphere 50 | log | 162 | 530 | 368 |
| sphere 50 | transport | 267 | 724 | 457 |

### Median speedup

| comparator | median ratio | cases |
|---|---|---|
| manifolds.jl | 2.0x | 21 |
| numba | 2.7x | 21 |
| numba-buffer | 0.9x | 12 |
| geomstats | 283.7x | 21 |
| geoopt | 979.7x | 9 |

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
- Two numba columns. `numba` is hand-written kernels with the same
  algorithms cartan uses, returning a value like every other column.
  `numba buf` additionally writes into a caller-owned buffer, uses scalar
  loops instead of array expressions, and enables `fastmath`. That removes
  every allocation, which is most of the cost at small dimension: sphere
  transport at dim 3 goes 92 ns to 15 ns. `fastmath` alone buys nothing
  until allocation is gone, then roughly doubles dim 50.
- `numba buf` therefore compares a mutating interface against a
  value-returning one, and permits reassociation cartan does not. It is
  the ceiling of what this algorithm reaches under numba, not a
  like-for-like API comparison. Values still agree to 5.6e-17.
- Both numba columns are the compiled kernel with the loop inside
  nopython mode. Reaching it from Python costs 109 ns to 724 ns of
  dispatch per call, which exceeds the kernel itself for every sphere
  case. The table below gives both.
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
| numpy | 2.4.6 |
| torch | 2.13.0 |
