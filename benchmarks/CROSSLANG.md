# Cross-language comparison

cartan against Manifolds.jl, geomstats and geoopt. Generated 2026-07-22.

All four libraries read the same fixture file, so agreement is measured
on identical inputs rather than on separately-seeded random draws.
Regenerate with `python/make_fixtures.py` and the three harnesses.

## Agreement

Every comparison agrees to better than `1e-12`. Largest deviation across all 84 comparisons: `7.994e-14`.

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
| spd 3 | dist | 291 | 1405 (4.8x) | 1641 (5.6x) | - | 12854 (44.1x) | - |
| spd 3 | exp | 483 | 3443 (7.1x) | 3585 (7.4x) | - | 32912 (68.1x) | - |
| spd 3 | log | 485 | 3492 (7.2x) | 3529 (7.3x) | - | 35717 (73.7x) | - |
| spd 6 | dist | 1278 | 2830 (2.2x) | 3452 (2.7x) | - | 15348 (12.0x) | - |
| spd 6 | exp | 2630 | 8399 (3.2x) | 8572 (3.3x) | - | 37080 (14.1x) | - |
| spd 6 | log | 2528 | 9060 (3.6x) | 8324 (3.3x) | - | 39704 (15.7x) | - |
| spd 10 | dist | 3457 | 6926 (2.0x) | 7402 (2.1x) | - | 21039 (6.1x) | - |
| spd 10 | exp | 7042 | 17462 (2.5x) | 19339 (2.7x) | - | 45114 (6.4x) | - |
| spd 10 | log | 6979 | 17403 (2.5x) | 20280 (2.9x) | - | 47820 (6.9x) | - |
| sphere 3 | dist | 5 | 8 (1.5x) | 42 (7.9x) | 5 (0.9x) | 7494 (1418.4x) | 6702 (1268.5x) |
| sphere 3 | exp | 14 | 32 (2.2x) | 34 (2.4x) | 16 (1.2x) | 18194 (1275.1x) | 14196 (994.9x) |
| sphere 3 | log | 14 | 45 (3.2x) | 56 (4.0x) | 15 (1.1x) | 15238 (1077.8x) | 20519 (1451.4x) |
| sphere 3 | transport | 6 | 32 (5.4x) | 92 (15.3x) | 15 (2.5x) | 25678 (4266.9x) | - |
| sphere 10 | dist | 6 | 11 (1.7x) | 39 (6.3x) | 6 (0.9x) | 7725 (1241.6x) | 6552 (1053.1x) |
| sphere 10 | exp | 21 | 37 (1.8x) | 45 (2.2x) | 20 (1.0x) | 19005 (927.0x) | 14187 (692.0x) |
| sphere 10 | log | 21 | 59 (2.8x) | 72 (3.5x) | 21 (1.0x) | 15759 (764.9x) | 20308 (985.7x) |
| sphere 10 | transport | 21 | 42 (2.0x) | 99 (4.8x) | 21 (1.0x) | 26380 (1272.0x) | - |
| sphere 50 | dist | 33 | 13 (0.4x) | 70 (2.1x) | 11 (0.3x) | 7514 (230.5x) | 6392 (196.1x) |
| sphere 50 | exp | 61 | 64 (1.1x) | 114 (1.9x) | 30 (0.5x) | 18645 (304.5x) | 13796 (225.3x) |
| sphere 50 | log | 51 | 82 (1.6x) | 162 (3.2x) | 39 (0.8x) | 15459 (300.9x) | 19858 (386.5x) |
| sphere 50 | transport | 88 | 89 (1.0x) | 267 (3.0x) | 56 (0.6x) | 26009 (294.6x) | - |

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
| manifolds.jl | 2.2x | 21 |
| numba | 3.3x | 21 |
| numba-buffer | 0.9x | 12 |
| geomstats | 294.6x | 21 |
| geoopt | 985.7x | 9 |

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
