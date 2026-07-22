# Cross-language comparison

cartan against Manifolds.jl, geomstats and geoopt. Generated 2026-07-22.

All four libraries read the same fixture file, so agreement is measured
on identical inputs rather than on separately-seeded random draws.
Regenerate with `python/make_fixtures.py` and the three harnesses.

## Agreement

Every comparison agrees to better than `1e-12`. Largest deviation across all 84 comparisons: `8.882e-14`.

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
| spd 3 | dist | 293 | 1405 (4.8x) | 1641 (5.6x) | - | 12854 (43.9x) | - |
| spd 3 | exp | 248 | 3443 (13.9x) | 3585 (14.5x) | - | 32912 (132.8x) | - |
| spd 3 | log | 308 | 3492 (11.3x) | 3529 (11.5x) | - | 35717 (116.1x) | - |
| spd 6 | dist | 1274 | 2830 (2.2x) | 3452 (2.7x) | - | 15348 (12.0x) | - |
| spd 6 | exp | 1545 | 8399 (5.4x) | 8572 (5.5x) | - | 37080 (24.0x) | - |
| spd 6 | log | 1477 | 9060 (6.1x) | 8324 (5.6x) | - | 39704 (26.9x) | - |
| spd 10 | dist | 3430 | 6926 (2.0x) | 7402 (2.2x) | - | 21039 (6.1x) | - |
| spd 10 | exp | 4071 | 17462 (4.3x) | 19339 (4.7x) | - | 45114 (11.1x) | - |
| spd 10 | log | 4135 | 17403 (4.2x) | 20280 (4.9x) | - | 47820 (11.6x) | - |
| sphere 3 | dist | 5 | 8 (1.5x) | 42 (7.8x) | 5 (0.9x) | 7494 (1398.3x) | 6702 (1250.5x) |
| sphere 3 | exp | 14 | 32 (2.2x) | 34 (2.4x) | 16 (1.2x) | 18194 (1273.9x) | 14196 (993.9x) |
| sphere 3 | log | 14 | 45 (3.2x) | 56 (4.0x) | 15 (1.1x) | 15238 (1085.5x) | 20519 (1461.7x) |
| sphere 3 | transport | 6 | 32 (5.2x) | 92 (14.7x) | 15 (2.4x) | 25678 (4095.2x) | - |
| sphere 10 | dist | 6 | 11 (1.7x) | 39 (6.1x) | 6 (0.9x) | 7725 (1206.2x) | 6552 (1023.1x) |
| sphere 10 | exp | 21 | 37 (1.8x) | 45 (2.2x) | 20 (1.0x) | 19005 (917.9x) | 14187 (685.2x) |
| sphere 10 | log | 22 | 59 (2.7x) | 72 (3.4x) | 21 (1.0x) | 15759 (732.8x) | 20308 (944.3x) |
| sphere 10 | transport | 21 | 42 (2.0x) | 99 (4.7x) | 21 (1.0x) | 26380 (1266.7x) | - |
| sphere 50 | dist | 32 | 13 (0.4x) | 70 (2.2x) | 11 (0.3x) | 7514 (234.9x) | 6392 (199.9x) |
| sphere 50 | exp | 61 | 64 (1.1x) | 114 (1.9x) | 30 (0.5x) | 18645 (306.7x) | 13796 (226.9x) |
| sphere 50 | log | 55 | 82 (1.5x) | 162 (3.0x) | 39 (0.7x) | 15459 (283.2x) | 19858 (363.8x) |
| sphere 50 | transport | 93 | 89 (1.0x) | 267 (2.9x) | 56 (0.6x) | 26009 (281.1x) | - |

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
| numba | 4.7x | 21 |
| numba-buffer | 0.9x | 12 |
| geomstats | 281.1x | 21 |
| geoopt | 944.3x | 9 |

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
