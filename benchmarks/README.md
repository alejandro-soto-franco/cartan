# cartan benchmarks

Two separate things live here.

**Cross-language comparison** ([CROSSLANG.md](CROSSLANG.md)) measures cartan
against Manifolds.jl, geomstats and geoopt. Agreement first, speed second: a
speed ratio between implementations that disagree is not a result, so every
timing row is gated on the corresponding values having matched.

**Homogenisation comparison** ([HOMOG.md](HOMOG.md), [BEREA.md](BEREA.md))
measures `cartan-homog` against an independent reference implementation.

## Cross-language comparison

All four libraries read one fixture file, so agreement is measured on identical
inputs rather than on separately-seeded random draws that only look alike.

### Setup

Python, via uv. There is no `pip install` step; the environment is pinned by
`requirements.txt`.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`geomstats` is pinned to a main-branch commit rather than to 2.8.0, because the
released version imports `numpy.trapz`, removed in numpy 2.0. Its PyTorch
backend imports the numpy backend transitively, so switching backends does not
avoid it. The fix is upstream but unreleased.

Julia, with its depot off-snapshot in the same idiom as `cargo-targets`:

```bash
export JULIA_DEPOT_PATH=/home/julia/depot
export PATH="/home/julia/juliaup/bin:$PATH"
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

### Running

```bash
# 1. Generate the shared fixtures. Every harness reads these.
.venv/bin/python python/make_fixtures.py

# 2. Each language, in any order.
cargo run --release -p cartan-bench --bin cartan-bench-crosslang
JULIA_DEPOT_PATH=/home/julia/depot julia --project=julia julia/bench_geometry.jl
.venv/bin/python python/bench_crosslang.py

# 3. Compare and write the report.
.venv/bin/python python/compare_crosslang.py --date "$(date +%F)"
```

Each harness writes `results/*_geometry.jsonl` holding both computed values and
timings. The comparison reads all three.

## Criterion benchmarks

Separate from the cross-language rig, and for a different purpose: guarding
against regressions in cartan itself, with the confidence intervals a
hand-rolled timing loop cannot give.

```bash
cargo bench -p cartan-manifolds
```

The two mechanisms coexist deliberately. Criterion's output is shaped for
regression tracking; the JSON-lines harness is shaped for cross-language
tables, which criterion's format does not suit.

## Native timing binaries

```bash
cargo run --release -p cartan-bench --bin cartan-bench -- --all
cargo run --release -p cartan-bench --bin cartan-bench-homog
cargo run --release -p cartan-bench --bin cartan-bench-pipeline
cargo run --release -p cartan-bench --bin cartan-bench-berea
```

## Figures

```bash
.venv/bin/python figures/plot_geometry.py
.venv/bin/python figures/plot_optimization.py
.venv/bin/python figures/plot_accuracy.py
```

Output lands in `figures/out/`.

## Notes

- Absolute timings are hardware-specific; ratios are the meaningful figure. The
  provenance table at the foot of each report records the machine and every
  library version that produced it, so no report is anonymous.
- Fixtures use seed 42 and are committed, so a rerun compares against the same
  inputs rather than regenerating them.
- A comparator that cannot express a case is reported and skipped, never
  silently dropped: a missing row must be distinguishable from a row nobody
  tried.
