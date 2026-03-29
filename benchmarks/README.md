# cartan Performance Benchmarks

Reproducing the figures on [cartan.sotofranco.dev/performance](https://cartan.sotofranco.dev/performance).

## Requirements

```bash
pip install cartan geomstats geoopt torch mpmath matplotlib pillow
```

Rust toolchain: stable (1.85+), release build.

## Running benchmarks

```bash
# 1. Python timing (geometry)
python python/bench_geometry.py

# 2. Python timing (optimisation)
python python/bench_optimization.py

# 3. Python accuracy
python python/bench_accuracy.py

# 4. Rust native timing
cd rust && cargo run --release -- --all

# 5. Generate figures
python figures/plot_geometry.py
python figures/plot_optimization.py
python figures/plot_accuracy.py
```

Output figures land in `figures/out/`. Copy them to `cartan-docs/public/performance/`.

## Notes

- Absolute timings depend on hardware; relative speedups are the meaningful metric.
- Published figures were generated on: (fill in CPU, OS, Python version, library versions).
- All benchmarks use seed 42 for reproducibility.
