# Performance Benchmarks Design Spec

## Goal

Build a comprehensive performance and accuracy benchmarking suite comparing cartan (Rust native + Python bindings) against geomstats and geoopt, with publication-quality figures integrated into cartan.sotofranco.dev under a new `/performance` section.

## Competitors

- **geomstats** (latest stable): primary competitor for manifold geometry operations (exp, log, dist, parallel transport, curvature). NumPy backend (default).
- **geoopt** (latest stable): Riemannian optimisation on PyTorch manifolds. Compared for RGD, RCG, Frechet mean.

## Manifolds

### Head-to-head (all three libraries where available)

| Manifold | cartan | geomstats | geoopt |
|----------|--------|-----------|--------|
| Sphere S(n) | `Sphere(n)` | `Hypersphere(n)` | `Sphere()` |
| SPD(n) | `SPD(n)` | `SPDMatrices(n)` | `SymmetricPositiveDefinite()` |
| SO(n) | `SO(n)` | `SpecialOrthogonal(n)` | n/a |
| Grassmann(n,k) | `Grassmann(n,k)` | `Grassmannian(n,k)` | `Stiefel()` (approximate) |
| Euclidean(n) | `Euclidean(n)` | `Euclidean(n)` | `Euclidean()` |

### cartan-only (no competitor equivalent)

| Manifold | Notes |
|----------|-------|
| SE(n) | Special Euclidean group; geomstats has it but limited trait coverage |
| Corr(n) | Correlation matrices with flat Frobenius metric |
| QTensor3 | Landau-de Gennes Q-tensor for liquid crystal physics |

These appear in the geometry page as a feature coverage table ("cartan offers these manifolds with no equivalent in geomstats or geoopt").

## Operations Benchmarked

### Core geometry (per manifold, dimension sweep)

- `exp(p, v)`: exponential map
- `log(p, q)`: logarithmic map
- `dist(p, q)`: geodesic distance
- `parallel_transport(p, q, v)`: parallel transport along geodesic

### Optimisation (per manifold, convergence + wall-clock)

- **RGD** (Riemannian gradient descent): cartan vs geoopt `RiemannianSGD`
- **RCG** (Riemannian conjugate gradient): cartan-only (geoopt has no CG; shown alongside RGD to demonstrate second-order convergence advantage)
- **Frechet mean**: cartan vs geomstats `FrechetMean` estimator
- **RTR** (Riemannian trust region): cartan-only, no competitor equivalent

## Dimension Sweep

Log-spaced points: n = 2, 3, 5, 10, 25, 50, 100, 250, 500, 1000 (10 points).

Manifold-specific caps:
- Sphere(n), Euclidean(n): full range to n=1000
- SO(n), SPD(n): cap at n=100 (cubic matrix operations)
- Grassmann(n,k): k = floor(n/2), cap at n=100

## Statistical Methodology

### Timing

- **Warmup**: 5 untimed calls per (library, manifold, operation, dimension) to allow JIT/cache warmup
- **Repetitions**: 200 for geometry microbenchmarks, 20 for optimisation
- **Timer**: `time.perf_counter_ns()` (Python), `std::time::Instant` (Rust)
- **Reporting**: median and interquartile range (Q1, Q3)
- **Seeds**: fixed random seeds (42) across all libraries for reproducibility
- **Isolation**: each benchmark runs in a fresh process to avoid cross-contamination

### Accuracy

- **Reference**: mpmath at 50-digit precision for analytical solutions
- **Sample**: 1000 random point pairs per (manifold, operation, dimension)
- **Metrics**: max absolute error, RMS error
- **Operations with analytical solutions**:
  - Sphere dist: `arccos(p . q)` at high precision (tests half-chord vs arccos)
  - SPD dist: `||log(A^{-1/2} B A^{-1/2})||_F` via mpmath
  - SO dist: `||log(R1^T R2)||_F / sqrt(2)` via mpmath
  - Euclidean dist: trivial (sanity check)
  - Frechet mean of 2 points: geodesic midpoint (known closed form)

## Repository Structure

```
cartan/
  benchmarks/
    rust/
      src/main.rs           # Rust timing binary, outputs JSON lines
      Cargo.toml            # standalone binary, depends on cartan workspace crates
    python/
      bench_geometry.py     # exp/log/dist/transport timing for all manifolds
      bench_optimization.py # RGD/RCG/frechet_mean timing
      bench_accuracy.py     # numerical accuracy vs analytical solutions
    figures/
      plot_geometry.py      # reads data, generates geometry page figures
      plot_optimization.py  # reads data, generates optimisation page figures
      plot_accuracy.py      # generates accuracy figures
      theme.py              # shared rcParams, LIGHT/DARK dicts, export helpers
    data/                   # generated timing CSV/JSON (gitignored)
    figures/out/            # generated images (gitignored)
    README.md               # how to reproduce benchmarks
  .gitignore                # includes benchmarks/data/ and benchmarks/figures/out/
```

The entire `benchmarks/data/` and `benchmarks/figures/out/` directories are gitignored. Generated figures are committed only to cartan-docs (not to the cartan repo).

## Rust Timing Binary

`benchmarks/rust/src/main.rs`:

- CLI: `cargo run --release -- --manifold sphere --op exp --dims 2,3,5,10,25,50,100,250,500,1000`
- For each dimension: 5 warmup + 200 timed iterations
- Output: one JSON line per (manifold, op, dim): `{"manifold":"sphere","op":"exp","dim":3,"median_ns":142,"q1_ns":138,"q3_ns":148}`
- Uses `cartan` workspace crates directly (no Python overhead)
- Same random seed (42) as Python harness
- Built as a standalone `[[bin]]` target in `benchmarks/rust/Cargo.toml`

## Figure Generation

### theme.py

```python
RC_BASE = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
}

LIGHT = {
    "bg": "#ffffff", "fg": "#1a1a1a",
    "accent": "#c9a84c", "blue": "#2563eb",
    "muted": "#6b7280", "border": "#dddddd",
}
DARK = {
    "bg": "#111111", "fg": "#e0e0e0",
    "accent": "#c9a84c", "blue": "#6ea8d9",
    "muted": "#484f58", "border": "#333333",
}
```

All figures:
- No grid lines (`ax.grid(False)`)
- Top and right spines off
- DPI 200
- Dual export: PNG then WebP (quality=90) via Pillow
- Light and dark variants with matching filenames
- British English in all labels, captions, and annotations
- Gold (#c9a84c) for cartan lines, blue for competitors
- Dashed gold for Rust native, solid gold for Python bindings
- IQR shaded bands (alpha=0.15)

### Naming convention

`geom_exp_sweep_{light,dark}.webp`, `geom_speedup_heatmap_{light,dark}.webp`, `geom_accuracy_{light,dark}.webp`, `optim_rgd_convergence_{light,dark}.webp`, etc.

### Copy to cartan-docs

A shell script or Makefile target copies final figures from `benchmarks/figures/out/` to `cartan-docs/public/performance/`. This is a manual step (not CI), run after regenerating benchmarks.

## cartan-docs Pages

### Sidebar addition

New "Performance" section in `lib/sidebar-config.ts`, positioned between "DEC" and "Demos":

```typescript
{
  title: "Performance",
  href: "/performance",
  children: [
    { title: "Geometry", href: "/performance/geometry" },
    { title: "Optimisation", href: "/performance/optimisation" },
  ],
}
```

### /performance (overview)

- `app/performance/page.mdx`
- Brief intro: what cartan is, why performance matters, methodology summary
- Headline numbers: "up to Nx faster than geomstats" (filled in after benchmarks run)
- Combined speedup heatmap (geometry + optimisation)
- Links to sub-pages
- Hardware and version footer

### /performance/geometry

- `app/performance/geometry/page.mdx`
- Methodology section: 200 runs, median + IQR, warmup, dimension sweep, hardware
- 4 dimension-sweep figures (exp, log, dist, parallel_transport), each as a 2x2 small-multiples grid (Sphere, SPD, SO, Grassmann; Euclidean omitted as trivial), shared log-log axes, common legend
- 1 speedup heatmap (manifolds x operations, at n=10)
- 1 accuracy comparison (bar chart, max absolute error per library per manifold)
- Cartan-only manifold feature table (SE, Corr, QTensor3)
- All figures via `BlogFigure` with light/dark variants

### /performance/optimisation

- `app/performance/optimisation/page.mdx`
- Methodology section: 20 runs, convergence metric, test problems
- 3 convergence-vs-wallclock figures (RGD, RCG, Frechet mean): x = wall-clock seconds, y = objective/error value
- 1 speedup heatmap (manifolds x optimisers)
- 1 Frechet mean accuracy figure (error vs sample size)
- RTR cartan-only highlight (standalone timing figure, no competitor)
- All figures via `BlogFigure` with light/dark variants

### /performance/layout.tsx

Shared layout wrapping `docs-layout.tsx`, same pattern as other section layouts.

## Optimisation Test Problems

- **RGD/RCG**: Rosenbrock-on-manifold. For Sphere: minimise f(p) = ||p - target||^2 from a random starting point. For SPD: minimise f(X) = tr(AX) + log(det(X)) (log-determinant divergence). Gradient computed analytically, projected to tangent space.
- **Frechet mean**: compute mean of K random points (K = 10, 50, 100, 500). Compare wall-clock and final error (distance to high-iteration reference mean).
- **RTR** (cartan-only): same test problems as RGD/RCG, showing convergence rate advantage of second-order method.

## Reproducibility

`benchmarks/README.md` documents:
- Required Python packages: `cartan`, `geomstats`, `geoopt`, `torch`, `mpmath`, `matplotlib`, `pillow`
- Rust toolchain: stable, `--release` build
- Hardware spec of the machine used for published figures
- Exact commands to reproduce: `python bench_geometry.py`, `cargo run --release -- ...`, `python plot_geometry.py`
- Note that absolute timings depend on hardware; relative speedups are the meaningful metric

## Out of Scope

- Windows benchmarks (Linux only for published figures)
- GPU backends (geomstats with PyTorch GPU, geoopt with CUDA): CPU-only comparison
- Batch/vectorised benchmarks (single-point operations only; batch benchmarks are a future extension)
- JAX backend for geomstats (adds complexity, NumPy backend is the default and most commonly used)
- Automated CI benchmark regression (future work)
