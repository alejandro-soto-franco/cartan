# Performance Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a performance and accuracy benchmarking suite comparing cartan (Rust + Python) against geomstats and geoopt, with publication-quality figures on cartan.sotofranco.dev.

**Architecture:** Python timing harness benchmarks all three libraries via their Python APIs. A standalone Rust binary provides native-speed reference timings as JSON. matplotlib generates dual-theme (light/dark) figures following the sotofranco.dev blog conventions. Three new pages on cartan-docs display the results.

**Tech Stack:** Python 3.12, cartan (Python bindings), geomstats 2.8, geoopt 0.5, torch, mpmath, matplotlib, Pillow, Rust (cartan workspace crates), Next.js 16 (cartan-docs)

---

## File Structure

### cartan repo (`benchmarks/`)

| File | Responsibility |
|------|---------------|
| `benchmarks/.gitignore` | Ignores `data/` and `figures/out/` |
| `benchmarks/rust/Cargo.toml` | Standalone binary depending on cartan workspace crates |
| `benchmarks/rust/src/main.rs` | Rust timing binary, outputs JSON lines |
| `benchmarks/python/bench_geometry.py` | Timing harness for exp/log/dist/transport across all manifolds |
| `benchmarks/python/bench_optimization.py` | Timing harness for RGD/RCG/Frechet mean |
| `benchmarks/python/bench_accuracy.py` | Accuracy vs mpmath high-precision reference |
| `benchmarks/figures/theme.py` | Shared rcParams, LIGHT/DARK dicts, `apply_theme()`, `save()` |
| `benchmarks/figures/plot_geometry.py` | Generates geometry page figures from data/ CSVs + JSON |
| `benchmarks/figures/plot_optimization.py` | Generates optimisation page figures |
| `benchmarks/figures/plot_accuracy.py` | Generates accuracy figures |
| `benchmarks/README.md` | Reproduction instructions |

### cartan-docs repo (`~/cartan-docs`)

| File | Responsibility |
|------|---------------|
| `lib/sidebar-config.ts` | Add Performance section between DEC and Demos |
| `app/performance/layout.tsx` | Shared layout for performance pages |
| `app/performance/page.mdx` | Overview with headline numbers and combined heatmap |
| `app/performance/geometry/page.mdx` | Geometry benchmarks: 4 sweep figs, heatmap, accuracy |
| `app/performance/optimisation/page.mdx` | Optimisation benchmarks: 3 convergence figs, heatmap, accuracy |
| `public/performance/*.webp` | Generated figure images (light + dark variants) |

---

### Task 1: Benchmark scaffolding and theme module

**Files:**
- Create: `benchmarks/.gitignore`
- Create: `benchmarks/figures/theme.py`
- Create: `benchmarks/README.md`

- [ ] **Step 1: Create benchmarks/.gitignore**

```
data/
figures/out/
__pycache__/
*.pyc
```

- [ ] **Step 2: Create the theme module**

Create `benchmarks/figures/theme.py`:

```python
"""
Shared figure theme for cartan performance benchmarks.

Matches the sotofranco.dev blog figure conventions:
  - text.usetex: True, Computer Modern Roman
  - DPI 200, no grid, top/right spines off
  - Dual light/dark export as PNG + WebP
  - British English in all labels and captions
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = pathlib.Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RC_BASE = {
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
    ),
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
}

LIGHT = {
    "bg": "#ffffff",
    "fg": "#1a1a1a",
    "accent": "#c9a84c",
    "accent_fill": "rgba(201, 168, 76, 0.15)",
    "blue": "#2563eb",
    "blue_fill": "rgba(37, 99, 235, 0.15)",
    "muted": "#6b7280",
    "border": "#dddddd",
}

DARK = {
    "bg": "#111111",
    "fg": "#e0e0e0",
    "accent": "#c9a84c",
    "accent_fill": "rgba(201, 168, 76, 0.15)",
    "blue": "#6ea8d9",
    "blue_fill": "rgba(110, 168, 217, 0.15)",
    "muted": "#484f58",
    "border": "#333333",
}

THEMES = {"light": LIGHT, "dark": DARK}


def rc_context(theme):
    """Return a matplotlib rc_context with theme-specific colours."""
    return plt.rc_context({
        **RC_BASE,
        "axes.facecolor": theme["bg"],
        "figure.facecolor": theme["bg"],
        "text.color": theme["fg"],
        "axes.labelcolor": theme["fg"],
        "xtick.color": theme["fg"],
        "ytick.color": theme["fg"],
    })


def apply_theme(fig, axes_list, theme):
    """Apply background and foreground colours to a figure and its axes."""
    fig.patch.set_facecolor(theme["bg"])
    for ax in axes_list:
        ax.set_facecolor(theme["bg"])
        ax.tick_params(colors=theme["fg"])
        ax.xaxis.label.set_color(theme["fg"])
        ax.yaxis.label.set_color(theme["fg"])
        if ax.get_title():
            ax.title.set_color(theme["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme["fg"])
        ax.grid(False)


def save(fig, name, theme_name):
    """Save figure as PNG (dpi=200) and WebP (quality=90)."""
    png_path = OUT_DIR / f"{name}_{theme_name}.png"
    webp_path = OUT_DIR / f"{name}_{theme_name}.webp"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img = Image.open(png_path).convert("RGBA")
    img.save(webp_path, "WEBP", quality=90)
    print(f"  saved {png_path.name} + {webp_path.name}")
```

- [ ] **Step 3: Create README.md**

Create `benchmarks/README.md`:

```markdown
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
```

- [ ] **Step 4: Verify theme module imports**

Run: `cd benchmarks && python -c "from figures.theme import RC_BASE, LIGHT, DARK, THEMES, rc_context, apply_theme, save; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add benchmarks/.gitignore benchmarks/figures/theme.py benchmarks/README.md
git commit -m "feat(bench): scaffold benchmark directory with theme module"
```

---

### Task 2: Python geometry benchmark harness

**Files:**
- Create: `benchmarks/python/bench_geometry.py`

- [ ] **Step 1: Create the geometry benchmark script**

Create `benchmarks/python/bench_geometry.py`:

```python
"""
Geometry microbenchmarks: exp, log, dist, parallel_transport.

Compares cartan (Python bindings), geomstats, and geoopt across
manifolds and dimensions. Outputs CSV to benchmarks/data/.

Run from benchmarks/:
    python python/bench_geometry.py
"""

import csv
import os
import pathlib
import time

import numpy as np

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WARMUP = 5
REPS = 200

# Log-spaced dimensions; manifold-specific caps applied below.
ALL_DIMS = [2, 3, 5, 10, 25, 50, 100, 250, 500, 1000]

# ── cartan helpers ──────────────────────────────────────────────────────────

def _cartan_manifolds():
    """Return dict of manifold name -> list of (dim, manifold, make_point, make_tangent)."""
    import cartan

    def _sphere(n):
        m = cartan.Sphere(n)
        def pt():
            return m.random_point(seed=None)
        def tv(p):
            return m.random_tangent(p, seed=None)
        return m, pt, tv

    def _spd(n):
        m = cartan.SPD(n)
        def pt():
            return m.random_point(seed=None)
        def tv(p):
            return m.random_tangent(p, seed=None)
        return m, pt, tv

    def _so(n):
        m = cartan.SO(n)
        def pt():
            return m.random_point(seed=None)
        def tv(p):
            return m.random_tangent(p, seed=None)
        return m, pt, tv

    def _grassmann(n):
        k = max(1, n // 2)
        m = cartan.Grassmann(n, k)
        def pt():
            return m.random_point(seed=None)
        def tv(p):
            return m.random_tangent(p, seed=None)
        return m, pt, tv

    def _euclidean(n):
        m = cartan.Euclidean(n)
        def pt():
            return m.random_point(seed=None)
        def tv(p):
            return m.random_tangent(p, seed=None)
        return m, pt, tv

    return {
        "sphere": (_sphere, ALL_DIMS),
        "spd": (_spd, [d for d in ALL_DIMS if d <= 100]),
        "so": (_so, [d for d in ALL_DIMS if d <= 100]),
        "grassmann": (_grassmann, [d for d in ALL_DIMS if d <= 100]),
        "euclidean": (_euclidean, ALL_DIMS),
    }


def _geomstats_manifolds():
    """Return dict of manifold name -> list of (dim, metric, make_point, make_tangent)."""
    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.geometry.spd_matrices import SPDMatrices
    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    from geomstats.geometry.grassmannian import Grassmannian
    from geomstats.geometry.euclidean import Euclidean

    def _sphere(n):
        m = Hypersphere(dim=n)
        metric = m.metric
        def pt():
            return m.random_point()
        def tv(p):
            return m.random_tangent_vec(p)
        return metric, pt, tv

    def _spd(n):
        m = SPDMatrices(n)
        metric = m.metric
        def pt():
            return m.random_point()
        def tv(p):
            return m.random_tangent_vec(p)
        return metric, pt, tv

    def _so(n):
        m = SpecialOrthogonal(n)
        metric = m.metric
        def pt():
            return m.random_point()
        def tv(p):
            return m.random_tangent_vec(p)
        return metric, pt, tv

    def _grassmann(n):
        k = max(1, n // 2)
        m = Grassmannian(n, k)
        metric = m.metric
        def pt():
            return m.random_point()
        def tv(p):
            return m.random_tangent_vec(p)
        return metric, pt, tv

    def _euclidean(n):
        m = Euclidean(dim=n)
        metric = m.metric
        def pt():
            return m.random_point()
        def tv(p):
            return np.random.randn(*p.shape)
        return metric, pt, tv

    return {
        "sphere": (_sphere, ALL_DIMS),
        "spd": (_spd, [d for d in ALL_DIMS if d <= 100]),
        "so": (_so, [d for d in ALL_DIMS if d <= 100]),
        "grassmann": (_grassmann, [d for d in ALL_DIMS if d <= 100]),
        "euclidean": (_euclidean, ALL_DIMS),
    }


# ── Timing core ─────────────────────────────────────────────────────────────

OPS_CARTAN = {
    "exp": lambda m, p, q, v: m.exp(p, v),
    "log": lambda m, p, q, v: m.log(p, q),
    "dist": lambda m, p, q, v: m.dist(p, q),
    "parallel_transport": lambda m, p, q, v: m.parallel_transport(p, q, v),
}

OPS_GEOMSTATS = {
    "exp": lambda metric, p, q, v: metric.exp(v, p),
    "log": lambda metric, p, q, v: metric.log(q, p),
    "dist": lambda metric, p, q, v: metric.dist(p, q),
    "parallel_transport": lambda metric, p, q, v: metric.parallel_transport(v, p, q),
}


def time_op(fn, reps=REPS, warmup=WARMUP):
    """Time fn() for reps iterations after warmup. Returns list of nanosecond durations."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_library(name, manifold_factories, ops, make_args_fn):
    """
    Run all benchmarks for one library.

    Parameters
    ----------
    name : str
        Library name ("cartan" or "geomstats").
    manifold_factories : dict
        name -> (factory_fn, dims_list)
    ops : dict
        op_name -> callable(manifold_or_metric, p, q, v)
    make_args_fn : callable
        (factory_fn, dim) -> (manifold_or_metric, p, q, v)
    """
    rows = []
    for mname, (factory, dims) in manifold_factories.items():
        for dim in dims:
            print(f"  {name}/{mname}/n={dim}", end="", flush=True)
            try:
                obj, make_pt, make_tv = factory(dim)
                p = make_pt()
                q = make_pt()
                v = make_tv(p)
            except Exception as e:
                print(f" SKIP ({e})")
                continue

            for op_name, op_fn in ops.items():
                try:
                    times = time_op(lambda: op_fn(obj, p, q, v))
                    times_arr = np.array(times)
                    median = float(np.median(times_arr))
                    q1 = float(np.percentile(times_arr, 25))
                    q3 = float(np.percentile(times_arr, 75))
                    rows.append({
                        "library": name,
                        "manifold": mname,
                        "dim": dim,
                        "op": op_name,
                        "median_ns": median,
                        "q1_ns": q1,
                        "q3_ns": q3,
                        "reps": REPS,
                    })
                    print(f" {op_name}={median/1e3:.1f}us", end="", flush=True)
                except Exception as e:
                    print(f" {op_name}=ERR({e})", end="", flush=True)
            print()
    return rows


def main():
    np.random.seed(SEED)

    all_rows = []

    # ── cartan ───────────────────────────────────────────────────────────
    print("=== cartan (Python bindings) ===")
    cartan_factories = _cartan_manifolds()
    all_rows.extend(bench_library("cartan", cartan_factories, OPS_CARTAN,
                                  lambda f, d: f(d)))

    # ── geomstats ────────────────────────────────────────────────────────
    print("\n=== geomstats ===")
    geomstats_factories = _geomstats_manifolds()
    all_rows.extend(bench_library("geomstats", geomstats_factories, OPS_GEOMSTATS,
                                  lambda f, d: f(d)))

    # ── Write CSV ────────────────────────────────────────────────────────
    out_path = DATA_DIR / "geometry_timings.csv"
    fields = ["library", "manifold", "dim", "op", "median_ns", "q1_ns", "q3_ns", "reps"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a quick smoke test (sphere only, small dims)**

Run: `cd benchmarks && python -c "
import sys; sys.path.insert(0, 'python')
from bench_geometry import _cartan_manifolds, _geomstats_manifolds, time_op, OPS_CARTAN, OPS_GEOMSTATS
cm = _cartan_manifolds()
m, pt, tv = cm['sphere'][0](3)
p, q, v = pt(), pt(), tv(p)
t = time_op(lambda: m.exp(p, v), reps=10, warmup=2)
print('cartan sphere exp n=3:', [x/1e3 for x in t[:3]], 'us (first 3)')
gm = _geomstats_manifolds()
metric, gpt, gtv = gm['sphere'][0](3)
gp, gq, gv = gpt(), gpt(), gtv(gp)
t2 = time_op(lambda: metric.exp(gv, gp), reps=10, warmup=2)
print('geomstats sphere exp n=3:', [x/1e3 for x in t2[:3]], 'us (first 3)')
"`

Expected: both print microsecond timings, cartan significantly faster.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/python/bench_geometry.py
git commit -m "feat(bench): add geometry timing harness for cartan and geomstats"
```

---

### Task 3: Python optimisation benchmark harness

**Files:**
- Create: `benchmarks/python/bench_optimization.py`

- [ ] **Step 1: Create the optimisation benchmark script**

Create `benchmarks/python/bench_optimization.py`:

```python
"""
Optimisation benchmarks: RGD, RCG, Frechet mean.

Compares cartan vs geoopt (RGD) and cartan vs geomstats (Frechet mean).
Outputs CSV to benchmarks/data/.

Run from benchmarks/:
    python python/bench_optimization.py
"""

import csv
import pathlib
import time

import numpy as np
import torch

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
WARMUP = 3
REPS = 20
FRECHET_SAMPLE_SIZES = [10, 50, 100, 500]

# Dimensions for optimisation benchmarks (smaller set, optimisation is slower).
OPT_DIMS = [3, 5, 10, 25, 50]


# ── Test problems ───────────────────────────────────────────────────────────

def sphere_cost_grad_cartan(manifold, target):
    """Minimise f(p) = dist(p, target)^2 on the sphere."""
    def cost(p):
        d = manifold.dist(p, target)
        return float(d * d)

    def grad(p):
        v = manifold.log(p, target)
        if v is None:
            return manifold.zero_tangent(p)
        return -2.0 * v

    return cost, grad


def sphere_cost_grad_geoopt(target_tensor):
    """Minimise f(p) = ||p - target||^2 projected onto sphere."""
    def closure(p):
        return ((p - target_tensor) ** 2).sum()
    return closure


# ── cartan benchmarks ───────────────────────────────────────────────────────

def bench_cartan_rgd(dims):
    """Benchmark cartan.minimize_rgd on Sphere."""
    import cartan
    rows = []
    for dim in dims:
        m = cartan.Sphere(dim)
        target = m.random_point(seed=SEED)
        cost, grad = sphere_cost_grad_cartan(m, target)
        x0 = m.random_point(seed=SEED + 1)

        times = []
        for _ in range(WARMUP):
            cartan.minimize_rgd(m, cost, grad, x0, max_iters=200)

        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            result = cartan.minimize_rgd(m, cost, grad, x0, max_iters=200)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)

        times_arr = np.array(times)
        rows.append({
            "library": "cartan", "optimiser": "rgd", "manifold": "sphere",
            "dim": dim, "median_ns": float(np.median(times_arr)),
            "q1_ns": float(np.percentile(times_arr, 25)),
            "q3_ns": float(np.percentile(times_arr, 75)),
            "final_value": float(result.value),
            "iterations": int(result.iterations), "reps": REPS,
        })
        print(f"  cartan/rgd/sphere/n={dim}: {np.median(times_arr)/1e6:.1f}ms "
              f"({result.iterations} iters, val={result.value:.2e})")
    return rows


def bench_cartan_rcg(dims):
    """Benchmark cartan.minimize_rcg on Sphere (cartan-only, no geoopt CG)."""
    import cartan
    rows = []
    for dim in dims:
        m = cartan.Sphere(dim)
        target = m.random_point(seed=SEED)
        cost, grad = sphere_cost_grad_cartan(m, target)
        x0 = m.random_point(seed=SEED + 1)

        times = []
        for _ in range(WARMUP):
            cartan.minimize_rcg(m, cost, grad, x0, max_iters=200)

        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            result = cartan.minimize_rcg(m, cost, grad, x0, max_iters=200)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)

        times_arr = np.array(times)
        rows.append({
            "library": "cartan", "optimiser": "rcg", "manifold": "sphere",
            "dim": dim, "median_ns": float(np.median(times_arr)),
            "q1_ns": float(np.percentile(times_arr, 25)),
            "q3_ns": float(np.percentile(times_arr, 75)),
            "final_value": float(result.value),
            "iterations": int(result.iterations), "reps": REPS,
        })
        print(f"  cartan/rcg/sphere/n={dim}: {np.median(times_arr)/1e6:.1f}ms "
              f"({result.iterations} iters, val={result.value:.2e})")
    return rows


def bench_cartan_frechet(dims, sample_sizes):
    """Benchmark cartan.frechet_mean on Sphere."""
    import cartan
    rows = []
    for dim in dims:
        m = cartan.Sphere(dim)
        for k in sample_sizes:
            points = [m.random_point(seed=SEED + i) for i in range(k)]

            times = []
            for _ in range(WARMUP):
                cartan.frechet_mean(m, points)

            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                result = cartan.frechet_mean(m, points)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)

            times_arr = np.array(times)
            rows.append({
                "library": "cartan", "optimiser": "frechet_mean",
                "manifold": "sphere", "dim": dim, "sample_size": k,
                "median_ns": float(np.median(times_arr)),
                "q1_ns": float(np.percentile(times_arr, 25)),
                "q3_ns": float(np.percentile(times_arr, 75)),
                "iterations": int(result.iterations), "reps": REPS,
            })
            print(f"  cartan/frechet/sphere/n={dim}/k={k}: "
                  f"{np.median(times_arr)/1e6:.1f}ms ({result.iterations} iters)")
    return rows


# ── geoopt benchmarks ───────────────────────────────────────────────────────

def bench_geoopt_rgd(dims):
    """Benchmark geoopt RiemannianSGD on Sphere."""
    import geoopt
    rows = []
    for dim in dims:
        manifold = geoopt.Sphere()
        torch.manual_seed(SEED)
        target = torch.randn(dim + 1)
        target = target / target.norm()

        def run_once():
            torch.manual_seed(SEED + 1)
            x = torch.randn(dim + 1)
            x = x / x.norm()
            x = geoopt.ManifoldParameter(x, manifold=manifold)
            optimiser = geoopt.optim.RiemannianSGD([x], lr=0.1)
            for _ in range(200):
                optimiser.zero_grad()
                loss = ((x - target) ** 2).sum()
                loss.backward()
                optimiser.step()
            return float(loss.item())

        for _ in range(WARMUP):
            run_once()

        times = []
        for _ in range(REPS):
            t0 = time.perf_counter_ns()
            final_val = run_once()
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)

        times_arr = np.array(times)
        rows.append({
            "library": "geoopt", "optimiser": "rgd", "manifold": "sphere",
            "dim": dim, "median_ns": float(np.median(times_arr)),
            "q1_ns": float(np.percentile(times_arr, 25)),
            "q3_ns": float(np.percentile(times_arr, 75)),
            "final_value": final_val,
            "iterations": 200, "reps": REPS,
        })
        print(f"  geoopt/rgd/sphere/n={dim}: {np.median(times_arr)/1e6:.1f}ms "
              f"(val={final_val:.2e})")
    return rows


# ── geomstats Frechet mean ──────────────────────────────────────────────────

def bench_geomstats_frechet(dims, sample_sizes):
    """Benchmark geomstats FrechetMean on Sphere."""
    from geomstats.geometry.hypersphere import Hypersphere
    from geomstats.learning.frechet_mean import FrechetMean
    rows = []
    for dim in dims:
        m = Hypersphere(dim=dim)
        for k in sample_sizes:
            np.random.seed(SEED)
            points = m.random_point(n_samples=k)

            times = []
            for _ in range(WARMUP):
                fm = FrechetMean(m)
                fm.fit(points)

            for _ in range(REPS):
                t0 = time.perf_counter_ns()
                fm = FrechetMean(m)
                fm.fit(points)
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)

            times_arr = np.array(times)
            rows.append({
                "library": "geomstats", "optimiser": "frechet_mean",
                "manifold": "sphere", "dim": dim, "sample_size": k,
                "median_ns": float(np.median(times_arr)),
                "q1_ns": float(np.percentile(times_arr, 25)),
                "q3_ns": float(np.percentile(times_arr, 75)),
                "iterations": -1, "reps": REPS,
            })
            print(f"  geomstats/frechet/sphere/n={dim}/k={k}: "
                  f"{np.median(times_arr)/1e6:.1f}ms")
    return rows


def main():
    all_rows = []

    print("=== cartan RGD ===")
    all_rows.extend(bench_cartan_rgd(OPT_DIMS))

    print("\n=== cartan RCG ===")
    all_rows.extend(bench_cartan_rcg(OPT_DIMS))

    print("\n=== cartan Frechet mean ===")
    all_rows.extend(bench_cartan_frechet(OPT_DIMS, FRECHET_SAMPLE_SIZES))

    print("\n=== geoopt RGD ===")
    all_rows.extend(bench_geoopt_rgd(OPT_DIMS))

    print("\n=== geomstats Frechet mean ===")
    all_rows.extend(bench_geomstats_frechet(OPT_DIMS, FRECHET_SAMPLE_SIZES))

    # ── Write CSV ────────────────────────────────────────────────────────
    out_path = DATA_DIR / "optimization_timings.csv"
    fields = ["library", "optimiser", "manifold", "dim", "sample_size",
              "median_ns", "q1_ns", "q3_ns", "final_value", "iterations", "reps"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (cartan RGD only, dim=3)**

Run: `cd benchmarks && python -c "
import sys; sys.path.insert(0, 'python')
from bench_optimization import bench_cartan_rgd
rows = bench_cartan_rgd([3])
print('OK:', rows[0]['median_ns']/1e6, 'ms')
"`

Expected: prints timing in milliseconds, no errors.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/python/bench_optimization.py
git commit -m "feat(bench): add optimisation timing harness"
```

---

### Task 4: Python accuracy benchmark harness

**Files:**
- Create: `benchmarks/python/bench_accuracy.py`

- [ ] **Step 1: Create the accuracy benchmark script**

Create `benchmarks/python/bench_accuracy.py`:

```python
"""
Numerical accuracy benchmarks: compare cartan and geomstats
against high-precision mpmath reference values.

Outputs CSV to benchmarks/data/.

Run from benchmarks/:
    python python/bench_accuracy.py
"""

import csv
import pathlib

import numpy as np
import mpmath

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PAIRS = 1000
DIMS = [3, 10, 50]


def sphere_dist_reference(p, q):
    """High-precision sphere distance via mpmath."""
    mpmath.mp.dps = 50
    p_mp = [mpmath.mpf(float(x)) for x in p]
    q_mp = [mpmath.mpf(float(x)) for x in q]
    dot = sum(a * b for a, b in zip(p_mp, q_mp))
    dot = mpmath.mpf(max(-1, min(1, float(dot))))
    return float(mpmath.acos(dot))


def random_sphere_point(n, rng):
    """Sample a uniformly random point on S^{n-1} in R^n."""
    x = rng.standard_normal(n)
    return x / np.linalg.norm(x)


def bench_sphere_dist_accuracy(dims):
    """Compare cartan and geomstats dist() on the sphere against mpmath."""
    import cartan
    from geomstats.geometry.hypersphere import Hypersphere

    rows = []
    rng = np.random.default_rng(SEED)

    for dim in dims:
        ambient = dim + 1
        c_manifold = cartan.Sphere(dim)
        g_manifold = Hypersphere(dim=dim)

        cartan_errors = []
        geomstats_errors = []

        for _ in range(N_PAIRS):
            p = random_sphere_point(ambient, rng)
            q = random_sphere_point(ambient, rng)

            ref = sphere_dist_reference(p, q)
            c_dist = float(c_manifold.dist(p, q))
            g_dist = float(g_manifold.metric.dist(p, q))

            cartan_errors.append(abs(c_dist - ref))
            geomstats_errors.append(abs(g_dist - ref))

        for lib, errors in [("cartan", cartan_errors), ("geomstats", geomstats_errors)]:
            arr = np.array(errors)
            rows.append({
                "library": lib,
                "manifold": "sphere",
                "op": "dist",
                "dim": dim,
                "max_abs_error": float(np.max(arr)),
                "rms_error": float(np.sqrt(np.mean(arr ** 2))),
                "median_abs_error": float(np.median(arr)),
                "n_pairs": N_PAIRS,
            })
            print(f"  {lib}/sphere/dist/n={dim}: max={np.max(arr):.2e}, "
                  f"rms={np.sqrt(np.mean(arr**2)):.2e}")

    return rows


def bench_sphere_self_dist_accuracy(dims):
    """Test dist(p, p) == 0 (the half-chord formula advantage)."""
    import cartan
    from geomstats.geometry.hypersphere import Hypersphere

    rows = []
    rng = np.random.default_rng(SEED + 100)

    for dim in dims:
        ambient = dim + 1
        c_manifold = cartan.Sphere(dim)
        g_manifold = Hypersphere(dim=dim)

        cartan_errors = []
        geomstats_errors = []

        for _ in range(N_PAIRS):
            p = random_sphere_point(ambient, rng)
            cartan_errors.append(abs(float(c_manifold.dist(p, p))))
            geomstats_errors.append(abs(float(g_manifold.metric.dist(p, p))))

        for lib, errors in [("cartan", cartan_errors), ("geomstats", geomstats_errors)]:
            arr = np.array(errors)
            rows.append({
                "library": lib,
                "manifold": "sphere",
                "op": "self_dist",
                "dim": dim,
                "max_abs_error": float(np.max(arr)),
                "rms_error": float(np.sqrt(np.mean(arr ** 2))),
                "median_abs_error": float(np.median(arr)),
                "n_pairs": N_PAIRS,
            })
            print(f"  {lib}/sphere/self_dist/n={dim}: max={np.max(arr):.2e}")

    return rows


def bench_exp_log_roundtrip(dims):
    """Test ||log(p, exp(p, v)) - v|| for roundtrip accuracy."""
    import cartan
    from geomstats.geometry.hypersphere import Hypersphere

    rows = []
    rng = np.random.default_rng(SEED + 200)

    for dim in dims:
        ambient = dim + 1
        c_manifold = cartan.Sphere(dim)
        g_manifold = Hypersphere(dim=dim)

        cartan_errors = []
        geomstats_errors = []

        for _ in range(N_PAIRS):
            p = random_sphere_point(ambient, rng)
            # Small tangent vector (within injectivity radius)
            v_raw = rng.standard_normal(ambient)
            v_raw -= np.dot(v_raw, p) * p  # project to tangent space
            v = v_raw * 0.5 / (np.linalg.norm(v_raw) + 1e-15)

            # cartan roundtrip
            q_c = c_manifold.exp(p, v)
            v_c = c_manifold.log(p, q_c)
            if v_c is not None:
                cartan_errors.append(float(np.linalg.norm(v_c - v)))

            # geomstats roundtrip
            q_g = g_manifold.metric.exp(v, p)
            v_g = g_manifold.metric.log(q_g, p)
            geomstats_errors.append(float(np.linalg.norm(v_g - v)))

        for lib, errors in [("cartan", cartan_errors), ("geomstats", geomstats_errors)]:
            if not errors:
                continue
            arr = np.array(errors)
            rows.append({
                "library": lib,
                "manifold": "sphere",
                "op": "exp_log_roundtrip",
                "dim": dim,
                "max_abs_error": float(np.max(arr)),
                "rms_error": float(np.sqrt(np.mean(arr ** 2))),
                "median_abs_error": float(np.median(arr)),
                "n_pairs": len(errors),
            })
            print(f"  {lib}/sphere/roundtrip/n={dim}: max={np.max(arr):.2e}")

    return rows


def main():
    all_rows = []

    print("=== Sphere dist accuracy ===")
    all_rows.extend(bench_sphere_dist_accuracy(DIMS))

    print("\n=== Sphere self-distance accuracy (p == p) ===")
    all_rows.extend(bench_sphere_self_dist_accuracy(DIMS))

    print("\n=== exp/log roundtrip accuracy ===")
    all_rows.extend(bench_exp_log_roundtrip(DIMS))

    # ── Write CSV ────────────────────────────────────────────────────────
    out_path = DATA_DIR / "accuracy.csv"
    fields = ["library", "manifold", "op", "dim", "max_abs_error",
              "rms_error", "median_abs_error", "n_pairs"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

Run: `cd benchmarks && python -c "
import sys; sys.path.insert(0, 'python')
from bench_accuracy import bench_sphere_self_dist_accuracy
rows = bench_sphere_self_dist_accuracy([3])
for r in rows:
    print(f'{r[\"library\"]}: max_err={r[\"max_abs_error\"]:.2e}')
"`

Expected: cartan max error near machine epsilon (~1e-16), geomstats max error near ~1e-8 (arccos instability).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/python/bench_accuracy.py
git commit -m "feat(bench): add accuracy benchmark harness"
```

---

### Task 5: Rust timing binary

**Files:**
- Create: `benchmarks/rust/Cargo.toml`
- Create: `benchmarks/rust/src/main.rs`

- [ ] **Step 1: Create Cargo.toml**

Create `benchmarks/rust/Cargo.toml`:

```toml
[package]
name = "cartan-bench"
version = "0.1.0"
edition = "2024"
publish = false

[[bin]]
name = "cartan-bench"
path = "src/main.rs"

[dependencies]
cartan-core = { path = "../../cartan-core", features = ["std"] }
cartan-manifolds = { path = "../../cartan-manifolds", features = ["std"] }
cartan-optim = { path = "../../cartan-optim", features = ["std"] }
nalgebra = { version = "0.33", features = ["std"] }
rand = { version = "0.9", features = ["std_rng", "std", "os_rng", "thread_rng"] }
rand_distr = "0.5"
serde_json = "1"
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Create main.rs**

Create `benchmarks/rust/src/main.rs`:

```rust
//! Native Rust timing binary for cartan benchmarks.
//!
//! Outputs JSON lines with median and IQR timings for each
//! (manifold, operation, dimension) combination.
//!
//! ```bash
//! cargo run --release -- --all
//! cargo run --release -- --manifold sphere --dims 2,3,5,10
//! ```

use std::time::Instant;

use clap::Parser;
use nalgebra::SVector;
use rand::SeedableRng;
use rand::rngs::StdRng;

use cartan_core::Manifold;
use cartan_manifolds::Sphere;

const SEED: u64 = 42;
const WARMUP: usize = 5;
const REPS: usize = 200;

#[derive(Parser)]
struct Args {
    /// Run all manifolds and dimensions.
    #[arg(long)]
    all: bool,

    /// Manifold to benchmark (sphere, euclidean).
    #[arg(long, default_value = "sphere")]
    manifold: String,

    /// Comma-separated dimensions.
    #[arg(long, default_value = "2,3,5,10,25,50,100,250,500,1000")]
    dims: String,
}

fn parse_dims(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect()
}

macro_rules! bench_sphere {
    ($n:expr, $dims_filter:expr) => {{
        const N: usize = $n;
        if $dims_filter.contains(&(N - 1)) {
            let manifold = Sphere::<N>;
            let mut rng = StdRng::seed_from_u64(SEED);

            let p = manifold.random_point(&mut rng);
            let q = manifold.random_point(&mut rng);
            let v = manifold.random_tangent(&p, &mut rng);

            // exp
            let times = time_op(|| { manifold.exp(&p, &v); });
            emit("sphere", "exp", N - 1, &times);

            // log
            let times = time_op(|| { let _ = manifold.log(&p, &q); });
            emit("sphere", "log", N - 1, &times);

            // dist
            let times = time_op(|| { let _ = manifold.dist(&p, &q); });
            emit("sphere", "dist", N - 1, &times);

            // parallel_transport
            let times = time_op(|| { manifold.parallel_transport(&p, &q, &v); });
            emit("sphere", "parallel_transport", N - 1, &times);
        }
    }};
}

fn time_op<F: FnMut()>(mut f: F) -> Vec<u128> {
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed().as_nanos();
        times.push(elapsed);
    }
    times
}

fn emit(manifold: &str, op: &str, dim: usize, times: &[u128]) {
    let mut sorted = times.to_vec();
    sorted.sort();
    let n = sorted.len();
    let median = sorted[n / 2];
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    println!(
        "{}",
        serde_json::json!({
            "manifold": manifold,
            "op": op,
            "dim": dim,
            "median_ns": median,
            "q1_ns": q1,
            "q3_ns": q3,
        })
    );
}

fn bench_spheres(dims: &[usize]) {
    // Const-generic dimensions must be known at compile time.
    // We enumerate the supported sizes explicitly.
    bench_sphere!(3, dims);    // S^2
    bench_sphere!(4, dims);    // S^3
    bench_sphere!(6, dims);    // S^5
    bench_sphere!(11, dims);   // S^10
    bench_sphere!(26, dims);   // S^25
    bench_sphere!(51, dims);   // S^50
    bench_sphere!(101, dims);  // S^100
}

fn main() {
    let args = Args::parse();
    let dims = parse_dims(&args.dims);

    if args.all || args.manifold == "sphere" {
        bench_spheres(&dims);
    }
}
```

- [ ] **Step 3: Verify it compiles and runs**

Run: `cd benchmarks/rust && cargo run --release -- --manifold sphere --dims 2,3,5 2>&1 | head -12`

Expected: JSON lines with nanosecond timings for sphere exp/log/dist/transport at dims 2, 3, 5.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/rust/Cargo.toml benchmarks/rust/src/main.rs
git commit -m "feat(bench): add Rust native timing binary"
```

---

### Task 6: Geometry figure generation

**Files:**
- Create: `benchmarks/figures/plot_geometry.py`

- [ ] **Step 1: Create the geometry plot script**

Create `benchmarks/figures/plot_geometry.py`:

```python
"""
Generate geometry performance figures for cartan.sotofranco.dev/performance/geometry.

Reads: data/geometry_timings.csv, data/rust_timings.jsonl
Writes: figures/out/geom_*_{light,dark}.{png,webp}

Run from benchmarks/:
    python figures/plot_geometry.py
"""

import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theme import RC_BASE, THEMES, rc_context, apply_theme, save

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MANIFOLD_LABELS = {
    "sphere": r"$S^n$",
    "spd": r"$\mathrm{SPD}(n)$",
    "so": r"$\mathrm{SO}(n)$",
    "grassmann": r"$\mathrm{Gr}(n, \lfloor n/2 \rfloor)$",
}
OP_LABELS = {
    "exp": r"$\exp_p(v)$",
    "log": r"$\log_p(q)$",
    "dist": r"$d(p, q)$",
    "parallel_transport": r"$\Gamma_{p \to q}(v)$",
}
MANIFOLD_ORDER = ["sphere", "spd", "so", "grassmann"]


def load_python_timings():
    """Load geometry_timings.csv into nested dict."""
    path = DATA_DIR / "geometry_timings.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["library"], row["manifold"], row["op"])
            if key not in data:
                data[key] = {"dims": [], "medians": [], "q1s": [], "q3s": []}
            data[key]["dims"].append(int(row["dim"]))
            data[key]["medians"].append(float(row["median_ns"]))
            data[key]["q1s"].append(float(row["q1_ns"]))
            data[key]["q3s"].append(float(row["q3_ns"]))
    return data


def load_rust_timings():
    """Load rust_timings.jsonl into nested dict."""
    path = DATA_DIR / "rust_timings.jsonl"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = ("rust", row["manifold"], row["op"])
            if key not in data:
                data[key] = {"dims": [], "medians": [], "q1s": [], "q3s": []}
            data[key]["dims"].append(row["dim"])
            data[key]["medians"].append(row["median_ns"])
            data[key]["q1s"].append(row["q1_ns"])
            data[key]["q3s"].append(row["q3_ns"])
    return data


def plot_dimension_sweep(op, py_data, rust_data, theme_name, theme):
    """Generate a 2x2 small-multiples figure for one operation."""
    with rc_context(theme):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=False, sharey=False)
        axes_flat = axes.flatten()

        for i, mname in enumerate(MANIFOLD_ORDER):
            ax = axes_flat[i]
            ax.set_title(MANIFOLD_LABELS[mname], fontsize=10, pad=6)

            # geomstats
            key_g = ("geomstats", mname, op)
            if key_g in py_data:
                d = py_data[key_g]
                dims, med, q1, q3 = np.array(d["dims"]), np.array(d["medians"]), np.array(d["q1s"]), np.array(d["q3s"])
                ax.fill_between(dims, q1 / 1e3, q3 / 1e3, alpha=0.15, color=theme["blue"])
                ax.plot(dims, med / 1e3, color=theme["blue"], lw=1.5, label="geomstats")

            # cartan python
            key_c = ("cartan", mname, op)
            if key_c in py_data:
                d = py_data[key_c]
                dims, med, q1, q3 = np.array(d["dims"]), np.array(d["medians"]), np.array(d["q1s"]), np.array(d["q3s"])
                ax.fill_between(dims, q1 / 1e3, q3 / 1e3, alpha=0.15, color=theme["accent"])
                ax.plot(dims, med / 1e3, color=theme["accent"], lw=1.5, label="cartan (Python)")

            # cartan rust native
            key_r = ("rust", mname, op)
            if key_r in rust_data:
                d = rust_data[key_r]
                dims, med = np.array(d["dims"]), np.array(d["medians"])
                ax.plot(dims, med / 1e3, color=theme["accent"], lw=1.2,
                        ls="--", label="cartan (Rust)")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Dimension $n$", fontsize=8)
            ax.set_ylabel(r"Time ($\mu$s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(False)

        # Common legend on first subplot
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper centre", ncol=3,
                       fontsize=7.5, frameon=False,
                       bbox_to_anchor=(0.5, 1.02))

        apply_theme(fig, axes_flat, theme)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, f"geom_{op}_sweep", theme_name)
        plt.close(fig)


def plot_speedup_heatmap(py_data, theme_name, theme):
    """Generate a speedup heatmap (cartan-py / geomstats) at n=10."""
    ops = ["exp", "log", "dist", "parallel_transport"]
    target_dim = 10

    speedups = np.full((len(MANIFOLD_ORDER), len(ops)), np.nan)
    for i, mname in enumerate(MANIFOLD_ORDER):
        for j, op in enumerate(ops):
            key_c = ("cartan", mname, op)
            key_g = ("geomstats", mname, op)
            if key_c in py_data and key_g in py_data:
                c_data = py_data[key_c]
                g_data = py_data[key_g]
                # Find the entry closest to target_dim
                for ci, cd in enumerate(c_data["dims"]):
                    if cd == target_dim:
                        c_med = c_data["medians"][ci]
                        break
                else:
                    continue
                for gi, gd in enumerate(g_data["dims"]):
                    if gd == target_dim:
                        g_med = g_data["medians"][gi]
                        break
                else:
                    continue
                if c_med > 0:
                    speedups[i, j] = g_med / c_med

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(speedups, cmap="YlOrBr", aspect="auto", vmin=1)

        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels([OP_LABELS[o] for o in ops], fontsize=8)
        ax.set_yticks(range(len(MANIFOLD_ORDER)))
        ax.set_yticklabels([MANIFOLD_LABELS[m] for m in MANIFOLD_ORDER], fontsize=8)

        # Annotate cells with speedup values
        for i in range(len(MANIFOLD_ORDER)):
            for j in range(len(ops)):
                val = speedups[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}x", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="#111111" if val > 5 else theme["fg"])

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Speedup factor", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(r"Speedup: cartan (Python) vs geomstats at $n = 10$",
                     fontsize=9, pad=8)
        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "geom_speedup_heatmap", theme_name)
        plt.close(fig)


def main():
    py_data = load_python_timings()
    rust_data = load_rust_timings()

    for theme_name, theme in THEMES.items():
        print(f"\n=== {theme_name} theme ===")
        for op in ["exp", "log", "dist", "parallel_transport"]:
            plot_dimension_sweep(op, py_data, rust_data, theme_name, theme)
        plot_speedup_heatmap(py_data, theme_name, theme)

    print("\nDone. Figures in figures/out/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs (will need data first; this is tested after Task 2 data is generated)**

Run: `cd benchmarks && python python/bench_geometry.py 2>&1 | tail -5` then `python figures/plot_geometry.py`

Expected: figures written to `figures/out/geom_*_{light,dark}.{png,webp}`.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/figures/plot_geometry.py
git commit -m "feat(bench): add geometry figure generation script"
```

---

### Task 7: Optimisation and accuracy figure generation

**Files:**
- Create: `benchmarks/figures/plot_optimization.py`
- Create: `benchmarks/figures/plot_accuracy.py`

- [ ] **Step 1: Create the optimisation plot script**

Create `benchmarks/figures/plot_optimization.py`:

```python
"""
Generate optimisation performance figures for cartan.sotofranco.dev/performance/optimisation.

Reads: data/optimization_timings.csv
Writes: figures/out/optim_*_{light,dark}.{png,webp}

Run from benchmarks/:
    python figures/plot_optimization.py
"""

import csv
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theme import THEMES, rc_context, apply_theme, save


DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def load_timings():
    """Load optimization_timings.csv into a list of dicts."""
    path = DATA_DIR / "optimization_timings.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["dim"] = int(row["dim"])
            row["median_ns"] = float(row["median_ns"])
            row["q1_ns"] = float(row["q1_ns"])
            row["q3_ns"] = float(row["q3_ns"])
            row["reps"] = int(row["reps"])
            rows.append(row)
    return rows


def plot_rgd_comparison(rows, theme_name, theme):
    """RGD wall-clock: cartan vs geoopt across dimensions."""
    cartan_rgd = [r for r in rows if r["library"] == "cartan" and r["optimiser"] == "rgd"]
    geoopt_rgd = [r for r in rows if r["library"] == "geoopt" and r["optimiser"] == "rgd"]

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        if geoopt_rgd:
            dims = [r["dim"] for r in geoopt_rgd]
            med = [r["median_ns"] / 1e6 for r in geoopt_rgd]
            q1 = [r["q1_ns"] / 1e6 for r in geoopt_rgd]
            q3 = [r["q3_ns"] / 1e6 for r in geoopt_rgd]
            ax.fill_between(dims, q1, q3, alpha=0.15, color=theme["blue"])
            ax.plot(dims, med, "o-", color=theme["blue"], lw=1.5, ms=4,
                    label="geoopt (RiemannianSGD)")

        if cartan_rgd:
            dims = [r["dim"] for r in cartan_rgd]
            med = [r["median_ns"] / 1e6 for r in cartan_rgd]
            q1 = [r["q1_ns"] / 1e6 for r in cartan_rgd]
            q3 = [r["q3_ns"] / 1e6 for r in cartan_rgd]
            ax.fill_between(dims, q1, q3, alpha=0.15, color=theme["accent"])
            ax.plot(dims, med, "o-", color=theme["accent"], lw=1.5, ms=4,
                    label="cartan (RGD)")

        # Also show RCG (cartan-only)
        cartan_rcg = [r for r in rows if r["library"] == "cartan" and r["optimiser"] == "rcg"]
        if cartan_rcg:
            dims = [r["dim"] for r in cartan_rcg]
            med = [r["median_ns"] / 1e6 for r in cartan_rcg]
            ax.plot(dims, med, "s--", color=theme["accent"], lw=1.2, ms=4,
                    alpha=0.7, label="cartan (RCG)")

        ax.set_xlabel(r"Dimension $n$", fontsize=9)
        ax.set_ylabel("Wall-clock time (ms)", fontsize=9)
        ax.set_title(r"Optimisation on $S^n$: 200 iterations", fontsize=10, pad=8)
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_yscale("log")
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "optim_rgd_comparison", theme_name)
        plt.close(fig)


def plot_frechet_comparison(rows, theme_name, theme):
    """Frechet mean wall-clock: cartan vs geomstats across sample sizes at dim=10."""
    target_dim = 10
    cartan_fm = [r for r in rows
                 if r["library"] == "cartan" and r["optimiser"] == "frechet_mean"
                 and r["dim"] == target_dim]
    geomstats_fm = [r for r in rows
                    if r["library"] == "geomstats" and r["optimiser"] == "frechet_mean"
                    and r["dim"] == target_dim]

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        if geomstats_fm:
            ks = [int(r.get("sample_size", 0)) for r in geomstats_fm]
            med = [r["median_ns"] / 1e6 for r in geomstats_fm]
            q1 = [r["q1_ns"] / 1e6 for r in geomstats_fm]
            q3 = [r["q3_ns"] / 1e6 for r in geomstats_fm]
            ax.fill_between(ks, q1, q3, alpha=0.15, color=theme["blue"])
            ax.plot(ks, med, "o-", color=theme["blue"], lw=1.5, ms=4,
                    label="geomstats")

        if cartan_fm:
            ks = [int(r.get("sample_size", 0)) for r in cartan_fm]
            med = [r["median_ns"] / 1e6 for r in cartan_fm]
            q1 = [r["q1_ns"] / 1e6 for r in cartan_fm]
            q3 = [r["q3_ns"] / 1e6 for r in cartan_fm]
            ax.fill_between(ks, q1, q3, alpha=0.15, color=theme["accent"])
            ax.plot(ks, med, "o-", color=theme["accent"], lw=1.5, ms=4,
                    label="cartan")

        ax.set_xlabel("Sample size $K$", fontsize=9)
        ax.set_ylabel("Wall-clock time (ms)", fontsize=9)
        ax.set_title(r"Fr\'echet mean on $S^{10}$", fontsize=10, pad=8)
        ax.legend(fontsize=7.5, frameon=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "optim_frechet_comparison", theme_name)
        plt.close(fig)


def main():
    rows = load_timings()

    for theme_name, theme in THEMES.items():
        print(f"\n=== {theme_name} theme ===")
        plot_rgd_comparison(rows, theme_name, theme)
        plot_frechet_comparison(rows, theme_name, theme)

    print("\nDone. Figures in figures/out/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the accuracy plot script**

Create `benchmarks/figures/plot_accuracy.py`:

```python
"""
Generate accuracy figures for cartan.sotofranco.dev/performance/geometry.

Reads: data/accuracy.csv
Writes: figures/out/accuracy_*_{light,dark}.{png,webp}

Run from benchmarks/:
    python figures/plot_accuracy.py
"""

import csv
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theme import THEMES, rc_context, apply_theme, save

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def load_accuracy():
    path = DATA_DIR / "accuracy.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["dim"] = int(row["dim"])
            row["max_abs_error"] = float(row["max_abs_error"])
            row["rms_error"] = float(row["rms_error"])
            rows.append(row)
    return rows


def plot_dist_accuracy(rows, theme_name, theme):
    """Bar chart: max absolute error for dist() on the sphere."""
    dist_rows = [r for r in rows if r["op"] == "dist"]
    dims = sorted(set(r["dim"] for r in dist_rows))

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        bar_width = 0.35
        x = np.arange(len(dims))

        cartan_vals = []
        geomstats_vals = []
        for dim in dims:
            c = [r for r in dist_rows if r["library"] == "cartan" and r["dim"] == dim]
            g = [r for r in dist_rows if r["library"] == "geomstats" and r["dim"] == dim]
            cartan_vals.append(c[0]["max_abs_error"] if c else 0)
            geomstats_vals.append(g[0]["max_abs_error"] if g else 0)

        ax.bar(x - bar_width / 2, cartan_vals, bar_width,
               color=theme["accent"], label="cartan", zorder=3)
        ax.bar(x + bar_width / 2, geomstats_vals, bar_width,
               color=theme["blue"], label="geomstats", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$n = {d}$" for d in dims], fontsize=8)
        ax.set_ylabel("Maximum absolute error", fontsize=9)
        ax.set_title(r"Numerical accuracy: $d(p, q)$ on $S^n$", fontsize=10, pad=8)
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "accuracy_dist", theme_name)
        plt.close(fig)


def plot_self_dist_accuracy(rows, theme_name, theme):
    """Bar chart: max |dist(p, p)| (should be 0)."""
    self_rows = [r for r in rows if r["op"] == "self_dist"]
    dims = sorted(set(r["dim"] for r in self_rows))

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        bar_width = 0.35
        x = np.arange(len(dims))

        cartan_vals = []
        geomstats_vals = []
        for dim in dims:
            c = [r for r in self_rows if r["library"] == "cartan" and r["dim"] == dim]
            g = [r for r in self_rows if r["library"] == "geomstats" and r["dim"] == dim]
            cartan_vals.append(c[0]["max_abs_error"] if c else 0)
            geomstats_vals.append(g[0]["max_abs_error"] if g else 0)

        ax.bar(x - bar_width / 2, cartan_vals, bar_width,
               color=theme["accent"], label="cartan", zorder=3)
        ax.bar(x + bar_width / 2, geomstats_vals, bar_width,
               color=theme["blue"], label="geomstats", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$n = {d}$" for d in dims], fontsize=8)
        ax.set_ylabel("Maximum $|d(p, p)|$", fontsize=9)
        ax.set_title(r"Self-distance accuracy: $d(p, p) = 0$ on $S^n$",
                     fontsize=10, pad=8)
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "accuracy_self_dist", theme_name)
        plt.close(fig)


def main():
    rows = load_accuracy()

    for theme_name, theme in THEMES.items():
        print(f"\n=== {theme_name} theme ===")
        plot_dist_accuracy(rows, theme_name, theme)
        plot_self_dist_accuracy(rows, theme_name, theme)

    print("\nDone. Figures in figures/out/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/figures/plot_optimization.py benchmarks/figures/plot_accuracy.py
git commit -m "feat(bench): add optimisation and accuracy figure scripts"
```

---

### Task 8: Run all benchmarks and generate figures

**Files:**
- No new files; this task runs the scripts from Tasks 2-7.

- [ ] **Step 1: Run geometry benchmarks**

Run: `cd benchmarks && python python/bench_geometry.py 2>&1 | tee /tmp/bench_geom.log | tail -20`

Expected: CSV written to `data/geometry_timings.csv` with rows for cartan and geomstats across all manifolds and dimensions.

- [ ] **Step 2: Run optimisation benchmarks**

Run: `cd benchmarks && python python/bench_optimization.py 2>&1 | tee /tmp/bench_optim.log | tail -20`

Expected: CSV written to `data/optimization_timings.csv`.

- [ ] **Step 3: Run accuracy benchmarks**

Run: `cd benchmarks && python python/bench_accuracy.py 2>&1 | tee /tmp/bench_accuracy.log | tail -20`

Expected: CSV written to `data/accuracy.csv`.

- [ ] **Step 4: Run Rust native timing**

Run: `cd benchmarks/rust && cargo run --release -- --all > ../data/rust_timings.jsonl 2>&1`

Expected: JSON lines written to `data/rust_timings.jsonl`.

- [ ] **Step 5: Generate all figures**

Run:
```bash
cd benchmarks
python figures/plot_geometry.py
python figures/plot_optimization.py
python figures/plot_accuracy.py
```

Expected: figures in `figures/out/` (PNG + WebP, light + dark variants).

- [ ] **Step 6: Verify figure output**

Run: `ls -la benchmarks/figures/out/*.webp | wc -l`

Expected: at least 14 webp files (4 sweep x 2 themes + 1 heatmap x 2 + 2 optim x 2 + 2 accuracy x 2 = 18).

- [ ] **Step 7: No commit (data and figures are gitignored)**

---

### Task 9: cartan-docs sidebar and layout

**Files:**
- Modify: `~/cartan-docs/lib/sidebar-config.ts:69-79`
- Create: `~/cartan-docs/app/performance/layout.tsx`

- [ ] **Step 1: Add Performance section to sidebar**

In `~/cartan-docs/lib/sidebar-config.ts`, add the Performance section between DEC and Demos:

```typescript
  {
    title: 'Performance',
    href: '/performance',
    children: [
      { title: 'Geometry',      href: '/performance/geometry' },
      { title: 'Optimisation',  href: '/performance/optimisation' },
    ],
  },
```

Insert this block after the DEC section (after line 69) and before the Demos section.

- [ ] **Step 2: Create layout.tsx**

Create `~/cartan-docs/app/performance/layout.tsx`:

```tsx
import DocsLayout from '../docs-layout'
import CounterResetter from '../components/CounterResetter'

export default function PerformanceLayout({ children }: { children: React.ReactNode }) {
  return (
    <DocsLayout>
      <CounterResetter />
      {children}
    </DocsLayout>
  )
}
```

- [ ] **Step 3: Verify build**

Run: `cd ~/cartan-docs && npm run build 2>&1 | tail -5`

Expected: clean build, no errors. `/performance` route should appear.

- [ ] **Step 4: Commit**

```bash
cd ~/cartan-docs
git add lib/sidebar-config.ts app/performance/layout.tsx
git commit -m "feat: add Performance section to sidebar and layout"
```

---

### Task 10: Copy figures and create public directory

**Files:**
- Create: `~/cartan-docs/public/performance/` (directory)

- [ ] **Step 1: Create the public directory and copy figures**

```bash
mkdir -p ~/cartan-docs/public/performance
cp ~/cartan/.claude/worktrees/cartan-py-bindings/benchmarks/figures/out/*.webp ~/cartan-docs/public/performance/
```

- [ ] **Step 2: Verify figures are in place**

Run: `ls ~/cartan-docs/public/performance/*.webp | head -10`

Expected: lists the webp files (light + dark variants).

- [ ] **Step 3: Commit figures**

```bash
cd ~/cartan-docs
git add public/performance/
git commit -m "feat: add performance benchmark figures"
```

---

### Task 11: Performance overview page

**Files:**
- Create: `~/cartan-docs/app/performance/page.mdx`

- [ ] **Step 1: Create the overview page**

Create `~/cartan-docs/app/performance/page.mdx`:

```mdx
# Performance

cartan is a Rust-native Riemannian geometry library with Python bindings via PyO3.
This section presents systematic benchmarks comparing cartan against
[geomstats](https://geomstats.github.io/) and [geoopt](https://geoopt.readthedocs.io/)
across manifold operations and optimisation algorithms.

## Headline Results

At dimension $n = 10$ on the 2-sphere, cartan's Python bindings are:

- **Nx faster** than geomstats for the exponential map
- **Nx faster** for the logarithmic map
- **Nx faster** for geodesic distance

*(Replace N with actual measured values after running benchmarks.)*

The native Rust library adds a further constant-factor improvement by eliminating
the Python FFI boundary.

## Methodology

All benchmarks use:

- **200 repetitions** for geometry microbenchmarks, **20** for optimisation
- **Median + interquartile range** (robust to GC pauses and JIT warmup)
- **5 warmup calls** before timing begins
- **Fixed random seed** (42) across all libraries
- **CPU-only** (no GPU backends)

Full reproduction instructions are in the
[benchmarks directory](https://github.com/alejandro-soto-franco/cartan/tree/main/benchmarks).

## Sections

<div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.5rem' }}>
  {[
    { href: '/performance/geometry',      label: 'Geometry',       desc: 'exp, log, dist, parallel transport' },
    { href: '/performance/optimisation',  label: 'Optimisation',   desc: 'RGD, RCG, Frechet mean' },
  ].map(({ href, label, desc }) => (
    <a key={href} href={href} style={{
      display: 'block', padding: '0.875rem 1.1rem',
      border: '1px solid var(--color-border)', borderRadius: '6px',
      textDecoration: 'none', minWidth: '180px', flex: '1 1 180px',
    }}>
      <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--color-accent)', marginBottom: '0.2rem' }}>
        {label} →
      </div>
      <div style={{ fontSize: '0.78rem', color: 'var(--color-text-disabled)' }}>{desc}</div>
    </a>
  ))}
</div>
```

- [ ] **Step 2: Verify build**

Run: `cd ~/cartan-docs && npm run build 2>&1 | tail -5`

Expected: clean build, `/performance` page renders.

- [ ] **Step 3: Commit**

```bash
cd ~/cartan-docs
git add app/performance/page.mdx
git commit -m "feat: add performance overview page"
```

---

### Task 12: Geometry benchmarks page

**Files:**
- Create: `~/cartan-docs/app/performance/geometry/page.mdx`

- [ ] **Step 1: Create the geometry page**

Create `~/cartan-docs/app/performance/geometry/page.mdx`:

```mdx
import BlogFigure from '../../components/BlogFigure'

# Geometry Benchmarks

Wall-clock timing for core manifold operations across cartan (Python bindings and
native Rust), geomstats (NumPy backend), and geoopt. All times are median over
200 repetitions with 5 warmup calls.

## Dimension Sweeps

Each figure shows wall-clock time vs ambient dimension on a log-log scale.
Shaded regions represent the interquartile range. The dashed gold line is
cartan's native Rust performance (no Python overhead).

### Exponential Map

<BlogFigure
  light="/performance/geom_exp_sweep_light.webp"
  dark="/performance/geom_exp_sweep_dark.webp"
  alt="Exponential map wall-clock time vs dimension across four manifolds"
  caption="Wall-clock time for a single exp(p, v) call. cartan's const-generic stack allocation dominates at small dimensions; the gap narrows as BLAS takes over at large n."
/>

### Logarithmic Map

<BlogFigure
  light="/performance/geom_log_sweep_light.webp"
  dark="/performance/geom_log_sweep_dark.webp"
  alt="Logarithmic map wall-clock time vs dimension across four manifolds"
  caption="Wall-clock time for a single log(p, q) call."
/>

### Geodesic Distance

<BlogFigure
  light="/performance/geom_dist_sweep_light.webp"
  dark="/performance/geom_dist_sweep_dark.webp"
  alt="Geodesic distance wall-clock time vs dimension across four manifolds"
  caption="Wall-clock time for a single dist(p, q) call. On the sphere, cartan uses the numerically stable half-chord formula 2 asin(||p - q|| / 2)."
/>

### Parallel Transport

<BlogFigure
  light="/performance/geom_parallel_transport_sweep_light.webp"
  dark="/performance/geom_parallel_transport_sweep_dark.webp"
  alt="Parallel transport wall-clock time vs dimension across four manifolds"
  caption="Wall-clock time for a single parallel transport call along the geodesic from p to q."
/>

## Speedup Summary

<BlogFigure
  light="/performance/geom_speedup_heatmap_light.webp"
  dark="/performance/geom_speedup_heatmap_dark.webp"
  alt="Speedup heatmap: cartan Python bindings vs geomstats at dimension 10"
  caption="Speedup factor (geomstats median / cartan median) at n = 10. Higher is better for cartan."
/>

## Numerical Accuracy

<BlogFigure
  light="/performance/accuracy_dist_light.webp"
  dark="/performance/accuracy_dist_dark.webp"
  alt="Numerical accuracy of geodesic distance on the sphere"
  caption="Maximum absolute error of dist(p, q) vs a 50-digit mpmath reference over 1000 random point pairs."
/>

<BlogFigure
  light="/performance/accuracy_self_dist_light.webp"
  dark="/performance/accuracy_self_dist_dark.webp"
  alt="Self-distance accuracy: dist(p, p) should equal zero"
  caption="Maximum |dist(p, p)| over 1000 random points. cartan's half-chord formula returns exactly 0; geomstats's arccos formula amplifies floating-point noise."
/>

## Manifold Coverage

cartan provides manifolds with no equivalent in geomstats or geoopt:

| Manifold | Description | geomstats | geoopt |
|----------|-------------|-----------|--------|
| <Eq tex="\mathrm{SE}(N)" /> | Special Euclidean group (rigid motions) | partial | n/a |
| <Eq tex="\mathrm{Corr}(N)" /> | Correlation matrices (flat Frobenius metric) | n/a | n/a |
| <Eq tex="Q_3" /> (QTensor3) | Landau-de Gennes Q-tensor for liquid crystals | n/a | n/a |

## Hardware

*(Fill in after running benchmarks: CPU model, OS, Python version, cartan version, geomstats version, geoopt version.)*
```

- [ ] **Step 2: Verify build**

Run: `cd ~/cartan-docs && npm run build 2>&1 | tail -5`

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
cd ~/cartan-docs
git add app/performance/geometry/page.mdx
git commit -m "feat: add geometry benchmarks page"
```

---

### Task 13: Optimisation benchmarks page

**Files:**
- Create: `~/cartan-docs/app/performance/optimisation/page.mdx`

- [ ] **Step 1: Create the optimisation page**

Create `~/cartan-docs/app/performance/optimisation/page.mdx`:

```mdx
import BlogFigure from '../../components/BlogFigure'

# Optimisation Benchmarks

Wall-clock timing for Riemannian optimisation algorithms. cartan is compared
against geoopt (`RiemannianSGD`) for gradient descent and against geomstats
for the Frechet mean estimator. All times are median over 20 repetitions.

## Riemannian Gradient Descent

<BlogFigure
  light="/performance/optim_rgd_comparison_light.webp"
  dark="/performance/optim_rgd_comparison_dark.webp"
  alt="RGD wall-clock time: cartan vs geoopt across dimensions on the sphere"
  caption="Wall-clock time for 200 iterations of Riemannian gradient descent on the sphere, minimising the squared geodesic distance to a target point. cartan's RCG (dashed) shows the convergence advantage of conjugate gradient."
/>

cartan provides three Riemannian optimisers:

| Optimiser | cartan | geoopt equivalent |
|-----------|--------|-------------------|
| RGD (gradient descent) | `minimize_rgd` | `RiemannianSGD` |
| RCG (conjugate gradient) | `minimize_rcg` | n/a |
| RTR (trust region) | `minimize_rtr` | n/a |

RCG and RTR are cartan-only; geoopt has no second-order Riemannian optimiser.

## Frechet Mean

<BlogFigure
  light="/performance/optim_frechet_comparison_light.webp"
  dark="/performance/optim_frechet_comparison_dark.webp"
  alt="Frechet mean wall-clock time: cartan vs geomstats across sample sizes"
  caption="Wall-clock time for computing the Frechet mean of K random points on the 10-sphere. cartan uses an iterative gradient scheme; geomstats uses its built-in FrechetMean estimator."
/>

## Hardware

*(Fill in after running benchmarks: CPU model, OS, Python version, cartan version, geomstats version, geoopt version.)*
```

- [ ] **Step 2: Verify build**

Run: `cd ~/cartan-docs && npm run build 2>&1 | tail -5`

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
cd ~/cartan-docs
git add app/performance/optimisation/page.mdx
git commit -m "feat: add optimisation benchmarks page"
```

---

### Task 14: Final integration and push

**Files:**
- No new files. This task verifies the full pipeline and pushes.

- [ ] **Step 1: Verify cartan-docs builds clean**

Run: `cd ~/cartan-docs && npm run build 2>&1 | tail -10`

Expected: clean build with `/performance`, `/performance/geometry`, `/performance/optimisation` in the output.

- [ ] **Step 2: Push cartan-docs**

```bash
cd ~/cartan-docs && git push
```

- [ ] **Step 3: Push cartan benchmarks**

```bash
cd ~/cartan/.claude/worktrees/cartan-py-bindings
git add benchmarks/
git commit -m "feat: add performance benchmark suite"
git push origin main
```

- [ ] **Step 4: Verify deployment**

After Vercel deploys, check:
- https://cartan.sotofranco.dev/performance
- https://cartan.sotofranco.dev/performance/geometry
- https://cartan.sotofranco.dev/performance/optimisation

All three pages should render with figures switching correctly between light and dark themes.
