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
OPT_DIMS = [3, 5, 7, 9]


# ── Test problems ───────────────────────────────────────────────────────────

def sphere_cost_grad_cartan(manifold, target):
    """Minimise f(p) = dist(p, target)^2 on the sphere."""
    def cost(p):
        d = manifold.dist(p, target)
        return float(d * d)

    def grad(p):
        v = manifold.log(p, target)
        if v is None:
            return np.zeros_like(p)
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
