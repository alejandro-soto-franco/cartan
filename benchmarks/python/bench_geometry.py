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
