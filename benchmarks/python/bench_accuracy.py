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

    # Write CSV
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
