"""geomstats and geoopt side of the cross-language comparison.

Reads the shared fixtures, computes the same operations as the Julia and Rust
harnesses, and writes values alongside timings. Values first: comparing the
speed of implementations that disagree tells you nothing.

Argument order differs between the libraries and is a live source of silent
error, so it is spelled out at each call site:

    cartan / Manifolds.jl   exp(p, v)              base point first
    geomstats               exp(tangent_vec, base) tangent first
    geoopt                  expmap(x, u)           base point first

Run:
    .venv/bin/python python/bench_crosslang.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures" / "geometry_cases.json"
OUT = ROOT / "results" / "python_geometry.jsonl"

# Enough repetitions that a median is stable, without making the SPD cases
# take minutes. The Julia and Rust sides use adaptive samplers; this one is
# fixed, so it is the least precise of the three and the report says so.
WARMUP = 20
REPS = 300


def time_ns(fn, *args) -> tuple[float, float, float]:
    """Median and interquartile range of `fn(*args)`, in nanoseconds."""
    for _ in range(WARMUP):
        fn(*args)
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        fn(*args)
        samples.append(time.perf_counter_ns() - t0)
    a = np.asarray(samples, dtype=float)
    return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))


def flat(x) -> list[float]:
    """Flatten a value to a plain list, whatever array type produced it."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).ravel().tolist()


def record(lib, case, op, value, timing) -> dict:
    med, q1, q3 = timing
    return {
        "lib": lib,
        "manifold": case["manifold"],
        "dim": case["dim"],
        "op": op,
        "value": flat(value),
        "median_ns": med,
        "q1_ns": q1,
        "q3_ns": q3,
    }


def run_geomstats(case) -> list[dict]:
    import geomstats.backend as gs

    kind, dim = case["manifold"], case["dim"]
    out = []

    if kind == "sphere":
        from geomstats.geometry.hypersphere import Hypersphere

        # geomstats Hypersphere(dim=d) is the unit sphere in R^(d+1).
        m = Hypersphere(dim=dim - 1)
        p = gs.array(case["p"])
        v = gs.array(case["v"])
        q = gs.array(case["q"])
        metric = m.metric

        out.append(record("geomstats", case, "exp", metric.exp(v, p),
                          time_ns(metric.exp, v, p)))
        out.append(record("geomstats", case, "log", metric.log(q, p),
                          time_ns(metric.log, q, p)))
        out.append(record("geomstats", case, "dist", [metric.dist(p, q)],
                          time_ns(metric.dist, p, q)))

        # Exact parallel transport, for comparison against cartan and
        # Manifolds.jl. geoopt deliberately does something else; see below.
        def gs_transport():
            return metric.parallel_transport(v, p, end_point=q)

        out.append(record("geomstats", case, "transport", gs_transport(),
                          time_ns(gs_transport)))
    elif kind == "spd":
        from geomstats.geometry.spd_matrices import SPDMatrices

        m = SPDMatrices(dim)
        p = gs.array(case["p"])
        v = gs.array(case["v"])
        q = gs.array(case["q"])
        metric = m.metric

        out.append(record("geomstats", case, "exp", metric.exp(v, p),
                          time_ns(metric.exp, v, p)))
        out.append(record("geomstats", case, "log", metric.log(q, p),
                          time_ns(metric.log, q, p)))
        out.append(record("geomstats", case, "dist", [metric.dist(p, q)],
                          time_ns(metric.dist, p, q)))
    return out


def run_geoopt(case) -> list[dict]:
    """geoopt covers the sphere only.

    Its SPD support is a different parameterisation, so including it would
    compare conventions rather than implementations.
    """
    if case["manifold"] != "sphere":
        return []

    import geoopt
    import torch

    m = geoopt.Sphere()
    p = torch.tensor(case["p"], dtype=torch.float64)
    v = torch.tensor(case["v"], dtype=torch.float64)
    q = torch.tensor(case["q"], dtype=torch.float64)

    out = [
        record("geoopt", case, "exp", m.expmap(p, v), time_ns(m.expmap, p, v)),
        record("geoopt", case, "log", m.logmap(p, q), time_ns(m.logmap, p, q)),
        record("geoopt", case, "dist", [m.dist(p, q)], time_ns(m.dist, p, q)),
        record("geoopt", case, "transport", m.transp(p, q, v),
               time_ns(m.transp, p, q, v)),
        # geoopt implements a projection-based vector transport, not exact
        # parallel transport. Emitting the bare projection lets the comparison
        # verify that classification instead of trusting a comment: if geoopt
        # ever switches to exact transport, the check fails loudly.
        record("geoopt", case, "transport_projection", m.proju(q, v),
               time_ns(m.proju, q, v)),
    ]
    return out


def main() -> None:
    data = json.loads(FIXTURES.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for case in data["cases"]:
        print(f"benchmarking {case['manifold']} dim={case['dim']}")
        for runner in (run_geomstats, run_geoopt):
            try:
                records.extend(runner(case))
            except Exception as exc:  # noqa: BLE001
                # A comparator that cannot express a case is reported and
                # skipped, never silently dropped: a missing row in the final
                # table must be distinguishable from a row nobody tried.
                print(f"  {runner.__name__} skipped {case['manifold']}"
                      f" dim={case['dim']}: {type(exc).__name__}: {exc}")

    with OUT.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
