#!/usr/bin/env python3
"""Homogenisation benchmark: time ECHOES across the same (scheme × shape × fraction)
sweep as cartan-bench-homog, emit JSON-line results.

Usage:
    conda activate echoes-homog
    python bench_homog.py --out ../results/homog_echoes.jsonl

Pair with `rust/src/homog_main.rs` via `plot_homog.py` for head-to-head figures.
"""
from __future__ import annotations
import argparse, json, time, pathlib

import numpy as np
from echoes import (
    rve, ellipsoid, spherical, spheroidal, stiff_kmu, homogenize,
    VOIGT, REUSS, DIL, DILD, MT, SC, ASC, MAX, PCW, DIFF, ISO,
    tId2,
)

SCHEME_MAP = {
    "VOIGT": VOIGT, "REUSS": REUSS, "DIL": DIL, "DILD": DILD,
    "MT": MT, "SC": SC, "ASC": ASC, "MAX": MAX, "PCW": PCW, "DIFF": DIFF,
}

WARMUP = 3
REPS = 50


def time_call(fn, *args, **kwargs) -> tuple[float, np.ndarray]:
    """Run fn WARMUP + REPS times; return (median_ns, last_result)."""
    for _ in range(WARMUP):
        fn(*args, **kwargs)
    times = []
    last = None
    for _ in range(REPS):
        t0 = time.perf_counter_ns()
        last = fn(*args, **kwargs)
        times.append(time.perf_counter_ns() - t0)
    return float(np.median(times)), np.asarray(last.array)


def bench_o2_sphere(scheme: str, phi: float):
    k_matrix = 1.0
    k_incl = 5.0
    def run():
        r = rve(matrix="M")
        r["M"] = ellipsoid(shape=spherical, prop={"K": k_matrix * tId2}, fraction=1.0 - phi)
        r["I"] = ellipsoid(shape=spherical, prop={"K": k_incl * tId2}, fraction=phi)
        return homogenize(prop="K", rve=r, scheme=SCHEME_MAP[scheme], verbose=False,
                          maxnb=500, epsrel=1.0e-10)
    return time_call(run)


def bench_o2_spheroid(scheme: str, phi: float, aspect: float):
    def run():
        r = rve(matrix="M")
        r["M"] = ellipsoid(shape=spherical, prop={"K": 1.0 * tId2}, fraction=1.0 - phi)
        r["I"] = ellipsoid(shape=spheroidal(aspect), prop={"K": 5.0 * tId2}, fraction=phi)
        return homogenize(prop="K", rve=r, scheme=SCHEME_MAP[scheme], verbose=False,
                          maxnb=500, epsrel=1.0e-10)
    return time_call(run)


def bench_o2_crack(scheme: str, rho: float):
    def run():
        r = rve(matrix="M")
        r["M"] = ellipsoid(shape=spherical, prop={"K": 1.0 * tId2}, fraction=1.0 - rho)
        r["C"] = ellipsoid(shape=spheroidal(1.0e-3), prop={"K": 5.0e-6 * tId2}, fraction=rho)
        return homogenize(prop="K", rve=r, scheme=SCHEME_MAP[scheme], verbose=False,
                          maxnb=500, epsrel=1.0e-10)
    return time_call(run)


def bench_o4_sphere(scheme: str, phi: float):
    def run():
        r = rve(matrix="M")
        r["M"] = ellipsoid(shape=spherical, prop={"C": stiff_kmu(72.0, 32.0)}, fraction=1.0 - phi)
        r["I"] = ellipsoid(shape=spherical, prop={"C": stiff_kmu(5.0, 2.0)}, fraction=phi)
        return homogenize(prop="C", rve=r, scheme=SCHEME_MAP[scheme], verbose=False,
                          maxnb=500, epsrel=1.0e-10)
    return time_call(run)


def safe_time(fn, *args) -> tuple[float | None, float | None]:
    try:
        ns, arr = fn(*args)
        return ns, float(arr[0, 0])
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--iterative", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fractions_sphere = [0.05, 0.10, 0.20, 0.30, 0.40]
    fractions_spheroid = [0.10, 0.20, 0.30]
    aspects = [0.1, 10.0]
    densities_crack = [0.05, 0.15, 0.30]
    schemes = ["VOIGT", "REUSS", "DIL", "DILD", "MT", "MAX", "PCW"]
    if args.iterative:
        schemes += ["SC", "ASC", "DIFF"]

    with args.out.open("w") as out:
        for phi in fractions_sphere:
            for sc in schemes:
                ns, k11 = safe_time(bench_o2_sphere, sc, phi)
                rec = {"library": "echoes", "order": "O2", "scheme": sc,
                       "shape": "sphere", "aspect": None,
                       "param": phi, "median_ns": ns, "k_eff_11": k11}
                out.write(json.dumps(rec) + "\n")
        for aspect in aspects:
            for phi in fractions_spheroid:
                for sc in schemes:
                    ns, k11 = safe_time(bench_o2_spheroid, sc, phi, aspect)
                    shape = "oblate" if aspect < 1.0 else "prolate"
                    rec = {"library": "echoes", "order": "O2", "scheme": sc,
                           "shape": shape, "aspect": aspect,
                           "param": phi, "median_ns": ns, "k_eff_11": k11}
                    out.write(json.dumps(rec) + "\n")
        for rho in densities_crack:
            for sc in schemes:
                ns, k11 = safe_time(bench_o2_crack, sc, rho)
                rec = {"library": "echoes", "order": "O2", "scheme": sc,
                       "shape": "crack", "aspect": None,
                       "param": rho, "median_ns": ns, "k_eff_11": k11}
                out.write(json.dumps(rec) + "\n")
        for phi in fractions_sphere:
            for sc in schemes:
                ns, k11 = safe_time(bench_o4_sphere, sc, phi)
                rec = {"library": "echoes", "order": "O4", "scheme": sc,
                       "shape": "sphere", "aspect": None,
                       "param": phi, "median_ns": ns, "k_eff_11": k11}
                out.write(json.dumps(rec) + "\n")

    print(f"Wrote {args.out.stat().st_size} bytes to {args.out}")


if __name__ == "__main__":
    main()
