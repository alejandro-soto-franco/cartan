#!/usr/bin/env python3
"""ECHOES timing on the Berea sandstone real-data benchmark.

Matches the Rust `cartan-bench-berea` input (porosity 0.195, dry pores in
Zimmerman-1991 mineral). Emits JSON-line output for head-to-head comparison.

Usage:
    conda activate echoes-homog
    python bench_berea_echoes.py --out ../results/berea_echoes.jsonl
"""
from __future__ import annotations
import argparse, json, time, pathlib

import numpy as np
from echoes import (
    rve, ellipsoid, spherical, stiff_kmu, homogenize,
    VOIGT, REUSS, DIL, DILD, MT, SC, MAX, DIFF, ISO,
)

# Published Berea parameters (match berea_main.rs).
PHI        = 0.195
K_MINERAL  = 39.75
MU_MINERAL = 31.34
K_PORE     = 1.0e-6
MU_PORE    = 1.0e-6

WARMUP = 3
REPS   = 20

SCHEME_MAP = {"VOIGT": VOIGT, "REUSS": REUSS, "DIL": DIL, "DILD": DILD,
              "MT": MT, "SC": SC, "MAX": MAX, "DIFF": DIFF}


def build_rve():
    c_mineral = stiff_kmu(K_MINERAL, MU_MINERAL)
    c_pore    = stiff_kmu(K_PORE,    MU_PORE)
    r = rve(matrix="MINERAL")
    r["MINERAL"] = ellipsoid(shape=spherical, symmetrize=[ISO],
                             prop={"C": c_mineral}, fraction=1.0 - PHI)
    r["PORE"]    = ellipsoid(shape=spherical, symmetrize=[ISO],
                             prop={"C": c_pore},    fraction=PHI)
    return r


def extract_k_mu(c_eff: np.ndarray) -> tuple[float, float]:
    """Extract isotropic bulk and shear moduli from a 6x6 KM stiffness tensor."""
    # J projector (hydrostatic): 1/3 * ones(3x3) in top-3 block
    j = np.zeros((6, 6))
    j[:3, :3] = 1.0 / 3.0
    k = np.eye(6) - j
    trace_j = np.trace(j)
    trace_k = np.trace(k)
    three_K = np.trace(j @ c_eff) / trace_j
    two_mu  = np.trace(k @ c_eff) / trace_k
    return three_K / 3.0, two_mu / 2.0


def time_scheme(scheme_name: str) -> tuple[float, float, float]:
    sch = SCHEME_MAP[scheme_name]
    for _ in range(WARMUP):
        homogenize(prop="C", rve=build_rve(), scheme=sch, verbose=False,
                   maxnb=500, epsrel=1.0e-10)
    times = []
    last_k, last_mu = None, None
    for _ in range(REPS):
        r = build_rve()
        t0 = time.perf_counter_ns()
        result = homogenize(prop="C", rve=r, scheme=sch, verbose=False,
                            maxnb=500, epsrel=1.0e-10)
        times.append(time.perf_counter_ns() - t0)
        arr = np.asarray(result.array)
        last_k, last_mu = extract_k_mu(arr)
    return last_k, last_mu, float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    schemes = ["VOIGT", "REUSS", "DIL", "DILD", "MT", "SC", "MAX", "DIFF"]

    print(f"\nBerea ECHOES benchmark (φ = {PHI}, mineral K={K_MINERAL} μ={MU_MINERAL} GPa)")
    print(f"  {'scheme':<6} {'K (GPa)':>10} {'μ (GPa)':>10} {'time (ns)':>14}")
    with args.out.open("w") as out:
        for sc in schemes:
            k, mu, ns = time_scheme(sc)
            print(f"  {sc:<6} {k:>10.3f} {mu:>10.3f} {ns:>14.0f}")
            rec = {"scheme": sc, "k_eff": k, "mu_eff": mu, "ns": ns, "library": "echoes"}
            out.write(json.dumps(rec) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
