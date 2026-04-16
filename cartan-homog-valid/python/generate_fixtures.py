#!/usr/bin/env python3
"""Generate ECHOES reference fixtures for cartan-homog validation.

Usage:
    python generate_fixtures.py --config configs/v1_test_matrix.yaml \\
        --out $CARTAN_HOMOG_FIXTURES_DIR/v1 [--mirror-basic]

Each case produces two files at <out>/mean_field_O<order>/<case_id>.{npz,json}:
  <case_id>.npz   — expected tensor (c_eff) as 3x3 (O2) or 6x6 (O4)
  <case_id>.json  — metadata: case_id, tensor_order, scheme, tolerance_tier, ...

Requires:
  - numpy<2 (ECHOES wheel built against NumPy 1.x)
  - echoes wheel from Zenodo DOI 10.5281/zenodo.14959866
  - pyyaml
"""
from __future__ import annotations
import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import yaml

from echoes import (
    rve, ellipsoid, crack, spherical, spheroidal, stiff_kmu, homogenize,
    VOIGT, REUSS, DIL, DILD, MT, SC, ASC, MAX, PCW, DIFF, ISO,
    tZ4, tId2, Id4, tJ4, tK4,
)

SCHEME_MAP = {
    "VOIGT": VOIGT, "REUSS": REUSS, "DIL": DIL, "DILD": DILD,
    "MT": MT, "SC": SC, "ASC": ASC, "MAX": MAX, "PCW": PCW, "DIFF": DIFF,
}

REPO_BASIC_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "basic" / "v1"

# Fixed contrast parameters across the v1 matrix.
K_MATRIX_O2 = 1.0
K_INCLUSION_O2 = 5.0
K_MATRIX_O4 = (72.0, 32.0)      # (bulk, shear)
K_INCLUSION_O4 = (5.0, 2.0)


def tolerance_tier(scheme: str) -> str:
    if scheme in {"VOIGT", "REUSS", "DIL", "DILD"}:
        return "exact"
    if scheme in {"MT", "MAX", "PCW", "DIFF"}:
        return "tight"
    if scheme in {"SC", "ASC"}:
        return "iterative"
    return "quadrature_sensitive"


def _inclusion_shape(inclusion: dict):
    """Map a YAML inclusion spec to an ECHOES shape token."""
    kind = inclusion["kind"]
    if kind == "sphere":
        return spherical
    if kind == "spheroid":
        return spheroidal(float(inclusion["aspect"]))
    if kind == "crack":
        # Penny crack: vanishingly oblate spheroid. ECHOES has a dedicated crack()
        # constructor that takes the spheroid limit internally, but for conductivity
        # it's equivalent to spheroidal(omega) with small omega.
        return spheroidal(1.0e-3)
    raise ValueError(f"unknown inclusion kind: {kind}")


def _matrix_symmetrize():
    """Matrix is spherical; ISO averaging is a no-op but kept for API clarity."""
    return [ISO]


def _inclusion_symmetrize(kind: str):
    """Spheres: ISO (no effect). Spheroids/cracks: axis-aligned (no orientation average),
    so that cartan-homog (which uses a fixed axis) reproduces exactly."""
    return [ISO] if kind == "sphere" else []


def build_rve_o2(param: float, ms: dict) -> "rve":
    """Order2 (conductivity) RVE for the current microstructure and parameter."""
    shape = _inclusion_shape(ms["inclusion"])
    kind = ms["inclusion"]["kind"]
    myrve = rve(matrix="MATRIX")
    k_inclusion = K_INCLUSION_O2 * 1.0e-6 if kind == "crack" else K_INCLUSION_O2
    myrve["MATRIX"] = ellipsoid(
        shape=spherical, symmetrize=_matrix_symmetrize(),
        prop={"K": K_MATRIX_O2 * tId2}, fraction=1.0 - param,
    )
    myrve["INCLUSION"] = ellipsoid(
        shape=shape, symmetrize=_inclusion_symmetrize(kind),
        prop={"K": k_inclusion * tId2}, fraction=param,
    )
    return myrve


def build_rve_o4(param: float, ms: dict) -> "rve":
    """Order4 (elasticity) RVE."""
    shape = _inclusion_shape(ms["inclusion"])
    kind = ms["inclusion"]["kind"]
    c_matrix = stiff_kmu(*K_MATRIX_O4)
    c_inclusion = stiff_kmu(1.0e-6, 1.0e-6) if kind == "crack" else stiff_kmu(*K_INCLUSION_O4)
    myrve = rve(matrix="MATRIX")
    myrve["MATRIX"] = ellipsoid(
        shape=spherical, symmetrize=_matrix_symmetrize(),
        prop={"C": c_matrix}, fraction=1.0 - param,
    )
    myrve["INCLUSION"] = ellipsoid(
        shape=shape, symmetrize=_inclusion_symmetrize(kind),
        prop={"C": c_inclusion}, fraction=param,
    )
    return myrve


def run_case(order: str, scheme_name: str, param: float, ms: dict):
    if order == "O2":
        myrve = build_rve_o2(param, ms)
        prop_key = "K"
    else:
        myrve = build_rve_o4(param, ms)
        prop_key = "C"
    sch = SCHEME_MAP[scheme_name]
    result = homogenize(prop=prop_key, rve=myrve, scheme=sch, verbose=False,
                        maxnb=500, epsrel=1.0e-10)
    return np.asarray(result.array)


def write_case(out_dir: pathlib.Path, case_id: str, order: str,
               scheme: str, c_eff: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{case_id}.npz"
    json_path = out_dir / f"{case_id}.json"
    np.savez(npz_path, c_eff=c_eff)
    meta = {
        "case_id": case_id,
        "tensor_order": 2 if order == "O2" else 4,
        "scheme": scheme,
        "tolerance_tier": tolerance_tier(scheme),
        "echoes_version": "1.0.0",
        "echoes_commit": "zenodo:10.5281/zenodo.14959866",
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    json_path.write_text(json.dumps(meta, indent=2))


def parameter_list(ms: dict) -> list[tuple[str, float]]:
    """Return [(label, value)] for the sweep parameter. Cracks use density, others use phi."""
    if ms["inclusion"]["kind"] == "crack":
        return [(f"rho={p:.2f}", p) for p in ms["densities"]]
    return [(f"phi={p:.2f}", p) for p in ms["fractions"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--mirror-basic", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    orders = cfg["tensor_orders"]
    schemes = cfg["schemes"]
    basic_ids = set(cfg.get("basic_subset_case_ids", []))

    n_written = 0
    n_basic = 0
    n_failed = 0
    for ms in cfg["microstructures"]:
        for label, param in parameter_list(ms):
            for order in orders:
                for scheme in schemes:
                    case_id = f"{order.lower()}_{scheme.lower()}_{ms['id']}_{label}"
                    out_dir = args.out / f"mean_field_{order}"
                    try:
                        c_eff = run_case(order, scheme, param, ms)
                        write_case(out_dir, case_id, order, scheme, c_eff)
                        n_written += 1
                        if args.mirror_basic and case_id in basic_ids:
                            basic_dir = REPO_BASIC_DIR / f"mean_field_{order}"
                            write_case(basic_dir, case_id, order, scheme, c_eff)
                            n_basic += 1
                    except Exception as exc:
                        print(f"  [FAIL] {case_id}: {exc}")
                        n_failed += 1

    print(f"[generate_fixtures] wrote {n_written} cases to {args.out}")
    print(f"[generate_fixtures] {n_failed} cases failed")
    if args.mirror_basic:
        print(f"[generate_fixtures] mirrored {n_basic} basic cases into repo")


if __name__ == "__main__":
    main()
