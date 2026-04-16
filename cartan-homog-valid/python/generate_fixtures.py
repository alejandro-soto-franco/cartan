#!/usr/bin/env python3
"""Generate ECHOES reference fixtures for cartan-homog validation.

Usage:
    python generate_fixtures.py --config configs/v1_test_matrix.yaml \\
        --out $CARTAN_HOMOG_FIXTURES_DIR/v1 [--mirror-basic]

Each case produces two files at <out>/mean_field_O<order>/<case_id>.{npz,json}:
  <case_id>.npz   — expected tensor (C_eff) as 3x3 (O2) or 6x6 (O4)
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

# ECHOES import — requires numpy<2 environment
from echoes import (
    rve, ellipsoid, spherical, stiff_kmu, homogenize,
    VOIGT, REUSS, DIL, DILD, MT, SC, ASC, MAX, PCW, DIFF, ISO,
    tZ4, tId2, Id4, tJ4, tK4,
)

SCHEME_MAP = {
    "VOIGT": VOIGT, "REUSS": REUSS, "DIL": DIL, "DILD": DILD,
    "MT": MT, "SC": SC, "ASC": ASC, "MAX": MAX, "PCW": PCW, "DIFF": DIFF,
}

REPO_BASIC_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "basic" / "v1"


def tolerance_tier(scheme: str) -> str:
    if scheme in {"VOIGT", "REUSS", "DIL", "DILD"}:
        return "exact"
    if scheme in {"MT", "MAX", "PCW", "DIFF"}:
        return "tight"
    if scheme in {"SC", "ASC"}:
        return "iterative"
    return "quadrature_sensitive"


def build_rve_o2(phi: float, inclusion_kind: str) -> "rve":
    """Conductivity (Order2) RVE: matrix k=1, inclusion k=5 for spheres."""
    k_matrix = 1.0
    k_inclusion = 5.0
    # For O2 in ECHOES, use the conductivity wrapper: property key "K", tensor = k*Id2.
    myrve = rve(matrix="MATRIX")
    myrve["MATRIX"] = ellipsoid(
        shape=spherical, symmetrize=[ISO],
        prop={"K": k_matrix * tId2}, fraction=1.0 - phi,
    )
    myrve["INCLUSION"] = ellipsoid(
        shape=spherical, symmetrize=[ISO],
        prop={"K": k_inclusion * tId2}, fraction=phi,
    )
    return myrve


def build_rve_o4(phi: float, inclusion_kind: str) -> "rve":
    """Elasticity (Order4) RVE: matrix (72, 32), inclusion (5, 2) in (k, mu)."""
    c_matrix = stiff_kmu(72.0, 32.0)
    c_inclusion = stiff_kmu(5.0, 2.0)
    myrve = rve(matrix="MATRIX")
    myrve["MATRIX"] = ellipsoid(
        shape=spherical, symmetrize=[ISO],
        prop={"C": c_matrix}, fraction=1.0 - phi,
    )
    myrve["INCLUSION"] = ellipsoid(
        shape=spherical, symmetrize=[ISO],
        prop={"C": c_inclusion}, fraction=phi,
    )
    return myrve


def run_case(order: str, scheme_name: str, phi: float, inclusion_kind: str):
    if order == "O2":
        myrve = build_rve_o2(phi, inclusion_kind)
        prop_key = "K"
    else:
        myrve = build_rve_o4(phi, inclusion_kind)
        prop_key = "C"
    sch = SCHEME_MAP[scheme_name]
    result = homogenize(prop=prop_key, rve=myrve, scheme=sch, verbose=False,
                        maxnb=300, epsrel=1.0e-10)
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
    for ms in cfg["microstructures"]:
        inclusion_kind = ms["inclusion"]
        for phi in ms["fractions"]:
            for order in orders:
                for scheme in schemes:
                    case_id = f"{order.lower()}_{scheme.lower()}_{ms['id']}_phi={phi:.2f}"
                    out_dir = args.out / f"mean_field_{order}"
                    try:
                        c_eff = run_case(order, scheme, phi, inclusion_kind)
                        write_case(out_dir, case_id, order, scheme, c_eff)
                        n_written += 1
                        if args.mirror_basic and case_id in basic_ids:
                            basic_dir = REPO_BASIC_DIR / f"mean_field_{order}"
                            write_case(basic_dir, case_id, order, scheme, c_eff)
                            n_basic += 1
                    except Exception as exc:
                        print(f"  [FAIL] {case_id}: {exc}")

    print(f"[generate_fixtures] wrote {n_written} cases to {args.out}")
    if args.mirror_basic:
        print(f"[generate_fixtures] mirrored {n_basic} basic cases into repo")


if __name__ == "__main__":
    main()
