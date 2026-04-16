#!/usr/bin/env python3
"""Head-to-head analysis of cartan-homog vs ECHOES timings + accuracy.

Reads the two JSON-line benchmark outputs, joins per (scheme, shape, param),
produces:
- head-to-head timing figure (log-log scatter, speedup histogram)
- accuracy scatter: k_eff_cartan vs k_eff_echoes
- summary table: median timing, max relative error, coverage.

Usage:
    python analyze_homog.py \\
        --cartan ../results/homog_cartan.jsonl \\
        --echoes ../results/homog_echoes.jsonl \\
        --out-dir ../figures/out
"""
from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows


def key(row: dict) -> tuple:
    return (row["order"], row["scheme"], row["shape"], row.get("aspect"), row["param"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartan", required=True, type=pathlib.Path)
    ap.add_argument("--echoes", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cartan = {key(r): r for r in load(args.cartan)}
    echoes = {key(r): r for r in load(args.echoes)}

    joined = []
    for k, cr in cartan.items():
        er = echoes.get(k)
        if er is None: continue
        joined.append((k, cr, er))

    print(f"Joined {len(joined)} cases (cartan={len(cartan)}, echoes={len(echoes)})")

    # Summary stats.
    timing_pairs = []
    accuracy_pairs = []
    for k, cr, er in joined:
        if cr["median_ns"] is not None and er["median_ns"] is not None:
            timing_pairs.append((cr["median_ns"], er["median_ns"]))
        if cr["k_eff_11"] is not None and er["k_eff_11"] is not None:
            accuracy_pairs.append((cr["k_eff_11"], er["k_eff_11"]))

    if timing_pairs:
        t_c = np.array([p[0] for p in timing_pairs])
        t_e = np.array([p[1] for p in timing_pairs])
        speedup = t_e / np.maximum(t_c, 1)
        print(f"\nTiming: {len(timing_pairs)} cases")
        print(f"  cartan median time: {np.median(t_c):.0f} ns")
        print(f"  echoes median time: {np.median(t_e):.0f} ns")
        print(f"  speedup: median = {np.median(speedup):.0f}x, range = [{speedup.min():.0f}, {speedup.max():.0f}]")

    if accuracy_pairs:
        c_vals = np.array([p[0] for p in accuracy_pairs])
        e_vals = np.array([p[1] for p in accuracy_pairs])
        rel_err = np.abs(c_vals - e_vals) / np.maximum(np.abs(e_vals), 1e-30)
        print(f"\nAccuracy: {len(accuracy_pairs)} cases")
        print(f"  median |rel err|: {np.median(rel_err):.2e}")
        print(f"  max    |rel err|: {np.max(rel_err):.2e}")
        print(f"  cases with rel err > 1e-6: {int(np.sum(rel_err > 1e-6))}")

    # JSON summary written alongside figures.
    summary = {
        "n_joined": len(joined),
        "timing": {
            "n": len(timing_pairs),
            "cartan_median_ns": float(np.median(t_c)) if timing_pairs else None,
            "echoes_median_ns": float(np.median(t_e)) if timing_pairs else None,
            "speedup_median":   float(np.median(speedup)) if timing_pairs else None,
            "speedup_min":      float(np.min(speedup)) if timing_pairs else None,
            "speedup_max":      float(np.max(speedup)) if timing_pairs else None,
        },
        "accuracy": {
            "n": len(accuracy_pairs),
            "rel_err_median": float(np.median(rel_err)) if accuracy_pairs else None,
            "rel_err_max":    float(np.max(rel_err)) if accuracy_pairs else None,
            "count_err_gt_1e-6": int(np.sum(rel_err > 1e-6)) if accuracy_pairs else 0,
        },
    }
    summary_path = args.out_dir / "homog_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")

    if HAVE_MPL and timing_pairs and accuracy_pairs:
        # Figure 1: head-to-head timing scatter.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.scatter(t_c, t_e, alpha=0.5, s=14, edgecolor="k", linewidth=0.3)
        lo = max(1, min(t_c.min(), t_e.min()))
        hi = max(t_c.max(), t_e.max())
        ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="parity")
        ax1.plot([lo, hi], [10 * lo, 10 * hi], ":", color="gray", lw=0.8, label="10x slower")
        ax1.plot([lo, hi], [100 * lo, 100 * hi], ":", color="gray", lw=0.8, label="100x slower")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("cartan median time (ns)")
        ax1.set_ylabel("ECHOES median time (ns)")
        ax1.set_title("Per-call timing, head-to-head")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="upper left", fontsize=8)

        ax2.hist(np.log10(speedup), bins=30, alpha=0.7, edgecolor="k")
        ax2.axvline(np.log10(np.median(speedup)), color="r", ls="--",
                    label=f"median = {np.median(speedup):.0f}x")
        ax2.set_xlabel("log10(ECHOES / cartan)")
        ax2.set_ylabel("count")
        ax2.set_title("Speedup distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = args.out_dir / "homog_timing.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"Timing figure -> {fig_path}")

        # Figure 2: accuracy scatter.
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(e_vals, c_vals, alpha=0.6, s=16, edgecolor="k", linewidth=0.3)
        lo = min(e_vals.min(), c_vals.min())
        hi = max(e_vals.max(), c_vals.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("ECHOES k_eff[0, 0]")
        ax.set_ylabel("cartan k_eff[0, 0]")
        ax.set_title(f"Agreement: |rel err| median = {np.median(rel_err):.1e}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = args.out_dir / "homog_accuracy.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        print(f"Accuracy figure -> {fig_path}")


if __name__ == "__main__":
    main()
