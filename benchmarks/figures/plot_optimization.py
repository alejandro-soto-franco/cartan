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
    """Frechet mean wall-clock: cartan vs geomstats across sample sizes."""
    fm_rows = [r for r in rows if r["optimiser"] == "frechet_mean"]
    if not fm_rows:
        return
    target_dim = max(r["dim"] for r in fm_rows)
    cartan_fm = [r for r in fm_rows
                 if r["library"] == "cartan" and r["dim"] == target_dim]
    geomstats_fm = [r for r in fm_rows
                    if r["library"] == "geomstats" and r["dim"] == target_dim]

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
        ax.set_title(rf"Fr\'echet mean on $S^{{{target_dim}}}$", fontsize=10, pad=8)
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
