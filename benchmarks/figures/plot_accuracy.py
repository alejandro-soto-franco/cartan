"""
Generate accuracy figures for cartan.sotofranco.dev/performance/geometry.

Reads: data/accuracy.csv
Writes: figures/out/accuracy_*_{light,dark}.{png,webp}

Run from benchmarks/:
    python figures/plot_accuracy.py
"""

import csv
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theme import THEMES, rc_context, apply_theme, save

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def load_accuracy():
    path = DATA_DIR / "accuracy.csv"
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["dim"] = int(row["dim"])
            row["max_abs_error"] = float(row["max_abs_error"])
            row["rms_error"] = float(row["rms_error"])
            rows.append(row)
    return rows


def plot_dist_accuracy(rows, theme_name, theme):
    """Bar chart: max absolute error for dist() on the sphere."""
    dist_rows = [r for r in rows if r["op"] == "dist"]
    dims = sorted(set(r["dim"] for r in dist_rows))

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        bar_width = 0.35
        x = np.arange(len(dims))

        cartan_vals = []
        geomstats_vals = []
        for dim in dims:
            c = [r for r in dist_rows if r["library"] == "cartan" and r["dim"] == dim]
            g = [r for r in dist_rows if r["library"] == "geomstats" and r["dim"] == dim]
            cartan_vals.append(c[0]["max_abs_error"] if c else 0)
            geomstats_vals.append(g[0]["max_abs_error"] if g else 0)

        ax.bar(x - bar_width / 2, cartan_vals, bar_width,
               color=theme["accent"], label="cartan", zorder=3)
        ax.bar(x + bar_width / 2, geomstats_vals, bar_width,
               color=theme["blue"], label="geomstats", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$n = {d}$" for d in dims], fontsize=8)
        ax.set_ylabel("Maximum absolute error", fontsize=9)
        ax.set_title(r"Numerical accuracy: $d(p, q)$ on $S^n$", fontsize=10, pad=8)
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "accuracy_dist", theme_name)
        plt.close(fig)


def plot_self_dist_accuracy(rows, theme_name, theme):
    """Bar chart: max |dist(p, p)| (should be 0)."""
    self_rows = [r for r in rows if r["op"] == "self_dist"]
    dims = sorted(set(r["dim"] for r in self_rows))

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        bar_width = 0.35
        x = np.arange(len(dims))

        cartan_vals = []
        geomstats_vals = []
        for dim in dims:
            c = [r for r in self_rows if r["library"] == "cartan" and r["dim"] == dim]
            g = [r for r in self_rows if r["library"] == "geomstats" and r["dim"] == dim]
            cartan_vals.append(c[0]["max_abs_error"] if c else 0)
            geomstats_vals.append(g[0]["max_abs_error"] if g else 0)

        ax.bar(x - bar_width / 2, cartan_vals, bar_width,
               color=theme["accent"], label="cartan", zorder=3)
        ax.bar(x + bar_width / 2, geomstats_vals, bar_width,
               color=theme["blue"], label="geomstats", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$n = {d}$" for d in dims], fontsize=8)
        ax.set_ylabel("Maximum $|d(p, p)|$", fontsize=9)
        ax.set_title(r"Self-distance accuracy: $d(p, p) = 0$ on $S^n$",
                     fontsize=10, pad=8)
        ax.set_yscale("log")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(False)

        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "accuracy_self_dist", theme_name)
        plt.close(fig)


def main():
    rows = load_accuracy()

    for theme_name, theme in THEMES.items():
        print(f"\n=== {theme_name} theme ===")
        plot_dist_accuracy(rows, theme_name, theme)
        plot_self_dist_accuracy(rows, theme_name, theme)

    print("\nDone. Figures in figures/out/")


if __name__ == "__main__":
    main()
