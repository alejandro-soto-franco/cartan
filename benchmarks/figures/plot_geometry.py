"""
Generate geometry performance figures for cartan.sotofranco.dev/performance/geometry.

Reads: data/geometry_timings.csv, data/rust_timings.jsonl
Writes: figures/out/geom_*_{light,dark}.{png,webp}

Run from benchmarks/:
    python figures/plot_geometry.py
"""

import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theme import RC_BASE, THEMES, rc_context, apply_theme, save

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
MANIFOLD_LABELS = {
    "sphere": r"$S^n$",
    "spd": r"$\mathrm{SPD}(n)$",
    "so": r"$\mathrm{SO}(n)$",
    "grassmann": r"$\mathrm{Gr}(n, \lfloor n/2 \rfloor)$",
}
OP_LABELS = {
    "exp": r"$\exp_p(v)$",
    "log": r"$\log_p(q)$",
    "dist": r"$d(p, q)$",
    "parallel_transport": r"$\Gamma_{p \to q}(v)$",
}
MANIFOLD_ORDER = ["sphere", "spd", "so", "grassmann"]


def load_python_timings():
    """Load geometry_timings.csv into nested dict."""
    path = DATA_DIR / "geometry_timings.csv"
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["library"], row["manifold"], row["op"])
            if key not in data:
                data[key] = {"dims": [], "medians": [], "q1s": [], "q3s": []}
            data[key]["dims"].append(int(row["dim"]))
            data[key]["medians"].append(float(row["median_ns"]))
            data[key]["q1s"].append(float(row["q1_ns"]))
            data[key]["q3s"].append(float(row["q3_ns"]))
    return data


def load_rust_timings():
    """Load rust_timings.jsonl into nested dict."""
    path = DATA_DIR / "rust_timings.jsonl"
    data = {}
    if not path.exists():
        return data
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            key = ("rust", row["manifold"], row["op"])
            if key not in data:
                data[key] = {"dims": [], "medians": [], "q1s": [], "q3s": []}
            data[key]["dims"].append(row["dim"])
            data[key]["medians"].append(row["median_ns"])
            data[key]["q1s"].append(row["q1_ns"])
            data[key]["q3s"].append(row["q3_ns"])
    return data


def plot_dimension_sweep(op, py_data, rust_data, theme_name, theme):
    """Generate a 2x2 small-multiples figure for one operation."""
    with rc_context(theme):
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.5), sharex=False, sharey=False)
        axes_flat = axes.flatten()

        for i, mname in enumerate(MANIFOLD_ORDER):
            ax = axes_flat[i]
            ax.set_title(MANIFOLD_LABELS[mname], fontsize=10, pad=6)

            # geomstats
            key_g = ("geomstats", mname, op)
            if key_g in py_data:
                d = py_data[key_g]
                dims, med, q1, q3 = np.array(d["dims"]), np.array(d["medians"]), np.array(d["q1s"]), np.array(d["q3s"])
                ax.fill_between(dims, q1 / 1e3, q3 / 1e3, alpha=0.15, color=theme["blue"])
                ax.plot(dims, med / 1e3, color=theme["blue"], lw=1.5, label="geomstats")

            # cartan python
            key_c = ("cartan", mname, op)
            if key_c in py_data:
                d = py_data[key_c]
                dims, med, q1, q3 = np.array(d["dims"]), np.array(d["medians"]), np.array(d["q1s"]), np.array(d["q3s"])
                ax.fill_between(dims, q1 / 1e3, q3 / 1e3, alpha=0.15, color=theme["accent"])
                ax.plot(dims, med / 1e3, color=theme["accent"], lw=1.5, label="cartan (Python)")

            # cartan rust native
            key_r = ("rust", mname, op)
            if key_r in rust_data:
                d = rust_data[key_r]
                dims, med = np.array(d["dims"]), np.array(d["medians"])
                ax.plot(dims, med / 1e3, color=theme["accent"], lw=1.2,
                        ls="--", label="cartan (Rust)")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"Dimension $n$", fontsize=8)
            ax.set_ylabel(r"Time ($\mu$s)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(False)

        # Common legend on first subplot
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3,
                       fontsize=7.5, frameon=False,
                       bbox_to_anchor=(0.5, 1.02))

        apply_theme(fig, axes_flat, theme)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save(fig, f"geom_{op}_sweep", theme_name)
        plt.close(fig)


def plot_speedup_heatmap(py_data, theme_name, theme):
    """Generate a speedup heatmap (cartan-py / geomstats) at n=3."""
    ops = ["exp", "log", "dist", "parallel_transport"]
    target_dim = 3

    speedups = np.full((len(MANIFOLD_ORDER), len(ops)), np.nan)
    for i, mname in enumerate(MANIFOLD_ORDER):
        for j, op in enumerate(ops):
            key_c = ("cartan", mname, op)
            key_g = ("geomstats", mname, op)
            if key_c in py_data and key_g in py_data:
                c_data = py_data[key_c]
                g_data = py_data[key_g]
                # Find the entry closest to target_dim
                for ci, cd in enumerate(c_data["dims"]):
                    if cd == target_dim:
                        c_med = c_data["medians"][ci]
                        break
                else:
                    continue
                for gi, gd in enumerate(g_data["dims"]):
                    if gd == target_dim:
                        g_med = g_data["medians"][gi]
                        break
                else:
                    continue
                if c_med > 0:
                    speedups[i, j] = g_med / c_med

    with rc_context(theme):
        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(speedups, cmap="YlOrBr", aspect="auto", vmin=1)

        ax.set_xticks(range(len(ops)))
        ax.set_xticklabels([OP_LABELS[o] for o in ops], fontsize=8)
        ax.set_yticks(range(len(MANIFOLD_ORDER)))
        ax.set_yticklabels([MANIFOLD_LABELS[m] for m in MANIFOLD_ORDER], fontsize=8)

        # Annotate cells with speedup values
        for i in range(len(MANIFOLD_ORDER)):
            for j in range(len(ops)):
                val = speedups[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}x", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="#111111" if val > 5 else theme["fg"])

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Speedup factor", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(r"Speedup: cartan (Python) vs geomstats at $n = 3$",
                     fontsize=9, pad=8)
        apply_theme(fig, [ax], theme)
        fig.tight_layout()
        save(fig, "geom_speedup_heatmap", theme_name)
        plt.close(fig)


def main():
    py_data = load_python_timings()
    rust_data = load_rust_timings()

    for theme_name, theme in THEMES.items():
        print(f"\n=== {theme_name} theme ===")
        for op in ["exp", "log", "dist", "parallel_transport"]:
            plot_dimension_sweep(op, py_data, rust_data, theme_name, theme)
        plot_speedup_heatmap(py_data, theme_name, theme)

    print("\nDone. Figures in figures/out/")


if __name__ == "__main__":
    main()
