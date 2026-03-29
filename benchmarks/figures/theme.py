"""
Shared figure theme for cartan performance benchmarks.

Matches the sotofranco.dev blog figure conventions:
  - text.usetex: True, Computer Modern Roman
  - DPI 200, no grid, top/right spines off
  - Dual light/dark export as PNG + WebP
  - British English in all labels and captions
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = pathlib.Path(__file__).parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RC_BASE = {
    "text.usetex": True,
    "text.latex.preamble": (
        r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
    ),
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
}

LIGHT = {
    "bg": "#ffffff",
    "fg": "#1a1a1a",
    "accent": "#c9a84c",
    "accent_fill": "rgba(201, 168, 76, 0.15)",
    "blue": "#2563eb",
    "blue_fill": "rgba(37, 99, 235, 0.15)",
    "muted": "#6b7280",
    "border": "#dddddd",
}

DARK = {
    "bg": "#111111",
    "fg": "#e0e0e0",
    "accent": "#c9a84c",
    "accent_fill": "rgba(201, 168, 76, 0.15)",
    "blue": "#6ea8d9",
    "blue_fill": "rgba(110, 168, 217, 0.15)",
    "muted": "#484f58",
    "border": "#333333",
}

THEMES = {"light": LIGHT, "dark": DARK}


def rc_context(theme):
    """Return a matplotlib rc_context with theme-specific colours."""
    return plt.rc_context({
        **RC_BASE,
        "axes.facecolor": theme["bg"],
        "figure.facecolor": theme["bg"],
        "text.color": theme["fg"],
        "axes.labelcolor": theme["fg"],
        "xtick.color": theme["fg"],
        "ytick.color": theme["fg"],
    })


def apply_theme(fig, axes_list, theme):
    """Apply background and foreground colours to a figure and its axes."""
    fig.patch.set_facecolor(theme["bg"])
    for ax in axes_list:
        ax.set_facecolor(theme["bg"])
        ax.tick_params(colors=theme["fg"])
        ax.xaxis.label.set_color(theme["fg"])
        ax.yaxis.label.set_color(theme["fg"])
        if ax.get_title():
            ax.title.set_color(theme["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(theme["fg"])
        ax.grid(False)


def save(fig, name, theme_name):
    """Save figure as PNG (dpi=200) and WebP (quality=90)."""
    png_path = OUT_DIR / f"{name}_{theme_name}.png"
    webp_path = OUT_DIR / f"{name}_{theme_name}.webp"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    img = Image.open(png_path).convert("RGBA")
    img.save(webp_path, "WEBP", quality=90)
    print(f"  saved {png_path.name} + {webp_path.name}")
