# Adapted from flowforms.
"""Visual brand for cartan-viz: palette, colormaps, fonts, and themes.

One palette serves both the figure profile (light background, publication) and
the cinema profile (dark background, emissive). Field colormaps are
perceptually uniform and colorblind-safe.
"""
from __future__ import annotations

import os
import shutil
from functools import lru_cache
from typing import Any

import cmocean
import matplotlib as mpl
import pyvista as pv

PALETTE: dict[str, str] = {
    "ink": "#171717",
    "paper": "#ffffff",
    "grid_bg": "#ffffff",
    "cine_bg": "#0a0a0a",
    "text_light": "#ededed",
    "edge_dark": "#666666",
    "edge_light": "#333333",
    "gold": "#E8B04B",
    "rust": "#C1440E",
    "blue": "#627eea",
    "muted": "#8A93A3",
}

# Ordered accent cycle for multi-series plots (CVD-safe).
ACCENT_CYCLE: list[str] = [PALETTE["blue"], PALETTE["rust"], PALETTE["gold"], PALETTE["muted"]]

_SEQUENTIAL: Any = getattr(cmocean.cm, "thermal")
_DIVERGING: Any = getattr(cmocean.cm, "balance")

_LATEX_PREAMBLE = (
    r"\usepackage{lmodern}"
    r"\usepackage{amsmath}"
    r"\usepackage{amssymb}"
    r"\renewcommand{\familydefault}{\sfdefault}"
)


def field_cmap(kind: str) -> Any:
    """Return a colormap for a field kind.

    Sequential for magnitudes (energy, B); diverging for signed fields.
    """
    signed = {"vorticity", "pressure", "helicity", "diverging"}
    return _DIVERGING if kind in signed else _SEQUENTIAL


@lru_cache(maxsize=1)
def latex_available() -> bool:
    """True if a system LaTeX is on PATH."""
    return shutil.which("latex") is not None


def cm_sans_font_file() -> str:
    """Return a path to a Computer Modern Sans TTF (cmss10) bundled with matplotlib."""
    base = os.path.dirname(mpl.__file__)
    path = os.path.join(base, "mpl-data", "fonts", "ttf", "cmss10.ttf")
    assert os.path.exists(path), f"cmss10.ttf not found at {path}"
    return path


@lru_cache(maxsize=1)
def _usetex_renders() -> bool:
    """Probe whether usetex can actually render a tiny figure."""
    if not latex_available():
        return False
    import matplotlib.pyplot as plt
    saved = dict(mpl.rcParams)
    try:
        mpl.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "text.latex.preamble": _LATEX_PREAMBLE,
            }
        )
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"$t$")
        fig.canvas.draw()
        plt.close(fig)
        return True
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return False
    finally:
        mpl.rcParams.update(saved)


def apply_figure_style(dark: bool = False) -> None:
    """Apply the house matplotlib style globally.

    Light (publication) by default; dark=True applies the cinematic dark theme.
    """
    if dark:
        bg = PALETTE["cine_bg"]
        text = PALETTE["text_light"]
        edge = PALETTE["edge_dark"]
    else:
        bg = PALETTE["paper"]
        text = PALETTE["ink"]
        edge = PALETTE["edge_light"]

    common = {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "axes.edgecolor": edge,
        "axes.labelcolor": text,
        "text.color": text,
        "xtick.color": text,
        "ytick.color": text,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": getattr(mpl, "cycler")(color=ACCENT_CYCLE),
        "lines.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    }

    if _usetex_renders():
        common.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "text.latex.preamble": _LATEX_PREAMBLE,
            }
        )
    else:
        common.update(
            {
                "text.usetex": False,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "CMU Sans Serif",
                    "Latin Modern Sans",
                    "cmss10",
                    "DejaVu Sans",
                ],
                "mathtext.fontset": "cm",
            }
        )
    mpl.rcParams.update(common)


def figure_pv_theme() -> Any:
    """A PyVista theme for light-background publication stills."""
    theme = getattr(pv.themes, "DocumentTheme")()
    theme.background = PALETTE["paper"]
    theme.font.color = PALETTE["ink"]
    theme.cmap = "viridis"
    theme.transparent_background = False
    return theme


def cinema_pv_theme() -> Any:
    """A PyVista theme for dark, emissive cinematic renders."""
    theme = getattr(pv.themes, "DarkTheme")()
    theme.background = PALETTE["cine_bg"]
    theme.font.color = PALETTE["text_light"]
    theme.cmap = "magma"
    theme.transparent_background = False
    return theme
