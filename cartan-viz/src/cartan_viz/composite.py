# Adapted from flowforms.
"""Composite panel functions: rolling plot, stacking, and animation encoding."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

import imageio.v3 as iio

from cartan_viz import brand
from cartan_viz import cine as _cine
from cartan_viz import chrome as _chrome
from cartan_viz.diagnostics import Diagnostics
from cartan_viz.scene import Scene

# Display labels for known diagnostic quantities.
_LABELS: dict[str, str] = {
    "energy": "energy U(t)",
    "magnetic_flux_residual": "flux residual",
}

# Re-export normalize_frames from cine for convenience.
normalize_frames = _cine.normalize_frames


def _fig_to_rgb(fig) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 3) uint8 array via Agg."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _crop_or_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Force an RGB array to exactly (h, w, 3) by cropping and/or zero-padding."""
    img = np.asarray(img)[..., :3]
    ch, cw = img.shape[0], img.shape[1]
    img = img[: min(ch, h), : min(cw, w), :]
    ch, cw = img.shape[0], img.shape[1]
    if ch == h and cw == w:
        return img
    out = np.zeros((h, w, 3), dtype=img.dtype)
    out[:ch, :cw, :] = img
    return out


def rolling_plot(
    diag: Diagnostics,
    quantity: str,
    t_now: float,
    *,
    size_px: tuple[int, int] = (1080, 360),
    ylim: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a growing diagnostic curve showing data up to t_now.

    Returns exactly (h, w, 3) uint8, where size_px = (w, h). The y-limits are
    fixed to the full-series range so the axis does not jump while the line grows.
    Uses the dark house style.
    """
    brand.apply_figure_style(dark=True)
    dpi = 100
    w, h = int(size_px[0]), int(size_px[1])
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    # Fixed axes box in figure fractions: constant regardless of content.
    ax = fig.add_axes((0.12, 0.22, 0.84, 0.70))
    t = diag.time
    y = diag.column(quantity)
    t_now = float(t_now)
    mask = t <= t_now
    if mask.sum() >= 1:
        ax.plot(t[mask], y[mask], color=brand.PALETTE["blue"], lw=2.0)
    t0 = float(t.min())
    ax.set_xlim(t0, t_now if t_now > t0 else t0 + 1e-9)
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymax <= ymin:
            ymax = ymin + 1e-9
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=13)
    ax.set_xlabel(r"$t$", fontsize=19)
    ax.set_ylabel(_LABELS.get(quantity, quantity), fontsize=19)
    img = _fig_to_rgb(fig)
    return _crop_or_pad(img, h, w)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((w, h)))


def stack(
    top: np.ndarray,
    bottom: np.ndarray,
    *,
    layout: str = "stacked",
    bg=None,
) -> np.ndarray:
    """Combine top and bottom panels.

    layout="stacked": resize bottom to top width and vstack.
    """
    top = np.asarray(top)[..., :3]
    bottom = np.asarray(bottom)[..., :3]
    if layout == "stacked":
        w = top.shape[1]
        bh = round(bottom.shape[0] * w / bottom.shape[1])
        bottom_r = _resize(bottom, w, bh)
        return np.vstack([top, bottom_r])
    if layout == "side_by_side":
        h = top.shape[0]
        bw = round(bottom.shape[1] * h / bottom.shape[0])
        bottom_r = _resize(bottom, bw, h)
        return np.hstack([top, bottom_r])
    raise ValueError(f"unknown layout: {layout!r}")


def render_composite_animation(
    series,
    diag: Diagnostics,
    scene: Scene,
    *,
    quantity: str = "energy",
    out,
    fps: int = 30,
    top_size: tuple[int, int] = (1080, 810),
    plot_size: tuple[int, int] = (1080, 360),
    layout: str = "stacked",
    formats: tuple[str, ...] = ("mp4", "webm"),
    title: str | None = None,
) -> list[Path]:
    """Render a stacked dashboard animation and write video files.

    For each frame: render the 3D top panel via cine.render_scene, draw the
    rolling plot bottom panel, stack them, optionally add chrome, then encode
    to each requested format. Also writes a poster PNG from the frame with the
    highest quantity value.

    Returns a list of written video Paths.
    """
    n = len(series)
    if n == 0:
        raise ValueError("empty series; nothing to render")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(series.times)
    frames_rgb: list[np.ndarray] = []

    for i in range(n):
        top = _cine.render_scene(series[i], scene, size=top_size)
        bottom = rolling_plot(diag, quantity, float(times[i]), size_px=plot_size)
        frame = stack(top, bottom, layout=layout)
        if title is not None:
            frame = _chrome.add_chrome(frame, title=title)
        frames_rgb.append(frame)

    frames_rgb = normalize_frames(frames_rgb)

    # Write poster from the frame with the peak quantity value.
    q_vals = diag.column(quantity)
    poster_idx = int(np.argmax(q_vals)) if len(q_vals) == n else 0
    poster_path = out.parent / (out.name + "_poster.png")
    Image.fromarray(frames_rgb[poster_idx]).save(poster_path)

    written: list[Path] = []
    for fmt in formats:
        path = out.parent / (out.name + f".{fmt}")
        iio.imwrite(
            str(path),
            frames_rgb,
            fps=fps,
            codec="libx264" if fmt == "mp4" else "libvpx-vp9",
            plugin="pyav",
        )
        written.append(path)

    return written
