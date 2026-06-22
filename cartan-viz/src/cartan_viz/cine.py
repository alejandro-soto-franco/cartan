# Adapted from flowforms.
"""Off-screen PyVista scene rendering for the cartan-viz top panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from cartan_viz import brand
from cartan_viz.io import Frame
from cartan_viz.scene import Scene


def _even(n: int, multiple: int = 16) -> int:
    """Round n up to the nearest multiple (default 16, required for h264)."""
    if n % multiple == 0:
        return n
    return n + (multiple - n % multiple)


def render_scene(
    frame: Frame,
    scene: Scene,
    *,
    size: tuple[int, int] = (1080, 810),
    camera_position=None,
) -> np.ndarray:
    """Render one Frame off-screen and return a (H, W, 3) uint8 numpy array.

    size is (width, height). The returned array has shape (height, width, 3).
    """
    pv.OFF_SCREEN = True
    pl = pv.Plotter(
        off_screen=True,
        theme=brand.cinema_pv_theme(),
        window_size=list(size),
    )

    if scene.background is not None:
        pl.set_background(scene.background)

    mesh = frame.mesh

    # Surface layer: colormap by B (or whatever scene.surface.field specifies).
    if scene.surface.enabled:
        cmap = scene.surface.cmap or brand.field_cmap("sequential")
        scalar_field = scene.surface.field
        pl.add_mesh(
            mesh,
            scalars=scalar_field,
            preference="cell",
            cmap=cmap,
            show_scalar_bar=True,
        )

    # Glyph layer: arrows at cell centers oriented by E.
    if scene.glyphs.enabled and scene.glyphs.field in mesh.cell_data:
        glyph_field = scene.glyphs.field
        centers = mesh.cell_centers()
        centers[glyph_field] = mesh.cell_data[glyph_field]
        glyphs = centers.glyph(
            orient=glyph_field,
            scale=glyph_field,
            factor=scene.glyphs.factor,
        )
        pl.add_mesh(glyphs, color=brand.PALETTE["gold"])

    if camera_position is not None:
        pl.camera_position = camera_position
    else:
        pl.camera_position = "xy"

    img = pl.screenshot(return_img=True)
    pl.close()
    return np.asarray(img)[..., :3].astype(np.uint8)


def normalize_frames(frames: list[np.ndarray], *, multiple: int = 16) -> list[np.ndarray]:
    """Resize every frame to one common shape whose width and height are a multiple of 16.

    Returns a uniform list so video encoders never see a frame-size change mid-stream.
    """
    arrs = [np.asarray(f)[..., :3] for f in frames]
    if not arrs:
        return arrs
    target_h = _even(max(a.shape[0] for a in arrs), multiple)
    target_w = _even(max(a.shape[1] for a in arrs), multiple)
    out = []
    for a in arrs:
        if a.shape[0] != target_h or a.shape[1] != target_w:
            a = np.asarray(
                Image.fromarray(a.astype(np.uint8)).resize((target_w, target_h))
            )
        out.append(np.ascontiguousarray(a.astype(np.uint8))[..., :3])
    return out
