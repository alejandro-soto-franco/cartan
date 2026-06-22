# Adapted from flowforms.
"""Brand chrome overlay: a subtle title (and optional handle/caption)."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cartan_viz import brand


def _font(size: int):
    """A Computer Modern Sans font at the given size, with safe fallbacks."""
    try:
        return ImageFont.truetype(brand.cm_sans_font_file(), size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def add_chrome(img: np.ndarray, *, title=None, handle=None, caption=None) -> np.ndarray:
    """Overlay text chrome onto an image array, returning the same shape (H, W, 3).

    If no text arguments are given, returns the input unchanged.
    """
    if not any((title, handle, caption)):
        return img
    pil = Image.fromarray(np.asarray(img)[..., :3].astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    if title:
        draw.text(
            (int(0.04 * w), int(0.04 * h)),
            title,
            fill=brand.PALETTE["text_light"],
            font=_font(max(18, w // 30)),
        )
    if caption:
        draw.text(
            (int(0.04 * w), int(0.90 * h)),
            caption,
            fill=brand.PALETTE["text_light"],
            font=_font(max(12, w // 55)),
        )
    if handle:
        draw.text(
            (int(0.80 * w), int(0.95 * h)),
            handle,
            fill=brand.PALETTE["muted"],
            font=_font(max(12, w // 60)),
        )
    return np.asarray(pil)
