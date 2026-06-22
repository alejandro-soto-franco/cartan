import numpy as np
from cartan_viz import brand, chrome, camera


def test_palette_and_style():
    assert "blue" in brand.PALETTE
    brand.apply_figure_style(dark=True)  # must not raise


def test_chrome_preserves_shape():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    out = chrome.add_chrome(img, title="cartan")
    assert out.shape == (120, 160, 3)


def test_orbit_positions_count():
    pos = camera.orbit_positions((0.0, 0.0, 0.0), 2.0, 8)
    assert len(pos) == 8
    # each entry is (position, focal_point, up)
    assert len(pos[0]) == 3
