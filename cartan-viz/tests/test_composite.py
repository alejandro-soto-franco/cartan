import numpy as np
from cartan_viz import io, diagnostics, composite


def test_rolling_plot_exact_size(run_dir):
    d = diagnostics.load(run_dir / "diagnostics.csv")
    img = composite.rolling_plot(d, "energy", 0.1, size_px=(200, 80))
    assert img.shape == (80, 200, 3)


def test_stack_dimensions():
    top = np.zeros((120, 200, 3), dtype=np.uint8)
    bottom = np.zeros((60, 100, 3), dtype=np.uint8)
    out = composite.stack(top, bottom, layout="stacked")
    # bottom resized to top width (200), heights add
    assert out.shape[1] == 200
    assert out.shape[0] >= 120
