import numpy as np
from cartan_viz import io, cine
from cartan_viz.scene import Scene


def test_render_scene_returns_requested_size(run_dir):
    frame = io.Series(run_dir / "frames.pvd")[0]
    img = cine.render_scene(frame, Scene.default(), size=(160, 120))
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8


def test_normalize_frames_uniform_multiple_of_16():
    a = np.zeros((100, 161, 3), dtype=np.uint8)
    b = np.zeros((97, 160, 3), dtype=np.uint8)
    out = cine.normalize_frames([a, b])
    h, w = out[0].shape[:2]
    assert h % 16 == 0 and w % 16 == 0
    assert all(f.shape == out[0].shape for f in out)
