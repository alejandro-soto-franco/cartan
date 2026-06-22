import imageio.v3 as iio
from cartan_viz import cli


def test_render_produces_readable_video(run_dir):
    out = run_dir / "dashboard"
    rc = cli.main([
        "render", str(run_dir),
        "--quantity", "energy",
        "--fps", "6",
        "--top-size", "160x120",
        "--plot-size", "160x60",
        "--formats", "mp4",
        "--out", str(out),
    ])
    assert rc == 0
    mp4 = run_dir / "dashboard.mp4"
    assert mp4.exists()
    first = iio.imread(mp4, index=0)
    # stacked frame: width 160, height >= 120 (top) + something (plot), multiple of 16
    assert first.shape[1] == 160
    assert first.shape[0] % 16 == 0
    poster = run_dir / "dashboard_poster.png"
    assert poster.exists()
