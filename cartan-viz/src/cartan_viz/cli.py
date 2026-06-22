# Adapted from flowforms.
"""cartan-viz command-line interface."""
from __future__ import annotations

import argparse
from pathlib import Path

from cartan_viz import composite, diagnostics, io
from cartan_viz.scene import Scene


def _parse_size(s: str) -> tuple[int, int]:
    """Parse a WxH string like '1080x810' into (width, height)."""
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"size must be WxH, got {s!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be WxH integers, got {s!r}")


def main(argv=None) -> int:
    """Entry point for the cartan-viz CLI. Returns exit code."""
    parser = argparse.ArgumentParser(prog="cartan-viz")
    sub = parser.add_subparsers(dest="command", required=True)

    render_p = sub.add_parser("render", help="render a stacked dashboard video from a run_dir")
    render_p.add_argument("run_dir", help="path to the cartan run_dir")
    render_p.add_argument("--quantity", default="energy",
                          help="diagnostic quantity to plot (default: energy)")
    render_p.add_argument("--layout", default="stacked",
                          help="panel layout: stacked or side_by_side (default: stacked)")
    render_p.add_argument("--fps", type=int, default=30, help="frames per second (default: 30)")
    render_p.add_argument("--top-size", default="1080x810", dest="top_size",
                          help="top panel size as WxH (default: 1080x810)")
    render_p.add_argument("--plot-size", default="1080x360", dest="plot_size",
                          help="plot panel size as WxH (default: 1080x360)")
    render_p.add_argument("--out", default=None,
                          help="output path stem (default: <run_dir>/dashboard)")
    render_p.add_argument("--formats", default="mp4,webm",
                          help="comma-separated output formats (default: mp4,webm)")
    render_p.add_argument("--title", default=None, help="optional title overlay")

    args = parser.parse_args(argv)

    if args.command == "render":
        run_dir = Path(args.run_dir)
        out = Path(args.out) if args.out is not None else run_dir / "dashboard"
        top_size = _parse_size(args.top_size)
        plot_size = _parse_size(args.plot_size)
        formats = tuple(f.strip() for f in args.formats.split(","))

        series = io.Series(run_dir / "frames.pvd")
        diag = diagnostics.load(run_dir / "diagnostics.csv")
        scene = Scene.default()

        composite.render_composite_animation(
            series,
            diag,
            scene,
            quantity=args.quantity,
            out=out,
            fps=args.fps,
            top_size=top_size,
            plot_size=plot_size,
            layout=args.layout,
            formats=formats,
            title=args.title,
        )
        return 0

    return 1
