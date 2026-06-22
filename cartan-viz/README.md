# cartan-viz

Flowforms-style stacked dashboard visualization for cartan `run_dir` outputs.

Turns a `run_dir/` (produced by `cartan-maxwell`'s `maxwell_record` example) into
a time-synced MP4/WebM dashboard video: PyVista 3D surface + E-glyphs on top,
energy/flux rolling plot on the bottom.

## Usage

```
cartan-viz render <run_dir> [--quantity energy] [--fps 30] [--formats mp4,webm]
```

## Development

```
uv sync
uv run pytest -q
```
