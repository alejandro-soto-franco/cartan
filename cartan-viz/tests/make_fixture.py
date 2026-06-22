"""Write a synthetic run_dir (3 frames) so cartan-viz tests need no cartan build."""
from pathlib import Path
import numpy as np
import pyvista as pv


def make_run_dir(root: Path) -> Path:
    root = Path(root)
    (root / "blender").mkdir(parents=True, exist_ok=True)
    times = [0.0, 0.1, 0.2]
    # a 2x2 triangulated unit square in the z=0 plane
    x, y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    pts = np.column_stack([x.ravel(), y.ravel(), np.zeros(9)])
    pvd_entries = []
    for i, t in enumerate(times):
        cloud = pv.PolyData(pts * (1.0 + 0.1 * t))
        surf = cloud.delaunay_2d()
        ncells = surf.n_cells
        surf.cell_data["B"] = np.linspace(0.0, 1.0, ncells) * (1.0 + t)
        evec = np.zeros((ncells, 3))
        evec[:, 0] = np.cos(np.arange(ncells) + t)
        evec[:, 1] = np.sin(np.arange(ncells) + t)
        surf.cell_data["E"] = evec
        name = f"frame_{i:04d}.vtu"
        surf.cast_to_unstructured_grid().save(root / name)
        pvd_entries.append((t, name))
    # frames.pvd
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="0.1">', "  <Collection>"]
    for t, name in pvd_entries:
        lines.append(f'    <DataSet timestep="{t}" file="{name}"/>')
    lines += ["  </Collection>", "</VTKFile>"]
    (root / "frames.pvd").write_text("\n".join(lines))
    # diagnostics.csv
    csv = ["time,energy,magnetic_flux_residual"]
    for t in times:
        csv.append(f"{t},{1.0 - 0.05 * t},0.0")
    (root / "diagnostics.csv").write_text("\n".join(csv) + "\n")
    return root
