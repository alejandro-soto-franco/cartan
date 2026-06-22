# Adapted from flowforms.
"""Load VTK/PVD output from cartan into a uniform Frame/Series.

The data contract for cartan run directories:
- frames.pvd: PVD collection referencing frame_XXXX.vtu files
- Each .vtu has cell data: B (scalar), E (3-vector)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, cast

import numpy as np
import pyvista as pv


class Frame:
    """A single time snapshot loaded from a .vtu file."""

    def __init__(self, mesh: pv.DataSet, time: float = 0.0):
        self.mesh = mesh
        self.time = float(time)

    def fields(self) -> list[str]:
        """Return all point and cell data field names."""
        return list(self.mesh.point_data.keys()) + list(self.mesh.cell_data.keys())

    def array(self, name: str) -> np.ndarray:
        """Return point or cell data array by name."""
        return np.asarray(self.mesh[name])

    def cell_array(self, name: str) -> np.ndarray:
        """Return a cell data array by name."""
        return np.asarray(self.mesh.cell_data[name])


class Series:
    """A PVD time series of Frame objects."""

    def __init__(self, pvd_path: str | Path):
        self._reader = pv.PVDReader(str(pvd_path))
        self.times = np.asarray(self._reader.time_values)

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, i: int) -> Frame:
        self._reader.set_active_time_point(i)
        block = cast(pv.DataSet, self._reader.read()[0])
        return Frame(block, float(self.times[i]))

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self[i]
