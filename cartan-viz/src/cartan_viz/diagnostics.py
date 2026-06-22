# Adapted from flowforms.
"""Read cartan diagnostics.csv scalar time-series logs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class Diagnostics:
    """Container for time-series diagnostic columns loaded from CSV."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self._cols = columns

    @property
    def columns(self) -> list[str]:
        """Return the list of column names."""
        return list(self._cols.keys())

    def column(self, name: str) -> np.ndarray:
        """Return a column by name."""
        if name not in self._cols:
            raise KeyError(f"no diagnostics column {name!r}; have {self.columns}")
        return self._cols[name]

    @property
    def time(self) -> np.ndarray:
        """Return the time column."""
        return self.column("time")


def load(path: str | Path) -> Diagnostics:
    """Load a diagnostics CSV file into a Diagnostics object."""
    df = pd.read_csv(path)
    cols = {name: np.asarray(df[name].to_numpy(), dtype=float) for name in df.columns}
    return Diagnostics(cols)
