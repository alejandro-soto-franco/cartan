import pyvista as pv
import pytest
from pathlib import Path
from make_fixture import make_run_dir

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return make_run_dir(tmp_path / "run")
