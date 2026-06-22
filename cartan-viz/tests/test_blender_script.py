import importlib
import py_compile
from pathlib import Path


def test_blender_script_compiles_and_imports_without_bpy():
    here = Path(__file__).resolve().parents[1] / "src" / "cartan_viz" / "blender" / "import_mdd.py"
    py_compile.compile(str(here), doraise=True)  # syntax check
    mod = importlib.import_module("cartan_viz.blender.import_mdd")
    assert hasattr(mod, "build")  # importable without Blender installed
