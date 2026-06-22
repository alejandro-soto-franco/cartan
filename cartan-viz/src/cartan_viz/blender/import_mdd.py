"""Blender script: import a cartan run_dir mesh and attach an MDD vertex-cache animation.

Invocation (from a terminal with Blender on PATH):

    blender --background --python import_mdd.py -- <run_dir>

The script imports <run_dir>/blender/base.obj, attaches a Mesh Cache modifier
pointing at <run_dir>/blender/motion.mdd, assigns a simple emissive material,
adds a camera and a sun lamp, and optionally sets the render output path to
<run_dir>/render/.

The bpy import is guarded so this module can be imported in plain Python
(for syntax-checking and testing) without a Blender installation.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import bpy
except ImportError:
    bpy = None  # type: ignore[assignment]


def build(run_dir: str | Path) -> None:
    """Set up the Blender scene for a cartan run_dir.

    Asserts that bpy is available (i.e., we are running inside Blender).
    Clears the default scene, imports base.obj, attaches the MDD modifier,
    creates a simple emissive material, adds a camera and a sun.
    """
    assert bpy is not None, "build() must be called from within Blender (bpy not available)"

    run_dir = Path(run_dir)
    obj_path = str(run_dir / "blender" / "base.obj")
    mdd_path = str(run_dir / "blender" / "motion.mdd")

    # Clear the default scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Import the base OBJ mesh.
    bpy.ops.wm.obj_import(filepath=obj_path)
    mesh_obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = mesh_obj

    # Attach a Mesh Cache (MDD) modifier.
    mod = mesh_obj.modifiers.new(name="MDD", type="MESH_CACHE")
    mod.cache_format = "MDD"
    mod.filepath = mdd_path

    # Assign a simple emissive material so the mesh is visible in renders.
    mat = bpy.data.materials.new(name="CartanEmissive")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.2, 0.6, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 2.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = mat
    else:
        mesh_obj.data.materials.append(mat)

    # Add a camera pointing at the mesh center.
    bpy.ops.object.camera_add(location=(2.0, -3.0, 2.0))
    cam_obj = bpy.context.object
    cam_obj.rotation_euler = (1.1, 0.0, 0.6)
    bpy.context.scene.camera = cam_obj

    # Add a sun lamp for lighting.
    bpy.ops.object.light_add(type="SUN", location=(3.0, 3.0, 5.0))

    # Optionally set render output path.
    render_out = run_dir / "render" / ""
    bpy.context.scene.render.filepath = str(render_out)
    bpy.context.scene.render.image_settings.file_format = "PNG"

    print(f"cartan-viz: scene built from {run_dir}")


if __name__ == "__main__":
    # Extract run_dir from argv after the '--' separator that Blender uses.
    argv = sys.argv
    if "--" in argv:
        args_after = argv[argv.index("--") + 1:]
    else:
        args_after = []

    if not args_after:
        print("Usage: blender --background --python import_mdd.py -- <run_dir>")
        sys.exit(1)

    build(args_after[0])
