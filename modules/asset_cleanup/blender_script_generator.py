"""
Blender Python script generator — Sprint 6.

Generates the Python source code that runs *inside* Blender's Python
interpreter via `blender -b --python <script.py>`.

The generated script:
  1. Clears the default scene.
  2. Imports the mesh (OBJ / PLY / FBX / GLB).
  3. Applies normalization: origin, ground alignment, scale, optional decimation.
  4. Exports as GLB.
  5. Exits with code 0 on success, 1 on error.
"""
from __future__ import annotations

from .mesh_normalization import NormalizationConfig


_SCRIPT_TEMPLATE = '''\
import sys
import bpy
import traceback

INPUT_PATH = {input_path!r}
OUTPUT_GLB = {output_glb!r}
ALIGN_ORIGIN = {align_to_origin!r}
ALIGN_GROUND = {align_ground_to_z_zero!r}
APPLY_SCALE = {apply_scale!r}
DECIMATE = {decimate_enabled!r}
DECIMATE_RATIO = {decimate_ratio!r}
DECIMATE_MIN_FACES = {decimate_min_faces!r}
FORWARD_AXIS = {forward_axis!r}
UP_AXIS = {up_axis!r}

try:
    # ── 1. clear default scene ──────────────────────────────────────
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)

    # ── 2. import ──────────────────────────────────────────────────
    ext = INPUT_PATH.rsplit(".", 1)[-1].lower()
    if ext == "obj":
        bpy.ops.wm.obj_import(filepath=INPUT_PATH)
    elif ext == "ply":
        bpy.ops.import_mesh.ply(filepath=INPUT_PATH)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=INPUT_PATH)
    elif ext == "glb" or ext == "gltf":
        bpy.ops.import_scene.gltf(filepath=INPUT_PATH)
    else:
        raise ValueError(f"Unsupported input format: {{ext}}")

    # ── 3. select all mesh objects ─────────────────────────────────
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh objects found after import")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]

    # ── 4. join into single mesh ──────────────────────────────────
    if len(mesh_objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    # ── 5. apply scale ────────────────────────────────────────────
    if APPLY_SCALE:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # ── 6. origin to geometry centre ──────────────────────────────
    if ALIGN_ORIGIN:
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        if ALIGN_GROUND:
            # move so bottom of bounding box sits at z=0
            obj.location.x = 0.0
            obj.location.y = 0.0
            bbox_z_min = min(obj.bound_box[i][2] for i in range(8))
            obj.location.z = -bbox_z_min
        else:
            obj.location = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # ── 7. optional decimation ────────────────────────────────────
    if DECIMATE:
        face_count = len(obj.data.polygons)
        if face_count >= DECIMATE_MIN_FACES:
            mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
            mod.ratio = DECIMATE_RATIO
            bpy.ops.object.modifier_apply(modifier=mod.name)

    # ── 8. export GLB ──────────────────────────────────────────────
    bpy.ops.export_scene.gltf(
        filepath=OUTPUT_GLB,
        export_format="GLB",
        export_yup=True,
        use_selection=False,
    )

    print(f"[blender_cleanup] exported to {{OUTPUT_GLB}}")
    sys.exit(0)

except Exception as exc:
    traceback.print_exc()
    sys.exit(1)
'''


def generate_cleanup_script(
    input_path: str,
    output_glb: str,
    config: NormalizationConfig,
) -> str:
    """Return the Python source string to run inside Blender."""
    return _SCRIPT_TEMPLATE.format(
        input_path=input_path,
        output_glb=output_glb,
        align_to_origin=config.align_to_origin,
        align_ground_to_z_zero=config.align_ground_to_z_zero,
        apply_scale=config.apply_scale,
        decimate_enabled=config.decimate_enabled,
        decimate_ratio=config.decimate_ratio,
        decimate_min_faces=config.decimate_min_faces,
        forward_axis=config.forward_axis,
        up_axis=config.up_axis,
    )
