"""Headless regressions for Remi Edit Mode selection tools.

Run with:
  blender --background --factory-startup --python tests/blender_edit_tools_regression.py
"""

from pathlib import Path
import math
import sys

import bmesh
import bpy
from mathutils import Matrix


ADDON_PARENT = Path(__file__).resolve().parents[2]
if str(ADDON_PARENT) not in sys.path:
    sys.path.insert(0, str(ADDON_PARENT))

import remi


def _clean_scene():
    if bpy.context.mode == "EDIT_MESH":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _activate_mesh(name, bm):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    return obj


def _selected_material_counts(obj):
    bm = bmesh.from_edit_mesh(obj.data)
    counts = {}
    for face in bm.faces:
        if face.select:
            counts[face.material_index] = counts.get(face.material_index, 0) + 1
    return counts


def test_mismatched_nested_triangulation():
    """The whole inner surface must win even when face pairs do not line up."""
    _clean_scene()
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
    outer_faces = set(bm.faces)
    rotation = Matrix.Rotation(math.radians(7.0), 4, "Z") @ Matrix.Rotation(
        math.radians(4.0), 4, "Y"
    )
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=0.90, matrix=rotation)
    inner_faces = [face for face in bm.faces if face not in outer_faces]
    bmesh.ops.reverse_faces(bm, faces=inner_faces)

    # Apply the same non-spherical radial shape to unrelated tessellations.
    for vertex in bm.verts:
        direction = vertex.co.normalized()
        azimuth = math.atan2(direction.y, direction.x)
        shape = 1.0 + 0.07 * math.sin(3.0 * azimuth) * (0.35 + direction.z * direction.z)
        vertex.co *= shape
        vertex.co.x *= 1.12
        vertex.co.y *= 0.91
        vertex.co.z *= 1.06

    for face in outer_faces:
        face.material_index = 0
    for face in inner_faces:
        face.material_index = 1
    expected_inner = len(inner_faces)
    bm.normal_update()

    obj = _activate_mesh("MismatchedDoubleShell", bm)
    assert bpy.ops.remi.select_inner_shell(include_connectors=False) == {"FINISHED"}
    counts = _selected_material_counts(obj)
    assert counts == {1: expected_inner}, (
        "inner-shell selection fragmented across nested surfaces: "
        f"selected={counts}, expected inner={expected_inner}"
    )
    print("PASS coherent nested double shell", counts)


def test_connector_wall_component():
    """Side walls joining the two layers should be selected as whole patches."""
    _clean_scene()
    bm = bmesh.new()
    segments = 24
    outer_bottom = []
    outer_top = []
    inner_bottom = []
    inner_top = []
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        cosine = math.cos(angle)
        sine = math.sin(angle)
        outer_bottom.append(bm.verts.new((cosine, sine, -1.0)))
        outer_top.append(bm.verts.new((cosine, sine, 1.0)))
        inner_bottom.append(bm.verts.new((0.90 * cosine, 0.90 * sine, -1.0)))
        inner_top.append(bm.verts.new((0.90 * cosine, 0.90 * sine, 1.0)))

    for index in range(segments):
        following = (index + 1) % segments
        outer = bm.faces.new(
            (outer_bottom[index], outer_bottom[following], outer_top[following], outer_top[index])
        )
        inner = bm.faces.new(
            (inner_bottom[index], inner_top[index], inner_top[following], inner_bottom[following])
        )
        top = bm.faces.new(
            (outer_top[index], outer_top[following], inner_top[following], inner_top[index])
        )
        bottom = bm.faces.new(
            (outer_bottom[index], inner_bottom[index], inner_bottom[following], outer_bottom[following])
        )
        outer.material_index = 0
        inner.material_index = 1
        top.material_index = 2
        bottom.material_index = 2

    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    expected = {}
    for face in bm.faces:
        if face.material_index in {1, 2}:
            expected[face.material_index] = expected.get(face.material_index, 0) + 1
    bm.normal_update()

    obj = _activate_mesh("ConnectedDoubleShell", bm)
    assert bpy.ops.remi.select_inner_shell(include_connectors=True) == {"FINISHED"}
    counts = _selected_material_counts(obj)
    assert counts == expected, (
        "inner surface or connector walls were incomplete: "
        f"selected={counts}, expected={expected}"
    )
    print("PASS coherent connector walls", counts)


def test_single_surface_is_rejected():
    """Normal closed meshes must not be mistaken for nearby double layers."""
    _clean_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    assert bpy.ops.remi.select_inner_shell() == {"CANCELLED"}
    assert _selected_material_counts(obj) == {}
    print("PASS single surface rejection")


remi.register()
try:
    test_mismatched_nested_triangulation()
    test_connector_wall_component()
    test_single_surface_is_rejected()
finally:
    _clean_scene()
    remi.unregister()

print("REMI_EDIT_TOOLS_REGRESSION_OK")
