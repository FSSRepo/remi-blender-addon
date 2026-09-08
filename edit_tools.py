"""Edit Mode selection and bridge-splitting tools for Remi."""

from collections import deque
import math

import bmesh
import bpy
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


class Remi_OT_BridgeBase:
    """Shared lobe/bridge analysis used by Remi's Edit Mode tools."""

    expand_steps: bpy.props.IntProperty(
        name="Expand Steps",
        description="Expand detected bridge by adjacent face rings",
        min=0, max=8, default=0,
    )
    seed_samples: bpy.props.IntProperty(
        name="Seed Samples",
        description="Candidate lobe pairs evaluated while finding a bridge",
        min=4, max=48, default=18,
    )
    min_face_ratio: bpy.props.FloatProperty(
        name="Min Face Ratio",
        description="Minimum smaller/larger face ratio accepted as a lobe split",
        min=0.0, max=1.0, default=0.08, subtype="FACTOR",
    )
    min_volume_ratio: bpy.props.FloatProperty(
        name="Min Volume Ratio",
        description="Minimum smaller/larger spatial-volume ratio accepted as a lobe split",
        min=0.0, max=1.0, default=0.06, subtype="FACTOR",
    )
    volume_balance_weight: bpy.props.FloatProperty(
        name="Volume Balance",
        description="How strongly detection favors two similarly sized spatial volumes",
        min=0.0, max=10.0, default=2.5,
    )
    mark_seam: bpy.props.BoolProperty(
        name="Mark Seams",
        description="Mark the detected bridge edges as seams",
        default=False,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH"

    @staticmethod
    def _dual_graph(bm, faces):
        adjacency = {face_index: set() for face_index in faces}
        edge_by_pair = {}
        for face_index in faces:
            for edge in bm.faces[face_index].edges:
                linked = [face for face in edge.link_faces if face.index in faces]
                if len(linked) != 2:
                    continue
                first, second = linked[0].index, linked[1].index
                key = (first, second) if first < second else (second, first)
                adjacency[first].add(second)
                adjacency[second].add(first)
                edge_by_pair.setdefault(key, edge)
        return adjacency, edge_by_pair

    @staticmethod
    def _components(nodes, adjacency):
        unvisited = set(nodes)
        components = []
        while unvisited:
            start = next(iter(unvisited))
            queue = deque([start])
            unvisited.remove(start)
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency.get(current, ()):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        queue.append(neighbor)
            components.append(component)
        return sorted(components, key=len, reverse=True)

    @staticmethod
    def _distances(start, allowed, adjacency):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, ()):
                if neighbor in allowed and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def _farthest_seeds(self, component, adjacency, first_seed):
        seeds = [first_seed]
        min_distances = {face: 10 ** 9 for face in component}
        for face, distance in self._distances(first_seed, component, adjacency).items():
            min_distances[face] = distance
        while len(seeds) < min(int(self.seed_samples), len(component)):
            candidate = max(component, key=lambda face: min_distances[face] if face not in seeds else -1)
            if candidate in seeds:
                break
            seeds.append(candidate)
            for face, distance in self._distances(candidate, component, adjacency).items():
                min_distances[face] = min(min_distances[face], distance)
        return seeds

    @staticmethod
    def _volume_proxy(bm, faces):
        """Estimate spatial volume with a sampled convex hull.

        AI geometry is frequently open or non-manifold, so signed mesh volume
        is unreliable.  A convex hull gives a stable comparison between two
        candidate object-like regions; an area/span proxy handles flat shells.
        """
        if not faces:
            return 0.0
        min_co = max_co = None
        area = 0.0
        used_vertices = set()
        for face_index in faces:
            face = bm.faces[face_index]
            area += face.calc_area()
            for vertex in face.verts:
                if vertex.index in used_vertices:
                    continue
                used_vertices.add(vertex.index)
                if min_co is None:
                    min_co = max_co = vertex.co.copy()
                else:
                    min_co.x = min(min_co.x, vertex.co.x)
                    min_co.y = min(min_co.y, vertex.co.y)
                    min_co.z = min(min_co.z, vertex.co.z)
                    max_co.x = max(max_co.x, vertex.co.x)
                    max_co.y = max(max_co.y, vertex.co.y)
                    max_co.z = max(max_co.z, vertex.co.z)
        fallback = area * max(1e-6, (max_co - min_co).length)
        coords = []
        used_vertices.clear()
        for face_index in faces:
            for vertex in bm.faces[face_index].verts:
                if vertex.index not in used_vertices:
                    used_vertices.add(vertex.index)
                    coords.append(vertex.co.copy())
        if len(coords) < 4:
            return max(1e-12, fallback)

        # Hull complexity is capped so bridge detection remains interactive.
        if len(coords) > 768:
            step = len(coords) / 768.0
            coords = [coords[int(index * step)] for index in range(768)]
        hull = bmesh.new()
        try:
            for coordinate in coords:
                hull.verts.new(coordinate)
            hull.verts.ensure_lookup_table()
            bmesh.ops.convex_hull(hull, input=list(hull.verts), use_existing_faces=False)
            volume = abs(float(hull.calc_volume(signed=False)))
        except Exception:
            volume = 0.0
        finally:
            hull.free()
        return max(1e-12, volume if volume > 1e-12 else fallback)

    def _partition(self, bm, component, adjacency, edge_by_pair):
        """Find the least disruptive balanced face-graph cut between two lobes."""
        start = next(iter(component))
        first_distances = self._distances(start, component, adjacency)
        seed_a = max(first_distances, key=first_distances.get)
        distances_a = self._distances(seed_a, component, adjacency)
        candidates = self._farthest_seeds(component, adjacency, seed_a)[1:]
        best = None
        infinity = 10 ** 9

        for seed_b in candidates:
            distances_b = self._distances(seed_b, component, adjacency)
            part_a = {face for face in component if distances_a.get(face, infinity) <= distances_b.get(face, infinity)}
            part_b = component - part_a
            if not part_a or not part_b:
                continue
            face_ratio = min(len(part_a), len(part_b)) / max(len(part_a), len(part_b))
            volume_a = self._volume_proxy(bm, part_a)
            volume_b = self._volume_proxy(bm, part_b)
            volume_ratio = min(volume_a, volume_b) / max(volume_a, volume_b)
            bridge_pairs = [
                pair for pair in edge_by_pair
                if (pair[0] in part_a) != (pair[1] in part_a)
            ]
            if not bridge_pairs:
                continue
            cut_length = sum(edge_by_pair[pair].calc_length() for pair in bridge_pairs)
            volume_scale = max(1e-9, min(volume_a, volume_b) ** (1.0 / 3.0))
            neck_score = cut_length / volume_scale
            # Favor every narrow connection between two object-like volumes.
            # Volume balance dominates face count because AI shells often have
            # very different tessellation densities on otherwise equal parts.
            score = (
                neck_score
                + (1.0 - volume_ratio) * float(self.volume_balance_weight)
                + (1.0 - face_ratio) * 0.2
            )
            candidate = (score, part_a, part_b, bridge_pairs, seed_a, seed_b, face_ratio, volume_ratio)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            return None
        _, part_a, part_b, bridge_pairs, seed_a, seed_b, face_ratio, volume_ratio = best
        return (
            part_a,
            part_b,
            {edge_by_pair[pair] for pair in bridge_pairs},
            seed_a,
            seed_b,
            face_ratio,
            volume_ratio,
        )

    @staticmethod
    def _clear_selection(bm):
        for vertex in bm.verts:
            vertex.select = False
        for edge in bm.edges:
            edge.select = False
        for face in bm.faces:
            face.select = False

    def _selected_component(self, bm):
        selected = {face.index for face in bm.faces if face.select}
        if len(selected) < 4:
            return None
        adjacency, edge_by_pair = self._dual_graph(bm, selected)
        components = self._components(selected, adjacency)
        if not components:
            return None
        component = set(components[0])
        return component, adjacency, edge_by_pair, len(components)

    def _expand_edges(self, edges, component):
        expanded = set(edges)
        for _ in range(self.expand_steps):
            ring_faces = {
                face
                for edge in expanded
                for face in edge.link_faces
                if face.index in component
            }
            for face in ring_faces:
                expanded.update(
                    edge for edge in face.edges
                    if any(linked.index in component for linked in edge.link_faces)
                )
        return expanded

    def _valid_partition(self, partition):
        if partition is None:
            self.report({"ERROR"}, "Could not find a bridge between two face lobes")
            return False
        if partition[5] < self.min_face_ratio or partition[6] < self.min_volume_ratio:
            self.report(
                {"ERROR"},
                "Split rejected as unbalanced. Narrow the selection or lower Min Face/Volume Ratio in the redo panel.",
            )
            return False
        return True


class Remi_OT_DetectBridge(Remi_OT_BridgeBase, bpy.types.Operator):
    """Select connector edges between the two main regions of the face selection."""

    bl_idname = "remi.detect_bridge"
    bl_label = "Detect Volume Bridges"
    bl_options = {"REGISTER", "UNDO"}

    clear_selection: bpy.props.BoolProperty(
        name="Clear Selection",
        description="Clear the face selection before selecting bridge edges",
        default=True,
    )

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        data = self._selected_component(bm)
        if data is None:
            self.report({"ERROR"}, "Select at least four connected faces")
            return {"CANCELLED"}
        component, adjacency, edge_by_pair, count = data
        partition = self._partition(bm, component, adjacency, edge_by_pair)
        if not self._valid_partition(partition):
            return {"CANCELLED"}
        _, _, edges, _, _, _, _ = partition
        edges = self._expand_edges(edges, component)
        if self.clear_selection:
            self._clear_selection(bm)
        for edge in edges:
            edge.select = True
            edge.seam = edge.seam or self.mark_seam
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)
        bpy.ops.mesh.select_mode(type="EDGE")
        suffix = " (largest connected selection used)" if count > 1 else ""
        self.report({"INFO"}, f"Bridge edges selected: {len(edges)}{suffix}")
        return {"FINISHED"}


class Remi_OT_SelectSplitPart(Remi_OT_BridgeBase, bpy.types.Operator):
    """Select the face lobe that Split By Bridge would turn into a new object."""

    bl_idname = "remi.select_split_part"
    bl_label = "Preview Fused Part"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        data = self._selected_component(bm)
        if data is None:
            self.report({"ERROR"}, "Select at least four connected faces")
            return {"CANCELLED"}
        component, adjacency, edge_by_pair, _ = data
        partition = self._partition(bm, component, adjacency, edge_by_pair)
        if not self._valid_partition(partition):
            return {"CANCELLED"}
        part_a, _, _, _, _, _, _ = partition
        self._clear_selection(bm)
        for index in part_a:
            bm.faces[index].select = True
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)
        bpy.ops.mesh.select_mode(type="FACE")
        self.report({"INFO"}, f"Selected split part faces: {len(part_a)}")
        return {"FINISHED"}


class Remi_OT_SplitByBridge(Remi_OT_BridgeBase, bpy.types.Operator):
    """Separate one lobe of the selected mesh by its detected bridge."""

    bl_idname = "remi.split_by_bridge"
    bl_label = "Separate Fused Volumes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        data = self._selected_component(bm)
        if data is None:
            self.report({"ERROR"}, "Select at least four connected faces")
            return {"CANCELLED"}
        component, adjacency, edge_by_pair, _ = data
        partition = self._partition(bm, component, adjacency, edge_by_pair)
        if not self._valid_partition(partition):
            return {"CANCELLED"}
        part_a, _, edges, _, _, _, _ = partition
        self._clear_selection(bm)
        for index in part_a:
            bm.faces[index].select = True
        if self.mark_seam:
            for edge in edges:
                edge.seam = True
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)
        bpy.ops.mesh.select_mode(type="FACE")
        before = {obj.name for obj in bpy.data.objects}
        result = bpy.ops.mesh.separate(type="SELECTED")
        if "FINISHED" not in result:
            self.report({"ERROR"}, "Could not separate the selected split part")
            return {"CANCELLED"}
        created = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
        self.report({"INFO"}, f"Bridge split complete: {len(created)} object created")
        return {"FINISHED"}


class Remi_OT_SmartSelectObject(Remi_OT_BridgeBase, bpy.types.Operator):
    """Select the lobe containing the currently selected face, edge, or vertex."""

    bl_idname = "remi.smart_select_object"
    bl_label = "Smart Select Object"
    bl_options = {"REGISTER", "UNDO"}

    refine_steps: bpy.props.IntProperty(
        name="Refine Steps",
        description="How many times to refine the lobe containing the picked element",
        min=0, max=8, default=2,
    )

    @staticmethod
    def _seed_face(bm):
        if bm.faces.active and bm.faces.active.select:
            return bm.faces.active.index
        for face in bm.faces:
            if face.select:
                return face.index
        for edge in bm.edges:
            if edge.select and edge.link_faces:
                return edge.link_faces[0].index
        for vertex in bm.verts:
            if vertex.select and vertex.link_faces:
                return vertex.link_faces[0].index
        return None

    def execute(self, context):
        bm = bmesh.from_edit_mesh(context.active_object.data)
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        seed = self._seed_face(bm)
        if seed is None:
            self.report({"ERROR"}, "Select a face, edge, or vertex to identify the desired lobe")
            return {"CANCELLED"}
        component = {face.index for face in bm.faces}
        adjacency, all_edges = self._dual_graph(bm, component)
        for _ in range(self.refine_steps):
            if len(component) < 8:
                break
            local_adjacency = {face: adjacency[face] & component for face in component}
            local_edges = {pair: edge for pair, edge in all_edges.items() if pair[0] in component and pair[1] in component}
            partition = self._partition(bm, component, local_adjacency, local_edges)
            if (
                partition is None
                or partition[5] < self.min_face_ratio
                or partition[6] < self.min_volume_ratio
            ):
                break
            part_a, part_b, _, _, _, _, _ = partition
            next_component = part_a if seed in part_a else part_b
            if len(next_component) == len(component):
                break
            component = next_component
        self._clear_selection(bm)
        for index in component:
            bm.faces[index].select = True
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)
        bpy.ops.mesh.select_mode(type="FACE")
        self.report({"INFO"}, f"Smart selected faces: {len(component)}")
        return {"FINISHED"}


class Remi_OT_DoubleShellBase:
    """Detect nearby opposite-facing layers in double-shell AI geometry."""

    max_thickness_ratio: bpy.props.FloatProperty(
        name="Max Shell Gap",
        description="Maximum layer separation as a fraction of the mesh diagonal",
        min=0.0001, max=0.25, default=0.035, precision=4, subtype="FACTOR",
    )
    opposite_angle: bpy.props.FloatProperty(
        name="Opposite Angle °",
        description="Minimum normal angle used to recognize opposing surface layers",
        min=90.0, max=180.0, default=135.0,
    )
    propagation_angle: bpy.props.FloatProperty(
        name="Surface Continuity °",
        description="Maximum angle across which inner-shell selection may propagate",
        min=1.0, max=179.0, default=70.0,
    )
    include_connectors: bpy.props.BoolProperty(
        name="Include Connectors",
        description="Include side-wall faces that directly join detected inner and outer layers",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and context.mode == "EDIT_MESH"

    @staticmethod
    def _clear_selection(bm):
        for vertex in bm.verts:
            vertex.select = False
        for edge in bm.edges:
            edge.select = False
        for face in bm.faces:
            face.select = False

    @staticmethod
    def _surface_patches(faces, continuity_dot):
        """Group adjacent, similarly oriented faces into coherent surfaces."""
        visible = {face.index for face in faces}
        face_to_patch = {}
        patches = []
        for seed in faces:
            if seed.index in face_to_patch:
                continue
            patch_index = len(patches)
            patch = set()
            queue = deque([seed])
            face_to_patch[seed.index] = patch_index
            while queue:
                face = queue.popleft()
                patch.add(face.index)
                for edge in face.edges:
                    for neighbor in edge.link_faces:
                        neighbor_index = neighbor.index
                        if neighbor_index not in visible or neighbor_index in face_to_patch:
                            continue
                        if face.normal.dot(neighbor.normal) < continuity_dot:
                            continue
                        face_to_patch[neighbor_index] = patch_index
                        queue.append(neighbor)
            patches.append(patch)
        return face_to_patch, patches

    @staticmethod
    def _patch_metrics(bm, patches, face_centers, center):
        metrics = []
        for patch in patches:
            area_sum = 0.0
            radius_sum = 0.0
            orientation_sum = 0.0
            for face_index in patch:
                face = bm.faces[face_index]
                area = max(1e-12, face.calc_area())
                radial = face_centers[face_index] - center
                area_sum += area
                radius_sum += area * radial.length
                if radial.length_squared > 1e-18:
                    orientation_sum += area * face.normal.dot(radial.normalized())
            metrics.append(
                {
                    "area": area_sum,
                    "radius": radius_sum / max(1e-12, area_sum),
                    "orientation": orientation_sum / max(1e-12, area_sum),
                }
            )
        return metrics

    def _detect_inner_shell(self, bm):
        bm.normal_update()
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        faces = [face for face in bm.faces if not face.hide]
        if len(faces) < 8:
            return None

        coordinates = [vertex.co for vertex in bm.verts if not vertex.hide]
        if not coordinates:
            return None
        center = sum(coordinates, coordinates[0] * 0.0) / len(coordinates)
        min_co = coordinates[0].copy()
        max_co = coordinates[0].copy()
        for coordinate in coordinates[1:]:
            min_co.x = min(min_co.x, coordinate.x)
            min_co.y = min(min_co.y, coordinate.y)
            min_co.z = min(min_co.z, coordinate.z)
            max_co.x = max(max_co.x, coordinate.x)
            max_co.y = max(max_co.y, coordinate.y)
            max_co.z = max(max_co.z, coordinate.z)
        diagonal = max(1e-9, (max_co - min_co).length)
        max_gap = diagonal * float(self.max_thickness_ratio)
        opposite_dot = math.cos(math.radians(float(self.opposite_angle)))
        continuity_dot = math.cos(math.radians(float(self.propagation_angle)))

        face_centers = {face.index: face.calc_center_median() for face in faces}
        visible = {face.index for face in faces}
        face_to_patch, patches = self._surface_patches(faces, continuity_dot)
        patch_metrics = self._patch_metrics(bm, patches, face_centers, center)

        tree = KDTree(len(faces))
        for face in faces:
            tree.insert(face_centers[face.index], face.index)
        tree.balance()

        # A ray cast along the back of each face finds the opposing layer even
        # when the two sides have unrelated triangle sizes and centroids.  The
        # KD fallback retains support for noisy scans whose layers are not
        # perfectly aligned along the face normal.
        ray_epsilon = max(1e-8, diagonal * 1e-7)
        surface_bvh = BVHTree.FromBMesh(bm, epsilon=ray_epsilon)
        matched_pairs = set()
        for face in faces:
            center_a = face_centers[face.index]
            partner = None
            direction = -face.normal
            hit = surface_bvh.ray_cast(
                center_a + direction * ray_epsilon,
                direction,
                max_gap,
            )
            if hit[2] is not None:
                other_index = int(hit[2])
                if (
                    other_index != face.index
                    and other_index in visible
                    and face_to_patch[other_index] != face_to_patch[face.index]
                    and face.normal.dot(bm.faces[other_index].normal) <= opposite_dot
                ):
                    partner = other_index

            if partner is None:
                best = None
                for _, other_index, distance in tree.find_n(center_a, min(64, len(faces))):
                    if (
                        other_index == face.index
                        or distance > max_gap
                        or face_to_patch[other_index] == face_to_patch[face.index]
                    ):
                        continue
                    other = bm.faces[other_index]
                    normal_dot = face.normal.dot(other.normal)
                    if normal_dot > opposite_dot:
                        continue
                    delta = face_centers[other_index] - center_a
                    if delta.length_squared <= 1e-18:
                        continue
                    pair_direction = delta.normalized()
                    # On a true double layer, each face points away from its
                    # counterpart. Reject lateral neighbors that only happen
                    # to have opposing normals on a folded surface.
                    facing = min(
                        -face.normal.dot(pair_direction),
                        other.normal.dot(pair_direction),
                    )
                    if facing < 0.15:
                        continue
                    score = (
                        distance / max_gap
                        + 0.35 * (1.0 + normal_dot)
                        + 0.35 * (1.0 - facing)
                    )
                    if best is None or score < best[0]:
                        best = (score, other_index)
                if best is not None:
                    partner = best[1]

            if partner is None:
                continue
            matched_pairs.add(tuple(sorted((face.index, partner))))

        minimum_pairs = max(4, min(32, len(faces) // 100))
        if len(matched_pairs) < minimum_pairs:
            return None

        # Consolidate pair evidence at the surface-patch level.  This is the
        # important distinction from the old per-face seed flood: one noisy
        # pair can no longer punch a zigzag hole through an otherwise coherent
        # inner surface.
        relations = {}
        for face_a, face_b in matched_pairs:
            patch_a = face_to_patch[face_a]
            patch_b = face_to_patch[face_b]
            if patch_a == patch_b:
                continue
            relation = tuple(sorted((patch_a, patch_b)))
            weight = min(
                max(1e-12, bm.faces[face_a].calc_area()),
                max(1e-12, bm.faces[face_b].calc_area()),
            )
            relations[relation] = relations.get(relation, 0.0) + weight

        if not relations:
            return None

        inner_votes = {patch_index: 0.0 for patch_index in range(len(patches))}
        outer_votes = {patch_index: 0.0 for patch_index in range(len(patches))}
        for (patch_a, patch_b), weight in relations.items():
            metrics_a = patch_metrics[patch_a]
            metrics_b = patch_metrics[patch_b]
            orientation_a = metrics_a["orientation"]
            orientation_b = metrics_b["orientation"]

            # Complete nested surfaces have opposite signed radial orientation:
            # the outer layer points away from the object and the inner layer
            # points into its cavity. Radius is the stable fallback for partial
            # or nearly tangent patches.
            if orientation_a <= -0.05 and orientation_b >= 0.05:
                inner, outer = patch_a, patch_b
            elif orientation_b <= -0.05 and orientation_a >= 0.05:
                inner, outer = patch_b, patch_a
            elif abs(metrics_a["radius"] - metrics_b["radius"]) > diagonal * 1e-6:
                inner, outer = (
                    (patch_a, patch_b)
                    if metrics_a["radius"] < metrics_b["radius"]
                    else (patch_b, patch_a)
                )
            else:
                inner, outer = (
                    (patch_a, patch_b)
                    if orientation_a < orientation_b
                    else (patch_b, patch_a)
                )
            inner_votes[inner] += weight
            outer_votes[outer] += weight

        inner_patches = {
            patch_index
            for patch_index in range(len(patches))
            if inner_votes[patch_index] > outer_votes[patch_index]
        }
        outer_patches = {
            patch_index
            for patch_index in range(len(patches))
            if outer_votes[patch_index] > inner_votes[patch_index]
        }
        if not inner_patches or not outer_patches:
            return None

        inner_faces = {
            face_index
            for patch_index in inner_patches
            for face_index in patches[patch_index]
        }

        connector_faces = set()
        if self.include_connectors:
            patch_adjacency = {patch_index: set() for patch_index in range(len(patches))}
            for face in faces:
                patch_a = face_to_patch[face.index]
                for edge in face.edges:
                    for neighbor in edge.link_faces:
                        if neighbor.index not in visible:
                            continue
                        patch_b = face_to_patch[neighbor.index]
                        if patch_a != patch_b:
                            patch_adjacency[patch_a].add(patch_b)

            # Connector walls can themselves be split into several sharp
            # patches. Include a whole neutral patch component only when it
            # topologically bridges a classified inner and outer surface.
            neutral = set(range(len(patches))) - inner_patches - outer_patches
            while neutral:
                start = neutral.pop()
                component = {start}
                queue = deque([start])
                boundary = set()
                while queue:
                    patch_index = queue.popleft()
                    for neighbor in patch_adjacency[patch_index]:
                        if neighbor in neutral:
                            neutral.remove(neighbor)
                            component.add(neighbor)
                            queue.append(neighbor)
                        elif neighbor not in component:
                            boundary.add(neighbor)
                if boundary & inner_patches and boundary & outer_patches:
                    connector_faces.update(
                        face_index
                        for patch_index in component
                        for face_index in patches[patch_index]
                    )

        selected = inner_faces | connector_faces
        if not selected or len(selected) >= int(len(faces) * 0.85):
            return None
        return selected, len(matched_pairs), len(connector_faces), max_gap


class Remi_OT_SelectInnerShell(Remi_OT_DoubleShellBase, bpy.types.Operator):
    """Preview the inner layer of double-sided AI-generated geometry."""

    bl_idname = "remi.select_inner_shell"
    bl_label = "Select Inner Shell"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        result = self._detect_inner_shell(bm)
        if result is None:
            self.report({"WARNING"}, "No confident double shell found; adjust Max Shell Gap in the redo panel")
            return {"CANCELLED"}
        selected, pair_count, connector_count, _ = result
        self._clear_selection(bm)
        for face_index in selected:
            bm.faces[face_index].select = True
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        bpy.ops.mesh.select_mode(type="FACE")
        self.report(
            {"INFO"},
            f"Inner shell selected: {len(selected)} faces, {pair_count} opposing pairs, {connector_count} connectors",
        )
        return {"FINISHED"}


class Remi_OT_RemoveInnerShell(Remi_OT_DoubleShellBase, bpy.types.Operator):
    """Remove the detected inner layer while retaining the outer surface."""

    bl_idname = "remi.remove_inner_shell"
    bl_label = "Remove Inner Shell"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        selected = {face.index for face in bm.faces if face.select}
        pair_count = connector_count = 0
        used_preview = 4 <= len(selected) < int(len(bm.faces) * 0.85)
        if not used_preview:
            result = self._detect_inner_shell(bm)
            if result is None:
                self.report({"WARNING"}, "No confident double shell found; use Select Inner Shell to tune detection first")
                return {"CANCELLED"}
            selected, pair_count, connector_count, _ = result
        faces_to_delete = [bm.faces[index] for index in selected if index < len(bm.faces)]
        bmesh.ops.delete(bm, geom=faces_to_delete, context="FACES")
        bmesh.update_edit_mesh(obj.data, loop_triangles=True, destructive=True)
        detail = "using the preview selection" if used_preview else f"from {pair_count} opposing pairs ({connector_count} connectors)"
        self.report({"INFO"}, f"Removed inner shell: {len(faces_to_delete)} faces {detail}")
        return {"FINISHED"}


bridge_classes = (
    Remi_OT_DetectBridge,
    Remi_OT_SelectSplitPart,
    Remi_OT_SplitByBridge,
    Remi_OT_SmartSelectObject,
)

double_shell_classes = (
    Remi_OT_SelectInnerShell,
    Remi_OT_RemoveInnerShell,
)

# Blender reads RNA properties from the concrete registered class.  Copy the
# shared analysis controls into each operator while keeping the implementation
# itself as a plain Python mixin.
for _operator_class in bridge_classes:
    _operator_class.__annotations__ = {
        **Remi_OT_BridgeBase.__annotations__,
        **getattr(_operator_class, "__annotations__", {}),
    }

for _operator_class in double_shell_classes:
    _operator_class.__annotations__ = {
        **Remi_OT_DoubleShellBase.__annotations__,
        **getattr(_operator_class, "__annotations__", {}),
    }

classes = bridge_classes + double_shell_classes


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
