"""Local triangulation of observed vertical planes; no scene template or GT input."""
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, QhullError

from .common import clean


def observed_vertical_planes(pcd, distance, max_edge, min_points=100, max_planes=24,
                             trajectory=None, voxel=.03):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    patches = o3d.geometry.TriangleMesh()
    stats = []
    if len(points) < min_points or normals.shape != points.shape:
        return patches, {"planes": [], "triangles": 0, "reason": "insufficient_or_unoriented_support"}
    # Sorting removes the voxel hash-map iteration order from seeded sampling.
    indices = np.lexsort(points.T[::-1])
    remaining = indices[np.abs(normals[indices, 2]) <= np.sin(np.deg2rad(20))]
    rng = np.random.default_rng(0)
    for _ in range(max_planes):
        if len(remaining) < min_points:
            break
        subset = remaining if len(remaining) <= 25000 else remaining[np.sort(rng.choice(len(remaining), 25000, replace=False))]
        # Fit a vertical plane directly in XY. Flattening a tilted 3D fit after
        # RANSAC can move its intersection beyond the fitting tolerance.
        xy_sample = points[subset, :2]
        pairs = rng.integers(0, len(subset), size=(500, 2))
        delta = xy_sample[pairs[:, 1]] - xy_sample[pairs[:, 0]]
        lengths = np.linalg.norm(delta, axis=1)
        valid = lengths > max_edge
        if not np.any(valid):
            break
        directions = delta[valid] / lengths[valid, None]
        candidates = np.column_stack([-directions[:, 1], directions[:, 0]])
        offsets = -np.einsum("ij,ij->i", xy_sample[pairs[valid, 0]], candidates)
        best_count, normal, offset = 0, None, None
        for first in range(0, len(candidates), 32):
            ns, ds = candidates[first:first + 32], offsets[first:first + 32]
            counts = (np.abs(xy_sample @ ns.T + ds) <= distance).sum(axis=0)
            winner = int(counts.argmax())
            if counts[winner] > best_count:
                best_count, normal, offset = counts[winner], np.r_[ns[winner], 0.], ds[winner]
        if best_count < max(20, round(min_points * len(subset) / len(remaining))):
            break
        selected = np.abs(points[remaining] @ normal + offset) <= distance
        # Refine a strictly vertical plane, so its local coordinates are height
        # and distance along the wall. This mode deliberately leaves slopes alone.
        fitting = points[remaining[selected]]
        center = fitting.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((fitting[:, :2] - center[:2]).T))
        normal = np.r_[vectors[:, 0], 0.]
        offset = -center @ normal
        if np.median(normals[remaining[selected]] @ normal) < 0:
            normal = -normal
            offset = -offset
        # Once fitted, use actual near-plane returns even where thin-wall or
        # floor adjacency makes their local PCA normal unreliable.
        supported = np.abs(points @ normal + offset) <= distance
        selected = np.abs(points[remaining] @ normal + offset) <= 3 * distance
        # Consume the noisy surface neighbourhood, not just its fitting core:
        # otherwise residual noise tails become parallel duplicate walls.
        remaining = remaining[~selected]
        support = points[supported]
        if len(support) < min_points:
            continue
        orientation = None
        if trajectory is not None and len(trajectory):
            from .common import orient_normals
            from .normals import correct_occluded_normals
            order = np.lexsort(support.T[::-1])
            chosen = order[np.linspace(0, len(order)-1, min(512, len(order))).astype(int)]
            probes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(support[chosen]))
            probes.normals = o3d.utility.Vector3dVector(np.tile(normal, (len(chosen), 1)))
            orient_normals(probes, trajectory)
            orientation = correct_occluded_normals(probes, trajectory, voxel, occluder_points=points)
            if np.median(np.asarray(probes.normals) @ normal) < 0:
                normal, offset = -normal, -offset
        u = np.array([-normal[1], normal[0], 0.])
        origin = support.mean(axis=0)
        origin -= (origin @ normal + offset) * normal
        xy = np.column_stack([(support-origin) @ u, support[:, 2]-origin[2]])
        # Average observations within local cells; keep actual sampled positions.
        cell = min(.04, max_edge / 4)
        keys = np.floor((xy-xy.min(axis=0))/cell).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        xy = np.column_stack([np.bincount(inverse, weights=xy[:, i])/counts for i in range(2)])
        if len(xy) < 3:
            continue
        try:
            triangulation = Delaunay(xy)
        except QhullError:
            continue
        faces = triangulation.simplices
        triangles = xy[faces]
        keep = np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2).max(axis=1) <= max_edge
        faces = faces[keep]
        if not len(faces):
            continue
        # A noisy observed boundary is uncertain within the fitting distance.
        # Expand only its local footprint by half that distance, not a room box.
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges.sort(axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = xy[np.unique(edges[counts == 1])]
        padding = distance / 2
        offsets = np.array([[padding, 0], [-padding, 0], [0, padding], [0, -padding]])
        xy = np.unique(np.vstack([xy, (boundary[:, None, :]+offsets).reshape(-1, 2)]), axis=0)
        try:
            faces = Delaunay(xy).simplices
        except QhullError:
            continue
        triangles = xy[faces]
        faces = faces[np.linalg.norm(triangles-np.roll(triangles, 1, axis=1), axis=2).max(axis=1) <= max_edge]
        vertices = origin + xy[:, 0, None]*u + xy[:, 1, None]*np.array([0., 0., 1.])
        patch = clean(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                               o3d.utility.Vector3iVector(faces)))
        # Coplanar interior triangles consume budget without improving geometry.
        patch = clean(patch.simplify_quadric_decimation(max(100, len(patch.triangles)//20), boundary_weight=100))
        patches += patch
        stats.append({"normal": normal.tolist(), "offset_m": float(offset),
                      "support_points": len(support), "triangles": len(patch.triangles),
                      "orientation": orientation})
    patches = clean(patches)
    return patches, {"planes": stats, "triangles": len(patches.triangles),
                     "distance_m": distance, "max_edge_m": max_edge,
                     "boundary_padding_m": distance / 2, "consumed_surface_distance_m": 3 * distance, "seed": 0, "scope": "observed local vertical footprints only"}


def uncovered_patches(patches, existing, clearance):
    """Keep observed patches only away from already reconstructed surfaces."""
    if not len(patches.triangles) or not len(existing.triangles):
        return patches, {"triangles_removed": 0, "clearance_m": clearance}
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(existing))
    vertices, faces = np.asarray(patches.vertices), np.asarray(patches.triangles)
    triangles = vertices[faces]
    probes = np.concatenate([triangles, (triangles + np.roll(triangles, 1, axis=1)) / 2,
                             triangles.mean(axis=1, keepdims=True)], axis=1)
    distances = scene.compute_distance(o3d.core.Tensor(probes.astype(np.float32)), nthreads=1).numpy()
    keep = distances.min(axis=1) > clearance
    result = clean(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.copy()),
                                            o3d.utility.Vector3iVector(faces[keep])))
    return result, {"triangles_removed": int((~keep).sum()), "clearance_m": clearance}


def orient_existing_faces(mesh, observed, distance):
    """Align winding only where every probe agrees with a nearby observed patch."""
    if not len(mesh.triangles) or not len(observed.triangles):
        return mesh, {"triangles_flipped": 0}
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles).copy()
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    probes = np.concatenate([triangles, triangles.mean(axis=1, keepdims=True)], axis=1)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(observed))
    nearest = scene.compute_closest_points(o3d.core.Tensor(probes.astype(np.float32)), nthreads=1)
    nearby = np.linalg.norm(probes - nearest['points'].numpy(), axis=2) <= distance
    opposed = np.einsum('ijk,ik->ij', nearest['primitive_normals'].numpy(), normal) < -.9
    flip = (nearby & opposed).all(axis=1)
    faces[flip] = faces[flip][:, [0, 2, 1]]
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.copy()),
                                      o3d.utility.Vector3iVector(faces))
    return result, {"triangles_flipped": int(flip.sum()), "distance_m": distance}
