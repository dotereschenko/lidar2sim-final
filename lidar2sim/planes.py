"""Local triangulation of observed vertical planes; no scene template or GT input."""
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, QhullError

from .common import clean


def observed_vertical_planes(pcd, distance, max_edge, min_points=100, max_planes=24):
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
        sample = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[subset]))
        # Open3D documents seed + probability=1 for a fixed RANSAC iteration count.
        o3d.utility.random.seed(0)
        model, inliers = sample.segment_plane(distance, 3, 500, probability=1.0)
        if len(inliers) < max(20, round(min_points * len(subset) / len(remaining))):
            break
        normal = np.asarray(model[:3])
        selected = np.abs(points[remaining] @ normal + model[3]) <= distance
        if abs(normal[2]) > np.sin(np.deg2rad(10)):
            remaining = remaining[~selected]
            continue
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
        supported = (np.abs(points @ normal + offset) <= distance) & (np.abs(normals @ normal) >= .65)
        selected = (np.abs(points[remaining] @ normal + offset) <= distance)
        # Always consume the fitted segment; degenerate footprints must terminate.
        remaining = remaining[~selected]
        support = points[supported]
        if len(support) < min_points:
            continue
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
        # Expand only its local footprint by that distance, not a room box.
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges.sort(axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary = xy[np.unique(edges[counts == 1])]
        padding = distance
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
                      "support_points": len(support), "triangles": len(patch.triangles)})
    patches = clean(patches)
    return patches, {"planes": stats, "triangles": len(patches.triangles),
                     "distance_m": distance, "max_edge_m": max_edge,
                     "boundary_padding_m": distance, "seed": 0, "scope": "observed local vertical footprints only"}
