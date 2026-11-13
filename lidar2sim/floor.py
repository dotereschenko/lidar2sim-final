"""Локальные заплатки горизонтального пола, поддержанные точками облака."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def repair_floor(mesh, pcd, cell=0.10, support_radius=0.25):
    if not np.isfinite([cell, support_radius]).all() or cell <= 0 or support_radius < cell:
        raise ValueError("floor_cell > 0 и floor_support_radius >= floor_cell обязательны")
    info = {"added_triangles": 0}
    points = np.asarray(pcd.points)
    if len(points) < 50 or not pcd.has_normals() or not mesh.has_triangles():
        return mesh, {**info, "skipped": "недостаточно геометрии или нормалей"}

    # shortcut: один почти горизонтальный этаж; для лестниц и нескольких этажей нужна сегментация.
    low = points[:, 2] <= np.quantile(points[:, 2], 0.30)
    horizontal = np.abs(np.asarray(pcd.normals)[:, 2]) > 0.90
    candidates = points[low & horizontal]
    if len(candidates) < 50:
        return mesh, {**info, "skipped": "не найден горизонтальный пол"}
    bins = np.floor(candidates[:, 2] / 0.03).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    height = (values[counts.argmax()] + 0.5) * 0.03
    floor = candidates[np.abs(candidates[:, 2] - height) < 0.05]
    if len(floor) < 50:
        return mesh, {**info, "skipped": "мало опорных точек"}
    origin = floor[:, :2].mean(axis=0)
    design = np.column_stack([floor[:, :2] - origin, np.ones(len(floor))])
    plane, _, rank, _ = np.linalg.lstsq(design, floor[:, 2], rcond=None)
    if rank < 3 or np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(5)):
        return mesh, {**info, "skipped": "пол не является почти горизонтальной плоскостью"}
    floor = floor[np.abs(design @ plane - floor[:, 2]) < 0.035]
    if len(floor) < 50:
        return mesh, {**info, "skipped": "плоскость не подтверждена точками"}

    lo = floor[:, :2].min(axis=0) - support_radius
    hi = floor[:, :2].max(axis=0) + support_radius
    lo = np.maximum(lo, points[:, :2].min(axis=0))
    hi = np.minimum(hi, points[:, :2].max(axis=0))
    shape = np.floor((hi - lo) / cell).astype(int)
    if np.prod(shape.astype(np.float64)) > 2_000_000:
        return mesh, {**info, "skipped": "слишком большая сетка пола; увеличьте floor_cell"}
    grid = np.indices(shape).reshape(2, -1).T
    corners = lo + (grid[:, None, :] + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * cell
    tree = cKDTree(floor[:, :2])
    distances = tree.query(corners.reshape(-1, 2))[0].reshape(-1, 4)
    corners = corners[(distances <= support_radius).all(axis=1)]
    if not len(corners):
        return mesh, {**info, "skipped": "нет поддержанных ячеек"}

    # Девять лучей на ячейку: центр, углы и середины сторон обнаруживают пропуски.
    probes = np.concatenate([corners, (corners + np.roll(corners, -1, axis=1)) / 2,
                             corners.mean(axis=1, keepdims=True)], axis=1)
    z = (probes - origin) @ plane[:2] + plane[2]
    starts = np.concatenate([probes, (z + 0.15)[..., None]], axis=2).reshape(-1, 3)
    rays = np.column_stack([starts, np.tile([0, 0, -1], (len(starts), 1))])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hit = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    missing = (np.abs(hit.reshape(-1, 9) - 0.15) > 0.04).any(axis=1)
    corners = corners[missing]
    xy = corners.reshape(-1, 2)
    vertices = np.column_stack([xy, (xy - origin) @ plane[:2] + plane[2]])
    faces = (np.arange(len(corners))[:, None, None] * 4 +
             np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
    patch = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                    o3d.utility.Vector3iVector(faces))
    patch.remove_duplicated_vertices()
    info.update(added_triangles=len(faces), area_m2=round(len(corners) * cell**2, 3),
                height_m=round(float(plane[2]), 4), support_points=len(floor))
    return mesh + patch, info
