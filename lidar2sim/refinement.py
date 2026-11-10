"""Поддержка Poisson реальными наблюдениями и ограниченная достройка пола.

Вход — только подготовленное облако и траектория. Эталон и тестовые позиции
не используются. Пол достраивается внутри локальных треугольников наблюдений,
а не по прямоугольнику сцены или выпуклой оболочке целиком.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, QhullError, cKDTree

from .common import clean


def trim_unsupported(mesh, pcd, distance):
    """Удалить грани с неподдержанными вершинами или центром, до упрощения."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tree = cKDTree(np.asarray(pcd.points))
    vertex_distance = tree.query(vertices, workers=1)[0]
    center_distance = tree.query(vertices[faces].mean(axis=1), workers=1)[0]
    keep = (vertex_distance[faces].max(axis=1) <= distance) & (center_distance <= distance)
    result = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.copy()),
                                      o3d.utility.Vector3iVector(faces[keep]))
    return clean(result), {"distance_m": distance, "triangles_before": len(faces),
                           "triangles_removed": int((~keep).sum())}


def _floor_plane(points, normals, traj, p):
    """Детерминированный RANSAC почти горизонтальной поверхности ниже лидара."""
    slope = np.tan(np.deg2rad(p["floor_max_slope_deg"]))
    horizontal = np.abs(normals[:, 2]) >= np.cos(np.deg2rad(25))
    if traj is not None and len(traj):
        nearest = cKDTree(traj[:, :2]).query(points[:, :2])[1]
        horizontal &= points[:, 2] <= traj[nearest, 2] - p["floor_clearance"]
    else:
        horizontal &= points[:, 2] <= np.quantile(points[:, 2], .35)
    candidates = points[horizontal]
    if len(candidates) < p["floor_min_points"]:
        return None
    rng = np.random.default_rng(0)
    # Плотно снятая столешница не должна побеждать редкий, но протяжённый пол.
    # RANSAC голосует по пространственным ячейкам, уточнение — по всем инлайерам.
    spacing = max(2 * p["floor_cell"], 4 * p["floor_distance"])
    cells = np.floor((candidates - candidates.min(axis=0)) /
                     [spacing, spacing, 2 * p["floor_distance"]]).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    sample = np.column_stack([np.bincount(inverse, weights=candidates[:, i]) / counts
                              for i in range(3)])
    if len(sample) > 10000:
        sample = sample[rng.choice(len(sample), 10000, replace=False)]
    if len(sample) < 3:
        return None
    best, best_count = None, 0
    for _ in range(128):
        xyz = sample[rng.choice(len(sample), 3, replace=False)]
        matrix = np.column_stack([xyz[:, :2], np.ones(3)])
        if abs(np.linalg.det(matrix)) < 1e-8:
            continue
        model = np.linalg.solve(matrix, xyz[:, 2])
        if np.linalg.norm(model[:2]) > slope:
            continue
        residual = np.abs(sample[:, 2] - sample[:, :2] @ model[:2] - model[2])
        count = int((residual <= p["floor_distance"]).sum())
        if count > best_count:
            best, best_count = model, count
    if best is None:
        return None
    for _ in range(3):
        residual = np.abs(candidates[:, 2] - candidates[:, :2] @ best[:2] - best[2])
        selected = candidates[residual <= p["floor_distance"]]
        if len(selected) < p["floor_min_points"]:
            return None
        best = np.linalg.lstsq(np.column_stack([selected[:, :2], np.ones(len(selected))]),
                               selected[:, 2], rcond=None)[0]
    if np.linalg.norm(best[:2]) > slope:
        return None
    return best


def recover_floor(mesh, pcd, traj, p):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    model = _floor_plane(points, normals, traj, p)
    if model is None:
        return mesh, {"status": "skipped", "reason": "not_enough_horizontal_floor_support"}
    residual = points[:, 2] - points[:, :2] @ model[:2] - model[2]
    inliers = (np.abs(residual) <= p["floor_distance"]) & (np.abs(normals[:, 2]) >= .8)
    observed = points[inliers, :2]
    if len(observed) < p["floor_min_points"]:
        return mesh, {"status": "skipped", "reason": "not_enough_plane_inliers"}

    # Одна вершина на занятую ячейку, в среднем положении её наблюдений.
    # Индексы относительно облака: сдвиг системы координат не меняет сетку.
    origin = observed.min(axis=0)
    cells = np.floor((observed - origin) / p["floor_cell"]).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    xy = np.column_stack([np.bincount(inverse, weights=observed[:, i]) / counts
                          for i in range(2)])
    if len(xy) < 3:
        return mesh, {"status": "skipped", "reason": "degenerate_floor_footprint"}
    try:
        triangulation = Delaunay(xy)
    except QhullError:
        return mesh, {"status": "skipped", "reason": "degenerate_floor_footprint"}
    faces = triangulation.simplices.copy()
    triangles = xy[faces]
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    keep = edge_lengths.max(axis=1) <= p["floor_max_edge"]

    # Возвраты ниже плоскости — свидетельство перепада/открытого провала.
    # Одиночный шум не считается таким свидетельством.
    lower = points[(residual < -2 * p["floor_distance"]) & (np.abs(normals[:, 2]) >= .8), :2]
    blocked = 0
    if len(lower) >= 3:
        queries = np.concatenate([triangles.reshape(-1, 2), triangles.mean(axis=1)])
        count = cKDTree(lower).query_ball_point(queries, p["floor_cell"] * 1.5, return_length=True)
        contradicted = (count[:len(faces) * 3].reshape(-1, 3).max(axis=1) >= 3) | (count[len(faces) * 3:] >= 3)
        blocked = int((keep & contradicted).sum())
        keep &= ~contradicted
    faces = faces[keep]
    if not len(faces):
        return mesh, {"status": "skipped", "reason": "no_local_floor_triangles"}
    floor_vertices = np.column_stack([xy, xy @ model[:2] + model[2]])

    # Delaunay даёт CCW в XY: односторонний коллайдер смотрит вверх.
    floor = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(floor_vertices),
                                     o3d.utility.Vector3iVector(faces))
    floor = clean(floor)
    vertices = np.asarray(mesh.vertices)
    old_faces = np.asarray(mesh.triangles)
    old_triangles = vertices[old_faces]
    old_center = old_triangles.mean(axis=1)
    old_z = old_triangles[:, :, 2] - old_triangles[:, :, :2] @ model[:2] - model[2]
    cross = np.cross(old_triangles[:, 1] - old_triangles[:, 0], old_triangles[:, 2] - old_triangles[:, 0])
    horizontal = np.abs(cross[:, 2]) >= .8 * np.linalg.norm(cross, axis=1)
    near_plane = np.abs(old_z).max(axis=1) <= 2 * p["floor_distance"]
    simplex = triangulation.find_simplex(old_center[:, :2])
    covered = simplex >= 0
    covered[covered] &= keep[simplex[covered]]
    replace = horizontal & near_plane & covered
    rest = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.copy()),
                                    o3d.utility.Vector3iVector(old_faces[~replace]))
    result = clean(rest + floor)
    ft = np.asarray(floor.vertices)[np.asarray(floor.triangles)]
    area = np.linalg.norm(np.cross(ft[:, 1] - ft[:, 0], ft[:, 2] - ft[:, 0]), axis=1).sum() / 2
    return result, {"status": "applied", "plane_z_ax_by_c": model.tolist(),
                    "observed_points": int(inliers.sum()), "floor_triangles": len(floor.triangles),
                    "floor_area_m2": round(float(area), 4), "replaced_triangles": int(replace.sum()),
                    "triangles_rejected_by_lower_observations": blocked}
