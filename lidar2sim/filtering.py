"""Удаление поверхностей Poisson, удалённых от измеренного облака."""
import numpy as np
from scipy.spatial import cKDTree

from .common import clean


def filter_supported(mesh, pcd, radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("support_radius должен быть положительным конечным числом")
    if not pcd.has_points() or not mesh.has_triangles():
        raise ValueError("Для фильтрации нужны непустые облако и сетка")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    tree = cKDTree(np.asarray(pcd.points))
    remove = np.zeros(len(faces), dtype=bool)
    # Пакеты ограничивают временную память на больших сценах.
    for start in range(0, len(faces), 100_000):
        centers = vertices[faces[start:start + 100_000]].mean(axis=1)
        remove[start:start + len(centers)] = tree.query(centers)[0] > radius
    if remove.all():
        raise ValueError("Фильтр удаляет всю сетку; проверьте облако и support_radius")
    report = {"radius_m": float(radius), "triangles_before": len(faces),
              "removed_triangles": int(remove.sum())}
    mesh.remove_triangles_by_mask(remove)
    return clean(mesh), report
