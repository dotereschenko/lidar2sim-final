"""Базовое решение: подготовка облака → Poisson → обрезка по плотности → упрощение.

Collision mesh — та же сетка, упрощённая сильнее. Это рабочая отправная точка:
конвейер проходит от облака до сцены Isaac Sim, но качество сцены не оптимизировано.
"""
from .common import Timer, clean, decimate, poisson, preprocess

PARAMS = {
    "voxel": 0.03,
    "outlier_neighbors": 20,
    "outlier_std": 2.0,
    "normal_radius": 0.10,
    "normal_max_nn": 30,
    "poisson_depth": 9,
    "density_quantile": 0.03,
    "visual_triangles": 200_000,
    "collision_triangles": 20_000,
}


def run(pcd, meta, traj, params=None):
    p = {**PARAMS, **(params or {})}
    t = Timer()
    with t("preprocess"):
        pcd = preprocess(pcd, meta, traj, p)
    with t("poisson"):
        mesh = poisson(pcd, p["poisson_depth"], p["density_quantile"])
        mesh = clean(mesh.crop(pcd.get_axis_aligned_bounding_box()))
    with t("decimate"):
        visual = decimate(mesh, p["visual_triangles"])
        collision = decimate(mesh, p["collision_triangles"])
    report = {"solution": "baseline", "params": p, "points_after_voxel": len(pcd.points),
              "timings_s": t.steps}
    return visual, collision, report
