"""Базовое решение: подготовка облака → Poisson → обрезка по плотности → упрощение.

Collision mesh — та же сетка, упрощённая сильнее. Это рабочая отправная точка:
конвейер проходит от облака до сцены Isaac Sim, но качество сцены не оптимизировано.
"""
from .common import Timer, clean, decimate, poisson, preprocess
from .floor import repair_floor
from .filtering import filter_supported

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
    "repair_floor": False,
    "floor_cell": 0.10,
    "floor_support_radius": 0.25,
    "support_radius": None,
}


def run(pcd, meta, traj, params=None):
    p = {**PARAMS, **(params or {})}
    t = Timer()
    with t("preprocess"):
        pcd = preprocess(pcd, meta, traj, p)
    with t("poisson"):
        mesh = poisson(pcd, p["poisson_depth"], p["density_quantile"])
        mesh = clean(mesh.crop(pcd.get_axis_aligned_bounding_box()))
    filter_report = None
    if p["support_radius"] is not None:
        with t("support_filter"):
            mesh, filter_report = filter_supported(mesh, pcd, p["support_radius"])
    with t("decimate"):
        visual = decimate(mesh, p["visual_triangles"])
        collision = decimate(mesh, p["collision_triangles"])
    floor_report = {}
    if p["repair_floor"]:
        # Заплатки после decimation: упрощение не должно снова открывать дыры.
        with t("repair_floor"):
            visual, floor_report["visual"] = repair_floor(
                visual, pcd, p["floor_cell"], p["floor_support_radius"])
            collision, floor_report["collision"] = repair_floor(
                collision, pcd, p["floor_cell"], p["floor_support_radius"])
    report = {"solution": "baseline", "params": p, "points_after_voxel": len(pcd.points),
              "timings_s": t.steps}
    if p["repair_floor"]:
        report.update(solution="baseline_floor", floor=floor_report)
    if filter_report is not None:
        report.update(solution=report["solution"] + "_filtered", support_filter=filter_report)
    return visual, collision, report
