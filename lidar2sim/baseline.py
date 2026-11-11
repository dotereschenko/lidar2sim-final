"""Базовое решение: подготовка облака → Poisson → обрезка по плотности → упрощение.

Collision mesh — та же сетка, упрощённая сильнее. Это рабочая отправная точка:
конвейер проходит от облака до сцены Isaac Sim, но качество сцены не оптимизировано.
"""
from .common import Timer, clean, decimate, poisson, preprocess
from .validation import validate_params

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
    "support_distance": 0.0,
    "floor_distance": 0.0,
    "floor_cell": 0.08,
    "floor_max_edge": 1.0,
    "floor_max_slope_deg": 10.0,
    "floor_min_points": 100,
    "floor_clearance": 0.4,
    "collision_floor_fraction": 0.0,
    "collision_boundary_weight": 1.0,
}


def run(pcd, meta, traj, params=None):
    p = validate_params({} if params is None else params, PARAMS)
    t = Timer()
    with t("preprocess"):
        pcd = preprocess(pcd, meta, traj, p)
    with t("poisson"):
        mesh = poisson(pcd, p["poisson_depth"], p["density_quantile"])
        mesh = clean(mesh.crop(pcd.get_axis_aligned_bounding_box()))
    refinement = {}
    floor_components = (None, None)
    if p["support_distance"] or p["floor_distance"]:
        from .refinement import recover_floor, trim_unsupported
        with t("refinement"):
            if p["support_distance"]:
                mesh, refinement["support"] = trim_unsupported(mesh, pcd, p["support_distance"])
            if p["floor_distance"]:
                mesh, refinement["floor"], *floor_components = recover_floor(
                    mesh, pcd, traj, p, return_components=True)
    with t("decimate"):
        visual = decimate(mesh, p["visual_triangles"])
        from .collision import build_collision
        collision, collision_info = build_collision(
            mesh, *floor_components, p["collision_triangles"],
            p["collision_floor_fraction"], p["collision_boundary_weight"])
    report = {"solution": "supported_poisson" if refinement else "baseline", "params": p,
              "points_after_voxel": len(pcd.points), "timings_s": t.steps}
    if refinement:
        report["refinement"] = refinement
    report["collision_build"] = collision_info
    return visual, collision, report
