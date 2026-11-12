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
    "visibility_normals": False,
    "poisson_depth": 9,
    "density_quantile": 0.03,
    "visual_triangles": 200_000,
    "collision_triangles": 20_000,
    "repair_floor": False,
    "floor_support_radius": 0.25,
    "support_radius": None,
    "support_distance": 0.0,
    "floor_distance": 0.0,
    "floor_boundary_pad": 0.0,
    "floor_cell": 0.08,
    "floor_max_edge": 1.0,
    "floor_max_slope_deg": 10.0,
    "floor_min_points": 100,
    "floor_clearance": 0.4,
    "collision_floor_fraction": 0.0,
    "collision_boundary_weight": 1.0,
    "plane_distance": 0.0,
    "plane_max_edge": 0.3,
}


def run(pcd, meta, traj, params=None):
    p = validate_params({} if params is None else params, PARAMS)
    t = Timer()
    normal_diagnostics = {}
    with t("preprocess"):
        pcd = preprocess(pcd, meta, traj, p, normal_diagnostics)
    with t("poisson"):
        mesh = poisson(pcd, p["poisson_depth"], p["density_quantile"])
        mesh = clean(mesh.crop(pcd.get_axis_aligned_bounding_box()))
    filter_report = None
    if p["support_radius"] is not None:
        from .filtering import filter_supported
        with t("support_filter"):
            mesh, filter_report = filter_supported(mesh, pcd, p["support_radius"])
    refinement = {}
    floor_components = (None, None)
    if p["support_distance"] or p["floor_distance"] or p["plane_distance"]:
        from .refinement import recover_floor, trim_unsupported
        with t("refinement"):
            if p["support_distance"]:
                mesh, refinement["support"] = trim_unsupported(mesh, pcd, p["support_distance"])
            if p["plane_distance"]:
                from .planes import observed_vertical_planes, uncovered_patches, orient_existing_faces
                patches, refinement["planes"] = observed_vertical_planes(
                    pcd, p["plane_distance"], p["plane_max_edge"], trajectory=traj, voxel=p["voxel"])
                if p["support_distance"]:
                    patches, refinement["planes"]["support_filter"] = trim_unsupported(patches, pcd, p["support_distance"])
                if traj is not None and len(traj):
                    mesh, refinement["planes"]["existing_orientation"] = orient_existing_faces(mesh, patches, p["plane_distance"])
                patches, refinement["planes"]["existing_surface_filter"] = uncovered_patches(patches, mesh, 3 * p["plane_distance"])
                mesh = clean(mesh + patches)
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
    if normal_diagnostics:
        report["normal_orientation"] = normal_diagnostics
    if refinement:
        report["refinement"] = refinement
    if filter_report is not None:
        report["support_filter"] = filter_report
    if p["repair_floor"]:
        from .floor import repair_floor
        floor_report = {}
        with t("repair_floor"):
            visual, floor_report["visual"] = repair_floor(visual, pcd, p["floor_cell"], p["floor_support_radius"])
            collision, floor_report["collision"] = repair_floor(collision, pcd, p["floor_cell"], p["floor_support_radius"])
        report.update(solution="baseline_floor_filtered" if filter_report is not None else "baseline_floor", floor=floor_report)
        collision_info["triangles"] = len(collision.triangles)
    report["collision_build"] = collision_info
    return visual, collision, report
