"""Общие шаги конвейера: чтение входа, подготовка облака, экспорт и отчёт."""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .validation import (ContractError, load_trajectory, validate_manifest,
                         validate_mesh, validate_outputs, validate_point_cloud)


def announce_stage(stage):
    if os.environ.get("LIDAR2SIM_WEB_PROGRESS") == "1":
        print("LIDAR2SIM_STAGE " + stage, file=sys.stderr, flush=True)
from .input_contract import validate_limits


class Timer:
    """Время каждого шага попадает в отчёт."""

    def __init__(self):
        self.steps = {}

    def __call__(self, name):
        timer = self

        class _Step:
            def __enter__(self):
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                timer.steps[name] = round(time.perf_counter() - self.t0, 3)

        return _Step()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(ply, manifest=None, trajectory=None):
    if Path(ply).suffix.lower() != ".ply" or not Path(ply).is_file():
        raise ContractError(f"{ply}: требуется существующий PLY-файл")
    try:
        validate_limits(file_bytes=Path(ply).stat().st_size)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    try:
        meta = json.loads(Path(manifest).read_text()) if manifest else {"units": "m", "up_axis": "Z"}
    except json.JSONDecodeError as exc:
        raise ContractError(f"manifest.json: некорректный JSON, строка {exc.lineno}") from exc
    validate_manifest(meta)
    pcd = o3d.io.read_point_cloud(str(ply))
    validate_point_cloud(pcd, str(ply))
    traj = load_trajectory(trajectory) if trajectory else None
    return pcd, meta, traj


def preprocess(pcd, meta, traj, p, normal_diagnostics=None, *, preserve_normals=False):
    """Обрезка, выбросы, воксель, нормали к ближайшей точке траектории."""
    if "bounds_min" in meta and "bounds_max" in meta:
        box = o3d.geometry.AxisAlignedBoundingBox(meta["bounds_min"], meta["bounds_max"])
        pcd = pcd.crop(box)
        validate_point_cloud(pcd, "облако после обрезки")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=p["outlier_neighbors"],
                                            std_ratio=p["outlier_std"])
    pcd = pcd.voxel_down_sample(p["voxel"])
    validate_point_cloud(pcd, "облако после фильтрации и прореживания")
    if preserve_normals and pcd.has_normals():
        normals = np.asarray(pcd.normals)
        if not np.isfinite(normals).all() or (np.linalg.norm(normals, axis=1) < 1e-8).any():
            raise ContractError("NKSR: входные нормали некорректны или взаимно погашаются")
        pcd.normalize_normals()
        if normal_diagnostics is not None:
            normal_diagnostics.update(method="provided", orientation_verified=False)
    else:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=p["normal_radius"],
                                                                  max_nn=p["normal_max_nn"]))
        if preserve_normals and traj is None:
            pcd.orient_normals_consistent_tangent_plane(min(30, len(pcd.points) - 1))
            normals = np.asarray(pcd.normals).copy()
            inward = np.asarray(pcd.points).mean(axis=0) - np.asarray(pcd.points)
            if np.einsum("ij,ij->i", normals, inward).sum() < 0:
                pcd.normals = o3d.utility.Vector3dVector(-normals)
            if normal_diagnostics is not None:
                normal_diagnostics.update(method="estimated_consistent_tangent_plane",
                    orientation_verified=False,
                    warning="Без позиций сенсора направление поверхностей неоднозначно; требуется проверка")
        else:
            orient_normals(pcd, traj)
            if preserve_normals and normal_diagnostics is not None:
                normal_diagnostics.update(method="estimated_nearest_trajectory", orientation_verified=False)
    if p["visibility_normals"]:
        from .normals import correct_occluded_normals
        info = correct_occluded_normals(pcd, traj, p["voxel"])
        if normal_diagnostics is not None:
            normal_diagnostics.update(info)
    return pcd


def orient_normals(pcd, traj):
    """Разворачивает нормали к лидару. Без траектории — к центру облака (для помещения это внутрь)."""
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals).copy()
    if traj is not None and len(traj):
        _, idx = cKDTree(traj).query(P)
        view = traj[idx] - P
    else:
        view = P.mean(axis=0) - P
    flip = np.einsum("ij,ij->i", N, view) < 0
    N[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)


def poisson(pcd, depth, density_quantile):
    # n_threads=1: в Open3D 0.20 многопоточный Poisson падает с "Failed to close loop"
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,
                                                                           n_threads=1)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    return mesh


def clean(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def decimate(mesh, target):
    if len(mesh.triangles) > target:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
    return clean(mesh)


def export_usd(path, visual, collision):
    """scene.usd: метры, ось Z вверх, Visual без коллизии, Collision — статический коллайдер."""
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    def add(prim_path, mesh):
        V = np.asarray(mesh.vertices, dtype=np.float32)
        F = np.asarray(mesh.triangles, dtype=np.int32)
        m = UsdGeom.Mesh.Define(stage, prim_path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, dtype=np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(True)
        return m

    vis = add("/World/Visual", visual)
    if visual.has_vertex_colors():
        vis.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(visual.vertex_colors, dtype=np.float32)))
    col = add("/World/Collision", collision)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)   # в рендер не попадает
    if not stage.GetRootLayer().Save():
        raise ContractError("scene.usd: не удалось сохранить слой")


def write_outputs(out, visual, collision, report, started_at=None):
    from .reporting import acceptance, file_record

    export_started = time.perf_counter()
    out = Path(out)
    validate_mesh(visual, "visual")
    validate_mesh(collision, "collision")
    out.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {
        "visual_triangles": len(visual.triangles),
        "collision_triangles": len(collision.triangles),
    }
    with tempfile.TemporaryDirectory(dir=out, prefix=".lidar2sim-") as tmp:
        staged = Path(tmp)
        for name, mesh in (("visual.ply", visual), ("collision.ply", collision)):
            if not o3d.io.write_triangle_mesh(str(staged / name), mesh):
                raise ContractError(f"{name}: не удалось записать PLY")
        export_usd(staged / "scene.usd", visual, collision)
        report["status"] = "generated"
        if report.get("schema_version") == 1:
            report["artifacts"] = {}
            for name in ("visual.ply", "collision.ply", "scene.usd"):
                record = file_record(staged / name)
                record["path"] = name
                report["artifacts"][name] = record
        (staged / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        contract = validate_outputs(staged)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("visual.ply", "collision.ply", "scene.usd"):
            os.replace(staged / name, out / name)
        if report.get("schema_version") == 1:
            report["checks"]["contract"] = contract
            report["acceptance"] = acceptance(report["checks"], report["threshold_profile"])
            report["timings_s"]["export_and_validation"] = round(time.perf_counter() - export_started, 6)
        if started_at is not None:
            report["total_s"] = round(time.perf_counter() - started_at, 6)
        # Время включает экспорт, обратное чтение файлов и первичную запись отчёта.
        (staged / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        os.replace(staged / "report.json", out / "report.json")


def write_failure_report(out, error, elapsed):
    """Ошибка нового запуска заменяет прежнюю отметку успешного результата."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "status": "failed", "solution": "baseline", "total_s": round(elapsed, 3),
              "error": {"type": type(error).__name__, "message": str(error)}}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        os.replace(tmp, out / "report.json")
    finally:
        tmp.unlink(missing_ok=True)
