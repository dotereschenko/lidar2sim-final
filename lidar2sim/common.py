"""Общие шаги конвейера: чтение входа, подготовка облака, экспорт и отчёт."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


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
    pcd = o3d.io.read_point_cloud(str(ply))
    if not pcd.has_points():
        raise ValueError(f"{ply}: пустое облако")
    meta = json.loads(Path(manifest).read_text()) if manifest else {}
    traj = None
    if trajectory:
        # формат: заголовок, затем t,x,y,z[,...] — берём координаты
        traj = np.loadtxt(trajectory, delimiter=",", skiprows=1, ndmin=2)[:, 1:4]
    return pcd, meta, traj


def preprocess(pcd, meta, traj, p):
    """Обрезка, выбросы, воксель, нормали к ближайшей точке траектории."""
    if "bounds_min" in meta and "bounds_max" in meta:
        box = o3d.geometry.AxisAlignedBoundingBox(meta["bounds_min"], meta["bounds_max"])
        pcd = pcd.crop(box)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=p["outlier_neighbors"],
                                            std_ratio=p["outlier_std"])
    pcd = pcd.voxel_down_sample(p["voxel"])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=p["normal_radius"],
                                                              max_nn=p["normal_max_nn"]))
    orient_normals(pcd, traj)
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

    add("/World/Visual", visual)
    col = add("/World/Collision", collision)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)   # в рендер не попадает
    stage.GetRootLayer().Save()


def write_outputs(out, visual, collision, report):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out / "visual.ply"), visual)
    o3d.io.write_triangle_mesh(str(out / "collision.ply"), collision)
    export_usd(out / "scene.usd", visual, collision)
    report["outputs"] = {
        "visual_triangles": len(visual.triangles),
        "collision_triangles": len(collision.triangles),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
