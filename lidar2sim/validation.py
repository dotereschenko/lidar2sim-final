"""Проверки контракта входов и артефактов, без запуска реконструкции."""
import csv
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml
from .input_contract import validate_limits


class ContractError(ValueError):
    """Данные или результат не соответствуют файловому контракту."""


def validate_manifest(meta):
    if not isinstance(meta, dict):
        raise ContractError("manifest.json: ожидается JSON-объект")
    if meta.get("units") != "m" or meta.get("up_axis") != "Z":
        raise ContractError("manifest.json: требуются units='m' и up_axis='Z'")
    if ("bounds_min" in meta) != ("bounds_max" in meta):
        raise ContractError("manifest.json: bounds_min и bounds_max задаются вместе")
    if "bounds_min" in meta:
        try:
            lo = np.asarray(meta["bounds_min"], dtype=float)
            hi = np.asarray(meta["bounds_max"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ContractError("manifest.json: границы должны быть числами") from exc
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all():
            raise ContractError("manifest.json: границы — три конечных числа")
        if not (lo < hi).all():
            raise ContractError("manifest.json: bounds_min должен быть меньше bounds_max")
    return meta


def validate_params(params, defaults):
    if not isinstance(params, dict):
        raise ContractError("config.yaml: ожидается словарь параметров")
    unknown = set(params) - set(defaults)
    if unknown:
        raise ContractError("config.yaml: неизвестные параметры: " + ", ".join(map(repr, unknown)))
    result = {**defaults, **params}
    for key in ("repair_floor", "visibility_normals"):
        if type(result[key]) is not bool:
            raise ContractError(f"config.yaml: {key} must be boolean")
    for key in ("support_radius", "floor_support_radius"):
        value = result[key]
        if key == "support_radius" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ContractError(f"config.yaml: {key} must be positive and finite")
    if result["repair_floor"] and result["floor_distance"]:
        raise ContractError("config.yaml: select one floor reconstruction method")
    if result["support_radius"] is not None and result["support_distance"]:
        raise ContractError("config.yaml: select one support filtering method")
    for key in ("voxel", "outlier_std", "normal_radius", "density_quantile",
                "support_distance", "floor_distance", "floor_boundary_pad", "floor_cell", "floor_max_edge",
                "floor_max_slope_deg", "floor_clearance", "collision_floor_fraction",
                "collision_boundary_weight", "plane_distance", "plane_max_edge"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ContractError(f"config.yaml: {key} должен быть конечным числом")
        if key == "density_quantile":
            if not 0 <= value < 1:
                raise ContractError("config.yaml: density_quantile должен быть в [0, 1)")
        elif key == "collision_floor_fraction":
            if not 0 <= value < 1:
                raise ContractError("config.yaml: collision_floor_fraction должен быть в [0, 1)")
        elif key in ("support_distance", "floor_distance", "floor_boundary_pad", "plane_distance"):
            if value < 0:
                raise ContractError(f"config.yaml: {key} должен быть неотрицательным")
        elif key == "floor_max_slope_deg":
            if not 0 < value <= 20:
                raise ContractError("config.yaml: floor_max_slope_deg должен быть в (0, 20]")
        elif value <= 0:
            raise ContractError(f"config.yaml: {key} должен быть больше нуля")
        result[key] = float(value)
    if result["floor_boundary_pad"]:
        if not result["floor_distance"] or result["floor_boundary_pad"] > result["floor_support_radius"]:
            raise ContractError("config.yaml: floor_boundary_pad requires floor_distance and must be within floor_support_radius")
    for key, minimum in (("outlier_neighbors", 2), ("normal_max_nn", 3),
                         ("poisson_depth", 2), ("visual_triangles", 1), ("collision_triangles", 1),
                         ("floor_min_points", 3)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ContractError(f"config.yaml: {key} должен быть целым числом ≥ {minimum}")
        result[key] = int(value)
    if result["floor_max_edge"] < 2 * result["floor_cell"]:
        raise ContractError("config.yaml: floor_max_edge должен быть ≥ 2 * floor_cell")
    if result["floor_distance"] >= result["floor_clearance"]:
        raise ContractError("config.yaml: floor_distance должен быть меньше floor_clearance")
    return result


def load_config(path, defaults):
    if path is None:
        return validate_params({}, defaults)
    try:
        with Path(path).open(encoding="utf-8") as stream:
            params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ContractError(f"config.yaml: некорректный YAML ({exc.problem})") from exc
    return validate_params(params, defaults)


def load_trajectory(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
        has_data = any(line.strip() and not line.lstrip().startswith("#") for line in stream)
    if [cell.strip() for cell in header[:4]] != ["t", "x", "y", "z"]:
        raise ContractError("trajectory.csv: первые столбцы должны называться t,x,y,z")
    if not has_data:
        raise ContractError("trajectory.csv: нет позиций лидара")
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2, usecols=(0, 1, 2, 3))
    except ValueError as exc:
        raise ContractError("trajectory.csv: ожидаются числовые строки t,x,y,z") from exc
    if not len(values) or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ContractError("trajectory.csv: нужна хотя бы одна строка с конечными числами")
    if (np.diff(values[:, 0]) <= 0).any():
        raise ContractError("trajectory.csv: время должно строго возрастать")
    return values[:, 1:4]


def validate_point_cloud(pcd, label="scene.ply"):
    points = np.asarray(pcd.points)
    try:
        validate_limits(point_count=len(points))
    except ValueError as exc:
        raise ContractError(f'{label}: {exc}') from exc
    if not np.isfinite(points).all():
        raise ContractError(f"{label}: координаты содержат NaN или бесконечность")


def _validate_arrays(vertices, faces, label):
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ContractError(f"{label}: нет вершин XYZ")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ContractError(f"{label}: нет треугольников")
    if not np.isfinite(vertices).all():
        raise ContractError(f"{label}: неконечные координаты вершин")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ContractError(f"{label}: индекс треугольника вне массива вершин")
    triangles = vertices[faces]
    if (np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                triangles[:, 2] - triangles[:, 0]), axis=1) == 0).any():
        raise ContractError(f"{label}: вырожденный треугольник")


def validate_mesh(mesh, label):
    _validate_arrays(np.asarray(mesh.vertices), np.asarray(mesh.triangles), label)


def validate_outputs(directory):
    """Повторно читает четыре файла и проверяет геометрию, USD и отчёт."""
    import open3d as o3d
    from pxr import Usd, UsdGeom, UsdPhysics

    directory = Path(directory)
    for name in ("visual.ply", "collision.ply", "scene.usd", "report.json"):
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise ContractError(f"выход: отсутствует или пуст {name}")
    meshes = {}
    for name in ("visual", "collision"):
        mesh = o3d.io.read_triangle_mesh(str(directory / (name + ".ply")))
        validate_mesh(mesh, name + ".ply")
        meshes[name] = mesh
    try:
        stage = Usd.Stage.Open(str(directory / "scene.usd"))
    except Exception as exc:
        raise ContractError("scene.usd: файл не открывается через USD API") from exc
    if not stage:
        raise ContractError("scene.usd: USD API не создал stage")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ContractError("scene.usd: требуется ось Z вверх")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ContractError("scene.usd: единицы должны быть метрами")
    if str(stage.GetDefaultPrim().GetPath()) != "/World":
        raise ContractError("scene.usd: defaultPrim должен быть /World")
    for name in ("visual", "collision"):
        prim = stage.GetPrimAtPath("/World/" + name.title())
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ContractError(f"scene.usd: отсутствует Mesh /World/{name.title()}")
        usd_mesh = UsdGeom.Mesh(prim)
        points = usd_mesh.GetPointsAttr().Get()
        counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or counts is None or indices is None or not len(counts):
            raise ContractError(f"scene.usd: пустая геометрия {name}")
        if not np.all(np.asarray(counts) == 3) or len(indices) != 3 * len(counts):
            raise ContractError(f"scene.usd: {name} должен состоять из треугольников")
        vertices = np.asarray(points)
        faces = np.asarray(indices).reshape(-1, 3)
        _validate_arrays(vertices, faces, "scene.usd: " + name)
        expected = meshes[name]
        if (vertices.shape != np.asarray(expected.vertices).shape
                or not np.allclose(vertices, np.asarray(expected.vertices), rtol=1e-6, atol=1e-6)
                or not np.array_equal(faces, np.asarray(expected.triangles))):
            raise ContractError(f"scene.usd: геометрия {name} отличается от PLY")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetParent().HasAPI(UsdPhysics.RigidBodyAPI):
            raise ContractError("scene.usd: сцена должна иметь статические коллизии")
        collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        if name == "visual" and collision:
            raise ContractError("scene.usd: коллизии должны быть только на Collision")
        if name == "collision":
            if not collision or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise ContractError("scene.usd: коллайдер отсутствует или выключен")
            if (not prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                    or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none"):
                raise ContractError("scene.usd: нужен статический треугольный коллайдер")
            if usd_mesh.GetPurposeAttr().Get() != UsdGeom.Tokens.guide:
                raise ContractError("scene.usd: Collision должен иметь purpose=guide")
    try:
        report = json.loads((directory / "report.json").read_text())
        if isinstance(report, dict) and report.get("schema_version") is not None:
            from .reporting import validate_report
            validate_report(report)
        if not isinstance(report, dict) or report.get("status") == "failed":
            raise ValueError("отчёт не содержит успешную генерацию")
        if not isinstance(report.get("params"), dict):
            raise ValueError("нет эффективных параметров")
        if not isinstance(report.get("solution"), str):
            raise ValueError("нет названия решения")
        elapsed = report.get("total_s")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("некорректное время")
        digest = report.get("input", {}).get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("нет SHA256 входа")
        for name, mesh in meshes.items():
            if report.get("outputs", {}).get(name + "_triangles") != len(mesh.triangles):
                raise ValueError("число треугольников не совпадает с " + name)
        build = report.get("collision_build")
        if build is not None:
            if build.get("triangles") != len(meshes["collision"].triangles):
                raise ValueError("collision_build: число треугольников не совпадает с collision.ply")
            if build.get("triangles") > build.get("target_triangles", 0):
                raise ValueError("collision_build: превышен бюджет треугольников")
            if build.get("mode") == "protected_floor" and (
                    build.get("floor_triangles", 0) + build.get("other_triangles", 0)
                    != build["triangles"]):
                raise ValueError("collision_build: сумма компонент не совпадает с collision.ply")
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContractError(f"report.json: {exc}") from exc
    return {"status": "passed", "files": ["visual.ply", "collision.ply", "scene.usd", "report.json"]}


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Проверить файловый контракт PLY/USD/report.json")
    parser.add_argument("directory", help="папка с четырьмя выходными файлами")
    args = parser.parse_args()
    try:
        result = validate_outputs(args.directory)
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
