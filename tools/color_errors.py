"""Диагностика по эталону: зелёный ≤3 см, жёлтый 3–5 см, красный >5 см.

Создаёт visual_errors.ply и collision_errors.ply; конвейер эти файлы не читает.
Цвет измерен в вершинах и не заменяет оценку равномерными выборками по площади.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def colorize(mesh, scene):
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if not mesh.has_triangles():
        raise ValueError("Нужна непустая сетка")
    distance = scene.compute_distance(o3d.core.Tensor(points)).numpy()
    colors = np.tile([0.15, 0.70, 0.40], (len(points), 1))
    colors[distance > 0.03] = [1.0, 0.72, 0.10]
    colors[distance > 0.05] = [1.0, 0.10, 0.05]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gt", type=Path)
    ap.add_argument("outputs", nargs="+", type=Path)
    args = ap.parse_args()
    gt = o3d.io.read_triangle_mesh(str(args.gt))
    if not gt.has_triangles():
        ap.error("Эталон отсутствует или пуст")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
    for folder in args.outputs:
        for kind in ("visual", "collision"):
            mesh = colorize(o3d.io.read_triangle_mesh(str(folder / f"{kind}.ply")), scene)
            path = folder / f"{kind}_errors.ply"
            if not o3d.io.write_triangle_mesh(str(path), mesh):
                raise OSError(f"Не удалось сохранить {path}")
            print(path)


if __name__ == "__main__":
    main()
