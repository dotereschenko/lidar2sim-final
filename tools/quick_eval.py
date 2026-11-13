"""Быстрые проверки на тестовой комнате. Не метрика задачи — удобно сравнивать свои версии.

python tools/quick_eval.py data/test_room out/baseline [out/my_version ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def scene(mesh):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def evaluate(data, out):
    gt = np.load(data / "gt_points.npy")
    t = json.loads((data / "manifest.json").read_text())["test"]
    vis = o3d.io.read_triangle_mesh(str(out / "visual.ply"))
    col = o3d.io.read_triangle_mesh(str(out / "collision.ply"))

    d = scene(vis).compute_distance(o3d.core.Tensor(gt.astype(np.float32))).numpy()
    samples = np.asarray(vis.sample_points_uniformly(100_000).points)
    extra, _ = cKDTree(gt).query(samples)

    # пол: луч вниз из сетки точек над полом, вне препятствий; попадание ±5 см от z=0
    L, W, _ = t["size"]
    xs, ys = np.meshgrid(np.arange(0.2, L - 0.2, 0.1), np.arange(0.2, W - 0.2, 0.1))
    xy = np.stack([xs.ravel(), ys.ravel()], 1)
    (bx0, by0, _), (bx1, by1, _) = t["box"]
    (cx, cy), r = t["column"]
    door = t["door"]
    free = ~((xy[:, 0] > bx0 - .1) & (xy[:, 0] < bx1 + .1) & (xy[:, 1] > by0 - .1) & (xy[:, 1] < by1 + .1))
    free &= (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2 > (r + .1) ** 2
    free &= np.abs(xy[:, 0] - door["x"]) > 0.15
    xy = xy[free]
    rays = np.hstack([xy, np.full((len(xy), 1), 0.5), np.tile([0, 0, -1], (len(xy), 1))])
    hit = scene(col).cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    floor_ok = np.abs(0.5 - hit) < 0.05
    (hx0, hy0), (hx1, hy1) = t["floor_hole"]
    in_hole = (xy[:, 0] > hx0) & (xy[:, 0] < hx1) & (xy[:, 1] > hy0) & (xy[:, 1] < hy1)

    # проём: горизонтальные лучи через проём вдоль x на высотах 0,3–1,7 м
    ys = np.linspace(door["y"][0] + 0.1, door["y"][1] - 0.1, 7)
    zs = np.linspace(0.3, 1.7, 8)
    Yd, Zd = np.meshgrid(ys, zs)
    rays = np.stack([np.full(Yd.size, door["x"] - 0.5), Yd.ravel(), Zd.ravel(),
                     np.ones(Yd.size), np.zeros(Yd.size), np.zeros(Yd.size)], 1)
    hit = scene(col).cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    door_blocked = (hit < 1.0).mean()

    rep = json.loads((out / "report.json").read_text())
    return {
        "версия": out.name,
        "эталон→сетка, медиана, см": round(float(np.median(d)) * 100, 2),
        "эталон→сетка, p95, см": round(float(np.percentile(d, 95)) * 100, 2),
        "сетка в 10 см от эталона, %": round(float((extra < 0.10).mean()) * 100, 1),
        "пол держит (лучи вниз), %": round(float(floor_ok.mean()) * 100, 1),
        "пол в пятне без точек, %": round(float(floor_ok[in_hole].mean()) * 100, 1),
        "проём перекрыт, %": round(float(door_blocked) * 100, 1),
        "треугольников visual / collision": f'{rep["outputs"]["visual_triangles"]} / {rep["outputs"]["collision_triangles"]}',
        "время, с": rep["total_s"],
    }


def main(data, *outs):
    rows = [evaluate(Path(data), Path(o)) for o in outs]
    for k in rows[0]:
        print(f"{k:34}" + "".join(f"{str(r[k]):>18}" for r in rows))


if __name__ == "__main__":
    main(*sys.argv[1:])
