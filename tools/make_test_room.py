"""Синтетическая комната для проверки конвейера без данных хакатона.

Помещение 6×4×2,7 м, перегородка с проёмом 0,9×2,0 м, ящик и круглая колонна.
Шум дальности ~2 см вдоль луча от траектории, 5% случайных выпадений и пятно пола,
которое лидар не видел. Это не замена сцене хакатона: только проверка, что код работает.

python tools/make_test_room.py data/test_room
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

rng = np.random.default_rng(0)
STEP = 0.02
L, W, H = 6.0, 4.0, 2.7
WALL_X, WALL_T = 4.0, 0.10                 # перегородка
DOOR_Y, DOOR_H = (1.55, 2.45), 2.0
BOX = ((1.2, 0.8, 0.0), (2.0, 1.4, 0.75))  # min, max
COL = ((2.6, 3.0), 0.20)                   # центр xy, радиус
HOLE = ((2.9, 1.8), (3.5, 2.4))            # пятно пола без точек


def rect(o, a, b):
    """Точки на прямоугольнике o + s*a + t*b, s,t ∈ [0,1]."""
    na, nb = max(int(np.linalg.norm(a) / STEP), 1), max(int(np.linalg.norm(b) / STEP), 1)
    s, t = np.meshgrid(np.linspace(0, 1, na), np.linspace(0, 1, nb))
    return np.asarray(o) + s.reshape(-1, 1) * a + t.reshape(-1, 1) * b


def surfaces():
    X, Y, Z = np.eye(3)
    parts = [
        rect((0, 0, 0), L * X, W * Y),           # пол
        rect((0, 0, H), L * X, W * Y),           # потолок
        rect((0, 0, 0), L * X, H * Z), rect((0, W, 0), L * X, H * Z),
        rect((0, 0, 0), W * Y, H * Z), rect((L, 0, 0), W * Y, H * Z),
    ]
    for x in (WALL_X - WALL_T / 2, WALL_X + WALL_T / 2):      # перегородка, две стороны
        parts += [rect((x, 0, 0), DOOR_Y[0] * Y, H * Z),
                  rect((x, DOOR_Y[1], 0), (W - DOOR_Y[1]) * Y, H * Z),
                  rect((x, DOOR_Y[0], DOOR_H), (DOOR_Y[1] - DOOR_Y[0]) * Y, (H - DOOR_H) * Z)]
    parts += [rect((WALL_X - WALL_T / 2, DOOR_Y[0], DOOR_H), WALL_T * X, (DOOR_Y[1] - DOOR_Y[0]) * Y),
              rect((WALL_X - WALL_T / 2, DOOR_Y[0], 0), WALL_T * X, DOOR_H * Z),
              rect((WALL_X - WALL_T / 2, DOOR_Y[1], 0), WALL_T * X, DOOR_H * Z)]
    (x0, y0, z0), (x1, y1, z1) = BOX
    parts += [rect((x0, y0, z1), (x1 - x0) * X, (y1 - y0) * Y),
              rect((x0, y0, 0), (x1 - x0) * X, z1 * Z), rect((x0, y1, 0), (x1 - x0) * X, z1 * Z),
              rect((x0, y0, 0), (y1 - y0) * Y, z1 * Z), rect((x1, y0, 0), (y1 - y0) * Y, z1 * Z)]
    (cx, cy), r = COL
    a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / STEP), endpoint=False)
    z = np.arange(0, H, STEP)
    A, Zc = np.meshgrid(a, z)
    parts.append(np.stack([cx + r * np.cos(A.ravel()), cy + r * np.sin(A.ravel()), Zc.ravel()], 1))
    return np.concatenate(parts)


def hidden(p):
    """Поверхность, которую лидар не видит: пол под ящиком и колонной, пятно пола."""
    (x0, y0, _), (x1, y1, _) = BOX
    on_floor = p[:, 2] < 1e-6
    under_box = on_floor & (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    (cx, cy), r = COL
    under_col = on_floor & ((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2 < r * r)
    (hx0, hy0), (hx1, hy1) = HOLE
    hole = on_floor & (p[:, 0] > hx0) & (p[:, 0] < hx1) & (p[:, 1] > hy0) & (p[:, 1] < hy1)
    return under_box | under_col, hole


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gt = surfaces()
    covered, hole = hidden(gt)
    gt_visible = gt[~covered]                       # эталон: всё, что физически можно увидеть
    np.save(out / "gt_points.npy", gt_visible)

    traj = np.stack([np.linspace(1.0, 5.2, 50), np.full(50, 2.0), np.full(50, 1.2)], 1)
    pts = gt[~covered & ~hole]
    pts = pts[rng.random(len(pts)) > 0.05]          # случайные выпадения
    nearest = traj[np.argmin(((pts[:, None, :2] - traj[None, :, :2]) ** 2).sum(-1), 1)]
    ray = pts - nearest
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    pts = pts + ray * rng.normal(0, 0.02, (len(pts), 1))   # шум дальности вдоль луча

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    o3d.io.write_point_cloud(str(out / "scene.ply"), pcd)
    np.savetxt(out / "trajectory.csv", np.column_stack([np.arange(len(traj)) * 0.1, traj]),
               delimiter=",", header="t,x,y,z", comments="")
    (out / "manifest.json").write_text(json.dumps({
        "units": "m", "up_axis": "Z",
        "bounds_min": [-0.2, -0.2, -0.2], "bounds_max": [L + 0.2, W + 0.2, H + 0.2],
        "source": "tools/make_test_room.py", "noise_range_sigma_m": 0.02, "dropout": 0.05,
        "test": {"door": {"x": WALL_X, "y": DOOR_Y, "h": DOOR_H}, "box": BOX,
                 "column": COL, "floor_hole": HOLE, "size": [L, W, H]},
    }, indent=2))
    print(f"{len(pts)} точек → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/test_room")
