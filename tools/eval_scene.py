"""Проверки геометрии на сцене хакатона по выданным данным: облаку A и точной геометрии сцены.

    python tools/eval_scene.py data/scene_a out/baseline [out/other ...]

1. Лишняя геометрия: точки, набранные равномерно по площади сетки, — доля в пределах 5 см
   от точной геометрии. Порог — не меньше 92%. Считается для visual и для collision.
2. Полнота: точки точной геометрии в области, которую видел A, — расстояние до сетки,
   медиана ≤ 3 см и p95 ≤ 10 см. Областью считается поверхность, у которой не меньше 10 точек A
   в радиусе 10 см: за то, чего в облаке нет, решение не отвечает.

На приёмке вторую проверку считаем по отдельному облаку B той же сцены, снятому с другого маршрута.
B участникам не выдаётся, поэтому здесь её заменяет точная геометрия: числа близкие, но не те же.
Контрольные размеры появятся вместе с разметкой сцены.
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

THRESHOLDS = {"extra_5cm_pct": 92.0, "median_cm": 3.0, "p95_cm": 10.0}
SAMPLES = 300_000


def raycasting(mesh):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return s


def region(data):
    """Точки точной геометрии в области, которую видел A. Считается один раз на набор данных."""
    gt = o3d.io.read_triangle_mesh(str(data / "gt_mesh.ply"))
    A = np.asarray(o3d.io.read_point_cloud(str(data / "A.ply")).points)
    pts = np.asarray(gt.sample_points_uniformly(SAMPLES).points)
    seen = cKDTree(A).query_ball_point(pts, 0.10, return_length=True) >= 10
    return gt, pts[seen], seen.mean()


def extra_within(mesh, gt_scene, dist):
    s = np.asarray(mesh.sample_points_uniformly(SAMPLES).points)
    d = gt_scene.compute_distance(o3d.core.Tensor(s.astype(np.float32))).numpy()
    return float((d < dist).mean()) * 100


def evaluate(out, gt_scene, ref):
    vis = o3d.io.read_triangle_mesh(str(out / "visual.ply"))
    col = o3d.io.read_triangle_mesh(str(out / "collision.ply"))
    d = raycasting(vis).compute_distance(o3d.core.Tensor(ref.astype(np.float32))).numpy()
    r = {
        "extra_5cm_pct": round(extra_within(vis, gt_scene, 0.05), 1),
        "extra_5cm_pct_collision": round(extra_within(col, gt_scene, 0.05), 1),
        "median_cm": round(float(np.median(d)) * 100, 2),
        "p95_cm": round(float(np.percentile(d, 95)) * 100, 2),
    }
    r["passed"] = {
        "лишняя геометрия, visual": r["extra_5cm_pct"] >= THRESHOLDS["extra_5cm_pct"],
        "лишняя геометрия, collision": r["extra_5cm_pct_collision"] >= THRESHOLDS["extra_5cm_pct"],
        "полнота, медиана": r["median_cm"] <= THRESHOLDS["median_cm"],
        "полнота, p95": r["p95_cm"] <= THRESHOLDS["p95_cm"],
    }
    return r


def main(data, *outs):
    o3d.utility.random.seed(0)   # выборка точек случайная: с зерном числа повторяются
    data = Path(data)
    gt, ref, share = region(data)
    gt_scene = raycasting(gt)
    print(f"оценка идёт по области, которую видел A: {share * 100:.0f}% всей поверхности сцены\n")
    rows = {Path(o).name: evaluate(Path(o), gt_scene, ref) for o in outs}
    labels = [
        ("extra_5cm_pct", f"сетка в 5 см от эталона, % (≥ {THRESHOLDS['extra_5cm_pct']:g})"),
        ("extra_5cm_pct_collision", "  то же для collision, %"),
        ("median_cm", f"эталон → сетка, медиана, см (≤ {THRESHOLDS['median_cm']:g})"),
        ("p95_cm", f"эталон → сетка, p95, см (≤ {THRESHOLDS['p95_cm']:g})"),
    ]
    w = max(12, *(len(n) + 2 for n in rows))
    print(f"{'':46}" + "".join(f"{n:>{w}}" for n in rows))
    for key, label in labels:
        print(f"{label:46}" + "".join(f"{r[key]:>{w}}" for r in rows.values()))
    print(f"{'все проверки пройдены':46}" + "".join(f"{('да' if all(r['passed'].values()) else 'нет'):>{w}}"
                                                  for r in rows.values()))
    for name, r in rows.items():
        (Path(outs[list(rows).index(name)]) / "eval_scene.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
