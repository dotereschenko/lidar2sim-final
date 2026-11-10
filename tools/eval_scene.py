"""Диагностика геометрии на A и GT (не независимая приёмка по B).

    python tools/eval_scene.py data/scene_a out/baseline [out/other ...] [--profile rules.yaml]

Полнота visual измеряется на GT в области с ≥10 точками A в радиусе 10 см.
Полнота collision, B, размеры, физика и остальные группы PoC здесь не проверяются.
"""
import argparse
import json
import sys
from pathlib import Path

# Source-checkout scripts also run by absolute path outside the repository cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.acceptance import geometry_checks, load_profile, overall_status

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def read_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if (not len(vertices) or not len(triangles) or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)
            or not np.isfinite(mesh.get_surface_area()) or mesh.get_surface_area() <= 0):
        raise ValueError(f'empty or invalid mesh: {path}')
    return mesh


def raycasting(mesh):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def sample(mesh, profile):
    g = profile['geometry']
    o3d.utility.random.seed(g['seed'])
    return np.asarray(mesh.sample_points_uniformly(g['samples']).points)


def region(data, profile):
    """GT sample in the observed A region; the coverage rule is diagnostic only."""
    gt = read_mesh(data / 'gt_mesh.ply')
    points_a = np.asarray(o3d.io.read_point_cloud(str(data / 'A.ply')).points)
    if not len(points_a) or not np.isfinite(points_a).all():
        raise ValueError('A point cloud must be nonempty and finite')
    pts = sample(gt, profile)
    seen = cKDTree(points_a).query_ball_point(pts, 0.10, return_length=True) >= 10
    if not seen.any():
        raise ValueError('empty reference region observed by A')
    return gt, pts[seen], float(seen.mean())


def extra_within(mesh, gt_scene, profile):
    points = sample(mesh, profile)
    distances = gt_scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()
    if not np.isfinite(distances).all():
        raise ValueError('nonfinite precision distances')
    return float((distances <= profile['geometry']['precision_distance_m']).mean()) * 100


def evaluate(out, gt_scene, ref, profile, provenance):
    if not len(ref) or not np.isfinite(ref).all():
        raise ValueError('reference must be nonempty and finite')
    vis = read_mesh(out / 'visual.ply')
    col = read_mesh(out / 'collision.ply')
    distances = raycasting(vis).compute_distance(o3d.core.Tensor(ref.astype(np.float32))).numpy()
    if not np.isfinite(distances).all():
        raise ValueError('nonfinite completeness distances')
    raw = {
        'extra_5cm_pct': extra_within(vis, gt_scene, profile),
        'extra_5cm_pct_collision': extra_within(col, gt_scene, profile),
        'median_cm': float(np.median(distances)) * 100,
        'p95_cm': float(np.percentile(distances, 95, method='linear')) * 100,
    }
    passed = geometry_checks(raw, profile)
    status = 'passed' if all(passed.values()) else 'failed'
    return {
        **{key: round(value, 1 if key.startswith('extra_') else 2) for key, value in raw.items()},
        'passed': passed, 'raw_metrics': raw, 'acceptance_profile': provenance,
        'evaluation_mode': 'scene_a_gt_diagnostic',
        'coverage': {'collision_completeness': 'not_run'},
        'geometry_status': status,
        'poc_status': overall_status({'implemented_geometry': status, 'remaining_groups': 'not_run'}),
    }


def main(data, *outs, profile=None):
    rules, provenance = load_profile(profile)
    gt, ref, share = region(Path(data), rules)
    gt_scene = raycasting(gt)
    print(f'диагностика A/GT: видимая область {share * 100:.0f}% поверхности сцены')
    rows = [(Path(out), evaluate(Path(out), gt_scene, ref, rules, provenance)) for out in outs]
    for out, result in rows:
        (out / 'eval_scene.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(f'{out}: {result["extra_5cm_pct"]}% visual, {result["extra_5cm_pct_collision"]}% collision; '
              f'медиана {result["median_cm"]} см, p95 {result["p95_cm"]} см; '
              f'четыре реализованные проверки: {result["geometry_status"]}; PoC: {result["poc_status"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data')
    ap.add_argument('outs', nargs='+')
    ap.add_argument('--profile')
    args = ap.parse_args()
    try:
        main(args.data, *args.outs, profile=args.profile)
    except (ValueError, OSError) as exc:
        ap.exit(1, f'evaluation error: {exc}\n')
