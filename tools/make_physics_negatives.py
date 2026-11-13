"""Build deliberately damaged GT fixtures for HACKATHON-12, never reconstruction input."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.common import export_usd


def remove_region(mesh, lo, hi):
    result = copy.deepcopy(mesh)
    triangles = np.asarray(result.vertices)[np.asarray(result.triangles)]
    remove = np.all((triangles.max(1) >= lo) & (triangles.min(1) <= hi), axis=1)
    if not remove.any():
        raise ValueError('Negative fixture region contains no triangles')
    result.remove_triangles_by_mask(remove)
    result.remove_unreferenced_vertices()
    return result


def main(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((data / 'test_config.json').read_text())
    gt = o3d.io.read_triangle_mesh(str(data / 'gt_mesh.ply'))
    if not gt.has_triangles():
        raise ValueError('Empty GT mesh')
    # Remove a patch under the first floor test, preserving walls above the slab.
    floor = cfg['floor'][0]
    x, y = floor['xy']
    missing_floor = remove_region(gt, [x - .8, y - .8, floor['floor_z'] - .3],
                                 [x + .8, y + .8, floor['floor_z'] + .04])
    # Clear the expected obstacle ahead of approach 0, leaving the ground intact.
    obstacle = cfg['obstacles'][0]
    direction = np.array(obstacle['dir'])
    support = np.dot(np.abs(direction), np.array(cfg['body']['size'][:2]) / 2)
    xy = np.array(obstacle['start']) + direction * (obstacle['free_m'] + support)
    missing_obstacle = remove_region(gt, [xy[0] - .7, xy[1] - .7, obstacle['floor_z'] + .04],
                                    [xy[0] + .7, xy[1] + .7, obstacle['floor_z'] + 2])
    # A new cross-corridor wall blocks route 0 after the initial waypoints.
    route = cfg['routes'][0]
    xy = np.array(route['points'][4])
    wall = o3d.geometry.TriangleMesh.create_box(4, .2, 1.2)
    wall.translate([xy[0] - 2, xy[1] - .1, route.get('floor_z', 0)])
    blocked_passage = gt + wall
    cases = [('missing_floor', missing_floor, 'floor', 0),
             ('missing_obstacle', missing_obstacle, 'obstacles', 0),
             ('blocked_passage', blocked_passage, 'routes', 0)]
    manifest = {'source_gt_sha256': hashlib.sha256((data / 'gt_mesh.ply').read_bytes()).hexdigest(),
                'source_gt_scene_sha256': hashlib.sha256((data / 'gt_scene.usd').read_bytes()).hexdigest(),
                'config_sha256': hashlib.sha256((data / 'test_config.json').read_bytes()).hexdigest(), 'cases': []}
    for name, mesh, group, index in cases:
        path = out / (name + '.usd')
        export_usd(path, mesh, mesh)
        manifest['cases'].append({'scene': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'expected_failure': {'group': group, 'index': index},
                                  'triangles': len(mesh.triangles)})
    (out / 'cases.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data')
    parser.add_argument('out')
    args = parser.parse_args()
    main(args.data, args.out)
