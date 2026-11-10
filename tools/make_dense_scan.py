"""Seeded dense LiDAR raycasting of GT for the 10M-point performance dataset.

This is a performance input, not the original noisy SLAM scan A or independent B.
"""
import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def generate(data, out, points=10_000_000, seed=20261110, noise_sigma=.02, batch_size=262144):
    if (type(points) is not int or points <= 0 or type(seed) is not int or seed < 0 or
            type(batch_size) is not int or batch_size <= 0 or not np.isfinite(noise_sigma) or noise_sigma < 0):
        raise ValueError('Expected positive point/batch counts, nonnegative seed and finite noise sigma')
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    source = json.loads((data/'manifest.json').read_text())
    if source['units'] != 'm' or source['up_axis'] != 'Z':
        raise ValueError('Source must use meters and Z-up')
    trajectory_path = data/'trajectory_A.csv'
    if trajectory_path.read_text().splitlines()[0].strip() != 't,x,y,z':
        raise ValueError('Trajectory header must be t,x,y,z')
    trajectory = np.loadtxt(trajectory_path, delimiter=',', skiprows=1, ndmin=2)
    if (trajectory.shape[1] != 4 or not len(trajectory) or not np.isfinite(trajectory).all()
            or (np.diff(trajectory[:, 0]) <= 0).any()):
        raise ValueError('Trajectory must contain finite t,x,y,z with strictly increasing time')
    gt_path = data/'gt_mesh.ply'
    mesh = o3d.io.read_triangle_mesh(str(gt_path))
    if not mesh.has_triangles():
        raise ValueError('Source GT has no triangles')
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rng = np.random.default_rng(seed)
    max_range = 20.0
    out.mkdir(parents=True)
    partial = out/'scene_10M.ply.partial'
    counts = np.zeros(len(trajectory), dtype=np.int64)
    written, attempted, hits_total = 0, 0, 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    noise_sum, noise_sq = 0.0, 0.0
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {points}\nproperty float x\nproperty float y\nproperty float z\nend_header\n').encode()
    with partial.open('wb') as stream:
        stream.write(header)
        while written < points:
            if attempted >= max(points * 100, batch_size * 10):
                raise ValueError('Too few ray hits: check trajectory, GT alignment and range')
            # Random directions and pose selection give independent returns, not repeated A points.
            pose_ids = rng.integers(len(trajectory), size=batch_size)
            origins = trajectory[pose_ids, 1:].astype(np.float32)
            azimuth = rng.uniform(-np.pi, np.pi, batch_size)
            sin_elevation = rng.uniform(-np.sqrt(.5), np.sqrt(.5), batch_size)
            cos_elevation = np.sqrt(1 - sin_elevation**2)
            directions = np.column_stack([cos_elevation*np.cos(azimuth), cos_elevation*np.sin(azimuth), sin_elevation]).astype(np.float32)
            rays = np.column_stack([origins, directions]).astype(np.float32)
            distances = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            valid = np.isfinite(distances) & (distances > .05) & (distances <= max_range)
            attempted += batch_size
            hits_total += int(valid.sum())
            ids = np.flatnonzero(valid)[:points-written]
            noise = rng.normal(0, noise_sigma, len(ids))
            cloud = origins[ids] + directions[ids] * (distances[ids] + noise)[:, None]
            cloud = cloud.astype('<f4')
            if not np.isfinite(cloud).all():
                raise ValueError('Non-finite raycast output')
            stream.write(cloud.tobytes())
            written += len(ids)
            counts += np.bincount(pose_ids[ids], minlength=len(counts))
            noise_sum += float(noise.sum())
            noise_sq += float(np.dot(noise, noise))
            if len(cloud):
                lo = np.minimum(lo, cloud.min(0))
                hi = np.maximum(hi, cloud.max(0))
            print(f'{written}/{points} points; {attempted} rays', flush=True)
    if partial.stat().st_size != len(header) + points * 12:
        raise ValueError('PLY byte count mismatch')
    result = out/'scene_10M.ply'
    partial.replace(result)
    shutil.copyfile(trajectory_path, out/'trajectory.csv')
    manifest = {
        'schema_version': 1, 'purpose': 'performance_only', 'units': 'm', 'up_axis': 'Z',
        'bounds_min': lo.tolist(), 'bounds_max': hi.tolist(), 'point_count': written,
        'source_bounds_min': source['bounds_min'], 'source_bounds_max': source['bounds_max'],
        'seed': seed, 'batch_size': batch_size, 'raycast_threads': 1,
        'sensor': {'max_range_m': max_range, 'min_range_m': .05, 'azimuth_deg': [-180, 180],
                   'elevation_deg': [-45, 45], 'elevation_sampling': 'uniform_sine',
                   'range_noise_sigma_m': noise_sigma, 'dropout': 0, 'drift': 0, 'ghost_fraction': 0},
        'range_noise_measured_mean_m': noise_sum / written,
        'range_noise_measured_sigma_m': float(np.sqrt(max(0, noise_sq/written - (noise_sum/written)**2))),
        'rays_attempted': attempted, 'valid_ray_hits_before_final_truncation': hits_total,
        'accepted_points_per_pose': counts.tolist(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'open3d': o3d.__version__},
        'source_gt_sha256': digest(gt_path), 'generator_sha256': digest(__file__),
        'sha256': {'scene_10M.ply': digest(result), 'trajectory.csv': digest(out/'trajectory.csv')},
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data')
    parser.add_argument('out')
    parser.add_argument('--points', type=int, default=10_000_000)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--noise-sigma', type=float, default=.02)
    args = parser.parse_args()
    generate(args.data, args.out, points=args.points, seed=args.seed, noise_sigma=args.noise_sigma)
