"""Generate independent candidate A/B acquisitions; B is evaluator-only."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.make_dense_scan import digest


def sector_mask(azimuth, heading, sectors):
    relative = (azimuth-heading+np.pi) % (2*np.pi)-np.pi
    masked = np.zeros(np.shape(relative), dtype=bool)
    for start, end in sectors:
        masked |= (relative >= np.deg2rad(start)-1e-12) & (relative <= np.deg2rad(end)+1e-12)
    return masked


def load_path(path):
    path = Path(path)
    if path.read_text().splitlines()[0].strip() != 't,x,y,z':
        raise ValueError('Expected trajectory header t,x,y,z')
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if values.shape[1] != 4 or len(values) < 2 or not np.isfinite(values).all() or (np.diff(values[:,0]) <= 0).any():
        raise ValueError('Need at least two finite poses with strictly increasing times')
    return values


def generate(data, out, points=750000, seed_a=1301, seed_b=1302, batch_size=65536,
             trajectory_path=None, trajectory_b_path=None, sectors_a=None):
    if (type(points) is not int or points <= 0 or type(batch_size) is not int or batch_size <= 0
            or any(type(s) is not int or s < 0 for s in (seed_a, seed_b)) or seed_a == seed_b):
        raise ValueError('Positive integer counts and distinct nonnegative seeds required')
    sectors_a = [] if sectors_a is None else sectors_a
    if any(len(pair) != 2 or not all(type(v) in (float,int) and np.isfinite(v) for v in pair)
           or not -180 <= pair[0] < pair[1] <= 180 for pair in sectors_a):
        raise ValueError('Sector intervals must satisfy -180 <= start < end <= 180')
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    source = json.loads((data/'manifest.json').read_text())
    if source['units'] != 'm' or source['up_axis'] != 'Z':
        raise ValueError('Source must use meters and Z-up')
    path = Path(trajectory_path) if trajectory_path is not None else data/'trajectory_A.csv'
    ta = load_path(path)
    if trajectory_b_path is not None:
        tb = load_path(trajectory_b_path)
    else:
        tb = (ta[:-1] + ta[1:]) / 2
        tb[:, 3] += .10
    # Independent viewpoints on the same route, not a transformation of A returns.
    if cKDTree(ta[:, 1:]).query(tb[:, 1:])[0].min() < .05:
        raise ValueError('B viewpoints must be distinct from A')
    mesh = o3d.io.read_triangle_mesh(str(data/'gt_mesh.ply'))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Finite nonempty GT triangles required')
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for trajectory in (ta, tb):
        xyz = trajectory[:, 1:].astype(np.float32)
        clearance = scene.compute_distance(o3d.core.Tensor(xyz), nthreads=1).numpy()
        if clearance.min() < .05:
            raise ValueError('Trajectory pose has less than 5 cm surface clearance')
        delta = np.diff(xyz, axis=0)
        length = np.linalg.norm(delta, axis=1)
        moving = length > 1e-6
        if moving.any():
            rays = np.column_stack([xyz[:-1][moving], delta[moving]/length[moving, None]])
            hit = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            if (hit < length[moving] - 1e-4).any():
                raise ValueError('Trajectory segment crosses GT geometry')
    # One fixed, uniform-area GT sample set for comparable diagnostic coverage.
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    if not np.isfinite(area.sum()) or area.sum() <= 0:
        raise ValueError('GT must have positive finite surface area')
    coverage_rng = np.random.default_rng(1318)
    chosen = tri[coverage_rng.choice(len(tri), 30000, p=area/area.sum())]
    u = np.sqrt(coverage_rng.random(30000))[:, None]
    v = coverage_rng.random(30000)[:, None]
    samples = (1-u)*chosen[:, 0] + u*(1-v)*chosen[:, 1] + u*v*chosen[:, 2]
    out.mkdir(parents=True)
    manifest = {
        'schema_version': 1, 'purpose': 'candidate_independent_scans', 'units': 'm', 'up_axis': 'Z',
        'acceptance_status': 'pending_HACKATHON-3_A_count_decision',
        'B_usage': 'evaluation_only', 'batch_size': batch_size, 'raycast_threads': 1,
        'trajectory_B_derivation': 'explicit independent trajectory' if trajectory_b_path is not None else 'adjacent A pose midpoints; sensor height +0.10 m',
        'source_trajectory_B_sha256': digest(trajectory_b_path) if trajectory_b_path is not None else None,
        'viewpoint_separation_m': {'minimum': float(cKDTree(ta[:,1:]).query(tb[:,1:])[0].min()),
                                   'median': float(np.median(cKDTree(ta[:,1:]).query(tb[:,1:])[0]))},
        'source_gt_sha256': digest(data/'gt_mesh.ply'), 'source_trajectory_sha256': digest(path),
        'generator_sha256': digest(__file__), 'digest_helper_sha256': digest(Path(__file__).with_name('make_dense_scan.py')),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'open3d': o3d.__version__},
        'coverage': {'method': 'uniform_area_GT_samples_to_cloud', 'samples': 30000, 'seed': 1318,
                     'distance_m': .10, 'scope': 'whole_GT_diagnostic_not_common_eval_region'},
        'scans': {}, 'sha256': {},
    }
    for name, trajectory, seed, sigma, dropout in [('A', ta, seed_a, .02, .05), ('B', tb, seed_b, .01, 0.)]:
        rng = np.random.default_rng(seed)
        clouds, noises = [], []
        counts = np.zeros(len(trajectory), dtype=np.int64)
        written = attempted = valid_total = dropped_total = sector_total = available_total = 0
        sectors = sectors_a if name == 'A' else []
        tangent = np.gradient(trajectory[:,1:3], axis=0) if len(trajectory) > 1 else np.zeros((1,2))
        headings = np.arctan2(tangent[:,1], tangent[:,0])
        while written < points:
            if attempted >= max(points*100, batch_size*10):
                raise ValueError('Too few ray hits; check GT/trajectory alignment')
            pose_ids = rng.integers(len(trajectory), size=batch_size)
            origins = trajectory[pose_ids, 1:].astype(np.float32)
            azimuth = rng.uniform(-np.pi, np.pi, batch_size)
            z = rng.uniform(-np.sqrt(.5), np.sqrt(.5), batch_size)
            xy = np.sqrt(1-z*z)
            directions = np.column_stack([xy*np.cos(azimuth), xy*np.sin(azimuth), z]).astype(np.float32)
            rays = np.column_stack([origins, directions])
            distance = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            ids = np.flatnonzero(np.isfinite(distance) & (distance > .05) & (distance <= 20))
            valid_total += len(ids)
            blocked = sector_mask(azimuth[ids], headings[pose_ids[ids]], sectors)
            sector_total += int(blocked.sum())
            ids = ids[~blocked]
            available_total += len(ids)
            dropped = rng.random(len(ids)) < dropout
            attempted += batch_size
            dropped_total += int(dropped.sum())
            ids = ids[~dropped][:points-written]
            noise = rng.normal(0, sigma, len(ids))
            cloud = (origins[ids] + directions[ids]*(distance[ids]+noise)[:, None]).astype('<f4')
            if not np.isfinite(cloud).all():
                raise ValueError('Non-finite acquisition')
            clouds.append(cloud)
            noises.append(noise)
            written += len(ids)
            counts += np.bincount(pose_ids[ids], minlength=len(trajectory))
        cloud, noise = np.concatenate(clouds), np.concatenate(noises)
        header = ('ply\nformat binary_little_endian 1.0\n'
                  f'element vertex {points}\nproperty float x\nproperty float y\nproperty float z\nend_header\n').encode()
        with (out/f'{name}.ply').open('wb') as stream:
            stream.write(header)
            stream.write(cloud.tobytes())
        np.savetxt(out/f'trajectory_{name}.csv', trajectory, delimiter=',', header='t,x,y,z', comments='', fmt='%.9f')
        distances = scene.compute_distance(o3d.core.Tensor(cloud), nthreads=1).numpy()
        coverage = cKDTree(cloud).query(samples, workers=1)[0] <= .10
        manifest['scans'][name] = {
            'point_count': len(cloud), 'seed': seed,
            'sensor': {'range_m': [.05, 20], 'azimuth_deg': [-180, 180], 'elevation_deg': [-45, 45],
                       'elevation_sampling': 'uniform_sine', 'range_noise_sigma_m': sigma,
                       'random_dropout_fraction': dropout, 'sector_dropout_deg': sectors,
                       'sector_frame': 'azimuth relative to XY trajectory tangent',
                       'heading_rad_per_pose': headings.tolist(),
                       'drift_m_per_s': [0, 0, 0], 'ghost_fraction': 0},
            'range_noise_mean_m': float(noise.mean()), 'range_noise_sigma_m': float(noise.std()),
            'cloud_to_gt_median_m': float(np.median(distances)), 'cloud_to_gt_p95_m': float(np.percentile(distances, 95)),
            'gt_area_sample_coverage_fraction': float(coverage.mean()),
            'rays_attempted': attempted, 'valid_returns_before_dropout': valid_total,
            'sector_dropped_returns': sector_total, 'sector_dropout_fraction_measured': sector_total/valid_total,
            'valid_returns_after_sector_dropout': available_total,
            'random_dropped_returns': dropped_total, 'random_dropout_fraction_measured': dropped_total/available_total if available_total else 0,
            'random_dropout_denominator': 'valid_returns_after_sector_dropout',
            'accepted_points_per_pose': counts.tolist(),
            'bounds_min': cloud.min(0).tolist(), 'bounds_max': cloud.max(0).tolist(),
        }
        for file in (f'{name}.ply', f'trajectory_{name}.csv'):
            manifest['sha256'][file] = digest(out/file)
        print(name, json.dumps({k: v for k, v in manifest['scans'][name].items() if k not in ['sensor', 'accepted_points_per_pose']}), flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data')
    parser.add_argument('out')
    parser.add_argument('--trajectory-a', help='Explicit corrected candidate trajectory; original remains unchanged')
    parser.add_argument('--trajectory-b', help='Independent B path; omit only for legacy midpoint acquisition')
    parser.add_argument('--sector-a', nargs=2, type=float, action='append', default=[], metavar=('START_DEG','END_DEG'))
    parser.add_argument('--points', type=int, default=750000)
    parser.add_argument('--seed-a', type=int, default=1301)
    parser.add_argument('--seed-b', type=int, default=1302)
    args = parser.parse_args()
    generate(args.data, args.out, args.points, args.seed_a, args.seed_b, trajectory_path=args.trajectory_a, trajectory_b_path=args.trajectory_b, sectors_a=args.sector_a)
