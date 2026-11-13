"""Normalize an uploaded scan, then invoke the existing reconstruction CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar2sim.common import load_inputs
from lidar2sim.validation import load_trajectory


def normalize(points, scale, up):
    points = np.asarray(points).copy() * scale
    if up == 'Y':
        points = points[:, [0, 2, 1]]
        points[:, 1] *= -1
    return points


def main():
    folder = Path(sys.argv[1])
    settings = json.loads((folder/'settings.json').read_text())
    cloud, _, _ = load_inputs(folder/'source.ply')
    cloud.points = o3d.utility.Vector3dVector(normalize(cloud.points, settings['scale'], settings['up']))
    if settings['method'] == 'nksr' and cloud.has_normals():
        cloud.normals = o3d.utility.Vector3dVector(normalize(cloud.normals, 1, settings['up']))
    else:
        cloud.normals = o3d.utility.Vector3dVector(np.empty((0, 3)))
    if not o3d.io.write_point_cloud(str(folder/'input.ply'), cloud):
        raise OSError('Не удалось сохранить нормализованное облако')
    preview = cloud.uniform_down_sample(max(1, int(np.ceil(len(cloud.points)/100000))))
    if not o3d.io.write_point_cloud(str(folder/'.preview.ply'), preview):
        raise OSError('Не удалось сохранить предпросмотр')
    (folder/'.preview.ply').replace(folder/'preview.ply')
    cmd = [sys.executable, '-m', 'lidar2sim', '--input', str(folder/'input.ply'),
           '--config', str(folder/'config.yaml'), '--out', str(folder/'result')]
    if settings['method'] == 'nksr':
        cmd += ['--backend', 'nksr']
    trajectory = folder/'trajectory.csv'
    if trajectory.exists():
        xyz = normalize(load_trajectory(trajectory), settings['scale'], settings['up'])
        np.savetxt(folder/'normalized-trajectory.csv', np.column_stack([np.loadtxt(trajectory, delimiter=",", skiprows=1, usecols=0, ndmin=1), xyz]),
                   delimiter=',', header='t,x,y,z', comments='')
        cmd += ['--trajectory', str(folder/'normalized-trajectory.csv')]
    code = subprocess.call(cmd, env={**os.environ, 'LIDAR2SIM_WEB_PROGRESS': '1'})
    if code == 0:
        from lidar2sim.reporting import file_record, validate_report
        path = folder/'result/report.json'
        report = json.loads(path.read_text())
        report['web_upload'] = {'source': file_record(folder/'source.ply'),
                                'trajectory': file_record(trajectory) if trajectory.exists() else None,
                                'normalization': settings}
        validate_report(report)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print("WEB_ERROR " + json.dumps(str(exc), ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(2)
