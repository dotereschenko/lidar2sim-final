"""Prove CUDA gravity/contact execution before enabling a Newton UI backend.

Run in the isolated deployment/requirements-newton.txt environment. No renderer,
Isaac runtime, GPU reset, or system configuration changes are needed.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

# Validate allocation before importing Newton, which initializes Warp devices.
_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if not _visible.startswith('GPU-') or ',' in _visible:
    raise RuntimeError('Set CUDA_VISIBLE_DEVICES to one explicitly allocated GPU UUID')

import numpy as np
import newton
import warp as wp


def drop(ground: bool):
    builder = newton.ModelBuilder()
    if ground:
        builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 2), wp.quat_identity()))
    builder.add_shape_box(body, hx=.2, hy=.2, hz=.2)
    model = builder.finalize(device='cuda:0')
    assert model.device.is_cuda, 'CPU fallback is forbidden'
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    current, following = model.state(), model.state()
    control, contacts = model.control(), model.contacts()
    positions = []
    max_contacts = 0
    for step in range(480):
        current.clear_forces()
        model.collide(current, contacts)
        solver.step(current, following, control, contacts, 1/120)
        current, following = following, current
        if step % 12 == 11:
            positions.append(current.body_q.numpy()[0, :3].tolist())
            max_contacts = max(max_contacts, int(contacts.rigid_contact_count.numpy()[0]))
    positions = np.asarray(positions)
    assert np.isfinite(positions).all(), 'Non-finite body position'
    return {'final_z_m': float(positions[-1, 2]),
            'min_z_m': float(positions[:, 2].min()),
            'max_contacts': max_contacts, 'positions_xyz_m': positions.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible:
        parser.error('Set CUDA_VISIBLE_DEVICES to one explicitly allocated GPU UUID')
    start = time.monotonic()
    wp.init()
    device = wp.get_device('cuda:0')
    assert device.is_cuda, 'CUDA device unavailable'
    with wp.ScopedDevice(device):
        ground = drop(True)
        freefall = drop(False)
        wall = hit_wall(True)
        no_wall = hit_wall(False)
    assert .17 <= ground['final_z_m'] <= .23, 'Body did not settle on the floor'
    assert ground['min_z_m'] >= .15, 'Excessive floor penetration'
    assert ground['max_contacts'] > 0, 'No physical contacts recorded'
    assert freefall['final_z_m'] < -50, 'Negative control did not fall freely'
    assert freefall['max_contacts'] == 0, 'Unexpected collision in negative control'
    assert wall['final_x_m'] < -.25, 'Body passed through the wall'
    assert no_wall['final_x_m'] > .8, 'Removed-wall control did not pass'
    report = {'status': 'passed', 'backend': 'newton', 'solver': 'XPBD',
              'device': str(device), 'device_name': device.name, 'gpu_uuid': visible,
              'versions': {name: importlib.metadata.version(name)
                           for name in ('newton', 'warp-lang', 'numpy', 'newton-actuators')},
              'physics_dt_s': 1/120, 'steps_per_case': 480,
              'elapsed_s': time.monotonic()-start,
              'ground': ground, 'negative_control': freefall,
              'wall': wall, 'removed_wall': no_wall,
              'limitations': ['primitive box and plane only', 'no scene A acceptance',
                              'no wheel dynamics', 'no rendering or UI integration']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('ground', 'negative_control')}, indent=2))


def hit_wall(present):
    builder = newton.ModelBuilder()
    if present: builder.add_shape_box(-1, hx=.1, hy=2, hz=10)
    body = builder.add_body(xform=wp.transform(wp.vec3(-1, 0, 1), wp.quat_identity()))
    builder.add_shape_box(body, hx=.2, hy=.2, hz=.2)
    model = builder.finalize(device='cuda:0')
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    current, following = model.state(), model.state()
    current.body_qd.assign(np.asarray([[2, 0, 0, 0, 0, 0]], dtype=np.float32))
    control, contacts = model.control(), model.contacts()
    for _ in range(120):
        current.clear_forces(); model.collide(current, contacts)
        solver.step(current, following, control, contacts, 1/120)
        current, following = following, current
    return {'final_x_m': float(current.body_q.numpy()[0, 0]), 'steps': 120}


if __name__ == '__main__':
    main()
