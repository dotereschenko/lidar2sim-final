"""Headless CUDA physics for the reconstructed USD; produces measured poses, not RTX video."""
import argparse
import hashlib
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
from pxr import Usd, UsdGeom


@wp.kernel
def drive(vel: wp.array(dtype=wp.spatial_vector), force: wp.array(dtype=wp.spatial_vector),
          target: wp.vec3, mass: float):
    v = wp.spatial_top(vel[0])
    acceleration = (target-v)/0.02
    force[0] = wp.spatial_vector(wp.clamp(acceleration[0], -4.0, 4.0)*mass,
                                 wp.clamp(acceleration[1], -4.0, 4.0)*mass,
                                 0.0, 0.0, 0.0, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('scene', 'config', 'out'): p.add_argument('--'+key, type=Path, required=True)
    for key in ('route', 'start-index', 'end-index'): p.add_argument('--'+key, type=int, required=True)
    a = p.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.startswith('GPU-') or ',' in visible: p.error('One allocated GPU UUID required')
    cfg = json.loads(a.config.read_text())
    if not 0 <= a.route < len(cfg['routes']): p.error('Invalid route')
    route = cfg['routes'][a.route]['points']
    if not 0 <= a.start_index < a.end_index < len(route): p.error('Invalid start/goal')
    route = np.asarray(route[a.start_index:a.end_index+1], dtype=float)
    size, mass, speed = np.asarray(cfg['body']['size']), float(cfg['body']['mass']), float(cfg['speed'])
    assert np.isfinite(route).all() and (size > 0).all() and mass > 0 and speed > 0
    a.out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    stage = Usd.Stage.Open(str(a.scene))
    assert stage and UsdGeom.GetStageUpAxis(stage) == 'Z'
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    prim = stage.GetPrimAtPath('/World/Collision')
    mesh = UsdGeom.Mesh(prim)
    assert mesh and np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.asarray([matrix.Transform(x) for x in mesh.GetPointsAttr().Get()], dtype=np.float32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    assert len(indices) and np.isfinite(vertices).all()
    wp.init()
    with wp.ScopedDevice('cuda:0'):
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, indices, compute_inertia=False, is_solid=False))
        body = builder.add_body(xform=wp.transform(wp.vec3(*route[0], .65), wp.quat_identity()))
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=mass/float(np.prod(size)), mu=.3)
        builder.add_shape_box(body, hx=size[0]/2, hy=size[1]/2, hz=size[2]/2, cfg=shape_cfg)
        model = builder.finalize(device='cuda:0')
        assert model.device.is_cuda, 'CPU fallback forbidden'
        solver = newton.solvers.SolverXPBD(model, iterations=10)
        current, following = model.state(), model.state()
        control, contacts = model.control(), model.contacts()
        poses = []; target_index = 1; outcome = 'timeout'; max_contacts = 0
        last_progress, last_time = route[0].copy(), 2.0
        duration = min(180.0, float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())/speed+12)
        print('NEWTON_STAGE physics; collision mesh loaded; no synthetic ground', flush=True)
        for frame in range(int(duration*20)):
            q = current.body_q.numpy()[0]
            delta = route[target_index]-q[:2]
            if frame >= 40 and np.linalg.norm(delta) < .15:
                if target_index == len(route)-1: outcome = 'reached'
                else: target_index += 1; delta = route[target_index]-q[:2]
            velocity = delta/max(float(np.linalg.norm(delta)), 1e-9)*speed
            for _ in range(6):
                current.clear_forces()
                if frame >= 40:
                    wp.launch(drive, 1, inputs=[current.body_qd, current.body_f,
                              wp.vec3(*velocity, 0), mass], device='cuda:0')
                model.collide(current, contacts)
                solver.step(current, following, control, contacts, 1/120)
                current, following = following, current
            q = current.body_q.numpy()[0]
            assert np.isfinite(q).all(), 'Non-finite physical state'
            n = int(contacts.rigid_contact_count.numpy()[0]); max_contacts = max(max_contacts, n)
            t = (frame+1)/20
            poses.append({'t': t, 'position': q[:3].tolist(), 'quaternion_xyzw': q[3:].tolist(), 'contacts': n})
            if q[2] < -.3: outcome = 'fell'
            elif frame >= 40:
                if np.linalg.norm(q[:2]-last_progress) >= .05: last_progress, last_time = q[:2].copy(), t
                elif t-last_time > 5 and outcome != 'reached': outcome = 'blocked'
            if frame % 20 == 0: print(f'NEWTON_PROGRESS {t:.1f}s z={q[2]:.3f} contacts={n}', flush=True)
            if outcome != 'timeout': break
        report = dict(backend='newton', solver='XPBD', device=str(model.device),
                      device_name=model.device.name, gpu_uuid=visible,
                      scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
                      config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),
                      robot='velocity_body', route=a.route, start=a.start_index, end=a.end_index,
                      outcome=outcome, body_size_m=size.tolist(), body_mass_kg=mass,
                      wheel_dynamics=False, synthetic_ground=False, collision_modified=False,
                      physics_dt_s=1/120, fps=20, duration_s=poses[-1]['t'],
                      trajectory=poses, max_contacts=max_contacts, elapsed_s=time.monotonic()-began,
                      mode='demonstration_not_acceptance', video_available=False)
        (a.out/'recording.json').write_text(json.dumps(report, allow_nan=False))
        print(f'NEWTON_RESULT {outcome}; {len(poses)} measured frames', flush=True)


if __name__ == '__main__': main()
