"""Build an asset-free static GT from a private box/trajectory recipe.

Keep hidden-scene recipes and outputs outside the participant repository.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.acceptance import load_profile, validate_physics_config
from lidar2sim.common import clean, export_usd, sha256


def trajectory(waypoints, spacing=.3, speed=.2):
    points = np.asarray(waypoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Need at least two finite XYZ waypoints')
    rows = [np.r_[0., points[0]]]; elapsed = 0.
    for start, end in zip(points[:-1], points[1:]):
        length = np.linalg.norm(end-start)
        if length <= 0: raise ValueError('Consecutive waypoints must differ')
        count = int(np.ceil(length/spacing))
        for i in range(1,count+1):rows.append(np.r_[elapsed+length*i/count/speed,start+(end-start)*i/count])
        elapsed += length/speed
    return np.asarray(rows)


def build(recipe, out, profile):
    recipe, out = Path(recipe), Path(out)
    if out.exists(): raise FileExistsError(out)
    spec = json.loads(recipe.read_text())
    if spec.get('schema_version') != 1 or spec.get('units') != 'm' or spec.get('up_axis') != 'Z':
        raise ValueError('Expected schema1, meters, Z up')
    rules, provenance = load_profile(profile)
    validate_physics_config(spec['physics'], rules)
    paths = {n:trajectory(spec['trajectories'][n]) for n in ['A','B']}
    if not spec['boxes']:raise ValueError('Empty geometry')
    mesh = o3d.geometry.TriangleMesh()
    for box in spec['boxes']:
        origin, size = np.asarray(box['origin'],float), np.asarray(box['size'],float)
        if origin.shape != (3,) or size.shape != (3,) or not np.isfinite([origin,size]).all() or (size<=0).any():
            raise ValueError('Box needs finite XYZ origin and positive size')
        mesh += o3d.geometry.TriangleMesh.create_box(*size).translate(origin)
    mesh = clean(mesh)
    out.mkdir(parents=True,mode=0o700)
    o3d.io.write_triangle_mesh(str(out/'gt_mesh.ply'),mesh)
    export_usd(out/'gt_scene.usd',mesh,mesh)
    for n,path in paths.items():np.savetxt(out/f'trajectory_{n}.csv',path,delimiter=',',header='t,x,y,z',comments='',fmt='%.9f')
    for name,key in [('test_config.json','physics'),('definition.json','annotations')]:
        (out/name).write_text(json.dumps(spec[key],indent=2)+'\n')
    manifest={'schema_version':1,'units':'m','up_axis':'Z','source':'private_box_recipe',
              'recipe_sha256':sha256(recipe),'builder_sha256':sha256(__file__),'profile':provenance,
              'sha256':{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recipe');p.add_argument('out');p.add_argument('--profile',required=True)
    a=p.parse_args();print(json.dumps(build(a.recipe,a.out,a.profile)['sha256'],indent=2))
