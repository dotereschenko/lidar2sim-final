"""Build pinned candidate dimensions, door rectangles and common A/B region."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar2sim.annotations import observed_by_both, dimension, door, require_observed_dimensions
from tools.make_dense_scan import digest
from tools.eval_scene import read_mesh, raycasting


def build(gt_data, scans, out, definition, physics_evidence):
    gt_data, scans, out = map(Path, (gt_data, scans, out))
    definition, physics_evidence = Path(definition), Path(physics_evidence)
    if out.exists():
        raise FileExistsError(out)
    spec = json.loads(definition.read_text())
    manifest = json.loads((scans/'manifest.json').read_text())
    evidence = json.loads(physics_evidence.read_text())
    if any(x.get('units') != 'm' or x.get('up_axis') != 'Z' for x in (spec,manifest)):
        raise ValueError('Expected meters and Z-up')
    if spec['schema_version'] != 1 or len(spec['dimensions']) != 10 or len(spec['doors']) < 2:
        raise ValueError('Need schema v1, ten dimensions and at least two doors')
    if len({v['id'] for v in spec['dimensions']}) != 10:
        raise ValueError('Dimension IDs must be unique')
    for key in ('samples','seed'):
        if type(spec[key]) is not int or spec[key] < (1 if key=='samples' else 0):
            raise ValueError(f'Invalid {key}')
    minimum = spec['region']['min_common_samples']
    if type(minimum) is not int or not 1 <= minimum <= spec['samples']:
        raise ValueError('Invalid minimum common sample count')
    for name in ('A.ply','B.ply','trajectory_A.csv','trajectory_B.csv'):
        if digest(scans/name) != manifest['sha256'][name]:
            raise ValueError(f'Scan manifest hash mismatch: {name}')
    if digest(gt_data/'gt_mesh.ply') != manifest['source_gt_sha256']:
        raise ValueError('GT differs from acquisition source')
    physical = evidence['cases']['gt']
    if (not evidence['qualification']['passed'] or not physical['summary']['passed']
            or digest(gt_data/'gt_scene.usd') != physical['inputs']['scene_sha256']
            or digest(gt_data/'test_config.json') != physical['inputs']['config_sha256']):
        raise ValueError('Physics evidence is failed or refers to different inputs')
    config = json.loads((gt_data/'test_config.json').read_text())
    if (len(config['floor']),len(config['obstacles']),len(config['routes'])) != (10,10,5):
        raise ValueError('Expected qualified 10/10/5 physical positions')
    mesh = read_mesh(gt_data/'gt_mesh.ply')
    scene = raycasting(mesh)
    dimensions = [dimension(scene, item) for item in spec['dimensions']]
    doors = [door(scene, item, config['body']['size']) for item in spec['doors']]
    # NumPy sampling avoids Open3D global RNG state and samples by true triangle area.
    triangle = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    area = np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)/2
    rng = np.random.default_rng(spec['seed'])
    chosen = triangle[rng.choice(len(triangle),spec['samples'],p=area/area.sum())]
    u = np.sqrt(rng.random(spec['samples']))[:,None]
    v = rng.random(spec['samples'])[:,None]
    points = ((1-u)*chosen[:,0]+u*(1-v)*chosen[:,1]+u*v*chosen[:,2]).astype(np.float32)
    a,b = [np.asarray(o3d.io.read_point_cloud(str(scans/f'{name}.ply')).points) for name in ('A','B')]
    require_observed_dimensions(dimensions,a,b,spec['region']['radius_m'],spec['region']['min_neighbors'])
    mask, ca, cb = observed_by_both(points,a,b,spec['region']['radius_m'],spec['region']['min_neighbors'])
    if int(mask.sum()) < minimum:
        raise ValueError('Insufficient common observable GT samples')
    out.mkdir(parents=True)
    np.savez_compressed(out/'eval_region.npz',points_m=points,common_mask=mask,a_neighbors=ca,b_neighbors=cb)
    report = {
        'schema_version':1,'profile_id':spec['profile_id'],'units':'m','up_axis':'Z',
        'status':'candidate_pending_H3_H13','dimensions':dimensions,'doors':doors,
        'region':{**spec['region'],'predicate':'neighbors_in_A >= min_neighbors AND neighbors_in_B >= min_neighbors',
                  'sampling':'uniform_triangle_area','seed':spec['seed'],'samples':len(points),
                  'common_samples':int(mask.sum()),'common_fraction_of_gt_area':float(mask.mean()),
                  'a_observed_samples':int((ca>=spec['region']['min_neighbors']).sum()),
                  'b_observed_samples':int((cb>=spec['region']['min_neighbors']).sum()),
                  'file':'eval_region.npz','query_contract':'apply the identical predicate to query points using the pinned A and B; never infer region from reconstructed meshes'},
        'physics':{'evidence_file':str(physics_evidence),'evidence_sha256':digest(physics_evidence),
                   'summary':physical['summary'],'inputs':physical['inputs'],'body':config['body'],
                   'scope':'existing GT physical qualification reused after exact input hash checks; no new Isaac run'},
        'inputs_sha256':{'gt_mesh.ply':digest(gt_data/'gt_mesh.ply'),'scan_manifest.json':digest(scans/'manifest.json'),
                         'definition.json':digest(definition),**manifest['sha256']},
        'code_sha256':{'builder':digest(__file__),'annotations':digest(Path(__file__).resolve().parents[1]/'lidar2sim/annotations.py')},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'open3d':o3d.__version__},
        'artifacts_sha256':{'eval_region.npz':digest(out/'eval_region.npz')},
    }
    (out/'annotations.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'region':report['region'],'dimensions':[(x['id'],x['reference_m']) for x in dimensions], 'doors':[(x['id'],x['width_m'],x['height_m']) for x in doors]},indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gt_data');parser.add_argument('scans');parser.add_argument('out')
    parser.add_argument('--definition',default='configs/scans/annotations-v1.json')
    parser.add_argument('--physics-evidence',default='docs/validation/hackathon-12-physics.json')
    args=parser.parse_args()
    build(args.gt_data,args.scans,args.out,args.definition,args.physics_evidence)
