"""Fast offline regression of geometry and dimensions on controlled meshes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lidar2sim.acceptance import load_profile
from lidar2sim.annotations import dimension, observed_by_both
from tools.eval_scene import evaluate_common, load_common_reference, raycasting
from tools.make_dense_scan import digest


def run_regression(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    data,scans,package=[out/n for n in ('data','scans','annotations')]
    for p in (data,scans,package):p.mkdir()
    gt=o3d.geometry.TriangleMesh.create_box(2,2,2)
    o3d.io.write_triangle_mesh(str(data/'gt_mesh.ply'),gt)
    cloud=[]
    for axis in range(3):
        other=[i for i in range(3) if i!=axis]
        for side in (0,2):
            for x in np.linspace(0,2,51):
                for y in np.linspace(0,2,51):
                    p=np.zeros(3);p[axis]=side;p[other]=[x,y];cloud.append(p)
    a=np.array(cloud);b=a+np.random.default_rng(22).normal(0,.001,a.shape)
    for name,points in [('A',a),('B',b)]:
        o3d.io.write_point_cloud(str(scans/f'{name}.ply'),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    profile,_=load_profile();profile['geometry']['samples']=12000
    rules_path=out/'regression-profile.yaml';rules_path.write_text(yaml.safe_dump(profile))
    profile,provenance=load_profile(rules_path)
    o3d.utility.random.seed(22)
    reference=np.asarray(gt.sample_points_uniformly(12000).points).astype(np.float32)
    mask,ca,cb=observed_by_both(reference,a,b)
    np.savez_compressed(package/'eval_region.npz',points_m=reference,common_mask=mask,a_neighbors=ca,b_neighbors=cb)
    gt_scene=raycasting(gt)
    definitions=[]
    for i in range(10):
        origin=[.7+(i%2)*.3, .8+(i%3)*.1, .9+(i%2)*.1]
        axis=np.eye(3)[i%3].tolist()
        definitions.append(dimension(gt_scene,{'id':f'D{i+1:02d}','label':f'Box span {i+1}','origin_m':origin,'axis':axis}))
    meta={'schema_version':1,'units':'m','up_axis':'Z','status':'regression_fixture','profile_id':'box-regression-v1',
          'region':{'radius_m':.1,'min_neighbors':10,'min_common_samples':1000,'samples':len(reference),'common_samples':int(mask.sum())},
          'inputs_sha256':{'gt_mesh.ply':digest(data/'gt_mesh.ply'),**{f'{n}.ply':digest(scans/f'{n}.ply') for n in ('A','B')}},
          'artifacts_sha256':{'eval_region.npz':digest(package/'eval_region.npz')},'dimensions':definitions}
    (package/'annotations.json').write_text(json.dumps(meta,indent=2)+'\n')
    shifted=o3d.geometry.TriangleMesh(gt).translate([.1,0,0])
    expanded=o3d.geometry.TriangleMesh(gt);np.asarray(expanded.vertices)[:,0]*=1.05
    missing=o3d.geometry.TriangleMesh(gt)
    remove=(np.asarray(missing.vertices)[np.asarray(missing.triangles),0]==2).all(axis=1)
    missing.remove_triangles_by_mask(remove);missing.remove_unreferenced_vertices()
    cases={'gt':gt,'shift_10cm':shifted,'missing_wall':missing,'expanded_width':expanded}
    for name,mesh in cases.items():
        folder=out/name;folder.mkdir()
        for part in ('visual','collision'):o3d.io.write_triangle_mesh(str(folder/f'{part}.ply'),mesh)
    checks={};reports={}
    for mode in ('gt','b'):
        ctx=load_common_reference(data,scans,package,mode,profile)
        def evaluate(name):return evaluate_common(out/name,gt_scene,ctx,profile,provenance,definitions)
        forward={name:evaluate(name) for name in cases}
        reverse={name:evaluate(name) for name in reversed(cases)}
        alone={name:evaluate(name) for name in cases}
        # Complete report equality is stronger than the requested 1e-12 numeric tolerance.
        checks[f'{mode}_ordering_and_single_runs']=forward==reverse==alone
        reports[mode]=forward
        for part in ('visual','collision'):
            positive=forward['gt']['meshes'][part]
            limit=2e-6 if mode=='gt' else .01
            prefix=f'{mode}_{part}'
            checks[prefix+'_gt']=all(positive['passed'].values()) and positive['dimensions']['passed_count']==10 and positive['raw_metrics']['p95_m']<=limit
            checks[prefix+'_shift_precision']=not forward['shift_10cm']['meshes'][part]['passed']['precision']
            wall=forward['missing_wall']['meshes'][part]
            checks[prefix+'_missing_wall']=not wall['passed']['p95'] and wall['dimensions']['items'][0]['status']=='error'
            expansion=forward['expanded_width']['meshes'][part]['dimensions']['items'][0]
            checks[prefix+'_expanded_width']=expansion['status']=='failed' and abs(expansion['absolute_error_m']-.1)<=2e-6
        for name,r in forward.items():
            (out/name/f'eval_{mode}.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    result={'passed':all(checks.values()),'checks':checks,'acceptance_profile':provenance,
            'sampling_note':'12000 samples for fixture regression; production thresholds unchanged',
            'numerical_tolerances':{'gt_distance_m':2e-6,'b_p95_m':.01,'distortion_error_m':2e-6,'order':1e-12},
            'code_sha256':{'runner':digest(__file__),'evaluator':digest(Path(__file__).with_name('eval_scene.py')),
                           'dimensions':digest(Path(__file__).resolve().parents[1]/'lidar2sim/dimensions.py')},
            'fixture_inputs_sha256':meta['inputs_sha256'],'reports':reports}
    (out/'regression.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':checks},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out');args=parser.parse_args()
    sys.exit(0 if run_regression(args.out)['passed'] else 1)
