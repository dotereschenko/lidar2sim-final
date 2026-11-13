import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar2sim.acceptance import load_profile
from tools.make_dense_scan import digest


class IndependentEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        data, scans, package, out = [root/n for n in ['data','scans','package','result']]
        for p in [data,scans,package,out]:p.mkdir()
        mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]))
        for p in [data/'gt_mesh.ply',out/'visual.ply']:o3d.io.write_triangle_mesh(str(p),mesh)
        shifted=o3d.geometry.TriangleMesh(mesh).translate([0,0,.07]);o3d.io.write_triangle_mesh(str(out/'collision.ply'),shifted)
        points=np.array([[x,y,0] for x in np.linspace(0,1,41) for y in np.linspace(0,1,41)],dtype=np.float32)
        for n,z in [('A',0),('B',.01)]:
            cloud=points.copy();cloud[:,2]=z;o3d.io.write_point_cloud(str(scans/f'{n}.ply'),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)))
        np.savez_compressed(package/'eval_region.npz',points_m=points,common_mask=np.ones(len(points),dtype=bool),a_neighbors=np.full(len(points),20),b_neighbors=np.full(len(points),20))
        meta={'schema_version':1,'units':'m','up_axis':'Z','profile_id':'test-region','status':'candidate',
              'region':{'radius_m':.1,'min_neighbors':10,'min_common_samples':1000,'samples':len(points),'common_samples':len(points)},
              'inputs_sha256':{'gt_mesh.ply':digest(data/'gt_mesh.ply'),**{f'{n}.ply':digest(scans/f'{n}.ply') for n in ['A','B']}},
              'artifacts_sha256':{'eval_region.npz':digest(package/'eval_region.npz')}}
        (package/'annotations.json').write_text(json.dumps(meta))
        return data,scans,package,out

    def test_b_is_independent_and_collision_has_its_own_failures(self):
        from tools.eval_scene import load_common_reference, evaluate_common, raycasting, read_mesh
        with tempfile.TemporaryDirectory() as tmp:
            data,scans,package,out=self.fixture(Path(tmp))
            profile,provenance=load_profile();profile['geometry']['samples']=2000
            ctx=load_common_reference(data,scans,package,'b',profile)
            report=evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),ctx,profile,provenance)
            self.assertEqual(report['evaluation_mode'],'independent_b_common_region')
            self.assertAlmostEqual(report['meshes']['visual']['raw_metrics']['median_m'],.01,places=5)
            self.assertAlmostEqual(report['meshes']['collision']['raw_metrics']['median_m'],.06,places=5)
            self.assertTrue(all(report['meshes']['visual']['passed'].values()))
            self.assertFalse(report['meshes']['collision']['passed']['median'])
            self.assertFalse(report['meshes']['collision']['passed']['precision'])
            self.assertEqual(report['poc_status'],'failed')
            second=evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),ctx,profile,provenance)
            self.assertEqual(report,second)
            gt=load_common_reference(data,scans,package,'gt',profile)
            result=evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),gt,profile,provenance)
            self.assertLess(result['meshes']['visual']['raw_metrics']['median_m'],1e-6)
            self.assertEqual(result['evaluation_mode'],'gt_common_region')

    def test_changed_package_and_undersampled_mesh_are_errors(self):
        from tools.eval_scene import load_common_reference,evaluate_common,raycasting,read_mesh
        with tempfile.TemporaryDirectory() as tmp:
            data,scans,package,out=self.fixture(Path(tmp))
            profile,provenance=load_profile();profile['geometry']['samples']=2000
            ctx=load_common_reference(data,scans,package,'b',profile)
            mesh=read_mesh(out/'collision.ply').translate([100,0,0]);o3d.io.write_triangle_mesh(str(out/'collision.ply'),mesh)
            far=evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),ctx,profile,provenance)
            self.assertFalse(far['meshes']['collision']['passed']['precision'])
            profile['geometry']['samples']=500
            with self.assertRaisesRegex(ValueError,'sample'):
                evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),ctx,profile,provenance)
            with (package/'eval_region.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):
                load_common_reference(data,scans,package,'b',profile)

    def test_explicit_cli_is_order_invariant_and_errors_do_not_overwrite(self):
        import subprocess
        import sys
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,scans,package,out=self.fixture(root)
            other=root/'other';other.mkdir()
            for name in ['visual.ply','collision.ply']:
                (other/name).write_bytes((out/name).read_bytes())
            profile,_=load_profile();profile['geometry']['samples']=2000
            config=root/'rules.yaml';config.write_text(yaml.safe_dump(profile))
            tool=Path(__file__).resolve().parents[1]/'tools/eval_scene.py'
            def run(order):
                return subprocess.run([sys.executable,str(tool),str(data),*map(str,order),'--mode','b','--scans',str(scans),'--annotations',str(package),'--profile',str(config)],capture_output=True,text=True,cwd=root)
            result=run([out,other]);self.assertEqual(result.returncode,0,result.stderr)
            before=[(p/'eval_scene.json').read_bytes() for p in [out,other]]
            result=run([other,out]);self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(before,[(p/'eval_scene.json').read_bytes() for p in [out,other]])
            (other/'collision.ply').unlink()
            result=run([out,other]);self.assertNotEqual(result.returncode,0)
            self.assertEqual(before[0],(out/'eval_scene.json').read_bytes())

    def test_b_sampling_does_not_overweight_dense_return_clusters(self):
        from tools.eval_scene import load_common_reference
        with tempfile.TemporaryDirectory() as tmp:
            data,scans,package,out=self.fixture(Path(tmp))
            path=scans/'B.ply';points=np.asarray(o3d.io.read_point_cloud(str(path)).points)
            points=np.concatenate([points,np.tile([.1,.1,.01],(10000,1))])
            o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
            meta=json.loads((package/'annotations.json').read_text());meta['inputs_sha256']['B.ply']=digest(path)
            (package/'annotations.json').write_text(json.dumps(meta))
            profile,_=load_profile();profile['geometry']['samples']=2000
            ctx=load_common_reference(data,scans,package,'b',profile)
            self.assertGreater(float(ctx['reference'][:,0].mean()),.45)
            self.assertLess(float(ctx['reference'][:,0].mean()),.55)

    def test_dimensions_cli_reports_missing_surfaces_without_false_success(self):
        import subprocess
        import sys
        import yaml
        from lidar2sim.annotations import dimension
        from tools.eval_scene import raycasting
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,scans,package,out=self.fixture(root)
            box=o3d.geometry.TriangleMesh.create_box(4,4,3)
            spec=dimension(raycasting(box),{'id':'D01','origin_m':[2,2,1.5],'axis':[1,0,0]})
            meta=json.loads((package/'annotations.json').read_text())
            meta['dimensions']=[{**spec,'id':f'D{i:02d}'} for i in range(1,11)]
            (package/'annotations.json').write_text(json.dumps(meta))
            profile,_=load_profile();profile['geometry']['samples']=2000
            config=root/'rules.yaml';config.write_text(yaml.safe_dump(profile))
            tool=Path(__file__).resolve().parents[1]/'tools/eval_scene.py'
            result=subprocess.run([sys.executable,str(tool),str(data),str(out),'--mode','b','--scans',str(scans),'--annotations',str(package),'--profile',str(config),'--dimensions'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            r=json.loads((out/'eval_scene.json').read_text())
            self.assertEqual(r['coverage']['dimensions'],'measured')
            self.assertEqual(r['dimensions_status'],'failed')
            for mesh in r['meshes'].values():
                self.assertEqual(mesh['dimensions']['passed_count'],0)
                self.assertEqual(len(mesh['dimensions']['items']),10)
                self.assertTrue(all(x['status']=='error' for x in mesh['dimensions']['items']))

    def test_v2_precision_uses_b_and_refuses_gt_substitution(self):
        from tools.eval_scene import load_common_reference, evaluate_common, raycasting, read_mesh, main
        with tempfile.TemporaryDirectory() as tmp:
            data, scans, package, out = self.fixture(Path(tmp))
            path = Path(__file__).resolve().parents[1]/'configs/acceptance/lidar2sim-rules-v2.yaml'
            profile, provenance = load_profile(path)
            profile['geometry']['samples'] = 2000
            # Displace B while keeping it inside the common observation radius.
            # A mesh at z=.15 is farther than .10 from GT but within .10 of B=.07.
            b = o3d.io.read_point_cloud(str(scans/'B.ply')); b.translate([0,0,.06])
            o3d.io.write_point_cloud(str(scans/'B.ply'), b)
            meta = json.loads((package/'annotations.json').read_text())
            meta['inputs_sha256']['B.ply'] = digest(scans/'B.ply')
            (package/'annotations.json').write_text(json.dumps(meta))
            mesh = read_mesh(data/'gt_mesh.ply').translate([0,0,.15])
            for name in ['visual','collision']:o3d.io.write_triangle_mesh(str(out/f'{name}.ply'),mesh)
            ctx = load_common_reference(data,scans,package,'b',profile)
            report = evaluate_common(out,raycasting(read_mesh(data/'gt_mesh.ply')),ctx,profile,provenance)
            self.assertEqual(ctx['provenance']['reference_weighting'],'uniform_B_returns')
            for result in report['meshes'].values():
                self.assertEqual(result['precision_reference'],'B_returns')
                self.assertTrue(result['passed']['precision'])
                self.assertFalse(result['passed']['median'])
            with self.assertRaisesRegex(ValueError,'requires'):
                load_common_reference(data,scans,package,'gt',profile)
            with self.assertRaisesRegex(ValueError,'requires'):
                main(data,out,profile=path)
