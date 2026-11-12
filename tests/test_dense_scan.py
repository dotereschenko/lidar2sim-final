import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


class DenseScanTests(unittest.TestCase):
    def test_seeded_raycast_count_coordinates_and_hashes(self):
        from tools.make_dense_scan import generate
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'source'
            data.mkdir()
            mesh = o3d.geometry.TriangleMesh.create_box(4, 4, 3)
            o3d.io.write_triangle_mesh(str(data/'gt_mesh.ply'), mesh)
            (data/'manifest.json').write_text(json.dumps({'units': 'm', 'up_axis': 'Z', 'bounds_min': [0, 0, 0], 'bounds_max': [4, 4, 3]}))
            (data/'trajectory_A.csv').write_text('t,x,y,z\n0,2,2,1.3\n1,1,1,1.3\n')
            outputs = [Path(tmp)/name for name in ['a', 'b', 'c']]
            for out, seed in zip(outputs, [7, 7, 8]):
                generate(data, out, points=1000, seed=seed, noise_sigma=0, batch_size=512)
            a = o3d.io.read_point_cloud(str(outputs[0]/'scene_10M.ply'))
            pts = np.asarray(a.points)
            self.assertEqual(pts.shape, (1000, 3))
            self.assertEqual(len(np.unique(pts, axis=0)), 1000)
            self.assertTrue(np.isfinite(pts).all())
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            distances = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
            self.assertLess(float(distances.max()), 1e-5)
            self.assertEqual((outputs[0]/'scene_10M.ply').read_bytes(), (outputs[1]/'scene_10M.ply').read_bytes())
            self.assertNotEqual((outputs[0]/'scene_10M.ply').read_bytes(), (outputs[2]/'scene_10M.ply').read_bytes())
            report = json.loads((outputs[0]/'manifest.json').read_text())
            self.assertEqual(report['point_count'], 1000)
            self.assertEqual(sum(report['accepted_points_per_pose']), 1000)
            self.assertEqual((outputs[0]/'trajectory.csv').read_bytes(), (data/'trajectory_A.csv').read_bytes())
            with self.assertRaises(FileExistsError):
                generate(data, outputs[0], points=1000)

    def test_bad_count_rejected_before_creating_output(self):
        from tools.make_dense_scan import generate
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'bad'
            with self.assertRaises(ValueError):
                generate(Path(tmp), out, points=0)
            self.assertFalse(out.exists())
