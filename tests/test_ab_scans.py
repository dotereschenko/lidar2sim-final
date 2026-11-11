import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


class ABScanTests(unittest.TestCase):
    def source(self, root):
        data = root/'source'
        data.mkdir()
        mesh = o3d.geometry.TriangleMesh.create_box(4, 4, 3)
        o3d.io.write_triangle_mesh(str(data/'gt_mesh.ply'), mesh)
        (data/'manifest.json').write_text(json.dumps({'units': 'm', 'up_axis': 'Z'}))
        (data/'trajectory_A.csv').write_text('t,x,y,z\n0,1,1,1.3\n1,2,1,1.3\n2,2,2,1.3\n')
        return data

    def test_independent_raycast_noise_dropout_and_repeatability(self):
        from tools.make_ab_scans import generate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.source(root)
            a = generate(data, root/'one', points=20000, batch_size=4096)
            b = generate(data, root/'two', points=20000, batch_size=4096)
            self.assertEqual(a, b)
            for name, sigma in [('A', .02), ('B', .01)]:
                stats = a['scans'][name]
                self.assertAlmostEqual(stats['range_noise_sigma_m'], sigma, delta=sigma*.05)
                self.assertLess(abs(stats['range_noise_mean_m']), .001)
                self.assertGreater(stats['cloud_to_gt_median_m'], .003)
                self.assertLess(stats['cloud_to_gt_median_m'], .02)
                cloud = np.asarray(o3d.io.read_point_cloud(str(root/'one'/f'{name}.ply')).points)
                self.assertEqual(len(cloud), 20000)
                self.assertTrue(np.isfinite(cloud).all())
                self.assertEqual(len(np.unique(cloud, axis=0)), 20000)
                self.assertEqual((root/'one'/f'{name}.ply').read_bytes(), (root/'two'/f'{name}.ply').read_bytes())
                self.assertEqual(sum(stats['accepted_points_per_pose']), 20000)
            rate = a['scans']['A']['random_dropout_fraction_measured']
            self.assertTrue(.04 < rate < .06)
            self.assertEqual(a['scans']['B']['random_dropout_fraction_measured'], 0)
            ta = np.loadtxt(root/'one'/'trajectory_A.csv', delimiter=',', skiprows=1)
            tb = np.loadtxt(root/'one'/'trajectory_B.csv', delimiter=',', skiprows=1)
            self.assertGreater(np.linalg.norm(ta[:, None, 1:] - tb[None, :, 1:], axis=2).min(), .09)
            with self.assertRaises(FileExistsError):
                generate(data, root/'one', points=20000)

    def test_invalid_input_and_collision_do_not_create_outputs(self):
        from tools.make_ab_scans import generate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.source(root)
            for kwargs in [{'points': 0}, {'seed_a': 5, 'seed_b': 5}]:
                with self.assertRaises(ValueError):
                    generate(data, root/'bad', **kwargs)
                self.assertFalse((root/'bad').exists())
            (data/'trajectory_A.csv').write_text('t,x,y,z\n0,0,1,1.3\n1,1,1,1.3\n')
            with self.assertRaises(ValueError):
                generate(data, root/'bad', points=10)
            self.assertFalse((root/'bad').exists())

    def test_explicit_trajectory_is_used_and_crossing_is_rejected(self):
        from tools.make_ab_scans import generate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.source(root)
            trajectory = root/'corrected.csv'
            trajectory.write_text('t,x,y,z\n0,1,2,1.3\n1,2,2,1.3\n')
            generate(data, root/'ok', points=100, trajectory_path=trajectory)
            actual = np.loadtxt(root/'ok'/'trajectory_A.csv', delimiter=',', skiprows=1)
            np.testing.assert_allclose(actual, [[0,1,2,1.3],[1,2,2,1.3]])
            # Endpoints are clear, but the segment crosses the enclosing box wall.
            trajectory.write_text('t,x,y,z\n0,-1,1,1.3\n1,2,1,1.3\n')
            with self.assertRaisesRegex(ValueError, 'crosses'):
                generate(data, root/'bad', points=100, trajectory_path=trajectory)
            self.assertFalse((root/'bad').exists())

    def test_explicit_b_and_sectors_are_reproducible(self):
        from tools.make_ab_scans import generate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = self.source(root)
            path = root/'independent.csv'
            path.write_text('t,x,y,z\n0,3,3,1.6\n1,3,1,1.6\n2,1,3,1.6\n')
            kwargs = dict(points=20000, batch_size=4096, trajectory_b_path=path,
                          sectors_a=[[-15,15],[165,180],[-180,-165]])
            a = generate(data, root/'one', **kwargs)
            b = generate(data, root/'two', **kwargs)
            self.assertEqual(a,b)
            np.testing.assert_allclose(np.loadtxt(root/'one/trajectory_B.csv',delimiter=',',skiprows=1),
                                       np.loadtxt(path,delimiter=',',skiprows=1))
            self.assertEqual(a['trajectory_B_derivation'],'explicit independent trajectory')
            stats=a['scans']['A']
            self.assertGreater(stats['sector_dropped_returns'],0)
            self.assertTrue(.04 < stats['random_dropout_fraction_measured'] < .06)
            self.assertEqual(stats['sensor']['sector_dropout_deg'],kwargs['sectors_a'])
            self.assertEqual(a['scans']['B']['sector_dropped_returns'],0)
            for sectors in [[[20,10]], [[-190,0]], [[0,float('nan')]]]:
                with self.assertRaises(ValueError):generate(data,root/'bad',sectors_a=sectors)
                self.assertFalse((root/'bad').exists())

    def test_sector_mask_uses_sensor_heading_and_boundaries(self):
        from tools.make_ab_scans import sector_mask
        angles=np.deg2rad([-180,-16,-15,0,15,16,179])
        np.testing.assert_array_equal(sector_mask(angles,np.zeros(7),[[-15,15]]),
                                      [False,False,True,True,True,False,False])
        self.assertTrue(sector_mask(np.array([np.pi/2]),np.array([np.pi/2]),[[-10,10]])[0])
