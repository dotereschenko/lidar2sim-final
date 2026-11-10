import unittest

import numpy as np
import open3d as o3d

from lidar2sim.filtering import filter_supported


class SupportFilterTest(unittest.TestCase):
    def setUp(self):
        self.mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                       [0, 0, 1], [1, 0, 1], [0, 1, 1]]),
            o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
        self.pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([[1/3, 1/3, 0]]))

    def test_removes_unsupported_surface(self):
        mesh, report = filter_supported(self.mesh, self.pcd, 0.08)
        self.assertEqual(report["removed_triangles"], 1)
        self.assertEqual(len(mesh.triangles), 1)
        np.testing.assert_array_equal(np.asarray(mesh.vertices)[:, 2], 0)

    def test_retains_supported_surfaces(self):
        self.pcd.points = o3d.utility.Vector3dVector([[1/3, 1/3, 0], [1/3, 1/3, 1]])
        mesh, report = filter_supported(self.mesh, self.pcd, 0.08)
        self.assertEqual(report["removed_triangles"], 0)
        self.assertEqual(len(mesh.triangles), 2)

    def test_rejects_invalid_radius(self):
        for radius in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                filter_supported(self.mesh, self.pcd, radius)

    def test_rejects_empty_result_without_mutating_mesh(self):
        self.pcd.translate([0, 0, 10])
        with self.assertRaises(ValueError):
            filter_supported(self.mesh, self.pcd, 0.08)
        self.assertEqual(len(self.mesh.triangles), 2)
