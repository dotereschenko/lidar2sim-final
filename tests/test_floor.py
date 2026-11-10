import unittest

import numpy as np
import open3d as o3d

from lidar2sim.floor import repair_floor


def cloud(x_ranges=((0, 2),)):
    points = []
    for start, stop in x_ranges:
        x, y = np.meshgrid(np.arange(start, stop + 0.01, 0.05), np.arange(0, 2.01, 0.05))
        points.append(np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(points)))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(pcd.points), 1)))
    return pcd


def plane():
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[-1, -1, 0], [5, -1, 0], [5, 3, 0], [-1, 3, 0]]),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))


def hits(mesh, xy):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = [[x, y, 0.5, 0, 0, -1] for x, y in xy]
    return scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()


class FloorRepairTest(unittest.TestCase):
    def test_existing_floor_is_unchanged(self):
        mesh, report = repair_floor(plane(), cloud())
        self.assertEqual(report["added_triangles"], 0)
        self.assertEqual(len(mesh.triangles), 2)

    def test_fills_supported_gap_with_upward_faces(self):
        original = plane().translate([0, 0, -1])
        mesh, report = repair_floor(original, cloud())
        self.assertGreater(report["added_triangles"], 0)
        self.assertAlmostEqual(float(hits(mesh, [(1.03, 1.02)])[0]), 0.5, places=4)
        mesh.compute_triangle_normals()
        self.assertTrue((np.asarray(mesh.triangle_normals)[2:, 2] > 0.99).all())
        self.assertEqual(len(original.triangles), 2)

    def test_does_not_bridge_unobserved_space(self):
        mesh, _ = repair_floor(plane().translate([0, 0, -1]), cloud(((0, 1), (3, 4))))
        np.testing.assert_allclose(hits(mesh, [(0.53, 1.02), (2.03, 1.02), (3.53, 1.02)]),
                                   [0.5, 1.5, 0.5], atol=1e-5)

    def test_floor_height_is_inferred(self):
        pcd = cloud().translate([0, 0, 0.2])
        mesh, report = repair_floor(plane().translate([0, 0, -1]), pcd)
        self.assertAlmostEqual(report["height_m"], 0.2, places=4)
        self.assertAlmostEqual(float(hits(mesh, [(1.03, 1.02)])[0]), 0.3, places=4)

    def test_vertical_cloud_is_skipped(self):
        pcd = cloud()
        pcd.normals = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(pcd.points), 1)))
        _, report = repair_floor(plane(), pcd)
        self.assertIn("skipped", report)

    def test_invalid_parameters(self):
        for cell, radius in [(0, 0.2), (0.1, 0.01), (float("nan"), 0.2)]:
            with self.assertRaises(ValueError):
                repair_floor(plane(), cloud(), cell, radius)


if __name__ == "__main__":
    unittest.main()
