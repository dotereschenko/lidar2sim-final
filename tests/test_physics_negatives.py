import unittest
import numpy as np
import open3d as o3d


class NegativeFixtureTests(unittest.TestCase):
    def test_removed_region_leaves_other_geometry_intact(self):
        from tools.make_physics_negatives import remove_region
        vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0],
                             [2, 0, 0], [2.1, 0, 0], [2, 0.1, 0],
                             [0, 0, 1], [0.1, 0, 1], [0, 0.1, 1]])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                       o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
        result = remove_region(mesh, [-0.2, -0.2, -0.1], [0.2, 0.2, 0.1])
        self.assertEqual(len(mesh.triangles), 3)
        self.assertEqual(len(result.triangles), 2)
        centers = np.asarray(result.vertices)[np.asarray(result.triangles)].mean(axis=1)
        self.assertTrue(any(p[0] > 2 for p in centers))
        self.assertTrue(any(p[2] == 1 for p in centers))
        with self.assertRaisesRegex(ValueError, 'no triangles'):
            remove_region(mesh, [9, 9, 9], [10, 10, 10])
