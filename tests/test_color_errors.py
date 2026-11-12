import unittest

import numpy as np
import open3d as o3d

from tools.color_errors import colorize


class ErrorColorsTest(unittest.TestCase):
    def test_colors_follow_distance_without_changing_geometry(self):
        gt = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector([[-1, -1, 0], [3, -1, 0], [-1, 3, 0]]),
            o3d.utility.Vector3iVector([[0, 1, 2]]))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
        vertices = np.array([[0, 0, 0], [1, 0, 0.04], [0, 1, 0.08]])
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                        o3d.utility.Vector3iVector([[0, 1, 2]]))
        colored = colorize(mesh, scene)
        np.testing.assert_array_equal(np.asarray(colored.vertices), vertices)
        np.testing.assert_allclose(np.asarray(colored.vertex_colors),
                                   [[0.15, 0.70, 0.40], [1, 0.72, 0.10], [1, 0.10, 0.05]])
