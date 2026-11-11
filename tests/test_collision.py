import numpy as np
import open3d as o3d
import pytest

from lidar2sim.baseline import PARAMS
from lidar2sim.collision import build_collision
from lidar2sim.refinement import recover_floor
from lidar2sim.validation import ContractError, validate_params


def test_protected_collision_keeps_floor_gap_and_wall_within_budget():
    x, y = np.meshgrid(np.arange(0, 3.01, .06), np.arange(0, 2.01, .06))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    points = points[(points[:, 0] < .8) | (points[:, 0] > 2.2)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(points), 1)))
    wall = o3d.geometry.TriangleMesh.create_box(.1, .7, .8).translate([.3, .6, 0])
    params = {**PARAMS, "floor_distance": .025, "floor_max_edge": .8}
    mesh, _, rest, floor = recover_floor(wall, pcd, None, params, return_components=True)
    collision, stats = build_collision(mesh, rest, floor, 300, .4, 100.)
    assert stats["mode"] == "protected_floor"
    assert len(collision.triangles) <= 300
    assert stats["floor_faces_upward"]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(collision))
    rays = np.array([[.5, 1., 2., 0., 0., -1.],
                     [1.5, 1., 2., 0., 0., -1.],
                     [2.5, 1., 2., 0., 0., -1.],
                     [0., .9, .4, 1., 0., 0.]], dtype=np.float32)
    distances = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    assert abs(distances[0] - 2.) < .01
    assert np.isinf(distances[1])
    assert abs(distances[2] - 2.) < .01
    assert abs(distances[3] - .3) < .01


def test_collision_without_recovered_floor_keeps_global_decimation():
    mesh = o3d.geometry.TriangleMesh.create_box()
    collision, stats = build_collision(mesh, None, None, 8, .25, 100.)
    assert stats["mode"] == "global"
    assert len(collision.triangles) <= 8


@pytest.mark.parametrize("value", [-1, 1, True, float("nan")])
def test_invalid_collision_floor_fraction(value):
    with pytest.raises(ContractError, match="collision_floor_fraction"):
        validate_params({"collision_floor_fraction": value}, PARAMS)


@pytest.mark.parametrize("value", [0, -1, True, float("inf")])
def test_invalid_collision_boundary_weight(value):
    with pytest.raises(ContractError, match="collision_boundary_weight"):
        validate_params({"collision_boundary_weight": value}, PARAMS)
