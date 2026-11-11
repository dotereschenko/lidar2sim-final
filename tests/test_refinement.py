import numpy as np
import open3d as o3d
import pytest

from lidar2sim.baseline import PARAMS
from lidar2sim.refinement import recover_floor, trim_unsupported
from lidar2sim.validation import ContractError, validate_params


def cloud(points, normals=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is None:
        normals = np.tile([0., 0., 1.], (len(points), 1))
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def floor_points():
    x, y = np.meshgrid(np.arange(0, 3.01, .1), np.arange(0, 2.01, .1))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def floor_params(**overrides):
    return {**PARAMS, "floor_distance": .025, "floor_cell": .08,
            "floor_max_edge": .8, **overrides}


def height_at(mesh, xy, origin_z=2.):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.column_stack([xy, np.full(len(xy), origin_z), np.tile([0, 0, -1], (len(xy), 1))])
    return origin_z - scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()


def test_support_filter_keeps_observed_surface_and_removes_floating_sheet():
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0],
                         [0, 0, 1], [.1, 0, 1], [0, .1, 1.]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]]))
    pcd = cloud(np.vstack([vertices[:3], vertices[:3].mean(axis=0)]))
    result, stats = trim_unsupported(mesh, pcd, .03)
    assert len(result.triangles) == 1
    assert stats["triangles_removed"] == 1
    assert np.asarray(result.vertices)[:, 2].max() == 0


def test_supported_corners_do_not_justify_large_unsupported_triangle():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0.]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector([[0, 1, 2]]))
    result, _ = trim_unsupported(mesh, cloud(vertices), .05)
    assert len(result.triangles) == 0


def test_floor_recovers_local_gap_with_upward_faces():
    points = floor_points()
    gap = (points[:, 0] > 1.2) & (points[:, 0] < 1.8) & (points[:, 1] > .7) & (points[:, 1] < 1.3)
    floor, stats = recover_floor(o3d.geometry.TriangleMesh(), cloud(points[~gap]), None, floor_params())
    assert stats["status"] == "applied"
    np.testing.assert_allclose(height_at(floor, [[1.5, 1.]]), 0, atol=1e-6)
    triangles = np.asarray(floor.vertices)[np.asarray(floor.triangles)]
    assert (np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])[:, 2] > 0).all()


def test_floor_does_not_span_large_unobserved_gap_or_extend_beyond_observations():
    points = floor_points()
    points = points[(points[:, 0] < .8) | (points[:, 0] > 2.2)]
    floor, _ = recover_floor(o3d.geometry.TriangleMesh(), cloud(points), None, floor_params())
    heights = height_at(floor, [[.4, 1.], [1.5, 1.], [3.4, 1.]])
    assert abs(heights[0]) < 1e-6
    assert np.isneginf(heights[1:]).all()


@pytest.mark.parametrize("max_edge", [.8, 2.0])
def test_observed_lower_level_is_not_covered_by_completed_floor(max_edge):
    points = floor_points()
    hole = (points[:, 0] >= 1.2) & (points[:, 0] <= 1.8) & (points[:, 1] >= .7) & (points[:, 1] <= 1.3)
    points[hole, 2] = -.4
    floor, stats = recover_floor(o3d.geometry.TriangleMesh(), cloud(points), None, floor_params(floor_max_edge=max_edge))
    assert stats["triangles_rejected_by_lower_observations"] > 0
    assert np.isneginf(height_at(floor, [[1.5, 1.]])[0])


def test_floor_recovery_preserves_vertical_obstacle():
    box = o3d.geometry.TriangleMesh.create_box(.4, .4, .6).translate([1.2, .8, 0])
    result, _ = recover_floor(box, cloud(floor_points()), None, floor_params())
    assert abs(height_at(result, [[1.4, 1.]])[0] - .6) < 1e-6
    np.testing.assert_allclose(height_at(result, [[.5, .5]]), 0, atol=1e-6)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(result))
    ray = o3d.core.Tensor([[.5, 1., .3, 1., 0., 0.]], dtype=o3d.core.Dtype.Float32)
    assert abs(scene.cast_rays(ray)["t_hit"].numpy()[0] - .7) < 1e-5


def test_sloped_translated_floor_uses_observed_height_instead_of_world_zero():
    points = floor_points()
    points[:, 2] = .05 * points[:, 0] + 4.
    points[:, :2] += [17, -34]
    n = np.array([-.05, 0, 1.])
    n /= np.linalg.norm(n)
    pcd = cloud(points, np.tile(n, (len(points), 1)))
    traj = np.array([[17, -33, 5.3], [20, -33, 5.45]])
    floor, _ = recover_floor(o3d.geometry.TriangleMesh(), pcd, traj, floor_params())
    np.testing.assert_allclose(height_at(floor, [[18.5, -33]], 6), 4.075, atol=1e-5)


def test_no_floor_evidence_leaves_mesh_unchanged():
    mesh = o3d.geometry.TriangleMesh.create_box()
    points = floor_points()
    pcd = cloud(points, np.tile([1., 0, 0], (len(points), 1)))
    result, stats = recover_floor(mesh, pcd, None, floor_params())
    assert result is mesh
    assert stats["status"] == "skipped"


def test_dense_tabletop_does_not_replace_spatially_extensive_floor():
    x, y = np.meshgrid(np.arange(1., 2., .005), np.arange(.5, 1.5, .005))
    tabletop = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .75)])
    pcd = cloud(np.vstack([floor_points(), tabletop]))
    traj = np.array([[.5, 1, 1.3], [2.5, 1, 1.3]])
    floor, stats = recover_floor(o3d.geometry.TriangleMesh(), pcd, traj, floor_params())
    assert stats["status"] == "applied"
    np.testing.assert_allclose(height_at(floor, [[.5, .5]]), 0, atol=1e-6)


def test_completed_floor_replaces_existing_floor_instead_of_stacking_on_it():
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0]]),
                                     o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
    result, stats = recover_floor(mesh, cloud(floor_points()), None, floor_params())
    assert stats["replaced_triangles"] == 2
    assert len(result.triangles) == stats["floor_triangles"]


@pytest.mark.parametrize("update", [{"support_distance": -.1}, {"floor_distance": True},
                                   {"floor_min_points": 2}, {"floor_cell": 0},
                                   {"floor_max_edge": .1}, {"floor_max_slope_deg": 90},
                                   {"floor_distance": .5}, {"floor_clearance": float("nan")}])
def test_invalid_refinement_parameters_fail_before_reconstruction(update):
    with pytest.raises(ContractError, match="config"):
        validate_params(update, PARAMS)
