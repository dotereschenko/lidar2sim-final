import numpy as np
import open3d as o3d
import pytest

from lidar2sim.common import orient_normals
from lidar2sim.normals import correct_occluded_normals
from lidar2sim.baseline import PARAMS
from lidar2sim.validation import ContractError, validate_params


def cloud(xs):
    y, z = np.meshgrid(np.arange(-.3, .301, .01), np.arange(-.3, .301, .01))
    points = np.concatenate([np.column_stack((np.full(y.size, x), y.ravel(), z.ravel())) for x in xs])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.normals = o3d.utility.Vector3dVector(np.tile([1., 0, 0], (len(points), 1)))
    return pcd


@pytest.mark.parametrize("angle", [0., .7])
def test_occluded_nearest_pose_reverses_only_front_sheet(angle):
    pcd = cloud([0., .1])
    trajectory = np.array([[.4, 0, 0], [-.6, 0, 0]])
    points = np.asarray(pcd.points).copy()
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([.2, angle, .3])
    shift = np.array([7., -3., 2.])
    pcd.rotate(rotation, center=(0, 0, 0)).translate(shift)
    trajectory = trajectory @ rotation.T + shift
    orient_normals(pcd, trajectory)
    result = correct_occluded_normals(pcd, trajectory, .03)
    normals = np.asarray(pcd.normals) @ rotation
    center = (np.abs(points[:, 1:]) < .03).all(axis=1)
    assert (normals[center & (points[:, 0] == 0), 0] < -.99).all()
    assert (normals[center & (points[:, 0] == .1), 0] > .99).all()
    assert result['flipped'] > 0


@pytest.mark.parametrize('trajectory', [None, [[.4, 0, 0]], [[.4, 0, 0], [.6, 0, 0]], [[.4, 0, 0], [-.6, 0, 0]]])
def test_clear_or_one_sided_rays_keep_normals(trajectory):
    pcd = cloud([0.])
    trajectory = None if trajectory is None else np.array(trajectory)
    orient_normals(pcd, trajectory)
    before = np.asarray(pcd.normals).copy()
    assert correct_occluded_normals(pcd, trajectory, .03)['flipped'] == 0
    np.testing.assert_array_equal(pcd.normals, before)


def test_isolated_outlier_is_not_an_occluder():
    pcd = cloud([0.])
    pcd.points.append([.1, 0, 0])
    pcd.normals.append([1, 0, 0])
    trajectory = np.array([[.4, 0, 0], [-.6, 0, 0]])
    orient_normals(pcd, trajectory)
    before = np.asarray(pcd.normals).copy()
    correct_occluded_normals(pcd, trajectory, .03)
    np.testing.assert_array_equal(np.asarray(pcd.normals)[:-1], before[:-1])


def test_flag_defaults_off_and_rejects_non_boolean():
    assert validate_params({}, PARAMS)['visibility_normals'] is False
    with pytest.raises(ContractError, match='visibility_normals'):
        validate_params({'visibility_normals': 'true'}, PARAMS)


def test_grazing_nearest_alternative_does_not_hide_visible_pose():
    pcd = cloud([0., .1])
    trajectory = np.array([[.4, 0, 0], [-.01, .5, 0], [-.7, 0, 0]])
    orient_normals(pcd, trajectory)
    correct_occluded_normals(pcd, trajectory, .03)
    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    center = (points[:, 0] == 0) & (np.abs(points[:, 1:]) < .02).all(axis=1)
    assert (normals[center, 0] < -.99).all()
