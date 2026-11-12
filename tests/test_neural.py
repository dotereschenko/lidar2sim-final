"""Input/output guarantees that do not require an NKSR installation or GPU."""
import numpy as np
import open3d as o3d
import pytest

from lidar2sim.baseline import PARAMS
from lidar2sim.common import export_usd, preprocess
from lidar2sim.web.worker import normalize


def sphere():
    i = np.arange(1200)
    z = 1 - 2 * (i + .5) / len(i)
    angle = i * 2.399963229728653
    r = np.sqrt(1-z*z)
    points = np.column_stack((r*np.cos(angle), r*np.sin(angle), z))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return cloud


def test_provided_normals_survive_filtering_without_center_flip():
    cloud = sphere()
    cloud.normals = o3d.utility.Vector3dVector(np.asarray(cloud.points) * 3)
    diagnostics = {}
    output = preprocess(cloud, {}, None, PARAMS, diagnostics, preserve_normals=True)
    np.testing.assert_allclose(np.linalg.norm(output.normals, axis=1), 1)
    assert (np.einsum('ij,ij->i', output.points, output.normals) > .99).all()
    assert diagnostics == {'method': 'provided', 'orientation_verified': False}


def test_estimated_normals_are_finite_and_not_marked_verified():
    diagnostics = {}
    output = preprocess(sphere(), {}, None, PARAMS, diagnostics, preserve_normals=True)
    assert np.isfinite(output.normals).all()
    np.testing.assert_allclose(np.linalg.norm(output.normals, axis=1), 1, atol=1e-6)
    assert diagnostics['method'] == 'estimated_consistent_tangent_plane'
    assert diagnostics['orientation_verified'] is False


def test_zero_normals_are_rejected():
    cloud = sphere()
    cloud.normals = o3d.utility.Vector3dVector(np.zeros_like(cloud.points))
    with pytest.raises(ValueError, match='нормали'):
        preprocess(cloud, {}, None, PARAMS, preserve_normals=True)


def test_normal_rotation_uses_no_unit_scale():
    np.testing.assert_allclose(normalize([[0, 1, 0]], 1, 'Y'), [[0, 0, 1]])


def test_usd_preserves_visual_vertex_colors(tmp_path):
    from pxr import Usd, UsdGeom
    visual = o3d.geometry.TriangleMesh.create_box()
    visual.paint_uniform_color([.2, .4, .6])
    collision = o3d.geometry.TriangleMesh.create_box()
    path = tmp_path/'scene.usd'
    export_usd(path, visual, collision)
    stage = Usd.Stage.Open(str(path))
    colors = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Visual')).GetDisplayColorPrimvar()
    assert colors.GetInterpolation() == 'vertex'
    np.testing.assert_allclose(np.asarray(colors.Get()), np.asarray(visual.vertex_colors))


def test_failed_neural_run_cannot_be_reported_as_baseline(tmp_path):
    import json
    from lidar2sim.common import write_failure_report
    write_failure_report(tmp_path, RuntimeError('CUDA unavailable'), .1, solution='nksr')
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['status'] == 'failed'
    assert report['solution'] == 'nksr'


def test_millimetres_misread_as_metres_rejected_before_cuda(monkeypatch):
    from lidar2sim import neural
    cloud=sphere();cloud.scale(1000,center=[0,0,0])
    monkeypatch.setattr(neural,'runtime',lambda:pytest.fail('CUDA must not initialize for invalid extent'))
    with pytest.raises(ValueError,match='Проверьте единицы'):
        neural.run(cloud,{},None)
    cloud.scale(.001,center=[0,0,0]);neural.validate_extent(cloud)
