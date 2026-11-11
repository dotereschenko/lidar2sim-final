import json
import subprocess
import sys

import numpy as np
import open3d as o3d
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from lidar2sim import baseline
from lidar2sim.common import load_inputs, write_outputs
from lidar2sim.validation import ContractError, load_config, validate_outputs


@pytest.fixture
def inputs(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = tmp_path / "scene.ply"
    assert o3d.io.write_point_cloud(str(cloud), pcd)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"units": "m", "up_axis": "Z", "bounds_min": [-1] * 3,
                                    "bounds_max": [2] * 3}))
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text("t,x,y,z\n0,0,0,0\n1,1,1,1\n")
    return cloud, manifest, trajectory


def test_load_inputs_preserves_points_and_trajectory(inputs):
    pcd, meta, trajectory = load_inputs(*inputs)
    assert len(pcd.points) == 4
    assert meta["units"] == "m"
    np.testing.assert_equal(trajectory, [[0, 0, 0], [1, 1, 1]])


@pytest.mark.parametrize("update", [{"units": "cm"}, {"up_axis": "Y"},
                                   {"bounds_min": [3, 3, 3]}, {"bounds_max": [1, 2]},
                                   {"bounds_min": [float("nan"), 0, 0]}])
def test_invalid_manifest_is_rejected(inputs, update):
    cloud, manifest, trajectory = inputs
    meta = json.loads(manifest.read_text())
    meta.update(update)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ContractError, match="manifest"):
        load_inputs(cloud, manifest, trajectory)


@pytest.mark.parametrize("contents", ["t,x,y\n0,0,0\n", "t,x,y,z\n0,0,0,0\n0,1,1,1\n",
                                      "t,x,y,z\n0,nan,0,0\n", "t,x,y,z\n"])
def test_invalid_trajectory_is_rejected(inputs, contents):
    cloud, manifest, trajectory = inputs
    trajectory.write_text(contents)
    with pytest.raises(ContractError, match="trajectory"):
        load_inputs(cloud, manifest, trajectory)


def test_nonfinite_cloud_is_not_silently_filtered(inputs):
    cloud, manifest, trajectory = inputs
    pcd = o3d.io.read_point_cloud(str(cloud))
    np.asarray(pcd.points)[0, 0] = np.nan
    assert o3d.io.write_point_cloud(str(cloud), pcd)
    with pytest.raises(ContractError, match="NaN"):
        load_inputs(cloud, manifest, trajectory)


@pytest.mark.parametrize("contents", ["[]", "voxel: 0", "voxel: .nan", "poisson_depth: true",
                                      "poisson_depth: 3.5", "density_quantile: 1", "voxle: 0.03"])
def test_invalid_config_is_rejected(tmp_path, contents):
    config = tmp_path / "config.yaml"
    config.write_text(contents)
    with pytest.raises(ContractError, match="config"):
        load_config(config, baseline.PARAMS)


@pytest.fixture
def outputs(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    report = {"solution": "test", "params": {}, "input": {"sha256": "0" * 64}, "total_s": 0.1}
    directory = tmp_path / "out"
    write_outputs(directory, mesh, mesh, report)
    return directory


def test_export_roundtrip_has_static_collider_and_matching_report(outputs):
    assert validate_outputs(outputs)["status"] == "passed"
    report = json.loads((outputs / "report.json").read_text())
    assert report["status"] == "generated"
    assert report["outputs"] == {"visual_triangles": 12, "collision_triangles": 12}


@pytest.mark.parametrize("corruption", ["axis", "units", "disabled_collider", "visual_collider",
                                       "rigid_body", "missing_prim", "changed_mesh"])
def test_corrupt_usd_is_rejected(outputs, corruption):
    stage = Usd.Stage.Open(str(outputs / "scene.usd"))
    collision = stage.GetPrimAtPath("/World/Collision")
    visual = stage.GetPrimAtPath("/World/Visual")
    if corruption == "axis":
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    elif corruption == "units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    elif corruption == "disabled_collider":
        UsdPhysics.CollisionAPI(collision).CreateCollisionEnabledAttr(False)
    elif corruption == "visual_collider":
        UsdPhysics.CollisionAPI.Apply(visual)
    elif corruption == "rigid_body":
        UsdPhysics.RigidBodyAPI.Apply(collision)
    elif corruption == "missing_prim":
        stage.RemovePrim("/World/Collision")
    else:
        points = np.asarray(UsdGeom.Mesh(visual).GetPointsAttr().Get()).copy()
        points[:, 0] += 0.2
        UsdGeom.Mesh(visual).GetPointsAttr().Set(points.tolist())
    stage.GetRootLayer().Save()
    with pytest.raises(ContractError, match="scene.usd"):
        validate_outputs(outputs)


def test_incomplete_output_is_rejected(outputs):
    (outputs / "collision.ply").unlink()
    with pytest.raises(ContractError, match="collision.ply"):
        validate_outputs(outputs)


def test_report_with_wrong_triangle_count_is_rejected(outputs):
    path = outputs / "report.json"
    report = json.loads(path.read_text())
    report["outputs"]["collision_triangles"] = 999
    path.write_text(json.dumps(report))
    with pytest.raises(ContractError, match="report.json"):
        validate_outputs(outputs)


def test_report_with_wrong_collision_component_count_is_rejected(outputs):
    path = outputs / "report.json"
    report = json.loads(path.read_text())
    report["collision_build"] = {"mode": "protected_floor", "target_triangles": 20,
                                 "triangles": 12, "floor_fraction": .4, "boundary_weight": 1.,
                                 "floor_triangles": 5, "other_triangles": 6,
                                 "floor_faces_upward": True}
    path.write_text(json.dumps(report))
    with pytest.raises(ContractError, match="collision_build"):
        validate_outputs(outputs)


def test_failed_cli_replaces_previous_success_report(inputs, tmp_path):
    directory = tmp_path / "output"
    directory.mkdir()
    (directory / "report.json").write_text('{"status":"generated"}')
    config = tmp_path / "bad.yaml"
    config.write_text("voxel: -1")
    result = subprocess.run([sys.executable, "-m", "lidar2sim", "--input", str(inputs[0]),
                             "--config", str(config), "--out", str(directory)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert json.loads(result.stderr)["status"] == "failed"
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed"
    assert "voxel" in report["error"]["message"]
