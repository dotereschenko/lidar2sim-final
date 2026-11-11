import hashlib
import json
import time

import open3d as o3d
import pytest
from pxr import Usd

from lidar2sim.common import write_outputs
from lidar2sim.reporting import evaluation_record, initialize_report, merge_evaluations, validate_report
from lidar2sim.validation import ContractError, validate_outputs


@pytest.fixture
def report_output(tmp_path):
    cloud = tmp_path / "input.ply"
    cloud.write_bytes(b"fixed test input")
    report = initialize_report({"solution": "test", "params": {"voxel": 0.03}, "total_s": 0,
                                "input": {"path": str(cloud), "sha256": hashlib.sha256(cloud.read_bytes()).hexdigest(),
                                          "points": 4}, "timings_s": {}}, cloud=cloud)
    box = o3d.geometry.TriangleMesh.create_box()
    directory = tmp_path / "result"
    write_outputs(directory, box, box, report, started_at=time.perf_counter())
    return directory


def test_generated_report_is_valid_and_does_not_claim_unevaluated_quality(report_output):
    report = json.loads((report_output / "report.json").read_text())
    validate_report(report)
    assert report["checks"]["contract"]["status"] == "passed"
    assert report["checks"]["geometry"]["status"] == "not_run"
    assert report["checks"]["physics"]["status"] == "not_run"
    assert report["acceptance"]["status"] == "not_run"
    assert len(report["code"]["source_sha256"]) == 64
    assert report["sources"]["cloud"]["sha256"] == report["input"]["sha256"]
    assert report["sources"]["trajectory"] is None
    assert report["threshold_profile"]["status"] == "not_provided"
    assert report["runtime"]["packages"]["open3d"] == "0.20.0"
    assert report["total_s"] >= report["timings_s"]["export_and_validation"]
    for name, record in report["artifacts"].items():
        assert record["sha256"] == hashlib.sha256((report_output / name).read_bytes()).hexdigest()


def test_merging_legacy_geometry_preserves_failure_details_and_runtime(report_output):
    path = report_output / "eval_scene.json"
    raw = {"median_cm": 1.7, "passed": {"surface": False, "distance": True}}
    path.write_text(json.dumps(raw))
    before = json.loads((report_output / "report.json").read_text())
    report = merge_evaluations(report_output, geometry=path)
    assert report["checks"]["geometry"]["details"] == raw
    assert report["checks"]["geometry"]["failed_checks"] == ["surface"]
    assert report["checks"]["geometry"]["wall_s"] is None
    assert report["checks"]["physics"]["status"] == "not_run"
    assert report["acceptance"]["status"] == "failed"
    assert report["total_s"] == before["total_s"]
    assert validate_outputs(report_output)["status"] == "passed"


def test_pass_requires_contract_geometry_and_physics(report_output):
    geometry = report_output / "eval_scene.json"
    physics = report_output / "physics.json"
    geometry.write_text('{"passed":{"surface":true,"distance":true}}')
    physics.write_text('{"passed":true,"summary":{"passed":true}}')
    partial = merge_evaluations(report_output, geometry=geometry)
    assert partial["acceptance"]["status"] == "incomplete"
    complete = merge_evaluations(report_output, physics=physics)
    assert complete["acceptance"]["status"] == "incomplete"
    assert complete["checks"]["geometry"]["status"] == "passed"
    profile = report_output / "profile.yaml"
    profile.write_text("version: 1\ngeometry: {median_cm: 3}\n")
    final = merge_evaluations(report_output, profile=profile)
    assert final["acceptance"]["status"] == "passed"
    assert final["threshold_profile"]["source"]["sha256"] == hashlib.sha256(profile.read_bytes()).hexdigest()


@pytest.mark.parametrize("payload", [{"passed": {}}, {"passed": {"a": 1}}, {"passed": "true"},
                                     {"median_cm": 1}, [],
                                     {"passed": True, "summary": {"passed": False}}])
def test_malformed_evaluator_result_cannot_pass(tmp_path, payload):
    path = tmp_path / "physics.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ContractError):
        evaluation_record(path, "physics")


def test_modified_artifact_cannot_reuse_original_report(report_output):
    stage = Usd.Stage.Open(str(report_output / "scene.usd"))
    stage.GetRootLayer().customLayerData = {"edited": "yes"}
    stage.GetRootLayer().Save()
    with pytest.raises(ContractError, match="изменился"):
        merge_evaluations(report_output)


def test_schema_and_json_finiteness_are_enforced(report_output):
    report = json.loads((report_output / "report.json").read_text())
    del report["code"]["source_sha256"]
    with pytest.raises(ContractError, match="source_sha256"):
        validate_report(report)
    report["code"]["source_sha256"] = "0" * 64
    report["checks"]["geometry"]["details"] = {"distance": float("nan")}
    with pytest.raises(ContractError, match="NaN"):
        validate_report(report)
