import hashlib
import json
from pathlib import Path
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


def test_legacy_geometry_cannot_claim_full_acceptance(report_output):
    geometry = report_output / "eval_scene.json"
    physics = report_output / "physics.json"
    geometry.write_text('{"passed":{"surface":true,"distance":true}}')
    from lidar2sim.acceptance import load_profile, DEFAULT_PROFILE
    config = Path(__file__).resolve().parents[1]/"data/scene_a/test_config.json"
    rules, provenance = load_profile()
    payload = {"passed":True,"summary":{"passed":True},"acceptance_profile":provenance,
               "inputs":{"scene_sha256":hashlib.sha256((report_output/"scene.usd").read_bytes()).hexdigest(),
                         "config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
                         "runner_sha256":hashlib.sha256((config.parents[2]/"tools/isaac_scene_test.py").read_bytes()).hexdigest()}}
    payload.update({g:[{"ok":True} for _ in range(rules["physics"][f"{g}_count"])] for g in ("floor","obstacles","routes")})
    physics.write_text(json.dumps(payload))
    partial = merge_evaluations(report_output, geometry=geometry)
    assert partial["acceptance"]["status"] == "incomplete"
    complete = merge_evaluations(report_output, physics=physics, physics_config=config)
    assert complete["acceptance"]["status"] == "incomplete"
    assert complete["checks"]["geometry"]["status"] == "passed"
    profile = report_output / "profile.yaml"
    profile.write_bytes(DEFAULT_PROFILE.read_bytes())
    final = merge_evaluations(report_output, profile=profile)
    assert final["acceptance"]["status"] == "incomplete"
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


def test_arbitrary_profile_cannot_create_final_acceptance(report_output):
    path=report_output/'fake-profile.yaml';path.write_text('anything: true\n')
    with pytest.raises(ContractError,match='profile'):
        merge_evaluations(report_output,profile=path)


def test_qualified_geometry_requires_matching_meshes_and_keeps_missing_evidence(report_output):
    from lidar2sim.acceptance import load_profile, DEFAULT_PROFILE
    profile, provenance=load_profile()
    raw={'evaluation_mode':'independent_b_common_region','acceptance_profile':provenance,
         'geometry_status':'passed','dimensions_status':'passed','meshes':{}}
    for name in ['visual','collision']:
        raw['meshes'][name]={'sha256':hashlib.sha256((report_output/f'{name}.ply').read_bytes()).hexdigest(),
            'passed':{'precision':True,'median':True,'p95':True},
            'dimensions':{'status':'passed','required_count':10,'passed_count':10,
                          'items':[{'id':f'D{i}','passed':True,'status':'passed'} for i in range(10)]}}
    path=report_output/'eval_scene.json';path.write_text(json.dumps(raw))
    result=merge_evaluations(report_output,geometry=path,profile=DEFAULT_PROFILE)
    assert result['checks']['geometry']['status']=='passed'
    assert result['checks']['dimensions']['status']=='passed'
    assert result['checks']['performance']['status']=='not_run'
    assert result['acceptance']['status']=='incomplete'
    raw['meshes']['collision']['sha256']='0'*64;path.write_text(json.dumps(raw))
    with pytest.raises(ContractError,match='hash'):
        merge_evaluations(report_output,geometry=path)


def test_unbound_physics_result_cannot_be_reused(report_output):
    path=report_output/'physics.json';path.write_text('{"passed":true,"summary":{"passed":true}}')
    with pytest.raises(ContractError,match='provenance'):
        merge_evaluations(report_output,physics=path)
