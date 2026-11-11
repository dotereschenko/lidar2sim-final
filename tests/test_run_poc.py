import json

import open3d as o3d

from lidar2sim.common import write_outputs
from lidar2sim.reporting import initialize_report
from tools.run_poc import mark_physics_unavailable, record_failure


def scene(tmp_path):
    mesh = o3d.geometry.TriangleMesh.create_box()
    out = tmp_path / "scene"
    cloud = tmp_path / "cloud.ply"
    cloud.write_text("test fixture")
    report = initialize_report({"solution": "test", "params": {},
                                "input": {"path": str(cloud), "sha256": "0" * 64,
                                          "points": 4}, "total_s": 0.1,
                                "timings_s": {}}, cloud=cloud)
    write_outputs(out, mesh, mesh, report)
    return out


def test_unavailable_isaac_is_explicitly_incomplete(tmp_path):
    out = scene(tmp_path)
    report_path = out / "report.json"
    before = json.loads(report_path.read_text())
    before["checks"]["geometry"] = {"status": "passed"}
    report_path.write_text(json.dumps(before))
    status = mark_physics_unavailable(out)
    report = json.loads((out / "report.json").read_text())
    assert status == "incomplete"
    assert report["checks"]["physics"] == {"status": "not_run",
                                            "reason": "Isaac runner not configured"}
    assert report["acceptance"]["status"] == "incomplete"


def test_external_evaluator_failure_propagates_to_report(tmp_path):
    out = scene(tmp_path)
    record_failure(out, "physics", "runner timed out")
    report = json.loads((out / "report.json").read_text())
    assert report["checks"]["physics"]["status"] == "failed"
    assert report["acceptance"]["status"] == "failed"
    assert "runner timed out" in report["checks"]["physics"]["reason"]
