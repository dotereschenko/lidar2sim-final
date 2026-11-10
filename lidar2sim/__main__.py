"""python -m lidar2sim --input scene.ply [--manifest manifest.json] [--trajectory trajectory.csv]
                     [--config configs/baseline.yaml] --out out/"""
import argparse
import json
import sys
import time

from . import baseline
from .common import load_inputs, sha256, write_failure_report, write_outputs
from .validation import load_config


def main():
    ap = argparse.ArgumentParser(prog="lidar2sim")
    ap.add_argument("--input", required=True, help="облако точек, PLY, метры")
    ap.add_argument("--manifest", help="manifest.json: границы сцены, оси, версия данных")
    ap.add_argument("--trajectory", help="trajectory.csv: положения лидара t,x,y,z")
    ap.add_argument("--config", help="YAML с параметрами; не заданные берутся по умолчанию")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    try:
        params = load_config(a.config, baseline.PARAMS)
        pcd, meta, traj = load_inputs(a.input, a.manifest, a.trajectory)
        visual, collision, report = baseline.run(pcd, meta, traj, params)
        report["input"] = {"path": a.input, "sha256": sha256(a.input), "points": len(pcd.points)}
        report["config"] = a.config
        report["total_s"] = round(time.perf_counter() - t0, 3)
        write_outputs(a.out, visual, collision, report)
    except (ValueError, OSError, RuntimeError) as exc:
        error = {"status": "failed", "error": str(exc)}
        try:
            write_failure_report(a.out, exc, time.perf_counter() - t0)
        except OSError as report_error:
            error["report_error"] = str(report_error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2 if isinstance(exc, (ValueError, OSError)) else 1
    print(json.dumps({k: report[k] for k in ("solution", "total_s", "outputs")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
