"""python -m lidar2sim --input scene.ply [--manifest manifest.json] [--trajectory trajectory.csv]
                     [--config configs/baseline.yaml] --out out/"""
import argparse
import json
import sys
import time

_IMPORT_STARTED = time.perf_counter()

from . import baseline
from .common import announce_stage, load_inputs, write_failure_report, write_outputs
from .validation import load_config
from .reporting import initialize_report

_IMPORT_S = time.perf_counter() - _IMPORT_STARTED


def main():
    ap = argparse.ArgumentParser(prog="lidar2sim")
    ap.add_argument("--input", required=True, help="облако точек, PLY, метры")
    ap.add_argument("--manifest", help="manifest.json: границы сцены, оси, версия данных")
    ap.add_argument("--trajectory", help="trajectory.csv: положения лидара t,x,y,z")
    ap.add_argument("--config", help="YAML с параметрами; не заданные берутся по умолчанию")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("poisson", "nksr"), default="poisson")
    a = ap.parse_args()

    t0 = time.perf_counter() - _IMPORT_S
    try:
        announce_stage("loading")
        params = load_config(a.config, baseline.PARAMS)
        pcd, meta, traj = load_inputs(a.input, a.manifest, a.trajectory)
        loaded = time.perf_counter()
        context = initialize_report({}, cloud=a.input, manifest=a.manifest,
                                    trajectory=a.trajectory, config=a.config)
        prepared = time.perf_counter()
        backend = baseline
        if a.backend == "nksr":
            from . import neural
            backend = neural
        announce_stage("reconstructing")
        visual, collision, report = backend.run(pcd, meta, traj, params)
        report.update(context)
        report["timings_s"]["load"] = round(loaded - (t0 + _IMPORT_S), 6)
        report["timings_s"]["provenance"] = round(prepared - loaded, 6)
        report["timings_s"]["imports"] = round(_IMPORT_S, 6)
        report["input"] = {"path": a.input, "sha256": context["sources"]["cloud"]["sha256"],
                           "points": len(pcd.points)}
        report["config"] = a.config
        report["total_s"] = round(time.perf_counter() - t0, 3)
        announce_stage("exporting")
        write_outputs(a.out, visual, collision, report, started_at=t0)
    except (ValueError, OSError, RuntimeError) as exc:
        error = {"status": "failed", "error": str(exc)}
        try:
            write_failure_report(a.out, exc, time.perf_counter() - t0,
                                 solution="nksr" if a.backend == "nksr" else "baseline")
        except OSError as report_error:
            error["report_error"] = str(report_error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2 if isinstance(exc, (ValueError, OSError)) else 1
    print(json.dumps({k: report[k] for k in ("solution", "total_s", "outputs")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
