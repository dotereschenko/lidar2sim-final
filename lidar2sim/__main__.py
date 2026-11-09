"""python -m lidar2sim --input scene.ply [--manifest manifest.json] [--trajectory trajectory.csv]
                     [--config configs/baseline.yaml] --out out/"""
import argparse
import json
import time

import yaml

from . import baseline
from .common import load_inputs, sha256, write_outputs


def main():
    ap = argparse.ArgumentParser(prog="lidar2sim")
    ap.add_argument("--input", required=True, help="облако точек, PLY, метры")
    ap.add_argument("--manifest", help="manifest.json: границы сцены, оси, версия данных")
    ap.add_argument("--trajectory", help="trajectory.csv: положения лидара t,x,y,z")
    ap.add_argument("--config", help="YAML с параметрами; не заданные берутся по умолчанию")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.perf_counter()
    pcd, meta, traj = load_inputs(a.input, a.manifest, a.trajectory)
    params = yaml.safe_load(open(a.config)) if a.config else None
    visual, collision, report = baseline.run(pcd, meta, traj, params)
    report["input"] = {"path": a.input, "sha256": sha256(a.input), "points": len(pcd.points)}
    report["config"] = a.config
    report["total_s"] = round(time.perf_counter() - t0, 3)
    write_outputs(a.out, visual, collision, report)
    print(json.dumps({k: report[k] for k in ("solution", "total_s", "outputs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
