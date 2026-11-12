"""Measure full process wall time and peak RSS for one 10M-point reconstruction.

The clock starts before Python import/input read and stops after the CLI exits,
including PLY/USD export and report close. Run this script once per config.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def memory_bytes():
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    if Path("/proc/meminfo").exists():
        first = Path("/proc/meminfo").read_text().splitlines()[0]
        return int(first.split()[1]) * 1024
    return None


def cpu_model():
    if sys.platform == "darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       text=True).strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("model name", "Hardware", "Model")):
                return line.partition(":")[2].strip()
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    cloud = args.data / "scene_10M.ply"
    trajectory = args.data / "trajectory.csv"
    manifest = args.data / "manifest.json"
    for path in (cloud, trajectory, manifest, args.config):
        if not path.is_file():
            parser.error(f"missing {path}")
    data_info = json.loads(manifest.read_text())
    if data_info.get("point_count") != 10_000_000:
        parser.error("manifest must declare exactly 10,000,000 points")
    if digest(cloud) != data_info["sha256"]["scene_10M.ply"]:
        parser.error("point cloud hash does not match manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "lidar2sim", "--input", str(cloud),
               "--manifest", str(manifest), "--trajectory", str(trajectory),
               "--config", str(args.config), "--out", str(args.out)]
    started = time.monotonic()
    timed_out = False
    with (args.out / "reconstruction.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            exit_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            exit_code = process.returncode
    wall = time.monotonic() - started
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    measurement = {
        "kind": "full_reconstruction_10m", "input_sha256": digest(cloud),
        "point_count": data_info["point_count"], "config_sha256": digest(args.config),
        "command": command, "timeout_s": args.timeout, "timed_out": timed_out,
        "exit_code": exit_code, "wall_s": round(wall, 3), "peak_rss_bytes": peak_bytes,
        "under_1800_s": bool(exit_code == 0 and wall <= 1800),
        "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                    "cpu_model": cpu_model(), "cpu_count": os.cpu_count(),
                    "ram_bytes": memory_bytes()},
        "runtime": {"python": platform.python_version(), "open3d": metadata.version("open3d"),
                    "numpy": metadata.version("numpy")},
        "report_status": report.get("status") if report else None,
        "report_commit": report.get("code", {}).get("commit") if report else None,
        "report_dirty": report.get("code", {}).get("dirty") if report else None,
        "output_hashes": {name: digest(args.out / name) for name in
                          ("visual.ply", "collision.ply", "scene.usd")
                          if (args.out / name).is_file()},
    }
    (args.out / "benchmark.json").write_text(json.dumps(measurement, indent=2) + "\n")
    print(json.dumps({k: measurement[k] for k in
                      ("exit_code", "wall_s", "peak_rss_bytes", "under_1800_s")},
                     indent=2))
    return 0 if measurement["under_1800_s"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
