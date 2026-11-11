"""Repeat the public-A candidate reconstruction and Isaac physics on Spark.

This does not use B, C, or jury-only data. It expects the frozen public A input,
the prebuilt candidate image, and an existing physics work template.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "scans", "config", "template", "jury_reconstruct"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--first", type=int, default=2)
    parser.add_argument("--last", type=int, default=5)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("jury_reconstruct", args.jury_reconstruct)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scans = args.root / "public-a-scans"
    scans.mkdir(exist_ok=True)
    for name, source in (("A.ply", args.scans / "scene.ply"),
                         ("manifest.json", args.scans / "manifest.json"),
                         ("trajectory_A.csv", args.scans / "trajectory_A.csv")):
        link = scans / name
        if not link.exists():
            link.symlink_to(source)

    records = []
    for repeat in range(args.first, args.last + 1):
        output = args.root / f"a-boundary-pad-five-r{repeat}"
        physics = args.root / f"a-physics-boundary-pad-five-r{repeat}"
        if output.exists() or physics.exists():
            raise FileExistsError(f"Repeat {repeat} already exists")

        isolation = module.run(scans, args.config, output, args.image)
        if isolation["exit_code"] or not isolation["required_outputs_present"]:
            raise RuntimeError(f"Public A reconstruction failed in repeat {repeat}")

        physics.mkdir()
        for name in ("isaac_slot.py", "profile.yaml", "test_config.json"):
            shutil.copy2(args.template / name, physics / name)
        for name in ("tools", "lidar2sim"):
            shutil.copytree(args.template / name, physics / name)
        shutil.copy2(output / "scene.usd", physics / "scene.usd")
        container = f"lidar2sim-final-a-five-r{repeat}"
        command = ["docker", "run", "--name", container, "--rm", "--runtime", "nvidia",
                   "--network", "none", "--cpus", "4", "--memory", "16g",
                   "-e", "ACCEPT_EULA=Y", "-e", "OMNI_KIT_ALLOW_ROOT=1",
                   "--entrypoint", "/bin/bash", "-v", f"{physics}:/work",
                   "isaac-sim-xvfb:6.2", "-lc",
                   "set -euo pipefail; Xvfb :99 -screen 0 1280x720x24 -nolisten tcp "
                   "> /work/xvfb.log 2>&1 & export DISPLAY=:99; "
                   "/isaac-sim/python.sh /work/tools/isaac_scene_test.py "
                   "--scene /work/scene.usd --config /work/test_config.json "
                   "--out /work/physics.json --cpu-threads 4 --profile /work/profile.yaml "
                   "> /work/physics.log 2>&1"]
        job = {"command": command, "container_name": container, "timeout_s": 1800,
               "pause_known_sessions": ["isaac-stream", "isaac-flight"],
               "lock_path": "/tmp/lidar2sim-isaac-slot.lock"}
        (physics / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        with (physics / "launcher.log").open("w") as log:
            result = subprocess.run([sys.executable, str(physics / "isaac_slot.py"),
                                     str(physics / "job.json")], stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1900)
        if result.returncode or not (physics / "physics.json").is_file():
            raise RuntimeError(f"Isaac physics failed in repeat {repeat}")
        record = {"repeat": repeat, "reconstruction": str(output),
                  "physics": str(physics), "isolation": isolation}
        records.append(record)
        (args.root / "public-a-five-progress.json").write_text(
            json.dumps(records, indent=2) + "\n")
        print(f"Finished public A repeat {repeat}", flush=True)


if __name__ == "__main__":
    main()
