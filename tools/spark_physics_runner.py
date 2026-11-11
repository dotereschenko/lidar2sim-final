#!/usr/bin/env python3
"""Stage one USD on Spark and run the Isaac evaluator in an isolated container.

Configuration JSON supplies SSH target/port and the existing Isaac image name.
The stage directory is retained on Spark for physics and Xvfb logs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid


def run(command, timeout):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command[0]}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    required = ("ssh_target", "remote_root", "image")
    if any(not isinstance(settings.get(key), str) or not settings[key] for key in required):
        parser.error("settings need ssh_target, remote_root and image")
    root = settings["remote_root"].rstrip("/")
    if not root.startswith("/") or not re.fullmatch(r"[A-Za-z0-9_./-]+", root):
        parser.error("remote_root must be an absolute simple path without spaces")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    port = int(settings.get("ssh_port", 22))
    ssh_options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if settings.get("ssh_control_path"):
        ssh_options += ["-o", "ControlPath=" + settings["ssh_control_path"]]
    ssh = ["ssh", *ssh_options, "-p", str(port), settings["ssh_target"]]
    scp = ["scp", *ssh_options, "-P", str(port)]
    stage = f"{root}/{uuid.uuid4().hex}"
    script = Path(__file__).with_name("isaac_scene_test.py")
    sources = ((args.scene, "scene.usd"), (args.config, "test_config.json"),
               (script, "isaac_scene_test.py"))
    for source, _ in sources:
        if not source.is_file():
            parser.error(f"missing file: {source}")
    timeout = int(settings.get("transfer_timeout_s", 180))
    run([*ssh, "mkdir -p " + shlex.quote(stage)], timeout)
    print("Spark stage:", stage, flush=True)
    for source, name in sources:
        run([*scp, str(source), f"{settings['ssh_target']}:{stage}/{name}"], timeout)
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        actual = run([*ssh, "sha256sum " + shlex.quote(stage + "/" + name)], timeout).split()[0]
        if actual != expected:
            raise RuntimeError(f"Spark transfer hash mismatch: {name}")

    image = settings["image"]
    cpus = int(settings.get("cpus", args.cpu_threads))
    memory_gib = int(settings.get("memory_gib", 16))
    runtime = settings.get("docker_runtime", "nvidia")
    inner = ("set -euo pipefail; Xvfb :99 -screen 0 1280x720x24 -nolisten tcp "
             "> /work/xvfb.log 2>&1 & export DISPLAY=:99; "
             f"/isaac-sim/python.sh /work/isaac_scene_test.py --scene /work/scene.usd "
             f"--config /work/test_config.json --out /work/physics.json "
             f"--cpu-threads {args.cpu_threads} > /work/physics.log 2>&1; "
             "cat /work/physics.log")
    docker = ["docker", "run", "--rm", "--runtime", runtime, "--network", "none",
              "--cpus", str(cpus), "--memory", f"{memory_gib}g",
              "-e", "ACCEPT_EULA=Y", "-e", "OMNI_KIT_ALLOW_ROOT=1",
              "--entrypoint", "/bin/bash", "-v", f"{stage}:/work",
              image, "-lc", inner]
    run([*ssh, shlex.join(docker)], int(settings.get("physics_timeout_s", 1800)))
    run([*scp, f"{settings['ssh_target']}:{stage}/physics.json", str(args.out)], timeout)
    result = json.loads(args.out.read_text())
    if type(result.get("passed")) is not bool:
        raise RuntimeError("Spark physics.json has no boolean passed")
    print("Spark physics result:", result.get("summary"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
        print(f"Spark physics runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
