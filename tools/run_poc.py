"""Run reconstruction and geometry locally, then attach an Isaac physics result.

The physics runner has the same CLI as tools/isaac_scene_test.py and may run in
another environment after arranging access to the scene and test configuration.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from lidar2sim.reporting import acceptance, merge_evaluations, validate_report
from lidar2sim.validation import ContractError, validate_outputs


ROOT = Path(__file__).resolve().parent.parent


class StageError(Exception):
    def __init__(self, stage, message):
        self.stage = stage
        super().__init__(message)


def run_command(stage, command, timeout, log):
    started = time.monotonic()
    with log.open("a") as stream:
        stream.write(f"\n[{stage}] started\n")
        stream.flush()
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise StageError(stage, f"timeout after {timeout} s; see {log}") from exc
        except OSError as exc:
            raise StageError(stage, f"could not start: {exc}; see {log}") from exc
        elapsed = time.monotonic() - started
        stream.write(f"[{stage}] exit={result.returncode} elapsed={elapsed:.2f} s\n")
    if result.returncode:
        raise StageError(stage, f"exit code {result.returncode}; see {log}")


def record_failure(out, stage, message):
    path = out / "report.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text())
    if report.get("status") != "generated":
        return
    check = "geometry" if stage == "geometry" else "physics" if stage == "physics" else "contract"
    report["checks"][check] = {"status": "failed", "reason": message}
    report["acceptance"] = acceptance(report["checks"], report["threshold_profile"])
    validate_report(report)
    temporary = out / ".report.run_poc.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mark_physics_unavailable(out):
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["checks"]["physics"] = {"status": "not_run", "reason": "Isaac runner not configured"}
    report["acceptance"] = acceptance(report["checks"], report["threshold_profile"])
    validate_report(report)
    temporary = out / ".report.run_poc.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return report["acceptance"]["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder with A.ply, manifest, trajectory, GT and test config")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/poc.yaml")
    parser.add_argument("--reuse", action="store_true", help="evaluate existing PLY/USD without reconstruction")
    parser.add_argument("--profile", type=Path, help="final threshold profile, when available")
    physics = parser.add_mutually_exclusive_group()
    physics.add_argument("--isaac-python", type=Path, help="Isaac python.sh in the current environment")
    physics.add_argument("--physics-runner", type=Path, help="runner accepting --scene, --config, --out and --cpu-threads")
    physics.add_argument("--physics-result", type=Path, help="existing physics.json from the same USD")
    parser.add_argument("--runner-arg", action="append", default=[], help="argument passed to --physics-runner before test arguments")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cpu-timeout", type=int, default=600)
    parser.add_argument("--geometry-timeout", type=int, default=600)
    parser.add_argument("--physics-timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1 or min(args.cpu_timeout, args.geometry_timeout, args.physics_timeout) < 1:
        parser.error("thread count and timeouts must be positive")
    if args.runner_arg and not args.physics_runner:
        parser.error("--runner-arg requires --physics-runner")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    cloud = args.data / "A.ply"
    manifest = args.data / "manifest.json"
    trajectory = args.data / "trajectory_A.csv"
    test_config = args.data / "test_config.json"
    try:
        if not args.reuse:
            run_command("reconstruction", [sys.executable, "-m", "lidar2sim",
                        "--input", str(cloud), "--manifest", str(manifest),
                        "--trajectory", str(trajectory), "--config", str(args.config),
                        "--out", str(out)], args.cpu_timeout, log)
        try:
            validate_outputs(out)
        except (ContractError, OSError) as exc:
            raise StageError("contract", str(exc)) from exc

        run_command("geometry", [sys.executable, str(ROOT / "tools/eval_scene.py"),
                    str(args.data), str(out)], args.geometry_timeout, log)
        try:
            merge_evaluations(out, out / "eval_scene.json", profile=args.profile)
        except (ContractError, OSError) as exc:
            raise StageError("geometry", str(exc)) from exc

        physics_path = out / "physics.json"
        if args.physics_result:
            try:
                if args.physics_result.resolve() != physics_path:
                    shutil.copyfile(args.physics_result, physics_path)
            except OSError as exc:
                raise StageError("physics", str(exc)) from exc
        elif args.isaac_python or args.physics_runner:
            command = ([str(args.isaac_python), str(ROOT / "tools/isaac_scene_test.py")]
                       if args.isaac_python else [str(args.physics_runner), *args.runner_arg])
            command += ["--scene", str(out / "scene.usd"), "--config", str(test_config),
                        "--out", str(physics_path), "--cpu-threads", str(args.cpu_threads)]
            run_command("physics", command, args.physics_timeout, log)
        else:
            status = mark_physics_unavailable(out)
            print(json.dumps({"status": status, "physics": "not_run", "reason": "Isaac runner not configured"}))
            return 1

        try:
            report = merge_evaluations(out, physics=physics_path)
        except (ContractError, OSError) as exc:
            raise StageError("physics", str(exc)) from exc
        status = report["acceptance"]["status"]
        print(json.dumps({"status": status,
                          "checks": {k: v["status"] for k, v in report["checks"].items()}},
                         ensure_ascii=False))
        return 0 if status == "passed" else 1
    except (StageError, OSError) as exc:
        stage = exc.stage if isinstance(exc, StageError) else "contract"
        message = str(exc)
        try:
            record_failure(out, stage, message)
        except (ContractError, OSError, KeyError, ValueError):
            pass
        print(json.dumps({"status": "failed", "stage": stage, "error": message},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
