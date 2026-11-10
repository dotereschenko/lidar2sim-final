"""Версионируемый отчёт запуска и присоединение результатов оценщиков."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata, resources
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from jsonschema import Draft202012Validator
import yaml

from .validation import ContractError


def file_record(path):
    if path is None:
        return None
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def code_record():
    package = Path(__file__).resolve().parent
    root = package.parent
    paths = [*package.rglob("*.py"), *package.glob("schemas/*.json")]
    paths += [root / name for name in ("pyproject.toml", "uv.lock", ".python-version")]
    hashes = {str(p.relative_to(root)): file_record(p)["sha256"] for p in sorted(paths) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()

    def git(*args):
        try:
            proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
            return proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    commit = git("rev-parse", "HEAD") if (root / "pyproject.toml").is_file() else None
    return {"commit": commit, "branch": git("branch", "--show-current") if commit else None,
            "dirty": bool(git("status", "--porcelain")) if commit else None,
            "source_sha256": fingerprint, "files": hashes}


def runtime_record():
    packages = {}
    for name in ("open3d", "numpy", "scipy", "usd-core", "pyyaml"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "architecture": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages}


def acceptance(checks, profile=None):
    statuses = [checks[name]["status"] for name in ("contract", "geometry", "physics")]
    if "failed" in statuses:
        return {"status": "failed"}
    if all(status == "passed" for status in statuses):
        if profile and profile.get("status") == "provided":
            return {"status": "passed"}
        return {"status": "incomplete", "reason": "Профиль порогов не передан"}
    if checks["geometry"]["status"] == checks["physics"]["status"] == "not_run":
        return {"status": "not_run"}
    return {"status": "incomplete"}


def initialize_report(report, *, cloud, manifest=None, trajectory=None, config=None):
    report["schema_version"] = 1
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "generated"
    report["sources"] = {name: file_record(path) for name, path in
                         (("cloud", cloud), ("manifest", manifest),
                          ("trajectory", trajectory), ("config", config))}
    report["code"] = code_record()
    report["runtime"] = runtime_record()
    report["checks"] = {name: {"status": "not_run", "reason": "Проверка ещё не выполнена"}
                        for name in ("contract", "geometry", "physics")}
    report["threshold_profile"] = {"status": "not_provided", "source": None, "details": None}
    report["acceptance"] = acceptance(report["checks"])
    return report


def validate_report(report):
    try:
        json.dumps(report, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("report.json: нужны JSON-значения без NaN и бесконечностей") from exc
    schema = json.loads(resources.files("lidar2sim").joinpath("schemas/report.schema.json").read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: error.json_path)
    if errors:
        error = errors[0]
        raise ContractError(f"report.json: {error.json_path}: {error.message}")


def evaluation_record(path, kind):
    if path is None:
        return {"status": "not_run", "reason": "Отчёт оценщика не передан"}
    try:
        raw = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ContractError(f"{kind}: некорректный JSON оценщика") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"{kind}: отчёт оценщика должен быть объектом")
    passed = raw.get("passed")
    if isinstance(passed, dict):
        if not passed or any(type(value) is not bool for value in passed.values()):
            raise ContractError(f"{kind}: passed должен содержать непустой набор boolean-проверок")
        ok = all(passed.values())
    elif type(passed) is bool:
        ok = passed
    else:
        raise ContractError(f"{kind}: отсутствует результат passed")
    if kind == "physics" and isinstance(raw.get("summary"), dict):
        summary_passed = raw["summary"].get("passed")
        if summary_passed is not None and (type(summary_passed) is not bool or summary_passed != ok):
            raise ContractError("physics: passed противоречит summary.passed")
    failed = ([name for name, value in passed.items() if not value]
              if isinstance(passed, dict) else ([] if ok else [kind]))
    wall = raw.get("wall_s")
    if wall is not None and (type(wall) not in (int, float) or wall < 0):
        raise ContractError(f"{kind}: wall_s должен быть неотрицательным числом")
    return {"status": "passed" if ok else "failed", "source": file_record(path),
            "details": raw, "failed_checks": failed, "wall_s": wall}


def merge_evaluations(directory, geometry=None, physics=None, profile=None):
    from .validation import validate_outputs

    directory = Path(directory)
    validate_outputs(directory)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("schema_version") != 1:
        raise ContractError("Для объединения нужен report.json версии 1 из нового конвейера")
    for name, record in report["artifacts"].items():
        if file_record(directory / name)["sha256"] != record["sha256"]:
            raise ContractError(f"{name}: артефакт изменился после генерации")
    started = time.perf_counter()
    for kind, path in (("geometry", geometry), ("physics", physics)):
        if path is not None:
            report["checks"][kind] = evaluation_record(path, kind)
    if profile is not None:
        try:
            raw_profile = yaml.safe_load(Path(profile).read_text())
        except (yaml.YAMLError, UnicodeError) as exc:
            raise ContractError("Некорректный JSON/YAML профиля порогов") from exc
        if not isinstance(raw_profile, dict) or not raw_profile:
            raise ContractError("Профиль порогов должен быть непустым объектом")
        report["threshold_profile"] = {"status": "provided", "source": file_record(profile),
                                       "details": raw_profile}
    report["acceptance"] = acceptance(report["checks"], report["threshold_profile"])
    report["timings_s"]["report_merge"] = round(time.perf_counter() - started, 6)
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    validate_report(report)
    temporary = directory / ".report.json.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Добавить результаты оценки в report.json")
    parser.add_argument("directory", help="папка результата конвейера")
    parser.add_argument("--geometry", help="eval_scene.json, полученный для этого результата")
    parser.add_argument("--physics", help="physics.json, полученный для этого результата")
    parser.add_argument("--profile", help="версионированный JSON/YAML с порогами оценки")
    args = parser.parse_args()
    try:
        report = merge_evaluations(args.directory, args.geometry, args.physics, args.profile)
    except (ContractError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"acceptance": report["acceptance"],
                      "checks": {key: value["status"] for key, value in report["checks"].items()}},
                     ensure_ascii=False))
    return 0 if report["acceptance"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
