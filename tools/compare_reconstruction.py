"""Геометрическая абляция на A; GT загружается только отдельным оценщиком.

    python tools/compare_reconstruction.py --data data/scene_a --out out/comparison

Poisson считается один раз, чтобы сравнить отсечение и достройку на одной сетке.
Это эксперимент с PLY, а не полный экспорт или измерение времени конвейера.
Итоговую конфигурацию запускайте отдельно через python -m lidar2sim.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import open3d as o3d

from lidar2sim.baseline import PARAMS
from lidar2sim.common import clean, decimate, load_inputs, poisson, preprocess, sha256
from lidar2sim.refinement import recover_floor, trim_unsupported
from lidar2sim.reporting import code_record, runtime_record
from lidar2sim.validation import load_config

# Скрипт запускается как tools/compare_reconstruction.py, рядом с оценщиком Дениса.
from eval_scene import THRESHOLDS, evaluate, raycasting, region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--support-distances", type=float, nargs="+", default=[.04, .06, .08, .10])
    args = parser.parse_args()
    p = load_config(args.config, PARAMS)
    distances = args.support_distances
    if not distances or any(not 0 < d < 1 for d in distances):
        parser.error("support-distances должны быть в (0, 1) м")
    data, out = args.data, args.out
    pcd, meta, traj = load_inputs(data / "A.ply", data / "manifest.json", data / "trajectory_A.csv")
    pcd = preprocess(pcd, meta, traj, p)
    raw = clean(poisson(pcd, p["poisson_depth"], p["density_quantile"]).crop(pcd.get_axis_aligned_bounding_box()))
    result = {"kind": "geometry_ablation", "cached_poisson": True, "code": code_record(),
              "runtime": runtime_record(),
              "inputs_sha256": {name: sha256(data / name)
                                for name in ("A.ply", "manifest.json", "trajectory_A.csv")},
              "config_sha256": sha256(args.config), "raw_triangles": len(raw.triangles),
              "thresholds": THRESHOLDS, "variants": {}}
    variants = [("baseline", 0., 0.)]
    variants += [(f"support-{d:.3f}", d, 0.) for d in distances]
    variants += [("floor-only", 0., p["floor_distance"]),
                 ("poc", p["support_distance"], p["floor_distance"])]

    # У реконструкции нет доступа к GT; он используется далее только для оценки.
    o3d.utility.random.seed(0)
    gt, reference, share = region(data)
    gt_scene = raycasting(gt)
    result["evaluation_gt_sha256"] = sha256(data / "gt_mesh.ply")
    result["observed_reference_share"] = float(share)
    for name, support, floor in variants:
        params = {**p, "support_distance": support, "floor_distance": floor}
        mesh = copy.deepcopy(raw)
        details = {}
        if support:
            mesh, details["support"] = trim_unsupported(mesh, pcd, support)
        if floor:
            mesh, details["floor"] = recover_floor(mesh, pcd, traj, params)
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        counts = {}
        for kind, budget in (("visual", p["visual_triangles"]), ("collision", p["collision_triangles"])):
            exported = decimate(mesh, budget)
            if not o3d.io.write_triangle_mesh(str(directory / (kind + ".ply")), exported):
                raise OSError(f"Не удалось записать {kind}.ply")
            counts[kind] = len(exported.triangles)
        o3d.utility.random.seed(0)
        metrics = evaluate(directory, gt_scene, reference)
        result["variants"][name] = {"params": params, "refinement": details,
                                     "triangles": counts, "geometry": metrics}
        (out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({"variant": name, **metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
