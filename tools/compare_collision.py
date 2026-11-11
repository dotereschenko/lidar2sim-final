"""Сравнить упрощение коллайдера на одном Poisson-меше сцены A.

    python tools/compare_collision.py --data data/scene_a --out out/collision-comparison

GT используется только оценщиком. Входы реконструкции — A, manifest и траектория.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar2sim.baseline import PARAMS
from lidar2sim.collision import build_collision
from lidar2sim.common import clean, load_inputs, poisson, preprocess, sha256
from lidar2sim.refinement import recover_floor, trim_unsupported
from lidar2sim.validation import load_config
from eval_scene import extra_within, raycasting, region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    args = parser.parse_args()
    p = load_config(args.config, PARAMS)
    args.out.mkdir(parents=True, exist_ok=True)
    cloud, meta, traj = load_inputs(args.data / "A.ply", args.data / "manifest.json",
                                   args.data / "trajectory_A.csv")
    prepared = preprocess(cloud, meta, traj, p)
    mesh = clean(poisson(prepared, p["poisson_depth"], p["density_quantile"])
                 .crop(prepared.get_axis_aligned_bounding_box()))
    mesh, support = trim_unsupported(mesh, prepared, p["support_distance"])
    mesh, floor_info, rest, floor = recover_floor(mesh, prepared, traj, p,
                                                  return_components=True)
    if floor is None:
        raise RuntimeError("На сцене A не удалось выделить опорный пол")

    o3d.utility.random.seed(0)
    gt, reference, observed_share = region(args.data)
    gt_scene = raycasting(gt)
    variants = [("global-20k", 20000, 0., 1.),
                ("global-30k", 30000, 0., 1.)]
    variants += [(f"protected-{budget // 1000}k-floor-{fraction:g}-boundary-{weight:g}",
                  budget, fraction, weight)
                 for budget in (20000, 30000, 40000)
                 for fraction in (.15, .25, .3, .4)
                 for weight in (1., 100.)]
    report = {"kind": "collision_ablation", "cached_poisson": True,
              "input_sha256": sha256(args.data / "A.ply"),
              "gt_sha256": sha256(args.data / "gt_mesh.ply"),
              "config_sha256": sha256(args.config),
              "observed_reference_share": float(observed_share),
              "support": support, "floor": floor_info, "variants": {}}
    for name, budget, fraction, weight in variants:
        candidate, stats = build_collision(mesh, rest, floor, budget, fraction, weight)
        o3d.utility.random.seed(0)
        accuracy = round(extra_within(candidate, gt_scene, .05), 1)
        distances = raycasting(candidate).compute_distance(
            o3d.core.Tensor(reference.astype(np.float32))).numpy()
        report["variants"][name] = {"collision": stats, "extra_5cm_pct": accuracy,
                                    "median_cm": round(float(np.median(distances)) * 100, 2),
                                    "p95_cm": round(float(np.percentile(distances, 95)) * 100, 2)}
        (args.out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"{name:49} {len(candidate.triangles):6d} triangles  {accuracy:4.1f}%", flush=True)


if __name__ == "__main__":
    main()
