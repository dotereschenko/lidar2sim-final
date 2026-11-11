# H13: independent acquisition v2

Update: adopted by the frozen release v2 in `configs/releases/scene-a-v2.json`; see [H3/H13 closure](release-contract-v2.md). Pending statements below describe the original acquisition checkpoint.

The candidate replaces the derived B path with independently authored room/corridor waypoints, interpolated at at most 0.35 m, at height 1.65 m. Each scan casts its own rays with separate seeds. B is not a transformed A cloud. This is an independently planned acquisition, not a claim of statistical independence from the shared GT. Both paths pass surface clearance and segment intersection checks.

Reproduce from the repository root in the locked environment (output directories must not exist):

```sh
uv run --locked python tools/make_ab_scans.py data/scene_a data/scene_ab_v2 \
  --trajectory-a configs/scans/trajectory_A_candidate.csv \
  --trajectory-b configs/scans/trajectory_B_independent_v2.csv \
  --sector-a -15 15 --sector-a 165 180 --sector-a -180 -165
uv run --locked python tools/annotate_scene.py data/scene_a data/scene_ab_v2 \
  out/annotations-v2-independent --definition configs/scans/annotations-v2.json
```

A and B each contain 750,000 finite unique points. Two independent builds have identical manifests and file hashes; both annotation builds also match byte for byte. Measured range noise sigma is 0.0200023 m / 0.0100086 m; cloud-to-GT medians are 0.0083339 m / 0.0040948 m. These are different statistics. A loses 17.1467% of otherwise valid returns to sectors and then 4.9586% to random dropout. Sectors rotate with the XY trajectory tangent; drift and ghosts are explicitly zero. Whole-GT area coverage at 10 cm is 35.49% / 36.34%; the qualified common region is 79,022 of 300,000 sampled GT points. These coverage values do not imply whole-scene reconstruction quality.

B viewpoints are at least 0.32376 m from the A polyline. Exact hashes, runtime versions, counts, trajectories and measurements are in [the audit](hackathon-13-independent-scans-v2.json).

D05 moved 0.30 m along the same doorway, from x=-15.5 to -15.2, because the original section is no longer sufficiently observed in sector-degraded A. Its GT height remains 2.4999997616 m. Both endpoints now have 37/26 A and 38/68 B neighbors, above the unchanged minimum 10 within 10 cm. This choice used GT and scan visibility before reconstruction. All ten dimensions pass the annotation visibility requirement.

The existing detailed Poisson pipeline was rerun on this candidate. Geometry passes the current repository profile; dimensions are 9/10 for both meshes: D01 has insufficient visual surface and an unstable collision surface. See [the evaluation](../../benchmarks/ab-v2-2026-11-11/eval_scene.json). This is a reconstruction handoff failure, not a reason to alter the scan or measurement locations. Physics has not been rerun on this new reconstructed USD. Earlier physics evidence applies only to the earlier USD; annotation physics evidence qualifies unchanged GT only.

The candidate remains pending H3: choose and record the official count and threshold profile. The original document and repository currently disagree on precision reference/threshold and penetration. No organizer approval is asserted. H2 remains open while release, calibration, jury and rehearsal children remain unfinished.

Tests: 142 passed, 185 subtests passed. Files are staged on Spark under `/home/unknown641/lidar2sim-data/scene_ab_v2` and `/home/unknown641/lidar2sim-data/annotations-v2-independent`.
