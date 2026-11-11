# Sector A/B v2: high-detail reconstruction

The new sector-degraded A exposed unstable doorway D01 surfaces in the existing detailed Poisson reconstruction. The defect was present in raw Poisson, before support trimming and simplification. The accepted v2 configuration is an additional `configs/poc-high-detail.yaml`: depth 10, original normals, visual budget 200,000, collision budget 200,000 including 5,000 protected floor triangles. No scan, annotation, evaluator or threshold was changed.

## Verified results

| Check | Existing detailed profile on v2 | High-detail profile on v2 |
|---|---:|---:|
| Visual dimensions | 9/10 | 10/10 |
| Collision dimensions | 9/10 | 10/10 |
| Visual mesh→GT precision within 5 cm | 99.28% | 99.33% |
| Collision mesh→GT precision within 5 cm | 99.46% | 99.31% |
| B→visual completeness p95 | 4.91 cm | 4.76 cm |
| B→collision completeness p95 | 5.27 cm | 4.76 cm |
| Isaac floor | not run | 10/10 |
| Isaac obstacles | not run | 10/10 |
| Isaac routes | not run | 5/5 |

Maximum recorded obstacle penetration is 0.0006 m (0.6 mm). These results use the current repository profile; they do not establish the original document's differently defined precision against B. Isaac tests use the configured box surrogate, not autonomous robot navigation.

Exact qualified USD SHA256: `3d415b3e2305757f9b32c21218078b45522e4f0aefad52093875234841bd0deb`. Spark stage: `/home/unknown641/lidar2sim-poc-runs/ad7419afd5ad4522a9c6160e663d5a6b`. The guard restored both paused Isaac sessions; `isaac-stream` was healthy, `isaac-flight` running/starting after restore, and Foxglove remained running.

The full CLI reconstruction took 51.72 s locally. A separate HTTP upload of A and its trajectory using `method=high-detail` completed in 46.31 s and produced the **same USD hash**. All four downloadable artifact bytes matched the job files. The page rendered in Chrome, but native automation did not confirm selector interaction; API/config selection and worker execution were checked separately.

A single full 10M-point reconstruction on Apple M1 Pro, 10 CPUs, 32 GiB RAM took **109.460 s**, peak RSS **1,897,103,360 bytes** (1.77 GiB). This is a local speed/memory measurement, not qualification on the designated Ubuntu machine or proof of 10M geometry quality.

## Why this is opt-in

Changing normal radius alone did not fix D01. Increasing depth alone restored visual D01 but the 60k collider lost detail; 120k passed D01 but lost the tabletop; 200k passed all ten v2 dimensions. Broader normals with depth 10 regressed v2 to 9/10 visual and 8/10 collision.

Depth 10 also exposed unstable tabletop D09 on the earlier acquisition (9/10 dimensions for both meshes). Consequently the previous `poc-detailed.yaml` and UI default are unchanged. High detail is an explicit alternative, not a universal quality improvement; hidden-scene generalization remains unqualified. Failed experiments are retained alongside the final report.

## Reproduce

```sh
uv run --locked python tools/run_poc.py \
  --data data/scene_ab_v2 --reference-data data/scene_a \
  --scans data/scene_ab_v2 --annotations out/annotations-v2-independent \
  --evaluation-mode b --config configs/poc-high-detail.yaml \
  --out out/poc-ab-v2-high-detail \
  --physics-runner tools/spark_physics_runner.py \
  --runner-arg=--settings --runner-arg=/path/to/operator-spark.json
```

For UI, start `python -m lidar2sim.web.server`, select «Высокая детализация · Poisson», and upload A plus its matching trajectory. Existing running servers require a restart to load the new API method; the review instance is on port 8766 with a separate job directory.

Full existing suite plus three configuration-selection cases passes (145 tests, including 185 subtests); JS syntax check passes. Evidence: [final geometry, physics and report](../../benchmarks/sector-doorway-2026-11-11/high-detail-v2/), [HTTP verification](../../benchmarks/sector-doorway-2026-11-11/web-check.json), [local 10M benchmark](../../benchmarks/sector-doorway-2026-11-11/benchmark-10m-local.json).

Aggregate acceptance remains `incomplete`: H3 approval and designated-machine performance evidence are absent. This result closes the observed v2 reconstruction defect, not the whole Denis epic, release package, jury rehearsal or hidden-scene work.
