# Independent A/B Scans Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task; execute inline under the user's autonomous-work instruction.

**Goal:** Produce reproducible independent candidate A/B acquisitions for H13.
**Architecture:** A standalone Open3D raycaster consumes pinned GT and trajectory_A. It writes separate scan and trajectory files plus provenance and diagnostic metrics. Existing performance generator and reconstruction inputs remain unchanged.
**Tech Stack:** Python 3.11, installed NumPy/Open3D, unittest.
**Spec:** docs/superpowers/specs/2026-11-10-hackathon-13-independent-scans-design.md

## Global Constraints

Meters, Z-up; no new dependencies; no source data overwrite; candidate A count 750000 pending H3. B is evaluator-only. Coverage is diagnostic, not H18 eval_region.

## Task 1: Independent acquisition and provenance

Files: create tools/make_ab_scans.py and tests/test_ab_scans.py.
Interface: generate(data, out, points=750000, seed_a=1301, seed_b=1302, batch_size=65536, trajectory_path=None) returns JSON-compatible manifest.

- [x] Write box-fixture tests: 20000 points per scan; cloud-to-box median A between .003 and .02 m, sigma within 5%; dropout between .04 and .06; B poses not in A; two runs have identical manifests and PLY bytes. Reject duplicate seeds, nonpositive count, colliding poses and existing output.
- [x] Run `.venv/bin/python -m unittest tests.test_ab_scans -v`; confirm missing generator fails.
- [x] Implement batched first-hit raycasting and exact accepted counts, noise/dropout statistics, trajectory checks and fixed-seed area sampling coverage. Write strict JSON and binary PLY, hashes for all outputs.
- [x] Run targeted tests and all existing tests. Commit generator, tests, spec and plan; push branch.

## Task 2: Office validation and handoff

Files: README.md and docs/validation/hackathon-13-independent-scans.json.

- [x] Run `.venv/bin/python tools/make_ab_scans.py data/scene_a data/scene_ab_candidate --trajectory-a configs/scans/trajectory_A_candidate.csv` and again into data/scene_ab_repeat.
- [x] Compare manifests and every output hash; audit finite/unique points and record actual metrics in tracked validation JSON.
- [x] Copy candidate files into spark:/home/unknown641/lidar2sim-data/scene_ab_candidate and verify SHA256 remotely.
- [x] Document CLI, B isolation, metrics and unresolved H3 decision. Commit and push evidence; update Plane with branch and concrete results, retaining In Progress.
