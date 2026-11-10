# Evaluation Annotations Implementation Plan

> Execute inline with superpowers:executing-plans under the user's autonomous-work authorization.

**Goal:** Pin a common A/B observation rule and ten GT dimensions with two checked doors.
**Architecture:** A shared observation predicate and annotation builder export candidate JSON/NPZ. A standalone overlay provides visual QA. Existing evaluator stays unchanged until H19/H20.
**Tech Stack:** Python, NumPy, SciPy cKDTree, Open3D, standard library.
**Spec:** docs/superpowers/specs/2026-11-11-hackathon-18-annotations-design.md

## Global constraints

Meters/Z-up; preserve original inputs; no new dependencies; minimum common sample 1000 of 300000; no organizer approval inferred.

## Task 1: Region and dimensions

Files: lidar2sim/annotations.py, tools/annotate_scene.py, configs/scans/annotations-v1.json, tests/test_annotations.py.
- [ ] Write failing tests for A/B intersection, nonfinite/empty inputs, box dimensions, missing hits and obstructed door interior.
- [ ] Run `.venv/bin/python -m unittest tests.test_annotations -v` to confirm missing implementation.
- [ ] Implement `observed_by_both(points, a, b, radius=.10, min_neighbors=10)` and `dimension(scene, definition)` with unit directions, finite hits and GT endpoint boxes.
- [ ] Implement `door(scene, definition, body_size)` and builder CLI, uniform-area sampling and provenance validation against H12.
- [ ] Run targeted/full tests, commit and push.

## Task 2: Data and visual validation

Files: docs/validation/hackathon-18-annotations.json, docs/img/annotations-v1.png, README.md.
- [ ] Build `out/annotations-v1` and repeat; compare artifact hashes.
- [ ] Inspect the generated overlay against GT/A/B, check dimension/door placements and record candidate limitations.
- [ ] Copy package to Spark, verify hashes, commit/push evidence and update Plane.
