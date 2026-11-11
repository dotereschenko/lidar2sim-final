# Active PoC Integration Implementation Plan

> Execute inline with SDD under the user's instruction to complete active tasks.

**Goal:** Join qualified evaluation with the implemented PoC and close active tasks only against real evidence.
**Architecture:** Merge histories, adapt existing reporting/orchestration, then validate candidate quality, shared Isaac access, performance and demo in dependency order.
**Tech Stack:** Existing Python/Open3D/USD/pytest/Isaac/Docker/SSH; no new framework.
**Spec:** ../specs/2026-11-11-in-progress-integration-design.md

## Task 1: Preserve and validate both implementations

Files: lidar2sim/baseline.py, validation.py, tools/isaac_scene_test.py, README.md.
- [x] Resolve merge conflicts preserving legacy filtered/floor settings, PR protected collision and qualified physics material/contact/profile.
- [x] Test legacy/new config validation, qualified physics preflight and all imported tests.
- [x] Commit and push integration merge; preserve main and original branches.

## Task 2: H23 qualified evaluator handoff

Files: lidar2sim/reporting.py, tools/run_poc.py, tests/test_reporting.py, tests/test_run_poc.py.
- [x] Add failing tests for qualified geometry/dimensions and mismatched report/profile/artifacts.
- [x] Pass --mode/--scans/--annotations/--dimensions/--profile to geometry, same profile to physics; adapt structured results.
- [x] Refuse stale evidence and preserve explicit missing performance/physics/signoff status.
- [x] Run original/candidate reconstruction and --reuse; save/push reports and update Plane.

## Task 3: H8/H16/H17/H29 remaining live verification

Files: tools/spark_physics_runner.py, configs/spark.example.json, docs, benchmark evidence.
- [x] Test staging of qualified evaluator dependencies and host-wide locking/conflict behavior.
- [x] Obtain a managed shared-GPU slot, run exact candidate physics and record evidence; resolve route floor gaps. Current profile passes 9/10 floor, 10/10 obstacles, 4/5 routes; remaining failures documented.
- [ ] Run pinned 10M benchmark on chosen machine; record full wall/RSS and quality separately.
- [ ] Record final candidate demo, baseline/final comparison and hashes.
- [ ] Finalize H3/H13 decision record from user/organizer choices; close only satisfied Plane tasks.
