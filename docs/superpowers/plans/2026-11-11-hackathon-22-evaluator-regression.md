# H22 implementation plan

Spec: ../specs/2026-11-11-hackathon-22-evaluator-regression-design.md.
Execute inline; NumPy/Open3D/SciPy and existing evaluator only.

- [x] Add failing test for the complete positive/negative regression contract.
- [x] Implement deterministic box fixtures and runner in tools/check_evaluator.py; reuse production region, measurement and evaluation functions.
- [x] Fix any evaluator defect exposed by controls, with tests; run both modes and alone/forward/reverse ordering.
- [x] Run full suite and standalone CLI; record tracked audit and representative reports; commit/push and update Plane.
