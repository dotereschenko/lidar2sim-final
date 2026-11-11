# H20 implementation plan

Spec: ../specs/2026-11-11-hackathon-20-dimension-measurement-design.md.
Execute inline under autonomous authorization; existing Python/Open3D/NumPy only.

- [ ] Add failing real-geometry tests for width/height distortion, absent/ambiguous surfaces and invalid annotations.
- [ ] Implement validated probe-bundle dimension measurement in lidar2sim/dimensions.py.
- [ ] Integrate optional --dimensions with common-region evaluator and JSON/PoC status; test CLI and compatibility.
- [ ] Run GT/baseline/filtered with all ten dimensions, publish evidence, commit/push branch and update Plane.
