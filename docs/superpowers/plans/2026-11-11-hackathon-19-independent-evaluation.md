# H19 implementation plan

Execute inline under autonomous authorization. Python/NumPy/Open3D/SciPy only.
Spec: docs/superpowers/specs/2026-11-11-hackathon-19-independent-evaluation-design.md.

- [ ] Add tests using dense planar A/B and two planar meshes; prove B completeness differs from GT and a shifted collision fails while visual passes. Corrupt NPZ or scan hash and require error. Check swapped solution ordering.
- [ ] Implement package loading/validation and explicit common-region evaluator in tools/eval_scene.py, preserving default CLI/report.
- [ ] Run full unittest suite, commit and push code.
- [ ] Run actual GT/baseline/filtered in both modes with pinned H18 package; record outputs, counts and input hashes.
- [ ] Document metric directions and B return weighting; commit/push results and update Plane.
