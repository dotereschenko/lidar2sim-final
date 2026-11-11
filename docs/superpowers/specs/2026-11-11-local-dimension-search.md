# SDD: local search-box dimension measurement

The frozen annotation contract says to find reconstructed surfaces in the two endpoint search boxes, without substituting GT endpoints. The current implementation instead uses the first hit from the dimension origin. Unrelated foreground geometry can hide a valid endpoint and falsely produce `missing_surface_in_search_box`.

A scene-independent counterexample is a 4 m room with an additional small object between the center and one wall. The wall positions and room width are unchanged, but the old implementation rejects the measurement. This also explains one C failure: its global probe reaches a foreground doorway edge before its target box. Do not move C annotations, change a reference length, or widen tolerances to repair this implementation error.

Version the dimension method as `local-box-ray-bundle-v3`. Clip each of the five directed probe rays to the entry of its assigned box, then measure actual triangle hits and enforce the existing box bounds, center-probe requirement, >=4/5 valid probes, <=2 cm spread, and <=5 cm length error. Support arbitrary unit axes. Keep profile v2 numerical criteria and all frozen datasets unchanged. Missing target surfaces, 10 cm distortion and ambiguous stepped surfaces must still fail.

Pin the dimension implementation SHA in release report merging and jury manifests, rejecting mixed old/new methods. Preserve all old reports; re-evaluate copies with explicit provenance. Existing physics may be reused only with unchanged USD/config/profile/runner hashes, and must not be represented as a new physical run. This fix does not resolve the separate three C obstacle failures.
