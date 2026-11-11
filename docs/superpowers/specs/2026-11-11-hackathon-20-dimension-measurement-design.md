# H20: measure reconstructed dimensions

Add optional --dimensions to explicit common-region B/GT evaluation. The ten H18 definitions supply probe origins, unit axes, search boxes and reference lengths; measurement coordinates must come exclusively from ray intersections with each reconstructed mesh.

For each side, cast a central ray and four transverse probes offset 5 mm. Require the central hit and at least four of five finite, axis-aligned hits inside the specified search box. Reject surface variation above 2 cm along the measurement direction as ambiguous. Compare the central reconstructed span to the reference with the profile's inclusive 5 cm tolerance, before rounding. Missing/unstable surfaces yield an error result, never a passing value. Persist actual central surface points, spread, valid probe counts, reference/measured/error in meters, and method parameters. Search boxes delimit correspondence; they do not supply the measured endpoints.

Validate all ten definitions before evaluation. Both visual/collision have individual dimensions results and aggregate status; failed or erroneous dimensions make PoC failed. Passing dimensions do not imply physics or performance acceptance. Legacy diagnostic and H19 invocations without --dimensions remain compatible.

Tests: real box GT, 10 cm width/height distortion, absent surface, stepped/ambiguous local surface, invalid annotation, CLI integration. Run all ten actual GT dimensions and baseline/filtered, pin hashes and publish reports. No new dependencies or Isaac run.
