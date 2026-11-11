# H3/H13 — release v2 adopted and checked

On 2026-11-11 the user delegated all Denis tasks. The implementation adopts the original document's thresholds and records the working decisions in `configs/acceptance/release-decision-v2.json`: 750000 A/B points, box surrogate, mentor-operated Spark, mandatory hidden C. This does not claim consent from other organizers. The decision pins exact profile bytes; a modified profile cannot inherit its adoption status.

v2 precision samples mesh area and queries independent B (95% within10cm); completeness samples actual eligible B returns and queries mesh triangles (median3cm/p9510cm). Both use the fixed common-observation rule. Legacy v1 keeps GT precision and area-anchor-weighted completeness for reproduction only. v2 refuses diagnostic/GT substitution. The report merger rejects a v2 report that labels GT as the precision/completeness reference.

| Item | v1 historical | v2 release |
|---|---|---|
| Precision reference | GT triangles | Independent B returns |
| Precision threshold | 92% within5cm | 95% within10cm |
| Completeness samples | GT-area anchors mapped to B | Uniform eligible B returns |
| Floor | 9/10 | 10/10 |
| Obstacles | 10/10, penetration3.5cm | 10/10, penetration2cm |
| Routes | 4/5 | 5/5 |
| A count | Original248702 / candidate750000 | Frozen750000 |

All distances/axes/CLI/output contracts are defined in [rules](../rules.md). Hidden C, five repeats, packages and jury rehearsal are release requirements tracked separately; a single-scene report does not prove them.

The frozen A/B acquisition has independent B waypoints, heading-relative sector dropout A, measured sigma .0200023/.0100086m and conditional random loss4.9586%, repeatable bytes, and all10 annotated dimensions visible. `configs/releases/scene-a-v2.json` adopts exact immutable scan/annotation hashes. Their embedded candidate status remains historical; it is not rewritten to manufacture a new acquisition.

On the qualified high-detail A reconstruction under v2: visual precision99.999638%, collision100%; completeness p95 approximately2.483cm for both, dimensions10/10 each. Fresh Isaac run with the v2 profile/config: floor10/10, obstacles10/10, routes5/5; maximum penetration0.0006m. GT and negative evaluator controls pass, including shifted geometry, missing wall, wrong dimensions and order independence. Evidence is in `benchmarks/release-v2/`.

Scene-A aggregate remains incomplete because designated-machine performance is not attached. H3 defines and implements the contract; it does not claim all downstream release work finished. H13 acquisition is complete under the adopted750000-point decision.
