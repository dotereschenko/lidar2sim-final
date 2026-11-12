# H22: executable evaluator regression

Provide one short command generating a deterministic box reference, independent dense A/B surface returns, a pinned common-region package and ten span annotations. Evaluate real Open3D triangles through the production common-region evaluator, in both B and GT modes. Fixtures: exact GT; translation +0.10 m in X; removal of the X=2 wall; expansion of X dimension by 0.10 m.

Require exact GT geometry and ten dimensions to pass for both meshes; GT-mode distances <=2e-6 m, B-mode p95 <=0.01 m. Translation must fail precision (<92% within 5 cm). Removed wall must fail completeness p95 (>10 cm) and an X dimension. Expansion must fail an X dimension with error 0.10±2e-6 m. Compare complete results for each solution alone, forward batch and reversed batch; numerical tolerance 1e-12. Save positive/negative JSON, fixture/input/code hashes, thresholds and all checks. CLI exits nonzero on a failed expectation.

Use 12000 uniform-area mesh samples with unchanged acceptance thresholds for a fast fixture-only regression. This is evaluator calibration, not replacement Office acceptance or H3 signoff. No network or Isaac needed. Preserve real Office validation separately.

If negative controls reveal masking of geometric errors, fix the shared evaluator and add a focused regression before publishing. Do not relax thresholds or redesign negative fixtures to make broken evaluation pass.

Regression exposed self-exclusion of shifted mesh surfaces under direct observation filtering. Precision membership now queries the identical A/B predicate at the closest GT surface point; distance still uses the original prediction. This reference-defined region prevents geometric displacement from hiding errors. Reports explicitly version the membership method; previous H19 precision values are historical, not interchangeable.
