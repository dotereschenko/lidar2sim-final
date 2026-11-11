# SDD: retain observed doorway surfaces under sector dropout

## Failure

Pinned A/B v2 acquisition passes reproducibility and annotation visibility checks, but the detailed Poisson output fails D01 on both visual and collision meshes. Five rays within a 5 mm bundle see an approximately 8 cm spread even on raw Poisson, before density/support trimming, floor recovery or decimation. Thus increasing polygon budgets or loosening the evaluator does not address the upstream defect.

## Scope

Investigate normal estimation on the sparse/noisy observations. Prefer an existing reconstruction parameter change if it restores stable surfaces. Use only A and its trajectory in reconstruction. Keep scan hashes, measurement definitions, geometry/physics thresholds and evaluator unchanged. Changes are isolated on fix/sector-scan-doorway-reconstruction.

## Acceptance

Run the common evaluator on v2: both meshes must pass all ten dimensions and existing geometry checks. Check the earlier A/B candidate for regressions. Run the qualified Isaac tests on the exact new USD; record hashes and measured penetration. Report the pending H3 contract and performance qualification independently: passing implementation checks is not organizer approval. Preserve the failed v2 report for comparison. Record both experiments and the final configuration, commit and push, and update Plane and project memory.

## Resulting design

Normal-radius changes alone failed. Depth 10 restored the v2 visual doorway, but the 60k collider lost detail. A 120k collider passed the doorway but failed the tabletop; 200k with a separate 5k floor budget passed all ten v2 dimensions. The original normals are retained. Depth 10 regressed D09 on the earlier acquisition, so this is an additive opt-in `poc-high-detail.yaml`, not a replacement of `poc-detailed.yaml`. The web method `high-detail` pins that config per upload and explains the time/memory tradeoff. Existing baseline/improved defaults remain unchanged.

Verification uses the full v2 evaluator, earlier-dataset regression evidence, existing unit/negative tests, web API upload with trajectory and artifact downloads, and the exact v2 USD in Isaac. Generalization to a hidden scene remains unqualified.
