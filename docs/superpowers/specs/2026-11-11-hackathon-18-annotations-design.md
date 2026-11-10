# HACKATHON-18: versioned evaluation annotations

Create a candidate annotation package pinned to GT, candidate A/B and the existing physical test configuration. Original scene_a and reconstruction inputs remain unchanged. The new files are inputs for H19/H20, not a claim that those evaluators are implemented.

The common observable region is the intersection of points having at least 10 A neighbors and 10 B neighbors within 0.10 m. Save 300000 uniform-area GT samples (seed 18), counts and a common mask in eval_region.npz. Preserve this same predicate for later queries on visual/collision; never define an independent mask for each solution. Reject empty/undersampled intersections (minimum 1000 samples). Report total-area fraction and hash of both input clouds. Noise/coverage thresholds are candidate data definitions, not organizer approval.

Ten dimensions cover three open-door widths and heights, south-room depth/height, tabletop depth and pillar depth. Each has label, category, probe origin, unit axis, two GT surface hits, reference length in meters and local search boxes for H20. Verify finite hits, opposing directions and surface normal alignment. A box specifies where to search the reconstructed surface; its GT coordinates are not the future measurement result.

Three open door rectangles include width/height, plane normal, vertices and a collision-clear interior grid through wall thickness. Check a 0.4×0.4×0.3 m body's fit. Existing 10 floor / 10 obstacle / 5 route positions retain the verified H12 physics evidence, but only after checking GT USD and configuration hashes match exactly.

Produce a visual overlay of GT/A/B, common samples, dimensions and doors. Save parameters, code/input/output hashes and versions. Two builds must be identical. Mark package provisional while H13/H3 count approval remains open. No new dependencies; existing Open3D, NumPy, SciPy and standard library suffice.

GT inspection: frames at (-18,22.5) and (-18,16.5) are 0.94 m wide but closed by leaves (425/425 blocked rays). Exclude them as traversable openings. Annotate southwest, northeast and southeast open doors instead; retain the excluded-frame explanation in the definition file.

Visual QA rejected dimensions behind closed doors and the hidden side of the pillar. Both endpoints of every retained dimension must satisfy the exact common A/B predicate; save neighbor counts and fail the build otherwise. The package does not require a reconstruction to infer inaccessible cabinet/corridor interiors.
