# H21: private scene C

A distinct asset-free layout was built with the generic `tools/build_box_scene.py` from a private JSON recipe. It is not a transformed copy of A. The recipe, layout, seeds, GT, scan clouds, annotations and detailed test reports are outside the participant checkout in the operator's private jury directory. No private recipe or detailed C reports are committed here.

C uses the same independent raycasting acquisition, count750000 per scan, range sigma0.02/0.01m, A5% random dropout after heading-relative sectors and common-observation annotation rules as A. GT source builds repeat byte-for-byte; both repeated acquisition manifests and file hashes match. Both clouds have750000 finite unique points. All ten dimensions are observed in both scans; two door rectangles are unobstructed and admit the box.

GT passes the v2 independent-B geometry evaluation, all ten dimensions on both meshes, and fresh Isaac physics10/10 floor,10/10 obstacles,5/5 routes. Generic v2 negative controls are separately recorded under `benchmarks/release-v2/evaluator-regression.json`; the C GT run is positive qualification and is not represented as a new negative-control suite.

A visibility-only revision of the acquisition path added missing observation of an obstacle side before reconstruction evaluation. The final dataset is frozen. Subsequent evaluation found baseline dimensions9/10 visual and8/10 collision, high-detail9/10 each. C was not changed to make the candidate pass. This is an unresolved candidate generalization failure for calibration/jury, not a defect concealed by relaxing the data or metric.

The public [redacted audit](hidden-scene-v2.json) records hashes and aggregate results. Full private freeze and recipes are stored locally under the operator's `.local/share/lidar2sim-jury/scene-c-v2` and on Spark under `/home/unknown641/lidar2sim-jury-private/scene-c-v2`, behind owner-only directory permissions. The jury package will include these privately; the participant archive will not.
