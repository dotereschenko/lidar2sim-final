# Hybrid observed surfaces

Experimental follow-up to the visibility-normal diagnostics. The remaining C
collision defect has the far partition surface but lacks its near surface.
Poisson alone merges/suppresses observed thin surfaces; correct normal direction
is insufficient. Reuse the existing experimental local-plane implementation from
`feat/observed-planar-surfaces` and keep visibility correction disabled.

Build strictly vertical local patches from observed A points using seeded RANSAC,
local Delaunay and a maximum edge length; combine with the supported Poisson mesh
before the existing support filter and floor recovery. No B/GT/scene-specific
coordinates are reconstruction inputs. Door-sized gaps must remain open.

The previous planar candidate lost a doorway dimension by 5.71 cm. Bound the
observed footprint extension by the full plane-fitting uncertainty (2 cm), instead
of half of it. This is a local sampling-boundary uncertainty allowance, not a
changed dimension tolerance. Disable density quantile removal to retain sparse
observed surfaces. These hypotheses require unchanged A/C dimensions and physics;
failed candidates are retained, never substituted for qualification.

Generic tests: two close opposing surfaces remain distinct; small acquisition hole
can close while a doorway stays open; translation invariance; no invented walls
from sparse/horizontal support; padding stays within declared uncertainty; invalid
configuration rejected. Release gates: A/C geometry and all dimensions, fresh
25/25 physics on both, target 10M runtime, five-repeat qualification. Default
`plane_distance=0` preserves existing release behavior until these gates pass.

Measured revision: full boundary padding did not fix A D03 (still 5.71 cm).
Its measured gap is too narrow, caused by a fitted patch offset, not insufficient
boundary coverage. Revert padding to half the fitting distance. Fit vertical
planes directly as seeded XY line RANSAC, then refine in XY. Do not fit a tilted
3D plane and force it vertical afterward: that changes intersections beyond the
initial residual bound. The generic two-surface, doorway and translation tests
exercise the new fit. Candidate v1 C export failed from local disk exhaustion;
it is an infrastructure failure, not a geometry/physics result.

Next hypothesis (unit tested, full-scene run pending): retain all observed returns
within a fitted plane's distance, even when their local PCA normals are unreliable.
Normals select fitting seeds, but must not discard the lower part of a thin wall
once its plane is established. A generic test verifies low-wall recovery with
corrupted local normals. Heavy runs paused on user report of disk exhaustion;
subsequent full-scene runs should execute on Spark with private evidence storage.

## Suppress repeated extraction of noise tails

Spark d307ce3 reconstructed a single wall as several parallel patches. The
remaining C span error is 6.16 cm, caused by the inward copies of two boundaries.
A generic single noisy wall reproduces duplicate extraction. After fitting,
consume candidate seeds within three fitting distances of that surface; continue
using the original distance for actual patch support and the original edge bound.
The acceptance tolerance remains unchanged. Existing nearby-opposing-surface
and doorway tests must still pass. Limitation: distinct surfaces closer than this
consumed neighbourhood may not both receive patches; the Poisson input is intact.

## Avoid overlapping collision surfaces

349a43f passes A/C geometry and all dimensions, but fresh A physics regresses
(floor8/10, obstacles2/10, routes4/5). Treat overlapping planar and Poisson
surfaces as a candidate cause, not a proven explanation. Add patches only where
all triangle vertices, edge midpoints and centre are farther than three fitting
distances from the already supported mesh. Both existing mesh and patches retain
the original A-support filter; floor recovery remains unchanged. A generic test
keeps the missing opposite side and removes the duplicate existing side. Fresh
physics is mandatory; successful geometry alone must not promote the candidate.

Fresh C physics of 349a43f is24/25: the remaining approach hits a reconstructed
near face whose winding points away from the approaching body. Ray diagnostics
confirm the surface is present, unlike the old missing-surface defect. Orient
only the fitted patches by voting over up to512 deterministic support samples:
nearest-pose orientation followed by the existing full-ray occlusion check against
all A returns. This leaves Poisson normals unchanged. A generic two-sided wall
test starts with both signs wrong on one side and requires opposite outward
orientations. No test positions or ground truth enter this correction.

Ray inspection of 764e928 shows that the surviving near face is part of the
original Poisson surface and is still reversed; the correctly oriented patch is
removed as overlapping. Align existing triangle winding to nearby observed
patches before removing overlap. Require all three vertices and centre to lie
within the fitting distance, and all patch normals to oppose the original normal
by cosine >0.9. Only use this with a trajectory. Positions/topology and thresholds
are unchanged. Unit test verifies exact unchanged vertices and triangle sets,
corrected winding, and no change to a distinct farther surface. This remains an
experimental physical correction until fresh Isaac results are available.
