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
