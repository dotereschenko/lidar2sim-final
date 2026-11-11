# Visibility-aware normal orientation (experimental)

Status: hypothesis under test; release configuration unchanged.

Nearest trajectory pose can lie behind an observed thin partition. Orienting a
surface normal toward that pose can reverse one side of the partition and corrupt
Poisson reconstruction. Input is only the submitted A cloud and trajectory.

Add an opt-in correction after nearest-pose orientation. In bounded batches find
the nearest pose in each normal hemisphere (among 64 closest poses). Probe the
first half metre of both sight lines, excluding the target surface. Flip only
when the currently selected side crosses a supported observed sheet and the
opposite side is clear. Require consecutive occupied probes and multiple nearby
points so a lone outlier cannot trigger a correction. Unsupported, grazing and
one-sided cases preserve existing normals. No scene IDs, evaluation coordinates,
B, ground truth, annotations or acceptance thresholds enter reconstruction.

Acceptance: generic two-sheet test corrects the occluded direction; clear rays,
one-sided trajectory and isolated outliers preserve existing orientation; flag
is strictly boolean and defaults off. Record corrected normal count in report.
Then measure unchanged A/C geometry, dimensions and fresh physics. A regression
or C obstacle failure keeps the method experimental. Release qualification still
requires target-machine runtime and repeated real reconstruction/physics runs.
