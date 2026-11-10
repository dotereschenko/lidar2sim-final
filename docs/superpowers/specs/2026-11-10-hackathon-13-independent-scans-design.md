# HACKATHON-13: reproducible independent A/B scans

Generate a new candidate dataset from pinned Office GT, without changing original scene_a or the reconstruction pipeline. Output A and B have separate pose files and random seeds. B uses intermediate positions along the supplied path, at a sensor height 0.10 m above A; it is independently raycast, never transformed from A points. Validate finite monotonic trajectories, pose clearance and line-of-sight between successive poses before writing outputs.

Use meters, Z-up, first-hit rays, 0.05–20 m range, full azimuth and ±45° elevation sampled uniformly in sine. Generate exactly 750000 accepted points per scan by continuing acquisition after dropped returns. A has Gaussian range noise sigma 0.02 m and 5% random dropout; B sigma 0.01 m and zero dropout. Sector dropout, drift and ghost fractions are explicitly zero in this initial profile. These are synthetic sensor measurements, not a simulation of a complete SLAM system.

Store seed, acquisition counts, per-pose counts, sensor profile, versions, source/code/output hashes, trajectory derivation and candidate status in manifest. Separately report measured signed range-noise sigma, cloud-to-GT triangle median/p95, and area-weighted GT sample coverage within 0.10 m of each cloud (30000 samples, fixed seed). Coverage is diagnostic, not the common observable evaluation region required by H18.

The 750000-point A is provisional until H3 resolves the original 248702-point input versus 0.5–1M requirement. H13 stays In Progress pending that decision even if technical checks pass. B remains evaluator-only. Do not introduce B into reconstruction commands or claim full PoC acceptance.

Validation: small box tests for coordinates, noise, dropout, independent poses, deterministic hashes, unsafe trajectories and overwrite rejection; two complete Office builds with equal hashes; audit finite/unique points and copy dataset to server. No new dependency or Isaac session is needed.

Office preflight found source pose t=63.2 s only 0.028 m from GT, with segment 78→79 crossing geometry. The tracked candidate trajectory changes this pose x from -18.942 to -18.742 m; both derived trajectories pass the same clearance and segment checks. Pass it explicitly with --trajectory-a configs/scans/trajectory_A_candidate.csv. This does not modify source A or its original trajectory.
