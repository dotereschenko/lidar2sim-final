# SDD: finish active PoC tasks

## Scope and existing evidence

Integrate Denis's qualified H12/H18/H19/H20/H22 evaluator branch with PR #1 at f17d967 (Yasin reconstruction, collision, orchestration and benchmark). Preserve both commit histories and existing configurations. Do not reimplement delivered components. Tasks H3/H8/H13/H16/H17/H23/H29 stay open until their concrete acceptance criteria are met; a successful merge is not acceptance.

## Work packages

1. H23 integration: merge PR #1, preserve legacy floor/filter options and qualified PhysX settings; consume common-region geometry plus dimensions. Pin profile, input, output and evaluator hashes; reject mismatched or stale physics. Repeat evaluation without reconstruction. All missing mandatory evidence remains not_run/incomplete. CLI failures must persist in report and return nonzero.
2. H8 execution environment: stage the qualified evaluator plus its Python/profile dependencies. Serialize Isaac jobs with a host-wide lock and refuse conflicting sessions; never silently stop shared workloads. Preserve existing image. Document operator handoff, isolation and artifact retrieval. Live GPU runs need a free slot; running services are not evidence of permission to disrupt other tasks.
3. H16 reconstruction: rebuild the PR candidate on the selected input; run qualified geometry, all dimensions and physics. Record failures and improve collision from input-only geometry. Never substitute GT into reconstruction or tune acceptance thresholds to hide failure.
4. H17 performance: use pinned H14 10M SHA, measure full child process including export and peak RSS on the selected scoring machine. Existing M5 figures are prior evidence, not a claim about Spark/x86. Record repeatability and quality separately.
5. H29 demo: record the actual final USD hash, bind baseline/final reports to identical data/profile/evaluator, show input, both meshes and an Isaac sequence. Historical filtered video is not final candidate evidence.
6. H3/H13: prepare a concrete decision record from existing profile and candidate A=750000. User selection of scoring input/machine is pending; do not invent organizer approval. Threshold and hidden-scene decisions must be explicit before final signoff.

## Integration invariants

No arbitrary YAML counts as an approved threshold profile. Geometry records must include both meshes and measured dimensions to count as full geometry. Attaching a report requires matching mesh/USD hashes and the same profile. Previously attached evidence must be invalidated when relevant inputs change. Reconstruction A timing cannot satisfy a 10M performance requirement. Missing physics, performance, or organizer signoff cannot yield full PoC passed.

## Validation

Run the union of unittest and pytest suites, not just one test style. Test qualified-record adapters, missing/mismatched profile and mesh hashes, failed dimensions, stale physics reuse, unavailable runner, stage failure/timeout and --reuse. Rebuild a clean candidate and evaluate it using the exact branch version before any physics or demo. Publish immutable evidence and remaining gaps for each Plane task.
