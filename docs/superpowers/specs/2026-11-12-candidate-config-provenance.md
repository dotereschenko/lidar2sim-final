# Candidate configuration provenance

Remote final-prep adds `calibrate_release --final-config`, but the jury protocol
still pins the old high-detail config. Also, placing a config path in the execution
environment makes it differ from the 10M execution record even with identical
image/resource limits. This blocks valid candidate results.

Keep environment.json restricted to image, CPU, memory and network settings.
Write the selected final config path and SHA256 to configuration.json. The protocol
requires the same pinned hash across all batches, uses it for final acceptance,
and rejects mixed or partially missing configuration metadata. Historical batches
without this metadata continue to use the old high-detail config hash. Performance
accepts the same --final-config file and mounts it read-only instead of selecting
a different embedded config. Do not alter geometry thresholds or input datasets.

Server tests exercise selected config propagation, environment compatibility,
mixed-config rejection, legacy batches, and existing jury tamper rejection.
