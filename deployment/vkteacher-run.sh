#!/bin/sh
# User service; no root, Docker, or machine configuration changes.
set -eu
cd "${1:?Usage: vkteacher-run.sh PROJECT_DIRECTORY GPU_UUID}"
gpu=${2:?Pass the UUID of an available GPU}
test -x "$PWD/.venv/bin/python"
mkdir -p "$PWD/jobs" "$PWD/weights"
chmod 700 "$PWD/jobs"
exec systemd-run --user --unit=lidar2sim-ui \
  --property="WorkingDirectory=$PWD" \
  --property=Nice=10 --property=CPUAffinity=0-3 --property=TasksMax=256 \
  --property=Restart=on-failure --property=RestartSec=10 \
  --setenv="CUDA_VISIBLE_DEVICES=$gpu" --setenv="TORCH_HOME=$PWD/weights" \
  --setenv=OMP_NUM_THREADS=4 --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 --setenv=LIDAR2SIM_JOB_TIMEOUT_S=600 \
  --setenv="LIDAR2SIM_NEWTON_COMMAND=${LIDAR2SIM_NEWTON_COMMAND:-}" \
  --setenv=LIDAR2SIM_MIN_FREE_BYTES=10737418240 \
  "$PWD/.venv/bin/python" -m lidar2sim.web.server --port 8768 --jobs "$PWD/jobs"
