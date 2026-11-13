#!/bin/bash
# Run inside the isolated project directory, after installing CPU dependencies.
set -euo pipefail
uv="$PWD/uv-x86_64-unknown-linux-gnu/uv"
export PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES= TORCH_HOME="$PWD/weights"
"$uv" pip install --python .venv/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
"$uv" pip install --python .venv/bin/python ninja GitPython python-pycg pykdtree omegaconf setuptools wheel pip
# Published CUDA wheels require glibc >=2.32; this host has glibc 2.31.
FORCE_CUDA=1 python -m pip install --force-reinstall --no-build-isolation --no-deps --no-binary=torch-scatter torch-scatter==2.1.2
test "$(git -C nksr-source rev-parse HEAD)" = e40336845e67761343a756788e5a98b827d4a143
cd nksr-source
if ! test -f .lidar2sim-patched; then
  sed -i '1i#include <thrust/tuple.h>' package/csrc/pcproc/cuda_kdtree.cu
  sed -i 's/ lerp(/ nksr_lerp(/g' package/csrc/common/math_util.h
  python ../deployment/patch_nksr_error_checks.py
  touch .lidar2sim-patched
fi
python -m pip install --no-build-isolation --no-deps ./package
cd ..
"$uv" pip freeze --python .venv/bin/python > installed-vkteacher.txt
