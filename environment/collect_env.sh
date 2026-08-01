#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/environment/generated"
mkdir -p "${OUT}"

{
  date --iso-8601=seconds
  cat /etc/os-release
  python --version
  gcc --version | head -n 1
  g++ --version | head -n 1
} > "${OUT}/system_info.txt" 2>&1

nvidia-smi > "${OUT}/nvidia_smi.txt"
nvcc --version > "${OUT}/nvcc_version.txt"
python -m pip freeze > "${OUT}/pip_freeze.txt"
conda list > "${OUT}/conda_list.txt"
python -m jittor.test.test_cuda 2>&1 | tee "${OUT}/jittor_cuda_test.log"

echo "Environment evidence written to ${OUT}"
