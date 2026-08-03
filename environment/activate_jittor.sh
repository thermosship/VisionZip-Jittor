#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${VISIONZIP_JITTOR_ENV:-/root/autodl-tmp/envs/visionzip-jittor}"
CACHE_ROOT="${VISIONZIP_CACHE_ROOT:-/root/autodl-tmp/cache}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_PREFIX}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"

cd "${PROJECT_ROOT}"
printf 'Activated: %s\n' "${CONDA_PREFIX}"
printf 'Project root: %s\n' "${PROJECT_ROOT}"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
