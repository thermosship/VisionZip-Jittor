#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export PIP_CACHE_DIR=/root/autodl-tmp/cache/pip

cd /root/autodl-tmp/VisionZip-Jittor
printf 'Activated: %s\n' "${CONDA_PREFIX}"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
