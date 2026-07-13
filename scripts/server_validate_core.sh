#!/usr/bin/env bash
set -euo pipefail
cd /home/sophgo13/cjl
DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
REPO=/home/sophgo13/cjl/parameter-importance
PY="$DATA_ROOT/envs/parameter-importance/bin/python"

[[ -x $PY ]] || { echo "ERROR: offline venv is not ready" >&2; exit 2; }
mkdir -p "$DATA_ROOT/manifests" "$DATA_ROOT/cache/huggingface" "$DATA_ROOT/cache/torch" "$DATA_ROOT/tmp"
export DATA_ROOT
export HF_HOME="$DATA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export TMPDIR="$DATA_ROOT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_PROXY='*' no_proxy='*'
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy HF_ENDPOINT HF_TOKEN

"$PY" -m pip check | tee "$DATA_ROOT/manifests/core-pip-check.txt"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$REPO/scripts/nccl_smoke.py" | tee "$DATA_ROOT/manifests/nccl-smoke.txt"
CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/scripts/verify_offline_assets.py" | tee "$DATA_ROOT/manifests/offline-assets-output.txt"

{
  echo "date=$(date -Is)"
  echo "python=$($PY --version 2>&1)"
  echo "kernel=$(uname -srmo)"
  "$PY" - <<'PY'
import datasets
import torch
import transformers
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"cudnn={torch.backends.cudnn.version()}")
print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$DATA_ROOT/manifests/environment-core.txt"

date -Is > "$DATA_ROOT/manifests/CORE_READY"
echo "CORE_READY: offline venv, 4-GPU NCCL, models, and datasets passed"
