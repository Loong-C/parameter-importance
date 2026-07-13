#!/usr/bin/env bash
set -euo pipefail
cd /home/sophgo13/cjl
DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
REPO=/home/sophgo13/cjl/parameter-importance
PY="$DATA_ROOT/envs/parameter-importance/bin/python"
PILE="$DATA_ROOT/datasets/pile-deduped-pythia-preshuffled"
BIN="$PILE/document-00000-of-00020.bin"
IDX="$PILE/document.idx"

[[ -x $PY ]] || { echo "ERROR: venv is not ready" >&2; exit 2; }
[[ -f $BIN && -f $IDX ]] || { echo "ERROR: Pile prefix is not ready" >&2; exit 2; }
source_dir=$(find "$DATA_ROOT/source" -maxdepth 1 -type d -name 'pythia-*' | sort | tail -1)
[[ -n $source_dir ]] || { echo "ERROR: official Pythia source is not extracted" >&2; exit 2; }

export DATA_ROOT
export HF_HOME="$DATA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export TMPDIR="$DATA_ROOT/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NO_PROXY='*' no_proxy='*'
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy HF_ENDPOINT HF_TOKEN

"$PY" "$REPO/scripts/verify_pile_prefix.py" \
  --idx "$IDX" --bin "$BIN" --samples $((512 * 1024)) --tokens-per-sample 2049 \
  --output "$DATA_ROOT/manifests/prefix_coverage.json"
"$PY" "$REPO/scripts/compare_batch_viewer.py" \
  --idx "$IDX" --bin "$BIN" --pythia-source "$source_dir" \
  --steps 0 1 511 --output "$DATA_ROOT/manifests/batch-viewer-comparison.json"

"$PY" -m pip check
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$REPO/scripts/nccl_smoke.py" | tee "$DATA_ROOT/manifests/nccl-smoke.txt"
CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO/scripts/verify_offline_assets.py"

{
  echo "date=$(date -Is)"
  echo "python=$($PY --version 2>&1)"
  echo "kernel=$(uname -srmo)"
  echo "source_dir=$source_dir"
  "$PY" - <<'PY'
import torch, transformers, datasets
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"transformers={transformers.__version__}")
print(f"datasets={datasets.__version__}")
print(f"cudnn={torch.backends.cudnn.version()}")
PY
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
} > "$DATA_ROOT/manifests/environment.txt"

date -Is > "$DATA_ROOT/manifests/READY"
echo "READY: minimum-loop environment passed all offline checks"
