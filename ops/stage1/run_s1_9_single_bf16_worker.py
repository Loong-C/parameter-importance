"""UUID-isolated single-GPU BF16 worker for the S1.9 formalizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_DETERMINISM_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _require_pre_cuda_policy(*, cuda_visible_devices: str) -> None:
    """Reject a child not started with the frozen deterministic policy."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != cuda_visible_devices or any(os.environ.get(name) != value for name, value in _DETERMINISM_ENV.items()):
        raise SystemExit("S1_9_SINGLE_WORKER_PRE_CUDA_POLICY_INVALID")


def _version_wire(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def _configure_correctness_policy(torch: object, *, cuda_visible_devices: str, local_gpu_uuid: str) -> dict[str, object]:
    """Set torch flags before any CUDA operation and preserve the allowlist."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    nccl = getattr(torch.cuda, "nccl", None)
    summary = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": _version_wire(torch.version.cuda),
        "cudnn_version": _version_wire(torch.backends.cudnn.version()),
        "nccl_version": _version_wire(None if nccl is None else nccl.version()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cuda_visible_devices": cuda_visible_devices,
        "local_rank": 0,
        "local_gpu_uuid": local_gpu_uuid,
    }
    if not (summary["deterministic_algorithms"] and summary["cudnn_deterministic"] and not summary["cudnn_benchmark"]):
        raise SystemExit("S1_9_SINGLE_WORKER_CORRECTNESS_POLICY_INVALID")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--execution-commit", required=True); parser.add_argument("--run-token", required=True); parser.add_argument("--approved-gpu-uuid", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.execution_commit) is None or re.fullmatch(r"[0-9a-f]{64}", args.run_token) is None or _UUID.fullmatch(args.approved_gpu_uuid) is None or os.environ.get("CUDA_VISIBLE_DEVICES") != args.approved_gpu_uuid:
        raise SystemExit("S1_9_SINGLE_WORKER_PLAN_INVALID")
    _require_pre_cuda_policy(cuda_visible_devices=args.approved_gpu_uuid)
    import torch

    environment_summary = _configure_correctness_policy(torch, cuda_visible_devices=args.approved_gpu_uuid, local_gpu_uuid=args.approved_gpu_uuid)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("S1_9_SINGLE_WORKER_UUID_ISOLATION_INVALID")
    if str(args.repository / "src") not in sys.path:
        sys.path.insert(0, str(args.repository / "src"))
    from param_importance_nlp.stage1_precision import run_stage1_s19_bf16_smoke

    observation = run_stage1_s19_bf16_smoke(source_root=args.repository, device="cuda:0", checkpoint_dir=args.checkpoint_dir)
    payload = {"schema_version": "stage1-s1-9-single-bf16-worker-v1", "status": observation["status"], "execution_commit": args.execution_commit, "run_token": args.run_token, "approved_gpu_uuid": args.approved_gpu_uuid, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "environment_summary": environment_summary, "observation": observation}
    payload["artifact_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
    if args.output.exists():
        raise SystemExit("S1_9_SINGLE_WORKER_OUTPUT_EXISTS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
