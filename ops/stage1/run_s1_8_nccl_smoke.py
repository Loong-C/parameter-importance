"""Minimal S1.8 lease-post-NCCL collective smoke; executed only by formalizer."""

from __future__ import annotations

import os
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--report", required=True); args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 4 or local_rank != rank:
        raise RuntimeError("S18_NCCL_SMOKE_TORCHRUN_IDENTITY_INVALID")
    torch.cuda.set_device(local_rank)
    allocation = torch.empty((256,), dtype=torch.float32, device=torch.device("cuda", local_rank))
    allocation.fill_(float(rank))
    dist.init_process_group("nccl")
    try:
        value = torch.tensor([rank + 1], dtype=torch.int64, device=torch.device("cuda", local_rank))
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if int(value.item()) != 10:
            raise RuntimeError("S18_NCCL_SMOKE_ALL_REDUCE_INVALID")
        gathered: list[object] = [None] * world
        dist.all_gather_object(gathered, {"rank": rank, "uuid": os.environ["CUDA_VISIBLE_DEVICES"].split(",")[rank], "input": rank + 1, "output": int(value.item()), "cuda_initialized": bool(torch.cuda.is_initialized()), "allocation_bytes": int(allocation.numel() * allocation.element_size())})
        if rank == 0:
            body = {"schema_version": "stage1-s1-8-nccl-smoke-v1", "status": "PASS", "backend": "nccl", "world_size": world, "allocation_probe": "torch.empty.float32.256", "rank_records": gathered}
            body["artifact_hash"] = canonical_json_hash(body)
            write_canonical_json(Path(args.report), body)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
