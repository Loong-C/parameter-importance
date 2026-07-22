from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

from param_importance_nlp.runtime import TorchDistributedReducer


_GLOO_UNAVAILABLE_REASON: str | None = None


def _gloo_worker(
    rank: int,
    world_size: int,
    init_uri: str,
    result_path: str,
) -> None:
    """spawn 子进程入口必须位于模块顶层，Windows 才能 pickle。"""

    dist.init_process_group(
        backend="gloo",
        init_method=init_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        reducer = TorchDistributedReducer()
        # 固定 global batch 为八个 sample。每个 rank 只看互不重叠的局部切片，
        # 归约充分统计量后再除全局计数，结果应与串行平均完全一致。
        samples = torch.arange(1, 9, dtype=torch.float64)
        local = samples[rank::world_size]
        reduced = reducer.sum_tensors({"sum": local.sum().reshape(1)})["sum"]
        count = reducer.sum_int(int(local.numel()))
        if rank == 0:
            Path(result_path).write_text(
                json.dumps(
                    {
                        "backend": reducer.capabilities.backend,
                        "world_size": reducer.capabilities.world_size,
                        "count": count,
                        "mean": float((reduced / count).item()),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gloo_reducer_matches_same_global_batch_semantics(
    tmp_path: Path, world_size: int
) -> None:
    global _GLOO_UNAVAILABLE_REASON

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("local torch build has no Gloo backend")
    if _GLOO_UNAVAILABLE_REASON is not None:
        pytest.skip(_GLOO_UNAVAILABLE_REASON)
    rendezvous = tmp_path / f"gloo-{world_size}.rendezvous"
    result = tmp_path / f"gloo-{world_size}.json"
    try:
        mp.spawn(
            _gloo_worker,
            args=(world_size, rendezvous.as_uri(), str(result)),
            nprocs=world_size,
            join=True,
        )
    except ProcessRaisedException as exc:
        # 某些 Windows CPU wheel 暴露 ``is_gloo_available=True``，但没有编译
        # 可用的 Gloo device。此时属于本机能力阻塞，不能降级成 LocalReducer
        # 再把 2/4-rank 场景伪装为通过。
        if "unsupported gloo device" not in str(exc):
            raise
        _GLOO_UNAVAILABLE_REASON = (
            "BLOCKED: installed Windows torch wheel reports Gloo but has no supported device"
        )
        pytest.skip(_GLOO_UNAVAILABLE_REASON)
    observed = json.loads(result.read_text(encoding="utf-8"))
    assert observed == {
        "backend": "gloo",
        "world_size": world_size,
        "count": 8,
        "mean": 4.5,
    }
