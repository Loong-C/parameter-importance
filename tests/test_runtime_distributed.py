from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException

from param_importance_nlp.providers import (
    DeterministicBatchCursor,
    TorchModelAdapter,
    TrainingMicrobatch,
)
from param_importance_nlp.runtime import TorchDistributedReducer
from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec


_GLOO_UNAVAILABLE_REASON: str | None = None


class _TokenModel(torch.nn.Module):
    """只有 dense 参数的极小 causal-LM，用于比较完整训练事务。"""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4, dtype=torch.float64)
        self.output = torch.nn.Linear(4, 7, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.arange(28, dtype=torch.float64).reshape(7, 4) / 37.0
            )
            self.output.weight.copy_(
                torch.arange(28, dtype=torch.float64).reshape(7, 4).flip(0) / 41.0
            )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del attention_mask
        return self.output(self.embedding(input_ids))


class _CountingNoSyncModule(torch.nn.Module):
    """记录 TrainingEngine 是否把每个 microbatch 放进 DDP.no_sync。"""

    def __init__(self, ddp: torch.nn.Module) -> None:
        super().__init__()
        self.ddp = ddp
        self.no_sync_calls = 0

    def forward(self, **payload: torch.Tensor) -> torch.Tensor:
        return self.ddp(**payload)

    def no_sync(self):
        self.no_sync_calls += 1
        return self.ddp.no_sync()


def _global_token_microbatches() -> tuple[TrainingMicrobatch, ...]:
    """八个统计单元具有不同有效 token 数，暴露错误的等权 rank 平均。"""

    batches: list[TrainingMicrobatch] = []
    for index in range(8):
        length = 2 + index % 4
        tokens = torch.tensor(
            [[(index + offset) % 7 for offset in range(5)]], dtype=torch.int64
        )
        mask = torch.tensor(
            [[1 if offset < length else 0 for offset in range(5)]], dtype=torch.int64
        )
        batches.append(
            TrainingMicrobatch(
                f"token-{index}",
                {"input_ids": tokens, "labels": tokens, "attention_mask": mask},
                (f"sample-{index}",),
            )
        )
    return tuple(batches)


def _flatten_parameters(module: torch.nn.Module) -> list[float]:
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    ).tolist()


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


def _gloo_training_worker(
    rank: int,
    world_size: int,
    init_uri: str,
    result_path: str,
) -> None:
    """执行一个带 token 加权、梯度累计和在线 U 的真实 DDP step。"""

    dist.init_process_group(
        backend="gloo",
        init_method=init_uri,
        rank=rank,
        world_size=world_size,
    )
    try:
        ddp = torch.nn.parallel.DistributedDataParallel(_TokenModel())
        counted = _CountingNoSyncModule(ddp)
        optimizer = torch.optim.SGD(counted.parameters(), lr=0.05)
        local_micros = _global_token_microbatches()[rank::world_size]
        result = TrainingEngine(
            spec=TrainingRunSpec(
                "distributed-token-fixture",
                "local_fixture",
                max_steps=1,
                max_attempts=1,
                importance_enabled=True,
                estimator_name="u",
                accumulation_dtype="float64",
                weights_exogenous=True,
                common_mean_assumption=True,
            ),
            model=TorchModelAdapter(counted, task_type="causal_lm"),
            optimizer=optimizer,
            cursor=DeterministicBatchCursor((local_micros,)),
            reducer=TorchDistributedReducer(),
            rank=rank,
        ).run()
        assert result.importance_snapshot is not None
        if rank == 0:
            Path(result_path).write_text(
                json.dumps(
                    {
                        "parameters": _flatten_parameters(ddp.module),
                        "mean_loss": result.records[0].mean_loss,
                        "effective_count": result.records[0].effective_count,
                        "no_sync_calls": counted.no_sync_calls,
                        "importance_summaries": dict(
                            result.importance_snapshot.scalar_summaries
                        ),
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _serial_token_reference() -> dict[str, object]:
    model = _TokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    result = TrainingEngine(
        spec=TrainingRunSpec(
            "distributed-token-fixture",
            "local_fixture",
            max_steps=1,
            max_attempts=1,
            importance_enabled=True,
            estimator_name="u",
            accumulation_dtype="float64",
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=TorchModelAdapter(model, task_type="causal_lm"),
        optimizer=optimizer,
        cursor=DeterministicBatchCursor((_global_token_microbatches(),)),
    ).run()
    assert result.importance_snapshot is not None
    return {
        "parameters": _flatten_parameters(model),
        "mean_loss": result.records[0].mean_loss,
        "effective_count": result.records[0].effective_count,
        "importance_summaries": dict(result.importance_snapshot.scalar_summaries),
    }


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


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_gloo_training_engine_matches_serial_token_weighted_accumulation(
    tmp_path: Path, world_size: int
) -> None:
    """相同 global batch 不得因 rank 划分、累计次数或 DDP hook 改变结果。"""

    global _GLOO_UNAVAILABLE_REASON

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("BLOCKED: local torch build has no Gloo backend")
    if _GLOO_UNAVAILABLE_REASON is not None:
        pytest.skip(_GLOO_UNAVAILABLE_REASON)
    rendezvous = tmp_path / f"engine-gloo-{world_size}.rendezvous"
    result_path = tmp_path / f"engine-gloo-{world_size}.json"
    try:
        mp.spawn(
            _gloo_training_worker,
            args=(world_size, rendezvous.as_uri(), str(result_path)),
            nprocs=world_size,
            join=True,
        )
    except ProcessRaisedException as exc:
        if "unsupported gloo device" not in str(exc):
            raise
        _GLOO_UNAVAILABLE_REASON = (
            "BLOCKED: installed Windows torch wheel reports Gloo but has no supported device"
        )
        pytest.skip(_GLOO_UNAVAILABLE_REASON)

    observed = json.loads(result_path.read_text(encoding="utf-8"))
    expected = _serial_token_reference()
    assert observed["effective_count"] == expected["effective_count"] == 20
    assert observed["no_sync_calls"] == 8 // world_size
    assert observed["mean_loss"] == pytest.approx(expected["mean_loss"], abs=1e-12)
    assert observed["parameters"] == pytest.approx(expected["parameters"], abs=1e-12)
    for name, expected_value in expected["importance_summaries"].items():
        assert observed["importance_summaries"][name] == pytest.approx(
            expected_value, abs=1e-12
        )
