"""可恢复训练循环与在线参数重要性事务。

该模块把 Stage 1 冻结的数学核心接到真实 optimizer step：每个 microbatch 的
mean gradient 先独立取得，再按有效统计单位合并为 global mean；同一批充分量
同时产生 raw/U/double 在线分数。optimizer 可见的 ``.grad`` 只安装一次，开启
重要性追踪不会改变梯度、随机数消费或参数更新。

一次成功尝试的主要边界为：

``microbatch gradients -> global mean -> unscale -> finite -> clip``
``-> estimator staging -> optimizer -> parameter-post -> accumulator``
``-> scheduler/scaler/RNG -> attempt-commit``。

checkpoint 保存 model/buffer、optimizer、scheduler、scaler、RNG、cursor、训练
计数和 importance accumulator，仍由 :class:`CheckpointStore` 的不可变对象加
权威 commit 两阶段协议发布。所有状态都是 primitive/tensor tree，不使用 pickle。
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from ..atomic import stable_json_hash
from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.seed import SeedPlan
from ..core.accumulator import ImportanceAccumulator
from ..core.estimators import (
    EstimatorResult,
    NO_UNBIASEDNESS_CLAIM,
    PLUGIN_SAME_BATCH_CLIP,
    UNBIASED_FIXED_STATE,
    double_sample_importance,
    raw_importance,
)
from ..core.registry import ParameterRegistry
from ..core.sufficient_statistics import WeightedSufficientStatistics
from ..core.tensors import TensorMap
from ..providers.training import BatchCursor, ModelAdapter, TrainingMicrobatch
from .checkpoint import CheckpointStore
from .events import EventRecord, EventSink, EventType
from .gradients import GradientAttempt, GradientPhase
from .optimizer import OptimizerBridge, StepOutcome
from .reducers import LocalReducer, Reducer
from .transactions import StepTransaction


_ESTIMATORS = {"raw", "u", "double"}
_AUTOCAST_DTYPES = {"none", "float16", "bfloat16"}
_GATE_ACCEPTED = {"PASS", "CONDITIONALLY_ACCEPTED"}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def install_training_rng(
    master_seed: int,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> SeedPlan:
    """把冻结 SeedPlan 的 runtime 域安装到当前训练进程。

    模型/数据 provider 可以使用局部 :class:`torch.Generator` 或 ``fork_rng``，但
    optimizer step 之间保存的 Python、NumPy、Torch RNG 必须有一个显式起点；否则
    两个内容完全相同的 fresh process 会因为调用前的全局 RNG 状态不同而产生不同
    ``attempt_commit_state_hash``。本函数只应在一个 phase 的 fresh engine 构造前
    调用；resume 随后会用 checkpoint 中的完整 RNG 状态覆盖这里安装的初始状态。

    各库使用彼此独立的一级域，并再按 rank 派生二级流。NumPy legacy RNG 只接受
    32-bit seed，因此仅在调用 API 的最后一步取模；SeedPlan artifact 仍保留完整
    63-bit 身份。CUDA 使用独立 ``rank_training`` 流，且只有 CUDA 实际可用时才安装。
    """

    plan = SeedPlan.from_master_seed(master_seed, world_size=world_size)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("TRAINING_RNG_RANK_OUT_OF_RANGE")
    python_seed = plan.derive_subseed("python_runtime", "rank", rank)
    numpy_seed = plan.derive_subseed("numpy_runtime", "rank", rank)
    torch_seed = plan.derive_subseed("torch_cpu_runtime", "rank", rank)
    random.seed(python_seed)
    np.random.seed(numpy_seed % (2**32))
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(plan.seed_for("rank_training", rank=rank))
    return plan


def _state_tree_hash(value: object) -> str:
    """稳定摘要 primitive/tensor 状态树，用于 step 边界而非 artifact codec。"""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().to(device="cpu").contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, str(candidate))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple(")
            for child in item:
                visit(child)
            digest.update(b"]" if isinstance(item, list) else b")")
            return
        digest.update(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        )
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def _capture_rng_state() -> dict[str, object]:
    """捕获 Python/NumPy/Torch CPU/CUDA RNG，不把设备可用性当成 Gate。"""

    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.cpu() for state in torch.cuda.get_rng_state_all())
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": cuda_states,
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise ValueError("TRAINING_RNG_STATE_FIELDS_MISMATCH")
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    cpu = state["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8:
        raise TypeError("TRAINING_TORCH_CPU_RNG_STATE_INVALID")
    torch.set_rng_state(cpu.cpu())
    cuda = state["torch_cuda"]
    if not isinstance(cuda, tuple) or not all(isinstance(item, torch.Tensor) for item in cuda):
        raise TypeError("TRAINING_TORCH_CUDA_RNG_STATE_INVALID")
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("TRAINING_CHECKPOINT_REQUIRES_CUDA_RNG")
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("TRAINING_CUDA_RNG_DEVICE_COUNT_MISMATCH")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda])


def _dtype_from_name(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError("TRAINING_ACCUMULATION_DTYPE_UNSUPPORTED")


@dataclass(frozen=True, slots=True)
class TrainingRunSpec:
    """单个可恢复训练 phase 的严格执行合同。"""

    run_id: str
    run_intent: str
    max_steps: int
    max_attempts: int
    importance_enabled: bool = True
    estimator_name: str = "u"
    accumulation_dtype: str = "float32"
    max_grad_norm: float | None = None
    autocast_dtype: str = "none"
    checkpoint_every_steps: int = 1
    log_every_steps: int = 1
    weights_exogenous: bool = False
    common_mean_assumption: bool = False
    estimator_decision_hash: str | None = None
    estimator_gate_status: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    checkpoint_segments: tuple[Mapping[str, int | None], ...] = ()
    schema_version: str = "training-run-spec-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "training-run-spec-v1" or not self.run_id:
            raise ValueError("TRAINING_RUN_SPEC_ID_OR_VERSION_INVALID")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise ValueError("TRAINING_RUN_INTENT_INVALID")
        for name in ("max_steps", "max_attempts", "checkpoint_every_steps", "log_every_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"TRAINING_RUN_SPEC_POSITIVE_INTEGER_REQUIRED:{name}")
        if self.max_attempts < self.max_steps:
            raise ValueError("TRAINING_MAX_ATTEMPTS_LT_MAX_STEPS")
        if self.estimator_name not in _ESTIMATORS:
            raise ValueError("TRAINING_ESTIMATOR_UNSUPPORTED")
        _dtype_from_name(self.accumulation_dtype)
        if self.autocast_dtype not in _AUTOCAST_DTYPES:
            raise ValueError("TRAINING_AUTOCAST_DTYPE_UNSUPPORTED")
        if self.max_grad_norm is not None and (
            isinstance(self.max_grad_norm, bool)
            or not isinstance(self.max_grad_norm, (int, float))
            or not math.isfinite(float(self.max_grad_norm))
            or float(self.max_grad_norm) < 0
        ):
            raise ValueError("TRAINING_MAX_GRAD_NORM_INVALID")
        if type(self.importance_enabled) is not bool:
            raise TypeError("TRAINING_IMPORTANCE_ENABLED_NOT_BOOL")
        if type(self.weights_exogenous) is not bool or type(self.common_mean_assumption) is not bool:
            raise TypeError("TRAINING_WEIGHT_ASSUMPTIONS_NOT_BOOL")
        if self.run_intent == "formal" and self.importance_enabled:
            if self.estimator_decision_hash is None or len(self.estimator_decision_hash) != 64:
                raise ValueError("FORMAL_TRAINING_ESTIMATOR_DECISION_REQUIRED")
            if self.estimator_gate_status not in _GATE_ACCEPTED:
                raise ValueError("FORMAL_TRAINING_ESTIMATOR_GATE_NOT_ACCEPTED")
        if self.run_intent == "local_fixture" and self.estimator_gate_status is not None:
            raise ValueError("LOCAL_FIXTURE_CANNOT_CLAIM_FORMAL_ESTIMATOR_GATE")
        segments: list[Mapping[str, int | None]] = []
        previous_end: int | None = None
        for index, raw in enumerate(self.checkpoint_segments):
            if not isinstance(raw, Mapping) or set(raw) != {
                "start_step", "end_step", "every_steps"
            }:
                raise ValueError("TRAINING_CHECKPOINT_SEGMENT_FIELDS_INVALID")
            start = raw["start_step"]
            end = raw["end_step"]
            every = raw["every_steps"]
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or start < 0
                or isinstance(every, bool)
                or not isinstance(every, int)
                or every <= 0
                or (
                    end is not None
                    and (isinstance(end, bool) or not isinstance(end, int) or end <= start)
                )
            ):
                raise ValueError("TRAINING_CHECKPOINT_SEGMENT_VALUE_INVALID")
            if index == 0 and start != 0:
                raise ValueError("TRAINING_CHECKPOINT_SEGMENTS_MUST_START_AT_ZERO")
            if index > 0 and previous_end != start:
                raise ValueError("TRAINING_CHECKPOINT_SEGMENTS_NOT_CONTIGUOUS")
            if end is None and index != len(self.checkpoint_segments) - 1:
                raise ValueError("TRAINING_CHECKPOINT_OPEN_SEGMENT_NOT_LAST")
            segments.append(
                MappingProxyType(
                    {"start_step": start, "end_step": end, "every_steps": every}
                )
            )
            previous_end = end
        if segments and previous_end is not None and previous_end < self.max_steps:
            raise ValueError("TRAINING_CHECKPOINT_SEGMENTS_DO_NOT_COVER_RUN")
        object.__setattr__(self, "checkpoint_segments", tuple(segments))
        object.__setattr__(
            self, "metadata", freeze_json_mapping(self.metadata, field="TrainingRunSpec.metadata")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_intent": self.run_intent,
            "max_steps": self.max_steps,
            "max_attempts": self.max_attempts,
            "importance_enabled": self.importance_enabled,
            "estimator_name": self.estimator_name,
            "accumulation_dtype": self.accumulation_dtype,
            "max_grad_norm": self.max_grad_norm,
            "autocast_dtype": self.autocast_dtype,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "log_every_steps": self.log_every_steps,
            "weights_exogenous": self.weights_exogenous,
            "common_mean_assumption": self.common_mean_assumption,
            "estimator_decision_hash": self.estimator_decision_hash,
            "estimator_gate_status": self.estimator_gate_status,
            "metadata": thaw_json_value(self.metadata),
            "checkpoint_segments": [dict(segment) for segment in self.checkpoint_segments],
        }

    def should_checkpoint(self, successful_step: int) -> bool:
        """按成功 optimizer step 判断是否位于冻结的分段保存点。"""

        if isinstance(successful_step, bool) or not isinstance(successful_step, int):
            raise TypeError("TRAINING_CHECKPOINT_STEP_NOT_INTEGER")
        if successful_step <= 0:
            return False
        if not self.checkpoint_segments:
            return successful_step % self.checkpoint_every_steps == 0
        # segment 使用左闭右开区间；step 正好等于边界时属于下一段。最终 phase-end
        # checkpoint 仍由 run() 的独立规则保证，不依赖整除关系。
        for segment in self.checkpoint_segments:
            start = int(segment["start_step"] or 0)
            end = segment["end_step"]
            if successful_step >= start and (end is None or successful_step < int(end)):
                return (successful_step - start) % int(segment["every_steps"] or 1) == 0
        return False

    @property
    def spec_hash(self) -> str:
        return stable_json_hash(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingRunSpec":
        expected = {
            "schema_version", "run_id", "run_intent", "max_steps", "max_attempts",
            "importance_enabled", "estimator_name", "accumulation_dtype", "max_grad_norm",
            "autocast_dtype", "checkpoint_every_steps", "log_every_steps",
            "weights_exogenous", "common_mean_assumption", "estimator_decision_hash",
            "estimator_gate_status", "metadata", "checkpoint_segments",
        }
        # 0.3.x 的 wire spec 没有 checkpoint_segments；读取时迁移为空分段，仍按
        # checkpoint_every_steps 工作。写出始终使用新的完整字段集合。
        if set(value) == expected - {"checkpoint_segments"}:
            value = {**dict(value), "checkpoint_segments": []}
        elif set(value) != expected:
            raise ValueError("TRAINING_RUN_SPEC_FIELDS_MISMATCH")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("TRAINING_RUN_SPEC_METADATA_NOT_OBJECT")
        raw_segments = value["checkpoint_segments"]
        if not isinstance(raw_segments, list) or not all(
            isinstance(item, Mapping) for item in raw_segments
        ):
            raise TypeError("TRAINING_CHECKPOINT_SEGMENTS_NOT_ARRAY")
        normalized = dict(value)
        normalized["checkpoint_segments"] = tuple(dict(item) for item in raw_segments)
        return cls(**normalized)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingState:
    """训练 phase 的控制状态；``global_step`` 只统计成功更新。"""

    global_step: int = 0
    attempt_index: int = 0
    skipped_steps: int = 0
    event_sequence: int = 0
    last_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if min(self.global_step, self.attempt_index, self.skipped_steps, self.event_sequence) < 0:
            raise ValueError("TRAINING_STATE_NEGATIVE_COUNTER")
        if self.global_step + self.skipped_steps != self.attempt_index:
            raise ValueError("TRAINING_STATE_ATTEMPT_ACCOUNTING_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "attempt_index": self.attempt_index,
            "skipped_steps": self.skipped_steps,
            "event_sequence": self.event_sequence,
            "last_checkpoint_id": self.last_checkpoint_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingState":
        if set(value) != {
            "global_step", "attempt_index", "skipped_steps", "event_sequence",
            "last_checkpoint_id",
        }:
            raise ValueError("TRAINING_STATE_FIELDS_MISMATCH")
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class TrainingStepRecord:
    """一个 attempt 的标量诊断与两种 post-state 摘要。"""

    attempt_index: int
    global_step: int
    status: str
    batch_ids: tuple[str, ...]
    # 非有限梯度对应的 attempt 仍需留下可恢复记录，但 canonical JSON 禁止 NaN/Inf。
    # 因而此字段以 null 表示“本次 loss 不可报告”，而不是把非有限数写进 artifact。
    mean_loss: float | None
    effective_count: int
    global_gradient_norm: float | None
    clip_factor: float | None
    estimator_name: str | None
    parameter_post_state_hash: str | None
    attempt_commit_state_hash: str | None
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "global_step": self.global_step,
            "status": self.status,
            "batch_ids": list(self.batch_ids),
            "mean_loss": self.mean_loss,
            "effective_count": self.effective_count,
            "global_gradient_norm": self.global_gradient_norm,
            "clip_factor": self.clip_factor,
            "estimator_name": self.estimator_name,
            "parameter_post_state_hash": self.parameter_post_state_hash,
            "attempt_commit_state_hash": self.attempt_commit_state_hash,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingStepRecord":
        """从 checkpoint primitive tree 严格恢复单步诊断。"""

        expected = {
            "attempt_index", "global_step", "status", "batch_ids", "mean_loss",
            "effective_count", "global_gradient_norm", "clip_factor", "estimator_name",
            "parameter_post_state_hash", "attempt_commit_state_hash", "skip_reason",
        }
        if set(value) != expected:
            raise ValueError("TRAINING_STEP_RECORD_FIELDS_MISMATCH")
        batch_ids = value["batch_ids"]
        if not isinstance(batch_ids, list) or not all(
            isinstance(item, str) and item for item in batch_ids
        ):
            raise TypeError("TRAINING_STEP_RECORD_BATCH_IDS_INVALID")
        return cls(
            attempt_index=value["attempt_index"],  # type: ignore[arg-type]
            global_step=value["global_step"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            batch_ids=tuple(batch_ids),
            mean_loss=value["mean_loss"],  # type: ignore[arg-type]
            effective_count=value["effective_count"],  # type: ignore[arg-type]
            global_gradient_norm=value["global_gradient_norm"],  # type: ignore[arg-type]
            clip_factor=value["clip_factor"],  # type: ignore[arg-type]
            estimator_name=value["estimator_name"],  # type: ignore[arg-type]
            parameter_post_state_hash=value["parameter_post_state_hash"],  # type: ignore[arg-type]
            attempt_commit_state_hash=value["attempt_commit_state_hash"],  # type: ignore[arg-type]
            skip_reason=value["skip_reason"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class GradientReadyEvent:
    """optimizer 前端点：参数、未裁剪 global mean 与 batch 身份。"""

    global_step: int
    attempt_index: int
    parameters_pre: TensorMap
    mean_gradient: TensorMap
    optimizer_gradient: TensorMap
    microbatch_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParameterPostEvent:
    """optimizer 后、scheduler/scaler/RNG commit 前的参数边界。"""

    transaction: StepTransaction
    parameters_post: TensorMap
    outcome: StepOutcome


@dataclass(frozen=True, slots=True)
class AttemptCommitEvent:
    """scheduler/scaler 与 cursor/RNG 均完成后的权威 attempt 边界。"""

    transaction: StepTransaction
    control_state_hash: str
    cursor_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SkippedAttemptEvent:
    """全局 nonfinite 导致未修改参数的 attempt。"""

    transaction: StepTransaction
    microbatch_ids: tuple[str, ...]


@runtime_checkable
class TrainingStepObserver(Protocol):
    """只读 step observer；任何模型状态修改都会被训练引擎拒绝。"""

    def on_gradient_ready(self, event: GradientReadyEvent) -> None: ...

    def on_parameter_post(self, event: ParameterPostEvent) -> None: ...

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None: ...

    def on_skip(self, event: SkippedAttemptEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class ImportanceSnapshot:
    """长期累计器的安全状态引用与可比较标量摘要。"""

    registry_hash: str
    successful_steps: int
    skipped_steps: int
    state_hash: str
    scalar_summaries: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalar_summaries", MappingProxyType(dict(self.scalar_summaries)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "importance-snapshot-v1",
            "registry_hash": self.registry_hash,
            "successful_steps": self.successful_steps,
            "skipped_steps": self.skipped_steps,
            "state_hash": self.state_hash,
            "scalar_summaries": dict(self.scalar_summaries),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ImportanceSnapshot":
        expected = {
            "schema_version", "registry_hash", "successful_steps", "skipped_steps",
            "state_hash", "scalar_summaries",
        }
        if set(value) != expected or value.get("schema_version") != "importance-snapshot-v1":
            raise ValueError("IMPORTANCE_SNAPSHOT_FIELDS_OR_VERSION_INVALID")
        summaries = value["scalar_summaries"]
        if not isinstance(summaries, Mapping):
            raise TypeError("IMPORTANCE_SNAPSHOT_SUMMARIES_NOT_OBJECT")
        return cls(
            registry_hash=value["registry_hash"],  # type: ignore[arg-type]
            successful_steps=value["successful_steps"],  # type: ignore[arg-type]
            skipped_steps=value["skipped_steps"],  # type: ignore[arg-type]
            state_hash=value["state_hash"],  # type: ignore[arg-type]
            scalar_summaries={str(key): float(item) for key, item in summaries.items()},
        )


@dataclass(frozen=True, slots=True)
class ImportanceTrajectoryPoint:
    """一个 checkpoint commit 对应的重要性累计快照。"""

    global_step: int
    checkpoint_id: str
    snapshot: ImportanceSnapshot

    def __post_init__(self) -> None:
        if isinstance(self.global_step, bool) or not isinstance(self.global_step, int) or self.global_step <= 0:
            raise ValueError("IMPORTANCE_TRAJECTORY_STEP_INVALID")
        if not self.checkpoint_id:
            raise ValueError("IMPORTANCE_TRAJECTORY_CHECKPOINT_ID_EMPTY")
        if self.snapshot.successful_steps != self.global_step:
            raise ValueError("IMPORTANCE_TRAJECTORY_SNAPSHOT_STEP_MISMATCH")

    def to_dict(self) -> dict[str, object]:
        return {
            "global_step": self.global_step,
            "checkpoint_id": self.checkpoint_id,
            "snapshot": self.snapshot.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ImportanceTrajectoryPoint":
        if set(value) != {"global_step", "checkpoint_id", "snapshot"}:
            raise ValueError("IMPORTANCE_TRAJECTORY_POINT_FIELDS_INVALID")
        snapshot = value["snapshot"]
        if not isinstance(snapshot, Mapping):
            raise TypeError("IMPORTANCE_TRAJECTORY_POINT_SNAPSHOT_INVALID")
        return cls(
            value["global_step"],  # type: ignore[arg-type]
            value["checkpoint_id"],  # type: ignore[arg-type]
            ImportanceSnapshot.from_mapping(snapshot),
        )


@dataclass(frozen=True, slots=True)
class ImportanceTrajectory:
    """按 checkpoint 边界发布的单调 importance trajectory。"""

    registry_hash: str
    estimator_decision_hash: str | None
    points: tuple[ImportanceTrajectoryPoint, ...]

    def __post_init__(self) -> None:
        if len(self.registry_hash) != 64:
            raise ValueError("IMPORTANCE_TRAJECTORY_REGISTRY_HASH_INVALID")
        if self.estimator_decision_hash is not None and len(self.estimator_decision_hash) != 64:
            raise ValueError("IMPORTANCE_TRAJECTORY_DECISION_HASH_INVALID")
        points = tuple(self.points)
        if any(point.snapshot.registry_hash != self.registry_hash for point in points):
            raise ValueError("IMPORTANCE_TRAJECTORY_REGISTRY_MISMATCH")
        steps = [point.global_step for point in points]
        if steps != sorted(set(steps)):
            raise ValueError("IMPORTANCE_TRAJECTORY_STEPS_NOT_STRICTLY_INCREASING")
        object.__setattr__(self, "points", points)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "importance-trajectory-v1",
            "registry_hash": self.registry_hash,
            "estimator_decision_hash": self.estimator_decision_hash,
            "points": [point.to_dict() for point in self.points],
        }
        payload["artifact_hash"] = stable_json_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ImportanceTrajectory":
        expected = {
            "schema_version", "registry_hash", "estimator_decision_hash", "points",
            "artifact_hash",
        }
        if set(value) != expected or value.get("schema_version") != "importance-trajectory-v1":
            raise ValueError("IMPORTANCE_TRAJECTORY_FIELDS_OR_VERSION_INVALID")
        raw_points = value["points"]
        if not isinstance(raw_points, list) or not all(isinstance(item, Mapping) for item in raw_points):
            raise TypeError("IMPORTANCE_TRAJECTORY_POINTS_INVALID")
        result = cls(
            value["registry_hash"],  # type: ignore[arg-type]
            value["estimator_decision_hash"],  # type: ignore[arg-type]
            tuple(ImportanceTrajectoryPoint.from_mapping(item) for item in raw_points),
        )
        if value["artifact_hash"] != result.to_dict()["artifact_hash"]:
            raise ValueError("IMPORTANCE_TRAJECTORY_HASH_MISMATCH")
        return result


class OnlineImportanceTracker:
    """从同一步 microbatch 梯度生成在线分数并事务累计。"""

    def __init__(self, registry: ParameterRegistry, spec: TrainingRunSpec) -> None:
        if not spec.importance_enabled:
            raise ValueError("IMPORTANCE_TRACKER_REQUIRES_ENABLED_SPEC")
        self.registry = registry
        self.spec = spec
        template = TensorMap(
            {name: registry.parameter(name).detach() for name in registry.eligible_names},
            registry=registry,
        )
        self.accumulator = ImportanceAccumulator(
            template, accumulation_dtype=_dtype_from_name(spec.accumulation_dtype)
        )
        self.accumulator.set_initial_parameters(template)

    @staticmethod
    def _weighted_mean(samples: Sequence[TensorMap], weights: Sequence[float]) -> TensorMap:
        result = TensorMap.zeros_like(samples[0])
        total = float(sum(weights))
        for sample, weight in zip(samples, weights, strict=True):
            result = result + sample * float(weight)
        return result / total

    def stage(
        self,
        micro_gradients: Sequence[TensorMap],
        weights: Sequence[float],
        learning_rates: Mapping[str, float],
        *,
        clip_factor: float,
    ) -> tuple[EstimatorResult, EstimatorResult, EstimatorResult]:
        """生成主 estimator 与 raw 对照；不修改 accumulator。"""

        return self.stage_distributed(
            micro_gradients,
            weights,
            learning_rates,
            clip_factor=clip_factor,
            reducer=LocalReducer(),
            rank=0,
        )

    def stage_distributed(
        self,
        micro_gradients: Sequence[TensorMap],
        weights: Sequence[float],
        learning_rates: Mapping[str, float],
        *,
        clip_factor: float,
        reducer: Reducer[torch.Tensor],
        rank: int,
    ) -> tuple[EstimatorResult, EstimatorResult, EstimatorResult]:
        """归约全局充分统计量后生成相同的 rank-local 结果。

        U 只需要 ``G1/G2/N1/N2/count``，无需 all-gather 每个参数梯度。double
        采用 ``(rank + local_micro_index) % 2`` 的稳定两半映射，并分别归约两半
        的加权梯度和权重。所有 rank 因而得到完全相同的 estimator score。
        """

        if len(micro_gradients) != len(weights) or not micro_gradients:
            raise ValueError("IMPORTANCE_MICRO_GRADIENT_WEIGHT_MISMATCH")
        dtype = _dtype_from_name(self.spec.accumulation_dtype)
        converted = [sample.to(dtype=dtype) for sample in micro_gradients]
        g1 = TensorMap.zeros_like(converted[0], dtype=dtype)
        g2 = TensorMap.zeros_like(converted[0], dtype=dtype)
        for sample, weight in zip(converted, weights, strict=True):
            g1 = g1 + sample * float(weight)
            g2 = g2 + sample.map(torch.square) * float(weight) ** 2
        reference_tensor = next(iter(g1.values()))
        scalars = {
            "__n1__": torch.tensor(sum(weights), dtype=torch.float64, device=reference_tensor.device),
            "__n2__": torch.tensor(
                sum(float(value) ** 2 for value in weights),
                dtype=torch.float64,
                device=reference_tensor.device,
            ),
        }
        reduced = reducer.sum_tensors(
            {
                **{f"g1:{name}": value for name, value in g1.items()},
                **{f"g2:{name}": value for name, value in g2.items()},
                **scalars,
            }
        )
        global_statistics = WeightedSufficientStatistics(
            count=reducer.sum_int(len(converted)),
            g1=TensorMap(
                {name: reduced[f"g1:{name}"] for name in g1}, registry=self.registry
            ),
            g2=TensorMap(
                {name: reduced[f"g2:{name}"] for name in g2}, registry=self.registry
            ),
            n1=float(reduced["__n1__"].detach().cpu().item()),
            n2=float(reduced["__n2__"].detach().cpu().item()),
            accumulation_dtype=dtype,
            statistical_unit="microbatch_mean_gradient",
            weight_unit="effective_loss_units",
            sampling_design="ordered_disjoint_microbatches",
            weights_exogenous=self.spec.weights_exogenous,
            common_mean_assumption=self.spec.common_mean_assumption,
            metadata={"world_size": reducer.capabilities.world_size},  # type: ignore[attr-defined]
        )
        mean = global_statistics.mean_gradient
        clip_source = "none" if clip_factor == 1.0 else "same_batch_mean_gradient"
        raw_unclipped = EstimatorResult.from_core(
            "raw_same_batch_gradient_importance",
            raw_importance(mean),
            learning_rates,
            clip_factor=1.0,
            clip_source="none",
            unbiasedness_claim=NO_UNBIASEDNESS_CLAIM,
        )
        raw_clipped = EstimatorResult.from_core(
            "raw_same_batch_gradient_importance_clipped",
            raw_importance(mean),
            learning_rates,
            clip_factor=clip_factor,
            clip_source=clip_source,
            unbiasedness_claim=(
                PLUGIN_SAME_BATCH_CLIP if clip_source != "none" else NO_UNBIASEDNESS_CLAIM
            ),
        )
        if self.spec.estimator_name == "raw":
            return raw_clipped, raw_unclipped, raw_clipped
        if global_statistics.count < 2:
            raise ValueError("IMPORTANCE_ESTIMATOR_REQUIRES_AT_LEAST_TWO_MICROBATCHES")
        if self.spec.estimator_name == "u":
            main = EstimatorResult.from_weighted_u(
                global_statistics,
                learning_rates,
                clip_factor=clip_factor,
                clip_source=clip_source,
            )
            return main, raw_unclipped, raw_clipped
        half_a = TensorMap.zeros_like(converted[0], dtype=dtype)
        half_b = TensorMap.zeros_like(converted[0], dtype=dtype)
        weight_a = 0.0
        weight_b = 0.0
        for index, (sample, weight) in enumerate(zip(converted, weights, strict=True)):
            if (rank + index) % 2 == 0:
                half_a = half_a + sample * float(weight)
                weight_a += float(weight)
            else:
                half_b = half_b + sample * float(weight)
                weight_b += float(weight)
        half_reduced = reducer.sum_tensors(
            {
                **{f"a:{name}": value for name, value in half_a.items()},
                **{f"b:{name}": value for name, value in half_b.items()},
                "__wa__": torch.tensor(weight_a, dtype=torch.float64, device=reference_tensor.device),
                "__wb__": torch.tensor(weight_b, dtype=torch.float64, device=reference_tensor.device),
            }
        )
        global_weight_a = float(half_reduced["__wa__"].detach().cpu().item())
        global_weight_b = float(half_reduced["__wb__"].detach().cpu().item())
        if global_weight_a <= 0 or global_weight_b <= 0:
            raise ValueError("DOUBLE_SAMPLE_SPLIT_EMPTY")
        mean_a = TensorMap(
            {name: half_reduced[f"a:{name}"] / global_weight_a for name in half_a},
            registry=self.registry,
        )
        mean_b = TensorMap(
            {name: half_reduced[f"b:{name}"] / global_weight_b for name in half_b},
            registry=self.registry,
        )
        main = EstimatorResult.from_core(
            "double_sample_gradient_importance",
            double_sample_importance(mean_a, mean_b),
            learning_rates,
            clip_factor=clip_factor,
            clip_source=clip_source,
            unbiasedness_claim=(
                PLUGIN_SAME_BATCH_CLIP
                if clip_source != "none"
                else (
                    UNBIASED_FIXED_STATE
                    if self.spec.weights_exogenous and self.spec.common_mean_assumption
                    else NO_UNBIASEDNESS_CLAIM
                )
            ),
            metadata={
                "split_mapping": "(rank+local_micro_index)%2",
                "statistical_unit": global_statistics.statistical_unit,
                "weight_unit": global_statistics.weight_unit,
                "sampling_design": "ordered_disjoint_microbatches",
                "weights_exogenous": self.spec.weights_exogenous,
                "common_mean_assumption": self.spec.common_mean_assumption,
            },
        )
        return main, raw_unclipped, raw_clipped

    def commit(
        self,
        main: EstimatorResult,
        raw_unclipped: EstimatorResult,
        raw_clipped: EstimatorResult,
        outcome: StepOutcome,
    ) -> None:
        data = TensorMap(outcome.data_delta, registry=self.registry)
        total = TensorMap(outcome.total_delta, registry=self.registry)
        parameters = TensorMap(
            {name: self.registry.parameter(name).detach() for name in self.registry.eligible_names},
            registry=self.registry,
        )
        self.accumulator.add_step(
            main.score,
            raw=raw_unclipped.score,
            raw_clipped=raw_clipped.score,
            data_update=data,
            total_update=total,
            weight_decay_update=TensorMap(
                outcome.weight_decay_delta, registry=self.registry
            ),
            current_parameters=parameters,
        )

    def record_skip(self) -> None:
        self.accumulator.record_skip()

    def snapshot(self) -> ImportanceSnapshot:
        self.accumulator.validate_invariants()
        state = self.accumulator.state_dict()
        views = {
            "signed": self.accumulator.signed,
            "positive": self.accumulator.positive,
            "negative_mass": self.accumulator.negative_mass,
            "absolute": self.accumulator.absolute,
            "raw": self.accumulator.raw,
            "raw_clipped": self.accumulator.raw_clipped,
            "data_movement": self.accumulator.data_movement,
            "net_data_movement": self.accumulator.net_data_movement,
            "total_endpoint_movement": self.accumulator.total_endpoint_movement,
            "weight_decay_movement": self.accumulator.weight_decay_movement,
            "magnitude": self.accumulator.magnitude,
        }
        summaries = {
            name: float(value.scalar_sum(dtype=torch.float64).item())
            for name, value in views.items()
        }
        return ImportanceSnapshot(
            self.registry.coordinate_registry_hash,
            self.accumulator.successful_steps,
            self.accumulator.skipped_steps,
            _state_tree_hash(state),
            summaries,
        )


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """训练执行的机器可读终态。"""

    run_id: str
    status: str
    state: TrainingState
    registry_hash: str
    optimizer_contract_hash: str
    records: tuple[TrainingStepRecord, ...]
    checkpoint_ids: tuple[str, ...]
    importance_snapshot: ImportanceSnapshot | None
    importance_trajectory: ImportanceTrajectory | None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": "training-run-result-v1",
            "run_id": self.run_id,
            "status": self.status,
            "state": self.state.to_dict(),
            "registry_hash": self.registry_hash,
            "optimizer_contract_hash": self.optimizer_contract_hash,
            "records": [record.to_dict() for record in self.records],
            "checkpoint_ids": list(self.checkpoint_ids),
            "importance_snapshot": (
                None if self.importance_snapshot is None else self.importance_snapshot.to_dict()
            ),
            "importance_trajectory": (
                None if self.importance_trajectory is None else self.importance_trajectory.to_dict()
            ),
        }
        payload["artifact_hash"] = stable_json_hash(payload)
        return payload


class TrainingEngine:
    """纯 Torch、provider 驱动的单 phase 训练执行器。"""

    def __init__(
        self,
        *,
        spec: TrainingRunSpec,
        model: ModelAdapter,
        optimizer: torch.optim.Optimizer,
        cursor: BatchCursor,
        scheduler: object | None = None,
        scaler: object | None = None,
        reducer: Reducer[torch.Tensor] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        event_sink: EventSink | None = None,
        experiment_id: str = "local-experiment",
        attempt_id: str = "attempt-0000",
        session_id: str = "session-0000",
        rank: int = 0,
        observers: Sequence[TrainingStepObserver] = (),
    ) -> None:
        self.spec = spec
        self.model = model
        self.optimizer = optimizer
        self.cursor = cursor
        self.scheduler = scheduler
        self.scaler = scaler
        self.reducer = reducer or LocalReducer()
        self.checkpoint_store = checkpoint_store
        self.event_sink = event_sink
        self.experiment_id = experiment_id
        self.attempt_id = attempt_id
        self.session_id = session_id
        self.rank = rank
        self.observers = tuple(observers)
        if any(not isinstance(observer, TrainingStepObserver) for observer in self.observers):
            raise TypeError("TRAINING_STEP_OBSERVER_PROTOCOL_INVALID")
        self.registry = ParameterRegistry.from_model(model.module, optimizer)
        self.named_parameters = {
            name: self.registry.parameter(name) for name in self.registry.eligible_names
        }
        self.bridge = OptimizerBridge(self.named_parameters, optimizer)
        self.tracker = OnlineImportanceTracker(self.registry, spec) if spec.importance_enabled else None
        self.state = TrainingState()
        self._records: list[TrainingStepRecord] = []
        self._checkpoint_ids: list[str] = []
        self._importance_points: list[ImportanceTrajectoryPoint] = []

    def register_observer(self, observer: TrainingStepObserver) -> None:
        """在训练开始前注册只读事务 observer。

        训练 runner 只有在模型、optimizer、registry 与恢复点都构造完成后，才能
        建立真实 endpoint 状态捕获器，因此不能要求所有 observer 都在
        ``TrainingEngine.__init__`` 之前存在。本方法只允许在尚未执行任何 attempt
        时调用；恢复后的 engine 也可注册，因为恢复本身不属于新的训练 attempt。
        """

        if not isinstance(observer, TrainingStepObserver):
            raise TypeError("TRAINING_STEP_OBSERVER_PROTOCOL_INVALID")
        if self._records or self.state.attempt_index != 0:
            # fresh-process resume 会先安装 checkpoint，再注册 observer；其
            # attempt_index 可以大于零，但内存中的本 session records 已经随
            # checkpoint 恢复。调用方应在 resume 之前注册，避免漏过恢复边界。
            raise RuntimeError("TRAINING_OBSERVER_MUST_REGISTER_BEFORE_RESUME_OR_RUN")
        self.observers = (*self.observers, observer)

    def capture_observer_state(
        self,
        *,
        optimizer_gradient: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, object]:
        """为只读 observer 复制完整、可安全编码的训练状态树。

        返回值只含 primitive/tensor/NumPy 状态，可直接交给安全 TensorBundle；
        所有 tensor 都是 detached CPU clone，调用方无法借此修改活动训练状态。
        ``optimizer_gradient`` 只应在 ``gradient-ready`` 边界传入，它表示真正将
        安装给 optimizer 的已 unscale/已 clip 梯度，用于 fresh-state replay。
        """

        gradients = None
        if optimizer_gradient is not None:
            if set(optimizer_gradient) != set(self.named_parameters):
                raise ValueError("OBSERVER_OPTIMIZER_GRADIENT_NAMES_MISMATCH")
            gradients = {
                name: optimizer_gradient[name].detach().cpu().clone()
                for name in self.registry.eligible_names
            }
        return {
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self.named_parameters.items()
            },
            "buffers": {
                name: buffer.detach().cpu().clone()
                for name, buffer in self.model.module.named_buffers(remove_duplicate=True)
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                None if self.scheduler is None else self.scheduler.state_dict()
            ),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "rng": _capture_rng_state(),
            "cursor": dict(self.cursor.state_dict()),
            "training_state": self.state.to_dict(),
            "model_modes": {
                name: module.training
                for name, module in self.model.module.named_modules()
            },
            "optimizer_gradient": gradients,
        }

    def _notify(self, method_name: str, event: object) -> None:
        """调用只读 observer，并以完整 model state hash 防止观测扰动训练。"""

        if not self.observers:
            return
        before = _state_tree_hash(self.model.module.state_dict())
        for observer in self.observers:
            method = getattr(observer, method_name)
            method(event)
        after = _state_tree_hash(self.model.module.state_dict())
        if after != before:
            raise RuntimeError(f"TRAINING_OBSERVER_MUTATED_MODEL_STATE:{method_name}")

    def _emit(self, event_type: EventType, payload: dict[str, object], *, critical: bool = False) -> None:
        if self.event_sink is None:
            return
        deterministic = self.spec.run_intent == "local_fixture"
        event_identity = (
            _sha256(
                f"{self.spec.spec_hash}:{self.rank}:{self.state.event_sequence}:{event_type.value}".encode(
                    "utf-8"
                )
            )[:32]
            if deterministic
            else None
        )
        event = EventRecord.create(
            experiment_id=self.experiment_id,
            run_id=self.spec.run_id,
            attempt_id=self.attempt_id,
            session_id=self.session_id,
            rank=self.rank,
            event_type=event_type,
            sequence=self.state.event_sequence,
            payload=payload,
            event_id=event_identity,
            occurred_at="1970-01-01T00:00:00+00:00" if deterministic else None,
        )
        self.event_sink.append(event, critical=critical)
        self.state = TrainingState(
            self.state.global_step,
            self.state.attempt_index,
            self.state.skipped_steps,
            self.state.event_sequence + 1,
            self.state.last_checkpoint_id,
        )

    def _autocast(self):
        if self.spec.autocast_dtype == "none":
            return nullcontext()
        device = next(iter(self.named_parameters.values())).device
        dtype = torch.float16 if self.spec.autocast_dtype == "float16" else torch.bfloat16
        if device.type == "cpu" and dtype == torch.float16:
            raise RuntimeError("CPU_FLOAT16_AUTOCAST_UNSUPPORTED")
        return torch.autocast(device_type=device.type, dtype=dtype)

    def _loss_scale(self) -> float:
        if self.scaler is None:
            return 1.0
        getter = getattr(self.scaler, "get_scale", None)
        if not callable(getter):
            raise TypeError("TRAINING_SCALER_MISSING_GET_SCALE")
        value = float(getter())
        if not math.isfinite(value) or value <= 0:
            raise ValueError("TRAINING_SCALER_SCALE_INVALID")
        return value

    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return loss
        scale = getattr(self.scaler, "scale", None)
        if not callable(scale):
            raise TypeError("TRAINING_SCALER_MISSING_SCALE")
        return scale(loss)

    def _learning_rates(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for index, group in enumerate(self.optimizer.param_groups):
            lr = float(group["lr"])
            if not math.isfinite(lr) or lr < 0:
                raise ValueError("TRAINING_LEARNING_RATE_INVALID")
            values[f"group_{index:04d}"] = lr
        return values

    def _collect_micro_gradients(
        self, microbatches: Sequence[TrainingMicrobatch]
    ) -> tuple[list[dict[str, torch.Tensor]], list[float], float, int]:
        """取得 scaled micro gradients；不写 ``Parameter.grad``。"""

        parameters = tuple(self.named_parameters.values())
        gradients: list[dict[str, torch.Tensor]] = []
        weights: list[float] = []
        loss_numerator = 0.0
        effective_count = 0
        for microbatch in microbatches:
            # 若 module 是 DistributedDataParallel，所有 microbatch 都放在
            # ``no_sync`` 中；全局 mean 与 U 充分量由项目 reducer 恰好归约一次，
            # 避免 DDP hook 与显式统计归约重复通信/重复除 world size。
            no_sync = getattr(self.model.module, "no_sync", None)
            sync_context = no_sync() if callable(no_sync) else nullcontext()
            with sync_context:
                with self._autocast():
                    loss = self.model.loss(microbatch)
                scaled_loss = self._scale_loss(loss.mean_loss)
                values = torch.autograd.grad(
                    scaled_loss,
                    parameters,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
            current: dict[str, torch.Tensor] = {}
            for name, parameter, gradient in zip(
                self.named_parameters, parameters, values, strict=True
            ):
                current[name] = (
                    torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
                )
            gradients.append(current)
            weights.append(float(loss.effective_count))
            loss_numerator += float(loss.loss_numerator.detach().cpu().item())
            effective_count += loss.effective_count
        return gradients, weights, loss_numerator, effective_count

    def _global_mean_gradient(
        self,
        scaled_micro_gradients: Sequence[Mapping[str, torch.Tensor]],
        weights: Sequence[float],
        *,
        loss_scale: float,
    ) -> tuple[dict[str, torch.Tensor], list[TensorMap], int]:
        """按有效单元做跨 rank sum；返回 unscaled global mean 和本地 samples。"""

        local_weight = int(sum(weights))
        global_weight = self.reducer.sum_int(local_weight)
        if global_weight <= 0:
            raise ValueError("TRAINING_GLOBAL_EFFECTIVE_COUNT_ZERO")
        weighted: dict[str, torch.Tensor] = {
            name: torch.zeros_like(parameter)
            for name, parameter in self.named_parameters.items()
        }
        unscaled_samples: list[TensorMap] = []
        for gradient_map, weight in zip(scaled_micro_gradients, weights, strict=True):
            unscaled = {
                name: gradient.detach() / loss_scale for name, gradient in gradient_map.items()
            }
            unscaled_samples.append(
                TensorMap(unscaled, registry=self.registry, require_finite=False)
            )
            for name, gradient in gradient_map.items():
                weighted[name].add_(gradient, alpha=float(weight))
        reduced = self.reducer.sum_tensors(weighted)
        return (
            {name: gradient / (loss_scale * global_weight) for name, gradient in reduced.items()},
            unscaled_samples,
            global_weight,
        )

    def _install_for_unscale(self, mean: Mapping[str, torch.Tensor], loss_scale: float) -> None:
        for name, parameter in self.named_parameters.items():
            value = mean[name].to(device=parameter.device, dtype=parameter.dtype) * loss_scale
            parameter.grad = value.detach().clone()
        if self.scaler is not None:
            unscale = getattr(self.scaler, "unscale_", None)
            if not callable(unscale):
                raise TypeError("TRAINING_SCALER_MISSING_UNSCALE")
            unscale(self.optimizer)

    def _update_scaler(self) -> None:
        if self.scaler is not None:
            update = getattr(self.scaler, "update", None)
            if not callable(update):
                raise TypeError("TRAINING_SCALER_MISSING_UPDATE")
            update()

    def _parameter_state(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters.items()}

    def _control_state(self) -> dict[str, object]:
        return {
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "rng": _capture_rng_state(),
            "cursor": dict(self.cursor.state_dict()),
            "training_state": self.state.to_dict(),
        }

    def _checkpoint_state(self) -> dict[str, object]:
        return {
            "schema_version": "training-checkpoint-state-v1",
            "run_spec_hash": self.spec.spec_hash,
            "registry_hash": self.registry.coordinate_registry_hash,
            "optimizer_contract_hash": self.registry.optimizer_contract_hash,
            "runtime_layout_hash": self.registry.runtime_layout_hash,
            "training_state": self.state.to_dict(),
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "rng": _capture_rng_state(),
            "cursor": dict(self.cursor.state_dict()),
            "importance": None if self.tracker is None else self.tracker.accumulator.state_dict(),
            "records": [record.to_dict() for record in self._records],
            "importance_trajectory_points": [
                point.to_dict() for point in self._importance_points
            ],
        }

    def save_checkpoint(self) -> str:
        if self.checkpoint_store is None:
            raise RuntimeError("TRAINING_CHECKPOINT_STORE_NOT_CONFIGURED")
        checkpoint_id = f"{self.spec.run_id}-step-{self.state.global_step:08d}"
        metadata = {
            "run_spec_hash": self.spec.spec_hash,
            "registry_hash": self.registry.coordinate_registry_hash,
            "optimizer_contract_hash": self.registry.optimizer_contract_hash,
            "world_size": self.reducer.capabilities.world_size,  # type: ignore[attr-defined]
        }
        previous_state = self.state
        # 先在将要发布的状态里保留 checkpoint 事件序号和自身 ID；对象若发布失败
        # 则恢复内存状态，不能让未提交对象推进 lineage。
        self.state = TrainingState(
            self.state.global_step,
            self.state.attempt_index,
            self.state.skipped_steps,
            self.state.event_sequence + (1 if self.event_sink is not None else 0),
            checkpoint_id,
        )
        point: ImportanceTrajectoryPoint | None = None
        if self.tracker is not None:
            point = ImportanceTrajectoryPoint(
                self.state.global_step,
                checkpoint_id,
                self.tracker.snapshot(),
            )
            self._importance_points.append(point)
        try:
            self.checkpoint_store.publish(
                checkpoint_id,
                self._checkpoint_state(),
                generation=self.state.global_step,
                metadata=metadata,
                parent_checkpoint_id=previous_state.last_checkpoint_id,
            )
        except Exception:
            if point is not None:
                self._importance_points.pop()
            self.state = previous_state
            raise
        self._checkpoint_ids.append(checkpoint_id)
        if self.event_sink is not None:
            deterministic = self.spec.run_intent == "local_fixture"
            sequence = previous_state.event_sequence
            self.event_sink.append(
                EventRecord.create(
                    experiment_id=self.experiment_id,
                    run_id=self.spec.run_id,
                    attempt_id=self.attempt_id,
                    session_id=self.session_id,
                    rank=self.rank,
                    event_type=EventType.CHECKPOINT,
                    sequence=sequence,
                    payload={"checkpoint_id": checkpoint_id},
                    event_id=(
                        _sha256(
                            f"{self.spec.spec_hash}:{self.rank}:{sequence}:checkpoint".encode(
                                "utf-8"
                            )
                        )[:32]
                        if deterministic
                        else None
                    ),
                    occurred_at=(
                        "1970-01-01T00:00:00+00:00" if deterministic else None
                    ),
                ),
                critical=True,
            )
        return checkpoint_id

    def resume_checkpoint(self, checkpoint_id: str) -> str:
        """从显式指定的权威 checkpoint commit 恢复。

        ``checkpoint_id`` 来自已经通过路径边界检查的 ``recovery.resume_ref``。
        与 ``resume_latest`` 相比，本入口不会根据目录内容猜测恢复点，因此适合
        正式任务和 fresh-process 故障恢复。
        """

        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("TRAINING_RESUME_CHECKPOINT_ID_INVALID")
        return self._resume_from_checkpoint_store(checkpoint_id)

    def resume_latest(self) -> str:
        """从已核验的权威 latest 索引恢复。

        该兼容入口只应在上层已经把显式 ``resume_ref`` 解析为本 rank 的 checkpoint
        根目录后使用；新任务的 ``run`` 路径不会再自动调用它。
        """

        return self._resume_from_checkpoint_store(None)

    def _resume_from_checkpoint_store(self, checkpoint_id: str | None) -> str:
        """加载、验证并原子安装一个 checkpoint 状态。"""

        if self.checkpoint_store is None:
            raise RuntimeError("TRAINING_CHECKPOINT_STORE_NOT_CONFIGURED")
        expected = {
            "run_spec_hash": self.spec.spec_hash,
            "registry_hash": self.registry.coordinate_registry_hash,
            "optimizer_contract_hash": self.registry.optimizer_contract_hash,
            "world_size": self.reducer.capabilities.world_size,  # type: ignore[attr-defined]
        }
        if checkpoint_id is None:
            raw, commit = self.checkpoint_store.load_latest(expected_metadata=expected)
        else:
            raw, commit = self.checkpoint_store.load(
                checkpoint_id,
                expected_metadata=expected,
            )
        if not isinstance(raw, Mapping) or raw.get("schema_version") != "training-checkpoint-state-v1":
            raise ValueError("TRAINING_CHECKPOINT_STATE_INVALID")
        if raw.get("run_spec_hash") != self.spec.spec_hash:
            raise ValueError("TRAINING_CHECKPOINT_SPEC_HASH_MISMATCH")
        state_value = raw.get("training_state")
        if not isinstance(state_value, Mapping):
            raise TypeError("TRAINING_CHECKPOINT_CONTROL_STATE_INVALID")
        staged_state = TrainingState.from_mapping(state_value)
        cursor_value = raw.get("cursor")
        rng_value = raw.get("rng")
        if not isinstance(cursor_value, Mapping) or not isinstance(rng_value, Mapping):
            raise TypeError("TRAINING_CHECKPOINT_CURSOR_OR_RNG_INVALID")
        self.model.module.load_state_dict(raw["model"], strict=True)  # type: ignore[arg-type]
        self.optimizer.load_state_dict(raw["optimizer"])  # type: ignore[arg-type]
        if self.scheduler is None:
            if raw.get("scheduler") is not None:
                raise ValueError("TRAINING_CHECKPOINT_SCHEDULER_MISMATCH")
        else:
            self.scheduler.load_state_dict(raw["scheduler"])  # type: ignore[attr-defined]
        if self.scaler is None:
            if raw.get("scaler") is not None:
                raise ValueError("TRAINING_CHECKPOINT_SCALER_MISMATCH")
        else:
            self.scaler.load_state_dict(raw["scaler"])  # type: ignore[attr-defined]
        self.cursor.load_state_dict(cursor_value)
        raw_records = raw.get("records", [])
        if not isinstance(raw_records, list) or not all(
            isinstance(item, Mapping) for item in raw_records
        ):
            raise TypeError("TRAINING_CHECKPOINT_RECORDS_INVALID")
        staged_records = [TrainingStepRecord.from_mapping(item) for item in raw_records]
        raw_points = raw.get("importance_trajectory_points", [])
        if not isinstance(raw_points, list) or not all(
            isinstance(item, Mapping) for item in raw_points
        ):
            raise TypeError("TRAINING_CHECKPOINT_TRAJECTORY_POINTS_INVALID")
        staged_points = [ImportanceTrajectoryPoint.from_mapping(item) for item in raw_points]
        if self.tracker is None:
            if raw.get("importance") is not None:
                raise ValueError("TRAINING_CHECKPOINT_IMPORTANCE_MISMATCH")
        else:
            importance = raw.get("importance")
            if not isinstance(importance, Mapping):
                raise TypeError("TRAINING_CHECKPOINT_IMPORTANCE_STATE_INVALID")
            self.tracker.accumulator.load_state_dict(importance)
        _restore_rng_state(rng_value)
        # optimizer.load_state_dict 允许替换 param-group 字典，因此恢复后重新绑定
        # bridge，避免继续读取恢复前的学习率或静态 options 引用。
        self.bridge = OptimizerBridge(self.named_parameters, self.optimizer)
        self.state = TrainingState(
            staged_state.global_step,
            staged_state.attempt_index,
            staged_state.skipped_steps,
            staged_state.event_sequence,
            commit.checkpoint_id,
        )
        self._records = staged_records
        self._importance_points = staged_points
        self._checkpoint_ids.append(commit.checkpoint_id)
        return commit.checkpoint_id

    def _run_attempt(self, microbatches: Sequence[TrainingMicrobatch]) -> TrainingStepRecord:
        self.model.module.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        next_attempt = self.state.attempt_index + 1
        parameters_pre = TensorMap(
            {name: parameter.detach().clone() for name, parameter in self.named_parameters.items()},
            registry=self.registry,
        )
        scaled_micros, weights, local_loss_numerator, _ = self._collect_micro_gradients(microbatches)
        loss_scale = self._loss_scale()
        mean_gradient, micro_samples, global_count = self._global_mean_gradient(
            scaled_micros, weights, loss_scale=loss_scale
        )
        self._install_for_unscale(mean_gradient, loss_scale)
        attempt = GradientAttempt.capture(
            {name: parameter.grad for name, parameter in self.named_parameters.items()},
            gradient_scale=loss_scale,
            scaled=False,
        ).check_finite()
        reference_parameter = next(iter(self.named_parameters.values()))
        reduced_loss = self.reducer.sum_tensors(
            {
                "loss_numerator": torch.tensor(
                    local_loss_numerator,
                    dtype=torch.float64,
                    device=reference_parameter.device,
                )
            }
        )["loss_numerator"]
        mean_loss_value = float(reduced_loss.detach().cpu().item()) / global_count
        mean_loss = mean_loss_value if math.isfinite(mean_loss_value) else None
        transaction = StepTransaction(self.state.global_step, next_attempt)
        if attempt.phase is GradientPhase.SKIPPED:
            transaction = transaction.skip(attempt.skip_reason or "NONFINITE")
            self._notify(
                "on_skip",
                SkippedAttemptEvent(
                    transaction,
                    tuple(batch.batch_id for batch in microbatches),
                ),
            )
            if self.tracker is not None:
                self.tracker.record_skip()
            self._update_scaler()
            self.optimizer.zero_grad(set_to_none=True)
            self.state = TrainingState(
                self.state.global_step,
                next_attempt,
                self.state.skipped_steps + 1,
                self.state.event_sequence,
                self.state.last_checkpoint_id,
            )
            return TrainingStepRecord(
                next_attempt,
                self.state.global_step,
                "SKIPPED",
                tuple(batch.batch_id for batch in microbatches),
                mean_loss,
                global_count,
                None,
                None,
                None,
                None,
                None,
                transaction.skip_reason,
            )
        if self.spec.max_grad_norm is not None:
            attempt = attempt.clip(float(self.spec.max_grad_norm))
        clip_factor = 1.0 if attempt.clip_factor is None else attempt.clip_factor
        self._notify(
            "on_gradient_ready",
            GradientReadyEvent(
                self.state.global_step,
                next_attempt,
                parameters_pre,
                TensorMap(mean_gradient, registry=self.registry),
                TensorMap(attempt.gradients, registry=self.registry),
                tuple(batch.batch_id for batch in microbatches),
                tuple(sample_id for batch in microbatches for sample_id in batch.sample_ids),
            ),
        )
        attempt.install(self.named_parameters)
        main: EstimatorResult | None = None
        raw_unclipped: EstimatorResult | None = None
        raw_clipped: EstimatorResult | None = None
        if self.tracker is not None:
            # 每个 rank 提交本地 microbatch 充分统计量，再由 reducer 求全局 S1/S2；
            # 因而在线 U/double 与参数更新共享同一个 global-batch 语义。
            main, raw_unclipped, raw_clipped = self.tracker.stage_distributed(
                micro_samples,
                weights,
                self._learning_rates(),
                clip_factor=clip_factor,
                reducer=self.reducer,
                rank=self.rank,
            )
        outcome = self.bridge.step()
        parameter_post_hash = _state_tree_hash(self._parameter_state())
        transaction = transaction.mark_parameter_post(parameter_post_hash)
        self._notify(
            "on_parameter_post",
            ParameterPostEvent(
                transaction,
                TensorMap(self._parameter_state(), registry=self.registry),
                outcome,
            ),
        )
        if self.tracker is not None:
            assert main is not None and raw_unclipped is not None and raw_clipped is not None
            self.tracker.commit(main, raw_unclipped, raw_clipped, outcome)
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore[attr-defined]
        self._update_scaler()
        self.optimizer.zero_grad(set_to_none=True)
        self.state = TrainingState(
            self.state.global_step + 1,
            next_attempt,
            self.state.skipped_steps,
            self.state.event_sequence,
            self.state.last_checkpoint_id,
        )
        commit_hash = _state_tree_hash(self._control_state())
        if commit_hash == parameter_post_hash:
            commit_hash = _sha256((commit_hash + ":attempt_commit").encode("ascii"))
        transaction = transaction.commit_attempt(commit_hash)
        self._notify(
            "on_attempt_commit",
            AttemptCommitEvent(
                transaction,
                commit_hash,
                dict(self.cursor.state_dict()),
            ),
        )
        return TrainingStepRecord(
            next_attempt,
            self.state.global_step,
            "COMMITTED",
            tuple(batch.batch_id for batch in microbatches),
            mean_loss,
            global_count,
            attempt.global_norm,
            clip_factor,
            None if main is None else main.estimator_name,
            transaction.parameter_post_state_hash,
            transaction.attempt_commit_state_hash,
        )

    def run(
        self,
        *,
        resume: bool = False,
        until_step: int | None = None,
    ) -> TrainingRunResult:
        """运行到目标 step、attempt 上限或数据耗尽，并发布最终 checkpoint。

        ``until_step`` 用于调度器安全暂停同一个冻结 run；它不进入 spec hash，
        因而后续 fresh process 可用原 spec 从 checkpoint 继续到 ``max_steps``。
        """

        if resume:
            self.resume_latest()
        self._emit(EventType.RUN_LIFECYCLE, {"status": "RUNNING"}, critical=True)
        if until_step is not None and (
            isinstance(until_step, bool)
            or not isinstance(until_step, int)
            or not self.state.global_step < until_step <= self.spec.max_steps
        ):
            raise ValueError("TRAINING_UNTIL_STEP_INVALID")
        target_step = self.spec.max_steps if until_step is None else until_step
        exhausted = False
        while (
            self.state.global_step < target_step
            and self.state.attempt_index < self.spec.max_attempts
        ):
            try:
                microbatches = self.cursor.next_microbatches()
            except StopIteration:
                exhausted = True
                break
            record = self._run_attempt(microbatches)
            self._records.append(record)
            if (
                record.status == "COMMITTED"
                and self.spec.should_checkpoint(self.state.global_step)
                and self.checkpoint_store is not None
            ):
                self.save_checkpoint()
            if self.state.attempt_index % self.spec.log_every_steps == 0:
                self._emit(EventType.OPTIMIZER_STEP, record.to_dict())
        if self.state.global_step == self.spec.max_steps:
            status = "COMPLETE"
        elif self.state.global_step == target_step and until_step is not None:
            status = "PAUSED"
        else:
            status = "DATA_EXHAUSTED" if exhausted else "MAX_ATTEMPTS_REACHED"
        if (
            self.checkpoint_store is not None
            and self.state.global_step > 0
            and self.state.last_checkpoint_id
            != f"{self.spec.run_id}-step-{self.state.global_step:08d}"
        ):
            self.save_checkpoint()
        self._emit(EventType.RUN_LIFECYCLE, {"status": status}, critical=True)
        final_snapshot = None if self.tracker is None else self.tracker.snapshot()
        trajectory = (
            None
            if self.tracker is None
            else ImportanceTrajectory(
                self.registry.coordinate_registry_hash,
                self.spec.estimator_decision_hash,
                tuple(self._importance_points),
            )
        )
        return TrainingRunResult(
            self.spec.run_id,
            status,
            self.state,
            self.registry.coordinate_registry_hash,
            self.registry.optimizer_contract_hash,
            tuple(self._records),
            tuple(self._checkpoint_ids),
            final_snapshot,
            trajectory,
        )


# 语义别名：一个 phase 的执行器就是冻结 TrainingEngine 的配置化包装。
TrainingPhaseRunner = TrainingEngine


__all__ = [
    "ImportanceSnapshot",
    "ImportanceTrajectory",
    "ImportanceTrajectoryPoint",
    "AttemptCommitEvent",
    "GradientReadyEvent",
    "OnlineImportanceTracker",
    "ParameterPostEvent",
    "SkippedAttemptEvent",
    "TrainingEngine",
    "TrainingPhaseRunner",
    "TrainingRunResult",
    "TrainingRunSpec",
    "TrainingState",
    "TrainingStepRecord",
    "TrainingStepObserver",
    "install_training_rng",
]
