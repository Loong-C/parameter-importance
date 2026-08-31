"""Stage 2 可恢复的正式编排核心。

该模块补足固定状态研究从“数学 kernel”到“可在服务器执行的编排层”之间的空白，
但不绑定 Transformers、Datasets 或任何模型实现。调用方只需提供
``FixedStateGradientProvider``、冻结 draws 和 repetition mapping：

* :class:`StreamingReferenceSizer` 以 FP64 充分统计流式构建 A/B reference，按
  block pair 发布不可变快照并从权威 commit 恢复；
* :class:`RecoverablePairedWaveRunner` 从同一基础梯度池产生 raw/double/U，逐
  repetition 两阶段提交，再流式汇总 Bias/Variance/MSE/MAE 与三种成本口径；
* :class:`Stage2RecommendationEngine` 只消费预注册阈值和完整 cell 观测，形成
  fixture 推荐或 formal 候选。候选默认不具备正式资格，必须再绑定本阶段 Gate。

本机测试可以完整运行这些状态机，但所有输出都保持
``formal_eligible=false``。真正 formal 运行必须传入
``FormalExecutionEvidence(run_intent="formal", ...)``；缺少冻结合同、资产或前置
Gate 时会在读取任何梯度之前 fail-closed。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

import numpy as np

from param_importance_nlp.contracts.artifacts import (
    validate_reference_result_artifact,
)
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.immutable import (
    freeze_json_mapping,
    thaw_json_value,
)
from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.stage23 import (
    ArtifactQualification,
    FormalExecutionEvidence,
    require_accepted_gate,
)
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.providers import FixedStateGradientProvider, GradientBatch
from param_importance_nlp.runtime.tensor_bundle import (
    load_tensor_bundle,
    publish_tensor_bundle,
)

from .sampling import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    RepetitionMapping,
)
from .stage2 import EstimatorDecision, PairedEstimatorRunner


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_COST_SEMANTICS = (
    "scientific_equal_sample_cost",
    "isolated_estimator_cost",
    "online_training_incremental_cost",
)

_FORMAL_PLAN_TASK_CONTRACT = {
    "stage2.05_paired_estimator_runner": (
        "pilot",
        "preregistered_development",
    ),
    "stage2.06_pilot_and_matrix_freeze": (
        "pilot",
        "preregistered_pilot",
    ),
    "stage2.07_main_sweep": (
        "confirmatory",
        "pilot_frozen_primary",
    ),
}


def _require_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} 必须是 1..160 字符的安全标识，只能含字母、数字、._-"
        )
    return value


def _require_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _require_gate_binding(gate: GateRecord, artifact_ref: str) -> None:
    if not artifact_ref or artifact_ref not in gate.evidence_refs:
        raise FormalRunRejected(
            f"FORMAL_GATE_DOES_NOT_BIND_ARTIFACT:{gate.gate_id}:{artifact_ref or '<missing>'}"
        )


def _logical_ref(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field_name} 必须是 POSIX workspace 逻辑引用")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} 发生路径逃逸")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class FormalExperimentPlan:
    """Stage 2 单个正式 paired wave 的冻结 B/M/R 执行计划。

    该对象只描述调度，不携带统计结论。``task_id`` 同时决定允许的随机流与
    ``selection_basis``：开发/ pilot 计划必须在观测结果前预注册，确认性计划则
    必须来自 pilot 冻结结果。运行器还会在装载时复核当前 SamplingPlan、
    FormalExecutionEvidence 与完整直接前驱 commit 集，因而不能把其他状态或其他
    配置的 B/M/R 静默搬入本次执行。

    ``microbatch_counts`` 是本 wave 需要同时计算的嵌套 M 集。确认性主实验只允许
    一个由 pilot 冻结的 ``M_primary>2``；double estimator 仍由相同 B 的前后两半
    自动计算，不要求把 ``M=2`` 伪装成主 U 分组。
    """

    plan_id: str
    task_id: str
    wave_id: str
    cell_id: str
    stream: str
    batch_size: int
    microbatch_counts: tuple[int, ...]
    repetitions: int
    sampling_plan_hash: str
    execution_evidence_hash: str
    source_artifact_refs: tuple[str, ...]
    selection_basis: str
    pilot_thresholds: Mapping[str, object] | None = None
    scope: str = "formal"
    state: str = "FROZEN"
    schema_version: str = "stage2-formal-experiment-plan-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage2-formal-experiment-plan-v1":
            raise ValueError("FORMAL_EXPERIMENT_PLAN_SCHEMA_UNSUPPORTED")
        _require_identifier(self.plan_id, field_name="plan_id")
        _require_identifier(self.wave_id, field_name="wave_id")
        _require_identifier(self.cell_id, field_name="cell_id")
        contract = _FORMAL_PLAN_TASK_CONTRACT.get(self.task_id)
        if contract is None:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_TASK_UNSUPPORTED")
        expected_stream, expected_basis = contract
        if self.stream != expected_stream:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_STREAM_TASK_MISMATCH")
        if self.selection_basis != expected_basis:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_SELECTION_BASIS_MISMATCH")
        if self.scope != "formal" or self.state != "FROZEN":
            raise FormalRunRejected("FORMAL_EXPERIMENT_PLAN_NOT_FROZEN")
        if self.batch_size not in CANDIDATE_BATCH_SIZES:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_BATCH_NOT_PREREGISTERED")
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions <= 0
        ):
            raise TypeError("FORMAL_EXPERIMENT_PLAN_REPETITIONS_INVALID")
        microbatch_counts = tuple(self.microbatch_counts)
        if not microbatch_counts or microbatch_counts != tuple(
            sorted(set(microbatch_counts))
        ):
            raise ValueError("FORMAL_EXPERIMENT_PLAN_MICROBATCH_COUNTS_NOT_SORTED_UNIQUE")
        if any(
            isinstance(item, bool)
            or item not in CANDIDATE_MICROBATCH_COUNTS
            or self.batch_size % item
            for item in microbatch_counts
        ):
            raise ValueError("FORMAL_EXPERIMENT_PLAN_MICROBATCH_NOT_PREREGISTERED_OR_DIVISOR")
        largest = max(microbatch_counts)
        if any(largest % item for item in microbatch_counts):
            raise ValueError("FORMAL_EXPERIMENT_PLAN_MICROBATCH_COUNTS_NOT_NESTED")
        if self.task_id == "stage2.07_main_sweep" and (
            len(microbatch_counts) != 1 or microbatch_counts[0] <= 2
        ):
            raise ValueError("FORMAL_CONFIRMATORY_REQUIRES_ONE_PRIMARY_M_GT_2")
        object.__setattr__(self, "microbatch_counts", microbatch_counts)
        threshold_fields = {
            "bias_margin",
            "max_corrected_nmse_ratio",
            "min_spearman",
            "min_topk_overlap",
            "max_online_cost_ratio",
        }
        if self.task_id == "stage2.06_pilot_and_matrix_freeze":
            raw_thresholds = self.pilot_thresholds
            if not isinstance(raw_thresholds, Mapping) or set(raw_thresholds) != (
                threshold_fields
            ):
                raise ValueError("FORMAL_PILOT_THRESHOLDS_REQUIRED")
            numeric = {name: float(raw_thresholds[name]) for name in threshold_fields}
            if any(not math.isfinite(item) for item in numeric.values()):
                raise ValueError("FORMAL_PILOT_THRESHOLDS_NON_FINITE")
            if (
                numeric["bias_margin"] < 0
                or numeric["max_corrected_nmse_ratio"] <= 0
                or numeric["max_online_cost_ratio"] <= 0
                or not -1 <= numeric["min_spearman"] <= 1
                or not 0 <= numeric["min_topk_overlap"] <= 1
            ):
                raise ValueError("FORMAL_PILOT_THRESHOLDS_OUT_OF_RANGE")
            object.__setattr__(
                self,
                "pilot_thresholds",
                freeze_json_mapping(
                    numeric,
                    field="FormalExperimentPlan.pilot_thresholds",
                ),
            )
        elif self.pilot_thresholds is not None:
            raise ValueError("FORMAL_NON_PILOT_PLAN_FORBIDS_THRESHOLDS")
        _require_sha256(self.sampling_plan_hash, field_name="sampling_plan_hash")
        _require_sha256(
            self.execution_evidence_hash,
            field_name="execution_evidence_hash",
        )
        refs = tuple(
            _logical_ref(item, field_name=f"source_artifact_refs[{index}]")
            for index, item in enumerate(self.source_artifact_refs)
        )
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("FORMAL_EXPERIMENT_PLAN_SOURCE_REFS_EMPTY_OR_DUPLICATE")
        # 前驱 commit 构成集合身份；排序后再进入 artifact hash，避免 CLI 参数顺序
        # 或不同调度器的枚举顺序改变同一份冻结计划的内容地址。
        object.__setattr__(self, "source_artifact_refs", tuple(sorted(refs)))

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "wave_id": self.wave_id,
            "cell_id": self.cell_id,
            "stream": self.stream,
            "batch_size": self.batch_size,
            "microbatch_counts": list(self.microbatch_counts),
            "repetitions": self.repetitions,
            "sampling_plan_hash": self.sampling_plan_hash,
            "execution_evidence_hash": self.execution_evidence_hash,
            "source_artifact_refs": list(self.source_artifact_refs),
            "selection_basis": self.selection_basis,
            "pilot_thresholds": (
                None
                if self.pilot_thresholds is None
                else thaw_json_value(self.pilot_thresholds)
            ),
            "scope": self.scope,
            "state": self.state,
            "formal_eligible": True,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FormalExperimentPlan":
        from param_importance_nlp.contracts.stage23 import validate_stage23_artifact

        validate_stage23_artifact(value)
        expected = {
            "schema_version", "plan_id", "task_id", "wave_id", "cell_id", "stream",
            "batch_size", "microbatch_counts", "repetitions", "sampling_plan_hash",
            "execution_evidence_hash", "source_artifact_refs", "selection_basis",
            "pilot_thresholds",
            "scope", "state", "formal_eligible", "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_FIELDS_MISMATCH")
        raw_m = value["microbatch_counts"]
        raw_refs = value["source_artifact_refs"]
        if not isinstance(raw_m, list) or not isinstance(raw_refs, list):
            raise TypeError("FORMAL_EXPERIMENT_PLAN_ARRAY_FIELD_INVALID")
        plan = cls(
            plan_id=value["plan_id"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            wave_id=value["wave_id"],  # type: ignore[arg-type]
            cell_id=value["cell_id"],  # type: ignore[arg-type]
            stream=value["stream"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            microbatch_counts=tuple(raw_m),  # type: ignore[arg-type]
            repetitions=value["repetitions"],  # type: ignore[arg-type]
            sampling_plan_hash=value["sampling_plan_hash"],  # type: ignore[arg-type]
            execution_evidence_hash=value["execution_evidence_hash"],  # type: ignore[arg-type]
            source_artifact_refs=tuple(raw_refs),  # type: ignore[arg-type]
            selection_basis=value["selection_basis"],  # type: ignore[arg-type]
            pilot_thresholds=value["pilot_thresholds"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            state=value["state"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["formal_eligible"] is not True:
            raise FormalRunRejected("FORMAL_EXPERIMENT_PLAN_ELIGIBILITY_REQUIRED")
        if value["artifact_hash"] != plan.artifact_hash:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_HASH_MISMATCH")
        return plan


def _as_array(value: object, *, field_name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[union-attr]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[union-attr]
    if hasattr(value, "numpy"):
        value = value.numpy()  # type: ignore[union-attr]
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} 包含 NaN/Inf")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _as_vector(value: object, *, field_name: str = "vector") -> dict[str, np.ndarray]:
    if hasattr(value, "items"):
        items = value.items()  # type: ignore[union-attr]
    else:
        raise TypeError(f"{field_name} 必须是 parameter-name -> tensor mapping")
    result: dict[str, np.ndarray] = {}
    for raw_name, item in items:
        name = str(raw_name)
        if not name or name in result:
            raise ValueError(f"{field_name} 包含空名称或重复参数名")
        result[name] = _as_array(item, field_name=f"{field_name}.{name}")
    if not result:
        raise ValueError(f"{field_name} 不能为空")
    return {name: result[name] for name in sorted(result)}


def _copy_vector(value: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    copied: dict[str, np.ndarray] = {}
    for name, item in _as_vector(value).items():
        item.setflags(write=False)
        copied[name] = item
    return MappingProxyType(copied)


def _vector_digest(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name, array in _as_vector(value).items():
        digest.update(len(name.encode("utf-8")).to_bytes(8, "big"))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _flatten(value: Mapping[str, object]) -> np.ndarray:
    vector = _as_vector(value)
    return np.concatenate([vector[name].reshape(-1) for name in sorted(vector)])


def _assert_compatible(
    left: Mapping[str, object], right: Mapping[str, object], *, field_name: str
) -> None:
    lhs = _as_vector(left, field_name=f"{field_name}.left")
    rhs = _as_vector(right, field_name=f"{field_name}.right")
    if tuple(lhs) != tuple(rhs):
        raise ValueError(f"{field_name} 参数名集合不一致")
    for name in lhs:
        if lhs[name].shape != rhs[name].shape:
            raise ValueError(f"{field_name}.{name} shape 不一致")


def _normalized_l1(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    _assert_compatible(left, right, field_name="normalized_l1")
    lhs, rhs = _flatten(left), _flatten(right)
    denominator = float(np.sum(np.abs(rhs)))
    numerator = float(np.sum(np.abs(lhs - rhs)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _draw_ids(draws: Sequence[object]) -> set[str]:
    return {
        str(getattr(draw, "draw_id"))
        for draw in draws
        if hasattr(draw, "draw_id")
    }


def _weighting_contract(provider: FixedStateGradientProvider) -> dict[str, object]:
    fields = (
        "statistical_unit",
        "weight_unit",
        "sampling_design",
        "weights_exogenous",
        "common_mean_assumption",
    )
    try:
        contract = {name: getattr(provider, name) for name in fields}
    except AttributeError as error:
        raise ValueError("GRADIENT_PROVIDER_WEIGHTING_CONTRACT_MISSING") from error
    for name in fields[:3]:
        if not isinstance(contract[name], str) or not str(contract[name]).strip():
            raise ValueError(f"GRADIENT_PROVIDER_{name.upper()}_MISSING")
    for name in fields[3:]:
        if type(contract[name]) is not bool:
            raise TypeError(f"GRADIENT_PROVIDER_{name.upper()}_MUST_BE_BOOL")
    return contract


class _GradientMoments:
    """一个 sampling stream 的 FP64 加权梯度充分统计量。"""

    def __init__(self) -> None:
        self.count = 0
        self.n1 = 0.0
        self.n2 = 0.0
        self.first_weight: float | None = None
        self.all_equal_weights = True
        self.g1: dict[str, np.ndarray] = {}
        self.g2: dict[str, np.ndarray] = {}

    def update(self, batch: GradientBatch, expected: Mapping[str, object]) -> None:
        if batch.weighting_assumptions != dict(expected):
            raise ValueError("GRADIENT_BATCH_WEIGHTING_CONTRACT_DRIFT")
        weight = float(batch.statistical_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("statistical_weight 必须是有限正数")
        vector = _as_vector(batch.gradients, field_name="gradient_batch.gradients")
        if self.g1:
            _assert_compatible(self.g1, vector, field_name="reference_gradient")
        else:
            self.g1 = {name: np.zeros_like(item) for name, item in vector.items()}
            self.g2 = {name: np.zeros_like(item) for name, item in vector.items()}
        if self.first_weight is None:
            self.first_weight = weight
        elif weight != self.first_weight:
            self.all_equal_weights = False
        for name, gradient in vector.items():
            self.g1[name] += weight * gradient
            self.g2[name] += weight**2 * np.square(gradient)
        self.count += 1
        self.n1 += weight
        self.n2 += weight**2

    def mean(self) -> dict[str, np.ndarray]:
        if self.count <= 0 or self.n1 <= 0:
            raise ValueError("空充分统计量没有 mean")
        return {name: value / self.n1 for name, value in self.g1.items()}

    def u(self, *, assumptions: Mapping[str, object]) -> dict[str, np.ndarray]:
        if self.count < 2:
            raise ValueError("U reference 至少需要两个 block")
        denominator = self.n1**2 - self.n2
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError("U reference 分母必须为有限正数")
        if not self.all_equal_weights and not (
            assumptions.get("weights_exogenous") is True
            and assumptions.get("common_mean_assumption") is True
        ):
            raise ValueError("WEIGHTED_REFERENCE_ASSUMPTIONS_NOT_DECLARED")
        return {
            name: (np.square(self.g1[name]) - self.g2[name]) / denominator
            for name in self.g1
        }

    def combine(self, other: "_GradientMoments") -> "_GradientMoments":
        if not self.g1:
            return other.copy()
        if not other.g1:
            return self.copy()
        _assert_compatible(self.g1, other.g1, field_name="reference_moments")
        result = _GradientMoments()
        result.count = self.count + other.count
        result.n1 = self.n1 + other.n1
        result.n2 = self.n2 + other.n2
        result.first_weight = self.first_weight
        result.all_equal_weights = bool(
            self.all_equal_weights
            and other.all_equal_weights
            and self.first_weight == other.first_weight
        )
        result.g1 = {name: self.g1[name] + other.g1[name] for name in self.g1}
        result.g2 = {name: self.g2[name] + other.g2[name] for name in self.g2}
        return result

    def copy(self) -> "_GradientMoments":
        return self.from_state(self.to_state())

    def to_state(self) -> dict[str, object]:
        return {
            "count": self.count,
            "n1": self.n1,
            "n2": self.n2,
            "first_weight": self.first_weight,
            "all_equal_weights": self.all_equal_weights,
            "g1": {name: np.array(value, copy=True) for name, value in self.g1.items()},
            "g2": {name: np.array(value, copy=True) for name, value in self.g2.items()},
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "_GradientMoments":
        required = {
            "count",
            "n1",
            "n2",
            "first_weight",
            "all_equal_weights",
            "g1",
            "g2",
        }
        if set(state) != required:
            raise ValueError("REFERENCE_MOMENTS_STATE_FIELDS_MISMATCH")
        result = cls()
        result.count = int(state["count"])
        result.n1 = float(state["n1"])
        result.n2 = float(state["n2"])
        first = state["first_weight"]
        result.first_weight = None if first is None else float(first)
        result.all_equal_weights = bool(state["all_equal_weights"])
        if not isinstance(state["g1"], Mapping) or not isinstance(state["g2"], Mapping):
            raise TypeError("REFERENCE_MOMENTS_TENSORS_NOT_MAPPINGS")
        result.g1 = _as_vector(state["g1"], field_name="reference_state.g1") if state["g1"] else {}
        result.g2 = _as_vector(state["g2"], field_name="reference_state.g2") if state["g2"] else {}
        if result.g1 or result.g2:
            _assert_compatible(result.g1, result.g2, field_name="reference_state")
        if result.count < 0 or result.n1 < 0 or result.n2 < 0:
            raise ValueError("REFERENCE_MOMENTS_STATE_NEGATIVE")
        return result


def _reference_vectors(
    a: _GradientMoments,
    b: _GradientMoments,
    assumptions: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    combined = a.combine(b)
    bias = combined.u(assumptions=assumptions)
    mean_a, mean_b = a.mean(), b.mean()
    _assert_compatible(mean_a, mean_b, field_name="reference_cross")
    cross = {name: mean_a[name] * mean_b[name] for name in mean_a}
    merged = combined.mean()
    ranking = {name: np.square(item) for name, item in merged.items()}
    return bias, cross, ranking


@dataclass(frozen=True, slots=True)
class ReferenceSizingPlan:
    """冻结 reference sizing ladder 与停止规则。"""

    reference_id: str
    candidate_sample_counts: tuple[int, ...]
    block_size: int
    convergence_tolerance: float
    required_consecutive: int
    execution: FormalExecutionEvidence = field(
        default_factory=lambda: FormalExecutionEvidence("local_fixture")
    )
    schema_version: str = "stage2-reference-sizing-plan-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage2-reference-sizing-plan-v1":
            raise ValueError("不支持的 ReferenceSizingPlan schema")
        _require_identifier(self.reference_id, field_name="reference_id")
        counts = tuple(self.candidate_sample_counts)
        if len(counts) < 2:
            raise ValueError("reference sizing 至少需要两个候选样本量")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in counts
        ):
            raise ValueError("candidate_sample_counts 必须是正整数")
        if tuple(sorted(set(counts))) != counts:
            raise ValueError("candidate_sample_counts 必须严格递增且无重复")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise ValueError("block_size 必须是正整数")
        if any(item % self.block_size for item in counts):
            raise ValueError("每个 candidate sample count 必须能被 block_size 整除")
        if not math.isfinite(self.convergence_tolerance) or self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance 必须是有限正数")
        if (
            isinstance(self.required_consecutive, bool)
            or not isinstance(self.required_consecutive, int)
            or self.required_consecutive <= 0
        ):
            raise ValueError("required_consecutive 必须是正整数")
        if self.execution.run_intent == "formal":
            self.execution.require_for_stage(2)

    @property
    def scope(self) -> str:
        return self.execution.run_intent

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "reference_id": self.reference_id,
            "candidate_sample_counts": list(self.candidate_sample_counts),
            "block_size": self.block_size,
            "convergence_tolerance": self.convergence_tolerance,
            "required_consecutive": self.required_consecutive,
            "execution_evidence_hash": self.execution.artifact_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}


@dataclass(frozen=True, slots=True)
class ReferenceSizingPoint:
    sample_count_per_stream: int
    block_count_total: int
    bias_reference_hash: str
    cross_reference_hash: str
    ranking_reference_hash: str
    normalized_l1_from_previous: float | None
    convergence_streak: int
    comparison_defined: bool
    comparison_reason: str | None

    def __post_init__(self) -> None:
        if self.sample_count_per_stream <= 0 or self.block_count_total < 2:
            raise ValueError("reference point 的样本数/block 数非法")
        for name in (
            "bias_reference_hash",
            "cross_reference_hash",
            "ranking_reference_hash",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        if self.normalized_l1_from_previous is not None and (
            self.normalized_l1_from_previous < 0
            or not math.isfinite(self.normalized_l1_from_previous)
        ):
            raise ValueError("normalized_l1_from_previous 必须是非负有限数或 null")
        if self.convergence_streak < 0:
            raise ValueError("convergence_streak 不能为负")
        if type(self.comparison_defined) is not bool:
            raise TypeError("comparison_defined 必须是显式 bool")
        if self.comparison_defined:
            if self.normalized_l1_from_previous is None or self.comparison_reason is not None:
                raise ValueError("defined comparison 必须有数值且不能有 reason")
        elif self.comparison_reason is None:
            raise ValueError("undefined comparison 必须给出 reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_count_per_stream": self.sample_count_per_stream,
            "block_count_total": self.block_count_total,
            "bias_reference_hash": self.bias_reference_hash,
            "cross_reference_hash": self.cross_reference_hash,
            "ranking_reference_hash": self.ranking_reference_hash,
            "normalized_l1_from_previous": self.normalized_l1_from_previous,
            "convergence_streak": self.convergence_streak,
            "comparison_defined": self.comparison_defined,
            "comparison_reason": self.comparison_reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReferenceSizingPoint":
        return cls(
            sample_count_per_stream=int(value["sample_count_per_stream"]),
            block_count_total=int(value["block_count_total"]),
            bias_reference_hash=str(value["bias_reference_hash"]),
            cross_reference_hash=str(value["cross_reference_hash"]),
            ranking_reference_hash=str(value["ranking_reference_hash"]),
            normalized_l1_from_previous=(
                None
                if value["normalized_l1_from_previous"] is None
                else float(value["normalized_l1_from_previous"])
            ),
            convergence_streak=int(value["convergence_streak"]),
            comparison_defined=bool(value["comparison_defined"]),
            comparison_reason=(
                None
                if value["comparison_reason"] is None
                else str(value["comparison_reason"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceSizingResult:
    """reference sizing 的当前/最终状态；正式候选也不会自行通过 Gate。"""

    plan_hash: str
    registry_hash: str
    provider_state_digest: str
    processed_sample_count_per_stream: int
    selected_sample_count_per_stream: int | None
    converged: bool
    points: tuple[ReferenceSizingPoint, ...]
    bias_reference: Mapping[str, object]
    cross_reference: Mapping[str, object]
    ranking_reference: Mapping[str, object]
    resumed_from_block_pairs: int
    scope: str
    qualification: ArtifactQualification
    weighting_assumptions: Mapping[str, object]
    schema_version: str = "stage2-reference-sizing-result-v1"

    def __post_init__(self) -> None:
        for name in ("plan_hash", "registry_hash", "provider_state_digest"):
            _require_sha256(getattr(self, name), field_name=name)
        if self.processed_sample_count_per_stream <= 0:
            raise ValueError("processed_sample_count_per_stream 必须为正")
        if self.selected_sample_count_per_stream is not None and not self.converged:
            raise ValueError("未收敛结果不能填写 selected sample count")
        if self.scope != self.qualification.scope:
            raise ValueError("ReferenceSizingResult scope 与 qualification 不一致")
        # 编排输出先是候选；资格化发生在生成 reference-result artifact 时。
        if self.qualification.formal_eligible:
            raise FormalRunRejected("REFERENCE_SIZING_RESULT_MUST_REMAIN_CANDIDATE")
        required_assumptions = {
            "statistical_unit",
            "weight_unit",
            "sampling_design",
            "weights_exogenous",
            "common_mean_assumption",
        }
        if set(self.weighting_assumptions) != required_assumptions:
            raise ValueError("ReferenceSizingResult weighting assumptions 不完整")
        object.__setattr__(
            self,
            "weighting_assumptions",
            freeze_json_mapping(
                self.weighting_assumptions,
                field="ReferenceSizingResult.weighting_assumptions",
            ),
        )
        object.__setattr__(self, "bias_reference", _copy_vector(self.bias_reference))
        object.__setattr__(self, "cross_reference", _copy_vector(self.cross_reference))
        object.__setattr__(self, "ranking_reference", _copy_vector(self.ranking_reference))
        _assert_compatible(self.bias_reference, self.cross_reference, field_name="reference")
        _assert_compatible(self.bias_reference, self.ranking_reference, field_name="reference")

    @property
    def status(self) -> str:
        if self.converged:
            return "FIXTURE_CONVERGED" if self.scope == "local_fixture" else "FORMAL_CANDIDATE"
        return "REFERENCE_UNRESOLVED"

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_hash": self.plan_hash,
            "registry_hash": self.registry_hash,
            "provider_state_digest": self.provider_state_digest,
            "processed_sample_count_per_stream": self.processed_sample_count_per_stream,
            "selected_sample_count_per_stream": self.selected_sample_count_per_stream,
            "converged": self.converged,
            "status": self.status,
            "points": [point.to_dict() for point in self.points],
            "bias_reference_hash": _vector_digest(self.bias_reference),
            "cross_reference_hash": _vector_digest(self.cross_reference),
            "ranking_reference_hash": _vector_digest(self.ranking_reference),
            "resumed_from_block_pairs": self.resumed_from_block_pairs,
            "scope": self.scope,
            "formal_eligible": False,
            "qualification_gate_hash": None,
            "weighting_assumptions": thaw_json_value(self.weighting_assumptions),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def to_reference_artifact(
        self,
        *,
        reference_id: str,
        block_size: int,
        tensor_bundle_ref: str,
        tensor_bundle_manifest_hash: str,
        qualification_gate: GateRecord | None = None,
        qualification_evidence_ref: str | None = None,
    ) -> dict[str, object]:
        """生成共享 reference manifest；正式资格必须显式绑定 Stage 2 Gate。"""

        qualified = False
        gate_hash: str | None = None
        if qualification_gate is not None:
            if self.scope != "formal" or not self.converged:
                raise FormalRunRejected("ONLY_CONVERGED_FORMAL_REFERENCE_CAN_BE_QUALIFIED")
            accepted = require_accepted_gate(qualification_gate, stage=2)
            if qualification_evidence_ref is None:
                raise FormalRunRejected("REFERENCE_QUALIFICATION_EVIDENCE_REF_REQUIRED")
            _require_gate_binding(accepted, qualification_evidence_ref)
            gate_hash = accepted.artifact_hash
            qualified = True
        sample_count = self.selected_sample_count_per_stream or self.processed_sample_count_per_stream
        payload: dict[str, object] = {
            "schema_version": "reference-result-v1",
            "reference_id": reference_id,
            "bias_reference_hash": _vector_digest(self.bias_reference),
            "cross_reference_hash": _vector_digest(self.cross_reference),
            "ranking_reference_hash": _vector_digest(self.ranking_reference),
            "sample_count_a": sample_count,
            "sample_count_b": sample_count,
            "block_size": block_size,
            "registry_hash": self.registry_hash,
            "scope": self.scope,
            "formal_eligible": qualified,
            "metadata": {
                "reference_sizing_result_hash": self.artifact_hash,
                "qualification_gate_hash": gate_hash,
                "converged": self.converged,
                "weighting_assumptions": thaw_json_value(
                    self.weighting_assumptions
                ),
            },
            "tensor_bundle_ref": tensor_bundle_ref,
            "tensor_bundle_manifest_hash": tensor_bundle_manifest_hash,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        validate_reference_result_artifact(payload)
        return payload


class _ReferenceSnapshotStore:
    """每个 block pair 发布一个不可变状态对象和独立权威 commit。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self._lock = self.root / "writer.lock"
        try:
            self._lock_fd = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("STAGE2_REFERENCE_WRITER_ALREADY_ACTIVE") from error

    def close(self) -> None:
        descriptor = getattr(self, "_lock_fd", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._lock_fd = -1
            self._lock.unlink(missing_ok=True)

    def __enter__(self) -> "_ReferenceSnapshotStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _state_digest(state: Mapping[str, object]) -> str:
        a = state["a"]
        b = state["b"]
        assert isinstance(a, Mapping) and isinstance(b, Mapping)
        payload = {
            key: value
            for key, value in state.items()
            if key not in {"a", "b", "last_bias"}
        }
        payload["a_g1_hash"] = _vector_digest(a["g1"]) if a["g1"] else None
        payload["a_g2_hash"] = _vector_digest(a["g2"]) if a["g2"] else None
        payload["b_g1_hash"] = _vector_digest(b["g1"]) if b["g1"] else None
        payload["b_g2_hash"] = _vector_digest(b["g2"]) if b["g2"] else None
        last = state["last_bias"]
        payload["last_bias_hash"] = _vector_digest(last) if last else None
        return canonical_json_hash(payload)

    def publish(self, sequence: int, state: Mapping[str, object]) -> None:
        state_digest = self._state_digest(state)
        object_path = self.objects / state_digest
        if not object_path.exists():
            bundle = publish_tensor_bundle(object_path, dict(state))
        else:
            restored, bundle = load_tensor_bundle(object_path)
            if not isinstance(restored, Mapping) or self._state_digest(restored) != state_digest:
                raise ValueError("REFERENCE_EXISTING_OBJECT_IDENTITY_MISMATCH")
        commit_payload: dict[str, object] = {
            "schema_version": "stage2-reference-progress-commit-v1",
            "sequence": sequence,
            "state_digest": state_digest,
            "object_ref": f"objects/{state_digest}",
            "object_manifest_hash": bundle.manifest_sha256,
        }
        commit_payload["artifact_hash"] = canonical_json_hash(commit_payload)
        commit_path = self.commits / f"{sequence:08d}.json"
        if commit_path.exists():
            existing = load_canonical_json(commit_path)
            if existing != commit_payload:
                raise ValueError("REFERENCE_COMMIT_CONFLICT")
            return
        write_canonical_json(commit_path, commit_payload)

    def latest(self) -> Mapping[str, object] | None:
        commits = sorted(self.commits.glob("*.json"))
        if not commits:
            return None
        commit = load_canonical_json(commits[-1])
        if not isinstance(commit, Mapping):
            raise ValueError("REFERENCE_COMMIT_NOT_OBJECT")
        required = {
            "schema_version",
            "sequence",
            "state_digest",
            "object_ref",
            "object_manifest_hash",
            "artifact_hash",
        }
        if set(commit) != required or commit["schema_version"] != "stage2-reference-progress-commit-v1":
            raise ValueError("REFERENCE_COMMIT_FIELDS_MISMATCH")
        payload = {key: value for key, value in commit.items() if key != "artifact_hash"}
        if canonical_json_hash(payload) != commit["artifact_hash"]:
            raise ValueError("REFERENCE_COMMIT_HASH_MISMATCH")
        relative = Path(str(commit["object_ref"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("REFERENCE_OBJECT_PATH_ESCAPE")
        state, bundle = load_tensor_bundle(self.root / relative)
        if bundle.manifest_sha256 != commit["object_manifest_hash"]:
            raise ValueError("REFERENCE_OBJECT_MANIFEST_HASH_MISMATCH")
        if not isinstance(state, Mapping) or self._state_digest(state) != commit["state_digest"]:
            raise ValueError("REFERENCE_STATE_DIGEST_MISMATCH")
        return state


class StreamingReferenceSizer:
    """按 A/B block pair 流式累计 reference，并可在任意 commit 边界恢复。"""

    def __init__(self, provider: FixedStateGradientProvider) -> None:
        self.provider = provider

    def run(
        self,
        plan: ReferenceSizingPlan,
        *,
        draws_a: Sequence[object],
        draws_b: Sequence[object],
        artifact_root: str | Path,
        max_new_block_pairs: int | None = None,
    ) -> ReferenceSizingResult:
        if plan.scope == "formal":
            plan.execution.require_for_stage(2)
        if max_new_block_pairs is not None and max_new_block_pairs <= 0:
            raise ValueError("max_new_block_pairs 必须为正整数或 null")
        maximum = plan.candidate_sample_counts[-1]
        if len(draws_a) < maximum or len(draws_b) < maximum:
            raise ValueError("冻结 A/B draws 少于 reference sizing 最大候选")
        if _draw_ids(draws_a[:maximum]).intersection(_draw_ids(draws_b[:maximum])):
            raise ValueError("reference A/B draw IDs 必须互不重用")
        for expected_stream, draws in (("reference_A", draws_a), ("reference_B", draws_b)):
            observed = {
                getattr(draw, "stream")
                for draw in draws[:maximum]
                if hasattr(draw, "stream")
            }
            if observed and observed != {expected_stream}:
                raise ValueError(f"{expected_stream} draws 来自错误 stream")

        provider_state = self.provider.state_digest()
        assumptions = _weighting_contract(self.provider)
        processed_pairs = 0
        points: list[ReferenceSizingPoint] = []
        streak = 0
        selected: int | None = None
        last_bias: dict[str, np.ndarray] | None = None
        moments_a, moments_b = _GradientMoments(), _GradientMoments()

        with _ReferenceSnapshotStore(artifact_root) as store:
            restored = store.latest()
            if restored is not None:
                if restored.get("plan_hash") != plan.artifact_hash:
                    raise ValueError("REFERENCE_RESUME_PLAN_HASH_MISMATCH")
                if restored.get("provider_state_digest") != provider_state:
                    raise ValueError("REFERENCE_RESUME_PROVIDER_STATE_MISMATCH")
                if restored.get("registry_hash") != self.provider.registry_hash:
                    raise ValueError("REFERENCE_RESUME_REGISTRY_HASH_MISMATCH")
                if restored.get("weighting_assumptions") != assumptions:
                    raise ValueError("REFERENCE_RESUME_WEIGHTING_CONTRACT_MISMATCH")
                processed_pairs = int(restored["processed_block_pairs"])
                streak = int(restored["convergence_streak"])
                selected_value = restored["selected_sample_count_per_stream"]
                selected = None if selected_value is None else int(selected_value)
                moments_a = _GradientMoments.from_state(restored["a"])  # type: ignore[arg-type]
                moments_b = _GradientMoments.from_state(restored["b"])  # type: ignore[arg-type]
                raw_points = restored["points"]
                if not isinstance(raw_points, list):
                    raise ValueError("REFERENCE_RESUME_POINTS_NOT_ARRAY")
                points = [ReferenceSizingPoint.from_mapping(item) for item in raw_points]
                raw_last = restored["last_bias"]
                if raw_last:
                    if not isinstance(raw_last, Mapping):
                        raise ValueError("REFERENCE_RESUME_LAST_BIAS_NOT_MAPPING")
                    last_bias = _as_vector(raw_last)

            resumed_from = processed_pairs
            new_pairs = 0
            total_pairs = maximum // plan.block_size
            while processed_pairs < total_pairs and selected is None:
                if max_new_block_pairs is not None and new_pairs >= max_new_block_pairs:
                    break
                start = processed_pairs * plan.block_size
                stop = start + plan.block_size
                batch_a = self.provider.gradient(draws_a[start:stop])
                batch_b = self.provider.gradient(draws_b[start:stop])
                moments_a.update(batch_a, assumptions)
                moments_b.update(batch_b, assumptions)
                self.provider.assert_unchanged(provider_state)
                processed_pairs += 1
                new_pairs += 1
                sample_count = processed_pairs * plan.block_size
                if sample_count in plan.candidate_sample_counts:
                    bias, cross, ranking = _reference_vectors(
                        moments_a, moments_b, assumptions
                    )
                    difference: float | None = None
                    comparison_defined = False
                    comparison_reason: str | None = "no_previous_reference"
                    if last_bias is not None:
                        # 分母使用当前更高样本量 reference，保持 sizing curve 与
                        # 后续“候选相对冻结 reference”的误差方向一致。
                        observed_difference = _normalized_l1(last_bias, bias)
                        if math.isfinite(observed_difference):
                            difference = observed_difference
                            comparison_defined = True
                            comparison_reason = None
                            streak = (
                                streak + 1
                                if difference <= plan.convergence_tolerance
                                else 0
                            )
                        else:
                            # JSON artifact 禁止 Infinity；零 reference L1 范数时将
                            # 比较显式标为未定义，而不是加 epsilon 伪造收敛。
                            streak = 0
                            comparison_reason = "zero_reference_l1_norm"
                    point = ReferenceSizingPoint(
                        sample_count_per_stream=sample_count,
                        block_count_total=moments_a.count + moments_b.count,
                        bias_reference_hash=_vector_digest(bias),
                        cross_reference_hash=_vector_digest(cross),
                        ranking_reference_hash=_vector_digest(ranking),
                        normalized_l1_from_previous=difference,
                        convergence_streak=streak,
                        comparison_defined=comparison_defined,
                        comparison_reason=comparison_reason,
                    )
                    points.append(point)
                    last_bias = bias
                    if streak >= plan.required_consecutive:
                        selected = sample_count
                state: dict[str, object] = {
                    "schema_version": "stage2-reference-progress-state-v1",
                    "plan_hash": plan.artifact_hash,
                    "provider_state_digest": provider_state,
                    "registry_hash": self.provider.registry_hash,
                    "weighting_assumptions": assumptions,
                    "processed_block_pairs": processed_pairs,
                    "convergence_streak": streak,
                    "selected_sample_count_per_stream": selected,
                    "points": [point.to_dict() for point in points],
                    "last_bias": {} if last_bias is None else last_bias,
                    "a": moments_a.to_state(),
                    "b": moments_b.to_state(),
                }
                store.publish(processed_pairs, state)

        if processed_pairs <= 0:
            raise RuntimeError("REFERENCE_SIZING_NO_COMMITTED_BLOCKS")
        bias, cross, ranking = _reference_vectors(moments_a, moments_b, assumptions)
        return ReferenceSizingResult(
            plan_hash=plan.artifact_hash,
            registry_hash=self.provider.registry_hash,
            provider_state_digest=provider_state,
            processed_sample_count_per_stream=processed_pairs * plan.block_size,
            selected_sample_count_per_stream=selected,
            converged=selected is not None,
            points=tuple(points),
            bias_reference=bias,
            cross_reference=cross,
            ranking_reference=ranking,
            resumed_from_block_pairs=resumed_from,
            scope=plan.scope,
            qualification=ArtifactQualification.candidate(plan.scope),
            weighting_assumptions=assumptions,
        )


class _WaveUnitStore:
    """paired repetition 的不可变对象/权威 commit store。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "writer.lock"
        try:
            self.lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("STAGE2_WAVE_WRITER_ALREADY_ACTIVE") from error

    def close(self) -> None:
        if getattr(self, "lock_fd", -1) >= 0:
            os.close(self.lock_fd)
            self.lock_fd = -1
            self.lock_path.unlink(missing_ok=True)

    def __enter__(self) -> "_WaveUnitStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def _scientific_digest(state: Mapping[str, object]) -> str:
        vectors = state.get("vectors")
        if not isinstance(vectors, Mapping):
            raise ValueError("WAVE_STATE_VECTORS_NOT_MAPPING")
        return canonical_json_hash(
            {
                "unit_id": state.get("unit_id"),
                "input_hash": state.get("input_hash"),
                "registry_hash": state.get("registry_hash"),
                "state_digest": state.get("state_digest"),
                "vectors": {
                    str(name): _vector_digest(value)  # type: ignore[arg-type]
                    for name, value in sorted(vectors.items())
                },
                "weighting_assumptions": state.get("weighting_assumptions"),
            }
        )

    def publish(self, state: Mapping[str, object]) -> None:
        unit_id = _require_identifier(str(state["unit_id"]), field_name="unit_id")
        digest = self._scientific_digest(state)
        object_path = self.objects / digest
        if not object_path.exists():
            bundle = publish_tensor_bundle(object_path, dict(state))
        else:
            restored, bundle = load_tensor_bundle(object_path)
            if not isinstance(restored, Mapping) or self._scientific_digest(restored) != digest:
                raise ValueError("WAVE_EXISTING_OBJECT_IDENTITY_MISMATCH")
        commit: dict[str, object] = {
            "schema_version": "stage2-wave-unit-commit-v1",
            "unit_id": unit_id,
            "input_hash": state["input_hash"],
            "scientific_digest": digest,
            "object_ref": f"objects/{digest}",
            "object_manifest_hash": bundle.manifest_sha256,
        }
        commit["artifact_hash"] = canonical_json_hash(commit)
        path = self.commits / f"{unit_id}.json"
        if path.exists():
            existing = load_canonical_json(path)
            if existing != commit:
                raise ValueError(f"WAVE_UNIT_COMMIT_CONFLICT:{unit_id}")
            return
        write_canonical_json(path, commit)

    def load_all(self) -> dict[str, Mapping[str, object]]:
        results: dict[str, Mapping[str, object]] = {}
        for path in sorted(self.commits.glob("*.json")):
            commit = load_canonical_json(path)
            if not isinstance(commit, Mapping):
                raise ValueError("WAVE_COMMIT_NOT_OBJECT")
            payload = {name: item for name, item in commit.items() if name != "artifact_hash"}
            if canonical_json_hash(payload) != commit.get("artifact_hash"):
                raise ValueError("WAVE_COMMIT_HASH_MISMATCH")
            relative = Path(str(commit["object_ref"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("WAVE_OBJECT_PATH_ESCAPE")
            state, bundle = load_tensor_bundle(self.root / relative)
            if bundle.manifest_sha256 != commit["object_manifest_hash"]:
                raise ValueError("WAVE_OBJECT_MANIFEST_HASH_MISMATCH")
            if not isinstance(state, Mapping):
                raise ValueError("WAVE_STATE_NOT_OBJECT")
            if self._scientific_digest(state) != commit["scientific_digest"]:
                raise ValueError("WAVE_SCIENTIFIC_DIGEST_MISMATCH")
            unit_id = str(commit["unit_id"])
            if unit_id in results:
                raise ValueError(f"WAVE_DUPLICATE_UNIT:{unit_id}")
            results[unit_id] = state
        return results

    def reconcile(self) -> dict[str, object]:
        committed = self.load_all()
        referenced = {
            str(load_canonical_json(path)["scientific_digest"])  # type: ignore[index]
            for path in self.commits.glob("*.json")
        }
        objects = {path.name for path in self.objects.iterdir() if path.is_dir()}
        return {
            "committed_unit_ids": sorted(committed),
            "orphan_objects": sorted(objects - referenced),
        }


def _unit_metrics(
    vectors: Mapping[str, Mapping[str, object]],
    reference: Mapping[str, object],
) -> dict[str, dict[str, float | int]]:
    expected = _flatten(reference)
    metrics: dict[str, dict[str, float | int]] = {}
    for method, vector in sorted(vectors.items()):
        _assert_compatible(vector, reference, field_name=f"wave.{method}")
        observed = _flatten(vector)
        error = observed - expected
        metrics[method] = {
            "coordinate_count": int(observed.size),
            "signed_error_sum": float(error.sum()),
            "absolute_error_sum": float(np.abs(error).sum()),
            "squared_error_sum": float(np.square(error).sum()),
            "positive_count": int(np.count_nonzero(observed > 0)),
            "zero_count": int(np.count_nonzero(observed == 0)),
            "negative_count": int(np.count_nonzero(observed < 0)),
            "positive_mass": float(observed[observed > 0].sum(initial=0.0)),
            "negative_mass": float((-observed[observed < 0]).sum(initial=0.0)),
        }
    return metrics


def _aggregate_wave(
    states: Sequence[Mapping[str, object]], reference: Mapping[str, object]
) -> tuple[dict[str, dict[str, float | int]], dict[str, dict[str, object]]]:
    by_method: dict[str, list[np.ndarray]] = {}
    gradient_evaluations = 0
    formula_seconds = 0.0
    wall_seconds = 0.0
    for state in states:
        vectors = state["vectors"]
        if not isinstance(vectors, Mapping):
            raise ValueError("WAVE_STATE_VECTORS_NOT_MAPPING")
        for method, vector in vectors.items():
            by_method.setdefault(str(method), []).append(_flatten(vector))  # type: ignore[arg-type]
        gradient_evaluations += int(state["gradient_evaluations"])
        formula_seconds += float(state["formula_seconds"])
        wall_seconds += float(state["wall_seconds"])
    expected = _flatten(reference)
    summaries: dict[str, dict[str, float | int]] = {}
    for method, values in sorted(by_method.items()):
        matrix = np.stack(values, axis=0)
        if matrix.shape[1:] != expected.shape:
            raise ValueError(f"WAVE_METHOD_REFERENCE_SHAPE_MISMATCH:{method}")
        mean = matrix.mean(axis=0)
        error = matrix - expected[None, :]
        variance = (
            np.zeros_like(mean)
            if matrix.shape[0] < 2
            else matrix.var(axis=0, ddof=1)
        )
        summaries[method] = {
            "repetitions": int(matrix.shape[0]),
            "coordinate_count": int(matrix.shape[1]),
            "bias": float((mean - expected).mean()),
            "absolute_bias": float(np.abs(mean - expected).mean()),
            "variance": float(variance.mean()),
            "mse": float(np.square(error).mean()),
            "mae": float(np.abs(error).mean()),
            "negative_fraction": float(np.count_nonzero(matrix < 0) / matrix.size),
            "positive_mass": float(matrix[matrix > 0].sum(initial=0.0)),
            "negative_mass": float((-matrix[matrix < 0]).sum(initial=0.0)),
        }
    costs = {
        "scientific_equal_sample_cost": {
            "defined": True,
            "gradient_evaluations": gradient_evaluations,
            "formula_seconds": formula_seconds,
            "wall_seconds": wall_seconds,
            "reason": None,
        },
        "isolated_estimator_cost": {
            "defined": False,
            "gradient_evaluations": None,
            "formula_seconds": None,
            "wall_seconds": None,
            "reason": "method_only_anchor_not_run",
        },
        "online_training_incremental_cost": {
            "defined": False,
            "gradient_evaluations": None,
            "formula_seconds": None,
            "wall_seconds": None,
            "reason": "online_training_adapter_not_run",
        },
    }
    return summaries, costs


@dataclass(frozen=True, slots=True)
class PairedWaveSummary:
    wave_id: str
    registry_hash: str
    reference_hash: str
    expected_unit_ids: tuple[str, ...]
    completed_unit_ids: tuple[str, ...]
    method_statistics: Mapping[str, Mapping[str, float | int]]
    cost_statistics: Mapping[str, Mapping[str, object]]
    scope: str
    resumed_unit_count: int
    weighting_assumptions: Mapping[str, object]
    schema_version: str = "stage2-paired-wave-summary-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.wave_id, field_name="wave_id")
        _require_sha256(self.registry_hash, field_name="registry_hash")
        _require_sha256(self.reference_hash, field_name="reference_hash")
        if len(set(self.expected_unit_ids)) != len(self.expected_unit_ids):
            raise ValueError("expected_unit_ids 不能重复")
        if not set(self.completed_unit_ids).issubset(self.expected_unit_ids):
            raise ValueError("completed_unit_ids 必须属于 expected_unit_ids")
        if set(self.cost_statistics) != set(_COST_SEMANTICS):
            raise ValueError("cost_statistics 必须完整记录三种成本口径")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("scope 不受支持")
        required_assumptions = {
            "statistical_unit",
            "weight_unit",
            "sampling_design",
            "weights_exogenous",
            "common_mean_assumption",
        }
        if set(self.weighting_assumptions) != required_assumptions:
            raise ValueError("PairedWaveSummary weighting assumptions 不完整")
        object.__setattr__(
            self,
            "weighting_assumptions",
            freeze_json_mapping(
                self.weighting_assumptions,
                field="PairedWaveSummary.weighting_assumptions",
            ),
        )

    @property
    def complete(self) -> bool:
        return set(self.completed_unit_ids) == set(self.expected_unit_ids)

    @property
    def status(self) -> str:
        if not self.complete:
            return "INCOMPLETE"
        return "FIXTURE_COMPLETE" if self.scope == "local_fixture" else "FORMAL_CANDIDATE"

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "wave_id": self.wave_id,
            "registry_hash": self.registry_hash,
            "reference_hash": self.reference_hash,
            "expected_unit_ids": list(self.expected_unit_ids),
            "completed_unit_ids": list(self.completed_unit_ids),
            "complete": self.complete,
            "status": self.status,
            "method_statistics": {
                name: dict(values) for name, values in sorted(self.method_statistics.items())
            },
            "cost_statistics": {
                name: dict(values) for name, values in sorted(self.cost_statistics.items())
            },
            "scope": self.scope,
            "formal_eligible": False,
            "qualification_gate_hash": None,
            "resumed_unit_count": self.resumed_unit_count,
            "weighting_assumptions": thaw_json_value(self.weighting_assumptions),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}


class RecoverablePairedWaveRunner:
    """逐 repetition 运行 paired estimators，并只归并权威 commits。"""

    def __init__(
        self,
        provider: FixedStateGradientProvider,
        *,
        execution: FormalExecutionEvidence | None = None,
        m2_tolerance: float = 1e-10,
    ) -> None:
        self.provider = provider
        self.execution = execution or FormalExecutionEvidence("local_fixture")
        if self.execution.run_intent == "formal":
            self.execution.require_for_stage(2)
        self.runner = PairedEstimatorRunner(provider, m2_tolerance=m2_tolerance)

    def run(
        self,
        *,
        wave_id: str,
        mappings: Sequence[RepetitionMapping],
        reference: Mapping[str, object],
        reference_hash: str,
        artifact_root: str | Path,
        max_new_units: int | None = None,
    ) -> PairedWaveSummary:
        _require_identifier(wave_id, field_name="wave_id")
        _require_sha256(reference_hash, field_name="reference_hash")
        if reference_hash != _vector_digest(reference):
            raise ValueError("WAVE_REFERENCE_HASH_MISMATCH")
        if not mappings:
            raise ValueError("paired wave 至少需要一个 repetition mapping")
        if max_new_units is not None and max_new_units <= 0:
            raise ValueError("max_new_units 必须为正整数或 null")
        ordered = tuple(sorted(mappings, key=lambda item: item.repetition_id))
        expected = tuple(mapping.repetition_id for mapping in ordered)
        if len(set(expected)) != len(expected):
            raise ValueError("repetition_id 不能重复")
        _as_vector(reference, field_name="reference")
        root = Path(artifact_root)
        root.mkdir(parents=True, exist_ok=True)
        wave_plan = {
            "schema_version": "stage2-paired-wave-plan-v1",
            "wave_id": wave_id,
            "reference_hash": reference_hash,
            "registry_hash": self.provider.registry_hash,
            "provider_state_digest": self.provider.state_digest(),
            "execution_evidence_hash": self.execution.artifact_hash,
            "weighting_assumptions": _weighting_contract(self.provider),
            "mappings": [
                {"unit_id": mapping.repetition_id, "mapping_hash": mapping.digest}
                for mapping in ordered
            ],
        }
        wave_plan["artifact_hash"] = canonical_json_hash(wave_plan)
        plan_path = root / "wave-plan.json"
        if plan_path.exists():
            if load_canonical_json(plan_path) != wave_plan:
                raise ValueError("WAVE_ROOT_PLAN_CONFLICT")
        else:
            write_canonical_json(plan_path, wave_plan)
        resumed_count = 0
        new_count = 0
        with _WaveUnitStore(root) as store:
            existing = store.load_all()
            resumed_count = len(set(existing).intersection(expected))
            for mapping in ordered:
                committed = existing.get(mapping.repetition_id)
                if committed is not None:
                    if committed.get("input_hash") != mapping.digest:
                        raise ValueError(
                            f"WAVE_RESUME_MAPPING_HASH_MISMATCH:{mapping.repetition_id}"
                        )
                    continue
                if max_new_units is not None and new_count >= max_new_units:
                    break
                started = time.perf_counter()
                result = self.runner.run(mapping)
                elapsed = time.perf_counter() - started
                vectors = {
                    name: _as_vector(value, field_name=f"wave.{name}")
                    for name, value in result.vectors.items()
                }
                state: dict[str, object] = {
                    "schema_version": "stage2-wave-unit-state-v1",
                    "unit_id": mapping.repetition_id,
                    "input_hash": mapping.digest,
                    "registry_hash": result.registry_hash,
                    "state_digest": result.state_digest,
                    "vectors": vectors,
                    "metrics": _unit_metrics(vectors, reference),
                    "gradient_evaluations": result.gradient_evaluations,
                    "formula_seconds": result.formula_seconds,
                    "wall_seconds": elapsed,
                    "sample_collision_count": result.sample_collision_count,
                    "m2_double_max_abs_error": result.m2_double_max_abs_error,
                    "weighting_assumptions": thaw_json_value(
                        result.weighting_assumptions
                    ),
                }
                store.publish(state)
                new_count += 1
            states_by_id = store.load_all()
            selected_states = [states_by_id[unit] for unit in expected if unit in states_by_id]
        if selected_states:
            registries = {str(state["registry_hash"]) for state in selected_states}
            if registries != {self.provider.registry_hash}:
                raise ValueError("WAVE_REGISTRY_HASH_DRIFT")
            assumption_hashes = {
                canonical_json_hash(state["weighting_assumptions"])
                for state in selected_states
            }
            if assumption_hashes != {
                canonical_json_hash(_weighting_contract(self.provider))
            }:
                raise ValueError("WAVE_WEIGHTING_CONTRACT_DRIFT")
            method_statistics, costs = _aggregate_wave(selected_states, reference)
        else:
            method_statistics = {}
            costs = {
                name: {
                    "defined": False,
                    "gradient_evaluations": None,
                    "formula_seconds": None,
                    "wall_seconds": None,
                    "reason": "no_committed_units",
                }
                for name in _COST_SEMANTICS
            }
        return PairedWaveSummary(
            wave_id=wave_id,
            registry_hash=self.provider.registry_hash,
            reference_hash=reference_hash,
            expected_unit_ids=expected,
            completed_unit_ids=tuple(unit for unit in expected if unit in states_by_id),
            method_statistics=MappingProxyType(method_statistics),
            cost_statistics=MappingProxyType(costs),
            scope=self.execution.run_intent,
            resumed_unit_count=resumed_count,
            weighting_assumptions=_weighting_contract(self.provider),
        )


@dataclass(frozen=True, slots=True)
class PilotThresholds:
    """pilot 前冻结的等价、非劣与在线增量成本阈值。"""

    bias_margin: float
    max_corrected_nmse_ratio: float
    min_spearman: float
    min_topk_overlap: float
    max_online_cost_ratio: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.bias_margin) or self.bias_margin < 0:
            raise ValueError("bias_margin 必须是非负有限数")
        for name in ("max_corrected_nmse_ratio", "max_online_cost_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        for name in ("min_spearman", "min_topk_overlap"):
            value = float(getattr(self, name))
            lower = -1.0 if name == "min_spearman" else 0.0
            if not math.isfinite(value) or not lower <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [{lower:g},1]")

    def to_dict(self) -> dict[str, float]:
        return {
            "bias_margin": self.bias_margin,
            "max_corrected_nmse_ratio": self.max_corrected_nmse_ratio,
            "min_spearman": self.min_spearman,
            "min_topk_overlap": self.min_topk_overlap,
            "max_online_cost_ratio": self.max_online_cost_ratio,
        }


@dataclass(frozen=True, slots=True)
class PilotCellObservation:
    cell_id: str
    estimator: str
    batch_size: int
    microbatch_count: int
    repetitions: int
    bias_interval_low: float
    bias_interval_high: float
    corrected_nmse_ratio: float
    spearman: float
    topk_overlap: float
    online_cost_ratio: float
    quality_complete: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.cell_id, field_name="cell_id")
        if self.estimator not in {"u", "weighted_u", "double"}:
            raise ValueError("estimator 不受支持")
        if self.batch_size <= 0 or self.microbatch_count < 2 or self.repetitions <= 0:
            raise ValueError("B/M/R 必须严格为正且 M>=2")
        if self.batch_size % self.microbatch_count:
            raise ValueError("microbatch_count 必须整除 batch_size")
        numeric = (
            self.bias_interval_low,
            self.bias_interval_high,
            self.corrected_nmse_ratio,
            self.spearman,
            self.topk_overlap,
            self.online_cost_ratio,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("PilotCellObservation 指标必须全部有限")
        if self.bias_interval_low > self.bias_interval_high:
            raise ValueError("bias interval 下界不能大于上界")
        if self.corrected_nmse_ratio < 0 or self.online_cost_ratio < 0:
            raise ValueError("NMSE/cost ratio 不能为负")
        if not -1 <= self.spearman <= 1 or not 0 <= self.topk_overlap <= 1:
            raise ValueError("spearman/topk_overlap 超出定义域")


@dataclass(frozen=True, slots=True)
class Stage2EstimatorRecommendation:
    recommendation_id: str
    status: str
    selected_estimator: str | None
    batch_size: int | None
    microbatch_count: int | None
    repetitions: int | None
    required_cells: tuple[str, ...]
    qualified_estimators: tuple[str, ...]
    thresholds: PilotThresholds
    scope: str
    execution_evidence_hash: str
    reasons: tuple[str, ...] = ()
    schema_version: str = "stage2-estimator-recommendation-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.recommendation_id, field_name="recommendation_id")
        if self.status not in {"FIXTURE_RECOMMENDATION", "FORMAL_CANDIDATE", "BLOCKED"}:
            raise ValueError("recommendation status 不受支持")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("scope 不受支持")
        if self.status == "BLOCKED":
            if self.selected_estimator is not None:
                raise ValueError("BLOCKED recommendation 不得选择 estimator")
        elif self.selected_estimator not in {"u", "weighted_u", "double"}:
            raise ValueError("非 BLOCKED recommendation 必须选择 estimator")
        _require_sha256(self.execution_evidence_hash, field_name="execution_evidence_hash")

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recommendation_id": self.recommendation_id,
            "status": self.status,
            "selected_estimator": self.selected_estimator,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "required_cells": list(self.required_cells),
            "qualified_estimators": list(self.qualified_estimators),
            "thresholds": self.thresholds.to_dict(),
            "scope": self.scope,
            "formal_eligible": False,
            "qualification_gate_hash": None,
            "execution_evidence_hash": self.execution_evidence_hash,
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def qualify(
        self,
        *,
        execution: FormalExecutionEvidence,
        gate: GateRecord,
        artifact_ref: str,
    ) -> EstimatorDecision:
        """把 formal 候选与 Stage 2 Gate 绑定为现有公共 EstimatorDecision。"""

        execution.require_for_stage(2)
        if self.scope != "formal" or self.status != "FORMAL_CANDIDATE":
            raise FormalRunRejected("STAGE2_RECOMMENDATION_NOT_FORMAL_CANDIDATE")
        if execution.artifact_hash != self.execution_evidence_hash:
            raise FormalRunRejected("STAGE2_RECOMMENDATION_EXECUTION_EVIDENCE_MISMATCH")
        accepted = require_accepted_gate(gate, stage=2)
        if not artifact_ref or artifact_ref.startswith(("/", "\\")) or ".." in Path(artifact_ref).parts:
            raise ValueError("artifact_ref 必须是无路径逃逸的相对引用")
        _require_gate_binding(accepted, artifact_ref)
        assert self.selected_estimator is not None
        payload: dict[str, object] = {
            "schema_version": "estimator-decision-v1",
            "decision_id": self.recommendation_id,
            "selected_estimator": self.selected_estimator,
            "scope": "formal",
            # 决策本身表示“已被 Gate 资格化”，不把 CONDITIONALLY_ACCEPTED
            # 擅自改写成无条件 PASS；真实 Gate 状态保留在 gate_status 字段。
            "status": "QUALIFIED",
            "state": "FROZEN",
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "gate_id": accepted.gate_id,
            "gate_status": accepted.status.value,
            "artifact_ref": artifact_ref,
            "metadata": {
                "formal_eligible": True,
                "recommendation_hash": self.artifact_hash,
                "qualification_gate_hash": accepted.artifact_hash,
                "execution_evidence_hash": execution.artifact_hash,
            },
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        decision = EstimatorDecision.from_mapping(payload)
        decision.require_formal()
        return decision


class Stage2RecommendationEngine:
    """按冻结 cell 集合执行 intersection-union，再按固定 estimator 顺序选择。"""

    def recommend(
        self,
        *,
        recommendation_id: str,
        observations: Sequence[PilotCellObservation],
        required_cells: Sequence[str],
        thresholds: PilotThresholds,
        execution: FormalExecutionEvidence | None = None,
        estimator_preference: Sequence[str] = ("u", "double", "weighted_u"),
    ) -> Stage2EstimatorRecommendation:
        execution = execution or FormalExecutionEvidence("local_fixture")
        if execution.run_intent == "formal":
            execution.require_for_stage(2)
        cells = tuple(required_cells)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("required_cells 必须非空且无重复")
        if not observations:
            raise ValueError("observations 不能为空")
        preference = tuple(estimator_preference)
        if len(set(preference)) != len(preference) or any(
            item not in {"u", "weighted_u", "double"} for item in preference
        ):
            raise ValueError("estimator_preference 非法")
        by_estimator: dict[str, dict[str, PilotCellObservation]] = {}
        for observation in observations:
            bucket = by_estimator.setdefault(observation.estimator, {})
            if observation.cell_id in bucket:
                raise ValueError(
                    f"DUPLICATE_PILOT_CELL:{observation.estimator}:{observation.cell_id}"
                )
            bucket[observation.cell_id] = observation

        qualified: list[str] = []
        reasons: list[str] = []
        for estimator in preference:
            bucket = by_estimator.get(estimator, {})
            missing = sorted(set(cells) - set(bucket))
            if missing:
                reasons.append(f"{estimator}:missing_cells={','.join(missing)}")
                continue
            passed = True
            for cell in cells:
                item = bucket[cell]
                checks = (
                    item.quality_complete,
                    item.bias_interval_low >= -thresholds.bias_margin,
                    item.bias_interval_high <= thresholds.bias_margin,
                    item.corrected_nmse_ratio <= thresholds.max_corrected_nmse_ratio,
                    item.spearman >= thresholds.min_spearman,
                    item.topk_overlap >= thresholds.min_topk_overlap,
                    item.online_cost_ratio <= thresholds.max_online_cost_ratio,
                )
                if not all(checks):
                    passed = False
                    reasons.append(f"{estimator}:{cell}:threshold_failed")
            if passed:
                qualified.append(estimator)

        selected = next((name for name in preference if name in qualified), None)
        exemplar: PilotCellObservation | None = None
        if selected is not None:
            selected_rows = [by_estimator[selected][cell] for cell in cells]
            identities = {
                (row.batch_size, row.microbatch_count, row.repetitions)
                for row in selected_rows
            }
            if len(identities) != 1:
                reasons.append(f"{selected}:B_M_R_not_unique_across_cells")
                selected = None
            else:
                exemplar = selected_rows[0]
        status = "BLOCKED"
        if selected is not None:
            status = (
                "FIXTURE_RECOMMENDATION"
                if execution.run_intent == "local_fixture"
                else "FORMAL_CANDIDATE"
            )
        return Stage2EstimatorRecommendation(
            recommendation_id=recommendation_id,
            status=status,
            selected_estimator=selected,
            batch_size=None if exemplar is None else exemplar.batch_size,
            microbatch_count=None if exemplar is None else exemplar.microbatch_count,
            repetitions=None if exemplar is None else exemplar.repetitions,
            required_cells=cells,
            qualified_estimators=tuple(qualified),
            thresholds=thresholds,
            scope=execution.run_intent,
            execution_evidence_hash=execution.artifact_hash,
            reasons=tuple(reasons),
        )


__all__ = [
    "FormalExperimentPlan",
    "PairedWaveSummary",
    "PilotCellObservation",
    "PilotThresholds",
    "RecoverablePairedWaveRunner",
    "ReferenceSizingPlan",
    "ReferenceSizingPoint",
    "ReferenceSizingResult",
    "Stage2EstimatorRecommendation",
    "Stage2RecommendationEngine",
    "StreamingReferenceSizer",
]
