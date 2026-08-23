"""Stage 2 S2.6 pilot, sizing and matrix-freeze contracts.

This module is deliberately independent from the model runner.  It consumes only
the immutable sampling/repetition objects and measurements emitted by S2.4/S2.5.
The formal constructor is fail-closed: local fixture evidence can never be
promoted to a G2.4b matrix or to a confirmatory formal mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..contracts.errors import FormalRunRejected
from ..contracts.jsonio import canonical_json_hash
from ..contracts.stage23 import FormalExecutionEvidence
from .sampling import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    MICROBATCH_SELECTION_ORDER,
    Draw,
    RepetitionMapping,
    SamplingPlan,
)


PILOT_MATRIX_SCHEMA = "stage2-pilot-matrix-freeze-v1"
CONFIRMATORY_MAPPING_SCHEMA = "stage2-confirmatory-mapping-v1"
COST_SEMANTICS = (
    "scientific_equal_sample_cost",
    "isolated_estimator_cost",
    "online_training_incremental_cost",
)
ANCHOR_IDS = tuple(
    f"{model}.{stage}"
    for model in ("pythia-14m", "pythia-31m-deduped")
    for stage in ("initialization", "early", "mid_late")
)


def _id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160:
        raise ValueError(f"{field} 必须是非空安全标识")
    if any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for ch in value):
        raise ValueError(f"{field} 含非法字符")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class CostSemantics:
    """冻结三种成本含义；``cost_io_quiescent`` 缺失时成本不可用。"""

    scientific_equal_sample_cost: Mapping[str, object]
    isolated_estimator_cost: Mapping[str, object]
    online_training_incremental_cost: Mapping[str, object]
    cost_io_quiescent: bool

    def __post_init__(self) -> None:
        for name in COST_SEMANTICS:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} 必须是 object")
            if "defined" not in value or type(value["defined"]) is not bool:
                raise ValueError(f"{name}.defined 必须显式声明")
        if type(self.cost_io_quiescent) is not bool:
            raise TypeError("cost_io_quiescent 必须是 bool")

    def to_dict(self) -> dict[str, object]:
        return {name: dict(getattr(self, name)) for name in COST_SEMANTICS} | {
            "cost_io_quiescent": self.cost_io_quiescent
        }


@dataclass(frozen=True, slots=True)
class ArtificialCalibrationReport:
    """人工梯度分布 calibration，数值字段不参与候选选择。"""

    batch_size: int
    m_values: tuple[int, ...]
    raw_bias_by_m: Mapping[int, float]
    u_mean_by_m: Mapping[int, float]
    double_mean: float
    m2_max_abs_error: float
    variance_by_m: Mapping[int, float]
    weighted_u_max_abs_error: float
    status: str = "PASS"
    scope: str = "local_fixture"

    def __post_init__(self) -> None:
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("人工校准 B 必须是偶数且不小于 2")
        values = tuple(sorted(set(self.m_values)))
        if not values or any(m < 2 or self.batch_size % m for m in values):
            raise ValueError("人工校准 m_values 必须是 B 的合法因子")
        object.__setattr__(self, "m_values", values)
        for mapping in (self.raw_bias_by_m, self.u_mean_by_m, self.variance_by_m):
            if set(mapping) != set(values) or any(not math.isfinite(float(v)) for v in mapping.values()):
                raise ValueError("人工校准指标必须覆盖全部 M 且有限")
        if not math.isfinite(self.double_mean) or not math.isfinite(self.m2_max_abs_error) or not math.isfinite(self.weighted_u_max_abs_error):
            raise ValueError("人工校准标量必须有限")
        if self.scope != "local_fixture":
            raise FormalRunRejected("人工校准只能是 local_fixture")

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "stage2-artificial-calibration-v1",
            "batch_size": self.batch_size,
            "m_values": list(self.m_values),
            "raw_bias_by_m": {str(k): float(v) for k, v in self.raw_bias_by_m.items()},
            "u_mean_by_m": {str(k): float(v) for k, v in self.u_mean_by_m.items()},
            "double_mean": self.double_mean,
            "m2_max_abs_error": self.m2_max_abs_error,
            "variance_by_m": {str(k): float(v) for k, v in self.variance_by_m.items()},
            "weighted_u_max_abs_error": self.weighted_u_max_abs_error,
            "status": self.status,
            "scope": self.scope,
            "formal_eligible": False,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def _u(groups: np.ndarray) -> np.ndarray:
    count = groups.shape[0]
    if count < 2:
        raise ValueError("U-statistic 至少需要两个 microbatch")
    return (np.square(groups.sum(axis=0)) - np.square(groups).sum(axis=0)) / (count * (count - 1))


def _weighted_u(groups: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 1 or len(weights) != len(groups) or np.any(weights <= 0):
        raise ValueError("weighted U 权重必须是有限正数")
    denominator = np.square(weights.sum()) - np.square(weights).sum()
    if denominator <= 0:
        raise ValueError("weighted U 分母必须为正")
    return ((weights[:, None] * groups).sum(axis=0) ** 2 - ((weights**2)[:, None] * groups**2).sum(axis=0)) / denominator


def run_artificial_distribution_calibration(
    samples: Sequence[Sequence[float]] | np.ndarray,
    *,
    batch_size: int | None = None,
    m_values: Iterable[int] = (2, 4, 8, 16, 32),
    mean: float | None = None,
    variance: float | None = None,
) -> ArtificialCalibrationReport:
    """Run finite Gaussian/discrete calibration without touching model state.

    ``samples`` may contain several consecutive B-sized repetitions.  All M use
    the same rows, and M=2 compares with the exact two-half double mapping.
    """

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("人工样本必须是有限二维数组")
    b = int(batch_size or array.shape[0])
    if b < 2 or b % 2 or array.shape[0] % b:
        raise ValueError("batch_size 必须整除样本数且为偶数")
    values = tuple(sorted(set(int(m) for m in m_values)))
    if not values or any(m < 2 or b % m for m in values):
        raise ValueError("m_values 必须是 batch_size 的合法因子")
    repetitions = array.reshape(array.shape[0] // b, b, -1)
    raw_bias: dict[int, float] = {}
    u_means: dict[int, float] = {}
    variances: dict[int, float] = {}
    double_values: list[float] = []
    u2_values: list[np.ndarray] = []
    for m in values:
        u_rows: list[np.ndarray] = []
        raw_rows: list[np.ndarray] = []
        for block in repetitions:
            groups = block.reshape(m, b // m, -1).mean(axis=1)
            u_rows.append(_u(groups))
            raw_rows.append(np.square(block.mean(axis=0)))
            if m == 2:
                double_values.append(float(np.prod(groups, axis=0).mean()))
        u_array = np.asarray(u_rows)
        raw_array = np.asarray(raw_rows)
        u_means[m] = float(u_array.mean())
        u2_values.append(u_array.reshape(-1))
        target = float(variance / b if variance is not None else 0.0)
        raw_bias[m] = float(raw_array.mean() - u_means[m] - target)
        variances[m] = float(np.var(u_array, ddof=1) if u_array.size > 1 else 0.0)
    m2_error = abs(u_means[2] - (float(np.mean(double_values)) if double_values else u_means[2]))
    groups = repetitions[0].reshape(values[-1], b // values[-1], -1).mean(axis=1)
    weights = np.arange(1, len(groups) + 1, dtype=np.float64)
    weighted_expected = _weighted_u(groups, weights)
    denominator = np.square(weights.sum()) - np.square(weights).sum()
    weighted_direct = np.zeros_like(weighted_expected)
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            weighted_direct += 2.0 * weights[left] * weights[right] * groups[left] * groups[right]
    weighted_direct /= denominator
    return ArtificialCalibrationReport(
        batch_size=b,
        m_values=values,
        raw_bias_by_m=raw_bias,
        u_mean_by_m=u_means,
        double_mean=float(np.mean(double_values)) if double_values else u_means[2],
        m2_max_abs_error=float(m2_error),
        variance_by_m=variances,
        weighted_u_max_abs_error=float(np.max(np.abs(weighted_expected - weighted_direct))),
    )


@dataclass(frozen=True, slots=True)
class AnchorPilotResult:
    """盲化 anchor × candidate 运行摘要；不保存方法均值、方向或显著性。"""

    anchor_id: str
    batch_size: int
    microbatch_count: int
    repetitions: int
    anchors_runnable: bool
    finite: bool
    aggregation_overhead_ratio: float
    required_repetitions: int
    storage_bytes: int
    gpu_hours: float
    resource_within_budget: bool
    cost_io_quiescent: bool

    def __post_init__(self) -> None:
        _id(self.anchor_id, "anchor_id")
        if self.batch_size not in CANDIDATE_BATCH_SIZES or self.microbatch_count not in CANDIDATE_MICROBATCH_COUNTS or self.batch_size % self.microbatch_count:
            raise ValueError("anchor B/M 不在预注册网格或不能整除")
        if self.repetitions < 1 or self.required_repetitions < 1 or self.storage_bytes < 0 or self.gpu_hours < 0:
            raise ValueError("anchor 计数/资源字段非法")
        if not 0 <= self.aggregation_overhead_ratio <= 1 or not math.isfinite(self.aggregation_overhead_ratio):
            raise ValueError("aggregation_overhead_ratio 必须位于 [0,1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "anchors_runnable": self.anchors_runnable,
            "finite": self.finite,
            "aggregation_overhead_ratio": self.aggregation_overhead_ratio,
            "required_repetitions": self.required_repetitions,
            "storage_bytes": self.storage_bytes,
            "gpu_hours": self.gpu_hours,
            "resource_within_budget": self.resource_within_budget,
            "cost_io_quiescent": self.cost_io_quiescent,
        }


def required_repetitions(
    *,
    estimator_variance: float,
    delta_sci: float,
    reference_half_width: float = 0.0,
    confidence_z: float = 1.96,
    minimum: int = 200,
    maximum: int = 1000,
) -> int:
    """Sizing formula using only variance/precision inputs, never method means."""

    for name, value in (("estimator_variance", estimator_variance), ("delta_sci", delta_sci), ("reference_half_width", reference_half_width), ("confidence_z", confidence_z)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} 必须是有限非负数")
    if delta_sci <= 0 or reference_half_width > delta_sci / 4:
        raise ValueError("delta_sci 必须为正且 reference 半宽需满足 h_ref<=delta_sci/4")
    if minimum < 1 or maximum < minimum:
        raise ValueError("R 上下限非法")
    # Reference uncertainty consumes the precision budget directly.  The
    # prerequisite h_ref <= delta/4 guarantees this margin is positive while
    # retaining the intended dependence of R on the one-shot reference width.
    available = delta_sci - reference_half_width
    raw = math.ceil((confidence_z**2 * estimator_variance) / (available**2))
    return min(maximum, max(minimum, raw))


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    batch_size: int
    microbatch_count: int
    m_candidate: bool
    anchors_runnable: bool
    finite: bool
    aggregation_overhead_ratio: float
    r_required: int
    resource_within_budget: bool
    selected: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "m_candidate": self.m_candidate,
            "anchors_runnable": self.anchors_runnable,
            "finite": self.finite,
            "aggregation_overhead_ratio": self.aggregation_overhead_ratio,
            "r_required": self.r_required,
            "resource_within_budget": self.resource_within_budget,
            "selected": self.selected,
            "reasons": list(self.reasons),
        }


def scan_candidates(
    observations: Sequence[AnchorPilotResult],
    *,
    required_anchor_ids: Sequence[str] = ANCHOR_IDS,
    r_max: int = 1000,
    overhead_limit: float = 0.25,
) -> tuple[CandidateEvaluation, ...]:
    """Apply preregistered M then B ordering using blinded operational fields."""

    if r_max < 1 or overhead_limit < 0:
        raise ValueError("r_max/overhead_limit 非法")
    required = tuple(required_anchor_ids)
    if not required or len(set(required)) != len(required):
        raise ValueError("required_anchor_ids 必须非空且唯一")
    by_pair: dict[tuple[int, int], list[AnchorPilotResult]] = {}
    for item in observations:
        by_pair.setdefault((item.batch_size, item.microbatch_count), []).append(item)
    output: list[CandidateEvaluation] = []
    chosen_m: dict[int, int] = {}
    for b in CANDIDATE_BATCH_SIZES:
        for m in MICROBATCH_SELECTION_ORDER:
            reasons: list[str] = []
            rows = by_pair.get((b, m), [])
            if b % m:
                reasons.append("M_DOES_NOT_DIVIDE_B")
            if not rows:
                reasons.append("OBSERVATION_MISSING")
            else:
                seen = {row.anchor_id for row in rows}
                if set(required) != seen:
                    reasons.append("SIX_ANCHORS_INCOMPLETE")
                if not all(row.anchors_runnable for row in rows):
                    reasons.append("ANCHOR_NOT_RUNNABLE")
                if not all(row.finite for row in rows):
                    reasons.append("NON_FINITE")
                if max(row.aggregation_overhead_ratio for row in rows) > overhead_limit:
                    reasons.append("AGGREGATION_OVERHEAD_EXCEEDED")
            m_candidate = not reasons and b not in chosen_m
            if m_candidate:
                chosen_m[b] = m
                if max(row.required_repetitions for row in rows) > r_max:
                    reasons.append("R_REQUIRED_EXCEEDED")
                if not all(row.resource_within_budget for row in rows):
                    reasons.append("RESOURCE_BUDGET_EXCEEDED")
                if not all(row.cost_io_quiescent for row in rows):
                    reasons.append("COST_IO_NOT_QUIESCENT")
            elif not reasons and b in chosen_m:
                reasons.append("LOWER_PRIORITY_THAN_SELECTED_M")
            output.append(CandidateEvaluation(
                b, m, m_candidate, not any(x in reasons for x in ("ANCHOR_NOT_RUNNABLE", "SIX_ANCHORS_INCOMPLETE")),
                not "NON_FINITE" in reasons,
                max((row.aggregation_overhead_ratio for row in rows), default=1.0),
                max((row.required_repetitions for row in rows), default=r_max + 1),
                all(row.resource_within_budget for row in rows) if rows else False,
                False,
                tuple(reasons),
            ))
    selected_pair: tuple[int, int] | None = None
    for b in CANDIDATE_BATCH_SIZES:
        m = chosen_m.get(b)
        if m is None:
            continue
        row = next(item for item in output if item.batch_size == b and item.microbatch_count == m)
        if not row.reasons:
            selected_pair = (b, m)
            break
    if selected_pair is not None:
        output = [CandidateEvaluation(**(row.to_dict() | {"selected": (row.batch_size, row.microbatch_count) == selected_pair, "reasons": tuple(row.reasons)})) for row in output]
    return tuple(output)


@dataclass(frozen=True, slots=True)
class MatrixFreeze:
    freeze_id: str
    scope: str
    status: str
    anchor_ids: tuple[str, ...]
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    b_primary: int | None
    m_primary: int | None
    r_primary: int | None
    completion_denominator: int
    cost_semantics: CostSemantics
    pilot_draw_stream: str = "pilot"
    confirmatory_draw_stream: str = "confirmatory"

    def __post_init__(self) -> None:
        _id(self.freeze_id, "freeze_id")
        if self.scope not in {"local_fixture", "formal"} or self.status not in {"UNFROZEN", "FIXTURE_FROZEN", "FORMAL_FROZEN", "BLOCKED"}:
            raise ValueError("matrix scope/status 非法")
        if not self.anchor_ids or len(set(self.anchor_ids)) != len(self.anchor_ids):
            raise ValueError("matrix anchor_ids 必须非空唯一")
        if self.scope == "local_fixture" and self.status == "FORMAL_FROZEN":
            raise FormalRunRejected("LOCAL_FIXTURE_CANNOT_BE_FORMAL_FROZEN")
        if self.status in {"FIXTURE_FROZEN", "FORMAL_FROZEN"}:
            if self.b_primary not in CANDIDATE_BATCH_SIZES or self.m_primary not in (4, 8, 16, 32) or self.r_primary is None:
                raise ValueError("冻结矩阵必须包含 B/M/R")
            if self.completion_denominator <= 0:
                raise ValueError("冻结矩阵完成分母必须为正")
        if self.completion_denominator < 0:
            raise ValueError("completion_denominator 不能为负")

    @property
    def formal_eligible(self) -> bool:
        return False

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": PILOT_MATRIX_SCHEMA,
            "freeze_id": self.freeze_id,
            "scope": self.scope,
            "status": self.status,
            "anchor_ids": list(self.anchor_ids),
            "candidate_evaluations": [item.to_dict() for item in self.candidate_evaluations],
            "b_primary": self.b_primary,
            "m_primary": self.m_primary,
            "r_primary": self.r_primary,
            "completion_denominator": self.completion_denominator,
            "cost_semantics": list(COST_SEMANTICS),
            "cost_observations": {
                name: dict(getattr(self.cost_semantics, name)) for name in COST_SEMANTICS
            },
            "cost_io_quiescent": self.cost_semantics.cost_io_quiescent,
            "pilot_draw_stream": self.pilot_draw_stream,
            "confirmatory_draw_stream": self.confirmatory_draw_stream,
            "formal_eligible": False,
            "qualification_gate_hash": None,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def freeze_fixture_matrix(
    observations: Sequence[AnchorPilotResult],
    *,
    freeze_id: str = "fixture-stage2-s206",
    required_anchor_ids: Sequence[str] = ANCHOR_IDS,
    r_max: int = 1000,
    cost_semantics: CostSemantics | None = None,
) -> MatrixFreeze:
    evaluations = scan_candidates(observations, required_anchor_ids=required_anchor_ids, r_max=r_max)
    selected = next((row for row in evaluations if row.selected), None)
    costs = cost_semantics or CostSemantics(
        **{name: {"defined": False, "reason": "local_fixture_timing_not_formal"} for name in COST_SEMANTICS},
        cost_io_quiescent=False,
    )
    return MatrixFreeze(
        freeze_id=freeze_id,
        scope="local_fixture",
        status="FIXTURE_FROZEN" if selected else "BLOCKED",
        anchor_ids=tuple(required_anchor_ids),
        candidate_evaluations=evaluations,
        b_primary=None if selected is None else selected.batch_size,
        m_primary=None if selected is None else selected.microbatch_count,
        r_primary=None if selected is None else selected.r_required,
        completion_denominator=0 if selected is None else int(selected.r_required * len(tuple(required_anchor_ids))),
        cost_semantics=costs,
    )


def freeze_formal_matrix(*args: object, execution: FormalExecutionEvidence, **kwargs: object) -> MatrixFreeze:
    """Formal matrix requires current accepted G2.3/G2.4a evidence."""

    if not isinstance(execution, FormalExecutionEvidence):
        raise FormalRunRejected("formal matrix requires FormalExecutionEvidence")
    execution.require_for_stage(2)
    raise FormalRunRejected("G2.4b formal matrix requires external authorization and is not generated by local S2.6")


def _mapping_draws(mappings: Sequence[RepetitionMapping]) -> tuple[Draw, ...]:
    return tuple(draw for mapping in mappings for draw in mapping.draws)


@dataclass(frozen=True, slots=True)
class ConfirmatoryMappingManifest:
    mapping_id: str
    freeze_hash: str
    sampling_plan_hash: str
    pilot_draw_ids: tuple[str, ...]
    mappings: tuple[RepetitionMapping, ...]
    scope: str = "local_fixture"
    stream: str = "confirmatory"

    def __post_init__(self) -> None:
        _id(self.mapping_id, "mapping_id")
        _hash(self.freeze_hash, "freeze_hash")
        _hash(self.sampling_plan_hash, "sampling_plan_hash")
        if self.scope not in {"local_fixture", "formal"} or self.stream != "confirmatory":
            raise ValueError("mapping scope/stream 非法")
        if self.scope == "formal":
            raise FormalRunRejected("formal confirmatory mapping requires external G2.4b authorization")
        if not self.mappings or len({item.repetition_id for item in self.mappings}) != len(self.mappings):
            raise ValueError("confirmatory mappings 必须非空且 repetition_id 唯一")
        pilot_ids = set(self.pilot_draw_ids)
        draws = _mapping_draws(self.mappings)
        if len({draw.draw_id for draw in draws}) != len(draws):
            raise ValueError("confirmatory draw ID 必须全局唯一")
        if pilot_ids.intersection(draw.draw_id for draw in draws):
            raise ValueError("pilot/confirmatory draw ID namespace 冲突")
        if any(draw.stream != "confirmatory" for draw in draws):
            raise ValueError("mapping 必须消费 confirmatory stream")

    @property
    def sample_id_collision_count(self) -> int:
        draws = _mapping_draws(self.mappings)
        return len(draws) - len({draw.sample_id for draw in draws})

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": CONFIRMATORY_MAPPING_SCHEMA,
            "mapping_id": self.mapping_id,
            "scope": self.scope,
            "stream": self.stream,
            "freeze_hash": self.freeze_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "pilot_draw_ids": list(self.pilot_draw_ids),
            "confirmatory_draw_ids": [draw.draw_id for draw in _mapping_draws(self.mappings)],
            "mappings": [
                {
                    "repetition_id": mapping.repetition_id,
                    "batch_size": mapping.batch_size,
                    "m_values": list(mapping.m_values),
                    "mapping_hash": mapping.digest,
                    "draw_ids": [draw.draw_id for draw in mapping.draws],
                    "double_half_draw_ids": [list(half) for half in mapping.to_manifest()["double_half_draw_ids"]],
                }
                for mapping in self.mappings
            ],
            "draw_id_unique": True,
            "sample_id_collision_count": self.sample_id_collision_count,
            "complete": True,
            "formal_eligible": False,
            "qualification_gate_hash": None,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value


def build_confirmatory_mapping(
    matrix: MatrixFreeze,
    sampling: SamplingPlan,
    *,
    pilot_draw_ids: Sequence[str],
    mapping_id: str = "fixture-confirmatory-mapping",
) -> ConfirmatoryMappingManifest:
    if matrix.status not in {"FIXTURE_FROZEN", "FORMAL_FROZEN"} or matrix.b_primary is None or matrix.m_primary is None or matrix.r_primary is None:
        raise ValueError("只有完整冻结矩阵才能生成 mapping")
    if matrix.scope == "formal":
        raise FormalRunRejected("formal mapping requires external G2.4b authorization")
    count = matrix.b_primary * matrix.r_primary
    draws = sampling.draws("confirmatory", count)
    mappings = tuple(
        RepetitionMapping.create(
            repetition_id=f"{mapping_id}.rep-{index:04d}",
            draws=draws[index * matrix.b_primary : (index + 1) * matrix.b_primary],
            m_values=(2, matrix.m_primary),
        )
        for index in range(matrix.r_primary)
    )
    return ConfirmatoryMappingManifest(
        mapping_id=mapping_id,
        freeze_hash=matrix.artifact_hash,
        sampling_plan_hash=sampling.digest,
        pilot_draw_ids=tuple(pilot_draw_ids),
        mappings=mappings,
    )


__all__ = [
    "ANCHOR_IDS",
    "COST_SEMANTICS",
    "PILOT_MATRIX_SCHEMA",
    "CONFIRMATORY_MAPPING_SCHEMA",
    "AnchorPilotResult",
    "ArtificialCalibrationReport",
    "CandidateEvaluation",
    "ConfirmatoryMappingManifest",
    "CostSemantics",
    "MatrixFreeze",
    "build_confirmatory_mapping",
    "freeze_fixture_matrix",
    "freeze_formal_matrix",
    "required_repetitions",
    "run_artificial_distribution_calibration",
    "scan_candidates",
]
