"""Pure numerical metrics for the Stage 3 path analysis.

The Stage 3 runner deliberately keeps orchestration, artifact validation and
the numerical metric layer separate.  This module contains only deterministic
array operations and returns the core ``MetricResult`` type for quantities that
can be undefined.  In particular, it never adds an epsilon to a denominator,
silently drops a unit, or changes a pre-registered threshold.

The grouping functions operate on already aligned parameter coordinates.  The
caller is responsible for supplying the canonical registry order and explicit
layer/module labels; no parameter-name guessing is performed here.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from ..core.metrics import (
    MetricResult,
    cosine_similarity as _cosine_similarity,
    normalized_l1_error as _normalized_l1_error,
    normalized_l2_error as _normalized_l2_error,
    normalized_linf_error as _normalized_linf_error,
    sign_agreement as _sign_agreement,
    spearman_correlation as _spearman_correlation,
    top_k_jaccard as _top_k_jaccard,
    top_k_overlap as _top_k_overlap,
)


DEFAULT_TOP_Q: tuple[float, ...] = (0.001, 0.01, 0.05)
"""The Stage 3 pre-registered top-q proportions."""


def _as_vector(values: object, *, name: str) -> np.ndarray:
    """Convert a tensor/array/sequence to a finite, non-empty FP64 vector."""

    # Avoid importing torch just for this adapter.  This handles torch tensors
    # and other tensor-like objects without making the metric module stateful.
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy = getattr(candidate, "numpy", None)
    if callable(numpy):
        candidate = numpy()
    try:
        vector = np.asarray(candidate, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须可转换为数值向量") from error
    if vector.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} 不能包含 NaN/Inf")
    return vector


def _paired_vectors(
    left: object,
    right: object,
    *,
    left_name: str = "candidate",
    right_name: str = "reference",
) -> tuple[np.ndarray, np.ndarray]:
    lhs = _as_vector(left, name=left_name)
    rhs = _as_vector(right, name=right_name)
    if lhs.shape != rhs.shape:
        raise ValueError(
            f"{left_name} 与 {right_name} 必须有相同坐标数，"
            f"实际为 {lhs.size} 与 {rhs.size}"
        )
    return lhs, rhs


# These wrappers give Stage 3 a stable, discoverable API while retaining the
# core metric contract (MetricResult, explicit undefined values, no epsilon).
def normalized_l1(candidate: object, reference: object) -> MetricResult:
    """Return ``||candidate-reference||_1 / ||reference||_1``."""

    return _normalized_l1_error(*_paired_vectors(candidate, reference))


def normalized_l2(candidate: object, reference: object) -> MetricResult:
    """Return ``||candidate-reference||_2 / ||reference||_2``."""

    return _normalized_l2_error(*_paired_vectors(candidate, reference))


def normalized_linf(candidate: object, reference: object) -> MetricResult:
    """Return ``||candidate-reference||_inf / ||reference||_inf``."""

    return _normalized_linf_error(*_paired_vectors(candidate, reference))


def cosine(candidate: object, reference: object) -> MetricResult:
    """Return cosine similarity for two aligned contribution vectors."""

    return _cosine_similarity(*_paired_vectors(candidate, reference))


def active_set_spearman(
    candidate: object,
    reference: object,
    *,
    active_threshold: float = 0.0,
) -> MetricResult:
    """Return Spearman correlation on the reference-defined active set.

    A coordinate is active when ``abs(reference) > active_threshold``.  The
    threshold is a caller-supplied frozen numerical-bottom-noise constant; it
    is never inferred from the observed vectors.  Average ranks are used for
    ties by the shared core implementation.  A result with fewer than two
    active coordinates is explicitly undefined.
    """

    if isinstance(active_threshold, bool) or not math.isfinite(float(active_threshold)):
        raise ValueError("active_threshold 必须是有限数")
    if active_threshold < 0:
        raise ValueError("active_threshold 不能为负")
    lhs, rhs = _paired_vectors(candidate, reference)
    active = np.abs(rhs) > float(active_threshold)
    active_count = int(active.sum())
    if active_count == 0:
        return MetricResult.undefined(
            "empty_active_set",
            active_threshold=float(active_threshold),
            active_count=0,
            active_basis="reference",
        )
    if active_count < 2:
        return MetricResult.undefined(
            "fewer_than_two_active_coordinates",
            active_threshold=float(active_threshold),
            active_count=active_count,
            active_basis="reference",
        )
    result = _spearman_correlation(lhs[active], rhs[active])
    if not result.defined:
        return MetricResult.undefined(
            result.reason or "active_set_spearman_undefined",
            active_threshold=float(active_threshold),
            active_count=active_count,
            active_basis="reference",
        )
    assert result.value is not None
    return MetricResult.ok(
        result.value,
        active_threshold=float(active_threshold),
        active_count=active_count,
        active_basis="reference",
        tie_policy="average_rank",
    )


def sign_consistency(
    candidate: object,
    reference: object,
    *,
    active_threshold: float = 0.0,
) -> MetricResult:
    """Return signed agreement on coordinates active in the reference."""

    if isinstance(active_threshold, bool) or not math.isfinite(float(active_threshold)):
        raise ValueError("active_threshold 必须是有限数")
    if active_threshold < 0:
        raise ValueError("active_threshold 不能为负")
    lhs, rhs = _paired_vectors(candidate, reference)
    result = _sign_agreement(lhs, rhs, active_threshold=float(active_threshold))
    if not result.defined:
        return result
    assert result.value is not None
    return MetricResult.ok(
        result.value,
        **dict(result.details),
        active_basis="reference",
    )


def _validate_q(q: float) -> float:
    if isinstance(q, bool) or not math.isfinite(float(q)) or not 0 < float(q) <= 1:
        raise ValueError("q 必须位于 (0, 1]")
    return float(q)


def _validate_coordinate_ids(
    coordinate_ids: Sequence[str] | None,
    size: int,
) -> tuple[str, ...] | None:
    if coordinate_ids is None:
        return None
    ids = tuple(str(item) for item in coordinate_ids)
    if len(ids) != size or len(set(ids)) != size:
        raise ValueError("coordinate_ids 必须与向量等长且唯一")
    return ids


def top_q_metrics(
    candidate: object,
    reference: object,
    q_values: Sequence[float] = DEFAULT_TOP_Q,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> dict[float, dict[str, object]]:
    """Return overlap and Jaccard for every pre-registered top-q proportion.

    ``K=max(1, ceil(q*P))`` is used exactly.  The input values are ranked as
    supplied; callers must pass the intended signed/positive/absolute view
    explicitly.  Without canonical coordinate IDs, a tied top-k boundary is
    returned as an undefined metric rather than being resolved arbitrarily.
    """

    lhs, rhs = _paired_vectors(candidate, reference)
    if not q_values:
        raise ValueError("q_values 不能为空")
    qs = tuple(_validate_q(q) for q in q_values)
    if len(set(qs)) != len(qs):
        raise ValueError("q_values 不能重复")
    ids = _validate_coordinate_ids(coordinate_ids, lhs.size)
    result: dict[float, dict[str, object]] = {}
    for q in qs:
        k = max(1, math.ceil(q * lhs.size))
        result[q] = {
            "q": q,
            "k": k,
            "overlap": _top_k_overlap(lhs, rhs, k, coordinate_ids=ids),
            "jaccard": _top_k_jaccard(lhs, rhs, k, coordinate_ids=ids),
        }
    return result


def top_q_overlap(
    candidate: object,
    reference: object,
    q_values: Sequence[float] = DEFAULT_TOP_Q,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> dict[float, MetricResult]:
    """Return only the overlap component of :func:`top_q_metrics`."""

    return {
        q: row["overlap"]  # type: ignore[return-value]
        for q, row in top_q_metrics(
            candidate,
            reference,
            q_values=q_values,
            coordinate_ids=coordinate_ids,
        ).items()
    }


def top_q_jaccard(
    candidate: object,
    reference: object,
    q_values: Sequence[float] = DEFAULT_TOP_Q,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> dict[float, MetricResult]:
    """Return only the Jaccard component of :func:`top_q_metrics`."""

    return {
        q: row["jaccard"]  # type: ignore[return-value]
        for q, row in top_q_metrics(
            candidate,
            reference,
            q_values=q_values,
            coordinate_ids=coordinate_ids,
        ).items()
    }


def _group_labels(
    groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    size: int,
) -> tuple[Hashable, ...]:
    """Normalize either per-coordinate labels or an explicit index mapping."""

    if isinstance(groups, Mapping):
        labels: list[Hashable | None] = [None] * size
        for group, indices in groups.items():
            if isinstance(indices, (str, bytes)):
                raise ValueError("group index 集合不能是字符串")
            for raw_index in indices:
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise ValueError("group index 必须是整数")
                if raw_index < 0 or raw_index >= size:
                    raise ValueError("group index 超出向量范围")
                if labels[raw_index] is not None:
                    raise ValueError("每个坐标只能属于一个 group")
                labels[raw_index] = group
        if any(label is None for label in labels):
            raise ValueError("group 映射必须覆盖全部坐标")
        return tuple(label for label in labels if label is not None)
    labels = tuple(groups)
    if len(labels) != size:
        raise ValueError("每个坐标必须有一个 group label")
    if any(label is None for label in labels):
        raise ValueError("group label 不能为 None")
    return labels


def _quality_values(values: np.ndarray, quality_view: str) -> np.ndarray:
    if quality_view == "absolute":
        return np.abs(values)
    if quality_view == "positive":
        return np.maximum(values, 0.0)
    raise ValueError("quality_view 只能是 absolute 或 positive")


def _sorted_groups(labels: Sequence[Hashable]) -> tuple[Hashable, ...]:
    unique = set(labels)
    return tuple(sorted(unique, key=lambda item: (type(item).__name__, repr(item))))


def aggregate_by_group(
    values: object,
    groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    *,
    quality_view: str = "absolute",
) -> dict[Hashable, dict[str, float | int | None]]:
    """Aggregate one aligned vector by explicit layer/module-style labels.

    ``total`` and ``mean`` use the values exactly as supplied (usually signed),
    while ``quality_total`` and ``quality_fraction`` use the explicitly chosen
    non-negative ``absolute`` or ``positive`` view.  A zero total quality has
    an undefined fraction represented by ``None``; no epsilon is inserted.
    """

    vector = _as_vector(values, name="values")
    labels = _group_labels(groups, vector.size)
    quality = _quality_values(vector, quality_view)
    totals: dict[Hashable, float] = {}
    quality_totals: dict[Hashable, float] = {}
    counts: dict[Hashable, int] = {}
    for value, mass, group in zip(vector, quality, labels, strict=True):
        totals[group] = totals.get(group, 0.0) + float(value)
        quality_totals[group] = quality_totals.get(group, 0.0) + float(mass)
        counts[group] = counts.get(group, 0) + 1
    quality_sum = float(sum(quality_totals.values()))
    return {
        group: {
            "count": counts[group],
            "total": totals[group],
            "mean": totals[group] / counts[group],
            "quality_total": quality_totals[group],
            "quality_fraction": (
                quality_totals[group] / quality_sum if quality_sum > 0 else None
            ),
        }
        for group in _sorted_groups(labels)
    }


def quality_total_variation(
    candidate: object,
    reference: object,
    groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    *,
    quality_view: str = "absolute",
) -> MetricResult:
    """Return TV distance between candidate/reference group quality masses."""

    lhs, rhs = _paired_vectors(candidate, reference)
    labels = _group_labels(groups, lhs.size)
    lhs_quality = _quality_values(lhs, quality_view)
    rhs_quality = _quality_values(rhs, quality_view)
    lhs_by_group: dict[Hashable, float] = {}
    rhs_by_group: dict[Hashable, float] = {}
    for left, right, group in zip(lhs_quality, rhs_quality, labels, strict=True):
        lhs_by_group[group] = lhs_by_group.get(group, 0.0) + float(left)
        rhs_by_group[group] = rhs_by_group.get(group, 0.0) + float(right)
    left_total = float(sum(lhs_by_group.values()))
    right_total = float(sum(rhs_by_group.values()))
    if left_total == 0 or right_total == 0:
        return MetricResult.undefined(
            "zero_total_quality",
            quality_view=quality_view,
            candidate_total=left_total,
            reference_total=right_total,
        )
    all_groups = _sorted_groups(labels)
    value = 0.5 * sum(
        abs(
            lhs_by_group.get(group, 0.0) / left_total
            - rhs_by_group.get(group, 0.0) / right_total
        )
        for group in all_groups
    )
    return MetricResult.ok(value, quality_view=quality_view, group_count=len(all_groups))


def aggregate_group_metrics(
    candidate: object,
    reference: object,
    groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    *,
    quality_view: str = "absolute",
) -> dict[str, object]:
    """Return candidate/reference totals, means, quality fractions and TV."""

    lhs, rhs = _paired_vectors(candidate, reference)
    labels = _group_labels(groups, lhs.size)
    candidate_rows = aggregate_by_group(lhs, labels, quality_view=quality_view)
    reference_rows = aggregate_by_group(rhs, labels, quality_view=quality_view)
    groups_out: dict[Hashable, dict[str, float | int | None]] = {}
    for group in _sorted_groups(labels):
        left = candidate_rows[group]
        right = reference_rows[group]
        groups_out[group] = {
            "count": left["count"],
            "candidate_total": left["total"],
            "reference_total": right["total"],
            "candidate_mean": left["mean"],
            "reference_mean": right["mean"],
            "candidate_quality_total": left["quality_total"],
            "reference_quality_total": right["quality_total"],
            "candidate_quality_fraction": left["quality_fraction"],
            "reference_quality_fraction": right["quality_fraction"],
        }
    return {
        "quality_view": quality_view,
        "groups": groups_out,
        "quality_total_variation": quality_total_variation(
            lhs, rhs, labels, quality_view=quality_view
        ),
    }


def aggregate_layer_module_metrics(
    candidate: object,
    reference: object,
    *,
    layer_groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    module_groups: Sequence[Hashable] | Mapping[Hashable, Sequence[int]],
    quality_view: str = "absolute",
) -> dict[str, object]:
    """Compute the same aggregation independently for layers and modules."""

    return {
        "layer": aggregate_group_metrics(
            candidate, reference, layer_groups, quality_view=quality_view
        ),
        "module": aggregate_group_metrics(
            candidate, reference, module_groups, quality_view=quality_view
        ),
    }


@dataclass(frozen=True, slots=True)
class UnitMetricObservation:
    """One raw finite metric value from one independent Stage 3 unit."""

    unit_id: str
    value: float
    strata: Mapping[str, Hashable]

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("unit_id 不能为空")
        if not math.isfinite(float(self.value)):
            raise ValueError("observation value 必须有限，不能包含 NaN/Inf")
        if any(not isinstance(key, str) or not key for key in self.strata):
            raise ValueError("strata key 必须是非空字符串")
        try:
            for value in self.strata.values():
                hash(value)
        except TypeError as error:
            raise ValueError("strata value 必须可哈希") from error
        object.__setattr__(self, "strata", MappingProxyType(dict(self.strata)))


def _observation_mapping(item: object, *, unit_key: str, value_key: str) -> UnitMetricObservation:
    if isinstance(item, UnitMetricObservation):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("每条 observation 必须是 mapping 或 UnitMetricObservation")
    if unit_key not in item or value_key not in item:
        raise ValueError(f"observation 缺少 {unit_key!r} 或 {value_key!r}")
    unit_id = item[unit_key]
    if not isinstance(unit_id, str) or not unit_id:
        raise ValueError("unit_id 必须是非空字符串")
    try:
        value = float(item[value_key])
    except (TypeError, ValueError) as error:
        raise ValueError("observation value 必须是数值") from error
    strata = {
        str(key): value_item
        for key, value_item in item.items()
        if key not in {unit_key, value_key}
    }
    return UnitMetricObservation(unit_id, value, strata)


def summarize_unit_observations(
    observations: Sequence[Mapping[str, object] | UnitMetricObservation],
    *,
    unit_key: str = "unit_id",
    value_key: str = "value",
    strata_keys: Sequence[str] | None = None,
    worst: str = "max",
) -> dict[str, object]:
    """Summarize raw unit values as median, p95 and worst by frozen strata.

    Each ``(stratum, unit_id)`` may occur only once.  This prevents multiple
    candidate/probe rows from being accidentally treated as independent units.
    The returned ``raw_observations`` preserve every input unit and the grouped
    summaries are sorted deterministically.  ``worst=max`` is the default for
    error metrics; ``worst=min`` is available for metrics where lower values
    are not adverse, but it must be chosen explicitly by the caller.
    """

    if not observations:
        raise ValueError("observations 不能为空")
    if worst not in {"max", "min"}:
        raise ValueError("worst 只能是 max 或 min")
    parsed = tuple(
        _observation_mapping(item, unit_key=unit_key, value_key=value_key)
        for item in observations
    )
    if strata_keys is None:
        keys = tuple(sorted({key for item in parsed for key in item.strata}))
    else:
        keys = tuple(str(key) for key in strata_keys)
        if len(set(keys)) != len(keys):
            raise ValueError("strata_keys 不能重复")
    for item in parsed:
        missing = set(keys) - set(item.strata)
        if missing:
            raise ValueError(f"observation 缺少 strata 字段：{sorted(missing)}")
    grouped: dict[tuple[Hashable, ...], list[UnitMetricObservation]] = {}
    seen: set[tuple[tuple[Hashable, ...], str]] = set()
    for item in parsed:
        key = tuple(item.strata[name] for name in keys)
        marker = (key, item.unit_id)
        if marker in seen:
            raise ValueError(f"同一 stratum/unit 重复观测：{key!r}/{item.unit_id!r}")
        seen.add(marker)
        grouped.setdefault(key, []).append(item)

    group_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=repr):
        rows = sorted(grouped[key], key=lambda item: item.unit_id)
        values = np.asarray([item.value for item in rows], dtype=np.float64)
        summary_worst = float(np.max(values) if worst == "max" else np.min(values))
        group_rows.append(
            {
                "strata": dict(zip(keys, key, strict=True)),
                "unit_count": len(rows),
                "unit_ids": [item.unit_id for item in rows],
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95.0, method="linear")),
                "worst": summary_worst,
                "raw_values": [float(item.value) for item in rows],
            }
        )
    return {
        "strata_keys": list(keys),
        "worst_policy": worst,
        "unit_count": len(parsed),
        "raw_observations": [
            {
                "unit_id": item.unit_id,
                "value": float(item.value),
                "strata": {key: item.strata[key] for key in keys},
            }
            for item in sorted(parsed, key=lambda item: (repr(tuple(item.strata[key] for key in keys)), item.unit_id))
        ],
        "groups": group_rows,
    }


# Descriptive aliases make the intended Stage 3 vocabulary available without
# duplicating implementations.
normalized_l1_error = normalized_l1
normalized_l2_error = normalized_l2
normalized_linf_error = normalized_linf
cosine_similarity = cosine
sign_agreement = sign_consistency
summarize_by_strata = summarize_unit_observations
aggregate_layer_module = aggregate_layer_module_metrics


__all__ = [
    "DEFAULT_TOP_Q",
    "MetricResult",
    "UnitMetricObservation",
    "active_set_spearman",
    "aggregate_by_group",
    "aggregate_group_metrics",
    "aggregate_layer_module",
    "aggregate_layer_module_metrics",
    "cosine",
    "cosine_similarity",
    "normalized_l1",
    "normalized_l1_error",
    "normalized_l2",
    "normalized_l2_error",
    "normalized_linf",
    "normalized_linf_error",
    "quality_total_variation",
    "sign_agreement",
    "sign_consistency",
    "summarize_by_strata",
    "summarize_unit_observations",
    "top_q_jaccard",
    "top_q_metrics",
    "top_q_overlap",
]
