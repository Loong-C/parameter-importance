"""Stage 2/3/9 共用的确定性统计指标。

所有可能在退化输入上“看起来像数字但没有统计意义”的指标统一返回
``MetricResult(defined=False, reason=...)``。零总质量、零范数、常量向量、单坐标
以及 top-k 边界并列且缺少 canonical ID 等情况均不会通过 epsilon 被伪造成有效值。
epsilon 只允许用于已经定义的数值残差尺度，不用于修补未定义统计量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from .errors import CoreContractError, NumericalError


@dataclass(frozen=True, slots=True)
class MetricResult:
    """一个可序列化指标及其定义状态。"""

    defined: bool
    value: float | None
    reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.defined:
            if self.value is None or not math.isfinite(float(self.value)):
                raise NumericalError("defined metric 必须具有有限 value")
            if self.reason is not None:
                raise CoreContractError("defined metric 不应携带 undefined reason")
        else:
            if self.value is not None:
                raise CoreContractError("undefined metric 的 value 必须为 None")
            if not self.reason:
                raise CoreContractError("undefined metric 必须说明 reason")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    @classmethod
    def ok(cls, value: float, **details: object) -> "MetricResult":
        return cls(True, float(value), None, details)

    @classmethod
    def undefined(cls, reason: str, **details: object) -> "MetricResult":
        return cls(False, None, reason, details)


def _vector(values: Sequence[float] | np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().to(torch.float64).cpu().numpy()
    else:
        array = np.asarray(values, dtype=np.float64)
    array = array.reshape(-1)
    if array.size == 0:
        raise CoreContractError(f"{name} 不能为空")
    if not np.isfinite(array).all():
        raise NumericalError(f"{name} 含 NaN/Inf")
    return array


def _paired_vectors(
    left: Sequence[float] | np.ndarray | torch.Tensor,
    right: Sequence[float] | np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    x = _vector(left, name="left")
    y = _vector(right, name="right")
    if x.shape != y.shape:
        raise CoreContractError("成对指标输入必须具有相同元素数")
    return x, y


def bias(
    estimates: Sequence[float] | np.ndarray | torch.Tensor,
    reference: Sequence[float] | np.ndarray | torch.Tensor | float,
) -> MetricResult:
    """返回所有 repetition/coordinate 上的平均有符号误差。"""

    estimate_array = np.asarray(
        estimates.detach().cpu().numpy() if isinstance(estimates, torch.Tensor) else estimates,
        dtype=np.float64,
    )
    reference_array = np.asarray(
        reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference,
        dtype=np.float64,
    )
    try:
        error = estimate_array - reference_array
    except ValueError as exc:
        raise CoreContractError("estimates 与 reference 无法按预期广播") from exc
    if error.size == 0 or not np.isfinite(error).all():
        raise NumericalError("bias 输入为空或含 NaN/Inf")
    return MetricResult.ok(float(error.mean()), count=int(error.size))


def variance(estimates: Sequence[float] | np.ndarray | torch.Tensor) -> MetricResult:
    """返回样本方差（ddof=1）；不足两个观测时未定义。"""

    values = _vector(estimates, name="estimates")
    if values.size < 2:
        return MetricResult.undefined("fewer_than_two_observations", count=int(values.size))
    return MetricResult.ok(float(values.var(ddof=1)), count=int(values.size))


def mse(
    estimates: Sequence[float] | np.ndarray | torch.Tensor,
    reference: Sequence[float] | np.ndarray | torch.Tensor | float,
) -> MetricResult:
    estimate_array = np.asarray(
        estimates.detach().cpu().numpy() if isinstance(estimates, torch.Tensor) else estimates,
        dtype=np.float64,
    )
    reference_array = np.asarray(
        reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference,
        dtype=np.float64,
    )
    try:
        error = estimate_array - reference_array
    except ValueError as exc:
        raise CoreContractError("estimates 与 reference 无法按预期广播") from exc
    if error.size == 0 or not np.isfinite(error).all():
        raise NumericalError("MSE 输入为空或含 NaN/Inf")
    return MetricResult.ok(float(np.mean(error**2)), count=int(error.size))


def mae(
    estimates: Sequence[float] | np.ndarray | torch.Tensor,
    reference: Sequence[float] | np.ndarray | torch.Tensor | float,
) -> MetricResult:
    estimate_array = np.asarray(
        estimates.detach().cpu().numpy() if isinstance(estimates, torch.Tensor) else estimates,
        dtype=np.float64,
    )
    reference_array = np.asarray(
        reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else reference,
        dtype=np.float64,
    )
    try:
        error = estimate_array - reference_array
    except ValueError as exc:
        raise CoreContractError("estimates 与 reference 无法按预期广播") from exc
    if error.size == 0 or not np.isfinite(error).all():
        raise NumericalError("MAE 输入为空或含 NaN/Inf")
    return MetricResult.ok(float(np.mean(np.abs(error))), count=int(error.size))


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> MetricResult:
    x, y = _paired_vectors(left, right)
    if x.size < 2:
        return MetricResult.undefined("fewer_than_two_coordinates", count=int(x.size))
    if np.all(x == x[0]) or np.all(y == y[0]):
        return MetricResult.undefined("constant_vector")
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    if denominator == 0:
        return MetricResult.undefined("zero_centered_norm")
    value = float(np.dot(centered_x, centered_y) / denominator)
    return MetricResult.ok(max(-1.0, min(1.0, value)), count=int(x.size))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """使用平均秩处理并列值；秩从 1 开始。"""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> MetricResult:
    x, y = _paired_vectors(left, right)
    if x.size < 2:
        return MetricResult.undefined("fewer_than_two_coordinates", count=int(x.size))
    if np.all(x == x[0]) or np.all(y == y[0]):
        return MetricResult.undefined("constant_vector")
    return pearson_correlation(_average_ranks(x), _average_ranks(y))


def normalized_l1_error(candidate: Sequence[float], reference: Sequence[float]) -> MetricResult:
    candidate_array, reference_array = _paired_vectors(candidate, reference)
    denominator = float(np.abs(reference_array).sum())
    if denominator == 0:
        return MetricResult.undefined("zero_reference_l1_norm")
    return MetricResult.ok(float(np.abs(candidate_array - reference_array).sum() / denominator))


def normalized_l2_error(candidate: Sequence[float], reference: Sequence[float]) -> MetricResult:
    candidate_array, reference_array = _paired_vectors(candidate, reference)
    denominator = float(np.linalg.norm(reference_array))
    if denominator == 0:
        return MetricResult.undefined("zero_reference_l2_norm")
    return MetricResult.ok(float(np.linalg.norm(candidate_array - reference_array) / denominator))


def normalized_linf_error(candidate: Sequence[float], reference: Sequence[float]) -> MetricResult:
    candidate_array, reference_array = _paired_vectors(candidate, reference)
    denominator = float(np.max(np.abs(reference_array)))
    if denominator == 0:
        return MetricResult.undefined("zero_reference_linf_norm")
    return MetricResult.ok(float(np.max(np.abs(candidate_array - reference_array)) / denominator))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> MetricResult:
    x, y = _paired_vectors(left, right)
    if x.size < 2:
        return MetricResult.undefined("fewer_than_two_coordinates", count=int(x.size))
    norm_product = float(np.linalg.norm(x) * np.linalg.norm(y))
    if norm_product == 0:
        return MetricResult.undefined("zero_vector_norm")
    value = float(np.dot(x, y) / norm_product)
    return MetricResult.ok(max(-1.0, min(1.0, value)))


def sign_agreement(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    active_threshold: float = 0.0,
) -> MetricResult:
    candidate_array, reference_array = _paired_vectors(candidate, reference)
    if not math.isfinite(active_threshold) or active_threshold < 0:
        raise CoreContractError("active_threshold 必须为有限非负数")
    active = np.abs(reference_array) > active_threshold
    if not active.any():
        return MetricResult.undefined("empty_active_set", active_threshold=active_threshold)
    agreement = np.sign(candidate_array[active]) == np.sign(reference_array[active])
    return MetricResult.ok(float(agreement.mean()), active_count=int(active.sum()))


def _top_indices(
    values: np.ndarray,
    k: int,
    coordinate_ids: Sequence[str] | None,
) -> tuple[np.ndarray | None, str | None]:
    if k <= 0 or k > values.size:
        raise CoreContractError("k 必须位于 [1,P]")
    if coordinate_ids is not None:
        if len(coordinate_ids) != values.size or len(set(coordinate_ids)) != values.size:
            raise CoreContractError("coordinate_ids 必须与向量等长且唯一")
        order = sorted(range(values.size), key=lambda i: (-float(values[i]), coordinate_ids[i]))
        return np.asarray(order[:k], dtype=np.int64), None
    sorted_values = np.sort(values)[::-1]
    if k < values.size and sorted_values[k - 1] == sorted_values[k]:
        return None, "non_unique_top_k_boundary_without_coordinate_ids"
    order = np.argsort(-values, kind="mergesort")
    return order[:k], None


def top_k_overlap(
    left: Sequence[float],
    right: Sequence[float],
    k: int,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> MetricResult:
    x, y = _paired_vectors(left, right)
    left_indices, left_reason = _top_indices(x, k, coordinate_ids)
    right_indices, right_reason = _top_indices(y, k, coordinate_ids)
    reason = left_reason or right_reason
    if reason:
        return MetricResult.undefined(reason, k=k)
    assert left_indices is not None and right_indices is not None
    intersection = len(set(left_indices.tolist()) & set(right_indices.tolist()))
    return MetricResult.ok(intersection / k, k=k, intersection=intersection)


def top_k_jaccard(
    left: Sequence[float],
    right: Sequence[float],
    k: int,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> MetricResult:
    x, y = _paired_vectors(left, right)
    left_indices, left_reason = _top_indices(x, k, coordinate_ids)
    right_indices, right_reason = _top_indices(y, k, coordinate_ids)
    reason = left_reason or right_reason
    if reason:
        return MetricResult.undefined(reason, k=k)
    assert left_indices is not None and right_indices is not None
    left_set, right_set = set(left_indices.tolist()), set(right_indices.tolist())
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return MetricResult.ok(intersection / union, k=k, intersection=intersection, union=union)


def _mass_distribution(masses: Sequence[float]) -> tuple[np.ndarray | None, MetricResult | None]:
    values = _vector(masses, name="masses")
    if values.size < 2:
        return None, MetricResult.undefined("fewer_than_two_coordinates", count=int(values.size))
    if np.any(values < 0):
        raise CoreContractError("质量指标要求所有输入非负")
    total = float(values.sum())
    if total == 0:
        return None, MetricResult.undefined("zero_total_mass")
    return values / total, None


def gini_coefficient(masses: Sequence[float]) -> MetricResult:
    probabilities, undefined = _mass_distribution(masses)
    if undefined:
        return undefined
    assert probabilities is not None
    sorted_values = np.sort(probabilities)
    n = sorted_values.size
    coefficient = float((2 * np.dot(np.arange(1, n + 1), sorted_values) / n) - (n + 1) / n)
    return MetricResult.ok(max(0.0, min(1.0, coefficient)), count=int(n))


def entropy(masses: Sequence[float], *, normalized: bool = False) -> MetricResult:
    probabilities, undefined = _mass_distribution(masses)
    if undefined:
        return undefined
    assert probabilities is not None
    positive = probabilities[probabilities > 0]
    value = float(-np.sum(positive * np.log(positive)))
    if normalized:
        value /= math.log(probabilities.size)
    return MetricResult.ok(value, normalized=normalized, count=int(probabilities.size))


def hhi(masses: Sequence[float]) -> MetricResult:
    probabilities, undefined = _mass_distribution(masses)
    if undefined:
        return undefined
    assert probabilities is not None
    return MetricResult.ok(float(np.sum(probabilities**2)), count=int(probabilities.size))


def effective_parameter_count(masses: Sequence[float]) -> MetricResult:
    hhi_result = hhi(masses)
    if not hhi_result.defined:
        return hhi_result
    assert hhi_result.value is not None
    return MetricResult.ok(1.0 / hhi_result.value)


def top_q_mass(
    masses: Sequence[float],
    q: float,
    *,
    coordinate_ids: Sequence[str] | None = None,
) -> MetricResult:
    probabilities, undefined = _mass_distribution(masses)
    if undefined:
        return undefined
    if not math.isfinite(q) or not 0 < q <= 1:
        raise CoreContractError("q 必须位于 (0,1]")
    assert probabilities is not None
    k = max(1, math.ceil(q * probabilities.size))
    indices, reason = _top_indices(probabilities, k, coordinate_ids)
    if reason:
        return MetricResult.undefined(reason, q=q, k=k)
    assert indices is not None
    return MetricResult.ok(float(probabilities[indices].sum()), q=q, k=k)


def confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> MetricResult:
    """计算独立重复的双侧 Student-t 均值置信区间半宽。

    返回值是 half-width；中心、上下界写入 details。SciPy 仅在调用该指标时延迟
    导入，因而其余纯核心不依赖 SciPy。
    """

    samples = _vector(values, name="values")
    if samples.size < 2:
        return MetricResult.undefined("fewer_than_two_observations", count=int(samples.size))
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise CoreContractError("confidence 必须位于 (0,1)")
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:  # pragma: no cover - 锁定环境应包含 scipy
        raise CoreContractError("confidence_interval 需要可选依赖 scipy") from exc
    mean = float(samples.mean())
    standard_error = float(samples.std(ddof=1) / math.sqrt(samples.size))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, samples.size - 1))
    half_width = critical * standard_error
    return MetricResult.ok(
        half_width,
        mean=mean,
        lower=mean - half_width,
        upper=mean + half_width,
        confidence=confidence,
        count=int(samples.size),
    )


def damage_auc(ratios: Sequence[float], damages: Sequence[float]) -> MetricResult:
    """按剪枝比例网格对性能损伤曲线执行梯形积分。"""

    x, y = _paired_vectors(ratios, damages)
    if x.size < 2:
        return MetricResult.undefined("fewer_than_two_curve_points", count=int(x.size))
    if np.any(x < 0) or np.any(x > 1) or np.any(np.diff(x) <= 0):
        raise CoreContractError("剪枝比例必须位于 [0,1] 且严格递增")
    return MetricResult.ok(float(np.trapezoid(y, x)), points=int(x.size))
