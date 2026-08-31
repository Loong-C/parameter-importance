"""Stage 9 可复核统计指标与显式未定义语义。

任何退化情形都返回 ``MetricResult(defined=False, reason=...)``，而不是添加
epsilon 伪造数字。相关系数、质量分布和 top-k 集合各有不同的可定义条件；
调用方必须保存 ``reason``，不能只丢下一个空值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..contracts.immutable import freeze_json_mapping


@dataclass(frozen=True, slots=True)
class MetricResult:
    """一个可能未定义的标量统计量。"""

    defined: bool
    value: float | None
    reason: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.defined:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("defined metric 必须具有有限 value")
            if self.reason is not None:
                raise ValueError("defined metric 不应携带 undefined reason")
        elif self.value is not None or not self.reason:
            raise ValueError("undefined metric 必须是 value=None 且提供 reason")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))


def _undefined(reason: str, **metadata: object) -> MetricResult:
    return MetricResult(False, None, reason, metadata)


def _defined(value: float, **metadata: object) -> MetricResult:
    if not math.isfinite(float(value)):
        return _undefined("NON_FINITE_RESULT", **metadata)
    return MetricResult(True, float(value), metadata=metadata)


def _finite_array(values: object, *, minimum_size: int = 1) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size < minimum_size or not np.all(np.isfinite(array)):
        return None
    return array


def _error_arrays(
    estimates: object, reference: object
) -> tuple[np.ndarray, np.ndarray] | None:
    sample = _finite_array(estimates)
    target = _finite_array(reference)
    if sample is None or target is None:
        return None
    if sample.ndim == target.ndim:
        sample = sample[np.newaxis, ...]
    if sample.ndim != target.ndim + 1 or sample.shape[1:] != target.shape:
        return None
    return sample, target


def bias(estimates: object, reference: object) -> MetricResult:
    """返回重复均值相对 reference 的逐坐标 signed bias 的平均。"""

    arrays = _error_arrays(estimates, reference)
    if arrays is None:
        return _undefined("INVALID_OR_NON_FINITE_SHAPE")
    sample, target = arrays
    coordinate_bias = np.mean(sample, axis=0) - target
    return _defined(float(np.mean(coordinate_bias)), repetitions=sample.shape[0])


def variance(estimates: object) -> MetricResult:
    """返回逐坐标 repetition population variance 的平均。

    这里使用 ``ddof=0``，从而与有限重复样本上的经验 MSE 分解一致；若论文
    需要无偏样本方差，应另行明确报告，不得混用名称。
    """

    sample = _finite_array(estimates)
    if sample is None or sample.ndim < 1:
        return _undefined("INVALID_OR_NON_FINITE_INPUT")
    if sample.shape[0] < 2:
        return _undefined("INSUFFICIENT_REPETITIONS", repetitions=sample.shape[0])
    return _defined(
        float(np.mean(np.var(sample, axis=0, ddof=0))),
        repetitions=sample.shape[0],
        ddof=0,
    )


def mse(estimates: object, reference: object) -> MetricResult:
    """返回全部 repetition×coordinate 的均方误差。"""

    arrays = _error_arrays(estimates, reference)
    if arrays is None:
        return _undefined("INVALID_OR_NON_FINITE_SHAPE")
    sample, target = arrays
    return _defined(float(np.mean(np.square(sample - target))), repetitions=sample.shape[0])


def mae(estimates: object, reference: object) -> MetricResult:
    """返回全部 repetition×coordinate 的平均绝对误差。"""

    arrays = _error_arrays(estimates, reference)
    if arrays is None:
        return _undefined("INVALID_OR_NON_FINITE_SHAPE")
    sample, target = arrays
    return _defined(float(np.mean(np.abs(sample - target))), repetitions=sample.shape[0])


def mean_squared_bias(estimates: object, reference: object) -> MetricResult:
    """返回逐坐标 bias 平方的平均，而不是 pooled signed bias 的平方。"""

    arrays = _error_arrays(estimates, reference)
    if arrays is None:
        return _undefined("INVALID_OR_NON_FINITE_SHAPE")
    sample, target = arrays
    coordinate_bias = np.mean(sample, axis=0) - target
    return _defined(float(np.mean(np.square(coordinate_bias))))


def error_summary(estimates: object, reference: object) -> Mapping[str, MetricResult]:
    """一次返回 Bias/Variance/MSE/MAE 与 mean squared bias。"""

    return MappingProxyType(
        {
            "bias": bias(estimates, reference),
            "variance": variance(estimates),
            "mse": mse(estimates, reference),
            "mae": mae(estimates, reference),
            "mean_squared_bias": mean_squared_bias(estimates, reference),
        }
    )


def pearson(left: object, right: object) -> MetricResult:
    """计算 Pearson；P<2、常量向量或非有限输入显式未定义。"""

    lhs = _finite_array(left, minimum_size=2)
    rhs = _finite_array(right, minimum_size=2)
    if lhs is None or rhs is None:
        return _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
    lhs = lhs.reshape(-1)
    rhs = rhs.reshape(-1)
    if lhs.shape != rhs.shape:
        return _undefined("SHAPE_MISMATCH")
    if lhs.size < 2:
        return _undefined("P_LESS_THAN_TWO")
    lhs_centered = lhs - np.mean(lhs)
    rhs_centered = rhs - np.mean(rhs)
    lhs_norm = float(np.linalg.norm(lhs_centered))
    rhs_norm = float(np.linalg.norm(rhs_centered))
    if lhs_norm == 0 or rhs_norm == 0:
        return _undefined("CONSTANT_VECTOR")
    return _defined(float(np.dot(lhs_centered, rhs_centered) / (lhs_norm * rhs_norm)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        # 传统 rank 从 1 开始；相关系数对整体平移不敏感。
        average = (index + 1 + end) / 2.0
        ranks[order[index:end]] = average
        index = end
    return ranks


def spearman(left: object, right: object) -> MetricResult:
    """使用平均秩处理并列值后计算 Spearman。"""

    lhs = _finite_array(left, minimum_size=2)
    rhs = _finite_array(right, minimum_size=2)
    if lhs is None or rhs is None:
        return _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
    lhs = lhs.reshape(-1)
    rhs = rhs.reshape(-1)
    if lhs.shape != rhs.shape:
        return _undefined("SHAPE_MISMATCH")
    return pearson(_average_ranks(lhs), _average_ranks(rhs))


def _top_k_set(
    values: np.ndarray,
    k: int,
    canonical_ids: Sequence[str] | None,
) -> tuple[set[int] | None, str | None]:
    if values.size < 2:
        return None, "P_LESS_THAN_TWO"
    if k <= 0 or k > values.size:
        return None, "INVALID_K"
    if canonical_ids is not None:
        if len(canonical_ids) != values.size or len(set(canonical_ids)) != values.size:
            return None, "INVALID_CANONICAL_IDS"
        order = sorted(range(values.size), key=lambda index: (-values[index], canonical_ids[index]))
        return set(order[:k]), None
    if k < values.size:
        sorted_values = np.sort(values)[::-1]
        if sorted_values[k - 1] == sorted_values[k]:
            return None, "NON_UNIQUE_TOP_K_BOUNDARY"
    order = np.argsort(-values, kind="mergesort")
    return set(int(index) for index in order[:k]), None


def top_k_overlap(
    left: object,
    right: object,
    k: int,
    *,
    canonical_ids: Sequence[str] | None = None,
) -> MetricResult:
    """返回两个 top-k 集合交集占 k 的比例。"""

    lhs = _finite_array(left, minimum_size=2)
    rhs = _finite_array(right, minimum_size=2)
    if lhs is None or rhs is None:
        return _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
    lhs = lhs.reshape(-1)
    rhs = rhs.reshape(-1)
    if lhs.shape != rhs.shape:
        return _undefined("SHAPE_MISMATCH")
    left_set, left_reason = _top_k_set(lhs, k, canonical_ids)
    right_set, right_reason = _top_k_set(rhs, k, canonical_ids)
    if left_set is None or right_set is None:
        return _undefined(left_reason or right_reason or "TOP_K_UNDEFINED", k=k)
    return _defined(len(left_set.intersection(right_set)) / k, k=k)


def top_k_jaccard(
    left: object,
    right: object,
    k: int,
    *,
    canonical_ids: Sequence[str] | None = None,
) -> MetricResult:
    """返回两个 top-k 集合的 Jaccard 系数。"""

    lhs = _finite_array(left, minimum_size=2)
    rhs = _finite_array(right, minimum_size=2)
    if lhs is None or rhs is None:
        return _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
    lhs = lhs.reshape(-1)
    rhs = rhs.reshape(-1)
    if lhs.shape != rhs.shape:
        return _undefined("SHAPE_MISMATCH")
    left_set, left_reason = _top_k_set(lhs, k, canonical_ids)
    right_set, right_reason = _top_k_set(rhs, k, canonical_ids)
    if left_set is None or right_set is None:
        return _undefined(left_reason or right_reason or "TOP_K_UNDEFINED", k=k)
    union = left_set.union(right_set)
    if not union:
        return _undefined("EMPTY_TOP_K_UNION", k=k)
    return _defined(len(left_set.intersection(right_set)) / len(union), k=k)


def _mass_distribution(values: object) -> tuple[np.ndarray | None, str | None]:
    mass = _finite_array(values)
    if mass is None:
        return None, "EMPTY_OR_NON_FINITE_INPUT"
    mass = mass.reshape(-1)
    if mass.size < 2:
        return None, "P_LESS_THAN_TWO"
    if np.any(mass < 0):
        return None, "NEGATIVE_MASS"
    total = float(np.sum(mass))
    if total == 0:
        return None, "ZERO_TOTAL_MASS"
    return mass / total, None


def gini(values: object) -> MetricResult:
    """计算非负质量的 Gini 系数。"""

    probability, reason = _mass_distribution(values)
    if probability is None:
        return _undefined(reason or "MASS_UNDEFINED")
    sorted_mass = np.sort(probability)
    n = sorted_mass.size
    numerator = 2.0 * float(np.dot(np.arange(1, n + 1), sorted_mass))
    return _defined(numerator / n - (n + 1) / n)


def entropy(values: object) -> MetricResult:
    """计算非负质量分布的 Shannon entropy（自然对数）。"""

    probability, reason = _mass_distribution(values)
    if probability is None:
        return _undefined(reason or "MASS_UNDEFINED")
    positive = probability[probability > 0]
    return _defined(float(-np.sum(positive * np.log(positive))), log_base="e")


def hhi(values: object) -> MetricResult:
    """计算 Herfindahl-Hirschman index ``sum(p**2)``。"""

    probability, reason = _mass_distribution(values)
    if probability is None:
        return _undefined(reason or "MASS_UNDEFINED")
    return _defined(float(np.sum(np.square(probability))))


def effective_parameter_count(values: object) -> MetricResult:
    """返回基于 HHI 的有效参数数 ``1 / sum(p_k**2)``。

    这里的 ``p_k`` 是输入非负质量归一化后的概率。质量分布契约已经保证
    总质量严格大于零，因此 HHI 也严格大于零；函数不添加 epsilon，也不把
    Shannon entropy 的 perplexity ``exp(H)`` 混用为本项目冻结的有效参数数。
    """

    hhi_result = hhi(values)
    if not hhi_result.defined:
        return _undefined(hhi_result.reason or "HHI_UNDEFINED")
    assert hhi_result.value is not None
    return _defined(1.0 / hhi_result.value)


def top_q_mass(values: object, q: float) -> MetricResult:
    """返回最大 ``K=max(1,ceil(qP))`` 个坐标占总质量的比例。"""

    if not math.isfinite(q) or not 0 < q <= 1:
        return _undefined("INVALID_Q")
    probability, reason = _mass_distribution(values)
    if probability is None:
        return _undefined(reason or "MASS_UNDEFINED")
    k = max(1, math.ceil(q * probability.size))
    selected = np.sort(probability)[-k:]
    return _defined(float(np.sum(selected)), q=q, k=k)


def mean_confidence_interval(
    values: object,
    *,
    confidence: float = 0.95,
) -> tuple[MetricResult, MetricResult, MetricResult]:
    """返回 mean、双侧 Student-t lower 与 upper 三个结果。

    置信单位必须由上层决定（例如独立 repetition/checkpoint/seed），本函数绝不
    把参数坐标自动当作独立样本。样本标准差使用 ``ddof=1``，自由度为
    ``n-1``；临界值来自 SciPy 的 Student-t 分布，与 ``core.metrics`` 的冻结
    语义一致。该计算不加入 epsilon：零方差样本会自然得到零宽区间。
    """

    sample = _finite_array(values, minimum_size=2)
    if sample is None:
        undefined = _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
        return undefined, undefined, undefined
    sample = sample.reshape(-1)
    if not 0 < confidence < 1:
        undefined = _undefined("INVALID_CONFIDENCE")
        return undefined, undefined, undefined
    try:
        from scipy.stats import t as student_t
    except ImportError as exc:  # pragma: no cover - Windows CPU 锁定环境包含 SciPy
        raise RuntimeError("mean_confidence_interval 需要可选依赖 scipy") from exc
    mean_value = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1) / math.sqrt(sample.size))
    degrees_of_freedom = sample.size - 1
    critical_value = float(
        student_t.ppf(0.5 + confidence / 2.0, degrees_of_freedom)
    )
    half_width = critical_value * standard_error
    metadata = {
        "confidence": confidence,
        "n": sample.size,
        "degrees_of_freedom": degrees_of_freedom,
        "method": "student_t",
    }
    return (
        _defined(mean_value, **metadata),
        _defined(mean_value - half_width, **metadata),
        _defined(mean_value + half_width, **metadata),
    )


def damage_auc(ratios: object, damages: object) -> MetricResult:
    """按严格递增剪枝比例计算损伤曲线的梯形面积。"""

    x = _finite_array(ratios, minimum_size=2)
    y = _finite_array(damages, minimum_size=2)
    if x is None or y is None:
        return _undefined("INSUFFICIENT_OR_NON_FINITE_INPUT")
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.shape != y.shape:
        return _undefined("SHAPE_MISMATCH")
    if np.any(np.diff(x) <= 0):
        return _undefined("RATIOS_NOT_STRICTLY_INCREASING")
    return _defined(float(np.trapezoid(y, x)), points=x.size)
