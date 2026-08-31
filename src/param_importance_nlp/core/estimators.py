"""局部梯度空间重要性的纯张量估计器。

所有 kernel 只返回“未乘学习率、未裁剪”的 ``core``。学习率和裁剪属于训练
step 的尺度合同，由 :meth:`EstimatorResult.from_core` 统一应用：每个参数张量
按其静态 group ID 读取该 step 的实际学习率，裁剪因子最多乘一次。

严格声明边界：固定状态、未裁剪的 U core 在 i.i.d./共同均值等前提下可保留
无偏性声明；若 clip factor 来自同一批随机梯度，乘积是 plug-in 在线分数，不能
再声称严格无偏。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from ..contracts.immutable import freeze_json_mapping
from .errors import CoreContractError, NumericalError
from .sufficient_statistics import EqualSufficientStatistics, WeightedSufficientStatistics
from .tensors import TensorMap


UNBIASED_FIXED_STATE = "unbiased_fixed_state_under_declared_sampling_assumptions"
PLUGIN_SAME_BATCH_CLIP = "plugin_same_batch_clip_no_strict_unbiasedness"
NO_UNBIASEDNESS_CLAIM = "no_unbiasedness_claim"

_STRICT_FIXED_STATE_ESTIMATORS = {
    "local_gradient_space_importance_u",
    "local_gradient_space_importance_u_weighted",
    "double_sample_gradient_importance",
    "cross_u_gradient_importance",
}


def _square(tensors: TensorMap) -> TensorMap:
    result = tensors.map(torch.square)
    result.assert_finite()
    return result


def raw_importance(mean_gradient: TensorMap) -> TensorMap:
    """返回同批平均梯度平方 ``g_bar**2``（未乘学习率）。

    该估计器包含随机梯度方差导致的正偏差，保留它是为了构造关键对照，而不是
    作为 U-statistic 的别名。
    """

    mean_gradient.assert_finite()
    return _square(mean_gradient)


def double_sample_importance(
    mean_gradient_a: TensorMap,
    mean_gradient_b: TensorMap,
) -> TensorMap:
    """返回两个独立样本均值梯度的逐坐标乘积。"""

    mean_gradient_a.assert_compatible(mean_gradient_b)
    mean_gradient_a.assert_finite()
    mean_gradient_b.assert_finite()
    result = mean_gradient_a * mean_gradient_b
    result.assert_finite()
    return result


def equal_u_importance(statistics: EqualSufficientStatistics) -> TensorMap:
    """计算等权 U core ``(S1**2-S2)/(M*(M-1))``。

    ``M=1`` 无法删除对角项，必须抛出异常；负输出是有限样本无偏估计器的正常
    波动，绝不能在 kernel 内 clamp。
    """

    if statistics.count < 2:
        raise CoreContractError("等权 U-statistic 至少需要两个统计单元")
    denominator = statistics.count * (statistics.count - 1)
    result = (statistics.s1 * statistics.s1 - statistics.s2) / denominator
    result.assert_finite()
    return result


def weighted_u_importance(
    statistics: WeightedSufficientStatistics,
    *,
    require_unbiasedness_assumptions: bool = False,
) -> TensorMap:
    """计算不等权 U core ``(G1**2-G2)/(N1**2-N2)``。

    ``require_unbiasedness_assumptions=True`` 时，权重非外生或共同均值假设未声明
    会直接失败；关闭时仍可计算描述性 plug-in 数值，但上层不得自动生成无偏声明。
    """

    if statistics.count < 2:
        raise CoreContractError("加权 U-statistic 至少需要两个统计单元")
    denominator = statistics.denominator
    if not math.isfinite(denominator) or denominator <= 0:
        raise CoreContractError("加权 U-statistic 分母 N1**2-N2 必须为正")
    if require_unbiasedness_assumptions and not (
        statistics.weights_exogenous and statistics.common_mean_assumption
    ):
        raise CoreContractError("未满足加权 U 的外生权重与共同均值声明")
    result = (statistics.g1 * statistics.g1 - statistics.g2) / denominator
    result.assert_finite()
    return result


def cross_u_importance(
    x_samples: Sequence[TensorMap],
    y_samples: Sequence[TensorMap],
    *,
    x_weights: Sequence[float | int] | None = None,
    y_weights: Sequence[float | int] | None = None,
    exclude_matching_pairs: bool = True,
) -> TensorMap:
    """计算一般的交叉 U-statistic。

    对成对观测 ``(X_m, Y_m)``，默认删除同索引对角项：

    ``[(sum wx*X)(sum wy*Y) - sum wx*wy*X*Y] /``
    ``[(sum wx)(sum wy) - sum wx*wy]``。

    该公式估计 ``E[X]E[Y]``，允许同一观测内部的 X/Y 相关。若两组样本来自彼此
    独立的 sampling stream，可设 ``exclude_matching_pairs=False``；此时退化为
    两个加权均值的乘积（double sample）。
    """

    if not x_samples or not y_samples:
        raise CoreContractError("cross-U 的 X/Y 样本都不能为空")
    reference = x_samples[0]
    for sample in (*x_samples, *y_samples):
        reference.assert_compatible(sample)
        sample.assert_finite()
    wx = [1.0] * len(x_samples) if x_weights is None else [float(v) for v in x_weights]
    wy = [1.0] * len(y_samples) if y_weights is None else [float(v) for v in y_weights]
    if len(wx) != len(x_samples) or len(wy) != len(y_samples):
        raise CoreContractError("cross-U 权重数量必须与样本数量一致")
    if any(not math.isfinite(v) or v <= 0 for v in (*wx, *wy)):
        raise CoreContractError("cross-U 权重必须为有限正数")

    sx = TensorMap.zeros_like(reference)
    sy = TensorMap.zeros_like(reference)
    for sample, weight in zip(x_samples, wx, strict=True):
        sx = sx + sample * weight
    for sample, weight in zip(y_samples, wy, strict=True):
        sy = sy + sample * weight

    if not exclude_matching_pairs:
        return (sx / sum(wx)) * (sy / sum(wy))
    if len(x_samples) != len(y_samples):
        raise CoreContractError("删除 matching pair 时 X/Y 样本数量必须相同")
    diagonal = TensorMap.zeros_like(reference)
    diagonal_weight = 0.0
    for x_sample, y_sample, x_weight, y_weight in zip(
        x_samples, y_samples, wx, wy, strict=True
    ):
        diagonal = diagonal + (x_sample * y_sample) * (x_weight * y_weight)
        diagonal_weight += x_weight * y_weight
    denominator = sum(wx) * sum(wy) - diagonal_weight
    if denominator <= 0 or not math.isfinite(denominator):
        raise CoreContractError("cross-U 删除对角项后的分母必须为正")
    result = (sx * sy - diagonal) / denominator
    result.assert_finite()
    return result


def _learning_rate_for_name(
    core: TensorMap,
    name: str,
    learning_rates: Mapping[str, float],
) -> float:
    if core.registry is not None:
        group_id = core.registry.record(name).group_id
        if group_id is None:
            raise CoreContractError(f"eligible 参数 {name!r} 缺少 optimizer group ID")
        if group_id not in learning_rates:
            raise CoreContractError(f"缺少参数组 {group_id!r} 的实际学习率")
        return float(learning_rates[group_id])
    if name in learning_rates:
        return float(learning_rates[name])
    if "default" in learning_rates:
        return float(learning_rates["default"])
    if "*" in learning_rates:
        return float(learning_rates["*"])
    if len(learning_rates) == 1:
        return float(next(iter(learning_rates.values())))
    raise CoreContractError(f"无 registry TensorMap 无法确定 {name!r} 的学习率")


@dataclass(frozen=True, slots=True)
class EstimatorResult:
    """冻结 estimator core 与训练尺度后的公开 score。

    ``core`` 永远不含学习率和裁剪；``score`` 按 ``eta[group] * clip_factor`` 只缩放
    一次。``clip_source`` 必须明确写出 ``none``、``same_batch_mean_gradient``、
    ``independent`` 或 ``external_constant``，以免把同批随机裁剪误写成严格无偏。
    """

    estimator_name: str
    core: TensorMap
    score: TensorMap
    learning_rates: Mapping[str, float]
    clip_factor: float
    clip_source: str
    unbiasedness_claim: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.core.assert_compatible(self.score)
        self.core.assert_finite()
        self.score.assert_finite()
        if not self.estimator_name:
            raise CoreContractError("estimator_name 不能为空")
        if self.clip_source not in {
            "none",
            "same_batch_mean_gradient",
            "independent",
            "external_constant",
        }:
            raise CoreContractError(f"未知 clip_source: {self.clip_source!r}")
        if not math.isfinite(self.clip_factor) or not 0 <= self.clip_factor <= 1:
            raise CoreContractError("clip_factor 必须是 [0,1] 内有限数")
        if (
            self.clip_source == "same_batch_mean_gradient"
            and self.unbiasedness_claim != PLUGIN_SAME_BATCH_CLIP
        ):
            raise CoreContractError("同批随机 clip 只能声明为 plug-in 在线分数")
        if self.unbiasedness_claim == UNBIASED_FIXED_STATE:
            if self.clip_source != "none" or self.clip_factor != 1.0:
                raise CoreContractError("被裁剪的 estimator 不能保留 fixed-state 严格无偏声明")
            if self.estimator_name not in _STRICT_FIXED_STATE_ESTIMATORS:
                raise CoreContractError(
                    f"estimator {self.estimator_name!r} 不属于允许声明 fixed-state 无偏的核"
                )
            if self.estimator_name == "local_gradient_space_importance_u_weighted":
                required = {
                    "weights_exogenous": True,
                    "common_mean_assumption": True,
                }
                for key, expected in required.items():
                    if self.metadata.get(key) is not expected:
                        raise CoreContractError(f"加权 U 严格声明缺少 {key}=true")
                for key in ("statistical_unit", "weight_unit", "sampling_design"):
                    if not isinstance(self.metadata.get(key), str) or not self.metadata[key]:
                        raise CoreContractError(f"加权 U 严格声明缺少 {key}")
        object.__setattr__(
            self,
            "learning_rates",
            MappingProxyType({str(k): float(v) for k, v in self.learning_rates.items()}),
        )
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @classmethod
    def from_core(
        cls,
        estimator_name: str,
        core: TensorMap,
        learning_rates: Mapping[str, float],
        *,
        clip_factor: float = 1.0,
        clip_source: str = "none",
        unbiasedness_claim: str = NO_UNBIASEDNESS_CLAIM,
        metadata: Mapping[str, object] | None = None,
    ) -> "EstimatorResult":
        """按每参数组实际学习率构造 score，并实施 clip 声明防线。"""

        if not math.isfinite(float(clip_factor)) or not 0 <= float(clip_factor) <= 1:
            raise CoreContractError("clip_factor 必须是 [0,1] 内有限数")
        if clip_source == "none" and float(clip_factor) != 1.0:
            raise CoreContractError("clip_factor != 1 时必须提供真实 clip_source")
        if clip_source == "same_batch_mean_gradient":
            unbiasedness_claim = PLUGIN_SAME_BATCH_CLIP
        scaled: dict[str, torch.Tensor] = {}
        normalized_lr: dict[str, float] = {}
        for key, value in learning_rates.items():
            lr = float(value)
            if not math.isfinite(lr) or lr < 0:
                raise CoreContractError(f"学习率 {key!r} 必须为有限非负数")
            normalized_lr[str(key)] = lr
        for name, value in core.items():
            lr = _learning_rate_for_name(core, name, normalized_lr)
            scaled[name] = value * lr * float(clip_factor)
        score = TensorMap(scaled, registry=core.registry)
        return cls(
            estimator_name=estimator_name,
            core=core.clone(),
            score=score,
            learning_rates=normalized_lr,
            clip_factor=float(clip_factor),
            clip_source=clip_source,
            unbiasedness_claim=unbiasedness_claim,
            metadata=metadata or {},
        )

    @classmethod
    def from_equal_u(
        cls,
        statistics: EqualSufficientStatistics,
        learning_rates: Mapping[str, float],
        *,
        clip_factor: float = 1.0,
        clip_source: str = "none",
        metadata: Mapping[str, object] | None = None,
    ) -> "EstimatorResult":
        """由等权统计量构造 U 结果，并自动绑定正确的声明边界。"""

        merged_metadata = {
            "statistical_unit": statistics.statistical_unit,
            "sampling_design": statistics.sampling_design,
            "microbatch_count": statistics.count,
            **dict(metadata or {}),
        }
        claim = (
            UNBIASED_FIXED_STATE
            if clip_source == "none" and float(clip_factor) == 1.0
            else NO_UNBIASEDNESS_CLAIM
        )
        public_name = (
            "local_gradient_space_importance_u"
            if clip_source == "none" and float(clip_factor) == 1.0
            else "local_gradient_space_importance_u_clipped"
        )
        return cls.from_core(
            public_name,
            equal_u_importance(statistics),
            learning_rates,
            clip_factor=clip_factor,
            clip_source=clip_source,
            unbiasedness_claim=claim,
            metadata=merged_metadata,
        )

    @classmethod
    def from_weighted_u(
        cls,
        statistics: WeightedSufficientStatistics,
        learning_rates: Mapping[str, float],
        *,
        clip_factor: float = 1.0,
        clip_source: str = "none",
        metadata: Mapping[str, object] | None = None,
    ) -> "EstimatorResult":
        """由加权统计量构造 U 结果，并把全部成立假设写入 metadata。"""

        assumptions_hold = statistics.weights_exogenous and statistics.common_mean_assumption
        merged_metadata = {
            "statistical_unit": statistics.statistical_unit,
            "weight_unit": statistics.weight_unit,
            "sampling_design": statistics.sampling_design,
            "weights_exogenous": statistics.weights_exogenous,
            "common_mean_assumption": statistics.common_mean_assumption,
            "microbatch_count": statistics.count,
            **dict(metadata or {}),
        }
        claim = (
            UNBIASED_FIXED_STATE
            if assumptions_hold and clip_source == "none" and float(clip_factor) == 1.0
            else NO_UNBIASEDNESS_CLAIM
        )
        public_name = (
            "local_gradient_space_importance_u_weighted"
            if clip_source == "none" and float(clip_factor) == 1.0
            else "local_gradient_space_importance_u_clipped"
        )
        return cls.from_core(
            public_name,
            weighted_u_importance(
                statistics,
                require_unbiasedness_assumptions=assumptions_hold,
            ),
            learning_rates,
            clip_factor=clip_factor,
            clip_source=clip_source,
            unbiasedness_claim=claim,
            metadata=merged_metadata,
        )


def global_clip_factor(
    mean_gradient: TensorMap,
    max_norm: float,
    *,
    epsilon: float = 1e-12,
) -> float:
    """从全局平均梯度计算单一裁剪因子，但不修改输入张量。"""

    if not math.isfinite(max_norm) or max_norm <= 0:
        raise CoreContractError("max_norm 必须为有限正数")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise CoreContractError("epsilon 必须为有限正数")
    mean_gradient.assert_finite()
    squared_norm = sum(
        float(value.detach().to(torch.float64).square().sum().item())
        for value in mean_gradient.values()
    )
    norm = math.sqrt(squared_norm)
    return min(1.0, max_norm / (norm + epsilon))
