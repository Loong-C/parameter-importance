"""剪枝与相关性比较所需的经典参数级基线。

这些函数与主 U-statistic 完全分离，避免分析层根据缺失 artifact 静默回退。每个
函数都要求明确输入并返回同坐标 ``TensorMap``；调用方负责在报告中保留基线名称
和数据来源。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .errors import CoreContractError
from .tensors import TensorMap


@dataclass(frozen=True, slots=True)
class BaselineProduction:
    """一次基线生产的确定性结果。

    ``scores`` 只包含数学输入足够、已经通过有限性和坐标一致性检查的基线；
    ``unavailable`` 则逐方法给出不能生产的机器原因。调用方不得把 unavailable
    方法补零后伪装成可用 artifact。
    """

    scores: Mapping[str, TensorMap]
    unavailable: Mapping[str, str]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.scores:
            raise CoreContractError("BaselineProduction 至少需要一个可用 score")
        if set(self.scores).intersection(self.unavailable):
            raise CoreContractError("同一 baseline 不能同时可用和 unavailable")
        first = next(iter(self.scores.values()))
        for name, score in self.scores.items():
            if not isinstance(name, str) or not name:
                raise CoreContractError("baseline method 必须是非空字符串")
            first.assert_compatible(score)
            score.assert_finite()
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "unavailable", MappingProxyType(dict(self.unavailable)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _state_tensor_map(
    state: Mapping[str, object],
    field: str,
    template: TensorMap,
) -> TensorMap:
    value = state.get(field)
    if not isinstance(value, Mapping) or not value:
        raise CoreContractError(f"importance accumulator 缺少 tensor field: {field}")
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in value.items()):
        raise CoreContractError(f"importance accumulator field {field} 不是 tensor mapping")
    result = TensorMap(value)  # type: ignore[arg-type]
    template.assert_compatible(result)
    result.assert_finite()
    return result


def produce_baseline_scores(
    parameters: TensorMap,
    accumulator_state: Mapping[str, object],
    *,
    estimator_name: str,
    per_unit_gradients: Sequence[TensorMap] | None = None,
    per_unit_weights: Sequence[float | int] | None = None,
    si_gradients: Sequence[TensorMap] | None = None,
    si_data_updates: Sequence[TensorMap] | None = None,
    optimizer_type: str | None = None,
    optimizer_momentum: float = 0.0,
    clip_mode: str = "none",
    si_xi: float = 1e-3,
) -> BaselineProduction:
    """把训练累计器与可选梯度观测生产为 Stage 7 可消费的全部基线。

    固定提供 ``magnitude``、``movement`` 与 ``raw``；累计器的主 estimator 只会
    发布成实际使用的 ``u`` 或 ``double``，不会用同一张量冒充另一种方法。
    empirical Fisher 需要显式的统计单位梯度。SI 优先使用完整梯度/数据更新历史；
    若训练合同严格为无 momentum、无裁剪 SGD，则累计 ``raw=Σηg²`` 与 SI 分子
    ``-ΣgΔθ_data`` 相等，可在记录该假设后精确恢复。其他 optimizer 不作猜测。

    参数：
        parameters: checkpoint 的最终 eligible 参数，shape/name 决定坐标合同。
        accumulator_state: :class:`ImportanceAccumulator.state_dict` 的 v2 primitive tree。
        estimator_name: 训练实际使用的 ``u``、``weighted_u``、``double`` 或 ``raw``。
        per_unit_gradients: empirical Fisher 的统计单位梯度，而非 batch 均值平方替代物。
        si_gradients/si_data_updates: 同一步对齐的梯度和纯数据更新。
    """

    if accumulator_state.get("version") not in {1, 2}:
        raise CoreContractError("baseline producer 只接受 accumulator state v1/v2")
    parameters.assert_finite()
    magnitude = final_magnitude(parameters).to(dtype=torch.float64)
    movement = _state_tensor_map(accumulator_state, "data_movement", parameters).to(
        dtype=torch.float64
    )
    raw = _state_tensor_map(accumulator_state, "raw", parameters).to(dtype=torch.float64)
    positive = _state_tensor_map(accumulator_state, "positive", parameters).to(
        dtype=torch.float64
    )
    negative = _state_tensor_map(accumulator_state, "negative_mass", parameters).to(
        dtype=torch.float64
    )
    main_signed = positive - negative

    scores: dict[str, TensorMap] = {
        "magnitude": magnitude,
        "movement": movement,
        "raw": raw,
    }
    unavailable: dict[str, str] = {}
    normalized_estimator = estimator_name.casefold()
    if normalized_estimator in {"u", "weighted_u"}:
        scores["u"] = main_signed
        unavailable["double"] = "training_estimator_not_double"
    elif normalized_estimator == "double":
        scores["double"] = main_signed
        unavailable["u"] = "training_estimator_not_u"
    elif normalized_estimator == "raw":
        unavailable["u"] = "training_estimator_raw"
        unavailable["double"] = "training_estimator_raw"
    else:
        raise CoreContractError(f"不支持的训练 estimator_name: {estimator_name!r}")

    if per_unit_gradients:
        scores["empirical_fisher"] = empirical_fisher(
            per_unit_gradients,
            weights=per_unit_weights,
            accumulation_dtype=torch.float64,
        )
    else:
        unavailable["empirical_fisher"] = "per_unit_gradients_unavailable"

    si_derivation = "unavailable"
    if si_gradients is not None or si_data_updates is not None:
        if si_gradients is None or si_data_updates is None:
            raise CoreContractError("SI gradients 与 data updates 必须同时提供")
        initial = _state_tensor_map(accumulator_state, "initial_parameters", parameters)
        final = _state_tensor_map(accumulator_state, "last_parameters", parameters)
        scores["si"] = synaptic_intelligence(
            si_gradients,
            si_data_updates,
            initial,
            final,
            xi=si_xi,
            accumulation_dtype=torch.float64,
        )
        si_derivation = "explicit_gradient_update_history"
    elif (
        optimizer_type is not None
        and optimizer_type.casefold() == "sgd"
        and float(optimizer_momentum) == 0.0
        and clip_mode == "none"
        and accumulator_state.get("has_initial_parameters") is True
    ):
        initial = _state_tensor_map(accumulator_state, "initial_parameters", parameters).to(
            dtype=torch.float64
        )
        final = _state_tensor_map(accumulator_state, "last_parameters", parameters).to(
            dtype=torch.float64
        )
        denominator = (final - initial).map(lambda value: value.square() + float(si_xi))
        scores["si"] = raw.zip_map(denominator, torch.div)
        si_derivation = "exact_unclipped_zero_momentum_sgd_from_raw"
    else:
        unavailable["si"] = "gradient_update_history_unavailable"

    return BaselineProduction(
        scores=scores,
        unavailable=unavailable,
        metadata={
            "estimator_name": estimator_name,
            "empirical_fisher_statistical_unit": (
                None if not per_unit_gradients else "declared_per_unit_gradient"
            ),
            "si_derivation": si_derivation,
            "si_xi": float(si_xi),
        },
    )


def final_magnitude(parameters: TensorMap) -> TensorMap:
    """最终参数幅值 ``abs(theta_T)``。"""

    parameters.assert_finite()
    return parameters.map(torch.abs)


def cumulative_movement(
    updates: Sequence[TensorMap],
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> TensorMap:
    """累计绝对数据位移 ``sum_t abs(delta_theta_t_data)``。"""

    if not updates:
        raise CoreContractError("cumulative_movement 至少需要一个 update")
    if accumulation_dtype not in {torch.float32, torch.float64}:
        raise CoreContractError("movement accumulation dtype 必须为 FP32/FP64")
    result = TensorMap.zeros_like(updates[0], dtype=accumulation_dtype)
    for update in updates:
        result.assert_compatible(update)
        update.assert_finite()
        converted = update.to(dtype=accumulation_dtype)
        converted.assert_finite()
        result = result + converted.map(torch.abs)
    return result


def net_movement(initial: TensorMap, final: TensorMap) -> TensorMap:
    """首尾净位移 ``abs(theta_T-theta_0)``。"""

    initial.assert_compatible(final, require_dtype_device=True)
    initial.assert_finite()
    final.assert_finite()
    return (final - initial).map(torch.abs)


def empirical_fisher(
    per_unit_gradients: Sequence[TensorMap],
    *,
    weights: Sequence[float | int] | None = None,
    accumulation_dtype: torch.dtype = torch.float64,
) -> TensorMap:
    """经验 Fisher 对角 ``sum_i w_i*g_i**2 / sum_i w_i``。

    输入必须是评价统计单元的梯度，而不是 batch mean 的平方；后者会改变数学对象。
    """

    if not per_unit_gradients:
        raise CoreContractError("empirical_fisher 至少需要一个梯度统计单元")
    numeric_weights = (
        [1.0] * len(per_unit_gradients)
        if weights is None
        else [float(weight) for weight in weights]
    )
    if len(numeric_weights) != len(per_unit_gradients):
        raise CoreContractError("Fisher weights 与梯度数量不一致")
    if any(not math.isfinite(weight) or weight <= 0 for weight in numeric_weights):
        raise CoreContractError("Fisher weights 必须为有限正数")
    result = TensorMap.zeros_like(per_unit_gradients[0], dtype=accumulation_dtype)
    for gradient, weight in zip(per_unit_gradients, numeric_weights, strict=True):
        result.assert_compatible(gradient)
        gradient.assert_finite()
        converted = gradient.to(dtype=accumulation_dtype)
        converted.assert_finite()
        result = result + converted.map(torch.square) * weight
    return result / sum(numeric_weights)


def synaptic_intelligence(
    gradients: Sequence[TensorMap],
    data_updates: Sequence[TensorMap],
    initial_parameters: TensorMap,
    final_parameters: TensorMap,
    *,
    xi: float,
    accumulation_dtype: torch.dtype = torch.float64,
) -> TensorMap:
    """计算 SI 基线 ``[ -sum(g_t*delta_t) ]_+ / ((theta_T-theta_0)^2+xi)``。

    ``data_updates`` 不应包含解耦 weight decay；否则该基线与主 data movement 的
    比较口径不一致。``xi`` 是 SI 定义中的稳定项，不用于伪造零质量统计量。
    """

    if not gradients or len(gradients) != len(data_updates):
        raise CoreContractError("SI gradients/data_updates 必须非空且数量一致")
    if not math.isfinite(xi) or xi <= 0:
        raise CoreContractError("SI xi 必须为有限正数")
    initial_parameters.assert_compatible(final_parameters)
    numerator = TensorMap.zeros_like(initial_parameters, dtype=accumulation_dtype)
    for gradient, update in zip(gradients, data_updates, strict=True):
        initial_parameters.assert_compatible(gradient)
        initial_parameters.assert_compatible(update)
        gradient.assert_finite()
        update.assert_finite()
        g = gradient.to(dtype=accumulation_dtype)
        delta = update.to(dtype=accumulation_dtype)
        numerator = numerator - g * delta
    task_delta = final_parameters.to(dtype=accumulation_dtype) - initial_parameters.to(
        dtype=accumulation_dtype
    )
    numerator = numerator.map(lambda value: value.clamp_min(0))
    denominator = task_delta.map(lambda value: value.square() + xi)
    result = numerator.zip_map(denominator, torch.div)
    result.assert_finite()
    return result
