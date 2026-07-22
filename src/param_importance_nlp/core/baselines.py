"""剪枝与相关性比较所需的经典参数级基线。

这些函数与主 U-statistic 完全分离，避免分析层根据缺失 artifact 静默回退。每个
函数都要求明确输入并返回同坐标 ``TensorMap``；调用方负责在报告中保留基线名称
和数据来源。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .errors import CoreContractError
from .tensors import TensorMap


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
