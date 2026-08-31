"""Microbatch 梯度的可合并充分统计量。

等权统计保存 ``S1=sum(g_m)`` 与 ``S2=sum(g_m**2)``。加权统计保存
``G1=sum(w_m*g_m)``、``G2=sum(w_m**2*g_m**2)``、``N1=sum(w_m)`` 和
``N2=sum(w_m**2)``。所有转换到 accumulation dtype 的动作都在构造时显式
执行并记录，避免 estimator 内部发生不可追溯的隐式精度变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import torch

from ..contracts.immutable import freeze_json_mapping
from .errors import CoreContractError, NumericalError
from .tensors import TensorMap


def _validate_accumulation_dtype(dtype: torch.dtype) -> None:
    if dtype not in {torch.float32, torch.float64}:
        raise CoreContractError("充分统计量 accumulation dtype 必须为 FP32 或 FP64")


def _validate_tensor_map_dtype(
    value: TensorMap,
    *,
    expected: torch.dtype,
    field: str,
) -> None:
    """确认充分统计张量与声明的累计精度逐坐标完全一致。

    ``accumulation_dtype`` 是 artifact 的数值语义，而不是提示。若只校验声明
    本身，调用方就能构造“声明 FP32、实际 FP64”的对象，后续合并与误差比较
    会在没有 provenance 的情况下改变精度。因此这里不做隐式转换，只给出
    第一个不一致坐标并 fail-closed。
    """

    for name, tensor in value.items():
        if tensor.dtype != expected:
            raise CoreContractError(
                f"{field}[{name!r}] dtype={tensor.dtype}，"
                f"与 accumulation_dtype={expected} 不一致"
            )


def _ensure_samples(samples: Sequence[TensorMap]) -> None:
    if not samples:
        raise CoreContractError("至少需要一个梯度统计单元")
    reference = samples[0]
    reference.assert_finite()
    for sample in samples[1:]:
        reference.assert_compatible(sample)
        sample.assert_finite()


@dataclass(frozen=True, slots=True)
class EqualSufficientStatistics:
    """等权 microbatch 梯度的不可变充分统计量。"""

    count: int
    s1: TensorMap
    s2: TensorMap
    accumulation_dtype: torch.dtype
    statistical_unit: str = "microbatch"
    sampling_design: str = "iid_disjoint_microbatches"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise CoreContractError("EqualSufficientStatistics.count 必须为正")
        _validate_accumulation_dtype(self.accumulation_dtype)
        self.s1.assert_compatible(self.s2)
        _validate_tensor_map_dtype(
            self.s1, expected=self.accumulation_dtype, field="s1"
        )
        _validate_tensor_map_dtype(
            self.s2, expected=self.accumulation_dtype, field="s2"
        )
        self.s1.assert_finite()
        self.s2.assert_finite()
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[TensorMap],
        *,
        accumulation_dtype: torch.dtype = torch.float32,
        statistical_unit: str = "microbatch",
        sampling_design: str = "iid_disjoint_microbatches",
        metadata: Mapping[str, object] | None = None,
    ) -> "EqualSufficientStatistics":
        """以流式等价公式累计 S1/S2；输入顺序不影响数学结果。"""

        _validate_accumulation_dtype(accumulation_dtype)
        _ensure_samples(samples)
        first = samples[0].to(dtype=accumulation_dtype)
        s1_values = {name: torch.zeros_like(value) for name, value in first.items()}
        s2_values = {name: torch.zeros_like(value) for name, value in first.items()}
        for sample in samples:
            converted = sample.to(dtype=accumulation_dtype)
            for name, gradient in converted.items():
                s1_values[name] = s1_values[name] + gradient
                s2_values[name] = s2_values[name] + gradient.square()
        return cls(
            len(samples),
            TensorMap(s1_values, registry=first.registry),
            TensorMap(s2_values, registry=first.registry),
            accumulation_dtype,
            statistical_unit,
            sampling_design,
            metadata or {},
        )

    @property
    def mean_gradient(self) -> TensorMap:
        return self.s1 / self.count

    def merge(self, other: "EqualSufficientStatistics") -> "EqualSufficientStatistics":
        """按加法归并 shard，不重新读取原始梯度。"""

        if self.accumulation_dtype != other.accumulation_dtype:
            raise CoreContractError("不同 accumulation dtype 的统计量不能隐式 merge")
        if self.statistical_unit != other.statistical_unit:
            raise CoreContractError("不同 statistical_unit 的统计量不能 merge")
        if self.sampling_design != other.sampling_design:
            raise CoreContractError("不同 sampling_design 的统计量不能 merge")
        return EqualSufficientStatistics(
            self.count + other.count,
            self.s1 + other.s1,
            self.s2 + other.s2,
            self.accumulation_dtype,
            self.statistical_unit,
            self.sampling_design,
            {"merged": True},
        )


@dataclass(frozen=True, slots=True)
class WeightedSufficientStatistics:
    """不等有效单元数 microbatch 的加权充分统计量。

    严格无偏性只在权重相对被估计梯度为外生、各 microbatch 梯度具有共同总体
    均值且统计单元满足声明的 sampling design 时成立。该假设不会由数值代码自动
    推断，必须作为结构化字段随 artifact 保存。
    """

    count: int
    g1: TensorMap
    g2: TensorMap
    n1: float
    n2: float
    accumulation_dtype: torch.dtype
    statistical_unit: str
    weight_unit: str
    sampling_design: str
    weights_exogenous: bool
    common_mean_assumption: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise CoreContractError("WeightedSufficientStatistics.count 必须为正")
        _validate_accumulation_dtype(self.accumulation_dtype)
        if not math.isfinite(self.n1) or not math.isfinite(self.n2):
            raise NumericalError("权重充分统计量 n1/n2 必须有限")
        if self.n1 <= 0 or self.n2 <= 0:
            raise CoreContractError("权重必须为正，因此 n1/n2 必须为正")
        self.g1.assert_compatible(self.g2)
        _validate_tensor_map_dtype(
            self.g1, expected=self.accumulation_dtype, field="g1"
        )
        _validate_tensor_map_dtype(
            self.g2, expected=self.accumulation_dtype, field="g2"
        )
        self.g1.assert_finite()
        self.g2.assert_finite()
        for field_name in ("statistical_unit", "weight_unit", "sampling_design"):
            if not getattr(self, field_name):
                raise CoreContractError(f"{field_name} 不能为空")
        if type(self.weights_exogenous) is not bool:
            raise CoreContractError("weights_exogenous 必须是显式 bool")
        if type(self.common_mean_assumption) is not bool:
            raise CoreContractError("common_mean_assumption 必须是显式 bool")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[TensorMap],
        weights: Sequence[float | int],
        *,
        accumulation_dtype: torch.dtype = torch.float32,
        statistical_unit: str,
        weight_unit: str,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
        metadata: Mapping[str, object] | None = None,
    ) -> "WeightedSufficientStatistics":
        _validate_accumulation_dtype(accumulation_dtype)
        _ensure_samples(samples)
        if len(samples) != len(weights):
            raise CoreContractError("weights 数量必须与梯度统计单元数量一致")
        numeric_weights = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight <= 0 for weight in numeric_weights):
            raise CoreContractError("每个 microbatch 权重必须为有限正数")

        first = samples[0].to(dtype=accumulation_dtype)
        g1_values = {name: torch.zeros_like(value) for name, value in first.items()}
        g2_values = {name: torch.zeros_like(value) for name, value in first.items()}
        for sample, weight in zip(samples, numeric_weights, strict=True):
            converted = sample.to(dtype=accumulation_dtype)
            for name, gradient in converted.items():
                g1_values[name] = g1_values[name] + weight * gradient
                g2_values[name] = g2_values[name] + weight**2 * gradient.square()
        return cls(
            count=len(samples),
            g1=TensorMap(g1_values, registry=first.registry),
            g2=TensorMap(g2_values, registry=first.registry),
            n1=sum(numeric_weights),
            n2=sum(weight**2 for weight in numeric_weights),
            accumulation_dtype=accumulation_dtype,
            statistical_unit=statistical_unit,
            weight_unit=weight_unit,
            sampling_design=sampling_design,
            weights_exogenous=weights_exogenous,
            common_mean_assumption=common_mean_assumption,
            metadata=metadata or {},
        )

    @property
    def denominator(self) -> float:
        return self.n1**2 - self.n2

    @property
    def mean_gradient(self) -> TensorMap:
        return self.g1 / self.n1
