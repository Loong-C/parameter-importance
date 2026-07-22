"""固定参数状态梯度源的最小协议。

Stage 2 的统计对象要求模型参数、buffer、模型模式以及模型/全局 RNG 在一次
研究单元内保持不变。这里刻意不规定模型框架：provider 可以返回 NumPy
数组、Torch 张量，或由核心层认识的其他张量值。实验层只依赖映射的键和
shape 稳定，并把真正的公式计算交给核心估计器适配器。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping, Protocol, Sequence, TypeVar, runtime_checkable


TensorValue = TypeVar("TensorValue")


@dataclass(frozen=True, slots=True)
class GradientBatch:
    """一次固定状态梯度求值的结果。

    Parameters
    ----------
    gradients:
        以 canonical parameter name 为键的平均梯度。所有值必须来自同一
        loss reduction，且映射顺序不得被解释为参数身份；参数身份由名称和
        ``registry_hash`` 共同确定。
    statistical_weight:
        本次平均梯度代表的统计权重，例如 sequence 数或有效目标 token 数。
        它必须严格为正；加权 U 公式需要这个原始权重，而不是归一化后的比例。
    statistical_unit:
        一次传给估计器的独立统计单元名称，例如 ``target_token_microbatch``。
        该字段描述“随机变量是什么”，不能由 runner 根据张量 shape 猜测。
    weight_unit:
        ``statistical_weight`` 的物理/统计单位，例如 ``effective_target_tokens``。
        即使当前各批权重恰好相等也必须填写，以免恢复后换成非等权数据时静默
        改变 estimand。
    sampling_design:
        产生这些统计单元的抽样设计。它必须说明独立性、是否放回以及分组方式；
        字符串本身是审计记录，不会被实现“猜成”某个更强假设。
    weights_exogenous:
        权重是否相对于被估计的梯度随机变量外生。加权 U 若要保留严格无偏性
        声明，该值必须由 provider 明确给出 ``True``。
    common_mean_assumption:
        不同权重统计单元是否共享同一目标均值。该假设与权重外生性共同构成
        加权 U 的 fail-closed 前提；缺失或 ``False`` 时 runner 不得代为补真。
    sample_ids:
        构成本次求值的有限经验分布样本 ID。不同 draw 允许命中同一 sample
        ID，所以这里不去重。
    loss:
        可选的同 reduction 标量损失，仅用于诊断，不参与估计器公式。
    """

    gradients: Mapping[str, object]
    statistical_weight: float
    statistical_unit: str
    weight_unit: str
    sampling_design: str
    weights_exogenous: bool
    common_mean_assumption: bool
    sample_ids: tuple[Hashable, ...]
    loss: float | None = None

    def __post_init__(self) -> None:
        if not self.gradients:
            raise ValueError("gradients 不能为空")
        if not math.isfinite(self.statistical_weight) or self.statistical_weight <= 0:
            raise ValueError("statistical_weight 必须是严格为正的有限数")
        for field_name in ("statistical_unit", "weight_unit", "sampling_design"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
        if type(self.weights_exogenous) is not bool:
            raise TypeError("weights_exogenous 必须是显式 bool")
        if type(self.common_mean_assumption) is not bool:
            raise TypeError("common_mean_assumption 必须是显式 bool")
        if not self.sample_ids:
            raise ValueError("sample_ids 不能为空")

    @property
    def weighting_assumptions(self) -> dict[str, object]:
        """返回可直接写入 artifact 的完整统计假设副本。

        返回新字典而不是内部可变对象，调用方可以安全地把它规范化或冻结。
        五个字段始终齐全，因此任何 estimator adapter 都不需要用默认值填空。
        """

        return {
            "statistical_unit": self.statistical_unit,
            "weight_unit": self.weight_unit,
            "sampling_design": self.sampling_design,
            "weights_exogenous": self.weights_exogenous,
            "common_mean_assumption": self.common_mean_assumption,
        }


@runtime_checkable
class GradientProvider(Protocol):
    """能按 draw 顺序返回平均梯度的数据源协议。

    provider 必须保留输入顺序和 sample-ID 碰撞。有放回抽样中的两个 draw
    即使指向同一 sample ID，也仍是两个统计 draw，实验层不会替 provider
    去重。
    """

    @property
    def registry_hash(self) -> str:
        """返回 canonical parameter registry 的稳定摘要。"""

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """返回 canonical 参数名顺序。"""

    @property
    def statistical_unit(self) -> str:
        """返回 provider 所产生独立梯度统计单元的名称。"""

    @property
    def weight_unit(self) -> str:
        """返回 ``GradientBatch.statistical_weight`` 的单位。"""

    @property
    def sampling_design(self) -> str:
        """返回固定的抽样设计声明；不得由 runner 推断或增强。"""

    @property
    def weights_exogenous(self) -> bool:
        """返回权重外生性假设是否被 provider 明确声明为成立。"""

    @property
    def common_mean_assumption(self) -> bool:
        """返回不同权重统计单元共享同一目标均值的显式假设。"""

    def gradient(self, draws: Sequence[object]) -> GradientBatch:
        """计算 ``draws`` 对应的平均梯度，不推进训练状态。"""


@runtime_checkable
class FixedStateGradientProvider(GradientProvider, Protocol):
    """带固定状态证明的梯度源协议。

    ``state_digest`` 只覆盖会改变梯度函数的模型状态，不覆盖 provider 的调用
    计数、性能计时等观测元数据。runner 应在单元前后比较该摘要。
    """

    @property
    def fixed_state_id(self) -> str:
        """返回不可变 checkpoint/人工分布状态的逻辑 ID。"""

    def state_digest(self) -> str:
        """返回当前模型状态摘要；只读调用前后必须完全相同。"""

    def assert_unchanged(self, expected_digest: str) -> None:
        """若当前状态与 ``expected_digest`` 不一致则抛出异常。"""
