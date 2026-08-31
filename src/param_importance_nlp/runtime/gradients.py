"""梯度 scale/unscale、有限性检查与全局裁剪的显式生命周期。

AMP 或外部 loss scaling 会先产生 ``scaled`` 梯度。参数重要性公式、全局范数和
optimizer 都只能消费同一份已 ``unscaled`` 的梯度；因此本模块把顺序写成不可
跳步的状态机，而不是依赖调用者记住若干布尔标志：

``SCALED -> UNSCALED -> FINITE -> CLIPPED``

若有限性检查失败，则进入 ``SKIPPED``，之后既不能裁剪，也不能回写参数的
``.grad``。``grad=None`` 被作为稳定的缺失集合记录，不会擅自补零；稀疏梯度在
收集边界直接拒绝。所有范数都先转换到 CPU/FP64 标量语义，实际梯度仍保持声明
的 dtype/device。状态对象对输入和输出均做 clone，避免调用方原地修改已经审计
过的快照。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, Mapping


class GradientPhase(StrEnum):
    """一次梯度尝试允许出现的阶段。"""

    SCALED = "SCALED"
    UNSCALED = "UNSCALED"
    FINITE = "FINITE"
    CLIPPED = "CLIPPED"
    SKIPPED = "SKIPPED"


def _require_scale(value: float) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("GRADIENT_SCALE_MUST_BE_FINITE_POSITIVE")
    return scale


def _clone_dense_gradients(
    gradients: Mapping[str, Any | None],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    """验证并复制梯度；返回非空梯度 mapping 与 ``grad=None`` 名称。"""

    import torch

    if not gradients:
        raise ValueError("GRADIENT_MAPPING_EMPTY")
    copied: dict[str, Any] = {}
    missing: list[str] = []
    for name, gradient in gradients.items():
        if not isinstance(name, str) or not name:
            raise TypeError("GRADIENT_NAME_MUST_BE_NONEMPTY_STRING")
        if gradient is None:
            missing.append(name)
            continue
        if not isinstance(gradient, torch.Tensor):
            raise TypeError(f"GRADIENT_NOT_TENSOR:{name}")
        if gradient.layout is not torch.strided or gradient.is_sparse:
            raise TypeError(f"SPARSE_GRADIENT_UNSUPPORTED:{name}")
        if not gradient.is_floating_point():
            raise TypeError(f"GRADIENT_DTYPE_NOT_FLOATING:{name}")
        copied[name] = gradient.detach().clone()
    if not copied:
        raise ValueError("ALL_GRADIENTS_ARE_NONE")
    return MappingProxyType(dict(sorted(copied.items()))), tuple(sorted(missing))


def _all_finite(gradients: Mapping[str, Any]) -> bool:
    import torch

    return all(bool(torch.isfinite(value).all()) for value in gradients.values())


def _global_norm_fp64(gradients: Mapping[str, Any]) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64, device="cpu")
    for gradient in gradients.values():
        total.add_(
            gradient.detach().to(device="cpu", dtype=torch.float64).square().sum()
        )
    return math.sqrt(float(total.item()))


@dataclass(frozen=True, slots=True)
class GradientAttempt:
    """一个不可变梯度快照及其数值处理阶段。

    不应直接调用构造器；使用 :meth:`capture` 收集梯度。``gradient_scale`` 是
    loss/梯度实际使用的缩放因子，不进入参数坐标身份。``global_norm`` 始终指
    未裁剪、已 unscale 梯度的 FP64 全局 L2 范数。
    """

    phase: GradientPhase
    gradient_scale: float
    _gradients: Mapping[str, Any]
    missing_names: tuple[str, ...]
    global_norm: float | None = None
    clip_factor: float | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        scale = _require_scale(self.gradient_scale)
        object.__setattr__(self, "gradient_scale", scale)
        if not isinstance(self.phase, GradientPhase):
            object.__setattr__(self, "phase", GradientPhase(self.phase))
        if set(self._gradients).intersection(self.missing_names):
            raise ValueError("GRADIENT_PRESENT_AND_MISSING_NAME_OVERLAP")
        # 构造边界也复制一次，防止扩展调用者绕过 capture 后保留可变引用。
        copied, unexpectedly_missing = _clone_dense_gradients(self._gradients)
        if unexpectedly_missing:
            raise ValueError("GRADIENT_SNAPSHOT_CANNOT_CONTAIN_NONE")
        object.__setattr__(self, "_gradients", copied)
        object.__setattr__(self, "missing_names", tuple(sorted(self.missing_names)))
        if self.global_norm is not None and (
            not math.isfinite(self.global_norm) or self.global_norm < 0
        ):
            raise ValueError("GRADIENT_GLOBAL_NORM_INVALID")
        if self.clip_factor is not None and (
            not math.isfinite(self.clip_factor) or not 0 <= self.clip_factor <= 1
        ):
            raise ValueError("GRADIENT_CLIP_FACTOR_INVALID")
        if self.phase is GradientPhase.SKIPPED:
            if not self.skip_reason:
                raise ValueError("GRADIENT_SKIP_REASON_REQUIRED")
        elif self.skip_reason is not None:
            raise ValueError("GRADIENT_SKIP_REASON_ONLY_FOR_SKIPPED")

    @classmethod
    def capture(
        cls,
        gradients: Mapping[str, Any | None],
        *,
        gradient_scale: float = 1.0,
        scaled: bool = False,
    ) -> "GradientAttempt":
        """复制一次 ``name -> grad`` 快照，并声明它是否仍处于 scaled 空间。

        Parameters
        ----------
        gradients:
            稳定 canonical name 到 dense 浮点梯度或 ``None`` 的映射。
        gradient_scale:
            生成当前梯度的正有限缩放因子。即使 ``scaled=False`` 也要求记录真实
            值；这种情况下它仅作为 lineage 元数据，不会再次相除。
        scaled:
            ``True`` 表示必须先调用 :meth:`unscale`；``False`` 表示上游已经按
            同一因子完成 unscale，可直接进入有限性检查。
        """

        copied, missing = _clone_dense_gradients(gradients)
        return cls(
            phase=GradientPhase.SCALED if scaled else GradientPhase.UNSCALED,
            gradient_scale=_require_scale(gradient_scale),
            _gradients=copied,
            missing_names=missing,
        )

    @property
    def gradients(self) -> Mapping[str, Any]:
        """返回防御性复制；调用方不能污染已审计状态。"""

        copied, missing = _clone_dense_gradients(self._gradients)
        assert not missing
        return copied

    def unscale(self) -> "GradientAttempt":
        """恰好一次除以 ``gradient_scale``；重复或越序调用立即失败。"""

        if self.phase is not GradientPhase.SCALED:
            raise ValueError(f"GRADIENT_UNSCALE_FROM_INVALID_PHASE:{self.phase}")
        values = {
            name: gradient / self.gradient_scale
            for name, gradient in self._gradients.items()
        }
        return GradientAttempt(
            phase=GradientPhase.UNSCALED,
            gradient_scale=self.gradient_scale,
            _gradients=values,
            missing_names=self.missing_names,
        )

    def check_finite(self) -> "GradientAttempt":
        """在 unscale 之后检查全部坐标；失败返回 ``SKIPPED`` 状态。"""

        if self.phase is not GradientPhase.UNSCALED:
            raise ValueError(f"GRADIENT_FINITE_CHECK_FROM_INVALID_PHASE:{self.phase}")
        if not _all_finite(self._gradients):
            return replace(
                self,
                phase=GradientPhase.SKIPPED,
                skip_reason="NONFINITE_UNSCALED_GRADIENT",
            )
        return replace(
            self,
            phase=GradientPhase.FINITE,
            global_norm=_global_norm_fp64(self._gradients),
        )

    def clip(self, max_norm: float) -> "GradientAttempt":
        """按单一全局因子裁剪有限梯度，因子不会隐式进入 estimator core。

        ``max_norm=0`` 合法并产生全零更新；负值、NaN/Inf 直接拒绝。返回对象的
        ``global_norm`` 仍是裁剪前范数，便于记录 same-batch clip 来源。
        """

        if self.phase is not GradientPhase.FINITE:
            raise ValueError(f"GRADIENT_CLIP_FROM_INVALID_PHASE:{self.phase}")
        maximum = float(max_norm)
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError("GRADIENT_MAX_NORM_MUST_BE_FINITE_NONNEGATIVE")
        assert self.global_norm is not None
        factor = min(1.0, maximum / (self.global_norm + 1e-12))
        values = {
            name: gradient * factor for name, gradient in self._gradients.items()
        }
        return GradientAttempt(
            phase=GradientPhase.CLIPPED,
            gradient_scale=self.gradient_scale,
            _gradients=values,
            missing_names=self.missing_names,
            global_norm=self.global_norm,
            clip_factor=factor,
        )

    def install(self, named_parameters: Mapping[str, Any]) -> None:
        """在全部 shape/device/dtype 验证后原子式回写 ``Parameter.grad``。

        仅 ``FINITE``/``CLIPPED`` 状态可以回写。先验证所有参数，再执行 copy，
        因而 shape 或名称错误不会造成半写入；记录为 ``grad=None`` 的参数保持
        ``None``。本方法不调用 optimizer，也不改变参数值。
        """

        import torch

        if self.phase not in {GradientPhase.FINITE, GradientPhase.CLIPPED}:
            raise ValueError(f"GRADIENT_INSTALL_FROM_INVALID_PHASE:{self.phase}")
        expected = set(self._gradients).union(self.missing_names)
        if set(named_parameters) != expected:
            raise ValueError("GRADIENT_PARAMETER_NAME_SET_MISMATCH")
        staged: dict[str, Any] = {}
        for name, gradient in self._gradients.items():
            parameter = named_parameters[name]
            if not isinstance(parameter, torch.Tensor):
                raise TypeError(f"GRADIENT_TARGET_NOT_TENSOR:{name}")
            if tuple(parameter.shape) != tuple(gradient.shape):
                raise ValueError(f"GRADIENT_TARGET_SHAPE_MISMATCH:{name}")
            staged[name] = gradient.to(device=parameter.device, dtype=parameter.dtype)
        for name in self.missing_names:
            if not isinstance(named_parameters[name], torch.Tensor):
                raise TypeError(f"GRADIENT_TARGET_NOT_TENSOR:{name}")
        for name, parameter in named_parameters.items():
            if name in self.missing_names:
                parameter.grad = None
                continue
            converted = staged[name]
            if parameter.grad is None:
                parameter.grad = converted.clone()
            else:
                parameter.grad.copy_(converted)


__all__ = ["GradientAttempt", "GradientPhase"]
