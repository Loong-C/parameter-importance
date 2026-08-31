"""优化器边界、全局裁剪和实际位移分解。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """一次成功 optimizer step 的逐参数位移。

    ``data_delta`` 排除 AdamW 的 decoupled weight decay；``total_delta`` 是实际
    端点差。对于含 coupled weight decay 的 SGD，本实现拒绝把两者伪装成可分离。
    """

    total_delta: dict[str, Any]
    data_delta: dict[str, Any]
    weight_decay_delta: dict[str, Any]
    learning_rates: dict[str, float]


def compute_global_clip_factor(
    gradients: Mapping[str, Any], max_norm: float | None, *, eps: float = 1e-12
) -> tuple[float, float]:
    """以 FP64 标量计算全局 L2 范数，返回 ``(norm, factor)``。

    该 factor 若由同批随机梯度得到，只能用于 plug-in 在线分数，不能让 U 核心
    获得额外无偏性声明。调用方负责在 score 中至多乘一次。
    """

    import torch

    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("CLIP_EPS_MUST_BE_FINITE_POSITIVE")
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("CLIP_EPS_MUST_BE_FINITE_POSITIVE")
    if max_norm is not None:
        if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
            raise TypeError("CLIP_MAX_NORM_MUST_BE_FINITE_NONNEGATIVE")
        max_norm = float(max_norm)
        if not math.isfinite(max_norm) or max_norm < 0.0:
            raise ValueError("CLIP_MAX_NORM_MUST_BE_FINITE_NONNEGATIVE")

    total = torch.zeros((), dtype=torch.float64)
    for name, gradient in gradients.items():
        if gradient is None:
            continue
        if not isinstance(gradient, torch.Tensor):
            raise TypeError(f"GRADIENT_NOT_TENSOR:{name}")
        if gradient.is_sparse:
            raise TypeError("SPARSE_GRADIENT_UNSUPPORTED")
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"NONFINITE_GRADIENT:{name}")
        total += gradient.detach().to(dtype=torch.float64).square().sum()
    norm = math.sqrt(float(total.item()))
    if not math.isfinite(norm):  # 前述逐张量检查后的防御性总量边界。
        raise ValueError("NONFINITE_GLOBAL_GRADIENT_NORM")
    if max_norm is None:
        return norm, 1.0
    factor = min(1.0, max_norm / (norm + eps))
    return norm, factor


class OptimizerBridge:
    """把 PyTorch optimizer step 转换成可审计的端点位移。"""

    def __init__(self, named_parameters: Mapping[str, Any], optimizer: Any) -> None:
        import torch

        supplied_parameters = dict(named_parameters)
        self.optimizer = optimizer
        # 在任何参数发生变化之前冻结支持边界。若把检查放到 ``step`` 之后，
        # 不受支持的优化器也会先修改模型，调用方即使收到异常也无法安全重试。
        self._is_adamw = isinstance(optimizer, torch.optim.AdamW)
        self._is_sgd = isinstance(optimizer, torch.optim.SGD)
        if not (self._is_adamw or self._is_sgd):
            raise TypeError(f"OPTIMIZER_UNSUPPORTED:{type(optimizer).__name__}")
        self._group_by_identity: dict[int, dict[str, Any]] = {}
        supplied_names_by_identity: dict[int, list[str]] = {}
        for name, parameter in supplied_parameters.items():
            supplied_names_by_identity.setdefault(id(parameter), []).append(name)
        if any(len(names) > 1 for names in supplied_names_by_identity.values()):
            raise ValueError("OPTIMIZER_BRIDGE_ALIASES_REQUIRE_CANONICAL_REGISTRY")
        for group in optimizer.param_groups:
            if group.get("fused") or group.get("foreach"):
                raise ValueError("OPTIMIZER_FUSED_OR_FOREACH_NOT_VERIFIED")
            if self._is_sgd and float(group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError("SGD_COUPLED_WEIGHT_DECAY_NOT_DECOMPOSABLE")
            for parameter in group["params"]:
                identity = id(parameter)
                if identity in self._group_by_identity:
                    raise ValueError("OPTIMIZER_PARAMETER_IN_MULTIPLE_GROUPS")
                self._group_by_identity[identity] = group
                if parameter.requires_grad and identity not in supplied_names_by_identity:
                    raise ValueError("OPTIMIZER_PARAMETER_NOT_NAMED")
        for name, parameter in supplied_parameters.items():
            if parameter.requires_grad and id(parameter) not in self._group_by_identity:
                raise ValueError(f"OPTIMIZER_PARAMETER_MISSING:{name}")
        # 冻结参数或不属于 optimizer 的只读参数不参与 step 位移合同，避免后续
        # group lookup 把它们误当成 eligible coordinate。
        self.named_parameters = {
            name: parameter
            for name, parameter in supplied_parameters.items()
            if parameter.requires_grad and id(parameter) in self._group_by_identity
        }

    def step(self) -> StepOutcome:
        import torch

        # 完整预检必须发生在 ``optimizer.step`` 之前。这样即使调用方没有通过
        # GradientAttempt 高层状态机，也不能把 sparse/NaN/Inf 写入参数或动量状态。
        for name, parameter in self.named_parameters.items():
            gradient = parameter.grad
            if gradient is not None and gradient.is_sparse:
                raise TypeError(f"SPARSE_GRADIENT_UNSUPPORTED:{name}")
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"NONFINITE_GRADIENT_STEP_SKIPPED:{name}")
        pre = {name: parameter.detach().clone() for name, parameter in self.named_parameters.items()}
        had_gradient = {
            name: parameter.grad is not None
            for name, parameter in self.named_parameters.items()
        }
        self.optimizer.step()
        total: dict[str, Any] = {}
        data: dict[str, Any] = {}
        decay: dict[str, Any] = {}
        rates: dict[str, float] = {}
        for name, parameter in self.named_parameters.items():
            group = self._group_by_identity[id(parameter)]
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            rates[name] = lr
            total_delta = parameter.detach() - pre[name]
            if self._is_adamw:
                # PyTorch AdamW 对 ``grad is None`` 的参数完全跳过，包括 decoupled
                # weight decay；不能伪造 decay 再用反向 data delta 抵消。
                decay_delta = (
                    -lr * weight_decay * pre[name]
                    if had_gradient[name]
                    else torch.zeros_like(total_delta)
                )
                data_delta = total_delta - decay_delta
            else:
                decay_delta = torch.zeros_like(total_delta)
                data_delta = total_delta
            total[name] = total_delta.clone()
            data[name] = data_delta.clone()
            decay[name] = decay_delta.clone()
        return StepOutcome(total, data, decay, rates)
