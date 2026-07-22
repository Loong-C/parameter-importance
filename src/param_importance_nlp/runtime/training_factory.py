"""严格配置到 Torch optimizer、scheduler 与 scaler 的构造边界。

这里不读取 YAML/JSON，也不提供宽松默认；调用方必须先通过 ResolvedConfig。
动态学习率只存在于 optimizer/scheduler 状态，不进入坐标 registry hash。为了让
``OptimizerBridge`` 能审计实际位移，fused/foreach/capturable 优化器均明确拒绝。
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


def _finite_number(value: object, *, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} 必须是数值")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} 必须为有限且 >= {minimum}")
    return result


def build_optimizer(
    parameters: object,
    base_options: Mapping[str, object],
    runtime_options: Mapping[str, object] | None = None,
) -> torch.optim.Optimizer:
    """构造受支持 optimizer；参数组高级映射由上游 registry adapter 展开。"""

    kind = base_options.get("type")
    if kind not in {"sgd", "momentum", "adamw"}:
        raise ValueError(f"OPTIMIZER_TYPE_UNSUPPORTED:{kind}")
    if base_options.get("fused") is not False or base_options.get("foreach") is not False:
        raise ValueError("OPTIMIZER_FUSED_OR_FOREACH_NOT_VERIFIED")
    lr = _finite_number(base_options.get("learning_rate"), field="optimizer.learning_rate")
    weight_decay = _finite_number(
        base_options.get("weight_decay", 0.0), field="optimizer.weight_decay"
    )
    runtime = {} if runtime_options is None else dict(runtime_options)
    if kind in {"sgd", "momentum"}:
        momentum = _finite_number(
            base_options.get("momentum", 0.0), field="optimizer.momentum"
        )
        if kind == "momentum" and momentum <= 0:
            raise ValueError("MOMENTUM_OPTIMIZER_REQUIRES_POSITIVE_MOMENTUM")
        # coupled SGD decay 无法和 data movement 唯一分解，在线重要性路径拒绝。
        if weight_decay != 0.0:
            raise ValueError("SGD_COUPLED_WEIGHT_DECAY_NOT_DECOMPOSABLE")
        return torch.optim.SGD(
            parameters,  # type: ignore[arg-type]
            lr=lr,
            momentum=momentum,
            dampening=_finite_number(runtime.get("dampening", 0.0), field="optimizer.dampening"),
            nesterov=bool(runtime.get("nesterov", False)),
            weight_decay=0.0,
            foreach=False,
            fused=False,
        )
    betas = runtime.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise TypeError("optimizer.betas 必须是两个数")
    beta1 = _finite_number(betas[0], field="optimizer.beta1")
    beta2 = _finite_number(betas[1], field="optimizer.beta2")
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("ADAMW_BETAS_OUT_OF_RANGE")
    eps = _finite_number(runtime.get("eps", 1e-8), field="optimizer.eps")
    if eps <= 0:
        raise ValueError("ADAMW_EPS_MUST_BE_POSITIVE")
    return torch.optim.AdamW(
        parameters,  # type: ignore[arg-type]
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=bool(runtime.get("amsgrad", False)),
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    options: Mapping[str, object],
) -> object | None:
    """构造 constant/linear/cosine warmup scheduler，并支持 state_dict 恢复。"""

    kind = options.get("kind")
    if kind == "none":
        return None
    if kind not in {"constant", "linear", "cosine"}:
        raise ValueError("SCHEDULER_KIND_UNSUPPORTED")
    warmup = options.get("warmup_steps")
    total = options.get("total_steps")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("SCHEDULER_WARMUP_INVALID")
    if isinstance(total, bool) or not isinstance(total, int) or total <= warmup:
        raise ValueError("SCHEDULER_TOTAL_STEPS_INVALID")

    def scale(step: int) -> float:
        # LambdaLR 首次 ``scheduler.step`` 接收 step=1；使用 step+1 会在 optimizer
        # 构造阶段隐式改变初始 LR，因此这里直接按已完成更新数定义。
        if warmup > 0 and step < warmup:
            return max(0.0, float(step + 1) / warmup)
        if kind == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
        if kind == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def build_grad_scaler(
    options: Mapping[str, object],
    *,
    device_type: str,
) -> object | None:
    """只在显式启用时构造 Torch GradScaler；BF16/FP32 默认不使用 scaler。"""

    enabled = options.get("grad_scaler_enabled", False)
    if type(enabled) is not bool:
        raise TypeError("precision_runtime.grad_scaler_enabled 必须是 bool")
    if not enabled:
        return None
    if device_type != "cuda":
        raise RuntimeError("GRAD_SCALER_REQUIRES_CUDA")
    return torch.amp.GradScaler(
        device="cuda",
        init_scale=_finite_number(
            options.get("initial_scale", 65536.0), field="precision_runtime.initial_scale"
        ),
        growth_factor=_finite_number(
            options.get("growth_factor", 2.0), field="precision_runtime.growth_factor"
        ),
        backoff_factor=_finite_number(
            options.get("backoff_factor", 0.5), field="precision_runtime.backoff_factor"
        ),
        growth_interval=int(options.get("growth_interval", 2000)),
        enabled=True,
    )


__all__ = ["build_grad_scaler", "build_optimizer", "build_scheduler"]
