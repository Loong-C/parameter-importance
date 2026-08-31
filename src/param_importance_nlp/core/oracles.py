"""Stage 1 的独立 FP64 解析 oracle 与确定性数值 fixture。

本模块刻意不调用 :mod:`core.estimators` 或充分统计量实现。raw、double、U 与
cross-U 都从原始梯度数组通过显式循环重算，使测试不会因“生产实现与测试共享同
一公式 helper”而共同出错。所有 oracle 输入会显式转换到 CPU/FP64；这次转换只
发生在比较/真值边界，并由返回对象记录，不能静默改变在线 estimator 的 dtype
合同。

除离线 estimator 真值外，这里还提供：

* 中心有限差分，用于将闭式梯度、autograd 与数值扰动三方交叉核验；
* 常梯度线性损失与对角二次损失 fixture，可解析计算逐坐标路径贡献；
* 从单样本高斯变量开始生成的零均值噪声 fixture，并给出 raw/U 均值的理论标准误；
* 按 Stage 1 冻结规范实现的 FP64 比较器，同时报告原单位误差、尺度化误差、
  normalized-L2、非有限数数量和最差参数张量。

这些工具只服务于 CPU 小型 fixture，不读取模型、数据或服务器资产，也不生成任
何正式实验结论。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

import torch

from .errors import CoreContractError, NumericalError
from .tensors import TensorMap


ScalarLoss = Callable[[TensorMap], torch.Tensor | float]


def _fp64_cpu(value: TensorMap) -> TensorMap:
    """返回断开旧计算图的 CPU/FP64 副本，同时保留 registry 坐标身份。"""

    converted = {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).clone()
        for name, tensor in value.items()
    }
    return TensorMap(converted, registry=value.registry)


def _validate_samples(samples: Sequence[TensorMap], *, name: str) -> TensorMap:
    if not samples:
        raise CoreContractError(f"{name} 至少需要一个梯度统计单元")
    reference = samples[0]
    reference.assert_finite()
    for sample in samples[1:]:
        reference.assert_compatible(sample)
        sample.assert_finite()
    return reference


def fp64_mean_gradient_oracle(samples: Sequence[TensorMap]) -> TensorMap:
    """用显式逐样本加法重构 FP64 平均梯度。

    输入 shape 是若干个同构 ``TensorMap``；返回每个坐标的算术平均，dtype 固定
    为 ``torch.float64`` 且 device 固定为 CPU。
    """

    reference = _validate_samples(samples, name="mean-gradient oracle")
    total = {
        name: torch.zeros_like(tensor, device="cpu", dtype=torch.float64)
        for name, tensor in reference.items()
    }
    for sample in samples:
        for name, tensor in sample.items():
            total[name].add_(tensor.detach().to(device="cpu", dtype=torch.float64))
    count = float(len(samples))
    return TensorMap(
        {name: value / count for name, value in total.items()},
        registry=reference.registry,
    )


def fp64_raw_oracle(mean_gradient: TensorMap) -> TensorMap:
    """独立计算 raw core ``mean_gradient**2``，不乘学习率或 clip。"""

    mean = _fp64_cpu(mean_gradient)
    return TensorMap(
        {name: tensor * tensor for name, tensor in mean.items()},
        registry=mean.registry,
    )


def fp64_double_sample_oracle(
    mean_gradient_a: TensorMap,
    mean_gradient_b: TensorMap,
) -> TensorMap:
    """独立计算两个 sampling stream 的 FP64 逐坐标均值乘积。"""

    mean_gradient_a.assert_compatible(mean_gradient_b)
    left = _fp64_cpu(mean_gradient_a)
    right = _fp64_cpu(mean_gradient_b)
    return TensorMap(
        {name: left[name] * right[name] for name in left},
        registry=left.registry,
    )


def fp64_equal_u_oracle(samples: Sequence[TensorMap]) -> TensorMap:
    """用显式 ``i != j`` 双循环计算等权 ordered-pair U 真值。

    该实现不使用 ``S1**2-S2`` 快速恒等式，因而能独立发现生产流式统计量或去对
    角公式中的共同错误。``M=1`` 没有合法 ordered pair，必须失败。
    """

    reference = _validate_samples(samples, name="equal-U oracle")
    if len(samples) < 2:
        raise CoreContractError("equal-U oracle 至少需要两个统计单元")
    converted = [_fp64_cpu(sample) for sample in samples]
    total = {
        name: torch.zeros_like(tensor, device="cpu", dtype=torch.float64)
        for name, tensor in reference.items()
    }
    pair_count = 0
    for left_index, left in enumerate(converted):
        for right_index, right in enumerate(converted):
            if left_index == right_index:
                continue
            pair_count += 1
            for name in total:
                total[name].add_(left[name] * right[name])
    assert pair_count == len(samples) * (len(samples) - 1)
    return TensorMap(
        {name: value / float(pair_count) for name, value in total.items()},
        registry=reference.registry,
    )


def fp64_weighted_u_oracle(
    samples: Sequence[TensorMap],
    weights: Sequence[float | int],
) -> TensorMap:
    """用显式加权 ordered pairs 计算 weighted microbatch U 真值。"""

    reference = _validate_samples(samples, name="weighted-U oracle")
    if len(samples) < 2 or len(weights) != len(samples):
        raise CoreContractError("weighted-U oracle 的样本/权重数量非法")
    normalized = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight <= 0 for weight in normalized):
        raise CoreContractError("weighted-U oracle 权重必须为有限正数")
    converted = [_fp64_cpu(sample) for sample in samples]
    total = {
        name: torch.zeros_like(tensor, device="cpu", dtype=torch.float64)
        for name, tensor in reference.items()
    }
    denominator = 0.0
    for left_index, left in enumerate(converted):
        for right_index, right in enumerate(converted):
            if left_index == right_index:
                continue
            pair_weight = normalized[left_index] * normalized[right_index]
            denominator += pair_weight
            for name in total:
                total[name].add_(left[name] * right[name], alpha=pair_weight)
    if not math.isfinite(denominator) or denominator <= 0:
        raise CoreContractError("weighted-U oracle ordered-pair 总权重必须为有限正数")
    return TensorMap(
        {name: value / denominator for name, value in total.items()},
        registry=reference.registry,
    )


def fp64_cross_u_oracle(
    x_samples: Sequence[TensorMap],
    y_samples: Sequence[TensorMap],
    *,
    x_weights: Sequence[float | int] | None = None,
    y_weights: Sequence[float | int] | None = None,
    exclude_matching_pairs: bool = True,
) -> TensorMap:
    """用显式允许对集合计算一般 cross-U 或独立 double-sample 真值。

    ``exclude_matching_pairs=True`` 表示 X/Y 是成对观测，删除相同下标以消除组内
    相关；为 ``False`` 时保留笛卡尔积全部 pair，得到两个加权均值的乘积。
    """

    reference = _validate_samples(x_samples, name="cross-U X oracle")
    _validate_samples(y_samples, name="cross-U Y oracle")
    for sample in y_samples:
        reference.assert_compatible(sample)
    if exclude_matching_pairs and len(x_samples) != len(y_samples):
        raise CoreContractError("删除 matching pair 时 X/Y 数量必须相同")
    wx = [1.0] * len(x_samples) if x_weights is None else [float(v) for v in x_weights]
    wy = [1.0] * len(y_samples) if y_weights is None else [float(v) for v in y_weights]
    if len(wx) != len(x_samples) or len(wy) != len(y_samples):
        raise CoreContractError("cross-U oracle 权重数量必须与样本数量一致")
    if any(not math.isfinite(v) or v <= 0 for v in (*wx, *wy)):
        raise CoreContractError("cross-U oracle 权重必须为有限正数")
    x64 = [_fp64_cpu(sample) for sample in x_samples]
    y64 = [_fp64_cpu(sample) for sample in y_samples]
    total = {
        name: torch.zeros_like(tensor, device="cpu", dtype=torch.float64)
        for name, tensor in reference.items()
    }
    denominator = 0.0
    for x_index, x_sample in enumerate(x64):
        for y_index, y_sample in enumerate(y64):
            if exclude_matching_pairs and x_index == y_index:
                continue
            pair_weight = wx[x_index] * wy[y_index]
            denominator += pair_weight
            for name in total:
                total[name].add_(x_sample[name] * y_sample[name], alpha=pair_weight)
    if not math.isfinite(denominator) or denominator <= 0:
        raise CoreContractError("cross-U oracle 没有正权重允许 pair")
    return TensorMap(
        {name: value / denominator for name, value in total.items()},
        registry=reference.registry,
    )


def fp64_apply_group_learning_rates(
    core: TensorMap,
    learning_rates: Mapping[str, float | int],
) -> TensorMap:
    """在独立 oracle 路径中按参数组应用实际 step 学习率。

    有 registry 时只按静态 ``group_id`` 查找；无 registry 的解析 fixture 可按参数
    名、``default`` 或 ``*`` 指定。动态学习率只影响返回 score，不参与坐标 hash。
    """

    if not learning_rates:
        raise CoreContractError("learning_rates 不能为空")
    normalized = {str(key): float(value) for key, value in learning_rates.items()}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise CoreContractError("oracle 学习率必须为有限非负数")
    core64 = _fp64_cpu(core)
    result: dict[str, torch.Tensor] = {}
    for name, value in core64.items():
        if core64.registry is not None:
            group_id = core64.registry.record(name).group_id
            if group_id is None or group_id not in normalized:
                raise CoreContractError(f"oracle 缺少参数 {name!r} 的 optimizer group 学习率")
            rate = normalized[group_id]
        elif name in normalized:
            rate = normalized[name]
        elif "default" in normalized:
            rate = normalized["default"]
        elif "*" in normalized:
            rate = normalized["*"]
        elif len(normalized) == 1:
            rate = next(iter(normalized.values()))
        else:
            raise CoreContractError(f"oracle 无法确定参数 {name!r} 的学习率")
        result[name] = value * rate
    return TensorMap(result, registry=core64.registry)


def _loss_scalar(loss_fn: ScalarLoss, state: TensorMap) -> float:
    raw = loss_fn(state)
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 0:
            raise CoreContractError("有限差分 loss_fn 必须返回标量")
        value = float(raw.detach().to(device="cpu", dtype=torch.float64).item())
    else:
        value = float(raw)
    if not math.isfinite(value):
        raise NumericalError("有限差分 loss_fn 返回 NaN/Inf")
    return value


def central_difference_gradient(
    loss_fn: ScalarLoss,
    point: TensorMap,
    *,
    step: float | Mapping[str, float | torch.Tensor] = 1e-5,
) -> TensorMap:
    """在 CPU/FP64 上逐坐标计算中心有限差分梯度。

    对坐标 ``k`` 使用 ``[L(theta+h e_k)-L(theta-h e_k)]/(2h)``。``step`` 可为
    全局正标量，也可为参数名到正标量/同 shape 张量的映射，以便记录不同参数角
    色的扰动尺度。每次调用 loss 都获得新的状态副本，原 ``point`` 不会被修改。
    本函数面向小型抽查，时间复杂度为两次前向乘坐标数，不应替代完整 autograd。
    """

    base = _fp64_cpu(point)

    def steps_for(name: str, value: torch.Tensor) -> torch.Tensor:
        raw: float | torch.Tensor
        if isinstance(step, Mapping):
            if name not in step:
                raise CoreContractError(f"有限差分缺少参数 {name!r} 的 step")
            raw = step[name]
        else:
            raw = step
        if isinstance(raw, torch.Tensor):
            tensor = raw.detach().to(device="cpu", dtype=torch.float64)
            if tuple(tensor.shape) != tuple(value.shape):
                raise CoreContractError(f"参数 {name!r} 的有限差分 step shape 不匹配")
        else:
            scalar = float(raw)
            tensor = torch.full_like(value, scalar, dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
            raise CoreContractError("有限差分 step 必须逐坐标为有限正数")
        return tensor

    gradient: dict[str, torch.Tensor] = {}
    base_values = {name: value.detach().clone() for name, value in base.items()}
    for name, value in base_values.items():
        parameter_steps = steps_for(name, value)
        flat_steps = parameter_steps.reshape(-1)
        coordinate_gradient = torch.empty_like(value, dtype=torch.float64, device="cpu")
        flat_gradient = coordinate_gradient.reshape(-1)
        for coordinate in range(value.numel()):
            plus_values = {key: tensor.clone() for key, tensor in base_values.items()}
            minus_values = {key: tensor.clone() for key, tensor in base_values.items()}
            h = float(flat_steps[coordinate].item())
            plus_values[name].reshape(-1)[coordinate] += h
            minus_values[name].reshape(-1)[coordinate] -= h
            plus = TensorMap(plus_values, registry=base.registry)
            minus = TensorMap(minus_values, registry=base.registry)
            flat_gradient[coordinate] = (
                _loss_scalar(loss_fn, plus) - _loss_scalar(loss_fn, minus)
            ) / (2.0 * h)
        gradient[name] = coordinate_gradient
    return TensorMap(gradient, registry=base.registry)


@dataclass(frozen=True, slots=True)
class ConstantGradientFixture:
    """线性标量损失 ``L(theta)=sum_k c_k*theta_k + constant``。"""

    gradient: TensorMap
    constant: float = 0.0

    def __post_init__(self) -> None:
        constant = float(self.constant)
        if not math.isfinite(constant):
            raise CoreContractError("线性 fixture constant 必须有限")
        object.__setattr__(self, "gradient", _fp64_cpu(self.gradient))
        object.__setattr__(self, "constant", constant)

    def loss(self, state: TensorMap) -> torch.Tensor:
        """返回 FP64 标量损失；保留输入计算图以支持 autograd 交叉核验。"""

        self.gradient.assert_compatible(state)
        terms = [
            (
                self.gradient[name]
                * state[name].to(device="cpu", dtype=torch.float64)
            ).sum()
            for name in self.gradient
        ]
        return torch.stack(terms).sum() + self.constant

    def gradient_at(self, state: TensorMap) -> TensorMap:
        """返回与位置无关的解析梯度副本。"""

        self.gradient.assert_compatible(state)
        return self.gradient.clone()

    def path_contribution(self, pre: TensorMap, post: TensorMap) -> TensorMap:
        """返回符号约定 ``-(post-pre)*gradient`` 的逐坐标闭式贡献。"""

        pre.assert_compatible(post)
        self.gradient.assert_compatible(pre)
        pre64, post64 = _fp64_cpu(pre), _fp64_cpu(post)
        return TensorMap(
            {
                name: -(post64[name] - pre64[name]) * self.gradient[name]
                for name in self.gradient
            },
            registry=self.gradient.registry,
        )


@dataclass(frozen=True, slots=True)
class QuadraticLossFixture:
    """可分离二次损失 ``0.5*d*theta**2 + linear*theta + constant``。

    ``diagonal``、``linear`` 与参数同 shape；不要求 ``diagonal`` 为正，因此也能
    覆盖鞍点，但所有系数必须有限。沿线性参数路径时梯度关于 alpha 为一次函数，
    梯形法应逐坐标精确。
    """

    diagonal: TensorMap
    linear: TensorMap
    constant: float = 0.0

    def __post_init__(self) -> None:
        self.diagonal.assert_compatible(self.linear)
        diagonal = _fp64_cpu(self.diagonal)
        linear = _fp64_cpu(self.linear)
        constant = float(self.constant)
        if not math.isfinite(constant):
            raise CoreContractError("二次 fixture constant 必须有限")
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "constant", constant)

    def loss(self, state: TensorMap) -> torch.Tensor:
        """返回解析二次损失的 CPU/FP64 标量，并保留输入 autograd 图。"""

        self.diagonal.assert_compatible(state)
        terms = [
            (
                0.5
                * self.diagonal[name]
                * state[name].to(device="cpu", dtype=torch.float64).square()
                + self.linear[name]
                * state[name].to(device="cpu", dtype=torch.float64)
            ).sum()
            for name in self.diagonal
        ]
        return torch.stack(terms).sum() + self.constant

    def gradient_at(self, state: TensorMap) -> TensorMap:
        """返回闭式梯度 ``d*theta + linear``。"""

        self.diagonal.assert_compatible(state)
        state64 = _fp64_cpu(state)
        return TensorMap(
            {
                name: self.diagonal[name] * state64[name] + self.linear[name]
                for name in self.diagonal
            },
            registry=self.diagonal.registry,
        )

    def path_contribution(self, pre: TensorMap, post: TensorMap) -> TensorMap:
        """逐坐标返回 ``L(pre)-L(post)`` 的闭式分解。"""

        self.diagonal.assert_compatible(pre)
        pre.assert_compatible(post)
        pre64, post64 = _fp64_cpu(pre), _fp64_cpu(post)
        contribution: dict[str, torch.Tensor] = {}
        for name in self.diagonal:
            before = (
                0.5 * self.diagonal[name] * pre64[name].square()
                + self.linear[name] * pre64[name]
            )
            after = (
                0.5 * self.diagonal[name] * post64[name].square()
                + self.linear[name] * post64[name]
            )
            contribution[name] = before - after
        return TensorMap(contribution, registry=self.diagonal.registry)


@dataclass(frozen=True, slots=True)
class ZeroMeanNoiseFixture:
    """从单样本高斯梯度生成确定性、理论均值为零的 microbatch fixture。

    先生成 ``h[r,m,i,k] ~ N(0,sigma**2)``，再对样本轴 ``i`` 求均值得到
    ``g[r,m,k]``，不会混淆单样本方差与 microbatch-mean 方差。返回的每个重复相
    互独立；它只能作为代码性质 smoke，不能替代真实梯度 pilot。
    """

    seed: int
    sigma: float
    microbatch_size: int
    microbatch_count: int
    repetitions: int
    coordinate_shapes: Mapping[str, tuple[int, ...]]

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise CoreContractError("noise fixture seed 必须非负")
        if not math.isfinite(float(self.sigma)) or float(self.sigma) <= 0:
            raise CoreContractError("noise fixture sigma 必须为有限正数")
        if self.microbatch_size <= 0 or self.microbatch_count < 2 or self.repetitions <= 0:
            raise CoreContractError("noise fixture 要求 b>0、M>=2、R>0")
        normalized: dict[str, tuple[int, ...]] = {}
        for name, shape in self.coordinate_shapes.items():
            if not name or any(int(size) <= 0 for size in shape):
                raise CoreContractError("noise fixture 参数名与 shape 必须有效")
            normalized[str(name)] = tuple(int(size) for size in shape)
        if not normalized:
            raise CoreContractError("noise fixture 至少需要一个参数张量")
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "coordinate_shapes", MappingProxyType(normalized))

    @property
    def microbatch_variance(self) -> float:
        return self.sigma**2 / self.microbatch_size

    @property
    def raw_expectation(self) -> float:
        """零均值下 full-batch raw core 的理论正偏 ``sigma²/(bM)``。"""

        return self.microbatch_variance / self.microbatch_count

    @property
    def raw_mean_standard_error(self) -> float:
        """R 次 raw core 均值的解析标准误（使用高斯四阶矩）。"""

        return self.raw_expectation * math.sqrt(2.0 / self.repetitions)

    @property
    def u_mean_standard_error(self) -> float:
        """R 次等权 U core 均值的解析标准误。"""

        variance = (
            2.0
            * self.microbatch_variance**2
            / (self.microbatch_count * (self.microbatch_count - 1))
        )
        return math.sqrt(variance / self.repetitions)

    def repetitions_iter(self) -> Iterator[tuple[TensorMap, ...]]:
        """按 seed 确定性地产生 R 组、每组 M 个 microbatch mean 梯度。"""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        generated: dict[str, torch.Tensor] = {}
        for name, shape in self.coordinate_shapes.items():
            sample_shape = (
                self.repetitions,
                self.microbatch_count,
                self.microbatch_size,
                *shape,
            )
            base = torch.randn(sample_shape, generator=generator, dtype=torch.float64)
            generated[name] = (base * self.sigma).mean(dim=2)
        for repetition in range(self.repetitions):
            yield tuple(
                TensorMap(
                    {
                        name: values[repetition, microbatch].clone()
                        for name, values in generated.items()
                    }
                )
                for microbatch in range(self.microbatch_count)
            )


def _stable_l2(vector: torch.Tensor) -> float:
    """通过最大值缩放计算 L2，避免极大/极小 FP64 值平方溢出或下溢。"""

    if vector.numel() == 0:
        return 0.0
    maximum = float(vector.abs().max().item())
    if maximum == 0.0:
        return 0.0
    if not math.isfinite(maximum):
        return math.inf
    scaled = vector / maximum
    return maximum * math.sqrt(float(torch.sum(scaled * scaled).item()))


@dataclass(frozen=True, slots=True)
class FP64Comparison:
    """一次 TensorMap 比较的完整、机器可判定结果。"""

    passed: bool
    branch: str
    comparison_dtype: str
    natural_scale: float
    atol: float
    rtol: float
    normalized_l2_limit: float
    near_zero_threshold: float
    absolute_threshold: float
    max_absolute_error: float
    scaled_max_error: float | None
    normalized_l2_error: float | None
    nonfinite_count: int
    worst_parameter: str | None

    def to_dict(self) -> dict[str, object]:
        """返回只含 canonical-JSON primitive 的确定性报告字段。

        内存结果用 ``math.inf`` 表示无法计算的误差，发布 artifact 时必须改为
        ``null``，并由 ``passed=false`` 与 ``nonfinite_count`` 解释失败原因，绝不
        把 JSON 禁止的 Infinity/NaN 写入证据文件。
        """

        def finite_or_none(value: float | None) -> float | None:
            if value is None or not math.isfinite(value):
                return None
            return value

        return {
            "passed": self.passed,
            "branch": self.branch,
            "comparison_dtype": self.comparison_dtype,
            "natural_scale": self.natural_scale,
            "atol": self.atol,
            "rtol": self.rtol,
            "normalized_l2_limit": self.normalized_l2_limit,
            "near_zero_threshold": self.near_zero_threshold,
            "absolute_threshold": self.absolute_threshold,
            "max_absolute_error": finite_or_none(self.max_absolute_error),
            "scaled_max_error": finite_or_none(self.scaled_max_error),
            "normalized_l2_error": finite_or_none(self.normalized_l2_error),
            "nonfinite_count": self.nonfinite_count,
            "worst_parameter": self.worst_parameter,
        }


def compare_tensor_maps_fp64(
    actual: TensorMap,
    oracle: TensorMap,
    *,
    natural_scale: float,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    normalized_l2_limit: float = 1e-10,
) -> FP64Comparison:
    """按 Stage 1 双分支容差规范比较两个 TensorMap。

    比较前两侧都转到 CPU/FP64。``natural_scale`` 必须来自解析输入设计，不能从
    ``actual`` 反推。若 oracle 的无穷范数不超过 ``10*atol*natural_scale``，只
    使用原单位绝对阈值；否则按 oracle 无穷范数缩放，同时 Gate 尺度化最大误差
    和稳定 normalized-L2。任何 NaN/Inf 都计数并使结果失败。
    """

    actual.assert_compatible(oracle)
    natural_scale = float(natural_scale)
    atol = float(atol)
    rtol = float(rtol)
    normalized_l2_limit = float(normalized_l2_limit)
    if not math.isfinite(natural_scale) or natural_scale <= 0:
        raise CoreContractError("comparison natural_scale 必须为有限正数")
    if any(
        not math.isfinite(value) or value < 0
        for value in (atol, rtol, normalized_l2_limit)
    ):
        raise CoreContractError("comparison 容差必须为有限非负数")

    actual64 = _fp64_cpu_allow_nonfinite(actual)
    oracle64 = _fp64_cpu_allow_nonfinite(oracle)
    actual_parts: list[torch.Tensor] = []
    oracle_parts: list[torch.Tensor] = []
    worst_parameter: str | None = None
    worst_error = -1.0
    nonfinite_count = 0
    for name in actual64:
        left = actual64[name].reshape(-1)
        right = oracle64[name].reshape(-1)
        left_finite = torch.isfinite(left)
        right_finite = torch.isfinite(right)
        finite = left_finite & right_finite
        local_nonfinite = int((~left_finite).sum().item() + (~right_finite).sum().item())
        nonfinite_count += local_nonfinite
        if local_nonfinite and (
            worst_error != math.inf
            or worst_parameter is None
            or name < worst_parameter
        ):
            worst_error = math.inf
            worst_parameter = name
        if bool(finite.any()):
            local_error = float((left[finite] - right[finite]).abs().max().item())
            if local_error > worst_error or (
                local_error == worst_error
                and (worst_parameter is None or name < worst_parameter)
            ):
                worst_error = local_error
                worst_parameter = name
        actual_parts.append(left)
        oracle_parts.append(right)
    actual_vector = torch.cat(actual_parts)
    oracle_vector = torch.cat(oracle_parts)
    finite_all = torch.isfinite(actual_vector) & torch.isfinite(oracle_vector)
    if bool(finite_all.all()):
        difference = actual_vector - oracle_vector
        max_absolute_error = float(difference.abs().max().item())
        oracle_inf = float(oracle_vector.abs().max().item())
    else:
        max_absolute_error = math.inf
        oracle_inf = (
            float(oracle_vector[torch.isfinite(oracle_vector)].abs().max().item())
            if bool(torch.isfinite(oracle_vector).any())
            else math.inf
        )
    near_zero_threshold = 10.0 * atol * natural_scale
    absolute_threshold = atol * natural_scale
    near_zero = math.isfinite(oracle_inf) and oracle_inf <= near_zero_threshold

    if near_zero:
        return FP64Comparison(
            passed=nonfinite_count == 0 and max_absolute_error <= absolute_threshold,
            branch="near_zero_absolute",
            comparison_dtype="torch.float64_cpu",
            natural_scale=natural_scale,
            atol=atol,
            rtol=rtol,
            normalized_l2_limit=normalized_l2_limit,
            near_zero_threshold=near_zero_threshold,
            absolute_threshold=absolute_threshold,
            max_absolute_error=max_absolute_error,
            scaled_max_error=None,
            normalized_l2_error=None,
            nonfinite_count=nonfinite_count,
            worst_parameter=worst_parameter,
        )

    if nonfinite_count:
        scaled_max_error = normalized_l2_error = math.inf
        passed = False
    else:
        scaled_actual = actual_vector / oracle_inf
        scaled_oracle = oracle_vector / oracle_inf
        scaled_difference = scaled_actual - scaled_oracle
        scaled_max_error = float(scaled_difference.abs().max().item())
        scale_bound = max(
            float(scaled_actual.abs().max().item()),
            float(scaled_oracle.abs().max().item()),
        )
        numerator = _stable_l2(scaled_difference)
        denominator = max(
            _stable_l2(scaled_actual),
            _stable_l2(scaled_oracle),
            1e-300,
        )
        normalized_l2_error = numerator / denominator
        passed = (
            scaled_max_error <= atol + rtol * scale_bound
            and normalized_l2_error <= normalized_l2_limit
        )
    return FP64Comparison(
        passed=passed,
        branch="scaled_nonzero",
        comparison_dtype="torch.float64_cpu",
        natural_scale=natural_scale,
        atol=atol,
        rtol=rtol,
        normalized_l2_limit=normalized_l2_limit,
        near_zero_threshold=near_zero_threshold,
        absolute_threshold=absolute_threshold,
        max_absolute_error=max_absolute_error,
        scaled_max_error=scaled_max_error,
        normalized_l2_error=normalized_l2_error,
        nonfinite_count=nonfinite_count,
        worst_parameter=worst_parameter,
    )


def _fp64_cpu_allow_nonfinite(value: TensorMap) -> TensorMap:
    """比较器专用转换：保留 NaN/Inf 以计数，而不是在构造时提前抛错。"""

    return TensorMap(
        {
            name: tensor.detach().to(device="cpu", dtype=torch.float64).clone()
            for name, tensor in value.items()
        },
        registry=value.registry,
        require_finite=False,
    )
