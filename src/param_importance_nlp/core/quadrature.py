"""模型无关的参数线性路径与数值求积引擎。

所有规则统一返回区间 ``[0,1]`` 上的 FP64 节点/权重，且权重和严格检查为 1。
路径引擎对每个节点只请求一次“全部研究参数梯度”，以高精度流式累计路径平均
梯度，再计算 ``-delta_parameter * average_gradient``。输入端点会被克隆，回调
得到的插值状态也不会与端点共享可写 storage。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import torch

from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from .errors import CoreContractError, NumericalError
from .tensors import TensorMap


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    """归一化到 ``[0,1]`` 的冻结求积规则。"""

    name: str
    version: str
    kind: str
    nodes: torch.Tensor
    weights: torch.Tensor
    subintervals: int | None
    exact_polynomial_degree: int
    theoretical_order: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.name, self.version, self.kind)
        ):
            raise CoreContractError("QuadratureRule name/version/kind 不能为空")
        nodes = torch.as_tensor(self.nodes, dtype=torch.float64).detach().clone()
        weights = torch.as_tensor(self.weights, dtype=torch.float64).detach().clone()
        if nodes.ndim != 1 or weights.ndim != 1 or nodes.numel() != weights.numel():
            raise CoreContractError("求积 nodes/weights 必须是一维且长度相同")
        if nodes.numel() == 0:
            raise CoreContractError("求积规则至少需要一个节点")
        if not bool(torch.isfinite(nodes).all()) or not bool(torch.isfinite(weights).all()):
            raise NumericalError("求积节点或权重含 NaN/Inf")
        if bool(((nodes < 0) | (nodes > 1)).any()):
            raise CoreContractError("求积节点必须位于 [0,1]")
        if bool((weights < 0).any()):
            raise CoreContractError("当前合同只支持非负求积权重")
        if not math.isclose(float(weights.sum().item()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise CoreContractError("求积权重之和必须为 1")
        if self.subintervals is not None and (
            isinstance(self.subintervals, bool)
            or not isinstance(self.subintervals, int)
            or self.subintervals <= 0
        ):
            raise CoreContractError("subintervals 必须为正整数或 null")
        if (
            isinstance(self.exact_polynomial_degree, bool)
            or not isinstance(self.exact_polynomial_degree, int)
            or self.exact_polynomial_degree < 0
        ):
            raise CoreContractError("exact_polynomial_degree 必须是非负整数")
        if (
            isinstance(self.theoretical_order, bool)
            or not isinstance(self.theoretical_order, int)
            or self.theoretical_order <= 0
        ):
            raise CoreContractError("theoretical_order 必须是正整数")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="QuadratureRule.metadata"),
        )

    @property
    def node_count(self) -> int:
        return int(self.nodes.numel())

    @property
    def unique_gradient_evaluations(self) -> int:
        return len({float(node) for node in self.nodes.tolist()})

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": "quadrature-rule-v1",
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "nodes": [float(item) for item in self.nodes.tolist()],
            "weights": [float(item) for item in self.weights.tolist()],
            "subintervals": self.subintervals,
            "exact_polynomial_degree": self.exact_polynomial_degree,
            "theoretical_order": self.theoretical_order,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def artifact_hash(self) -> str:
        """返回包含 FP64 节点/权重的 canonical artifact SHA-256。"""

        from ..contracts.jsonio import canonical_json_hash

        return canonical_json_hash(self._payload_without_hash())

    def to_dict(self) -> dict[str, object]:
        """序列化为独立公共 artifact；数组顺序承载求积语义。"""

        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "QuadratureRule":
        """严格加载并复算节点/权重 artifact hash。"""

        required = {
            "schema_version",
            "name",
            "version",
            "kind",
            "nodes",
            "weights",
            "subintervals",
            "exact_polynomial_degree",
            "theoretical_order",
            "metadata",
            "artifact_hash",
        }
        if set(value) != required:
            raise CoreContractError(
                "QuadratureRule 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        if value["schema_version"] != "quadrature-rule-v1":
            raise CoreContractError("不支持的 QuadratureRule schema")
        for name in ("name", "version", "kind", "artifact_hash"):
            if not isinstance(value[name], str):
                raise CoreContractError(f"QuadratureRule {name} 必须是字符串")
        nodes, weights = value["nodes"], value["weights"]
        if not isinstance(nodes, list) or not isinstance(weights, list):
            raise CoreContractError("QuadratureRule nodes/weights 必须是数组")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (*nodes, *weights)):
            raise CoreContractError("QuadratureRule nodes/weights 必须是数值且不能是 bool")
        subintervals = value["subintervals"]
        if subintervals is not None and (
            isinstance(subintervals, bool) or not isinstance(subintervals, int)
        ):
            raise CoreContractError("QuadratureRule subintervals 必须是整数或 null")
        for name in ("exact_polynomial_degree", "theoretical_order"):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise CoreContractError(f"QuadratureRule {name} 必须是整数")
        if not isinstance(value["metadata"], Mapping):
            raise CoreContractError("QuadratureRule metadata 必须是 object")
        rule = cls(
            name=value["name"],
            version=value["version"],
            kind=value["kind"],
            nodes=torch.tensor(nodes, dtype=torch.float64),
            weights=torch.tensor(weights, dtype=torch.float64),
            subintervals=subintervals,
            exact_polynomial_degree=value["exact_polynomial_degree"],
            theoretical_order=value["theoretical_order"],
            metadata=value["metadata"],
        )
        if value["artifact_hash"] != rule.artifact_hash:
            raise CoreContractError("QuadratureRule artifact_hash 与完整内容不一致")
        return rule


def left_rule() -> QuadratureRule:
    return QuadratureRule("left", "1", "single_interval", [0.0], [1.0], None, 0, 1)


def right_rule() -> QuadratureRule:
    return QuadratureRule("right", "1", "single_interval", [1.0], [1.0], None, 0, 1)


def midpoint_rule() -> QuadratureRule:
    return QuadratureRule("midpoint", "1", "single_interval", [0.5], [1.0], None, 1, 2)


def trapezoid_rule() -> QuadratureRule:
    return QuadratureRule(
        "trapezoid", "1", "single_interval", [0.0, 1.0], [0.5, 0.5], None, 1, 2
    )


def simpson_rule() -> QuadratureRule:
    return QuadratureRule(
        "simpson",
        "1",
        "single_interval",
        [0.0, 0.5, 1.0],
        [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0],
        2,
        3,
        4,
    )


def composite_rule(method: str, subintervals: int) -> QuadratureRule:
    """构造复合等距 Newton-Cotes 规则。

    ``method`` 支持 ``left``、``right``、``midpoint``、``trapezoid``、
    ``simpson``。复合 Simpson 严格要求正偶数子区间，不自动取整。
    """

    if not isinstance(subintervals, int) or isinstance(subintervals, bool) or subintervals <= 0:
        raise CoreContractError("subintervals 必须是正整数")
    normalized = method.strip().lower().replace("composite_", "")
    n = subintervals
    if normalized == "left":
        nodes = torch.arange(n, dtype=torch.float64) / n
        weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
        degree, order = 0, 1
    elif normalized == "right":
        nodes = torch.arange(1, n + 1, dtype=torch.float64) / n
        weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
        degree, order = 0, 1
    elif normalized == "midpoint":
        nodes = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
        degree, order = 1, 2
    elif normalized == "trapezoid":
        nodes = torch.arange(n + 1, dtype=torch.float64) / n
        weights = torch.full((n + 1,), 1.0 / n, dtype=torch.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        degree, order = 1, 2
    elif normalized == "simpson":
        if n % 2 != 0:
            raise CoreContractError("复合 Simpson 的 subintervals 必须为正偶数")
        nodes = torch.arange(n + 1, dtype=torch.float64) / n
        coefficients = torch.ones(n + 1, dtype=torch.float64)
        coefficients[1:-1:2] = 4.0
        coefficients[2:-1:2] = 2.0
        weights = coefficients / (3.0 * n)
        degree, order = 3, 4
    else:
        raise CoreContractError(f"未知复合求积方法: {method!r}")
    return QuadratureRule(
        f"composite_{normalized}",
        "1",
        "composite_equispaced",
        nodes,
        weights,
        n,
        degree,
        order,
    )


def composite_left_rule(subintervals: int) -> QuadratureRule:
    return composite_rule("left", subintervals)


def composite_right_rule(subintervals: int) -> QuadratureRule:
    return composite_rule("right", subintervals)


def composite_midpoint_rule(subintervals: int) -> QuadratureRule:
    return composite_rule("midpoint", subintervals)


def composite_trapezoid_rule(subintervals: int) -> QuadratureRule:
    return composite_rule("trapezoid", subintervals)


def composite_simpson_rule(subintervals: int) -> QuadratureRule:
    return composite_rule("simpson", subintervals)


def gauss_legendre_rule(points: int) -> QuadratureRule:
    """构造 Q 点 Gauss-Legendre，并从 ``[-1,1]`` 映射到 ``[0,1]``。"""

    if not isinstance(points, int) or isinstance(points, bool) or points <= 0:
        raise CoreContractError("Gauss-Legendre points 必须是正整数")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(points)
    nodes = (raw_nodes + 1.0) / 2.0
    weights = raw_weights / 2.0
    return QuadratureRule(
        f"gauss_legendre_{points}",
        "1",
        "gauss_legendre",
        torch.from_numpy(nodes),
        torch.from_numpy(weights),
        None,
        2 * points - 1,
        2 * points,
        {"numpy_version": np.__version__, "generation_dtype": "float64"},
    )


def default_quadrature_rules() -> Mapping[str, QuadratureRule]:
    """返回低成本基础规则注册表；正式默认方法仍由冻结决策 artifact 指定。"""

    rules = [left_rule(), right_rule(), midpoint_rule(), trapezoid_rule(), simpson_rule()]
    return MappingProxyType({rule.name: rule for rule in rules})


@dataclass(frozen=True, slots=True)
class PathSpec:
    """两个不可变参数端点之间的线性路径合同。"""

    parameter_pre_state: TensorMap
    parameter_post_state: TensorMap
    path_id: str = "linear_actual_endpoint_path"
    probe_id: str = "unspecified_probe"
    loss_id: str = "unspecified_loss"
    accumulation_dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        self.parameter_pre_state.assert_compatible(self.parameter_post_state)
        self.parameter_pre_state.assert_finite()
        self.parameter_post_state.assert_finite()
        if self.accumulation_dtype not in {torch.float32, torch.float64}:
            raise CoreContractError("路径 accumulation dtype 必须为 FP32 或 FP64")
        if not self.path_id or not self.probe_id or not self.loss_id:
            raise CoreContractError("path_id/probe_id/loss_id 不能为空")
        for name in self.parameter_pre_state:
            before = self.parameter_pre_state[name]
            after = self.parameter_post_state[name]
            if before.dtype != after.dtype or before.device != after.device:
                raise CoreContractError(f"路径端点 {name!r} 的 dtype/device 必须一致")
        # 克隆端点，防止调用方在积分过程中原地修改源 mapping。
        object.__setattr__(self, "parameter_pre_state", self.parameter_pre_state.clone())
        object.__setattr__(self, "parameter_post_state", self.parameter_post_state.clone())

    @property
    def delta(self) -> TensorMap:
        return self.parameter_post_state - self.parameter_pre_state

    def interpolate(self, alpha: float) -> TensorMap:
        """返回 ``pre + alpha * (post-pre)``；默认拒绝反事实外推。"""

        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise CoreContractError("路径 alpha 必须是 [0,1] 内有限数")
        return self.parameter_pre_state + self.delta * alpha

    @property
    def identity_hash(self) -> str:
        def tensor_digest(tensor: torch.Tensor) -> str:
            contiguous = tensor.detach().cpu().contiguous()
            raw = contiguous.view(torch.uint8).numpy().tobytes()
            return hashlib.sha256(raw).hexdigest()

        payload = {
            "path_id": self.path_id,
            "probe_id": self.probe_id,
            "loss_id": self.loss_id,
            "registry_hash": self.parameter_pre_state.registry_hash,
            "names": list(self.parameter_pre_state),
            "shapes": {name: list(self.parameter_pre_state[name].shape) for name in self.parameter_pre_state},
            "dtype": {name: str(self.parameter_pre_state[name].dtype) for name in self.parameter_pre_state},
            "device": {name: str(self.parameter_pre_state[name].device) for name in self.parameter_pre_state},
            "parameter_pre_sha256": {
                name: tensor_digest(self.parameter_pre_state[name]) for name in self.parameter_pre_state
            },
            "parameter_post_sha256": {
                name: tensor_digest(self.parameter_post_state[name]) for name in self.parameter_post_state
            },
            "accumulation_dtype": str(self.accumulation_dtype),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_artifact(
        self,
        *,
        parameter_pre_bundle_ref: str,
        parameter_pre_bundle_manifest_hash: str,
        parameter_post_bundle_ref: str,
        parameter_post_bundle_manifest_hash: str,
    ) -> dict[str, object]:
        """生成双端点安全 bundle 引用的路径合同 manifest。"""

        from ..contracts.artifacts import validate_path_spec_artifact
        from ..contracts.jsonio import canonical_json_hash

        payload: dict[str, object] = {
            "schema_version": "path-spec-v1",
            "path_id": self.path_id,
            "probe_id": self.probe_id,
            "loss_id": self.loss_id,
            "accumulation_dtype": (
                "float64" if self.accumulation_dtype == torch.float64 else "float32"
            ),
            "registry_hash": self.parameter_pre_state.registry_hash,
            "parameter_names": list(self.parameter_pre_state),
            "parameter_pre_bundle_ref": parameter_pre_bundle_ref,
            "parameter_pre_bundle_manifest_hash": parameter_pre_bundle_manifest_hash,
            "parameter_post_bundle_ref": parameter_post_bundle_ref,
            "parameter_post_bundle_manifest_hash": parameter_post_bundle_manifest_hash,
            "path_identity_hash": self.identity_hash,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        validate_path_spec_artifact(payload)
        return payload


@dataclass(frozen=True, slots=True)
class PathIntegralResult:
    """单条路径、单个求积规则的逐坐标归因结果。"""

    rule: QuadratureRule
    path_identity_hash: str
    average_gradient: TensorMap
    signed: TensorMap
    positive: TensorMap
    negative_mass: TensorMap
    absolute: TensorMap
    endpoint_loss_pre: float | None
    endpoint_loss_post: float | None
    loss_drop: float | None
    completeness_absolute_residual: float | None
    completeness_relative_residual: float | None
    completeness_l1_scaled_residual: float | None
    node_losses: tuple[float | None, ...]
    unique_gradient_evaluations: int

    @property
    def contributions(self) -> TensorMap:
        """``signed`` 的可读别名。"""

        return self.signed

    def to_artifact(
        self,
        *,
        contribution_bundle_ref: str,
        contribution_bundle_manifest_hash: str,
    ) -> dict[str, object]:
        """生成标量完备性结果与贡献 bundle 共同绑定的 manifest。"""

        from ..contracts.artifacts import validate_path_integral_result_artifact
        from ..contracts.jsonio import canonical_json_hash

        payload: dict[str, object] = {
            "schema_version": "path-integral-result-v1",
            "path_identity_hash": self.path_identity_hash,
            "rule": {
                "name": self.rule.name,
                "version": self.rule.version,
                "kind": self.rule.kind,
                "nodes": [float(item) for item in self.rule.nodes.tolist()],
                "weights": [float(item) for item in self.rule.weights.tolist()],
                "subintervals": self.rule.subintervals,
                "exact_polynomial_degree": self.rule.exact_polynomial_degree,
                "theoretical_order": self.rule.theoretical_order,
            },
            "contribution_bundle_ref": contribution_bundle_ref,
            "contribution_bundle_manifest_hash": contribution_bundle_manifest_hash,
            "views": ["signed", "positive", "negative_mass", "absolute"],
            "endpoint_loss_pre": self.endpoint_loss_pre,
            "endpoint_loss_post": self.endpoint_loss_post,
            "loss_drop": self.loss_drop,
            "completeness_absolute_residual": self.completeness_absolute_residual,
            "completeness_relative_residual": self.completeness_relative_residual,
            "completeness_l1_scaled_residual": self.completeness_l1_scaled_residual,
            "node_losses": list(self.node_losses),
            "unique_gradient_evaluations": self.unique_gradient_evaluations,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        validate_path_integral_result_artifact(payload)
        return payload

    def view(self, name: str) -> TensorMap:
        """按冻结名称读取 signed/positive/negative_mass/absolute 视图。"""

        if name not in {"signed", "positive", "negative_mass", "absolute"}:
            raise CoreContractError(f"未知路径贡献视图: {name!r}")
        return getattr(self, name)

    def ranked_coordinates(
        self,
        view: str = "absolute",
        *,
        descending: bool = True,
    ) -> tuple[tuple[str, float], ...]:
        """按数值排序全部标量坐标，并以 canonical ID 稳定解决并列。

        coordinate ID 采用 ``tensor_name#十二位扁平索引``，不依赖 Python 对象地址。
        """

        selected = self.view(view)
        rows: list[tuple[str, float]] = []
        for name, tensor in selected.items():
            for index, value in enumerate(tensor.detach().to(torch.float64).reshape(-1).tolist()):
                rows.append((f"{name}#{index:012d}", float(value)))
        if descending:
            rows.sort(key=lambda row: (-row[1], row[0]))
        else:
            rows.sort(key=lambda row: (row[1], row[0]))
        return tuple(rows)

    def group_summary(
        self,
        view: str = "absolute",
        *,
        tag: str = "layer",
    ) -> Mapping[str, Mapping[str, float | int | None]]:
        """返回层/模块的总量、每参数均值与非负质量占比。

        signed 不是概率质量，因此其 ``mass_fraction`` 固定为 ``None``；positive、
        negative_mass 和 absolute 在总质量为零时也返回 ``None``，而不是用 epsilon
        伪造占比。
        """

        selected = self.view(view)
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name, tensor in selected.items():
            if selected.registry is not None:
                group = selected.registry.record(name).tags.get(tag)
                if group is None:
                    raise CoreContractError(f"参数 {name!r} 缺少聚合 tag {tag!r}")
            elif tag == "module":
                group = name.rsplit(".", 1)[0] if "." in name else "<root>"
            elif tag == "layer":
                group = name.split(".", 1)[0]
            else:
                raise CoreContractError("无 registry TensorMap 只支持 layer/module 聚合")
            totals[group] = totals.get(group, 0.0) + float(
                tensor.detach().to(torch.float64).sum().item()
            )
            counts[group] = counts.get(group, 0) + tensor.numel()
        total_mass = sum(totals.values()) if view != "signed" else 0.0
        return MappingProxyType(
            {
                group: MappingProxyType(
                    {
                        "sum": totals[group],
                        "mean_per_parameter": totals[group] / counts[group],
                        "parameter_count": counts[group],
                        "mass_fraction": (
                            totals[group] / total_mass
                            if view != "signed" and total_mass > 0
                            else None
                        ),
                    }
                )
                for group in sorted(totals)
            }
        )


GradientCallback = Callable[[float, TensorMap], TensorMap]
LossCallback = Callable[[TensorMap], torch.Tensor | float]


def _loss_value(loss_fn: LossCallback, state: TensorMap) -> float:
    raw = loss_fn(state.clone())
    if isinstance(raw, torch.Tensor):
        if raw.ndim != 0:
            raise CoreContractError("路径 loss callback 必须返回标量")
        value = float(raw.detach().to(torch.float64).item())
    else:
        value = float(raw)
    if not math.isfinite(value):
        raise NumericalError("路径 loss callback 返回 NaN/Inf")
    return value


def integrate_path(
    path: PathSpec,
    rule: QuadratureRule,
    gradient_fn: GradientCallback,
    *,
    loss_fn: LossCallback | None = None,
    residual_epsilon: float = 1e-12,
) -> PathIntegralResult:
    """沿线性端点路径执行求积并返回完备性视图。

    ``gradient_fn(alpha, state)`` 必须返回与 path 同坐标的梯度；它不得修改输入
    ``state`` 或任何训练状态。若提供 ``loss_fn``，两个端点与可选节点 loss 都由
    同一个 callback 计算，杜绝从训练日志混入另一 reduction 的端点损失。
    """

    if not math.isfinite(residual_epsilon) or residual_epsilon <= 0:
        raise CoreContractError("residual_epsilon 必须为有限正数")
    template = path.parameter_pre_state.to(dtype=path.accumulation_dtype)
    average = TensorMap.zeros_like(template, dtype=path.accumulation_dtype)
    node_losses: list[float | None] = []
    for node_tensor, weight_tensor in zip(rule.nodes, rule.weights, strict=True):
        alpha = float(node_tensor.item())
        weight = float(weight_tensor.item())
        state = path.interpolate(alpha)
        gradient = gradient_fn(alpha, state.clone())
        template.assert_compatible(gradient)
        gradient.assert_finite()
        average = average + gradient.to(dtype=path.accumulation_dtype) * weight
        node_losses.append(None if loss_fn is None else _loss_value(loss_fn, state))
    average.assert_finite()
    delta = path.delta.to(dtype=path.accumulation_dtype)
    signed = -(delta * average)
    signed.assert_finite()
    positive = signed.map(lambda value: value.clamp_min(0))
    negative = signed.map(lambda value: (-value).clamp_min(0))
    absolute = positive + negative

    if loss_fn is None:
        loss_pre = loss_post = loss_drop = None
        residual_abs = residual_rel = residual_l1 = None
    else:
        loss_pre = _loss_value(loss_fn, path.parameter_pre_state)
        loss_post = _loss_value(loss_fn, path.parameter_post_state)
        loss_drop = loss_pre - loss_post
        signed_sum = float(signed.scalar_sum(dtype=torch.float64).item())
        absolute_mass = float(absolute.scalar_sum(dtype=torch.float64).item())
        residual_abs = abs(signed_sum - loss_drop)
        residual_rel = residual_abs / (abs(loss_drop) + residual_epsilon)
        residual_l1 = residual_abs / (absolute_mass + residual_epsilon)
    return PathIntegralResult(
        rule=rule,
        path_identity_hash=path.identity_hash,
        average_gradient=average,
        signed=signed,
        positive=positive,
        negative_mass=negative,
        absolute=absolute,
        endpoint_loss_pre=loss_pre,
        endpoint_loss_post=loss_post,
        loss_drop=loss_drop,
        completeness_absolute_residual=residual_abs,
        completeness_relative_residual=residual_rel,
        completeness_l1_scaled_residual=residual_l1,
        node_losses=tuple(node_losses),
        unique_gradient_evaluations=rule.unique_gradient_evaluations,
    )


def integrate_scalar_function(
    function: Callable[[float], float],
    rule: QuadratureRule,
) -> float:
    """用于解析 fixture 的标量求积 helper。"""

    total = 0.0
    for node, weight in zip(rule.nodes.tolist(), rule.weights.tolist(), strict=True):
        value = float(function(float(node)))
        if not math.isfinite(value):
            raise NumericalError("标量被积函数返回 NaN/Inf")
        total += float(weight) * value
    if not math.isfinite(total):
        raise NumericalError("标量求积结果为 NaN/Inf")
    return total
