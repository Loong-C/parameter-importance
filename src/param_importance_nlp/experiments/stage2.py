"""Stage 2 固定状态 reference、paired runner 与 decision artifact。

本模块负责把冻结 draw manifest 交给 provider，并调用 ``core`` 中唯一的公式
实现。它不复制 raw/double/U 数学内核，也不训练模型。所有便捷构造器默认
生成 ``local_fixture`` scope；正式 decision 只能从外部已验收 artifact 加载，
不能由本机 synthetic 结果升级得到。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from param_importance_nlp.contracts import GateStatus, validate_gate_id
from param_importance_nlp.contracts.artifacts import validate_estimator_decision_artifact
from param_importance_nlp.contracts.immutable import (
    freeze_json_mapping,
    thaw_json_value,
)
from param_importance_nlp.contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from param_importance_nlp.providers import FixedStateGradientProvider, GradientBatch
from param_importance_nlp.runtime.tensor_bundle import (
    load_tensor_bundle,
    publish_tensor_bundle,
)

from .sampling import FormalDecisionBlocked, PrimaryPairDecision, RepetitionMapping


COST_SEMANTICS = (
    "scientific_equal_sample_cost",
    "isolated_estimator_cost",
    "online_training_incremental_cost",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _tensor_items(value: object) -> Iterable[tuple[str, object]]:
    if not isinstance(value, Mapping):
        raise TypeError("估计器向量必须实现 Mapping[str, tensor]")
    return ((str(name), tensor) for name, tensor in value.items())


def _to_numpy(tensor: object) -> np.ndarray:
    """只读地把 Torch/NumPy 张量转换成 CPU FP64 数组。"""

    if hasattr(tensor, "detach"):
        tensor = tensor.detach()  # type: ignore[union-attr]
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()  # type: ignore[union-attr]
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()  # type: ignore[union-attr]
    array = np.asarray(tensor, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("估计向量包含 NaN/Inf")
    return array


def _vector_digest(value: object) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(_tensor_items(value), key=lambda item: item[0]):
        array = np.ascontiguousarray(_to_numpy(tensor))
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _clone_vector(value: object) -> object:
    """冻结小型向量，避免 shard 发布后被调用方原地修改。"""

    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copied: dict[str, np.ndarray] = {}
    for name, tensor in _tensor_items(value):
        array = np.array(_to_numpy(tensor), copy=True)
        array.setflags(write=False)
        copied[name] = array
    return MappingProxyType(copied)


def _max_abs_difference(left: object, right: object) -> float:
    left_items = dict(_tensor_items(left))
    right_items = dict(_tensor_items(right))
    if set(left_items) != set(right_items):
        raise ValueError("比较向量的参数名集合不一致")
    maximum = 0.0
    for name in left_items:
        lhs = _to_numpy(left_items[name])
        rhs = _to_numpy(right_items[name])
        if lhs.shape != rhs.shape:
            raise ValueError(f"参数 {name!r} 的 shape 不一致")
        maximum = max(maximum, float(np.max(np.abs(lhs - rhs), initial=0.0)))
    return maximum


_WEIGHTING_CONTRACT_FIELDS = (
    "statistical_unit",
    "weight_unit",
    "sampling_design",
    "weights_exogenous",
    "common_mean_assumption",
)


def _shared_weighting_contract(
    provider: FixedStateGradientProvider,
    batches: Sequence[GradientBatch],
) -> dict[str, object]:
    """校验并返回一组梯度批次共享的加权 U 假设。

    统计语义由 provider 声明、由每个 ``GradientBatch`` 随数值一起传输。这里
    同时比较两边，防止 adapter 在不同调用间悄悄改变 statistical unit，或
    runner 因为看到等权样本就替 provider 补上外生性/同均值假设。缺字段、
    跨批不一致、provider/batch 不一致都会在任何 U 计算之前失败。
    """

    if not batches:
        raise ValueError("GRADIENT_BATCHES_EMPTY")
    try:
        declared = {
            field_name: getattr(provider, field_name)
            for field_name in _WEIGHTING_CONTRACT_FIELDS
        }
        observed = batches[0].weighting_assumptions
    except AttributeError as exc:
        raise ValueError("GRADIENT_WEIGHTING_CONTRACT_MISSING") from exc
    if declared != observed:
        raise ValueError("GRADIENT_PROVIDER_BATCH_CONTRACT_MISMATCH")
    for batch in batches[1:]:
        if batch.weighting_assumptions != observed:
            raise ValueError("GRADIENT_BATCH_WEIGHTING_CONTRACT_DRIFT")
    return dict(observed)


class EstimatorKernel(Protocol):
    """paired runner 所需的核心估计器适配器。"""

    def tensor_map(self, batch: GradientBatch) -> object:
        """把 provider 梯度映射转换成核心层张量容器。"""

    def raw(self, mean_gradient: object) -> object:
        """计算同批平均梯度平方。"""

    def weighted_mean(
        self, gradients: Sequence[object], weights: Sequence[float]
    ) -> object:
        """把互斥统计单元的平均梯度按原始统计权重合并。"""

    def double(self, left: object, right: object) -> object:
        """计算两个独立半区平均梯度的乘积。"""

    def u(
        self,
        gradients: Sequence[object],
        weights: Sequence[float],
        *,
        statistical_unit: str,
        weight_unit: str,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> object:
        """按等权或加权去对角公式计算 signed U，并透传统计假设。"""


class CoreEstimatorKernel:
    """对 :mod:`param_importance_nlp.core` 的延迟导入适配器。

    导入发生在真正执行 synthetic runner 时，因此仅查看配置或导入实验模块
    不要求 Torch，更不会导入 Hugging Face。在线 profile 用 FP32，reference
    fixture 可显式选择 FP64；dtype 选择会保存在结果元数据中。
    """

    def __init__(self, *, accumulation_dtype: str = "float64") -> None:
        if accumulation_dtype not in {"float32", "float64"}:
            raise ValueError("accumulation_dtype 只能是 float32 或 float64")
        self.accumulation_dtype = accumulation_dtype

    def _api(self) -> tuple[object, ...]:
        try:
            import torch

            from param_importance_nlp.core.estimators import (
                double_sample_importance,
                equal_u_importance,
                raw_importance,
                weighted_u_importance,
            )
            from param_importance_nlp.core.sufficient_statistics import (
                EqualSufficientStatistics,
                WeightedSufficientStatistics,
            )
            from param_importance_nlp.core.tensors import TensorMap
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - 环境防线
            raise RuntimeError("执行 Stage 2 runner 需要已安装的 Torch 与 core 模块") from exc
        dtype = torch.float32 if self.accumulation_dtype == "float32" else torch.float64
        return (
            torch,
            TensorMap,
            EqualSufficientStatistics,
            WeightedSufficientStatistics,
            raw_importance,
            double_sample_importance,
            equal_u_importance,
            weighted_u_importance,
            dtype,
        )

    def tensor_map(self, batch: GradientBatch) -> object:
        torch, TensorMap, *_rest, dtype = self._api()
        values = {
            name: torch.as_tensor(value, dtype=dtype).detach().clone()
            for name, value in batch.gradients.items()
        }
        return TensorMap(values)

    def raw(self, mean_gradient: object) -> object:
        *_, raw_importance, _double, _equal_u, _weighted_u, _dtype = self._api()
        return raw_importance(mean_gradient)

    def weighted_mean(
        self, gradients: Sequence[object], weights: Sequence[float]
    ) -> object:
        if not gradients or len(gradients) != len(weights):
            raise ValueError("gradients/weights 必须非空且数量一致")
        numeric = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight <= 0 for weight in numeric):
            raise ValueError("合并平均梯度的权重必须为有限正数")
        accumulator = gradients[0] * numeric[0]
        for gradient, weight in zip(gradients[1:], numeric[1:], strict=True):
            accumulator = accumulator + gradient * weight
        return accumulator / sum(numeric)

    def double(self, left: object, right: object) -> object:
        *_, _raw, double_sample_importance, _equal_u, _weighted_u, _dtype = self._api()
        return double_sample_importance(left, right)

    def u(
        self,
        gradients: Sequence[object],
        weights: Sequence[float],
        *,
        statistical_unit: str,
        weight_unit: str,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> object:
        (
            _torch,
            _TensorMap,
            EqualSufficientStatistics,
            WeightedSufficientStatistics,
            _raw,
            _double,
            equal_u_importance,
            weighted_u_importance,
            dtype,
        ) = self._api()
        if len(gradients) != len(weights):
            raise ValueError("gradients 与 weights 数量必须一致")
        if not gradients:
            raise ValueError("U-statistic 至少需要两个梯度统计单元")
        for field_name, value in (
            ("statistical_unit", statistical_unit),
            ("weight_unit", weight_unit),
            ("sampling_design", sampling_design),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 必须由 provider 显式声明")
        if type(weights_exogenous) is not bool:
            raise TypeError("weights_exogenous 必须是显式 bool")
        if type(common_mean_assumption) is not bool:
            raise TypeError("common_mean_assumption 必须是显式 bool")
        if all(math.isclose(weight, weights[0], rel_tol=0.0, abs_tol=0.0) for weight in weights):
            statistics = EqualSufficientStatistics.from_samples(
                gradients,
                accumulation_dtype=dtype,
                statistical_unit=statistical_unit,
                sampling_design=sampling_design,
            )
            return equal_u_importance(statistics)
        if not weights_exogenous or not common_mean_assumption:
            raise ValueError(
                "WEIGHTED_U_UNBIASEDNESS_ASSUMPTIONS_NOT_DECLARED: "
                "非等权 U 要求 weights_exogenous=true 且 "
                "common_mean_assumption=true"
            )
        statistics = WeightedSufficientStatistics.from_samples(
            gradients,
            weights,
            accumulation_dtype=dtype,
            statistical_unit=statistical_unit,
            weight_unit=weight_unit,
            sampling_design=sampling_design,
            weights_exogenous=weights_exogenous,
            common_mean_assumption=common_mean_assumption,
        )
        return weighted_u_importance(
            statistics,
            require_unbiasedness_assumptions=True,
        )


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    """有限样本 reference 三视图及其身份。

    ``bias_reference`` 是 block means 的去对角 U，``cross_reference`` 是独立
    A/B 均值梯度乘积，``ranking_reference`` 是合并均值平方。最后一项只供
    排序，不能替代无偏 bias reference。
    """

    reference_id: str
    bias_reference: object
    cross_reference: object
    ranking_reference: object
    sample_count_a: int
    sample_count_b: int
    block_size: int
    registry_hash: str
    scope: str = "local_fixture"
    formal_eligible: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope != "local_fixture" or self.formal_eligible:
            raise FormalDecisionBlocked("本机 ReferenceRunner 只能发布 local_fixture reference")
        if self.sample_count_a <= 0 or self.sample_count_b <= 0 or self.block_size <= 0:
            raise ValueError("reference 样本数与 block_size 必须严格为正")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="ReferenceResult.metadata"),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "reference_id": self.reference_id,
                    "bias": _vector_digest(self.bias_reference),
                    "cross": _vector_digest(self.cross_reference),
                    "ranking": _vector_digest(self.ranking_reference),
                    "sample_count_a": self.sample_count_a,
                    "sample_count_b": self.sample_count_b,
                    "block_size": self.block_size,
                    "registry_hash": self.registry_hash,
                    "scope": self.scope,
                }
            )
        ).hexdigest()

    def to_artifact(
        self,
        *,
        tensor_bundle_ref: str,
        tensor_bundle_manifest_hash: str,
    ) -> dict[str, object]:
        """生成只引用安全 tensor bundle 的公共 reference manifest。

        三个数值视图不嵌入 JSON；调用方必须先把它们发布到同一不可变 bundle，
        再将权威 manifest SHA-256 传入。此方法只描述当前 scope，不改变正式资格。
        """

        payload: dict[str, object] = {
            "schema_version": "reference-result-v1",
            "reference_id": self.reference_id,
            "bias_reference_hash": _vector_digest(self.bias_reference),
            "cross_reference_hash": _vector_digest(self.cross_reference),
            "ranking_reference_hash": _vector_digest(self.ranking_reference),
            "sample_count_a": self.sample_count_a,
            "sample_count_b": self.sample_count_b,
            "block_size": self.block_size,
            "registry_hash": self.registry_hash,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "metadata": thaw_json_value(self.metadata),
            "tensor_bundle_ref": tensor_bundle_ref,
            "tensor_bundle_manifest_hash": tensor_bundle_manifest_hash,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        # 复用共享边界验证器，确保调用方不能生成绝对路径或未绑定 manifest。
        from param_importance_nlp.contracts.artifacts import (
            validate_reference_result_artifact,
        )

        validate_reference_result_artifact(payload)
        return payload


class ReferenceRunner:
    """流式概念等价的本机 reference 构建器。

    本机 fixture 很小，所以这里保留 block mean 张量；正式服务器实现可以用
    同一接口替换为不可变 block shards，而不改变 reference 的三种数学身份。
    """

    def __init__(
        self,
        provider: FixedStateGradientProvider,
        *,
        kernel: EstimatorKernel | None = None,
    ) -> None:
        self.provider = provider
        self.kernel = kernel or CoreEstimatorKernel(accumulation_dtype="float64")

    def run(
        self,
        *,
        reference_id: str,
        draws_a: Sequence[object],
        draws_b: Sequence[object],
        block_size: int,
    ) -> ReferenceResult:
        """完整消费已冻结 A/B draws，不执行 early stopping。"""

        if not reference_id:
            raise ValueError("reference_id 不能为空")
        if not draws_a or not draws_b:
            raise ValueError("reference A/B 都不能为空")
        if len(draws_a) != len(draws_b):
            raise ValueError("one-shot reference A/B 必须使用相同的冻结样本量")
        if block_size <= 0:
            raise ValueError("block_size 必须严格为正")
        if len(draws_a) % block_size or len(draws_b) % block_size:
            raise ValueError("A/B 长度必须都能被 block_size 整除")
        if (len(draws_a) + len(draws_b)) // block_size < 2:
            raise ValueError("bias reference 至少需要两个独立 blocks")
        draw_ids_a = {
            str(getattr(draw, "draw_id")) for draw in draws_a if hasattr(draw, "draw_id")
        }
        draw_ids_b = {
            str(getattr(draw, "draw_id")) for draw in draws_b if hasattr(draw, "draw_id")
        }
        if draw_ids_a.intersection(draw_ids_b):
            raise ValueError("reference A/B 的 draw IDs 必须互不重用")
        streams_a = {getattr(draw, "stream") for draw in draws_a if hasattr(draw, "stream")}
        streams_b = {getattr(draw, "stream") for draw in draws_b if hasattr(draw, "stream")}
        if streams_a and streams_a != {"reference_A"}:
            raise ValueError("reference A 必须来自 reference_A stream")
        if streams_b and streams_b != {"reference_B"}:
            raise ValueError("reference B 必须来自 reference_B stream")

        before = self.provider.state_digest()
        block_batches: list[GradientBatch] = []
        split_index = 0
        for draw_index, draws in enumerate((draws_a, draws_b)):
            for start in range(0, len(draws), block_size):
                block_batches.append(self.provider.gradient(draws[start : start + block_size]))
            if draw_index == 0:
                split_index = len(block_batches)
        weighting_contract = _shared_weighting_contract(self.provider, block_batches)
        block_maps = [self.kernel.tensor_map(batch) for batch in block_batches]
        bias_reference = self.kernel.u(
            block_maps,
            [batch.statistical_weight for batch in block_batches],
            **weighting_contract,
        )
        mean_a = self.kernel.weighted_mean(
            block_maps[:split_index],
            [batch.statistical_weight for batch in block_batches[:split_index]],
        )
        mean_b = self.kernel.weighted_mean(
            block_maps[split_index:],
            [batch.statistical_weight for batch in block_batches[split_index:]],
        )
        cross_reference = self.kernel.double(mean_a, mean_b)
        merged_mean = self.kernel.weighted_mean(
            block_maps,
            [batch.statistical_weight for batch in block_batches],
        )
        ranking_reference = self.kernel.raw(merged_mean)
        self.provider.assert_unchanged(before)
        return ReferenceResult(
            reference_id=reference_id,
            bias_reference=bias_reference,
            cross_reference=cross_reference,
            ranking_reference=ranking_reference,
            sample_count_a=len(draws_a),
            sample_count_b=len(draws_b),
            block_size=block_size,
            registry_hash=self.provider.registry_hash,
            metadata={
                "fixed_state_id": self.provider.fixed_state_id,
                "state_digest": before,
                "one_shot": True,
                "early_stopping": False,
                "weighting_assumptions": weighting_contract,
            },
        )


@dataclass(frozen=True, slots=True)
class PairedEstimatorResult:
    """一次公平预算 repetition 的 signed estimator vectors。"""

    unit_id: str
    mapping_digest: str
    registry_hash: str
    raw: object
    double: object
    u_by_m: Mapping[int, object]
    sample_collision_count: int
    m2_double_max_abs_error: float
    gradient_evaluations: int
    formula_seconds: float
    state_digest: str
    weighting_assumptions: Mapping[str, object]
    scope: str = "local_fixture"
    formal_eligible: bool = False

    def __post_init__(self) -> None:
        if self.scope != "local_fixture" or self.formal_eligible:
            raise FormalDecisionBlocked("本机 runner 结果不能标记为 formal")
        if 2 not in self.u_by_m:
            raise ValueError("paired result 必须包含 M=2 U/double 不变量")
        if self.m2_double_max_abs_error < 0 or not math.isfinite(
            self.m2_double_max_abs_error
        ):
            raise ValueError("M=2 误差必须是非负有限数")
        object.__setattr__(self, "u_by_m", MappingProxyType(dict(self.u_by_m)))
        object.__setattr__(
            self,
            "weighting_assumptions",
            freeze_json_mapping(
                self.weighting_assumptions,
                field="PairedEstimatorResult.weighting_assumptions",
            ),
        )

    @property
    def vectors(self) -> Mapping[str, object]:
        values: dict[str, object] = {"raw": self.raw, "double": self.double}
        values.update({f"u_m{m}": value for m, value in self.u_by_m.items()})
        return MappingProxyType(values)

    @property
    def digest(self) -> str:
        # 性能时间会受机器噪声影响，因此不进入科学结果身份摘要。
        return hashlib.sha256(
            _canonical_json(
                {
                    "unit_id": self.unit_id,
                    "mapping_digest": self.mapping_digest,
                    "registry_hash": self.registry_hash,
                    "vectors": {
                        name: _vector_digest(value)
                        for name, value in sorted(self.vectors.items())
                    },
                    "sample_collision_count": self.sample_collision_count,
                    "m2_double_max_abs_error": self.m2_double_max_abs_error,
                    "state_digest": self.state_digest,
                    "weighting_assumptions": thaw_json_value(
                        self.weighting_assumptions
                    ),
                }
            )
        ).hexdigest()


class PairedEstimatorRunner:
    """一次梯度池配对计算 raw、double 与所有冻结 M 的 U。"""

    def __init__(
        self,
        provider: FixedStateGradientProvider,
        *,
        kernel: EstimatorKernel | None = None,
        m2_tolerance: float = 1e-10,
    ) -> None:
        if m2_tolerance < 0 or not math.isfinite(m2_tolerance):
            raise ValueError("m2_tolerance 必须是非负有限数")
        self.provider = provider
        self.kernel = kernel or CoreEstimatorKernel(accumulation_dtype="float64")
        self.m2_tolerance = m2_tolerance

    def run(self, mapping: RepetitionMapping) -> PairedEstimatorResult:
        """在不修改 provider 固定状态的前提下执行一个 repetition。"""

        before = self.provider.state_digest()
        # 只求值 M_max 个基础 microbatches；raw、double 和更粗 M 都从同一梯度池
        # 合并，避免把 B/M 变化与额外梯度抽样混在一起。
        max_m = max(mapping.m_values)
        base_batches = [self.provider.gradient(group) for group in mapping.groups(max_m)]
        weighting_contract = _shared_weighting_contract(self.provider, base_batches)
        base_maps = [self.kernel.tensor_map(batch) for batch in base_batches]
        base_weights = [batch.statistical_weight for batch in base_batches]
        full_mean = self.kernel.weighted_mean(base_maps, base_weights)
        formula_start = time.perf_counter()
        raw = self.kernel.raw(full_mean)

        base_half = max_m // 2
        left_mean = self.kernel.weighted_mean(
            base_maps[:base_half], base_weights[:base_half]
        )
        right_mean = self.kernel.weighted_mean(
            base_maps[base_half:], base_weights[base_half:]
        )
        double = self.kernel.double(left_mean, right_mean)

        u_by_m: dict[int, object] = {}
        for microbatch_count in mapping.m_values:
            merge_width = max_m // microbatch_count
            gradients: list[object] = []
            weights: list[float] = []
            for start in range(0, max_m, merge_width):
                chunk_maps = base_maps[start : start + merge_width]
                chunk_weights = base_weights[start : start + merge_width]
                gradients.append(self.kernel.weighted_mean(chunk_maps, chunk_weights))
                weights.append(sum(chunk_weights))
            u_by_m[microbatch_count] = self.kernel.u(
                gradients,
                weights,
                **weighting_contract,
            )
        formula_seconds = time.perf_counter() - formula_start
        m2_error = _max_abs_difference(u_by_m[2], double)
        if m2_error > self.m2_tolerance:
            raise RuntimeError(
                f"M=2 U 与 double 不一致: max_abs={m2_error:.6g}, "
                f"tolerance={self.m2_tolerance:.6g}"
            )
        self.provider.assert_unchanged(before)
        return PairedEstimatorResult(
            unit_id=mapping.repetition_id,
            mapping_digest=mapping.digest,
            registry_hash=self.provider.registry_hash,
            raw=raw,
            double=double,
            u_by_m=u_by_m,
            sample_collision_count=mapping.sample_collision_count,
            m2_double_max_abs_error=m2_error,
            gradient_evaluations=max_m,
            formula_seconds=formula_seconds,
            state_digest=before,
            weighting_assumptions=weighting_contract,
        )


@dataclass(frozen=True, slots=True)
class SufficientStatisticShard:
    """不可变 repetition 估计向量 shard。

    正式实现可把完整向量替换为 count/sum/sum-of-squares；本机 fixture 保留
    小向量用于独立验证 reducer。``digest`` 不包含 attempt 时间等非确定字段。
    """

    unit_id: str
    attempt_id: str
    input_hash: str
    vectors: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.unit_id or not self.attempt_id or not self.input_hash:
            raise ValueError("unit_id、attempt_id 与 input_hash 都不能为空")
        if not self.vectors:
            raise ValueError("shard vectors 不能为空")
        object.__setattr__(
            self,
            "vectors",
            MappingProxyType(
                {name: _clone_vector(value) for name, value in self.vectors.items()}
            ),
        )

    @classmethod
    def from_result(
        cls,
        result: PairedEstimatorResult,
        *,
        attempt_id: str,
    ) -> "SufficientStatisticShard":
        return cls(
            unit_id=result.unit_id,
            attempt_id=attempt_id,
            input_hash=result.mapping_digest,
            vectors=result.vectors,
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "unit_id": self.unit_id,
                    "attempt_id": self.attempt_id,
                    "input_hash": self.input_hash,
                    "vectors": {
                        name: _vector_digest(value)
                        for name, value in sorted(self.vectors.items())
                    },
                }
            )
        ).hexdigest()


SHARD_COMMIT_SCHEMA = "stage2.sufficient-stat-shard-commit.v1"


class ShardArtifactStore:
    """不可变 sufficient-stat shard 的本地两阶段 artifact store。

    张量先发布到 ``objects/<shard-digest>`` 的安全 TensorBundle；只有独立的
    canonical commit 发布后，该 shard 才属于恢复集合。对象发布后、commit
    发布前崩溃只会留下可诊断孤儿，不会导致重复计数。writer lock 使用排他创建，
    同一目录在任意时刻只允许一个 reducer 写入。
    """

    def __init__(self, root: str | Path, *, acquire_writer: bool = True) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / "writer.lock"
        self._lock_descriptor = -1
        self._lock_token: str | None = None
        if acquire_writer:
            self.acquire_writer()

    def acquire_writer(self) -> None:
        """排他取得单写者身份；已有 writer 时绝不偷锁。"""

        if self._lock_token is not None:
            return
        descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Windows 的 msvcrt byte-range lock 要求被锁范围已有至少一个字节。
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows 是本任务的主验证平台
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            token = uuid.uuid4().hex
            value = {
                "schema_version": "stage2.shard-writer-lock.v1",
                "pid": os.getpid(),
                "token": token,
            }
            payload = canonical_json_bytes(value)
            os.lseek(descriptor, 0, os.SEEK_SET)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, payload)
            os.fsync(descriptor)
        except BaseException as exc:
            os.close(descriptor)
            raise RuntimeError(f"SHARD_WRITER_ALREADY_ACTIVE:{self.root}") from exc
        self._lock_descriptor = descriptor
        self._lock_token = token

    def close(self) -> None:
        """仅释放自己持有的 writer lock，不改变任何 shard artifact。"""

        if self._lock_token is None or self._lock_descriptor < 0:
            return
        try:
            descriptor = self._lock_descriptor
            os.lseek(descriptor, 0, os.SEEK_SET)
            payload = os.read(descriptor, os.fstat(descriptor).st_size)
            value = loads_strict_json(payload)
            if (
                canonical_json_bytes(value) != payload
                or not isinstance(value, dict)
                or value.get("token") != self._lock_token
            ):
                raise RuntimeError("SHARD_WRITER_LOCK_OWNERSHIP_LOST")
        finally:
            descriptor = self._lock_descriptor
            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - Windows 是本任务的主验证平台
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                self._lock_descriptor = -1
            self._lock_token = None

    def __enter__(self) -> "ShardArtifactStore":
        if self._lock_token is None:
            self.acquire_writer()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _commit_filename(unit_id: str) -> str:
        return hashlib.sha256(unit_id.encode("utf-8")).hexdigest() + ".json"

    @staticmethod
    def _state(shard: SufficientStatisticShard) -> dict[str, object]:
        return {
            "vectors": {
                method: {
                    name: np.array(_to_numpy(tensor), copy=True)
                    for name, tensor in _tensor_items(vector)
                }
                for method, vector in shard.vectors.items()
            }
        }

    @staticmethod
    def _from_state(
        state: object,
        *,
        unit_id: str,
        attempt_id: str,
        input_hash: str,
    ) -> SufficientStatisticShard:
        if not isinstance(state, dict) or set(state) != {"vectors"}:
            raise ValueError("SHARD_BUNDLE_STATE_FIELDS_MISMATCH")
        vectors = state["vectors"]
        if not isinstance(vectors, dict) or not vectors:
            raise ValueError("SHARD_BUNDLE_VECTORS_INVALID")
        normalized: dict[str, Mapping[str, object]] = {}
        for method, value in vectors.items():
            if not isinstance(method, str) or not isinstance(value, dict) or not value:
                raise ValueError("SHARD_BUNDLE_METHOD_INVALID")
            if any(not isinstance(name, str) for name in value):
                raise ValueError("SHARD_BUNDLE_PARAMETER_NAME_INVALID")
            normalized[method] = value
        return SufficientStatisticShard(
            unit_id=unit_id,
            attempt_id=attempt_id,
            input_hash=input_hash,
            vectors=normalized,
        )

    @staticmethod
    def _validate_commit(value: object, *, path: Path) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError("SHARD_COMMIT_ROOT_NOT_OBJECT")
        expected = {
            "schema_version",
            "unit_id",
            "attempt_id",
            "input_hash",
            "shard_digest",
            "object_relative_path",
            "bundle_manifest_sha256",
            "commit_sha256",
        }
        if set(value) != expected:
            raise ValueError("SHARD_COMMIT_FIELDS_MISMATCH")
        if value.get("schema_version") != SHARD_COMMIT_SCHEMA:
            raise ValueError("SHARD_COMMIT_SCHEMA_MISMATCH")
        for field_name in ("unit_id", "attempt_id", "input_hash"):
            if not isinstance(value.get(field_name), str) or not value[field_name]:
                raise ValueError(f"SHARD_COMMIT_INVALID_{field_name.upper()}")
        for field_name in ("shard_digest", "bundle_manifest_sha256", "commit_sha256"):
            digest = value.get(field_name)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"SHARD_COMMIT_INVALID_{field_name.upper()}")
        if path.name != ShardArtifactStore._commit_filename(str(value["unit_id"])):
            raise ValueError("SHARD_COMMIT_NONCANONICAL_FILENAME")
        expected_object = f"objects/{value['shard_digest']}"
        if value.get("object_relative_path") != expected_object:
            raise ValueError("SHARD_COMMIT_OBJECT_PATH_MISMATCH")
        payload = dict(value)
        observed_hash = str(payload.pop("commit_sha256"))
        if canonical_json_hash(payload) != observed_hash:
            raise ValueError("SHARD_COMMIT_HASH_MISMATCH")
        return value

    def _load_commit(self, path: Path) -> tuple[SufficientStatisticShard, dict[str, object]]:
        value = self._validate_commit(load_canonical_json(path), path=path)
        object_path = self.root / str(value["object_relative_path"])
        state, bundle = load_tensor_bundle(object_path)
        if bundle.manifest_sha256 != value["bundle_manifest_sha256"]:
            raise ValueError("SHARD_BUNDLE_MANIFEST_HASH_MISMATCH")
        shard = self._from_state(
            state,
            unit_id=str(value["unit_id"]),
            attempt_id=str(value["attempt_id"]),
            input_hash=str(value["input_hash"]),
        )
        if shard.digest != value["shard_digest"]:
            raise ValueError("SHARD_CONTENT_DIGEST_MISMATCH")
        return shard, value

    def publish(self, shard: SufficientStatisticShard) -> bool:
        """幂等发布 shard；同 unit 的不同 attempt/content 立即拒绝。"""

        if self._lock_token is None:
            raise RuntimeError("SHARD_STORE_WRITER_LOCK_REQUIRED")
        commit_path = self.commits / self._commit_filename(shard.unit_id)
        if commit_path.exists():
            existing, _ = self._load_commit(commit_path)
            if existing.digest == shard.digest:
                return False
            raise ValueError(f"SHARD_UNIT_ALREADY_COMMITTED:{shard.unit_id}")

        object_path = self.objects / shard.digest
        if object_path.exists():
            state, bundle = load_tensor_bundle(object_path)
            observed = self._from_state(
                state,
                unit_id=shard.unit_id,
                attempt_id=shard.attempt_id,
                input_hash=shard.input_hash,
            )
            if observed.digest != shard.digest:
                raise ValueError("SHARD_ORPHAN_OBJECT_DIGEST_MISMATCH")
        else:
            bundle = publish_tensor_bundle(object_path, self._state(shard))
        # 重新加载完整对象后才允许发布权威 commit。
        state, verified = load_tensor_bundle(object_path)
        observed = self._from_state(
            state,
            unit_id=shard.unit_id,
            attempt_id=shard.attempt_id,
            input_hash=shard.input_hash,
        )
        if observed.digest != shard.digest or verified.manifest_sha256 != bundle.manifest_sha256:
            raise ValueError("SHARD_POST_PUBLISH_VERIFICATION_FAILED")
        payload: dict[str, object] = {
            "schema_version": SHARD_COMMIT_SCHEMA,
            "unit_id": shard.unit_id,
            "attempt_id": shard.attempt_id,
            "input_hash": shard.input_hash,
            "shard_digest": shard.digest,
            "object_relative_path": f"objects/{shard.digest}",
            "bundle_manifest_sha256": verified.manifest_sha256,
        }
        payload["commit_sha256"] = canonical_json_hash(payload)
        write_canonical_json(commit_path, payload)
        # commit 发布完成后立即按读取路径复核，避免内存状态领先于磁盘权威状态。
        restored, _ = self._load_commit(commit_path)
        if restored.digest != shard.digest:
            raise ValueError("SHARD_COMMIT_POST_PUBLISH_DRIFT")
        return True

    def restore(self) -> tuple[SufficientStatisticShard, ...]:
        """严格加载全部 commits，并按 canonical unit ID 排序。"""

        restored = [
            self._load_commit(path)[0]
            for path in sorted(self.commits.glob("*.json"))
        ]
        unit_ids = [shard.unit_id for shard in restored]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("SHARD_DUPLICATE_UNIT_COMMIT")
        return tuple(sorted(restored, key=lambda shard: shard.unit_id))

    def reconcile(self) -> dict[str, object]:
        """只读报告有效/损坏 commit 与孤儿对象，不删除任何内容。"""

        valid: list[str] = []
        invalid: list[dict[str, str]] = []
        referenced: set[str] = set()
        for path in sorted(self.commits.glob("*.json")):
            try:
                shard, value = self._load_commit(path)
            except Exception as exc:
                invalid.append({"commit": path.name, "reason": str(exc)})
            else:
                valid.append(shard.unit_id)
                referenced.add(Path(str(value["object_relative_path"])).name)
        return {
            "schema_version": "stage2.shard-artifact-reconcile.v1",
            "valid_unit_ids": sorted(valid),
            "invalid_commits": invalid,
            "orphan_objects": sorted(
                path.name
                for path in self.objects.iterdir()
                if path.is_dir() and path.name not in referenced
            ),
        }


@dataclass(frozen=True, slots=True)
class ReducedMoments:
    """逐方法、逐参数的 count/mean/M2 汇总。"""

    count: int
    mean: Mapping[str, np.ndarray]
    m2: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.count <= 0 or set(self.mean) != set(self.m2):
            raise ValueError("ReducedMoments count/参数集合无效")
        frozen_mean: dict[str, np.ndarray] = {}
        frozen_m2: dict[str, np.ndarray] = {}
        for name in self.mean:
            mean = np.array(self.mean[name], dtype=np.float64, copy=True)
            m2 = np.array(self.m2[name], dtype=np.float64, copy=True)
            if mean.shape != m2.shape or not np.all(np.isfinite(mean)) or not np.all(
                np.isfinite(m2)
            ):
                raise ValueError(f"ReducedMoments[{name!r}] shape/有限性无效")
            mean.setflags(write=False)
            m2.setflags(write=False)
            frozen_mean[name] = mean
            frozen_m2[name] = m2
        object.__setattr__(self, "mean", MappingProxyType(frozen_mean))
        object.__setattr__(self, "m2", MappingProxyType(frozen_m2))


@dataclass(frozen=True, slots=True)
class ReducedSufficientStatistics:
    """按 canonical unit ID 确定性归并后的结果。"""

    unit_ids: tuple[str, ...]
    methods: Mapping[str, ReducedMoments]
    digest: str

    def __post_init__(self) -> None:
        if not self.unit_ids or not self.methods:
            raise ValueError("ReducedSufficientStatistics 不能为空")
        if len(self.digest) != 64:
            raise ValueError("reducer digest 必须是 SHA-256")
        object.__setattr__(self, "methods", MappingProxyType(dict(self.methods)))


class DeterministicShardReducer:
    """单写入、排序归并、幂等读取的 fixture reducer。

    同一 shard 被重复发现时按 digest 幂等忽略；同一 unit 出现不同 attempt 或
    不同内容时立即失败，避免重试被重复计数。
    """

    def __init__(self, artifact_root: str | Path | None = None) -> None:
        self._shards: dict[str, SufficientStatisticShard] = {}
        self._artifact_store: ShardArtifactStore | None = None
        if artifact_root is not None:
            store = ShardArtifactStore(artifact_root)
            try:
                for shard in store.restore():
                    self._add_memory(shard)
            except BaseException:
                # 恢复失败时也必须释放本进程刚取得的 lock，否则一次损坏诊断会让
                # 后续只读修复误判为“仍有 writer 活动”。artifact 本身不做改写。
                store.close()
                raise
            self._artifact_store = store

    @classmethod
    def resume(cls, artifact_root: str | Path) -> "DeterministicShardReducer":
        """从全部权威 shard commits 恢复单写 reducer。"""

        return cls(artifact_root)

    def close(self) -> None:
        """释放持久化 store 的单写者锁；内存 reducer 无操作。"""

        if self._artifact_store is not None:
            self._artifact_store.close()

    def __enter__(self) -> "DeterministicShardReducer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _add_memory(self, shard: SufficientStatisticShard) -> bool:
        existing = self._shards.get(shard.unit_id)
        if existing is None:
            self._shards[shard.unit_id] = shard
            return True
        if existing.digest == shard.digest:
            return False
        raise ValueError(
            f"unit {shard.unit_id!r} 已有 attempt {existing.attempt_id!r}，拒绝重复计数"
        )

    def add(self, shard: SufficientStatisticShard) -> bool:
        existing = self._shards.get(shard.unit_id)
        if existing is not None:
            return self._add_memory(shard)
        if self._artifact_store is not None:
            self._artifact_store.publish(shard)
        return self._add_memory(shard)

    @property
    def persisted(self) -> bool:
        return self._artifact_store is not None

    def reconcile_artifacts(self) -> dict[str, object]:
        """返回持久化证据健康状态；内存 reducer 明确拒绝。"""

        if self._artifact_store is None:
            raise RuntimeError("SHARD_REDUCER_NOT_PERSISTENT")
        return self._artifact_store.reconcile()

    def reduce(self) -> ReducedSufficientStatistics:
        if not self._shards:
            raise ValueError("没有可归并 shards")
        unit_ids = tuple(sorted(self._shards))
        method_names = tuple(sorted(next(iter(self._shards.values())).vectors))
        for shard in self._shards.values():
            if tuple(sorted(shard.vectors)) != method_names:
                raise ValueError("不同 shard 的 method 集合不一致")

        methods: dict[str, ReducedMoments] = {}
        digest_payload: dict[str, object] = {"unit_ids": list(unit_ids), "methods": {}}
        for method in method_names:
            first_items = dict(_tensor_items(self._shards[unit_ids[0]].vectors[method]))
            names = tuple(sorted(first_items))
            means = {name: np.zeros_like(_to_numpy(first_items[name])) for name in names}
            m2 = {name: np.zeros_like(means[name]) for name in names}
            count = 0
            for unit_id in unit_ids:
                items = dict(_tensor_items(self._shards[unit_id].vectors[method]))
                if tuple(sorted(items)) != names:
                    raise ValueError("不同 shard 的参数名集合不一致")
                count += 1
                for name in names:
                    value = _to_numpy(items[name])
                    if value.shape != means[name].shape:
                        raise ValueError(f"参数 {name!r} 的 shard shape 不一致")
                    delta = value - means[name]
                    means[name] = means[name] + delta / count
                    m2[name] = m2[name] + delta * (value - means[name])
            methods[method] = ReducedMoments(
                count=count,
                mean=MappingProxyType(means),
                m2=MappingProxyType(m2),
            )
            digest_payload["methods"][method] = {  # type: ignore[index]
                "count": count,
                "mean": _mapping_array_digest(means),
                "m2": _mapping_array_digest(m2),
            }
        digest = hashlib.sha256(_canonical_json(digest_payload)).hexdigest()
        return ReducedSufficientStatistics(
            unit_ids=unit_ids,
            methods=MappingProxyType(methods),
            digest=digest,
        )


def _mapping_array_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(values[name], dtype=np.float64)
        digest.update(name.encode("utf-8"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EstimatorDecision:
    """Stage 2 主估计器 decision artifact 的最小跨阶段合同。"""

    decision_id: str
    selected_estimator: str | None
    scope: str
    status: str
    artifact_hash: str
    batch_size: int | None = None
    microbatch_count: int | None = None
    repetitions: int | None = None
    gate_id: str = "stage2.G2.7b"
    state: str = "UNFROZEN"
    gate_status: str | None = None
    artifact_ref: str | None = None
    schema_version: str = "estimator-decision-v1"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.decision_id:
            raise ValueError("decision_id 不能为空")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("decision scope 只能是 local_fixture 或 formal")
        if self.selected_estimator not in {None, "u", "weighted_u", "double"}:
            raise ValueError("selected_estimator 只能是 u、weighted_u、double 或 None")
        if self.schema_version != "estimator-decision-v1":
            raise ValueError("不支持的 EstimatorDecision schema")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("EstimatorDecision status 不能为空")
        if self.state not in {"UNFROZEN", "FROZEN", "SELECTED", "PASS", "READY"}:
            raise ValueError("EstimatorDecision state 不受支持")
        validate_gate_id(self.gate_id, stage=2)
        if self.gate_status is not None:
            GateStatus(self.gate_status)
        for field_name, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("microbatch_count", self.microbatch_count, 2),
            ("repetitions", self.repetitions, 1),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
            ):
                raise ValueError(f"EstimatorDecision {field_name} 必须 >= {minimum}")
        if len(self.artifact_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_hash
        ):
            raise ValueError("artifact_hash 必须是小写 SHA-256")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="EstimatorDecision.metadata"),
        )
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("EstimatorDecision artifact_hash 与完整 wire object 不一致")

    @property
    def formal_eligible(self) -> bool:
        """只有外部 formal/PASS artifact 才能解锁正式在线 importance。"""

        return (
            self.scope == "formal"
            and self.state in {"FROZEN", "SELECTED", "PASS", "READY"}
            and self.status in {"PASS", "SELECTED", "QUALIFIED", "READY"}
            and self.selected_estimator in {"u", "weighted_u", "double"}
            and self.gate_status in {"PASS", "CONDITIONALLY_ACCEPTED"}
            and self.gate_id.startswith("stage2.G")
            and self.batch_size is not None
            and self.microbatch_count is not None
            and self.repetitions is not None
            and self.batch_size % self.microbatch_count == 0
            and self.artifact_ref is not None
        )

    def require_formal(self) -> None:
        if not self.formal_eligible:
            raise FormalDecisionBlocked(
                "正式训练必须读取 scope=formal、status=PASS 且 Gate 已通过的 "
                "Stage 2 decision artifact"
            )

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "selected_estimator": self.selected_estimator,
            "scope": self.scope,
            "status": self.status,
            "state": self.state,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "gate_id": self.gate_id,
            "gate_status": self.gate_status,
            "artifact_ref": self.artifact_ref,
            "metadata": thaw_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "EstimatorDecision":
        """加载外部验收 artifact；不在此处把 fixture 升级为 formal。"""

        validate_estimator_decision_artifact(payload)

        required = {
            "schema_version",
            "decision_id",
            "selected_estimator",
            "scope",
            "status",
            "state",
            "batch_size",
            "microbatch_count",
            "repetitions",
            "gate_id",
            "gate_status",
            "artifact_ref",
            "metadata",
            "artifact_hash",
        }
        if set(payload) != required:
            raise ValueError(
                "EstimatorDecision 字段集合不匹配："
                f"missing={sorted(required-set(payload))}, extra={sorted(set(payload)-required)}"
            )
        if not isinstance(payload["metadata"], Mapping):
            raise ValueError("EstimatorDecision metadata 必须是 object")
        for field_name in (
            "schema_version",
            "decision_id",
            "scope",
            "status",
            "state",
            "gate_id",
            "artifact_hash",
        ):
            if not isinstance(payload[field_name], str):
                raise TypeError(f"EstimatorDecision {field_name} 必须是字符串")
        if payload["selected_estimator"] is not None and not isinstance(
            payload["selected_estimator"], str
        ):
            raise TypeError("EstimatorDecision selected_estimator 必须是字符串或 null")
        for field_name in ("batch_size", "microbatch_count", "repetitions"):
            field_value = payload[field_name]
            if field_value is not None and (
                isinstance(field_value, bool) or not isinstance(field_value, int)
            ):
                raise TypeError(
                    f"EstimatorDecision {field_name} 必须是整数或 null，不能是 bool"
                )
        for field_name in ("gate_status", "artifact_ref"):
            if payload[field_name] is not None and not isinstance(
                payload[field_name], str
            ):
                raise TypeError(f"EstimatorDecision {field_name} 必须是字符串或 null")
        return cls(
            decision_id=payload["decision_id"],
            selected_estimator=payload["selected_estimator"],
            scope=payload["scope"],
            status=payload["status"],
            artifact_hash=payload["artifact_hash"],
            batch_size=payload["batch_size"],
            microbatch_count=payload["microbatch_count"],
            repetitions=payload["repetitions"],
            gate_id=payload["gate_id"],
            state=payload["state"],
            gate_status=payload["gate_status"],
            artifact_ref=payload["artifact_ref"],
            schema_version=payload["schema_version"],
            metadata=payload["metadata"],
        )


def build_fixture_estimator_decision(
    primary_pair: PrimaryPairDecision,
    *,
    selected_estimator: str = "u",
    repetitions: int = 2,
) -> EstimatorDecision:
    """从 synthetic 选择结果生成明确不可 formal 化的 fixture decision。"""

    if primary_pair.status != "FIXTURE_SELECTED":
        raise ValueError("只有已选择 B/M 的 fixture 才能生成 decision")
    if selected_estimator not in {"u", "double"}:
        raise ValueError("fixture estimator 只能是 u 或 double")
    if repetitions <= 0:
        raise ValueError("repetitions 必须严格为正")
    payload = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "",
        "selected_estimator": selected_estimator,
        "scope": "local_fixture",
        "status": "FIXTURE_ONLY",
        "state": "UNFROZEN",
        "batch_size": primary_pair.batch_size,
        "microbatch_count": primary_pair.microbatch_count,
        "repetitions": repetitions,
        "gate_id": "stage2.G2.7b",
        "gate_status": "NOT_RUN",
        "artifact_ref": None,
        "metadata": {
            "formal_eligible": False,
            "source": "synthetic_local_fixture",
        },
    }
    identity_hash = canonical_json_hash(payload)
    decision_id = f"fixture-{identity_hash[:16]}"
    payload["decision_id"] = decision_id
    artifact_hash = canonical_json_hash(payload)
    return EstimatorDecision(
        decision_id=decision_id,
        selected_estimator=selected_estimator,
        scope="local_fixture",
        status="FIXTURE_ONLY",
        artifact_hash=artifact_hash,
        batch_size=primary_pair.batch_size,
        microbatch_count=primary_pair.microbatch_count,
        repetitions=repetitions,
        state="UNFROZEN",
        gate_status="NOT_RUN",
        metadata={"formal_eligible": False, "source": "synthetic_local_fixture"},
    )


class Stage2FixtureStudy:
    """只用于 CPU synthetic 验证的最小 Stage 2 状态机。

    状态按 ``CREATED -> REFERENCE_READY -> MATRIX_FROZEN -> DECIDED ->
    FIXTURE_COMPLETE`` 单向推进；任一前置失败进入 ``BLOCKED``。该对象没有
    formal 转换方法，因而无法把本机 reference/pilot 冒充服务器 Gate 证据。
    """

    def __init__(self, study_id: str) -> None:
        if not study_id:
            raise ValueError("study_id 不能为空")
        self.study_id = study_id
        self.state = "CREATED"
        self.reference: ReferenceResult | None = None
        self.primary_pair: PrimaryPairDecision | None = None
        self.decision: EstimatorDecision | None = None
        self.confirmatory_result_hashes: tuple[str, ...] = ()

    def register_reference(self, reference: ReferenceResult) -> None:
        if self.state != "CREATED":
            raise RuntimeError(f"状态 {self.state} 不能登记 reference")
        if reference.formal_eligible or reference.scope != "local_fixture":
            raise FormalDecisionBlocked("fixture 状态机只接受 local reference")
        self.reference = reference
        self.state = "REFERENCE_READY"

    def freeze_matrix(self, primary_pair: PrimaryPairDecision) -> None:
        if self.state != "REFERENCE_READY":
            raise RuntimeError(f"状态 {self.state} 不能冻结 fixture matrix")
        self.primary_pair = primary_pair
        if primary_pair.status != "FIXTURE_SELECTED":
            self.state = "BLOCKED"
            return
        self.state = "MATRIX_FROZEN"

    def select_estimator(self, *, selected_estimator: str = "u", repetitions: int = 2) -> None:
        if self.state != "MATRIX_FROZEN" or self.primary_pair is None:
            raise RuntimeError(f"状态 {self.state} 不能生成 estimator decision")
        self.decision = build_fixture_estimator_decision(
            self.primary_pair,
            selected_estimator=selected_estimator,
            repetitions=repetitions,
        )
        self.state = "DECIDED"

    def complete(self, results: Sequence[PairedEstimatorResult]) -> None:
        if self.state != "DECIDED" or self.decision is None:
            raise RuntimeError(f"状态 {self.state} 不能完成 fixture study")
        if not results:
            raise ValueError("至少需要一个 paired fixture result")
        if any(result.formal_eligible for result in results):
            raise FormalDecisionBlocked("fixture 状态机拒绝 formal result")
        self.confirmatory_result_hashes = tuple(sorted(result.digest for result in results))
        self.state = "FIXTURE_COMPLETE"

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "study_id": self.study_id,
                    "state": self.state,
                    "reference": None if self.reference is None else self.reference.digest,
                    "primary_pair": None
                    if self.primary_pair is None
                    else {
                        "status": self.primary_pair.status,
                        "B": self.primary_pair.batch_size,
                        "M": self.primary_pair.microbatch_count,
                    },
                    "decision": None
                    if self.decision is None
                    else self.decision.artifact_hash,
                    "confirmatory_result_hashes": list(self.confirmatory_result_hashes),
                }
            )
        ).hexdigest()
