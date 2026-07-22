"""Stage 3 端点、固定 probe、只读路径单元与参考选择编排。

求积节点与逐参数积分公式位于 ``core.quadrature``。本模块只冻结实验身份、
验证 update/probe 不重叠、保护分析状态，并确保本机结果永远不能冒充正式默认
求积方案。optimizer 更新后的 ``parameter_post_state`` 与 scheduler/scaler/RNG
推进完成后的 ``attempt_commit_state`` 是两个独立身份，不能再用一个模糊的
“post checkpoint”代替。
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Hashable, Mapping, Protocol, Sequence

import numpy as np

from ..contracts.immutable import freeze_json_mapping
from .sampling import FormalDecisionBlocked


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} 必须是小写 SHA-256")


class StateMutationError(RuntimeError):
    """路径分析改变了受保护状态时抛出的硬错误。"""


@dataclass(frozen=True, slots=True)
class EndpointState:
    """一个不可变运行状态对象的结构化身份。

    除参数/buffer 外显式绑定 optimizer、scheduler、scaler、RNG、数据游标和
    模型模式，避免两个只换 artifact ID 的对象冒充不同提交阶段。
    """

    artifact_id: str
    artifact_hash: str
    parameter_hash: str
    buffer_hash: str
    optimizer_hash: str
    scheduler_hash: str
    scaler_hash: str
    rng_hash: str
    data_cursor_hash: str
    model_mode_hash: str

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id 不能为空")
        for name in (
            "artifact_hash",
            "parameter_hash",
            "buffer_hash",
            "optimizer_hash",
            "scheduler_hash",
            "scaler_hash",
            "rng_hash",
            "data_cursor_hash",
            "model_mode_hash",
        ):
            _validate_sha256(name, getattr(self, name))

    @property
    def control_state_hash(self) -> str:
        """返回 scheduler/scaler/RNG/游标/模式的联合摘要。"""

        return _sha256(
            {
                "scheduler_hash": self.scheduler_hash,
                "scaler_hash": self.scaler_hash,
                "rng_hash": self.rng_hash,
                "data_cursor_hash": self.data_cursor_hash,
                "model_mode_hash": self.model_mode_hash,
            }
        )

    @property
    def identity_hash(self) -> str:
        """绑定权威 artifact 与全部可恢复组件的端点身份。

        ``artifact_hash`` 是外部两阶段发布对象的身份，不能假设它必然等于本类
        局部字段的摘要；但路径缓存也不能只信任这一项，否则手工组装或损坏的
        manifest 可以在 parameter/buffer/control component 漂移后复用旧节点。
        因此路径单元使用这个联合摘要，同时保存外部对象身份与所有恢复组件。
        """

        return _sha256(
            {
                "artifact_id": self.artifact_id,
                "artifact_hash": self.artifact_hash,
                "parameter_hash": self.parameter_hash,
                "buffer_hash": self.buffer_hash,
                "optimizer_hash": self.optimizer_hash,
                "scheduler_hash": self.scheduler_hash,
                "scaler_hash": self.scaler_hash,
                "rng_hash": self.rng_hash,
                "data_cursor_hash": self.data_cursor_hash,
                "model_mode_hash": self.model_mode_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """一次真实 optimizer transition 的 pre/post/commit 契约。

    ``parameter_post_state`` 必须在 optimizer 更新后、scheduler/scaler/RNG 进入
    下一语义步骤前捕获；``attempt_commit_state`` 则表示整个训练 attempt 已经
    原子提交的恢复点。两者可以共享参数哈希，但 artifact 身份不能相同。
    """

    path_state_id: str
    source_run_id: str
    optimizer_step: int
    parameter_registry_hash: str
    pre_state: EndpointState
    parameter_post_state: EndpointState
    attempt_commit_state: EndpointState
    attempt_commit_parent_hash: str
    probe_buffer_snapshot_hash: str
    full_update_delta_hash: str
    update_sample_ids: tuple[Hashable, ...]
    replay_verified: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.path_state_id or not self.source_run_id:
            raise ValueError("path_state_id 与 source_run_id 不能为空")
        if self.optimizer_step < 0:
            raise ValueError("optimizer_step 不能为负")
        for name in (
            "parameter_registry_hash",
            "probe_buffer_snapshot_hash",
            "full_update_delta_hash",
            "attempt_commit_parent_hash",
        ):
            _validate_sha256(name, getattr(self, name))
        if self.pre_state.buffer_hash != self.parameter_post_state.buffer_hash:
            raise ValueError("参数路径合同要求 optimizer transition 前后 buffer 完全相同")
        if self.pre_state.buffer_hash != self.probe_buffer_snapshot_hash:
            raise ValueError("probe_buffer_snapshot 必须绑定共同 pre/post buffer")
        if self.pre_state.parameter_hash == self.parameter_post_state.parameter_hash:
            # 零位移可以作为解析 fixture，但真实端点默认不接受。调用方必须显式标注。
            if not bool(self.metadata.get("allow_zero_delta_fixture", False)):
                raise ValueError("端点参数哈希相同；非 fixture 路径必须有实际位移")
        if self.parameter_post_state.artifact_id == self.attempt_commit_state.artifact_id:
            raise ValueError(
                "parameter_post_state 与 attempt_commit_state 必须使用不同 artifact 身份"
            )
        if self.attempt_commit_parent_hash != self.parameter_post_state.artifact_hash:
            raise ValueError("attempt_commit_state 未绑定权威 parameter_post_state hash")
        if self.parameter_post_state.parameter_hash != self.attempt_commit_state.parameter_hash:
            raise ValueError("attempt commit 前不得在 optimizer post-state 之后再次改变参数")
        if self.parameter_post_state.buffer_hash != self.attempt_commit_state.buffer_hash:
            raise ValueError("attempt commit 前不得静默改变 probe 相关 buffer")
        if self.parameter_post_state.optimizer_hash != self.attempt_commit_state.optimizer_hash:
            raise ValueError("scheduler/scaler/RNG 提交阶段不得再次改变 optimizer state")
        if (
            self.parameter_post_state.control_state_hash
            == self.attempt_commit_state.control_state_hash
        ):
            raise ValueError("attempt_commit_state 未证明 scheduler/scaler/RNG/游标已推进")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    @property
    def digest(self) -> str:
        return _sha256(
            {
                "path_state_id": self.path_state_id,
                "source_run_id": self.source_run_id,
                "optimizer_step": self.optimizer_step,
                "parameter_registry_hash": self.parameter_registry_hash,
                "pre_state": self.pre_state.identity_hash,
                "parameter_post_state": self.parameter_post_state.identity_hash,
                "attempt_commit_state": self.attempt_commit_state.identity_hash,
                "attempt_commit_parent_hash": self.attempt_commit_parent_hash,
                "probe_buffer_snapshot_hash": self.probe_buffer_snapshot_hash,
                "full_update_delta_hash": self.full_update_delta_hash,
                "update_sample_ids": list(self.update_sample_ids),
                "replay_verified": self.replay_verified,
            }
        )


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """所有路径节点共享的确定性有限 probe panel。"""

    probe_id: str
    sample_ids: tuple[Hashable, ...]
    content_hash: str
    loss_contract_hash: str
    effective_weight_unit: str = "target_token"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probe_id or not self.sample_ids:
            raise ValueError("probe_id 与 sample_ids 不能为空")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("一个固定 probe panel 内不允许重复统计单元")
        _validate_sha256("content_hash", self.content_hash)
        _validate_sha256("loss_contract_hash", self.loss_contract_hash)
        if not self.effective_weight_unit:
            raise ValueError("effective_weight_unit 不能为空")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def assert_independent_from(self, endpoint: EndpointRecord) -> None:
        overlap = set(self.sample_ids).intersection(endpoint.update_sample_ids)
        if overlap:
            raise ValueError(
                f"probe 与 update batch 发生统计单元重叠，共 {len(overlap)} 个 sample IDs"
            )

    @property
    def digest(self) -> str:
        return _sha256(
            {
                "probe_id": self.probe_id,
                "sample_ids": list(self.sample_ids),
                "content_hash": self.content_hash,
                "loss_contract_hash": self.loss_contract_hash,
                "effective_weight_unit": self.effective_weight_unit,
            }
        )


@dataclass(frozen=True, slots=True)
class PathStateUnit:
    """Stage 3 最小可恢复原子单元身份。"""

    endpoint: EndpointRecord
    probe: ProbeSpec
    path_type: str = "full_update_path"
    precision: str = "float64"
    run_intent: str = "local_fixture"

    def __post_init__(self) -> None:
        if self.path_type not in {"full_update_path", "data_update_path"}:
            raise ValueError("未知 path_type")
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision 只能是 float32 或 float64")
        if self.run_intent != "local_fixture":
            raise FormalDecisionBlocked("本机 Stage 3 编排只允许 local_fixture")
        if not self.endpoint.replay_verified:
            raise ValueError("端点重放未验证，不能进入路径分析")
        self.probe.assert_independent_from(self.endpoint)

    @property
    def unit_id(self) -> str:
        digest = _sha256(
            {
                "endpoint": self.endpoint.digest,
                "probe": self.probe.digest,
                "path_type": self.path_type,
                "precision": self.precision,
                "run_intent": self.run_intent,
            }
        )
        return f"path-unit-{digest[:20]}"


class StateController(Protocol):
    """只读分析上下文需要的状态快照/恢复接口。"""

    def digest(self) -> str:
        """返回参数、buffer、模式、RNG 和游标的联合摘要。"""

    def restore(self) -> None:
        """恢复进入分析上下文时保存的全部状态。"""


class ReadOnlyPathContext(AbstractContextManager["ReadOnlyPathContext"]):
    """分析结束后验证并恢复状态的上下文管理器。

    只要检测到摘要变化，即使恢复成功也会抛出 ``StateMutationError``，因为该
    单元已经违反无副作用合同，不能进入聚合。
    """

    def __init__(self, controller: StateController) -> None:
        self.controller = controller
        self._before: str | None = None

    def __enter__(self) -> "ReadOnlyPathContext":
        self._before = self.controller.digest()
        if not self._before:
            raise ValueError("state digest 不能为空")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        assert self._before is not None
        after = self.controller.digest()
        if after != self._before:
            self.controller.restore()
            restored = self.controller.digest()
            if restored != self._before:
                raise StateMutationError("路径分析污染状态且恢复失败") from exc
            raise StateMutationError("路径分析曾污染状态；已恢复但该单元必须失败") from exc
        return False


@dataclass(frozen=True, slots=True)
class NodeCacheKey:
    """公共节点缓存的完整身份；任一字段变化都会产生新键。"""

    path_unit_id: str
    alpha: float
    precision: str
    parameter_registry_hash: str
    loss_contract_hash: str

    def __post_init__(self) -> None:
        if not self.path_unit_id:
            raise ValueError("path_unit_id 不能为空")
        if not 0.0 <= self.alpha <= 1.0 or not math.isfinite(self.alpha):
            raise ValueError("alpha 必须是 [0,1] 内有限数")
        # IEEE-754 中 ``-0.0`` 与 ``0.0`` 数值相等、但 ``float.hex`` 不同。
        # 路径左端点只有一个数学身份，因此在生成 digest 前先规范化为正零。
        if self.alpha == 0.0:
            object.__setattr__(self, "alpha", 0.0)
        if self.precision not in {"float32", "float64"}:
            raise ValueError("precision 只能是 float32 或 float64")
        _validate_sha256("parameter_registry_hash", self.parameter_registry_hash)
        _validate_sha256("loss_contract_hash", self.loss_contract_hash)

    @property
    def digest(self) -> str:
        # float.hex 避免十进制 JSON 格式差异改变节点身份。
        return _sha256(
            {
                "path_unit_id": self.path_unit_id,
                "alpha_hex": self.alpha.hex(),
                "precision": self.precision,
                "parameter_registry_hash": self.parameter_registry_hash,
                "loss_contract_hash": self.loss_contract_hash,
            }
        )


def _clone_cached_gradient(value: object) -> object:
    """复制一个梯度缓存值，隔离缓存所有权与调用方所有权。

    Stage 3 的正式梯度值通常是 :class:`core.TensorMap`。它内部虽然使用只读
    mapping，但其中的 ``torch.Tensor`` 仍可原地修改，因此不能只复制容器。
    这里延迟导入 Torch 相关类型，保证仅阅读契约时不会强制加载数值运行时；
    对扩展 adapter 返回的普通状态树则递归复制，未知对象最后使用
    :func:`copy.deepcopy`。无法安全复制的对象会 fail-closed，而不是把同一可变
    引用交给缓存和调用方。
    """

    try:
        from param_importance_nlp.core.tensors import TensorMap
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - 极简契约环境
        TensorMap = None  # type: ignore[assignment,misc]
    if TensorMap is not None and isinstance(value, TensorMap):
        return value.clone()

    # 不在模块导入阶段依赖 torch；实际缓存裸 Tensor 时才加载它。
    try:
        import torch
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - 极简契约环境
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(value.requires_grad)
        return cloned

    if isinstance(value, dict):
        return {key: _clone_cached_gradient(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cached_gradient(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cached_gradient(item) for item in value)
    if isinstance(value, MappingProxyType):
        return MappingProxyType(
            {key: _clone_cached_gradient(item) for key, item in value.items()}
        )
    try:
        return copy.deepcopy(value)
    except Exception as exc:  # pragma: no cover - adapter 扩展对象防线
        raise TypeError(
            f"节点梯度类型 {type(value).__qualname__!r} 无法安全复制，不能写入缓存"
        ) from exc


class NodeGradientCache:
    """进程内、不可覆盖且防御性复制的 Stage 3 节点梯度缓存。

    缓存键同时绑定路径原子单元、精确 ``alpha``、累计精度、参数注册表和 loss
    合同。缓存只保存节点梯度，不保存插值状态或模型对象。``get`` 与 ``put`` 两端
    都执行深复制，因此求积器、测试代码或外部 adapter 对返回值的原地修改不会
    反向污染后续规则。

    ``PathAnalysisRunner`` 会先把本次新节点放入私有暂存区，直到求积、端点 loss
    检查和 ``ReadOnlyPathContext`` 全部成功后才调用 :meth:`publish_many`。这使缓存
    发布与一次路径分析的成功边界一致；异常或状态恢复不会遗留半成品节点。

    该实现用可重入锁保护单进程并发访问，并按 digest 排序导出键，便于确定性
    审计。它有意不提供覆盖接口：同一键发布后即视为不可变事实。
    """

    def __init__(self) -> None:
        self._values: dict[NodeCacheKey, object] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._values

    def get(self, key: NodeCacheKey) -> object:
        """返回缓存值的独立副本；键不存在时抛出 ``KeyError``。"""

        if not isinstance(key, NodeCacheKey):
            raise TypeError("节点缓存只接受 NodeCacheKey")
        with self._lock:
            value = self._values[key]
            return _clone_cached_gradient(value)

    def publish_many(self, entries: Mapping[NodeCacheKey, object]) -> int:
        """原子发布一批私有暂存节点，并返回实际新增节点数。

        发布前先复制全部值；只要其中一个值不可安全复制，就不会写入任何节点。
        已存在键保持首次发布值，避免并发运行以完成先后顺序覆盖缓存事实。
        """

        prepared: list[tuple[NodeCacheKey, object]] = []
        for key, value in entries.items():
            if not isinstance(key, NodeCacheKey):
                raise TypeError("节点缓存只接受 NodeCacheKey")
            prepared.append((key, _clone_cached_gradient(value)))
        prepared.sort(key=lambda item: item[0].digest)
        added = 0
        with self._lock:
            for key, value in prepared:
                if key not in self._values:
                    self._values[key] = value
                    added += 1
        return added

    def keys(self) -> tuple[NodeCacheKey, ...]:
        """按 key digest 返回确定性快照，不暴露内部字典。"""

        with self._lock:
            return tuple(sorted(self._values, key=lambda key: key.digest))

    def clear(self) -> None:
        """清空本进程缓存；调用方应只在开始新的独立 session 时使用。"""

        with self._lock:
            self._values.clear()


class PathIntegrator(Protocol):
    """core 求积引擎的实验层适配协议。"""

    def integrate(
        self,
        *,
        path_spec: object,
        rule: object,
        gradient_callback: Callable[[float, object], object],
        loss_callback: Callable[[object], object],
    ) -> object:
        """返回 core 的 PathIntegralResult。"""


class CorePathIntegrator:
    """延迟导入 ``core.quadrature.integrate_path`` 的默认适配器。"""

    def integrate(
        self,
        *,
        path_spec: object,
        rule: object,
        gradient_callback: Callable[[float, object], object],
        loss_callback: Callable[[object], object],
    ) -> object:
        try:
            from param_importance_nlp.core.quadrature import integrate_path
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - 环境防线
            raise RuntimeError("执行路径积分需要 core.quadrature") from exc
        return integrate_path(path_spec, rule, gradient_callback, loss_fn=loss_callback)


@dataclass(frozen=True, slots=True)
class PathEvaluation:
    """一个本机路径单元与规则的结果包装。"""

    unit_id: str
    rule_name: str
    result: object
    cache_hits: int = 0
    cache_misses: int = 0
    cache_entries_published: int = 0
    scope: str = "local_fixture"
    formal_eligible: bool = False


class PathAnalysisRunner:
    """在只读状态保护下调用核心求积引擎并复用公共节点梯度。

    一个 runner 默认持有一个 session 级 :class:`NodeGradientCache`。连续用
    trapezoid、Simpson 或不同加密级别分析同一 ``PathStateUnit`` 时，完全相同的
    节点只调用一次昂贵的梯度 provider；不同路径、精度、registry 或 loss 合同
    生成不同 key，不会发生跨实验串用。
    """

    def __init__(
        self,
        integrator: PathIntegrator | None = None,
        *,
        node_cache: NodeGradientCache | None = None,
    ) -> None:
        self.integrator = integrator or CorePathIntegrator()
        # 不能写成 ``node_cache or ...``：空缓存的 ``len`` 为零，但它仍是调用方
        # 希望跨多个 runner 共享的有效对象。
        self.node_cache = node_cache if node_cache is not None else NodeGradientCache()

    @staticmethod
    def _node_key(unit: PathStateUnit, alpha: float) -> NodeCacheKey:
        """由冻结路径单元构造完整节点身份。"""

        return NodeCacheKey(
            path_unit_id=unit.unit_id,
            alpha=alpha,
            precision=unit.precision,
            parameter_registry_hash=unit.endpoint.parameter_registry_hash,
            loss_contract_hash=unit.probe.loss_contract_hash,
        )

    def run(
        self,
        *,
        unit: PathStateUnit,
        path_spec: object,
        rule: object,
        gradient_callback: Callable[[float, object], object],
        loss_callback: Callable[[object], object],
        state_controller: StateController,
        expected_loss_pre: float | None = None,
        expected_loss_post: float | None = None,
    ) -> PathEvaluation:
        """执行一个本机 ``PathStateUnit``。

        该兼容入口保留 Stage 3 fixture 的既有合同；实际缓存事务统一委托给
        :meth:`run_bound`，从而让 formal runner 可以使用已经由 endpoint/probe
        artifact 冻结的等价身份，而不必伪造一个只允许 ``local_fixture`` 的
        :class:`PathStateUnit`。
        """

        return self.run_bound(
            unit_id=unit.unit_id,
            precision=unit.precision,
            parameter_registry_hash=unit.endpoint.parameter_registry_hash,
            loss_contract_hash=unit.probe.loss_contract_hash,
            path_spec=path_spec,
            rule=rule,
            gradient_callback=gradient_callback,
            loss_callback=loss_callback,
            state_controller=state_controller,
            expected_loss_pre=expected_loss_pre,
            expected_loss_post=expected_loss_post,
            scope="local_fixture",
            formal_eligible=False,
        )

    def run_bound(
        self,
        *,
        unit_id: str,
        precision: str,
        parameter_registry_hash: str,
        loss_contract_hash: str,
        path_spec: object,
        rule: object,
        gradient_callback: Callable[[float, object], object],
        loss_callback: Callable[[object], object],
        state_controller: StateController,
        expected_loss_pre: float | None = None,
        expected_loss_post: float | None = None,
        scope: str = "local_fixture",
        formal_eligible: bool = False,
    ) -> PathEvaluation:
        """按已冻结的路径身份执行同一套缓存与只读状态事务。

        ``unit_id``、精度、registry hash 与 loss hash 必须来自已经验证过的
        endpoint/probe/path artifact。该入口不会替 formal 运行做资格判断；它只把
        调用方给出的身份装配成 :class:`NodeCacheKey`。节点梯度先进入本次私有
        ``pending``，只有求积、端点 loss 校验和状态不变性检查全部成功后，才通过
        cache 的 ``publish_many`` 一次发布。因此进程内缓存和持久化两阶段缓存共享
        完全相同的失败边界。
        """

        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("unit_id 不能为空")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision 只能是 float32 或 float64")
        for field_name, digest in (
            ("parameter_registry_hash", parameter_registry_hash),
            ("loss_contract_hash", loss_contract_hash),
        ):
            _validate_sha256(field_name, digest)
        if scope not in {"local_fixture", "formal"}:
            raise ValueError("scope 只能是 local_fixture 或 formal")
        if formal_eligible and scope != "formal":
            raise ValueError("只有 formal scope 可以声明 formal_eligible")
        for name, value in (
            ("expected_loss_pre", expected_loss_pre),
            ("expected_loss_post", expected_loss_post),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} 必须有限")
        rule_name = str(getattr(rule, "name", "unknown-rule"))
        pending: dict[NodeCacheKey, object] = {}
        cache_hits = 0
        cache_misses = 0

        def cached_gradient(alpha: float, state: object) -> object:
            """先查已发布/本次暂存节点，缺失时才调用真实梯度 provider。"""

            nonlocal cache_hits, cache_misses
            key = NodeCacheKey(
                path_unit_id=unit_id,
                alpha=alpha,
                precision=precision,
                parameter_registry_hash=parameter_registry_hash,
                loss_contract_hash=loss_contract_hash,
            )
            try:
                cached = self.node_cache.get(key)
            except KeyError:
                if key in pending:
                    cache_hits += 1
                    return _clone_cached_gradient(pending[key])
                computed = gradient_callback(alpha, state)
                # 暂存区和本次求积器也不能共享同一个可变对象：求积器拿到的是
                # 副本，只有完整运行成功后暂存值才有资格原子发布。
                pending[key] = _clone_cached_gradient(computed)
                cache_misses += 1
                return _clone_cached_gradient(computed)
            else:
                cache_hits += 1
                return cached

        with ReadOnlyPathContext(state_controller):
            result = self.integrator.integrate(
                path_spec=path_spec,
                rule=rule,
                gradient_callback=cached_gradient,
                loss_callback=loss_callback,
            )
        actual_pre = getattr(result, "endpoint_loss_pre", None)
        actual_post = getattr(result, "endpoint_loss_post", None)
        if expected_loss_pre is not None and not math.isclose(
            float(actual_pre), expected_loss_pre, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("核心求积结果的 pre probe loss 与冻结端点不一致")
        if expected_loss_post is not None and not math.isclose(
            float(actual_post), expected_loss_post, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("核心求积结果的 post probe loss 与冻结端点不一致")
        published = self.node_cache.publish_many(pending)
        return PathEvaluation(
            unit_id=unit_id,
            rule_name=rule_name,
            result=result,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_entries_published=published,
            scope=scope,
            formal_eligible=formal_eligible,
        )


@dataclass(frozen=True, slots=True)
class ReferenceLevel:
    """一条参考家族的一个连续加密级别。"""

    family: str
    level: int
    unique_nodes: int
    contribution: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.family not in {"gauss_legendre", "composite_simpson"}:
            raise ValueError("参考 family 必须是 Gauss-Legendre 或 composite Simpson")
        if self.level < 0 or self.unique_nodes <= 0:
            raise ValueError("level 必须非负且 unique_nodes 必须严格为正")


def _flatten_vector(value: Mapping[str, object]) -> np.ndarray:
    if not value:
        raise ValueError("贡献向量不能为空")
    parts: list[np.ndarray] = []
    for name in sorted(value):
        part = value[name]
        if hasattr(part, "detach"):
            part = part.detach()  # type: ignore[union-attr]
        if hasattr(part, "cpu"):
            part = part.cpu()  # type: ignore[union-attr]
        if hasattr(part, "numpy"):
            part = part.numpy()  # type: ignore[union-attr]
        array = np.asarray(part, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError("参考贡献包含 NaN/Inf")
        parts.append(array)
    return np.concatenate(parts)


def _normalized_l1(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    if set(left) != set(right):
        raise ValueError("参考向量参数名集合不一致")
    for name in left:
        left_shape = np.asarray(left[name]).shape if not hasattr(left[name], "shape") else left[name].shape
        right_shape = (
            np.asarray(right[name]).shape
            if not hasattr(right[name], "shape")
            else right[name].shape
        )
        if tuple(left_shape) != tuple(right_shape):
            raise ValueError(f"参考向量参数 {name!r} 的 shape 不一致")
    lhs = _flatten_vector(left)
    rhs = _flatten_vector(right)
    if lhs.shape != rhs.shape:
        raise ValueError("参考向量 shape 不一致")
    denominator = float(np.sum(np.abs(rhs)))
    if denominator == 0:
        return 0.0 if np.array_equal(lhs, rhs) else math.inf
    return float(np.sum(np.abs(lhs - rhs)) / denominator)


@dataclass(frozen=True, slots=True)
class ReferenceConvergenceResult:
    """两参考家族内部与跨家族的保守误差代理。"""

    converged: bool
    gauss_within_error: float
    simpson_within_error: float
    cross_family_error: float
    conservative_error: float
    tolerance: float
    scope: str = "local_fixture"
    formal_eligible: bool = False


def assess_reference_convergence(
    levels: Sequence[ReferenceLevel],
    *,
    tolerance: float,
    scope: str = "local_fixture",
) -> ReferenceConvergenceResult:
    """要求两个不同家族各至少两级，并取三种误差代理最大值。"""

    if scope != "local_fixture":
        raise FormalDecisionBlocked("本机参考收敛只能形成 fixture 证据")
    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance 必须是有限正数")
    grouped: dict[str, list[ReferenceLevel]] = {
        "gauss_legendre": [],
        "composite_simpson": [],
    }
    for level in levels:
        grouped[level.family].append(level)
    for family_levels in grouped.values():
        family_levels.sort(key=lambda item: item.level)
        if len(family_levels) < 2:
            raise ValueError("每个参考家族至少需要两个连续加密级别")
        if any(
            right.unique_nodes <= left.unique_nodes
            for left, right in zip(family_levels, family_levels[1:])
        ):
            raise ValueError("参考加密级别的 unique_nodes 必须严格递增")
    gauss = grouped["gauss_legendre"]
    simpson = grouped["composite_simpson"]
    gauss_error = _normalized_l1(gauss[-2].contribution, gauss[-1].contribution)
    simpson_error = _normalized_l1(simpson[-2].contribution, simpson[-1].contribution)
    cross_error = _normalized_l1(gauss[-1].contribution, simpson[-1].contribution)
    conservative = max(gauss_error, simpson_error, cross_error)
    return ReferenceConvergenceResult(
        converged=math.isfinite(conservative) and conservative <= tolerance,
        gauss_within_error=gauss_error,
        simpson_within_error=simpson_error,
        cross_family_error=cross_error,
        conservative_error=conservative,
        tolerance=tolerance,
    )


@dataclass(frozen=True, slots=True)
class QuadratureDecision:
    """Stage 3 求积方案 decision；本机只能形成 fixture 推荐。"""

    decision_id: str
    status: str
    default_rule: str | None
    fallback_rule: str | None
    scope: str
    artifact_hash: str
    formal_eligible: bool
    unresolved_fields: tuple[str, ...] = (
        "formal_thresholds",
        "formal_probe_count",
        "formal_node_budget",
    )
    schema_version: str = "quadrature-decision-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "quadrature-decision-v1":
            raise ValueError("不支持的 QuadratureDecision schema")
        if not self.decision_id:
            raise ValueError("QuadratureDecision decision_id 不能为空")
        if self.status not in {"FIXTURE_RECOMMENDATION", "BLOCKED"}:
            raise ValueError("QuadratureDecision status 不受支持")
        if self.status == "FIXTURE_RECOMMENDATION" and not self.default_rule:
            raise ValueError("FIXTURE_RECOMMENDATION 必须指定 default_rule")
        if self.status == "BLOCKED" and self.default_rule is not None:
            raise ValueError("BLOCKED decision 不得指定 default_rule")
        if self.fallback_rule is not None and not self.fallback_rule:
            raise ValueError("fallback_rule 必须是非空字符串或 null")
        if not isinstance(self.formal_eligible, bool):
            raise TypeError("QuadratureDecision formal_eligible 必须是布尔值")
        if (
            not self.unresolved_fields
            or not all(isinstance(item, str) and item for item in self.unresolved_fields)
            or len(set(self.unresolved_fields)) != len(self.unresolved_fields)
        ):
            raise ValueError("unresolved_fields 必须是非空、无重复字段数组")
        _validate_sha256("artifact_hash", self.artifact_hash)
        if self.scope != "local_fixture" or self.formal_eligible:
            raise FormalDecisionBlocked("本机 QuadratureDecision 不能标记为 formal")
        if _sha256(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("QuadratureDecision artifact_hash 与完整 wire object 不一致")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "status": self.status,
            "default_rule": self.default_rule,
            "fallback_rule": self.fallback_rule,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "unresolved_fields": list(self.unresolved_fields),
        }

    def to_dict(self) -> dict[str, object]:
        """返回严格、hash 绑定且明确非正式的 fixture decision。"""

        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "QuadratureDecision":
        """严格加载 decision；不会把本机推荐提升为正式默认方法。"""

        required = {
            "schema_version",
            "decision_id",
            "status",
            "default_rule",
            "fallback_rule",
            "scope",
            "formal_eligible",
            "unresolved_fields",
            "artifact_hash",
        }
        if set(value) != required:
            raise ValueError(
                "QuadratureDecision 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        unresolved = value["unresolved_fields"]
        if not isinstance(unresolved, list) or not all(
            isinstance(item, str) and item for item in unresolved
        ):
            raise TypeError("QuadratureDecision unresolved_fields 必须是非空字符串数组")
        for field_name in ("decision_id", "status", "scope", "artifact_hash"):
            if not isinstance(value[field_name], str):
                raise TypeError(f"QuadratureDecision {field_name} 必须是字符串")
        for field_name in ("default_rule", "fallback_rule"):
            if value[field_name] is not None and not isinstance(value[field_name], str):
                raise TypeError(f"QuadratureDecision {field_name} 必须是字符串或 null")
        if not isinstance(value["formal_eligible"], bool):
            raise TypeError("QuadratureDecision formal_eligible 必须是布尔值")
        return cls(
            decision_id=value["decision_id"],
            status=value["status"],
            default_rule=value["default_rule"],
            fallback_rule=value["fallback_rule"],
            scope=value["scope"],
            artifact_hash=value["artifact_hash"],
            formal_eligible=value["formal_eligible"],
            unresolved_fields=tuple(unresolved),
            schema_version=str(value["schema_version"]),
        )


def build_fixture_quadrature_decision(
    *,
    passing_rules_by_cost: Sequence[str],
    fallback_rule: str | None = None,
) -> QuadratureDecision:
    """从解析 fixture 的通过规则生成非正式推荐；空集合保持 BLOCKED。"""

    default = passing_rules_by_cost[0] if passing_rules_by_cost else None
    status = "FIXTURE_RECOMMENDATION" if default is not None else "BLOCKED"
    if fallback_rule is None and len(passing_rules_by_cost) > 1:
        fallback_rule = passing_rules_by_cost[1]
    identity_payload = {
        "status": status,
        "default_rule": default,
        "fallback_rule": fallback_rule,
        "scope": "local_fixture",
        "formal_eligible": False,
        "unresolved_fields": [
            "formal_thresholds",
            "formal_probe_count",
            "formal_node_budget",
        ],
    }
    identity_digest = _sha256(identity_payload)
    decision_id = f"fixture-quadrature-{identity_digest[:16]}"
    payload = {
        "schema_version": "quadrature-decision-v1",
        "decision_id": decision_id,
        **identity_payload,
    }
    digest = _sha256(payload)
    return QuadratureDecision(
        decision_id=decision_id,
        status=status,
        default_rule=default,
        fallback_rule=fallback_rule,
        scope="local_fixture",
        artifact_hash=digest,
        formal_eligible=False,
    )
