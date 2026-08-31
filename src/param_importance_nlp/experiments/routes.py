"""Stage 4–6 训练路线 DAG、checkpoint lineage 与 estimator 决策防线。

训练执行器不在这里实现；本模块把 pretrain、direct supervised 与 finetune
编译为稳定的 phase DAG。所有模型规模、任务、数据和 checkpoint 频率均由
asset/config 字段驱动，不出现 Pythia、SST-2 等硬编码路径。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..contracts import GateRecord, GateStatus
from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from .sampling import FormalDecisionBlocked
from .stage2 import EstimatorDecision


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _decision_value(decision: object, name: str) -> object:
    if isinstance(decision, Mapping):
        return decision.get(name)
    return getattr(decision, name, None)


def _validate_estimator_decision(
    decision: object,
    *,
    run_intent: str,
    estimator_gate: object | None,
) -> tuple[EstimatorDecision, GateRecord | None]:
    """重算 decision hash，并在 formal 路径绑定独立 GateRecord。"""

    try:
        normalized = (
            decision
            if isinstance(decision, EstimatorDecision)
            else EstimatorDecision.from_mapping(decision)  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalDecisionBlocked("EstimatorDecision wire object 无效或 hash 不一致") from exc
    if run_intent == "formal":
        normalized.require_formal()
        if estimator_gate is None:
            raise FormalDecisionBlocked("formal 路线必须同时提供独立的 Stage 2 GateRecord")
        try:
            gate = (
                estimator_gate
                if isinstance(estimator_gate, GateRecord)
                else GateRecord.from_mapping(dict(estimator_gate))  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise FormalDecisionBlocked("formal EstimatorDecision GateRecord 无效") from exc
        effective = gate.effective_status()
        if gate.gate_id != normalized.gate_id or not gate.gate_id.startswith("stage2.G"):
            raise FormalDecisionBlocked("formal decision 与 Stage 2 GateRecord ID 不匹配")
        if effective not in {GateStatus.PASS, GateStatus.CONDITIONALLY_ACCEPTED}:
            raise FormalDecisionBlocked("formal EstimatorDecision 的独立 Gate 未通过")
        if normalized.gate_status != effective.value:
            raise FormalDecisionBlocked("decision 自报 Gate 状态与 GateRecord 不一致")
    elif run_intent == "local_fixture":
        if normalized.scope != "local_fixture" or normalized.status != "FIXTURE_ONLY":
            raise ValueError("本机路线必须显式绑定 local_fixture EstimatorDecision")
        if estimator_gate is not None:
            raise ValueError("local_fixture 路线不得附加 formal GateRecord")
    else:
        raise ValueError("run_intent 只能是 local_fixture 或 formal")
    return normalized, gate if run_intent == "formal" else None


@dataclass(frozen=True, slots=True)
class TrainingPhaseSpec:
    """训练路线中的一个不可变 phase 节点。

    ``output_checkpoint_id`` 是该 phase 的 lineage 出口。finetune 的
    ``input_checkpoint_id`` 必须等于其 pretrain 父节点的出口，不能通过路径
    猜测 lineage。
    """

    phase_id: str
    phase_type: str
    base_initialization_id: str
    model_asset_id: str
    dataset_asset_id: str
    output_checkpoint_id: str
    checkpoint_frequency_steps: int
    parent_phase_id: str | None = None
    input_checkpoint_id: str | None = None
    task_id: str | None = None
    importance_enabled: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase_type not in {"pretrain", "direct_supervised", "finetune"}:
            raise ValueError(f"未知 phase_type: {self.phase_type!r}")
        for name in (
            "phase_id",
            "base_initialization_id",
            "model_asset_id",
            "dataset_asset_id",
            "output_checkpoint_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} 不能为空")
        if self.checkpoint_frequency_steps <= 0:
            raise ValueError("checkpoint_frequency_steps 必须严格为正")
        if self.phase_type == "finetune":
            if self.parent_phase_id is None or self.input_checkpoint_id is None:
                raise ValueError("finetune 必须声明 parent_phase_id 与 input_checkpoint_id")
            if self.task_id is None:
                raise ValueError("finetune 必须声明 task_id")
        else:
            if self.parent_phase_id is not None or self.input_checkpoint_id is not None:
                raise ValueError("pretrain/direct_supervised 根节点不能伪造 checkpoint 父 lineage")
        if self.phase_type == "direct_supervised" and self.task_id is None:
            raise ValueError("direct_supervised 必须声明 task_id")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="TrainingPhaseSpec.metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "phase_type": self.phase_type,
            "base_initialization_id": self.base_initialization_id,
            "model_asset_id": self.model_asset_id,
            "dataset_asset_id": self.dataset_asset_id,
            "output_checkpoint_id": self.output_checkpoint_id,
            "checkpoint_frequency_steps": self.checkpoint_frequency_steps,
            "parent_phase_id": self.parent_phase_id,
            "input_checkpoint_id": self.input_checkpoint_id,
            "task_id": self.task_id,
            "importance_enabled": self.importance_enabled,
            "metadata": thaw_json_value(self.metadata),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingPhaseSpec":
        """严格加载 phase 节点，禁止未知字段和 ``bool -> int`` 宽松转换。"""

        required = {
            "phase_id",
            "phase_type",
            "base_initialization_id",
            "model_asset_id",
            "dataset_asset_id",
            "output_checkpoint_id",
            "checkpoint_frequency_steps",
            "parent_phase_id",
            "input_checkpoint_id",
            "task_id",
            "importance_enabled",
            "metadata",
        }
        if set(value) != required:
            raise ValueError(
                "TrainingPhaseSpec 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        string_fields = (
            "phase_id",
            "phase_type",
            "base_initialization_id",
            "model_asset_id",
            "dataset_asset_id",
            "output_checkpoint_id",
        )
        if any(not isinstance(value[name], str) for name in string_fields):
            raise TypeError("TrainingPhaseSpec 必填身份字段必须是字符串")
        for name in ("parent_phase_id", "input_checkpoint_id", "task_id"):
            if value[name] is not None and not isinstance(value[name], str):
                raise TypeError(f"TrainingPhaseSpec {name} 必须是字符串或 null")
        frequency = value["checkpoint_frequency_steps"]
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError("checkpoint_frequency_steps 必须是整数且不能是 bool")
        if not isinstance(value["importance_enabled"], bool):
            raise TypeError("importance_enabled 必须是布尔值")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("TrainingPhaseSpec metadata 必须是 object")
        return cls(
            phase_id=value["phase_id"],
            phase_type=value["phase_type"],
            base_initialization_id=value["base_initialization_id"],
            model_asset_id=value["model_asset_id"],
            dataset_asset_id=value["dataset_asset_id"],
            output_checkpoint_id=value["output_checkpoint_id"],
            checkpoint_frequency_steps=frequency,
            parent_phase_id=value["parent_phase_id"],
            input_checkpoint_id=value["input_checkpoint_id"],
            task_id=value["task_id"],
            importance_enabled=value["importance_enabled"],
            metadata=value["metadata"],
        )


@dataclass(frozen=True, slots=True)
class TrainingRouteSpec:
    """经过完整 DAG/lineage 校验的 Stage 4–6 路线。"""

    route_id: str
    phases: tuple[TrainingPhaseSpec, ...]
    run_intent: str
    estimator_decision: object | None
    schema_version: str = "training-route-v1"
    metadata: Mapping[str, object] = field(default_factory=dict)
    estimator_gate: object | None = None

    def __post_init__(self) -> None:
        if not self.route_id or not self.phases:
            raise ValueError("route_id 与 phases 不能为空")
        if self.schema_version != "training-route-v1":
            raise ValueError(f"不支持的 TrainingRouteSpec 版本: {self.schema_version!r}")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise ValueError("run_intent 只能是 local_fixture 或 formal")
        by_id = {phase.phase_id: phase for phase in self.phases}
        if len(by_id) != len(self.phases):
            raise ValueError("phase_id 必须唯一")
        if len({phase.output_checkpoint_id for phase in self.phases}) != len(self.phases):
            raise ValueError("output_checkpoint_id 必须唯一")

        # 先做父引用与 finetune checkpoint lineage 校验，再用拓扑排序识别环。
        for phase in self.phases:
            if phase.parent_phase_id is None:
                continue
            parent = by_id.get(phase.parent_phase_id)
            if parent is None:
                raise ValueError(f"phase {phase.phase_id!r} 引用了不存在的父节点")
            if phase.phase_type != "finetune" or parent.phase_type != "pretrain":
                raise ValueError("finetune 只能直接消费 pretrain 父节点")
            if phase.input_checkpoint_id != parent.output_checkpoint_id:
                raise ValueError("finetune input checkpoint 不属于声明的 pretrain lineage")
            if phase.base_initialization_id != parent.base_initialization_id:
                raise ValueError("finetune 与 pretrain 必须追溯到同一 base initialization")
            if phase.model_asset_id != parent.model_asset_id:
                raise ValueError("finetune 与父 pretrain 必须使用同一模型架构资产")

        order = _topological_ids(self.phases)
        if len(order) != len(self.phases):  # pragma: no cover - 防御性，helper 已抛错
            raise ValueError("phase DAG 不完整")

        roots = [
            phase
            for phase in self.phases
            if phase.phase_type in {"pretrain", "direct_supervised"}
        ]
        if roots and len({phase.base_initialization_id for phase in roots}) != 1:
            raise ValueError("直接监督与预训练路线必须共享同一 base initialization")
        if any(phase.importance_enabled for phase in self.phases):
            if self.estimator_decision is None:
                raise FormalDecisionBlocked("启用在线 importance 时必须绑定 EstimatorDecision")
            normalized_decision, normalized_gate = _validate_estimator_decision(
                self.estimator_decision,
                run_intent=self.run_intent,
                estimator_gate=self.estimator_gate,
            )
            object.__setattr__(self, "estimator_decision", normalized_decision)
            object.__setattr__(self, "estimator_gate", normalized_gate)
        elif self.estimator_decision is not None or self.estimator_gate is not None:
            raise ValueError("全部 phase 关闭 importance 时不得附带未消费的 decision/Gate")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="TrainingRouteSpec.metadata"),
        )

    @property
    def topological_order(self) -> tuple[str, ...]:
        """以 phase_id 为稳定 tie-breaker 返回拓扑序。"""

        return _topological_ids(self.phases)

    @property
    def lineage_hash(self) -> str:
        """返回不含动态训练结果的路线身份摘要。"""

        decision_hash = (
            None
            if self.estimator_decision is None
            else _decision_value(self.estimator_decision, "artifact_hash")
        )
        return _digest(
            {
                "schema_version": self.schema_version,
                "route_id": self.route_id,
                "run_intent": self.run_intent,
                "phases": [
                    next(phase for phase in self.phases if phase.phase_id == phase_id).to_dict()
                    for phase_id in self.topological_order
                ],
                "estimator_decision_hash": decision_hash,
                "estimator_gate_hash": (
                    None
                    if self.estimator_gate is None
                    else _decision_value(self.estimator_gate, "artifact_hash")
                ),
                "metadata": thaw_json_value(self.metadata),
            }
        )

    def phase(self, phase_id: str) -> TrainingPhaseSpec:
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        raise KeyError(phase_id)

    def to_dict(self) -> dict[str, object]:
        """返回按拓扑序排列并绑定 lineage hash 的公共路线 artifact。"""

        decision: object
        if self.estimator_decision is None:
            decision = None
        elif isinstance(self.estimator_decision, EstimatorDecision):
            decision = self.estimator_decision.to_dict()
        else:  # pragma: no cover - __post_init__ 已规范化启用 importance 的对象
            decision = dict(self.estimator_decision)  # type: ignore[arg-type]
        gate: object
        if self.estimator_gate is None:
            gate = None
        elif isinstance(self.estimator_gate, GateRecord):
            gate = self.estimator_gate.to_dict()
        else:
            gate = dict(self.estimator_gate)  # type: ignore[arg-type]
        return {
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "phases": [
                self.phase(phase_id).to_dict() for phase_id in self.topological_order
            ],
            "run_intent": self.run_intent,
            "estimator_decision": decision,
            "estimator_gate": gate,
            "metadata": thaw_json_value(self.metadata),
            "lineage_hash": self.lineage_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingRouteSpec":
        """严格加载路线，并通过构造器重新核对 DAG、decision 与独立 Gate。"""

        required = {
            "schema_version",
            "route_id",
            "phases",
            "run_intent",
            "estimator_decision",
            "estimator_gate",
            "metadata",
            "lineage_hash",
        }
        if set(value) != required:
            raise ValueError(
                "TrainingRouteSpec 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        if not isinstance(value["route_id"], str) or not isinstance(
            value["run_intent"], str
        ):
            raise TypeError("route_id 与 run_intent 必须是字符串")
        if not isinstance(value["phases"], list) or not all(
            isinstance(item, Mapping) for item in value["phases"]
        ):
            raise TypeError("TrainingRouteSpec phases 必须是 object 数组")
        if value["estimator_decision"] is not None and not isinstance(
            value["estimator_decision"], Mapping
        ):
            raise TypeError("estimator_decision 必须是 object 或 null")
        if value["estimator_gate"] is not None and not isinstance(
            value["estimator_gate"], Mapping
        ):
            raise TypeError("estimator_gate 必须是 object 或 null")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("TrainingRouteSpec metadata 必须是 object")
        route = cls(
            route_id=value["route_id"],
            phases=tuple(TrainingPhaseSpec.from_mapping(item) for item in value["phases"]),
            run_intent=value["run_intent"],
            estimator_decision=value["estimator_decision"],
            estimator_gate=value["estimator_gate"],
            schema_version=str(value["schema_version"]),
            metadata=value["metadata"],
        )
        if value["lineage_hash"] != route.lineage_hash:
            raise ValueError("TrainingRouteSpec lineage_hash 与完整路线内容不一致")
        return route


def _topological_ids(phases: Sequence[TrainingPhaseSpec]) -> tuple[str, ...]:
    by_id = {phase.phase_id: phase for phase in phases}
    indegree = {phase.phase_id: 0 for phase in phases}
    children: dict[str, list[str]] = {phase.phase_id: [] for phase in phases}
    for phase in phases:
        if phase.parent_phase_id is not None:
            if phase.parent_phase_id not in by_id:
                raise ValueError(f"不存在的父 phase: {phase.parent_phase_id!r}")
            indegree[phase.phase_id] += 1
            children[phase.parent_phase_id].append(phase.phase_id)
    ready = sorted(phase_id for phase_id, degree in indegree.items() if degree == 0)
    result: list[str] = []
    while ready:
        phase_id = ready.pop(0)
        result.append(phase_id)
        for child in sorted(children[phase_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(result) != len(phases):
        raise ValueError("TrainingRouteSpec phase graph 存在环")
    return tuple(result)


def validate_comparable_routes(routes: Sequence[TrainingRouteSpec]) -> str:
    """验证多条训练范式路线共享同一 base initialization 并返回其 ID。"""

    if len(routes) < 2:
        raise ValueError("路线比较至少需要两条 TrainingRouteSpec")
    initialization_ids = {
        phase.base_initialization_id
        for route in routes
        for phase in route.phases
        if phase.phase_type in {"pretrain", "direct_supervised"}
    }
    if len(initialization_ids) != 1:
        raise ValueError("被比较路线没有共享唯一 base initialization")
    return next(iter(initialization_ids))
