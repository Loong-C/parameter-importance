"""Stage 0--9 可执行任务目录。

计划文档描述的是研究与工程要求，而运行时需要稳定、不可含糊的机器标识。本模块
把 Stage 0--3 的每一份编号计划文件，以及 Stage 4--9 的顶层实验任务，冻结为
``TaskDefinition``。每个定义都明确回答五个问题：

* 应由哪一类 runner 执行；
* 必须从 resolved config 读取哪些区域；
* 成功时必须发布哪些 artifact；
* 中断后从哪个安全边界恢复；
* 正式运行还缺哪些合同、Gate、能力或 estimator decision。

目录只描述合同，不导入训练、服务器或分析实现。这样本机可以完整审计任务覆盖率，
而缺少 GPU、真实资产或 runner 时，运行时会生成结构化 ``BLOCKED``，不会把接口
存在误写成任务已经完成。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Final

from .jsonio import JSONValue, canonical_json_hash


TASK_CATALOG_SCHEMA_VERSION: Final = "task-catalog-v2"
TASK_DEFINITION_SCHEMA_VERSION: Final = "task-definition-v2"

_TASK_ID_RE = re.compile(r"^stage(?P<stage>[0-9])\.[a-z0-9][a-z0-9_.-]*$")
_GATE_ID_RE = re.compile(r"^stage[0-9]\.G[0-9]+(?:[.-][A-Za-z0-9]+)*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CONFIG_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_SCHEMA_REF_RE = re.compile(
    r"^https://parameter-importance\.invalid/schemas/[a-z0-9_./-]+\.json$"
)
_RULE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# 任务目录只记录项目内 canonical schema 的绝对 ``$id``。这里不接受任意 URL，
# 因为任意 URL 会把离线执行变成隐式网络依赖，也无法保证 catalog hash 在不同机器
# 上有相同含义。文件是否真实存在由仓库级契约测试逐条核对。
RESOLVED_CONFIG_V2_SCHEMA_REF: Final = (
    "https://parameter-importance.invalid/schemas/shared/resolved-config-v2.json"
)
TASK_OUTPUT_ARTIFACT_SCHEMA_REF: Final = (
    "https://parameter-importance.invalid/schemas/shared/task-output-artifact-v1.json"
)
TASK_OUTPUT_PAYLOAD_SCHEMA_REF: Final = (
    "https://parameter-importance.invalid/schemas/shared/task-output-payload-v1.json"
)
TASK_OUTPUT_COMMIT_SCHEMA_REF: Final = (
    "https://parameter-importance.invalid/schemas/shared/task-output-commit-v1.json"
)


class TaskCatalogError(ValueError):
    """任务目录字段、覆盖范围或摘要无效。"""


class RunnerKind(str, Enum):
    """稳定 runner 类别；类别描述能力边界，不等同于某个 Python 类名。"""

    AUDIT = "audit"
    STORAGE = "storage"
    ENVIRONMENT = "environment"
    ASSET = "asset"
    CONTRACT = "contract"
    TRAINING = "training"
    DISTRIBUTED_TRAINING = "distributed_training"
    OBSERVABILITY = "observability"
    CHECKPOINT = "checkpoint"
    CAPACITY = "capacity"
    TEST_MATRIX = "test_matrix"
    DELIVERY = "delivery"
    REGISTRY = "registry"
    ORACLE = "oracle"
    VALIDATION = "validation"
    ESTIMATOR = "estimator"
    REFERENCE = "reference"
    ESTIMATOR_EXPERIMENT = "estimator_experiment"
    PILOT = "pilot"
    STATISTICS = "statistics"
    REPORTING = "reporting"
    PATH_INTEGRATION = "path_integration"
    ROUTE_TRAINING = "route_training"
    PRUNING = "pruning"
    ABLATION = "ablation"
    ANALYSIS = "analysis"


class RecoveryMode(str, Enum):
    """任务级恢复策略。

    ``resume_*`` 只允许消费已提交的不可变边界；临时目录或半写 shard 永远不能作为
    恢复点。``manual_external`` 用于 Git/多端同步等需要外部授权的动作。
    """

    RESTART_IDEMPOTENT = "restart_idempotent"
    RESUME_CHECKPOINT = "resume_checkpoint"
    RESUME_SHARDS = "resume_shards"
    RECONCILE_STATE = "reconcile_state"
    REBUILD_DERIVED = "rebuild_derived"
    MANUAL_EXTERNAL = "manual_external"


class SafeBoundary(str, Enum):
    """恢复时可被权威读取的最小提交边界。"""

    NONE = "none"
    IMMUTABLE_PUBLISH = "immutable_publish"
    ATTEMPT_COMMIT_STATE = "attempt_commit_state"
    SHARD_COMMIT = "shard_commit"
    CHECKPOINT_COMMIT = "checkpoint_commit"
    CANONICAL_SOURCE = "canonical_source"


class ReplayStrategy(str, Enum):
    """任务重放时允许采用的唯一状态来源。"""

    RESTART_FROM_CONFIG = "restart_from_config"
    RESUME_FROM_COMMIT = "resume_from_commit"
    RECONCILE_COMMITTED_STATE = "reconcile_committed_state"
    REBUILD_FROM_FROZEN_SOURCES = "rebuild_from_frozen_sources"
    MANUAL_EXTERNAL_REVIEW = "manual_external_review"


class ExistingOutputPolicy(str, Enum):
    """重放遇到同一逻辑输出时的 fail-closed 行为。"""

    REUSE_IDENTICAL_REJECT_DRIFT = "reuse_identical_reject_drift"


@dataclass(frozen=True, slots=True)
class InputArtifactContract:
    """一个下游任务必须消费的上游 artifact 合同。

    ``producer_task_ids`` 记录逻辑生产者；实际运行仍必须从生产者的权威 commit
    发现对象，并按 ``schema_ref`` 校验 envelope。目录不会把普通路径、临时文件或
    未提交对象误当输入。
    """

    input_id: str
    schema_ref: str
    required: bool
    producer_task_ids: tuple[str, ...]
    artifact_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_string(self.input_id, field="input_artifacts.input_id", pattern=_NAME_RE)
        _require_string(self.schema_ref, field="input_artifacts.schema_ref", pattern=_SCHEMA_REF_RE)
        if type(self.required) is not bool:
            raise TaskCatalogError("input_artifacts.required 必须是 bool")
        producers = _string_tuple(
            self.producer_task_ids,
            field="input_artifacts.producer_task_ids",
            pattern=_TASK_ID_RE,
        )
        artifact_kinds = _string_tuple(
            self.artifact_kinds,
            field="input_artifacts.artifact_kinds",
            pattern=_NAME_RE,
        )
        object.__setattr__(self, "producer_task_ids", producers)
        object.__setattr__(self, "artifact_kinds", artifact_kinds)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "input_id": self.input_id,
            "schema_ref": self.schema_ref,
            "required": self.required,
            "producer_task_ids": list(self.producer_task_ids),
            "artifact_kinds": list(self.artifact_kinds),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "InputArtifactContract":
        if set(value) != {
            "input_id", "schema_ref", "required", "producer_task_ids", "artifact_kinds"
        }:
            raise TaskCatalogError("input_artifacts 字段集合无效")
        return cls(
            input_id=value["input_id"],  # type: ignore[arg-type]
            schema_ref=value["schema_ref"],  # type: ignore[arg-type]
            required=value["required"],  # type: ignore[arg-type]
            producer_task_ids=tuple(value["producer_task_ids"]),  # type: ignore[arg-type]
            artifact_kinds=tuple(value["artifact_kinds"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class OutputArtifactContract:
    """一个 required output 的 canonical envelope 与业务 payload schema 绑定。"""

    artifact_kind: str
    schema_ref: str
    payload_schema_ref: str

    def __post_init__(self) -> None:
        _require_string(self.artifact_kind, field="output_artifacts.artifact_kind", pattern=_NAME_RE)
        _require_string(self.schema_ref, field="output_artifacts.schema_ref", pattern=_SCHEMA_REF_RE)
        _require_string(
            self.payload_schema_ref,
            field="output_artifacts.payload_schema_ref",
            pattern=_SCHEMA_REF_RE,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "artifact_kind": self.artifact_kind,
            "schema_ref": self.schema_ref,
            "payload_schema_ref": self.payload_schema_ref,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OutputArtifactContract":
        if set(value) != {"artifact_kind", "schema_ref", "payload_schema_ref"}:
            raise TaskCatalogError("output_artifacts 字段集合无效")
        return cls(
            artifact_kind=value["artifact_kind"],  # type: ignore[arg-type]
            schema_ref=value["schema_ref"],  # type: ignore[arg-type]
            payload_schema_ref=value["payload_schema_ref"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
    """重放绑定规则；同配置、同输入与已存在输出的处理彼此独立冻结。"""

    strategy: ReplayStrategy
    requires_same_config_hash: bool
    requires_same_input_hashes: bool
    existing_output_policy: ExistingOutputPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ReplayStrategy):
            raise TaskCatalogError("replay_policy.strategy 必须是 ReplayStrategy")
        if type(self.requires_same_config_hash) is not bool:
            raise TaskCatalogError("requires_same_config_hash 必须是 bool")
        if type(self.requires_same_input_hashes) is not bool:
            raise TaskCatalogError("requires_same_input_hashes 必须是 bool")
        if not isinstance(self.existing_output_policy, ExistingOutputPolicy):
            raise TaskCatalogError("existing_output_policy 类型无效")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "strategy": self.strategy.value,
            "requires_same_config_hash": self.requires_same_config_hash,
            "requires_same_input_hashes": self.requires_same_input_hashes,
            "existing_output_policy": self.existing_output_policy.value,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReplayPolicy":
        expected = {
            "strategy",
            "requires_same_config_hash",
            "requires_same_input_hashes",
            "existing_output_policy",
        }
        if set(value) != expected:
            raise TaskCatalogError("replay_policy 字段集合无效")
        try:
            strategy = ReplayStrategy(value["strategy"])
            output_policy = ExistingOutputPolicy(value["existing_output_policy"])
        except (TypeError, ValueError) as error:
            raise TaskCatalogError("replay_policy 枚举值无效") from error
        return cls(
            strategy=strategy,
            requires_same_config_hash=value["requires_same_config_hash"],  # type: ignore[arg-type]
            requires_same_input_hashes=value["requires_same_input_hashes"],  # type: ignore[arg-type]
            existing_output_policy=output_policy,
        )


@dataclass(frozen=True, slots=True)
class LocalFixturePolicy:
    """本机 fixture 的执行资格与 formal 隔离规则。"""

    supported: bool
    artifact_formal_eligible: bool
    may_satisfy_formal_gate: bool
    gate_status_on_success: str

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise TaskCatalogError("local_fixture.supported 必须是 bool")
        if type(self.artifact_formal_eligible) is not bool:
            raise TaskCatalogError("local_fixture.artifact_formal_eligible 必须是 bool")
        if type(self.may_satisfy_formal_gate) is not bool:
            raise TaskCatalogError("local_fixture.may_satisfy_formal_gate 必须是 bool")
        if self.artifact_formal_eligible or self.may_satisfy_formal_gate:
            raise TaskCatalogError("local_fixture 不能获得 formal 资格或满足 formal Gate")
        if self.gate_status_on_success != "NOT_RUN":
            raise TaskCatalogError("local_fixture 成功后 formal gate_status 必须保持 NOT_RUN")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "supported": self.supported,
            "artifact_formal_eligible": self.artifact_formal_eligible,
            "may_satisfy_formal_gate": self.may_satisfy_formal_gate,
            "gate_status_on_success": self.gate_status_on_success,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LocalFixturePolicy":
        expected = {
            "supported",
            "artifact_formal_eligible",
            "may_satisfy_formal_gate",
            "gate_status_on_success",
        }
        if set(value) != expected:
            raise TaskCatalogError("local_fixture 字段集合无效")
        return cls(
            supported=value["supported"],  # type: ignore[arg-type]
            artifact_formal_eligible=value["artifact_formal_eligible"],  # type: ignore[arg-type]
            may_satisfy_formal_gate=value["may_satisfy_formal_gate"],  # type: ignore[arg-type]
            gate_status_on_success=value["gate_status_on_success"],  # type: ignore[arg-type]
        )


def _require_string(value: object, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise TaskCatalogError(f"{field} 必须是非空字符串")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise TaskCatalogError(f"{field} 格式无效：{value!r}")
    return value


def _string_tuple(
    values: Sequence[object],
    *,
    field: str,
    pattern: re.Pattern[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TaskCatalogError(f"{field} 必须是字符串数组")
    normalized = tuple(
        _require_string(item, field=f"{field}[{index}]", pattern=pattern)
        for index, item in enumerate(values)
    )
    if not allow_empty and not normalized:
        raise TaskCatalogError(f"{field} 不能为空")
    if len(set(normalized)) != len(normalized):
        raise TaskCatalogError(f"{field} 不允许重复值")
    return normalized


@dataclass(frozen=True, slots=True)
class FormalEligibilityPolicy:
    """正式运行资格要求；它描述前置证据，不表示这些证据当前已经存在。"""

    supported: bool
    required_contract_stages: tuple[int, ...]
    required_gate_ids: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    requires_estimator_decision: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.supported, bool):
            raise TaskCatalogError("formal_eligibility.supported 必须是 bool")
        stages = tuple(self.required_contract_stages)
        if any(isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9 for stage in stages):
            raise TaskCatalogError("required_contract_stages 只能包含 0--9 的整数")
        if len(set(stages)) != len(stages) or stages != tuple(sorted(stages)):
            raise TaskCatalogError("required_contract_stages 必须唯一且升序")
        gates = _string_tuple(
            self.required_gate_ids,
            field="formal_eligibility.required_gate_ids",
            pattern=_GATE_ID_RE,
            allow_empty=True,
        )
        capabilities = _string_tuple(
            self.required_capabilities,
            field="formal_eligibility.required_capabilities",
            pattern=_NAME_RE,
            allow_empty=True,
        )
        if not isinstance(self.requires_estimator_decision, bool):
            raise TaskCatalogError("requires_estimator_decision 必须是 bool")
        if not self.supported and (stages or gates or capabilities or self.requires_estimator_decision):
            raise TaskCatalogError("不支持 formal 的任务不能声明 formal 前置条件")
        object.__setattr__(self, "required_contract_stages", stages)
        object.__setattr__(self, "required_gate_ids", gates)
        object.__setattr__(self, "required_capabilities", capabilities)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "supported": self.supported,
            "required_contract_stages": list(self.required_contract_stages),
            "required_gate_ids": list(self.required_gate_ids),
            "required_capabilities": list(self.required_capabilities),
            "requires_estimator_decision": self.requires_estimator_decision,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FormalEligibilityPolicy":
        expected = {
            "supported",
            "required_contract_stages",
            "required_gate_ids",
            "required_capabilities",
            "requires_estimator_decision",
        }
        if set(value) != expected:
            raise TaskCatalogError("formal_eligibility 字段集合不符合 v2 合同")
        return cls(
            supported=value["supported"],  # type: ignore[arg-type]
            required_contract_stages=tuple(value["required_contract_stages"]),  # type: ignore[arg-type]
            required_gate_ids=tuple(value["required_gate_ids"]),  # type: ignore[arg-type]
            required_capabilities=tuple(value["required_capabilities"]),  # type: ignore[arg-type]
            requires_estimator_decision=value["requires_estimator_decision"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """一个 canonical 任务的静态执行合同。"""

    task_id: str
    stage: int
    title: str
    plan_ref: str
    runner_kind: RunnerKind
    config_schema_ref: str
    config_paths: tuple[str, ...]
    predecessor_task_ids: tuple[str, ...]
    input_artifacts: tuple[InputArtifactContract, ...]
    artifact_kinds: tuple[str, ...]
    output_artifacts: tuple[OutputArtifactContract, ...]
    recovery_mode: RecoveryMode
    safe_boundary: SafeBoundary
    replay_policy: ReplayPolicy
    completion_rules: tuple[str, ...]
    failure_rules: tuple[str, ...]
    local_fixture: LocalFixturePolicy
    formal_eligibility: FormalEligibilityPolicy

    def __post_init__(self) -> None:
        match = _TASK_ID_RE.fullmatch(self.task_id) if isinstance(self.task_id, str) else None
        if match is None:
            raise TaskCatalogError(f"task_id 格式无效：{self.task_id!r}")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise TaskCatalogError("stage 必须是 0--9 的整数")
        if int(match.group("stage")) != self.stage:
            raise TaskCatalogError("task_id 的 stage 前缀与 stage 字段不一致")
        _require_string(self.title, field="title")
        _require_string(self.plan_ref, field="plan_ref")
        if not isinstance(self.runner_kind, RunnerKind):
            raise TaskCatalogError("runner_kind 必须是 RunnerKind")
        _require_string(
            self.config_schema_ref,
            field="config_schema_ref",
            pattern=_SCHEMA_REF_RE,
        )
        paths = _string_tuple(self.config_paths, field="config_paths", pattern=_CONFIG_PATH_RE)
        predecessors = _string_tuple(
            self.predecessor_task_ids,
            field="predecessor_task_ids",
            pattern=_TASK_ID_RE,
            allow_empty=True,
        )
        if self.task_id in predecessors:
            raise TaskCatalogError("任务不能把自身声明为 predecessor")
        inputs = tuple(self.input_artifacts)
        if any(not isinstance(item, InputArtifactContract) for item in inputs):
            raise TaskCatalogError("input_artifacts 只能包含 InputArtifactContract")
        input_ids = tuple(item.input_id for item in inputs)
        if len(set(input_ids)) != len(input_ids):
            raise TaskCatalogError("input_artifacts.input_id 不允许重复")
        covered_predecessors = {
            producer
            for item in inputs
            if item.required
            for producer in item.producer_task_ids
        }
        if covered_predecessors != set(predecessors):
            raise TaskCatalogError("每个 predecessor 必须且只能由 required input artifact 覆盖")
        artifacts = _string_tuple(self.artifact_kinds, field="artifact_kinds", pattern=_NAME_RE)
        outputs = tuple(self.output_artifacts)
        if any(not isinstance(item, OutputArtifactContract) for item in outputs):
            raise TaskCatalogError("output_artifacts 只能包含 OutputArtifactContract")
        output_kinds = tuple(item.artifact_kind for item in outputs)
        if output_kinds != artifacts:
            raise TaskCatalogError("output_artifacts 必须按顺序完整覆盖 artifact_kinds")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise TaskCatalogError("recovery_mode 必须是 RecoveryMode")
        if not isinstance(self.safe_boundary, SafeBoundary):
            raise TaskCatalogError("safe_boundary 必须是 SafeBoundary")
        if self.recovery_mode is RecoveryMode.RESTART_IDEMPOTENT and self.safe_boundary is SafeBoundary.ATTEMPT_COMMIT_STATE:
            raise TaskCatalogError("restart_idempotent 不能把未完成 attempt 当作恢复边界")
        if not isinstance(self.replay_policy, ReplayPolicy):
            raise TaskCatalogError("replay_policy 类型错误")
        if self.replay_policy.requires_same_input_hashes != bool(inputs):
            raise TaskCatalogError("replay_policy 的输入 hash 绑定必须与 input_artifacts 是否为空一致")
        completion_rules = _string_tuple(
            self.completion_rules,
            field="completion_rules",
            pattern=_RULE_ID_RE,
        )
        failure_rules = _string_tuple(
            self.failure_rules,
            field="failure_rules",
            pattern=_RULE_ID_RE,
        )
        if set(completion_rules) & set(failure_rules):
            raise TaskCatalogError("completion_rules 与 failure_rules 不能复用同一 rule ID")
        if not isinstance(self.local_fixture, LocalFixturePolicy):
            raise TaskCatalogError("local_fixture 类型错误")
        if not isinstance(self.formal_eligibility, FormalEligibilityPolicy):
            raise TaskCatalogError("formal_eligibility 类型错误")
        object.__setattr__(self, "config_paths", paths)
        object.__setattr__(self, "predecessor_task_ids", predecessors)
        object.__setattr__(self, "input_artifacts", inputs)
        object.__setattr__(self, "artifact_kinds", artifacts)
        object.__setattr__(self, "output_artifacts", outputs)
        object.__setattr__(self, "completion_rules", completion_rules)
        object.__setattr__(self, "failure_rules", failure_rules)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_DEFINITION_SCHEMA_VERSION,
            "task_id": self.task_id,
            "stage": self.stage,
            "title": self.title,
            "plan_ref": self.plan_ref,
            "runner_kind": self.runner_kind.value,
            "config_schema_ref": self.config_schema_ref,
            "config_paths": list(self.config_paths),
            "predecessor_task_ids": list(self.predecessor_task_ids),
            "input_artifacts": [item.to_dict() for item in self.input_artifacts],
            "artifact_kinds": list(self.artifact_kinds),
            "output_artifacts": [item.to_dict() for item in self.output_artifacts],
            "recovery": {
                "mode": self.recovery_mode.value,
                "safe_boundary": self.safe_boundary.value,
            },
            "replay_policy": self.replay_policy.to_dict(),
            "completion_rules": list(self.completion_rules),
            "failure_rules": list(self.failure_rules),
            "local_fixture": self.local_fixture.to_dict(),
            "formal_eligibility": self.formal_eligibility.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskDefinition":
        expected = {
            "schema_version",
            "task_id",
            "stage",
            "title",
            "plan_ref",
            "runner_kind",
            "config_schema_ref",
            "config_paths",
            "predecessor_task_ids",
            "input_artifacts",
            "artifact_kinds",
            "output_artifacts",
            "recovery",
            "replay_policy",
            "completion_rules",
            "failure_rules",
            "local_fixture",
            "formal_eligibility",
        }
        if set(value) != expected or value.get("schema_version") != TASK_DEFINITION_SCHEMA_VERSION:
            raise TaskCatalogError("TaskDefinition 字段集合或 schema_version 无效")
        recovery = value["recovery"]
        replay_policy = value["replay_policy"]
        local_fixture = value["local_fixture"]
        policy = value["formal_eligibility"]
        if not isinstance(recovery, Mapping) or set(recovery) != {"mode", "safe_boundary"}:
            raise TaskCatalogError("recovery 字段集合无效")
        if not isinstance(policy, Mapping):
            raise TaskCatalogError("formal_eligibility 必须是 object")
        if not isinstance(replay_policy, Mapping):
            raise TaskCatalogError("replay_policy 必须是 object")
        if not isinstance(local_fixture, Mapping):
            raise TaskCatalogError("local_fixture 必须是 object")
        raw_inputs = value["input_artifacts"]
        raw_outputs = value["output_artifacts"]
        if isinstance(raw_inputs, (str, bytes)) or not isinstance(raw_inputs, Sequence):
            raise TaskCatalogError("input_artifacts 必须是数组")
        if isinstance(raw_outputs, (str, bytes)) or not isinstance(raw_outputs, Sequence):
            raise TaskCatalogError("output_artifacts 必须是数组")
        inputs: list[InputArtifactContract] = []
        for index, item in enumerate(raw_inputs):
            if not isinstance(item, Mapping):
                raise TaskCatalogError(f"input_artifacts[{index}] 必须是 object")
            inputs.append(InputArtifactContract.from_mapping(item))
        outputs: list[OutputArtifactContract] = []
        for index, item in enumerate(raw_outputs):
            if not isinstance(item, Mapping):
                raise TaskCatalogError(f"output_artifacts[{index}] 必须是 object")
            outputs.append(OutputArtifactContract.from_mapping(item))
        try:
            runner_kind = RunnerKind(value["runner_kind"])
            recovery_mode = RecoveryMode(recovery["mode"])
            safe_boundary = SafeBoundary(recovery["safe_boundary"])
        except (TypeError, ValueError) as error:
            raise TaskCatalogError("runner/recovery 枚举值无效") from error
        return cls(
            task_id=value["task_id"],  # type: ignore[arg-type]
            stage=value["stage"],  # type: ignore[arg-type]
            title=value["title"],  # type: ignore[arg-type]
            plan_ref=value["plan_ref"],  # type: ignore[arg-type]
            runner_kind=runner_kind,
            config_schema_ref=value["config_schema_ref"],  # type: ignore[arg-type]
            config_paths=tuple(value["config_paths"]),  # type: ignore[arg-type]
            predecessor_task_ids=tuple(value["predecessor_task_ids"]),  # type: ignore[arg-type]
            input_artifacts=tuple(inputs),
            artifact_kinds=tuple(value["artifact_kinds"]),  # type: ignore[arg-type]
            output_artifacts=tuple(outputs),
            recovery_mode=recovery_mode,
            safe_boundary=safe_boundary,
            replay_policy=ReplayPolicy.from_mapping(replay_policy),
            completion_rules=tuple(value["completion_rules"]),  # type: ignore[arg-type]
            failure_rules=tuple(value["failure_rules"]),  # type: ignore[arg-type]
            local_fixture=LocalFixturePolicy.from_mapping(local_fixture),
            formal_eligibility=FormalEligibilityPolicy.from_mapping(policy),
        )


@dataclass(frozen=True, slots=True)
class TaskCatalog:
    """按 ``task_id`` 排序并由 SHA-256 绑定的不可变任务目录。"""

    tasks: tuple[TaskDefinition, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise TaskCatalogError("任务目录不能为空")
        if any(not isinstance(task, TaskDefinition) for task in self.tasks):
            raise TaskCatalogError("tasks 只能包含 TaskDefinition")
        ordered = tuple(sorted(self.tasks, key=lambda item: item.task_id))
        identifiers = tuple(task.task_id for task in ordered)
        if len(set(identifiers)) != len(identifiers):
            raise TaskCatalogError("task_id 不能重复")
        plan_refs = [task.plan_ref for task in ordered if task.stage <= 3]
        if len(set(plan_refs)) != len(plan_refs):
            raise TaskCatalogError("Stage 0--3 的 plan_ref 必须一一对应")
        by_id = {task.task_id: task for task in ordered}
        roots: list[str] = []
        for task in ordered:
            if not task.predecessor_task_ids:
                roots.append(task.task_id)
            for predecessor_id in task.predecessor_task_ids:
                predecessor = by_id.get(predecessor_id)
                if predecessor is None:
                    raise TaskCatalogError(
                        f"predecessor_task_ids 引用了目录外任务：{predecessor_id}"
                    )
                if predecessor.stage > task.stage:
                    raise TaskCatalogError("任务不能依赖未来 Stage 的 predecessor")
        if roots != ["stage0.01_baseline_and_safety"]:
            raise TaskCatalogError("完整 StageTaskCatalog 只能有 stage0.01 一个根任务")

        # 显式 DFS 检查同 Stage 内的环。只比较 stage 数无法发现 A -> B -> A，若不在
        # catalog 冻结时拒绝，运行器可能永远等待一个不可能发布的上游 commit。
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise TaskCatalogError(f"任务 predecessor 图存在环：{task_id}")
            visiting.add(task_id)
            for predecessor_id in by_id[task_id].predecessor_task_ids:
                visit(predecessor_id)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in identifiers:
            visit(task_id)
        object.__setattr__(self, "tasks", ordered)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def get(self, task_id: str) -> TaskDefinition:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise TaskCatalogError(f"未知 canonical task_id：{task_id!r}")

    @property
    def catalog_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_CATALOG_SCHEMA_VERSION,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_dict(self) -> dict[str, JSONValue]:
        payload = self._payload()
        payload["catalog_hash"] = self.catalog_hash
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskCatalog":
        if set(value) != {"schema_version", "tasks", "catalog_hash"}:
            raise TaskCatalogError("TaskCatalog 字段集合不符合 v2 合同")
        if value["schema_version"] != TASK_CATALOG_SCHEMA_VERSION:
            raise TaskCatalogError("TaskCatalog schema_version 不受支持")
        raw_tasks = value["tasks"]
        if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
            raise TaskCatalogError("tasks 必须是数组")
        tasks: list[TaskDefinition] = []
        for index, item in enumerate(raw_tasks):
            if not isinstance(item, Mapping):
                raise TaskCatalogError(f"tasks[{index}] 必须是 object")
            tasks.append(TaskDefinition.from_mapping(item))
        catalog = cls(tuple(tasks))
        if value["catalog_hash"] != catalog.catalog_hash:
            raise TaskCatalogError("catalog_hash 与任务目录内容不一致")
        return catalog


_CONFIG_PATHS_BY_KIND: Final[dict[RunnerKind, tuple[str, ...]]] = {
    RunnerKind.AUDIT: ("base_config.identity", "base_config.runtime", "execution", "artifacts"),
    RunnerKind.STORAGE: ("base_config.runtime", "base_config.checkpoint", "execution", "recovery", "artifacts"),
    RunnerKind.ENVIRONMENT: ("base_config.runtime", "execution", "artifacts"),
    RunnerKind.ASSET: ("base_config.model", "base_config.data", "execution", "recovery", "artifacts"),
    RunnerKind.CONTRACT: ("base_config.identity", "base_config.runtime", "execution", "artifacts"),
    RunnerKind.TRAINING: (
        "base_config.model", "base_config.data", "base_config.loss", "base_config.batching",
        "base_config.precision", "base_config.optimizer", "base_config.importance", "training",
        "scheduler", "data_loader", "providers", "evaluation", "profiling",
        "checkpoint_schedule", "precision_runtime", "optimizer_runtime", "launcher",
        "orchestration", "recovery", "artifacts",
    ),
    RunnerKind.DISTRIBUTED_TRAINING: (
        "base_config.model", "base_config.data", "base_config.batching", "base_config.distributed",
        "base_config.precision", "base_config.optimizer", "base_config.importance", "training",
        "scheduler", "data_loader", "providers", "evaluation", "profiling",
        "checkpoint_schedule", "precision_runtime", "optimizer_runtime", "launcher",
        "orchestration", "recovery", "artifacts",
    ),
    RunnerKind.OBSERVABILITY: ("base_config.logging", "base_config.checkpoint", "execution", "recovery", "artifacts"),
    RunnerKind.CHECKPOINT: ("base_config.checkpoint", "base_config.distributed", "checkpoint_schedule", "launcher", "recovery", "artifacts"),
    RunnerKind.CAPACITY: ("base_config.runtime", "base_config.model", "base_config.distributed", "providers", "profiling", "launcher", "execution", "artifacts"),
    RunnerKind.TEST_MATRIX: ("base_config.identity", "base_config.runtime", "execution", "recovery", "artifacts"),
    RunnerKind.DELIVERY: ("base_config.identity", "base_config.runtime", "execution", "recovery", "artifacts"),
    RunnerKind.REGISTRY: ("base_config.model", "base_config.optimizer", "base_config.precision", "providers", "optimizer_runtime", "artifacts"),
    RunnerKind.ORACLE: ("base_config.model", "base_config.data", "base_config.loss", "base_config.precision", "providers", "artifacts"),
    RunnerKind.VALIDATION: ("base_config.loss", "base_config.batching", "base_config.precision", "providers", "precision_runtime", "launcher", "execution", "artifacts"),
    RunnerKind.ESTIMATOR: ("base_config.importance", "base_config.sampling", "base_config.precision", "orchestration", "execution", "artifacts"),
    RunnerKind.REFERENCE: ("base_config.model", "base_config.data", "base_config.sampling", "providers", "precision_runtime", "launcher", "orchestration", "recovery", "artifacts"),
    RunnerKind.ESTIMATOR_EXPERIMENT: ("base_config.sampling", "base_config.importance", "providers", "precision_runtime", "launcher", "orchestration", "recovery", "artifacts"),
    RunnerKind.PILOT: ("base_config.sampling", "base_config.importance", "providers", "profiling", "launcher", "orchestration", "execution", "recovery", "artifacts"),
    RunnerKind.STATISTICS: ("base_config.analysis", "execution", "recovery", "artifacts"),
    RunnerKind.REPORTING: ("base_config.analysis", "execution", "recovery", "artifacts"),
    RunnerKind.PATH_INTEGRATION: ("base_config.path_integration", "base_config.precision", "providers", "precision_runtime", "launcher", "orchestration", "execution", "recovery", "artifacts"),
    RunnerKind.ROUTE_TRAINING: (
        "base_config.model", "base_config.data", "base_config.optimizer", "base_config.importance",
        "training", "scheduler", "data_loader", "providers", "evaluation", "profiling",
        "checkpoint_schedule", "precision_runtime", "optimizer_runtime", "launcher",
        "orchestration", "recovery", "artifacts",
    ),
    RunnerKind.PRUNING: ("base_config.pruning", "base_config.analysis", "providers", "evaluation", "launcher", "orchestration", "execution", "recovery", "artifacts"),
    RunnerKind.ABLATION: ("base_config.analysis", "base_config.sampling", "providers", "evaluation", "launcher", "orchestration", "execution", "recovery", "artifacts"),
    RunnerKind.ANALYSIS: ("base_config.analysis", "orchestration", "execution", "recovery", "artifacts"),
}


def _formal_policy(
    stage: int,
    *,
    gates: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    estimator_decision: bool = False,
) -> FormalEligibilityPolicy:
    return FormalEligibilityPolicy(
        supported=True,
        required_contract_stages=tuple(range(stage + 1)),
        required_gate_ids=tuple(gates),
        required_capabilities=tuple(capabilities),
        requires_estimator_decision=estimator_decision,
    )


def _replay_strategy(recovery_mode: RecoveryMode) -> ReplayStrategy:
    """把恢复机制投影成对外稳定的 replay 语义。"""

    return {
        RecoveryMode.RESTART_IDEMPOTENT: ReplayStrategy.RESTART_FROM_CONFIG,
        RecoveryMode.RESUME_CHECKPOINT: ReplayStrategy.RESUME_FROM_COMMIT,
        RecoveryMode.RESUME_SHARDS: ReplayStrategy.RESUME_FROM_COMMIT,
        RecoveryMode.RECONCILE_STATE: ReplayStrategy.RECONCILE_COMMITTED_STATE,
        RecoveryMode.REBUILD_DERIVED: ReplayStrategy.REBUILD_FROM_FROZEN_SOURCES,
        RecoveryMode.MANUAL_EXTERNAL: ReplayStrategy.MANUAL_EXTERNAL_REVIEW,
    }[recovery_mode]


def _task(
    task_id: str,
    title: str,
    plan_ref: str,
    runner_kind: RunnerKind,
    artifacts: Sequence[str],
    recovery_mode: RecoveryMode,
    safe_boundary: SafeBoundary,
    *,
    gates: Sequence[str] = (),
    capabilities: Sequence[str] = (),
    estimator_decision: bool = False,
) -> TaskDefinition:
    match = _TASK_ID_RE.fullmatch(task_id)
    if match is None:  # pragma: no cover - 常量定义错误会在导入时立即暴露
        raise TaskCatalogError(f"内部 task_id 无效：{task_id}")
    stage = int(match.group("stage"))
    return TaskDefinition(
        task_id=task_id,
        stage=stage,
        title=title,
        plan_ref=plan_ref,
        runner_kind=runner_kind,
        config_schema_ref=RESOLVED_CONFIG_V2_SCHEMA_REF,
        config_paths=_CONFIG_PATHS_BY_KIND[runner_kind],
        predecessor_task_ids=(),
        input_artifacts=(),
        artifact_kinds=tuple(artifacts),
        output_artifacts=tuple(
            OutputArtifactContract(
                artifact_kind=artifact_kind,
                schema_ref=TASK_OUTPUT_ARTIFACT_SCHEMA_REF,
                payload_schema_ref=TASK_OUTPUT_PAYLOAD_SCHEMA_REF,
            )
            for artifact_kind in artifacts
        ),
        recovery_mode=recovery_mode,
        safe_boundary=safe_boundary,
        replay_policy=ReplayPolicy(
            strategy=_replay_strategy(recovery_mode),
            requires_same_config_hash=True,
            requires_same_input_hashes=False,
            existing_output_policy=ExistingOutputPolicy.REUSE_IDENTICAL_REJECT_DRIFT,
        ),
        completion_rules=(
            "all_required_artifact_commits_published",
            "artifact_hashes_verified",
            "artifact_scope_matches_run_intent",
        ),
        failure_rules=(
            "unknown_or_missing_config_fails_closed",
            "input_contract_mismatch_fails_closed",
            "artifact_commit_conflict_fails_closed",
            "missing_runtime_prerequisite_returns_blocked",
        ),
        local_fixture=LocalFixturePolicy(
            supported=True,
            artifact_formal_eligible=False,
            may_satisfy_formal_gate=False,
            gate_status_on_success="NOT_RUN",
        ),
        formal_eligibility=_formal_policy(
            stage,
            gates=gates,
            capabilities=capabilities,
            estimator_decision=estimator_decision,
        ),
    )


_TASKS_RAW: Final = (
    # Stage 0：每一项都对应 plan/stage0 下同名编号文件。
    _task("stage0.01_baseline_and_safety", "基线、安全与硬件现状冻结", "plan/stage0/01_baseline_and_safety.md", RunnerKind.AUDIT, ("baseline_report", "safety_report", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, capabilities=("git", "server")),
    _task("stage0.02_storage_and_layout", "存储边界、布局与持久性", "plan/stage0/02_storage_and_layout.md", RunnerKind.STORAGE, ("storage_layout_manifest", "storage_validation_report", "persistence_decision"), RecoveryMode.RECONCILE_STATE, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G0-C",), capabilities=("server",)),
    _task("stage0.03_runtime_and_dependencies", "运行环境与依赖重建", "plan/stage0/03_runtime_and_dependencies.md", RunnerKind.ENVIRONMENT, ("environment_manifest", "dependency_audit", "offline_rebuild_report"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G0-C", "stage0.G1"), capabilities=("server", "wheelhouse")),
    _task("stage0.04_assets_and_manifests", "模型、数据与资产 manifest", "plan/stage0/04_assets_and_manifests.md", RunnerKind.ASSET, ("asset_manifest", "asset_audit", "asset_resolution"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage0.G0-C", "stage0.G1"), capabilities=("server", "model_assets", "data_assets")),
    _task("stage0.05_config_run_identity_and_seeds", "配置、运行身份、seed 与 provenance", "plan/stage0/05_config_run_identity_and_seeds.md", RunnerKind.CONTRACT, ("resolved_config", "run_identity", "seed_plan", "provenance_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G0-C", "stage0.G1")),
    _task("stage0.06_single_gpu_smoke", "单 GPU 训练 smoke", "plan/stage0/06_single_gpu_smoke.md", RunnerKind.TRAINING, ("training_smoke_result", "event_stream", "checkpoint_commit"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage0.G0-G", "stage0.G2", "stage0.G3-S1", "stage0.G4"), capabilities=("server", "cuda", "model_assets", "data_assets")),
    _task("stage0.07_ddp_and_gradient_semantics", "DDP 与全局梯度语义", "plan/stage0/07_ddp_and_gradient_semantics.md", RunnerKind.DISTRIBUTED_TRAINING, ("distributed_validation", "gradient_semantics_report", "communication_report"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage0.G0-G", "stage0.G5"), capabilities=("server", "cuda", "nccl")),
    _task("stage0.08_logging_and_tracking", "事件日志、追踪与 canonical lineage", "plan/stage0/08_logging_and_tracking.md", RunnerKind.OBSERVABILITY, ("event_stream", "lineage_manifest", "logging_overhead_report"), RecoveryMode.RECONCILE_STATE, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage0.G1", "stage0.G4", "stage0.G5"), capabilities=("server",)),
    _task("stage0.09_checkpoint_and_resume", "Checkpoint、恢复与保留", "plan/stage0/09_checkpoint_and_resume.md", RunnerKind.CHECKPOINT, ("checkpoint_commit", "resume_equivalence_report", "retention_report"), RecoveryMode.RECONCILE_STATE, SafeBoundary.CHECKPOINT_COMMIT, gates=("stage0.G5", "stage0.G6"), capabilities=("server", "cuda")),
    _task("stage0.10_capacity_and_operations", "容量、安全运行与故障处置", "plan/stage0/10_capacity_and_operations.md", RunnerKind.CAPACITY, ("capacity_envelope", "operations_preflight", "fault_report"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G0-G", "stage0.G5", "stage0.G6", "stage0.G7"), capabilities=("server", "cuda")),
    _task("stage0.11_test_quality_and_replay", "测试分层与确定性重放", "plan/stage0/11_test_quality_and_replay.md", RunnerKind.TEST_MATRIX, ("test_report", "replay_report", "gate_summary"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G8-C",), capabilities=("server",)),
    _task("stage0.12_delivery_and_sync", "交付、工作日志与多端同步", "plan/stage0/12_delivery_and_sync.md", RunnerKind.DELIVERY, ("delivery_manifest", "worklog", "sync_report"), RecoveryMode.MANUAL_EXTERNAL, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G9",), capabilities=("git", "github", "server")),

    # Stage 1：公式正确性、训练 step、真实单卡/DDP 和恢复 Gate。
    _task("stage1.01_entry_and_contract", "进入条件与数学合同冻结", "plan/stage1/01_entry_and_contract.md", RunnerKind.CONTRACT, ("stage_contract", "requirements_matrix", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G10",)),
    _task("stage1.02_architecture_and_parameter_registry", "架构边界与参数 registry", "plan/stage1/02_architecture_and_parameter_registry.md", RunnerKind.REGISTRY, ("parameter_registry", "registry_validation_report", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-ENTRY", "stage1.G1-CONTRACT")),
    _task("stage1.03_fixtures_and_oracles", "Fixture 与独立 oracle", "plan/stage1/03_fixtures_and_oracles.md", RunnerKind.ORACLE, ("fixture_manifest", "oracle_bundle", "oracle_validation_report"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-CONTRACT", "stage1.G1-REGISTRY")),
    _task("stage1.04_loss_and_gradient_scale", "Loss reduction 与梯度尺度", "plan/stage1/04_loss_and_gradient_scale.md", RunnerKind.VALIDATION, ("gradient_scale_report", "comparison_table", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-CONTRACT", "stage1.G1-REGISTRY", "stage1.G1-ORACLE")),
    _task("stage1.05_estimators", "Raw、double 与 U 估计器", "plan/stage1/05_estimators.md", RunnerKind.ESTIMATOR, ("estimator_validation_report", "estimator_tensor_bundle", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-ORACLE", "stage1.G1-GRAD")),
    _task("stage1.06_training_integration_and_accumulators", "训练 step 集成与累计器", "plan/stage1/06_training_integration_and_accumulators.md", RunnerKind.TRAINING, ("step_validation_report", "importance_trajectory", "gate_record"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage1.G1-GRAD", "stage1.G1-EST")),
    _task("stage1.07_single_gpu_pythia14m", "Pythia-14M 单 GPU 验证", "plan/stage1/07_single_gpu_pythia14m.md", RunnerKind.TRAINING, ("single_gpu_report", "importance_trajectory", "checkpoint_commit"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage1.G1-ENTRY", "stage1.G1-REGISTRY", "stage1.G1-ORACLE", "stage1.G1-GRAD", "stage1.G1-EST", "stage1.G1-STEP"), capabilities=("server", "cuda", "model_assets", "data_assets")),
    _task("stage1.08_ddp_and_gradient_accumulation", "DDP、梯度累积与 no_sync", "plan/stage1/08_ddp_and_gradient_accumulation.md", RunnerKind.DISTRIBUTED_TRAINING, ("ddp_equivalence_report", "distributed_statistics", "gate_record"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage1.G1-SINGLE",), capabilities=("server", "cuda", "nccl")),
    _task("stage1.09_precision_clipping_and_optimizer_boundaries", "AMP、裁剪与 optimizer 边界", "plan/stage1/09_precision_clipping_and_optimizer_boundaries.md", RunnerKind.VALIDATION, ("numeric_boundary_report", "skip_lifecycle_report", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-SINGLE",), capabilities=("server", "cuda")),
    _task("stage1.10_checkpoint_resume_and_artifacts", "重要性状态 checkpoint 与恢复", "plan/stage1/10_checkpoint_resume_and_artifacts.md", RunnerKind.CHECKPOINT, ("training_state_manifest", "resume_equivalence_report", "gate_record"), RecoveryMode.RECONCILE_STATE, SafeBoundary.CHECKPOINT_COMMIT, gates=("stage1.G1-STEP", "stage1.G1-SINGLE", "stage1.G1-NUMERIC"), capabilities=("server", "cuda")),
    _task("stage1.11_reporting_and_exit_gate", "Stage 1 报告与退出 Gate", "plan/stage1/11_reporting_and_exit_gate.md", RunnerKind.REPORTING, ("stage_report", "requirements_matrix", "gate_summary", "delivery_manifest"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage1.G1-DDP", "stage1.G1-RESUME"), capabilities=("server", "git")),

    # Stage 2：固定状态 reference、成对实验、统计决策与交付。
    _task("stage2.01_scope_hypotheses_and_preregistration", "范围、假设与预注册", "plan/stage2/01_scope_hypotheses_and_preregistration.md", RunnerKind.CONTRACT, ("preregistration", "hypothesis_contract", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage1.G1-EXIT",)),
    _task("stage2.02_stage1_handoff_and_fixed_state_contract", "Stage 1 交接与固定状态合同", "plan/stage2/02_stage1_handoff_and_fixed_state_contract.md", RunnerKind.AUDIT, ("handoff_manifest", "fixed_state_contract", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage2.G2.0", "stage1.G1-EXIT")),
    _task("stage2.03_assets_checkpoints_and_sampling", "资产、checkpoint 与抽样流", "plan/stage2/03_assets_checkpoints_and_sampling.md", RunnerKind.ASSET, ("sampling_plan", "draw_manifest", "asset_resolution", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage2.G2.1",), capabilities=("server", "cuda", "model_assets", "data_assets")),
    _task("stage2.04_reference_target", "高精度 reference 目标", "plan/stage2/04_reference_target.md", RunnerKind.REFERENCE, ("reference_result", "reference_convergence_report", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage2.G2.2",), capabilities=("server", "cuda", "model_assets", "data_assets")),
    _task("stage2.05_paired_estimator_runner", "成对估计器 runner", "plan/stage2/05_paired_estimator_runner.md", RunnerKind.ESTIMATOR_EXPERIMENT, ("paired_runner_report", "sufficient_stat_shards", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage2.G2.1", "stage2.G2.2-dev"), capabilities=("server", "cuda")),
    _task("stage2.06_pilot_and_matrix_freeze", "Pilot 与正式矩阵冻结", "plan/stage2/06_pilot_and_matrix_freeze.md", RunnerKind.PILOT, ("pilot_report", "frozen_experiment_matrix", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage2.G2.2", "stage2.G2.3", "stage2.G2.4a"), capabilities=("server", "cuda")),
    _task("stage2.07_main_sweep", "确认性主实验矩阵", "plan/stage2/07_main_sweep.md", RunnerKind.ESTIMATOR_EXPERIMENT, ("confirmatory_results", "sufficient_stat_shards", "completeness_report"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage2.G2.3", "stage2.G2.4b"), capabilities=("server", "cuda")),
    _task("stage2.08_statistics_and_robustness", "统计有效性与稳健性", "plan/stage2/08_statistics_and_robustness.md", RunnerKind.STATISTICS, ("frozen_source_table", "quality_gates", "hypothesis_decisions"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage2.G2.3", "stage2.G2.5")),
    _task("stage2.09_cost_and_system_validation", "成本与系统验证", "plan/stage2/09_cost_and_system_validation.md", RunnerKind.CAPACITY, ("cost_table", "system_validation_report", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage2.G2.5",), capabilities=("server", "cuda", "nccl")),
    _task("stage2.10_visualization_reporting_and_decision", "可视化、报告与 estimator 决策", "plan/stage2/10_visualization_reporting_and_decision.md", RunnerKind.REPORTING, ("analysis_report", "chart_artifacts", "estimator_decision", "gate_record"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage2.G2.6", "stage2.G2.7a")),
    _task("stage2.11_delivery_and_exit_gate", "Stage 2 交付与退出 Gate", "plan/stage2/11_delivery_and_exit_gate.md", RunnerKind.DELIVERY, ("delivery_manifest", "replay_report", "gate_summary", "sync_report"), RecoveryMode.MANUAL_EXTERNAL, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage2.G2.7b",), capabilities=("git", "github", "server")),

    # Stage 3：端点状态、求积 reference、正式矩阵与唯一方法决策。
    _task("stage3.01_prerequisites_and_scope", "Stage 3 前置 Gate 与范围", "plan/stage3/01_prerequisites_and_scope.md", RunnerKind.AUDIT, ("prerequisite_report", "scope_freeze", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage0.G10", "stage1.G1-EXIT", "stage2.G2.7b", "stage2.G2.8"), capabilities=("server",), estimator_decision=True),
    _task("stage3.02_math_and_metric_contract", "路径积分数学与指标合同", "plan/stage3/02_math_and_metric_contract.md", RunnerKind.CONTRACT, ("path_math_contract", "metric_contract", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage3.G3-0",), estimator_decision=True),
    _task("stage3.03_endpoint_and_probe_pipeline", "端点、probe 与状态管线", "plan/stage3/03_endpoint_and_probe_pipeline.md", RunnerKind.PATH_INTEGRATION, ("path_spec", "probe_manifest", "state_restoration_report", "gate_record"), RecoveryMode.RECONCILE_STATE, SafeBoundary.CHECKPOINT_COMMIT, gates=("stage3.G3-0", "stage3.G3-1"), capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage3.04_quadrature_engine_and_unit_tests", "求积引擎与解析测试", "plan/stage3/04_quadrature_engine_and_unit_tests.md", RunnerKind.VALIDATION, ("quadrature_rules", "analytic_validation_report", "gate_record"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage3.G3-1",), estimator_decision=True),
    _task("stage3.05_reference_integral_and_precision", "参考积分与精度预算", "plan/stage3/05_reference_integral_and_precision.md", RunnerKind.REFERENCE, ("path_integral_reference", "precision_budget", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage3.G3-2", "stage3.G3-3"), capabilities=("server", "cuda"), estimator_decision=True),
    _task("stage3.06_pilot_and_threshold_freeze", "Pilot、阈值与预算冻结", "plan/stage3/06_pilot_and_threshold_freeze.md", RunnerKind.PILOT, ("quadrature_pilot_report", "threshold_freeze", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage3.G3-4",), capabilities=("server", "cuda"), estimator_decision=True),
    _task("stage3.07_formal_experiment_matrix", "路径积分正式实验矩阵", "plan/stage3/07_formal_experiment_matrix.md", RunnerKind.PATH_INTEGRATION, ("formal_path_results", "completeness_report", "gate_record"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, gates=("stage3.G3-5",), capabilities=("server", "cuda"), estimator_decision=True),
    _task("stage3.08_error_analysis_and_stability", "误差分析与排序稳定性", "plan/stage3/08_error_analysis_and_stability.md", RunnerKind.STATISTICS, ("path_error_table", "stability_report", "frozen_source_table"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage3.G3-6",), estimator_decision=True),
    _task("stage3.09_cost_and_method_selection", "成本与求积方法选择", "plan/stage3/09_cost_and_method_selection.md", RunnerKind.ANALYSIS, ("cost_accuracy_table", "quadrature_decision", "gate_record"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage3.G3-6",), estimator_decision=True),
    _task("stage3.10_reports_visualizations_and_handoff", "报告、图表与后续交接", "plan/stage3/10_reports_visualizations_and_handoff.md", RunnerKind.REPORTING, ("analysis_report", "chart_artifacts", "handoff_manifest", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, gates=("stage3.G3-7",), capabilities=("git", "server"), estimator_decision=True),

    # Stage 4--9 的旧顶层 ID 保持兼容；其后叶任务才是可独立调度的原子单元。
    _task("stage4.minimal_complete_loop", "160M 最小完整闭环", "plan/stage4/README.md#stage4-minimal-complete-loop", RunnerKind.ROUTE_TRAINING, ("training_route", "importance_trajectory", "pruning_results", "stage_report"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage3.G3-8",), capabilities=("server", "cuda", "nccl", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage5.formal_pretraining", "410M 正式预训练与重要性轨迹", "plan/stage5/README.md#stage5-formal-pretraining", RunnerKind.ROUTE_TRAINING, ("training_route", "checkpoint_lineage", "importance_trajectory", "stage_report"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, capabilities=("server", "cuda", "nccl", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage6.training_route_comparison", "直接监督与预训练微调路线比较", "plan/stage6/README.md#stage6-training-route-comparison", RunnerKind.ROUTE_TRAINING, ("training_routes", "route_comparison_table", "importance_reuse_report", "stage_report"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, capabilities=("server", "cuda", "nccl", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage7.functional_pruning_validation", "参数重要性的功能性剪枝验证", "plan/stage7/README.md#stage7-functional-pruning-validation", RunnerKind.PRUNING, ("pruning_plan", "pruning_results", "damage_auc_table", "stage_report"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage8.ablation_and_robustness", "消融矩阵与稳健性验证", "plan/stage8/README.md#stage8-ablation-and-robustness", RunnerKind.ABLATION, ("ablation_matrix", "ablation_results", "robustness_report", "stage_report"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage9.analysis_visualization_reporting", "统计分析、可视化与报告重建", "plan/stage9/README.md#stage9-analysis-visualization-reporting", RunnerKind.ANALYSIS, ("frozen_source_table", "analysis_report", "chart_artifacts", "reproduction_manifest"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 4：路线冻结、三条训练 phase、轨迹、剪枝与报告各自可恢复。
    _task("stage4.route", "160M 闭环路线编译与 lineage 校验", "plan/stage4/README.md#stage4-route", RunnerKind.CONTRACT, ("training_route", "route_validation_report"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, gates=("stage3.G3-8",), estimator_decision=True),
    _task("stage4.pretrain", "160M 短程预训练", "plan/stage4/README.md#stage4-pretrain", RunnerKind.ROUTE_TRAINING, ("checkpoint_lineage", "training_metrics", "importance_trajectory"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage3.G3-8",), capabilities=("server", "cuda", "nccl", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage4.direct_supervised", "160M 统一初始化直接监督训练", "plan/stage4/README.md#stage4-direct-supervised", RunnerKind.ROUTE_TRAINING, ("checkpoint_lineage", "training_metrics", "importance_trajectory"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage3.G3-8",), capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage4.finetune", "160M 预训练后监督微调", "plan/stage4/README.md#stage4-finetune", RunnerKind.ROUTE_TRAINING, ("checkpoint_lineage", "training_metrics", "importance_trajectory"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, gates=("stage3.G3-8",), capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage4.importance_trajectory", "160M 重要性轨迹归约", "plan/stage4/README.md#stage4-importance-trajectory", RunnerKind.STATISTICS, ("importance_trajectory_table", "layer_module_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage4.pruning_validation", "160M 初步剪枝闭环", "plan/stage4/README.md#stage4-pruning-validation", RunnerKind.PRUNING, ("pruning_plan", "pruning_results", "damage_auc_table"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage4.report", "Stage 4 闭环报告", "plan/stage4/README.md#stage4-report", RunnerKind.REPORTING, ("stage_report", "chart_artifacts", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 5：正式预训练与派生分析分离，报告不重新执行训练。
    _task("stage5.pretrain", "410M 正式预训练", "plan/stage5/README.md#stage5-pretrain", RunnerKind.ROUTE_TRAINING, ("checkpoint_lineage", "training_metrics", "importance_trajectory"), RecoveryMode.RESUME_CHECKPOINT, SafeBoundary.ATTEMPT_COMMIT_STATE, capabilities=("server", "cuda", "nccl", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage5.importance_trajectory", "410M 重要性形成轨迹", "plan/stage5/README.md#stage5-importance-trajectory", RunnerKind.STATISTICS, ("importance_trajectory_table", "concentration_table", "topk_stability_table"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage5.checkpoint_analysis", "410M checkpoint 层级与模块分析", "plan/stage5/README.md#stage5-checkpoint-analysis", RunnerKind.ANALYSIS, ("checkpoint_analysis_table", "layer_module_summary", "heatmap_sources"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage5.report", "Stage 5 预训练报告", "plan/stage5/README.md#stage5-report", RunnerKind.REPORTING, ("stage_report", "chart_artifacts", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 6：冻结路线矩阵后再运行评价、配对比较和复用分析。
    _task("stage6.route_matrix", "监督与预训练微调路线矩阵", "plan/stage6/README.md#stage6-route-matrix", RunnerKind.CONTRACT, ("training_routes", "route_matrix", "matrix_validation_report"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, estimator_decision=True),
    _task("stage6.evaluate", "多任务路线评价", "plan/stage6/README.md#stage6-evaluate", RunnerKind.VALIDATION, ("evaluation_results", "paired_route_metrics"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage6.compare", "训练范式配对比较", "plan/stage6/README.md#stage6-compare", RunnerKind.STATISTICS, ("route_comparison_table", "confidence_intervals", "quality_gates"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage6.importance_reuse", "预训练参数重要性复用分析", "plan/stage6/README.md#stage6-importance-reuse", RunnerKind.ANALYSIS, ("importance_reuse_table", "topk_overlap_table", "layer_module_difference"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage6.report", "Stage 6 路线比较报告", "plan/stage6/README.md#stage6-report", RunnerKind.REPORTING, ("stage_report", "chart_artifacts", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 7：mask 矩阵、实际评价和归约分开，便于按不可变 cell 恢复。
    _task("stage7.matrix", "剪枝矩阵与 canonical mask 计划", "plan/stage7/README.md#stage7-matrix", RunnerKind.PRUNING, ("pruning_matrix", "pruning_plans", "mask_manifest"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, estimator_decision=True),
    _task("stage7.evaluate", "剪枝 cell 功能评价", "plan/stage7/README.md#stage7-evaluate", RunnerKind.PRUNING, ("pruning_evaluation_results", "damage_curves"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage7.reduce", "剪枝结果归约与置信区间", "plan/stage7/README.md#stage7-reduce", RunnerKind.STATISTICS, ("pruning_summary_table", "damage_auc_table", "confidence_intervals"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage7.report", "Stage 7 功能性验证报告", "plan/stage7/README.md#stage7-report", RunnerKind.REPORTING, ("stage_report", "chart_artifacts", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 8：矩阵冻结、cell 执行、归约、推荐和报告不可混为一步。
    _task("stage8.freeze", "消融矩阵与单因素合同冻结", "plan/stage8/README.md#stage8-freeze", RunnerKind.CONTRACT, ("ablation_matrix", "single_factor_validation", "matrix_freeze"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, estimator_decision=True),
    _task("stage8.execute", "消融矩阵 cell 执行", "plan/stage8/README.md#stage8-execute", RunnerKind.ABLATION, ("ablation_cell_results", "cell_lineage_manifest"), RecoveryMode.RESUME_SHARDS, SafeBoundary.SHARD_COMMIT, capabilities=("server", "cuda", "model_assets", "data_assets"), estimator_decision=True),
    _task("stage8.reduce", "消融与稳健性结果归约", "plan/stage8/README.md#stage8-reduce", RunnerKind.STATISTICS, ("ablation_summary_table", "robustness_table", "quality_gates"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage8.recommend", "方法适用范围与配置推荐", "plan/stage8/README.md#stage8-recommend", RunnerKind.ANALYSIS, ("configuration_recommendation", "applicability_report", "limitation_table"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage8.report", "Stage 8 消融报告", "plan/stage8/README.md#stage8-report", RunnerKind.REPORTING, ("stage_report", "chart_artifacts", "gate_summary"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),

    # Stage 9：所有表图与报告都只消费 hash-bound source table。
    _task("stage9.ingest", "冻结源表摄取与 lineage 验证", "plan/stage9/README.md#stage9-ingest", RunnerKind.ANALYSIS, ("frozen_source_table", "source_lineage_manifest", "ingest_report"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage9.statistics", "统一统计指标与置信区间", "plan/stage9/README.md#stage9-statistics", RunnerKind.STATISTICS, ("statistics_table", "confidence_intervals", "undefined_metric_report"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage9.tables", "论文表格重建", "plan/stage9/README.md#stage9-tables", RunnerKind.REPORTING, ("table_specs", "table_artifacts"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage9.charts", "论文图表重建", "plan/stage9/README.md#stage9-charts", RunnerKind.REPORTING, ("chart_specs", "chart_artifacts"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage9.report", "最终分析报告重建", "plan/stage9/README.md#stage9-report", RunnerKind.REPORTING, ("analysis_report", "claim_evidence_index"), RecoveryMode.REBUILD_DERIVED, SafeBoundary.CANONICAL_SOURCE, estimator_decision=True),
    _task("stage9.bundle", "论文与复现产物 bundle", "plan/stage9/README.md#stage9-bundle", RunnerKind.DELIVERY, ("reproduction_manifest", "delivery_manifest", "artifact_inventory"), RecoveryMode.MANUAL_EXTERNAL, SafeBoundary.IMMUTABLE_PUBLISH, capabilities=("git",), estimator_decision=True),
    _task("stage9.replay", "空目录确定性重放", "plan/stage9/README.md#stage9-replay", RunnerKind.TEST_MATRIX, ("replay_report", "hash_comparison", "gate_summary"), RecoveryMode.RESTART_IDEMPOTENT, SafeBoundary.IMMUTABLE_PUBLISH, estimator_decision=True),
)


# Stage 4--9 同时保留了兼容用的顶层任务与可独立调度的叶任务，因此不能依赖源码
# 声明顺序推断 DAG。该表显式冻结科学工作流；新增叶任务若没有在此声明会在导入时
# 立即失败，避免它悄悄成为一个“无需任何输入”的伪根任务。
_LATE_STAGE_PREDECESSORS: Final[dict[str, tuple[str, ...]]] = {
    "stage4.minimal_complete_loop": ("stage3.10_reports_visualizations_and_handoff",),
    "stage5.formal_pretraining": ("stage4.minimal_complete_loop",),
    "stage6.training_route_comparison": ("stage5.formal_pretraining",),
    "stage7.functional_pruning_validation": ("stage6.training_route_comparison",),
    "stage8.ablation_and_robustness": ("stage7.functional_pruning_validation",),
    "stage9.analysis_visualization_reporting": ("stage8.ablation_and_robustness",),
    "stage4.route": ("stage3.10_reports_visualizations_and_handoff",),
    "stage4.pretrain": ("stage4.route",),
    "stage4.direct_supervised": ("stage4.route",),
    "stage4.finetune": ("stage4.pretrain",),
    "stage4.importance_trajectory": (
        "stage4.pretrain", "stage4.direct_supervised", "stage4.finetune",
    ),
    "stage4.pruning_validation": ("stage4.importance_trajectory",),
    "stage4.report": ("stage4.pruning_validation",),
    "stage5.pretrain": ("stage4.report",),
    "stage5.importance_trajectory": ("stage5.pretrain",),
    "stage5.checkpoint_analysis": ("stage5.importance_trajectory",),
    "stage5.report": ("stage5.checkpoint_analysis",),
    # Stage 6 的比较矩阵必须直接绑定两条已经真实执行并发布 checkpoint/evaluation
    # 证据的路线。若只依赖 Stage 5 报告，报告里的摘要无法证明具体 checkpoint、
    # probe panel 与 metric lineage，也无法让后续 evaluate 安全重放。
    "stage6.route_matrix": ("stage4.direct_supervised", "stage4.finetune"),
    "stage6.evaluate": ("stage6.route_matrix",),
    "stage6.compare": ("stage6.evaluate",),
    "stage6.importance_reuse": ("stage6.compare",),
    "stage6.report": ("stage6.importance_reuse",),
    "stage7.matrix": ("stage6.report",),
    "stage7.evaluate": ("stage7.matrix",),
    "stage7.reduce": ("stage7.evaluate",),
    "stage7.report": ("stage7.reduce",),
    "stage8.freeze": ("stage7.report",),
    "stage8.execute": ("stage8.freeze",),
    "stage8.reduce": ("stage8.execute",),
    "stage8.recommend": ("stage8.reduce",),
    "stage8.report": ("stage8.recommend",),
    "stage9.ingest": ("stage8.report",),
    "stage9.statistics": ("stage9.ingest",),
    "stage9.tables": ("stage9.statistics",),
    "stage9.charts": ("stage9.statistics",),
    "stage9.report": ("stage9.tables", "stage9.charts"),
    "stage9.bundle": ("stage9.report",),
    "stage9.replay": ("stage9.bundle",),
}


def _freeze_dependency_contracts(
    raw_tasks: Sequence[TaskDefinition],
) -> tuple[TaskDefinition, ...]:
    """补全 predecessor 与 input artifact 合同并返回新的不可变任务元组。

    Stage 0--3 的编号计划天然形成线性审计链：同 Stage 的 ``NN`` 消费 ``NN-1``，
    新 Stage 的 ``01`` 消费前一 Stage 的退出任务。Stage 4--9 则使用上面的显式
    DAG。输入合同绑定上游 ``task-output-commit-v1``，这与运行时只从权威 commit
    发现对象的两阶段发布协议一致。
    """

    numbered_by_stage: dict[int, list[tuple[int, str]]] = {stage: [] for stage in range(4)}
    for task in raw_tasks:
        if task.stage <= 3:
            match = re.match(r"^stage[0-3]\.(?P<number>[0-9]{2})_", task.task_id)
            if match is None:
                raise TaskCatalogError(f"Stage 0--3 task_id 缺少两位计划编号：{task.task_id}")
            numbered_by_stage[task.stage].append((int(match.group("number")), task.task_id))

    numbered_predecessors: dict[str, tuple[str, ...]] = {}
    previous_stage_exit: str | None = None
    for stage in range(4):
        entries = sorted(numbered_by_stage[stage])
        numbers = [number for number, _ in entries]
        if numbers != list(range(1, len(entries) + 1)):
            raise TaskCatalogError(f"Stage {stage} 编号任务必须从 01 连续递增")
        for index, (_, task_id) in enumerate(entries):
            if index > 0:
                predecessors = (entries[index - 1][1],)
            elif previous_stage_exit is not None:
                predecessors = (previous_stage_exit,)
            else:
                predecessors = ()
            numbered_predecessors[task_id] = predecessors
        previous_stage_exit = entries[-1][1]

    late_ids = {task.task_id for task in raw_tasks if task.stage >= 4}
    if late_ids != set(_LATE_STAGE_PREDECESSORS):
        missing = sorted(late_ids - set(_LATE_STAGE_PREDECESSORS))
        stale = sorted(set(_LATE_STAGE_PREDECESSORS) - late_ids)
        raise TaskCatalogError(
            f"Stage 4--9 predecessor 表与任务目录漂移：missing={missing}, stale={stale}"
        )

    frozen: list[TaskDefinition] = []
    raw_by_id = {task.task_id: task for task in raw_tasks}
    for task in raw_tasks:
        predecessors = (
            numbered_predecessors[task.task_id]
            if task.stage <= 3
            else _LATE_STAGE_PREDECESSORS[task.task_id]
        )
        inputs = tuple(
            InputArtifactContract(
                input_id=f"upstream_{index:02d}",
                schema_ref=TASK_OUTPUT_COMMIT_SCHEMA_REF,
                required=True,
                producer_task_ids=(predecessor_id,),
                artifact_kinds=raw_by_id[predecessor_id].artifact_kinds,
            )
            for index, predecessor_id in enumerate(predecessors, start=1)
        )
        frozen.append(
            replace(
                task,
                predecessor_task_ids=predecessors,
                input_artifacts=inputs,
                replay_policy=replace(
                    task.replay_policy,
                    requires_same_input_hashes=bool(inputs),
                ),
            )
        )
    return tuple(frozen)


_TASKS: Final = _freeze_dependency_contracts(_TASKS_RAW)


DEFAULT_TASK_CATALOG: Final = TaskCatalog(_TASKS)

# 公共语义名：强调该目录覆盖完整 Stage，而不是任意任务列表。保留 TaskCatalog
# 原名以兼容 0.3.x 调用方。
StageTaskCatalog = TaskCatalog


__all__ = [
    "DEFAULT_TASK_CATALOG",
    "RESOLVED_CONFIG_V2_SCHEMA_REF",
    "TASK_CATALOG_SCHEMA_VERSION",
    "TASK_DEFINITION_SCHEMA_VERSION",
    "TASK_OUTPUT_ARTIFACT_SCHEMA_REF",
    "TASK_OUTPUT_PAYLOAD_SCHEMA_REF",
    "TASK_OUTPUT_COMMIT_SCHEMA_REF",
    "ExistingOutputPolicy",
    "FormalEligibilityPolicy",
    "InputArtifactContract",
    "LocalFixturePolicy",
    "OutputArtifactContract",
    "RecoveryMode",
    "ReplayPolicy",
    "ReplayStrategy",
    "RunnerKind",
    "SafeBoundary",
    "StageTaskCatalog",
    "TaskCatalog",
    "TaskCatalogError",
    "TaskDefinition",
]
