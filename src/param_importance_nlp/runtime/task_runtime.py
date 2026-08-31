"""统一任务运行协议与结构化 ``BLOCKED`` 结果。

本模块不实现任何具体训练或服务器操作。它负责在进入 runner 之前检查 catalog 与
``ResolvedConfigV2`` 的绑定，区分“代码执行失败”和“外部前置条件不存在”：

* 缺 runner、服务器、GPU、资产、Gate、冻结合同或 estimator decision 返回
  ``status=BLOCKED``，并附稳定 blocker code；
* runner 已启动但异常退出返回 ``status=FAIL``；
* 只有满足全部 formal 前置条件、非 dry-run 且 runner 发布完整 artifact 集合时，
  结果才可以标记 ``formal_eligible=true``。

这样以后即使服务器仍不可达，也可以运行同一个 orchestration 入口获得可审计证据，
而不需要临时修改代码、吞掉 ImportError 或伪造 formal PASS。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
import re
from typing import Any, Protocol, runtime_checkable

from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.errors import DependencyUnavailable
from param_importance_nlp.contracts.freeze import ContractFreeze
from param_importance_nlp.contracts.jsonio import (
    JSONValue,
    canonical_json_bytes,
    canonical_json_hash,
)
from param_importance_nlp.contracts.runtime_evidence import RuntimeCapabilityEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.contracts.task_catalog import (
    DEFAULT_TASK_CATALOG,
    RecoveryMode,
    RunnerKind,
    TaskCatalog,
    TaskDefinition,
)
from param_importance_nlp.runtime.task_artifacts import (
    LoadedTaskArtifact,
    load_committed_task_artifact,
)


TASK_RUN_RESULT_SCHEMA_VERSION = "task-run-result-v2"
TASK_RUNTIME_ENVIRONMENT_SCHEMA_VERSION = "task-runtime-environment-v1"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class TaskRuntimeError(RuntimeError):
    """runner 注册或返回值违反统一运行协议。"""


class TaskRunStatus(str, Enum):
    """任务执行状态；与 formal Gate 状态分离。"""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class BlockerCode(str, Enum):
    """可由编排器稳定识别并决定是否重试的阻塞原因。"""

    RUNNER_UNAVAILABLE = "runner_unavailable"
    SERVER_UNREACHABLE = "server_unreachable"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ASSET_UNAVAILABLE = "asset_unavailable"
    DEVICE_UNAVAILABLE = "device_unavailable"
    GATE_NOT_READY = "gate_not_ready"
    CONTRACT_UNFROZEN = "contract_unfrozen"
    ESTIMATOR_DECISION_UNAVAILABLE = "estimator_decision_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    FORMAL_NOT_SUPPORTED = "formal_not_supported"
    FORMAL_DRY_RUN = "formal_dry_run"
    EXTERNAL_APPROVAL_REQUIRED = "external_approval_required"


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskRuntimeError(f"{field_name} 必须是非空字符串")
    return value


def _logical_ref(value: object, *, field_name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _non_empty(value, field_name=field_name)
    if "\\" in text:
        raise TaskRuntimeError(f"{field_name} 必须使用 POSIX 逻辑路径")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise TaskRuntimeError(f"{field_name} 发生路径逃逸")
    return str(path)


def _strict_string_tuple(
    value: Sequence[object],
    *,
    field_name: str,
    logical_refs: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskRuntimeError(f"{field_name} 必须是数组")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if logical_refs:
            parsed = _logical_ref(item, field_name=f"{field_name}[{index}]")
            assert parsed is not None
            normalized.append(parsed)
        else:
            normalized.append(_non_empty(item, field_name=f"{field_name}[{index}]"))
    if len(set(normalized)) != len(normalized):
        raise TaskRuntimeError(f"{field_name} 不能重复")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class TaskBlocker:
    """一个未满足前置条件；同一结果可以同时记录多个 blocker。"""

    code: BlockerCode
    requirement: str
    message: str
    retryable: bool
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, BlockerCode):
            raise TaskRuntimeError("TaskBlocker.code 类型错误")
        _non_empty(self.requirement, field_name="blocker.requirement")
        _non_empty(self.message, field_name="blocker.message")
        if not isinstance(self.retryable, bool):
            raise TaskRuntimeError("blocker.retryable 必须是 bool")
        refs = _strict_string_tuple(
            self.evidence_refs,
            field_name="blocker.evidence_refs",
            logical_refs=True,
        )
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "code": self.code.value,
            "requirement": self.requirement,
            "message": self.message,
            "retryable": self.retryable,
            "evidence_refs": list(self.evidence_refs),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskBlocker":
        expected = {"code", "requirement", "message", "retryable", "evidence_refs"}
        if set(value) != expected:
            raise TaskRuntimeError("TaskBlocker 字段集合不符合 v2 合同")
        try:
            code = BlockerCode(value["code"])
        except (TypeError, ValueError) as error:
            raise TaskRuntimeError("TaskBlocker.code 不受支持") from error
        return cls(
            code=code,
            requirement=value["requirement"],  # type: ignore[arg-type]
            message=value["message"],  # type: ignore[arg-type]
            retryable=value["retryable"],  # type: ignore[arg-type]
            evidence_refs=tuple(value["evidence_refs"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class TaskRuntimeEnvironment:
    """编排器已验证的外部能力和 formal 证据快照。

    所有字段只是已经验证的 ID，不在此处探测网络、GPU 或文件系统。探测器应先发布
    自己的 artifact，再把通过的能力填入本对象。
    """

    capabilities: frozenset[str] = frozenset()
    frozen_contract_stages: frozenset[int] = frozenset()
    passed_gate_ids: frozenset[str] = frozenset()
    estimator_decision_ref: str | None = None
    evidence_refs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        capabilities = frozenset(self.capabilities)
        if any(not isinstance(item, str) or _NAME_RE.fullmatch(item) is None for item in capabilities):
            raise TaskRuntimeError("capabilities 只能包含 snake_case 能力名")
        stages = frozenset(self.frozen_contract_stages)
        if any(isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9 for stage in stages):
            raise TaskRuntimeError("frozen_contract_stages 只能包含 0--9")
        gates = frozenset(self.passed_gate_ids)
        if any(not isinstance(gate, str) or not gate.startswith("stage") for gate in gates):
            raise TaskRuntimeError("passed_gate_ids 包含无效 ID")
        decision_ref = _logical_ref(
            self.estimator_decision_ref,
            field_name="estimator_decision_ref",
            optional=True,
        )
        if not isinstance(self.evidence_refs, Mapping):
            raise TaskRuntimeError("evidence_refs 必须是 object")
        refs: dict[str, str] = {}
        for key, value in self.evidence_refs.items():
            if not isinstance(key, str) or _NAME_RE.fullmatch(key) is None:
                raise TaskRuntimeError("evidence_refs 的键必须是 snake_case")
            parsed = _logical_ref(value, field_name=f"evidence_refs.{key}")
            assert parsed is not None
            refs[key] = parsed
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "frozen_contract_stages", stages)
        object.__setattr__(self, "passed_gate_ids", gates)
        object.__setattr__(self, "estimator_decision_ref", decision_ref)
        object.__setattr__(self, "evidence_refs", refs)

    def _payload(self) -> dict[str, JSONValue]:
        """返回不含自引用 hash 的稳定环境证据快照。

        capability 名只说明“上游探测器已经验证并发布证据”，本对象本身不会访问
        网络、探测 GPU 或检查资产。数组排序保证从不同集合构造时具有相同身份。
        """

        return {
            "schema_version": TASK_RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
            "capabilities": sorted(self.capabilities),
            "frozen_contract_stages": sorted(self.frozen_contract_stages),
            "passed_gate_ids": sorted(self.passed_gate_ids),
            "estimator_decision_ref": self.estimator_decision_ref,
            "evidence_refs": dict(sorted(self.evidence_refs.items())),
        }

    @property
    def environment_hash(self) -> str:
        """返回完整快照的 canonical SHA-256。"""

        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["environment_hash"] = self.environment_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskRuntimeEnvironment":
        """严格加载环境 artifact；未知字段和 hash 漂移立即失败。"""

        expected = {
            "schema_version",
            "capabilities",
            "frozen_contract_stages",
            "passed_gate_ids",
            "estimator_decision_ref",
            "evidence_refs",
            "environment_hash",
        }
        if set(value) != expected:
            raise TaskRuntimeError("TaskRuntimeEnvironment 字段集合无效")
        if value.get("schema_version") != TASK_RUNTIME_ENVIRONMENT_SCHEMA_VERSION:
            raise TaskRuntimeError("TaskRuntimeEnvironment.schema_version 无效")
        for field_name in (
            "capabilities",
            "frozen_contract_stages",
            "passed_gate_ids",
        ):
            member = value[field_name]
            if isinstance(member, (str, bytes)) or not isinstance(member, Sequence):
                raise TaskRuntimeError(f"environment.{field_name} 必须是数组")
        evidence = value["evidence_refs"]
        if not isinstance(evidence, Mapping):
            raise TaskRuntimeError("environment.evidence_refs 必须是 object")
        try:
            environment = cls(
                capabilities=frozenset(value["capabilities"]),  # type: ignore[arg-type]
                frozen_contract_stages=frozenset(  # type: ignore[arg-type]
                    value["frozen_contract_stages"]
                ),
                passed_gate_ids=frozenset(value["passed_gate_ids"]),  # type: ignore[arg-type]
                estimator_decision_ref=value["estimator_decision_ref"],  # type: ignore[arg-type]
                evidence_refs=evidence,  # type: ignore[arg-type]
            )
        except TypeError as error:
            raise TaskRuntimeError("TaskRuntimeEnvironment 数组包含不可哈希值") from error
        if value["environment_hash"] != environment.environment_hash:
            raise TaskRuntimeError("environment_hash 与环境内容不一致")
        return environment


@dataclass(frozen=True, slots=True)
class TaskRunSpec:
    """从 ResolvedConfigV2 派生的最小调度合同。

    它不复制训练超参数；调度器用它决定 runner、输入、输出与恢复边界，具体 runner
    仍读取完整配置。这样 status/replay 系统无需理解模型或统计配置。
    """

    task_id: str
    stage: int
    runner_kind: RunnerKind
    run_intent: str
    config_hash: str
    output_dir: str
    input_refs: tuple[str, ...]
    recovery_mode: RecoveryMode
    safe_boundary: str
    resume_ref: str | None

    @classmethod
    def from_config(cls, config: ResolvedConfigV2) -> "TaskRunSpec":
        task = config.task_definition
        artifacts = config.section("artifacts")
        orchestration = config.section("orchestration")
        recovery = config.section("recovery")
        assert isinstance(artifacts, dict)
        assert isinstance(orchestration, dict)
        assert isinstance(recovery, dict)
        return cls(
            task_id=task.task_id,
            stage=task.stage,
            runner_kind=task.runner_kind,
            run_intent=config.run_intent,
            config_hash=config.config_hash,
            output_dir=str(artifacts["output_dir"]),
            input_refs=tuple(str(item) for item in orchestration["input_result_refs"]),
            recovery_mode=task.recovery_mode,
            safe_boundary=str(recovery["safe_boundary"]),
            resume_ref=(
                None
                if recovery["resume_ref"] is None
                else str(recovery["resume_ref"])
            ),
        )

    def __post_init__(self) -> None:
        _non_empty(self.task_id, field_name="TaskRunSpec.task_id")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise TaskRuntimeError("TaskRunSpec.stage 必须是 0--9")
        if not isinstance(self.runner_kind, RunnerKind):
            raise TaskRuntimeError("TaskRunSpec.runner_kind 类型错误")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise TaskRuntimeError("TaskRunSpec.run_intent 非法")
        if not isinstance(self.config_hash, str) or re.fullmatch(r"[0-9a-f]{64}", self.config_hash) is None:
            raise TaskRuntimeError("TaskRunSpec.config_hash 非法")
        output = _logical_ref(self.output_dir, field_name="TaskRunSpec.output_dir")
        inputs = (
            ()
            if not self.input_refs
            else _strict_string_tuple(
                self.input_refs,
                field_name="TaskRunSpec.input_refs",
                logical_refs=True,
            )
        )
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise TaskRuntimeError("TaskRunSpec.recovery_mode 类型错误")
        _non_empty(self.safe_boundary, field_name="TaskRunSpec.safe_boundary")
        resume_ref = (
            None
            if self.resume_ref is None
            else _logical_ref(self.resume_ref, field_name="TaskRunSpec.resume_ref")
        )
        object.__setattr__(self, "output_dir", output)
        object.__setattr__(self, "input_refs", inputs)
        object.__setattr__(self, "resume_ref", resume_ref)

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": "task-run-spec-v1",
            "task_id": self.task_id,
            "stage": self.stage,
            "runner_kind": self.runner_kind.value,
            "run_intent": self.run_intent,
            "config_hash": self.config_hash,
            "output_dir": self.output_dir,
            "input_refs": list(self.input_refs),
            "recovery_mode": self.recovery_mode.value,
            "safe_boundary": self.safe_boundary,
            "resume_ref": self.resume_ref,
        }
        payload["spec_hash"] = canonical_json_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskRunSpec":
        expected = {
            "schema_version", "task_id", "stage", "runner_kind", "run_intent",
            "config_hash", "output_dir", "input_refs", "recovery_mode", "safe_boundary",
            "resume_ref",
            "spec_hash",
        }
        if set(value) != expected or value.get("schema_version") != "task-run-spec-v1":
            raise TaskRuntimeError("TaskRunSpec 字段或版本无效")
        raw_inputs = value["input_refs"]
        if not isinstance(raw_inputs, list):
            raise TaskRuntimeError("TaskRunSpec.input_refs 必须是数组")
        try:
            result = cls(
                task_id=value["task_id"],  # type: ignore[arg-type]
                stage=value["stage"],  # type: ignore[arg-type]
                runner_kind=RunnerKind(value["runner_kind"]),
                run_intent=value["run_intent"],  # type: ignore[arg-type]
                config_hash=value["config_hash"],  # type: ignore[arg-type]
                output_dir=value["output_dir"],  # type: ignore[arg-type]
                input_refs=tuple(raw_inputs),  # type: ignore[arg-type]
                recovery_mode=RecoveryMode(value["recovery_mode"]),
                safe_boundary=value["safe_boundary"],  # type: ignore[arg-type]
                resume_ref=value["resume_ref"],  # type: ignore[arg-type]
            )
        except ValueError as error:
            raise TaskRuntimeError("TaskRunSpec 枚举字段无效") from error
        if value["spec_hash"] != result.to_dict()["spec_hash"]:
            raise TaskRuntimeError("TaskRunSpec.spec_hash 不匹配")
        return result


@dataclass(frozen=True, slots=True)
class TaskExecutionRequest:
    """传给 runner 的只读执行请求。"""

    config: ResolvedConfigV2
    task: TaskDefinition
    environment: TaskRuntimeEnvironment

    def __post_init__(self) -> None:
        if self.config.task_id != self.task.task_id:
            raise TaskRuntimeError("TaskExecutionRequest 的 config/task 不一致")


@runtime_checkable
class TaskRunner(Protocol):
    """具体 runner 必须实现的最小同步协议。"""

    runner_kind: RunnerKind

    def run(self, request: TaskExecutionRequest) -> "TaskRunResult":
        """执行一个任务原子单元并返回 hash-bound 结果。"""


class TaskBlockedError(RuntimeError):
    """runner 在启动后发现可恢复前置条件缺失时使用的结构化异常。"""

    def __init__(self, *blockers: TaskBlocker) -> None:
        if not blockers:
            raise TaskRuntimeError("TaskBlockedError 至少需要一个 blocker")
        self.blockers = tuple(blockers)
        super().__init__("; ".join(blocker.message for blocker in blockers))


@dataclass(frozen=True, slots=True)
class TaskRunResult:
    """统一任务结果；``result_hash`` 绑定全部字段但不包含自身。"""

    task_id: str
    stage: int
    runner_kind: RunnerKind
    run_intent: str
    status: TaskRunStatus
    config_hash: str
    formal_eligible: bool
    artifact_refs: Mapping[str, str]
    checkpoint_ref: str | None
    blockers: tuple[TaskBlocker, ...]
    error_code: str | None
    message: str
    recovery_mode: RecoveryMode
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.task_id, field_name="task_id")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise TaskRuntimeError("stage 必须是 0--9")
        if not isinstance(self.runner_kind, RunnerKind):
            raise TaskRuntimeError("runner_kind 类型错误")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise TaskRuntimeError("run_intent 不受支持")
        if not isinstance(self.status, TaskRunStatus):
            raise TaskRuntimeError("status 类型错误")
        if not isinstance(self.config_hash, str) or re.fullmatch(r"[0-9a-f]{64}", self.config_hash) is None:
            raise TaskRuntimeError("config_hash 必须是小写 SHA-256")
        if not isinstance(self.formal_eligible, bool):
            raise TaskRuntimeError("formal_eligible 必须是 bool")
        if not isinstance(self.artifact_refs, Mapping):
            raise TaskRuntimeError("artifact_refs 必须是 object")
        artifact_refs: dict[str, str] = {}
        for kind, reference in self.artifact_refs.items():
            if not isinstance(kind, str) or _NAME_RE.fullmatch(kind) is None:
                raise TaskRuntimeError("artifact_refs 的键必须是 artifact kind")
            parsed = _logical_ref(reference, field_name=f"artifact_refs.{kind}")
            assert parsed is not None
            artifact_refs[kind] = parsed
        checkpoint_ref = _logical_ref(
            self.checkpoint_ref,
            field_name="checkpoint_ref",
            optional=True,
        )
        blockers = tuple(self.blockers)
        if any(not isinstance(blocker, TaskBlocker) for blocker in blockers):
            raise TaskRuntimeError("blockers 只能包含 TaskBlocker")
        if self.error_code is not None:
            _non_empty(self.error_code, field_name="error_code")
        _non_empty(self.message, field_name="message")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise TaskRuntimeError("recovery_mode 类型错误")
        if not isinstance(self.metadata, Mapping):
            raise TaskRuntimeError("metadata 必须是 object")
        metadata = dict(self.metadata)
        try:
            canonical_json_bytes(metadata)
        except ValueError as error:
            raise TaskRuntimeError("metadata 不是严格 JSON object") from error

        if self.status is TaskRunStatus.PASS:
            if blockers or self.error_code is not None:
                raise TaskRuntimeError("PASS 不能携带 blockers/error_code")
        elif self.status is TaskRunStatus.BLOCKED:
            if not blockers or self.error_code is not None:
                raise TaskRuntimeError("BLOCKED 必须携带 blocker 且不能携带 error_code")
        elif self.status is TaskRunStatus.FAIL:
            if blockers or self.error_code is None:
                raise TaskRuntimeError("FAIL 必须携带 error_code 且不能携带 blocker")
        else:  # SKIPPED
            if blockers or self.error_code is not None:
                raise TaskRuntimeError("SKIPPED 不能携带 blockers/error_code")
        if self.formal_eligible and not (
            self.status is TaskRunStatus.PASS and self.run_intent == "formal"
        ):
            raise TaskRuntimeError("只有 formal PASS 才能 formal_eligible=true")

        object.__setattr__(self, "artifact_refs", artifact_refs)
        object.__setattr__(self, "checkpoint_ref", checkpoint_ref)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_RUN_RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "stage": self.stage,
            "runner_kind": self.runner_kind.value,
            "run_intent": self.run_intent,
            "status": self.status.value,
            "config_hash": self.config_hash,
            "formal_eligible": self.formal_eligible,
            "artifact_refs": dict(self.artifact_refs),
            "checkpoint_ref": self.checkpoint_ref,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
            "error_code": self.error_code,
            "message": self.message,
            "recovery_mode": self.recovery_mode.value,
            "metadata": dict(self.metadata),
        }

    @property
    def result_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        result = self._payload()
        result["result_hash"] = self.result_hash
        return result

    @classmethod
    def passed(
        cls,
        request: TaskExecutionRequest,
        *,
        artifact_refs: Mapping[str, str],
        checkpoint_ref: str | None = None,
        message: str = "task completed",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> "TaskRunResult":
        execution = request.config.section("execution")
        assert isinstance(execution, dict)
        formal_eligible = request.config.run_intent == "formal" and not bool(execution["dry_run"])
        return cls(
            task_id=request.task.task_id,
            stage=request.task.stage,
            runner_kind=request.task.runner_kind,
            run_intent=request.config.run_intent,
            status=TaskRunStatus.PASS,
            config_hash=request.config.config_hash,
            formal_eligible=formal_eligible,
            artifact_refs=artifact_refs,
            checkpoint_ref=checkpoint_ref,
            blockers=(),
            error_code=None,
            message=message,
            recovery_mode=request.task.recovery_mode,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def blocked(
        cls,
        config: ResolvedConfigV2,
        task: TaskDefinition,
        blockers: Sequence[TaskBlocker],
        *,
        message: str = "task prerequisites are blocked",
    ) -> "TaskRunResult":
        return cls(
            task_id=task.task_id,
            stage=task.stage,
            runner_kind=task.runner_kind,
            run_intent=config.run_intent,
            status=TaskRunStatus.BLOCKED,
            config_hash=config.config_hash,
            formal_eligible=False,
            artifact_refs={},
            checkpoint_ref=None,
            blockers=tuple(blockers),
            error_code=None,
            message=message,
            recovery_mode=task.recovery_mode,
            metadata={},
        )

    @classmethod
    def failed(
        cls,
        config: ResolvedConfigV2,
        task: TaskDefinition,
        *,
        error_code: str,
        message: str,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> "TaskRunResult":
        return cls(
            task_id=task.task_id,
            stage=task.stage,
            runner_kind=task.runner_kind,
            run_intent=config.run_intent,
            status=TaskRunStatus.FAIL,
            config_hash=config.config_hash,
            formal_eligible=False,
            artifact_refs={},
            checkpoint_ref=None,
            blockers=(),
            error_code=error_code,
            message=message,
            recovery_mode=task.recovery_mode,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskRunResult":
        expected = {
            "schema_version", "task_id", "stage", "runner_kind", "run_intent", "status",
            "config_hash", "formal_eligible", "artifact_refs", "checkpoint_ref", "blockers",
            "error_code", "message", "recovery_mode", "metadata", "result_hash",
        }
        if set(value) != expected or value.get("schema_version") != TASK_RUN_RESULT_SCHEMA_VERSION:
            raise TaskRuntimeError("TaskRunResult 字段集合或 schema_version 无效")
        raw_blockers = value["blockers"]
        if isinstance(raw_blockers, (str, bytes)) or not isinstance(raw_blockers, Sequence):
            raise TaskRuntimeError("blockers 必须是数组")
        blockers = []
        for index, item in enumerate(raw_blockers):
            if not isinstance(item, Mapping):
                raise TaskRuntimeError(f"blockers[{index}] 必须是 object")
            blockers.append(TaskBlocker.from_mapping(item))
        try:
            result = cls(
                task_id=value["task_id"],  # type: ignore[arg-type]
                stage=value["stage"],  # type: ignore[arg-type]
                runner_kind=RunnerKind(value["runner_kind"]),
                run_intent=value["run_intent"],  # type: ignore[arg-type]
                status=TaskRunStatus(value["status"]),
                config_hash=value["config_hash"],  # type: ignore[arg-type]
                formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
                artifact_refs=value["artifact_refs"],  # type: ignore[arg-type]
                checkpoint_ref=value["checkpoint_ref"],  # type: ignore[arg-type]
                blockers=tuple(blockers),
                error_code=value["error_code"],  # type: ignore[arg-type]
                message=value["message"],  # type: ignore[arg-type]
                recovery_mode=RecoveryMode(value["recovery_mode"]),
                metadata=value["metadata"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            raise TaskRuntimeError("TaskRunResult 枚举字段无效") from error
        if value["result_hash"] != result.result_hash:
            raise TaskRuntimeError("result_hash 与 TaskRunResult 内容不一致")
        return result


def _capability_blocker(capability: str, evidence_refs: Mapping[str, str]) -> TaskBlocker:
    evidence = tuple([evidence_refs[capability]]) if capability in evidence_refs else ()
    if capability == "server":
        return TaskBlocker(BlockerCode.SERVER_UNREACHABLE, capability, "服务器当前不可达", True, evidence)
    if capability in {"cuda", "nccl"}:
        return TaskBlocker(BlockerCode.DEVICE_UNAVAILABLE, capability, f"运行设备能力不可用：{capability}", True, evidence)
    if capability in {"model_assets", "data_assets"}:
        return TaskBlocker(BlockerCode.ASSET_UNAVAILABLE, capability, f"真实资产未 ready：{capability}", True, evidence)
    if capability == "wheelhouse":
        return TaskBlocker(BlockerCode.DEPENDENCY_UNAVAILABLE, capability, "离线 wheelhouse 不可用", True, evidence)
    if capability in {"github"}:
        return TaskBlocker(BlockerCode.EXTERNAL_APPROVAL_REQUIRED, capability, f"外部能力或授权不可用：{capability}", True, evidence)
    return TaskBlocker(BlockerCode.CAPABILITY_UNAVAILABLE, capability, f"运行能力不可用：{capability}", True, evidence)


class TaskRuntime:
    """按 ``RunnerKind`` 注册 runner，并统一执行 preflight 与结果审计。"""

    def __init__(
        self,
        *,
        catalog: TaskCatalog = DEFAULT_TASK_CATALOG,
        workspace_root: str | Path | None = None,
    ) -> None:
        self._catalog = catalog
        self._runners: dict[RunnerKind, TaskRunner] = {}
        self._workspace_root = (
            Path.cwd().resolve()
            if workspace_root is None
            else Path(workspace_root).resolve()
        )

    def _load_environment_evidence(self, reference: str) -> LoadedTaskArtifact:
        """读取 workspace 内的权威 formal commit，不接受直接 object/裸 JSON。

        证据链在这里统一经过 commit、内容寻址 envelope、对象 hash 与 formal scope
        验证。上层的合同、Gate、capability 和 estimator 解析器只负责各自语义，
        从而不会因某一分支忘记检查 envelope 而把本机 fixture 升格。
        """

        try:
            return load_committed_task_artifact(
                self._workspace_root,
                reference,
                require_formal=True,
            )
        except Exception as error:
            raise TaskRuntimeError(
                f"FORMAL_EVIDENCE_COMMIT_INVALID:{type(error).__name__}:{error}"
            ) from error

    @staticmethod
    def _extract_schema_payload(
        value: Mapping[str, object],
        schema_version: str,
    ) -> Mapping[str, object]:
        """提取精确或唯一嵌套的 schema payload，拒绝歧义候选。"""

        if value.get("schema_version") == schema_version:
            return value
        candidates = [
            item
            for item in value.values()
            if isinstance(item, Mapping)
            and item.get("schema_version") == schema_version
        ]
        if len(candidates) != 1:
            raise TaskRuntimeError(
                f"FORMAL_EVIDENCE_PAYLOAD_NOT_UNIQUE:{schema_version}"
            )
        return candidates[0]

    @staticmethod
    def _gate_evidence_key(gate_id: str) -> str:
        return "gate_" + re.sub(r"[^a-z0-9]+", "_", gate_id.casefold()).strip("_")

    def _verified_contract_ref(
        self,
        environment: TaskRuntimeEnvironment,
        stage: int,
    ) -> tuple[bool, tuple[str, ...]]:
        key = f"contract_stage_{stage}"
        reference = environment.evidence_refs.get(key)
        if reference is None:
            return False, ()
        try:
            loaded = self._load_environment_evidence(reference)
            value = self._extract_schema_payload(
                loaded.payload,
                "contract-freeze-v1",
            )
            freeze = ContractFreeze.from_mapping(dict(value))
        except Exception:
            return False, (reference,)
        return freeze.stage == stage and freeze.formal_eligible, (reference,)

    def _verified_gate_ref(
        self,
        environment: TaskRuntimeEnvironment,
        gate_id: str,
    ) -> tuple[bool, tuple[str, ...]]:
        reference = environment.evidence_refs.get(self._gate_evidence_key(gate_id))
        if reference is None:
            return False, ()
        try:
            loaded = self._load_environment_evidence(reference)
            value = self._extract_schema_payload(
                loaded.payload,
                "gate-record-v1",
            )
            gate = GateRecord.from_mapping(dict(value))
        except Exception:
            return False, (reference,)
        return gate.gate_id == gate_id and gate.status is GateStatus.PASS, (reference,)

    def _verified_capability_ref(
        self,
        environment: TaskRuntimeEnvironment,
        capability: str,
    ) -> tuple[bool, tuple[str, ...]]:
        reference = environment.evidence_refs.get(f"capability_{capability}")
        if reference is None:
            return False, ()
        try:
            loaded = self._load_environment_evidence(reference)
            value = self._extract_schema_payload(
                loaded.payload,
                "runtime-capability-evidence-v1",
            )
            evidence = RuntimeCapabilityEvidence.from_mapping(value)
        except Exception:
            return False, (reference,)
        return evidence.capability == capability and evidence.verified, (reference,)

    def register(self, runner: TaskRunner) -> None:
        kind = getattr(runner, "runner_kind", None)
        if not isinstance(kind, RunnerKind):
            raise TaskRuntimeError("runner.runner_kind 必须是 RunnerKind")
        if kind in self._runners:
            raise TaskRuntimeError(f"RunnerKind {kind.value!r} 已注册，禁止静默覆盖")
        if not callable(getattr(runner, "run", None)):
            raise TaskRuntimeError("runner 必须实现 run(request)")
        self._runners[kind] = runner

    @property
    def registered_kinds(self) -> tuple[RunnerKind, ...]:
        return tuple(sorted(self._runners, key=lambda item: item.value))

    @property
    def catalog(self) -> TaskCatalog:
        """返回构造时绑定的只读任务目录。"""

        return self._catalog

    @property
    def workspace_root(self) -> Path:
        """返回 formal 证据解析所绑定的绝对 workspace 根目录。"""

        return self._workspace_root

    def _formal_input_blockers(
        self,
        config: ResolvedConfigV2,
        task: TaskDefinition,
    ) -> tuple[TaskBlocker, ...]:
        """验证 catalog required inputs 的 formal commit 身份与完整集合。"""

        orchestration = config.section("orchestration")
        assert isinstance(orchestration, dict)
        references = tuple(str(item) for item in orchestration["input_result_refs"])
        blockers: list[TaskBlocker] = []
        loaded_by_identity: dict[tuple[str, str], str] = {}
        contracts = tuple(task.input_artifacts)
        for reference in references:
            try:
                loaded = self._load_environment_evidence(reference)
            except Exception:
                blockers.append(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_commit",
                        f"正式输入 commit 缺失、损坏或不是 formal envelope：{reference}",
                        True,
                        (reference,),
                    )
                )
                continue
            identity = loaded.identity
            matched = any(
                identity.task_id in contract.producer_task_ids
                and identity.artifact_kind in contract.artifact_kinds
                for contract in contracts
            )
            if not matched:
                blockers.append(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_contract",
                        (
                            "正式输入 commit 的 producer/artifact kind 不属于任务目录："
                            f"{identity.task_id}:{identity.artifact_kind}"
                        ),
                        False,
                        (reference,),
                    )
                )
                continue
            key = (identity.task_id, identity.artifact_kind)
            if key in loaded_by_identity:
                blockers.append(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_duplicate",
                        f"正式输入身份重复：{identity.task_id}:{identity.artifact_kind}",
                        False,
                        (loaded_by_identity[key], reference),
                    )
                )
            else:
                loaded_by_identity[key] = reference

        for contract in contracts:
            if not contract.required:
                continue
            for artifact_kind in contract.artifact_kinds:
                candidates = [
                    reference
                    for (producer, kind), reference in loaded_by_identity.items()
                    if producer in contract.producer_task_ids and kind == artifact_kind
                ]
                if not candidates:
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.ASSET_UNAVAILABLE,
                            f"input:{contract.input_id}:{artifact_kind}",
                            (
                                "缺少 catalog 要求的正式上游 commit："
                                f"{contract.input_id}/{artifact_kind}"
                            ),
                            True,
                        )
                    )
        return tuple(blockers)

    def _preflight(
        self,
        config: ResolvedConfigV2,
        task: TaskDefinition,
        environment: TaskRuntimeEnvironment,
    ) -> tuple[TaskBlocker, ...]:
        blockers: list[TaskBlocker] = []
        if task.runner_kind not in self._runners:
            blockers.append(
                TaskBlocker(
                    BlockerCode.RUNNER_UNAVAILABLE,
                    task.runner_kind.value,
                    f"尚未注册 {task.runner_kind.value} runner",
                    False,
                )
            )
        if config.run_intent == "formal":
            policy = task.formal_eligibility
            if not policy.supported:
                blockers.append(
                    TaskBlocker(
                        BlockerCode.FORMAL_NOT_SUPPORTED,
                        task.task_id,
                        "任务目录声明该任务不支持 formal 运行",
                        False,
                    )
                )
            execution = config.section("execution")
            assert isinstance(execution, dict)
            if execution["dry_run"]:
                blockers.append(
                    TaskBlocker(
                        BlockerCode.FORMAL_DRY_RUN,
                        "execution.dry_run",
                        "dry-run 不能发布 formal 合格结果",
                        True,
                    )
                )
            blockers.extend(self._formal_input_blockers(config, task))
            for stage in policy.required_contract_stages:
                claimed = stage in environment.frozen_contract_stages
                verified, evidence = self._verified_contract_ref(environment, stage)
                if not claimed or not verified:
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.CONTRACT_UNFROZEN,
                            f"stage{stage}",
                            (
                                f"Stage {stage} 合同尚未冻结或已漂移"
                                if not claimed
                                else f"Stage {stage} 缺少可核验的 formal ContractFreeze"
                            ),
                            True,
                            evidence,
                        )
                    )
            for gate_id in policy.required_gate_ids:
                claimed = gate_id in environment.passed_gate_ids
                verified, evidence = self._verified_gate_ref(environment, gate_id)
                if not claimed or not verified:
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.GATE_NOT_READY,
                            gate_id,
                            (
                                f"前置 Gate 未通过：{gate_id}"
                                if not claimed
                                else f"前置 Gate 缺少匹配且状态为 PASS 的 GateRecord：{gate_id}"
                            ),
                            True,
                            evidence,
                        )
                    )
            for capability in policy.required_capabilities:
                claimed = capability in environment.capabilities
                verified, evidence = self._verified_capability_ref(
                    environment, capability
                )
                if not claimed or not verified:
                    blocker = _capability_blocker(
                        capability,
                        (
                            environment.evidence_refs
                            if not evidence
                            else {capability: evidence[0]}
                        ),
                    )
                    if claimed and not verified:
                        blocker = TaskBlocker(
                            blocker.code,
                            blocker.requirement,
                            f"声明的能力缺少 VERIFIED formal evidence：{capability}",
                            blocker.retryable,
                            evidence,
                        )
                    blockers.append(blocker)
            if policy.requires_estimator_decision:
                decision_ref = environment.estimator_decision_ref
                decision_valid = False
                if decision_ref is not None:
                    try:
                        from ..experiments.stage2 import EstimatorDecision

                        loaded = self._load_environment_evidence(decision_ref)
                        value = self._extract_schema_payload(
                            loaded.payload,
                            "estimator-decision-v1",
                        )
                        decision = EstimatorDecision.from_mapping(value)
                        decision.require_formal()
                    except Exception:
                        decision_valid = False
                    else:
                        decision_valid = True
                if not decision_valid:
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
                            "estimator_decision",
                            "formal 任务缺少已通过 Gate 且 hash-bound 的 EstimatorDecision",
                            True,
                            (() if decision_ref is None else (decision_ref,)),
                        )
                    )
        return tuple(
            sorted(blockers, key=lambda item: (item.code.value, item.requirement, item.message))
        )

    def preflight(
        self,
        config: ResolvedConfigV2,
        *,
        environment: TaskRuntimeEnvironment | None = None,
    ) -> tuple[TaskBlocker, ...]:
        """公开、只读地执行与 :meth:`execute` 完全相同的前置检查。

        返回空 tuple 表示当前快照已具备进入 runner 的条件，并不代表任务、Gate 或
        科学结果已经通过。该方法不调用 runner，也不触碰网络、设备或输出目录。
        """

        if not isinstance(config, ResolvedConfigV2):
            raise TypeError("config 必须是 ResolvedConfigV2")
        task = self._catalog.get(config.task_id)
        snapshot = TaskRuntimeEnvironment() if environment is None else environment
        if not isinstance(snapshot, TaskRuntimeEnvironment):
            raise TypeError("environment 必须是 TaskRuntimeEnvironment")
        return self._preflight(config, task, snapshot)

    def _validate_result(
        self,
        result: TaskRunResult,
        request: TaskExecutionRequest,
    ) -> None:
        task = request.task
        config = request.config
        if (
            result.task_id != task.task_id
            or result.stage != task.stage
            or result.runner_kind is not task.runner_kind
            or result.run_intent != config.run_intent
            or result.config_hash != config.config_hash
            or result.recovery_mode is not task.recovery_mode
        ):
            raise TaskRuntimeError("runner 返回结果与执行请求身份不一致")
        if result.status is TaskRunStatus.PASS:
            if tuple(result.artifact_refs) != task.artifact_kinds:
                raise TaskRuntimeError(
                    "PASS 必须按任务目录顺序发布完整 artifact 集合"
                )
            execution = config.section("execution")
            assert isinstance(execution, dict)
            expected_formal = config.run_intent == "formal" and not bool(execution["dry_run"])
            if result.formal_eligible != expected_formal:
                raise TaskRuntimeError("PASS 的 formal_eligible 与 preflight 语义不一致")

    def execute(
        self,
        config: ResolvedConfigV2,
        *,
        environment: TaskRuntimeEnvironment | None = None,
    ) -> TaskRunResult:
        """执行一个 v2 任务；缺失外部条件始终返回结构化 ``BLOCKED``。"""

        if not isinstance(config, ResolvedConfigV2):
            raise TypeError("config 必须是 ResolvedConfigV2")
        task = self._catalog.get(config.task_id)
        snapshot = TaskRuntimeEnvironment() if environment is None else environment
        blockers = self.preflight(config, environment=snapshot)
        if blockers:
            return TaskRunResult.blocked(config, task, blockers)

        request = TaskExecutionRequest(config=config, task=task, environment=snapshot)
        runner = self._runners[task.runner_kind]
        try:
            result = runner.run(request)
            if not isinstance(result, TaskRunResult):
                raise TaskRuntimeError("runner.run 必须返回 TaskRunResult")
            self._validate_result(result, request)
            return result
        except TaskBlockedError as error:
            return TaskRunResult.blocked(config, task, error.blockers, message=str(error))
        except DependencyUnavailable as error:
            blocker = TaskBlocker(
                BlockerCode.DEPENDENCY_UNAVAILABLE,
                error.dependency,
                str(error),
                True,
            )
            return TaskRunResult.blocked(config, task, (blocker,))
        except Exception as error:  # runner bug/输入外的实际失败；KeyboardInterrupt 不吞掉
            return TaskRunResult.failed(
                config,
                task,
                error_code=(
                    "runner_contract_violation"
                    if isinstance(error, TaskRuntimeError)
                    else "runner_exception"
                ),
                message=f"{type(error).__name__}: {error}",
                metadata={"exception_type": type(error).__name__},
            )


__all__ = [
    "TASK_RUN_RESULT_SCHEMA_VERSION",
    "TASK_RUNTIME_ENVIRONMENT_SCHEMA_VERSION",
    "BlockerCode",
    "TaskBlockedError",
    "TaskBlocker",
    "TaskExecutionRequest",
    "TaskRunResult",
    "TaskRunSpec",
    "TaskRunStatus",
    "TaskRunner",
    "TaskRuntime",
    "TaskRuntimeEnvironment",
    "TaskRuntimeError",
]

# 0.4 公共语义别名；保留 TaskRunResult 以兼容已有代码。
TaskResult = TaskRunResult
__all__.append("TaskResult")
