"""任务结果的只读状态、确定性重放和最终确认合同。

任务 runner 只负责产生 :class:`~param_importance_nlp.runtime.task_runtime.TaskRunResult`；
本模块负责 runner 之外的三个生命周期动作：

``status``
    把一个已经通过严格 loader 校验的结果投影成稳定、便于调度器读取的状态快照。
``replay``
    比较原结果与一次重新执行的结果。比较包含完整 ``result_hash``，同时单独列出
    status、artifact 引用和 formal 资格，避免只比较退出码而漏掉结果漂移。
``finalize``
    仅允许确认 ``PASS`` 结果。确认记录绑定原结果 hash 和全部 artifact 逻辑引用；
    它不把本机 fixture 升格为 formal，也不宣称自己重新校验了 artifact 内容。

这些对象都不保存机器绝对路径。文件路径只是 CLI 的定位信息，进入合同前必须转换
成仓库/输出根下的 POSIX 逻辑引用。发布函数使用“临时文件 + 同目录硬链接”实现
不可变发布：目标已存在且内容相同视为幂等，内容不同则 fail-closed，绝不覆盖。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Final

from ..contracts.jsonio import (
    JSONValue,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
)
from ..contracts.task_catalog import RecoveryMode, RunnerKind
from .task_runtime import TaskRunResult, TaskRunStatus


TASK_STATUS_SCHEMA_VERSION: Final = "task-status-snapshot-v1"
TASK_REPLAY_SCHEMA_VERSION: Final = "task-replay-record-v1"
TASK_FINALIZATION_SCHEMA_VERSION: Final = "task-finalization-record-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TaskLifecycleError(ValueError):
    """任务生命周期 artifact 字段、hash 或发布目标无效。"""


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise TaskLifecycleError(f"{field_name} 必须是小写 SHA-256")
    return value


def logical_reference(value: str, *, field_name: str = "reference") -> str:
    """验证平台无关逻辑引用，拒绝绝对路径、反斜杠和 ``..`` 逃逸。"""

    if not isinstance(value, str) or not value:
        raise TaskLifecycleError(f"{field_name} 必须是非空字符串")
    if "\\" in value or ":" in value:
        raise TaskLifecycleError(f"{field_name} 必须是 POSIX 逻辑引用")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TaskLifecycleError(f"{field_name} 发生路径逃逸")
    return str(path)


def _result_identity(result: TaskRunResult) -> dict[str, JSONValue]:
    return {
        "task_id": result.task_id,
        "stage": result.stage,
        "runner_kind": result.runner_kind.value,
        "run_intent": result.run_intent,
        "config_hash": result.config_hash,
        "recovery_mode": result.recovery_mode.value,
    }


@dataclass(frozen=True, slots=True)
class TaskStatusSnapshot:
    """一个 TaskRunResult 的纯派生、hash-bound 状态视图。"""

    result_ref: str
    result_hash: str
    task_id: str
    stage: int
    runner_kind: RunnerKind
    run_intent: str
    config_hash: str
    recovery_mode: RecoveryMode
    status: TaskRunStatus
    formal_eligible: bool
    artifact_count: int
    blocker_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_ref",
            logical_reference(self.result_ref, field_name="result_ref"),
        )
        _require_hash(self.result_hash, field_name="result_hash")
        _require_hash(self.config_hash, field_name="config_hash")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TaskLifecycleError("task_id 必须是非空字符串")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise TaskLifecycleError("stage 必须是 0--9")
        if not isinstance(self.runner_kind, RunnerKind):
            raise TaskLifecycleError("runner_kind 类型无效")
        if not isinstance(self.recovery_mode, RecoveryMode):
            raise TaskLifecycleError("recovery_mode 类型无效")
        if not isinstance(self.status, TaskRunStatus):
            raise TaskLifecycleError("status 类型无效")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise TaskLifecycleError("run_intent 类型无效")
        if not isinstance(self.formal_eligible, bool):
            raise TaskLifecycleError("formal_eligible 必须是 bool")
        if self.formal_eligible and not (
            self.run_intent == "formal" and self.status is TaskRunStatus.PASS
        ):
            raise TaskLifecycleError("只有 formal PASS 可以 formal_eligible=true")
        if (
            isinstance(self.artifact_count, bool)
            or not isinstance(self.artifact_count, int)
            or self.artifact_count < 0
        ):
            raise TaskLifecycleError("artifact_count 必须是非负整数")
        codes = tuple(self.blocker_codes)
        if any(not isinstance(code, str) or not code for code in codes):
            raise TaskLifecycleError("blocker_codes 只能包含非空字符串")
        if len(set(codes)) != len(codes):
            raise TaskLifecycleError("blocker_codes 不得重复")
        object.__setattr__(self, "blocker_codes", codes)

    @classmethod
    def from_result(
        cls,
        result: TaskRunResult,
        *,
        result_ref: str,
    ) -> "TaskStatusSnapshot":
        return cls(
            result_ref=result_ref,
            result_hash=result.result_hash,
            task_id=result.task_id,
            stage=result.stage,
            runner_kind=result.runner_kind,
            run_intent=result.run_intent,
            config_hash=result.config_hash,
            recovery_mode=result.recovery_mode,
            status=result.status,
            formal_eligible=result.formal_eligible,
            artifact_count=len(result.artifact_refs),
            # 一个 formal preflight 可能同时缺少多个 stage 合同或 Gate；这些
            # blocker 的 code 相同但 requirement 不同。状态快照只呈现 code 集合，
            # 详细 requirement 仍保留在源 TaskRunResult 中。
            blocker_codes=tuple(sorted({blocker.code.value for blocker in result.blockers})),
        )

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_STATUS_SCHEMA_VERSION,
            "result_ref": self.result_ref,
            "result_hash": self.result_hash,
            **_status_identity(self),
            "status": self.status.value,
            "formal_eligible": self.formal_eligible,
            "artifact_count": self.artifact_count,
            "blocker_codes": list(self.blocker_codes),
        }

    @property
    def snapshot_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["snapshot_hash"] = self.snapshot_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskStatusSnapshot":
        expected = {
            "schema_version",
            "result_ref",
            "result_hash",
            "task_id",
            "stage",
            "runner_kind",
            "run_intent",
            "config_hash",
            "recovery_mode",
            "status",
            "formal_eligible",
            "artifact_count",
            "blocker_codes",
            "snapshot_hash",
        }
        if set(value) != expected or value.get("schema_version") != TASK_STATUS_SCHEMA_VERSION:
            raise TaskLifecycleError("TaskStatusSnapshot 字段集合或 schema_version 无效")
        codes = value["blocker_codes"]
        if isinstance(codes, (str, bytes)) or not isinstance(codes, list):
            raise TaskLifecycleError("blocker_codes 必须是数组")
        try:
            snapshot = cls(
                result_ref=value["result_ref"],  # type: ignore[arg-type]
                result_hash=value["result_hash"],  # type: ignore[arg-type]
                task_id=value["task_id"],  # type: ignore[arg-type]
                stage=value["stage"],  # type: ignore[arg-type]
                runner_kind=RunnerKind(value["runner_kind"]),
                run_intent=value["run_intent"],  # type: ignore[arg-type]
                config_hash=value["config_hash"],  # type: ignore[arg-type]
                recovery_mode=RecoveryMode(value["recovery_mode"]),
                status=TaskRunStatus(value["status"]),
                formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
                artifact_count=value["artifact_count"],  # type: ignore[arg-type]
                blocker_codes=tuple(codes),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as error:
            raise TaskLifecycleError("TaskStatusSnapshot 枚举字段无效") from error
        if value["snapshot_hash"] != snapshot.snapshot_hash:
            raise TaskLifecycleError("snapshot_hash 与内容不一致")
        return snapshot


def _status_identity(snapshot: TaskStatusSnapshot) -> dict[str, JSONValue]:
    return {
        "task_id": snapshot.task_id,
        "stage": snapshot.stage,
        "runner_kind": snapshot.runner_kind.value,
        "run_intent": snapshot.run_intent,
        "config_hash": snapshot.config_hash,
        "recovery_mode": snapshot.recovery_mode.value,
    }


@dataclass(frozen=True, slots=True)
class TaskReplayRecord:
    """原结果与重放结果的完整等价性比较。"""

    source_result_ref: str
    source_result_hash: str
    replay_result_ref: str
    replay_result_hash: str
    task_id: str
    config_hash: str
    status_match: bool
    artifact_refs_match: bool
    formal_eligibility_match: bool
    equivalent: bool

    def __post_init__(self) -> None:
        for field_name in ("source_result_ref", "replay_result_ref"):
            object.__setattr__(
                self,
                field_name,
                logical_reference(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "source_result_hash",
            "replay_result_hash",
            "config_hash",
        ):
            _require_hash(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TaskLifecycleError("task_id 必须是非空字符串")
        flags = (
            self.status_match,
            self.artifact_refs_match,
            self.formal_eligibility_match,
            self.equivalent,
        )
        if any(not isinstance(flag, bool) for flag in flags):
            raise TaskLifecycleError("replay 比较字段必须是 bool")
        expected = (
            self.source_result_hash == self.replay_result_hash
            and self.status_match
            and self.artifact_refs_match
            and self.formal_eligibility_match
        )
        if self.equivalent is not expected:
            raise TaskLifecycleError("equivalent 与重放比较字段不一致")

    @classmethod
    def compare(
        cls,
        source: TaskRunResult,
        replay: TaskRunResult,
        *,
        source_result_ref: str,
        replay_result_ref: str,
    ) -> "TaskReplayRecord":
        if _result_identity(source) != _result_identity(replay):
            raise TaskLifecycleError("重放结果与源结果的任务/config 身份不一致")
        status_match = source.status is replay.status
        artifact_refs_match = dict(source.artifact_refs) == dict(replay.artifact_refs)
        formal_match = source.formal_eligible is replay.formal_eligible
        hashes_match = source.result_hash == replay.result_hash
        return cls(
            source_result_ref=source_result_ref,
            source_result_hash=source.result_hash,
            replay_result_ref=replay_result_ref,
            replay_result_hash=replay.result_hash,
            task_id=source.task_id,
            config_hash=source.config_hash,
            status_match=status_match,
            artifact_refs_match=artifact_refs_match,
            formal_eligibility_match=formal_match,
            equivalent=(
                hashes_match and status_match and artifact_refs_match and formal_match
            ),
        )

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_REPLAY_SCHEMA_VERSION,
            "source_result_ref": self.source_result_ref,
            "source_result_hash": self.source_result_hash,
            "replay_result_ref": self.replay_result_ref,
            "replay_result_hash": self.replay_result_hash,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "status_match": self.status_match,
            "artifact_refs_match": self.artifact_refs_match,
            "formal_eligibility_match": self.formal_eligibility_match,
            "equivalent": self.equivalent,
        }

    @property
    def replay_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["replay_hash"] = self.replay_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskReplayRecord":
        expected = {
            "schema_version",
            "source_result_ref",
            "source_result_hash",
            "replay_result_ref",
            "replay_result_hash",
            "task_id",
            "config_hash",
            "status_match",
            "artifact_refs_match",
            "formal_eligibility_match",
            "equivalent",
            "replay_hash",
        }
        if set(value) != expected or value.get("schema_version") != TASK_REPLAY_SCHEMA_VERSION:
            raise TaskLifecycleError("TaskReplayRecord 字段集合或 schema_version 无效")
        record = cls(
            source_result_ref=value["source_result_ref"],  # type: ignore[arg-type]
            source_result_hash=value["source_result_hash"],  # type: ignore[arg-type]
            replay_result_ref=value["replay_result_ref"],  # type: ignore[arg-type]
            replay_result_hash=value["replay_result_hash"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            config_hash=value["config_hash"],  # type: ignore[arg-type]
            status_match=value["status_match"],  # type: ignore[arg-type]
            artifact_refs_match=value["artifact_refs_match"],  # type: ignore[arg-type]
            formal_eligibility_match=value["formal_eligibility_match"],  # type: ignore[arg-type]
            equivalent=value["equivalent"],  # type: ignore[arg-type]
        )
        if value["replay_hash"] != record.replay_hash:
            raise TaskLifecycleError("replay_hash 与内容不一致")
        return record


@dataclass(frozen=True, slots=True)
class TaskFinalizationRecord:
    """对一个完整 PASS 结果的不可变最终确认。

    ``scope`` 由源结果派生，调用方不能把 ``local_fixture`` 指定成 ``formal``。
    该记录只确认任务结果及其逻辑引用已经冻结；artifact 内容仍应由各自 schema
    loader 或 TensorBundle loader 逐一验证。
    """

    result_ref: str
    result_hash: str
    task_id: str
    config_hash: str
    scope: str
    formal_eligible: bool
    artifact_refs: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_ref",
            logical_reference(self.result_ref, field_name="result_ref"),
        )
        _require_hash(self.result_hash, field_name="result_hash")
        _require_hash(self.config_hash, field_name="config_hash")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise TaskLifecycleError("task_id 必须是非空字符串")
        if self.scope not in {"local_fixture", "formal"}:
            raise TaskLifecycleError("finalization scope 无效")
        if self.formal_eligible is not (self.scope == "formal"):
            raise TaskLifecycleError("formal_eligible 必须严格等于 (scope == 'formal')")
        refs: dict[str, str] = {}
        for kind, reference in self.artifact_refs.items():
            if not isinstance(kind, str) or not kind:
                raise TaskLifecycleError("artifact kind 必须是非空字符串")
            refs[kind] = logical_reference(
                reference,
                field_name=f"artifact_refs.{kind}",
            )
        if not refs:
            raise TaskLifecycleError("finalization 必须绑定至少一个 artifact")
        object.__setattr__(self, "artifact_refs", refs)

    @classmethod
    def from_result(
        cls,
        result: TaskRunResult,
        *,
        result_ref: str,
    ) -> "TaskFinalizationRecord":
        if result.status is not TaskRunStatus.PASS:
            raise TaskLifecycleError("只有 PASS TaskRunResult 可以 finalize")
        return cls(
            result_ref=result_ref,
            result_hash=result.result_hash,
            task_id=result.task_id,
            config_hash=result.config_hash,
            scope="formal" if result.formal_eligible else "local_fixture",
            formal_eligible=result.formal_eligible,
            artifact_refs=result.artifact_refs,
        )

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TASK_FINALIZATION_SCHEMA_VERSION,
            "result_ref": self.result_ref,
            "result_hash": self.result_hash,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "artifact_refs": dict(self.artifact_refs),
        }

    @property
    def finalization_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["finalization_hash"] = self.finalization_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TaskFinalizationRecord":
        expected = {
            "schema_version",
            "result_ref",
            "result_hash",
            "task_id",
            "config_hash",
            "scope",
            "formal_eligible",
            "artifact_refs",
            "finalization_hash",
        }
        if set(value) != expected or value.get("schema_version") != TASK_FINALIZATION_SCHEMA_VERSION:
            raise TaskLifecycleError("TaskFinalizationRecord 字段集合或 schema_version 无效")
        refs = value["artifact_refs"]
        if not isinstance(refs, Mapping):
            raise TaskLifecycleError("artifact_refs 必须是 object")
        record = cls(
            result_ref=value["result_ref"],  # type: ignore[arg-type]
            result_hash=value["result_hash"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            config_hash=value["config_hash"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            artifact_refs=refs,  # type: ignore[arg-type]
        )
        if value["finalization_hash"] != record.finalization_hash:
            raise TaskLifecycleError("finalization_hash 与内容不一致")
        return record


def load_task_run_result(path: str | Path) -> TaskRunResult:
    """从内部 canonical JSON 边界加载并完整复核 ``result_hash``。"""

    value = ensure_json_object(load_canonical_json(path), field="task result")
    return TaskRunResult.from_mapping(value)


def publish_canonical_immutable(path: str | Path, value: Mapping[str, object]) -> Path:
    """不可变地发布 canonical JSON；相同内容重试幂等，异内容拒绝覆盖。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 不允许借助已有 symlink/junction 把“输出路径”重定向到调用方没有审阅的位置。
    # 对尚不存在的末端文件不做 resolve；只逐级检查真实存在的父链。
    absolute = Path(os.path.abspath(target))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        is_junction = bool(getattr(current, "is_junction", lambda: False)())
        if current.is_symlink() or is_junction:
            raise TaskLifecycleError(f"不可变发布路径不得包含 symlink/junction：{current}")
    payload = canonical_json_bytes(value)
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TaskLifecycleError(f"不可变目标已存在且内容不同：{target}")
        return target

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.read_bytes() != payload:
                raise TaskLifecycleError(f"并发发布产生异内容冲突：{target}")
        return target
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "TASK_FINALIZATION_SCHEMA_VERSION",
    "TASK_REPLAY_SCHEMA_VERSION",
    "TASK_STATUS_SCHEMA_VERSION",
    "TaskFinalizationRecord",
    "TaskLifecycleError",
    "TaskReplayRecord",
    "TaskStatusSnapshot",
    "load_task_run_result",
    "logical_reference",
    "publish_canonical_immutable",
]
