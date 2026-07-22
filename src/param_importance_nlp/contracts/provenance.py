"""运行 provenance 的白名单字段合同。

provenance 只保存复现所需的稳定身份和 artifact 引用，不接受任意 ``metadata``，
从结构上减少令牌、Cookie、签名 URL 或整份环境变量被写入结果的机会。正式运行必须
来自干净工作树；本机 smoke 若允许 dirty，则必须同时记录基线提交、完整可应用补丁
引用和补丁 SHA-256，不能只写一句“dirty”。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re
from typing import Any, Final

from .errors import ProvenanceContractError
from .identity import RunIdentity
from .jsonio import JSONValue, canonical_json_hash


SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class ProvenanceStatus(StrEnum):
    """provenance 生命周期；STARTED 是唯一允许没有结束时间的状态。"""

    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


def _timestamp(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ProvenanceContractError(f"{field} 必须是带时区 ISO-8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProvenanceContractError(f"{field} 不是有效 ISO-8601 时间") from error
    if parsed.tzinfo is None:
        raise ProvenanceContractError(f"{field} 必须包含时区")
    return parsed


def _text(value: Any, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ProvenanceContractError(f"{field} 必须是非空字符串")
    if any(ord(character) < 32 for character in value):
        raise ProvenanceContractError(f"{field} 不能包含控制字符")
    return value


def _stable_ref(value: str | None, *, field: str, required: bool = False) -> None:
    if value is None:
        if required:
            raise ProvenanceContractError(f"{field} 不能为空")
        return
    text = _text(value, field=field)
    if "?" in text or "://" in text:
        raise ProvenanceContractError(
            f"{field} 只能是稳定 artifact ID/路径，不能是 URL 或查询字符串"
        )


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """一次 attempt 的最小、白名单化且可哈希 provenance。"""

    identity: RunIdentity
    config_hash: str
    resolved_config_ref: str
    seed_plan_hash: str
    git_commit: str
    git_branch: str
    worktree_clean: bool
    environment_id: str
    hardware_snapshot_ref: str
    device_mapping: tuple[str, ...]
    model_manifest_id: str
    data_manifest_id: str
    started_at: str
    status: ProvenanceStatus
    scope: str = "local_fixture"
    tokenizer_manifest_id: str | None = None
    ended_at: str | None = None
    resume_checkpoint_id: str | None = None
    dirty_base_commit: str | None = None
    dirty_patch_ref: str | None = None
    dirty_patch_hash: str | None = None
    artifact_refs: tuple[str, ...] = ()
    schema_version: str = "provenance-record-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "provenance-record-v1":
            raise ProvenanceContractError(
                "ProvenanceRecord.schema_version 必须是 provenance-record-v1"
            )
        if not isinstance(self.identity, RunIdentity):
            raise ProvenanceContractError("identity 必须是 RunIdentity")
        if self.scope not in {"local_fixture", "formal"}:
            raise ProvenanceContractError("scope 必须是 local_fixture 或 formal")
        for field, digest in (
            ("config_hash", self.config_hash),
            ("seed_plan_hash", self.seed_plan_hash),
        ):
            if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
                raise ProvenanceContractError(f"{field} 必须是小写 SHA-256")
        if not isinstance(self.git_commit, str) or GIT_COMMIT_PATTERN.fullmatch(self.git_commit) is None:
            raise ProvenanceContractError("git_commit 必须是完整 40/64 位小写提交摘要")
        _text(self.git_branch, field="git_branch", maximum=512)
        if not isinstance(self.worktree_clean, bool):
            raise ProvenanceContractError("worktree_clean 必须是 bool")
        for field, value in (
            ("resolved_config_ref", self.resolved_config_ref),
            ("environment_id", self.environment_id),
            ("hardware_snapshot_ref", self.hardware_snapshot_ref),
            ("model_manifest_id", self.model_manifest_id),
            ("data_manifest_id", self.data_manifest_id),
        ):
            _stable_ref(value, field=field, required=True)
        _stable_ref(self.tokenizer_manifest_id, field="tokenizer_manifest_id")
        _stable_ref(self.resume_checkpoint_id, field="resume_checkpoint_id")
        if not isinstance(self.device_mapping, tuple) or not self.device_mapping:
            raise ProvenanceContractError("device_mapping 必须至少记录 cpu 或一个设备 ID")
        for index, device in enumerate(self.device_mapping):
            _stable_ref(device, field=f"device_mapping[{index}]", required=True)
        if len(self.device_mapping) != len(set(self.device_mapping)):
            raise ProvenanceContractError("device_mapping 不能重复")
        for index, reference in enumerate(self.artifact_refs):
            _stable_ref(reference, field=f"artifact_refs[{index}]", required=True)
        if len(self.artifact_refs) != len(set(self.artifact_refs)):
            raise ProvenanceContractError("artifact_refs 不能重复")

        started = _timestamp(self.started_at, field="started_at")
        try:
            normalized_status = ProvenanceStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ProvenanceContractError(f"未知 provenance status：{self.status!r}") from error
        object.__setattr__(self, "status", normalized_status)
        if normalized_status is ProvenanceStatus.STARTED:
            if self.ended_at is not None:
                raise ProvenanceContractError("STARTED provenance 不能有 ended_at")
        else:
            if self.ended_at is None:
                raise ProvenanceContractError(f"{normalized_status.value} 必须记录 ended_at")
            if _timestamp(self.ended_at, field="ended_at") < started:
                raise ProvenanceContractError("ended_at 不能早于 started_at")

        dirty_values = (
            self.dirty_base_commit,
            self.dirty_patch_ref,
            self.dirty_patch_hash,
        )
        if self.worktree_clean:
            if any(value is not None for value in dirty_values):
                raise ProvenanceContractError("干净工作树不能携带 dirty patch 字段")
        else:
            if any(value is None for value in dirty_values):
                raise ProvenanceContractError(
                    "dirty smoke 必须同时记录 base commit、完整 patch 引用和 patch hash"
                )
            if GIT_COMMIT_PATTERN.fullmatch(self.dirty_base_commit or "") is None:
                raise ProvenanceContractError("dirty_base_commit 必须是完整提交摘要")
            _stable_ref(self.dirty_patch_ref, field="dirty_patch_ref", required=True)
            if SHA256_PATTERN.fullmatch(self.dirty_patch_hash or "") is None:
                raise ProvenanceContractError("dirty_patch_hash 必须是小写 SHA-256")

    @property
    def formal_eligible(self) -> bool:
        """只有 formal scope 且工作树干净时才允许进入正式证据链。"""

        return self.scope == "formal" and self.worktree_clean

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "config_hash": self.config_hash,
            "resolved_config_ref": self.resolved_config_ref,
            "seed_plan_hash": self.seed_plan_hash,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "worktree_clean": self.worktree_clean,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "dirty_base_commit": self.dirty_base_commit,
            "dirty_patch_ref": self.dirty_patch_ref,
            "dirty_patch_hash": self.dirty_patch_hash,
            "environment_id": self.environment_id,
            "hardware_snapshot_ref": self.hardware_snapshot_ref,
            "device_mapping": list(self.device_mapping),
            "model_manifest_id": self.model_manifest_id,
            "tokenizer_manifest_id": self.tokenizer_manifest_id,
            "data_manifest_id": self.data_manifest_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status.value,
            "resume_checkpoint_id": self.resume_checkpoint_id,
            "artifact_refs": list(self.artifact_refs),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ProvenanceRecord":
        required = {
            "schema_version",
            "identity",
            "config_hash",
            "resolved_config_ref",
            "seed_plan_hash",
            "git_commit",
            "git_branch",
            "worktree_clean",
            "scope",
            "formal_eligible",
            "dirty_base_commit",
            "dirty_patch_ref",
            "dirty_patch_hash",
            "environment_id",
            "hardware_snapshot_ref",
            "device_mapping",
            "model_manifest_id",
            "tokenizer_manifest_id",
            "data_manifest_id",
            "started_at",
            "ended_at",
            "status",
            "resume_checkpoint_id",
            "artifact_refs",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise ProvenanceContractError(
                f"ProvenanceRecord 字段错误：missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if not isinstance(value["identity"], dict):
            raise ProvenanceContractError("identity 必须是 object")
        for field in ("device_mapping", "artifact_refs"):
            if not isinstance(value[field], list):
                raise ProvenanceContractError(f"{field} 必须是数组")
        record = cls(
            identity=RunIdentity.from_mapping(value["identity"]),
            config_hash=value["config_hash"],
            resolved_config_ref=value["resolved_config_ref"],
            seed_plan_hash=value["seed_plan_hash"],
            git_commit=value["git_commit"],
            git_branch=value["git_branch"],
            worktree_clean=value["worktree_clean"],
            scope=value["scope"],
            environment_id=value["environment_id"],
            hardware_snapshot_ref=value["hardware_snapshot_ref"],
            device_mapping=tuple(value["device_mapping"]),
            model_manifest_id=value["model_manifest_id"],
            data_manifest_id=value["data_manifest_id"],
            started_at=value["started_at"],
            status=value["status"],
            tokenizer_manifest_id=value["tokenizer_manifest_id"],
            ended_at=value["ended_at"],
            resume_checkpoint_id=value["resume_checkpoint_id"],
            dirty_base_commit=value["dirty_base_commit"],
            dirty_patch_ref=value["dirty_patch_ref"],
            dirty_patch_hash=value["dirty_patch_hash"],
            artifact_refs=tuple(value["artifact_refs"]),
            schema_version=value["schema_version"],
        )
        if value["formal_eligible"] is not record.formal_eligible:
            raise ProvenanceContractError("formal_eligible 与 scope/worktree_clean 派生值不一致")
        if value["artifact_hash"] != record.artifact_hash:
            raise ProvenanceContractError("ProvenanceRecord.artifact_hash 与内容不一致")
        return record
