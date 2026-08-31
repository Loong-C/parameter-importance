"""正式运行能力的不可变证据合同。

``TaskRuntimeEnvironment`` 中的字符串集合只是一张索引，不能自行证明服务器、
CUDA、资产或外部授权真的可用。本模块定义最小的 hash-bound 证明：每个声明的
capability 必须引用一个 scope=formal 的 ``RuntimeCapabilityEvidence``。探测动作
由受控环境完成；本机运行时只做严格解析、哈希和状态验证，不会尝试 SSH、下载或
把本机 fixture 冒充正式证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .jsonio import JSONValue, canonical_json_bytes, canonical_json_hash


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class RuntimeEvidenceError(ValueError):
    """能力证据字段、状态或内容哈希无效。"""


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeEvidenceError("checked_at 必须是带时区的 ISO-8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeEvidenceError("checked_at 不是合法 ISO-8601") from error
    if parsed.tzinfo is None:
        raise RuntimeEvidenceError("checked_at 必须包含时区")
    return value


def _logical_ref(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeEvidenceError(f"{field_name} 必须是 POSIX workspace 逻辑引用")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeEvidenceError(f"{field_name} 发生路径逃逸")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityEvidence:
    """一个外部 capability 的正式、内容寻址验证结果。

    ``VERIFIED`` 只表示指定探测器在 ``checked_at`` 发布了可审计证据；它不会使
    任何科学 Gate 自动通过。``BLOCKED`` 可作为诊断 artifact 保存，但不能解锁
    formal runner。
    """

    capability: str
    status: str
    checked_at: str
    evidence_refs: tuple[str, ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    scope: str = "formal"
    schema_version: str = "runtime-capability-evidence-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "runtime-capability-evidence-v1":
            raise RuntimeEvidenceError("RuntimeCapabilityEvidence schema_version 无效")
        if not isinstance(self.capability, str) or _NAME_RE.fullmatch(self.capability) is None:
            raise RuntimeEvidenceError("capability 必须是 snake_case 名称")
        if self.scope != "formal":
            raise RuntimeEvidenceError("正式能力证据 scope 必须是 formal")
        if self.status not in {"VERIFIED", "BLOCKED", "STALE"}:
            raise RuntimeEvidenceError("status 只能是 VERIFIED/BLOCKED/STALE")
        object.__setattr__(self, "checked_at", _timestamp(self.checked_at))
        raw_refs = tuple(self.evidence_refs)
        if not raw_refs:
            raise RuntimeEvidenceError("evidence_refs 必须至少包含一个稳定引用")
        refs = tuple(
            _logical_ref(ref, field_name=f"evidence_refs[{index}]")
            for index, ref in enumerate(raw_refs)
        )
        if len(refs) != len(set(refs)):
            raise RuntimeEvidenceError("evidence_refs 不允许重复")
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.metadata, Mapping):
            raise RuntimeEvidenceError("metadata 必须是 object")
        normalized = dict(self.metadata)
        canonical_json_bytes(normalized)
        object.__setattr__(self, "metadata", normalized)

    @property
    def verified(self) -> bool:
        return self.status == "VERIFIED"

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "capability": self.capability,
            "status": self.status,
            "checked_at": self.checked_at,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeCapabilityEvidence":
        expected = {
            "schema_version",
            "scope",
            "capability",
            "status",
            "checked_at",
            "evidence_refs",
            "metadata",
            "artifact_hash",
        }
        if set(value) != expected:
            raise RuntimeEvidenceError("RuntimeCapabilityEvidence 字段集合无效")
        raw_refs = value["evidence_refs"]
        if not isinstance(raw_refs, list):
            raise RuntimeEvidenceError("evidence_refs 必须是数组")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise RuntimeEvidenceError("metadata 必须是 object")
        result = cls(
            capability=value["capability"],
            status=value["status"],
            checked_at=value["checked_at"],
            evidence_refs=tuple(raw_refs),
            metadata=metadata,
            scope=value["scope"],
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != result.artifact_hash:
            raise RuntimeEvidenceError("RuntimeCapabilityEvidence artifact_hash 不匹配")
        return result


__all__ = [
    "RuntimeCapabilityEvidence",
    "RuntimeEvidenceError",
]
