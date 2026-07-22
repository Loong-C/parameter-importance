"""Artifact 人工审核与批准的不可变合同。

审核（review）记录“审阅了哪个 hash、给出什么决定”；批准（approval）是第二个
独立 artifact，只能引用一个 ``APPROVE`` review。两层分离可让后续调度器要求双人
复核，也能防止直接改写 review 把 ``REJECT`` 变为通过。

最关键的安全边界是 scope 不可升级：被审 artifact 的 ``formal_eligible`` 必须由其
自身严格 loader 推导并写入 review；formal approval 仅接受
``artifact_scope=formal`` 且 ``artifact_formal_eligible=true`` 的 review。本机 fixture
即使所有数值检查通过，也只能得到 ``local_validation`` approval。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import re
from typing import Any, Final, Mapping

from .jsonio import JSONValue, canonical_json_hash
from .immutable import freeze_json_mapping, thaw_json_value


ARTIFACT_REVIEW_SCHEMA_VERSION: Final = "artifact-review-v1"
ARTIFACT_APPROVAL_SCHEMA_VERSION: Final = "artifact-approval-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACTOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")


class ArtifactReviewError(ValueError):
    """审核/批准字段、来源 hash 或 scope 无效。"""


class ReviewDecision(StrEnum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    NEEDS_CHANGES = "NEEDS_CHANGES"


class ApprovalScope(StrEnum):
    LOCAL_VALIDATION = "local_validation"
    FORMAL = "formal"


def _text(value: object, *, field_name: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ArtifactReviewError(f"{field_name} 必须是非空字符串")
    if any(ord(character) < 32 for character in value):
        raise ArtifactReviewError(f"{field_name} 不得包含控制字符")
    return value


def _hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactReviewError(f"{field_name} 必须是小写 SHA-256")
    return value


def _actor(value: object, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, maximum=128)
    if _ACTOR_RE.fullmatch(text) is None:
        raise ArtifactReviewError(f"{field_name} 必须使用安全 actor ID")
    return text


def _artifact_ref(value: object) -> str:
    text = _text(value, field_name="artifact_ref")
    if "\\" in text or ":" in text or text.startswith("/"):
        raise ArtifactReviewError("artifact_ref 必须是 POSIX 逻辑引用")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise ArtifactReviewError("artifact_ref 发生路径逃逸")
    return text


@dataclass(frozen=True, slots=True)
class ArtifactReview:
    """绑定某一精确 artifact hash 的审核决定。"""

    artifact_kind: str
    artifact_ref: str
    artifact_hash: str
    artifact_scope: str
    artifact_formal_eligible: bool
    reviewer: str
    decision: ReviewDecision
    findings: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = ARTIFACT_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_REVIEW_SCHEMA_VERSION:
            raise ArtifactReviewError("ArtifactReview.schema_version 无效")
        _text(self.artifact_kind, field_name="artifact_kind", maximum=128)
        object.__setattr__(self, "artifact_ref", _artifact_ref(self.artifact_ref))
        _hash(self.artifact_hash, field_name="artifact_hash")
        if self.artifact_scope not in {"local_fixture", "formal"}:
            raise ArtifactReviewError("artifact_scope 必须是 local_fixture/formal")
        if not isinstance(self.artifact_formal_eligible, bool):
            raise ArtifactReviewError("artifact_formal_eligible 必须是 bool")
        if self.artifact_scope == "local_fixture" and self.artifact_formal_eligible:
            raise ArtifactReviewError("local_fixture artifact 不可能 formal_eligible")
        _actor(self.reviewer, field_name="reviewer")
        try:
            decision = ReviewDecision(self.decision)
        except (TypeError, ValueError) as error:
            raise ArtifactReviewError("review decision 无效") from error
        findings = tuple(
            _text(item, field_name=f"findings[{index}]")
            for index, item in enumerate(self.findings)
        )
        if decision is not ReviewDecision.APPROVE and not findings:
            raise ArtifactReviewError("REJECT/NEEDS_CHANGES 必须至少给出一个 finding")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "artifact_ref": self.artifact_ref,
            "artifact_hash": self.artifact_hash,
            "artifact_scope": self.artifact_scope,
            "artifact_formal_eligible": self.artifact_formal_eligible,
            "reviewer": self.reviewer,
            "decision": self.decision.value,
            "findings": list(self.findings),
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def review_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["review_hash"] = self.review_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactReview":
        expected = {
            "schema_version",
            "artifact_kind",
            "artifact_ref",
            "artifact_hash",
            "artifact_scope",
            "artifact_formal_eligible",
            "reviewer",
            "decision",
            "findings",
            "metadata",
            "review_hash",
        }
        if set(value) != expected:
            raise ArtifactReviewError("ArtifactReview 字段集合无效")
        if not isinstance(value["findings"], list):
            raise ArtifactReviewError("findings 必须是数组")
        if not isinstance(value["metadata"], Mapping):
            raise ArtifactReviewError("metadata 必须是 object")
        review = cls(
            artifact_kind=value["artifact_kind"],
            artifact_ref=value["artifact_ref"],
            artifact_hash=value["artifact_hash"],
            artifact_scope=value["artifact_scope"],
            artifact_formal_eligible=value["artifact_formal_eligible"],
            reviewer=value["reviewer"],
            decision=value["decision"],
            findings=tuple(value["findings"]),
            metadata=value["metadata"],
            schema_version=value["schema_version"],
        )
        if value["review_hash"] != review.review_hash:
            raise ArtifactReviewError("review_hash 与 ArtifactReview 内容不一致")
        return review


@dataclass(frozen=True, slots=True)
class ArtifactApproval:
    """引用一个 APPROVE review 的独立批准 artifact。"""

    review_ref: str
    review_hash: str
    artifact_kind: str
    artifact_ref: str
    artifact_hash: str
    artifact_scope: str
    artifact_formal_eligible: bool
    approval_scope: ApprovalScope
    approver: str
    schema_version: str = ARTIFACT_APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_APPROVAL_SCHEMA_VERSION:
            raise ArtifactReviewError("ArtifactApproval.schema_version 无效")
        object.__setattr__(self, "review_ref", _artifact_ref(self.review_ref))
        _hash(self.review_hash, field_name="review_hash")
        _text(self.artifact_kind, field_name="artifact_kind", maximum=128)
        object.__setattr__(self, "artifact_ref", _artifact_ref(self.artifact_ref))
        _hash(self.artifact_hash, field_name="artifact_hash")
        if self.artifact_scope not in {"local_fixture", "formal"}:
            raise ArtifactReviewError("artifact_scope 必须是 local_fixture/formal")
        if not isinstance(self.artifact_formal_eligible, bool):
            raise ArtifactReviewError("artifact_formal_eligible 必须是 bool")
        if self.artifact_scope == "local_fixture" and self.artifact_formal_eligible:
            raise ArtifactReviewError("local_fixture artifact 不可能 formal_eligible")
        try:
            scope = ApprovalScope(self.approval_scope)
        except (TypeError, ValueError) as error:
            raise ArtifactReviewError("approval_scope 无效") from error
        object.__setattr__(self, "approval_scope", scope)
        if scope is ApprovalScope.FORMAL and not (
            self.artifact_scope == "formal" and self.artifact_formal_eligible
        ):
            raise ArtifactReviewError(
                "formal approval 只接受 artifact_scope=formal 且 formal_eligible=true"
            )
        _actor(self.approver, field_name="approver")

    @classmethod
    def from_review(
        cls,
        review: ArtifactReview,
        *,
        review_ref: str,
        approval_scope: ApprovalScope | str,
        approver: str,
    ) -> "ArtifactApproval":
        if review.decision is not ReviewDecision.APPROVE:
            raise ArtifactReviewError("只有 APPROVE review 可以生成 approval")
        try:
            scope = ApprovalScope(approval_scope)
        except (TypeError, ValueError) as error:
            raise ArtifactReviewError("approval_scope 无效") from error
        if scope is ApprovalScope.FORMAL and not (
            review.artifact_scope == "formal" and review.artifact_formal_eligible
        ):
            raise ArtifactReviewError(
                "formal approval 只接受 artifact_scope=formal 且 formal_eligible=true"
            )
        return cls(
            review_ref=review_ref,
            review_hash=review.review_hash,
            artifact_kind=review.artifact_kind,
            artifact_ref=review.artifact_ref,
            artifact_hash=review.artifact_hash,
            artifact_scope=review.artifact_scope,
            artifact_formal_eligible=review.artifact_formal_eligible,
            approval_scope=scope,
            approver=approver,
        )

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "review_ref": self.review_ref,
            "review_hash": self.review_hash,
            "artifact_kind": self.artifact_kind,
            "artifact_ref": self.artifact_ref,
            "artifact_hash": self.artifact_hash,
            "artifact_scope": self.artifact_scope,
            "artifact_formal_eligible": self.artifact_formal_eligible,
            "approval_scope": self.approval_scope.value,
            "approver": self.approver,
        }

    @property
    def approval_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._payload()
        value["approval_hash"] = self.approval_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactApproval":
        expected = {
            "schema_version",
            "review_ref",
            "review_hash",
            "artifact_kind",
            "artifact_ref",
            "artifact_hash",
            "artifact_scope",
            "artifact_formal_eligible",
            "approval_scope",
            "approver",
            "approval_hash",
        }
        if set(value) != expected:
            raise ArtifactReviewError("ArtifactApproval 字段集合无效")
        approval = cls(
            review_ref=value["review_ref"],
            review_hash=value["review_hash"],
            artifact_kind=value["artifact_kind"],
            artifact_ref=value["artifact_ref"],
            artifact_hash=value["artifact_hash"],
            artifact_scope=value["artifact_scope"],
            artifact_formal_eligible=value["artifact_formal_eligible"],
            approval_scope=value["approval_scope"],
            approver=value["approver"],
            schema_version=value["schema_version"],
        )
        if value["approval_hash"] != approval.approval_hash:
            raise ArtifactReviewError("approval_hash 与 ArtifactApproval 内容不一致")
        return approval


__all__ = [
    "ARTIFACT_APPROVAL_SCHEMA_VERSION",
    "ARTIFACT_REVIEW_SCHEMA_VERSION",
    "ApprovalScope",
    "ArtifactApproval",
    "ArtifactReview",
    "ArtifactReviewError",
    "ReviewDecision",
]
