"""正式 Gate、合同状态和本机验证状态的机器可读合同。

本模块最重要的不变量是：正式 Gate 与本机验证永远使用不同类型和不同 schema。
即使 ``LocalValidationRecord`` 的全部检查通过，它也始终携带
``formal_eligible=false``，因此不能被正式入口误当成 GPU、资产或科学 Gate。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import re
from typing import Any, Final

from .errors import GateContractError
from .jsonio import JSONValue, canonical_json_hash, canonical_json_bytes


class GateStatus(StrEnum):
    """正式 Gate 的完整状态集合；值使用稳定的大写 wire representation。"""

    PASS = "PASS"
    CONDITIONALLY_ACCEPTED = "CONDITIONALLY_ACCEPTED"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    STALE = "STALE"
    NOT_RUN = "NOT_RUN"


class LocalValidationStatus(StrEnum):
    """仅表示本机检查结果，不承载任何正式推进资格。"""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    NOT_RUN = "NOT_RUN"


class ContractState(StrEnum):
    """版本化合同或决策是否已经冻结并仍可使用。"""

    FROZEN = "FROZEN"
    UNFROZEN = "UNFROZEN"
    BLOCKED = "BLOCKED"
    STALE = "STALE"


GATE_ID_PATTERN: Final = re.compile(
    r"^stage(?P<stage>[0-9])\.(?P<gate>G[0-9]+(?:[.-][A-Za-z0-9]+)*)$"
)
LOCAL_VALIDATION_ID_PATTERN: Final = re.compile(
    r"^local\.[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


def _validate_timestamp(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise GateContractError(f"{field} 必须是带时区的 ISO-8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GateContractError(f"{field} 不是有效 ISO-8601 时间") from error
    if parsed.tzinfo is None:
        raise GateContractError(f"{field} 必须包含时区")
    return value


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _validate_text(value: Any, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise GateContractError(f"{field} 必须是非空字符串，且长度不超过 {maximum}")
    if any(ord(character) < 32 for character in value):
        raise GateContractError(f"{field} 不能包含控制字符")
    return value


def _validate_evidence_refs(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for index, value in enumerate(values):
        text = _validate_text(value, field=f"{field}[{index}]")
        if "?" in text:
            raise GateContractError(
                f"{field}[{index}] 必须是稳定引用，不能是查询或签名 URL"
            )
        normalized.append(text)
    if len(normalized) != len(set(normalized)):
        raise GateContractError(f"{field} 不能包含重复引用")
    return tuple(normalized)


def validate_gate_id(gate_id: str, *, stage: int | None = None) -> str:
    """验证 ``stageN.G...`` 全限定 Gate ID，并可核对 stage 字段。

    允许计划中已经出现的 ``stage0.G0-C``、``stage2.G2.7b`` 等后缀，但拒绝
    无 stage 前缀的 ``G1``，从根源上消除跨阶段同名 Gate 冲突。
    """

    if not isinstance(gate_id, str):
        raise GateContractError("gate_id 必须是字符串")
    matched = GATE_ID_PATTERN.fullmatch(gate_id)
    if matched is None:
        raise GateContractError(
            "gate_id 必须使用全限定格式，例如 stage0.G1 或 stage1.G1-CONTRACT"
        )
    parsed_stage = int(matched.group("stage"))
    if stage is not None and parsed_stage != stage:
        raise GateContractError(
            f"gate_id 的 stage{parsed_stage} 与 stage={stage} 不一致"
        )
    return gate_id


@dataclass(frozen=True, slots=True)
class GateRecord:
    """一个不可变的正式 Gate 判断。

    ``measured`` 与 ``threshold`` 保留 JSON 原生单位，不把数值压成字符串；
    ``evidence_refs`` 只能引用稳定 artifact。PASS 必须有证据，失败/阻塞/过期必须
    给出原因，条件接受还必须有条件和失效时间。
    """

    gate_id: str
    stage: int
    status: GateStatus
    checked_at: str
    measured: JSONValue = None
    threshold: JSONValue = None
    evidence_refs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    expires_at: str | None = None
    schema_version: str = "gate-record-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "gate-record-v1":
            raise GateContractError("GateRecord.schema_version 必须是 gate-record-v1")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise GateContractError("GateRecord.stage 必须是 0..9 的整数")
        validate_gate_id(self.gate_id, stage=self.stage)
        try:
            normalized_status = GateStatus(self.status)
        except (TypeError, ValueError) as error:
            raise GateContractError(f"未知 GateStatus：{self.status!r}") from error
        object.__setattr__(self, "status", normalized_status)
        _validate_timestamp(self.checked_at, field="checked_at")
        refs = _validate_evidence_refs(tuple(self.evidence_refs), field="evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        reasons = tuple(
            _validate_text(item, field=f"reasons[{index}]")
            for index, item in enumerate(self.reasons)
        )
        conditions = tuple(
            _validate_text(item, field=f"conditions[{index}]")
            for index, item in enumerate(self.conditions)
        )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "conditions", conditions)
        if self.expires_at is not None:
            _validate_timestamp(self.expires_at, field="expires_at")
            if _timestamp_value(self.expires_at) <= _timestamp_value(self.checked_at):
                raise GateContractError("expires_at 必须晚于 checked_at")

        if normalized_status is GateStatus.PASS:
            if not refs:
                raise GateContractError("PASS Gate 必须至少引用一份证据")
            if conditions:
                raise GateContractError("PASS Gate 不能携带 conditions")
        elif normalized_status is GateStatus.CONDITIONALLY_ACCEPTED:
            if not refs or not conditions or self.expires_at is None:
                raise GateContractError(
                    "CONDITIONALLY_ACCEPTED 必须同时有证据、条件和 expires_at"
                )
        elif not reasons:
            raise GateContractError(
                f"{normalized_status.value} Gate 必须给出至少一个 reason"
            )

        # 在构造时确认 measured/threshold 都属于严格 JSON 数据模型。
        canonical_json_bytes(self.measured)
        canonical_json_bytes(self.threshold)

    def payload_dict(self) -> dict[str, JSONValue]:
        """返回不含自引用哈希的稳定 wire payload。"""

        return {
            "schema_version": self.schema_version,
            "scope": "formal",
            "gate_id": self.gate_id,
            "stage": self.stage,
            "status": self.status.value,
            "checked_at": self.checked_at,
            "measured": self.measured,
            "threshold": self.threshold,
            "evidence_refs": list(self.evidence_refs),
            "reasons": list(self.reasons),
            "conditions": list(self.conditions),
            "expires_at": self.expires_at,
        }

    @property
    def artifact_hash(self) -> str:
        """返回 Gate payload 的 canonical SHA-256。"""

        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "GateRecord":
        """严格读取 Gate 对象；未知、缺失字段和哈希漂移均失败。"""

        required = {
            "schema_version",
            "scope",
            "gate_id",
            "stage",
            "status",
            "checked_at",
            "measured",
            "threshold",
            "evidence_refs",
            "reasons",
            "conditions",
            "expires_at",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise GateContractError(
                f"GateRecord 字段错误：missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if value["scope"] != "formal":
            raise GateContractError("GateRecord.scope 必须是 formal")
        for field in ("evidence_refs", "reasons", "conditions"):
            if not isinstance(value[field], list) or not all(
                isinstance(item, str) for item in value[field]
            ):
                raise GateContractError(f"{field} 必须是字符串数组")
        record = cls(
            gate_id=value["gate_id"],
            stage=value["stage"],
            status=value["status"],
            checked_at=value["checked_at"],
            measured=value["measured"],
            threshold=value["threshold"],
            evidence_refs=tuple(value["evidence_refs"]),
            reasons=tuple(value["reasons"]),
            conditions=tuple(value["conditions"]),
            expires_at=value["expires_at"],
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != record.artifact_hash:
            raise GateContractError("GateRecord.artifact_hash 与内容不一致")
        return record

    def effective_status(self, *, at: datetime | None = None) -> GateStatus:
        """在给定时间计算有效状态；过期的通过记录按 STALE 处理。"""

        if self.expires_at is None:
            return self.status
        instant = at or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            raise GateContractError("effective_status.at 必须带时区")
        if instant.astimezone(timezone.utc) >= _timestamp_value(self.expires_at):
            return GateStatus.STALE
        return self.status


@dataclass(frozen=True, slots=True)
class LocalValidationRecord:
    """本机 CPU/fixture 检查记录；它永远不能满足正式 Gate。"""

    validation_id: str
    status: LocalValidationStatus
    checked_at: str
    checks: dict[str, bool]
    evidence_refs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    host_profile: str = "windows-cpu-local"
    schema_version: str = "local-validation-record-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "local-validation-record-v1":
            raise GateContractError(
                "LocalValidationRecord.schema_version 必须是 local-validation-record-v1"
            )
        if not isinstance(self.validation_id, str) or LOCAL_VALIDATION_ID_PATTERN.fullmatch(
            self.validation_id
        ) is None:
            raise GateContractError(
                "validation_id 必须使用 local.<namespace> 格式且仅含小写安全字符"
            )
        try:
            normalized_status = LocalValidationStatus(self.status)
        except (TypeError, ValueError) as error:
            raise GateContractError(
                f"未知 LocalValidationStatus：{self.status!r}"
            ) from error
        object.__setattr__(self, "status", normalized_status)
        _validate_timestamp(self.checked_at, field="checked_at")
        if not isinstance(self.checks, dict) or any(
            not isinstance(key, str) or not key or not isinstance(result, bool)
            for key, result in self.checks.items()
        ):
            raise GateContractError("checks 必须是非空字符串到 bool 的对象")
        refs = _validate_evidence_refs(tuple(self.evidence_refs), field="evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        reasons = tuple(
            _validate_text(item, field=f"reasons[{index}]")
            for index, item in enumerate(self.reasons)
        )
        object.__setattr__(self, "reasons", reasons)
        _validate_text(self.host_profile, field="host_profile", maximum=256)
        if normalized_status is LocalValidationStatus.PASS:
            if not self.checks or not all(self.checks.values()):
                raise GateContractError("本机 PASS 要求 checks 非空且全部为 true")
        elif normalized_status is LocalValidationStatus.FAIL:
            if not self.checks or all(self.checks.values()):
                raise GateContractError("本机 FAIL 至少需要一个 false check")
        elif not reasons:
            raise GateContractError(
                f"本机 {normalized_status.value} 必须给出至少一个 reason"
            )

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "scope": "local_fixture",
            "formal_eligible": False,
            "validation_id": self.validation_id,
            "status": self.status.value,
            "checked_at": self.checked_at,
            "checks": dict(sorted(self.checks.items())),
            "evidence_refs": list(self.evidence_refs),
            "reasons": list(self.reasons),
            "host_profile": self.host_profile,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LocalValidationRecord":
        required = {
            "schema_version",
            "scope",
            "formal_eligible",
            "validation_id",
            "status",
            "checked_at",
            "checks",
            "evidence_refs",
            "reasons",
            "host_profile",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise GateContractError(
                "LocalValidationRecord 字段错误："
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if value["scope"] != "local_fixture" or value["formal_eligible"] is not False:
            raise GateContractError("本机验证必须是 local_fixture 且 formal_eligible=false")
        if not isinstance(value["checks"], dict):
            raise GateContractError("checks 必须是对象")
        if not isinstance(value["evidence_refs"], list) or not isinstance(value["reasons"], list):
            raise GateContractError("evidence_refs/reasons 必须是数组")
        record = cls(
            validation_id=value["validation_id"],
            status=value["status"],
            checked_at=value["checked_at"],
            checks=dict(value["checks"]),
            evidence_refs=tuple(value["evidence_refs"]),
            reasons=tuple(value["reasons"]),
            host_profile=value["host_profile"],
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != record.artifact_hash:
            raise GateContractError("LocalValidationRecord.artifact_hash 与内容不一致")
        return record
