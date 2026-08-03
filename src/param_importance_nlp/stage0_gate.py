"""Strict, hash-bound evidence records for Stage 0 gates.

Stage 0 mixes correctness, safety, operational and performance requirements.  A
single free-form ``status`` field is therefore too easy to misuse: in particular,
an approved performance exception must never make a failed correctness check look
like a pass.  This module provides the small common envelope used by the G4--G10
formal runners and by the independent replay audit.

The envelope deliberately does not execute a gate.  Gate-specific runners must
produce the measurements and evidence references; this module only validates and
aggregates them without weakening their meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from os import PathLike
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GATE_ID_RE = re.compile(r"^stage0\.G(?:[0-9]|10)(?:-[A-Z0-9]+)?$")
_CHECK_ID_RE = re.compile(r"^stage0\.[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENVIRONMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class Stage0GateEvidenceError(ValueError):
    """Raised when a Stage 0 gate record is ambiguous or internally inconsistent."""


class Stage0CheckClass(StrEnum):
    CORRECTNESS = "correctness"
    SAFETY = "safety"
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    CAPACITY = "capacity"


class Stage0CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    APPROVED_EXCEPTION = "APPROVED_EXCEPTION"


class Stage0GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    CONDITIONALLY_ACCEPTED = "CONDITIONALLY_ACCEPTED"


def _require_non_empty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage0GateEvidenceError(f"{field} must be a non-empty string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    text = _require_non_empty(value, field=field)
    if _SHA256_RE.fullmatch(text) is None:
        raise Stage0GateEvidenceError(f"{field} must be a lowercase SHA-256")
    return text


def _require_timestamp(value: object, *, field: str) -> str:
    text = _require_non_empty(value, field=field)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as error:
        raise Stage0GateEvidenceError(f"{field} must be RFC3339") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0GateEvidenceError(f"{field} must include a UTC offset")
    return text


def _logical_ref(value: object, *, field: str) -> str:
    text = _require_non_empty(value, field=field)
    if "\\" in text:
        raise Stage0GateEvidenceError(f"{field} must use POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage0GateEvidenceError(f"{field} must be a canonical logical reference")
    if str(path) != text:
        raise Stage0GateEvidenceError(f"{field} is not canonical")
    return text


def _json_mapping(value: object, *, field: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0GateEvidenceError(f"{field} must be an object with string keys")
    # canonical_json_hash is also a recursive JSON-type validator.
    normalized = dict(value)
    try:
        canonical_json_hash(normalized)
    except (TypeError, ValueError) as error:
        raise Stage0GateEvidenceError(f"{field} is not canonical JSON") from error
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Stage0EvidenceRef:
    ref: str
    sha256: str
    schema_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _logical_ref(self.ref, field="evidence.ref"))
        object.__setattr__(
            self, "sha256", _require_sha256(self.sha256, field="evidence.sha256")
        )
        object.__setattr__(
            self,
            "schema_version",
            _require_non_empty(self.schema_version, field="evidence.schema_version"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "ref": self.ref,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage0EvidenceRef":
        if set(value) != {"ref", "sha256", "schema_version"}:
            raise Stage0GateEvidenceError("evidence reference fields are invalid")
        return cls(
            ref=value["ref"],  # type: ignore[arg-type]
            sha256=value["sha256"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Stage0ExceptionApproval:
    approval_ref: str
    approval_sha256: str
    approved_by: str
    approved_at: str
    expires_at: str
    scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approval_ref",
            _logical_ref(self.approval_ref, field="approval.approval_ref"),
        )
        object.__setattr__(
            self,
            "approval_sha256",
            _require_sha256(
                self.approval_sha256, field="approval.approval_sha256"
            ),
        )
        object.__setattr__(
            self,
            "approved_by",
            _require_non_empty(self.approved_by, field="approval.approved_by"),
        )
        approved = _require_timestamp(self.approved_at, field="approval.approved_at")
        expires = _require_timestamp(self.expires_at, field="approval.expires_at")
        object.__setattr__(self, "approved_at", approved)
        object.__setattr__(self, "expires_at", expires)
        approved_dt = datetime.fromisoformat(approved.replace("Z", "+00:00"))
        expires_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        if expires_dt <= approved_dt:
            raise Stage0GateEvidenceError("approval.expires_at must follow approved_at")
        object.__setattr__(
            self, "scope", _require_non_empty(self.scope, field="approval.scope")
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "approval_ref": self.approval_ref,
            "approval_sha256": self.approval_sha256,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "Stage0ExceptionApproval":
        expected = {
            "approval_ref",
            "approval_sha256",
            "approved_by",
            "approved_at",
            "expires_at",
            "scope",
        }
        if set(value) != expected:
            raise Stage0GateEvidenceError("exception approval fields are invalid")
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Stage0GateCheck:
    check_id: str
    check_class: Stage0CheckClass | str
    status: Stage0CheckStatus | str
    summary: str
    exception_eligible: bool = False
    measurements: Mapping[str, JSONValue] | None = None
    evidence_refs: tuple[str, ...] = ()
    approval: Stage0ExceptionApproval | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or _CHECK_ID_RE.fullmatch(self.check_id) is None:
            raise Stage0GateEvidenceError("check_id is invalid")
        try:
            check_class = Stage0CheckClass(self.check_class)
            status = Stage0CheckStatus(self.status)
        except ValueError as error:
            raise Stage0GateEvidenceError("check class/status is invalid") from error
        object.__setattr__(self, "check_class", check_class)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "summary", _require_non_empty(self.summary, field="check.summary")
        )
        if not isinstance(self.exception_eligible, bool):
            raise Stage0GateEvidenceError("exception_eligible must be bool")
        if self.exception_eligible and check_class not in {
            Stage0CheckClass.PERFORMANCE,
            Stage0CheckClass.CAPACITY,
        }:
            raise Stage0GateEvidenceError(
                "only performance/capacity checks may be exception eligible"
            )
        if status is Stage0CheckStatus.APPROVED_EXCEPTION:
            if not self.exception_eligible or self.approval is None:
                raise Stage0GateEvidenceError(
                    "approved exception requires eligibility and approval evidence"
                )
        elif self.approval is not None:
            raise Stage0GateEvidenceError(
                "approval evidence is only valid for APPROVED_EXCEPTION"
            )
        measurements = _json_mapping(
            {} if self.measurements is None else self.measurements,
            field="check.measurements",
        )
        object.__setattr__(self, "measurements", measurements)
        refs = tuple(
            _logical_ref(item, field=f"check.evidence_refs[{index}]")
            for index, item in enumerate(self.evidence_refs)
        )
        if len(set(refs)) != len(refs):
            raise Stage0GateEvidenceError("check evidence_refs must be unique")
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "check_id": self.check_id,
            "check_class": self.check_class.value,
            "status": self.status.value,
            "summary": self.summary,
            "exception_eligible": self.exception_eligible,
            "measurements": dict(self.measurements or {}),
            "evidence_refs": list(self.evidence_refs),
            "approval": None if self.approval is None else self.approval.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage0GateCheck":
        expected = {
            "check_id",
            "check_class",
            "status",
            "summary",
            "exception_eligible",
            "measurements",
            "evidence_refs",
            "approval",
        }
        if set(value) != expected:
            raise Stage0GateEvidenceError("gate check fields are invalid")
        refs = value["evidence_refs"]
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            raise Stage0GateEvidenceError("check.evidence_refs must be a string array")
        approval_value = value["approval"]
        if approval_value is not None and not isinstance(approval_value, Mapping):
            raise Stage0GateEvidenceError("check.approval must be object or null")
        return cls(
            check_id=value["check_id"],  # type: ignore[arg-type]
            check_class=value["check_class"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            summary=value["summary"],  # type: ignore[arg-type]
            exception_eligible=value["exception_eligible"],  # type: ignore[arg-type]
            measurements=value["measurements"],  # type: ignore[arg-type]
            evidence_refs=tuple(refs),
            approval=(
                None
                if approval_value is None
                else Stage0ExceptionApproval.from_mapping(approval_value)
            ),
        )


def aggregate_stage0_gate_status(
    checks: Sequence[Stage0GateCheck],
) -> Stage0GateStatus:
    if not checks:
        raise Stage0GateEvidenceError("a gate report must contain at least one check")
    statuses = {check.status for check in checks}
    if Stage0CheckStatus.FAIL in statuses:
        return Stage0GateStatus.FAIL
    if Stage0CheckStatus.BLOCKED in statuses:
        return Stage0GateStatus.BLOCKED
    if Stage0CheckStatus.APPROVED_EXCEPTION in statuses:
        return Stage0GateStatus.CONDITIONALLY_ACCEPTED
    return Stage0GateStatus.PASS


@dataclass(frozen=True, slots=True)
class Stage0GateReport:
    gate_id: str
    generated_at: str
    generator_git_commit: str
    environment_id: str
    checks: tuple[Stage0GateCheck, ...]
    input_evidence: tuple[Stage0EvidenceRef, ...] = ()
    config_hashes: Mapping[str, str] | None = None
    status: Stage0GateStatus | str | None = None
    schema_version: str = "stage0-gate-report-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, str) or _GATE_ID_RE.fullmatch(self.gate_id) is None:
            raise Stage0GateEvidenceError("gate_id is invalid")
        object.__setattr__(
            self,
            "generated_at",
            _require_timestamp(self.generated_at, field="generated_at"),
        )
        if (
            not isinstance(self.generator_git_commit, str)
            or _GIT_COMMIT_RE.fullmatch(self.generator_git_commit) is None
        ):
            raise Stage0GateEvidenceError("generator_git_commit must be a full commit")
        if (
            not isinstance(self.environment_id, str)
            or _ENVIRONMENT_ID_RE.fullmatch(self.environment_id) is None
        ):
            raise Stage0GateEvidenceError("environment_id is invalid")
        checks = tuple(self.checks)
        if any(not isinstance(item, Stage0GateCheck) for item in checks):
            raise Stage0GateEvidenceError("checks must contain Stage0GateCheck")
        check_ids = [item.check_id for item in checks]
        if len(set(check_ids)) != len(check_ids):
            raise Stage0GateEvidenceError("check_id values must be unique")
        object.__setattr__(self, "checks", checks)
        evidence = tuple(self.input_evidence)
        if any(not isinstance(item, Stage0EvidenceRef) for item in evidence):
            raise Stage0GateEvidenceError("input_evidence entries are invalid")
        refs = [item.ref for item in evidence]
        if len(set(refs)) != len(refs):
            raise Stage0GateEvidenceError("input evidence references must be unique")
        object.__setattr__(self, "input_evidence", evidence)
        hashes = dict(self.config_hashes or {})
        for key, value in hashes.items():
            if _ENVIRONMENT_ID_RE.fullmatch(key) is None:
                raise Stage0GateEvidenceError("config_hashes key is invalid")
            _require_sha256(value, field=f"config_hashes.{key}")
        object.__setattr__(self, "config_hashes", dict(sorted(hashes.items())))
        aggregate = aggregate_stage0_gate_status(checks)
        if self.status is not None:
            try:
                claimed = Stage0GateStatus(self.status)
            except ValueError as error:
                raise Stage0GateEvidenceError("gate status is invalid") from error
            if claimed is not aggregate:
                raise Stage0GateEvidenceError(
                    f"gate status {claimed.value} disagrees with checks {aggregate.value}"
                )
        object.__setattr__(self, "status", aggregate)
        if self.schema_version != "stage0-gate-report-v1":
            raise Stage0GateEvidenceError("schema_version is invalid")

    def _identity_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "status": self.status.value,
            "generated_at": self.generated_at,
            "generator_git_commit": self.generator_git_commit,
            "environment_id": self.environment_id,
            "config_hashes": dict(self.config_hashes or {}),
            "input_evidence": [item.to_dict() for item in self.input_evidence],
            "checks": [item.to_dict() for item in self.checks],
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self._identity_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self._identity_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage0GateReport":
        expected = {
            "schema_version",
            "gate_id",
            "status",
            "generated_at",
            "generator_git_commit",
            "environment_id",
            "config_hashes",
            "input_evidence",
            "checks",
            "artifact_hash",
        }
        if set(value) != expected:
            raise Stage0GateEvidenceError("gate report fields are invalid")
        raw_checks = value["checks"]
        raw_evidence = value["input_evidence"]
        if not isinstance(raw_checks, list) or any(
            not isinstance(item, Mapping) for item in raw_checks
        ):
            raise Stage0GateEvidenceError("checks must be an object array")
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, Mapping) for item in raw_evidence
        ):
            raise Stage0GateEvidenceError("input_evidence must be an object array")
        report = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            gate_id=value["gate_id"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            generated_at=value["generated_at"],  # type: ignore[arg-type]
            generator_git_commit=value["generator_git_commit"],  # type: ignore[arg-type]
            environment_id=value["environment_id"],  # type: ignore[arg-type]
            config_hashes=value["config_hashes"],  # type: ignore[arg-type]
            input_evidence=tuple(
                Stage0EvidenceRef.from_mapping(item) for item in raw_evidence
            ),
            checks=tuple(Stage0GateCheck.from_mapping(item) for item in raw_checks),
        )
        claimed_hash = _require_sha256(value["artifact_hash"], field="artifact_hash")
        if claimed_hash != report.artifact_hash:
            raise Stage0GateEvidenceError("artifact_hash does not match report content")
        return report


def load_stage0_gate_report(path: str | PathLike[str]) -> Stage0GateReport:
    value = load_canonical_json(path)
    if not isinstance(value, dict):
        raise Stage0GateEvidenceError("gate report root must be an object")
    return Stage0GateReport.from_mapping(value)


def utc_now_rfc3339() -> str:
    """Return a canonical seconds-resolution UTC timestamp for evidence writers."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


__all__ = [
    "Stage0CheckClass",
    "Stage0CheckStatus",
    "Stage0EvidenceRef",
    "Stage0ExceptionApproval",
    "Stage0GateCheck",
    "Stage0GateEvidenceError",
    "Stage0GateReport",
    "Stage0GateStatus",
    "aggregate_stage0_gate_status",
    "load_stage0_gate_report",
    "utc_now_rfc3339",
]
