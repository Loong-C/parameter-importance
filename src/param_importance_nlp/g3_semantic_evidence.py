"""Shared validation for Stage 0 G3 semantic-verification evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Final

from .assets import validate_asset_path
from .contracts import canonical_json_bytes, canonical_json_hash


SEMANTIC_EVIDENCE_SCHEMA_VERSION: Final = (
    "stage0-asset-semantic-verification-v1"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SEMANTIC_FIELDS: Final = frozenset(
    {
        "schema_version",
        "formal",
        "asset_id",
        "candidate_id",
        "logical_name",
        "kind",
        "requirements_ref",
        "requirements_sha256",
        "checks",
        "network_attempts",
        "checked_at",
        "generator_git_commit",
        "artifact_hash",
    }
)
_SEMANTIC_CHECK_FIELDS: Final = frozenset(
    {"check_id", "status", "summary", "details"}
)


class G3SemanticEvidenceError(ValueError):
    """Raised when semantic evidence is malformed or not formally admissible."""


def _raise(
    error_type: type[ValueError],
    message: str,
) -> None:
    raise error_type(message)


def _require_text(
    value: Any,
    *,
    field: str,
    error_type: type[ValueError],
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        _raise(error_type, f"{field} must be normalized non-empty text")
    return value


def _require_sha256(
    value: Any,
    *,
    field: str,
    error_type: type[ValueError],
) -> str:
    text = _require_text(value, field=field, error_type=error_type)
    if _SHA256.fullmatch(text) is None:
        _raise(error_type, f"{field} must be a lowercase SHA-256")
    return text


def _require_timestamp(
    value: Any,
    *,
    field: str,
    error_type: type[ValueError],
) -> str:
    text = _require_text(value, field=field, error_type=error_type)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise error_type(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        _raise(error_type, f"{field} must include a timezone")
    return text


def semantic_evidence_artifact_hash(value: Mapping[str, Any]) -> str:
    """Return the canonical identity hash, excluding the declared hash field."""

    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def validate_semantic_evidence(
    value: Mapping[str, Any],
    *,
    expected_check_ids: tuple[str, ...] | None = None,
    error_type: type[ValueError] = G3SemanticEvidenceError,
) -> None:
    """Validate the complete formal semantic-evidence envelope.

    ``error_type`` lets the publication API preserve its historical public
    exception class while the gate consumes this module without importing the
    publisher and creating a dependency cycle.
    """

    if not isinstance(value, Mapping) or set(value) != _SEMANTIC_FIELDS:
        _raise(error_type, "semantic evidence fields are invalid")
    if (
        value["schema_version"] != SEMANTIC_EVIDENCE_SCHEMA_VERSION
        or value["formal"] is not True
    ):
        _raise(error_type, "semantic evidence schema/formal marker invalid")
    _require_sha256(value["asset_id"], field="semantic.asset_id", error_type=error_type)
    _require_sha256(
        value["candidate_id"], field="semantic.candidate_id", error_type=error_type
    )
    _require_text(
        value["logical_name"], field="semantic.logical_name", error_type=error_type
    )
    _require_text(value["kind"], field="semantic.kind", error_type=error_type)
    validate_asset_path(
        _require_text(
            value["requirements_ref"],
            field="semantic.requirements_ref",
            error_type=error_type,
        )
    )
    _require_sha256(
        value["requirements_sha256"],
        field="semantic.requirements_sha256",
        error_type=error_type,
    )
    checks = value["checks"]
    if not isinstance(checks, list) or not checks:
        _raise(error_type, "semantic.checks must be non-empty")
    observed: list[str] = []
    for raw in checks:
        if not isinstance(raw, Mapping) or set(raw) != _SEMANTIC_CHECK_FIELDS:
            _raise(error_type, "semantic check fields are invalid")
        check_id = _require_text(
            raw["check_id"], field="semantic.check_id", error_type=error_type
        )
        if raw["status"] != "PASS":
            _raise(error_type, "semantic evidence may contain only PASS checks")
        _require_text(
            raw["summary"],
            field=f"semantic.{check_id}.summary",
            error_type=error_type,
        )
        if not isinstance(raw["details"], Mapping):
            _raise(error_type, "semantic check details must be an object")
        canonical_json_bytes(dict(raw["details"]))
        observed.append(check_id)
    if len(observed) != len(set(observed)):
        _raise(error_type, "semantic check IDs must be unique")
    if expected_check_ids is not None and tuple(observed) != expected_check_ids:
        _raise(error_type, "semantic check set/order does not match G3")
    if type(value["network_attempts"]) is not int or value["network_attempts"] != 0:
        _raise(error_type, "semantic network_attempts must be exactly zero")
    _require_timestamp(
        value["checked_at"], field="semantic.checked_at", error_type=error_type
    )
    commit = _require_text(
        value["generator_git_commit"],
        field="semantic.generator_git_commit",
        error_type=error_type,
    )
    if _GIT_COMMIT.fullmatch(commit) is None:
        _raise(error_type, "semantic generator commit is invalid")
    declared = _require_sha256(
        value["artifact_hash"],
        field="semantic.artifact_hash",
        error_type=error_type,
    )
    if declared != semantic_evidence_artifact_hash(value):
        _raise(error_type, "semantic artifact_hash mismatch")


__all__ = [
    "G3SemanticEvidenceError",
    "SEMANTIC_EVIDENCE_SCHEMA_VERSION",
    "semantic_evidence_artifact_hash",
    "validate_semantic_evidence",
]
