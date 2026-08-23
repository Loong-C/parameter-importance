"""Fail-closed consumer for the immutable Stage 1 G1-EXIT handoff.

Stage 2 must consume the formal S1.11 evidence closure, never a tracked fixture
or an unqualified gate candidate.  This module intentionally validates the
small final-role closure needed by Stage 2 without re-running Stage 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

from .jsonio import canonical_json_hash, load_canonical_json


STAGE1_G1_EXIT_PRODUCER_COMMIT = "3f18b04df8922be9894678ae4842bd999c7e8fd5"
STAGE1_G1_EXIT_TASK_ID = "stage1.11_reporting_and_exit_gate"
STAGE1_G1_EXIT_GATE_ID = "G1-EXIT"
STAGE1_G1_EXIT_INDEX_SCHEMA = "stage1-s1-11-formalization-index-v1"
STAGE1_G1_EXIT_CANONICAL_ROOT = "evidence/stage1/s1-11-formal"
_REQUIRED_ROLES = (
    "formal_observation",
    "stage_report",
    "delivery_manifest",
    "gate_summary",
    "requirements_matrix",
    "validation",
    "replay_validation",
)
_REQUIRED_REQUIREMENTS = tuple(f"S1.11-R{index:02d}" for index in range(1, 29))


class Stage1HandoffError(ValueError):
    """Raised when a Stage 1 formal handoff is absent, altered, or ineligible."""


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_HASH_INVALID")
    return value


def _commit(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_COMMIT_INVALID")
    return value


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_OBJECT_REQUIRED")
    return value


def _safe_relative(root: Path, ref: object, *, field: str) -> Path:
    if not isinstance(ref, str) or not ref or "\\" in ref:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_REF_INVALID")
    logical = PurePosixPath(ref)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_REF_INVALID")
    candidate = (root / Path(*logical.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1HandoffError(
            f"STAGE1_HANDOFF_{field.upper()}_REF_ESCAPES_ROOT"
        ) from error
    return candidate


def _role_path(index_path: Path, ref: object, *, field: str) -> Path:
    # S1.11 role refs are index-directory relative.  Rejecting a parent or an
    # absolute ref prevents a caller from silently mixing two evidence roots.
    return _safe_relative(index_path.parent, ref, field=field)


def _read(path: Path, *, field: str) -> Mapping[str, object]:
    try:
        return _object(load_canonical_json(path), field=field)
    except (FileNotFoundError, OSError) as error:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_MISSING") from error
    except (TypeError, ValueError) as error:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_JSON_INVALID") from error


def _self_hash(value: Mapping[str, object], *, field: str) -> str:
    supplied = _sha256(value.get("artifact_hash"), field=f"{field}.artifact")
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_ARTIFACT_HASH_MISMATCH")
    return supplied


def _file_hash(path: Path, *, expected: object, field: str) -> str:
    digest = _sha256(expected, field=field)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != digest:
        raise Stage1HandoffError(
            f"STAGE1_HANDOFF_{field.upper().replace('.', '_')}_FILE_HASH_MISMATCH"
        )
    return digest


def _identity(value: Mapping[str, object], *, field: str, schema: str) -> None:
    if value.get("schema_version") != schema:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_SCHEMA_INVALID")
    if value.get("status") != "PASS":
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_STATUS_NOT_PASS")
    if value.get("task_id") != STAGE1_G1_EXIT_TASK_ID:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_TASK_INVALID")
    if value.get("gate_id") != STAGE1_G1_EXIT_GATE_ID:
        raise Stage1HandoffError(f"STAGE1_HANDOFF_{field.upper()}_GATE_INVALID")


@dataclass(frozen=True, slots=True)
class Stage1ExitEvidence:
    """Validated identity of the immutable S1.11 formal evidence closure."""

    index_ref: str
    index_sha256: str
    index_artifact_hash: str
    producer_commit: str
    execution_commit: str
    formal_observation_ref: str
    formal_observation_sha256: str
    formal_observation_artifact_hash: str
    role_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _sha256(self.index_sha256, field="index")
        _sha256(self.index_artifact_hash, field="index_artifact")
        _sha256(self.formal_observation_sha256, field="formal_observation")
        _sha256(self.formal_observation_artifact_hash, field="formal_observation_artifact")
        _commit(self.producer_commit, field="producer")
        _commit(self.execution_commit, field="execution")

    def to_dict(self) -> dict[str, object]:
        return {
            "index_ref": self.index_ref,
            "index_sha256": self.index_sha256,
            "index_artifact_hash": self.index_artifact_hash,
            "producer_commit": self.producer_commit,
            "execution_commit": self.execution_commit,
            "formal_observation_ref": self.formal_observation_ref,
            "formal_observation_sha256": self.formal_observation_sha256,
            "formal_observation_artifact_hash": self.formal_observation_artifact_hash,
            "role_sha256": dict(self.role_sha256),
        }


def validate_stage1_exit_evidence(
    root: Path,
    index_ref: str,
    *,
    expected_producer: str = STAGE1_G1_EXIT_PRODUCER_COMMIT,
    evidence_root: Path | None = None,
) -> Stage1ExitEvidence:
    """Validate and return the exact Stage 1 G1-EXIT handoff identity.

    ``index_ref`` is workspace-relative and must point below the canonical
    ``evidence/stage1/s1-11-formal/<producer>/<attempt>/`` root.  The validator
    checks every role hash before inspecting role semantics, rejects tracked or
    temporary fixture paths, and binds both generator and consumer/execution
    commit to the released producer.  A caller using ``DATA_ROOT`` therefore
    passes ``evidence/...``; it must not pass a ``tmp/...`` attempt path.
    """

    if expected_producer != STAGE1_G1_EXIT_PRODUCER_COMMIT:
        raise Stage1HandoffError("STAGE1_HANDOFF_PRODUCER_POLICY_CANNOT_BE_RELAXED")
    path_parts = PurePosixPath(index_ref).parts
    if any(
        path_parts[index : index + 2] in (("reports", "stage1"), ("fixtures", "stage1"))
        for index in range(max(0, len(path_parts) - 1))
    ):
        # Tracked Stage 1 reports/fixtures may be NOT_RUN and are never formal
        # authority, even if a caller places a PASS-looking object there.
        raise Stage1HandoffError("STAGE1_HANDOFF_TRACKED_FIXTURE_REF_FORBIDDEN")
    canonical_root_parts = PurePosixPath(STAGE1_G1_EXIT_CANONICAL_ROOT).parts
    if path_parts[: len(canonical_root_parts)] != canonical_root_parts:
        if (
            path_parts
            and path_parts[0] == "tmp"
            and len(path_parts) > 1
            and path_parts[1].startswith("stage1")
        ):
            raise Stage1HandoffError("STAGE1_HANDOFF_TEMP_REF_FORBIDDEN")
        raise Stage1HandoffError("STAGE1_HANDOFF_CANONICAL_ROOT_REQUIRED")
    if len(path_parts) < 6 or path_parts[3] != expected_producer:
        raise Stage1HandoffError("STAGE1_HANDOFF_CANONICAL_PRODUCER_PATH_INVALID")
    # Formal callers pass the explicitly approved DATA_ROOT as ``evidence_root``;
    # workspace/result references remain rooted at ``root``.  Keeping the two
    # roots separate prevents a mutable repo/tmp copy from becoming authority.
    index_path = _safe_relative(
        evidence_root if evidence_root is not None else root,
        index_ref,
        field="index",
    )
    try:
        index = _read(index_path, field="index")
        index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    except OSError as error:
        raise Stage1HandoffError("STAGE1_HANDOFF_INDEX_MISSING") from error
    _self_hash(index, field="index")
    if index.get("schema_version") != STAGE1_G1_EXIT_INDEX_SCHEMA:
        raise Stage1HandoffError("STAGE1_HANDOFF_INDEX_SCHEMA_INVALID")
    _identity(index, field="index", schema=STAGE1_G1_EXIT_INDEX_SCHEMA)
    if index.get("generator_git_commit") != expected_producer:
        raise Stage1HandoffError("STAGE1_HANDOFF_GENERATOR_COMMIT_MISMATCH")
    if index.get("consumer_git_commit") != expected_producer:
        raise Stage1HandoffError("STAGE1_HANDOFF_CONSUMER_COMMIT_MISMATCH")
    if index.get("next_task_ids") != ["stage2", "stage3"]:
        raise Stage1HandoffError("STAGE1_HANDOFF_NEXT_TASKS_INVALID")

    role_refs = _object(index.get("role_refs"), field="index.role_refs")
    role_hashes = _object(index.get("role_sha256"), field="index.role_sha256")
    # Released r4 keeps validation outside ``role_refs``: its dedicated
    # validation_ref/validation_sha256 fields are part of the producer index
    # wire.  Older local/compatibility fixtures put validation in role_refs;
    # retain that path, but never accept an unbound or unhashed validation file.
    validation_in_roles = "validation" in role_refs or "validation" in role_hashes
    if validation_in_roles and not (
        "validation" in role_refs and "validation" in role_hashes
    ):
        raise Stage1HandoffError("STAGE1_HANDOFF_ROLE_VALIDATION_MISSING")
    validation_from_index = not validation_in_roles
    if validation_from_index:
        if index.get("validation_ref") != "validation.json":
            raise Stage1HandoffError("STAGE1_HANDOFF_ROLE_VALIDATION_REF_INVALID")
        if "validation_sha256" not in index:
            raise Stage1HandoffError("STAGE1_HANDOFF_ROLE_VALIDATION_HASH_MISSING")
    for role in _REQUIRED_ROLES:
        if role == "validation" and validation_from_index:
            continue
        if role not in role_refs or role not in role_hashes:
            raise Stage1HandoffError(f"STAGE1_HANDOFF_ROLE_{role.upper()}_MISSING")
    loaded: dict[str, Mapping[str, object]] = {}
    normalized_hashes: dict[str, str] = {}
    for role in _REQUIRED_ROLES:
        if role == "validation" and validation_from_index:
            role_ref = index["validation_ref"]
            expected_hash = index["validation_sha256"]
        else:
            role_ref = role_refs[role]
            expected_hash = role_hashes[role]
        role_path = _role_path(index_path, role_ref, field=f"role.{role}")
        try:
            digest = _file_hash(role_path, expected=expected_hash, field=f"role.{role}")
        except OSError as error:
            raise Stage1HandoffError(f"STAGE1_HANDOFF_ROLE_{role.upper()}_MISSING") from error
        loaded[role] = _read(role_path, field=f"role.{role}")
        _self_hash(loaded[role], field=f"role.{role}")
        normalized_hashes[role] = digest

    formal_observation = loaded["formal_observation"]
    _identity(
        formal_observation,
        field="formal_observation",
        schema="stage1-s1-11-formal-observation-v1",
    )
    execution_commit = _commit(
        formal_observation.get("execution_commit"), field="execution"
    )
    if execution_commit != expected_producer:
        raise Stage1HandoffError("STAGE1_HANDOFF_EXECUTION_COMMIT_MISMATCH")

    _identity(loaded["stage_report"], field="stage_report", schema="stage1-s1-11-stage-report-v1")
    delivery = loaded["delivery_manifest"]
    if (
        delivery.get("schema_version") != "stage1-s1-11-delivery-manifest-v1"
        or delivery.get("task_id") != STAGE1_G1_EXIT_TASK_ID
        or delivery.get("gate_id") != STAGE1_G1_EXIT_GATE_ID
    ):
        raise Stage1HandoffError("STAGE1_HANDOFF_DELIVERY_MANIFEST_IDENTITY_INVALID")
    gate_summary = loaded["gate_summary"]
    _identity(gate_summary, field="gate_summary", schema="stage1-s1-11-gate-summary-v1")
    if gate_summary.get("unresolved_failure_count") != 0:
        raise Stage1HandoffError("STAGE1_HANDOFF_UNRESOLVED_FAILURES")

    matrix = loaded["requirements_matrix"]
    if (
        matrix.get("schema_version") != "stage1-s1-11-requirements-matrix-v1"
        or matrix.get("task_id") != STAGE1_G1_EXIT_TASK_ID
        or matrix.get("gate_id") != STAGE1_G1_EXIT_GATE_ID
    ):
        raise Stage1HandoffError("STAGE1_HANDOFF_REQUIREMENTS_IDENTITY_INVALID")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != len(_REQUIRED_REQUIREMENTS):
        raise Stage1HandoffError("STAGE1_HANDOFF_REQUIREMENTS_COUNT_INVALID")
    observed_requirements: list[str] = []
    for row in rows:
        item = _object(row, field="requirements_matrix.row")
        requirement_id = item.get("requirement_id")
        if not isinstance(requirement_id, str) or item.get("status") != "PASS":
            raise Stage1HandoffError("STAGE1_HANDOFF_REQUIREMENTS_NOT_ALL_PASS")
        observed_requirements.append(requirement_id)
    if tuple(observed_requirements) != _REQUIRED_REQUIREMENTS:
        raise Stage1HandoffError("STAGE1_HANDOFF_REQUIREMENTS_ORDER_INVALID")

    validation = loaded["validation"]
    if validation_from_index:
        if (
            validation.get("schema_version") != "stage1-s1-11-validation-v1"
            or validation.get("status") != "PASS"
            or validation.get("task_id") != STAGE1_G1_EXIT_TASK_ID
        ):
            raise Stage1HandoffError("STAGE1_HANDOFF_VALIDATION_IDENTITY_INVALID")
    else:
        _identity(validation, field="validation", schema="stage1-s1-11-validation-v1")
    replay = loaded["replay_validation"]
    if replay.get("schema_version") != "stage1-s1-11-replay-validation-v1" or replay.get("status") != "PASS":
        raise Stage1HandoffError("STAGE1_HANDOFF_REPLAY_INVALID")

    formal_ref = str(role_refs["formal_observation"])
    return Stage1ExitEvidence(
        index_ref=index_ref,
        index_sha256=index_sha256,
        index_artifact_hash=_sha256(index.get("artifact_hash"), field="index_artifact"),
        producer_commit=expected_producer,
        execution_commit=execution_commit,
        formal_observation_ref=formal_ref,
        formal_observation_sha256=normalized_hashes["formal_observation"],
        formal_observation_artifact_hash=_sha256(
            formal_observation.get("artifact_hash"), field="formal_observation_artifact"
        ),
        role_sha256=tuple(sorted(normalized_hashes.items())),
    )


__all__ = [
    "STAGE1_G1_EXIT_CANONICAL_ROOT",
    "STAGE1_G1_EXIT_GATE_ID",
    "STAGE1_G1_EXIT_INDEX_SCHEMA",
    "STAGE1_G1_EXIT_PRODUCER_COMMIT",
    "STAGE1_G1_EXIT_TASK_ID",
    "Stage1ExitEvidence",
    "Stage1HandoffError",
    "validate_stage1_exit_evidence",
]
