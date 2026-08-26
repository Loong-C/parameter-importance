"""Strict Stage 2.11 delivery/replay control plane.

This module is intentionally a *consumer*.  It never runs a model, creates a
draw stream, or turns a local fixture into formal evidence.  The formal
server invocation supplies immutable S2.10/G2.7b output, the complete Stage 2
lineage, boundary manifests, and the result of one independently executed
31M confirmatory repetition replay.  Missing or stale inputs produce a
content-addressed ``BLOCKED`` delivery bundle; no replay success is inferred.

The public entry point is :func:`run_s211_g28`.  Output publication is
append-only: the destination must not exist, and a temporary sibling is
atomically renamed into place only after every JSON artifact and hash has
been verified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid
from typing import Any, Mapping, Sequence

from ..contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from ..contracts.status import GateRecord, GateStatus


S211_TASK_ID = "stage2.11_delivery_and_exit_gate"
S211_GATE_ID = "stage2.G2.8"
S211_DELIVERY_SCHEMA = "stage2-s211-delivery-manifest-v1"
S211_REPLAY_INSTRUCTIONS_SCHEMA = "stage2-s211-replay-instructions-v1"
S211_REPLAY_VALIDATOR_SCHEMA = "stage2-s211-replay-validator-v1"
S211_REPLAY_AUDIT_SCHEMA = "stage2-s211-replay-audit-v1"
S211_LINEAGE_SCHEMA = "stage2-s211-lineage-v1"
S211_GATE_SCHEMA = "stage2-s211-g28-gate-v1"
S211_OUTPUT_INVENTORY_SCHEMA = "stage2-s211-output-inventory-v1"
S210_GATE_SCHEMA = "stage2-s210-g27b-gate-v1"
S210_DECISION_SCHEMA = "stage2-s210-estimator-decision-v1"
S208_GATE_SCHEMA = "stage2-s208-g26-gate-v1"
S25_GATE_SCHEMA = "stage2-g24a-formal-evaluation-v1"
S23_GATE_SCHEMA = "stage2-g23-reference-evaluation-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_PRIMARY_CELL_IDS = (
    "pythia-14m:initialization",
    "pythia-14m:early",
    "pythia-14m:mid_late",
    "pythia-31m-deduped:initialization",
    "pythia-31m-deduped:early",
    "pythia-31m-deduped:mid_late",
)
_G26_PRIMARY_GATES = ("stage2.G2.3", "stage2.G2.4a", "stage2.G2.4b", "stage2.G2.5")
_STAGE2_TASKS = tuple(
    f"stage2.{index:02d}_{name}"
    for index, name in (
        (1, "scope_hypotheses_and_preregistration"),
        (2, "stage1_handoff_and_fixed_state_contract"),
        (3, "assets_checkpoints_and_sampling"),
        (4, "reference_target"),
        (5, "paired_estimator_runner"),
        (6, "pilot_and_matrix_freeze"),
        (7, "main_sweep"),
        (8, "statistics_and_robustness"),
        (9, "cost_and_system_validation"),
        (10, "visualization_reporting_and_decision"),
    )
)
_BOUNDARY_ROLES = (
    "environment",
    "assets",
    "reference",
    "pilot",
    "formal_14m",
    "formal_31m",
    "analysis",
    "decision",
)
_UPSTREAM_GATES = (
    "stage2.G2.0",
    "stage2.G2.1",
    "stage2.G2.2",
    "stage2.G2.3",
    "stage2.G2.4a",
    "stage2.G2.4b",
    "stage2.G2.5",
    "stage2.G2.6",
    "stage2.G2.7a",
    "stage2.G2.7b",
)
_DELIVERY_ROLES = (
    "plan",
    "task_catalog",
    "replay_report",
    "gate_summary",
    "sync_report",
    "estimator_decision",
    "large_artifact_index",
    "worklog",
    "dirty_head_evidence",
    "failure_retry_amendment_history",
)
_AGENT_DOCUMENTS = ("git.md", "local.md", "remote_access.md", "server.md", "sync.md", "worklogs.md")
_ROLE_SCHEMAS = {
    "plan": "stage2-s211-plan-v1",
    "task_catalog": "task-catalog-v1",
    "replay_report": "stage2-s211-replay-report-v1",
    "gate_summary": "stage2-s211-gate-summary-v1",
    "sync_report": "stage2-s211-sync-report-v1",
    "estimator_decision": "estimator-decision-v1",
    "large_artifact_index": "stage2-s211-large-artifact-index-v1",
    "worklog": "stage2-s211-worklog-v1",
    "dirty_head_evidence": "stage2-s211-dirty-head-v1",
    "failure_retry_amendment_history": "stage2-s211-history-v1",
}
_DECISION_FIELDS = {
    "schema_version",
    "decision_id",
    "selected_estimator",
    "scope",
    "status",
    "state",
    "batch_size",
    "microbatch_count",
    "repetitions",
    "gate_id",
    "gate_status",
    "artifact_ref",
    "metadata",
    "artifact_hash",
}


class S211DeliveryBlocked(RuntimeError):
    """Raised for unsupported arguments, malformed artifacts, or bad output roots."""


# Naming aliases follow the S2.8--S2.10 producer convention and keep detached
# server wrappers/tests free to use either the task or Gate spelling.
S211G28Blocked = S211DeliveryBlocked


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S211DeliveryBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise S211DeliveryBlocked(f"{field}:FULL_COMMIT_REQUIRED")
    return value


def _payload_identity(payload: Mapping[str, Any], *, field: str) -> tuple[str, bool]:
    """Return the canonical identity and whether a producer hash was declared."""

    declared = payload.get("artifact_hash")
    body = {key: item for key, item in payload.items() if key != "artifact_hash"}
    try:
        computed = canonical_json_hash(body if declared is not None else payload)
    except (TypeError, ValueError) as error:
        raise S211DeliveryBlocked(f"{field}:CANONICAL_HASH_FAILED") from error
    if declared is not None:
        _sha(declared, field=f"{field}.artifact_hash")
        if declared != computed:
            raise S211DeliveryBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
        return str(declared), True
    return computed, False


def _data_root(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise S211DeliveryBlocked(f"DATA_ROOT_INVALID:{type(error).__name__}") from error
    if not resolved.is_dir() or candidate.absolute() != resolved:
        raise S211DeliveryBlocked("DATA_ROOT_MUST_BE_REAL_DIRECTORY")
    return resolved


def _inside_data_root(value: str | Path, *, root: Path, field: str, must_exist: bool = True) -> Path:
    candidate = Path(value)
    lexical = candidate.absolute()
    try:
        lexical_common = os.path.commonpath((os.path.normcase(str(root)), os.path.normcase(str(lexical))))
    except ValueError as error:
        raise S211DeliveryBlocked(f"{field}:DATA_ROOT_PATH_ESCAPE") from error
    if lexical_common != os.path.normcase(str(root)):
        raise S211DeliveryBlocked(f"{field}:DATA_ROOT_PATH_ESCAPE")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as error:
        raise S211DeliveryBlocked(f"{field}:DATA_ROOT_REF_UNREADABLE") from error
    try:
        common = os.path.commonpath((os.path.normcase(str(root)), os.path.normcase(str(resolved))))
    except ValueError as error:
        raise S211DeliveryBlocked(f"{field}:DATA_ROOT_PATH_ESCAPE") from error
    if common != os.path.normcase(str(root)):
        raise S211DeliveryBlocked(f"{field}:DATA_ROOT_PATH_ESCAPE")
    # A symlink inside DATA_ROOT is not an acceptable formal reference even
    # when its resolved target happens to land back under DATA_ROOT.
    if candidate.absolute() != resolved:
        raise S211DeliveryBlocked(f"{field}:SYMLINK_REF_FORBIDDEN")
    return resolved


def _resolve_data_root_ref(
    value: str | Path,
    *,
    root: Path,
    field: str,
    must_exist: bool = True,
) -> Path:
    """Resolve a ref relative to DATA_ROOT before applying containment checks."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _inside_data_root(candidate, root=root, field=field, must_exist=must_exist)


def _check_declared_refs(values: Any, *, root: Path, field: str) -> None:
    """Reject absolute/escaping evidence refs without dereferencing labels.

    Gate evidence often uses stable relative object names that do not exist on
    the local consumer host.  Those labels are accepted, while every absolute
    path and every existing relative path is still checked for DATA_ROOT and
    symlink escape.
    """

    if not isinstance(values, (list, tuple)):
        return
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise S211DeliveryBlocked(f"{field}[{index}]:REF_STRING_REQUIRED")
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate
        _inside_data_root(candidate, root=root, field=f"{field}[{index}]", must_exist=False)


def _check_declared_ref(value: Any, *, root: Path, field: str) -> None:
    if not isinstance(value, str):
        return
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    _inside_data_root(candidate, root=root, field=field, must_exist=False)


def _ref_text(value: Mapping[str, Any] | str | Path | None) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return value
    return "inline"


def _read(
    value: Mapping[str, Any] | str | Path | None,
    *,
    field: str,
    data_root: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Read an inline mapping or strict canonical JSON path.

    ``bool`` says that the object carries an explicit producer artifact_hash.
    A missing value is not exceptional: it is an input blocker and allows the
    caller to publish an auditable BLOCKED bundle before formal assets exist.
    """

    if value is None:
        return None, None, False
    ref = "inline"
    try:
        if isinstance(value, Mapping):
            payload = dict(value)
        else:
            if data_root is None:
                raise S211DeliveryBlocked(f"{field}:DATA_ROOT_REQUIRED")
            # CLI references are logical paths rooted at DATA_ROOT.  Absolute
            # paths are still accepted only when they resolve inside that
            # root; resolving relative references against the process cwd
            # would make an otherwise valid server invocation fail closed (or,
            # worse, make its meaning depend on the launch directory).
            path = _resolve_data_root_ref(value, root=data_root, field=field)
            ref = path.as_posix()
            loaded = load_canonical_json(path)
            if not isinstance(loaded, Mapping):
                raise S211DeliveryBlocked(f"{field}:OBJECT_REQUIRED")
            payload = dict(loaded)
        identity, declared = _payload_identity(payload, field=field)
        return payload, identity, declared
    except S211DeliveryBlocked:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise S211DeliveryBlocked(f"{field}:CANONICAL_READ_FAILED:{type(error).__name__}") from error


def _status(payload: Mapping[str, Any]) -> str | None:
    for key in ("status", "gate_status", "state", "phase_status"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.upper()
    gate = payload.get("gate")
    if isinstance(gate, Mapping):
        value = gate.get("status")
        if isinstance(value, str):
            return value.upper()
    return None


def _forbidden_formal_source(payload: Mapping[str, Any]) -> bool:
    """Return true for explicit fixture/synthetic provenance markers."""

    return (
        payload.get("local_fixture") is True
        or payload.get("synthetic") is True
        or payload.get("fixture") is True
        or payload.get("scope") == "local_fixture"
    )


def _formal_eligible(payload: Mapping[str, Any]) -> bool:
    return not _forbidden_formal_source(payload) and (
        payload.get("formal_eligible") is True or payload.get("scope") == "formal"
    )


def _gate_record(
    value: Mapping[str, Any] | None,
    *,
    gate_id: str,
    field: str,
    data_root: Path | None = None,
) -> tuple[str | None, list[str]]:
    """Parse one immutable upstream GateRecord and bind its canonical hash.

    Server evidence commonly has a task-output wrapper around the GateRecord.
    The wrapper itself is hash checked first; the nested record is then parsed
    by the strict GateRecord contract.  A wrapper can never turn a local or
    unsealed record into a formal predecessor.
    """

    if value is None:
        return None, [f"UPSTREAM_GATE_MISSING:{gate_id}"]
    reasons: list[str] = []
    candidate = dict(value)
    outer_hash: str | None = None
    try:
        outer_hash, outer_declared = _payload_identity(candidate, field=field)
    except S211DeliveryBlocked as error:
        return None, [str(error)]
    if not outer_declared:
        reasons.append(f"UPSTREAM_GATE_ARTIFACT_HASH_MISSING:{gate_id}")
    nested = candidate.get("payload")
    wrapped = isinstance(nested, Mapping)
    if wrapped:
        if candidate.get("formal_eligible") is not True:
            reasons.append(f"UPSTREAM_GATE_NOT_FORMAL:{gate_id}")
        candidate = dict(nested)
    try:
        record = GateRecord.from_mapping(candidate)
    except Exception as error:
        return None, reasons + [f"UPSTREAM_GATE_INVALID:{gate_id}:{type(error).__name__}"]
    if record.gate_id != gate_id or record.stage != 2:
        reasons.append(f"UPSTREAM_GATE_ID_MISMATCH:{gate_id}")
    if record.effective_status() is not GateStatus.PASS:
        reasons.append(f"UPSTREAM_GATE_NOT_PASS:{gate_id}")
    if _forbidden_formal_source(value):
        reasons.append(f"UPSTREAM_GATE_FORBIDDEN_FIXTURE:{gate_id}")
    if data_root is not None:
        try:
            _check_declared_refs(record.evidence_refs, root=data_root, field=f"{field}.evidence_refs")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    if outer_hash is not None and not wrapped and outer_hash != record.artifact_hash:
        reasons.append(f"UPSTREAM_GATE_HASH_MISMATCH:{gate_id}")
    # Bind the GateRecord payload hash in the manifest.  The wrapper hash was
    # already checked above and remains part of the input evidence boundary;
    # downstream gates must reason about the immutable nested record itself.
    return record.artifact_hash, sorted(set(reasons))


def _producer_gate_identity(
    value: Mapping[str, Any] | None,
    *,
    gate_id: str,
    schema: str,
    field: str,
) -> tuple[str | None, list[str]]:
    """Check the common identity/status contract of a producer gate."""

    if value is None:
        return None, [f"UPSTREAM_GATE_MISSING:{gate_id}"]
    if value.get("schema_version") != schema:
        return None, [f"UPSTREAM_GATE_SCHEMA_MISMATCH:{gate_id}"]
    try:
        identity, declared = _payload_identity(value, field=field)
    except S211DeliveryBlocked as error:
        return None, [str(error)]
    reasons: list[str] = []
    if not declared:
        reasons.append(f"UPSTREAM_GATE_ARTIFACT_HASH_MISSING:{gate_id}")
    if value.get("gate_id") != gate_id:
        reasons.append(f"UPSTREAM_GATE_ID_MISMATCH:{gate_id}")
    if value.get("status") != "PASS":
        reasons.append(f"UPSTREAM_GATE_NOT_PASS:{gate_id}")
    if _forbidden_formal_source(value):
        reasons.append(f"UPSTREAM_GATE_FORBIDDEN_FIXTURE:{gate_id}")
    return identity, reasons


def _validate_g23(
    value: Mapping[str, Any] | None,
    *,
    field: str,
    data_root: Path | None = None,
) -> tuple[str | None, list[str]]:
    """Validate the exact producer-specific S2.4/G2.3 qualification object."""

    identity, reasons = _producer_gate_identity(
        value,
        gate_id="stage2.G2.3",
        schema=S23_GATE_SCHEMA,
        field=field,
    )
    if value is None or value.get("schema_version") != S23_GATE_SCHEMA:
        return identity, sorted(set(reasons))
    if value.get("formal_eligible") is not True:
        reasons.append("UPSTREAM_GATE_NOT_FORMAL:stage2.G2.3")
    if (
        value.get("required_cell_count") != len(_PRIMARY_CELL_IDS)
        or value.get("complete_cell_count") != len(_PRIMARY_CELL_IDS)
        or value.get("expected_cell_ids") != list(_PRIMARY_CELL_IDS)
    ):
        reasons.append("G2.3_SIX_CELL_COMPLETENESS_REQUIRED")
    cells = value.get("cells")
    if (
        not isinstance(cells, list)
        or len(cells) != len(_PRIMARY_CELL_IDS)
        or tuple(item.get("cell_id") if isinstance(item, Mapping) else None for item in cells)
        != _PRIMARY_CELL_IDS
    ):
        reasons.append("G2.3_CELL_ORDER_INVALID")
    elif any(
        not isinstance(item, Mapping)
        or item.get("status") != "PASS"
        or item.get("formal_eligible") is False
        for item in cells
    ):
        reasons.append("G2.3_CELL_PASS_REQUIRED")
    evidence_refs = value.get("evidence_refs")
    if data_root is not None and isinstance(evidence_refs, (list, tuple)):
        try:
            _check_declared_refs(evidence_refs, root=data_root, field=f"{field}.evidence_refs")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    return identity, sorted(set(reasons))


def _validate_g24a(
    value: Mapping[str, Any] | None,
    *,
    field: str,
    data_root: Path | None = None,
) -> tuple[str | None, list[str]]:
    """Validate the exact producer-specific S2.5/G2.4a qualification object."""

    identity, reasons = _producer_gate_identity(
        value,
        gate_id="stage2.G2.4a",
        schema=S25_GATE_SCHEMA,
        field=field,
    )
    if value is None or value.get("schema_version") != S25_GATE_SCHEMA:
        return identity, sorted(set(reasons))
    if value.get("formal_eligible") is not True:
        reasons.append("UPSTREAM_GATE_NOT_FORMAL:stage2.G2.4a")
    if value.get("cell_count") != len(_PRIMARY_CELL_IDS):
        reasons.append("G2.4A_SIX_CELL_COMPLETENESS_REQUIRED")
    results = value.get("results")
    if (
        not isinstance(results, list)
        or len(results) != len(_PRIMARY_CELL_IDS)
        or tuple(item.get("cell_id") if isinstance(item, Mapping) else None for item in results)
        != _PRIMARY_CELL_IDS
    ):
        reasons.append("G2.4A_CELL_ORDER_INVALID")
    elif any(
        not isinstance(item, Mapping)
        or item.get("status") != "PASS"
        or item.get("formal_eligible") is not True
        for item in results
    ):
        reasons.append("G2.4A_CELL_PASS_REQUIRED")
    execution_commit = value.get("execution_commit")
    if not isinstance(execution_commit, str) or _COMMIT.fullmatch(execution_commit) is None:
        reasons.append("G2.4A_EXECUTION_COMMIT_REQUIRED")
    evidence_refs = value.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not evidence_refs:
        reasons.append("G2.4A_EVIDENCE_REFS_MISSING")
    elif data_root is not None:
        try:
            _check_declared_refs(evidence_refs, root=data_root, field=f"{field}.evidence_refs")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    if value.get("confirmatory_draws_generated") is not False:
        reasons.append("G2.4A_CONFIRMATORY_DRAWS_FORBIDDEN")
    for name in ("rebind_plan_hash", "g23_evaluation_hash"):
        try:
            _sha(value.get(name), field=f"{field}.{name}")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    return identity, sorted(set(reasons))


def _validate_g26(
    value: Mapping[str, Any] | None,
    *,
    field: str,
) -> tuple[str | None, list[str]]:
    """Validate the exact producer-specific S2.8/G2.6 quality gate."""

    identity, reasons = _producer_gate_identity(
        value,
        gate_id="stage2.G2.6",
        schema=S208_GATE_SCHEMA,
        field=field,
    )
    if value is None or value.get("schema_version") != S208_GATE_SCHEMA:
        return identity, sorted(set(reasons))
    if value.get("stage") != 2:
        reasons.append("G2.6_STAGE_MISMATCH")
    if value.get("quality_gate_dependency") is not True:
        reasons.append("G2.6_QUALITY_GATE_DEPENDENCY_REQUIRED")
    if not isinstance(value.get("measured"), Mapping):
        reasons.append("G2.6_MEASURED_REQUIRED")
    if not isinstance(value.get("threshold"), Mapping):
        reasons.append("G2.6_THRESHOLD_REQUIRED")
    if value.get("reasons") != []:
        reasons.append("G2.6_REASONS_MUST_BE_EMPTY_FOR_PASS")
    upstream = value.get("upstream_gate_hashes")
    if not isinstance(upstream, Mapping) or set(upstream) != set(_G26_PRIMARY_GATES):
        reasons.append("G2.6_UPSTREAM_GATE_SET_INVALID")
    elif any(
        not isinstance(upstream.get(gate_id), str)
        or _SHA256.fullmatch(upstream.get(gate_id, "")) is None
        for gate_id in _G26_PRIMARY_GATES
    ):
        reasons.append("G2.6_UPSTREAM_GATE_HASHES_INVALID")
    return identity, sorted(set(reasons))


def _validate_s210_decision(
    value: Mapping[str, Any], *, field: str
) -> tuple[str | None, bool, list[str]]:
    """Validate the producer-specific S2.10 decision wire object.

    S2.10 intentionally uses ``stage2-s210-estimator-decision-v1`` while the
    cross-stage estimator contract uses ``estimator-decision-v1``.  Both are
    immutable, but they are not interchangeable schemas; accepting the
    producer object here avoids making the final delivery gate depend on a
    lossy or hand-transcribed conversion.
    """

    reasons: list[str] = []
    identity: str | None = None
    try:
        identity, declared = _payload_identity(value, field=field)
    except S211DeliveryBlocked as error:
        return None, False, [str(error)]
    if not declared:
        reasons.append(f"{field}:ARTIFACT_HASH_MISSING")
    if set(value) != _DECISION_FIELDS:
        reasons.append(f"{field}:FIELDS_INVALID")
    if value.get("schema_version") != S210_DECISION_SCHEMA:
        reasons.append(f"{field}:SCHEMA_MISMATCH")
    if _forbidden_formal_source(value):
        reasons.append(f"{field}:FORBIDDEN_FIXTURE")
    if not isinstance(value.get("decision_id"), str) or not value.get("decision_id"):
        reasons.append(f"{field}:DECISION_ID_REQUIRED")
    if value.get("selected_estimator") not in {"u", "weighted_u", "double"}:
        reasons.append(f"{field}:ESTIMATOR_INVALID")
    if value.get("scope") != "formal":
        reasons.append(f"{field}:SCOPE_NOT_FORMAL")
    if value.get("status") not in {"SELECTED", "PASS", "QUALIFIED", "READY"}:
        reasons.append(f"{field}:STATUS_NOT_PASS")
    if value.get("state") not in {"SELECTED", "FROZEN", "PASS", "READY"}:
        reasons.append(f"{field}:STATE_NOT_FROZEN")
    if value.get("gate_id") != "stage2.G2.7b" or value.get("gate_status") != "PASS":
        reasons.append(f"{field}:GATE_NOT_PASS")
    for name, minimum in (("batch_size", 1), ("microbatch_count", 2), ("repetitions", 1)):
        candidate = value.get(name)
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < minimum:
            reasons.append(f"{field}:{name.upper()}_INVALID")
    batch = value.get("batch_size")
    micro = value.get("microbatch_count")
    if isinstance(batch, int) and not isinstance(batch, bool) and isinstance(micro, int) and not isinstance(micro, bool) and micro > 0 and batch % micro:
        reasons.append(f"{field}:BATCH_MICROBATCH_NOT_DIVISIBLE")
    if not isinstance(value.get("artifact_ref"), str) or not value.get("artifact_ref"):
        reasons.append(f"{field}:ARTIFACT_REF_REQUIRED")
    if not isinstance(value.get("metadata"), Mapping):
        reasons.append(f"{field}:METADATA_REQUIRED")
    formal = not reasons
    return identity, formal, sorted(set(reasons))


def _validate_upstream_gates(
    values: Mapping[str, Mapping[str, Any] | str | Path | None] | None,
    *,
    g27b_gate: Mapping[str, Any] | None,
    data_root: Path | None = None,
) -> tuple[dict[str, str | None], list[str]]:
    """Require every explicitly planned predecessor, never G2.8 itself."""

    source = dict(values or {})
    if "stage2.G2.8" in source:
        return {}, ["G2.8_CANNOT_BE_ITS_OWN_PREDECESSOR"]
    unsupported = set(source) - set(_UPSTREAM_GATES)
    if unsupported:
        # The predecessor map is a closed contract.  Ignoring an additional
        # gate would make a caller's claimed upstream set differ from the
        # machine-checked set while still allowing G2.8 to pass.
        return {}, [
            "UPSTREAM_GATE_UNSUPPORTED:" + ",".join(sorted(map(str, unsupported)))
        ]
    if g27b_gate is not None:
        existing = source.get("stage2.G2.7b")
        if existing is not None and existing != g27b_gate:
            return {}, ["G2.7B_GATE_DUPLICATE_MISMATCH"]
        source["stage2.G2.7b"] = g27b_gate
    hashes: dict[str, str | None] = {}
    reasons: list[str] = []
    payloads: dict[str, Mapping[str, Any]] = {}
    for gate_id in _UPSTREAM_GATES:
        raw = source.get(gate_id)
        if raw is not None and not isinstance(raw, Mapping):
            try:
                loaded, _identity, _declared = _read(
                    raw, field=f"upstream.{gate_id}", data_root=data_root
                )
            except S211DeliveryBlocked as error:
                hashes[gate_id] = None
                reasons.append(str(error))
                continue
            raw = loaded
        if isinstance(raw, Mapping):
            payloads[gate_id] = raw
        if gate_id == "stage2.G2.7b":
            # S2.10 has a producer-specific Gate schema in addition to the
            # generic GateRecord wire form.  Keep this compatibility at the
            # boundary, but apply the same exact ID/status/hash checks before
            # it can unlock G2.8.
            identity, g27b_reasons = _validate_g27b(
                raw, field=f"upstream.{gate_id}", data_root=data_root
            )
            hashes[gate_id] = identity
            reasons.extend(g27b_reasons)
            continue
        if isinstance(raw, Mapping) and raw.get("schema_version") == S23_GATE_SCHEMA:
            identity, gate_reasons = _validate_g23(
                raw, field=f"upstream.{gate_id}", data_root=data_root
            )
            hashes[gate_id] = identity
            reasons.extend(gate_reasons)
            continue
        if isinstance(raw, Mapping) and raw.get("schema_version") == S25_GATE_SCHEMA:
            identity, gate_reasons = _validate_g24a(
                raw, field=f"upstream.{gate_id}", data_root=data_root
            )
            hashes[gate_id] = identity
            reasons.extend(gate_reasons)
            continue
        if isinstance(raw, Mapping) and raw.get("schema_version") == S208_GATE_SCHEMA:
            identity, gate_reasons = _validate_g26(raw, field=f"upstream.{gate_id}")
            hashes[gate_id] = identity
            reasons.extend(gate_reasons)
            continue
        identity, gate_reasons = _gate_record(
            raw, gate_id=gate_id, field=f"upstream.{gate_id}", data_root=data_root
        )
        hashes[gate_id] = identity
        reasons.extend(gate_reasons)
    # Producer-specific gates carry predecessor hashes in addition to their
    # own content hash.  Bind those claims to the exact objects supplied in
    # this invocation; syntax-only SHA validation is insufficient because it
    # would permit a valid G2.6/G2.4a artifact from a different run.
    g24a = payloads.get("stage2.G2.4a")
    if isinstance(g24a, Mapping):
        declared_g23 = g24a.get("g23_evaluation_hash")
        actual_g23 = hashes.get("stage2.G2.3")
        if isinstance(declared_g23, str) and actual_g23 is not None and declared_g23 != actual_g23:
            reasons.append("G2.4A_G23_HASH_MISMATCH")
    g26 = payloads.get("stage2.G2.6")
    if isinstance(g26, Mapping):
        declared_upstream = g26.get("upstream_gate_hashes")
        if isinstance(declared_upstream, Mapping):
            for dependency in _G26_PRIMARY_GATES:
                declared_hash = declared_upstream.get(dependency)
                actual_hash = hashes.get(dependency)
                if isinstance(declared_hash, str) and actual_hash is not None and declared_hash != actual_hash:
                    reasons.append(f"G2.6_UPSTREAM_HASH_MISMATCH:{dependency}")
    return hashes, sorted(set(reasons))


def _validate_g27b(
    value: Mapping[str, Any] | None,
    *,
    field: str,
    data_root: Path | None = None,
) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if value is None:
        return None, ["G2.7B_GATE_MISSING"]
    # A server task-output may wrap the producer Gate.  Verify the wrapper
    # identity and eligibility, then validate the nested S2.10 object instead
    # of treating the wrapper as a different Gate schema.
    nested = value.get("payload")
    if isinstance(nested, Mapping):
        expected_wrapper_fields = {
            "schema_version",
            "task_id",
            "artifact_kind",
            "config_hash",
            "run_intent",
            "formal_eligible",
            "source_refs",
            "payload",
            "artifact_hash",
        }
        if value.get("schema_version") != "task-output-artifact-v1":
            return None, ["G2.7B_WRAPPER_SCHEMA_MISMATCH"]
        if set(value) != expected_wrapper_fields:
            return None, ["G2.7B_WRAPPER_FIELDS_INVALID"]
        if (
            value.get("task_id") != "stage2.10_visualization_reporting_and_decision"
            or not isinstance(value.get("artifact_kind"), str)
            or not value.get("artifact_kind")
            or value.get("run_intent") != "formal"
            or value.get("formal_eligible") is not True
            or not isinstance(value.get("config_hash"), str)
            or _SHA256.fullmatch(value["config_hash"]) is None
        ):
            return None, ["G2.7B_WRAPPER_IDENTITY_INVALID"]
        source_refs = value.get("source_refs")
        if (
            not isinstance(source_refs, list)
            or not source_refs
            or any(not isinstance(ref, str) or not ref for ref in source_refs)
            or len(source_refs) != len(set(source_refs))
        ):
            return None, ["G2.7B_WRAPPER_SOURCE_REFS_INVALID"]
        if data_root is not None:
            try:
                _check_declared_refs(source_refs, root=data_root, field=f"{field}.source_refs")
            except S211DeliveryBlocked as error:
                return None, [str(error)]
        try:
            _wrapper_hash, wrapper_declared = _payload_identity(value, field=field)
        except S211DeliveryBlocked as error:
            return None, [str(error)]
        if not wrapper_declared:
            reasons.append("G2.7B_WRAPPER_ARTIFACT_HASH_MISSING")
        nested_hash, nested_reasons = _validate_g27b(
            dict(nested), field=f"{field}.payload", data_root=data_root
        )
        return nested_hash, sorted(set(reasons + nested_reasons))
    try:
        record = GateRecord.from_mapping(dict(value))
    except Exception as error:
        # S2.10's producer-specific gate is accepted only if it still exposes
        # the same exact identity/status contract.
        if value.get("schema_version") != S210_GATE_SCHEMA:
            return None, [f"G2.7B_GATE_INVALID:{type(error).__name__}"]
        try:
            identity, has_hash = _payload_identity(value, field=field)
        except S211DeliveryBlocked as blocked:
            return None, [str(blocked)]
        if not has_hash:
            reasons.append("G2.7B_GATE_ARTIFACT_HASH_MISSING")
        if value.get("gate_id") != "stage2.G2.7b" or value.get("stage") != 2:
            reasons.append("G2.7B_GATE_ID_MISMATCH")
        if value.get("status") != "PASS":
            reasons.append("G2.7B_NOT_PASS")
        if _forbidden_formal_source(value):
            reasons.append("G2.7B_FORBIDDEN_FIXTURE")
        if value.get("scope") not in {None, "formal"}:
            reasons.append("G2.7B_GATE_SCOPE_NOT_FORMAL")
        if (
            "formal_eligible" in value
            and value.get("formal_eligible") is not True
        ) or (
            "formal_eligible" not in value
            and value.get("scope") != "formal"
        ):
            reasons.append("G2.7B_GATE_NOT_FORMAL")
        if not isinstance(value.get("measured"), Mapping):
            reasons.append("G2.7B_GATE_MEASURED_MISSING")
        return identity, reasons
    if record.gate_id != "stage2.G2.7b" or record.stage != 2:
        reasons.append("G2.7B_GATE_ID_MISMATCH")
    if record.effective_status() is not GateStatus.PASS:
        reasons.append("G2.7B_NOT_PASS")
    if _forbidden_formal_source(value):
        reasons.append("G2.7B_FORBIDDEN_FIXTURE")
    declared = value.get("artifact_hash")
    if not isinstance(declared, str):
        reasons.append("G2.7B_GATE_ARTIFACT_HASH_MISSING")
    elif declared != record.artifact_hash:
        reasons.append("G2.7B_GATE_HASH_MISMATCH")
    return record.artifact_hash, reasons


def _lineage_entries(
    value: Mapping[str, Any] | None,
    *,
    field: str,
    data_root: Path | None = None,
) -> tuple[dict[str, Any], list[str], str | None]:
    if value is None:
        return [], ["STAGE2_LINEAGE_MISSING"], None
    identity, declared = _payload_identity(value, field=field)
    reasons: list[str] = []
    if value.get("schema_version") != S211_LINEAGE_SCHEMA:
        reasons.append("STAGE2_LINEAGE_SCHEMA_MISMATCH")
    if _forbidden_formal_source(value):
        reasons.append("STAGE2_LINEAGE_FORBIDDEN_FIXTURE")
    raw: Any = value.get("tasks", value.get("entries", value.get("lineage", value.get("artifacts"))))
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for task_id, item in raw.items():
            if not isinstance(item, Mapping):
                reasons.append("STAGE2_LINEAGE_ENTRY_INVALID")
                continue
            row = dict(item)
            if row.get("task_id") is not None and row.get("task_id") != task_id:
                reasons.append("STAGE2_LINEAGE_TASK_KEY_MISMATCH")
            row.setdefault("task_id", task_id)
            entries.append(row)
    elif isinstance(raw, list):
        entries = [dict(item) for item in raw if isinstance(item, Mapping)]
        if len(entries) != len(raw):
            reasons.append("STAGE2_LINEAGE_ENTRY_INVALID")
    if not entries:
        reasons.append("STAGE2_LINEAGE_ENTRIES_MISSING")
        return entries, reasons, identity
    by_task: dict[str, dict[str, Any]] = {}
    duplicate_tasks: set[str] = set()
    for item in entries:
        task_id = item.get("task_id")
        if not isinstance(task_id, str):
            reasons.append("STAGE2_LINEAGE_TASK_ID_INVALID")
            continue
        if task_id in by_task:
            duplicate_tasks.add(task_id)
        by_task[task_id] = item
    for task_id in sorted(duplicate_tasks):
        reasons.append(f"STAGE2_LINEAGE_DUPLICATE:{task_id}")
    unknown_tasks = set(by_task) - set(_STAGE2_TASKS)
    for task_id in sorted(unknown_tasks):
        reasons.append(f"STAGE2_LINEAGE_UNSUPPORTED_TASK:{task_id}")
    missing = [task_id for task_id in _STAGE2_TASKS if task_id not in by_task]
    if missing:
        reasons.append("STAGE2_LINEAGE_INCOMPLETE:" + ",".join(missing))
    for task_id, item in by_task.items():
        if task_id not in _STAGE2_TASKS:
            continue
        state = _status(item)
        if state not in {"PASS", "SEALED", "COMPLETED", "FROZEN"}:
            reasons.append(f"STAGE2_LINEAGE_NOT_PASS:{task_id}")
        if item.get("formal_eligible") is not True:
            reasons.append(f"STAGE2_LINEAGE_NOT_FORMAL:{task_id}")
        if _forbidden_formal_source(item):
            reasons.append(f"STAGE2_LINEAGE_FORBIDDEN_FIXTURE:{task_id}")
        refs = item.get("artifact_refs", item.get("artifacts"))
        if not isinstance(refs, (list, tuple)) or not refs:
            reasons.append(f"STAGE2_LINEAGE_ARTIFACT_REFS_MISSING:{task_id}")
        elif data_root is not None:
            try:
                _check_declared_refs(refs, root=data_root, field=f"stage2_lineage.{task_id}.artifact_refs")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
        artifact_hash = item.get("artifact_hash")
        if not isinstance(artifact_hash, str) or _SHA256.fullmatch(artifact_hash) is None:
            reasons.append(f"STAGE2_LINEAGE_ARTIFACT_HASH_MISSING:{task_id}")
        else:
            try:
                _payload_identity(item, field=f"stage2_lineage.{task_id}")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
    if not declared:
        reasons.append("STAGE2_LINEAGE_ARTIFACT_HASH_MISSING")
    return entries, sorted(set(reasons)), identity


def _boundary_record(
    role: str,
    value: Mapping[str, Any] | str | Path | None,
    *,
    data_root: Path | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    if value is None:
        return None, None, [f"BOUNDARY_MISSING:{role}"]
    try:
        payload, identity, declared = _read(value, field=f"boundary.{role}", data_root=data_root)
    except S211DeliveryBlocked as error:
        return None, None, [str(error)]
    assert payload is not None
    reasons: list[str] = []
    state = _status(payload)
    if state not in {"PASS", "READY", "SEALED", "COMPLETED", "FROZEN", "SELECTED"}:
        reasons.append(f"BOUNDARY_NOT_READY:{role}")
    if not _formal_eligible(payload):
        reasons.append(f"BOUNDARY_NOT_FORMAL:{role}")
    if not declared:
        reasons.append(f"BOUNDARY_ARTIFACT_HASH_MISSING:{role}")
    if data_root is not None:
        for ref_field in ("artifact_refs", "artifacts", "source_refs", "evidence_refs"):
            try:
                _check_declared_refs(payload.get(ref_field), root=data_root, field=f"boundary.{role}.{ref_field}")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
    assert identity is not None
    return (
        {
            "role": role,
            "ref": _ref_text(value),
            "schema_version": payload.get("schema_version"),
            "artifact_hash": identity,
            "status": state,
            "formal_eligible": _formal_eligible(payload),
        },
        identity,
        reasons,
    )


def _validate_replay(audit: Mapping[str, Any] | None, *, expected_31m_hash: str | None) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if audit is None:
        return (
            {
                "schema_version": S211_REPLAY_AUDIT_SCHEMA,
                "audit_type": "confirmatory_31m_repetition",
                "status": "BLOCKED",
                "formal_eligible": False,
                "replay_executed": False,
                "equivalent": False,
                "source_result_hash": None,
                "replay_result_hash": None,
                "reasons": ["REPLAY_AUDIT_MISSING"],
            },
            ["REPLAY_AUDIT_MISSING"],
        )
    try:
        identity, declared = _payload_identity(audit, field="replay_audit_31m")
    except S211DeliveryBlocked as error:
        return (
            {"schema_version": S211_REPLAY_AUDIT_SCHEMA, "status": "BLOCKED", "formal_eligible": False, "replay_executed": False, "equivalent": False, "reasons": [str(error)]},
            [str(error)],
        )
    if not declared:
        reasons.append("REPLAY_AUDIT_ARTIFACT_HASH_MISSING")
    if _status(audit) != "PASS":
        reasons.append("REPLAY_AUDIT_NOT_PASS")
    if audit.get("replay_executed") is not True:
        reasons.append("REPLAY_NOT_EXECUTED")
    if audit.get("equivalent") is not True:
        reasons.append("REPLAY_NOT_EQUIVALENT")
    model = str(audit.get("model", audit.get("model_id", audit.get("scale", "")))).casefold()
    if "31m" not in model and "31-m" not in model:
        reasons.append("REPLAY_MODEL_NOT_31M")
    kind = str(audit.get("audit_type", audit.get("replay_kind", ""))).casefold()
    if "confirmatory" not in kind or "repetition" not in kind:
        reasons.append("REPLAY_NOT_CONFIRMATORY_REPETITION")
    repetition = audit.get("repetition_id", audit.get("repetition"))
    if not isinstance(repetition, (str, int)) or isinstance(repetition, bool):
        reasons.append("REPLAY_REPETITION_ID_MISSING")
    source_hash = audit.get("source_result_hash")
    replay_hash = audit.get("replay_result_hash")
    for name, candidate in (("source_result_hash", source_hash), ("replay_result_hash", replay_hash)):
        if not isinstance(candidate, str) or _SHA256.fullmatch(candidate) is None:
            reasons.append(f"REPLAY_{name.upper()}_MISSING")
    if (
        isinstance(source_hash, str)
        and _SHA256.fullmatch(source_hash) is not None
        and isinstance(replay_hash, str)
        and _SHA256.fullmatch(replay_hash) is not None
        and source_hash != replay_hash
    ):
        reasons.append("REPLAY_RESULT_HASH_MISMATCH")
    source_artifact_hash = audit.get("source_artifact_hash")
    if not isinstance(source_artifact_hash, str) or _SHA256.fullmatch(source_artifact_hash) is None:
        reasons.append("REPLAY_31M_SOURCE_HASH_MISSING")
    elif not isinstance(expected_31m_hash, str):
        reasons.append("REPLAY_31M_BOUNDARY_HASH_MISSING")
    elif source_artifact_hash != expected_31m_hash:
        reasons.append("REPLAY_31M_SOURCE_HASH_MISMATCH")
    if audit.get("formal_eligible") is not True:
        reasons.append("REPLAY_NOT_FORMAL")
    if audit.get("network_used") is True or audit.get("synthetic") is True or audit.get("local_fixture") is True:
        reasons.append("REPLAY_FORBIDDEN_SYNTHETIC_OR_NETWORK")
    body = dict(audit)
    body["validated_artifact_hash"] = identity
    body["validation_reasons"] = sorted(set(reasons))
    return body, sorted(set(reasons))


def _validate_delivery_role(
    role: str,
    value: Mapping[str, Any] | str | Path | None,
    *,
    data_root: Path | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Validate a Stage 3 handoff role as a separately hashed artifact."""

    reasons: list[str] = []
    if value is None:
        return None, None, [f"DELIVERY_ROLE_MISSING:{role}"]
    try:
        payload, identity, declared = _read(value, field=f"delivery_role.{role}", data_root=data_root)
    except S211DeliveryBlocked as error:
        return None, None, [str(error)]
    assert payload is not None
    if not declared:
        reasons.append(f"DELIVERY_ROLE_ARTIFACT_HASH_MISSING:{role}")
    expected_schema = _ROLE_SCHEMAS.get(role)
    if role == "estimator_decision":
        accepted_schemas = {expected_schema, S210_DECISION_SCHEMA}
        if payload.get("schema_version") not in accepted_schemas:
            reasons.append(f"DELIVERY_ROLE_SCHEMA_MISMATCH:{role}")
    elif expected_schema is not None and payload.get("schema_version") != expected_schema:
        reasons.append(f"DELIVERY_ROLE_SCHEMA_MISMATCH:{role}")
    role_formal = payload.get("formal_eligible") is True
    if _forbidden_formal_source(payload):
        role_formal = False
        reasons.append(f"DELIVERY_ROLE_FORBIDDEN_FIXTURE:{role}")
    estimator_object: Any = None
    if role == "estimator_decision":
        if payload.get("schema_version") == S210_DECISION_SCHEMA:
            _, role_formal, decision_reasons = _validate_s210_decision(
                payload, field="delivery_role.estimator_decision"
            )
            reasons.extend(
                f"DELIVERY_ESTIMATOR_INVALID:{reason}" for reason in decision_reasons
            )
        else:
            try:
                from .stage2 import EstimatorDecision

                estimator_object = EstimatorDecision.from_mapping(payload)
                role_formal = estimator_object.formal_eligible
            except (TypeError, ValueError, RuntimeError) as error:
                reasons.append(f"DELIVERY_ESTIMATOR_INVALID:{type(error).__name__}")
    if not role_formal:
        reasons.append(f"DELIVERY_ROLE_NOT_FORMAL:{role}")
    state = _status(payload)
    allowed = {
        "plan": {"PASS", "SEALED", "FROZEN", "COMPLETED"},
        "task_catalog": {"PASS", "SEALED", "FROZEN", "COMPLETED"},
        "replay_report": {"PASS"},
        "gate_summary": {"PASS"},
        "sync_report": {"PASS", "SEALED", "COMPLETED"},
        "estimator_decision": {"SELECTED", "PASS"},
        "large_artifact_index": {"PASS", "SEALED", "COMPLETED"},
        "worklog": {"PASS", "SEALED", "COMPLETED"},
        "dirty_head_evidence": {"PASS", "SEALED", "COMPLETED"},
        "failure_retry_amendment_history": {"PASS", "SEALED", "COMPLETED"},
    }
    if state not in allowed.get(role, {"PASS"}):
        reasons.append(f"DELIVERY_ROLE_NOT_READY:{role}")
    if data_root is not None:
        for ref_field in ("artifact_ref", "source_ref", "plan_ref", "task_catalog_ref"):
            try:
                _check_declared_ref(payload.get(ref_field), root=data_root, field=f"delivery_role.{role}.{ref_field}")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
        for ref_field in ("artifact_refs", "source_refs", "evidence_refs"):
            try:
                _check_declared_refs(payload.get(ref_field), root=data_root, field=f"delivery_role.{role}.{ref_field}")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
    if role == "plan":
        if payload.get("task_id") != S211_TASK_ID or payload.get("gate_id") != S211_GATE_ID:
            reasons.append("DELIVERY_PLAN_BINDING_MISMATCH")
    elif role == "task_catalog":
        if payload.get("task_id") != S211_TASK_ID:
            reasons.append("DELIVERY_TASK_CATALOG_TASK_MISMATCH")
        outputs = payload.get("outputs", payload.get("declared_outputs"))
        required = {"delivery_manifest", "replay_report", "gate_summary", "sync_report"}
        if not isinstance(outputs, (list, tuple)) or not required.issubset(set(outputs)):
            reasons.append("DELIVERY_TASK_CATALOG_OUTPUTS_INCOMPLETE")
        gates = payload.get("gates", payload.get("required_gates"))
        if not isinstance(gates, (list, tuple)) or "stage2.G2.7b" not in gates:
            reasons.append("DELIVERY_TASK_CATALOG_GATE_BINDING_MISSING")
    elif role == "replay_report":
        if payload.get("replay_executed") is not True or payload.get("equivalent") is not True:
            reasons.append("DELIVERY_REPLAY_REPORT_NOT_EXECUTED")
    elif role == "gate_summary":
        gates = payload.get("gates", payload.get("gate_records"))
        if not isinstance(gates, Mapping):
            reasons.append("DELIVERY_GATE_SUMMARY_MISSING_GATES")
        else:
            for gate_id in _UPSTREAM_GATES:
                item = gates.get(gate_id)
                item_status = item.get("status") if isinstance(item, Mapping) else item
                if item_status != "PASS":
                    reasons.append(f"DELIVERY_GATE_SUMMARY_NOT_PASS:{gate_id}")
    elif role == "sync_report":
        docs = payload.get("agent_documents", payload.get("agent_files"))
        if not isinstance(docs, Mapping) or set(docs) != set(_AGENT_DOCUMENTS):
            reasons.append("AGENT_DOCUMENT_SET_INVALID")
        elif any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in docs.values()):
            reasons.append("AGENT_DOCUMENT_SHA_INVALID")
        refs = payload.get("agent_document_refs")
        if not isinstance(refs, Mapping) or set(refs) != set(_AGENT_DOCUMENTS):
            reasons.append("AGENT_DOCUMENT_REF_SET_INVALID")
        elif isinstance(docs, Mapping):
            for document_name in _AGENT_DOCUMENTS:
                ref = refs.get(document_name)
                try:
                    resolved = _resolve_data_root_ref(
                        ref,
                        root=data_root,
                        field=f"sync_report.agent_document_refs.{document_name}",
                        must_exist=True,
                    )
                    if resolved.name != document_name:
                        reasons.append(f"AGENT_DOCUMENT_BASENAME_MISMATCH:{document_name}")
                    actual_sha = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    if docs.get(document_name) != actual_sha:
                        reasons.append(f"AGENT_DOCUMENT_SHA_MISMATCH:{document_name}")
                except (S211DeliveryBlocked, OSError, TypeError):
                    reasons.append(f"AGENT_DOCUMENT_REF_INVALID:{document_name}")
        if payload.get("worktree_clean") is not True or payload.get("server_worktree_clean") is not True:
            reasons.append("SYNC_WORKTREE_NOT_CLEAN")
        if payload.get("user_files_excluded") is not True:
            reasons.append("SYNC_USER_FILES_NOT_EXCLUDED")
        for commit_field in ("github_commit", "server_commit", "target_execution_commit"):
            try:
                _commit(payload.get(commit_field), field=f"sync_report.{commit_field}")
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
        if "local_temp.md" in (docs or {}):
            reasons.append("LEGACY_LOCAL_TEMP_MUST_NOT_BE_BOUND")
    elif role == "estimator_decision":
        if payload.get("scope") != "formal" or payload.get("gate_status") != "PASS":
            reasons.append("DELIVERY_ESTIMATOR_DECISION_NOT_PASS")
        if not isinstance(payload.get("selected_estimator"), str) or not payload.get("selected_estimator"):
            reasons.append("DELIVERY_ESTIMATOR_MISSING")
    elif role == "large_artifact_index":
        entries = payload.get("entries", payload.get("artifacts"))
        if not isinstance(entries, list) or not entries:
            reasons.append("LARGE_ARTIFACT_INDEX_EMPTY")
        elif data_root is None:
            reasons.append("LARGE_ARTIFACT_INDEX_DATA_ROOT_REQUIRED")
        else:
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    reasons.append(f"LARGE_ARTIFACT_INDEX_ENTRY_INVALID:{index}")
                    continue
                path = entry.get("path", entry.get("ref"))
                try:
                    resolved = _resolve_data_root_ref(
                        path,
                        root=data_root,
                        field=f"large_artifact_index[{index}]",
                    )
                    expected = entry.get("sha256", entry.get("file_sha256"))
                    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                        reasons.append(f"LARGE_ARTIFACT_INDEX_SHA_MISSING:{index}")
                    elif hashlib.sha256(resolved.read_bytes()).hexdigest() != expected:
                        reasons.append(f"LARGE_ARTIFACT_INDEX_SHA_MISMATCH:{index}")
                except (S211DeliveryBlocked, OSError, TypeError):
                    reasons.append(f"LARGE_ARTIFACT_INDEX_PATH_INVALID:{index}")
    elif role == "worklog":
        if not isinstance(payload.get("entries", payload.get("sections")), list):
            reasons.append("WORKLOG_ENTRIES_MISSING")
    elif role == "dirty_head_evidence":
        if payload.get("worktree_clean_for_delivery") is not True:
            reasons.append("DIRTY_HEAD_DELIVERY_SCOPE_NOT_CLEAN")
        excluded = payload.get("excluded_files")
        if not isinstance(excluded, list):
            reasons.append("USER_FILE_EXCLUSION_EVIDENCE_MISSING")
        elif any(not isinstance(path, str) or Path(path).parts[:1] != ("presentation",) for path in excluded):
            reasons.append("USER_FILE_EXCLUSION_SCOPE_INVALID")
    elif role == "failure_retry_amendment_history":
        for key in ("failures", "retries", "amendments"):
            if not isinstance(payload.get(key), list):
                reasons.append(f"HISTORY_{key.upper()}_MISSING")
    record = {
        "role": role,
        "ref": _ref_text(value),
        "schema_version": payload.get("schema_version"),
        "artifact_hash": identity,
        "status": state,
        "formal_eligible": role_formal,
    }
    return record, identity, sorted(set(reasons))


def _write_json(root: Path, name: str, value: Mapping[str, Any]) -> str:
    target = root / name
    if target.exists():
        raise S211DeliveryBlocked(f"OUTPUT_ALREADY_EXISTS:{name}")
    write_canonical_json(target, value)
    return name


def _publish_atomic(destination: Path, files: Mapping[str, Mapping[str, Any]]) -> list[str]:
    destination = destination.absolute()
    if destination.exists():
        raise S211DeliveryBlocked("OUTPUT_ROOT_MUST_BE_NEW_AND_EMPTY")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise S211DeliveryBlocked("STAGING_PATH_COLLISION")
    staging.mkdir()
    try:
        names = [_write_json(staging, name, payload) for name, payload in files.items()]
        inventory_entries: list[dict[str, Any]] = []
        # Verify bytes, canonical identities, and file SHA before the directory
        # becomes visible.  The inventory itself is intentionally not listed
        # recursively, avoiding a self-referential hash.
        for name in names:
            target = staging / name
            loaded = load_canonical_json(target)
            if not isinstance(loaded, Mapping):
                raise S211DeliveryBlocked(f"PUBLISHED_OBJECT_INVALID:{name}")
            if "artifact_hash" in loaded:
                _payload_identity(dict(loaded), field=f"published.{name}")
            inventory_entries.append({
                "file": name,
                "bytes": target.stat().st_size,
                "file_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "artifact_hash": loaded.get("artifact_hash"),
            })
        inventory = {
            "schema_version": S211_OUTPUT_INVENTORY_SCHEMA,
            "status": "PASS",
            "formal_eligible": files.get("g2.8-gate.json", {}).get("status") == "PASS",
            "append_only": True,
            "files": inventory_entries,
        }
        inventory["artifact_hash"] = canonical_json_hash(inventory)
        names.append(_write_json(staging, "output_inventory.json", inventory))
        loaded_inventory = load_canonical_json(staging / "output_inventory.json")
        if not isinstance(loaded_inventory, Mapping):
            raise S211DeliveryBlocked("PUBLISHED_OBJECT_INVALID:output_inventory.json")
        _payload_identity(dict(loaded_inventory), field="published.output_inventory.json")
        os.replace(staging, destination)
        # Re-open every published file after the atomic rename.  This catches
        # path mix-ups and ensures the manifest's inventory describes bytes
        # that are actually visible to downstream Stage 3 consumers.
        for name in names:
            target = destination / name
            loaded = load_canonical_json(target)
            if not isinstance(loaded, Mapping):
                raise S211DeliveryBlocked(f"PUBLISHED_OBJECT_INVALID_AFTER_RENAME:{name}")
            if "artifact_hash" in loaded:
                _payload_identity(dict(loaded), field=f"published.after_rename.{name}")
        return names
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_s211_g28(
    *,
    g27b_gate: Mapping[str, Any] | str | Path | None = None,
    g27b_decision: Mapping[str, Any] | str | Path | None = None,
    g27b_lineage: Mapping[str, Any] | str | Path | None = None,
    stage2_lineage: Mapping[str, Any] | str | Path | None = None,
    boundary_refs: Mapping[str, Mapping[str, Any] | str | Path | None] | None = None,
    environment: Mapping[str, Any] | str | Path | None = None,
    assets: Mapping[str, Any] | str | Path | None = None,
    reference: Mapping[str, Any] | str | Path | None = None,
    pilot: Mapping[str, Any] | str | Path | None = None,
    formal_14m: Mapping[str, Any] | str | Path | None = None,
    formal_31m: Mapping[str, Any] | str | Path | None = None,
    analysis: Mapping[str, Any] | str | Path | None = None,
    decision: Mapping[str, Any] | str | Path | None = None,
    replay_audit_31m: Mapping[str, Any] | str | Path | None = None,
    output_root: str | Path | None = None,
    data_root: str | Path | None = None,
    predecessor_gates: Mapping[str, Mapping[str, Any] | str | Path | None] | None = None,
    plan: Mapping[str, Any] | str | Path | None = None,
    task_catalog: Mapping[str, Any] | str | Path | None = None,
    replay_report: Mapping[str, Any] | str | Path | None = None,
    gate_summary: Mapping[str, Any] | str | Path | None = None,
    sync_report: Mapping[str, Any] | str | Path | None = None,
    estimator_decision: Mapping[str, Any] | str | Path | None = None,
    large_artifact_index: Mapping[str, Any] | str | Path | None = None,
    worklog: Mapping[str, Any] | str | Path | None = None,
    dirty_head_evidence: Mapping[str, Any] | str | Path | None = None,
    failure_retry_amendment_history: Mapping[str, Any] | str | Path | None = None,
    delivery_refs: Mapping[str, Mapping[str, Any] | str | Path | None] | None = None,
    run_id: str = "s211-g28-delivery",
    producer_commit: str | None = None,
    consumer_commit: str | None = None,
    checked_at: str | None = None,
    replay_command: Sequence[str] | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Validate and atomically publish the formal S2.11/G2.8 bundle.

    The explicit boundary arguments are convenient for server launchers;
    ``boundary_refs`` is useful when a manifest already has role-to-ref
    entries.  Aliases are accepted only for stable descriptive spellings and
    unknown arguments fail closed.
    """

    g27b_gate = g27b_gate or aliases.pop("gate", None) or aliases.pop("g27b_gate_ref", None)
    g27b_decision = g27b_decision or aliases.pop("decision_ref", None)
    g27b_lineage = g27b_lineage or aliases.pop("lineage", None)
    stage2_lineage = stage2_lineage or aliases.pop("complete_lineage", None)
    replay_audit_31m = replay_audit_31m or aliases.pop("confirmatory_31m_replay_audit", None)
    predecessor_gates = predecessor_gates or aliases.pop("upstream_gates", None)
    data_root = data_root or aliases.pop("data_root_ref", None)
    plan = plan or aliases.pop("plan_ref", None)
    task_catalog = task_catalog or aliases.pop("task_catalog_ref", None)
    replay_report = replay_report or aliases.pop("replay_report_ref", None)
    gate_summary = gate_summary or aliases.pop("gate_summary_ref", None)
    sync_report = sync_report or aliases.pop("sync_report_ref", None)
    estimator_decision = estimator_decision or aliases.pop("estimator_decision_ref", None)
    large_artifact_index = large_artifact_index or aliases.pop("large_artifact_index_ref", None)
    worklog = worklog or aliases.pop("worklog_ref", None)
    dirty_head_evidence = dirty_head_evidence or aliases.pop("dirty_head_evidence_ref", None)
    failure_retry_amendment_history = failure_retry_amendment_history or aliases.pop("history_ref", None)
    if aliases:
        raise S211DeliveryBlocked("UNSUPPORTED_ARGUMENTS:" + ",".join(sorted(aliases)))
    if not isinstance(run_id, str) or not run_id:
        raise S211DeliveryBlocked("RUN_ID_REQUIRED")

    try:
        resolved_data_root = _data_root(data_root)
    except S211DeliveryBlocked:
        raise
    if resolved_data_root is None:
        raise S211DeliveryBlocked("DATA_ROOT_REQUIRED")
    role_values: dict[str, Any] = dict(delivery_refs or {})
    for role, value in {
        "plan": plan,
        "task_catalog": task_catalog,
        "replay_report": replay_report,
        "gate_summary": gate_summary,
        "sync_report": sync_report,
        "estimator_decision": estimator_decision,
        "large_artifact_index": large_artifact_index,
        "worklog": worklog,
        "dirty_head_evidence": dirty_head_evidence,
        "failure_retry_amendment_history": failure_retry_amendment_history,
    }.items():
        if value is not None:
            role_values[role] = value

    boundary_values: dict[str, Any] = dict(boundary_refs or {})
    for role, value in {
        "environment": environment,
        "assets": assets,
        "reference": reference,
        "pilot": pilot,
        "formal_14m": formal_14m,
        "formal_31m": formal_31m,
        "analysis": analysis,
        "decision": decision,
    }.items():
        if value is not None:
            boundary_values[role] = value
    reasons: list[str] = []
    for role in sorted(set(boundary_values) - set(_BOUNDARY_ROLES)):
        reasons.append(f"BOUNDARY_ROLE_UNSUPPORTED:{role}")
    for role in sorted(set(role_values) - set(_DELIVERY_ROLES)):
        reasons.append(f"DELIVERY_ROLE_UNSUPPORTED:{role}")
    try:
        g27b_payload, g27b_hash, g27b_declared = _read(g27b_gate, field="g27b_gate", data_root=resolved_data_root)
    except S211DeliveryBlocked as error:
        g27b_payload, g27b_hash, g27b_declared = None, None, False
        reasons.append(str(error))
    gate_hashes, gate_reasons = _validate_upstream_gates(
        predecessor_gates, g27b_gate=g27b_payload, data_root=resolved_data_root
    )
    reasons.extend(gate_reasons)
    gate_hash = gate_hashes.get("stage2.G2.7b")
    if g27b_payload is not None and not g27b_declared:
        reasons.append("G2.7B_GATE_ARTIFACT_HASH_MISSING")

    try:
        decision_payload, decision_hash, decision_declared = _read(
            estimator_decision or g27b_decision or boundary_values.get("decision"), field="g27b_decision", data_root=resolved_data_root
        )
    except S211DeliveryBlocked as error:
        decision_payload, decision_hash, decision_declared = None, None, False
        reasons.append(str(error))
    if decision_payload is None:
        reasons.append("G2.7B_DECISION_MISSING")
    else:
        if not decision_declared:
            reasons.append("G2.7B_DECISION_ARTIFACT_HASH_MISSING")
        if _status(decision_payload) not in {"SELECTED", "PASS"} or decision_payload.get("gate_status") != "PASS":
            reasons.append("G2.7B_DECISION_NOT_PASS")
        if decision_payload.get("scope") != "formal":
            reasons.append("G2.7B_DECISION_NOT_FORMAL")
        if not isinstance(decision_payload.get("selected_estimator"), str) or not decision_payload.get("selected_estimator"):
            reasons.append("G2.7B_DECISION_ESTIMATOR_MISSING")
        if decision_payload.get("schema_version") == S210_DECISION_SCHEMA:
            _, decision_formal, decision_reasons = _validate_s210_decision(
                decision_payload, field="g27b_decision"
            )
            reasons.extend(f"G2.7B_DECISION_INVALID:{reason}" for reason in decision_reasons)
            if not decision_formal:
                reasons.append("G2.7B_DECISION_NOT_FORMAL_ELIGIBLE")
        else:
            try:
                from .stage2 import EstimatorDecision

                parsed_decision = EstimatorDecision.from_mapping(decision_payload)
            except (TypeError, ValueError, RuntimeError) as error:
                reasons.append(f"G2.7B_DECISION_INVALID:{type(error).__name__}")
            else:
                if not parsed_decision.formal_eligible:
                    reasons.append("G2.7B_DECISION_NOT_FORMAL_ELIGIBLE")
    if estimator_decision is not None and g27b_decision is not None:
        try:
            _estimator_payload, estimator_hash, _ = _read(
                estimator_decision, field="estimator_decision", data_root=resolved_data_root
            )
            _legacy_payload, legacy_hash, _ = _read(
                g27b_decision, field="g27b_decision_legacy", data_root=resolved_data_root
            )
            if estimator_hash != legacy_hash:
                reasons.append("ESTIMATOR_DECISION_DUPLICATE_MISMATCH")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    if g27b_payload is not None:
        gate_payload = g27b_payload
        while isinstance(gate_payload.get("payload"), Mapping):
            gate_payload = dict(gate_payload["payload"])
        measured = gate_payload.get("measured")
        expected = measured.get("decision_hash") if isinstance(measured, Mapping) else None
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            reasons.append("G2.7B_DECISION_HASH_MISSING")
        elif decision_hash is None or expected != decision_hash:
            reasons.append("G2.7B_DECISION_HASH_MISMATCH")

    try:
        lineage_payload, lineage_hash, lineage_declared = _read(stage2_lineage or g27b_lineage, field="stage2_lineage", data_root=resolved_data_root)
    except S211DeliveryBlocked as error:
        lineage_payload, lineage_hash, lineage_declared = None, None, False
        reasons.append(str(error))
    entries, lineage_reasons, derived_lineage_hash = _lineage_entries(
        lineage_payload, field="stage2_lineage", data_root=resolved_data_root
    )
    reasons.extend(lineage_reasons)
    lineage_hash = lineage_hash or derived_lineage_hash
    if lineage_payload is not None and not lineage_declared:
        reasons.append("STAGE2_LINEAGE_ARTIFACT_HASH_MISSING")

    records: dict[str, dict[str, Any]] = {}
    boundary_hashes: dict[str, str | None] = {}
    for role in _BOUNDARY_ROLES:
        record, identity, boundary_reasons = _boundary_record(role, boundary_values.get(role), data_root=resolved_data_root)
        if record is not None:
            records[role] = record
        boundary_hashes[role] = identity
        reasons.extend(boundary_reasons)

    replay_payload: Mapping[str, Any] | None = None
    if replay_audit_31m is not None:
        try:
            replay_payload, _replay_hash, _replay_declared = _read(replay_audit_31m, field="replay_audit_31m", data_root=resolved_data_root)
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    replay_audit, replay_reasons = _validate_replay(
        replay_payload, expected_31m_hash=boundary_hashes.get("formal_31m")
    )
    reasons.extend(replay_reasons)
    role_records: dict[str, dict[str, Any]] = {}
    role_hashes: dict[str, str | None] = {}
    role_payloads: dict[str, dict[str, Any]] = {}
    role_reason_map: dict[str, list[str]] = {}
    for role in _DELIVERY_ROLES:
        record, identity, role_issues = _validate_delivery_role(
            role, role_values.get(role), data_root=resolved_data_root
        )
        if record is not None:
            role_records[role] = record
        role_hashes[role] = identity
        reasons.extend(role_issues)
        role_reason_map[role] = role_issues
        value = role_values.get(role)
        if value is not None:
            try:
                loaded, _identity, _declared = _read(
                    value, field=f"delivery_role.{role}", data_root=resolved_data_root
                )
            except S211DeliveryBlocked as error:
                reasons.append(str(error))
            else:
                if loaded is not None:
                    role_payloads[role] = loaded
    if producer_commit is None:
        reasons.append("PRODUCER_COMMIT_REQUIRED")
    else:
        try:
            producer_commit = _commit(producer_commit, field="producer_commit")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    if consumer_commit is not None:
        try:
            consumer_commit = _commit(consumer_commit, field="consumer_commit")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
            consumer_commit = None
    else:
        reasons.append("CONSUMER_COMMIT_REQUIRED")
    sync_execution_commit = role_payloads.get("sync_report", {}).get("target_execution_commit")
    if producer_commit is not None and sync_execution_commit is not None and sync_execution_commit != producer_commit:
        reasons.append("SYNC_TARGET_EXECUTION_COMMIT_MISMATCH")
    reasons = sorted(set(reasons))
    final_status = "PASS" if not reasons else "BLOCKED"
    formal_eligible = final_status == "PASS"
    timestamp = checked_at or _now()
    instruction_body: dict[str, Any] = {
        "schema_version": S211_REPLAY_INSTRUCTIONS_SCHEMA,
        "task_id": S211_TASK_ID,
        "replay_id": f"{run_id}-31m-confirmatory-repetition",
        "status": "READY" if not reasons else "BLOCKED",
        "formal_eligible": formal_eligible,
        "started_from_empty_output_root": True,
        "network_required": False,
        "forbidden_inputs": ["shard-5.part", "*.lock", "old_history_objects", "existing_replay_output"],
        "source_refs": {"g27b_gate": gate_hash, "g27b_decision": decision_hash, "stage2_lineage": lineage_hash, "formal_31m": boundary_hashes.get("formal_31m")},
        "command": list(replay_command) if replay_command is not None else None,
        "reasons": reasons,
    }
    instruction_body["artifact_hash"] = canonical_json_hash(instruction_body)
    validator_body: dict[str, Any] = {
        "schema_version": S211_REPLAY_VALIDATOR_SCHEMA,
        "validator_id": "stage2-s211-31m-repetition-v1",
        "required_model": "31M",
        "required_audit_type": "confirmatory_31m_repetition",
        "required_fields": ["source_result_hash", "replay_result_hash", "repetition_id", "equivalent", "replay_executed", "formal_eligible"],
        "network_forbidden": True,
        "synthetic_forbidden": True,
        "empty_output_root_required": True,
        "expected_source_artifact_hash": boundary_hashes.get("formal_31m"),
        "status": "READY" if not reasons else "BLOCKED",
        "reasons": reasons,
    }
    validator_body["artifact_hash"] = canonical_json_hash(validator_body)
    replay_audit_body = dict(replay_audit)
    # The source audit may use a producer-specific schema; the delivery copy
    # is always normalized to the S2.11 audit envelope before hashing.
    replay_audit_body["schema_version"] = S211_REPLAY_AUDIT_SCHEMA
    replay_audit_body["validation_status"] = "PASS" if not replay_reasons else "BLOCKED"
    replay_audit_body["formal_eligible"] = bool(not replay_reasons and replay_audit_body.get("formal_eligible") is True)
    replay_audit_body["artifact_hash"] = canonical_json_hash({key: item for key, item in replay_audit_body.items() if key != "artifact_hash"})

    delivery_body: dict[str, Any] = {
        "schema_version": S211_DELIVERY_SCHEMA,
        "task_id": S211_TASK_ID,
        "gate_id": S211_GATE_ID,
        "run_id": run_id,
        "checked_at": timestamp,
        "status": final_status,
        "formal_eligible": formal_eligible,
        "append_only": True,
        "producer_commit": producer_commit,
        "consumer_commit": consumer_commit,
        "g27b_gate_hash": gate_hash,
        "upstream_gate_hashes": gate_hashes,
        "g27b_decision_hash": decision_hash,
        "stage2_lineage_hash": lineage_hash,
        "boundary_hashes": boundary_hashes,
        "boundary_records": records,
        "replay_audit_31m_hash": replay_audit_body["artifact_hash"],
        "replay_instructions_hash": instruction_body["artifact_hash"],
        "replay_validator_hash": validator_body["artifact_hash"],
        "delivery_role_hashes": role_hashes,
        "delivery_role_records": role_records,
        "estimator_decision": role_payloads.get("estimator_decision", decision_payload),
        "replay_report": role_payloads.get("replay_report"),
        "gate_summary": role_payloads.get("gate_summary"),
        "sync_report": role_payloads.get("sync_report"),
        "large_artifact_index": role_payloads.get("large_artifact_index"),
        "worklog": role_payloads.get("worklog"),
        "dirty_head_evidence": role_payloads.get("dirty_head_evidence"),
        "failure_retry_amendment_history": role_payloads.get("failure_retry_amendment_history"),
        "append_only_history_retained": role_payloads.get("failure_retry_amendment_history") is not None,
        "reasons": reasons,
    }
    delivery_body["artifact_hash"] = canonical_json_hash(delivery_body)
    evidence_names = ("delivery_manifest.json", "replay_instructions.json", "replay_validator.json", "replay_audit_31m.json")
    gate = GateRecord(
        gate_id=S211_GATE_ID,
        stage=2,
        status=GateStatus.PASS if formal_eligible else GateStatus.BLOCKED,
        checked_at=timestamp,
        measured={
            "g27b_gate_hash": gate_hash,
            "upstream_gate_hashes": gate_hashes,
            "g27b_decision_hash": decision_hash,
            "stage2_lineage_hash": lineage_hash,
            "boundary_hashes": boundary_hashes,
            "replay_audit_31m_hash": replay_audit_body["artifact_hash"],
            "formal_eligible": formal_eligible,
        },
        threshold={"required_lineage_tasks": list(_STAGE2_TASKS), "required_boundary_roles": list(_BOUNDARY_ROLES), "confirmatory_replay": "31M_one_repetition", "network": False, "append_only": True},
        evidence_refs=evidence_names + tuple(f"{role}.json" for role in _DELIVERY_ROLES),
        reasons=tuple(reasons) if not formal_eligible else (),
    )
    gate_body = gate.to_dict()
    role_files: dict[str, dict[str, Any]] = {}
    for role in _DELIVERY_ROLES:
        payload = role_payloads.get(role)
        if payload is None:
            payload = {
                "schema_version": _ROLE_SCHEMAS.get(role, f"stage2-s211-{role}-v1"),
                "task_id": S211_TASK_ID,
                "role": role,
                "status": "BLOCKED",
                "formal_eligible": False,
                "reasons": role_reason_map.get(role, [f"DELIVERY_ROLE_MISSING:{role}"]),
            }
            payload["artifact_hash"] = canonical_json_hash(payload)
        role_files[f"{role}.json"] = payload
    files = {
        "delivery_manifest.json": delivery_body,
        "replay_instructions.json": instruction_body,
        "replay_validator.json": validator_body,
        "replay_audit_31m.json": replay_audit_body,
        "g2.8-gate.json": gate_body,
        "replay_report.json": role_files["replay_report.json"],
        "gate_summary.json": role_files["gate_summary.json"],
        "sync_report.json": role_files["sync_report.json"],
        "estimator_decision.json": role_files["estimator_decision.json"],
        "large_artifact_index.json": role_files["large_artifact_index.json"],
        "worklog.json": role_files["worklog.json"],
        "dirty_head_evidence.json": role_files["dirty_head_evidence.json"],
        "failure_retry_amendment_history.json": role_files["failure_retry_amendment_history.json"],
        "plan.json": role_files["plan.json"],
        "task_catalog.json": role_files["task_catalog.json"],
    }
    output_files: list[str] = []
    if output_root is not None:
        # Keep the publication target in DATA_ROOT and give relative targets
        # the same stable meaning as all artifact references.
        output_candidate = _resolve_data_root_ref(
            output_root,
            root=resolved_data_root,
            field="output_root",
            must_exist=False,
        )
        if output_candidate.exists() and output_candidate.is_symlink():
            raise S211DeliveryBlocked("output_root:SYMLINK_REF_FORBIDDEN")
        output_files = _publish_atomic(output_candidate, files)
    return {
        "status": final_status,
        "formal_eligible": formal_eligible,
        "delivery_manifest": delivery_body,
        "replay_instructions": instruction_body,
        "replay_validator": validator_body,
        "replay_audit_31m": replay_audit_body,
        "gate": gate_body,
        "upstream_gate_hashes": gate_hashes,
        "delivery_roles": role_files,
        "lineage_entries": entries,
        "output_files": output_files,
        "delivery_hash": delivery_body["artifact_hash"],
        "gate_hash": gate_body["artifact_hash"],
    }


validate_g28 = run_s211_g28
orchestrate_s211_delivery = run_s211_g28
run_s211_delivery = run_s211_g28
run_s211_g28_delivery = run_s211_g28

__all__ = [
    "S211DeliveryBlocked",
    "S211G28Blocked",
    "S211_DELIVERY_SCHEMA",
    "S211_GATE_ID",
    "S211_GATE_SCHEMA",
    "S210_DECISION_SCHEMA",
    "S211_OUTPUT_INVENTORY_SCHEMA",
    "S211_REPLAY_AUDIT_SCHEMA",
    "S211_REPLAY_INSTRUCTIONS_SCHEMA",
    "S211_REPLAY_VALIDATOR_SCHEMA",
    "run_s211_g28",
    "run_s211_delivery",
    "run_s211_g28_delivery",
    "orchestrate_s211_delivery",
    "validate_g28",
]
