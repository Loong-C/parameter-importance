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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
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
    computed = canonical_json_hash(body if declared is not None else payload)
    if declared is not None:
        _sha(declared, field=f"{field}.artifact_hash")
        if declared != computed:
            raise S211DeliveryBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
        return str(declared), True
    return computed, False


def _read(value: Mapping[str, Any] | str | Path | None, *, field: str) -> tuple[dict[str, Any] | None, str | None, bool]:
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
            path = Path(value)
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


def _formal_eligible(payload: Mapping[str, Any]) -> bool:
    return payload.get("formal_eligible") is True or payload.get("scope") == "formal"


def _validate_g27b(value: Mapping[str, Any] | None, *, field: str) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    if value is None:
        return None, ["G2.7B_GATE_MISSING"]
    try:
        record = GateRecord.from_mapping(dict(value))
    except Exception as error:
        # S2.10's producer-specific gate is accepted only if it still exposes
        # the same exact identity/status contract.
        if value.get("schema_version") != "stage2-s210-g27b-gate-v1":
            return None, [f"G2.7B_GATE_INVALID:{type(error).__name__}"]
        declared = value.get("artifact_hash")
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
        return identity, reasons
    if record.gate_id != "stage2.G2.7b" or record.stage != 2:
        reasons.append("G2.7B_GATE_ID_MISMATCH")
    if record.effective_status() is not GateStatus.PASS:
        reasons.append("G2.7B_NOT_PASS")
    declared = value.get("artifact_hash")
    if declared != record.artifact_hash:
        reasons.append("G2.7B_GATE_HASH_MISMATCH")
    return record.artifact_hash, reasons


def _lineage_entries(value: Mapping[str, Any] | None, *, field: str) -> tuple[dict[str, Any], list[str], str | None]:
    if value is None:
        return [], ["STAGE2_LINEAGE_MISSING"], None
    identity, declared = _payload_identity(value, field=field)
    reasons: list[str] = []
    raw: Any = value.get("tasks", value.get("entries", value.get("lineage", value.get("artifacts"))))
    entries: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for task_id, item in raw.items():
            if isinstance(item, Mapping):
                row = dict(item)
                row.setdefault("task_id", task_id)
                entries.append(row)
    elif isinstance(raw, list):
        entries = [dict(item) for item in raw if isinstance(item, Mapping)]
    if not entries:
        reasons.append("STAGE2_LINEAGE_ENTRIES_MISSING")
        return entries, reasons, identity
    by_task = {item.get("task_id"): item for item in entries if isinstance(item.get("task_id"), str)}
    missing = [task_id for task_id in _STAGE2_TASKS if task_id not in by_task]
    if missing:
        reasons.append("STAGE2_LINEAGE_INCOMPLETE:" + ",".join(missing))
    for task_id, item in by_task.items():
        if task_id not in _STAGE2_TASKS:
            continue
        state = _status(item)
        if state not in {"PASS", "SEALED", "COMPLETED", "FROZEN"}:
            reasons.append(f"STAGE2_LINEAGE_NOT_PASS:{task_id}")
        if item.get("formal_eligible") is False:
            reasons.append(f"STAGE2_LINEAGE_NOT_FORMAL:{task_id}")
        refs = item.get("artifact_refs", item.get("artifacts"))
        if refs is None:
            reasons.append(f"STAGE2_LINEAGE_ARTIFACT_REFS_MISSING:{task_id}")
    if not declared:
        reasons.append("STAGE2_LINEAGE_ARTIFACT_HASH_MISSING")
    return entries, sorted(set(reasons)), identity


def _boundary_record(
    role: str,
    value: Mapping[str, Any] | str | Path | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    if value is None:
        return None, None, [f"BOUNDARY_MISSING:{role}"]
    try:
        payload, identity, declared = _read(value, field=f"boundary.{role}")
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
    assert identity is not None
    return (
        {
            "role": role,
            "ref": value.as_posix() if isinstance(value, Path) else (value if isinstance(value, str) else "inline"),
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
    if isinstance(expected_31m_hash, str) and isinstance(audit.get("source_artifact_hash"), str) and audit.get("source_artifact_hash") != expected_31m_hash:
        reasons.append("REPLAY_31M_SOURCE_HASH_MISMATCH")
    if audit.get("formal_eligible") is not True:
        reasons.append("REPLAY_NOT_FORMAL")
    if audit.get("network_used") is True or audit.get("synthetic") is True or audit.get("local_fixture") is True:
        reasons.append("REPLAY_FORBIDDEN_SYNTHETIC_OR_NETWORK")
    body = dict(audit)
    body["validated_artifact_hash"] = identity
    body["validation_reasons"] = sorted(set(reasons))
    return body, sorted(set(reasons))


def _write_json(root: Path, name: str, value: Mapping[str, Any]) -> str:
    target = root / name
    if target.exists():
        raise S211DeliveryBlocked(f"OUTPUT_ALREADY_EXISTS:{name}")
    write_canonical_json(target, value)
    return name


def _publish_atomic(destination: Path, files: Mapping[str, Mapping[str, Any]]) -> list[str]:
    destination = destination.resolve()
    if destination.exists():
        raise S211DeliveryBlocked("OUTPUT_ROOT_MUST_BE_NEW_AND_EMPTY")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise S211DeliveryBlocked("STAGING_PATH_COLLISION")
    staging.mkdir()
    try:
        names = [_write_json(staging, name, payload) for name, payload in files.items()]
        # Verify bytes and hashes before the directory becomes visible.
        for name in names:
            loaded = load_canonical_json(staging / name)
            if not isinstance(loaded, Mapping):
                raise S211DeliveryBlocked(f"PUBLISHED_OBJECT_INVALID:{name}")
        os.replace(staging, destination)
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
    if aliases:
        raise S211DeliveryBlocked("UNSUPPORTED_ARGUMENTS:" + ",".join(sorted(aliases)))
    if not isinstance(run_id, str) or not run_id:
        raise S211DeliveryBlocked("RUN_ID_REQUIRED")

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
    try:
        g27b_payload, g27b_hash, g27b_declared = _read(g27b_gate, field="g27b_gate")
    except S211DeliveryBlocked as error:
        g27b_payload, g27b_hash, g27b_declared = None, None, False
        reasons.append(str(error))
    gate_hash, gate_reasons = _validate_g27b(g27b_payload, field="g27b_gate")
    reasons.extend(gate_reasons)
    if g27b_payload is not None and not g27b_declared:
        reasons.append("G2.7B_GATE_ARTIFACT_HASH_MISSING")

    try:
        decision_payload, decision_hash, decision_declared = _read(
            g27b_decision or boundary_values.get("decision"), field="g27b_decision"
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
    if g27b_payload is not None and isinstance(g27b_payload.get("measured"), Mapping) and decision_hash is not None:
        expected = g27b_payload["measured"].get("decision_hash")
        if expected is not None and expected != decision_hash:
            reasons.append("G2.7B_DECISION_HASH_MISMATCH")

    try:
        lineage_payload, lineage_hash, lineage_declared = _read(stage2_lineage or g27b_lineage, field="stage2_lineage")
    except S211DeliveryBlocked as error:
        lineage_payload, lineage_hash, lineage_declared = None, None, False
        reasons.append(str(error))
    entries, lineage_reasons, derived_lineage_hash = _lineage_entries(lineage_payload, field="stage2_lineage")
    reasons.extend(lineage_reasons)
    lineage_hash = lineage_hash or derived_lineage_hash
    if lineage_payload is not None and not lineage_declared:
        reasons.append("STAGE2_LINEAGE_ARTIFACT_HASH_MISSING")

    records: dict[str, dict[str, Any]] = {}
    boundary_hashes: dict[str, str | None] = {}
    for role in _BOUNDARY_ROLES:
        record, identity, boundary_reasons = _boundary_record(role, boundary_values.get(role))
        if record is not None:
            records[role] = record
        boundary_hashes[role] = identity
        reasons.extend(boundary_reasons)

    replay_payload: Mapping[str, Any] | None = None
    if replay_audit_31m is not None:
        try:
            replay_payload, _replay_hash, _replay_declared = _read(replay_audit_31m, field="replay_audit_31m")
        except S211DeliveryBlocked as error:
            reasons.append(str(error))
    replay_audit, replay_reasons = _validate_replay(
        replay_payload, expected_31m_hash=boundary_hashes.get("formal_31m")
    )
    reasons.extend(replay_reasons)
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
        "g27b_decision_hash": decision_hash,
        "stage2_lineage_hash": lineage_hash,
        "boundary_hashes": boundary_hashes,
        "boundary_records": records,
        "replay_audit_31m_hash": replay_audit_body["artifact_hash"],
        "replay_instructions_hash": instruction_body["artifact_hash"],
        "replay_validator_hash": validator_body["artifact_hash"],
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
            "g27b_decision_hash": decision_hash,
            "stage2_lineage_hash": lineage_hash,
            "boundary_hashes": boundary_hashes,
            "replay_audit_31m_hash": replay_audit_body["artifact_hash"],
            "formal_eligible": formal_eligible,
        },
        threshold={"required_lineage_tasks": list(_STAGE2_TASKS), "required_boundary_roles": list(_BOUNDARY_ROLES), "confirmatory_replay": "31M_one_repetition", "network": False, "append_only": True},
        evidence_refs=evidence_names,
        reasons=tuple(reasons) if not formal_eligible else (),
    )
    gate_body = gate.to_dict()
    files = {
        "delivery_manifest.json": delivery_body,
        "replay_instructions.json": instruction_body,
        "replay_validator.json": validator_body,
        "replay_audit_31m.json": replay_audit_body,
        "g2.8-gate.json": gate_body,
    }
    output_files: list[str] = []
    if output_root is not None:
        output_files = _publish_atomic(Path(output_root), files)
    return {
        "status": final_status,
        "formal_eligible": formal_eligible,
        "delivery_manifest": delivery_body,
        "replay_instructions": instruction_body,
        "replay_validator": validator_body,
        "replay_audit_31m": replay_audit_body,
        "gate": gate_body,
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
    "S211_REPLAY_AUDIT_SCHEMA",
    "S211_REPLAY_INSTRUCTIONS_SCHEMA",
    "S211_REPLAY_VALIDATOR_SCHEMA",
    "run_s211_g28",
    "run_s211_delivery",
    "run_s211_g28_delivery",
    "orchestrate_s211_delivery",
    "validate_g28",
]
