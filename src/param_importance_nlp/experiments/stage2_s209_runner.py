"""Production S2.9 profiler control plane.

The S2.9 reducer is intentionally a pure consumer.  This module supplies the
missing execution boundary: it binds the frozen G2.4b matrix and sealed S2.7
manifest, validates a live UUID inventory and I/O observation, and executes a
profiler worker for each missing measurement.  A worker must return measured
wall/phase/memory counters; this control plane never invents a timing,
throughput, or memory value.

The runner is detached-friendly and resumable at terminal measurement files.
Every attempt and failure is retained.  A reducer report and its G2.7a gate are
published together from a staging directory, so a partial Pareto/capacity
decision cannot be mistaken for an atomic gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord
from ..contracts.g21_formal_handoff import ALLOWED_DEVICES, EXCLUDED_UUID
from .stage2_s207_formal import APPROVED_GPU_UUIDS, EXCLUDED_GPU_UUID, EXCLUDED_PCI
from .stage2_s209_g27a import (
    S29_COUNT_FIELDS,
    S29_COST_SEMANTICS,
    S29_METHODS,
    S29_TIMING_FIELDS,
    S29FrozenInputs,
    S29_G25_GATE_ID,
    S29G27ABlocked,
    bind_s209_inputs,
    prepare_s209_measurement_plan,
    run_s209_g27a,
)


S29_RUNNER_SCHEMA = "stage2-s209-g27a-profiler-runner-v1"
S29_STATUS_SCHEMA = "stage2-s209-g27a-detached-status-v2"
S29_ATTEMPT_SCHEMA = "stage2-s209-g27a-measurement-attempt-v1"
S29_FAILURE_SCHEMA = "stage2-s209-g27a-failure-evidence-v1"
S29_IO_SCHEMA = "stage2-s209-g27a-io-quiescence-v1"
S29_INVENTORY_SCHEMA = "stage2-s209-g27a-gpu-inventory-v1"
S29_CAPACITY_SCHEMA = "stage2-s209-g27a-capacity-v1"
S29_ULIMIT_SCHEMA = "stage2-s209-g27a-ulimit-v1"
S29_ACCEPTED_INVENTORY_SCHEMAS = frozenset({
    S29_INVENTORY_SCHEMA,
    "stage2-s206-gpu-inventory-v1",
})
S29_LIVE_GPU_COUNT = 8
S29_INVENTORY_HEALTH_ALIASES = {
    "memory_used_mib": ("memory_used_mib", "memory_used", "memory.used"),
    "memory_total_mib": ("memory_total_mib", "memory_total", "memory.total"),
    "utilization_gpu_percent": (
        "utilization_gpu_percent",
        "utilization_percent",
        "utilization_gpu",
        "utilization.gpu",
    ),
    "ecc_uncorrected_volatile": (
        "ecc_uncorrected_volatile",
        "ecc_volatile_uncorrected",
        "ecc.errors.uncorrected.volatile.total",
    ),
    "ecc_uncorrected_aggregate": (
        "ecc_uncorrected_aggregate",
        "ecc_aggregate_uncorrected",
        "ecc.errors.uncorrected.aggregate.total",
    ),
    "row_remap_status": (
        "row_remap_status",
        "row_remap",
        "row_remap_pending",
    ),
    "gpu_recovery_action": ("gpu_recovery_action", "recovery_action"),
    "xid_errors": ("xid_errors", "xid_error_count"),
    "gpu_class": ("gpu_class", "product_name", "name"),
}
S29_STATUS_VALUES = ("PREPARED", "RUNNING", "PAUSED", "FAILED", "SEALED", "BLOCKED")
S29_TERMINAL_VALUES = frozenset({"FAILED", "SEALED", "BLOCKED"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class S29RunnerBlocked(RuntimeError):
    """Raised when a profiler run cannot safely proceed or resume."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise S29RunnerBlocked(f"{field}:SAFE_ID_REQUIRED")
    return value


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S29RunnerBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_json(value: Any, *, field: str = "value") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not (value == value and abs(value) != float("inf")):
            raise S29RunnerBlocked(f"{field}:NONFINITE")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, field=f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise S29RunnerBlocked(f"{field}:NON_STRING_KEY")
            _finite_json(item, field=f"{field}.{key}")
        return
    raise S29RunnerBlocked(f"{field}:NOT_JSON_VALUE")


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise S29RunnerBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _verify_hash(payload: Mapping[str, Any], *, field: str, key: str = "artifact_hash") -> str:
    declared = _sha(payload.get(key), field=f"{field}.{key}")
    body = {name: value for name, value in payload.items() if name != key}
    if canonical_json_hash(body) != declared:
        raise S29RunnerBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _logical(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S29RunnerBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE")
    target = (root.joinpath(*logical.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE") from error
    return target


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    if path.exists():
        try:
            existing = _load_object(path, field=field)
        except S29RunnerBlocked:
            raise
        if existing == dict(value):
            return
        raise S29RunnerBlocked(f"{field}:IMMUTABLE_OUTPUT_EXISTS")
    write_canonical_json(path, value)


def _canonical_pci(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise S29RunnerBlocked(f"{field}:PCI_REQUIRED")
    text = value.strip().upper()
    match = re.fullmatch(r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])", text)
    if match is None:
        raise S29RunnerBlocked(f"{field}:PCI_INVALID")
    return f"0000:{match.group(1)}:{match.group(2)}.{match.group(3)}"


def _canonical_uuid(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S29RunnerBlocked(f"{field}:UUID_REQUIRED")
    text = value.strip()
    if not text.upper().startswith("GPU-"):
        text = "GPU-" + text
    return text


def _normalise_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    compute_apps: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Validate the complete live inventory before any formal scheduling.

    The four approved cards and the known excluded card are checked as a
    bidirectional PCI/UUID mapping.  Every live row carries health fields so
    that the excluded device's ECC/remap facts remain auditable, while only
    approved cards with a clean/idle snapshot become schedulable.
    """

    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise S29RunnerBlocked("GPU_INVENTORY_REQUIRED")
    if len(inventory) != S29_LIVE_GPU_COUNT:
        raise S29RunnerBlocked(f"GPU_INVENTORY_LIVE_CARD_COUNT_INVALID:{len(inventory)}")
    rows: list[dict[str, Any]] = []
    seen_uuid: set[str] = set()
    seen_pci: set[str] = set()
    for index, source in enumerate(inventory):
        if not isinstance(source, Mapping):
            raise S29RunnerBlocked(f"GPU_INVENTORY_ROW_INVALID:{index}")
        uuid = _canonical_uuid(source.get("uuid", source.get("gpu_uuid")), field=f"gpu[{index}]")
        pci = _canonical_pci(source.get("pci_bus_id", source.get("pci")), field=f"gpu[{index}]")
        uuid_key = uuid.casefold()
        pci_key = pci.casefold()
        if uuid_key in {item.casefold() for item in APPROVED_GPU_UUIDS} and pci_key == EXCLUDED_PCI.casefold():
            raise S29RunnerBlocked("APPROVED_GPU_BOUND_TO_EXCLUDED_PCI")
        if uuid_key in seen_uuid:
            raise S29RunnerBlocked("GPU_INVENTORY_DUPLICATE_UUID")
        if pci_key in seen_pci:
            raise S29RunnerBlocked("GPU_INVENTORY_DUPLICATE_PCI")
        seen_uuid.add(uuid_key)
        seen_pci.add(pci_key)
        row = dict(source)
        row["uuid"] = uuid
        row["pci_bus_id"] = pci
        for canonical, aliases in S29_INVENTORY_HEALTH_ALIASES.items():
            found = next((row[name] for name in aliases if name in row), None)
            if found is None:
                raise S29RunnerBlocked(f"GPU_INVENTORY_HEALTH_FIELD_MISSING:{canonical}")
            row[canonical] = found
        row["selected"] = uuid_key in {item.casefold() for item in APPROVED_GPU_UUIDS}
        row["permanently_excluded"] = uuid_key == EXCLUDED_GPU_UUID.casefold() or pci_key == EXCLUDED_PCI.casefold()
        rows.append(row)

    expected = {pci.casefold(): uuid.casefold() for pci, uuid in ALLOWED_DEVICES}
    expected[EXCLUDED_PCI.casefold()] = EXCLUDED_UUID.casefold()
    observed = {
        str(row["pci_bus_id"]).casefold(): str(row["uuid"]).casefold()
        for row in rows
        if str(row["pci_bus_id"]).casefold() in expected
    }
    if observed != expected:
        raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_OR_EXCLUDED_IDENTITY_DRIFT")
    if not any(str(row["uuid"]).casefold() == EXCLUDED_GPU_UUID.casefold() for row in rows):
        raise S29RunnerBlocked("GPU_INVENTORY_EXCLUDED_CARD_REQUIRED")

    approved_keys = {item.casefold() for item in APPROVED_GPU_UUIDS}
    apps_by_uuid: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(compute_apps, Sequence) or isinstance(compute_apps, (str, bytes)):
        raise S29RunnerBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    live_keys = {str(row["uuid"]).casefold() for row in rows}
    for index, source in enumerate(compute_apps):
        if not isinstance(source, Mapping):
            raise S29RunnerBlocked(f"GPU_INVENTORY_COMPUTE_APP_INVALID:{index}")
        app = dict(source)
        app_uuid = _canonical_uuid(app.get("gpu_uuid", app.get("uuid")), field=f"compute_app[{index}]")
        if app_uuid.casefold() not in live_keys:
            raise S29RunnerBlocked("GPU_INVENTORY_COMPUTE_APP_GPU_UNKNOWN")
        app["gpu_uuid"] = app_uuid
        apps_by_uuid.setdefault(app_uuid.casefold(), []).append(app)

    approved_gpu_classes: set[str] = set()
    for row in rows:
        row_uuid = str(row["uuid"]).casefold()
        if row_uuid not in approved_keys:
            continue
        try:
            memory_used = float(row["memory_used_mib"])
            memory_total = float(row["memory_total_mib"])
            utilization = float(row["utilization_gpu_percent"])
            ecc_volatile = float(row["ecc_uncorrected_volatile"])
            ecc_aggregate = float(row["ecc_uncorrected_aggregate"])
            xid_errors = float(row["xid_errors"])
        except (TypeError, ValueError) as error:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_HEALTH_INVALID") from error
        if memory_total <= 0 or memory_used != 0 or utilization != 0:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
        if ecc_volatile != 0 or ecc_aggregate != 0:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_CARD_ECC_NOT_CLEAN")
        if xid_errors != 0:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_XID_NOT_CLEAN")
        gpu_class = str(row["gpu_class"]).strip()
        if not gpu_class:
            raise S29RunnerBlocked("GPU_INVENTORY_GPU_CLASS_REQUIRED")
        approved_gpu_classes.add(gpu_class)
        remap = str(row["row_remap_status"]).strip().casefold()
        if remap not in {"none", "0", "clean", "false", "not_pending", "not-pending", "n/a", "na"}:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_ROW_REMAP_NOT_CLEAN")
        recovery = str(row["gpu_recovery_action"]).strip().casefold()
        if recovery not in {"none", "0", "clean", "n/a", "na"}:
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_CARD_RECOVERY_NOT_CLEAN")
        if apps_by_uuid.get(row_uuid):
            raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
    if len(approved_gpu_classes) != 1:
        raise S29RunnerBlocked("GPU_INVENTORY_APPROVED_GPU_CLASS_DRIFT")
    return rows


def _load_inventory_envelope(
    value: Mapping[str, Any],
    *,
    root: Path | None = None,
    inventory_ref: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a hash-bound live inventory envelope and return rows/identity."""

    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("GPU_INVENTORY_ENVELOPE_REQUIRED")
    payload = dict(value)
    if payload.get("schema_version") not in S29_ACCEPTED_INVENTORY_SCHEMAS:
        raise S29RunnerBlocked("GPU_INVENTORY_SCHEMA_INVALID")
    source_ref = payload.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref or "\\" in source_ref:
        raise S29RunnerBlocked("GPU_INVENTORY_SOURCE_REF_REQUIRED")
    if root is not None:
        _logical(root, source_ref, field="gpu_inventory.source_ref")
        if inventory_ref is not None:
            try:
                expected_source_ref = Path(inventory_ref).resolve().relative_to(root.resolve()).as_posix()
            except ValueError as error:
                raise S29RunnerBlocked("GPU_INVENTORY_PATH_OUTSIDE_DATA_ROOT") from error
            if source_ref != expected_source_ref:
                raise S29RunnerBlocked("GPU_INVENTORY_SOURCE_REF_PATH_MISMATCH")
    rows = payload.get("rows")
    apps = payload.get("compute_apps")
    if not isinstance(rows, list) or not all(isinstance(item, Mapping) for item in rows):
        raise S29RunnerBlocked("GPU_INVENTORY_ROWS_REQUIRED")
    if not isinstance(apps, list) or not all(isinstance(item, Mapping) for item in apps):
        raise S29RunnerBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    artifact_hash = _verify_hash(payload, field="gpu_inventory")
    source_sha = _file_sha256(inventory_ref) if inventory_ref is not None else ""
    declared_source_sha = payload.get("source_sha256")
    if declared_source_sha is not None:
        if not isinstance(declared_source_sha, str) or declared_source_sha != source_sha:
            raise S29RunnerBlocked("GPU_INVENTORY_SOURCE_SHA256_MISMATCH")
    normalized = validate_s209_gpu_inventory(rows, compute_apps=apps)
    identity = {
        "source_ref": source_ref,
        "artifact_hash": artifact_hash,
        "source_sha256": source_sha,
        "schema_version": str(payload["schema_version"]),
    }
    normalized_summary = dict(normalized)
    normalized_summary.update(
        {
            "inventory_source_ref": source_ref,
            "inventory_artifact_hash": artifact_hash,
            "inventory_source_sha256": source_sha,
            "inventory_schema_version": str(payload["schema_version"]),
        }
    )
    return normalized_summary, identity


def validate_s209_gpu_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    compute_apps: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a canonical inventory summary without selecting by CUDA index."""

    rows = _normalise_inventory(inventory, compute_apps=compute_apps)
    summary: dict[str, Any] = {
        "schema_version": S29_INVENTORY_SCHEMA,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "selected_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "rows": rows,
        "inventory_count": len(rows),
        "excluded_present": any(row["uuid"] == EXCLUDED_GPU_UUID for row in rows),
    }
    summary["artifact_hash"] = canonical_json_hash(summary)
    return summary


def validate_s209_io_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an immutable I/O observation; no inferred quiescence is allowed."""

    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("IO_QUIESCENCE_EVIDENCE_REQUIRED")
    payload = dict(value)
    _verify_hash(payload, field="io_evidence")
    if payload.get("schema_version") != S29_IO_SCHEMA:
        raise S29RunnerBlocked("IO_EVIDENCE_SCHEMA_INVALID")
    if payload.get("status") not in {"QUIESCENT", "NON_QUIESCENT", "FAILED"}:
        raise S29RunnerBlocked("IO_EVIDENCE_STATUS_INVALID")
    for field in ("observed_at", "run_id", "probe_id"):
        _safe_id(payload.get(field), field=f"io_evidence.{field}") if field != "observed_at" else None
    if not isinstance(payload.get("active_transfers"), list):
        raise S29RunnerBlocked("IO_EVIDENCE_ACTIVE_TRANSFERS_REQUIRED")
    if not isinstance(payload.get("failure_evidence"), list):
        raise S29RunnerBlocked("IO_EVIDENCE_FAILURE_EVIDENCE_REQUIRED")
    if payload["status"] == "QUIESCENT" and payload["active_transfers"]:
        raise S29RunnerBlocked("IO_EVIDENCE_QUIESCENT_WITH_ACTIVE_TRANSFER")
    payload["cost_io_quiescent"] = payload["status"] == "QUIESCENT" and not payload["active_transfers"]
    return payload


def _load_capacity_evidence(root: Path, reference: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if reference is None:
        raise S29RunnerBlocked("CAPACITY_EVIDENCE_REQUIRED")
    path = _logical(root, reference, field="capacity")
    payload = _load_object(path, field="capacity")
    artifact_hash = _verify_hash(payload, field="capacity")
    if payload.get("schema_version") != S29_CAPACITY_SCHEMA:
        raise S29RunnerBlocked("CAPACITY_SCHEMA_INVALID")
    if not isinstance(payload.get("observed_at"), str) or not payload["observed_at"].strip():
        raise S29RunnerBlocked("CAPACITY_OBSERVED_AT_REQUIRED")
    for field in ("stage4_steps", "stage5_steps", "disk_free_bytes", "inode_free"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise S29RunnerBlocked(f"CAPACITY_{field.upper()}_INVALID")
    payload["capacity_evidence_hash"] = artifact_hash
    return payload, {"ref": reference, "artifact_hash": artifact_hash}


def _load_ulimit_evidence(root: Path, reference: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if reference is None:
        raise S29RunnerBlocked("ULIMIT_EVIDENCE_REQUIRED")
    path = _logical(root, reference, field="ulimit")
    payload = _load_object(path, field="ulimit")
    artifact_hash = _verify_hash(payload, field="ulimit")
    if payload.get("schema_version") != S29_ULIMIT_SCHEMA:
        raise S29RunnerBlocked("ULIMIT_SCHEMA_INVALID")
    if payload.get("resource") != "nofile":
        raise S29RunnerBlocked("ULIMIT_RESOURCE_INVALID")
    soft = payload.get("soft_nofile", payload.get("soft_limit"))
    hard = payload.get("hard_nofile", payload.get("hard_limit"))
    if isinstance(soft, bool) or not isinstance(soft, int) or soft != 1024:
        raise S29RunnerBlocked("ULIMIT_NOFILE_SOFT_MUST_BE_1024")
    if isinstance(hard, bool) or not isinstance(hard, int) or hard < soft:
        raise S29RunnerBlocked("ULIMIT_NOFILE_HARD_INVALID")
    payload["ulimit_evidence_hash"] = artifact_hash
    return payload, {"ref": reference, "artifact_hash": artifact_hash, "soft_nofile": soft}


@dataclass(frozen=True, slots=True)
class S29Preflight:
    """All identities and evidence bound immediately before execution."""

    root: Path
    frozen: S29FrozenInputs
    matrix: dict[str, Any]
    gate: dict[str, Any]
    g25_gate: dict[str, Any] | None
    raw_manifest: dict[str, Any]
    measurement_plan: dict[str, Any]
    inventory: dict[str, Any]
    io_evidence: dict[str, Any]
    matrix_ref: str
    gate_ref: str
    g25_gate_ref: str | None
    raw_manifest_ref: str
    measurement_plan_ref: str
    gpu_inventory_ref: str = "gpu-inventory.json"
    io_evidence_ref: str = "io-evidence.json"
    capacity_inputs: dict[str, Any] | None = None
    capacity_ref: str = "capacity.json"
    ulimit_evidence: dict[str, Any] | None = None
    ulimit_ref: str = "ulimit.json"

    @property
    def plan_hash(self) -> str:
        return str(self.measurement_plan["artifact_hash"])


def prepare_s209_plan(
    *,
    data_root: str | Path,
    matrix_ref: str,
    gate_ref: str,
    raw_manifest_ref: str,
    g25_gate_ref: str | None = None,
    run_id: str,
    anchor_ids: Sequence[str] = ("method-only-anchor-0", "method-only-anchor-1"),
    repetitions: int = 2,
    randomization_seed: int = 2909,
    inventory: Mapping[str, Any] | None = None,
    io_evidence: Mapping[str, Any] | None = None,
    output_ref: str | None = None,
) -> dict[str, Any]:
    """Prepare an immutable measurement plan after binding all S2.7 inputs."""

    root = Path(data_root).resolve()
    matrix_path = _logical(root, matrix_ref, field="g24b_matrix")
    gate_path = _logical(root, gate_ref, field="g24b_gate")
    raw_path = _logical(root, raw_manifest_ref, field="s27_raw_manifest")
    matrix = _load_object(matrix_path, field="g24b_matrix")
    gate = _load_object(gate_path, field="g24b_gate")
    raw = _load_object(raw_path, field="s27_raw_manifest")
    g25 = None
    if g25_gate_ref is not None:
        g25 = _load_object(_logical(root, g25_gate_ref, field="g25_gate"), field="g25_gate")
    frozen = bind_s209_inputs(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=raw,
        g25_gate=g25,
        require_g25=g25_gate_ref is not None,
        matrix_ref=matrix_ref,
        raw_manifest_ref=raw_manifest_ref,
    )
    if inventory is not None:
        _load_inventory_envelope(inventory, root=root)
    if io_evidence is not None:
        validate_s209_io_evidence(io_evidence)
    plan = prepare_s209_measurement_plan(frozen, run_id=run_id, anchor_ids=anchor_ids, repetitions=repetitions, randomization_seed=randomization_seed)
    if output_ref is not None:
        path = _logical(root, output_ref, field="measurement_plan_output")
        _write_once(path, plan, field="measurement_plan")
    return plan


def load_s209_preflight(
    *,
    data_root: str | Path,
    matrix_ref: str,
    gate_ref: str,
    raw_manifest_ref: str,
    g25_gate_ref: str | None = None,
    measurement_plan_ref: str,
    gpu_inventory_ref: str,
    io_evidence_ref: str,
    capacity_ref: str | None = None,
    ulimit_ref: str | None = None,
) -> S29Preflight:
    """Re-read and re-bind all producer artifacts at launch time."""

    if g25_gate_ref is None:
        raise S29RunnerBlocked("G25_GATE_REFERENCE_REQUIRED")
    root = Path(data_root).resolve()
    refs = {
        "matrix": _logical(root, matrix_ref, field="g24b_matrix"),
        "gate": _logical(root, gate_ref, field="g24b_gate"),
        "raw": _logical(root, raw_manifest_ref, field="s27_raw_manifest"),
        "plan": _logical(root, measurement_plan_ref, field="measurement_plan"),
        "inventory": _logical(root, gpu_inventory_ref, field="gpu_inventory"),
        "io": _logical(root, io_evidence_ref, field="io_evidence"),
    }
    if g25_gate_ref is not None:
        refs["g25"] = _logical(root, g25_gate_ref, field="g25_gate")
    matrix = _load_object(refs["matrix"], field="g24b_matrix")
    gate = _load_object(refs["gate"], field="g24b_gate")
    raw = _load_object(refs["raw"], field="s27_raw_manifest")
    g25 = _load_object(refs["g25"], field="g25_gate") if "g25" in refs else None
    frozen = bind_s209_inputs(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=raw,
        g25_gate=g25,
        require_g25=g25_gate_ref is not None,
        matrix_ref=matrix_ref,
        raw_manifest_ref=raw_manifest_ref,
    )
    plan = _load_object(refs["plan"], field="measurement_plan")
    # bind_s209_inputs is deliberately invoked before plan validation; a
    # stale plan cannot grant authority to a replaced S2.7 manifest.
    from .stage2_s209_g27a import _validate_measurement_plan  # private, same contract
    _validate_measurement_plan(plan, frozen=frozen)
    inventory_raw = _load_object(refs["inventory"], field="gpu_inventory")
    inventory, inventory_identity = _load_inventory_envelope(
        inventory_raw,
        root=root,
        inventory_ref=refs["inventory"],
    )
    io_evidence = validate_s209_io_evidence(_load_object(refs["io"], field="io_evidence"))
    capacity_inputs, capacity_identity = _load_capacity_evidence(root, capacity_ref)
    ulimit_evidence, ulimit_identity = _load_ulimit_evidence(root, ulimit_ref)
    inventory["inventory_identity"] = inventory_identity
    inventory["capacity_identity"] = capacity_identity
    inventory["ulimit_identity"] = ulimit_identity
    return S29Preflight(
        root=root,
        frozen=frozen,
        matrix=matrix,
        gate=gate,
        g25_gate=g25,
        raw_manifest=raw,
        measurement_plan=plan,
        inventory=inventory,
        io_evidence=io_evidence,
        matrix_ref=matrix_ref,
        gate_ref=gate_ref,
        g25_gate_ref=g25_gate_ref,
        raw_manifest_ref=raw_manifest_ref,
        measurement_plan_ref=measurement_plan_ref,
        gpu_inventory_ref=gpu_inventory_ref,
        io_evidence_ref=io_evidence_ref,
        capacity_inputs=capacity_inputs,
        capacity_ref=capacity_identity["ref"],
        ulimit_evidence=ulimit_evidence,
        ulimit_ref=ulimit_identity["ref"],
    )


@dataclass(frozen=True, slots=True)
class S29DetachedStatus:
    run_id: str
    plan_hash: str
    status: str
    completed_tasks: int
    expected_tasks: int
    updated_at: str
    owner_pid: int | None = None
    terminal_reason: str | None = None
    inventory_artifact_hash: str | None = None
    inventory_source_sha256: str | None = None
    matrix_hash: str | None = None
    raw_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        _safe_id(self.run_id, field="status.run_id")
        _sha(self.plan_hash, field="status.plan_hash")
        if self.status not in S29_STATUS_VALUES:
            raise S29RunnerBlocked("STATUS_VALUE_INVALID")
        if isinstance(self.completed_tasks, bool) or not isinstance(self.completed_tasks, int) or self.completed_tasks < 0:
            raise S29RunnerBlocked("STATUS_COMPLETED_TASKS_INVALID")
        if isinstance(self.expected_tasks, bool) or not isinstance(self.expected_tasks, int) or self.expected_tasks <= 0 or self.completed_tasks > self.expected_tasks:
            raise S29RunnerBlocked("STATUS_EXPECTED_TASKS_INVALID")
        if self.status in S29_TERMINAL_VALUES and not self.terminal_reason:
            raise S29RunnerBlocked("STATUS_TERMINAL_REASON_REQUIRED")
        for field, value in (
            ("status.inventory_artifact_hash", self.inventory_artifact_hash),
            ("status.inventory_source_sha256", self.inventory_source_sha256),
            ("status.matrix_hash", self.matrix_hash),
            ("status.raw_manifest_hash", self.raw_manifest_hash),
        ):
            if value is not None:
                _sha(value, field=field)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": S29_STATUS_SCHEMA,
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "status": self.status,
            "completed_tasks": self.completed_tasks,
            "expected_tasks": self.expected_tasks,
            "updated_at": self.updated_at,
            "owner_pid": self.owner_pid,
            "terminal_reason": self.terminal_reason,
            "inventory_artifact_hash": self.inventory_artifact_hash,
            "inventory_source_sha256": self.inventory_source_sha256,
            "matrix_hash": self.matrix_hash,
            "raw_manifest_hash": self.raw_manifest_hash,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        return body


class S29StatusStore:
    """Atomic status store with conservative resume transitions."""

    _ALLOWED = {
        "PREPARED": {"PREPARED", "RUNNING", "BLOCKED"},
        "RUNNING": {"RUNNING", "PAUSED", "FAILED", "SEALED", "BLOCKED"},
        "PAUSED": {"PAUSED", "RUNNING", "FAILED", "BLOCKED"},
        "FAILED": {"FAILED"},
        "SEALED": {"SEALED"},
        "BLOCKED": {"BLOCKED"},
    }

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        plan_hash: str,
        inventory_identity: Mapping[str, Any] | None = None,
        cost_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.run_id = _safe_id(run_id, field="status.run_id")
        self.plan_hash = _sha(plan_hash, field="status.plan_hash")
        self.inventory_identity = dict(inventory_identity) if isinstance(inventory_identity, Mapping) else None
        self.cost_identity = dict(cost_identity) if isinstance(cost_identity, Mapping) else None

    def _bind_identity(self, status: S29DetachedStatus) -> S29DetachedStatus:
        expected = {
            "inventory_artifact_hash": None if self.inventory_identity is None else self.inventory_identity.get("artifact_hash"),
            "inventory_source_sha256": None if self.inventory_identity is None else self.inventory_identity.get("source_sha256"),
            "matrix_hash": None if self.cost_identity is None else self.cost_identity.get("matrix_hash"),
            "raw_manifest_hash": None if self.cost_identity is None else self.cost_identity.get("raw_manifest_hash"),
        }
        for field, value in expected.items():
            actual = getattr(status, field)
            if value is not None and actual not in {None, value}:
                raise S29RunnerBlocked(f"STATUS_{field.upper()}_DRIFT")
        return replace(status, **{field: value if value is not None else getattr(status, field) for field, value in expected.items()})

    def load(self) -> S29DetachedStatus:
        value = _load_object(self.path, field="status")
        if value.get("schema_version") != S29_STATUS_SCHEMA:
            raise S29RunnerBlocked("STATUS_SCHEMA_INVALID")
        declared = _sha(value.get("artifact_hash"), field="status.artifact_hash")
        if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != declared:
            raise S29RunnerBlocked("STATUS_HASH_MISMATCH")
        if value.get("run_id") != self.run_id or value.get("plan_hash") != self.plan_hash:
            raise S29RunnerBlocked("STATUS_IDENTITY_MISMATCH")
        status = S29DetachedStatus(
            run_id=self.run_id,
            plan_hash=self.plan_hash,
            status=str(value["status"]),
            completed_tasks=int(value["completed_tasks"]),
            expected_tasks=int(value["expected_tasks"]),
            updated_at=str(value["updated_at"]),
            owner_pid=(None if value.get("owner_pid") is None else int(value["owner_pid"])),
            terminal_reason=(None if value.get("terminal_reason") is None else str(value["terminal_reason"])),
            inventory_artifact_hash=(None if value.get("inventory_artifact_hash") is None else str(value["inventory_artifact_hash"])),
            inventory_source_sha256=(None if value.get("inventory_source_sha256") is None else str(value["inventory_source_sha256"])),
            matrix_hash=(None if value.get("matrix_hash") is None else str(value["matrix_hash"])),
            raw_manifest_hash=(None if value.get("raw_manifest_hash") is None else str(value["raw_manifest_hash"])),
        )
        bound = self._bind_identity(status)
        if bound != status:
            raise S29RunnerBlocked("STATUS_IDENTITY_MISSING")
        return status

    def publish(self, status: S29DetachedStatus) -> None:
        status = self._bind_identity(status)
        if status.run_id != self.run_id or status.plan_hash != self.plan_hash:
            raise S29RunnerBlocked("STATUS_IDENTITY_MISMATCH")
        if self.path.exists():
            previous = self.load()
            if status.status not in self._ALLOWED[previous.status]:
                raise S29RunnerBlocked(f"STATUS_TRANSITION_FORBIDDEN:{previous.status}->{status.status}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(self.path, status.to_dict())

    def acquire(self, *, expected_tasks: int) -> S29DetachedStatus:
        current: S29DetachedStatus | None = None
        if self.path.exists():
            current = self.load()
            if current.expected_tasks != expected_tasks:
                raise S29RunnerBlocked("STATUS_EXPECTED_TASKS_DRIFT")
            if current.status in S29_TERMINAL_VALUES:
                raise S29RunnerBlocked(f"STATUS_TERMINAL_NO_RESUME:{current.status}")
            if current.status == "RUNNING" and current.owner_pid not in {None, os.getpid()}:
                try:
                    os.kill(int(current.owner_pid), 0)
                except OSError:
                    self.publish(S29DetachedStatus(self.run_id, self.plan_hash, "PAUSED", current.completed_tasks, current.expected_tasks, _now(), terminal_reason="OWNER_EXITED"))
                else:
                    raise S29RunnerBlocked("STATUS_OWNED_BY_OTHER_PROCESS")
        if current is None:
            current = S29DetachedStatus(self.run_id, self.plan_hash, "PREPARED", 0, expected_tasks, _now())
            self.publish(current)
        running = S29DetachedStatus(self.run_id, self.plan_hash, "RUNNING", current.completed_tasks, current.expected_tasks, _now(), os.getpid())
        self.publish(running)
        return running


class S29ProfilerExecutor(Protocol):
    def __call__(self, task: Mapping[str, Any], *, environment: Mapping[str, str]) -> Mapping[str, Any]: ...


def subprocess_profiler_executor(command: Sequence[str]) -> S29ProfilerExecutor:
    """Build an executor for a real profiler process.

    The child must print one JSON object (or ``{"row": object}``) to stdout.
    All required measurements are validated downstream; an absent field is a
    hard failure, never replaced with a model/default estimate.
    """

    if not command or any(not isinstance(item, str) or not item for item in command):
        raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
    frozen_command = tuple(command)

    def execute(task: Mapping[str, Any], *, environment: Mapping[str, str]) -> Mapping[str, Any]:
        child_env = os.environ.copy()
        child_env.update({str(key): str(value) for key, value in environment.items()})
        started = time.monotonic()
        completed = subprocess.run(frozen_command, env=child_env, capture_output=True, text=True, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise S29RunnerBlocked(f"PROFILER_PROCESS_FAILED:{completed.returncode}:{completed.stderr[-400:]}")
        try:
            decoded = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise S29RunnerBlocked("PROFILER_OUTPUT_JSON_REQUIRED") from error
        if not isinstance(decoded, Mapping):
            raise S29RunnerBlocked("PROFILER_OUTPUT_OBJECT_REQUIRED")
        result = decoded.get("row", decoded)
        if not isinstance(result, Mapping):
            raise S29RunnerBlocked("PROFILER_OUTPUT_ROW_REQUIRED")
        row = dict(result)
        row["profiler_process_wall_seconds"] = elapsed
        return row

    return execute


def _validate_measured_row(
    row: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    frozen: S29FrozenInputs,
    io_evidence: Mapping[str, Any],
    inventory_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Require actual profiler fields and bind immutable task identity."""

    output = dict(row)
    _finite_json(output, field="profiler_row")
    if output.get("measurement_kind") not in {"actual", "device_actual"} or output.get("measured") is not True:
        raise S29RunnerBlocked("PROFILER_MEASUREMENT_MARKER_REQUIRED")
    semantic = str(task["semantic"])
    method = str(task.get("method", "shared"))
    expected_identity = {
        "semantic": semantic,
        "method": method,
        "anchor_id": str(task["anchor_id"]),
        "repetition": int(task["repetition"]),
        "run_id": str(task["run_id"]),
        "gpu_uuid": str(task["gpu_uuid"]),
        "device_count": int(task["device_count"]),
    }
    if semantic != "anchor" and int(task["device_count"]) != 1:
        raise S29RunnerBlocked("METHOD_TASK_SINGLE_GPU_REQUIRED")
    # System anchors are the only multi-GPU measurements.  Bind the complete
    # UUID set reported by the worker; checking only device_count would allow a
    # nominal four-card anchor to run on one card while retaining a four-card
    # label in the generated evidence.
    if semantic == "anchor":
        expected_gpu_uuids = task.get("gpu_uuids")
        if (
            not isinstance(expected_gpu_uuids, list)
            or not expected_gpu_uuids
            or row.get("gpu_uuids") != expected_gpu_uuids
        ):
            raise S29RunnerBlocked("PROFILER_GPU_UUID_SET_INVALID")
    for name, expected in expected_identity.items():
        if name in row and row[name] != expected:
            raise S29RunnerBlocked(f"PROFILER_IDENTITY_DRIFT:{name}")
    immutable_identity = {
        "source_raw_run_id": frozen.raw_run_id,
        "matrix_hash": frozen.matrix_hash,
        "raw_manifest_hash": frozen.raw_manifest_hash,
        "batch_size": frozen.batch_size,
        "microbatch_count": frozen.microbatch_count,
        "inventory_artifact_hash": inventory_identity.get("artifact_hash"),
        "inventory_source_sha256": inventory_identity.get("source_sha256"),
        "io_evidence_hash": io_evidence.get("artifact_hash"),
        "cost_io_quiescent": io_evidence.get("cost_io_quiescent") is True,
        "anchor_kind": "shared_runner" if semantic == "scientific_equal_sample_cost" else "method_only",
    }
    for name, expected in immutable_identity.items():
        if name in row and row[name] != expected:
            raise S29RunnerBlocked(f"PROFILER_IDENTITY_DRIFT:{name}")
    output["semantic"] = semantic
    output["method"] = method
    output["anchor_id"] = str(task["anchor_id"])
    output["repetition"] = int(task["repetition"])
    output["run_id"] = str(task["run_id"])
    output["source_raw_run_id"] = frozen.raw_run_id
    output["matrix_hash"] = frozen.matrix_hash
    output["raw_manifest_hash"] = frozen.raw_manifest_hash
    output["batch_size"] = frozen.batch_size
    output["microbatch_count"] = frozen.microbatch_count
    output["gpu_uuid"] = str(task["gpu_uuid"])
    output["device_count"] = int(task["device_count"])
    output["inventory_artifact_hash"] = _sha(inventory_identity.get("artifact_hash"), field="inventory_artifact_hash")
    output["inventory_source_sha256"] = _sha(inventory_identity.get("source_sha256"), field="inventory_source_sha256")
    output["anchor_kind"] = "shared_runner" if semantic == "scientific_equal_sample_cost" else "method_only"
    output["cost_io_quiescent"] = io_evidence.get("cost_io_quiescent") is True
    output["io_evidence_hash"] = str(io_evidence["artifact_hash"])
    output["health_ok"] = output.get("health_ok") is True
    # Require every phase/counter and all three memory views from the child.
    for name in S29_TIMING_FIELDS + ("wall_seconds", "allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes"):
        if name not in row:
            raise S29RunnerBlocked(f"PROFILER_MEASUREMENT_MISSING:{name}")
    for name in S29_COUNT_FIELDS:
        if name not in row:
            raise S29RunnerBlocked(f"PROFILER_MEASUREMENT_MISSING:{name}")
    wall = output["wall_seconds"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not float(wall) > 0:
        raise S29RunnerBlocked("PROFILER_WALL_SECONDS_INVALID")
    if any(isinstance(output[name], bool) or not isinstance(output[name], (int, float)) or float(output[name]) < 0 for name in S29_TIMING_FIELDS):
        raise S29RunnerBlocked("PROFILER_PHASE_SECONDS_INVALID")
    if output["wall_seconds"] + 1e-9 < sum(float(output[name]) for name in S29_TIMING_FIELDS):
        raise S29RunnerBlocked("PROFILER_WALL_PHASE_TOTAL_INVALID")
    if any(isinstance(output[name], bool) or not isinstance(output[name], int) or output[name] < 0 for name in S29_COUNT_FIELDS):
        raise S29RunnerBlocked("PROFILER_COUNT_INVALID")
    for name in ("allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes"):
        if isinstance(output[name], bool) or not isinstance(output[name], int) or output[name] <= 0:
            raise S29RunnerBlocked(f"PROFILER_MEMORY_INVALID:{name}")
    if not output["allocated_peak_bytes"] <= output["reserved_peak_bytes"] <= output["device_peak_bytes"]:
        raise S29RunnerBlocked("PROFILER_MEMORY_ORDER_INVALID")
    output["throughput_sequences_per_second"] = float(output["sequence_count"]) / float(output["wall_seconds"])
    output["measurement_boundary"] = {
        "start": "sample_mapping_resolved_before_data_read",
        "end": "estimator_statistics_and_atomic_result_write_complete",
        "source": "profiler_worker",
    }
    return output


def _task_list(preflight: S29Preflight, *, run_id: str) -> list[dict[str, Any]]:
    if preflight.measurement_plan.get("run_id") != run_id:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    rows = preflight.measurement_plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_ROWS_REQUIRED")
    tasks: list[dict[str, Any]] = []
    # The prepared plan is the sole source of anchor/repetition/order identity
    # for every semantic.  Synthetic IDs or hard-coded 2x2 loops would allow a
    # launcher to measure rows that are absent from the frozen plan.
    for semantic in S29_COST_SEMANTICS:
        for source in rows:
            if not isinstance(source, Mapping):
                raise S29RunnerBlocked("MEASUREMENT_PLAN_ROW_INVALID")
            anchor_value = source.get("anchor_id")
            repetition_value = source.get("repetition")
            if not isinstance(anchor_value, str) or not anchor_value or isinstance(repetition_value, bool) or not isinstance(repetition_value, int) or repetition_value < 0:
                raise S29RunnerBlocked("MEASUREMENT_PLAN_ROW_INVALID")
            anchor = anchor_value
            repetition = repetition_value
            order = source.get("method_order")
            if not isinstance(order, list) or len(order) != len(S29_METHODS) or len(set(order)) != len(S29_METHODS) or set(order) != set(S29_METHODS):
                raise S29RunnerBlocked("MEASUREMENT_PLAN_METHOD_ORDER_INVALID")
            for method in order:
                tasks.append({
                    "semantic": semantic,
                    "method": str(method),
                    "anchor_id": anchor,
                    "repetition": repetition,
                    "run_id": run_id,
                    "gpu_uuid": APPROVED_GPU_UUIDS[0],
                    "device_count": 1,
                })
    return tasks


class S29ProfilerRunner:
    """Resumable measurement runner and atomic reducer publisher."""

    def __init__(
        self,
        *,
        preflight: S29Preflight,
        run_id: str,
        run_root: str | Path,
        profiler: S29ProfilerExecutor,
        single_gpu_anchor: Mapping[str, Any] | None = None,
        four_gpu_anchor: Mapping[str, Any] | None = None,
        shared_attribution_cross_check: Mapping[str, Any] | None = None,
        accuracy_rows: Sequence[Mapping[str, Any]] = (),
        capacity_inputs: Mapping[str, Any] | None = None,
    ) -> None:
        self.preflight = preflight
        self.run_id = _safe_id(run_id, field="runner.run_id")
        if self.preflight.measurement_plan.get("run_id") != self.run_id:
            raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
        self.run_root = Path(run_root).resolve()
        self.profiler = profiler
        self.single_gpu_anchor = single_gpu_anchor
        self.four_gpu_anchor = four_gpu_anchor
        self.shared_attribution_cross_check = shared_attribution_cross_check
        self.accuracy_rows = tuple(dict(row) for row in accuracy_rows)
        self.capacity_inputs = dict(capacity_inputs) if isinstance(capacity_inputs, Mapping) else capacity_inputs
        self._last_report_status: str | None = None
        self.measurement_root = self.run_root / "measurements"
        self.attempt_root = self.run_root / "attempts"
        self.failure_root = self.run_root / "failures"
        self.status_store = S29StatusStore(
            self.run_root / "status.json",
            run_id=self.run_id,
            plan_hash=preflight.plan_hash,
            inventory_identity=preflight.inventory.get("inventory_identity"),
            cost_identity={
                "matrix_hash": preflight.frozen.matrix_hash,
                "raw_manifest_hash": preflight.frozen.raw_manifest_hash,
            },
        )

    def _task_key(self, task: Mapping[str, Any]) -> str:
        return f"{task['semantic']}__{task.get('method','shared')}__{task['anchor_id']}__r{task['repetition']}"

    def _row_path(self, task: Mapping[str, Any]) -> Path:
        return self.measurement_root / f"{self._task_key(task)}.json"

    def _attempt_path(self, task: Mapping[str, Any]) -> Path:
        return self.attempt_root / f"{self._task_key(task)}.json"

    def _failure_path(self, task: Mapping[str, Any]) -> Path:
        return self.failure_root / f"{self._task_key(task)}.json"

    def _anchor_task(self, *, device_count: int) -> dict[str, Any]:
        if device_count not in {1, 4}:
            raise S29RunnerBlocked("ANCHOR_DEVICE_COUNT_INVALID")
        return {
            "semantic": "anchor",
            "method": "anchor",
            "anchor_id": "single-gpu-anchor" if device_count == 1 else "four-gpu-anchor",
            "repetition": 0,
            "run_id": self.run_id,
            "gpu_uuid": APPROVED_GPU_UUIDS[0],
            "gpu_uuids": list(APPROVED_GPU_UUIDS if device_count == 4 else (APPROVED_GPU_UUIDS[0],)),
            "device_count": device_count,
        }

    def _run_measured_anchor(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one real anchor and convert its measured row to reducer shape."""

        key = str(task["anchor_id"])
        anchor_root = self.run_root / "anchors"
        anchor_path = anchor_root / f"{key}.json"
        if anchor_path.exists():
            cached = _load_object(anchor_path, field=f"anchor.{key}")
            self._validate_anchor(cached, task)
            return cached
        env = {
            "S29_RUN_ID": self.run_id,
            "S29_PLAN_HASH": self.preflight.plan_hash,
            "S29_SEMANTIC": "anchor",
            "S29_METHOD": "anchor",
            "S29_ANCHOR_ID": key,
            "S29_REPETITION": "0",
            "S29_GPU_UUIDS": ",".join(task["gpu_uuids"]),
            "CUDA_VISIBLE_DEVICES": ",".join(task["gpu_uuids"]),
        }
        try:
            measured = self.profiler(task, environment=env)
            # Validate the system anchor using its own semantic.  Converting a
            # four-card anchor to a method row would trigger the single-GPU
            # guard and make every real four-card anchor impossible to run.
            row = _validate_measured_row(
                measured,
                task=task,
                frozen=self.preflight.frozen,
                io_evidence=self.preflight.io_evidence,
                inventory_identity=self.preflight.inventory["inventory_identity"],
            )
            anchor = {
                "status": "PASS",
                "matrix_hash": self.preflight.frozen.matrix_hash,
                "source_raw_run_id": self.preflight.frozen.raw_run_id,
                "device_count": int(task["device_count"]),
                "gpu_uuids": list(task["gpu_uuids"]),
                "inventory_artifact_hash": self.preflight.inventory["inventory_artifact_hash"],
                "inventory_source_sha256": self.preflight.inventory["inventory_source_sha256"],
                "measurement_plan_hash": self.preflight.plan_hash,
                "cost_io_quiescent": row["cost_io_quiescent"],
                "health_ok": row["health_ok"],
                "numeric_consistency": measured.get("numeric_consistency") is True,
                "batch_size": self.preflight.frozen.batch_size,
                "microbatch_count": self.preflight.frozen.microbatch_count,
                **{name: row[name] for name in S29_COUNT_FIELDS},
                "wall_seconds": row["wall_seconds"],
                "allocated_peak_bytes": row["allocated_peak_bytes"],
                "reserved_peak_bytes": row["reserved_peak_bytes"],
                "device_peak_bytes": row["device_peak_bytes"],
                "measurement_kind": "actual",
                "measurement_source": "profiler_worker",
                "io_evidence_hash": self.preflight.io_evidence["artifact_hash"],
            }
            anchor_root.mkdir(parents=True, exist_ok=True)
            _write_once(anchor_path, anchor, field=f"anchor.{key}")
            return anchor
        except Exception as error:
            self._record_failure(task, code="S29_ANCHOR_PROFILER_FAILED", reason=f"{type(error).__name__}:{error}")
            raise

    def _validate_anchor(self, anchor: Mapping[str, Any], task: Mapping[str, Any]) -> None:
        """Reject cached anchors whose producer or frozen identity drifted."""

        expected = {
            "status": "PASS",
            "matrix_hash": self.preflight.frozen.matrix_hash,
            "source_raw_run_id": self.preflight.frozen.raw_run_id,
            "device_count": int(task["device_count"]),
            "gpu_uuids": list(task["gpu_uuids"]),
            "cost_io_quiescent": True,
            "health_ok": True,
            "numeric_consistency": True,
            "batch_size": self.preflight.frozen.batch_size,
            "microbatch_count": self.preflight.frozen.microbatch_count,
            "inventory_artifact_hash": self.preflight.inventory["inventory_artifact_hash"],
            "inventory_source_sha256": self.preflight.inventory["inventory_source_sha256"],
            "measurement_plan_hash": self.preflight.plan_hash,
            "measurement_kind": "actual",
            "measurement_source": "profiler_worker",
            "io_evidence_hash": self.preflight.io_evidence["artifact_hash"],
        }
        for name, value in expected.items():
            if anchor.get(name) != value:
                raise S29RunnerBlocked(f"ANCHOR_IDENTITY_DRIFT:{task['anchor_id']}:{name}")
        for name in S29_COUNT_FIELDS:
            value = anchor.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise S29RunnerBlocked(f"ANCHOR_COUNT_INVALID:{task['anchor_id']}:{name}")
        for name in ("wall_seconds", "allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes"):
            value = anchor.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
                raise S29RunnerBlocked(f"ANCHOR_MEASUREMENT_INVALID:{task['anchor_id']}:{name}")

    def _load_completed(self, tasks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        for task in tasks:
            path = self._row_path(task)
            if not path.exists():
                continue
            row = _load_object(path, field=f"measurement.{self._task_key(task)}")
            normalized = _validate_measured_row(
                row,
                task=task,
                frozen=self.preflight.frozen,
                io_evidence=self.preflight.io_evidence,
                inventory_identity=self.preflight.inventory["inventory_identity"],
            )
            if canonical_json_hash(normalized) != canonical_json_hash(row):
                raise S29RunnerBlocked(f"MEASUREMENT_ROW_IDENTITY_DRIFT:{self._task_key(task)}")
            completed[self._task_key(task)] = row
        return completed

    def _record_failure(self, task: Mapping[str, Any], *, code: str, reason: str) -> None:
        body: dict[str, Any] = {
            "schema_version": S29_FAILURE_SCHEMA,
            "run_id": self.run_id,
            "plan_hash": self.preflight.plan_hash,
            "task": dict(task),
            "code": code,
            "reason": reason,
            "observed_at": _now(),
        }
        body["artifact_hash"] = canonical_json_hash(body)
        self.failure_root.mkdir(parents=True, exist_ok=True)
        _write_once(self._failure_path(task), body, field=f"failure.{self._task_key(task)}")

    def run(self) -> dict[str, Any]:
        tasks = _task_list(self.preflight, run_id=self.run_id)
        self.measurement_root.mkdir(parents=True, exist_ok=True)
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        self.failure_root.mkdir(parents=True, exist_ok=True)
        current = self.status_store.acquire(expected_tasks=len(tasks) + 2)
        if self.single_gpu_anchor is None:
            self.single_gpu_anchor = self._run_measured_anchor(self._anchor_task(device_count=1))
        if self.four_gpu_anchor is None:
            self.four_gpu_anchor = self._run_measured_anchor(self._anchor_task(device_count=4))
        completed = self._load_completed(tasks)
        for task in tasks:
            key = self._task_key(task)
            if key in completed:
                continue
            attempt = self._attempt_path(task)
            if attempt.exists():
                previous = _load_object(attempt, field=f"attempt.{key}")
                if previous.get("status") == "RUNNING":
                    self._record_failure(task, code="S29_INTERRUPTED_ATTEMPT", reason="non-terminal profiler attempt found during resume")
                    continue
                if previous.get("status") == "FAILED":
                    continue
                raise S29RunnerBlocked(f"ATTEMPT_LEDGER_INVALID:{key}")
            attempt_payload: dict[str, Any] = {
                "schema_version": S29_ATTEMPT_SCHEMA,
                "run_id": self.run_id,
                "plan_hash": self.preflight.plan_hash,
                "task": dict(task),
                "status": "RUNNING",
                "started_at": _now(),
            }
            attempt_payload["artifact_hash"] = canonical_json_hash(attempt_payload)
            _write_once(attempt, attempt_payload, field=f"attempt.{key}")
            env = {
                "S29_RUN_ID": self.run_id,
                "S29_PLAN_HASH": self.preflight.plan_hash,
                "S29_SEMANTIC": str(task["semantic"]),
                "S29_METHOD": str(task.get("method", "shared")),
                "S29_ANCHOR_ID": str(task["anchor_id"]),
                "S29_REPETITION": str(task["repetition"]),
                # Every method task is an isolated single-GPU process.  The
                # four-card set is reserved exclusively for _run_measured_anchor.
                "S29_GPU_UUIDS": str(task["gpu_uuid"]),
                "CUDA_VISIBLE_DEVICES": str(task["gpu_uuid"]),
            }
            try:
                measured = self.profiler(task, environment=env)
                row = _validate_measured_row(
                    measured,
                    task=task,
                    frozen=self.preflight.frozen,
                    io_evidence=self.preflight.io_evidence,
                    inventory_identity=self.preflight.inventory["inventory_identity"],
                )
                _write_once(self._row_path(task), row, field=f"measurement.{key}")
                terminal = dict(attempt_payload)
                terminal["status"] = "SUCCEEDED"
                terminal["finished_at"] = _now()
                terminal["artifact_hash"] = canonical_json_hash({k: v for k, v in terminal.items() if k != "artifact_hash"})
                write_canonical_json(attempt, terminal)
                completed[key] = row
            except Exception as error:
                self._record_failure(task, code="S29_PROFILER_FAILED", reason=f"{type(error).__name__}:{error}")
                failed = dict(attempt_payload)
                failed["status"] = "FAILED"
                failed["finished_at"] = _now()
                failed["failure_code"] = "S29_PROFILER_FAILED"
                failed["failure_reason"] = f"{type(error).__name__}:{error}"
                failed["artifact_hash"] = canonical_json_hash({k: v for k, v in failed.items() if k != "artifact_hash"})
                write_canonical_json(attempt, failed)
        done_count = len(completed)
        status_completed = done_count + 2  # the single/four-card anchors
        if done_count < len(tasks):
            self.status_store.publish(S29DetachedStatus(self.run_id, self.preflight.plan_hash, "BLOCKED", status_completed, len(tasks) + 2, _now(), terminal_reason="MEASUREMENT_FAILURE_OR_INCOMPLETE"))
            return {"status": "BLOCKED", "completed_tasks": status_completed, "expected_tasks": len(tasks) + 2, "failure_evidence": sorted(path.name for path in self.failure_root.glob("*.json"))}
        self.status_store.publish(S29DetachedStatus(self.run_id, self.preflight.plan_hash, "RUNNING", status_completed, len(tasks) + 2, _now(), os.getpid()))
        cost_observations: dict[str, Any] = {
            semantic: {"observations": []}
            for semantic in S29_COST_SEMANTICS
        }
        online_metadata = cost_observations["online_training_incremental_cost"]
        for field in ("randomized_method_order", "randomization_seed", "decision_ratio_threshold"):
            if field in self.preflight.measurement_plan:
                online_metadata[field] = self.preflight.measurement_plan[field]
        for row in completed.values():
            cost_observations[str(row["semantic"])] ["observations"].append(row)
        # Method-only measured rows are still required for all three semantics;
        # shared/isolated anchors are intentionally supplied as separately
        # measured, immutable inputs to this control plane.
        try:
            report_root = self._publish_reduced(cost_observations)
        except Exception as error:
            self.status_store.publish(S29DetachedStatus(self.run_id, self.preflight.plan_hash, "BLOCKED", status_completed, len(tasks) + 2, _now(), terminal_reason=f"REDUCTION_BLOCKED:{type(error).__name__}:{error}"))
            self._record_failure({"semantic": "reduction", "method": "all", "anchor_id": "gate", "repetition": 0, "run_id": self.run_id}, code="S29_REDUCTION_BLOCKED", reason=f"{type(error).__name__}:{error}")
            return {"status": "BLOCKED", "completed_tasks": status_completed, "expected_tasks": len(tasks) + 2, "failure_evidence": sorted(path.name for path in self.failure_root.glob("*.json"))}
        if self._last_report_status == "PASS":
            self.status_store.publish(S29DetachedStatus(self.run_id, self.preflight.plan_hash, "SEALED", status_completed, len(tasks) + 2, _now(), terminal_reason="G2.7A_ATOMIC_REPORT_PUBLISHED"))
            status = "SEALED"
        else:
            self.status_store.publish(S29DetachedStatus(self.run_id, self.preflight.plan_hash, "BLOCKED", status_completed, len(tasks) + 2, _now(), terminal_reason=f"G2.7A_GATE_{self._last_report_status or 'UNKNOWN'}"))
            status = "BLOCKED"
        return {"status": status, "gate_status": self._last_report_status, "completed_tasks": status_completed, "expected_tasks": len(tasks) + 2, "report_root": str(report_root)}

    def _publish_reduced(self, cost_observations: Mapping[str, Any]) -> Path:
        destination = self.run_root / "g27a"
        if destination.exists():
            raise S29RunnerBlocked("G27A_OUTPUT_ALREADY_EXISTS")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        try:
            approved_rows = [
                row for row in self.preflight.inventory["rows"]
                if str(row.get("uuid", "")).casefold() in {uuid.casefold() for uuid in APPROVED_GPU_UUIDS}
            ]
            gpu_classes = {str(row.get("gpu_class", "")).strip() for row in approved_rows}
            health_snapshot = {
                "healthy": all(
                    float(row["ecc_uncorrected_volatile"]) == 0
                    and float(row["ecc_uncorrected_aggregate"]) == 0
                    and str(row["row_remap_status"]).strip().casefold() in {"none", "0", "clean", "false", "not_pending", "not-pending", "n/a", "na"}
                    and str(row["gpu_recovery_action"]).strip().casefold() in {"none", "0", "clean", "n/a", "na"}
                    for row in approved_rows
                ),
                "idle": all(
                    float(row["memory_used_mib"]) == 0 and float(row["utilization_gpu_percent"]) == 0
                    for row in approved_rows
                ),
                "same_gpu_class": len(gpu_classes) == 1 and "" not in gpu_classes,
                "gpu_class": next(iter(gpu_classes), None),
                "gpu_uuids": list(self.preflight.inventory["selected_gpu_uuids"]),
                "ecc_errors": sum(
                    int(float(row["ecc_uncorrected_volatile"]) + float(row["ecc_uncorrected_aggregate"]))
                    for row in approved_rows
                ),
                "xid_errors": sum(int(float(row["xid_errors"])) for row in approved_rows),
                "cost_io_quiescent": self.preflight.io_evidence["cost_io_quiescent"],
                "inventory_artifact_hash": self.preflight.inventory["inventory_artifact_hash"],
                "inventory_source_sha256": self.preflight.inventory["inventory_source_sha256"],
            }
            capacity_inputs = dict(self.preflight.capacity_inputs or {})
            if self.preflight.ulimit_evidence is not None:
                capacity_inputs["ulimit_evidence_hash"] = self.preflight.ulimit_evidence["ulimit_evidence_hash"]
                capacity_inputs["ulimit_nofile_soft"] = self.preflight.ulimit_evidence.get("soft_nofile", self.preflight.ulimit_evidence.get("soft_limit"))
            report = run_s209_g27a(
                matrix=self.preflight.matrix,
                g24b_gate=self.preflight.gate,
                raw_manifest=self.preflight.raw_manifest,
                g25_gate=self.preflight.g25_gate,
                cost_observations=cost_observations,
                health_snapshot=health_snapshot,
                single_gpu_anchor=self.single_gpu_anchor,
                four_gpu_anchor=self.four_gpu_anchor,
                shared_attribution_cross_check=self.shared_attribution_cross_check,
                accuracy_rows=self.accuracy_rows,
                capacity_inputs=capacity_inputs,
                run_id=self.run_id,
                cost_io_quiescent=self.preflight.io_evidence["cost_io_quiescent"],
                output_root=staging,
                measurement_plan=self.preflight.measurement_plan,
                matrix_ref=self.preflight.matrix_ref,
                raw_manifest_ref=self.preflight.raw_manifest_ref,
            )
            if not isinstance(report, Mapping) or report.get("status") not in {"PASS", "PROVISIONAL", "BLOCKED"}:
                raise S29RunnerBlocked("G27A_REPORT_INVALID")
            self._last_report_status = str(report["status"])
            # Preserve the reducer's exact report/gate pair and attach runner
            # provenance before the directory is published atomically.
            report_path = staging / "cost-system-validation.json"
            gate_path = staging / "g2.7a-gate.json"
            report_payload = _load_object(report_path, field="g27a.report")
            gate_payload = _load_object(gate_path, field="g27a.gate")
            gate = GateRecord.from_mapping(gate_payload)
            evidence_set = set(gate.evidence_refs)
            evidence_set.update(
                {
                    self.preflight.matrix_ref,
                    self.preflight.raw_manifest_ref,
                    self.preflight.measurement_plan_ref,
                    self.preflight.gpu_inventory_ref,
                    self.preflight.io_evidence_ref,
                    self.preflight.capacity_ref,
                    self.preflight.ulimit_ref,
                }
            )
            if self.preflight.g25_gate_ref is not None:
                evidence_set.add(self.preflight.g25_gate_ref)
            evidence_refs = tuple(sorted(evidence_set))
            gate_payload = GateRecord(
                gate_id=gate.gate_id,
                stage=gate.stage,
                status=gate.status,
                checked_at=gate.checked_at,
                measured=gate.measured,
                threshold=gate.threshold,
                evidence_refs=evidence_refs,
                reasons=gate.reasons,
                conditions=gate.conditions,
                expires_at=gate.expires_at,
                schema_version=gate.schema_version,
            ).to_dict()
            write_canonical_json(gate_path, gate_payload)
            report_payload["profiler_runner"] = {
                "schema_version": S29_RUNNER_SCHEMA,
                "run_id": self.run_id,
                "measurement_plan_hash": self.preflight.plan_hash,
                "g24b_gate_hash": self.preflight.frozen.g24b_gate_hash,
                "g25_gate_hash": self.preflight.frozen.g25_gate_hash,
                "raw_manifest_hash": self.preflight.frozen.raw_manifest_hash,
                "inventory_hash": self.preflight.inventory["inventory_artifact_hash"],
                "inventory_source_sha256": self.preflight.inventory["inventory_source_sha256"],
                "capacity_evidence_hash": capacity_inputs.get("capacity_evidence_hash"),
                "ulimit_evidence_hash": capacity_inputs.get("ulimit_evidence_hash"),
                "io_evidence_hash": self.preflight.io_evidence["artifact_hash"],
                "gpu_inventory_ref": self.preflight.gpu_inventory_ref,
                "g25_gate_ref": self.preflight.g25_gate_ref,
                "io_evidence_ref": self.preflight.io_evidence_ref,
                "actual_measurements_only": True,
                "failure_evidence_ref": "../failures",
            }
            report_payload["gate"] = gate_payload
            report_payload["artifact_hash"] = canonical_json_hash({key: value for key, value in report_payload.items() if key != "artifact_hash"})
            write_canonical_json(report_path, report_payload)
            os.replace(staging, destination)
            return destination
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


__all__ = [
    "APPROVED_GPU_UUIDS",
    "EXCLUDED_GPU_UUID",
    "EXCLUDED_PCI",
    "S29_ATTEMPT_SCHEMA",
    "S29_FAILURE_SCHEMA",
    "S29_IO_SCHEMA",
    "S29_INVENTORY_SCHEMA",
    "S29_RUNNER_SCHEMA",
    "S29_STATUS_SCHEMA",
    "S29DetachedStatus",
    "S29Preflight",
    "S29ProfilerExecutor",
    "S29ProfilerRunner",
    "S29RunnerBlocked",
    "S29StatusStore",
    "load_s209_preflight",
    "prepare_s209_plan",
    "subprocess_profiler_executor",
    "validate_s209_gpu_inventory",
    "validate_s209_io_evidence",
]
