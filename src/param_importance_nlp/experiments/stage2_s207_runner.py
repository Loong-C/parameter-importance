"""Production S2.7 worker and detached queue.

This module is intentionally separate from the shared Stage 2 task runner.  It
is the execution adapter for the already frozen S2.7 plan: it reads the
S2.6/G2.4b mapping, constructs one real fixed-state provider for a selected
S2.3 checkpoint, and executes each repetition exactly once through
``RecoverablePairedWaveRunner``.  The worker never generates a draw, performs a
statistical reduction, or decides whether an estimator is useful.

The durable boundary is an attempt marker followed by one terminal raw-unit
record.  A committed paired-wave object can be recovered into a raw-unit
record; an abandoned attempt is converted to an explicit failure on recovery,
never silently retried.  The full reducer is deliberately the strict reducer
from :mod:`stage2_s207_formal` and is called only after all six cell waves have
terminal unit records.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.g21_formal_handoff import ALLOWED_DEVICES as APPROVED_GPU_BINDINGS
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..core.registry import ParameterRegistry
from ..g3_runtime_assets import FormalG3RuntimeAssets, G3RuntimeAssetError, formal_pile_route
from ..providers import (
    FixedStateGradientProvider,
    OfflineHuggingFaceModelAdapter,
    PythiaMMapFrozenSampleResolver,
    PretokenizedGlueDatasetAdapter,
    TorchFixedStateGradientProvider,
)
from ..runtime.task_artifacts import load_committed_task_artifact
from ..runtime.task_runtime import TaskExecutionRequest, TaskRuntimeEnvironment
from ..runtime.tensor_bundle import load_tensor_bundle
from ..experiments.stage2_formal import RecoverablePairedWaveRunner, _vector_digest
from ..experiments.sampling import RepetitionMapping
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
    S27CellPlan,
    S27G25Blocked,
    S27MappingUnit,
    S27Plan,
    S27RawUnit,
    S27StatusStore,
    StrictG25Reducer,
    _anchor_to_cell_id,
    validate_gpu_inventory,
)


S27_EXECUTION_SCHEMA = "stage2-s27-production-execution-v1"
S27_ATTEMPT_SCHEMA = "stage2-s27-unit-attempt-v1"
S27_WAVE_SEAL_SCHEMA = "stage2-s27-wave-seal-v1"
S27_LAUNCH_SCHEMA = "stage2-s27-detached-launch-v1"
S27_LAUNCH_STATUS_SCHEMA = "stage2-s27-launch-status-v1"
S27_RAW_RESULT_SCHEMA = "stage2-s27-raw-result-v1"
S27_SHARD_PLAN_SCHEMA = "stage2-s27-unit-shard-plan-v1"
S27_SHARD_SEAL_SCHEMA = "stage2-s27-unit-shard-seal-v1"
S27_DEFAULT_MAX_ATTEMPTS = 1
S27_DEFAULT_M2_TOLERANCE = 1e-10
S27_GPU_INVENTORY_SCHEMA = "stage2-s206-gpu-inventory-v1"
S27_LIVE_GPU_COUNT = 8
S27_INVENTORY_HEALTH_ALIASES = {
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
    "temperature_c": ("temperature_c", "temperature_gpu_c", "temperature.gpu"),
    "row_remap_status": ("row_remap_status", "row_remap", "row_remap_pending"),
    "gpu_recovery_action": ("gpu_recovery_action", "recovery_action"),
}
_S27_CLEAN_VALUES = {"none", "0", "clean", "false", "not_pending", "not-pending", "n/a", "na"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class S27ExecutionBlocked(RuntimeError):
    """Raised when a production input or recovery boundary is not safe."""


def _s27_canonical_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S27ExecutionBlocked(f"{field}:UUID_REQUIRED")
    text = value.strip()
    if not text.upper().startswith("GPU-"):
        text = "GPU-" + text
    return text


def _s27_canonical_pci(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise S27ExecutionBlocked(f"{field}:PCI_REQUIRED")
    match = re.fullmatch(
        r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])",
        value.strip().upper(),
    )
    if match is None:
        raise S27ExecutionBlocked(f"{field}:PCI_INVALID")
    return f"0000:{match.group(1)[-4:]}:{match.group(2)}.{match.group(3)}"


def _s27_finite_number(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"{field}:NUMBER_REQUIRED") from error
    if not np.isfinite(result):
        raise S27ExecutionBlocked(f"{field}:NONFINITE")
    return result


def validate_s27_gpu_inventory(
    inventory: Sequence[Mapping[str, object]],
    *,
    compute_apps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate the complete hash-bound live GPU identity/health snapshot.

    S2.7 runs after a potentially long S2.6 preparation interval.  A bound
    G2.4b Gate therefore cannot substitute for this launch-time check: the
    eight-card identity set, approved-card health, excluded-card ECC facts,
    row-remap state, temperature, and compute-app list must all be present in
    the same immutable envelope.
    """

    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise S27ExecutionBlocked("GPU_INVENTORY_ROWS_REQUIRED")
    if len(inventory) != S27_LIVE_GPU_COUNT:
        raise S27ExecutionBlocked(f"GPU_INVENTORY_LIVE_CARD_COUNT_INVALID:{len(inventory)}")
    if not isinstance(compute_apps, Sequence) or isinstance(compute_apps, (str, bytes)):
        raise S27ExecutionBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")

    approved = {uuid.casefold() for uuid in APPROVED_GPU_UUIDS}
    expected = {
        pci.casefold(): uuid.casefold()
        for pci, uuid in APPROVED_GPU_BINDINGS
    }
    expected[EXCLUDED_PCI.casefold()] = EXCLUDED_GPU_UUID.casefold()
    rows: list[dict[str, object]] = []
    seen_uuid: set[str] = set()
    seen_pci: set[str] = set()
    for index, source in enumerate(inventory):
        if not isinstance(source, Mapping):
            raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_INVALID:{index}")
        row = dict(source)
        uuid = _s27_canonical_uuid(row.get("uuid", row.get("gpu_uuid")), field=f"gpu[{index}]")
        pci = _s27_canonical_pci(row.get("pci_bus_id", row.get("pci")), field=f"gpu[{index}]")
        uuid_key = uuid.casefold()
        pci_key = pci.casefold()
        if uuid_key in seen_uuid:
            raise S27ExecutionBlocked("GPU_INVENTORY_DUPLICATE_UUID")
        if pci_key in seen_pci:
            raise S27ExecutionBlocked("GPU_INVENTORY_DUPLICATE_PCI")
        seen_uuid.add(uuid_key)
        seen_pci.add(pci_key)
        row["uuid"] = uuid
        row["pci_bus_id"] = pci
        for canonical, aliases in S27_INVENTORY_HEALTH_ALIASES.items():
            found = next((row[name] for name in aliases if name in row), None)
            if found is None:
                raise S27ExecutionBlocked(f"GPU_INVENTORY_HEALTH_FIELD_MISSING:{canonical}")
            row[canonical] = found
        # Require numeric evidence for every card, including the quarantined
        # card; only approved-card values are subject to clean-health gates.
        for field in (
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_percent",
            "ecc_uncorrected_volatile",
            "ecc_uncorrected_aggregate",
            "temperature_c",
        ):
            row[field] = _s27_finite_number(row[field], field=f"gpu[{index}].{field}")
        rows.append(row)

    observed = {
        str(row["pci_bus_id"]).casefold(): str(row["uuid"]).casefold()
        for row in rows
        if str(row["pci_bus_id"]).casefold() in expected
    }
    if observed != expected:
        raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_OR_EXCLUDED_IDENTITY_DRIFT")

    apps_by_uuid: dict[str, list[dict[str, object]]] = {}
    live_uuids = {str(row["uuid"]).casefold() for row in rows}
    for index, source in enumerate(compute_apps):
        if not isinstance(source, Mapping):
            raise S27ExecutionBlocked(f"GPU_INVENTORY_COMPUTE_APP_INVALID:{index}")
        app = dict(source)
        app_uuid = _s27_canonical_uuid(app.get("gpu_uuid", app.get("uuid")), field=f"compute_app[{index}]")
        if app_uuid.casefold() not in live_uuids:
            raise S27ExecutionBlocked("GPU_INVENTORY_COMPUTE_APP_GPU_UNKNOWN")
        app["gpu_uuid"] = app_uuid
        apps_by_uuid.setdefault(app_uuid.casefold(), []).append(app)

    for row in rows:
        uuid_key = str(row["uuid"]).casefold()
        if uuid_key not in approved:
            continue
        if apps_by_uuid.get(uuid_key):
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
        if (
            float(row["memory_total_mib"]) <= 0
            or float(row["memory_used_mib"]) != 0
            or float(row["utilization_gpu_percent"]) != 0
        ):
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
        if float(row["ecc_uncorrected_volatile"]) != 0 or float(row["ecc_uncorrected_aggregate"]) != 0:
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_CARD_ECC_NOT_CLEAN")
        if float(row["temperature_c"]) < 0 or float(row["temperature_c"]) >= 85:
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_CARD_TEMPERATURE_INVALID")
        if str(row["row_remap_status"]).strip().casefold() not in _S27_CLEAN_VALUES:
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_ROW_REMAP_NOT_CLEAN")
        if str(row["gpu_recovery_action"]).strip().casefold() not in _S27_CLEAN_VALUES:
            raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_CARD_RECOVERY_NOT_CLEAN")

    return {
        "schema_version": S27_GPU_INVENTORY_SCHEMA,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "inventory_count": len(rows),
        "inventory": rows,
        "compute_apps": [dict(item) for item in compute_apps],
    }


def _s27_inventory_path(root: Path, path: str | Path) -> Path:
    resolved_root = root.resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise S27ExecutionBlocked("GPU_INVENTORY_PATH_OUTSIDE_DATA_ROOT") from error
    return resolved


def _s27_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_S27_FORMAL_GPU_INVENTORY_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "status",
        "checked_at",
        "artifact_ref",
        "source_ref",
        "source_sha256",
        "rows",
        "compute_apps",
        "approved_gpu_uuids",
        "excluded_pci",
        "excluded_gpu_uuid",
        "artifact_hash",
    }
)
_S27_FORMAL_GPU_ROW_FIELDS = frozenset(
    {
        "uuid",
        "pci_bus_id",
        "gpu_name",
        "temperature_c",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
        "compute_mode",
        "ecc_uncorrected_volatile",
        "ecc_uncorrected_aggregate",
        "row_remap_failure",
        "row_remap_pending",
        "row_remap_status",
        "gpu_recovery_action",
        "health_state",
        "compute_apps",
    }
)
_S27_FORMAL_GPU_APP_FIELDS = frozenset({"pid", "gpu_uuid", "process_name", "used_memory"})


def _validate_s27_formal_gpu_inventory_schema(value: Mapping[str, object]) -> None:
    """Validate the exact S2.6 inventory schema before S2.7 normalization."""

    if "gpus" in value:
        raise S27ExecutionBlocked("GPU_INVENTORY_GPUS_ALIAS_FORBIDDEN")
    if "artifact_ref" not in value:
        raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_REF_REQUIRED")
    if "source_sha256" not in value:
        raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_SHA256_REQUIRED")
    unknown = set(value) - _S27_FORMAL_GPU_INVENTORY_FIELDS
    if unknown:
        raise S27ExecutionBlocked("GPU_INVENTORY_TOP_LEVEL_UNKNOWN_FIELDS")
    if set(value) != _S27_FORMAL_GPU_INVENTORY_FIELDS:
        raise S27ExecutionBlocked("GPU_INVENTORY_TOP_LEVEL_FIELDS_REQUIRED")
    if value.get("scope") != "formal":
        raise S27ExecutionBlocked("GPU_INVENTORY_SCOPE_INVALID")
    if value.get("status") != "OBSERVED":
        raise S27ExecutionBlocked("GPU_INVENTORY_STATUS_INVALID")
    checked_at = value.get("checked_at")
    if not isinstance(checked_at, str) or not checked_at:
        raise S27ExecutionBlocked("GPU_INVENTORY_CHECKED_AT_REQUIRED")
    try:
        checked = datetime.fromisoformat(checked_at)
    except ValueError as error:
        raise S27ExecutionBlocked("GPU_INVENTORY_CHECKED_AT_INVALID") from error
    if checked.tzinfo is None:
        raise S27ExecutionBlocked("GPU_INVENTORY_CHECKED_AT_TIMEZONE_REQUIRED")
    approved = value.get("approved_gpu_uuids")
    if not isinstance(approved, list) or tuple(approved) != APPROVED_GPU_UUIDS:
        raise S27ExecutionBlocked("GPU_INVENTORY_APPROVED_UUIDS_IDENTITY_DRIFT")
    if value.get("excluded_pci") != EXCLUDED_PCI or value.get("excluded_gpu_uuid") != EXCLUDED_GPU_UUID:
        raise S27ExecutionBlocked("GPU_INVENTORY_EXCLUDED_IDENTITY_DRIFT")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise S27ExecutionBlocked("GPU_INVENTORY_ROWS_REQUIRED")
    for index, item in enumerate(rows):
        if not isinstance(item, Mapping):
            raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_INVALID:{index}")
        if set(item) != _S27_FORMAL_GPU_ROW_FIELDS:
            raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_FIELDS_INVALID:{index}")
        for field in (
            "uuid",
            "pci_bus_id",
            "gpu_name",
            "compute_mode",
            "row_remap_status",
            "gpu_recovery_action",
            "health_state",
        ):
            if not isinstance(item[field], str) or not item[field]:
                raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_{field.upper()}_INVALID:{index}")
        for field in ("temperature_c", "utilization_gpu_percent"):
            number = item[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(float(number)):
                raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_{field.upper()}_INVALID:{index}")
        for field in (
            "memory_used_mib",
            "memory_total_mib",
            "ecc_uncorrected_volatile",
            "ecc_uncorrected_aggregate",
            "row_remap_failure",
            "row_remap_pending",
        ):
            number = item[field]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_{field.upper()}_INVALID:{index}")
        row_apps = item["compute_apps"]
        if not isinstance(row_apps, list):
            raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_COMPUTE_APPS_INVALID:{index}")
        for app_index, app in enumerate(row_apps):
            if not isinstance(app, Mapping) or set(app) != _S27_FORMAL_GPU_APP_FIELDS:
                raise S27ExecutionBlocked(f"GPU_INVENTORY_ROW_APP_FIELDS_INVALID:{index}:{app_index}")
    apps = value.get("compute_apps")
    if not isinstance(apps, list):
        raise S27ExecutionBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    for index, app in enumerate(apps):
        if not isinstance(app, Mapping) or set(app) != _S27_FORMAL_GPU_APP_FIELDS:
            raise S27ExecutionBlocked(f"GPU_INVENTORY_APP_FIELDS_INVALID:{index}")


def load_s27_gpu_inventory_envelope(
    path: str | Path | None,
    *,
    data_root: str | Path,
    _allow_legacy: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load the source-bound S2.6 inventory consumed by formal S2.7 paths.

    The formal default requires an artifact reference, a distinct raw-capture
    reference inside ``data_root``, and its declared SHA-256.  ``_allow_legacy``
    is a private non-formal fixture escape hatch only; all S2.7 launch and
    materialization callers use the strict default.
    """

    if path is None:
        raise S27ExecutionBlocked("GPU_INVENTORY_JSON_REQUIRED")
    root = Path(data_root).resolve()
    resolved = _s27_inventory_path(root, path)
    expected_artifact_ref = resolved.relative_to(root).as_posix()
    try:
        value = load_canonical_json(resolved)
    except (OSError, TypeError, ValueError) as error:
        raise S27ExecutionBlocked("GPU_INVENTORY_JSON_INVALID") from error
    if not isinstance(value, Mapping):
        raise S27ExecutionBlocked("GPU_INVENTORY_ENVELOPE_REQUIRED")
    payload = dict(value)
    if payload.get("schema_version") != S27_GPU_INVENTORY_SCHEMA:
        raise S27ExecutionBlocked("GPU_INVENTORY_SCHEMA_INVALID")
    if not _allow_legacy:
        _validate_s27_formal_gpu_inventory_schema(payload)
    source_ref = payload.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref or "\\" in source_ref:
        raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_REF_REQUIRED")
    artifact_ref = payload.get("artifact_ref")
    if artifact_ref is not None:
        if not isinstance(artifact_ref, str) or not artifact_ref or "\\" in artifact_ref:
            raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_REF_INVALID")
        if artifact_ref != expected_artifact_ref:
            raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_REF_PATH_MISMATCH")
        try:
            source_posix = PurePosixPath(source_ref)
            if (
                source_posix.is_absolute()
                or Path(source_ref).is_absolute()
                or Path(source_ref).drive
                or any(part in {"", ".", ".."} for part in source_posix.parts)
            ):
                raise ValueError(source_ref)
            source_path = (root / Path(*source_posix.parts)).resolve()
            source_path.relative_to(root)
        except (OSError, ValueError) as error:
            raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_REF_PATH_INVALID") from error
        if source_ref != source_path.relative_to(root).as_posix():
            raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_REF_NOT_CANONICAL")
        if source_path == resolved:
            raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_REF_SELF_REFERENCE")
    else:
        if not _allow_legacy:
            raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_REF_REQUIRED")
        # Legacy envelope: source_ref was the inventory itself.  A declared
        # source digest is checked below and rejected in that shape because it
        # would be a self-reference rather than a raw capture digest.
        if source_ref != expected_artifact_ref:
            raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_REF_PATH_MISMATCH")
        source_path = resolved
    rows = payload.get("rows")
    apps = payload.get("compute_apps")
    if not isinstance(rows, list) or not all(isinstance(item, Mapping) for item in rows):
        raise S27ExecutionBlocked("GPU_INVENTORY_ROWS_REQUIRED")
    if not isinstance(apps, list) or not all(isinstance(item, Mapping) for item in apps):
        raise S27ExecutionBlocked("GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    artifact_hash = payload.get("artifact_hash")
    if not isinstance(artifact_hash, str) or _SHA256.fullmatch(artifact_hash) is None:
        raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_HASH_REQUIRED")
    body = {key: item for key, item in payload.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != artifact_hash:
        raise S27ExecutionBlocked("GPU_INVENTORY_ARTIFACT_HASH_MISMATCH")
    source_sha = _s27_file_sha256(source_path)
    declared_source_sha = payload.get("source_sha256")
    if declared_source_sha is None and not _allow_legacy:
        raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_SHA256_REQUIRED")
    if declared_source_sha is not None and declared_source_sha != source_sha:
        raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_SHA256_MISMATCH")
    if declared_source_sha is not None and artifact_ref is None:
        raise S27ExecutionBlocked("GPU_INVENTORY_SOURCE_SHA256_SELF_REFERENCE")
    summary = validate_s27_gpu_inventory(rows, compute_apps=apps)
    identity = {
        "source_ref": source_ref,
        "artifact_ref": artifact_ref or expected_artifact_ref,
        "artifact_hash": artifact_hash,
        "source_sha256": source_sha,
        "schema_version": S27_GPU_INVENTORY_SCHEMA,
    }
    summary.update(
        {
            "inventory_source_ref": source_ref,
            "inventory_artifact_hash": artifact_hash,
            "inventory_source_sha256": source_sha,
            "inventory_schema_version": S27_GPU_INVENTORY_SCHEMA,
        }
    )
    return summary, identity


class S27ProviderFactory(Protocol):
    def __call__(
        self,
        cell: S27CellPlan,
        *,
        data_root: Path,
        checkpoint_root_ref: str,
        registry_ref: str,
        request: TaskExecutionRequest | None = None,
    ) -> "S27ProviderContext": ...


@dataclass(frozen=True, slots=True)
class S27ProviderContext:
    """The only provider state accepted by a production cell worker."""

    provider: FixedStateGradientProvider
    execution: FormalExecutionEvidence
    checkpoint_root_ref: str
    checkpoint_hash: str
    registry_ref: str
    registry_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, FixedStateGradientProvider):
            raise TypeError("S27_PROVIDER_MUST_IMPLEMENT_FIXED_STATE_CONTRACT")
        if not isinstance(self.execution, FormalExecutionEvidence):
            raise TypeError("S27_FORMAL_EXECUTION_EVIDENCE_REQUIRED")
        if not _SHA256.fullmatch(self.checkpoint_hash):
            raise ValueError("S27_CHECKPOINT_HASH_INVALID")
        if not _SHA256.fullmatch(self.registry_hash):
            raise ValueError("S27_REGISTRY_HASH_INVALID")


@dataclass(frozen=True, slots=True)
class S27RetryPolicy:
    """Frozen retry policy.

    A confirmatory unit has one attempt.  An OOM retry is permitted only if an
    upstream frozen artifact explicitly names that unit and the caller supplies
    a smaller-forward provider; this module does not invent such a provider.
    Consequently the default policy is intentionally no-retry.
    """

    max_attempts: int = S27_DEFAULT_MAX_ATTEMPTS
    pre_registered_oom_unit_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or self.max_attempts != 1:
            raise S27ExecutionBlocked("S27_RETRY_POLICY_MUST_BE_SINGLE_ATTEMPT")
        if len(set(self.pre_registered_oom_unit_ids)) != len(self.pre_registered_oom_unit_ids):
            raise S27ExecutionBlocked("S27_RETRY_POLICY_DUPLICATE_UNIT")
        if self.pre_registered_oom_unit_ids:
            # There is deliberately no implicit smaller-forward implementation.
            raise S27ExecutionBlocked("S27_OOM_RETRY_REQUIRES_EXPLICIT_SMALLER_FORWARD_ADAPTER")


@dataclass(frozen=True, slots=True)
class S27UnitShard:
    """Deterministic partition of one frozen checkpoint wave.

    A shard owns complete repetitions, never a prefix of a repetition.  The
    mapping identity hashes are carried in the plan so a worker cannot turn a
    shard into a draw generator or silently substitute another unit.
    """

    shard_index: int
    shard_count: int
    gpu_uuid: str
    unit_ids: tuple[str, ...]
    unit_mapping_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if isinstance(self.shard_index, bool) or not isinstance(self.shard_index, int) or self.shard_index < 0:
            raise ValueError("S27_SHARD_INDEX_INVALID")
        if isinstance(self.shard_count, bool) or not isinstance(self.shard_count, int) or self.shard_count <= 0 or self.shard_index >= self.shard_count:
            raise ValueError("S27_SHARD_COUNT_INVALID")
        if self.gpu_uuid not in APPROVED_GPU_UUIDS:
            raise ValueError("S27_SHARD_GPU_UNAPPROVED")
        if not self.unit_ids or any(not isinstance(unit_id, str) or not _SAFE_COMPONENT.fullmatch(unit_id) for unit_id in self.unit_ids) or len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("S27_SHARD_UNIT_IDS_INVALID")
        if tuple(unit_id for unit_id, _ in self.unit_mapping_hashes) != self.unit_ids:
            raise ValueError("S27_SHARD_UNIT_HASH_ORDER_INVALID")
        if any(not isinstance(mapping_hash, str) or not _SHA256.fullmatch(mapping_hash) for _, mapping_hash in self.unit_mapping_hashes):
            raise ValueError("S27_SHARD_MAPPING_HASH_INVALID")

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "gpu_uuid": self.gpu_uuid,
            "unit_ids": list(self.unit_ids),
            "unit_mapping_hashes": {unit_id: mapping_hash for unit_id, mapping_hash in self.unit_mapping_hashes},
        }


def _safe_ref(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S27ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE")
    current = root
    if root.is_symlink():
        raise S27ExecutionBlocked(f"{field}:SYMLINK_ROOT")
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise S27ExecutionBlocked(f"{field}:SYMLINK_COMPONENT")
    target = (root.joinpath(*logical.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    return target


def _relative_ref(root: Path, path: Path, *, field: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise S27ExecutionBlocked(f"{field}:OUTSIDE_DATA_ROOT") from error
    if not relative or ".." in PurePosixPath(relative).parts:
        raise S27ExecutionBlocked(f"{field}:INVALID_RELATIVE_REFERENCE")
    return relative


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_payload(value: Mapping[str, object], *, field: str) -> tuple[dict[str, object], str]:
    body = {str(key): item for key, item in value.items() if key != "artifact_hash"}
    try:
        digest = canonical_json_hash(body)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"{field}:NOT_CANONICAL_JSON") from error
    declared = value.get("artifact_hash")
    if declared != digest:
        raise S27ExecutionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return dict(value), digest


def _load_payload(root: Path, reference: str, *, field: str) -> tuple[dict[str, object], str, str]:
    """Load either a direct canonical payload or a formal task commit."""

    path = _safe_ref(root, reference, field=field)
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping):
        raise S27ExecutionBlocked(f"{field}:OBJECT_REQUIRED")
    if raw.get("schema_version") == "task-output-commit-v1":
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S27ExecutionBlocked(f"{field}:FORMAL_TASK_COMMIT_INVALID") from error
        payload = loaded.payload
        if not isinstance(payload, Mapping):
            raise S27ExecutionBlocked(f"{field}:TASK_PAYLOAD_INVALID")
        return dict(payload), loaded.identity.artifact_hash, reference
    payload, digest = _canonical_payload(raw, field=field)
    return payload, digest, reference


def load_s27_plan(data_root: str | Path, plan_ref: str) -> S27Plan:
    root = Path(data_root).resolve()
    payload, _, _ = _load_payload(root, plan_ref, field="s27_plan")
    try:
        return S27Plan.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_PLAN_INVALID:{error}") from error


def partition_s27_units(
    plan: S27Plan,
    cell_id: str,
    *,
    gpu_uuids: Sequence[str] = APPROVED_GPU_UUIDS,
) -> tuple[S27UnitShard, ...]:
    """Split one frozen cell into deterministic contiguous repetition shards."""

    if cell_id not in EXPECTED_CELL_IDS:
        raise S27ExecutionBlocked(f"S27_SHARD_CELL_UNKNOWN:{cell_id}")
    cards = tuple(gpu_uuids)
    if cards != APPROVED_GPU_UUIDS or len(set(cards)) != len(cards):
        raise S27ExecutionBlocked("S27_SHARD_GPU_ALLOWLIST_DRIFT")
    expected = tuple(unit for unit in plan.frozen_inputs.units if unit.cell_id == cell_id)
    if not expected:
        raise S27ExecutionBlocked(f"S27_SHARD_CELL_EMPTY:{cell_id}")
    q, remainder = divmod(len(expected), len(cards))
    shards: list[S27UnitShard] = []
    offset = 0
    for shard_index, gpu_uuid in enumerate(cards):
        count = q + (1 if shard_index < remainder else 0)
        selected = expected[offset : offset + count]
        offset += count
        if not selected:
            raise S27ExecutionBlocked("S27_SHARD_EMPTY_PARTITION")
        shards.append(
            S27UnitShard(
                shard_index=shard_index,
                shard_count=len(cards),
                gpu_uuid=gpu_uuid,
                unit_ids=tuple(unit.unit_id for unit in selected),
                unit_mapping_hashes=tuple((unit.unit_id, unit.mapping_hash) for unit in selected),
            )
        )
    if offset != len(expected) or set(item for shard in shards for item in shard.unit_ids) != {unit.unit_id for unit in expected}:
        raise S27ExecutionBlocked("S27_SHARD_COVERAGE_INTERNAL_ERROR")
    return tuple(shards)


def _shard_plan_path(run_root: Path, cell_id: str) -> Path:
    return run_root / "wave-shards" / cell_id.replace(":", "__") / "shard-plan.json"


def write_s27_shard_plan(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    run_id: str,
    cell_id: str,
) -> tuple[S27UnitShard, ...]:
    """Persist the immutable shard partition before any child is launched."""

    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    try:
        run.relative_to(root)
    except ValueError as error:
        raise S27ExecutionBlocked("S27_SHARD_RUN_ROOT_OUTSIDE_DATA_ROOT") from error
    shards = partition_s27_units(plan, cell_id)
    body: dict[str, object] = {
        "schema_version": S27_SHARD_PLAN_SCHEMA,
        "run_id": run_id,
        "plan_hash": plan.artifact_hash,
        "cell_id": cell_id,
        "shard_count": len(shards),
        "expected_unit_ids": [unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id],
        "shards": [shard.to_dict() for shard in shards],
    }
    body["artifact_hash"] = canonical_json_hash(body)
    _write_once(_shard_plan_path(run, cell_id), body, field=f"S27_SHARD_PLAN:{cell_id}")
    return shards


def load_s27_shard_plan(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    run_id: str,
    cell_id: str,
    shard_plan_ref: str | None = None,
) -> tuple[S27UnitShard, ...]:
    """Reload and rederive the exact four-card partition; never trust CLI IDs."""

    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    expected_path = _shard_plan_path(run, cell_id).resolve()
    path = expected_path if shard_plan_ref is None else _safe_ref(root, shard_plan_ref, field=f"S27_SHARD_PLAN_REF:{cell_id}")
    if path != expected_path:
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_REF_MISMATCH:{cell_id}")
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != S27_SHARD_PLAN_SCHEMA:
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_SCHEMA_INVALID:{cell_id}")
    if raw.get("artifact_hash") != canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}):
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_HASH_INVALID:{cell_id}")
    if raw.get("run_id") != run_id or raw.get("plan_hash") != plan.artifact_hash or raw.get("cell_id") != cell_id:
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_LINEAGE_INVALID:{cell_id}")
    expected = tuple(unit for unit in plan.frozen_inputs.units if unit.cell_id == cell_id)
    if raw.get("expected_unit_ids") != [unit.unit_id for unit in expected]:
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_EXPECTED_IDS_INVALID:{cell_id}")
    rows = raw.get("shards")
    if not isinstance(rows, list) or raw.get("shard_count") != len(APPROVED_GPU_UUIDS) or len(rows) != len(APPROVED_GPU_UUIDS):
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_COUNT_INVALID:{cell_id}")
    by_index: dict[int, S27UnitShard] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"shard_index", "shard_count", "gpu_uuid", "unit_ids", "unit_mapping_hashes"}:
            raise S27ExecutionBlocked(f"S27_SHARD_PLAN_ROW_INVALID:{cell_id}")
        ids = row.get("unit_ids")
        hashes = row.get("unit_mapping_hashes")
        if not isinstance(ids, list) or not isinstance(hashes, Mapping):
            raise S27ExecutionBlocked(f"S27_SHARD_PLAN_ROW_FIELDS_INVALID:{cell_id}")
        if set(hashes) != set(ids):
            raise S27ExecutionBlocked(f"S27_SHARD_PLAN_ROW_HASH_SET_INVALID:{cell_id}")
        try:
            shard = S27UnitShard(
                shard_index=row["shard_index"],
                shard_count=row["shard_count"],
                gpu_uuid=row["gpu_uuid"],
                unit_ids=tuple(ids),
                unit_mapping_hashes=tuple((unit_id, hashes.get(unit_id)) for unit_id in ids),
            )
        except (TypeError, ValueError) as error:
            raise S27ExecutionBlocked(f"S27_SHARD_PLAN_ROW_INVALID:{cell_id}") from error
        if shard.shard_index in by_index:
            raise S27ExecutionBlocked(f"S27_SHARD_PLAN_DUPLICATE_INDEX:{cell_id}")
        by_index[shard.shard_index] = shard
    observed = tuple(by_index[index] for index in sorted(by_index)) if set(by_index) == set(range(len(APPROVED_GPU_UUIDS))) else ()
    if observed != partition_s27_units(plan, cell_id):
        raise S27ExecutionBlocked(f"S27_SHARD_PLAN_PARTITION_DRIFT:{cell_id}")
    return observed


def load_s27_frozen_mappings(
    data_root: str | Path,
    plan: S27Plan,
    *,
    cell_id: str | None = None,
) -> dict[str, RepetitionMapping]:
    """Read the exact S2.6 mappings; this function never calls a draw API."""

    root = Path(data_root).resolve()
    payload, digest, _ = _load_payload(root, plan.frozen_inputs.mapping_ref, field="s27_mapping")
    if digest != plan.frozen_inputs.mapping_hash:
        raise S27ExecutionBlocked("S27_MAPPING_HASH_DRIFT")
    if payload.get("schema_version") != "stage2-formal-confirmatory-mapping-v1":
        raise S27ExecutionBlocked("S27_MAPPING_SCHEMA_INVALID")
    if payload.get("scope") != "formal" or payload.get("stream") != "confirmatory" or payload.get("formal_eligible") is not True or payload.get("complete") is not True or payload.get("draw_id_unique") is not True:
        raise S27ExecutionBlocked("S27_MAPPING_FORMAL_SCOPE_INVALID")
    if payload.get("freeze_hash") != plan.frozen_inputs.matrix_hash or payload.get("qualification_gate_hash") != plan.frozen_inputs.g24b_gate_hash or payload.get("sampling_plan_hash") != plan.frozen_inputs.sampling_plan_hash:
        raise S27ExecutionBlocked("S27_MAPPING_FREEZE_LINEAGE_INVALID")
    result: dict[str, RepetitionMapping] = {}
    observed_draw_ids: set[str] = set()
    observed_positions: set[tuple[str, int]] = set()
    expected_units = {
        unit.unit_id: unit
        for unit in plan.frozen_inputs.units
        if cell_id is None or unit.cell_id == cell_id
    }
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise S27ExecutionBlocked("S27_MAPPING_CELLS_REQUIRED")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise S27ExecutionBlocked("S27_MAPPING_CELL_NOT_OBJECT")
        anchor = cell.get("anchor_id")
        try:
            canonical_cell_id = _anchor_to_cell_id(anchor, field=f"S27_MAPPING_ANCHOR:{anchor}")
        except ValueError as error:
            raise S27ExecutionBlocked("S27_MAPPING_ANCHOR_INVALID") from error
        if cell_id is not None and canonical_cell_id != cell_id:
            continue
        rows = cell.get("mappings")
        if not isinstance(rows, list):
            raise S27ExecutionBlocked(f"S27_MAPPING_ROWS_REQUIRED:{anchor}")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise S27ExecutionBlocked(f"S27_MAPPING_ROW_NOT_OBJECT:{anchor}")
            try:
                mapping = RepetitionMapping.from_manifest(raw)
            except (TypeError, ValueError) as error:
                raise S27ExecutionBlocked(f"S27_MAPPING_REPETITION_INVALID:{anchor}") from error
            unit_id = f"{canonical_cell_id}::{mapping.repetition_id}"
            expected = expected_units.get(unit_id)
            if expected is None:
                raise S27ExecutionBlocked(f"S27_MAPPING_UNEXPECTED_UNIT:{unit_id}")
            if mapping.digest != expected.mapping_hash:
                raise S27ExecutionBlocked(f"S27_MAPPING_UNIT_HASH_DRIFT:{unit_id}")
            if mapping.batch_size != plan.frozen_inputs.batch_size or mapping.m_values != (2, plan.frozen_inputs.microbatch_count):
                raise S27ExecutionBlocked(f"S27_MAPPING_B_M_DRIFT:{unit_id}")
            for draw in mapping.draws:
                position = (str(draw.stream), int(draw.position))
                if draw.draw_id in observed_draw_ids or position in observed_positions:
                    raise S27ExecutionBlocked(f"S27_MAPPING_GLOBAL_DRAW_COLLISION:{unit_id}")
                observed_draw_ids.add(str(draw.draw_id))
                observed_positions.add(position)
            if tuple(draw.draw_id for draw in mapping.draws) != expected.draw_ids or tuple(draw.sample_id for draw in mapping.draws) != expected.sample_ids:
                raise S27ExecutionBlocked(f"S27_MAPPING_DRAW_SAMPLE_DRIFT:{unit_id}")
            if unit_id in result:
                raise S27ExecutionBlocked(f"S27_MAPPING_DUPLICATE_UNIT:{unit_id}")
            result[unit_id] = mapping
    if set(result) != set(expected_units):
        missing = sorted(set(expected_units) - set(result))
        raise S27ExecutionBlocked(f"S27_MAPPING_MISSING_UNIT:{','.join(missing[:8])}")
    return result


def load_s27_reference_views(
    data_root: str | Path,
    cell: S27CellPlan,
    *,
    expected_registry_hash: str | None = None,
    reference_output_root_ref: str | None = None,
) -> dict[str, Mapping[str, object]]:
    """Load the independent S2.4 reference bundle and its G2.3 PASS gate."""

    root = Path(data_root).resolve()
    reference, reference_digest, reference_ref = _load_payload(root, cell.reference_ref, field=f"reference.{cell.cell_id}")
    if reference_digest != cell.reference_hash:
        raise S27ExecutionBlocked(f"S27_REFERENCE_ARTIFACT_HASH_MISMATCH:{cell.cell_id}")
    try:
        gate_payload, gate_hash, _ = _load_payload(root, cell.reference_gate_ref, field=f"reference_gate.{cell.cell_id}")
        gate = GateRecord.from_mapping(dict(gate_payload))
    except (OSError, TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_INVALID:{cell.cell_id}") from error
    if gate_hash != cell.reference_gate_hash or gate.gate_id != "stage2.G2.3" or gate.status is not GateStatus.PASS:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_NOT_PASS:{cell.cell_id}")
    if cell.reference_ref not in gate.evidence_refs:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_DOES_NOT_BIND_REFERENCE:{cell.cell_id}")
    if reference.get("schema_version") != "reference-result-v1" or reference.get("scope") != "formal":
        raise S27ExecutionBlocked(f"S27_REFERENCE_SCOPE_INVALID:{cell.cell_id}")
    if expected_registry_hash is not None and reference.get("registry_hash") != expected_registry_hash:
        raise S27ExecutionBlocked(f"S27_REFERENCE_REGISTRY_MISMATCH:{cell.cell_id}")
    bundle_ref = reference.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str):
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_REF_MISSING:{cell.cell_id}")
    bundle_candidates: list[Path] = []
    try:
        bundle_candidates.append(_safe_ref(root, bundle_ref, field=f"reference_bundle.{cell.cell_id}"))
    except S27ExecutionBlocked:
        pass
    if reference_output_root_ref is not None:
        _safe_ref(root, reference_output_root_ref, field=f"reference_output_root.{cell.cell_id}")
        logical_bundle = PurePosixPath(reference_output_root_ref) / PurePosixPath(bundle_ref)
        bundle_candidates.append(
            _safe_ref(root, logical_bundle.as_posix(), field=f"reference_bundle.{cell.cell_id}.output_root")
        )
    reference_path = _safe_ref(root, reference_ref, field=f"reference.{cell.cell_id}")
    try:
        logical_parent = PurePosixPath(reference_ref).parent / PurePosixPath(bundle_ref)
        bundle_candidates.append(
            _safe_ref(root, logical_parent.as_posix(), field=f"reference_bundle.{cell.cell_id}.parent")
        )
    except (ValueError, S27ExecutionBlocked):
        pass
    bundle_path = next(
        (
            candidate
            for candidate in bundle_candidates
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        None,
    )
    if bundle_path is None:
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_MISSING:{cell.cell_id}")
    try:
        state, bundle = load_tensor_bundle(bundle_path)
    except (OSError, TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_INVALID:{cell.cell_id}") from error
    if bundle.manifest_sha256 != reference.get("tensor_bundle_manifest_hash") or not isinstance(state, Mapping):
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_HASH_MISMATCH:{cell.cell_id}")
    expected_names = {"bias_reference", "cross_reference", "ranking_reference"}
    if not expected_names.issubset(set(state)):
        raise S27ExecutionBlocked(f"S27_REFERENCE_VIEWS_INVALID:{cell.cell_id}")
    views: dict[str, Mapping[str, object]] = {}
    for short, long_name in (("bias", "bias_reference"), ("cross", "cross_reference"), ("ranking", "ranking_reference")):
        value = state[long_name]
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_REFERENCE_VIEW_NOT_MAPPING:{cell.cell_id}:{short}")
        declared = reference.get(f"{short}_reference_hash")
        if declared != _vector_digest(value):
            raise S27ExecutionBlocked(f"S27_REFERENCE_VIEW_HASH_MISMATCH:{cell.cell_id}:{short}")
        views[short] = value
    return views


@dataclass(frozen=True, slots=True)
class S27MaterializedCellInput:
    """S2.4 materialization row consumed by the S2.7 worker."""

    cell_id: str
    config_ref: str
    environment_ref: str
    checkpoint_root_ref: str
    registry_ref: str
    reference_output_root_ref: str = ""

    def __post_init__(self) -> None:
        if self.cell_id not in EXPECTED_CELL_IDS:
            raise ValueError("S27_MATERIALIZED_CELL_UNKNOWN")
        for field, value in (("config_ref", self.config_ref), ("environment_ref", self.environment_ref), ("checkpoint_root_ref", self.checkpoint_root_ref), ("registry_ref", self.registry_ref)):
            if not isinstance(value, str) or not value or "\\" in value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
                raise ValueError(f"S27_MATERIALIZED_{field.upper()}_INVALID")
        if self.reference_output_root_ref and ("\\" in self.reference_output_root_ref or PurePosixPath(self.reference_output_root_ref).is_absolute() or ".." in PurePosixPath(self.reference_output_root_ref).parts):
            raise ValueError("S27_MATERIALIZED_REFERENCE_OUTPUT_ROOT_REF_INVALID")


def load_s27_materialized_inputs(data_root: str | Path, index_ref: str) -> dict[str, S27MaterializedCellInput]:
    root = Path(data_root).resolve()
    index_path = _safe_ref(root, index_ref, field="s27_materialization_index")
    raw_index = load_canonical_json(index_path)
    if not isinstance(raw_index, Mapping):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_OBJECT_REQUIRED")
    if raw_index.get("schema_version") == "task-output-commit-v1":
        payload, _, _ = _load_payload(root, index_ref, field="s27_materialization_index")
    else:
        declared_index_hash = raw_index.get("index_hash")
        index_body = {key: item for key, item in raw_index.items() if key != "index_hash"}
        if declared_index_hash != canonical_json_hash(index_body):
            raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_HASH_INVALID")
        payload = dict(raw_index)
    if payload.get("schema_version") != "stage2-s204-six-cell-materialization-index-v1":
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_SCHEMA_INVALID")
    rows = payload.get("cells")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CELL_IDS):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_SIX_CELL_REQUIRED")
    manifest_ref = payload.get("six_cell_manifest_ref")
    if not isinstance(manifest_ref, str):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_MANIFEST_REF_MISSING")
    manifest_path = _safe_ref(root, manifest_ref, field="s27_six_cell_manifest")
    raw_manifest = load_canonical_json(manifest_path)
    if not isinstance(raw_manifest, Mapping):
        raise S27ExecutionBlocked("S27_SIX_CELL_MANIFEST_OBJECT_REQUIRED")
    if raw_manifest.get("schema_version") == "task-output-commit-v1":
        manifest, _, _ = _load_payload(root, manifest_ref, field="s27_six_cell_manifest")
    else:
        declared_manifest_hash = raw_manifest.get("manifest_hash")
        manifest_body = {key: item for key, item in raw_manifest.items() if key != "manifest_hash"}
        if declared_manifest_hash != canonical_json_hash(manifest_body):
            raise S27ExecutionBlocked("S27_SIX_CELL_MANIFEST_HASH_INVALID")
        manifest = dict(raw_manifest)
    checkpoint_rows = manifest.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != len(EXPECTED_CELL_IDS):
        raise S27ExecutionBlocked("S27_SIX_CELL_CHECKPOINT_ROWS_INVALID")
    checkpoint_by_cell: dict[str, Mapping[str, object]] = {}
    for checkpoint in checkpoint_rows:
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("cell_id"), str):
            raise S27ExecutionBlocked("S27_SIX_CELL_CHECKPOINT_ROW_INVALID")
        checkpoint_by_cell[str(checkpoint["cell_id"])] = checkpoint
    result: dict[str, S27MaterializedCellInput] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise S27ExecutionBlocked("S27_MATERIALIZATION_ROW_INVALID")
        cell_id = raw.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in result:
            raise S27ExecutionBlocked("S27_MATERIALIZATION_CELL_SET_INVALID")
        checkpoint = checkpoint_by_cell.get(cell_id)
        if checkpoint is None or not isinstance(checkpoint.get("checkpoint_root_ref"), str):
            raise S27ExecutionBlocked(f"S27_CHECKPOINT_ROOT_REF_MISSING:{cell_id}")
        config_ref = raw.get("config_ref")
        if not isinstance(config_ref, str):
            raise S27ExecutionBlocked(f"S27_CONFIG_REF_MISSING:{cell_id}")
        environment_ref = raw.get("environment_ref")
        if not isinstance(environment_ref, str) or not environment_ref:
            raise S27ExecutionBlocked(f"S27_ENVIRONMENT_REF_MISSING:{cell_id}")
        registry_ref = raw.get("registry_ref", raw.get("parameter_registry_ref"))
        if not isinstance(registry_ref, str) or not registry_ref:
            raise S27ExecutionBlocked(f"S27_REGISTRY_REF_MISSING:{cell_id}")
        config_path = _safe_ref(root, config_ref, field=f"config.{cell_id}")
        config_value = load_canonical_json(config_path)
        if not isinstance(config_value, Mapping):
            raise S27ExecutionBlocked(f"S27_CONFIG_OBJECT_REQUIRED:{cell_id}")
        artifacts = config_value.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
            raise S27ExecutionBlocked(f"S27_CONFIG_OUTPUT_DIR_MISSING:{cell_id}")
        # The six-cell manifest is the authority for the selected checkpoint
        # root; the index itself must not be allowed to point to an unbound root.
        result[cell_id] = S27MaterializedCellInput(
            cell_id=cell_id,
            config_ref=config_ref,
            environment_ref=environment_ref,
            checkpoint_root_ref=str(checkpoint["checkpoint_root_ref"]),
            registry_ref=registry_ref,
            reference_output_root_ref=str(artifacts["output_dir"]),
        )
    if tuple(result) != EXPECTED_CELL_IDS:
        raise S27ExecutionBlocked("S27_MATERIALIZATION_CELL_ORDER_INVALID")
    return result


def _load_checkpoint_root(
    root: Path,
    cell: S27CellPlan,
    materialized: S27MaterializedCellInput,
) -> Path:
    manifest_path = _safe_ref(root, cell.checkpoint_ref, field=f"checkpoint.{cell.cell_id}.manifest")
    manifest_value = load_canonical_json(manifest_path) if manifest_path.is_file() else None
    if not isinstance(manifest_value, Mapping) or canonical_json_hash(dict(manifest_value)) != cell.checkpoint_hash:
        raise S27ExecutionBlocked(f"S27_CHECKPOINT_MANIFEST_HASH_MISMATCH:{cell.cell_id}")
    checkpoint_root = _safe_ref(root, materialized.checkpoint_root_ref, field=f"checkpoint.{cell.cell_id}.root")
    if not checkpoint_root.is_dir():
        raise S27ExecutionBlocked(f"S27_CHECKPOINT_ROOT_MISSING:{cell.cell_id}")
    return checkpoint_root


def build_s27_torch_provider(
    cell: S27CellPlan,
    *,
    data_root: Path,
    checkpoint_root_ref: str,
    registry_ref: str,
    request: TaskExecutionRequest,
) -> S27ProviderContext:
    """Build the real selected-checkpoint Torch provider.

    ``FormalG3RuntimeAssets`` still authorizes the immutable data/tokenizer
    route and the G3 lineage.  The model itself is loaded from the hash-bound
    S2.3 checkpoint root, never from the step-0 alias used by the shared legacy
    adapter.  This function is intentionally lazy about optional HF imports so
    control-plane tests remain CPU-only.
    """

    if request.config.run_intent != "formal" or request.task.stage != 2:
        raise S27ExecutionBlocked("S27_FORMAL_REQUEST_REQUIRED")
    from .stage23_task_runners import (
        _formal_execution_evidence,
        _load_formal_parameter_registry,
        _load_formal_selected_checkpoint,
    )
    try:
        evidence, _ = _formal_execution_evidence(request, data_root)
        providers = request.config.section("providers")
        base = request.config.base_config
        model_config = base.section("model")
        data = base.section("data")
        runtime = base.section("runtime")
        identity = base.section("identity")
        if not all(isinstance(value, Mapping) for value in (providers, model_config, data, runtime, identity)):
            raise ValueError("S27_REQUEST_SECTIONS_INVALID")
        if providers.get("kind") != "offline_hf":
            raise S27ExecutionBlocked("S27_OFFLINE_HF_PROVIDER_REQUIRED")
        task_type = str(providers["task_type"])
        runtime_assets = FormalG3RuntimeAssets.from_request(request, data_root)
        architecture = model_config.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("S27_MODEL_ARCHITECTURE_REQUIRED")
        model_asset = runtime_assets.resolve(f"{architecture}-step0", expected_kind="model")
        expected_data_kind = "pile" if task_type == "causal_lm" else "glue_derived"
        data_asset = runtime_assets.resolve(str(data["asset_id"]), expected_kind=expected_data_kind)
        tokenizer_asset = runtime_assets.resolve(str(model_config["tokenizer_asset_id"]), expected_kind="tokenizer")
        lineage = runtime_assets.runtime_lineage_sha256(model_asset, data_asset, tokenizer_asset)
        if not set(item.ready_manifest_sha256 for item in (model_asset, data_asset, tokenizer_asset)).issubset(evidence.asset_manifest_hashes):
            raise S27ExecutionBlocked("S27_G3_ASSET_MANIFEST_LINEAGE_MISMATCH")
        # Reuse the shared S2.4 strict selected-checkpoint binding.  It
        # verifies the source manifest bytes, exact file inventory and every
        # model.safetensors digest before Transformers sees the directory.
        selected_checkpoint = _load_formal_selected_checkpoint(request, data_root)
        if selected_checkpoint is None:
            raise S27ExecutionBlocked("S27_FORMAL_SELECTED_CHECKPOINT_REQUIRED")
        if (
            selected_checkpoint.checkpoint_id != cell.checkpoint_id
            or selected_checkpoint.manifest_sha256 != cell.checkpoint_hash
            or selected_checkpoint.root_ref != checkpoint_root_ref
        ):
            raise S27ExecutionBlocked("S27_SELECTED_CHECKPOINT_LINEAGE_MISMATCH")
        checkpoint_root = selected_checkpoint.root
        model = OfflineHuggingFaceModelAdapter.from_local_directory(
            checkpoint_root,
            task_type=task_type,
            num_labels=providers.get("num_labels"),
            torch_dtype=__import__("torch").float32,
        )
        torch = __import__("torch")
        model.module.to(torch.device(str(runtime["device"])))
        model.module.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.module.parameters()):
            raise ValueError("S27_MODEL_DTYPE_NOT_FLOAT32")
        if task_type == "causal_lm":
            if data_asset.storage_kind != "pythia_mmap_shards":
                raise G3RuntimeAssetError("S27_PILE_STORAGE_KIND_INVALID")
            route = formal_pile_route(
                stage=int(identity["stage"]),
                evaluation=False,
                declared_sampling_design=str(data["sampling_design"]),
                configured_split=str(data["split"]),
            )
            start, stop = runtime_assets.pile_split_interval(data_asset, route.split)
            runtime_assets.validate_pile_budget(stage=int(identity["stage"]), split=route.split, requested_records=stop - start)
            resolver = PythiaMMapFrozenSampleResolver(
                runtime_assets.pythia_dataset(data_asset, split=route.split),
                asset_id=data_asset.resolved.asset_id,
                ready_manifest_sha256=data_asset.ready_manifest_sha256,
                qualification_sha256=data_asset.qualification_artifact_hash,
                g3_resolution_artifact_hash=runtime_assets.resolution_artifact_hash,
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=lineage,
                split_start=start,
                split_stop=stop,
                sampling_design=str(data["sampling_design"]),
                weights_exogenous=bool(data["weights_exogenous"]),
                common_mean_assumption=bool(data["common_mean_assumption"]),
            )
        elif task_type == "sequence_classification":
            if data_asset.storage_kind != "hf_load_from_disk":
                raise G3RuntimeAssetError("S27_GLUE_STORAGE_KIND_INVALID")
            glue_task = data_asset.require_glue_route(task_name=str(providers["task_name"]), split=str(data["split"]))
            resolver = PretokenizedGlueDatasetAdapter(
                data_asset.resolved.root,
                task_name=glue_task,
                split=str(data["split"]),
                dataset_id=data_asset.resolved.asset_id,
                microbatch_size=1,
                microbatches_per_step=1,
                expected_asset_hash=data_asset.directory_content_sha256,
                allowed_root=data_asset.resolved.root,
                g3_resolution_artifact_hash=runtime_assets.resolution_artifact_hash,
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=lineage,
            )
        else:
            raise ValueError("S27_TASK_TYPE_UNSUPPORTED")
        registry = _load_formal_parameter_registry(data_root, registry_ref, model.module)
        if registry.get("registry_hash") != selected_checkpoint.registry_hash:
            raise S27ExecutionBlocked("S27_SELECTED_CHECKPOINT_REGISTRY_MISMATCH")
        provider = TorchFixedStateGradientProvider(
            model,
            resolver,
            fixed_state_id=f"s27-{cell.checkpoint_id}-{cell.checkpoint_hash[:16]}",
            registry=registry,
            output_dtype=torch.float32,
            gradient_chunk_size=1,
            enable_formal_batched=True,
            formal_batch_chunk_size=4,
        )
        return S27ProviderContext(
            provider=provider,
            execution=evidence,
            checkpoint_root_ref=checkpoint_root_ref,
            checkpoint_hash=cell.checkpoint_hash,
            registry_ref=registry_ref,
            registry_hash=provider.registry_hash,
        )
    except S27ExecutionBlocked:
        raise
    except Exception as error:
        raise S27ExecutionBlocked(f"S27_TORCH_PROVIDER_BIND_FAILED:{cell.cell_id}:{type(error).__name__}:{error}") from error


def _unit_file_name(unit_id: str) -> str:
    return hashlib.sha256(unit_id.encode("utf-8")).hexdigest() + ".json"


def _read_json_if_exists(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise S27ExecutionBlocked(f"S27_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _write_once(path: Path, value: Mapping[str, object], *, field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_canonical_json(path)
        if existing != dict(value):
            raise S27ExecutionBlocked(f"{field}:OUTPUT_CONFLICT")
        return
    write_canonical_json(path, dict(value))


def _state_from_committed_wave(wave_root: Path, unit_id: str) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    commit_path = wave_root / "commits" / f"{unit_id}.json"
    if not commit_path.exists():
        return None
    commit = load_canonical_json(commit_path)
    if not isinstance(commit, Mapping):
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_INVALID:{unit_id}")
    if commit.get("artifact_hash") != canonical_json_hash({key: item for key, item in commit.items() if key != "artifact_hash"}):
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_HASH_INVALID:{unit_id}")
    relative = commit.get("object_ref")
    if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise S27ExecutionBlocked(f"S27_WAVE_OBJECT_REF_INVALID:{unit_id}")
    state, bundle = load_tensor_bundle((wave_root / relative).resolve())
    if not isinstance(state, Mapping):
        raise S27ExecutionBlocked(f"S27_WAVE_STATE_INVALID:{unit_id}")
    if bundle.manifest_sha256 != commit.get("object_manifest_hash") or commit.get("unit_id") != unit_id:
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_BINDING_INVALID:{unit_id}")
    return state, commit


def _cost_from_state(state: Mapping[str, object]) -> dict[str, object]:
    wall = state.get("wall_seconds")
    formula = state.get("formula_seconds")
    gradients = state.get("gradient_evaluations")
    valid = isinstance(wall, (int, float)) and not isinstance(wall, bool) and float(wall) >= 0 and isinstance(formula, (int, float)) and not isinstance(formula, bool) and float(formula) >= 0 and isinstance(gradients, int) and not isinstance(gradients, bool) and gradients >= 0
    return {
        "valid": bool(valid),
        "wall_seconds": wall,
        "formula_seconds": formula,
        "gradient_seconds": state.get("gradient_seconds"),
        "gradient_evaluations": gradients,
        "peak_memory_bytes": state.get("peak_memory_bytes"),
        "shared_gradient_pool": True,
    }


class _GradientAuditProxy:
    """Capture the immutable base pool for an observed mean-gradient audit."""

    def __init__(self, delegate: FixedStateGradientProvider) -> None:
        self._delegate = delegate
        self._batches: list[tuple[tuple[object, ...], object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    @property
    def registry_hash(self) -> str:
        return self._delegate.registry_hash

    @property
    def fixed_state_id(self) -> str:
        return self._delegate.fixed_state_id

    @property
    def statistical_unit(self) -> str:
        return self._delegate.statistical_unit

    @property
    def weight_unit(self) -> str:
        return self._delegate.weight_unit

    @property
    def sampling_design(self) -> str:
        return self._delegate.sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._delegate.weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._delegate.common_mean_assumption

    def state_digest(self) -> str:
        return self._delegate.state_digest()

    def assert_unchanged(self, expected_digest: str) -> None:
        self._delegate.assert_unchanged(expected_digest)

    def begin(self) -> None:
        self._batches.clear()

    def gradient(self, draws: Sequence[object]) -> object:
        batch = self._delegate.gradient(draws)
        self._batches.append((tuple(draws), batch))
        return batch

    @staticmethod
    def _weighted_mean(batches: Sequence[object]) -> dict[str, np.ndarray]:
        if not batches:
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_POOL_EMPTY")
        vectors = [getattr(batch, "gradients") for batch in batches]
        weights = [float(getattr(batch, "statistical_weight")) for batch in batches]
        return _GradientAuditProxy._weighted_vector_mean(vectors, weights)

    @staticmethod
    def _as_fp64_numpy(value: object) -> np.ndarray:
        """Materialize a gradient value without asking NumPy to read CUDA memory.

        The real Torch provider returns detached tensors, and a CUDA tensor's
        ``__array__`` implementation intentionally rejects direct NumPy
        conversion.  The audit is a CPU-side FP64 reduction, so detach first,
        move to CPU, then materialize the NumPy array and cast explicitly.
        The duck-typed calls keep this control-plane module importable without
        making Torch a hard dependency for its contract tests.
        """

        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _weighted_vector_mean(
        vectors: Sequence[Mapping[str, object]], weights: Sequence[float]
    ) -> dict[str, np.ndarray]:
        if not vectors or len(vectors) != len(weights):
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_VECTOR_POOL_INVALID")
        names = tuple(vectors[0].keys())
        total_weight = float(sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_WEIGHT_INVALID")
        result: dict[str, np.ndarray] = {}
        for name in names:
            value = np.zeros_like(
                _GradientAuditProxy._as_fp64_numpy(vectors[0][name]), dtype=np.float64
            )
            for gradients, weight in zip(vectors, weights):
                if tuple(gradients.keys()) != names:
                    raise S27ExecutionBlocked("S27_MEAN_GRADIENT_PARAMETER_SET_DRIFT")
                value += _GradientAuditProxy._as_fp64_numpy(gradients[name]) * weight
            result[name] = value / total_weight
        return result

    def finish(self, mapping: RepetitionMapping, *, tolerance: float) -> dict[str, object]:
        expected_count = max(mapping.m_values)
        if len(self._batches) != expected_count:
            return {"passed": False, "max_abs_error": None, "observed_batch_count": len(self._batches), "expected_batch_count": expected_count}
        batches = [batch for _, batch in self._batches]
        full = self._weighted_mean(batches)
        # Re-form every frozen M from the exact live base pool.  This is an
        # observed regrouping audit (not a metadata assertion): the same
        # GradientBatch values that fed raw/double/U are grouped at each M,
        # reduced with their true weights, and compared with the full mean.
        max_m = len(batches)
        per_m: dict[str, dict[str, object]] = {}
        errors: list[float] = []
        for microbatch_count in tuple(mapping.m_values):
            if microbatch_count <= 0 or max_m % microbatch_count:
                per_m[str(microbatch_count)] = {
                    "passed": False,
                    "max_abs_error": None,
                    "reason": "base_pool_not_divisible_by_m",
                }
                continue
            merge_width = max_m // microbatch_count
            grouped_vectors: list[Mapping[str, object]] = []
            grouped_weights: list[float] = []
            for start in range(0, max_m, merge_width):
                group = batches[start : start + merge_width]
                grouped_vectors.append(self._weighted_mean(group))
                grouped_weights.append(
                    float(sum(float(getattr(batch, "statistical_weight")) for batch in group))
                )
            regrouped = self._weighted_vector_mean(grouped_vectors, grouped_weights)
            m_errors = [
                float(np.max(np.abs(full[name] - regrouped[name])))
                for name in full
            ]
            m_error = float(max(m_errors, default=0.0))
            finite_m = bool(np.isfinite(m_error))
            if finite_m:
                errors.append(m_error)
            per_m[str(microbatch_count)] = {
                "passed": bool(finite_m and m_error <= tolerance),
                "max_abs_error": m_error if finite_m else None,
                "merge_width": merge_width,
                "group_count": microbatch_count,
            }
        error = float(max(errors, default=float("nan")))
        finite = bool(np.isfinite(error))
        all_m_passed = bool(per_m) and all(item.get("passed") is True for item in per_m.values())
        return {
            "passed": bool(all_m_passed and finite and error <= tolerance),
            "max_abs_error": error if finite else None,
            "observed_batch_count": len(self._batches),
            "expected_batch_count": expected_count,
            "per_m": per_m,
            "identity": "weighted_full_mean_equals_regrouped_nested_m_means",
            "tolerance": tolerance,
        }


def _success_record(
    plan: S27Plan,
    cell: S27CellPlan,
    unit: S27MappingUnit,
    state: Mapping[str, object],
    commit: Mapping[str, object],
    *,
    artifact_root: Path,
    data_root: Path,
    attempt_id: str,
    mean_gradient_audit: Mapping[str, object],
) -> S27RawUnit:
    vectors = state.get("vectors")
    if not isinstance(vectors, Mapping):
        raise S27ExecutionBlocked(f"S27_RAW_VECTORS_INVALID:{unit.unit_id}")
    if state.get("input_hash") != unit.mapping_hash or commit.get("input_hash") != unit.mapping_hash:
        raise S27ExecutionBlocked(f"S27_RAW_MAPPING_BINDING_INVALID:{unit.unit_id}")
    expected_methods = {"raw", "double", "u_m2", f"u_m{unit.microbatch_count}"}
    if set(vectors) != expected_methods:
        raise S27ExecutionBlocked(f"S27_RAW_METHOD_SET_INVALID:{unit.unit_id}")
    for method, vector in vectors.items():
        if not isinstance(vector, Mapping) or any(not np.all(np.isfinite(np.asarray(value))) for value in vector.values()):
            raise S27ExecutionBlocked(f"S27_RAW_NONFINITE_VECTOR:{unit.unit_id}:{method}")
    # ``object_ref`` is relative to the paired-wave artifact root, not the
    # enclosing S2.7 run root.  Binding the manifest to that exact immutable
    # object keeps the sealed raw manifest recoverable and prevents a
    # misleading path that happens to hash correctly but cannot be reopened.
    raw_ref = _relative_ref(data_root, (artifact_root / str(commit["object_ref"])).resolve(), field=f"raw_artifact.{unit.unit_id}")
    raw_hash = commit.get("object_manifest_hash")
    if not isinstance(raw_hash, str) or not _SHA256.fullmatch(raw_hash):
        raise S27ExecutionBlocked(f"S27_RAW_OBJECT_HASH_INVALID:{unit.unit_id}")
    state_digest = state.get("state_digest")
    state_after = state.get("state_digest_after")
    if state_digest != state_after:
        raise S27ExecutionBlocked(f"S27_PROVIDER_STATE_DRIFT:{unit.unit_id}")
    assumptions = state.get("weighting_assumptions")
    mean_consistent = mean_gradient_audit.get("passed") is True
    if cell.corrected_delta_sci_binding is None:
        raise S27ExecutionBlocked(f"S27_CORRECTED_DELTA_BINDING_MISSING:{cell.cell_id}")
    metrics: dict[str, object] = {
        "finite": True,
        "raw_double_shared_gradient_pool": True,
        "nested_m_shared_gradient_pool": True,
        "mean_gradient_audit": dict(mean_gradient_audit),
        "inner_attempt_id": state.get("attempt_id"),
        "corrected_delta_sci_binding": dict(cell.corrected_delta_sci_binding),
    }
    return S27RawUnit(
        unit_id=unit.unit_id,
        cell_id=unit.cell_id,
        repetition_id=unit.repetition_id,
        status="SUCCESS",
        attempt_id=attempt_id,
        matrix_hash=plan.frozen_inputs.matrix_hash,
        mapping_hash=unit.mapping_hash,
        sampling_plan_hash=plan.frozen_inputs.sampling_plan_hash,
        checkpoint_hash=cell.checkpoint_hash,
        reference_hash=cell.reference_hash,
        batch_size=unit.batch_size,
        microbatch_count=unit.microbatch_count,
        draw_ids=unit.draw_ids,
        sample_ids=unit.sample_ids,
        raw_artifact_ref=raw_ref,
        raw_artifact_hash=raw_hash,
        metrics=metrics,
        methods=tuple(sorted(expected_methods)),
        m2_identity_max_abs=float(state.get("m2_double_max_abs_error")),
        mean_gradient_consistent=bool(mean_consistent),
        clamp_applied=False,
        clip_mode="none",
        cost=_cost_from_state(state),
    )


def _failure_record(
    plan: S27Plan,
    cell: S27CellPlan,
    unit: S27MappingUnit,
    *,
    attempt_id: str,
    run_root: Path,
    data_root: Path,
    code: str,
    reason: str,
) -> S27RawUnit:
    if not _SAFE_COMPONENT.fullmatch(code):
        code = "S27_WORKER_FAILURE"
    reason = " ".join(str(reason).split())
    if not reason:
        reason = "unspecified production unit failure"
    body: dict[str, object] = {
        "schema_version": S27_RAW_RESULT_SCHEMA,
        "run_id": run_root.name,
        "unit_id": unit.unit_id,
        "cell_id": unit.cell_id,
        "repetition_id": unit.repetition_id,
        "status": "FAILED",
        "attempt_id": attempt_id,
        "failure_code": code,
        "failure_reason": reason,
        "matrix_hash": plan.frozen_inputs.matrix_hash,
        "mapping_hash": unit.mapping_hash,
        "sampling_plan_hash": plan.frozen_inputs.sampling_plan_hash,
        "checkpoint_hash": cell.checkpoint_hash,
        "reference_hash": cell.reference_hash,
        "draw_ids": list(unit.draw_ids),
        "sample_ids": list(unit.sample_ids),
    }
    if cell.corrected_delta_sci_binding is not None:
        body["corrected_delta_sci_binding"] = dict(cell.corrected_delta_sci_binding)
    body["artifact_hash"] = canonical_json_hash(body)
    artifact_path = run_root / "raw-artifacts" / f"{_unit_file_name(unit.unit_id)[:-5]}-failure.json"
    _write_once(artifact_path, body, field=f"S27_FAILURE_ARTIFACT:{unit.unit_id}")
    return S27RawUnit(
        unit_id=unit.unit_id,
        cell_id=unit.cell_id,
        repetition_id=unit.repetition_id,
        status="FAILED",
        attempt_id=attempt_id,
        matrix_hash=plan.frozen_inputs.matrix_hash,
        mapping_hash=unit.mapping_hash,
        sampling_plan_hash=plan.frozen_inputs.sampling_plan_hash,
        checkpoint_hash=cell.checkpoint_hash,
        reference_hash=cell.reference_hash,
        batch_size=unit.batch_size,
        microbatch_count=unit.microbatch_count,
        draw_ids=unit.draw_ids,
        sample_ids=unit.sample_ids,
        raw_artifact_ref=_relative_ref(data_root, artifact_path, field=f"S27_FAILURE_REF:{unit.unit_id}"),
        raw_artifact_hash=str(body["artifact_hash"]),
        metrics={
            "finite": False,
            "failed": True,
            **(
                {"corrected_delta_sci_binding": dict(cell.corrected_delta_sci_binding)}
                if cell.corrected_delta_sci_binding is not None
                else {}
            ),
        },
        methods=("raw", "double", "u_m2", f"u_m{unit.microbatch_count}"),
        m2_identity_max_abs=None,
        mean_gradient_consistent=False,
        clamp_applied=False,
        clip_mode="none",
        cost={"valid": False, "reason": "unit_failed"},
        failure_code=code,
        failure_reason=reason,
    )


def _attempt_payload(unit: S27MappingUnit, *, attempt_id: str, mapping_hash: str, status: str, code: str | None = None, reason: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": S27_ATTEMPT_SCHEMA,
        "unit_id": unit.unit_id,
        "mapping_hash": mapping_hash,
        "attempt_id": attempt_id,
        "status": status,
        "started_at": _now(),
        "failure_code": code,
        "failure_reason": reason,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


class S27ProductionWorker:
    """Run one immutable repetition shard on one approved GPU."""

    def __init__(
        self,
        *,
        plan: S27Plan,
        run_id: str,
        cell_id: str,
        gpu_uuid: str,
        data_root: str | Path,
        run_root: str | Path,
        provider_factory: S27ProviderFactory | None = None,
        request: TaskExecutionRequest | None = None,
        materialized_input: S27MaterializedCellInput | None = None,
        retry_policy: S27RetryPolicy | None = None,
        m2_tolerance: float = S27_DEFAULT_M2_TOLERANCE,
        unit_ids: Sequence[str] | None = None,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ) -> None:
        self.plan = plan
        self.run_id = run_id
        self.cell = next((item for item in plan.cells if item.cell_id == cell_id), None)
        if self.cell is None:
            raise S27ExecutionBlocked(f"S27_UNKNOWN_CELL:{cell_id}")
        if gpu_uuid not in APPROVED_GPU_UUIDS:
            raise S27ExecutionBlocked(f"S27_GPU_ASSIGNMENT_INVALID:{cell_id}")
        self.gpu_uuid = gpu_uuid
        self.data_root = Path(data_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.provider_factory = provider_factory or build_s27_torch_provider
        self.request = request
        self.materialized_input = materialized_input
        self.retry_policy = retry_policy or S27RetryPolicy()
        if unit_ids is None:
            if shard_index is not None or shard_count is not None:
                raise S27ExecutionBlocked("S27_SHARD_METADATA_WITHOUT_UNIT_IDS")
            self.shard_unit_ids: tuple[str, ...] | None = None
            self.shard_index = None
            self.shard_count = None
        else:
            if shard_index is None or shard_count is None:
                raise S27ExecutionBlocked("S27_SHARD_METADATA_REQUIRED")
            self.shard_unit_ids = tuple(unit_ids)
            try:
                self.shard_index = int(shard_index)
                self.shard_count = int(shard_count)
            except (TypeError, ValueError) as error:
                raise S27ExecutionBlocked("S27_SHARD_METADATA_INVALID") from error
            if self.shard_count != len(APPROVED_GPU_UUIDS) or self.shard_index < 0 or self.shard_index >= self.shard_count:
                raise S27ExecutionBlocked("S27_SHARD_METADATA_INVALID")
            if not self.shard_unit_ids or len(set(self.shard_unit_ids)) != len(self.shard_unit_ids):
                raise S27ExecutionBlocked("S27_SHARD_UNIT_IDS_INVALID")
            expected_ids = {unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id}
            if not set(self.shard_unit_ids).issubset(expected_ids):
                raise S27ExecutionBlocked("S27_SHARD_UNIT_ID_OUTSIDE_CELL")
            if gpu_uuid != APPROVED_GPU_UUIDS[self.shard_index]:
                raise S27ExecutionBlocked("S27_SHARD_GPU_INDEX_BINDING_INVALID")
        if unit_ids is None and gpu_uuid != self.cell.assigned_gpu_uuid:
            raise S27ExecutionBlocked(f"S27_GPU_ASSIGNMENT_INVALID:{cell_id}")
        if not isinstance(m2_tolerance, (int, float)) or isinstance(m2_tolerance, bool) or float(m2_tolerance) < 0:
            raise ValueError("S27_M2_TOLERANCE_INVALID")
        self.m2_tolerance = float(m2_tolerance)
        self.wave_root = self.run_root / "waves" / self.cell.cell_id.replace(":", "__")
        self.raw_root = self.run_root / "raw-units"
        self.attempt_root = self.run_root / "attempts"
        self.provider_registry_hash: str | None = None

    def _unit_path(self, unit_id: str) -> Path:
        return self.raw_root / _unit_file_name(unit_id)

    def _attempt_path(self, unit_id: str) -> Path:
        return self.attempt_root / _unit_file_name(unit_id)

    def _terminal_attempt_path(self, unit_id: str) -> Path:
        return self.attempt_root / ("terminal-" + _unit_file_name(unit_id))

    def _validate_attempt_ledger(self, record: S27RawUnit) -> Mapping[str, object]:
        attempt = _read_json_if_exists(self._attempt_path(record.unit_id))
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
        ):
            raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_BINDING_INVALID:{record.unit_id}")
        return attempt

    def _terminal_payload(self, record: S27RawUnit) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": S27_ATTEMPT_SCHEMA,
            "unit_id": record.unit_id,
            "mapping_hash": record.mapping_hash,
            "attempt_id": record.attempt_id,
            "status": record.status,
            "record_artifact_hash": record.artifact_hash,
            "finished_at": _now(),
        }
        body["artifact_hash"] = canonical_json_hash(body)
        return body

    def _publish_terminal(self, record: S27RawUnit) -> None:
        _write_once(
            self._terminal_attempt_path(record.unit_id),
            self._terminal_payload(record),
            field=f"S27_TERMINAL_ATTEMPT:{record.unit_id}",
        )

    def _validate_terminal_attempt(self, record: S27RawUnit) -> None:
        self._validate_attempt_ledger(record)
        terminal = _read_json_if_exists(self._terminal_attempt_path(record.unit_id))
        if (
            terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_TERMINAL_ATTEMPT_BINDING_MISSING:{record.unit_id}")

    def _load_existing_records(
        self,
        reducer: StrictG25Reducer,
        mappings: Mapping[str, RepetitionMapping],
        *,
        target_unit_ids: set[str] | None = None,
    ) -> set[str]:
        done: set[str] = set()
        for unit in self.plan.frozen_inputs.units:
            if unit.cell_id != self.cell.cell_id:
                continue
            if target_unit_ids is not None and unit.unit_id not in target_unit_ids:
                continue
            path = self._unit_path(unit.unit_id)
            raw = _read_json_if_exists(path)
            if raw is not None:
                record = S27RawUnit.from_mapping(raw)
                self._validate_attempt_ledger(record)
                if _read_json_if_exists(self._terminal_attempt_path(record.unit_id)) is None:
                    # A crash after the append-only raw record but before its
                    # terminal marker is recoverable: the raw record is the
                    # completion marker, and the outer RUNNING ledger binds it
                    # to the sole attempt.  Publish the missing marker without
                    # changing the raw object.
                    self._publish_terminal(record)
                self._validate_terminal_attempt(record)
                reducer.add(record)
                done.add(unit.unit_id)
                continue
            attempt = _read_json_if_exists(self._attempt_path(unit.unit_id))
            if attempt is not None and attempt.get("status") == "RUNNING":
                if attempt.get("artifact_hash") != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"}) or attempt.get("mapping_hash") != unit.mapping_hash:
                    raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_HASH_INVALID:{unit.unit_id}")
                # A process that was detached and killed has crossed the only
                # retry boundary.  It is a failure in the fixed denominator.
                failed = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=str(attempt.get("attempt_id", "attempt-recovery")),
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_INTERRUPTED_ATTEMPT",
                    reason="previous worker left a non-terminal attempt; recovery is fail-closed",
                )
                self._publish_record(reducer, failed)
                done.add(unit.unit_id)
        return done

    def _publish_record(self, reducer: StrictG25Reducer, record: S27RawUnit) -> None:
        path = self._unit_path(record.unit_id)
        _write_once(path, record.to_dict(), field=f"S27_RAW_UNIT:{record.unit_id}")
        reducer.add(record)
        self._publish_terminal(record)

    def run(self) -> dict[str, object]:
        if self.gpu_uuid not in APPROVED_GPU_UUIDS:
            raise S27ExecutionBlocked("S27_APPROVED_GPU_REQUIRED")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != self.gpu_uuid:
            raise S27ExecutionBlocked("S27_CUDA_VISIBLE_DEVICES_ASSIGNMENT_DRIFT")
        if self.plan.frozen_inputs.mapping_ref is None:
            raise S27ExecutionBlocked("S27_FROZEN_MAPPING_REQUIRED")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        mappings = load_s27_frozen_mappings(self.data_root, self.plan, cell_id=self.cell.cell_id)
        all_cell_unit_ids = {
            unit.unit_id for unit in self.plan.frozen_inputs.units if unit.cell_id == self.cell.cell_id
        }
        target_unit_ids = set(self.shard_unit_ids) if self.shard_unit_ids is not None else all_cell_unit_ids
        if not target_unit_ids or not target_unit_ids.issubset(all_cell_unit_ids):
            raise S27ExecutionBlocked("S27_WORKER_TARGET_UNIT_SET_INVALID")
        reducer = StrictG25Reducer(self.plan, run_id=self.run_id)
        done = self._load_existing_records(reducer, mappings, target_unit_ids=target_unit_ids)
        if done == target_unit_ids:
            # Recovery after a process exit between the last terminal raw
            # record and the shard/wave seal must not reload a model or create
            # another attempt.  The immutable records are sufficient to seal.
            return self._seal_shard(reducer) if self.shard_unit_ids is not None else self._seal_wave(reducer)
        cell_inputs = self.materialized_input
        context: S27ProviderContext | None = None
        provider_error: Exception | None = None
        try:
            if cell_inputs is None:
                raise S27ExecutionBlocked("S27_MATERIALIZED_CELL_INPUT_REQUIRED")
            _load_checkpoint_root(self.data_root, self.cell, cell_inputs)
            context = self.provider_factory(
                self.cell,
                data_root=self.data_root,
                checkpoint_root_ref=cell_inputs.checkpoint_root_ref,
                registry_ref=cell_inputs.registry_ref,
                request=self.request,
            )
            if context.checkpoint_hash != self.cell.checkpoint_hash:
                raise S27ExecutionBlocked("S27_PROVIDER_CHECKPOINT_HASH_DRIFT")
            self.provider_registry_hash = context.registry_hash
        except Exception as error:
            provider_error = error
        if context is None:
            # Provider construction failure still consumes every expected unit
            # as an explicit failed denominator row; no cell is silently skipped.
            for unit in self.plan.frozen_inputs.units:
                if unit.cell_id != self.cell.cell_id or unit.unit_id not in target_unit_ids or unit.unit_id in done:
                    continue
                provider_attempt_id = f"attempt-provider-{time.time_ns()}"
                _write_once(
                    self._attempt_path(unit.unit_id),
                    _attempt_payload(unit, attempt_id=provider_attempt_id, mapping_hash=unit.mapping_hash, status="RUNNING"),
                    field=f"S27_ATTEMPT:{unit.unit_id}",
                )
                failed = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=provider_attempt_id,
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_PROVIDER_BIND_FAILED",
                    reason=f"{type(provider_error).__name__}:{provider_error}",
                )
                self._publish_record(reducer, failed)
            sealed = self._seal_shard(reducer) if self.shard_unit_ids is not None else self._seal_wave(reducer)
            return {"status": "FAILED", "cell_id": self.cell.cell_id, "failed_units": len(target_unit_ids), "failure_code": "S27_PROVIDER_BIND_FAILED", "seal": sealed}

        reference_views = load_s27_reference_views(
            self.data_root,
            self.cell,
            expected_registry_hash=context.provider.registry_hash,
            reference_output_root_ref=(cell_inputs.reference_output_root_ref or None),
        )
        reference = reference_views["bias"]
        audit_provider = _GradientAuditProxy(context.provider)
        wave_runner = RecoverablePairedWaveRunner(audit_provider, execution=context.execution, m2_tolerance=self.m2_tolerance)
        for unit in self.plan.frozen_inputs.units:
            if unit.cell_id != self.cell.cell_id or unit.unit_id not in target_unit_ids or unit.unit_id in done:
                continue
            mapping = mappings[unit.unit_id]
            attempt_path = self._attempt_path(unit.unit_id)
            existing_attempt = _read_json_if_exists(attempt_path)
            if existing_attempt is not None:
                if existing_attempt.get("mapping_hash") != unit.mapping_hash or existing_attempt.get("status") != "RUNNING":
                    raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_CONFLICT:{unit.unit_id}")
                # The only legal pre-existing state was handled by recovery.
                raise S27ExecutionBlocked(f"S27_NONTERMINAL_ATTEMPT_NOT_RECOVERED:{unit.unit_id}")
            attempt_id = f"attempt-{time.time_ns()}"
            _write_once(attempt_path, _attempt_payload(unit, attempt_id=attempt_id, mapping_hash=unit.mapping_hash, status="RUNNING"), field=f"S27_ATTEMPT:{unit.unit_id}")
            try:
                audit_provider.begin()
                unit_wave_root = self.wave_root / "paired" / _unit_file_name(unit.unit_id).removesuffix(".json")
                wave_runner.run(
                    wave_id=f"s27-{self.cell.cell_id.replace(':', '-')}",
                    mappings=(mapping,),
                    reference=reference,
                    reference_hash=_vector_digest(reference),
                    references=reference_views,
                    artifact_root=unit_wave_root,
                    max_new_units=1,
                )
                committed = _state_from_committed_wave(unit_wave_root, mapping.repetition_id)
                if committed is None:
                    raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_MISSING:{unit.unit_id}")
                state, commit = committed
                if state.get("registry_hash") != context.provider.registry_hash:
                    raise S27ExecutionBlocked(f"S27_WAVE_REGISTRY_HASH_DRIFT:{unit.unit_id}")
                audit = audit_provider.finish(mapping, tolerance=self.m2_tolerance)
                record = _success_record(
                    self.plan,
                    self.cell,
                    unit,
                    state,
                    commit,
                    artifact_root=unit_wave_root,
                    data_root=self.data_root,
                    attempt_id=attempt_id,
                    mean_gradient_audit=audit,
                )
            except Exception as error:
                record = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=attempt_id,
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_UNIT_EXECUTION_FAILED",
                    reason=f"{type(error).__name__}:{error}",
                )
            self._publish_record(reducer, record)
        return self._seal_shard(reducer) if self.shard_unit_ids is not None else self._seal_wave(reducer)

    def _seal_shard(self, reducer: StrictG25Reducer) -> dict[str, object]:
        if self.shard_unit_ids is None or self.shard_index is None or self.shard_count is None:
            raise S27ExecutionBlocked("S27_SHARD_SEAL_METADATA_MISSING")
        expected = tuple(self.shard_unit_ids)
        records = tuple(record for record in reducer.records if record.cell_id == self.cell.cell_id)
        if {record.unit_id for record in records} != set(expected) or len(records) != len(expected):
            raise S27ExecutionBlocked(f"S27_SHARD_EXPECTED_UNITS_MISSING:{self.cell.cell_id}:{self.shard_index}")
        records = tuple(sorted(records, key=lambda item: item.unit_id))
        descriptors = [
            {
                "unit_id": record.unit_id,
                "status": record.status,
                "attempt_id": record.attempt_id,
                "unit_artifact_hash": record.artifact_hash,
            }
            for record in records
        ]
        body: dict[str, object] = {
            "schema_version": S27_SHARD_SEAL_SCHEMA,
            "run_id": self.run_id,
            "plan_hash": self.plan.artifact_hash,
            "cell_id": self.cell.cell_id,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "gpu_uuid": self.gpu_uuid,
            "checkpoint_hash": self.cell.checkpoint_hash,
            "reference_hash": self.cell.reference_hash,
            "provider_registry_hash": self.provider_registry_hash,
            "expected_unit_ids": list(expected),
            "completed_unit_count": len(records),
            "failed_unit_count": sum(record.status == "FAILED" for record in records),
            "units": descriptors,
            "sealed": True,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        path = self.run_root / "wave-shards" / self.cell.cell_id.replace(":", "__") / f"shard-{self.shard_index:02d}.json"
        _write_once(path, body, field=f"S27_SHARD_SEAL:{self.cell.cell_id}:{self.shard_index}")
        return {
            "status": "SHARD_SEALED",
            "cell_id": self.cell.cell_id,
            "shard_index": self.shard_index,
            "gpu_uuid": self.gpu_uuid,
            "expected_units": len(expected),
            "failed_units": body["failed_unit_count"],
            "shard_seal_ref": _relative_ref(self.data_root, path, field=f"S27_SHARD_SEAL_REF:{self.cell.cell_id}:{self.shard_index}"),
        }

    def _seal_wave(self, reducer: StrictG25Reducer) -> dict[str, object]:
        records = tuple(record for record in reducer.records if record.cell_id == self.cell.cell_id)
        path = self.run_root / "wave-seals" / f"{self.cell.cell_id.replace(':', '__')}.json"
        existing = _read_json_if_exists(path)
        checked_at = existing.get("checked_at") if isinstance(existing, Mapping) and isinstance(existing.get("checked_at"), str) else None
        body = build_s27_wave_seal_payload(
            self.plan,
            run_id=self.run_id,
            cell_id=self.cell.cell_id,
            records=records,
            gpu_uuids=(self.gpu_uuid,),
            checked_at=checked_at,
        )
        _write_once(path, body, field=f"S27_WAVE_SEAL:{self.cell.cell_id}")
        return {"status": "SEALED", "cell_id": self.cell.cell_id, "wave_seal_ref": _relative_ref(self.data_root, path, field="S27_WAVE_SEAL_REF"), "expected_units": body["expected_unit_count"], "failed_units": body["failed_unit_count"]}


def load_s27_raw_records(data_root: str | Path, plan: S27Plan, run_root: str | Path, *, run_id: str) -> StrictG25Reducer:
    root = Path(data_root).resolve()
    reducer = StrictG25Reducer(plan, run_id=run_id)
    record_root = Path(run_root).resolve() / "raw-units"
    if not record_root.is_dir():
        raise S27ExecutionBlocked("S27_RAW_UNIT_DIRECTORY_MISSING")
    for path in sorted(record_root.glob("*.json")):
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_RAW_UNIT_NOT_OBJECT:{path.name}")
        record = S27RawUnit.from_mapping(value)
        artifact_path = _safe_ref(root, record.raw_artifact_ref, field=f"S27_RAW_ARTIFACT:{record.unit_id}")
        if record.status == "SUCCESS":
            if not artifact_path.is_dir() or artifact_path.is_symlink():
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_DIRECTORY_MISSING:{record.unit_id}")
            try:
                state, bundle = load_tensor_bundle(artifact_path)
            except (OSError, TypeError, ValueError) as error:
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_INVALID:{record.unit_id}") from error
            if (
                not isinstance(state, Mapping)
                or state.get("schema_version") != "stage2-wave-unit-state-v1"
                or state.get("unit_id") != record.repetition_id
                or not isinstance(state.get("vectors"), Mapping)
                or bundle.manifest_sha256 != record.raw_artifact_hash
            ):
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_HASH_INVALID:{record.unit_id}")
        else:
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise S27ExecutionBlocked(f"S27_RAW_FAILURE_ARTIFACT_MISSING:{record.unit_id}")
            failure_payload = load_canonical_json(artifact_path)
            if (
                not isinstance(failure_payload, Mapping)
                or failure_payload.get("artifact_hash") != record.raw_artifact_hash
                or canonical_json_hash({key: item for key, item in failure_payload.items() if key != "artifact_hash"}) != record.raw_artifact_hash
                or failure_payload.get("unit_id") != record.unit_id
            ):
                raise S27ExecutionBlocked(f"S27_RAW_FAILURE_ARTIFACT_HASH_INVALID:{record.unit_id}")
        attempt_path = record_root.parent / "attempts" / _unit_file_name(record.unit_id)
        terminal_path = record_root.parent / "attempts" / ("terminal-" + _unit_file_name(record.unit_id))
        attempt = _read_json_if_exists(attempt_path)
        terminal = _read_json_if_exists(terminal_path)
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
            or terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_BINDING_INVALID:{record.unit_id}")
        reducer.add(record)
    return reducer


def build_s27_wave_seal_payload(
    plan: S27Plan,
    *,
    run_id: str,
    cell_id: str,
    records: Sequence[S27RawUnit],
    gpu_uuids: Sequence[str] = (),
    checked_at: str | None = None,
) -> dict[str, object]:
    """Build the deterministic merged checkpoint-wave seal from raw units."""

    expected = tuple(unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id)
    observed = tuple(record.unit_id for record in records if record.cell_id == cell_id)
    if cell_id not in EXPECTED_CELL_IDS or set(observed) != set(expected) or len(observed) != len(expected):
        raise S27ExecutionBlocked(f"S27_WAVE_MERGE_COVERAGE_INVALID:{cell_id}")
    cards = tuple(gpu_uuids)
    if any(card not in APPROVED_GPU_UUIDS for card in cards) or len(set(cards)) != len(cards):
        raise S27ExecutionBlocked(f"S27_WAVE_MERGE_GPU_INVALID:{cell_id}")
    ordered = tuple(sorted((record for record in records if record.cell_id == cell_id), key=lambda item: item.unit_id))
    descriptors = [
        {
            "unit_id": record.unit_id,
            "status": record.status,
            "attempt_id": record.attempt_id,
            "unit_artifact_hash": record.artifact_hash,
        }
        for record in ordered
    ]
    body: dict[str, object] = {
        "schema_version": S27_WAVE_SEAL_SCHEMA,
        "run_id": run_id,
        "plan_hash": plan.artifact_hash,
        "cell_id": cell_id,
        "gpu_uuid": cards[0] if len(cards) == 1 else None,
        "gpu_uuids": list(cards),
        "expected_unit_count": len(expected),
        "completed_unit_count": len(ordered),
        "failed_unit_count": sum(record.status == "FAILED" for record in ordered),
        "units": descriptors,
        "sealed": True,
        "checked_at": checked_at or _now(),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def validate_s27_shard_seals(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    run_id: str,
    cell_id: str,
) -> tuple[S27UnitShard, ...]:
    """Reject overlap, missing shard seals, GPU drift, or raw descriptor drift."""

    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    shards = load_s27_shard_plan(root, plan, run, run_id=run_id, cell_id=cell_id)
    reducer = load_s27_raw_records(root, plan, run, run_id=run_id)
    by_unit = {record.unit_id: record for record in reducer.records if record.cell_id == cell_id}
    observed: set[str] = set()
    registry_hashes: set[str] = set()
    missing_registry_for_success = False
    for shard in shards:
        path = run / "wave-shards" / cell_id.replace(":", "__") / f"shard-{shard.shard_index:02d}.json"
        raw = load_canonical_json(path) if path.is_file() else None
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != S27_SHARD_SEAL_SCHEMA
            or raw.get("artifact_hash") != canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"})
            or raw.get("run_id") != run_id
            or raw.get("plan_hash") != plan.artifact_hash
            or raw.get("cell_id") != cell_id
            or raw.get("shard_index") != shard.shard_index
            or raw.get("shard_count") != shard.shard_count
            or raw.get("gpu_uuid") != shard.gpu_uuid
            or raw.get("checkpoint_hash") != next(cell.checkpoint_hash for cell in plan.cells if cell.cell_id == cell_id)
            or raw.get("reference_hash") != next(cell.reference_hash for cell in plan.cells if cell.cell_id == cell_id)
            or raw.get("expected_unit_ids") != list(shard.unit_ids)
            or raw.get("sealed") is not True
        ):
            raise S27ExecutionBlocked(f"S27_SHARD_SEAL_INVALID:{cell_id}:{shard.shard_index}")
        registry_hash = raw.get("provider_registry_hash")
        if registry_hash is not None:
            if not isinstance(registry_hash, str) or not _SHA256.fullmatch(registry_hash):
                raise S27ExecutionBlocked(f"S27_SHARD_REGISTRY_HASH_INVALID:{cell_id}:{shard.shard_index}")
            registry_hashes.add(registry_hash)
        descriptors = raw.get("units")
        if not isinstance(descriptors, list) or len(descriptors) != len(shard.unit_ids) or raw.get("completed_unit_count") != len(descriptors):
            raise S27ExecutionBlocked(f"S27_SHARD_SEAL_UNIT_COUNT_INVALID:{cell_id}:{shard.shard_index}")
        descriptor_by_unit: dict[str, Mapping[str, object]] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("unit_id"), str) or descriptor["unit_id"] in descriptor_by_unit:
                raise S27ExecutionBlocked(f"S27_SHARD_SEAL_DESCRIPTOR_INVALID:{cell_id}:{shard.shard_index}")
            descriptor_by_unit[str(descriptor["unit_id"])] = descriptor
        if set(descriptor_by_unit) != set(shard.unit_ids) or observed.intersection(descriptor_by_unit):
            raise S27ExecutionBlocked(f"S27_SHARD_SEAL_OVERLAP_OR_COVERAGE_INVALID:{cell_id}:{shard.shard_index}")
        for unit_id in shard.unit_ids:
            record = by_unit.get(unit_id)
            descriptor = descriptor_by_unit.get(unit_id)
            if record is None or descriptor is None or descriptor.get("status") != record.status or descriptor.get("attempt_id") != record.attempt_id or descriptor.get("unit_artifact_hash") != record.artifact_hash:
                raise S27ExecutionBlocked(f"S27_SHARD_SEAL_RAW_BINDING_INVALID:{cell_id}:{shard.shard_index}:{unit_id}")
        if raw.get("failed_unit_count") != sum(by_unit[unit_id].status == "FAILED" for unit_id in shard.unit_ids):
            raise S27ExecutionBlocked(f"S27_SHARD_SEAL_FAILURE_COUNT_INVALID:{cell_id}:{shard.shard_index}")
        if registry_hash is None and any(by_unit[unit_id].status == "SUCCESS" for unit_id in shard.unit_ids):
            missing_registry_for_success = True
        observed.update(descriptor_by_unit)
    expected = {unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id}
    if observed != expected or set(by_unit) != expected:
        raise S27ExecutionBlocked(f"S27_SHARD_SEAL_GLOBAL_COVERAGE_INVALID:{cell_id}")
    if len(registry_hashes) > 1 or missing_registry_for_success:
        raise S27ExecutionBlocked(f"S27_SHARD_PROVIDER_IDENTITY_DRIFT:{cell_id}")
    return shards


def seal_s27_wave(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    run_id: str,
    cell_id: str,
    checked_at: str | None = None,
) -> dict[str, object]:
    """Atomically merge all four shard seals into one checkpoint-wave seal."""

    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    shards = validate_s27_shard_seals(root, plan, run, run_id=run_id, cell_id=cell_id)
    reducer = load_s27_raw_records(root, plan, run, run_id=run_id)
    path = run / "wave-seals" / f"{cell_id.replace(':', '__')}.json"
    existing = _read_json_if_exists(path)
    stable_checked_at = existing.get("checked_at") if isinstance(existing, Mapping) and isinstance(existing.get("checked_at"), str) else checked_at
    body = build_s27_wave_seal_payload(
        plan,
        run_id=run_id,
        cell_id=cell_id,
        records=reducer.records,
        gpu_uuids=tuple(shard.gpu_uuid for shard in shards),
        checked_at=stable_checked_at,
    )
    _write_once(path, body, field=f"S27_WAVE_SEAL:{cell_id}")
    return {
        "status": "SEALED",
        "cell_id": cell_id,
        "wave_seal_ref": _relative_ref(root, path, field=f"S27_WAVE_SEAL_REF:{cell_id}"),
        "expected_units": body["expected_unit_count"],
        "failed_units": body["failed_unit_count"],
        "shard_count": len(shards),
    }


def seal_s27_run(data_root: str | Path, plan: S27Plan, run_root: str | Path, *, run_id: str) -> dict[str, object]:
    """Seal only the complete denominator; no statistics are computed here."""

    reducer = load_s27_raw_records(data_root, plan, run_root, run_id=run_id)
    for cell in plan.cells:
        seal = Path(run_root).resolve() / "wave-seals" / f"{cell.cell_id.replace(':', '__')}.json"
        if not seal.is_file():
            raise S27ExecutionBlocked(f"S27_WAVE_SEAL_MISSING:{cell.cell_id}")
        payload = load_canonical_json(seal)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != S27_WAVE_SEAL_SCHEMA
            or payload.get("artifact_hash") != canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"})
            or payload.get("run_id") != run_id
            or payload.get("plan_hash") != plan.artifact_hash
            or payload.get("cell_id") != cell.cell_id
            or payload.get("sealed") is not True
        ):
            raise S27ExecutionBlocked(f"S27_WAVE_SEAL_INVALID:{cell.cell_id}")
        expected = {unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell.cell_id}
        records = {record.unit_id: record for record in reducer.records if record.cell_id == cell.cell_id}
        descriptors = payload.get("units")
        if not isinstance(descriptors, list) or len(descriptors) != len(expected):
            raise S27ExecutionBlocked(f"S27_WAVE_SEAL_UNIT_COUNT_INVALID:{cell.cell_id}")
        seen_descriptors: set[str] = set()
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise S27ExecutionBlocked(f"S27_WAVE_SEAL_DESCRIPTOR_INVALID:{cell.cell_id}")
            unit_id = descriptor.get("unit_id")
            if not isinstance(unit_id, str) or unit_id in seen_descriptors:
                raise S27ExecutionBlocked(f"S27_WAVE_SEAL_DESCRIPTOR_INVALID:{cell.cell_id}")
            seen_descriptors.add(unit_id)
            record = records.get(unit_id)
            if record is None or descriptor.get("status") != record.status or descriptor.get("attempt_id") != record.attempt_id or descriptor.get("unit_artifact_hash") != record.artifact_hash:
                raise S27ExecutionBlocked(f"S27_WAVE_SEAL_RAW_BINDING_INVALID:{cell.cell_id}")
        if seen_descriptors != expected or set(records) != expected:
            raise S27ExecutionBlocked(f"S27_WAVE_SEAL_COVERAGE_INVALID:{cell.cell_id}")
    return reducer.seal(Path(run_root).resolve() / "sealed")


def validate_s27_quality_wave(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    cell_id: str,
) -> dict[str, object]:
    """Validate the first 14M quality wave before any later child starts.

    This is an operational conjunction only.  It checks every terminal raw
    unit, finite/m2/observed-mean-gradient/cost evidence and raw artifact
    persistence.  It never reads or compares estimator means, ranking, bias,
    or any superiority metric.
    """

    if cell_id != EXPECTED_CELL_IDS[0]:
        raise S27ExecutionBlocked("S27_QUALITY_WAVE_MUST_BE_FIRST_14M_INITIALIZATION")
    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    expected = [unit for unit in plan.frozen_inputs.units if unit.cell_id == cell_id]
    found: list[S27RawUnit] = []
    for unit in expected:
        path = run / "raw-units" / _unit_file_name(unit.unit_id)
        if not path.is_file():
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_UNIT_MISSING:{unit.unit_id}")
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_UNIT_INVALID:{unit.unit_id}")
        record = S27RawUnit.from_mapping(value)
        if record.status != "SUCCESS":
            raise S27ExecutionBlocked(f"S27_QUALITY_UNIT_FAILED:{unit.unit_id}")
        if record.mean_gradient_consistent is not True or record.m2_identity_max_abs is None or record.m2_identity_max_abs > 1e-12 or record.clamp_applied or record.clip_mode != "none" or record.cost.get("valid") is not True:
            raise S27ExecutionBlocked(f"S27_QUALITY_INTEGRITY_FAILED:{unit.unit_id}")
        cell = next(item for item in plan.cells if item.cell_id == cell_id)
        if cell.corrected_delta_sci_binding is not None and record.metrics.get("corrected_delta_sci_binding") != dict(cell.corrected_delta_sci_binding):
            raise S27ExecutionBlocked(f"S27_QUALITY_CORRECTED_DELTA_BINDING_FAILED:{unit.unit_id}")
        audit = record.metrics.get("mean_gradient_audit")
        if not isinstance(audit, Mapping) or audit.get("passed") is not True:
            raise S27ExecutionBlocked(f"S27_QUALITY_MEAN_GRADIENT_AUDIT_FAILED:{unit.unit_id}")
        per_m = audit.get("per_m")
        expected_m = {str(value) for value in (2, plan.frozen_inputs.microbatch_count)}
        if (
            not isinstance(per_m, Mapping)
            or set(per_m) != expected_m
            or any(not isinstance(per_m[value], Mapping) or per_m[value].get("passed") is not True for value in expected_m)
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_MEAN_GRADIENT_PER_M_AUDIT_FAILED:{unit.unit_id}")
        attempt = _read_json_if_exists(run / "attempts" / _unit_file_name(unit.unit_id))
        terminal = _read_json_if_exists(run / "attempts" / ("terminal-" + _unit_file_name(unit.unit_id)))
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
            or terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_ATTEMPT_LEDGER_INVALID:{unit.unit_id}")
        artifact = _safe_ref(root, record.raw_artifact_ref, field=f"S27_QUALITY_RAW_ARTIFACT:{unit.unit_id}")
        if not artifact.is_dir() or artifact.is_symlink():
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_MISSING:{unit.unit_id}")
        try:
            state, bundle = load_tensor_bundle(artifact)
        except (OSError, TypeError, ValueError) as error:
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_INVALID:{unit.unit_id}") from error
        if (
            not isinstance(state, Mapping)
            or state.get("schema_version") != "stage2-wave-unit-state-v1"
            or state.get("unit_id") != record.repetition_id
            or bundle.manifest_sha256 != record.raw_artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_HASH_INVALID:{unit.unit_id}")
        found.append(record)
    return {
        "status": "QUALITY_PASS",
        "cell_id": cell_id,
        "expected_unit_count": len(expected),
        "completed_unit_count": len(found),
        "failed_unit_count": 0,
        "finite_outputs": True,
        "m2_identity": True,
        "mean_gradient_audit": True,
        "cost_profiler": True,
        "raw_artifacts_persisted": True,
        "statistical_conclusions": False,
    }


def normalized_gpu_inventory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate the four approved UUIDs and excluded PCI before launching."""

    normalized = []
    for row in rows:
        uuid = row.get("uuid")
        pci = row.get("pci_bus_id", row.get("pci"))
        if isinstance(uuid, str) and isinstance(pci, str):
            normalized.append({"uuid": uuid, "pci_bus_id": pci, **{str(k): v for k, v in row.items()}})
    return validate_gpu_inventory(normalized)


def nvidia_smi_inventory() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S27ExecutionBlocked(f"S27_GPU_INVENTORY_UNAVAILABLE:{type(error).__name__}") from error
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        rows.append({"uuid": parts[0], "pci_bus_id": parts[1], "memory_used_mib": parts[2] if len(parts) > 2 else None, "utilization_gpu": parts[3] if len(parts) > 3 else None})
    if not rows:
        raise S27ExecutionBlocked("S27_GPU_INVENTORY_EMPTY")
    return rows


@dataclass(frozen=True, slots=True)
class S27SubprocessSpec:
    cell_id: str
    gpu_uuid: str
    command: tuple[str, ...]
    log_path: Path


def build_s27_worker_command(
    *,
    python: str | Path,
    launcher_script: str | Path,
    data_root: str | Path,
    plan_ref: str,
    run_root: str,
    run_id: str,
    cell_id: str,
    gpu_uuid: str,
    materialization_index_ref: str,
    execution_evidence_ref: str,
    gpu_inventory_json: str | Path | None = None,
    shard_plan_ref: str | None = None,
    shard_index: int | None = None,
) -> tuple[str, ...]:
    if gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S27ExecutionBlocked("S27_WORKER_COMMAND_UNAPPROVED_GPU")
    command = [
        str(python),
        str(launcher_script),
        "--worker",
        "--data-root",
        str(data_root),
        "--plan-ref",
        plan_ref,
        "--run-root",
        run_root,
        "--run-id",
        run_id,
        "--cell-id",
        cell_id,
        "--gpu-uuid",
        gpu_uuid,
        "--materialization-index-ref",
        materialization_index_ref,
        "--execution-evidence-ref",
        execution_evidence_ref,
    ]
    if gpu_inventory_json is not None:
        command.extend(("--gpu-inventory-json", str(gpu_inventory_json)))
    if (shard_plan_ref is None) != (shard_index is None):
        raise S27ExecutionBlocked("S27_WORKER_COMMAND_SHARD_METADATA_INCOMPLETE")
    if shard_plan_ref is not None and shard_index is not None:
        if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0 or shard_index >= len(APPROVED_GPU_UUIDS):
            raise S27ExecutionBlocked("S27_WORKER_COMMAND_SHARD_INDEX_INVALID")
        command.extend(("--shard-plan-ref", shard_plan_ref, "--shard-index", str(shard_index)))
    return tuple(command)


class S27DetachedLauncher:
    """Four-slot within-wave queue for the six frozen checkpoint waves."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        plan_ref: str,
        run_root: str | Path,
        run_id: str,
        python: str | Path,
        launcher_script: str | Path,
        materialization_index_ref: str,
        execution_evidence_ref: str,
        approved_inventory: Sequence[Mapping[str, object]],
        gpu_inventory_json: str | Path | None = None,
        gpu_inventory_identity: Mapping[str, object] | None = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.plan = load_s27_plan(self.data_root, plan_ref)
        if any(cell.corrected_delta_sci_binding is None for cell in self.plan.cells):
            raise S27ExecutionBlocked("S27_CORRECTED_DELTA_BINDINGS_REQUIRED")
        self.plan_ref = plan_ref
        self.run_root = Path(run_root).resolve()
        self.run_id = run_id
        self.python = str(python)
        self.launcher_script = str(launcher_script)
        self.materialization_index_ref = materialization_index_ref
        self.execution_evidence_ref = execution_evidence_ref
        validate_s27_gpu_inventory(approved_inventory, compute_apps=())
        if gpu_inventory_json is None or gpu_inventory_identity is None:
            raise S27ExecutionBlocked("S27_GPU_INVENTORY_IDENTITY_REQUIRED")
        required_identity = {"source_ref", "artifact_hash", "source_sha256", "schema_version"}
        if set(gpu_inventory_identity) != required_identity:
            raise S27ExecutionBlocked("S27_GPU_INVENTORY_IDENTITY_FIELDS_INVALID")
        for field in required_identity:
            if not isinstance(gpu_inventory_identity.get(field), str) or not gpu_inventory_identity[field]:
                raise S27ExecutionBlocked(f"S27_GPU_INVENTORY_IDENTITY_{field.upper()}_INVALID")
        self.gpu_inventory_json = Path(gpu_inventory_json).resolve()
        self.gpu_inventory_identity = dict(gpu_inventory_identity)
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.run_root.is_relative_to(self.data_root) is False:
            raise S27ExecutionBlocked("S27_RUN_ROOT_OUTSIDE_DATA_ROOT")

    def _status_path(self) -> Path:
        return self.run_root / "launcher-status.json"

    def _publish_status(self, status: str, *, waves: Mapping[str, Mapping[str, object]], reason: str | None = None) -> None:
        body: dict[str, object] = {
            "schema_version": S27_LAUNCH_STATUS_SCHEMA,
            "run_id": self.run_id,
            "plan_ref": self.plan_ref,
            "plan_hash": self.plan.artifact_hash,
            "gpu_inventory_identity": dict(self.gpu_inventory_identity),
            "status": status,
            "wave_order": list(EXPECTED_CELL_IDS),
            "waves": {key: dict(value) for key, value in sorted(waves.items())},
            "updated_at": _now(),
            "reason": reason,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        _write_once(self._status_path(), body, field="S27_LAUNCH_STATUS") if status in {"PREPARED", "SEALED"} else write_canonical_json(self._status_path(), body)

    def _run_sharded_wave(self, cell: S27CellPlan) -> dict[str, object]:
        shards = write_s27_shard_plan(
            self.data_root,
            self.plan,
            self.run_root,
            run_id=self.run_id,
            cell_id=cell.cell_id,
        )
        shard_plan_path = _shard_plan_path(self.run_root, cell.cell_id)
        shard_plan_ref = _relative_ref(self.data_root, shard_plan_path, field=f"S27_SHARD_PLAN_REF:{cell.cell_id}")
        run_root_ref = _relative_ref(self.data_root, self.run_root, field="S27_RUN_ROOT_REF")

        def launch(shard: S27UnitShard) -> dict[str, object]:
            spec = build_s27_worker_command(
                python=self.python,
                launcher_script=self.launcher_script,
                data_root=self.data_root,
                plan_ref=self.plan_ref,
                run_root=run_root_ref,
                run_id=self.run_id,
                cell_id=cell.cell_id,
                gpu_uuid=shard.gpu_uuid,
                materialization_index_ref=self.materialization_index_ref,
                execution_evidence_ref=self.execution_evidence_ref,
                gpu_inventory_json=self.gpu_inventory_json,
                shard_plan_ref=shard_plan_ref,
                shard_index=shard.shard_index,
            )
            log_path = self.run_root / "logs" / f"{cell.cell_id.replace(':', '__')}__shard-{shard.shard_index:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": shard.gpu_uuid, "NVIDIA_VISIBLE_DEVICES": shard.gpu_uuid})
            try:
                with log_path.open("ab") as handle:
                    proc = subprocess.Popen(
                        spec,
                        cwd=str(Path(self.launcher_script).resolve().parents[2]),
                        env=dict(env),
                        stdin=subprocess.DEVNULL,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = int(proc.wait())
            except Exception as error:
                return {
                    "shard_index": shard.shard_index,
                    "gpu_uuid": shard.gpu_uuid,
                    "returncode": None,
                    "status": "FAILED",
                    "error": f"{type(error).__name__}:{error}",
                }
            return {
                "shard_index": shard.shard_index,
                "gpu_uuid": shard.gpu_uuid,
                "returncode": code,
                "status": "COMPLETE" if code == 0 else "FAILED",
                "log_ref": _relative_ref(self.data_root, log_path, field=f"S27_LOG_REF:{cell.cell_id}:{shard.shard_index}"),
            }

        with ThreadPoolExecutor(max_workers=len(APPROVED_GPU_UUIDS)) as pool:
            futures = [pool.submit(launch, shard) for shard in shards]
            outcomes = [future.result() for future in futures]
        outcomes = sorted(outcomes, key=lambda item: int(item["shard_index"]))
        if any(item.get("returncode") != 0 for item in outcomes):
            raise S27ExecutionBlocked(f"S27_SHARD_WORKER_FAILED:{cell.cell_id}")
        merged = seal_s27_wave(
            self.data_root,
            self.plan,
            self.run_root,
            run_id=self.run_id,
            cell_id=cell.cell_id,
        )
        return {
            "cell_id": cell.cell_id,
            "status": "COMPLETE",
            "shard_plan_ref": shard_plan_ref,
            "shard_count": len(shards),
            "shards": outcomes,
            "wave_seal": merged,
        }

    def execute(self) -> dict[str, object]:
        waves: dict[str, Mapping[str, object]] = {}
        if self._status_path().exists():
            raise S27ExecutionBlocked("S27_LAUNCH_STATUS_EXISTS_REQUIRES_EXPLICIT_RECOVERY_AUDIT")
        self._publish_status("PREPARED", waves=waves)
        quality_passed = False
        # Checkpoint barriers remain serial and canonical; only independent
        # repetitions inside the current checkpoint wave use four GPUs.
        for index, cell in enumerate(self.plan.cells):
            self._publish_status("RUNNING", waves=waves)
            try:
                result = self._run_sharded_wave(cell)
            except Exception as error:
                waves[cell.cell_id] = {"cell_id": cell.cell_id, "status": "FAILED", "error": f"{type(error).__name__}:{error}"}
                self._publish_status("FAILED", waves=waves, reason=f"S27_WAVE_FAILED:{cell.cell_id}")
                raise
            waves[cell.cell_id] = result
            if index == 0:
                try:
                    quality = validate_s27_quality_wave(self.data_root, self.plan, self.run_root, cell_id=cell.cell_id)
                except Exception as error:
                    self._publish_status("FAILED", waves=waves, reason=f"S27_QUALITY_WAVE_BLOCKED:{error}")
                    raise
                waves[cell.cell_id] = {**dict(result), "quality": quality}
                quality_passed = True
            try:
                reducer = load_s27_raw_records(self.data_root, self.plan, self.run_root, run_id=self.run_id)
                failed = sum(record.status == "FAILED" for record in reducer.records)
                fraction = failed / self.plan.frozen_inputs.completion_denominator
            except Exception as error:
                self._publish_status("FAILED", waves=waves, reason=f"S27_FAILURE_ACCOUNTING_BLOCKED:{error}")
                raise
            if fraction > self.plan.frozen_inputs.max_failure_fraction:
                self._publish_status("FAILED", waves=waves, reason="S27_FAILURE_FRACTION_EXCEEDED")
                raise S27ExecutionBlocked("S27_FAILURE_FRACTION_EXCEEDED")
            self._publish_status("RUNNING", waves=waves)
        if set(waves) != set(EXPECTED_CELL_IDS) or not quality_passed:
            self._publish_status("FAILED", waves=waves, reason="S27_QUEUE_INCOMPLETE")
            raise S27ExecutionBlocked("S27_QUEUE_INCOMPLETE")
        try:
            sealed = seal_s27_run(self.data_root, self.plan, self.run_root, run_id=self.run_id)
        except Exception as error:
            self._publish_status("FAILED", waves=waves, reason=f"{type(error).__name__}:{error}")
            raise
        result = {"status": "SEALED", "run_id": self.run_id, "plan_hash": self.plan.artifact_hash, "waves": waves, "sealed": {"manifest_hash": sealed["manifest"]["artifact_hash"], "gate_ref": sealed["gate_ref"]}}
        self._publish_status("SEALED", waves=waves)
        return result


__all__ = [
    "S27_ATTEMPT_SCHEMA",
    "S27_DEFAULT_MAX_ATTEMPTS",
    "S27_DEFAULT_M2_TOLERANCE",
    "S27_SHARD_PLAN_SCHEMA",
    "S27_SHARD_SEAL_SCHEMA",
    "S27DetachedLauncher",
    "S27ExecutionBlocked",
    "S27_GPU_INVENTORY_SCHEMA",
    "S27_LIVE_GPU_COUNT",
    "S27MaterializedCellInput",
    "S27ProductionWorker",
    "S27ProviderContext",
    "S27RetryPolicy",
    "S27UnitShard",
    "S27SubprocessSpec",
    "S27_WAVE_SEAL_SCHEMA",
    "build_s27_torch_provider",
    "build_s27_wave_seal_payload",
    "build_s27_worker_command",
    "load_s27_frozen_mappings",
    "load_s27_gpu_inventory_envelope",
    "load_s27_materialized_inputs",
    "load_s27_plan",
    "load_s27_raw_records",
    "load_s27_reference_views",
    "load_s27_shard_plan",
    "normalized_gpu_inventory",
    "nvidia_smi_inventory",
    "partition_s27_units",
    "seal_s27_wave",
    "seal_s27_run",
    "validate_s27_shard_seals",
    "validate_s27_gpu_inventory",
    "validate_s27_quality_wave",
    "write_s27_shard_plan",
]
