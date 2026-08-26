"""Producers for the immutable S2.6 formal-input boundary.

This module contains no implicit server discovery.  GPU collection is only
performed when :func:`collect_gpu_inventory` is called explicitly by the
producer CLI.  The emitted inventory keeps the JSON artifact hash separate
from the hash of the raw command capture; this avoids the impossible
``source_sha256`` self-reference that older consumers accepted.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Callable, Mapping, Sequence

from ..atomic import atomic_write_bytes
from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from .stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_UUIDS,
    EXCLUDED_PCI,
    GPU_INVENTORY_SCHEMA,
    LIVE_GPU_INVENTORY_COUNT,
    normalize_gpu_inventory,
    validate_gpu_inventory,
)
from .stage2_s207_runner import validate_s27_gpu_inventory
from .stage2_path_security import (
    DataRootPathError,
    resolve_data_root,
    resolve_data_root_ref,
)


class S206FormalInputError(ValueError):
    """Raised when an immutable S2.6 input cannot be safely produced."""


_SHA256 = 64
_GPU_QUERY_FIELDS = (
    "uuid",
    "pci.bus_id",
    "gpu_name",
    "temperature.gpu",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "compute_mode",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.total",
    "remapped_rows.failure",
    "remapped_rows.pending",
    "gpu_recovery_action",
)
_APP_QUERY_FIELDS = ("pid", "gpu_uuid", "process_name", "used_memory")
_CLEAN_VALUES = {"none", "0", "clean", "false", "not_pending", "not-pending", "n/a", "na"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise S206FormalInputError(f"SOURCE_CAPTURE_UNREADABLE:{path}") from error


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256 or any(c not in "0123456789abcdef" for c in value):
        raise S206FormalInputError(f"{field}:SHA256_REQUIRED")
    return value


def _logical_ref(root: Path, value: str, *, field: str) -> tuple[str, Path]:
    try:
        return resolve_data_root_ref(root, value, field=field, allow_absolute=False)
    except DataRootPathError as error:
        raise S206FormalInputError(str(error)) from error


def _write_immutable(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one canonical object without overwriting a different object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    if path.exists():
        try:
            existing = load_canonical_json(path)
        except (OSError, TypeError, ValueError) as error:
            raise S206FormalInputError(f"IMMUTABLE_TARGET_INVALID:{path}") from error
        if existing != value:
            raise S206FormalInputError(f"IMMUTABLE_TARGET_CONFLICT:{path}")
        return
    write_canonical_json(path, value)


def _canonical_uuid(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise S206FormalInputError("GPU_UUID_REQUIRED")
    return text if text.upper().startswith("GPU-") else f"GPU-{text}"


def _canonical_number(value: object, *, field: str, integer: bool = False) -> int | float:
    text = str(value).strip()
    if text in {"", "N/A", "[Not Supported]", "Not Supported"}:
        raise S206FormalInputError(f"{field}:NUMBER_REQUIRED")
    try:
        parsed = float(text)
    except (TypeError, ValueError) as error:
        raise S206FormalInputError(f"{field}:NUMBER_REQUIRED") from error
    if not parsed == parsed or parsed in (float("inf"), float("-inf")):
        raise S206FormalInputError(f"{field}:NONFINITE")
    if integer:
        if not parsed.is_integer():
            raise S206FormalInputError(f"{field}:INTEGER_REQUIRED")
        return int(parsed)
    return parsed


def _parse_gpu_csv(text: str) -> list[dict[str, object]]:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or any(len(row) != len(_GPU_QUERY_FIELDS) for row in rows):
        raise S206FormalInputError("GPU_QUERY_OUTPUT_INVALID")
    parsed: list[dict[str, object]] = []
    for index, raw in enumerate(rows):
        values = [item.strip() for item in raw]
        uuid = _canonical_uuid(values[0])
        pci = values[1].upper()
        memory_used = _canonical_number(values[4], field=f"gpu[{index}].memory_used_mib", integer=True)
        memory_total = _canonical_number(values[5], field=f"gpu[{index}].memory_total_mib", integer=True)
        utilization = _canonical_number(values[6], field=f"gpu[{index}].utilization_gpu_percent")
        volatile = _canonical_number(values[8], field=f"gpu[{index}].ecc_uncorrected_volatile", integer=True)
        aggregate = _canonical_number(values[9], field=f"gpu[{index}].ecc_uncorrected_aggregate", integer=True)
        remap_failure = _canonical_number(values[10], field=f"gpu[{index}].row_remap_failure", integer=True)
        remap_pending = _canonical_number(values[11], field=f"gpu[{index}].row_remap_pending", integer=True)
        recovery = values[12]
        remap_status = "pending" if remap_pending != 0 else ("failure" if remap_failure != 0 else "None")
        health_state = (
            "HEALTHY"
            if memory_used == 0 and utilization == 0 and volatile == 0 and aggregate == 0
            and remap_failure == 0 and remap_pending == 0 and recovery.casefold() in _CLEAN_VALUES
            else "UNHEALTHY"
        )
        parsed.append(
            {
                "uuid": uuid,
                "pci_bus_id": pci,
                "gpu_name": values[2],
                "temperature_c": _canonical_number(values[3], field=f"gpu[{index}].temperature_c"),
                "memory_used_mib": memory_used,
                "memory_total_mib": memory_total,
                "utilization_gpu_percent": utilization,
                "compute_mode": values[7],
                "ecc_uncorrected_volatile": volatile,
                "ecc_uncorrected_aggregate": aggregate,
                "row_remap_failure": remap_failure,
                "row_remap_pending": remap_pending,
                "row_remap_status": remap_status,
                "gpu_recovery_action": recovery,
                "health_state": health_state,
            }
        )
    return normalize_gpu_inventory(parsed)


def _parse_apps_csv(text: str) -> list[dict[str, object]]:
    if not text.strip():
        return []
    rows = list(csv.reader(io.StringIO(text)))
    if any(len(row) != len(_APP_QUERY_FIELDS) for row in rows):
        raise S206FormalInputError("COMPUTE_APPS_QUERY_OUTPUT_INVALID")
    apps: list[dict[str, object]] = []
    for row in rows:
        apps.append(
            {
                "pid": _canonical_number(row[0].strip(), field="compute_app.pid", integer=True),
                "gpu_uuid": _canonical_uuid(row[1]),
                "process_name": row[2].strip(),
                "used_memory": row[3].strip(),
            }
        )
    return apps


def _run_query(
    command: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[str, str]:
    try:
        completed = runner(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S206FormalInputError(f"NVIDIA_SMI_QUERY_FAILED:{type(error).__name__}") from error
    stdout = completed.stdout if isinstance(completed.stdout, str) else ""
    stderr = completed.stderr if isinstance(completed.stderr, str) else ""
    return stdout, stderr


def collect_gpu_inventory(
    *,
    output: str | Path,
    source_output: str | Path,
    data_root: str | Path | None = None,
    executable: str = "nvidia-smi",
    runner: Callable[..., Any] = subprocess.run,
    checked_at: str | None = None,
) -> dict[str, object]:
    """Collect and publish one explicit live eight-card inventory.

    No collection occurs on import.  ``source_output`` is a raw UTF-8 capture
    of both query responses and is the only input to ``source_sha256``; the
    inventory JSON's own bytes are never used for that field.
    """

    if data_root is None:
        # Keep the historical convenience API, but still establish and audit
        # an explicit lexical root before either output path is inspected.
        artifact_lexical = Path(output).absolute()
        root = resolve_data_root(artifact_lexical.parent)
    else:
        try:
            root = resolve_data_root(data_root)
        except DataRootPathError as error:
            raise S206FormalInputError(str(error)) from error
    try:
        artifact_ref, artifact_path = resolve_data_root_ref(root, output, field="GPU_INVENTORY_OUTPUT")
        source_ref, source_path = resolve_data_root_ref(root, source_output, field="GPU_INVENTORY_SOURCE")
    except DataRootPathError as error:
        raise S206FormalInputError("GPU_INVENTORY_OUTPUT_OUTSIDE_DATA_ROOT") from error
    if artifact_path == source_path:
        raise S206FormalInputError("GPU_INVENTORY_SOURCE_SELF_REFERENCE")
    observed_at = checked_at or _now()
    gpu_command = (
        executable,
        f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    app_command = (
        executable,
        f"--query-compute-apps={','.join(_APP_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    gpu_stdout, gpu_stderr = _run_query(gpu_command, runner=runner)
    app_stdout, app_stderr = _run_query(app_command, runner=runner)
    rows = _parse_gpu_csv(gpu_stdout)
    apps = _parse_apps_csv(app_stdout)
    if len(rows) != LIVE_GPU_INVENTORY_COUNT:
        raise S206FormalInputError(f"GPU_INVENTORY_LIVE_CARD_COUNT_INVALID:{len(rows)}")
    try:
        summary = validate_gpu_inventory(rows, compute_apps=apps)
        validate_s27_gpu_inventory(rows, compute_apps=apps)
    except Exception as error:
        raise S206FormalInputError(f"GPU_INVENTORY_NOT_FORMAL_READY:{error}") from error
    apps_by_uuid: dict[str, list[dict[str, object]]] = {}
    for app in apps:
        apps_by_uuid.setdefault(str(app["gpu_uuid"]).casefold(), []).append(dict(app))
    for row in rows:
        row["compute_apps"] = apps_by_uuid.get(str(row["uuid"]).casefold(), [])
    capture = (
        "# S2.6 live GPU inventory capture\n"
        f"# checked_at={observed_at}\n"
        f"$ {' '.join(gpu_command)}\n{gpu_stdout}"
        f"$ {' '.join(app_command)}\n{app_stdout}"
        + (f"# stderr_gpu={gpu_stderr}\n" if gpu_stderr else "")
        + (f"# stderr_compute_apps={app_stderr}\n" if app_stderr else "")
    ).encode("utf-8")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.exists() and source_path.read_bytes() != capture:
        raise S206FormalInputError(f"IMMUTABLE_SOURCE_CAPTURE_CONFLICT:{source_path}")
    if not source_path.exists():
        atomic_write_bytes(source_path, capture)
    source_sha = _sha256_file(source_path)
    payload: dict[str, object] = {
        "schema_version": GPU_INVENTORY_SCHEMA,
        "scope": "formal",
        "status": "OBSERVED",
        "checked_at": observed_at,
        "artifact_ref": artifact_ref,
        "source_ref": source_ref,
        "source_sha256": source_sha,
        "rows": rows,
        "compute_apps": apps,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_pci": EXCLUDED_PCI,
        "excluded_gpu_uuid": summary["excluded_gpu_uuid"],
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _write_immutable(artifact_path, payload)
    return payload


def build_cost_semantics_contract(
    *,
    scientific_measurement_ref: str | None = None,
    cost_io_quiescent: bool = False,
) -> dict[str, object]:
    """Build the plan-frozen S2.6 cost meanings without invented timings."""

    if type(cost_io_quiescent) is not bool:
        raise S206FormalInputError("COST_IO_QUIESCENT_BOOLEAN_REQUIRED")
    scientific: dict[str, object] = {
        "defined": True,
        "definition": "shared paired pilot gradient cost per equal sample budget B",
        "measurement_status": "PENDING_S2.9",
    }
    if scientific_measurement_ref is not None:
        if not scientific_measurement_ref or "\\" in scientific_measurement_ref:
            raise S206FormalInputError("scientific_measurement_ref:INVALID_REFERENCE")
        scientific["measurement_status"] = "OBSERVED"
        scientific["measurement_ref"] = scientific_measurement_ref
    payload: dict[str, object] = {
        "schema_version": "stage2-s206-cost-semantics-contract-v1",
        "scope": "formal",
        "status": "FROZEN",
        "measurement_boundary": {
            "isolated_estimator_cost": "stage2.09.capacity:estimator-only incremental cost under isolated conditions",
            "online_training_incremental_cost": "stage2.09.capacity:incremental cost over the frozen online training baseline",
        },
        "scientific_equal_sample_cost": scientific,
        "isolated_estimator_cost": {
            "defined": True,
            "definition": "estimator-only incremental cost under isolated pilot conditions",
            "measurement_status": "PENDING_S2.9",
        },
        "online_training_incremental_cost": {
            "defined": True,
            "definition": "incremental cost over the frozen online training baseline",
            "measurement_status": "PENDING_S2.9",
        },
        "cost_io_quiescent": cost_io_quiescent,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def build_retry_policy_contract(*, max_cell_attempts: int | None = None) -> dict[str, object]:
    """Build the explicit no-guess retry policy used by S2.6."""

    if max_cell_attempts is None:
        raise S206FormalInputError("MAX_CELL_ATTEMPTS_EXPLICIT_REQUIRED")
    if isinstance(max_cell_attempts, bool) or max_cell_attempts not in {1, 2, 3}:
        raise S206FormalInputError("MAX_CELL_ATTEMPTS_MUST_BE_1_TO_3")
    payload: dict[str, object] = {
        "schema_version": "stage2-s206-retry-policy-v1",
        "scope": "formal",
        "status": "FROZEN",
        "max_cell_attempts": max_cell_attempts,
        "reuse_mapping_on_retry": True,
        "new_pilot_draws_on_retry": False,
        "preserve_failure_records": True,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def _load_hashed_ref(root: Path, value: str, *, field: str) -> tuple[str, Path, dict[str, object], str]:
    ref, path = _logical_ref(root, value, field=field)
    try:
        raw = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S206FormalInputError(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(raw, Mapping):
        raise S206FormalInputError(f"{field}:OBJECT_REQUIRED")
    payload = dict(raw)
    declared = _hash(payload.get("artifact_hash"), f"{field}.artifact_hash")
    if canonical_json_hash({k: v for k, v in payload.items() if k != "artifact_hash"}) != declared:
        raise S206FormalInputError(f"{field}:ARTIFACT_HASH_MISMATCH")
    return ref, path, payload, declared


def _require_six_g23(payload: Mapping[str, object], *, field: str) -> None:
    if payload.get("schema_version") != "stage2-g23-reference-evaluation-v1" or payload.get("status") != "PASS" or payload.get("formal_eligible") is not True:
        raise S206FormalInputError(f"{field}:PASS_FORMAL_REQUIRED")
    if payload.get("required_cell_count") != 6 or payload.get("complete_cell_count") != 6:
        raise S206FormalInputError(f"{field}:SIX_CELL_COMPLETENESS_REQUIRED")
    cells = payload.get("cells")
    expected = tuple(anchor.replace(".", ":", 1) for anchor in ANCHOR_IDS)
    if not isinstance(cells, list) or tuple(item.get("cell_id") for item in cells if isinstance(item, Mapping)) != expected:
        raise S206FormalInputError(f"{field}:ANCHOR_SET_INVALID")
    if any(not isinstance(item, Mapping) or item.get("status") != "PASS" or item.get("formal_eligible") is not True for item in cells):
        raise S206FormalInputError(f"{field}:CELL_PASS_REQUIRED")


def _require_g24a(payload: Mapping[str, object], *, field: str, g23_ref: str, g23_hash: str) -> None:
    if payload.get("schema_version") != "stage2-g24a-formal-evaluation-v1" or payload.get("gate_id") != "stage2.G2.4a" or payload.get("status") != "PASS" or payload.get("formal_eligible") is not True:
        raise S206FormalInputError(f"{field}:PASS_FORMAL_REQUIRED")
    if payload.get("cell_count") != 6:
        raise S206FormalInputError(f"{field}:SIX_CELL_COMPLETENESS_REQUIRED")
    if payload.get("g23_evaluation_ref") != g23_ref or payload.get("g23_evaluation_hash") != g23_hash:
        raise S206FormalInputError(f"{field}:G23_BINDING_INVALID")
    results = payload.get("results")
    expected = tuple(anchor.replace(".", ":", 1) for anchor in ANCHOR_IDS)
    if not isinstance(results, list) or tuple(item.get("cell_id") for item in results if isinstance(item, Mapping)) != expected:
        raise S206FormalInputError(f"{field}:ANCHOR_SET_INVALID")
    for item in results:
        if not isinstance(item, Mapping) or item.get("status") != "PASS" or item.get("formal_eligible") is not True or not isinstance(item.get("metrics"), Mapping):
            raise S206FormalInputError(f"{field}:OUTPUT_DERIVED_METRICS_REQUIRED")


def _require_s205(payload: Mapping[str, object], *, g23_ref: str, g23_hash: str, field: str) -> None:
    if payload.get("schema_version") != "stage2-s205-rebind-plan-v1" or payload.get("status") != "READY" or payload.get("formal_eligible") is not True:
        raise S206FormalInputError(f"{field}:READY_FORMAL_REQUIRED")
    if payload.get("g23_evaluation_ref") != g23_ref or payload.get("g23_evaluation_hash") != g23_hash:
        raise S206FormalInputError(f"{field}:G23_BINDING_INVALID")
    rows = payload.get("cells")
    if not isinstance(rows, list) or len(rows) != len(ANCHOR_IDS):
        raise S206FormalInputError(f"{field}:SIX_CELL_ROWS_REQUIRED")
    expected = tuple(anchor.replace(".", ":", 1) for anchor in ANCHOR_IDS)
    if tuple(item.get("cell_id") for item in rows if isinstance(item, Mapping)) != expected:
        raise S206FormalInputError(f"{field}:ANCHOR_ORDER_INVALID")
    required = {"config_ref", "environment_ref", "formal_execution_ref", "reference_artifact_refs", "task_result_status_path", "config_hash", "result_hash"}
    for item in rows:
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise S206FormalInputError(f"{field}:CELL_IDENTITY_FIELDS_REQUIRED")
        for name in ("config_hash", "result_hash"):
            _hash(item[name], f"{field}.{name}")


def build_formal_execution_evidence(
    *,
    data_root: str | Path,
    g23_evaluation_ref: str,
    g24a_evaluation_ref: str,
    s205_rebind_ref: str,
    parent_evidence_ref: str | None = None,
    matrix_prerequisite_ref: str | None = None,
    contract_freeze_hash: str | None = None,
    asset_manifest_hashes: Sequence[str] = (),
    checked_at: str | None = None,
) -> dict[str, object]:
    """Produce a new S2.6 evidence amendment from exact upstream objects.

    The old G2.2 evidence is never accepted as a substitute: G2.3 and G2.4a
    must be loaded and bound by both reference and content hash.  S2.5's
    rebind is likewise checked against the exact G2.3 object.
    """

    root = Path(data_root).resolve()
    g23_ref, _g23_path, g23, g23_hash = _load_hashed_ref(root, g23_evaluation_ref, field="g23_evaluation")
    g24a_ref, _g24a_path, g24a, g24a_hash = _load_hashed_ref(root, g24a_evaluation_ref, field="g24a_evaluation")
    s205_ref, _s205_path, s205, s205_hash = _load_hashed_ref(root, s205_rebind_ref, field="s205_rebind")
    _require_six_g23(g23, field="g23_evaluation")
    _require_g24a(g24a, field="g24a_evaluation", g23_ref=g23_ref, g23_hash=g23_hash)
    _require_s205(s205, g23_ref=g23_ref, g23_hash=g23_hash, field="s205_rebind")

    parent: FormalExecutionEvidence | None = None
    parent_ref: str | None = None
    parent_hash: str | None = None
    if parent_evidence_ref is not None:
        parent_ref, _parent_path, parent_raw, parent_hash = _load_hashed_ref(root, parent_evidence_ref, field="parent_evidence")
        try:
            parent = FormalExecutionEvidence.from_mapping(parent_raw)
        except (TypeError, ValueError) as error:
            raise S206FormalInputError("parent_evidence:FORMAL_EVIDENCE_INVALID") from error
        if parent.run_intent != "formal":
            raise S206FormalInputError("parent_evidence:FORMAL_REQUIRED")

    at = checked_at or _now()
    gates: list[GateRecord] = []
    if parent is not None:
        gates.extend(parent.prerequisite_gates)
    gates = [gate for gate in gates if gate.gate_id not in {"stage2.G2.3", "stage2.G2.4a"}]
    gates.extend(
        [
            GateRecord(
                gate_id="stage2.G2.3",
                stage=2,
                status=GateStatus.PASS,
                checked_at=at,
                measured={"evaluation_ref": g23_ref, "evaluation_artifact_hash": g23_hash},
                threshold={"required_cell_count": 6, "status": "PASS"},
                evidence_refs=(g23_ref,),
            ),
            GateRecord(
                gate_id="stage2.G2.4a",
                stage=2,
                status=GateStatus.PASS,
                checked_at=at,
                measured={"evaluation_ref": g24a_ref, "evaluation_artifact_hash": g24a_hash},
                threshold={"cell_count": 6, "status": "PASS"},
                evidence_refs=(g24a_ref,),
            ),
        ]
    )
    metadata: dict[str, object] = {
        "producer": "stage2-s206-formal-inputs-v1",
        "g23_evaluation_ref": g23_ref,
        "g23_evaluation_hash": g23_hash,
        "g24a_evaluation_ref": g24a_ref,
        "g24a_evaluation_hash": g24a_hash,
        "s205_rebind_ref": s205_ref,
        "s205_rebind_hash": s205_hash,
        "s205_rebind_g23_ref": g23_ref,
        "s205_rebind_g23_hash": g23_hash,
    }
    if matrix_prerequisite_ref is not None:
        matrix_ref, _matrix_path, matrix, matrix_hash = _load_hashed_ref(root, matrix_prerequisite_ref, field="matrix_prerequisite")
        if matrix.get("scope") != "formal" or matrix.get("formal_eligible") is not True:
            raise S206FormalInputError("matrix_prerequisite:FORMAL_REQUIRED")
        metadata.update({"matrix_prerequisite_ref": matrix_ref, "matrix_prerequisite_hash": matrix_hash})
    if parent_ref is not None and parent_hash is not None:
        metadata.update({"amendment_parent_ref": parent_ref, "amendment_parent_hash": parent_hash})
    contract_hash = parent.contract_freeze_hash if parent is not None else contract_freeze_hash
    if contract_hash is None:
        raise S206FormalInputError("FORMAL_CONTRACT_FREEZE_REQUIRED")
    _hash(contract_hash, "contract_freeze_hash")
    assets = parent.asset_manifest_hashes if parent is not None else tuple(asset_manifest_hashes)
    for index, value in enumerate(assets):
        _hash(value, f"asset_manifest_hashes[{index}]")
    if not assets:
        raise S206FormalInputError("FORMAL_ASSET_MANIFESTS_REQUIRED")
    evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=contract_hash,
        asset_manifest_hashes=tuple(assets),
        prerequisite_gates=tuple(gates),
        metadata=metadata,
    )
    return evidence.to_dict()


__all__ = [
    "S206FormalInputError",
    "build_cost_semantics_contract",
    "build_formal_execution_evidence",
    "build_retry_policy_contract",
    "collect_gpu_inventory",
]
