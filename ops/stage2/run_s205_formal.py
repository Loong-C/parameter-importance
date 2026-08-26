"""Detached S2.5/G2.4a production launcher.

``--preflight`` is read-only and revalidates the fresh S2.4/G2.3 rebind,
sampling plan and preregistered pilot B/M/R plan.  ``--execute`` reruns that
preflight immediately before starting six UUID-bound workers through a
deterministic four-GPU LPT queue;
the workers call the S2.5-only strict runner and never create draws.  A final
G2.4a object is published only after all six immutable cell summaries pass.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_s25_formal import (
    APPROVED_GPU_UUIDS,
    EXPECTED_CELL_IDS,
    S25ExecutionBlocked,
    S25FormalRunner,
    load_s25_rebind_plan,
    preflight_s25,
    run_s25_runner_qualification,
)
from param_importance_nlp.experiments.stage2_s25_inputs import (
    S205_INPUT_INDEX_SCHEMA,
    S205InputBlocked,
    validate_s205_development_sweep,
)
from param_importance_nlp.experiments.sampling import SamplingPlan
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
from param_importance_nlp.contracts.g21_formal_handoff import (
    ALLOWED_DEVICES as APPROVED_GPU_BINDINGS,
    EXCLUDED_PCI,
    EXCLUDED_UUID,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, value: str, *, field: str, allow_missing: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S25ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE")
    root = root.resolve()
    candidate = root / path
    # Formal references are content-addressed inputs.  A symlink can otherwise
    # make a harmless-looking DATA_ROOT-relative ref resolve to an unrelated
    # mutable file after preflight, so reject every symlink component (including
    # the terminal path) before resolving it.
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise S25ExecutionBlocked(f"{field}:SYMLINK_NOT_ALLOWED")
    result = candidate.resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    if not allow_missing and not result.exists():
        raise S25ExecutionBlocked(f"{field}:MISSING")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _launcher_source_sha256() -> str:
    """Digest the exact launcher source used by every child worker."""

    return _file_sha256(Path(__file__).resolve())


def _require_launcher_in_repository(repository: Path) -> None:
    try:
        Path(__file__).resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise S25ExecutionBlocked("S205_LAUNCHER_SOURCE_OUTSIDE_REPOSITORY") from error


def _strict_inventory_path(root: Path, path: Path) -> Path:
    """Resolve an inventory path only when it is a non-symlink DATA_ROOT ref."""

    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_PATH_OUTSIDE_DATA_ROOT") from error
    # Reuse the logical-reference policy for symlink and escape checks.  The
    # caller may pass an absolute Path, while the envelope itself stores the
    # required DATA_ROOT-relative artifact_ref.
    return _logical(root, relative.as_posix(), field="gpu_inventory_ref")


_GPU_INVENTORY_SCHEMA = "stage2-s206-gpu-inventory-v1"
_GPU_INVENTORY_HEALTH_ALIASES = {
    "memory_used_mib": ("memory_used_mib", "memory_used", "memory.used"),
    "memory_total_mib": ("memory_total_mib", "memory_total", "memory.total"),
    "utilization_gpu_percent": (
        "utilization_gpu_percent", "utilization_percent", "utilization_gpu", "utilization.gpu"
    ),
    "ecc_uncorrected_volatile": (
        "ecc_uncorrected_volatile", "ecc_volatile_uncorrected",
        "ecc.errors.uncorrected.volatile.total",
    ),
    "ecc_uncorrected_aggregate": (
        "ecc_uncorrected_aggregate", "ecc_aggregate_uncorrected",
        "ecc.errors.uncorrected.aggregate.total",
    ),
    "temperature_c": ("temperature_c", "temperature_gpu_c", "temperature.gpu"),
    "row_remap_status": ("row_remap_status", "row_remap", "row_remap_pending"),
    "gpu_recovery_action": ("gpu_recovery_action", "recovery_action"),
}
_GPU_CLEAN_VALUES = {"none", "0", "clean", "false", "not_pending", "not-pending", "n/a", "na"}


def _inventory_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S25ExecutionBlocked(f"{field}:UUID_REQUIRED")
    text = value.strip()
    return text if text.upper().startswith("GPU-") else f"GPU-{text}"


def _inventory_pci(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise S25ExecutionBlocked(f"{field}:PCI_REQUIRED")
    match = re.fullmatch(
        r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])",
        value.strip().upper(),
    )
    if match is None:
        raise S25ExecutionBlocked(f"{field}:PCI_INVALID")
    return f"0000:{match.group(1)[-4:]}:{match.group(2)}.{match.group(3)}"


def _inventory_number(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise S25ExecutionBlocked(f"{field}:NUMBER_REQUIRED") from error
    if not math.isfinite(result):
        raise S25ExecutionBlocked(f"{field}:NONFINITE")
    return result


def _validate_s205_gpu_inventory(
    rows: Sequence[Mapping[str, object]],
    *,
    compute_apps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate the complete eight-card S2.6/S2.7 live inventory contract."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ROWS_REQUIRED")
    if len(rows) != 8:
        raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_LIVE_CARD_COUNT_INVALID:{len(rows)}")
    if not isinstance(compute_apps, Sequence) or isinstance(compute_apps, (str, bytes)):
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    expected = {pci.casefold(): uuid.casefold() for pci, uuid in APPROVED_GPU_BINDINGS}
    expected[EXCLUDED_PCI.casefold()] = EXCLUDED_UUID.casefold()
    approved = {uuid.casefold() for _pci, uuid in APPROVED_GPU_BINDINGS}
    seen_uuid: set[str] = set()
    seen_pci: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_ROW_INVALID:{index}")
        row = dict(source)
        uuid = _inventory_uuid(row.get("uuid", row.get("gpu_uuid")), field=f"gpu[{index}]")
        pci = _inventory_pci(row.get("pci_bus_id", row.get("pci")), field=f"gpu[{index}]")
        if uuid.casefold() in seen_uuid:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_DUPLICATE_UUID")
        if pci.casefold() in seen_pci:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_DUPLICATE_PCI")
        seen_uuid.add(uuid.casefold())
        seen_pci.add(pci.casefold())
        row["uuid"] = uuid
        row["pci_bus_id"] = pci
        for canonical, aliases in _GPU_INVENTORY_HEALTH_ALIASES.items():
            found = next((row[name] for name in aliases if name in row), None)
            if found is None:
                raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_HEALTH_FIELD_MISSING:{canonical}")
            row[canonical] = found
        for field in (
            "memory_used_mib", "memory_total_mib", "utilization_gpu_percent",
            "ecc_uncorrected_volatile", "ecc_uncorrected_aggregate", "temperature_c",
        ):
            row[field] = _inventory_number(row[field], field=f"gpu[{index}].{field}")
        normalized.append(row)
    observed = {
        str(row["pci_bus_id"]).casefold(): str(row["uuid"]).casefold()
        for row in normalized
        if str(row["pci_bus_id"]).casefold() in expected
    }
    if observed != expected:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_OR_EXCLUDED_IDENTITY_DRIFT")
    apps_by_uuid: dict[str, list[dict[str, object]]] = {}
    live_uuids = {str(row["uuid"]).casefold() for row in normalized}
    app_rows: list[dict[str, object]] = []
    for index, source in enumerate(compute_apps):
        if not isinstance(source, Mapping):
            raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_COMPUTE_APP_INVALID:{index}")
        app = dict(source)
        app_uuid = _inventory_uuid(app.get("gpu_uuid", app.get("uuid")), field=f"compute_app[{index}]")
        if app_uuid.casefold() not in live_uuids:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_COMPUTE_APP_GPU_UNKNOWN")
        app["gpu_uuid"] = app_uuid
        apps_by_uuid.setdefault(app_uuid.casefold(), []).append(app)
        app_rows.append(app)
    for row in normalized:
        uuid_key = str(row["uuid"]).casefold()
        if uuid_key not in approved:
            continue
        if apps_by_uuid.get(uuid_key):
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
        if (
            float(row["memory_total_mib"]) <= 0
            or float(row["memory_used_mib"]) != 0
            or float(row["utilization_gpu_percent"]) != 0
        ):
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_CARD_NOT_IDLE")
        if float(row["ecc_uncorrected_volatile"]) != 0 or float(row["ecc_uncorrected_aggregate"]) != 0:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_CARD_ECC_NOT_CLEAN")
        if float(row["temperature_c"]) < 0 or float(row["temperature_c"]) >= 85:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_CARD_TEMPERATURE_INVALID")
        if str(row["row_remap_status"]).strip().casefold() not in _GPU_CLEAN_VALUES:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_ROW_REMAP_NOT_CLEAN")
        if str(row["gpu_recovery_action"]).strip().casefold() not in _GPU_CLEAN_VALUES:
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_APPROVED_CARD_RECOVERY_NOT_CLEAN")
    return {
        "schema_version": _GPU_INVENTORY_SCHEMA,
        "approved_gpu_uuids": [uuid for _pci, uuid in APPROVED_GPU_BINDINGS],
        "excluded_gpu_uuid": EXCLUDED_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "inventory_count": len(normalized),
        "inventory": normalized,
        "compute_apps": app_rows,
    }


def _load_inventory_snapshot(
    path: Path | None,
    *,
    data_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load the strict S2.6/S2.7 source-bound inventory envelope.

    ``artifact_ref`` names the inventory JSON and ``source_ref`` names a
    distinct raw ``nvidia-smi`` capture.  This exact S2.6/S2.7 protocol checks
    the canonical artifact hash, raw-capture SHA-256, complete eight-row
    identity set, health fields and compute-app list.  The wrapper adds the
    common DATA_ROOT/no-symlink policy and an explicit path identity for S2.5
    status and detached receipts.
    """

    if path is None:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_JSON_REQUIRED")
    root = data_root.resolve()
    resolved = _strict_inventory_path(root, path)
    try:
        envelope = load_canonical_json(resolved)
    except (OSError, TypeError, ValueError) as error:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_JSON_INVALID") from error
    if not isinstance(envelope, Mapping):
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ENVELOPE_REQUIRED")
    source_ref = envelope.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SOURCE_REF_REQUIRED")
    # Validate the raw capture path independently so the shared loader's
    # resolved path cannot hide a symlink component.
    _logical(root, source_ref, field="gpu_inventory_source_ref")
    payload = dict(envelope)
    if payload.get("schema_version") != _GPU_INVENTORY_SCHEMA:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SCHEMA_INVALID")
    artifact_ref = payload.get("artifact_ref")
    expected_artifact_ref = resolved.relative_to(root).as_posix()
    if not isinstance(artifact_ref, str) or not artifact_ref:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ARTIFACT_REF_REQUIRED")
    if artifact_ref != expected_artifact_ref:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ARTIFACT_REF_PATH_MISMATCH")
    rows = payload.get("rows", payload.get("gpus"))
    apps = payload.get("compute_apps")
    if not isinstance(rows, list) or not all(isinstance(item, Mapping) for item in rows):
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ROWS_REQUIRED")
    if not isinstance(apps, list) or not all(isinstance(item, Mapping) for item in apps):
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_COMPUTE_APPS_REQUIRED")
    artifact_hash = payload.get("artifact_hash")
    if not isinstance(artifact_hash, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ARTIFACT_HASH_REQUIRED")
    if canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"}) != artifact_hash:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_ARTIFACT_HASH_MISMATCH")
    source_ref = payload.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SOURCE_REF_REQUIRED")
    source_path = _logical(root, source_ref, field="gpu_inventory_source_ref")
    if source_path == resolved:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SOURCE_REF_SELF_REFERENCE")
    source_sha = _file_sha256(source_path)
    declared_source_sha = payload.get("source_sha256")
    if not isinstance(declared_source_sha, str) or re.fullmatch(r"[0-9a-f]{64}", declared_source_sha) is None:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SOURCE_SHA256_REQUIRED")
    if declared_source_sha != source_sha:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_SOURCE_SHA256_MISMATCH")
    summary = _validate_s205_gpu_inventory(rows, compute_apps=apps)
    identity = {
        "source_ref": source_ref,
        "artifact_ref": artifact_ref,
        "artifact_hash": artifact_hash,
        "source_sha256": source_sha,
        "path": str(resolved),
        "schema_version": _GPU_INVENTORY_SCHEMA,
        "compute_apps": [dict(item) for item in apps],
    }
    return [dict(row) for row in summary["inventory"]], identity  # type: ignore[index]


def _inventory(path: Path | None, *, data_root: Path | None = None) -> list[dict[str, object]]:
    if path is not None:
        rows, _identity = _load_inventory_snapshot(
            path,
            data_root=(data_root.resolve() if data_root is not None else path.resolve().parent),
        )
        return rows
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_UNAVAILABLE:{type(error).__name__}") from error
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 8:
            rows.append({
                "index": fields[0], "uuid": fields[1], "pci_bus_id": fields[2],
                "memory_used_mib": fields[3], "memory_total_mib": fields[4],
                "utilization_gpu_percent": fields[5],
                "ecc_uncorrected_volatile": fields[6],
                "ecc_uncorrected_aggregate": fields[7],
                "gpu_recovery_action": "None",
            })
    if not rows:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_EMPTY")
    return rows


def _validate_inventory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    try:
        summary = _validate_s205_gpu_inventory(rows, compute_apps=())
    except Exception as error:
        raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_INVALID:{error}") from error
    return dict(summary)


def _write_status(path: Path, payload: Mapping[str, object]) -> None:
    value = {"schema_version": "stage2-s205-formal-status-v1", "updated_at": _now(), **dict(payload)}
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(path, value)


_S205_GATE_REQUIREMENTS = (
    "same_total_sample_pool_B_for_raw_u_double",
    "disjoint_double_halves_each_B_over_2",
    "m2_u_equals_double_with_stage1_tolerance",
    "complete_batch_mean_gradient_invariant_across_M",
    "signed_u_and_double_outputs_not_clamped",
    "gradient_formula_sample_token_and_memory_costs_recorded",
    "state_summary_unchanged_before_after",
    "failure_retry_replays_same_mapping_and_atomic_publish",
    "streaming_reducer_matches_offline_recompute",
    "concurrent_retry_deduplication_and_deterministic_reduction",
    "reference_topk_and_cross_M_summaries_retained_before_release",
)


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        existing = load_canonical_json(path)
        if existing != dict(payload):
            raise S25ExecutionBlocked(f"S205_OUTPUT_CONFLICT:{path}")
        return
    write_canonical_json(path, dict(payload))


def _repository_commit(repository: Path) -> str:
    try:
        dirty = subprocess.run(
            ["git", "-C", str(repository.resolve()), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        if dirty.stdout.strip():
            raise S25ExecutionBlocked("S205_REPOSITORY_DIRTY")
        completed = subprocess.run(
            ["git", "-C", str(repository.resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S25ExecutionBlocked("S205_RUNNER_COMMIT_UNAVAILABLE") from error
    commit = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise S25ExecutionBlocked("S205_RUNNER_COMMIT_INVALID")
    return commit


def _gate_only(args: argparse.Namespace) -> dict[str, object]:
    """Run the bounded qualification fixture; never run formal data."""

    root = args.data_root.resolve()
    plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
    operations = _logical(root, args.operations_root, field="operations_root", allow_missing=True)
    operations.mkdir(parents=True, exist_ok=True)
    runner_commit = _repository_commit(getattr(args, "repository", _REPOSITORY_ROOT))
    payload = run_s25_runner_qualification(
        artifact_root=operations / "g2.4a-runner-qualification" / runner_commit,
        rebind_plan=plan,
        runner_commit=runner_commit,
    )
    payload = {
        **payload,
        "rebind_ref": str(plan.get("_rebind_ref", args.s205_rebind_ref)),
        "rebind_hash": plan.get("artifact_hash"),
        "g23_evaluation_ref": plan.get("_g23_ref"),
        "g23_evaluation_hash": plan.get("g23_evaluation_hash"),
        "gate_requirements": list(_S205_GATE_REQUIREMENTS),
    }
    payload.pop("artifact_hash", None)
    payload["artifact_hash"] = canonical_json_hash(payload)
    _write_once(operations / "g2.4a-runner-qualification.json", payload)
    return payload


class _S205DynamicLPTQueue:
    """Deterministic four-GPU queue for the six frozen S2.5 cells."""

    _MODEL_RANK = {"pythia-14m": 1, "pythia-31m-deduped": 2}
    _STAGE_RANK = {"initialization": 1, "early": 2, "mid_late": 3}

    def __init__(self, cell_ids: Sequence[str], gpu_ids: Sequence[str]) -> None:
        if tuple(cell_ids) != tuple(EXPECTED_CELL_IDS):
            raise ValueError("S205_QUEUE_CELL_SET_MISMATCH")
        if tuple(gpu_ids) != tuple(APPROVED_GPU_UUIDS):
            raise ValueError("S205_QUEUE_GPU_SET_MISMATCH")
        self._pending = deque(sorted(cell_ids, key=self._lpt_key))
        self._free = deque(gpu_ids)
        self._active: dict[str, str] = {}
        self._completed: list[str] = []

    @classmethod
    def _lpt_key(cls, cell_id: str) -> tuple[int, int, int]:
        model, stage = cell_id.split(":", 1)
        return (-cls._MODEL_RANK[model], -cls._STAGE_RANK[stage], EXPECTED_CELL_IDS.index(cell_id))

    def fill(self) -> tuple[tuple[str, str], ...]:
        assignments: list[tuple[str, str]] = []
        while self._pending and self._free:
            cell_id = self._pending.popleft()
            gpu_uuid = self._free.popleft()
            self._active[cell_id] = gpu_uuid
            assignments.append((cell_id, gpu_uuid))
        return tuple(assignments)

    def complete(self, cell_id: str) -> tuple[tuple[str, str], ...]:
        if cell_id not in self._active:
            raise ValueError(f"S205_QUEUE_CELL_NOT_ACTIVE:{cell_id}")
        gpu_uuid = self._active.pop(cell_id)
        self._completed.append(cell_id)
        free = [*self._free, gpu_uuid]
        self._free = deque(sorted(free, key=APPROVED_GPU_UUIDS.index))
        return self.fill()

    def skip(self, cell_id: str) -> None:
        """Remove a previously completed cell before scheduling a resume."""

        if cell_id in self._active:
            raise ValueError(f"S205_QUEUE_CELL_ACTIVE:{cell_id}")
        try:
            self._pending.remove(cell_id)
        except ValueError as error:
            raise ValueError(f"S205_QUEUE_CELL_NOT_PENDING:{cell_id}") from error
        self._completed.append(cell_id)

    @property
    def active(self) -> dict[str, str]:
        return dict(self._active)

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)


def _load_materialization_index(root: Path, reference: str) -> dict[str, object]:
    """Load the producer-owned S2.5 input index and recheck its closure.

    The index is the handoff boundary between ``materialize_s205_formal_inputs``
    and this launcher.  Keeping this check here prevents a caller from mixing
    a valid sweep with a different sampling plan or formal-execution evidence.
    """

    path = _logical(root, reference, field="input_index_ref")
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_JSON_INVALID") from error
    if not isinstance(value, Mapping):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_OBJECT_REQUIRED")
    required = {
        "schema_version", "status", "formal_eligible", "sampling_plan_ref",
        "sampling_plan_hash", "development_sweep_plan_ref",
        "development_sweep_plan_hash", "preregistration_ref",
        "sampling_plan_task_ref", "formal_execution_ref",
        "primary_parameters_selected", "confirmatory_draws_generated",
        "reference_draws_generated", "artifact_hash",
    }
    if set(value) != required:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_FIELDS_MISMATCH")
    if (
        value.get("schema_version") != S205_INPUT_INDEX_SCHEMA
        or value.get("status") != "FROZEN"
        or value.get("formal_eligible") is not True
        or value.get("primary_parameters_selected") is not False
        or value.get("confirmatory_draws_generated") is not False
        or value.get("reference_draws_generated") is not False
    ):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SEMANTICS_INVALID")
    declared_hash = value.get("artifact_hash")
    if not isinstance(declared_hash, str) or declared_hash != canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_ARTIFACT_HASH_INVALID")

    sampling_ref = str(value["sampling_plan_ref"])
    sweep_ref = str(value["development_sweep_plan_ref"])
    if _logical(root, sampling_ref, field="input_index.sampling_plan_ref").parent != path.parent:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SAMPLING_PATH_MISMATCH")
    if _logical(root, sweep_ref, field="input_index.development_sweep_plan_ref").parent != path.parent:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SWEEP_PATH_MISMATCH")
    try:
        sampling_value = load_canonical_json(_logical(root, sampling_ref, field="sampling_plan_ref"))
        sweep_value = load_canonical_json(_logical(root, sweep_ref, field="development_sweep_plan_ref"))
        if not isinstance(sampling_value, Mapping) or not isinstance(sweep_value, Mapping):
            raise TypeError("sampling and sweep objects required")
        sampling = SamplingPlan.from_mapping(sampling_value)
        validate_s205_development_sweep(sweep_value, sampling=sampling)
    except (OSError, TypeError, ValueError, RuntimeError, S205InputBlocked) as error:
        raise S25ExecutionBlocked(f"S205_INPUT_INDEX_PAYLOAD_INVALID:{error}") from error
    if value.get("sampling_plan_hash") != sampling.digest:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SAMPLING_HASH_MISMATCH")
    if value.get("development_sweep_plan_hash") != sweep_value.get("artifact_hash"):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SWEEP_HASH_MISMATCH")

    source_refs = sweep_value.get("source_artifact_refs")
    required_sources = {
        str(value["preregistration_ref"]),
        str(value["sampling_plan_task_ref"]),
        str(value["formal_execution_ref"]),
    }
    if not isinstance(source_refs, list) or set(source_refs) != required_sources:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SOURCE_CLOSURE_INVALID")
    try:
        prereg = load_committed_task_artifact(
            root, str(value["preregistration_ref"]), require_formal=True
        )
        sampling_commit = load_committed_task_artifact(
            root, str(value["sampling_plan_task_ref"]), require_formal=True
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SOURCE_ARTIFACT_INVALID") from error
    if (
        prereg.identity.task_id != "stage2.01_scope_hypotheses_and_preregistration"
        or prereg.identity.artifact_kind != "preregistration"
        or prereg.identity.commit_ref != str(value["preregistration_ref"])
        or prereg.identity.artifact_hash != sweep_value.get("preregistration_artifact_hash")
        or canonical_json_hash(dict(prereg.payload)) != sweep_value.get("preregistration_payload_hash")
        or sampling_commit.identity.task_id != "stage2.03_assets_checkpoints_and_sampling"
        or sampling_commit.identity.artifact_kind != "sampling_plan"
        or sampling_commit.identity.commit_ref != str(value["sampling_plan_task_ref"])
        or dict(sampling_commit.payload) != dict(sampling_value)
    ):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_SOURCE_ARTIFACT_BINDING_INVALID")
    try:
        execution_value = load_canonical_json(_logical(root, str(value["formal_execution_ref"]), field="formal_execution_ref"))
        execution = FormalExecutionEvidence.from_mapping(execution_value)
        execution.require_for_stage(2)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_FORMAL_EXECUTION_INVALID") from error
    if execution.artifact_hash != sweep_value.get("execution_evidence_hash"):
        raise S25ExecutionBlocked("S205_INPUT_INDEX_FORMAL_EXECUTION_HASH_MISMATCH")
    return dict(value)


def _validate_s204_input_refs(
    root: Path,
    rebind: Mapping[str, object],
    index: Mapping[str, object],
) -> None:
    """Require every S2.4 config to carry the exact formal predecessor set."""

    expected_sampling_ref = str(index["sampling_plan_task_ref"])
    required = {
        "stage2.02_reference": {"handoff_manifest", "fixed_state_contract", "gate_record"},
        "stage2.03_reference": {"sampling_plan", "draw_manifest", "asset_resolution", "gate_record"},
    }
    common_refs: tuple[str, ...] | None = None
    rows = rebind.get("cells")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CELL_IDS):
        raise S25ExecutionBlocked("S205_S204_INPUT_REF_ROWS_INVALID")
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise S25ExecutionBlocked("S205_S204_INPUT_REF_ROW_INVALID")
        config_path = _logical(root, str(raw.get("config_ref")), field="s204.config_ref")
        try:
            config = load_canonical_json(config_path)
        except (OSError, TypeError, ValueError) as error:
            raise S25ExecutionBlocked("S205_S204_CONFIG_JSON_INVALID") from error
        if not isinstance(config, Mapping):
            raise S25ExecutionBlocked("S205_S204_CONFIG_OBJECT_REQUIRED")
        orchestration = config.get("orchestration")
        refs = orchestration.get("input_result_refs") if isinstance(orchestration, Mapping) else None
        if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
            raise S25ExecutionBlocked("S205_S204_INPUT_RESULT_REFS_INVALID")
        normalized_refs = tuple(str(ref) for ref in refs)
        if expected_sampling_ref not in normalized_refs:
            raise S25ExecutionBlocked("S205_S204_SAMPLING_REF_NOT_BOUND")
        kinds: dict[str, set[str]] = {key: set() for key in required}
        for ref in normalized_refs:
            try:
                loaded = load_committed_task_artifact(
                    root, _logical(root, ref, field="s204.input_result_ref").relative_to(root).as_posix(), require_formal=True
                )
            except (OSError, TypeError, ValueError, RuntimeError) as error:
                raise S25ExecutionBlocked("S205_S204_INPUT_ARTIFACT_INVALID") from error
            if loaded.identity.task_id == "stage2.02_stage1_handoff_and_fixed_state_contract":
                kinds["stage2.02_reference"].add(loaded.identity.artifact_kind)
            elif loaded.identity.task_id == "stage2.03_assets_checkpoints_and_sampling":
                kinds["stage2.03_reference"].add(loaded.identity.artifact_kind)
        if any(not expected.issubset(kinds[name]) for name, expected in required.items()):
            raise S25ExecutionBlocked("S205_S204_INPUT_ARTIFACT_SET_INCOMPLETE")
        if common_refs is None:
            common_refs = normalized_refs
        elif common_refs != normalized_refs:
            raise S25ExecutionBlocked("S205_S204_INPUT_RESULT_REFS_DRIFT")


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    index_ref = getattr(args, "input_index_ref", None)
    if not index_ref:
        raise S25ExecutionBlocked("S205_INPUT_INDEX_REF_REQUIRED")
    if index_ref:
        index = _load_materialization_index(args.data_root.resolve(), index_ref)
        for field, index_field in (
            ("sampling_plan_ref", "sampling_plan_ref"),
            ("experiment_plan_ref", "development_sweep_plan_ref"),
        ):
            supplied = getattr(args, field, None)
            if supplied is not None and supplied != index[index_field]:
                raise S25ExecutionBlocked(f"S205_INPUT_INDEX_{field.upper()}_MISMATCH")
        args.sampling_plan_ref = str(index["sampling_plan_ref"])
        args.experiment_plan_ref = str(index["development_sweep_plan_ref"])
    if not args.sampling_plan_ref or not args.experiment_plan_ref or not args.artifact_root:
        raise S25ExecutionBlocked("S205_FORMAL_PLAN_REFS_REQUIRED")
    root = args.data_root.resolve()
    _require_launcher_in_repository(args.repository)
    executor_commit = _repository_commit(args.repository)
    supplied_execution_commit = getattr(args, "execution_commit", None)
    if supplied_execution_commit is not None and supplied_execution_commit != executor_commit:
        raise S25ExecutionBlocked("S205_EXECUTION_COMMIT_MISMATCH")
    launcher_source_sha256 = _launcher_source_sha256()
    supplied_launcher_source = getattr(args, "launcher_source_sha256", None)
    if supplied_launcher_source is not None and supplied_launcher_source != launcher_source_sha256:
        raise S25ExecutionBlocked("S205_LAUNCHER_SOURCE_SHA256_MISMATCH")
    plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
    _validate_s204_input_refs(root, plan, index)
    if args.artifact_root != plan.get("s205_output_root"):
        raise S25ExecutionBlocked("S205_ARTIFACT_ROOT_MUST_MATCH_REBIND")
    if args.operations_root != plan.get("operations_root"):
        raise S25ExecutionBlocked("S205_OPERATIONS_ROOT_MUST_MATCH_REBIND")
    if args.gpu_inventory_json is None:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_JSON_REQUIRED")
    result = preflight_s25(
        root,
        rebind_ref=args.s205_rebind_ref,
        sampling_ref=args.sampling_plan_ref,
        experiment_plan_ref=args.experiment_plan_ref,
        artifact_root=args.artifact_root,
    )
    inventory, inventory_identity = _load_inventory_snapshot(args.gpu_inventory_json, data_root=root)
    result["gpu"] = inventory
    # ``preflight_s25`` exposes the S2.4 execution commit.  Preserve it under
    # an explicit name and reserve the launcher-level execution_commit for the
    # exact clean HEAD whose source will run the six workers.
    result["s204_execution_commit"] = result.pop("execution_commit")
    result["execution_commit"] = executor_commit
    result["launcher_source_sha256"] = launcher_source_sha256
    result["gpu_inventory_ref"] = inventory_identity["artifact_ref"]
    result["gpu_inventory_artifact_hash"] = inventory_identity["artifact_hash"]
    result["gpu_inventory_source_ref"] = inventory_identity["source_ref"]
    result["gpu_inventory_source_sha256"] = inventory_identity["source_sha256"]
    result["gpu_inventory_path"] = inventory_identity["path"]
    result["gpu_inventory_identity"] = {
        "artifact_ref": inventory_identity["artifact_ref"],
        "artifact_hash": inventory_identity["artifact_hash"],
        "source_ref": inventory_identity["source_ref"],
        "source_sha256": inventory_identity["source_sha256"],
    }
    result["launcher"] = {
        "repository": str(args.repository.resolve()),
        "data_root": str(root),
        "run_id": args.run_id,
        "execution_commit": executor_commit,
        "launcher_source_sha256": launcher_source_sha256,
    }
    result["preflight_artifact_hash"] = canonical_json_hash(result)
    return result


def _worker(args: argparse.Namespace) -> dict[str, object]:
    if not args.cell_id or not args.gpu_uuid or args.cell_id not in EXPECTED_CELL_IDS:
        raise S25ExecutionBlocked("S205_WORKER_CELL_AND_GPU_REQUIRED")
    if args.gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S25ExecutionBlocked("S205_WORKER_GPU_UNAPPROVED")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != args.gpu_uuid:
        raise S25ExecutionBlocked("S205_WORKER_GPU_BINDING_MISMATCH")
    root = args.data_root.resolve()
    preflight = _preflight(args)
    plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
    from param_importance_nlp.experiments.stage2_s25_formal import (
        _load_object,
        load_s25_experiment_plan,
    )
    _, sampling_value = _load_object(root, args.sampling_plan_ref, field="sampling_plan_ref")
    from param_importance_nlp.experiments.sampling import SamplingPlan

    sampling = SamplingPlan.from_mapping(sampling_value)
    experiment = load_s25_experiment_plan(root, args.experiment_plan_ref)
    runner = S25FormalRunner(
        data_root=root,
        rebind_plan=plan,
        experiment_plan=experiment,
        sampling_plan=sampling,
        artifact_root=_logical(root, args.artifact_root, field="artifact_root", allow_missing=True),
    )
    result = runner.run_cell(args.cell_id)
    return {
        "status": "CELL_COMPLETE",
        "cell_id": args.cell_id,
        "gpu_uuid": args.gpu_uuid,
        "execution_commit": preflight["execution_commit"],
        "launcher_source_sha256": preflight["launcher_source_sha256"],
        "preflight": preflight,
        "cell": result,
    }


def _execute(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
    root = args.data_root.resolve()
    operations = _logical(root, args.operations_root, field="operations_root", allow_missing=True)
    operations.mkdir(parents=True, exist_ok=True)
    status_path = operations / "status.json"
    resume = bool(getattr(args, "resume", False))
    previous: Mapping[str, object] | None = None
    if status_path.exists():
        try:
            loaded = load_canonical_json(status_path)
        except (OSError, TypeError, ValueError) as error:
            raise S25ExecutionBlocked("S205_STATUS_JSON_INVALID") from error
        if not isinstance(loaded, Mapping) or loaded.get("artifact_hash") != canonical_json_hash(
            {key: item for key, item in loaded.items() if key != "artifact_hash"}
        ):
            raise S25ExecutionBlocked("S205_STATUS_ARTIFACT_HASH_INVALID")
        previous = loaded
        if loaded.get("run_id") != args.run_id:
            raise S25ExecutionBlocked("S205_STATUS_RUN_ID_MISMATCH")
        # Terminal status is still bound to the exact clean executor source;
        # never let a later HEAD substitution or inventory drift masquerade as
        # a successful resumable run.
        if loaded.get("preflight") != preflight:
            raise S25ExecutionBlocked("S205_STATUS_PREFLIGHT_IDENTITY_MISMATCH")
        if loaded.get("execution_commit") != preflight.get("execution_commit"):
            raise S25ExecutionBlocked("S205_STATUS_EXECUTION_COMMIT_MISMATCH")
        if loaded.get("launcher_source_sha256") != preflight.get("launcher_source_sha256"):
            raise S25ExecutionBlocked("S205_STATUS_LAUNCHER_SOURCE_SHA256_MISMATCH")
        old_records = loaded.get("completed_cells")
        if not isinstance(old_records, list):
            raise S25ExecutionBlocked("S205_STATUS_COMPLETED_CELL_RECORDS_INVALID")
        for old_record in old_records:
            if not isinstance(old_record, Mapping):
                raise S25ExecutionBlocked("S205_STATUS_COMPLETED_CELL_RECORD_INVALID")
            if old_record.get("returncode") == 0 and (
                old_record.get("execution_commit") != preflight.get("execution_commit")
                or old_record.get("launcher_source_sha256") != preflight.get("launcher_source_sha256")
            ):
                raise S25ExecutionBlocked("S205_STATUS_COMPLETED_CELL_IDENTITY_MISMATCH")
        if loaded.get("stage") in {"G2.4A_PASS", "G2.4A_BLOCKED", "BLOCKED"}:
            if loaded.get("stage") == "G2.4A_PASS":
                return dict(loaded)
            raise S25ExecutionBlocked("S205_STATUS_TERMINAL_NO_RESUME")
        if not resume:
            raise S25ExecutionBlocked("S205_STATUS_EXISTS_USE_RESUME")
        owner_pid = loaded.get("owner_pid")
        if isinstance(owner_pid, int) and owner_pid != os.getpid():
            try:
                os.kill(owner_pid, 0)
            except OSError:
                pass
            else:
                raise S25ExecutionBlocked("S205_STATUS_OWNED_BY_OTHER_PROCESS")
    status: dict[str, object] = {
        "run_id": args.run_id,
        "stage": "PREPARED",
        "owner_pid": os.getpid(),
        "formal_eligible": True,
        "execution_commit": preflight["execution_commit"],
        "s204_execution_commit": preflight["s204_execution_commit"],
        "launcher_source_sha256": preflight["launcher_source_sha256"],
        "preflight": preflight,
        "completed_cells": list(previous.get("completed_cells", [])) if previous else [],
        "confirmatory_draws_generated": False,
    }
    _write_status(status_path, status)
    base = [
        str(args.python), str(Path(__file__).resolve()), "--worker",
        "--data-root", str(root), "--s205-rebind-ref", args.s205_rebind_ref,
        "--sampling-plan-ref", args.sampling_plan_ref, "--experiment-plan-ref", args.experiment_plan_ref,
        "--artifact-root", args.artifact_root, "--operations-root", args.operations_root,
        "--run-id", args.run_id,
        "--repository", str(args.repository.resolve()),
        "--execution-commit", str(preflight["execution_commit"]),
        "--launcher-source-sha256", str(preflight["launcher_source_sha256"]),
    ]
    if getattr(args, "input_index_ref", None):
        base += ["--input-index-ref", args.input_index_ref]
    if args.gpu_inventory_json is not None:
        base += ["--gpu-inventory-json", str(args.gpu_inventory_json.resolve())]
    records: list[dict[str, object]] = [
        dict(item) for item in (previous.get("completed_cells", []) if previous else [])
        if isinstance(item, Mapping)
    ]
    queue = _S205DynamicLPTQueue(EXPECTED_CELL_IDS, APPROVED_GPU_UUIDS)
    if previous:
        completed_ids = {
            str(item.get("cell_id"))
            for item in records
            if item.get("returncode") == 0 and isinstance(item.get("cell_id"), str)
        }
        if not completed_ids.issubset(set(EXPECTED_CELL_IDS)):
            raise S25ExecutionBlocked("S205_STATUS_COMPLETED_CELL_SET_INVALID")
        for cell_id in EXPECTED_CELL_IDS:
            if cell_id in completed_ids:
                queue.skip(cell_id)
    try:
        with ThreadPoolExecutor(max_workers=len(APPROVED_GPU_UUIDS)) as pool:
            futures = {}

            def submit(assignments: Sequence[tuple[str, str]]) -> None:
                for cell_id, gpu_uuid in assignments:
                    command = [*base, "--cell-id", cell_id, "--gpu-uuid", gpu_uuid]
                    log = operations / "workers" / f"{cell_id.replace(':', '__')}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    handle = log.open("a", encoding="utf-8")
                    future = pool.submit(
                        subprocess.run,
                        command,
                        cwd=args.repository.resolve(),
                        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu_uuid, "NVIDIA_VISIBLE_DEVICES": gpu_uuid, "PYTHONPATH": str(args.repository.resolve() / "src")},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    futures[future] = (cell_id, gpu_uuid, log, handle)

            submit(queue.fill())
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: futures[item][0]):
                    cell_id, gpu_uuid, log, handle = futures.pop(future)
                    completed = future.result()
                    handle.close()
                    record = {
                        "cell_id": cell_id,
                        "gpu_uuid": gpu_uuid,
                        "returncode": completed.returncode,
                        "log_ref": str(log.relative_to(root)),
                        "execution_commit": preflight["execution_commit"],
                        "launcher_source_sha256": preflight["launcher_source_sha256"],
                    }
                    records.append(record)
                    status.update({"stage": "RUNNING", "completed_cells": list(records)})
                    _write_status(status_path, status)
                    if completed.returncode != 0:
                        raise S25ExecutionBlocked(f"S205_CELL_WORKER_FAILED:{cell_id}:{completed.returncode}")
                    # Refill immediately on each individual completion; there is
                    # intentionally no second wave/barrier.
                    submit(queue.complete(cell_id))
        if queue.pending or queue.active or set(queue.completed) != set(EXPECTED_CELL_IDS):
            raise S25ExecutionBlocked("S205_QUEUE_DID_NOT_COMPLETE_ALL_CELLS")
        from param_importance_nlp.experiments.stage2_s25_formal import _load_object
        _, sampling_value = _load_object(root, args.sampling_plan_ref, field="sampling_plan_ref")
        from param_importance_nlp.experiments.sampling import SamplingPlan

        plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
        runner = S25FormalRunner(
            data_root=root,
            rebind_plan=plan,
            experiment_plan=load_s25_experiment_plan(root, args.experiment_plan_ref),
            sampling_plan=SamplingPlan.from_mapping(sampling_value),
            artifact_root=_logical(root, args.artifact_root, field="artifact_root", allow_missing=True),
        )
        gate = runner.run_all()
        status.update({"stage": "G2.4A_PASS" if gate.get("status") == "PASS" else "G2.4A_BLOCKED", "formal_eligible": gate.get("formal_eligible"), "completed_cells": list(records), "gate_ref": str((_logical(root, args.artifact_root, field="artifact_root", allow_missing=True) / "g2.4a-evaluation.json").relative_to(root)), "gate_hash": gate.get("artifact_hash")})
        _write_status(status_path, status)
        return status
    except Exception as error:
        status.update({"stage": "BLOCKED", "formal_eligible": False, "completed_cells": list(records), "reason": f"{type(error).__name__}:{error}"})
        _write_status(status_path, status)
        raise


def _status(args: argparse.Namespace, *, wait: bool) -> int:
    path = _logical(args.data_root.resolve(), f"{args.operations_root}/status.json", field="status")
    deadline = None if args.timeout_seconds is None else time.monotonic() + float(args.timeout_seconds)
    while True:
        if path.exists():
            value = load_canonical_json(path)
            if not isinstance(value, Mapping) or value.get("artifact_hash") != canonical_json_hash(
                {key: item for key, item in value.items() if key != "artifact_hash"}
            ):
                raise S25ExecutionBlocked("S205_STATUS_ARTIFACT_HASH_INVALID")
            preflight = value.get("preflight")
            if not isinstance(preflight, Mapping):
                raise S25ExecutionBlocked("S205_STATUS_PREFLIGHT_MISSING")
            execution_commit = value.get("execution_commit")
            launcher_source_sha256 = value.get("launcher_source_sha256")
            if execution_commit != preflight.get("execution_commit"):
                raise S25ExecutionBlocked("S205_STATUS_EXECUTION_COMMIT_MISMATCH")
            if value.get("s204_execution_commit") != preflight.get("s204_execution_commit"):
                raise S25ExecutionBlocked("S205_STATUS_S204_EXECUTION_COMMIT_MISMATCH")
            if launcher_source_sha256 != preflight.get("launcher_source_sha256"):
                raise S25ExecutionBlocked("S205_STATUS_LAUNCHER_SOURCE_SHA256_MISMATCH")
            if execution_commit != _repository_commit(args.repository):
                raise S25ExecutionBlocked("S205_STATUS_EXECUTION_COMMIT_DRIFT")
            if launcher_source_sha256 != _launcher_source_sha256():
                raise S25ExecutionBlocked("S205_STATUS_LAUNCHER_SOURCE_SHA256_DRIFT")
            launcher = preflight.get("launcher")
            if (
                not isinstance(launcher, Mapping)
                or launcher.get("repository") != str(args.repository.resolve())
                or launcher.get("data_root") != str(args.data_root.resolve())
                or launcher.get("run_id") != value.get("run_id")
                or launcher.get("execution_commit") != execution_commit
                or launcher.get("launcher_source_sha256") != launcher_source_sha256
            ):
                raise S25ExecutionBlocked("S205_STATUS_LAUNCHER_IDENTITY_MISMATCH")
            declared_preflight_hash = preflight.get("preflight_artifact_hash")
            if not isinstance(declared_preflight_hash, str) or declared_preflight_hash != canonical_json_hash(
                {key: item for key, item in preflight.items() if key != "preflight_artifact_hash"}
            ):
                raise S25ExecutionBlocked("S205_STATUS_PREFLIGHT_ARTIFACT_HASH_INVALID")
            inventory_ref = preflight.get("gpu_inventory_ref")
            inventory_hash = preflight.get("gpu_inventory_artifact_hash")
            inventory_source_ref = preflight.get("gpu_inventory_source_ref")
            inventory_source_sha = preflight.get("gpu_inventory_source_sha256")
            if (
                not isinstance(inventory_ref, str)
                or not isinstance(inventory_hash, str)
                or not isinstance(inventory_source_ref, str)
                or not isinstance(inventory_source_sha, str)
            ):
                raise S25ExecutionBlocked("S205_STATUS_GPU_IDENTITY_MISSING")
            if preflight.get("gpu_inventory_identity") != {
                "artifact_ref": inventory_ref,
                "artifact_hash": inventory_hash,
                "source_ref": inventory_source_ref,
                "source_sha256": inventory_source_sha,
            }:
                raise S25ExecutionBlocked("S205_STATUS_GPU_IDENTITY_MISMATCH")
            inventory_path = _logical(args.data_root.resolve(), inventory_ref, field="gpu_inventory_ref")
            _rows, inventory_identity = _load_inventory_snapshot(inventory_path, data_root=args.data_root.resolve())
            if (
                inventory_identity.get("artifact_ref") != inventory_ref
                or inventory_identity.get("artifact_hash") != inventory_hash
                or inventory_identity.get("source_ref") != inventory_source_ref
                or inventory_identity.get("source_sha256") != inventory_source_sha
            ):
                raise S25ExecutionBlocked("S205_STATUS_GPU_IDENTITY_DRIFT")
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            if not wait or (isinstance(value, Mapping) and value.get("stage") in {"G2.4A_PASS", "G2.4A_BLOCKED", "BLOCKED"}):
                return 0 if isinstance(value, Mapping) and value.get("stage") == "G2.4A_PASS" else 3
        if not wait or (deadline is not None and time.monotonic() >= deadline):
            return 4
        time.sleep(max(0.1, float(args.poll_seconds)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.5/G2.4a production launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true")
    action.add_argument("--gate-only", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    parser.add_argument("--detach", action="store_true", help="detach --execute and return a PID")
    parser.add_argument("--detached-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="resume a non-terminal status/run root")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s205-rebind-ref", required=True)
    parser.add_argument("--sampling-plan-ref")
    parser.add_argument("--experiment-plan-ref")
    parser.add_argument("--input-index-ref", help="producer-owned materialized S2.5 input index")
    parser.add_argument("--artifact-root")
    parser.add_argument("--operations-root", required=True)
    parser.add_argument("--run-id", default="s205-formal-g24a")
    parser.add_argument("--gpu-inventory-json", type=Path)
    parser.add_argument("--repository", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--execution-commit")
    parser.add_argument("--launcher-source-sha256")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cell-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def _detach(args: argparse.Namespace, raw_argv: Sequence[str] | None = None) -> int:
    if not args.execute:
        raise S25ExecutionBlocked("S205_DETACH_REQUIRES_EXECUTE")
    preflight = _preflight(args)
    root = args.data_root.resolve()
    operations = _logical(root, args.operations_root, field="operations_root", allow_missing=True)
    operations.mkdir(parents=True, exist_ok=True)
    # Receipts are append-only per launch attempt.  A stale receipt from a
    # crashed launcher must not make ``--execute --detach --resume``
    # permanently unusable, while a live PID still blocks a duplicate run.
    receipt_candidates = [operations / "launcher.pid.json"]
    attempts = operations / "attempts"
    if attempts.is_symlink():
        raise S25ExecutionBlocked("S205_DETACHED_PID_INVALID")
    if attempts.exists():
        receipt_candidates.extend(attempts.glob("*/launcher.pid.json"))
    for existing_path in receipt_candidates:
        if existing_path.is_symlink():
            raise S25ExecutionBlocked("S205_DETACHED_PID_INVALID")
        if not existing_path.exists():
            continue
        if not existing_path.is_file():
            raise S25ExecutionBlocked("S205_DETACHED_PID_INVALID")
        try:
            existing = load_canonical_json(existing_path)
        except (OSError, TypeError, ValueError) as error:
            raise S25ExecutionBlocked("S205_DETACHED_PID_INVALID") from error
        if not isinstance(existing, Mapping) or existing.get("artifact_hash") != canonical_json_hash(
            {key: item for key, item in existing.items() if key != "artifact_hash"}
        ) or not isinstance(existing.get("pid"), int):
            raise S25ExecutionBlocked("S205_DETACHED_PID_INVALID")
        try:
            os.kill(int(existing["pid"]), 0)
        except OSError:
            continue
        raise S25ExecutionBlocked(f"S205_DETACHED_LAUNCH_ALREADY_RUNNING:{existing['pid']}")

    attempts.mkdir(parents=True, exist_ok=True)
    safe_run_id = re.sub(r"[^A-Za-z0-9._-]", "_", str(args.run_id))[:80] or "run"
    attempt_id = f"{safe_run_id}-{time.time_ns()}-{os.getpid()}"
    attempt_root = attempts / attempt_id
    try:
        attempt_root.mkdir()
    except FileExistsError as error:
        raise S25ExecutionBlocked("S205_DETACHED_ATTEMPT_COLLISION") from error
    pid_path = attempt_root / "launcher.pid.json"
    log_path = attempt_root / "launcher.log"
    child = list(sys.argv[1:] if raw_argv is None else raw_argv)
    try:
        child[child.index("--detach")] = "--execute"
    except ValueError as error:
        raise S25ExecutionBlocked("S205_DETACH_ACTION_NOT_FOUND") from error
    if "--launcher-source-sha256" in child:
        source_index = child.index("--launcher-source-sha256")
        if source_index + 1 >= len(child):
            raise S25ExecutionBlocked("S205_LAUNCHER_SOURCE_SHA256_REQUIRED")
        child[source_index + 1] = str(preflight["launcher_source_sha256"])
    else:
        child += ["--launcher-source-sha256", str(preflight["launcher_source_sha256"])]
    if "--execution-commit" in child:
        commit_index = child.index("--execution-commit")
        if commit_index + 1 >= len(child):
            raise S25ExecutionBlocked("S205_EXECUTION_COMMIT_REQUIRED")
        child[commit_index + 1] = str(preflight["execution_commit"])
    else:
        child += ["--execution-commit", str(preflight["execution_commit"])]
    if "--detached-child" not in child:
        child.append("--detached-child")
    if "--repository" in child:
        repository_index = child.index("--repository")
        if repository_index + 1 >= len(child):
            raise S25ExecutionBlocked("S205_REPOSITORY_REQUIRED")
        child[repository_index + 1] = str(args.repository.resolve())
    else:
        child += ["--repository", str(args.repository.resolve())]
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [str(args.python), str(Path(__file__).resolve()), *child],
            cwd=args.repository.resolve(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload: dict[str, object] = {
        "schema_version": "stage2-s205-detached-launch-v1",
        "pid": int(process.pid),
        "attempt_id": attempt_id,
        "run_id": args.run_id,
        "operations_root": args.operations_root,
        "attempt_root": str(attempt_root.relative_to(root)),
        "log_ref": str(log_path.relative_to(root)),
        "status_ref": str((operations / "status.json").relative_to(root)),
        "preflight_artifact_hash": preflight["preflight_artifact_hash"],
        "execution_commit": preflight["execution_commit"],
        "launcher_source_sha256": preflight["launcher_source_sha256"],
        "gpu_inventory_identity": preflight["gpu_inventory_identity"],
        "confirmatory_draws_generated": False,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _write_once(pid_path, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.detach:
            return _detach(args, argv)
        if args.resume and not args.execute:
            raise S25ExecutionBlocked("S205_RESUME_REQUIRES_EXECUTE")
        if args.resume and not args.detach and not args.detached_child:
            raise S25ExecutionBlocked("S205_RESUME_REQUIRES_DETACH")
        if args.status:
            return _status(args, wait=False)
        if args.wait:
            return _status(args, wait=True)
        if args.gate_only:
            payload = _gate_only(args)
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            return 0 if payload.get("status") == "PASS" else 3
        if args.preflight:
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.worker:
            print(json.dumps(_worker(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        print(json.dumps(_execute(args), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (S25ExecutionBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.5/G2.4a blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
