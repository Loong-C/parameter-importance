"""Detached, resumable Stage 3 production orchestrator.

This module is intentionally a thin control-plane boundary.  It never creates a
model endpoint, probe, or numerical result itself.  Real task commands are
provided by a hash-bound manifest and are executed with ``shell=False``.  The
orchestrator only admits committed formal inputs, publishes independent
GateRecords for S3.2--S3.6, and records every unit/attempt transition in an
atomic ledger.

The file is standalone so it can be copied with the source tree and run on the
lab host after the source commit has been verified.  It does not probe or write
the server during import, and ``--dry-validate`` performs no subprocess launch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import errno
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable


STAGE3_ORCHESTRATOR_SCHEMA = "stage3-production-orchestrator-v1"
LEDGER_SCHEMA = "stage3-unit-ledger-v1"
HEALTH_SCHEMA = "stage3-gpu-health-v1"
CONTROL_TASK_ID = "stage3.formal_control_plane"

APPROVED_GPU_UUIDS = frozenset(
    {
        "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
        "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
        "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
        "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    }
)
EXCLUDED_GPU_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
EXCLUDED_PCI_BUS_ID = "0000:50:00.0"
EXPECTED_STAGE2_RUN_ID = "pythia-grid-20260826T145530Z"
EXPECTED_STAGE2_ESTIMATOR = "U"
EXPECTED_STAGE2_BATCH_SIZE = 32
EXPECTED_STAGE2_DATA_VARIANT = "Raw"

PILOT_ENDPOINTS = 6
FORMAL_ENDPOINTS = 33
PILOT_PROBES = 2
FORMAL_PROBES = 3
PILOT_UNITS = 12
FORMAL_UNITS = 99
STAGES = ("early", "middle", "late")

TASK_ORDER = (
    "stage3.01_prerequisites_and_scope",
    "stage3.02_math_and_metric_contract",
    "stage3.03_endpoint_and_probe_pipeline",
    "stage3.04_quadrature_engine_and_unit_tests",
    "stage3.05_reference_integral_and_precision",
    "stage3.06_pilot_and_threshold_freeze",
    "stage3.07_formal_experiment_matrix",
    "stage3.08_error_analysis_and_stability",
    "stage3.09_cost_and_method_selection",
    "stage3.10_reports_visualizations_and_handoff",
)
PILOT_TASK_ORDER = TASK_ORDER[:6]
FORMAL_TASK_ORDER = TASK_ORDER[6:]
LOCAL_GATE_BY_TASK = {
    "stage3.02_math_and_metric_contract": "stage3.G3-1",
    "stage3.03_endpoint_and_probe_pipeline": "stage3.G3-2",
    "stage3.04_quadrature_engine_and_unit_tests": "stage3.G3-3",
    "stage3.05_reference_integral_and_precision": "stage3.G3-4",
    "stage3.06_pilot_and_threshold_freeze": "stage3.G3-5",
}
REQUIRED_GATES_BY_TASK = {
    "stage3.01_prerequisites_and_scope": ("stage3.G3-0",),
    "stage3.02_math_and_metric_contract": ("stage3.G3-0",),
    "stage3.03_endpoint_and_probe_pipeline": ("stage3.G3-1",),
    "stage3.04_quadrature_engine_and_unit_tests": ("stage3.G3-1",),
    "stage3.05_reference_integral_and_precision": ("stage3.G3-2", "stage3.G3-3"),
    "stage3.06_pilot_and_threshold_freeze": ("stage3.G3-4",),
    "stage3.07_formal_experiment_matrix": ("stage3.G3-5",),
    "stage3.08_error_analysis_and_stability": ("stage3.G3-5",),
    "stage3.09_cost_and_method_selection": ("stage3.G3-6",),
    "stage3.10_reports_visualizations_and_handoff": ("stage3.G3-7",),
}
EXTERNAL_GATE_BY_TASK = {
    "stage3.08_error_analysis_and_stability": "stage3.G3-6",
    "stage3.09_cost_and_method_selection": "stage3.G3-7",
}
TASK_ENVIRONMENT_EVIDENCE_REQUIREMENTS = {
    # The formal matrix plan and the independent G3-5 Gate are phase-level
    # authorities.  They must be carried by the task environment rather than
    # smuggled into per-unit predecessor refs.  The plan execution evidence
    # ref is intentionally optional: the plan's committed source refs may
    # already prove the required execution ancestry.
    "stage3.07_formal_experiment_matrix": frozenset(
        {
            "formal_stage3_matrix_plan",
            "gate_stage3_g3_5",
        }
    ),
}
EXPECTED_OUTPUTS = {
    "stage3.01_prerequisites_and_scope": (
        "prerequisite_report", "scope_freeze", "gate_record"
    ),
    "stage3.02_math_and_metric_contract": ("path_math_contract", "metric_contract"),
    "stage3.03_endpoint_and_probe_pipeline": (
        "path_spec", "probe_manifest", "state_restoration_report"
    ),
    "stage3.04_quadrature_engine_and_unit_tests": (
        "quadrature_rules", "analytic_validation_report"
    ),
    "stage3.05_reference_integral_and_precision": (
        "path_integral_reference", "precision_budget"
    ),
    "stage3.06_pilot_and_threshold_freeze": (
        "quadrature_pilot_report", "threshold_freeze"
    ),
    "stage3.07_formal_experiment_matrix": (
        "formal_path_results", "completeness_report"
    ),
    "stage3.08_error_analysis_and_stability": (
        "path_error_table", "stability_report", "frozen_source_table"
    ),
    "stage3.09_cost_and_method_selection": (
        "cost_accuracy_table", "quadrature_decision"
    ),
    "stage3.10_reports_visualizations_and_handoff": (
        "analysis_report", "chart_artifacts", "handoff_manifest", "gate_summary"
    ),
}


class Stage3OrchestratorError(RuntimeError):
    """A fail-closed manifest, authority, resource, or recovery error."""


def _fail(code: str, detail: object | None = None) -> Stage3OrchestratorError:
    return Stage3OrchestratorError(code if detail is None else f"{code}:{detail}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise _fail("HASH_INVALID", field)
    return value


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise _fail("JSON_NOT_CANONICAL", type(error).__name__) from error


def _canonical_hash(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _fail("JSON_LOAD_FAILED", path) from error
    if not isinstance(value, Mapping):
        raise _fail("JSON_OBJECT_REQUIRED", path)
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write canonical state through a task-local sibling then atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_bytes(value))
        os.replace(temporary, path)
    except OSError as error:
        raise _fail("ATOMIC_WRITE_FAILED", path) from error
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _safe_rel(value: object, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail("LOGICAL_REF_INVALID", field)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _fail("LOGICAL_REF_ESCAPE", field)
    return path


def _resolve_ref(value: object, *, roots: Sequence[Path], field: str) -> Path:
    if not isinstance(value, str):
        raise _fail("REFERENCE_INVALID", field)
    if Path(value).is_absolute():
        candidate = Path(value).resolve()
        if not any(_within(candidate, root) for root in roots):
            raise _fail("REFERENCE_OUTSIDE_ALLOWED_ROOT", field)
        return candidate
    relative = _safe_rel(value, field)
    candidates = list(
        dict.fromkeys(root.joinpath(*relative.parts).resolve() for root in roots)
    )
    existing = [candidate for candidate in candidates if candidate.exists()]
    if len(existing) > 1:
        raise _fail("REFERENCE_AMBIGUOUS", field)
    if existing:
        return existing[0]
    # Return the first safe candidate so callers can report the precise missing ref.
    return candidates[0]


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _logical_from(path: Path, root: Path, field: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise _fail("REFERENCE_OUTSIDE_WORKSPACE", field) from error


def _forbidden_ref(value: object, field: str) -> None:
    if not isinstance(value, str):
        return
    lowered = value.casefold()
    if "fixture" in lowered or "synthetic" in lowered:
        raise _fail("FORMAL_FIXTURE_OR_SYNTHETIC_REF", field)


def _validate_task_environment_evidence(
    task_id: str,
    evidence_refs: Mapping[str, str],
) -> None:
    """Require phase authorities before a formal task can be launched."""

    required = TASK_ENVIRONMENT_EVIDENCE_REQUIREMENTS.get(task_id)
    if required is None:
        return
    missing = sorted(key for key in required if key not in evidence_refs)
    if missing:
        raise _fail(
            "TASK_ENVIRONMENT_EVIDENCE_MISSING",
            f"{task_id}:{','.join(missing)}",
        )
    for key in required:
        value = evidence_refs.get(key)
        if not isinstance(value, str) or not value:
            raise _fail("TASK_ENVIRONMENT_EVIDENCE_INVALID", f"{task_id}:{key}")


def _walk(value: object):
    yield value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(key)
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk(child)


def _require_no_formal_false(value: Mapping[str, Any], field: str) -> None:
    if value.get("scope") == "local_fixture" or value.get("run_intent") == "local_fixture":
        raise _fail("FORMAL_OUTPUT_SCOPE_INVALID", field)
    # Numbered Stage 3 tasks intentionally publish domain candidates with
    # ``formal_eligible=false`` before an independent Gate exists.  The task
    # envelope has already been required to be a formal commit; scientific
    # eligibility is established only by the gate-specific reload below.


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        # ``os.kill(pid, 0)`` is not a reliable non-mutating existence probe on
        # Windows and has hung under the desktop test host.  Querying a limited
        # process handle is read-only and returns immediately for live PIDs.
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED => live
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM
    return True


@dataclass(frozen=True, slots=True)
class GPUHealth:
    """One immutable, externally captured GPU health snapshot."""

    snapshot_hash: str
    selected_gpu_uuid: str
    selected_pci_bus_id: str
    checked_at: str

    @classmethod
    def from_file(cls, path: Path) -> "GPUHealth":
        value = _load_json(path)
        if value.get("schema_version") != HEALTH_SCHEMA:
            raise _fail("GPU_HEALTH_SCHEMA_INVALID")
        declared = value.get("artifact_hash")
        if not isinstance(declared, str) or declared != _canonical_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        ):
            raise _fail("GPU_HEALTH_HASH_INVALID")
        selected = value.get("selected_gpu_uuid")
        pci = value.get("selected_pci_bus_id")
        devices = value.get("devices")
        if not isinstance(selected, str) or not isinstance(pci, str) or not isinstance(devices, list):
            raise _fail("GPU_HEALTH_FIELDS_INVALID")
        if selected not in APPROVED_GPU_UUIDS or selected == EXCLUDED_GPU_UUID:
            raise _fail("GPU_NOT_APPROVED", selected)
        if pci == EXCLUDED_PCI_BUS_ID:
            raise _fail("GPU_PCI_EXCLUDED", pci)
        matching = [item for item in devices if isinstance(item, Mapping) and item.get("uuid") == selected]
        if len(matching) != 1:
            raise _fail("GPU_SELECTED_DEVICE_NOT_UNIQUE")
        device = matching[0]
        if device.get("pci_bus_id") != pci or device.get("health") != "PASS":
            raise _fail("GPU_SELECTED_DEVICE_NOT_HEALTHY")
        if device.get("active_processes") not in ([], 0, None):
            raise _fail("GPU_SELECTED_DEVICE_BUSY")
        if device.get("uncorrected_ecc", 0) not in (0, None):
            raise _fail("GPU_SELECTED_DEVICE_ECC")
        return cls(
            snapshot_hash=declared,
            selected_gpu_uuid=selected,
            selected_pci_bus_id=pci,
            checked_at=str(value.get("checked_at", "")),
        )


def verify_gpu_health_once(path: Path) -> GPUHealth:
    """Perform the single complete pre-launch health check for this session."""

    return GPUHealth.from_file(path)


def _stage2_values(value: object, names: set[str], out: list[object]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in names:
                out.append(child)
            _stage2_values(child, names, out)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _stage2_values(child, names, out)


def validate_stage2_identity(
    payload: Mapping[str, Any],
    *,
    run_id: str = EXPECTED_STAGE2_RUN_ID,
    estimator: str = EXPECTED_STAGE2_ESTIMATOR,
    batch_size: int = EXPECTED_STAGE2_BATCH_SIZE,
    data_variant: str = EXPECTED_STAGE2_DATA_VARIANT,
) -> None:
    """Require the real Stage2 U-32/B=32/Raw identity without relabelling it."""

    run_values: list[object] = []
    estimator_values: list[object] = []
    batch_values: list[object] = []
    variant_values: list[object] = []
    _stage2_values(payload, {"run_id", "stage2_run_id", "source_run_id"}, run_values)
    _stage2_values(payload, {"default_estimator", "selected_estimator", "estimator"}, estimator_values)
    _stage2_values(payload, {"batch_size", "reference_batch_size"}, batch_values)
    _stage2_values(payload, {"data_variant", "sampling_variant", "weighting_variant", "view"}, variant_values)
    if run_id not in run_values:
        raise _fail("STAGE2_RUN_ID_MISMATCH", run_id)
    if not any(isinstance(item, str) and item.casefold() in {estimator.casefold(), "u-32", "equal_u", "equal-u"} for item in estimator_values):
        raise _fail("STAGE2_ESTIMATOR_MISMATCH", estimator)
    if batch_size not in batch_values:
        raise _fail("STAGE2_BATCH_SIZE_MISMATCH", batch_size)
    if not any(isinstance(item, str) and item.casefold() == data_variant.casefold() for item in variant_values):
        raise _fail("STAGE2_DATA_VARIANT_MISSING", data_variant)
    for ref in (run_id,):
        _forbidden_ref(ref, "stage2.run_id")


@dataclass(frozen=True, slots=True)
class UnitRecord:
    unit_id: str
    model: str
    seed: int
    stage: str
    endpoint_hash: str
    probe_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "model": self.model,
            "seed": self.seed,
            "stage": self.stage,
            "endpoint_hash": self.endpoint_hash,
            "probe_id": self.probe_id,
        }


def load_unit_index(path: Path, *, scope: str) -> tuple[str, tuple[UnitRecord, ...]]:
    """Load and structurally revalidate the repository ProductionUnitIndex."""

    try:
        from param_importance_nlp.experiments.stage3_production_plan import load_production_unit_index
    except (ImportError, SyntaxError) as error:
        raise _fail("PRODUCTION_INDEX_API_UNAVAILABLE") from error
    try:
        index = load_production_unit_index(path, expected_scope=scope)
    except Exception as error:
        raise _fail("PRODUCTION_INDEX_INVALID", type(error).__name__) from error
    expected = PILOT_UNITS if scope == "pilot" else FORMAL_UNITS
    if index.unit_count != expected:
        raise _fail("PRODUCTION_UNIT_COUNT_INVALID", f"expected={expected},actual={index.unit_count}")
    records = tuple(
        UnitRecord(
            unit_id=unit.path_unit_id,
            model=unit.model,
            seed=unit.seed,
            stage=unit.stage,
            endpoint_hash=unit.endpoint_digest,
            probe_id=unit.probe,
        )
        for unit in index.units
    )
    if len({record.unit_id for record in records}) != len(records):
        raise _fail("PRODUCTION_UNIT_ID_DUPLICATE")
    return str(index.artifact_hash), records


class UnitLedger:
    """Atomic, append-only-attempt ledger for endpoint×probe units."""

    def __init__(self, path: Path, *, scope: str, config_hash: str, unit_index_hash: str, units: Sequence[UnitRecord]) -> None:
        self.path = path
        self.scope = scope
        self.config_hash = _hash(config_hash, "config_hash")
        self.unit_index_hash = _hash(unit_index_hash, "unit_index_hash")
        self.units = tuple(units)
        self._value = self._load_or_initialize()

    def _body(self) -> dict[str, Any]:
        return {key: value for key, value in self._value.items() if key != "ledger_hash"}

    def _load_or_initialize(self) -> dict[str, Any]:
        if not self.path.exists():
            now = _now()
            value = {
                "schema_version": LEDGER_SCHEMA,
                "scope": self.scope,
                "config_hash": self.config_hash,
                "unit_index_hash": self.unit_index_hash,
                "expected_unit_count": len(self.units),
                "units": {
                    record.unit_id: {
                        **record.to_dict(),
                        "status": "PENDING",
                        "last_attempt_id": None,
                        "last_error": None,
                    }
                    for record in self.units
                },
                "attempts": [],
                "created_at": now,
                "updated_at": now,
            }
            value["ledger_hash"] = _canonical_hash(value)
            _write_atomic(self.path, value)
            return value
        value = dict(_load_json(self.path))
        declared = value.get("ledger_hash")
        if not isinstance(declared, str) or declared != _canonical_hash({key: item for key, item in value.items() if key != "ledger_hash"}):
            raise _fail("UNIT_LEDGER_HASH_INVALID")
        if value.get("schema_version") != LEDGER_SCHEMA or value.get("scope") != self.scope or value.get("config_hash") != self.config_hash or value.get("unit_index_hash") != self.unit_index_hash or value.get("expected_unit_count") != len(self.units):
            raise _fail("UNIT_LEDGER_IDENTITY_DRIFT")
        rows = value.get("units")
        if not isinstance(rows, Mapping) or set(rows) != {record.unit_id for record in self.units}:
            raise _fail("UNIT_LEDGER_COVERAGE_DRIFT")
        return value

    @property
    def value(self) -> Mapping[str, Any]:
        return dict(self._value)

    @property
    def complete(self) -> bool:
        rows = self._value["units"]
        return all(isinstance(item, Mapping) and item.get("status") == "PASS" for item in rows.values())

    def _save(self) -> None:
        self._value["updated_at"] = _now()
        self._value["ledger_hash"] = _canonical_hash(self._body())
        _write_atomic(self.path, self._value)

    def record_attempt(self, unit_id: str, status: str, *, attempt_id: str, error: str | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        if unit_id not in self._value["units"]:
            raise _fail("UNIT_UNKNOWN", unit_id)
        if status not in {"RUNNING", "PASS", "FAIL", "PENDING"}:
            raise _fail("UNIT_STATUS_INVALID", status)
        rows = self._value["units"]
        row = dict(rows[unit_id])
        if row.get("status") == "PASS" and status != "PASS":
            raise _fail("UNIT_PASS_IMMUTABLE", unit_id)
        now = _now()
        row["status"] = status
        row["last_attempt_id"] = attempt_id
        row["last_error"] = error
        rows[unit_id] = row
        self._value["attempts"].append({
            "attempt_id": attempt_id,
            "unit_id": unit_id,
            "status": status,
            "recorded_at": now,
            "error": error,
            "metadata": {} if metadata is None else dict(metadata),
        })
        self._save()

    def reconcile(self, path: Path) -> None:
        value = _load_json(path)
        if value.get("schema_version") != "stage3-unit-status-v1":
            raise _fail("UNIT_STATUS_SCHEMA_INVALID")
        if value.get("config_hash") != self.config_hash or value.get("unit_index_hash") != self.unit_index_hash:
            raise _fail("UNIT_STATUS_IDENTITY_DRIFT")
        entries = value.get("units")
        if not isinstance(entries, list) or any(not isinstance(item, Mapping) for item in entries):
            raise _fail("UNIT_STATUS_COVERAGE_INVALID")
        unit_ids = [item.get("unit_id") for item in entries]
        expected_ids = set(self._value["units"])
        if any(not isinstance(unit_id, str) for unit_id in unit_ids) or len(unit_ids) != len(set(unit_ids)) or set(unit_ids) != expected_ids:
            raise _fail("UNIT_STATUS_COVERAGE_INVALID")
        for item in entries:
            if item.get("status") not in {"PASS", "FAIL"}:
                raise _fail("UNIT_STATUS_ENTRY_INVALID")
            self.record_attempt(str(item["unit_id"]), str(item["status"]), attempt_id=f"reconcile-{uuid.uuid4().hex}", error=(None if item["status"] == "PASS" else str(item.get("error", "unit reported failure"))), metadata={key: child for key, child in item.items() if key not in {"unit_id", "status", "error"}})


class InstanceLock:
    """PID lock; a live lock is never overwritten and stale locks are evidence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = {"pid": os.getpid(), "started_at": _now()}
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            try:
                existing = _load_json(self.path)
            except Exception as read_error:
                raise _fail("ORCHESTRATOR_LOCK_UNREADABLE") from read_error
            if _pid_alive(existing.get("pid")):
                raise _fail("ORCHESTRATOR_ALREADY_RUNNING", existing.get("pid"))
            raise _fail("ORCHESTRATOR_STALE_LOCK", self.path) from error
        try:
            os.write(fd, _canonical_bytes(content))
        finally:
            os.close(fd)
        self.owned = True

    def release(self) -> None:
        if self.owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.owned = False

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    config_ref: str
    config_hash: str
    environment_ref: str | None
    evidence_refs: Mapping[str, str]
    command: tuple[str, ...]
    output_refs: Mapping[str, str]
    output_dir: str
    result_ref: str | None = None
    unit_status_ref: str | None = None
    external_gate_ref: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TaskSpec":
        expected = {"task_id", "config_ref", "config_hash", "environment_ref", "evidence_refs", "command", "output_refs", "output_dir", "result_ref", "unit_status_ref", "external_gate_ref"}
        if set(value) != expected:
            raise _fail("TASK_SPEC_FIELDS_INVALID")
        task_id = value.get("task_id")
        if task_id not in TASK_ORDER:
            raise _fail("TASK_SPEC_TASK_INVALID", task_id)
        command = value.get("command")
        if isinstance(command, (str, bytes)) or not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
            raise _fail("TASK_SPEC_COMMAND_REQUIRED", task_id)
        outputs = value.get("output_refs")
        if not isinstance(outputs, Mapping):
            raise _fail("TASK_SPEC_OUTPUTS_REQUIRED", task_id)
        expected_outputs = set(EXPECTED_OUTPUTS[task_id])
        if set(outputs) != expected_outputs or any(not isinstance(item, str) or not item for item in outputs.values()):
            raise _fail("TASK_SPEC_OUTPUT_COVERAGE_INVALID", task_id)
        config_hash = _hash(value.get("config_hash"), f"{task_id}.config_hash")
        evidence_refs = value.get("evidence_refs")
        if not isinstance(evidence_refs, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or not item
            for key, item in evidence_refs.items()
        ):
            raise _fail("TASK_SPEC_EVIDENCE_REFS_INVALID", task_id)
        _validate_task_environment_evidence(
            str(task_id),
            {str(key): str(item) for key, item in evidence_refs.items()},
        )
        return cls(
            task_id=task_id,
            config_ref=str(value["config_ref"]),
            config_hash=config_hash,
            environment_ref=(None if value["environment_ref"] is None else str(value["environment_ref"])),
            evidence_refs={str(key): str(item) for key, item in evidence_refs.items()},
            command=tuple(command),
            output_refs={str(key): str(item) for key, item in outputs.items()},
            output_dir=str(value["output_dir"]),
            result_ref=(None if value["result_ref"] is None else str(value["result_ref"])),
            unit_status_ref=(None if value["unit_status_ref"] is None else str(value["unit_status_ref"])),
            external_gate_ref=(None if value["external_gate_ref"] is None else str(value["external_gate_ref"])),
        )


def _load_formal_commit(root: Path, ref: str, *, field: str) -> Any:
    _forbidden_ref(ref, field)
    try:
        from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except Exception as error:
        raise _fail("FORMAL_COMMIT_INVALID", field) from error
    _require_no_formal_false(dict(loaded.payload), field)
    return loaded


def _load_gate(root: Path, ref: str, expected_gate_id: str) -> Any:
    loaded = _load_formal_commit(root, ref, field="gate_ref")
    try:
        from param_importance_nlp.contracts.status import GateRecord, GateStatus
        gate = GateRecord.from_mapping(dict(loaded.payload))
    except Exception as error:
        raise _fail("GATE_RECORD_INVALID", ref) from error
    if gate.gate_id != expected_gate_id or gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
        raise _fail("GATE_NOT_LIVE_PASS", expected_gate_id)
    return loaded, gate


def _require_content_hash(payload: Mapping[str, Any], *, field: str) -> None:
    supplied = payload.get("artifact_hash")
    if supplied is None:
        return
    if supplied != _canonical_hash(
        {key: item for key, item in payload.items() if key != "artifact_hash"}
    ):
        raise _fail("DOMAIN_ARTIFACT_HASH_INVALID", field)


def _validate_local_gate_outputs(
    task_id: str,
    loaded: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Independently reload the scientific meaning of G3-1..G3-5 outputs."""

    payloads = {kind: dict(item.payload) for kind, item in loaded.items()}
    if task_id == "stage3.02_math_and_metric_contract":
        math_contract = payloads["path_math_contract"]
        metric_contract = payloads["metric_contract"]
        required_metrics = {
            "normalized_l1_error", "normalized_l2_error",
            "normalized_linf_error", "cosine_similarity", "active_spearman",
            "sign_consistency", "completeness_absolute_residual",
            "completeness_relative_residual",
            "completeness_l1_scaled_residual", "spearman", "top_q_overlap",
            "top_q_jaccard", "layer_total_variation",
            "module_total_variation", "real_gradient_evaluation_cost",
        }
        if (
            math_contract.get("schema_version")
            != "stage3-task-path-math-contract-v1"
            or math_contract.get("signed_contribution")
            != "-delta_theta*integral_0^1 gradient(theta(alpha)) d_alpha"
            or math_contract.get(
                "parameter_post_state_distinct_from_attempt_commit_state"
            )
            is not True
            or math_contract.get("quadrature_weight_dtype") != "float64"
            or metric_contract.get("schema_version")
            != "stage3-task-metric-contract-v1"
            or metric_contract.get("undefined_policy")
            != "defined_false_with_reason_no_epsilon"
            or set(metric_contract.get("metrics", ())) < required_metrics
            or metric_contract.get("strata")
            != ["model", "stage", "update", "probe"]
        ):
            raise _fail("G31_SCIENTIFIC_CONTRACT_INVALID")
        return (
            {"math_contract_hash": loaded["path_math_contract"].identity.artifact_hash},
            {"exact_path_and_metric_contract": True},
        )

    if task_id == "stage3.03_endpoint_and_probe_pipeline":
        path_spec = payloads["path_spec"]
        panel = payloads["probe_manifest"]
        restoration = payloads["state_restoration_report"]
        _require_content_hash(path_spec, field="path_spec")
        _require_content_hash(panel, field="probe_manifest")
        entries = panel.get("entries")
        sample_ids: list[object] = []
        if not isinstance(entries, list):
            raise _fail("G32_PROBE_PANEL_INVALID")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(
                entry.get("sample_ids"), list
            ):
                raise _fail("G32_PROBE_PANEL_INVALID")
            sample_ids.extend(entry["sample_ids"])
        panel_scope = panel.get("scope")
        if panel_scope not in {"pilot", "formal"}:
            raise _fail("G32_PROBE_PANEL_SCOPE_INVALID")
        expected_probe_role = "formal" if panel_scope == "formal" else "pilot"
        expected_probe_count = 3 if panel_scope == "formal" else 2
        if (
            path_spec.get("schema_version") != "path-spec-v1"
            or path_spec.get("accumulation_dtype") != "float64"
            or panel.get("schema_version") != "stage3-probe-panel-v1"
            or panel.get("formal_eligible") is not (panel_scope == "formal")
            or len(entries) != expected_probe_count
            or sum(entry.get("role") == expected_probe_role for entry in entries)
            != expected_probe_count
            or len(sample_ids) != len(set(sample_ids))
            or restoration.get("replay_verified") is not True
            or restoration.get("parameter_post_is_attempt_commit") is not False
            or restoration.get("failure_restore_boundary") != "pre_state"
            or restoration.get("scope") != panel_scope
            or restoration.get("execution_evidence_hash")
            != panel.get("execution_evidence_hash")
        ):
            raise _fail("G32_ENDPOINT_PROBE_REPLAY_INVALID")
        return (
            {"probe_count": len(entries), "replay_verified": True},
            {
                (
                    "minimum_independent_formal_probes"
                    if panel_scope == "formal"
                    else "minimum_independent_pilot_probes"
                ): expected_probe_count,
                "full_update_endpoint": True,
            },
        )

    if task_id == "stage3.04_quadrature_engine_and_unit_tests":
        registry = payloads["quadrature_rules"]
        validation = payloads["analytic_validation_report"]
        rules = registry.get("rules")
        required_rules = {
            "left", "right", "midpoint", "trapezoid", "simpson",
            "composite_trapezoid_4", "composite_trapezoid_8",
            "composite_simpson_4", "composite_simpson_8",
            "composite_simpson_16", "gauss_legendre_2",
            "gauss_legendre_4", "gauss_legendre_8",
        }
        if (
            registry.get("schema_version")
            != "stage3-task-quadrature-rules-v1"
            or not isinstance(rules, Mapping)
            or set(rules) < required_rules
            or registry.get("registry_hash") != _canonical_hash(dict(rules))
            or validation.get("passed") is not True
            or validation.get("local_validation_status") != "PASS"
            or validation.get("formal_gate_status") != "NOT_RUN"
            or any(
                row.get("finite") is not True
                for row in validation.get("polynomial_rows", ())
                if isinstance(row, Mapping)
            )
            or any(
                row.get("error_decreased") is not True
                for row in validation.get("refinement_rows", ())
                if isinstance(row, Mapping)
            )
        ):
            raise _fail("G33_ANALYTIC_QUADRATURE_VALIDATION_INVALID")
        return (
            {"registered_rule_count": len(rules), "analytic_validation": "PASS"},
            {"required_rule_registry_complete": True, "refinement_error_decreased": True},
        )

    if task_id == "stage3.05_reference_integral_and_precision":
        reference = payloads["path_integral_reference"]
        precision = payloads["precision_budget"]
        refinement = reference.get("refinement")
        if (
            reference.get("schema_version")
            != "stage3-task-path-integral-reference-v1"
            or not isinstance(refinement, Mapping)
            or refinement.get("converged") is not True
            or refinement.get("status") != "FORMAL_CANDIDATE"
            or not isinstance(reference.get("reference_aggregate_ref"), str)
            or not isinstance(reference.get("reference_aggregate_hash"), str)
            or precision.get("two_independent_rule_families") is not True
            or precision.get("continuous_refinement") is not True
            or precision.get("gradient_dtype") != "float64"
            or precision.get("path_accumulation_dtype") != "float64"
        ):
            raise _fail("G34_REFERENCE_CONVERGENCE_INVALID")
        return (
            {
                "reference_status": "CONVERGED",
                "conservative_error": refinement.get("conservative_error"),
                "selected_level": refinement.get("selected_level"),
            },
            {"two_independent_rule_families": True, "continuous_refinement": True},
        )

    if task_id == "stage3.06_pilot_and_threshold_freeze":
        pilot = payloads["quadrature_pilot_report"]
        freeze = payloads["threshold_freeze"]
        progress = pilot.get("observation_progress")
        recommendation = pilot.get("recommendation")
        thresholds = freeze.get("thresholds")
        if isinstance(thresholds, Mapping):
            try:
                from param_importance_nlp.experiments.stage3_formal import (
                    QuadratureThresholds,
                )

                QuadratureThresholds(**dict(thresholds)).require_formal_contract()
            except Exception as error:
                raise _fail("G35_FORMAL_THRESHOLD_CONTRACT_INVALID") from error
        if (
            pilot.get("schema_version")
            != "stage3-task-quadrature-pilot-report-v1"
            or not isinstance(progress, Mapping)
            or progress.get("required_unit_count") != PILOT_UNITS
            or progress.get("observed_unit_count") != PILOT_UNITS
            or progress.get("missing_unit_ids") != []
            or not isinstance(recommendation, Mapping)
            or recommendation.get("status") != "FORMAL_CANDIDATE"
            or freeze.get("schema_version")
            != "stage3-task-threshold-freeze-v1"
            or freeze.get("formal_freeze_status") != "PENDING_GATE_REVIEW"
            or not isinstance(thresholds, Mapping)
            or freeze.get("thresholds_hash") != _canonical_hash(dict(thresholds))
        ):
            raise _fail("G35_PILOT_FREEZE_INVALID")
        return (
            {"pilot_unit_count": PILOT_UNITS, "thresholds_hash": freeze["thresholds_hash"]},
            {"exact_pilot_unit_coverage": PILOT_UNITS, "threshold_contract_frozen": True},
        )
    raise _fail("LOCAL_GATE_TASK_UNSUPPORTED", task_id)


class GateAuthorityPublisher:
    """Publish a real formal GateRecord and the next execution-evidence commit."""

    def __init__(self, workspace_root: Path, *, control_task_id: str = CONTROL_TASK_ID) -> None:
        self.workspace_root = workspace_root.resolve()
        self.control_task_id = control_task_id

    def _publish(self, output_dir: str, *, artifact_kind: str, config_hash: str, payload: Mapping[str, Any], source_refs: Sequence[str]) -> str:
        try:
            from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
            store = TaskArtifactStore(self.workspace_root, output_dir)
            published = store.publish(
                task_id=self.control_task_id,
                artifact_kind=artifact_kind,
                config_hash=config_hash,
                run_intent="formal",
                payload=dict(payload),
                formal_eligible=True,
                source_refs=tuple(dict.fromkeys(source_refs)),
            )
            return published.commit_ref
        except Exception as error:
            raise _fail("FORMAL_AUTHORITY_PUBLISH_FAILED", artifact_kind) from error

    def publish(
        self,
        *,
        output_dir: str,
        config_hash: str,
        task_id: str,
        gate_id: str,
        output_refs: Mapping[str, str],
        previous_evidence_ref: str,
        contract_freeze_hash: str,
        asset_manifest_hashes: Sequence[str],
        measured: Mapping[str, Any] | None = None,
        threshold: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, Any]:
        """Validate task outputs and publish GateRecord before evidence.

        Candidate ``gate_record`` outputs are intentionally excluded from
        authority and can never be used as a PASS.  A missing/invalid output
        produces a formal BLOCKED record, preserving failure evidence while
        preventing downstream execution.
        """

        expected = set(EXPECTED_OUTPUTS[task_id])
        if set(output_refs) != expected:
            raise _fail("AUTHORITY_OUTPUT_COVERAGE_INVALID", task_id)
        _hash(config_hash, "config_hash")
        _hash(contract_freeze_hash, "contract_freeze_hash")
        hashes = tuple(_hash(item, "asset_manifest_hash") for item in asset_manifest_hashes)
        loaded: dict[str, Any] = {}
        reasons: list[str] = []
        refs: list[str] = [previous_evidence_ref]
        for kind in expected:
            ref = output_refs[kind]
            try:
                item = _load_formal_commit(self.workspace_root, ref, field=f"output.{kind}")
                if item.identity.config_hash != config_hash or item.identity.task_id != task_id:
                    raise _fail("AUTHORITY_OUTPUT_IDENTITY_DRIFT", kind)
                loaded[kind] = item
                refs.append(ref)
                # A Gate must preserve the producer lineage of the committed
                # task outputs it independently reloads.  Keeping only the
                # outer commit refs loses the preregistered pilot plan and
                # other hash-bound inputs behind those envelopes.
                refs.extend(item.source_refs)
            except Stage3OrchestratorError as error:
                reasons.append(str(error))
        semantic_measured: Mapping[str, Any] = {}
        semantic_threshold: Mapping[str, Any] = {}
        if not reasons:
            try:
                semantic_measured, semantic_threshold = (
                    _validate_local_gate_outputs(task_id, loaded)
                )
            except Stage3OrchestratorError as error:
                reasons.append(str(error))
        gate_status = "PASS" if not reasons else "BLOCKED"
        checked_at = _now()
        try:
            from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
            from param_importance_nlp.contracts.status import GateRecord, GateStatus
            previous = _load_formal_commit(self.workspace_root, previous_evidence_ref, field="previous_evidence")
            evidence = FormalExecutionEvidence.from_mapping(dict(previous.payload))
            existing_ids = {item.gate_id for item in evidence.prerequisite_gates}
            if gate_id in existing_ids:
                raise _fail("GATE_ALREADY_PRESENT", gate_id)
            gate = GateRecord(
                gate_id=gate_id,
                stage=3,
                status=GateStatus.PASS if gate_status == "PASS" else GateStatus.BLOCKED,
                checked_at=checked_at,
                measured={
                    "task_id": task_id,
                    "expected_output_kinds": sorted(expected),
                    "validated_output_count": len(loaded),
                    **dict(semantic_measured),
                    **({} if measured is None else dict(measured)),
                },
                threshold={
                    "all_required_formal_commits": True,
                    **dict(semantic_threshold),
                    **({} if threshold is None else dict(threshold)),
                },
                evidence_refs=tuple(dict.fromkeys(refs)) if gate_status == "PASS" else tuple(refs),
                reasons=tuple(reasons),
            )
            gate_ref = self._publish(
                output_dir,
                artifact_kind="gate_record",
                config_hash=config_hash,
                payload=gate.to_dict(),
                source_refs=refs,
            )
            next_evidence = FormalExecutionEvidence(
                run_intent="formal",
                contract_freeze_hash=evidence.contract_freeze_hash or contract_freeze_hash,
                asset_manifest_hashes=evidence.asset_manifest_hashes or hashes,
                prerequisite_gates=(*evidence.prerequisite_gates, gate),
                metadata=dict(evidence.metadata),
            )
            evidence_ref = self._publish(
                f"{output_dir}/execution-evidence",
                artifact_kind="formal_execution_evidence",
                config_hash=config_hash,
                payload=next_evidence.to_dict(),
                source_refs=(*refs, gate_ref, previous_evidence_ref),
            )
        except Stage3OrchestratorError:
            raise
        except Exception as error:
            raise _fail("GATE_AUTHORITY_CONSTRUCTION_FAILED", task_id) from error
        return gate_ref, evidence_ref, gate

    def publish_external(
        self,
        *,
        output_dir: str,
        config_hash: str,
        gate_ref: str,
        previous_evidence_ref: str,
        gate_id: str,
    ) -> tuple[str, Any]:
        loaded, gate = _load_gate(self.workspace_root, gate_ref, gate_id)
        previous = _load_formal_commit(self.workspace_root, previous_evidence_ref, field="previous_evidence")
        from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence

        evidence = FormalExecutionEvidence.from_mapping(dict(previous.payload))
        if gate_id in {item.gate_id for item in evidence.prerequisite_gates}:
            raise _fail("EXTERNAL_GATE_ALREADY_PRESENT", gate_id)
        next_evidence = FormalExecutionEvidence(
            run_intent="formal",
            contract_freeze_hash=evidence.contract_freeze_hash,
            asset_manifest_hashes=evidence.asset_manifest_hashes,
            prerequisite_gates=(*evidence.prerequisite_gates, gate),
            metadata=dict(evidence.metadata),
        )
        evidence_ref = self._publish(
            f"{output_dir}/execution-evidence",
            artifact_kind="formal_execution_evidence",
            config_hash=config_hash,
            payload=next_evidence.to_dict(),
            source_refs=(previous_evidence_ref, gate_ref),
        )
        return evidence_ref, gate


class Stage3Orchestrator:
    """Validate and execute the manifest's ordered, resumable Stage 3 DAG."""

    def __init__(self, manifest: Mapping[str, Any], *, workspace_root: Path, data_root: Path, python_executable: str | None = None) -> None:
        self.manifest = dict(manifest)
        self.workspace_root = workspace_root.resolve()
        self.data_root = data_root.resolve()
        self.python_executable = python_executable or sys.executable
        self._validate_manifest_shape()
        self.scope = str(self.manifest["scope"])
        self.config_hash = _hash(self.manifest["config_hash"], "config_hash")
        self.state_dir = _resolve_ref(self.manifest["state_dir"], roots=(self.data_root,), field="state_dir")
        if not _within(self.state_dir, self.data_root):
            raise _fail("STATE_DIR_OUTSIDE_DATA_ROOT")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.task_specs = {spec.task_id: spec for spec in (TaskSpec.from_mapping(item) for item in self.manifest["tasks"])}
        self.task_order = (
            PILOT_TASK_ORDER if self.scope == "pilot" else FORMAL_TASK_ORDER
        )
        if set(self.task_specs) != set(self.task_order):
            raise _fail("TASK_DAG_COVERAGE_INVALID")
        self.unit_index_path = _resolve_ref(self.manifest["unit_index_ref"], roots=(self.data_root, self.workspace_root), field="unit_index_ref")
        unit_hash, units = load_unit_index(self.unit_index_path, scope=self.scope)
        self.unit_index_hash = unit_hash
        self.units = units
        self.ledger = UnitLedger(self.state_dir / "unit-ledger.json", scope=self.scope, config_hash=self.config_hash, unit_index_hash=unit_hash, units=units)
        self.gate_authority = GateAuthorityPublisher(self.data_root)

    def _validate_manifest_shape(self) -> None:
        expected = {"schema_version", "scope", "config_hash", "state_dir", "unit_index_ref", "stage2_authority_ref", "initial_execution_evidence_ref", "initial_execution_config_hash", "initial_environment_ref", "g30_gate_ref", "health_snapshot_ref", "scope_decision_ref", "contract_freeze_hash", "asset_manifest_hashes", "tasks"}
        if set(self.manifest) != expected or self.manifest.get("schema_version") != STAGE3_ORCHESTRATOR_SCHEMA:
            raise _fail("ORCHESTRATOR_MANIFEST_FIELDS_INVALID")
        if self.manifest.get("scope") not in {"pilot", "formal"}:
            raise _fail("ORCHESTRATOR_SCOPE_INVALID")
        if not isinstance(self.manifest.get("tasks"), list):
            raise _fail("ORCHESTRATOR_TASKS_INVALID")
        for field in ("stage2_authority_ref", "initial_execution_evidence_ref", "initial_environment_ref", "g30_gate_ref", "health_snapshot_ref", "scope_decision_ref", "unit_index_ref", "state_dir"):
            if not isinstance(self.manifest.get(field), str) or not self.manifest[field]:
                raise _fail("ORCHESTRATOR_REFERENCE_MISSING", field)
        if not isinstance(self.manifest.get("asset_manifest_hashes"), list) or not self.manifest["asset_manifest_hashes"]:
            raise _fail("ORCHESTRATOR_ASSET_MANIFESTS_REQUIRED")
        _hash(
            self.manifest.get("initial_execution_config_hash"),
            "initial_execution_config_hash",
        )

    def _roots(self) -> tuple[Path, Path]:
        return self.data_root, self.workspace_root

    def _prepare_authority(self) -> tuple[str, Any]:
        stage2_path = _resolve_ref(self.manifest["stage2_authority_ref"], roots=self._roots(), field="stage2_authority_ref")
        validate_stage2_identity(_load_json(stage2_path))
        g30_ref = str(self.manifest["g30_gate_ref"])
        _, g30 = _load_gate(self.data_root, g30_ref, "stage3.G3-0")
        evidence_ref = str(self.manifest["initial_execution_evidence_ref"])
        evidence = _load_formal_commit(self.data_root, evidence_ref, field="initial_execution_evidence")
        if evidence.identity.config_hash != self.manifest["initial_execution_config_hash"]:
            raise _fail("INITIAL_EVIDENCE_CONFIG_DRIFT")
        from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence

        execution = FormalExecutionEvidence.from_mapping(dict(evidence.payload))
        execution.require_for_stage(3)
        if not any(item.gate_id == "stage3.G3-0" and item.artifact_hash == g30.artifact_hash for item in execution.prerequisite_gates):
            raise _fail("INITIAL_EVIDENCE_G30_BINDING_MISSING")
        env_path = _resolve_ref(self.manifest["initial_environment_ref"], roots=self._roots(), field="initial_environment_ref")
        env = _load_json(env_path)
        if env.get("environment_hash") != _canonical_hash({key: item for key, item in env.items() if key != "environment_hash"}):
            raise _fail("INITIAL_ENVIRONMENT_HASH_INVALID")
        if "stage3.G3-0" not in env.get("passed_gate_ids", []):
            raise _fail("INITIAL_ENVIRONMENT_G30_MISSING")
        return evidence_ref, execution

    def _health_snapshot(self, *, probe: Callable[[Path], GPUHealth] = verify_gpu_health_once) -> GPUHealth:
        state_path = self.state_dir / "health.json"
        if state_path.exists():
            value = _load_json(state_path)
            declared = value.get("snapshot_hash")
            selected = value.get("selected_gpu_uuid")
            pci = value.get("selected_pci_bus_id")
            checked_at = value.get("checked_at")
            if not isinstance(declared, str) or not isinstance(selected, str) or not isinstance(pci, str) or not isinstance(checked_at, str):
                raise _fail("HEALTH_STATE_INVALID")
            if selected not in APPROVED_GPU_UUIDS or selected == EXCLUDED_GPU_UUID or pci == EXCLUDED_PCI_BUS_ID:
                raise _fail("HEALTH_STATE_DEVICE_INVALID")
            return GPUHealth(snapshot_hash=declared, selected_gpu_uuid=selected, selected_pci_bus_id=pci, checked_at=checked_at)
        snapshot_path = _resolve_ref(self.manifest["health_snapshot_ref"], roots=self._roots(), field="health_snapshot_ref")
        snapshot = probe(snapshot_path)
        _write_atomic(state_path, {"schema_version": "stage3-health-state-v1", "snapshot_hash": snapshot.snapshot_hash, "selected_gpu_uuid": snapshot.selected_gpu_uuid, "selected_pci_bus_id": snapshot.selected_pci_bus_id, "checked_at": snapshot.checked_at})
        return snapshot

    def _task_environment(self, spec: TaskSpec, current_ref: str) -> Path:
        return _resolve_ref(current_ref, roots=(self.data_root,), field="dynamic_environment_ref")

    def _expand_command(self, spec: TaskSpec, *, environment: Path) -> list[str]:
        config = _resolve_ref(spec.config_ref, roots=self._roots(), field=f"{spec.task_id}.config_ref")
        config_payload = _load_json(config)
        if config_payload.get("config_hash") != spec.config_hash:
            raise _fail("TASK_CONFIG_HASH_MISMATCH", spec.task_id)
        result = None if spec.result_ref is None else _resolve_ref(spec.result_ref, roots=(self.data_root,), field=f"{spec.task_id}.result_ref")
        substitutions = {
            "{python}": self.python_executable,
            "{config}": str(config),
            "{environment}": str(environment),
            "{workspace_root}": str(self.workspace_root),
            "{data_root}": str(self.data_root),
            "{state_dir}": str(self.state_dir),
            "{task_id}": spec.task_id,
            "{scope}": self.scope,
            "{unit_index}": str(self.unit_index_path),
        }
        if result is not None:
            substitutions["{result}"] = str(result)
        output: list[str] = []
        for token in spec.command:
            value = token
            for key, replacement in substitutions.items():
                value = value.replace(key, replacement)
            if "{" in value or "}" in value:
                raise _fail("TASK_COMMAND_UNKNOWN_PLACEHOLDER", spec.task_id)
            output.append(value)
        return output

    def _verify_result(self, spec: TaskSpec) -> None:
        if spec.result_ref is None:
            return
        path = _resolve_ref(spec.result_ref, roots=(self.data_root,), field=f"{spec.task_id}.result_ref")
        value = _load_json(path)
        if value.get("schema_version") == "task-run-result-v2":
            if value.get("task_id") != spec.task_id or value.get("config_hash") != spec.config_hash or value.get("status") != "PASS":
                raise _fail("TASK_RESULT_NOT_FORMAL_PASS", spec.task_id)

    def _verify_outputs(self, spec: TaskSpec) -> None:
        for kind, ref in spec.output_refs.items():
            loaded = _load_formal_commit(self.data_root, ref, field=f"{spec.task_id}.output.{kind}")
            if loaded.identity.task_id != spec.task_id or loaded.identity.artifact_kind != kind or loaded.identity.config_hash != spec.config_hash:
                raise _fail("TASK_OUTPUT_IDENTITY_INVALID", f"{spec.task_id}:{kind}")

    def _write_environment(
        self,
        evidence_ref: str,
        gate_refs: Mapping[str, str],
        *,
        task_id: str,
        task_evidence_refs: Mapping[str, str],
    ) -> str:
        """Materialize a task-runtime environment bound to the latest evidence."""

        previous = _load_formal_commit(self.data_root, evidence_ref, field="current_execution_evidence")
        from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
        execution = FormalExecutionEvidence.from_mapping(dict(previous.payload))
        passed = [item.gate_id for item in execution.prerequisite_gates if item.status.value == "PASS"]
        evidence_refs = {
            "stage3_scope_decision": str(self.manifest.get("scope_decision_ref", "stage3-scope-decision.json")),
            "stage3_g30_gate": str(self.manifest["g30_gate_ref"]),
            "formal_execution": evidence_ref,
        }
        for gate_id, gate_ref in gate_refs.items():
            evidence_refs["gate_" + gate_id.casefold().replace(".", "_").replace("-", "_")] = gate_ref
        # Preserve all pre-validated capability/contract refs from the initial
        # environment.  They are never invented by this orchestrator.
        base = _load_json(_resolve_ref(self.manifest["initial_environment_ref"], roots=self._roots(), field="initial_environment_ref"))
        for key, value in base.get("evidence_refs", {}).items():
            if key not in evidence_refs:
                evidence_refs[key] = value
        for key, value in task_evidence_refs.items():
            _forbidden_ref(value, f"{task_id}.evidence_refs.{key}")
            evidence_refs[key] = value
        _validate_task_environment_evidence(task_id, evidence_refs)
        env_payload = {
            "schema_version": "task-runtime-environment-v1",
            "capabilities": list(base.get("capabilities", [])),
            "frozen_contract_stages": list(base.get("frozen_contract_stages", [])),
            "passed_gate_ids": sorted(set(passed)),
            "estimator_decision_ref": base.get("estimator_decision_ref"),
            "evidence_refs": evidence_refs,
        }
        env_payload["environment_hash"] = _canonical_hash(env_payload)
        ref = f"{self.state_dir.relative_to(self.data_root).as_posix()}/environments/{task_id.replace('.', '_')}.json"
        path = self.data_root / Path(*PurePosixPath(ref).parts)
        _write_atomic(path, env_payload)
        return ref

    def run(self, *, dry_validate: bool = False, health_check: Callable[[Path], GPUHealth] = verify_gpu_health_once) -> Mapping[str, Any]:
        lock = InstanceLock(self.state_dir / "orchestrator.lock")
        with lock:
            if dry_validate:
                # Validate the same immutable authorities and unit counts, but
                # do not perform the health probe or spawn any process.
                current_ref, _ = self._prepare_authority()
                _ = current_ref
                return {"status": "VALIDATED", "scope": self.scope, "unit_count": len(self.units), "task_count": len(self.task_order)}
            stored = self._health_snapshot(probe=health_check)
            current_evidence_ref, _execution = self._prepare_authority()
            gate_refs: dict[str, str] = {"stage3.G3-0": str(self.manifest["g30_gate_ref"])}
            completed: list[str] = []
            for task_id in self.task_order:
                spec = self.task_specs[task_id]
                required = REQUIRED_GATES_BY_TASK[task_id]
                previous = _load_formal_commit(self.data_root, current_evidence_ref, field="current_execution_evidence")
                from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
                evidence = FormalExecutionEvidence.from_mapping(dict(previous.payload))
                live = {item.gate_id for item in evidence.prerequisite_gates if item.status.value == "PASS"}
                if not set(required).issubset(live):
                    raise _fail("DAG_GATE_PRECONDITION_MISSING", f"{task_id}:{required}")
                environment_ref = self._write_environment(
                    current_evidence_ref,
                    gate_refs,
                    task_id=task_id,
                    task_evidence_refs=spec.evidence_refs,
                )
                env_path = self._task_environment(spec, environment_ref)
                command = self._expand_command(spec, environment=env_path)
                attempt_id = f"{task_id}-{uuid.uuid4().hex}"
                # Control-plane tasks (S3.02--S3.06 and the analysis tasks)
                # do not own production endpoint×probe units.  Only a task
                # that declares a status artifact may claim pending units;
                # this prevents a failed contract task from poisoning the
                # formal unit ledger with unrelated RUNNING rows.
                touched_units: list[str] = []
                if spec.unit_status_ref is not None:
                    for unit in self.units:
                        if self.ledger.value["units"][unit.unit_id]["status"] == "PENDING":
                            self.ledger.record_attempt(unit.unit_id, "RUNNING", attempt_id=attempt_id)
                            touched_units.append(unit.unit_id)
                process = subprocess.run(command, cwd=self.workspace_root, shell=False, check=False)
                if process.returncode != 0:
                    for unit_id in touched_units:
                        self.ledger.record_attempt(unit_id, "FAIL", attempt_id=attempt_id, error=f"TASK_EXIT_{process.returncode}")
                    raise _fail("TASK_PROCESS_FAILED", f"{task_id}:{process.returncode}")
                self._verify_result(spec)
                self._verify_outputs(spec)
                if spec.unit_status_ref is not None:
                    self.ledger.reconcile(_resolve_ref(spec.unit_status_ref, roots=(self.data_root,), field=f"{task_id}.unit_status_ref"))
                if task_id in LOCAL_GATE_BY_TASK:
                    gate_ref, next_evidence_ref, gate = self.gate_authority.publish(
                        output_dir=spec.output_dir,
                        config_hash=spec.config_hash,
                        task_id=task_id,
                        gate_id=LOCAL_GATE_BY_TASK[task_id],
                        output_refs=spec.output_refs,
                        previous_evidence_ref=current_evidence_ref,
                        contract_freeze_hash=str(self.manifest["contract_freeze_hash"]),
                        asset_manifest_hashes=tuple(self.manifest["asset_manifest_hashes"]),
                    )
                    if gate.status.value != "PASS":
                        raise _fail("STAGE3_GATE_BLOCKED", gate.gate_id)
                    gate_refs[gate.gate_id] = gate_ref
                    current_evidence_ref = next_evidence_ref
                external_gate_id = EXTERNAL_GATE_BY_TASK.get(task_id)
                if external_gate_id is not None:
                    if spec.external_gate_ref is None:
                        raise _fail("EXTERNAL_GATE_REF_REQUIRED", external_gate_id)
                    next_evidence_ref, gate = self.gate_authority.publish_external(
                        output_dir=spec.output_dir,
                        config_hash=spec.config_hash,
                        gate_ref=spec.external_gate_ref,
                        previous_evidence_ref=current_evidence_ref,
                        gate_id=external_gate_id,
                    )
                    if gate.status.value != "PASS":
                        raise _fail("EXTERNAL_GATE_BLOCKED", external_gate_id)
                    gate_refs[external_gate_id] = str(spec.external_gate_ref)
                    current_evidence_ref = next_evidence_ref
                completed.append(task_id)
            if not self.ledger.complete:
                raise _fail("FORMAL_UNIT_LEDGER_INCOMPLETE")
            return {"status": "COMPLETE", "scope": self.scope, "unit_count": len(self.units), "completed_tasks": completed, "execution_evidence_ref": current_evidence_ref, "gpu_uuid": stored.selected_gpu_uuid}


def launch_detached(argv: Sequence[str], *, log_path: Path, cwd: Path) -> int:
    """Launch one detached child; caller retains the PID as the only handle."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("ab")
    kwargs: dict[str, Any] = {
        "args": list(argv),
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": handle,
        "stderr": subprocess.STDOUT,
        "shell": False,
        "close_fds": True,
        "start_new_session": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    try:
        process = subprocess.Popen(**kwargs)
    finally:
        handle.close()
    return int(process.pid)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--dry-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    if args.detach:
        log_path = args.log or (args.data_root.resolve() / "operations" / "stage3" / "orchestrator.log")
        child_args = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(manifest_path), "--workspace-root", str(args.workspace_root.resolve()), "--data-root", str(args.data_root.resolve())]
        if args.dry_validate:
            child_args.append("--dry-validate")
        pid = launch_detached(child_args, log_path=log_path.resolve(), cwd=args.workspace_root.resolve())
        print(json.dumps({"status": "DETACHED", "pid": pid, "log": str(log_path.resolve())}, ensure_ascii=False))
        return 0
    try:
        result = Stage3Orchestrator(manifest, workspace_root=args.workspace_root, data_root=args.data_root).run(dry_validate=args.dry_validate)
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVED_GPU_UUIDS",
    "EXCLUDED_GPU_UUID",
    "EXCLUDED_PCI_BUS_ID",
    "EXPECTED_STAGE2_RUN_ID",
    "GateAuthorityPublisher",
    "GPUHealth",
    "InstanceLock",
    "Stage3Orchestrator",
    "Stage3OrchestratorError",
    "UnitLedger",
    "UnitRecord",
    "TASK_ENVIRONMENT_EVIDENCE_REQUIREMENTS",
    "_validate_task_environment_evidence",
    "launch_detached",
    "load_unit_index",
    "validate_stage2_identity",
    "verify_gpu_health_once",
]
