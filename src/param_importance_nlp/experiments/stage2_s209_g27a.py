"""Fail-closed S2.9/G2.7a cost and system validation.

This module is a detached consumer.  It does not start a worker, enumerate a
GPU, or launch a CUDA wave.  A caller supplies the already sealed G2.4b
matrix, the S2.7 raw-results manifest, health evidence, and profiler rows.
The small control plane deliberately keeps the three cost meanings separate:
``scientific_equal_sample_cost`` is a shared paired-run accounting,
``isolated_estimator_cost`` is a fixed-state method run, and
``online_training_incremental_cost`` is the only cost used for the method
decision.  Missing four-card evidence or non-quiescent I/O can therefore
never be turned into a formal PASS by a reducer or a report writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord, GateStatus
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import APPROVED_GPU_UUIDS, EXCLUDED_GPU_UUID, EXCLUDED_PCI, S27_RAW_MANIFEST_SCHEMA


S29_SCHEMA = "stage2-s209-g27a-cost-system-validation-v1"
S29_GATE_SCHEMA = "stage2-s209-g27a-gate-v1"
S29_MEASUREMENT_PLAN_SCHEMA = "stage2-s209-g27a-measurement-plan-v1"
S29_TASK_ID = "stage2.09_cost_and_system_validation"
S29_MATRIX_SCHEMA = "stage2-formal-pilot-matrix-freeze-v1"
S29_COST_SEMANTICS = (
    "scientific_equal_sample_cost",
    "isolated_estimator_cost",
    "online_training_incremental_cost",
)
S29_G25_GATE_ID = "stage2.G2.5"
S29_METHODS = ("raw", "double", "u")
S29_CELL_IDS = EXPECTED_CELL_IDS
S29_DECISION_RATIO = 1.25
S29_CROSSCHECK_TOLERANCE = 0.25
S29_TIMING_FIELDS = (
    "data_wait_seconds",
    "forward_seconds",
    "backward_seconds",
    "gradient_aggregation_seconds",
    "formula_seconds",
    "statistics_seconds",
    "communication_seconds",
    "write_seconds",
)
S29_COUNT_FIELDS = (
    "sequence_count",
    "token_count",
    "backward_count",
    "communication_bytes",
    "output_bytes",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class S29G27ABlocked(RuntimeError):
    """Raised for malformed identity, lineage, or cost observations."""


def _finite(value: Any, *, field: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise S29G27ABlocked(f"{field}:NONFINITE")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, field=f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise S29G27ABlocked(f"{field}:NON_STRING_KEY")
            _finite(item, field=f"{field}.{key}")
        return
    raise S29G27ABlocked(f"{field}:NOT_JSON_VALUE")


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S29G27ABlocked(f"{field}:SHA256_REQUIRED")
    return value


def _id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise S29G27ABlocked(f"{field}:SAFE_ID_REQUIRED")
    return value


def _positive(value: Any, *, field: str, zero_ok: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value < 0 if zero_ok else value <= 0):
        raise S29G27ABlocked(f"{field}:INTEGER_REQUIRED")
    return value


def _number(value: Any, *, field: str, zero_ok: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise S29G27ABlocked(f"{field}:NUMBER_REQUIRED")
    result = float(value)
    if not math.isfinite(result) or (result < 0 if zero_ok else result <= 0):
        raise S29G27ABlocked(f"{field}:FINITE_NONNEGATIVE_REQUIRED")
    return result


def _load(value: Mapping[str, Any] | str | Path, *, field: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        try:
            loaded = load_canonical_json(value)
        except Exception as error:  # pragma: no cover - stable public error below
            raise S29G27ABlocked(f"{field}:CANONICAL_READ_FAILED") from error
        if not isinstance(loaded, Mapping):
            raise S29G27ABlocked(f"{field}:OBJECT_REQUIRED")
        result = dict(loaded)
    _finite(result, field=field)
    return result


def _verify_hash(value: Mapping[str, Any], *, field: str) -> str:
    digest = _sha(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if digest != canonical_json_hash(body):
        raise S29G27ABlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return digest


def _method(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise S29G27ABlocked(f"{field}:METHOD_REQUIRED")
    lowered = value.lower()
    if lowered == "u" or lowered.startswith("u_m"):
        return "u"
    if lowered in {"raw", "double"}:
        return lowered
    raise S29G27ABlocked(f"{field}:UNKNOWN_METHOD")


@dataclass(frozen=True, slots=True)
class S29FrozenInputs:
    """Content identities consumed by every S2.9 observation."""

    matrix_hash: str
    g24b_gate_hash: str
    raw_manifest_hash: str
    raw_run_id: str
    plan_hash: str
    mapping_hash: str
    sampling_plan_hash: str
    expected_unit_ids: tuple[str, ...]
    batch_size: int
    microbatch_count: int
    repetitions: int
    completion_denominator: int
    g25_gate_hash: str | None = None
    matrix_ref: str = "g24b-matrix.json"
    raw_manifest_ref: str = "raw-results-manifest.json"

    def __post_init__(self) -> None:
        for name in ("matrix_hash", "g24b_gate_hash", "raw_manifest_hash", "plan_hash", "mapping_hash", "sampling_plan_hash"):
            _sha(getattr(self, name), field=name)
        if self.g25_gate_hash is not None:
            _sha(self.g25_gate_hash, field="g25_gate_hash")
        _id(self.raw_run_id, field="raw_run_id")
        if not self.expected_unit_ids or len(set(self.expected_unit_ids)) != len(self.expected_unit_ids):
            raise S29G27ABlocked("expected_unit_ids:NONEMPTY_UNIQUE_REQUIRED")
        _positive(self.batch_size, field="batch_size")
        _positive(self.microbatch_count, field="microbatch_count")
        _positive(self.repetitions, field="repetitions")
        _positive(self.completion_denominator, field="completion_denominator")
        if self.completion_denominator != len(self.expected_unit_ids):
            raise S29G27ABlocked("completion_denominator:UNIT_COUNT_MISMATCH")

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_hash": self.matrix_hash,
            "g24b_gate_hash": self.g24b_gate_hash,
            "raw_manifest_hash": self.raw_manifest_hash,
            "raw_run_id": self.raw_run_id,
            "plan_hash": self.plan_hash,
            "mapping_hash": self.mapping_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "expected_unit_ids": list(self.expected_unit_ids),
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "completion_denominator": self.completion_denominator,
            "g25_gate_hash": self.g25_gate_hash,
            "matrix_ref": self.matrix_ref,
            "raw_manifest_ref": self.raw_manifest_ref,
        }


def prepare_s209_measurement_plan(
    frozen: S29FrozenInputs,
    *,
    run_id: str,
    anchor_ids: Sequence[str] = ("method-only-anchor-0", "method-only-anchor-1"),
    repetitions: int = 2,
    randomization_seed: int = 2909,
) -> dict[str, Any]:
    """Create the detached work order consumed by a formal profiler runner.

    The function only emits an immutable plan; it never starts a process or
    touches a GPU.  A runner must record the resulting method order and all
    fields in :data:`S29_TIMING_FIELDS`/:data:`S29_COUNT_FIELDS` in its rows.
    """

    run_id = _id(run_id, field="measurement_plan.run_id")
    if not anchor_ids or len(set(anchor_ids)) != len(anchor_ids):
        raise S29G27ABlocked("measurement_plan.anchor_ids:UNIQUE_REQUIRED")
    normalized_anchors = [_id(anchor, field="measurement_plan.anchor_id") for anchor in anchor_ids]
    if repetitions < 2:
        raise S29G27ABlocked("measurement_plan.repetitions:AT_LEAST_TWO_REQUIRED")
    if isinstance(randomization_seed, bool) or not isinstance(randomization_seed, int):
        raise S29G27ABlocked("measurement_plan.randomization_seed:INTEGER_REQUIRED")
    rng = random.Random(randomization_seed)
    rows: list[dict[str, Any]] = []
    for anchor in normalized_anchors:
        for repetition in range(repetitions):
            order = list(S29_METHODS)
            rng.shuffle(order)
            rows.append({"anchor_id": anchor, "repetition": repetition, "method_order": order})
    body: dict[str, Any] = {
        "schema_version": S29_MEASUREMENT_PLAN_SCHEMA,
        "task_id": S29_TASK_ID,
        "run_id": run_id,
        "source_raw_run_id": frozen.raw_run_id,
        "matrix_hash": frozen.matrix_hash,
        "raw_manifest_hash": frozen.raw_manifest_hash,
        "g25_gate_hash": frozen.g25_gate_hash,
        "cost_semantics": list(S29_COST_SEMANTICS),
        "method_only": True,
        "randomized_method_order": True,
        "randomization_seed": randomization_seed,
        "anchor_ids": normalized_anchors,
        "repetitions": repetitions,
        "required_methods": list(S29_METHODS),
        "rows": rows,
        "required_timing_fields": list(S29_TIMING_FIELDS),
        "required_count_fields": list(S29_COUNT_FIELDS),
        "single_gpu_required": True,
        "four_gpu_required": True,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "online_decision_ratio": S29_DECISION_RATIO,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _validate_measurement_plan(value: Any, *, frozen: S29FrozenInputs) -> dict[str, Any]:
    payload = _load(value, field="measurement_plan")
    if payload.get("schema_version") != S29_MEASUREMENT_PLAN_SCHEMA or payload.get("task_id") != S29_TASK_ID:
        raise S29G27ABlocked("MEASUREMENT_PLAN_SCHEMA_INVALID")
    _verify_hash(payload, field="measurement_plan")
    for field, expected in (("matrix_hash", frozen.matrix_hash), ("raw_manifest_hash", frozen.raw_manifest_hash), ("source_raw_run_id", frozen.raw_run_id), ("g25_gate_hash", frozen.g25_gate_hash)):
        if payload.get(field) != expected:
            raise S29G27ABlocked(f"MEASUREMENT_PLAN_{field.upper()}_MISMATCH")
    if payload.get("cost_semantics") != list(S29_COST_SEMANTICS) or payload.get("required_methods") != list(S29_METHODS) or payload.get("method_only") is not True or payload.get("randomized_method_order") is not True:
        raise S29G27ABlocked("MEASUREMENT_PLAN_COST_OR_METHOD_CONTRACT_INVALID")
    if payload.get("approved_gpu_uuids") != list(APPROVED_GPU_UUIDS) or payload.get("excluded_gpu_uuid") != EXCLUDED_GPU_UUID or payload.get("excluded_pci") != EXCLUDED_PCI:
        raise S29G27ABlocked("MEASUREMENT_PLAN_GPU_ALLOWLIST_DRIFT")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise S29G27ABlocked("MEASUREMENT_PLAN_ROWS_REQUIRED")
    anchor_ids = payload.get("anchor_ids")
    repetitions = payload.get("repetitions")
    if not isinstance(anchor_ids, list) or not anchor_ids or len(set(anchor_ids)) != len(anchor_ids) or isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 2:
        raise S29G27ABlocked("MEASUREMENT_PLAN_ANCHOR_CONTRACT_INVALID")
    expected_rows = len(anchor_ids) * repetitions
    if len(rows) != expected_rows:
        raise S29G27ABlocked("MEASUREMENT_PLAN_ROW_COUNT_INVALID")
    seen: set[tuple[str, int]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"anchor_id", "repetition", "method_order"} or row.get("anchor_id") not in anchor_ids or isinstance(row.get("repetition"), bool) or not isinstance(row.get("repetition"), int) or not 0 <= row["repetition"] < repetitions:
            raise S29G27ABlocked(f"MEASUREMENT_PLAN_ROW_INVALID:{index}")
        method_order = row["method_order"]
        if not isinstance(method_order, list) or len(method_order) != len(S29_METHODS) or len(set(method_order)) != len(S29_METHODS) or set(method_order) != set(S29_METHODS):
            raise S29G27ABlocked(f"MEASUREMENT_PLAN_METHOD_ORDER_INVALID:{index}")
        key = (str(row["anchor_id"]), int(row["repetition"]))
        if key in seen:
            raise S29G27ABlocked(f"MEASUREMENT_PLAN_DUPLICATE_ROW:{index}")
        seen.add(key)
    return dict(payload)


def bind_s209_inputs(
    *,
    matrix: Mapping[str, Any] | str | Path,
    g24b_gate: Mapping[str, Any] | str | Path,
    raw_manifest: Mapping[str, Any] | str | Path,
    g25_gate: Mapping[str, Any] | str | Path | None = None,
    require_g25: bool = False,
    matrix_ref: str = "g24b-matrix.json",
    raw_manifest_ref: str = "raw-results-manifest.json",
) -> S29FrozenInputs:
    """Validate and bind G2.4b's frozen matrix to the sealed S2.7 manifest."""

    matrix_payload = _load(matrix, field="g24b_matrix")
    matrix_hash = _verify_hash(matrix_payload, field="g24b_matrix")
    if matrix_payload.get("schema_version") != S29_MATRIX_SCHEMA:
        raise S29G27ABlocked("G24B_MATRIX_SCHEMA_INVALID")
    if matrix_payload.get("scope") != "formal" or matrix_payload.get("status") != "FORMAL_FROZEN" or matrix_payload.get("formal_eligible") is not True:
        raise S29G27ABlocked("G24B_FORMAL_FROZEN_MATRIX_REQUIRED")
    if tuple(matrix_payload.get("anchor_ids", ())) != S29_CELL_IDS:
        raise S29G27ABlocked("G24B_MATRIX_SIX_CELL_ORDER_INVALID")
    if set(matrix_payload.get("cost_semantics", ())) != set(S29_COST_SEMANTICS):
        raise S29G27ABlocked("G24B_COST_SEMANTICS_INCOMPLETE")
    gate_payload = _load(g24b_gate, field="g24b_gate")
    try:
        gate = GateRecord.from_mapping(dict(gate_payload))
    except Exception as error:
        raise S29G27ABlocked("G24B_GATE_RECORD_INVALID") from error
    if gate.gate_id != "stage2.G2.4b" or gate.effective_status() is not GateStatus.PASS:
        raise S29G27ABlocked("G24B_GATE_PASS_REQUIRED")
    gate_hash = gate.artifact_hash
    if matrix_payload.get("qualification_gate_hash") != gate_hash:
        raise S29G27ABlocked("G24B_GATE_HASH_MISMATCH")

    manifest_payload = _load(raw_manifest, field="s27_raw_manifest")
    manifest_hash = _verify_hash(manifest_payload, field="s27_raw_manifest")
    if manifest_payload.get("schema_version") != S27_RAW_MANIFEST_SCHEMA or manifest_payload.get("status") != "SEALED" or manifest_payload.get("formal_eligible") is not True:
        raise S29G27ABlocked("S27_SEALED_FORMAL_MANIFEST_REQUIRED")
    if manifest_payload.get("matrix_hash") != matrix_hash:
        raise S29G27ABlocked("S27_MATRIX_HASH_MISMATCH")
    raw_run_id = _id(manifest_payload.get("run_id"), field="s27_raw_manifest.run_id")
    expected = _positive(manifest_payload.get("expected_unit_count"), field="expected_unit_count")
    completed = _positive(manifest_payload.get("completed_unit_count"), field="completed_unit_count")
    if expected != completed or manifest_payload.get("failed_unit_count") != 0 or manifest_payload.get("failure_fraction") != 0.0:
        raise S29G27ABlocked("S27_MANIFEST_INCOMPLETE")
    units = manifest_payload.get("units")
    if not isinstance(units, list) or len(units) != expected:
        raise S29G27ABlocked("S27_UNIT_DENOMINATOR_INVALID")
    unit_ids: list[str] = []
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping):
            raise S29G27ABlocked(f"S27_UNIT_INVALID:{index}")
        unit_ids.append(_id(unit.get("unit_id"), field=f"s27.units[{index}].unit_id"))
        _sha(unit.get("unit_artifact_hash"), field=f"s27.units[{index}].unit_artifact_hash")
        _sha(unit.get("raw_artifact_hash"), field=f"s27.units[{index}].raw_artifact_hash")
    if len(set(unit_ids)) != len(unit_ids):
        raise S29G27ABlocked("S27_UNIT_IDS_NOT_UNIQUE")
    g25_gate_hash: str | None = None
    if g25_gate is None:
        if require_g25:
            raise S29G27ABlocked("G25_GATE_REQUIRED")
    else:
        gate25_payload = _load(g25_gate, field="g25_gate")
        try:
            gate25 = GateRecord.from_mapping(dict(gate25_payload))
        except Exception as error:
            raise S29G27ABlocked("G25_GATE_RECORD_INVALID") from error
        if gate25.gate_id != S29_G25_GATE_ID or gate25.effective_status() is not GateStatus.PASS:
            raise S29G27ABlocked("G25_GATE_PASS_REQUIRED")
        g25_gate_hash = gate25.artifact_hash
        measured = gate25.measured
        if measured.get("raw_manifest_hash") != manifest_hash:
            raise S29G27ABlocked("G25_RAW_MANIFEST_HASH_MISMATCH")
    plan_hash = _sha(manifest_payload.get("plan_hash"), field="s27.plan_hash")
    mapping_hash = _sha(manifest_payload.get("mapping_hash"), field="s27.mapping_hash")
    sampling_hash = _sha(manifest_payload.get("sampling_plan_hash"), field="s27.sampling_plan_hash")
    batch_size = _positive(matrix_payload.get("b_primary"), field="matrix.b_primary")
    microbatch_count = _positive(matrix_payload.get("m_primary"), field="matrix.m_primary")
    repetitions = _positive(matrix_payload.get("r_primary"), field="matrix.r_primary")
    return S29FrozenInputs(
        matrix_hash=matrix_hash,
        g24b_gate_hash=gate_hash,
        raw_manifest_hash=manifest_hash,
        raw_run_id=raw_run_id,
        plan_hash=plan_hash,
        mapping_hash=mapping_hash,
        sampling_plan_hash=sampling_hash,
        expected_unit_ids=tuple(unit_ids),
        batch_size=batch_size,
        microbatch_count=microbatch_count,
        repetitions=repetitions,
        completion_denominator=expected,
        g25_gate_hash=g25_gate_hash,
        matrix_ref=matrix_ref,
        raw_manifest_ref=raw_manifest_ref,
    )


def _semantic_rows(value: Any, *, semantic: str) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    if isinstance(value, list):
        if any(not isinstance(item, Mapping) for item in value):
            raise S29G27ABlocked(f"{semantic}:OBSERVATION_OBJECT_REQUIRED")
        return {}, [dict(item) for item in value]
    if not isinstance(value, Mapping):
        raise S29G27ABlocked(f"{semantic}:OBSERVATIONS_OBJECT_REQUIRED")
    rows = value.get("observations", value.get("rows"))
    if not isinstance(rows, list):
        raise S29G27ABlocked(f"{semantic}:OBSERVATIONS_REQUIRED")
    if any(not isinstance(item, Mapping) for item in rows):
        raise S29G27ABlocked(f"{semantic}:OBSERVATION_OBJECT_REQUIRED")
    return dict(value), [dict(item) for item in rows]


def _normalize_record(
    row: Mapping[str, Any],
    *,
    semantic: str,
    frozen: S29FrozenInputs,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(row)
    declared_semantic = result.get("semantic")
    if declared_semantic is not None and declared_semantic != semantic:
        raise S29G27ABlocked(f"{semantic}:SEMANTIC_MIXED")
    result["semantic"] = semantic
    result["method"] = _method(result.get("method", "shared" if semantic == S29_COST_SEMANTICS[0] else None), field=f"{semantic}.method") if semantic != S29_COST_SEMANTICS[0] else str(result.get("method", "shared"))
    if semantic != S29_COST_SEMANTICS[0] and result["method"] not in S29_METHODS:
        raise S29G27ABlocked(f"{semantic}:METHOD_REQUIRED")
    result["run_id"] = _id(result.get("run_id", frozen.raw_run_id), field=f"{semantic}.run_id")
    source_run = result.get("source_raw_run_id", result.get("raw_run_id", frozen.raw_run_id))
    if source_run != frozen.raw_run_id:
        raise S29G27ABlocked(f"{semantic}:{result['run_id']}:RAW_RUN_ID_MISMATCH")
    result["source_raw_run_id"] = frozen.raw_run_id
    for name, expected in (("matrix_hash", frozen.matrix_hash), ("raw_manifest_hash", frozen.raw_manifest_hash)):
        if name in result and result[name] != expected:
            raise S29G27ABlocked(f"{semantic}:{name}:LINEAGE_MISMATCH")
        result[name] = expected
    for name in ("inventory_artifact_hash", "inventory_source_sha256"):
        result[name] = _sha(result.get(name), field=f"{semantic}.{name}")
    result["anchor_id"] = _id(result.get("anchor_id", f"{semantic}-anchor"), field=f"{semantic}.anchor_id")
    result["repetition"] = _positive(result.get("repetition", 0), field=f"{semantic}.repetition", zero_ok=True)
    result["gpu_uuid"] = str(result.get("gpu_uuid", ""))
    if result["gpu_uuid"] not in APPROVED_GPU_UUIDS:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:UNAPPROVED_GPU_UUID")
    result["device_count"] = _positive(result.get("device_count", 1), field=f"{semantic}.device_count")
    for name in S29_COUNT_FIELDS:
        alias = {"output_bytes": "result_bytes"}.get(name)
        raw = result.get(name, result.get(alias) if alias else None)
        result[name] = _positive(raw, field=f"{semantic}.{name}", zero_ok=name.endswith("bytes"))
    for name in S29_TIMING_FIELDS:
        aliases = {
            "gradient_aggregation_seconds": ("gradient_seconds", "gradient_read_seconds"),
            "communication_seconds": ("all_reduce_seconds",),
        }.get(name, ())
        raw = result.get(name)
        if raw is None:
            for alias in aliases:
                if alias in result:
                    raw = result[alias]
                    break
        result[name] = _number(raw, field=f"{semantic}.{name}")
    result["wall_seconds"] = _number(result.get("wall_seconds"), field=f"{semantic}.wall_seconds", zero_ok=False)
    result["allocated_peak_bytes"] = _positive(result.get("allocated_peak_bytes"), field=f"{semantic}.allocated_peak_bytes")
    result["reserved_peak_bytes"] = _positive(result.get("reserved_peak_bytes"), field=f"{semantic}.reserved_peak_bytes")
    result["device_peak_bytes"] = _positive(result.get("device_peak_bytes"), field=f"{semantic}.device_peak_bytes")
    if not result["allocated_peak_bytes"] <= result["reserved_peak_bytes"] <= result["device_peak_bytes"]:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:PEAK_MEMORY_ORDER_INVALID")
    phase_total = sum(result[name] for name in S29_TIMING_FIELDS)
    if result["wall_seconds"] + 1e-9 < phase_total:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:WALL_PHASE_TOTAL_INVALID")
    result["timing_total_seconds"] = phase_total
    result["cost_io_quiescent"] = result.get("cost_io_quiescent", result.get("io_quiescent"))
    if type(result["cost_io_quiescent"]) is not bool:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:COST_IO_QUIESCENCE_REQUIRED")
    result["health_ok"] = result.get("health_ok")
    if type(result["health_ok"]) is not bool:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:HEALTH_MARKER_REQUIRED")
    if result.get("batch_size", frozen.batch_size) != frozen.batch_size or result.get("microbatch_count", frozen.microbatch_count) != frozen.microbatch_count:
        raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:B_M_DRIFT")
    result["batch_size"] = frozen.batch_size
    result["microbatch_count"] = frozen.microbatch_count
    source_units = result.get("source_unit_ids", result.get("unit_ids"))
    if source_units is not None:
        if not isinstance(source_units, list) or not source_units or any(item not in frozen.expected_unit_ids for item in source_units):
            raise S29G27ABlocked(f"{semantic}:{result['anchor_id']}:SOURCE_UNIT_ID_INVALID")
        result["source_unit_ids"] = list(dict.fromkeys(source_units))
    else:
        result["source_unit_ids"] = []
    default_kind = "shared_runner" if semantic == S29_COST_SEMANTICS[0] else "method_only"
    result["anchor_kind"] = result.get("anchor_kind", metadata.get("anchor_kind", default_kind))
    if semantic == S29_COST_SEMANTICS[0] and result["anchor_kind"] != "shared_runner":
        raise S29G27ABlocked(f"{semantic}:SHARED_ANCHOR_REQUIRED")
    if semantic != S29_COST_SEMANTICS[0] and result["anchor_kind"] not in {"method_only", "fixed_state_method_only"}:
        raise S29G27ABlocked(f"{semantic}:METHOD_ONLY_ANCHOR_REQUIRED")
    return result


class StrictS29Reducer:
    """Single-writer reducer for cost rows; exact duplicate replay is idempotent."""

    def __init__(self, frozen: S29FrozenInputs, *, run_id: str) -> None:
        self.frozen = frozen
        self.run_id = _id(run_id, field="s29.run_id")
        self._rows: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
        self._sealed = False

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def add(self, record: Mapping[str, Any], *, semantic: str | None = None, metadata: Mapping[str, Any] | None = None) -> bool:
        if self._sealed:
            raise S29G27ABlocked("SEALED_REDUCER_IS_IMMUTABLE")
        row = dict(record)
        selected = semantic or row.get("semantic")
        if selected not in S29_COST_SEMANTICS:
            raise S29G27ABlocked("COST_SEMANTIC_REQUIRED")
        normalized = _normalize_record(row, semantic=selected, frozen=self.frozen, metadata=metadata or {})
        method = str(normalized["method"])
        key = (selected, method, str(normalized["anchor_id"]), int(normalized["repetition"]), str(normalized["run_id"]))
        digest = canonical_json_hash(normalized)
        existing = self._rows.get(key)
        if existing is not None:
            if canonical_json_hash(existing) == digest:
                return False
            raise S29G27ABlocked(f"DUPLICATE_OR_RETRY_ATTEMPT:{key}")
        self._rows[key] = normalized
        return True

    def seal(self) -> tuple[Mapping[str, Any], ...]:
        if self._sealed:
            raise S29G27ABlocked("SEALED_REDUCER_IS_IMMUTABLE")
        if not self._rows:
            raise S29G27ABlocked("COST_OBSERVATIONS_EMPTY")
        semantics = {str(row["semantic"]) for row in self._rows.values()}
        missing = set(S29_COST_SEMANTICS) - semantics
        if missing:
            raise S29G27ABlocked(f"COST_SEMANTICS_MISSING:{','.join(sorted(missing))}")
        self._sealed = True
        return self.records


def _health_reasons(snapshot: Any, *, expected_io: bool) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    provisional: list[str] = []
    if not isinstance(snapshot, Mapping):
        return ["HEALTH_SNAPSHOT_REQUIRED"], provisional
    required = (
        "healthy",
        "idle",
        "same_gpu_class",
        "gpu_class",
        "gpu_uuids",
        "ecc_errors",
        "xid_errors",
        "inventory_artifact_hash",
        "inventory_source_sha256",
    )
    missing = [name for name in required if name not in snapshot]
    if missing:
        blockers.append("HEALTH_FIELDS_MISSING:" + ",".join(missing))
        return blockers, provisional
    if snapshot.get("healthy") is not True or snapshot.get("idle") is not True or snapshot.get("same_gpu_class") is not True:
        blockers.append("GPU_HEALTH_OR_IDLE_FAILED")
    uuids = snapshot.get("gpu_uuids")
    if not isinstance(uuids, list) or not uuids or len(set(uuids)) != len(uuids) or any(uuid not in APPROVED_GPU_UUIDS for uuid in uuids):
        blockers.append("APPROVED_GPU_UUIDS_REQUIRED")
    if snapshot.get("ecc_errors") != 0 or snapshot.get("xid_errors") != 0:
        blockers.append("GPU_ERROR_COUNTER_NONZERO")
    try:
        _sha(snapshot.get("inventory_artifact_hash"), field="health.inventory_artifact_hash")
        _sha(snapshot.get("inventory_source_sha256"), field="health.inventory_source_sha256")
    except S29G27ABlocked as error:
        blockers.append(str(error))
    io = snapshot.get("cost_io_quiescent")
    if type(io) is not bool:
        blockers.append("HEALTH_IO_QUIESCENCE_REQUIRED")
    elif io is not True or expected_io is not True:
        provisional.append("COST_IO_NOT_QUIESCENT")
    return blockers, provisional


def _anchor_reasons(anchor: Any, *, frozen: S29FrozenInputs, expected_devices: int) -> list[str]:
    if not isinstance(anchor, Mapping):
        return ["FOUR_CARD_ANCHOR_MISSING" if expected_devices == 4 else "SINGLE_CARD_ANCHOR_MISSING"]
    prefix = "FOUR_CARD" if expected_devices == 4 else "SINGLE_CARD"
    reasons: list[str] = []
    if anchor.get("status") != "PASS":
        reasons.append(prefix + "_ANCHOR_NOT_PASS")
    if anchor.get("matrix_hash") != frozen.matrix_hash:
        reasons.append(prefix + "_MATRIX_HASH_MISMATCH")
    if anchor.get("source_raw_run_id", anchor.get("raw_run_id")) != frozen.raw_run_id:
        reasons.append(prefix + "_RAW_RUN_ID_MISMATCH")
    if anchor.get("device_count") != expected_devices:
        reasons.append(prefix + "_DEVICE_COUNT_INVALID")
    uuids = anchor.get("gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != expected_devices or len(set(uuids)) != len(uuids) or any(uuid not in APPROVED_GPU_UUIDS for uuid in uuids):
        reasons.append(prefix + "_APPROVED_UUIDS_REQUIRED")
    if expected_devices == 4 and set(anchor.get("gpu_uuids", [])) != set(APPROVED_GPU_UUIDS):
        reasons.append("FOUR_CARD_COMPLETE_APPROVED_SET_REQUIRED")
    if anchor.get("cost_io_quiescent") is not True:
        reasons.append(prefix + "_IO_NOT_QUIESCENT")
    if anchor.get("health_ok") is not True or anchor.get("numeric_consistency") is not True:
        reasons.append(prefix + "_HEALTH_OR_NUMERIC_CONSISTENCY_FAILED")
    for name in S29_COUNT_FIELDS:
        if name not in anchor:
            reasons.append(prefix + "_COUNT_FIELDS_MISSING:" + name)
    if anchor.get("batch_size") != frozen.batch_size or anchor.get("microbatch_count") != frozen.microbatch_count:
        reasons.append(prefix + "_B_M_DRIFT")
    return reasons


def _group(rows: Iterable[Mapping[str, Any]], *, semantic: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["semantic"] == semantic:
            grouped.setdefault(str(row["anchor_id"]), []).append(row)
    return grouped


def _online_checks(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    blockers: list[str] = []
    groups = _group(rows, semantic="online_training_incremental_cost")
    if not groups:
        return ["ONLINE_COST_OBSERVATIONS_MISSING"], {}, {}
    methods = {str(row["method"]) for row in rows if row["semantic"] == "online_training_incremental_cost"}
    if methods != set(S29_METHODS):
        blockers.append("ONLINE_METHOD_SET_INCOMPLETE")
    repetition_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for anchor_id, anchor_rows in groups.items():
        for row in anchor_rows:
            repetition_groups.setdefault((anchor_id, int(row["repetition"])), []).append(row)
    for (anchor_id, repetition), repetition_rows in repetition_groups.items():
        if {str(row["method"]) for row in repetition_rows} != set(S29_METHODS):
            blockers.append(f"METHOD_ONLY_ANCHOR_METHOD_SET_INVALID:{anchor_id}:r{repetition}")
        if any(row.get("cost_io_quiescent") is not True or row.get("health_ok") is not True for row in repetition_rows):
            blockers.append(f"METHOD_ONLY_ANCHOR_HEALTH_INVALID:{anchor_id}:r{repetition}")
    if len(repetition_groups) < 2:
        blockers.append("METHOD_ONLY_ANCHOR_REPETITIONS_LT_TWO")
    randomized = metadata.get("randomized_method_order")
    seed = metadata.get("randomization_seed")
    if randomized is not True or isinstance(seed, bool) or not isinstance(seed, int):
        blockers.append("METHOD_ONLY_RANDOMIZATION_EVIDENCE_REQUIRED")
    aggregates: dict[str, dict[str, float]] = {}
    for method in S29_METHODS:
        method_rows = [row for row in rows if row["semantic"] == "online_training_incremental_cost" and row["method"] == method]
        if not method_rows:
            continue
        aggregates[method] = {
            "wall_seconds": statistics.median(float(row["wall_seconds"]) for row in method_rows),
            "allocated_peak_bytes": statistics.median(float(row["allocated_peak_bytes"]) for row in method_rows),
            "reserved_peak_bytes": statistics.median(float(row["reserved_peak_bytes"]) for row in method_rows),
            "device_peak_bytes": statistics.median(float(row["device_peak_bytes"]) for row in method_rows),
            "sequence_count": float(statistics.median(int(row["sequence_count"]) for row in method_rows)),
            "token_count": float(statistics.median(int(row["token_count"]) for row in method_rows)),
            "backward_count": float(statistics.median(int(row["backward_count"]) for row in method_rows)),
            "communication_bytes": float(statistics.median(int(row["communication_bytes"]) for row in method_rows)),
            "output_bytes": float(statistics.median(int(row["output_bytes"]) for row in method_rows)),
        }
    ratios: dict[str, Any] = {"threshold": S29_DECISION_RATIO, "source": "online_training_incremental_cost", "methods": {}}
    if metadata.get("decision_ratio_threshold", S29_DECISION_RATIO) != S29_DECISION_RATIO:
        blockers.append("ONLINE_DECISION_THRESHOLD_NOT_FROZEN_1_25")
    baseline = aggregates.get("raw")
    if baseline is None or baseline.get("wall_seconds", 0) <= 0:
        blockers.append("ONLINE_RAW_BASELINE_MISSING")
    else:
        for method in ("double", "u"):
            current = aggregates.get(method)
            if current is None:
                continue
            values = {
                "wall_seconds": current["wall_seconds"] / baseline["wall_seconds"],
                "allocated_peak_bytes": current["allocated_peak_bytes"] / baseline["allocated_peak_bytes"],
                "reserved_peak_bytes": current["reserved_peak_bytes"] / baseline["reserved_peak_bytes"],
                "device_peak_bytes": current["device_peak_bytes"] / baseline["device_peak_bytes"],
            }
            ratios["methods"][method] = values
            if any(value > S29_DECISION_RATIO for value in values.values()):
                blockers.append(f"ONLINE_RATIO_EXCEEDS_1_25:{method}")
    return blockers, aggregates, ratios


def _crosscheck(rows: Sequence[Mapping[str, Any]], value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["SHARED_ATTRIBUTION_CROSSCHECK_REQUIRED"]
    reasons: list[str] = []
    shared = [row for row in rows if row["semantic"] == "scientific_equal_sample_cost"]
    isolated = _group(rows, semantic="isolated_estimator_cost")
    for method in S29_METHODS:
        direct = [row for row in shared if row.get("method") == method]
        payload = value.get(method)
        if not direct or not isinstance(payload, Mapping) or not any(row.get("method") == method for group in isolated.values() for row in group):
            reasons.append(f"SHARED_ATTRIBUTION_ROW_MISSING:{method}")
            continue
        shared_seconds = statistics.median(float(row["wall_seconds"]) for row in direct)
        isolated_rows = [row for group in isolated.values() for row in group if row.get("method") == method]
        isolated_seconds = statistics.median(float(row["wall_seconds"]) for row in isolated_rows)
        declared_shared = _number(payload.get("shared_wall_seconds"), field=f"crosscheck.{method}.shared_wall_seconds")
        declared_isolated = _number(payload.get("isolated_wall_seconds"), field=f"crosscheck.{method}.isolated_wall_seconds")
        declared_delta = _number(payload.get("relative_difference"), field=f"crosscheck.{method}.relative_difference")
        if not math.isclose(declared_shared, shared_seconds, rel_tol=1e-6, abs_tol=1e-6) or not math.isclose(declared_isolated, isolated_seconds, rel_tol=1e-6, abs_tol=1e-6):
            reasons.append(f"SHARED_ATTRIBUTION_DECLARED_VALUE_MISMATCH:{method}")
        expected_delta = abs(shared_seconds - isolated_seconds) / max(isolated_seconds, 1e-12)
        if not math.isclose(declared_delta, expected_delta, rel_tol=1e-6, abs_tol=1e-6) or declared_delta > S29_CROSSCHECK_TOLERANCE:
            reasons.append(f"SHARED_ATTRIBUTION_DISAGREEMENT:{method}")
    return reasons


def _consistency(rows: Sequence[Mapping[str, Any]], frozen: S29FrozenInputs) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "matrix_b_m": True,
        "finite_timing": True,
        "peak_memory_order": True,
        "sequence_token_backward_present": True,
        "method_counts_equal_per_anchor": True,
    }
    for row in rows:
        checks["matrix_b_m"] &= row.get("batch_size") == frozen.batch_size and row.get("microbatch_count") == frozen.microbatch_count
        checks["finite_timing"] &= all(math.isfinite(float(row[name])) for name in S29_TIMING_FIELDS + ("wall_seconds",))
        checks["peak_memory_order"] &= row["allocated_peak_bytes"] <= row["reserved_peak_bytes"] <= row["device_peak_bytes"]
    online = _group(rows, semantic="online_training_incremental_cost")
    for group in online.values():
        triples = {(row["sequence_count"], row["token_count"], row["backward_count"]) for row in group}
        checks["method_counts_equal_per_anchor"] &= len(triples) == 1
    return {"checks": checks, "all_pass": all(checks.values())}


def _pareto(rows: Sequence[Mapping[str, Any]], accuracy_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    if not accuracy_rows:
        return {"status": "BLOCKED", "rows": [], "dominated": []}, ["PARETO_ACCURACY_ROWS_REQUIRED"]
    normalized: list[dict[str, Any]] = []
    reasons: list[str] = []
    online: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["semantic"] == "online_training_incremental_cost":
            online.setdefault(str(row["method"]), {})["wall_seconds"] = float(row["wall_seconds"])
            online.setdefault(str(row["method"]), {})["device_peak_bytes"] = float(row["device_peak_bytes"])
    for index, source in enumerate(accuracy_rows):
        if not isinstance(source, Mapping):
            reasons.append(f"PARETO_ROW_INVALID:{index}")
            continue
        method = _method(source.get("method"), field=f"pareto[{index}].method")
        cell = source.get("cell_id")
        if cell not in S29_CELL_IDS:
            reasons.append(f"PARETO_CELL_INVALID:{index}")
            continue
        fields = ("corrected_nmse", "mse", "spearman", "overlap_1pct")
        if any(name not in source for name in fields):
            reasons.append(f"PARETO_METRIC_MISSING:{index}")
            continue
        nmse = _number(source["corrected_nmse"], field=f"pareto[{index}].corrected_nmse")
        mse = _number(source["mse"], field=f"pareto[{index}].mse")
        spearman = float(source["spearman"])
        overlap = float(source["overlap_1pct"])
        if not -1 <= spearman <= 1 or not 0 <= overlap <= 1:
            reasons.append(f"PARETO_RANK_METRIC_INVALID:{index}")
            continue
        if method not in online:
            reasons.append(f"PARETO_ONLINE_COST_MISSING:{method}")
            continue
        token_count = _positive(source.get("token_count", 1), field=f"pareto[{index}].token_count")
        normalized.append({
            "row_id": str(source.get("row_id", f"{cell}:{method}:{index}")),
            "cell_id": cell,
            "method": method,
            "corrected_nmse": nmse,
            "mse": mse,
            "spearman": spearman,
            "overlap_1pct": overlap,
            "wall_seconds": online[method]["wall_seconds"],
            "device_peak_bytes": online[method]["device_peak_bytes"],
            "token_count": token_count,
            "time_per_mse": None if mse == 0 else online[method]["wall_seconds"] / mse,
            "time_per_sorting_quality": online[method]["wall_seconds"] / max(overlap, 1e-12),
            "mse_per_token": mse / token_count,
        })
    dominated: list[str] = []
    for candidate in normalized:
        for other in normalized:
            if candidate is other:
                continue
            no_worse = (
                other["corrected_nmse"] <= candidate["corrected_nmse"]
                and other["wall_seconds"] <= candidate["wall_seconds"]
                and other["device_peak_bytes"] <= candidate["device_peak_bytes"]
                and other["spearman"] >= candidate["spearman"]
                and other["overlap_1pct"] >= candidate["overlap_1pct"]
            )
            strict = any(
                other[left] < candidate[left] for left in ("corrected_nmse", "wall_seconds", "device_peak_bytes")
            ) or any(other[left] > candidate[left] for left in ("spearman", "overlap_1pct"))
            if no_worse and strict:
                dominated.append(str(candidate["row_id"]))
                break
    return {"status": "PASS" if not reasons else "BLOCKED", "rows": normalized, "dominated": sorted(set(dominated))}, reasons


def _capacity(rows: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(inputs, Mapping):
        return {
            "status": "BLOCKED",
            "steps": {"stage4": 0, "stage5": 0, "total": 0},
            "forecasts": {},
            "warnings": ["CAPACITY_INPUTS_REQUIRED"],
        }
    required_hashes = ("capacity_evidence_hash", "ulimit_evidence_hash")
    if any(not isinstance(inputs.get(name), str) or _SHA256.fullmatch(str(inputs.get(name))) is None for name in required_hashes):
        return {
            "status": "BLOCKED",
            "steps": {"stage4": 0, "stage5": 0, "total": 0},
            "forecasts": {},
            "warnings": ["CAPACITY_OR_ULIMIT_EVIDENCE_HASH_REQUIRED"],
        }
    if inputs.get("ulimit_nofile_soft") != 1024:
        return {
            "status": "BLOCKED",
            "steps": {"stage4": 0, "stage5": 0, "total": 0},
            "forecasts": {},
            "warnings": ["ULIMIT_NOFILE_SOFT_MUST_BE_1024"],
        }
    for field in ("disk_free_bytes", "inode_free"):
        value = inputs.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return {
                "status": "BLOCKED",
                "steps": {"stage4": 0, "stage5": 0, "total": 0},
                "forecasts": {},
                "warnings": [f"CAPACITY_{field.upper()}_REQUIRED"],
            }
    try:
        steps = int(inputs.get("stage4_steps", 0))
        stage5_steps = int(inputs.get("stage5_steps", 0))
    except (TypeError, ValueError):
        return {
            "status": "BLOCKED",
            "steps": {"stage4": 0, "stage5": 0, "total": 0},
            "forecasts": {},
            "warnings": ["CAPACITY_STAGE_STEPS_INVALID"],
        }
    online = _group(rows, semantic="online_training_incremental_cost")
    if steps <= 0 or stage5_steps <= 0:
        return {
            "status": "BLOCKED",
            "steps": {"stage4": steps, "stage5": stage5_steps, "total": max(0, steps) + max(0, stage5_steps)},
            "forecasts": {},
            "warnings": ["CAPACITY_STAGE_STEPS_REQUIRED"],
        }
    total_steps = steps + stage5_steps
    forecasts: dict[str, Any] = {}
    for method in S29_METHODS:
        sample = [row for group in online.values() for row in group if row["method"] == method]
        if not sample:
            continue
        wall = statistics.median(float(row["wall_seconds"]) for row in sample)
        devices = statistics.median(int(row["device_count"]) for row in sample)
        output = statistics.median(int(row["output_bytes"]) for row in sample)
        forecasts[method] = {
            "observed_wall_seconds_median": wall,
            "observed_wall_seconds_min": min(float(row["wall_seconds"]) for row in sample),
            "observed_wall_seconds_max": max(float(row["wall_seconds"]) for row in sample),
            "projected_stage4_a100_hours": wall * devices * steps / 3600.0,
            "projected_stage5_a100_hours": wall * devices * stage5_steps / 3600.0,
            "projected_total_a100_hours": wall * devices * total_steps / 3600.0,
            "projected_output_bytes": output * total_steps,
            "device_count": devices,
        }
    warnings: list[str] = []
    if isinstance(inputs, Mapping):
        budget = inputs.get("a100_hours_budget")
        if budget is not None and forecasts and any(value["projected_total_a100_hours"] > float(budget) for value in forecasts.values()):
            warnings.append("A100_HOURS_BUDGET_EXCEEDED")
        byte_budget = inputs.get("output_bytes_budget")
        if byte_budget is not None and forecasts and any(value["projected_output_bytes"] > int(byte_budget) for value in forecasts.values()):
            warnings.append("OUTPUT_BYTES_BUDGET_EXCEEDED")
    return {"status": "PASS", "steps": {"stage4": steps, "stage5": stage5_steps, "total": total_steps}, "forecasts": forecasts, "warnings": warnings, "capacity_evidence_hash": inputs["capacity_evidence_hash"], "ulimit_evidence_hash": inputs["ulimit_evidence_hash"], "ulimit_nofile_soft": inputs["ulimit_nofile_soft"]}


def run_s209_g27a(
    *,
    matrix: Mapping[str, Any] | str | Path,
    g24b_gate: Mapping[str, Any] | str | Path,
    raw_manifest: Mapping[str, Any] | str | Path,
    g25_gate: Mapping[str, Any] | str | Path | None = None,
    cost_observations: Mapping[str, Any],
    health_snapshot: Mapping[str, Any],
    single_gpu_anchor: Mapping[str, Any] | None,
    four_gpu_anchor: Mapping[str, Any] | None,
    shared_attribution_cross_check: Mapping[str, Any] | None,
    accuracy_rows: Sequence[Mapping[str, Any]] = (),
    capacity_inputs: Mapping[str, Any] | None = None,
    run_id: str = "s209-g27a",
    cost_io_quiescent: bool | None = None,
    checked_at: str | None = None,
    output_root: str | Path | None = None,
    measurement_plan: Mapping[str, Any] | str | Path | None = None,
    matrix_ref: str = "g24b-matrix.json",
    raw_manifest_ref: str = "raw-results-manifest.json",
) -> dict[str, Any]:
    """Reduce supplied formal profiler rows and produce the G2.7a decision.

    This function performs no process or device-management operation.  It is
    safe to run against CPU fixtures; such fixtures remain blocked unless all
    formal evidence fields (including four-card data) are present.
    """

    frozen = bind_s209_inputs(
        matrix=matrix,
        g24b_gate=g24b_gate,
        raw_manifest=raw_manifest,
        g25_gate=g25_gate,
        require_g25=g25_gate is not None,
        matrix_ref=matrix_ref,
        raw_manifest_ref=raw_manifest_ref,
    )
    _id(run_id, field="s29.run_id")
    validated_measurement_plan = _validate_measurement_plan(measurement_plan, frozen=frozen) if measurement_plan is not None else None
    if not isinstance(cost_observations, Mapping) or set(cost_observations) != set(S29_COST_SEMANTICS):
        raise S29G27ABlocked("COST_SEMANTICS_EXACTLY_THREE_REQUIRED")
    reducer = StrictS29Reducer(frozen, run_id=run_id)
    metadata: dict[str, Mapping[str, Any]] = {}
    for semantic in S29_COST_SEMANTICS:
        meta, rows = _semantic_rows(cost_observations[semantic], semantic=semantic)
        metadata[semantic] = meta
        for row in rows:
            reducer.add(row, semantic=semantic, metadata=meta)
    reduced = reducer.seal()
    expected_io = cost_io_quiescent
    if expected_io is None:
        values = [row.get("cost_io_quiescent") for row in reduced]
        expected_io = bool(values) and all(value is True for value in values)
    if type(expected_io) is not bool:
        raise S29G27ABlocked("cost_io_quiescent:BOOL_REQUIRED")
    blockers, provisional = _health_reasons(health_snapshot, expected_io=expected_io)
    health_gpu_uuids = set(health_snapshot.get("gpu_uuids", ())) if isinstance(health_snapshot, Mapping) else set()
    health_gpu_class = health_snapshot.get("gpu_class") if isinstance(health_snapshot, Mapping) else None
    health_inventory_hash = health_snapshot.get("inventory_artifact_hash") if isinstance(health_snapshot, Mapping) else None
    health_source_sha256 = health_snapshot.get("inventory_source_sha256") if isinstance(health_snapshot, Mapping) else None
    for row in reduced:
        if row["cost_io_quiescent"] is not True:
            provisional.append(f"COST_IO_NOT_QUIESCENT:{row['semantic']}:{row['anchor_id']}")
        if row["health_ok"] is not True:
            blockers.append(f"OBSERVATION_HEALTH_FAILED:{row['semantic']}:{row['anchor_id']}")
        if health_gpu_uuids and row["gpu_uuid"] not in health_gpu_uuids:
            blockers.append(f"OBSERVATION_GPU_NOT_IN_HEALTH_SNAPSHOT:{row['semantic']}:{row['anchor_id']}")
        if row.get("gpu_class") is not None and row.get("gpu_class") != health_gpu_class:
            blockers.append(f"OBSERVATION_GPU_CLASS_DRIFT:{row['semantic']}:{row['anchor_id']}")
        if row.get("inventory_artifact_hash") != health_inventory_hash or row.get("inventory_source_sha256") != health_source_sha256:
            blockers.append(f"OBSERVATION_GPU_INVENTORY_IDENTITY_DRIFT:{row['semantic']}:{row['anchor_id']}")
    blockers.extend(_anchor_reasons(single_gpu_anchor, frozen=frozen, expected_devices=1))
    four_reasons = _anchor_reasons(four_gpu_anchor, frozen=frozen, expected_devices=4)
    if four_reasons:
        provisional.append("FOUR_CARD_EVIDENCE_MISSING_OR_INVALID")
        blockers.extend(four_reasons)
    consistency = _consistency(reduced, frozen)
    if not consistency["all_pass"]:
        blockers.append("COST_SYSTEM_CONSISTENCY_FAILED")
    online_blockers, online_aggregates, online_ratios = _online_checks(reduced, metadata["online_training_incremental_cost"])
    blockers.extend(online_blockers)
    blockers.extend(_crosscheck(reduced, shared_attribution_cross_check))
    pareto, pareto_reasons = _pareto(reduced, accuracy_rows)
    blockers.extend(pareto_reasons)
    capacity = _capacity(reduced, capacity_inputs)
    if capacity.get("status") != "PASS":
        blockers.extend(str(item) for item in capacity.get("warnings", ["CAPACITY_EVIDENCE_INVALID"]))
    only_provisional = bool(provisional) and not blockers
    if not blockers and provisional:
        # A missing four-card anchor is deliberately not a formal PASS.  The
        # wire GateRecord remains BLOCKED because the shared Gate contract has
        # no PROVISIONAL status; the report exposes the clearer PROVISIONAL
        # state for downstream S2.10.
        only_provisional = True
    report_status = "PASS" if not blockers and not provisional else ("PROVISIONAL" if only_provisional else "BLOCKED")
    now = checked_at or datetime.now(timezone.utc).isoformat()
    gate_status = GateStatus.PASS if report_status == "PASS" else GateStatus.BLOCKED
    reasons = tuple(sorted(set(blockers + provisional)))
    measured = {
        "matrix_hash": frozen.matrix_hash,
        "raw_manifest_hash": frozen.raw_manifest_hash,
        "raw_run_id": frozen.raw_run_id,
        "g25_gate_hash": frozen.g25_gate_hash,
        "s29_run_id": run_id,
        "cost_io_quiescent": expected_io,
        "record_count": len(reduced),
        "online_ratios": online_ratios,
        "four_card_complete": not bool(four_reasons),
        "inventory_artifact_hash": health_snapshot.get("inventory_artifact_hash") if isinstance(health_snapshot, Mapping) else None,
        "inventory_source_sha256": health_snapshot.get("inventory_source_sha256") if isinstance(health_snapshot, Mapping) else None,
        "capacity_evidence_hash": capacity.get("capacity_evidence_hash"),
        "ulimit_evidence_hash": capacity.get("ulimit_evidence_hash"),
        "ulimit_nofile_soft": capacity.get("ulimit_nofile_soft"),
    }
    gate = GateRecord(
        gate_id="stage2.G2.7a",
        stage=2,
        status=gate_status,
        checked_at=now,
        measured=measured,
        threshold={"online_training_incremental_cost_ratio": S29_DECISION_RATIO, "shared_crosscheck_relative_difference": S29_CROSSCHECK_TOLERANCE, "four_gpu_required": True, "cost_io_quiescent": True},
        evidence_refs=(matrix_ref, raw_manifest_ref, f"s209/{run_id}/cost-system-validation.json"),
        reasons=reasons if gate_status is not GateStatus.PASS else (),
    )
    semantic_summary: dict[str, Any] = {}
    for semantic in S29_COST_SEMANTICS:
        subset = [row for row in reduced if row["semantic"] == semantic]
        semantic_summary[semantic] = {
            "defined": bool(subset),
            "observation_count": len(subset),
            "run_ids": sorted({str(row["run_id"]) for row in subset}),
            "method_set": sorted({str(row["method"]) for row in subset}),
            "wall_seconds_median": statistics.median(float(row["wall_seconds"]) for row in subset) if subset else None,
            "decision_eligible": semantic == "online_training_incremental_cost",
        }
    report: dict[str, Any] = {
        "schema_version": S29_SCHEMA,
        "task_id": S29_TASK_ID,
        "scope": "formal",
        "formal_eligible": report_status == "PASS",
        "status": report_status,
        "run_id": run_id,
        "frozen_inputs": frozen.to_dict(),
        "cost_semantics": semantic_summary,
        "cost_rows": [dict(row) for row in reduced],
        "health_snapshot": dict(health_snapshot),
        "single_gpu_anchor": dict(single_gpu_anchor) if isinstance(single_gpu_anchor, Mapping) else None,
        "four_gpu_anchor": dict(four_gpu_anchor) if isinstance(four_gpu_anchor, Mapping) else None,
        "cost_io_quiescent": expected_io,
        "consistency": consistency,
        "shared_attribution_cross_check": dict(shared_attribution_cross_check) if isinstance(shared_attribution_cross_check, Mapping) else None,
        "online_training_incremental_cost": {"aggregates": online_aggregates, "ratios": online_ratios, "decision_source": "online_training_incremental_cost"},
        "pareto": pareto,
        "capacity": capacity,
        "measurement_plan": validated_measurement_plan,
        "gate": gate.to_dict(),
        "reasons": list(reasons),
    }
    report["artifact_hash"] = canonical_json_hash(report)
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        write_canonical_json(root / "cost-system-validation.json", report)
        write_canonical_json(root / "g2.7a-gate.json", gate.to_dict())
    return report


# Descriptive aliases make the detached entry point convenient without adding
# a second implementation or a second formal identity.
orchestrate_s209_g27a = run_s209_g27a
validate_g27a = run_s209_g27a


__all__ = [
    "APPROVED_GPU_UUIDS",
    "EXCLUDED_GPU_UUID",
    "EXCLUDED_PCI",
    "S29_COST_SEMANTICS",
    "S29_CROSSCHECK_TOLERANCE",
    "S29_DECISION_RATIO",
    "S29_MEASUREMENT_PLAN_SCHEMA",
    "S29FrozenInputs",
    "S29G27ABlocked",
    "StrictS29Reducer",
    "bind_s209_inputs",
    "orchestrate_s209_g27a",
    "prepare_s209_measurement_plan",
    "run_s209_g27a",
    "validate_g27a",
]
