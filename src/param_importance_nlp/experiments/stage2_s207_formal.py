"""Fail-closed formal S2.7 orchestration and G2.5 reduction.

This module is deliberately a S2.7-only control plane.  It consumes already
published S2.3/S2.4/S2.6 evidence and per-repetition raw records; it never
creates draws, loads a provider, starts a process, or upgrades fixture
evidence.  The caller can therefore use it for detached preparation and
recovery without giving the reducer authority to silently continue a partial
formal sweep.

The S2.6 implementation publishes the formal matrix and confirmatory mapping
under ``stage2-formal-*`` schemas.  S2.7 binds those artifacts here and adds
the information those producer artifacts intentionally do not contain:
checkpoint/reference identities for each cell, exact expected unit IDs,
failure denominator semantics, raw artifact sealing, and the G2.5 gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from ..atomic import sha256_file
from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord, GateStatus
from .sampling import RepetitionMapping
from .stage2_s204_ids import EXPECTED_CELL_IDS


S27_PLAN_SCHEMA = "stage2-s27-six-cell-plan-v1"
S27_STATUS_SCHEMA = "stage2-s27-detached-status-v1"
S27_UNIT_SCHEMA = "stage2-s27-raw-unit-v1"
S27_RAW_MANIFEST_SCHEMA = "stage2-s27-sealed-raw-manifest-v1"
S27_SEAL_SCHEMA = "stage2-s27-sealed-marker-v1"
S27_GATE_SCHEMA = "stage2-s27-g25-gate-v1"
S27_TASK_ID = "stage2.07_main_sweep"
S27_SCOPE = "formal"
S27_STREAM = "confirmatory"
S27_M2_IDENTITY_TOLERANCE = 1e-12
S27_MATRIX_SCHEMA = "stage2-formal-pilot-matrix-freeze-v1"
S27_MAPPING_SCHEMA = "stage2-formal-confirmatory-mapping-v1"
S27_REFERENCE_TASK = "stage2.04_reference_target"
S27_CHECKPOINT_TASK = "stage2.03_assets_checkpoints_and_sampling"
S27_REQUIRED_GATES = ("stage2.G2.2", "stage2.G2.3", "stage2.G2.4a", "stage2.G2.4b")
S27_WAVE_ORDER = EXPECTED_CELL_IDS
# S2.6 keeps its historical dot anchor spelling in matrix/mapping objects,
# while G2.3/S2.7 use the canonical colon cell identity.
S27_ANCHOR_ORDER = tuple(cell.replace(":", ".", 1) for cell in EXPECTED_CELL_IDS)
S27_CORRECTED_DELTA_BINDING_FIELDS = frozenset(
    {
        "cell_id",
        "config_hash",
        "result_hash",
        "corrected_delta_sci_hash",
        "corrected_delta_sci_ref",
        "corrected_delta_sci_batch_sizes",
        "delta_sci_source",
    }
)
S27_CORRECTED_DELTA_BATCH_SIZES = (32, 64, 128, 256)
S27_CORRECTED_DELTA_SOURCE = "g23_output_derived_corrected_sidecar"
APPROVED_GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)
EXCLUDED_GPU_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
EXCLUDED_PCI = "0000:50:00.0"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,511}$")
_UNIT_STATUSES = ("SUCCESS", "FAILED")
_TERMINAL_STATUS = ("FAILED", "SEALED", "BLOCKED")
_STATUS_VALUES = ("PREPARED", "RUNNING", "PAUSED", "FAILED", "SEALED", "BLOCKED")
_METHOD_NAMES = ("raw", "double", "u_m2")


class S27PreparationBlocked(RuntimeError):
    """Raised when formal S2.7 identity or upstream qualification is absent."""


class S27G25Blocked(RuntimeError):
    """Raised when the raw denominator cannot support a strict G2.5 gate."""


def _text(value: object, *, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field}:NON_EMPTY_TRIMMED_STRING_REQUIRED")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field}:CONTROL_CHARACTER")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{field}:UNSAFE_VALUE")
    return value


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field}:SHA256_REQUIRED")
    return value


def _ref(value: object, *, field: str) -> str:
    return _text(value, field=field, pattern=_SAFE_REF)


def _positive_int(value: object, *, field: str, zero_ok: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value < 0 if zero_ok else value <= 0):
        raise ValueError(f"{field}:POSITIVE_INTEGER_REQUIRED")
    return value


def _finite(value: object, *, field: str = "value") -> None:
    """Reject NaN/Inf recursively before an artifact can be hashed."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field}:NONFINITE")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, field=f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field}:NON_STRING_KEY")
            _finite(item, field=f"{field}.{key}")
        return
    raise ValueError(f"{field}:NOT_JSON_VALUE")


def _verify_artifact(value: Mapping[str, object], *, field: str) -> str:
    declared = _sha(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    _finite(body, field=field)
    if declared != canonical_json_hash(body):
        raise ValueError(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _verify_gate(value: Mapping[str, object], *, gate_id: str, field: str) -> str:
    try:
        gate = GateRecord.from_mapping(dict(value))
    except Exception as error:  # keep the public blocker stable across contract versions
        raise S27PreparationBlocked(f"{field}:GATE_RECORD_INVALID") from error
    if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
        raise S27PreparationBlocked(f"{field}:PASS_REQUIRED")
    return gate.artifact_hash


def _safe_hash_without_artifact(value: Mapping[str, object], *, field: str) -> str:
    return _verify_artifact(value, field=field)


def _map_ref(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field}:REFERENCE_REQUIRED")
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"{field}:UNSAFE_REFERENCE")
    return _ref(value, field=field)


def _anchor_to_cell_id(value: object, *, field: str = "anchor_id") -> str:
    """Map one exact S2.6 dot anchor to the canonical S2.7 colon cell."""

    if not isinstance(value, str) or value not in S27_ANCHOR_ORDER:
        raise ValueError(f"{field}:S206_DOT_ANCHOR_REQUIRED")
    return EXPECTED_CELL_IDS[S27_ANCHOR_ORDER.index(value)]


def _validate_corrected_delta_binding(
    value: object,
    *,
    expected_cell_id: str,
    field: str,
) -> dict[str, object]:
    """Validate the exact per-cell S2.6/G2.3 sidecar binding."""

    if not isinstance(value, Mapping) or set(value) != S27_CORRECTED_DELTA_BINDING_FIELDS:
        raise ValueError(f"{field}:FIELDS_INVALID")
    binding = dict(value)
    if binding.get("cell_id") != expected_cell_id:
        raise ValueError(f"{field}:CELL_IDENTITY_MISMATCH")
    for name in ("config_hash", "result_hash", "corrected_delta_sci_hash"):
        _sha(binding.get(name), field=f"{field}.{name}")
    ref = binding.get("corrected_delta_sci_ref")
    if not isinstance(ref, str) or not ref or "\\" in ref or ref.startswith("/") or ref.endswith("/"):
        raise ValueError(f"{field}.corrected_delta_sci_ref:INVALID")
    ref_path = PurePosixPath(ref)
    if ref_path.is_absolute() or any(part in {"", ".", ".."} for part in ref_path.parts):
        raise ValueError(f"{field}.corrected_delta_sci_ref:INVALID")
    expected_suffix = f"g2.3-corrected-delta-sci/{binding['corrected_delta_sci_hash']}.json"
    if not ref.endswith(expected_suffix):
        raise ValueError(f"{field}.corrected_delta_sci_ref:CONTENT_ADDRESS_INVALID")
    if binding.get("corrected_delta_sci_batch_sizes") != list(S27_CORRECTED_DELTA_BATCH_SIZES):
        raise ValueError(f"{field}.corrected_delta_sci_batch_sizes:INVALID")
    if binding.get("delta_sci_source") != S27_CORRECTED_DELTA_SOURCE:
        raise ValueError(f"{field}.delta_sci_source:INVALID")
    return binding


@dataclass(frozen=True, slots=True)
class S27MappingUnit:
    """Immutable expected identity for one confirmatory repetition."""

    unit_id: str
    cell_id: str
    repetition_id: str
    batch_size: int
    microbatch_count: int
    draw_ids: tuple[str, ...]
    sample_ids: tuple[object, ...]
    mapping_hash: str

    def __post_init__(self) -> None:
        _text(self.unit_id, field="unit_id", pattern=_SAFE_ID)
        _text(self.cell_id, field="cell_id", pattern=_SAFE_ID)
        _text(self.repetition_id, field="repetition_id", pattern=_SAFE_ID)
        _positive_int(self.batch_size, field="batch_size")
        if self.microbatch_count <= 2 or self.batch_size % self.microbatch_count:
            raise ValueError("microbatch_count must be primary M>2 and divide B")
        if len(self.draw_ids) != self.batch_size or len(self.sample_ids) != self.batch_size:
            raise ValueError("mapping draw/sample count must equal B")
        if len(set(self.draw_ids)) != len(self.draw_ids):
            raise ValueError("mapping draw IDs must be unique within a repetition")
        if any(not isinstance(item, str) or not item for item in self.draw_ids):
            raise ValueError("mapping draw IDs must be non-empty strings")
        _sha(self.mapping_hash, field="mapping_hash")

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "cell_id": self.cell_id,
            "repetition_id": self.repetition_id,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "draw_ids": list(self.draw_ids),
            "sample_ids": list(self.sample_ids),
            "mapping_hash": self.mapping_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "S27MappingUnit":
        required = {"unit_id", "cell_id", "repetition_id", "batch_size", "microbatch_count", "draw_ids", "sample_ids", "mapping_hash"}
        if set(value) != required or not isinstance(value.get("draw_ids"), list) or not isinstance(value.get("sample_ids"), list):
            raise ValueError("mapping unit fields mismatch")
        return cls(
            unit_id=value["unit_id"], cell_id=value["cell_id"], repetition_id=value["repetition_id"],
            batch_size=value["batch_size"], microbatch_count=value["microbatch_count"],
            draw_ids=tuple(value["draw_ids"]), sample_ids=tuple(value["sample_ids"]), mapping_hash=value["mapping_hash"],
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class S27FrozenInputs:
    """G2.4b matrix and formal mapping binding consumed by S2.7."""

    matrix_ref: str
    matrix_hash: str
    mapping_ref: str
    mapping_hash: str
    g24b_gate_ref: str
    g24b_gate_hash: str
    sampling_plan_hash: str
    batch_size: int
    microbatch_count: int
    repetitions: int
    completion_denominator: int
    max_failure_fraction: float
    cost_required: bool
    units: tuple[S27MappingUnit, ...]

    def __post_init__(self) -> None:
        _map_ref(self.matrix_ref, field="matrix_ref")
        _sha(self.matrix_hash, field="matrix_hash")
        _map_ref(self.mapping_ref, field="mapping_ref")
        _sha(self.mapping_hash, field="mapping_hash")
        _map_ref(self.g24b_gate_ref, field="g24b_gate_ref")
        _sha(self.g24b_gate_hash, field="g24b_gate_hash")
        _sha(self.sampling_plan_hash, field="sampling_plan_hash")
        if self.batch_size not in (32, 64, 128, 256) or self.microbatch_count not in (4, 8, 16, 32):
            raise ValueError("formal B/M candidate outside preregistered set")
        if self.batch_size % self.microbatch_count:
            raise ValueError("formal M must divide B")
        if not 200 <= self.repetitions <= 1000:
            raise ValueError("formal R must be in [200,1000]")
        if self.completion_denominator != len(EXPECTED_CELL_IDS) * self.repetitions:
            raise ValueError("completion denominator must cover six cells")
        if isinstance(self.max_failure_fraction, bool) or not isinstance(self.max_failure_fraction, float):
            raise TypeError("max_failure_fraction must be a float")
        if not 0.0 <= self.max_failure_fraction < 1.0:
            raise ValueError("max_failure_fraction must be in [0,1)")
        if type(self.cost_required) is not bool:
            raise TypeError("cost_required must be bool")
        if len(self.units) != self.completion_denominator:
            raise ValueError("mapping unit denominator mismatch")
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("mapping unit IDs must be globally unique")
        if set(unit.cell_id for unit in self.units) != set(EXPECTED_CELL_IDS):
            raise ValueError("mapping must cover all six cells")
        for unit in self.units:
            if unit.batch_size != self.batch_size or unit.microbatch_count != self.microbatch_count:
                raise ValueError("mapping B/M drift from frozen matrix")

    @property
    def expected_unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.units)

    @property
    def expected_draw_ids(self) -> tuple[str, ...]:
        return tuple(draw_id for unit in self.units for draw_id in unit.draw_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "matrix_ref": self.matrix_ref,
            "matrix_hash": self.matrix_hash,
            "mapping_ref": self.mapping_ref,
            "mapping_hash": self.mapping_hash,
            "g24b_gate_ref": self.g24b_gate_ref,
            "g24b_gate_hash": self.g24b_gate_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "repetitions": self.repetitions,
            "completion_denominator": self.completion_denominator,
            "max_failure_fraction": self.max_failure_fraction,
            "cost_required": self.cost_required,
            "unit_ids": list(self.expected_unit_ids),
            "expected_draw_id_count": len(self.expected_draw_ids),
            "expected_draw_id_hash": canonical_json_hash(list(self.expected_draw_ids)),
            "units": [unit.to_dict() for unit in self.units],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "S27FrozenInputs":
        required = {"matrix_ref", "matrix_hash", "mapping_ref", "mapping_hash", "g24b_gate_ref", "g24b_gate_hash", "sampling_plan_hash", "batch_size", "microbatch_count", "repetitions", "completion_denominator", "max_failure_fraction", "cost_required", "unit_ids", "expected_draw_id_count", "expected_draw_id_hash", "units"}
        if set(value) != required or not isinstance(value.get("units"), list):
            raise ValueError("frozen input fields mismatch")
        units = tuple(S27MappingUnit.from_mapping(item) for item in value["units"] if isinstance(item, Mapping))
        if len(units) != len(value["units"]):
            raise ValueError("frozen input units must be objects")
        result = cls(
            matrix_ref=value["matrix_ref"], matrix_hash=value["matrix_hash"], mapping_ref=value["mapping_ref"], mapping_hash=value["mapping_hash"],
            g24b_gate_ref=value["g24b_gate_ref"], g24b_gate_hash=value["g24b_gate_hash"], sampling_plan_hash=value["sampling_plan_hash"],
            batch_size=value["batch_size"], microbatch_count=value["microbatch_count"], repetitions=value["repetitions"], completion_denominator=value["completion_denominator"], max_failure_fraction=value["max_failure_fraction"], cost_required=value["cost_required"], units=units,
        )  # type: ignore[arg-type]
        if value["unit_ids"] != list(result.expected_unit_ids) or value["expected_draw_id_count"] != len(result.expected_draw_ids) or value["expected_draw_id_hash"] != canonical_json_hash(list(result.expected_draw_ids)):
            raise ValueError("frozen input derived identity mismatch")
        return result


@dataclass(frozen=True, slots=True)
class S27CellPlan:
    cell_id: str
    model_id: str
    training_stage: str
    checkpoint_ref: str
    checkpoint_hash: str
    checkpoint_id: str
    reference_ref: str
    reference_hash: str
    reference_gate_ref: str
    reference_gate_hash: str
    expected_unit_ids: tuple[str, ...]
    assigned_gpu_uuid: str
    corrected_delta_sci_binding: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.cell_id not in EXPECTED_CELL_IDS:
            raise ValueError(f"unknown S2.7 cell: {self.cell_id}")
        _text(self.model_id, field="model_id", pattern=_SAFE_ID)
        _text(self.training_stage, field="training_stage", pattern=_SAFE_ID)
        _map_ref(self.checkpoint_ref, field="checkpoint_ref")
        _sha(self.checkpoint_hash, field="checkpoint_hash")
        _text(self.checkpoint_id, field="checkpoint_id", pattern=_SAFE_ID)
        _map_ref(self.reference_ref, field="reference_ref")
        _sha(self.reference_hash, field="reference_hash")
        _map_ref(self.reference_gate_ref, field="reference_gate_ref")
        _sha(self.reference_gate_hash, field="reference_gate_hash")
        if not self.expected_unit_ids or len(set(self.expected_unit_ids)) != len(self.expected_unit_ids):
            raise ValueError("cell expected unit IDs must be non-empty and unique")
        if self.assigned_gpu_uuid not in APPROVED_GPU_UUIDS or self.assigned_gpu_uuid == EXCLUDED_GPU_UUID:
            raise ValueError("cell assigned to unapproved GPU")
        if self.corrected_delta_sci_binding is not None:
            _validate_corrected_delta_binding(
                self.corrected_delta_sci_binding,
                expected_cell_id=self.cell_id,
                field=f"corrected_delta_sci_binding.{self.cell_id}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "model_id": self.model_id,
            "training_stage": self.training_stage,
            "checkpoint_ref": self.checkpoint_ref,
            "checkpoint_hash": self.checkpoint_hash,
            "checkpoint_id": self.checkpoint_id,
            "reference_ref": self.reference_ref,
            "reference_hash": self.reference_hash,
            "reference_gate_ref": self.reference_gate_ref,
            "reference_gate_hash": self.reference_gate_hash,
            "expected_unit_ids": list(self.expected_unit_ids),
            "assigned_gpu_uuid": self.assigned_gpu_uuid,
            "corrected_delta_sci_binding": (
                None
                if self.corrected_delta_sci_binding is None
                else dict(self.corrected_delta_sci_binding)
            ),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "S27CellPlan":
        required = {"cell_id", "model_id", "training_stage", "checkpoint_ref", "checkpoint_hash", "checkpoint_id", "reference_ref", "reference_hash", "reference_gate_ref", "reference_gate_hash", "expected_unit_ids", "assigned_gpu_uuid", "corrected_delta_sci_binding"}
        if set(value) != required or not isinstance(value.get("expected_unit_ids"), list):
            raise ValueError("cell plan fields mismatch")
        binding_value = value.get("corrected_delta_sci_binding")
        if not isinstance(binding_value, Mapping):
            raise ValueError("cell plan corrected delta binding required")
        return cls(
            cell_id=value["cell_id"], model_id=value["model_id"], training_stage=value["training_stage"], checkpoint_ref=value["checkpoint_ref"], checkpoint_hash=value["checkpoint_hash"], checkpoint_id=value["checkpoint_id"], reference_ref=value["reference_ref"], reference_hash=value["reference_hash"], reference_gate_ref=value["reference_gate_ref"], reference_gate_hash=value["reference_gate_hash"], expected_unit_ids=tuple(value["expected_unit_ids"]), assigned_gpu_uuid=value["assigned_gpu_uuid"],
            corrected_delta_sci_binding=dict(binding_value),
        )  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class S27Plan:
    plan_id: str
    plan_ref: str
    frozen_inputs: S27FrozenInputs
    cells: tuple[S27CellPlan, ...]
    source_artifact_refs: tuple[str, ...]
    approved_gpu_uuids: tuple[str, ...] = APPROVED_GPU_UUIDS
    excluded_pci: str = EXCLUDED_PCI
    status: str = "READY"
    formal_eligible: bool = True

    def __post_init__(self) -> None:
        _text(self.plan_id, field="plan_id", pattern=_SAFE_ID)
        _map_ref(self.plan_ref, field="plan_ref")
        if tuple(cell.cell_id for cell in self.cells) != EXPECTED_CELL_IDS:
            raise ValueError("S2.7 cells must be exactly six cells in canonical order")
        for cell in self.cells:
            expected = tuple(unit.unit_id for unit in self.frozen_inputs.units if unit.cell_id == cell.cell_id)
            if cell.expected_unit_ids != expected:
                raise ValueError(f"cell expected units mismatch: {cell.cell_id}")
        if tuple(self.approved_gpu_uuids) != APPROVED_GPU_UUIDS:
            raise ValueError("approved GPU allowlist drift")
        if self.excluded_pci != EXCLUDED_PCI:
            raise ValueError("excluded PCI drift")
        if self.status != "READY" or self.formal_eligible is not True:
            raise S27PreparationBlocked("S27_PLAN_NOT_FORMAL_READY")
        refs = tuple(_map_ref(item, field="source_artifact_refs") for item in self.source_artifact_refs)
        if len(refs) != len(set(refs)):
            raise ValueError("source artifact refs must be unique")

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    @property
    def expected_unit_ids(self) -> tuple[str, ...]:
        return self.frozen_inputs.expected_unit_ids

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": S27_PLAN_SCHEMA,
            "task_id": S27_TASK_ID,
            "plan_id": self.plan_id,
            "plan_ref": self.plan_ref,
            "scope": S27_SCOPE,
            "stream": S27_STREAM,
            "status": self.status,
            "formal_eligible": self.formal_eligible,
            "frozen_inputs": self.frozen_inputs.to_dict(),
            "cells": [item.to_dict() for item in self.cells],
            "corrected_delta_sci_bindings": [
                None
                if item.corrected_delta_sci_binding is None
                else dict(item.corrected_delta_sci_binding)
                for item in self.cells
            ],
            "corrected_delta_sci_bindings_hash": canonical_json_hash({
                "bindings": [
                    None
                    if item.corrected_delta_sci_binding is None
                    else dict(item.corrected_delta_sci_binding)
                    for item in self.cells
                ]
            }),
            "source_artifact_refs": list(self.source_artifact_refs),
            "approved_gpu_uuids": list(self.approved_gpu_uuids),
            "excluded_pci": self.excluded_pci,
            "wave_order": list(S27_WAVE_ORDER),
            "expected_unit_count": len(self.expected_unit_ids),
            "expected_sample_budget": sum(unit.batch_size for unit in self.frozen_inputs.units),
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "S27Plan":
        required = {"schema_version", "task_id", "plan_id", "plan_ref", "scope", "stream", "status", "formal_eligible", "frozen_inputs", "cells", "corrected_delta_sci_bindings", "corrected_delta_sci_bindings_hash", "source_artifact_refs", "approved_gpu_uuids", "excluded_pci", "wave_order", "expected_unit_count", "expected_sample_budget", "artifact_hash"}
        if set(value) != required or value.get("schema_version") != S27_PLAN_SCHEMA or value.get("task_id") != S27_TASK_ID or value.get("scope") != S27_SCOPE or value.get("stream") != S27_STREAM:
            raise ValueError("S2.7 plan fields or identity mismatch")
        declared = _sha(value.get("artifact_hash"), field="s27_plan.artifact_hash")
        if declared != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
            raise ValueError("S2.7 plan artifact hash mismatch")
        if not isinstance(value.get("frozen_inputs"), Mapping) or not isinstance(value.get("cells"), list) or not isinstance(value.get("corrected_delta_sci_bindings"), list) or not isinstance(value.get("source_artifact_refs"), list) or not isinstance(value.get("approved_gpu_uuids"), list):
            raise TypeError("S2.7 plan nested fields invalid")
        frozen = S27FrozenInputs.from_mapping(value["frozen_inputs"])
        cells = tuple(S27CellPlan.from_mapping(item) for item in value["cells"] if isinstance(item, Mapping))
        if len(cells) != len(value["cells"]):
            raise ValueError("S2.7 plan cells must be objects")
        bindings = value["corrected_delta_sci_bindings"]
        if len(bindings) != len(cells) or any(not isinstance(item, Mapping) for item in bindings):
            raise ValueError("S2.7 plan corrected delta bindings invalid")
        if value["corrected_delta_sci_bindings_hash"] != canonical_json_hash({"bindings": bindings}):
            raise ValueError("S2.7 plan corrected delta bindings hash mismatch")
        for cell, binding in zip(cells, bindings):
            if cell.corrected_delta_sci_binding != dict(binding):
                raise ValueError(f"S2.7 plan corrected delta binding drift: {cell.cell_id}")
        result = cls(plan_id=value["plan_id"], plan_ref=value["plan_ref"], frozen_inputs=frozen, cells=cells, source_artifact_refs=tuple(value["source_artifact_refs"]), approved_gpu_uuids=tuple(value["approved_gpu_uuids"]), excluded_pci=value["excluded_pci"], status=value["status"], formal_eligible=value["formal_eligible"])  # type: ignore[arg-type]
        if value["wave_order"] != list(S27_WAVE_ORDER) or value["expected_unit_count"] != len(result.expected_unit_ids) or value["expected_sample_budget"] != sum(item.batch_size for item in frozen.units) or declared != result.artifact_hash:
            raise ValueError("S2.7 plan derived identity mismatch")
        return result


def _extract_checkpoint(row: Mapping[str, object], *, cell_id: str) -> tuple[str, str, str]:
    ref = row.get("checkpoint_ref", row.get("checkpoint_manifest_ref"))
    digest = row.get("checkpoint_hash", row.get("manifest_sha256"))
    checkpoint_id = row.get("checkpoint_id")
    if not isinstance(ref, str) or not isinstance(digest, str) or not isinstance(checkpoint_id, str):
        raise S27PreparationBlocked(f"checkpoint.{cell_id}:IDENTITY_INCOMPLETE")
    try:
        return _map_ref(ref, field=f"checkpoint.{cell_id}.ref"), _sha(digest, field=f"checkpoint.{cell_id}.hash"), _text(checkpoint_id, field=f"checkpoint.{cell_id}.id", pattern=_SAFE_ID)
    except ValueError as error:
        raise S27PreparationBlocked(str(error)) from error


def _extract_reference(row: Mapping[str, object], *, cell_id: str) -> tuple[str, str, str, str]:
    try:
        ref = _map_ref(row.get("reference_ref"), field=f"reference.{cell_id}.ref")
        digest = _sha(row.get("reference_hash"), field=f"reference.{cell_id}.hash")
        gate_ref = _map_ref(row.get("gate_ref", row.get("reference_gate_ref")), field=f"reference.{cell_id}.gate_ref")
        gate_hash = _sha(row.get("gate_hash", row.get("reference_gate_hash")), field=f"reference.{cell_id}.gate_hash")
    except ValueError as error:
        raise S27PreparationBlocked(str(error)) from error
    if row.get("task_id", S27_REFERENCE_TASK) != S27_REFERENCE_TASK:
        raise S27PreparationBlocked(f"reference.{cell_id}:WRONG_PRODUCER")
    if row.get("gate_id", "stage2.G2.3") != "stage2.G2.3" or row.get("gate_status", "PASS") != "PASS":
        raise S27PreparationBlocked(f"reference.{cell_id}:G23_PASS_REQUIRED")
    if row.get("scope", "formal") != "formal" or row.get("formal_eligible", True) is not True:
        raise S27PreparationBlocked(f"reference.{cell_id}:FORMAL_REFERENCE_REQUIRED")
    if row.get("independent", True) is not True:
        raise S27PreparationBlocked(f"reference.{cell_id}:INDEPENDENT_REFERENCE_REQUIRED")
    return ref, digest, gate_ref, gate_hash


def _mapping_units(mapping: Mapping[str, object], *, matrix_hash: str, mapping_hash: str, batch_size: int, microbatch_count: int, repetitions: int, sampling_plan_hash: str) -> tuple[S27MappingUnit, ...]:
    cells = mapping.get("cells")
    if not isinstance(cells, list) or tuple(item.get("anchor_id") for item in cells if isinstance(item, Mapping)) != S27_ANCHOR_ORDER:
        raise S27PreparationBlocked("confirmatory mapping must cover six cells in canonical order")
    pilot_ids = mapping.get("pilot_draw_ids")
    if pilot_ids is None:
        # The S2.6 formal mapping intentionally publishes the pilot namespace
        # as count+hash rather than replaying pilot IDs.  Require that compact
        # audit identity instead of pretending the namespace was inspected.
        pilot_count = mapping.get("pilot_draw_id_count")
        pilot_hash = mapping.get("pilot_draw_id_hash")
        if isinstance(pilot_count, bool) or not isinstance(pilot_count, int) or pilot_count < 0:
            raise S27PreparationBlocked("pilot draw audit count missing")
        if not isinstance(pilot_hash, str) or _SHA256.fullmatch(pilot_hash) is None:
            raise S27PreparationBlocked("pilot draw audit hash missing")
        pilot_ids = []
    if not isinstance(pilot_ids, list) or len(set(pilot_ids)) != len(pilot_ids):
        raise S27PreparationBlocked("pilot draw IDs must be unique")
    pilot_set = set(pilot_ids)
    all_draw_ids: list[str] = []
    all_positions: set[tuple[str, int]] = set()
    output: list[S27MappingUnit] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise S27PreparationBlocked("confirmatory mapping cell must be an object")
        try:
            cell_id = _anchor_to_cell_id(cell.get("anchor_id"))
        except ValueError as error:
            raise S27PreparationBlocked(f"mapping.{cell.get('anchor_id')}:DOT_ANCHOR_REQUIRED") from error
        rows = cell.get("mappings")
        if not isinstance(rows, list) or len(rows) != repetitions:
            raise S27PreparationBlocked(f"mapping.{cell_id}:REPETITION_DENOMINATOR_MISMATCH")
        for row in rows:
            if not isinstance(row, Mapping):
                raise S27PreparationBlocked(f"mapping.{cell_id}:REPETITION_NOT_OBJECT")
            try:
                repetition = RepetitionMapping.from_manifest(row)
            except Exception as error:
                raise S27PreparationBlocked(f"mapping.{cell_id}:REPETITION_INVALID") from error
            if repetition.batch_size != batch_size or repetition.m_values != (2, microbatch_count):
                raise S27PreparationBlocked(f"mapping.{cell_id}:{repetition.repetition_id}:B_M_MISMATCH")
            if any(draw.stream != S27_STREAM for draw in repetition.draws):
                raise S27PreparationBlocked(f"mapping.{cell_id}:{repetition.repetition_id}:NON_CONFIRMATORY_DRAW")
            positions = [(str(draw.stream), int(draw.position)) for draw in repetition.draws]
            if len(set(positions)) != len(positions) or all_positions.intersection(positions):
                raise S27PreparationBlocked(f"mapping.{cell_id}:{repetition.repetition_id}:STREAM_POSITION_COLLISION")
            all_positions.update(positions)
            draw_ids = tuple(str(draw.draw_id) for draw in repetition.draws)
            if pilot_set.intersection(draw_ids):
                raise S27PreparationBlocked("pilot/confirmatory draw namespace overlap")
            all_draw_ids.extend(draw_ids)
            output.append(
                S27MappingUnit(
                    unit_id=f"{cell_id}::{repetition.repetition_id}",
                    cell_id=cell_id,
                    repetition_id=repetition.repetition_id,
                    batch_size=batch_size,
                    microbatch_count=microbatch_count,
                    draw_ids=draw_ids,
                    sample_ids=tuple(draw.sample_id for draw in repetition.draws),
                    mapping_hash=repetition.digest,
                )
            )
    if len(all_draw_ids) != len(set(all_draw_ids)):
        raise S27PreparationBlocked("confirmatory draw IDs collide across cells")
    declared_draw_count = mapping.get("confirmatory_draw_id_count")
    declared_draw_hash = mapping.get("confirmatory_draw_id_hash")
    if declared_draw_count != len(all_draw_ids) or declared_draw_hash != canonical_json_hash(all_draw_ids):
        raise S27PreparationBlocked("confirmatory draw audit hash/count mismatch")
    # S2.6's formal manifest names this aggregate ``sample_id_collision_count``;
    # accept that frozen producer spelling without inventing a new mapping or
    # weakening the collision audit.  The legacy alias remains readable for
    # already-published candidate fixtures.
    observed_collision_count = mapping.get(
        "sample_collision_count",
        mapping.get("sample_id_collision_count"),
    )
    if observed_collision_count != sum(unit.batch_size - len(set(unit.sample_ids)) for unit in output):
        raise S27PreparationBlocked("sample reuse audit mismatch")
    if mapping.get("draw_id_unique") is not True or mapping.get("stream") != S27_STREAM:
        raise S27PreparationBlocked("confirmatory mapping uniqueness/stream marker invalid")
    if mapping.get("scope") != S27_SCOPE or mapping.get("formal_eligible") is not True:
        raise S27PreparationBlocked("confirmatory mapping is not formal eligible")
    if mapping.get("freeze_hash") != matrix_hash or mapping.get("sampling_plan_hash") != sampling_plan_hash:
        raise S27PreparationBlocked("confirmatory mapping is not bound to frozen matrix/sampling plan")
    if not isinstance(mapping.get("qualification_gate_hash"), str) or not _SHA256.fullmatch(str(mapping.get("qualification_gate_hash"))):
        raise S27PreparationBlocked("confirmatory mapping qualification hash missing")
    return tuple(output)


def prepare_s27_plan(
    *,
    plan_id: str,
    plan_ref: str,
    matrix_ref: str,
    matrix: Mapping[str, object],
    mapping_ref: str,
    mapping: Mapping[str, object],
    g24b_gate_ref: str,
    g24b_gate: Mapping[str, object],
    checkpoints: Mapping[str, Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
    source_artifact_refs: Sequence[str],
    max_failure_fraction: float,
    cost_required: bool = True,
) -> S27Plan:
    """Build a six-cell formal plan without creating any confirmatory draw."""

    if matrix.get("schema_version") != S27_MATRIX_SCHEMA:
        raise S27PreparationBlocked("G24B_MATRIX_SCHEMA_INVALID")
    try:
        matrix_hash = _verify_artifact(matrix, field="g24b_matrix")
    except ValueError as error:
        raise S27PreparationBlocked(str(error)) from error
    if matrix.get("scope") != S27_SCOPE or matrix.get("status") != "FORMAL_FROZEN" or matrix.get("formal_eligible") is not True:
        raise S27PreparationBlocked("G24B_FORMAL_FROZEN_MATRIX_REQUIRED")
    if tuple(matrix.get("anchor_ids", ())) != S27_ANCHOR_ORDER:
        raise S27PreparationBlocked("G24B_MATRIX_SIX_CELL_ORDER_INVALID")
    raw_bindings = matrix.get("corrected_delta_sci_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != len(EXPECTED_CELL_IDS):
        raise S27PreparationBlocked("G24B_CORRECTED_DELTA_BINDINGS_REQUIRED")
    bindings: list[dict[str, object]] = []
    for index, expected_cell_id in enumerate(EXPECTED_CELL_IDS):
        try:
            bindings.append(
                _validate_corrected_delta_binding(
                    raw_bindings[index],
                    expected_cell_id=expected_cell_id,
                    field=f"g24b_matrix.corrected_delta_sci_bindings[{index}]",
                )
            )
        except (IndexError, TypeError, ValueError) as error:
            raise S27PreparationBlocked("G24B_CORRECTED_DELTA_BINDING_INVALID") from error
    if matrix.get("corrected_delta_sci_bindings_hash") != canonical_json_hash({"bindings": bindings}):
        raise S27PreparationBlocked("G24B_CORRECTED_DELTA_BINDINGS_HASH_INVALID")
    if not isinstance(matrix.get("candidate_evaluations"), list) or not matrix.get("candidate_evaluations"):
        raise S27PreparationBlocked("G24B_MATRIX_CANDIDATE_TABLE_MISSING")
    try:
        matrix_ref = _map_ref(matrix_ref, field="matrix_ref")
        mapping_ref = _map_ref(mapping_ref, field="mapping_ref")
        gate_ref = _map_ref(g24b_gate_ref, field="g24b_gate_ref")
        sampling_plan_hash = _sha(matrix.get("sampling_plan_hash"), field="matrix.sampling_plan_hash")
        matrix_gate_hash = _sha(matrix.get("qualification_gate_hash"), field="matrix.qualification_gate_hash")
        batch_size = _positive_int(matrix.get("b_primary"), field="matrix.b_primary")
        microbatch_count = _positive_int(matrix.get("m_primary"), field="matrix.m_primary")
        repetitions = _positive_int(matrix.get("r_primary"), field="matrix.r_primary")
        denominator = _positive_int(matrix.get("completion_denominator"), field="matrix.completion_denominator")
    except (TypeError, ValueError) as error:
        raise S27PreparationBlocked(f"G24B_MATRIX_FIELDS_INVALID:{error}") from error
    gate_hash = _verify_gate(g24b_gate, gate_id="stage2.G2.4b", field="g24b_gate")
    if gate_hash != matrix_gate_hash:
        raise S27PreparationBlocked("G24B_GATE_HASH_MISMATCH")
    try:
        mapping_hash = _verify_artifact(mapping, field="confirmatory_mapping")
    except ValueError as error:
        raise S27PreparationBlocked(str(error)) from error
    if mapping.get("schema_version") != S27_MAPPING_SCHEMA:
        raise S27PreparationBlocked("CONFIRMATORY_MAPPING_SCHEMA_INVALID")
    if mapping.get("freeze_hash") != matrix_hash or mapping.get("qualification_gate_hash") != gate_hash or mapping.get("complete") is not True:
        raise S27PreparationBlocked("CONFIRMATORY_MAPPING_FREEZE_OR_GATE_MISMATCH")
    if mapping.get("sampling_plan_hash") != sampling_plan_hash:
        raise S27PreparationBlocked("CONFIRMATORY_MAPPING_SAMPLING_HASH_MISMATCH")
    units = _mapping_units(mapping, matrix_hash=matrix_hash, mapping_hash=mapping_hash, batch_size=batch_size, microbatch_count=microbatch_count, repetitions=repetitions, sampling_plan_hash=sampling_plan_hash)
    frozen = S27FrozenInputs(
        matrix_ref=matrix_ref,
        matrix_hash=matrix_hash,
        mapping_ref=mapping_ref,
        mapping_hash=mapping_hash,
        g24b_gate_ref=gate_ref,
        g24b_gate_hash=gate_hash,
        sampling_plan_hash=sampling_plan_hash,
        batch_size=batch_size,
        microbatch_count=microbatch_count,
        repetitions=repetitions,
        completion_denominator=denominator,
        max_failure_fraction=float(max_failure_fraction),
        cost_required=cost_required,
        units=units,
    )
    if set(checkpoints) != set(EXPECTED_CELL_IDS) or set(references) != set(EXPECTED_CELL_IDS):
        raise S27PreparationBlocked("S27_CHECKPOINT_REFERENCE_COVERAGE_INVALID")
    cells: list[S27CellPlan] = []
    for index, cell_id in enumerate(EXPECTED_CELL_IDS):
        checkpoint_ref, checkpoint_hash, checkpoint_id = _extract_checkpoint(checkpoints[cell_id], cell_id=cell_id)
        reference_ref, reference_hash, reference_gate_ref, reference_gate_hash = _extract_reference(references[cell_id], cell_id=cell_id)
        model_id, training_stage = cell_id.split(":", 1)
        expected_ids = tuple(unit.unit_id for unit in units if unit.cell_id == cell_id)
        cells.append(S27CellPlan(
            cell_id=cell_id,
            model_id=model_id,
            training_stage=training_stage,
            checkpoint_ref=checkpoint_ref,
            checkpoint_hash=checkpoint_hash,
            checkpoint_id=checkpoint_id,
            reference_ref=reference_ref,
            reference_hash=reference_hash,
            reference_gate_ref=reference_gate_ref,
            reference_gate_hash=reference_gate_hash,
            expected_unit_ids=expected_ids,
            assigned_gpu_uuid=APPROVED_GPU_UUIDS[index % len(APPROVED_GPU_UUIDS)],
            corrected_delta_sci_binding=bindings[index],
        ))
    return S27Plan(
        plan_id=_text(plan_id, field="plan_id", pattern=_SAFE_ID),
        plan_ref=_map_ref(plan_ref, field="plan_ref"),
        frozen_inputs=frozen,
        cells=tuple(cells),
        source_artifact_refs=tuple(_map_ref(item, field="source_artifact_refs") for item in source_artifact_refs),
    )


@dataclass(frozen=True, slots=True)
class S27RawUnit:
    """One immutable success or explicit failure record for the reducer."""

    unit_id: str
    cell_id: str
    repetition_id: str
    status: str
    attempt_id: str
    matrix_hash: str
    mapping_hash: str
    sampling_plan_hash: str
    checkpoint_hash: str
    reference_hash: str
    batch_size: int
    microbatch_count: int
    draw_ids: tuple[str, ...]
    sample_ids: tuple[object, ...]
    raw_artifact_ref: str
    raw_artifact_hash: str
    metrics: Mapping[str, object]
    methods: tuple[str, ...]
    m2_identity_max_abs: float | None
    mean_gradient_consistent: bool
    clamp_applied: bool
    clip_mode: str
    cost: Mapping[str, object]
    failure_code: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.unit_id, field="unit_id", pattern=_SAFE_ID)
        _text(self.cell_id, field="cell_id", pattern=_SAFE_ID)
        _text(self.repetition_id, field="repetition_id", pattern=_SAFE_ID)
        if self.status not in _UNIT_STATUSES:
            raise ValueError("raw unit status must be SUCCESS or FAILED")
        _text(self.attempt_id, field="attempt_id", pattern=_SAFE_ID)
        for name, value in (("matrix_hash", self.matrix_hash), ("mapping_hash", self.mapping_hash), ("sampling_plan_hash", self.sampling_plan_hash), ("checkpoint_hash", self.checkpoint_hash), ("reference_hash", self.reference_hash), ("raw_artifact_hash", self.raw_artifact_hash)):
            _sha(value, field=name)
        _positive_int(self.batch_size, field="batch_size")
        if self.microbatch_count <= 2 or self.batch_size % self.microbatch_count:
            raise ValueError("raw unit M must be primary M>2 and divide B")
        if len(self.draw_ids) != self.batch_size or len(self.sample_ids) != self.batch_size:
            raise ValueError("raw unit draw/sample count must equal B")
        if len(set(self.draw_ids)) != len(self.draw_ids):
            raise ValueError("raw unit draw IDs must be unique")
        _ref(self.raw_artifact_ref, field="raw_artifact_ref")
        _finite(self.metrics, field="metrics")
        _finite(self.cost, field="cost")
        if self.status == "SUCCESS":
            expected_methods = set(_METHOD_NAMES) | {f"u_m{self.microbatch_count}"}
            if set(self.methods) != expected_methods:
                raise ValueError("success raw unit method set is incomplete")
            if self.m2_identity_max_abs is None or self.m2_identity_max_abs != self.m2_identity_max_abs or self.m2_identity_max_abs in (float("inf"), float("-inf")) or self.m2_identity_max_abs > S27_M2_IDENTITY_TOLERANCE:
                raise ValueError("success raw unit M=2 identity metric is required and finite")
            if self.mean_gradient_consistent is not True or self.clamp_applied is not False or self.clip_mode != "none":
                raise ValueError("success raw unit estimator integrity marker failed")
            if self.failure_code is not None or self.failure_reason is not None:
                raise ValueError("success raw unit cannot carry failure reason")
        else:
            if not self.failure_code or not self.failure_reason:
                raise ValueError("failed raw unit requires explicit code and reason")
            _text(self.failure_code, field="failure_code", pattern=_SAFE_ID)
            _text(self.failure_reason, field="failure_reason")

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": S27_UNIT_SCHEMA,
            "unit_id": self.unit_id,
            "cell_id": self.cell_id,
            "repetition_id": self.repetition_id,
            "status": self.status,
            "attempt_id": self.attempt_id,
            "matrix_hash": self.matrix_hash,
            "mapping_hash": self.mapping_hash,
            "sampling_plan_hash": self.sampling_plan_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "reference_hash": self.reference_hash,
            "batch_size": self.batch_size,
            "microbatch_count": self.microbatch_count,
            "draw_ids": list(self.draw_ids),
            "sample_ids": list(self.sample_ids),
            "raw_artifact_ref": self.raw_artifact_ref,
            "raw_artifact_hash": self.raw_artifact_hash,
            "metrics": dict(self.metrics),
            "methods": list(self.methods),
            "m2_identity_max_abs": self.m2_identity_max_abs,
            "mean_gradient_consistent": self.mean_gradient_consistent,
            "clamp_applied": self.clamp_applied,
            "clip_mode": self.clip_mode,
            "cost": dict(self.cost),
            "failure_code": self.failure_code,
            "failure_reason": self.failure_reason,
        }
        if include_hash:
            value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "S27RawUnit":
        if value.get("schema_version") != S27_UNIT_SCHEMA:
            raise ValueError("raw unit schema mismatch")
        declared = _sha(value.get("artifact_hash"), field="raw_unit.artifact_hash")
        body = {key: item for key, item in value.items() if key != "artifact_hash"}
        if declared != canonical_json_hash(body):
            raise ValueError("raw unit artifact hash mismatch")
        required = {"schema_version", "unit_id", "cell_id", "repetition_id", "status", "attempt_id", "matrix_hash", "mapping_hash", "sampling_plan_hash", "checkpoint_hash", "reference_hash", "batch_size", "microbatch_count", "draw_ids", "sample_ids", "raw_artifact_ref", "raw_artifact_hash", "metrics", "methods", "m2_identity_max_abs", "mean_gradient_consistent", "clamp_applied", "clip_mode", "cost", "failure_code", "failure_reason", "artifact_hash"}
        if set(value) != required:
            raise ValueError("raw unit field set mismatch")
        if not isinstance(value["draw_ids"], list) or not isinstance(value["sample_ids"], list) or not isinstance(value["methods"], list) or not isinstance(value["metrics"], Mapping) or not isinstance(value["cost"], Mapping):
            raise TypeError("raw unit list/object fields invalid")
        return cls(
            unit_id=value["unit_id"], cell_id=value["cell_id"], repetition_id=value["repetition_id"], status=value["status"], attempt_id=value["attempt_id"], matrix_hash=value["matrix_hash"], mapping_hash=value["mapping_hash"], sampling_plan_hash=value["sampling_plan_hash"], checkpoint_hash=value["checkpoint_hash"], reference_hash=value["reference_hash"], batch_size=value["batch_size"], microbatch_count=value["microbatch_count"], draw_ids=tuple(value["draw_ids"]), sample_ids=tuple(value["sample_ids"]), raw_artifact_ref=value["raw_artifact_ref"], raw_artifact_hash=value["raw_artifact_hash"], metrics=dict(value["metrics"]), methods=tuple(value["methods"]), m2_identity_max_abs=value["m2_identity_max_abs"], mean_gradient_consistent=value["mean_gradient_consistent"], clamp_applied=value["clamp_applied"], clip_mode=value["clip_mode"], cost=dict(value["cost"]), failure_code=value["failure_code"], failure_reason=value["failure_reason"],
        )  # type: ignore[arg-type]


class StrictG25Reducer:
    """Single-writer, idempotent reducer with a closed expected denominator."""

    def __init__(self, plan: S27Plan, *, run_id: str) -> None:
        self.plan = plan
        self.run_id = _text(run_id, field="run_id", pattern=_SAFE_ID)
        self._records: dict[str, S27RawUnit] = {}
        self._sealed = False
        self._expected = {unit.unit_id: unit for unit in plan.frozen_inputs.units}
        self._by_cell = {cell.cell_id: cell for cell in plan.cells}

    @property
    def records(self) -> tuple[S27RawUnit, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def add(self, record: S27RawUnit | Mapping[str, object]) -> bool:
        if self._sealed:
            raise S27G25Blocked("SEALED_REDUCER_IS_IMMUTABLE")
        unit = record if isinstance(record, S27RawUnit) else S27RawUnit.from_mapping(record)
        expected = self._expected.get(unit.unit_id)
        if expected is None:
            raise S27G25Blocked(f"UNEXPECTED_UNIT:{unit.unit_id}")
        if unit.cell_id != expected.cell_id or unit.repetition_id != expected.repetition_id:
            raise S27G25Blocked(f"UNIT_IDENTITY_MISMATCH:{unit.unit_id}")
        if (unit.batch_size, unit.microbatch_count) != (expected.batch_size, expected.microbatch_count):
            raise S27G25Blocked(f"UNIT_B_M_MISMATCH:{unit.unit_id}")
        if tuple(unit.draw_ids) != expected.draw_ids or tuple(unit.sample_ids) != expected.sample_ids:
            raise S27G25Blocked(f"UNIT_MAPPING_MISMATCH:{unit.unit_id}")
        cell = self._by_cell[unit.cell_id]
        if (unit.matrix_hash != self.plan.frozen_inputs.matrix_hash or unit.mapping_hash != expected.mapping_hash or unit.sampling_plan_hash != self.plan.frozen_inputs.sampling_plan_hash or unit.checkpoint_hash != cell.checkpoint_hash or unit.reference_hash != cell.reference_hash):
            raise S27G25Blocked(f"UNIT_LINEAGE_MISMATCH:{unit.unit_id}")
        if cell.corrected_delta_sci_binding is not None:
            observed_binding = unit.metrics.get("corrected_delta_sci_binding")
            if observed_binding != dict(cell.corrected_delta_sci_binding):
                raise S27G25Blocked(f"UNIT_CORRECTED_DELTA_BINDING_MISMATCH:{unit.unit_id}")
        previous = self._records.get(unit.unit_id)
        if previous is not None:
            if previous.artifact_hash == unit.artifact_hash and previous.attempt_id == unit.attempt_id:
                return False
            raise S27G25Blocked(f"DUPLICATE_OR_RETRY_ATTEMPT:{unit.unit_id}")
        observed_draws = {draw_id for item in self._records.values() for draw_id in item.draw_ids}
        if observed_draws.intersection(unit.draw_ids):
            raise S27G25Blocked(f"DRAW_ID_COLLISION:{unit.unit_id}")
        self._records[unit.unit_id] = unit
        return True

    def _validate_complete(self) -> tuple[int, int, float]:
        missing = sorted(set(self._expected) - set(self._records))
        if missing:
            raise S27G25Blocked(f"EXPECTED_UNITS_MISSING:{','.join(missing[:8])}")
        all_draws = [draw_id for record in self.records for draw_id in record.draw_ids]
        if len(all_draws) != len(set(all_draws)):
            raise S27G25Blocked("DRAW_ID_COLLISION_GLOBAL")
        failed = sum(record.status == "FAILED" for record in self.records)
        failure_fraction = failed / len(self._expected)
        if failure_fraction > self.plan.frozen_inputs.max_failure_fraction:
            raise S27G25Blocked("FAILURE_FRACTION_EXCEEDS_FROZEN_LIMIT")
        if self.plan.frozen_inputs.cost_required:
            invalid = [record.unit_id for record in self.records if record.status == "SUCCESS" and record.cost.get("valid") is not True]
            if invalid:
                raise S27G25Blocked(f"COST_RECORD_INVALID:{','.join(invalid[:8])}")
        return len(self._records), failed, failure_fraction

    def _manifest(self) -> dict[str, object]:
        completed, failed, failure_fraction = self._validate_complete()
        descriptors = []
        for record in self.records:
            descriptor: dict[str, object] = {
                "unit_id": record.unit_id,
                "cell_id": record.cell_id,
                "repetition_id": record.repetition_id,
                "status": record.status,
                "attempt_id": record.attempt_id,
                "draw_id_hash": canonical_json_hash(list(record.draw_ids)),
                "raw_artifact_ref": record.raw_artifact_ref,
                "raw_artifact_hash": record.raw_artifact_hash,
                "unit_artifact_hash": record.artifact_hash,
            }
            binding = self._by_cell[record.cell_id].corrected_delta_sci_binding
            if binding is not None:
                descriptor["corrected_delta_sci_binding"] = dict(binding)
            descriptors.append(descriptor)
        body: dict[str, object] = {
            "schema_version": S27_RAW_MANIFEST_SCHEMA,
            "run_id": self.run_id,
            "task_id": S27_TASK_ID,
            "scope": S27_SCOPE,
            "stream": S27_STREAM,
            "status": "SEALED",
            "formal_eligible": True,
            "plan_ref": self.plan.plan_ref,
            "plan_hash": self.plan.artifact_hash,
            "matrix_ref": self.plan.frozen_inputs.matrix_ref,
            "matrix_hash": self.plan.frozen_inputs.matrix_hash,
            "mapping_ref": self.plan.frozen_inputs.mapping_ref,
            "mapping_hash": self.plan.frozen_inputs.mapping_hash,
            "sampling_plan_hash": self.plan.frozen_inputs.sampling_plan_hash,
            "expected_unit_count": len(self._expected),
            "completed_unit_count": completed,
            "failed_unit_count": failed,
            "failure_fraction": failure_fraction,
            "max_failure_fraction": self.plan.frozen_inputs.max_failure_fraction,
            "cell_order": list(S27_WAVE_ORDER),
            "units": descriptors,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        return body

    def seal(self, output_root: str | Path, *, checked_at: str | None = None) -> dict[str, object]:
        if self._sealed:
            raise S27G25Blocked("SEALED_REDUCER_IS_IMMUTABLE")
        manifest = self._manifest()
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "raw-results-manifest.json"
        marker_path = root / "sealed.marker.json"
        gate_path = root / "g2.5-gate.json"
        if any(path.exists() for path in (manifest_path, marker_path, gate_path)):
            raise S27G25Blocked("SEALED_OUTPUT_ALREADY_EXISTS")
        write_canonical_json(manifest_path, manifest)
        manifest_file_hash = sha256_file(manifest_path)
        now = checked_at or datetime.now(timezone.utc).isoformat()
        marker_body: dict[str, object] = {
            "schema_version": S27_SEAL_SCHEMA,
            "run_id": self.run_id,
            "manifest_ref": manifest_path.name,
            "manifest_hash": manifest["artifact_hash"],
            "manifest_file_sha256": manifest_file_hash,
            "sealed": True,
            "checked_at": now,
        }
        marker_body["artifact_hash"] = canonical_json_hash(marker_body)
        write_canonical_json(marker_path, marker_body)
        completed = int(manifest["completed_unit_count"])
        failed = int(manifest["failed_unit_count"])
        failure_fraction = float(manifest["failure_fraction"])
        gate = GateRecord(
            gate_id="stage2.G2.5",
            stage=2,
            status=GateStatus.PASS,
            checked_at=now,
            measured={"expected_unit_count": completed, "failed_unit_count": failed, "failure_fraction": failure_fraction, "raw_manifest_hash": manifest["artifact_hash"], "sealed_marker_hash": marker_body["artifact_hash"]},
            threshold={"expected_unit_count": len(self._expected), "max_failure_fraction": self.plan.frozen_inputs.max_failure_fraction, "strict_mapping_audit": True, "strict_lineage_audit": True, "finite_outputs": True, "m2_identity": True, "no_clamp": True},
            evidence_refs=(self.plan.plan_ref, manifest_path.name, marker_path.name),
        )
        gate_payload = gate.to_dict()
        write_canonical_json(gate_path, gate_payload)
        self._sealed = True
        return {"manifest": manifest, "sealed_marker": marker_body, "gate": gate_payload, "manifest_ref": manifest_path.name, "sealed_marker_ref": marker_path.name, "gate_ref": gate_path.name}


@dataclass(frozen=True, slots=True)
class S27DetachedStatus:
    run_id: str
    plan_hash: str
    status: str
    wave_index: int
    updated_at: str
    owner_pid: int | None = None
    gpu_uuid: str | None = None
    heartbeat_unix: float | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.run_id, field="run_id", pattern=_SAFE_ID)
        _sha(self.plan_hash, field="plan_hash")
        if self.status not in _STATUS_VALUES:
            raise ValueError("invalid detached status")
        if isinstance(self.wave_index, bool) or not isinstance(self.wave_index, int) or self.wave_index < 0 or self.wave_index > len(S27_WAVE_ORDER):
            raise ValueError("invalid wave_index")
        _text(self.updated_at, field="updated_at")
        if self.owner_pid is not None and (isinstance(self.owner_pid, bool) or not isinstance(self.owner_pid, int) or self.owner_pid <= 0):
            raise ValueError("owner_pid must be positive")
        if self.gpu_uuid is not None and self.gpu_uuid not in APPROVED_GPU_UUIDS:
            raise ValueError("status uses unapproved GPU")
        if self.heartbeat_unix is not None and isinstance(self.heartbeat_unix, bool):
            raise ValueError("heartbeat_unix invalid")
        if self.status == "RUNNING" and (self.owner_pid is None or self.gpu_uuid is None or self.heartbeat_unix is None):
            raise S27G25Blocked("RUNNING_STATUS_OWNER_HEARTBEAT_REQUIRED")
        if self.status in _TERMINAL_STATUS and not self.terminal_reason:
            raise ValueError("terminal status requires reason")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": S27_STATUS_SCHEMA,
            "run_id": self.run_id,
            "plan_hash": self.plan_hash,
            "status": self.status,
            "wave_index": self.wave_index,
            "updated_at": self.updated_at,
            "owner_pid": self.owner_pid,
            "gpu_uuid": self.gpu_uuid,
            "heartbeat_unix": self.heartbeat_unix,
            "terminal_reason": self.terminal_reason,
        }
        if include_hash:
            value["artifact_hash"] = canonical_json_hash(value)
        return value


class S27StatusStore:
    """Atomic status publication with explicit transition and recovery checks."""

    def __init__(self, path: str | Path, *, run_id: str, plan_hash: str) -> None:
        self.path = Path(path)
        self.run_id = _text(run_id, field="run_id", pattern=_SAFE_ID)
        self.plan_hash = _sha(plan_hash, field="plan_hash")

    def load(self) -> S27DetachedStatus:
        try:
            value = load_canonical_json(self.path)
        except Exception as error:
            raise S27G25Blocked("STATUS_CANONICAL_READ_FAILED") from error
        if not isinstance(value, Mapping) or value.get("schema_version") != S27_STATUS_SCHEMA:
            raise S27G25Blocked("STATUS_SCHEMA_INVALID")
        declared = value.get("artifact_hash")
        body = {key: item for key, item in value.items() if key != "artifact_hash"}
        if declared != canonical_json_hash(body) or value.get("run_id") != self.run_id or value.get("plan_hash") != self.plan_hash:
            raise S27G25Blocked("STATUS_IDENTITY_OR_HASH_DRIFT")
        try:
            return S27DetachedStatus(run_id=value["run_id"], plan_hash=value["plan_hash"], status=value["status"], wave_index=value["wave_index"], updated_at=value["updated_at"], owner_pid=value["owner_pid"], gpu_uuid=value["gpu_uuid"], heartbeat_unix=value["heartbeat_unix"], terminal_reason=value["terminal_reason"])  # type: ignore[arg-type]
        except (TypeError, ValueError, S27G25Blocked) as error:
            raise S27G25Blocked("STATUS_FIELDS_INVALID") from error

    def publish(self, status: S27DetachedStatus) -> None:
        if status.run_id != self.run_id or status.plan_hash != self.plan_hash:
            raise S27G25Blocked("STATUS_IDENTITY_MISMATCH")
        if self.path.exists():
            previous = self.load()
            allowed = {"PREPARED": {"PREPARED", "RUNNING", "BLOCKED"}, "RUNNING": {"RUNNING", "PAUSED", "FAILED", "SEALED", "BLOCKED"}, "PAUSED": {"PAUSED", "RUNNING", "FAILED", "BLOCKED"}, "FAILED": {"FAILED"}, "SEALED": {"SEALED"}, "BLOCKED": {"BLOCKED"}}
            if status.status not in allowed[previous.status]:
                raise S27G25Blocked(f"STATUS_TRANSITION_FORBIDDEN:{previous.status}->{status.status}")
        write_canonical_json(self.path, status.to_dict())

    def require_recoverable(self) -> S27DetachedStatus:
        status = self.load()
        if status.status in {"SEALED", "FAILED", "BLOCKED"}:
            raise S27G25Blocked(f"STATUS_TERMINAL_NO_RESUME:{status.status}")
        if status.status == "RUNNING" and status.owner_pid is not None and status.owner_pid != os.getpid():
            raise S27G25Blocked("STATUS_OWNED_BY_OTHER_PROCESS")
        return status


def validate_gpu_inventory(inventory: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate a normalized inventory; never selects an excluded GPU."""

    by_uuid = {str(item.get("uuid")): item for item in inventory}
    missing = [uuid for uuid in APPROVED_GPU_UUIDS if uuid not in by_uuid]
    if missing:
        raise S27PreparationBlocked(f"APPROVED_GPU_MISSING:{','.join(missing)}")
    for item in inventory:
        uuid = str(item.get("uuid"))
        pci = str(item.get("pci_bus_id", "")).lower()
        if uuid in APPROVED_GPU_UUIDS and pci == EXCLUDED_PCI.lower():
            raise S27PreparationBlocked("APPROVED_GPU_BOUND_TO_EXCLUDED_PCI")
    return {"approved_gpu_uuids": list(APPROVED_GPU_UUIDS), "excluded_gpu_uuid": EXCLUDED_GPU_UUID, "excluded_pci": EXCLUDED_PCI, "inventory_count": len(inventory)}


__all__ = [
    "APPROVED_GPU_UUIDS",
    "EXCLUDED_GPU_UUID",
    "EXCLUDED_PCI",
    "S27CellPlan",
    "S27_ANCHOR_ORDER",
    "S27_CORRECTED_DELTA_BATCH_SIZES",
    "S27_CORRECTED_DELTA_SOURCE",
    "S27_CORRECTED_DELTA_BINDING_FIELDS",
    "S27DetachedStatus",
    "S27FrozenInputs",
    "S27G25Blocked",
    "S27MappingUnit",
    "S27Plan",
    "S27PreparationBlocked",
    "S27RawUnit",
    "S27StatusStore",
    "S27_GATE_SCHEMA",
    "S27_MAPPING_SCHEMA",
    "S27_M2_IDENTITY_TOLERANCE",
    "S27_MATRIX_SCHEMA",
    "S27_PLAN_SCHEMA",
    "S27_RAW_MANIFEST_SCHEMA",
    "S27_SEAL_SCHEMA",
    "S27_STATUS_SCHEMA",
    "S27_TASK_ID",
    "StrictG25Reducer",
    "prepare_s27_plan",
    "validate_gpu_inventory",
]
