"""Repository-contained S2.9 production profiler.

This is the executable adapter used by ``run_s209_profiler_worker.py``.  The
worker remains the identity boundary; this module supplies the real backend
behind it.  A config is produced from the frozen G2.4b matrix, S2.7 plan,
mapping, sealed raw manifest, materialized six-cell inputs, and formal
execution evidence.  At run time the adapter re-reads and re-hashes those
inputs before loading the selected checkpoint through the existing S2.7 Torch
provider.

The measurement path deliberately has no CPU/synthetic fallback.  It requires
CUDA, an approved UUID, a formal fixed-state provider, and a launch-time I/O
snapshot.  ``scientific_equal_sample_cost`` computes raw/double/U once from one
gradient pool; the other two semantics execute only their requested method.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stdout
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import queue as queue_module
import time
from typing import Any, Mapping, Sequence

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import canonical_json_bytes, canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG
from ..experiments.sampling import RepetitionMapping
from ..runtime.task_runtime import TaskExecutionRequest, TaskRuntimeEnvironment
from .stage2 import CoreEstimatorKernel, _vector_digest
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import APPROVED_GPU_UUIDS, S27CellPlan
from .stage2_s207_runner import (
    S27ExecutionBlocked,
    _load_checkpoint_root,
    build_s27_torch_provider,
    load_s27_frozen_mappings,
    load_s27_materialized_inputs,
    load_s27_plan,
    load_s27_raw_records,
    load_s27_reference_views,
)
from .stage2_s209_g27a import (
    S29_COUNT_FIELDS,
    S29_COST_SEMANTICS,
    S29_METHODS,
    S29_SHARED_POOL_SCHEMA,
    S29_SHARED_RUN_SCHEMA,
    S29_TIMING_FIELDS,
    _validate_measurement_plan,
    bind_s209_inputs,
)
from .stage2_s209_worker import S209_WORKER_CONFIG_SCHEMA, S209WorkerBlocked


S209_PRODUCTION_CONFIG_SCHEMA = "stage2-s209-profiler-production-config-v1"
_SHA256_LENGTH = 64
_METHOD_ONLY_SEMANTICS = S29_COST_SEMANTICS[1:]


class S209ProductionBlocked(RuntimeError):
    """Raised when a formal production input or device measurement is absent."""


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise S209ProductionBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _logical(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S209ProductionBlocked(f"{field}:LOGICAL_REF_REQUIRED")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S209ProductionBlocked(f"{field}:PATH_ESCAPE")
    result = (root / Path(*logical.parts)).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S209ProductionBlocked(f"{field}:PATH_ESCAPE") from error
    return result


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise S209ProductionBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S209ProductionBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _payload_identity(value: Mapping[str, Any], *, field: str) -> str:
    """Return the producer identity for canonical JSON artifacts."""

    for name in ("artifact_hash", "index_hash", "manifest_hash"):
        declared = value.get(name)
        if declared is not None:
            digest = _sha(declared, field=f"{field}.{name}")
            body = {key: item for key, item in value.items() if key != name}
            if digest != canonical_json_hash(body):
                raise S209ProductionBlocked(f"{field}:{name.upper()}_MISMATCH")
            return digest
    raise S209ProductionBlocked(f"{field}:CONTENT_ADDRESS_REQUIRED")


def _source_identity(root: Path, reference: str, *, expected: str, field: str) -> dict[str, Any]:
    path = _logical(root, reference, field=field)
    value = _load_object(path, field=field)
    observed = _payload_identity(value, field=field)
    if observed != expected:
        raise S209ProductionBlocked(f"{field}:HASH_DRIFT")
    return value


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = _load_object(path, field=field)
        if current != dict(value):
            raise S209ProductionBlocked(f"{field}:OUTPUT_CONFLICT")
        return
    write_canonical_json(path, dict(value))


def _check_config_shape(config: Mapping[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    if config.get("schema_version") != S209_WORKER_CONFIG_SCHEMA or config.get("formal_eligible") is not True:
        raise S209ProductionBlocked("WORKER_CONFIG_FORMAL_SCHEMA_REQUIRED")
    if config.get("production_schema") != S209_PRODUCTION_CONFIG_SCHEMA:
        raise S209ProductionBlocked("PRODUCTION_CONFIG_SCHEMA_REQUIRED")
    declared = config.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash({key: item for key, item in config.items() if key != "artifact_hash"}) != declared:
        raise S209ProductionBlocked("PRODUCTION_CONFIG_ARTIFACT_HASH_MISMATCH")
    if run_id is not None and config.get("run_id") != run_id:
        raise S209ProductionBlocked("PRODUCTION_CONFIG_RUN_ID_MISMATCH")
    for name in ("data_root", "s27_plan_ref", "materialization_index_ref", "execution_evidence_ref", "measurement_plan_ref"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise S209ProductionBlocked(f"PRODUCTION_CONFIG_{name.upper()}_REQUIRED")
    if not isinstance(config.get("sources"), Mapping) or not isinstance(config.get("cells"), list) or not isinstance(config.get("task_bindings"), list):
        raise S209ProductionBlocked("PRODUCTION_CONFIG_FROZEN_BINDINGS_REQUIRED")
    return dict(config)


def _config_source_map(frozen: Any, *, refs: Mapping[str, str]) -> dict[str, dict[str, str]]:
    required = ("matrix", "g24b_gate", "raw_manifest", "g25_gate", "s27_plan", "mapping", "materialization_index", "execution_evidence", "measurement_plan")
    if set(refs) != set(required):
        raise S209ProductionBlocked("PRODUCTION_CONFIG_SOURCE_SET_INVALID")
    output: dict[str, dict[str, str]] = {}
    for name in required:
        reference = refs[name]
        if not isinstance(reference, str):
            raise S209ProductionBlocked(f"PRODUCTION_CONFIG_SOURCE_REF_INVALID:{name}")
        output[name] = {"ref": reference}
    return output


def prepare_s209_profiler_config(
    *,
    data_root: str | Path,
    matrix_ref: str,
    g24b_gate_ref: str,
    raw_manifest_ref: str,
    g25_gate_ref: str,
    s27_plan_ref: str,
    materialization_index_ref: str,
    execution_evidence_ref: str,
    measurement_plan_ref: str,
    output_ref: str,
    run_id: str,
) -> dict[str, Any]:
    """Create a hash-bound config for the repository-contained backend."""

    if not isinstance(run_id, str) or not run_id:
        raise S209ProductionBlocked("RUN_ID_REQUIRED")
    root = Path(data_root).resolve()
    refs = {
        "matrix": matrix_ref,
        "g24b_gate": g24b_gate_ref,
        "raw_manifest": raw_manifest_ref,
        "g25_gate": g25_gate_ref,
        "s27_plan": s27_plan_ref,
        "mapping": None,
        "materialization_index": materialization_index_ref,
        "execution_evidence": execution_evidence_ref,
        "measurement_plan": measurement_plan_ref,
    }
    matrix = _load_object(_logical(root, matrix_ref, field="matrix_ref"), field="matrix")
    gate = _load_object(_logical(root, g24b_gate_ref, field="g24b_gate_ref"), field="g24b_gate")
    raw = _load_object(_logical(root, raw_manifest_ref, field="raw_manifest_ref"), field="raw_manifest")
    g25 = _load_object(_logical(root, g25_gate_ref, field="g25_gate_ref"), field="g25_gate")
    frozen = bind_s209_inputs(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=raw,
        g25_gate=g25,
        require_g25=True,
        matrix_ref=matrix_ref,
        raw_manifest_ref=raw_manifest_ref,
    )
    s27_plan = load_s27_plan(root, s27_plan_ref)
    if s27_plan.artifact_hash != frozen.plan_hash:
        raise S209ProductionBlocked("S27_PLAN_HASH_MISMATCH")
    if s27_plan.frozen_inputs.mapping_hash != frozen.mapping_hash or s27_plan.frozen_inputs.sampling_plan_hash != frozen.sampling_plan_hash:
        raise S209ProductionBlocked("S27_MAPPING_OR_SAMPLING_HASH_MISMATCH")
    mappings = load_s27_frozen_mappings(root, s27_plan)
    materialized = load_s27_materialized_inputs(root, materialization_index_ref)
    if tuple(materialized) != EXPECTED_CELL_IDS:
        raise S209ProductionBlocked("S27_MATERIALIZATION_SIX_CELL_ORDER_INVALID")
    for cell in s27_plan.cells:
        _load_checkpoint_root(root, cell, materialized[cell.cell_id])
        try:
            load_s27_reference_views(
                root,
                cell,
                reference_output_root_ref=materialized[cell.cell_id].reference_output_root_ref,
            )
        except Exception as error:
            raise S209ProductionBlocked(f"S27_REFERENCE_ARTIFACT_INVALID:{cell.cell_id}") from error
    # The sealed manifest is only a summary.  Re-open the S2.7 raw-unit
    # records, tensor bundles, and attempt ledgers so config production cannot
    # bind a summary whose underlying raw/ref artifacts were replaced.
    try:
        raw_reducer = load_s27_raw_records(
            root,
            s27_plan,
            _logical(root, raw_manifest_ref, field="raw_manifest_ref").parent,
            run_id=frozen.raw_run_id,
        )
        raw_reducer.seal()
    except Exception as error:
        raise S209ProductionBlocked("S27_RAW_ARTIFACTS_NOT_SEALED") from error

    measurement = _load_object(_logical(root, measurement_plan_ref, field="measurement_plan_ref"), field="measurement_plan")
    try:
        _validate_measurement_plan(measurement, frozen=frozen)
    except Exception as error:
        raise S209ProductionBlocked(f"MEASUREMENT_PLAN_INVALID:{error}") from error
    if measurement.get("run_id") != run_id:
        raise S209ProductionBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    mapping_ref = s27_plan.frozen_inputs.mapping_ref
    refs["mapping"] = mapping_ref
    assert isinstance(mapping_ref, str)

    source_values = {
        "matrix": matrix,
        "g24b_gate": gate,
        "raw_manifest": raw,
        "g25_gate": g25,
        "s27_plan": _load_object(_logical(root, s27_plan_ref, field="s27_plan_ref"), field="s27_plan"),
        "mapping": _load_object(_logical(root, mapping_ref, field="mapping_ref"), field="mapping"),
        "materialization_index": _load_object(_logical(root, materialization_index_ref, field="materialization_index_ref"), field="materialization_index"),
        "execution_evidence": _load_object(_logical(root, execution_evidence_ref, field="execution_evidence_ref"), field="execution_evidence"),
        "measurement_plan": measurement,
    }
    source_hashes = {name: _payload_identity(value, field=f"source.{name}") for name, value in source_values.items()}
    if source_hashes["s27_plan"] != s27_plan.artifact_hash or source_hashes["mapping"] != frozen.mapping_hash:
        raise S209ProductionBlocked("S27_SOURCE_HASH_BINDING_INVALID")
    try:
        evidence = FormalExecutionEvidence.from_mapping(source_values["execution_evidence"])
        evidence.require_for_stage(2)
    except Exception as error:
        raise S209ProductionBlocked(f"FORMAL_EXECUTION_EVIDENCE_INVALID:{error}") from error

    units_by_id = {unit.unit_id: unit for unit in s27_plan.frozen_inputs.units}
    ordered_units = list(s27_plan.frozen_inputs.units)
    cell_rows: list[dict[str, Any]] = []
    for cell in s27_plan.cells:
        material = materialized[cell.cell_id]
        cell_rows.append(
            {
                "cell_id": cell.cell_id,
                "model_id": cell.model_id,
                "training_stage": cell.training_stage,
                "checkpoint_ref": cell.checkpoint_ref,
                "checkpoint_hash": cell.checkpoint_hash,
                "checkpoint_id": cell.checkpoint_id,
                "checkpoint_root_ref": material.checkpoint_root_ref,
                "registry_ref": material.registry_ref,
                "config_ref": material.config_ref,
                "environment_ref": material.environment_ref,
                "unit_ids": list(cell.expected_unit_ids),
            }
        )

    # Each S2.9 anchor/repetition is bound to a real S2.7 unit in plan order.
    # All three cost semantics use the same route for a given anchor/repetition;
    # the semantic/method key remains explicit so a method cannot be rebound.
    bindings: list[dict[str, Any]] = []
    measurement_rows = measurement["rows"]
    assert isinstance(measurement_rows, list)
    for row_index, row in enumerate(measurement_rows):
        if not isinstance(row, Mapping):
            raise S209ProductionBlocked("MEASUREMENT_PLAN_ROW_INVALID")
        unit = ordered_units[row_index % len(ordered_units)]
        mapping = mappings.get(unit.unit_id)
        if mapping is None:
            raise S209ProductionBlocked(f"S27_MAPPING_MISSING:{unit.unit_id}")
        base = {"anchor_id": row["anchor_id"], "repetition": row["repetition"], "cell_id": unit.cell_id, "unit_id": unit.unit_id, "mapping_hash": mapping.digest}
        bindings.append({"semantic": "scientific_equal_sample_cost", "method": "shared", **base})
        for semantic in _METHOD_ONLY_SEMANTICS:
            for method in S29_METHODS:
                bindings.append({"semantic": semantic, "method": method, **base})

    # Preserve every frozen mapping row and its sample/draw identity in the
    # config.  The backend still reloads the authoritative artifact and checks
    # this copy, so a replaced mapping cannot be silently consumed.
    mapping_rows = [mapping.to_dict() for mapping in mappings.values()]
    mapping_rows.sort(key=lambda item: str(item["repetition_id"]))
    body: dict[str, Any] = {
        "schema_version": S209_WORKER_CONFIG_SCHEMA,
        "production_schema": S209_PRODUCTION_CONFIG_SCHEMA,
        "formal_eligible": True,
        "run_id": run_id,
        "data_root": str(root),
        "s27_plan_ref": s27_plan_ref,
        "materialization_index_ref": materialization_index_ref,
        "execution_evidence_ref": execution_evidence_ref,
        "measurement_plan_ref": measurement_plan_ref,
        "sources": {
            name: {"ref": refs[name], "hash": source_hashes[name]}
            for name in sorted(refs)
        },
        "frozen": {
            "matrix_hash": frozen.matrix_hash,
            "g24b_gate_hash": frozen.g24b_gate_hash,
            "g25_gate_hash": frozen.g25_gate_hash,
            "raw_manifest_hash": frozen.raw_manifest_hash,
            "raw_run_id": frozen.raw_run_id,
            "plan_hash": frozen.plan_hash,
            "mapping_hash": frozen.mapping_hash,
            "sampling_plan_hash": frozen.sampling_plan_hash,
            "batch_size": frozen.batch_size,
            "microbatch_count": frozen.microbatch_count,
            "repetitions": frozen.repetitions,
            "expected_unit_ids": list(frozen.expected_unit_ids),
        },
        "cells": cell_rows,
        "mapping_units": mapping_rows,
        "task_bindings": bindings,
        "approved_backend": "param_importance_nlp.experiments.stage2_s209_production:run_s209_production_backend",
        "cuda_required": True,
        "shared_gradient_pool_required": True,
        "measurement_contract": {
            "dtype": "float32_gradients_float64_reducer",
            "model_state": "fixed_eval_no_optimizer_step",
            "communication": "single_gpu_zero_bytes",
            "timing": "cuda_events_and_provider_hooks",
        },
    }
    body["artifact_hash"] = canonical_json_hash(body)
    output = _logical(root, output_ref, field="output_ref")
    _write_once(output, body, field="production_config")
    return body


def _verify_runtime_config(config: Mapping[str, Any], *, task: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    checked = _check_config_shape(config, run_id=str(task["run_id"]))
    root_value = checked.get("data_root")
    if not isinstance(root_value, str) or not root_value:
        raise S209ProductionBlocked("PRODUCTION_CONFIG_DATA_ROOT_REQUIRED")
    root = Path(root_value).resolve()
    if not root.is_dir():
        raise S209ProductionBlocked("PRODUCTION_CONFIG_DATA_ROOT_MISSING")
    sources = checked["sources"]
    assert isinstance(sources, Mapping)
    source_values: dict[str, Mapping[str, Any]] = {}
    for name, source in sources.items():
        if not isinstance(source, Mapping) or set(source) != {"ref", "hash"}:
            raise S209ProductionBlocked(f"PRODUCTION_CONFIG_SOURCE_INVALID:{name}")
        source_values[str(name)] = _source_identity(root, str(source["ref"]), expected=str(source["hash"]), field=f"source.{name}")
    frozen = checked.get("frozen")
    if not isinstance(frozen, Mapping):
        raise S209ProductionBlocked("PRODUCTION_CONFIG_FROZEN_REQUIRED")
    plan = load_s27_plan(root, str(checked["s27_plan_ref"]))
    if plan.artifact_hash != frozen.get("plan_hash"):
        raise S209ProductionBlocked("RUNTIME_S27_PLAN_HASH_DRIFT")
    mappings = load_s27_frozen_mappings(root, plan)
    if plan.frozen_inputs.mapping_hash != frozen.get("mapping_hash"):
        raise S209ProductionBlocked("RUNTIME_S27_MAPPING_HASH_DRIFT")
    if tuple(frozen.get("expected_unit_ids", ())) != plan.expected_unit_ids:
        raise S209ProductionBlocked("RUNTIME_S27_UNIT_ORDER_DRIFT")
    measurement = source_values.get("measurement_plan")
    if measurement is None:
        raise S209ProductionBlocked("RUNTIME_MEASUREMENT_PLAN_MISSING")
    try:
        _validate_measurement_plan(measurement, frozen=bind_s209_inputs(
            matrix=source_values["matrix"],
            g24b_gate=source_values["g24b_gate"],
            raw_manifest=source_values["raw_manifest"],
            g25_gate=source_values["g25_gate"],
            require_g25=True,
        ))
    except Exception as error:
        raise S209ProductionBlocked(f"RUNTIME_MEASUREMENT_PLAN_INVALID:{error}") from error
    materialized = load_s27_materialized_inputs(root, str(checked["materialization_index_ref"]))
    if tuple(materialized) != EXPECTED_CELL_IDS:
        raise S209ProductionBlocked("RUNTIME_MATERIALIZATION_ORDER_INVALID")
    for cell in plan.cells:
        _load_checkpoint_root(root, cell, materialized[cell.cell_id])
        try:
            load_s27_reference_views(
                root,
                cell,
                reference_output_root_ref=materialized[cell.cell_id].reference_output_root_ref,
            )
        except Exception as error:
            raise S209ProductionBlocked(f"RUNTIME_S27_REFERENCE_ARTIFACT_INVALID:{cell.cell_id}") from error
    try:
        raw_manifest_source = sources.get("raw_manifest")
        if not isinstance(raw_manifest_source, Mapping) or not isinstance(raw_manifest_source.get("ref"), str):
            raise S209ProductionBlocked("RUNTIME_RAW_MANIFEST_SOURCE_REQUIRED")
        load_s27_raw_records(
            root,
            plan,
            _logical(root, str(raw_manifest_source["ref"]), field="raw_manifest_ref").parent,
            run_id=str(frozen.get("raw_run_id")),
        ).seal()
    except Exception as error:
        raise S209ProductionBlocked("RUNTIME_S27_RAW_ARTIFACTS_NOT_SEALED") from error
    config_mappings = checked.get("mapping_units")
    if not isinstance(config_mappings, list) or len(config_mappings) != len(mappings):
        raise S209ProductionBlocked("PRODUCTION_CONFIG_MAPPING_ROWS_INVALID")
    by_unit = {str(item.get("unit_id")): item for item in config_mappings if isinstance(item, Mapping)}
    if set(by_unit) != set(mappings):
        raise S209ProductionBlocked("PRODUCTION_CONFIG_MAPPING_UNIT_SET_INVALID")
    for unit_id, mapping in mappings.items():
        if by_unit[unit_id].get("mapping_hash") != mapping.digest or by_unit[unit_id].get("draws") != [draw.to_manifest() for draw in mapping.draws]:
            raise S209ProductionBlocked(f"PRODUCTION_CONFIG_MAPPING_CONTENT_DRIFT:{unit_id}")
    return checked, root


def _request_for_cell(root: Path, config: Mapping[str, Any], cell: S27CellPlan) -> TaskExecutionRequest:
    """Rebind the materialized S2.4 request exactly as S2.7 does."""

    material = load_s27_materialized_inputs(root, str(config["materialization_index_ref"]))[cell.cell_id]
    config_value = _load_object(_logical(root, material.config_ref, field=f"config.{cell.cell_id}"), field=f"config.{cell.cell_id}")
    environment_value = _load_object(_logical(root, material.environment_ref, field=f"environment.{cell.cell_id}"), field=f"environment.{cell.cell_id}")
    try:
        source = ResolvedConfigV2.from_mapping(config_value)
        environment = TaskRuntimeEnvironment.from_mapping(environment_value)
    except Exception as error:
        raise S209ProductionBlocked(f"CELL_INPUT_INVALID:{cell.cell_id}:{error}") from error
    if source.run_intent != "formal":
        raise S209ProductionBlocked(f"CELL_CONFIG_NOT_FORMAL:{cell.cell_id}")
    if source.task_id != "stage2.07_main_sweep":
        payload = {key: value for key, value in source.to_dict().items() if key not in {"config_hash", "full_hash"}}
        execution = payload.get("execution")
        artifacts = payload.get("artifacts")
        if not isinstance(execution, Mapping) or not isinstance(artifacts, Mapping):
            raise S209ProductionBlocked("CELL_CONFIG_SECTIONS_MISSING")
        execution = dict(execution)
        execution["runner_kind"] = DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep").runner_kind.value
        payload["task_id"] = "stage2.07_main_sweep"
        payload["execution"] = execution
        artifacts = dict(artifacts)
        artifacts["required_kinds"] = list(DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep").artifact_kinds)
        payload["artifacts"] = artifacts
        try:
            source = ResolvedConfigV2(payload)
        except Exception as error:
            raise S209ProductionBlocked("CELL_CONFIG_DERIVATION_FAILED") from error
    refs = dict(environment.evidence_refs)
    refs["formal_execution"] = str(config["execution_evidence_ref"])
    environment = TaskRuntimeEnvironment(
        capabilities=environment.capabilities | frozenset({"server", "cuda", "model_assets", "data_assets"}),
        frozen_contract_stages=environment.frozen_contract_stages | frozenset({2}),
        passed_gate_ids=environment.passed_gate_ids | frozenset({"stage2.G2.4b"}),
        estimator_decision_ref=environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    return TaskExecutionRequest(config=source, task=DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep"), environment=environment)


@dataclass
class _MeasuredPool:
    batches: list[Any]
    maps: list[Any]
    weights: list[float]
    gradient_wall_seconds: float
    gradient_seconds: float
    forward_seconds: float
    backward_seconds: float
    data_wait_seconds: float
    sequence_count: int
    token_count: int
    backward_count: int
    state_digest: str
    state_digest_after: str
    allocated_peak_bytes: int
    reserved_peak_bytes: int
    device_peak_bytes: int
    sample_mapping_hash: str
    gradient_pool_hash: str
    gradient_aggregation_seconds: float


def _device_identity(expected_uuid: str) -> None:
    try:
        import torch
    except ImportError as error:
        raise S209ProductionBlocked("TORCH_REQUIRED") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise S209ProductionBlocked("CUDA_SINGLE_DEVICE_REQUIRED")
    try:
        properties = torch.cuda.get_device_properties(0)
        observed = getattr(properties, "uuid", None)
        if observed is not None and str(observed) != expected_uuid:
            raise S209ProductionBlocked("CUDA_UUID_VISIBLE_DEVICE_MISMATCH")
    except S209ProductionBlocked:
        raise
    except Exception:
        # Older supported torch builds do not expose a UUID property.  The
        # pre-import CUDA_VISIBLE_DEVICES mask plus NVML UUID handle below are
        # still required and remain the authoritative binding check.
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        # NVML's global index is not the CUDA-visible index after UUID based
        # masking.  Resolve by UUID so a launcher cannot claim that visible
        # device 0 is an approved card merely because it is numerically first.
        pynvml.nvmlDeviceGetHandleByUUID(expected_uuid.encode("utf-8"))
    except S209ProductionBlocked:
        raise
    except Exception as error:
        raise S209ProductionBlocked("CUDA_UUID_PROBE_REQUIRED") from error


def _device_identity_set(expected_uuids: Sequence[str]) -> None:
    """Verify one process sees exactly the approved four-card UUID set."""

    try:
        import torch
    except ImportError as error:
        raise S209ProductionBlocked("TORCH_REQUIRED") from error
    expected = tuple(expected_uuids)
    if len(expected) != 4 or len(set(expected)) != 4:
        raise S209ProductionBlocked("FOUR_GPU_UUID_SET_INVALID")
    if tuple(expected) != tuple(APPROVED_GPU_UUIDS):
        raise S209ProductionBlocked("FOUR_GPU_UUID_ALLOWLIST_DRIFT")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise S209ProductionBlocked("CUDA_FOUR_DEVICE_REQUIRED")
    try:
        import pynvml

        pynvml.nvmlInit()
        for uuid in expected:
            pynvml.nvmlDeviceGetHandleByUUID(uuid.encode("utf-8"))
    except Exception as error:
        raise S209ProductionBlocked("CUDA_FOUR_UUID_PROBE_REQUIRED") from error


_FOUR_GPU_BARRIER_TIMEOUT_SECONDS = 900.0


def _four_gpu_anchor_worker(
    gpu_uuid: str,
    *,
    config: Mapping[str, Any],
    root: Path,
    binding: Mapping[str, Any],
    barrier: Any,
    result_queue: Any,
) -> None:
    """Measure one fixed-state process of the synchronized four-card anchor.

    This function is intentionally top-level and pickleable for the Windows
    ``spawn`` context.  Each child receives one UUID mask, reloads the same
    hash-bound S2.7 checkpoint/mapping, and contributes an actual measurement
    record.  The parent aggregates only after the end barrier; no 4x scaling
    or label-only row is possible.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_uuid)
    try:
        # Backend/provider implementations are not allowed to leak logs into
        # the worker's exact-one-JSON stdout stream.
        with redirect_stdout(io.StringIO()):
            if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_uuid):
                raise S209ProductionBlocked("FOUR_GPU_CHILD_MASK_DRIFT")
            _device_identity(str(gpu_uuid))
            provider, mapping, unit, _cell = _load_provider_for_binding(root, config, binding)
            groups = mapping.groups(max(mapping.m_values))
            if not groups:
                raise S209ProductionBlocked("FOUR_GPU_ANCHOR_GROUPS_EMPTY")
            # Loading/model construction happens before the barrier.  The
            # reported wall interval starts only once all four peers are ready.
            barrier.wait(timeout=_FOUR_GPU_BARRIER_TIMEOUT_SECONDS)
            wall_start = time.perf_counter()
            pool = _measure_pool(provider, groups, mapping=mapping)
            value, formula_seconds = _method_value(CoreEstimatorKernel(accumulation_dtype="float64"), pool, "raw", mapping)
            statistics_start = time.perf_counter()
            estimate_hash = _vector_digest(value)
            statistics_seconds = time.perf_counter() - statistics_start
            phase = {
                "data_wait_seconds": float(pool.data_wait_seconds),
                "forward_seconds": float(pool.forward_seconds),
                "backward_seconds": float(pool.backward_seconds),
                "gradient_aggregation_seconds": float(pool.gradient_aggregation_seconds),
                "formula_seconds": float(formula_seconds),
                "statistics_seconds": float(statistics_seconds),
                "communication_seconds": 0.0,
                "write_seconds": 0.0,
            }
            if any(value < 0 or not math.isfinite(value) for value in phase.values()):
                raise S209ProductionBlocked("FOUR_GPU_PHASE_TIMING_INVALID")
            # End timing is barrier-bounded: the slowest peer's synchronization
            # tail is part of the actual system-anchor wall interval.
            barrier.wait(timeout=_FOUR_GPU_BARRIER_TIMEOUT_SECONDS)
            wall_seconds = float(time.perf_counter() - wall_start)
            if wall_seconds <= 0 or wall_seconds + 1e-9 < sum(phase.values()):
                raise S209ProductionBlocked("FOUR_GPU_BARRIER_WALL_INVALID")
            result_queue.put(
                {
                    "ok": True,
                    "gpu_uuid": str(gpu_uuid),
                    "health_ok": True,
                    "wall_seconds": wall_seconds,
                    "phase": phase,
                    "sequence_count": int(pool.sequence_count),
                    "token_count": int(pool.token_count),
                    "backward_count": int(pool.backward_count),
                    "communication_bytes": 0,
                    "allocated_peak_bytes": int(pool.allocated_peak_bytes),
                    "reserved_peak_bytes": int(pool.reserved_peak_bytes),
                    "device_peak_bytes": int(pool.device_peak_bytes),
                    "sample_mapping_hash": str(pool.sample_mapping_hash),
                    "gradient_pool_hash": str(pool.gradient_pool_hash),
                    "estimate_hash": str(estimate_hash),
                    "state_digest": str(pool.state_digest),
                    "state_digest_after": str(pool.state_digest_after),
                    "fixed_checkpoint_id": str(unit["checkpoint_id"]),
                    "cuda_gradient_seconds": float(pool.gradient_seconds),
                }
            )
    except BaseException as error:
        try:
            barrier.abort()
        except Exception:
            pass
        try:
            result_queue.put(
                {
                    "ok": False,
                    "gpu_uuid": str(gpu_uuid),
                    "error": f"{type(error).__name__}:{error}",
                }
            )
        except Exception:
            pass


def _aggregate_four_gpu_anchor_results(
    *,
    task: Mapping[str, Any],
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    io_hash: str,
) -> dict[str, Any]:
    """Strictly aggregate four actual process records into one anchor row."""

    expected = tuple(task.get("gpu_uuids", ()))
    if task.get("anchor_id") != "four-gpu-anchor" or task.get("device_count") != 4:
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_TASK_REQUIRED")
    if expected != tuple(APPROVED_GPU_UUIDS) or len(expected) != 4:
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_UUID_SET_INVALID")
    _sha(io_hash, field="four_gpu.io_evidence_hash")
    if len(results) != 4:
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_RESULT_COUNT_INVALID")
    by_uuid: dict[str, Mapping[str, Any]] = {}
    for result in results:
        if not isinstance(result, Mapping) or result.get("ok") is not True:
            raise S209ProductionBlocked("FOUR_GPU_ANCHOR_CHILD_FAILED")
        uuid = result.get("gpu_uuid")
        if uuid not in expected or uuid in by_uuid:
            raise S209ProductionBlocked("FOUR_GPU_ANCHOR_CHILD_UUID_DRIFT")
        by_uuid[str(uuid)] = result
    if set(by_uuid) != set(expected):
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_CHILD_UUID_SET_INVALID")
    required = {
        "health_ok", "wall_seconds", "phase", "sequence_count", "token_count", "backward_count",
        "communication_bytes", "allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes",
        "sample_mapping_hash", "gradient_pool_hash", "estimate_hash", "state_digest", "state_digest_after",
        "fixed_checkpoint_id", "cuda_gradient_seconds",
    }
    for uuid, result in by_uuid.items():
        if set(result) < required or result.get("health_ok") is not True:
            raise S209ProductionBlocked(f"FOUR_GPU_CHILD_HEALTH_INVALID:{uuid}")
        phase = result["phase"]
        if not isinstance(phase, Mapping) or set(phase) != set(S29_TIMING_FIELDS):
            raise S209ProductionBlocked(f"FOUR_GPU_CHILD_PHASE_FIELDS_INVALID:{uuid}")
        for name in S29_TIMING_FIELDS:
            value = phase[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise S209ProductionBlocked(f"FOUR_GPU_CHILD_PHASE_INVALID:{uuid}:{name}")
        wall = result["wall_seconds"]
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or float(wall) <= 0 or float(wall) + 1e-9 < sum(float(phase[name]) for name in S29_TIMING_FIELDS):
            raise S209ProductionBlocked(f"FOUR_GPU_CHILD_WALL_INVALID:{uuid}")
        for name in ("sequence_count", "token_count", "backward_count", "communication_bytes"):
            value = result[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise S209ProductionBlocked(f"FOUR_GPU_CHILD_COUNTER_INVALID:{uuid}:{name}")
        memory = [result[name] for name in ("allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes")]
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in memory) or not memory[0] <= memory[1] <= memory[2]:
            raise S209ProductionBlocked(f"FOUR_GPU_CHILD_MEMORY_INVALID:{uuid}")
        if not isinstance(result["fixed_checkpoint_id"], str) or not result["fixed_checkpoint_id"]:
            raise S209ProductionBlocked(f"FOUR_GPU_CHILD_CHECKPOINT_INVALID:{uuid}")
        for name in ("sample_mapping_hash", "gradient_pool_hash", "estimate_hash", "state_digest", "state_digest_after"):
            _sha(result[name], field=f"four_gpu.{uuid}.{name}")

    identity_fields = ("fixed_checkpoint_id", "sample_mapping_hash", "gradient_pool_hash", "estimate_hash", "state_digest", "state_digest_after")
    for name in identity_fields:
        if len({str(result[name]) for result in by_uuid.values()}) != 1:
            raise S209ProductionBlocked(f"FOUR_GPU_NUMERIC_CONSISTENCY_FAILED:{name}")
    slowest = max(by_uuid.values(), key=lambda result: float(result["wall_seconds"]))
    phase = {name: float(slowest["phase"][name]) for name in S29_TIMING_FIELDS}
    per_device = []
    for uuid in expected:
        result = by_uuid[uuid]
        per_device.append(
            {
                "gpu_uuid": uuid,
                "wall_seconds": float(result["wall_seconds"]),
                **{name: float(result["phase"][name]) for name in S29_TIMING_FIELDS},
                "sequence_count": int(result["sequence_count"]),
                "token_count": int(result["token_count"]),
                "backward_count": int(result["backward_count"]),
                "communication_bytes": int(result["communication_bytes"]),
                "allocated_peak_bytes": int(result["allocated_peak_bytes"]),
                "reserved_peak_bytes": int(result["reserved_peak_bytes"]),
                "device_peak_bytes": int(result["device_peak_bytes"]),
                "cuda_gradient_seconds": float(result["cuda_gradient_seconds"]),
            }
        )
    row: dict[str, Any] = {
        "measurement_kind": "device_actual",
        "measurement_source": "stage2_s209_production.synchronized_four_process",
        "measured": True,
        "semantic": "anchor",
        "method": "anchor",
        "run_id": task["run_id"],
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "gpu_uuids": list(expected),
        "device_count": 4,
        "health_ok": True,
        "numeric_consistency": True,
        "cost_io_quiescent": True,
        "io_evidence_hash": io_hash,
        **phase,
        "write_seconds": 0.0,
        "wall_seconds": float(slowest["wall_seconds"]),
        "allocated_peak_bytes": max(int(result["allocated_peak_bytes"]) for result in by_uuid.values()),
        "reserved_peak_bytes": max(int(result["reserved_peak_bytes"]) for result in by_uuid.values()),
        "device_peak_bytes": max(int(result["device_peak_bytes"]) for result in by_uuid.values()),
        "sequence_count": sum(int(result["sequence_count"]) for result in by_uuid.values()),
        "token_count": sum(int(result["token_count"]) for result in by_uuid.values()),
        "backward_count": sum(int(result["backward_count"]) for result in by_uuid.values()),
        "communication_bytes": sum(int(result["communication_bytes"]) for result in by_uuid.values()),
        "batch_size": config["frozen"]["batch_size"],
        "microbatch_count": config["frozen"]["microbatch_count"],
        "source_unit_ids": [str(task.get("unit_id", "four-gpu-fixed-anchor"))],
        "fixed_checkpoint_id": str(slowest["fixed_checkpoint_id"]),
        "cuda_gradient_seconds": max(float(result["cuda_gradient_seconds"]) for result in by_uuid.values()),
        "gradient_timing_source": "torch_cuda_event",
        "memory_counter_source": "torch_cuda_peak_and_mem_get_info",
        "system_anchor_mode": "synchronized_fixed_state_four_process",
        "communication_mode": "barrier_only_no_gradient_collective",
        "sample_mapping_hash": str(slowest["sample_mapping_hash"]),
        "gradient_pool_hash": str(slowest["gradient_pool_hash"]),
        "estimate_hash": str(slowest["estimate_hash"]),
        "four_process_identity_hash": canonical_json_hash(
            {
                "gpu_uuids": list(expected),
                "sample_mapping_hash": str(slowest["sample_mapping_hash"]),
                "gradient_pool_hash": str(slowest["gradient_pool_hash"]),
                "estimate_hash": str(slowest["estimate_hash"]),
                "state_digest": str(slowest["state_digest"]),
                "state_digest_after": str(slowest["state_digest_after"]),
            }
        ),
        "per_device_measurements": per_device,
    }
    if row["wall_seconds"] + 1e-9 < sum(float(row[name]) for name in S29_TIMING_FIELDS):
        raise S209ProductionBlocked("FOUR_GPU_AGGREGATE_WALL_INVALID")
    _row_bytes(row)
    write_start = time.perf_counter()
    _row_bytes(row)
    row["write_seconds"] = max(0.0, float(time.perf_counter() - write_start))
    row["wall_seconds"] = float(slowest["wall_seconds"]) + row["write_seconds"]
    return row


def _run_four_gpu_anchor(*, task: Mapping[str, Any], config: Mapping[str, Any], checked: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Launch the approved UUID set as four synchronized fixed-state workers."""

    expected = tuple(task.get("gpu_uuids", ()))
    if task.get("device_count") != 4 or expected != tuple(APPROVED_GPU_UUIDS):
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_UUID_BINDING_INVALID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(expected):
        raise S209ProductionBlocked("FOUR_GPU_PARENT_MASK_INVALID")
    _device_identity_set(expected)
    plan = load_s27_plan(root, str(checked["s27_plan_ref"]))
    cell_id = EXPECTED_CELL_IDS[0]
    unit_id = checked["frozen"]["expected_unit_ids"][0]
    mapping = load_s27_frozen_mappings(root, plan, cell_id=cell_id).get(unit_id)
    if mapping is None:
        raise S209ProductionBlocked("FOUR_GPU_ANCHOR_MAPPING_MISSING")
    binding = {"cell_id": cell_id, "unit_id": unit_id, "mapping_hash": mapping.digest}
    context = mp.get_context("spawn")
    barrier = context.Barrier(4)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_four_gpu_anchor_worker,
            kwargs={"gpu_uuid": uuid, "config": checked, "root": root, "binding": binding, "barrier": barrier, "result_queue": result_queue},
            name=f"s209-four-gpu-anchor-{index}",
        )
        for index, uuid in enumerate(expected)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=_FOUR_GPU_BARRIER_TIMEOUT_SECONDS + 60.0)
        alive = [process for process in processes if process.is_alive()]
        if alive:
            for process in alive:
                process.terminate()
            for process in alive:
                process.join(timeout=10.0)
            raise S209ProductionBlocked("FOUR_GPU_ANCHOR_PROCESS_TIMEOUT")
        if any(process.exitcode != 0 for process in processes):
            raise S209ProductionBlocked("FOUR_GPU_ANCHOR_PROCESS_FAILED")
        results: list[Mapping[str, Any]] = []
        for _ in expected:
            try:
                result = result_queue.get(timeout=10.0)
            except queue_module.Empty as error:
                raise S209ProductionBlocked("FOUR_GPU_ANCHOR_RESULT_MISSING") from error
            if not isinstance(result, Mapping):
                raise S209ProductionBlocked("FOUR_GPU_ANCHOR_RESULT_INVALID")
            results.append(dict(result))
        aggregate_task = dict(task)
        aggregate_task["unit_id"] = unit_id
        return _aggregate_four_gpu_anchor_results(task=aggregate_task, config=checked, results=results, io_hash=str(task["io_evidence_hash"]))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=10.0)


def _device_measurement_start() -> tuple[Any, float, int]:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    start.record()
    _total, free = torch.cuda.mem_get_info()
    return start, float(time.perf_counter()), int(free)


def _device_measurement_end(start_event: Any, wall_start: float, free_start: int) -> tuple[float, float, int, int, int]:
    import torch

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    end.synchronize()
    elapsed_cuda = float(start_event.elapsed_time(end)) / 1000.0
    wall = float(time.perf_counter() - wall_start)
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    _total, free_end = torch.cuda.mem_get_info()
    device_peak = max(int(_total - min(free_start, int(free_end))), reserved)
    if allocated <= 0 or reserved <= 0 or device_peak <= 0 or not allocated <= reserved <= device_peak:
        raise S209ProductionBlocked("CUDA_MEMORY_COUNTERS_INVALID")
    return wall, elapsed_cuda, allocated, reserved, device_peak


def _measure_pool(provider: Any, groups: Sequence[Sequence[Any]], *, mapping: RepetitionMapping) -> _MeasuredPool:
    import torch

    if not groups:
        raise S209ProductionBlocked("GRADIENT_GROUPS_EMPTY")
    before = provider.state_digest()
    forward_total = 0.0
    backward_total = 0.0
    forward_starts: list[float] = []
    backward_starts: list[float] = []
    module = getattr(getattr(provider, "_model", None), "module", None)
    handles: list[Any] = []
    if module is None or not hasattr(module, "register_forward_pre_hook"):
        raise S209ProductionBlocked("PROVIDER_MODEL_HOOKS_REQUIRED")

    def forward_pre(_module: Any, _inputs: Any) -> None:
        forward_starts.append(time.perf_counter())

    def forward_post(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal forward_total
        if not forward_starts:
            raise S209ProductionBlocked("FORWARD_HOOK_ORDER_INVALID")
        forward_total += time.perf_counter() - forward_starts.pop()

    def backward_pre(_module: Any, _grad_output: Any) -> None:
        backward_starts.append(time.perf_counter())

    def backward_post(_module: Any, _grad_input: Any, _grad_output: Any) -> None:
        nonlocal backward_total
        if not backward_starts:
            raise S209ProductionBlocked("BACKWARD_HOOK_ORDER_INVALID")
        backward_total += time.perf_counter() - backward_starts.pop()

    handles.extend(
        [
            module.register_forward_pre_hook(forward_pre),
            module.register_forward_hook(forward_post),
            module.register_full_backward_pre_hook(backward_pre),
            module.register_full_backward_hook(backward_post),
        ]
    )
    batches: list[Any] = []
    sample_ids: list[Any] = []
    start_event, wall_start, free_start = _device_measurement_start()
    try:
        for group in groups:
            batch = provider.gradient(group)
            if tuple(batch.sample_ids) != tuple(draw.sample_id for draw in group):
                raise S209ProductionBlocked("GRADIENT_SAMPLE_MAPPING_DRIFT")
            batches.append(batch)
            sample_ids.extend(batch.sample_ids)
    finally:
        for handle in handles:
            handle.remove()
    wall, cuda_seconds, allocated, reserved, device_peak = _device_measurement_end(start_event, wall_start, free_start)
    if forward_starts or backward_starts:
        raise S209ProductionBlocked("PROVIDER_HOOK_UNBALANCED")
    if forward_total <= 0 or backward_total <= 0:
        # A missing phase hook is a measurement failure, never a reason to
        # divide a combined timing into invented percentages.
        raise S209ProductionBlocked("PROVIDER_PHASE_TIMING_MISSING")
    if cuda_seconds <= 0 or wall <= 0:
        raise S209ProductionBlocked("CUDA_TIMING_COUNTERS_INVALID")
    if backward_total > wall + 1e-6 or forward_total > wall + 1e-6:
        raise S209ProductionBlocked("PROVIDER_PHASE_TIMING_INVALID")
    aggregation_start = time.perf_counter()
    kernel = CoreEstimatorKernel(accumulation_dtype="float64")
    maps = [kernel.tensor_map(batch) for batch in batches]
    aggregation_seconds = time.perf_counter() - aggregation_start
    if aggregation_seconds < 0:
        raise S209ProductionBlocked("GRADIENT_AGGREGATION_TIMING_INVALID")
    weights = [float(batch.statistical_weight) for batch in batches]
    total_tokens = sum(weights)
    if total_tokens <= 0 or not float(total_tokens).is_integer():
        raise S209ProductionBlocked("TOKEN_COUNTER_NOT_INTEGER")
    provider.assert_unchanged(before)
    after = provider.state_digest()
    mapping_hash = canonical_json_hash([draw.to_manifest() for draw in mapping.draws])
    gradient_identity = {
        "mapping_hash": mapping.digest,
        "sample_mapping_hash": mapping_hash,
        "groups": [list(batch.sample_ids) for batch in batches],
        "gradient_hashes": [_vector_digest(batch.gradients) for batch in batches],
    }
    gradient_hash = canonical_json_hash(gradient_identity)
    return _MeasuredPool(
        batches=batches,
        maps=maps,
        weights=weights,
        gradient_wall_seconds=float(wall),
        gradient_seconds=float(cuda_seconds),
        forward_seconds=float(forward_total),
        backward_seconds=float(backward_total),
        data_wait_seconds=max(0.0, float(wall) - float(forward_total) - float(backward_total)),
        sequence_count=mapping.batch_size,
        token_count=int(total_tokens),
        backward_count=len(groups),
        state_digest=before,
        state_digest_after=after,
        allocated_peak_bytes=allocated,
        reserved_peak_bytes=reserved,
        device_peak_bytes=device_peak,
        sample_mapping_hash=mapping_hash,
        gradient_pool_hash=gradient_hash,
        gradient_aggregation_seconds=float(aggregation_seconds),
    )


def _method_value(kernel: CoreEstimatorKernel, pool: _MeasuredPool, method: str, mapping: RepetitionMapping) -> tuple[Any, float]:
    start = time.perf_counter()
    if method == "raw":
        mean = kernel.weighted_mean(pool.maps, pool.weights)
        value = kernel.raw(mean)
    elif method == "double":
        half = len(pool.maps) // 2
        left = kernel.weighted_mean(pool.maps[:half], pool.weights[:half])
        right = kernel.weighted_mean(pool.maps[half:], pool.weights[half:])
        value = kernel.double(left, right)
    elif method == "u":
        weighting = pool.batches[0].weighting_assumptions
        value = kernel.u(pool.maps, pool.weights, **weighting)
    else:
        raise S209ProductionBlocked(f"METHOD_UNKNOWN:{method}")
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        raise S209ProductionBlocked("FORMULA_TIMING_INVALID")
    return value, float(elapsed)


def _row_bytes(row: dict[str, Any]) -> int:
    row["output_bytes"] = 0
    for _ in range(8):
        observed = len(canonical_json_bytes(row))
        if observed == row["output_bytes"]:
            return observed
        row["output_bytes"] = observed
    raise S209ProductionBlocked("OUTPUT_BYTE_COUNTER_UNSTABLE")


def _make_row(*, task: Mapping[str, Any], config: Mapping[str, Any], unit: Mapping[str, Any], method: str, pool: _MeasuredPool, formula_seconds: float, statistics_seconds: float, write_seconds: float, io_hash: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "measurement_kind": "device_actual",
        "measurement_source": "stage2_s209_production",
        "measured": True,
        "semantic": task["semantic"],
        "method": method,
        "run_id": task["run_id"],
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": task["device_count"],
        "health_ok": True,
        "cost_io_quiescent": task.get("cost_io_quiescent") is True,
        "io_evidence_hash": io_hash,
        "wall_seconds": 0.0,
        "cuda_gradient_seconds": pool.gradient_seconds,
        "gradient_timing_source": "torch_cuda_event",
        "memory_counter_source": "torch_cuda_peak_and_mem_get_info",
        "data_wait_seconds": pool.data_wait_seconds,
        "forward_seconds": pool.forward_seconds,
        "backward_seconds": pool.backward_seconds,
        "gradient_aggregation_seconds": pool.gradient_aggregation_seconds,
        "formula_seconds": formula_seconds,
        "statistics_seconds": statistics_seconds,
        "communication_seconds": 0.0,
        "write_seconds": write_seconds,
        "allocated_peak_bytes": pool.allocated_peak_bytes,
        "reserved_peak_bytes": pool.reserved_peak_bytes,
        "device_peak_bytes": pool.device_peak_bytes,
        "sequence_count": pool.sequence_count,
        "token_count": pool.token_count,
        "backward_count": pool.backward_count,
        "communication_bytes": 0,
        "source_unit_ids": [unit["unit_id"]],
        "batch_size": config["frozen"]["batch_size"],
        "microbatch_count": config["frozen"]["microbatch_count"],
        "fixed_checkpoint_id": unit["checkpoint_id"],
    }
    row["wall_seconds"] = (
        pool.gradient_wall_seconds
        + pool.gradient_aggregation_seconds
        + float(formula_seconds)
        + float(statistics_seconds)
        + float(write_seconds)
    )
    if row["wall_seconds"] <= 0:
        raise S209ProductionBlocked("ROW_WALL_TIMING_INVALID")
    _row_bytes(row)
    return row


def _binding(config: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    bindings = config.get("task_bindings")
    if not isinstance(bindings, list):
        raise S209ProductionBlocked("TASK_BINDINGS_REQUIRED")
    matches = [item for item in bindings if isinstance(item, Mapping) and item.get("semantic") == task["semantic"] and item.get("method") == task["method"] and item.get("anchor_id") == task["anchor_id"] and item.get("repetition") == task["repetition"]]
    if len(matches) != 1:
        raise S209ProductionBlocked("TASK_BINDING_NOT_UNIQUE")
    return dict(matches[0])


def _load_provider_for_binding(root: Path, config: Mapping[str, Any], binding: Mapping[str, Any]) -> tuple[Any, RepetitionMapping, dict[str, Any], S27CellPlan]:
    plan = load_s27_plan(root, str(config["s27_plan_ref"]))
    cell_id = binding.get("cell_id")
    unit_id = binding.get("unit_id")
    if cell_id not in EXPECTED_CELL_IDS or not isinstance(unit_id, str):
        raise S209ProductionBlocked("TASK_BINDING_CELL_OR_UNIT_INVALID")
    unit = next((item for item in plan.frozen_inputs.units if item.unit_id == unit_id and item.cell_id == cell_id), None)
    if unit is None:
        raise S209ProductionBlocked("TASK_BINDING_UNIT_NOT_IN_PLAN")
    mappings = load_s27_frozen_mappings(root, plan, cell_id=cell_id)
    mapping = mappings.get(unit_id)
    if mapping is None or mapping.digest != binding.get("mapping_hash"):
        raise S209ProductionBlocked("TASK_BINDING_MAPPING_HASH_DRIFT")
    materialized = load_s27_materialized_inputs(root, str(config["materialization_index_ref"]))[cell_id]
    cell = next(item for item in plan.cells if item.cell_id == cell_id)
    request = _request_for_cell(root, config, cell)
    context = build_s27_torch_provider(cell, data_root=root, checkpoint_root_ref=materialized.checkpoint_root_ref, registry_ref=materialized.registry_ref, request=request)
    cell_context = {
        "unit_id": unit_id,
        "checkpoint_id": cell.checkpoint_id,
        "checkpoint_hash": cell.checkpoint_hash,
        "registry_hash": context.registry_hash,
    }
    return context.provider, mapping, cell_context, cell


def run_s209_production_backend(*, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one UUID-bound S2.9 task with real fixed-state Torch gradients."""

    checked, root = _verify_runtime_config(config, task=task)
    if task.get("cost_io_quiescent") is not True or not isinstance(task.get("io_evidence_hash"), str):
        raise S209ProductionBlocked("LAUNCH_IO_EVIDENCE_REQUIRED")
    _sha(task["io_evidence_hash"], field="io_evidence_hash")
    if task.get("semantic") == "anchor" and task.get("anchor_id") == "four-gpu-anchor":
        return _run_four_gpu_anchor(task=task, config=config, checked=checked, root=root)
    if task.get("device_count") != 1:
        raise S209ProductionBlocked("PRODUCTION_BACKEND_FOUR_GPU_REQUIRES_DDP_ADAPTER")
    _device_identity(str(task["gpu_uuid"]))
    if task["semantic"] == "anchor":
        if task.get("anchor_id") != "single-gpu-anchor":
            raise S209ProductionBlocked("PRODUCTION_BACKEND_SYSTEM_FOUR_GPU_UNSUPPORTED")
        # The system anchor uses the first real frozen unit and executes one
        # complete estimator pool.  It is a measured runtime anchor, not a
        # label-only placeholder.
        binding = {"cell_id": EXPECTED_CELL_IDS[0], "unit_id": checked["frozen"]["expected_unit_ids"][0]}
        plan = load_s27_plan(root, str(checked["s27_plan_ref"]))
        mapping = load_s27_frozen_mappings(root, plan, cell_id=EXPECTED_CELL_IDS[0])[binding["unit_id"]]
        cell = next(item for item in plan.cells if item.cell_id == EXPECTED_CELL_IDS[0])
        material = load_s27_materialized_inputs(root, str(checked["materialization_index_ref"]))[cell.cell_id]
        provider = build_s27_torch_provider(cell, data_root=root, checkpoint_root_ref=material.checkpoint_root_ref, registry_ref=material.registry_ref, request=_request_for_cell(root, checked, cell)).provider
        pool = _measure_pool(provider, mapping.groups(max(mapping.m_values)), mapping=mapping)
        value, formula_seconds = _method_value(CoreEstimatorKernel(accumulation_dtype="float64"), pool, "raw", mapping)
        stats_start = time.perf_counter()
        _ = _vector_digest(value)
        statistics_seconds = time.perf_counter() - stats_start
        write_start = time.perf_counter()
        row = _make_row(task=task, config=checked, unit={"unit_id": binding["unit_id"], "checkpoint_id": cell.checkpoint_id}, method="anchor", pool=pool, formula_seconds=formula_seconds, statistics_seconds=statistics_seconds, write_seconds=0.0, io_hash=str(task["io_evidence_hash"]))
        row["numeric_consistency"] = True
        row["write_seconds"] = max(0.0, time.perf_counter() - write_start)
        row["wall_seconds"] = sum(float(row[name]) for name in S29_TIMING_FIELDS)
        _row_bytes(row)
        return row

    binding = _binding(checked, task)
    provider, mapping, unit, _cell = _load_provider_for_binding(root, checked, binding)
    kernel = CoreEstimatorKernel(accumulation_dtype="float64")
    method = "raw" if task["semantic"] == "scientific_equal_sample_cost" else str(task["method"])
    if task["semantic"] == "scientific_equal_sample_cost" and task["method"] != "shared":
        raise S209ProductionBlocked("SHARED_METHOD_ID_INVALID")
    if task["semantic"] != "scientific_equal_sample_cost" and method not in S29_METHODS:
        raise S209ProductionBlocked("METHOD_ID_INVALID")

    # Shared and isolated measurements deliberately choose their own provider
    # call graph.  Shared calls gradient() once per frozen primary-M group and
    # derives all three methods from those same returned batches.
    if task["semantic"] == "scientific_equal_sample_cost":
        groups = mapping.groups(max(mapping.m_values))
        pool = _measure_pool(provider, groups, mapping=mapping)
        values: dict[str, tuple[Any, float]] = {}
        for name in task["shared_method_order"]:
            values[name] = _method_value(kernel, pool, str(name), mapping)
        rows: list[dict[str, Any]] = []
        pool_body: dict[str, Any] = {
            "schema_version": S29_SHARED_POOL_SCHEMA,
            "paired_run_id": task["paired_run_id"],
            "paired_run_identity_hash": task["paired_run_identity_hash"],
            "measurement_plan_hash": task["measurement_plan_hash"],
            "matrix_hash": checked["frozen"]["matrix_hash"],
            "raw_manifest_hash": checked["frozen"]["raw_manifest_hash"],
            "source_raw_run_id": checked["frozen"]["raw_run_id"],
            "anchor_id": task["anchor_id"],
            "repetition": task["repetition"],
            "gpu_uuid": task["gpu_uuid"],
            "device_count": 1,
            "batch_size": checked["frozen"]["batch_size"],
            "microbatch_count": checked["frozen"]["microbatch_count"],
            "method_order": list(task["shared_method_order"]),
            "pool_id": canonical_json_hash({"paired_run_id": task["paired_run_id"], "mapping_hash": binding["mapping_hash"], "gradient_pool_hash": pool.gradient_pool_hash}),
            "sample_mapping_hash": pool.sample_mapping_hash,
            "gradient_pool_hash": pool.gradient_pool_hash,
            "cuda_gradient_seconds": pool.gradient_seconds,
            "gradient_timing_source": "torch_cuda_event",
            "sequence_count": pool.sequence_count,
            "token_count": pool.token_count,
            "backward_count": pool.backward_count,
            "cost_io_quiescent": task.get("cost_io_quiescent") is True,
            "shared_pool_ref": task["shared_pool_ref"],
        }
        pool_body["artifact_hash"] = canonical_json_hash(pool_body)
        for method_name in task["shared_method_order"]:
            stats_start = time.perf_counter()
            _ = _vector_digest(values[str(method_name)][0])
            stats_seconds = time.perf_counter() - stats_start
            write_start = time.perf_counter()
            row = _make_row(task=task, config=checked, unit=unit, method=str(method_name), pool=pool, formula_seconds=values[str(method_name)][1], statistics_seconds=stats_seconds, write_seconds=0.0, io_hash=str(task["io_evidence_hash"]))
            row.update({
                "paired_run_id": task["paired_run_id"],
                "paired_run_identity_hash": task["paired_run_identity_hash"],
                "measurement_plan_hash": task["measurement_plan_hash"],
                "shared_pool_id": pool_body["pool_id"],
                "shared_pool_artifact_hash": pool_body["artifact_hash"],
                "shared_pool_ref": task["shared_pool_ref"],
                "shared_sample_mapping_hash": pool_body["sample_mapping_hash"],
                "shared_gradient_pool_hash": pool_body["gradient_pool_hash"],
                "shared_method_order": list(task["shared_method_order"]),
                "shared_method_index": list(task["shared_method_order"]).index(str(method_name)),
                "shared_sample_sequence_count": pool_body["sequence_count"],
                "shared_sample_token_count": pool_body["token_count"],
            })
            row["write_seconds"] = max(0.0, time.perf_counter() - write_start)
            row["wall_seconds"] = sum(float(row[name]) for name in S29_TIMING_FIELDS)
            _row_bytes(row)
            rows.append(row)
        return {
            "schema_version": S29_SHARED_RUN_SCHEMA,
            "semantic": task["semantic"],
            "paired_run_id": task["paired_run_id"],
            "paired_run_identity_hash": task["paired_run_identity_hash"],
            "run_id": task["run_id"],
            "measurement_plan_hash": task["measurement_plan_hash"],
            "anchor_id": task["anchor_id"],
            "repetition": task["repetition"],
            "gpu_uuid": task["gpu_uuid"],
            "device_count": 1,
            "method_order": list(task["shared_method_order"]),
            "methods": list(S29_METHODS),
            "shared_pool": pool_body,
            "rows": rows,
        }

    if method == "double":
        groups = mapping.double_halves
    elif method == "u":
        groups = mapping.groups(max(mapping.m_values))
    else:
        groups = mapping.groups(max(mapping.m_values))
    pool = _measure_pool(provider, groups, mapping=mapping)
    stats_start = time.perf_counter()
    _value, formula_seconds = _method_value(kernel, pool, method, mapping)
    statistics_seconds = time.perf_counter() - stats_start
    write_start = time.perf_counter()
    row = _make_row(task=task, config=checked, unit=unit, method=method, pool=pool, formula_seconds=formula_seconds, statistics_seconds=statistics_seconds, write_seconds=0.0, io_hash=str(task["io_evidence_hash"]))
    row["write_seconds"] = max(0.0, time.perf_counter() - write_start)
    row["wall_seconds"] = sum(float(row[name]) for name in S29_TIMING_FIELDS)
    _row_bytes(row)
    return row


__all__ = [
    "S209_PRODUCTION_CONFIG_SCHEMA",
    "S209ProductionBlocked",
    "prepare_s209_profiler_config",
    "run_s209_production_backend",
]
