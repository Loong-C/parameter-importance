"""Executable S2.9 profiler-worker boundary.

The S2.9 runner owns scheduling (including the one serial shared paired
invocation).  This module owns the external-worker boundary: it binds the
runner's UUID/paired environment, invokes a real backend supplied by the
operator, and emits exactly one JSON object.  It deliberately never invents
timings, memory values, counters, or shared-pool identities.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from .stage2_s207_formal import APPROVED_GPU_UUIDS
from .stage2_s209_g27a import (
    S29_COUNT_FIELDS,
    S29_COST_SEMANTICS,
    S29_METHODS,
    S29_SHARED_POOL_SCHEMA,
    S29_SHARED_RUN_SCHEMA,
    S29_TIMING_FIELDS,
)


S209_WORKER_CONFIG_SCHEMA = "stage2-s209-profiler-worker-config-v1"
_ACTUAL_MARKERS = {"actual", "device_actual"}
_SHA256_LENGTH = 64


def _required_sha(environment: Mapping[str, str], name: str) -> str:
    value = _required_text(environment, name)
    if len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise S209WorkerBlocked(f"ENV_{name}_SHA256_REQUIRED")
    return value


class S209WorkerBlocked(RuntimeError):
    """Raised when the worker boundary cannot preserve formal identity."""


def _required_text(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise S209WorkerBlocked(f"ENV_{name}_REQUIRED")
    return value.strip()


def _required_int(environment: Mapping[str, str], name: str) -> int:
    value = _required_text(environment, name)
    try:
        result = int(value)
    except ValueError as error:
        raise S209WorkerBlocked(f"ENV_{name}_INTEGER_REQUIRED") from error
    if result < 0:
        raise S209WorkerBlocked(f"ENV_{name}_NONNEGATIVE_REQUIRED")
    return result


def _uuid_list(value: str, *, field: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise S209WorkerBlocked(f"{field}_INVALID")
    return result


def _task_from_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    semantic = _required_text(environment, "S29_SEMANTIC")
    method = _required_text(environment, "S29_METHOD")
    if semantic not in (*S29_COST_SEMANTICS, "anchor"):
        raise S209WorkerBlocked("ENV_S29_SEMANTIC_INVALID")
    if semantic == "anchor":
        if method != "anchor":
            raise S209WorkerBlocked("ENV_ANCHOR_METHOD_INVALID")
        if _required_text(environment, "S29_ANCHOR_ID") not in {"single-gpu-anchor", "four-gpu-anchor"}:
            raise S209WorkerBlocked("ENV_ANCHOR_ID_INVALID")
    elif semantic == "scientific_equal_sample_cost":
        if method != "shared":
            raise S209WorkerBlocked("ENV_SHARED_METHOD_REQUIRED")
    elif method not in S29_METHODS:
        raise S209WorkerBlocked("ENV_S29_METHOD_INVALID")
    visible = _uuid_list(_required_text(environment, "CUDA_VISIBLE_DEVICES"), field="CUDA_VISIBLE_DEVICES")
    declared = _uuid_list(_required_text(environment, "S29_GPU_UUIDS"), field="S29_GPU_UUIDS")
    if visible != declared:
        raise S209WorkerBlocked("ENV_GPU_UUID_BINDING_DRIFT")
    approved = list(APPROVED_GPU_UUIDS)
    if semantic != "anchor" and visible != [approved[0]]:
        raise S209WorkerBlocked("ENV_METHOD_SINGLE_APPROVED_GPU_REQUIRED")
    if semantic == "anchor":
        expected = [approved[0]] if len(visible) == 1 else approved
        if visible != expected:
            raise S209WorkerBlocked("ENV_ANCHOR_APPROVED_GPU_SET_INVALID")
    task: dict[str, Any] = {
        "semantic": semantic,
        "method": method,
        "run_id": _required_text(environment, "S29_RUN_ID"),
        "measurement_plan_hash": _required_sha(environment, "S29_PLAN_HASH"),
        "anchor_id": _required_text(environment, "S29_ANCHOR_ID"),
        "repetition": _required_int(environment, "S29_REPETITION"),
        "gpu_uuid": visible[0],
        "gpu_uuids": visible,
        "device_count": len(visible),
    }
    if semantic == "scientific_equal_sample_cost":
        if _required_text(environment, "S29_SHARED_RUN_SCHEMA") != S29_SHARED_RUN_SCHEMA:
            raise S209WorkerBlocked("ENV_SHARED_RUN_SCHEMA_INVALID")
        if _required_text(environment, "S29_SHARED_POOL_SCHEMA") != S29_SHARED_POOL_SCHEMA:
            raise S209WorkerBlocked("ENV_SHARED_POOL_SCHEMA_INVALID")
        try:
            method_order = json.loads(_required_text(environment, "S29_SHARED_METHOD_ORDER"))
        except (TypeError, ValueError) as error:
            raise S209WorkerBlocked("ENV_SHARED_METHOD_ORDER_JSON_INVALID") from error
        if not isinstance(method_order, list) or len(method_order) != len(S29_METHODS) or len(set(method_order)) != len(S29_METHODS) or set(method_order) != set(S29_METHODS):
            raise S209WorkerBlocked("ENV_SHARED_METHOD_ORDER_INVALID")
        task.update(
            {
                "paired_run_id": _required_text(environment, "S29_PAIRED_RUN_ID"),
                "paired_run_identity_hash": _required_sha(environment, "S29_PAIRED_RUN_IDENTITY_HASH"),
                "shared_method_order": method_order,
                "shared_pool_ref": _required_text(environment, "S29_SHARED_POOL_REF"),
            }
        )
    return task


def _load_config(path: str | Path, *, run_id: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise S209WorkerBlocked("WORKER_CONFIG_CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S209WorkerBlocked("WORKER_CONFIG_OBJECT_REQUIRED")
    config = dict(value)
    if config.get("schema_version") != S209_WORKER_CONFIG_SCHEMA or config.get("formal_eligible") is not True:
        raise S209WorkerBlocked("WORKER_CONFIG_FORMAL_SCHEMA_REQUIRED")
    declared = config.get("artifact_hash")
    if not isinstance(declared, str) or len(declared) != _SHA256_LENGTH:
        raise S209WorkerBlocked("WORKER_CONFIG_ARTIFACT_HASH_REQUIRED")
    if canonical_json_hash({key: item for key, item in config.items() if key != "artifact_hash"}) != declared:
        raise S209WorkerBlocked("WORKER_CONFIG_ARTIFACT_HASH_MISMATCH")
    config_run_id = config.get("run_id")
    if config_run_id is not None and config_run_id != run_id:
        raise S209WorkerBlocked("WORKER_CONFIG_RUN_ID_MISMATCH")
    return config


def _finite_json(value: Any, *, field: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise S209WorkerBlocked(f"{field}_NONFINITE")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, field=f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise S209WorkerBlocked(f"{field}_KEY_INVALID")
            _finite_json(item, field=f"{field}.{key}")
        return
    raise S209WorkerBlocked(f"{field}_JSON_INVALID")


def _require_actual_measurement(value: Mapping[str, Any], *, task: Mapping[str, Any]) -> None:
    if value.get("measurement_kind") not in _ACTUAL_MARKERS or value.get("measured") is not True:
        raise S209WorkerBlocked("PROFILER_MEASUREMENT_MARKER_REQUIRED")
    for name in S29_TIMING_FIELDS + ("wall_seconds", "allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes") + S29_COUNT_FIELDS:
        if name not in value:
            raise S209WorkerBlocked(f"PROFILER_MEASUREMENT_MISSING:{name}")
    if value.get("semantic") != task["semantic"] or value.get("run_id") != task["run_id"] or value.get("anchor_id") != task["anchor_id"] or value.get("repetition") != task["repetition"]:
        raise S209WorkerBlocked("PROFILER_TASK_IDENTITY_DRIFT")
    if value.get("gpu_uuid") != task["gpu_uuid"] or value.get("device_count") != task["device_count"]:
        raise S209WorkerBlocked("PROFILER_GPU_IDENTITY_DRIFT")
    if task["semantic"] != "anchor" and value.get("method") != task["method"]:
        raise S209WorkerBlocked("PROFILER_METHOD_IDENTITY_DRIFT")
    if task["semantic"] == "anchor" and value.get("gpu_uuids") != task["gpu_uuids"]:
        raise S209WorkerBlocked("PROFILER_ANCHOR_GPU_SET_DRIFT")
    if value.get("health_ok") is not True or value.get("cost_io_quiescent") is not True:
        raise S209WorkerBlocked("PROFILER_HEALTH_OR_IO_NOT_PASS")
    for name in S29_TIMING_FIELDS:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) < 0:
            raise S209WorkerBlocked(f"PROFILER_PHASE_SECONDS_INVALID:{name}")
    wall = value["wall_seconds"]
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or float(wall) <= 0:
        raise S209WorkerBlocked("PROFILER_WALL_SECONDS_INVALID")
    if float(wall) + 1e-9 < sum(float(value[name]) for name in S29_TIMING_FIELDS):
        raise S209WorkerBlocked("PROFILER_WALL_PHASE_TOTAL_INVALID")
    for name in S29_COUNT_FIELDS:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise S209WorkerBlocked(f"PROFILER_COUNT_INVALID:{name}")
    for name in ("allocated_peak_bytes", "reserved_peak_bytes", "device_peak_bytes"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise S209WorkerBlocked(f"PROFILER_MEMORY_INVALID:{name}")
    if not value["allocated_peak_bytes"] <= value["reserved_peak_bytes"] <= value["device_peak_bytes"]:
        raise S209WorkerBlocked("PROFILER_MEMORY_ORDER_INVALID")


def _validate_backend_output(value: Any, *, task: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S209WorkerBlocked("PROFILER_OUTPUT_OBJECT_REQUIRED")
    output = dict(value)
    _finite_json(output, field="profiler_output")
    if task["semantic"] == "scientific_equal_sample_cost":
        if output.get("schema_version") != S29_SHARED_RUN_SCHEMA or output.get("semantic") != task["semantic"]:
            raise S209WorkerBlocked("PROFILER_SHARED_RUN_SCHEMA_INVALID")
        if output.get("paired_run_id") != task["paired_run_id"] or output.get("paired_run_identity_hash") != task["paired_run_identity_hash"] or output.get("run_id") != task["run_id"] or output.get("measurement_plan_hash") != task["measurement_plan_hash"] or output.get("anchor_id") != task["anchor_id"] or output.get("repetition") != task["repetition"] or output.get("gpu_uuid") != task["gpu_uuid"] or output.get("device_count") != 1:
            raise S209WorkerBlocked("PROFILER_SHARED_RUN_IDENTITY_DRIFT")
        if output.get("method_order") != task["shared_method_order"] or output.get("methods") != list(S29_METHODS):
            raise S209WorkerBlocked("PROFILER_SHARED_METHOD_ORDER_INVALID")
        pool = output.get("shared_pool")
        required_pool_fields = {
            "schema_version", "paired_run_id", "paired_run_identity_hash", "measurement_plan_hash",
            "matrix_hash", "raw_manifest_hash", "source_raw_run_id", "anchor_id", "repetition",
            "gpu_uuid", "device_count", "batch_size", "microbatch_count", "method_order", "pool_id",
            "sample_mapping_hash", "gradient_pool_hash", "sequence_count", "token_count", "backward_count",
            "cost_io_quiescent", "shared_pool_ref", "artifact_hash",
        }
        if not isinstance(pool, Mapping) or not required_pool_fields.issubset(pool) or pool.get("schema_version") != S29_SHARED_POOL_SCHEMA or pool.get("paired_run_id") != task["paired_run_id"] or pool.get("paired_run_identity_hash") != task["paired_run_identity_hash"] or pool.get("measurement_plan_hash") != task["measurement_plan_hash"] or pool.get("shared_pool_ref") != task["shared_pool_ref"] or pool.get("gpu_uuid") != task["gpu_uuid"] or pool.get("device_count") != 1 or pool.get("cost_io_quiescent") is not True:
            raise S209WorkerBlocked("PROFILER_SHARED_POOL_IDENTITY_INVALID")
        if pool.get("method_order") != task["shared_method_order"]:
            raise S209WorkerBlocked("PROFILER_SHARED_POOL_CONTENT_INVALID")
        for name in ("sequence_count", "token_count", "backward_count"):
            item = pool.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise S209WorkerBlocked(f"PROFILER_SHARED_POOL_COUNT_INVALID:{name}")
        if canonical_json_hash({key: item for key, item in pool.items() if key != "artifact_hash"}) != pool.get("artifact_hash"):
            raise S209WorkerBlocked("PROFILER_SHARED_POOL_HASH_MISMATCH")
        for name in ("matrix_hash", "raw_manifest_hash", "pool_id", "sample_mapping_hash", "gradient_pool_hash"):
            item = pool.get(name)
            if not isinstance(item, str) or len(item) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in item):
                raise S209WorkerBlocked(f"PROFILER_SHARED_POOL_SHA256_REQUIRED:{name}")
        rows = output.get("rows")
        if not isinstance(rows, list) or len(rows) != len(S29_METHODS):
            raise S209WorkerBlocked("PROFILER_SHARED_RUN_ROWS_REQUIRED")
        methods = {str(row.get("method")) for row in rows if isinstance(row, Mapping)}
        if methods != set(S29_METHODS):
            raise S209WorkerBlocked("PROFILER_SHARED_RUN_METHOD_SET_INVALID")
        for row in rows:
            if not isinstance(row, Mapping):
                raise S209WorkerBlocked("PROFILER_SHARED_RUN_ROW_INVALID")
            row_task = dict(task)
            row_task["method"] = str(row.get("method"))
            _require_actual_measurement(row, task=row_task)
            required_shared_fields = {
                "paired_run_id", "paired_run_identity_hash", "measurement_plan_hash", "shared_pool_id",
                "shared_pool_artifact_hash", "shared_pool_ref", "shared_sample_mapping_hash",
                "shared_gradient_pool_hash", "shared_method_order", "shared_method_index",
                "shared_sample_sequence_count", "shared_sample_token_count",
            }
            if not required_shared_fields.issubset(row):
                raise S209WorkerBlocked("PROFILER_SHARED_ROW_FIELDS_REQUIRED")
            expected_shared = {
                "paired_run_id": task["paired_run_id"],
                "paired_run_identity_hash": task["paired_run_identity_hash"],
                "measurement_plan_hash": task["measurement_plan_hash"],
                "shared_pool_id": pool["pool_id"],
                "shared_pool_artifact_hash": pool["artifact_hash"],
                "shared_pool_ref": task["shared_pool_ref"],
                "shared_sample_mapping_hash": pool["sample_mapping_hash"],
                "shared_gradient_pool_hash": pool["gradient_pool_hash"],
                "shared_method_order": task["shared_method_order"],
                "shared_method_index": task["shared_method_order"].index(str(row.get("method"))),
                "shared_sample_sequence_count": pool["sequence_count"],
                "shared_sample_token_count": pool["token_count"],
            }
            if any(row.get(name) != expected for name, expected in expected_shared.items()):
                raise S209WorkerBlocked("PROFILER_SHARED_ROW_IDENTITY_DRIFT")
            if row.get("sequence_count") != pool["sequence_count"] or row.get("token_count") != pool["token_count"] or row.get("backward_count") != pool["backward_count"]:
                raise S209WorkerBlocked("PROFILER_SHARED_SAMPLE_COUNT_MISMATCH")
        return output
    _require_actual_measurement(output, task=task)
    return output


def run_s209_profiler_worker(
    *,
    backend: Callable[..., Mapping[str, Any]],
    config: Mapping[str, Any],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Invoke one real backend under the frozen runner environment."""

    env = dict(os.environ if environment is None else environment)
    task = _task_from_environment(env)
    if not isinstance(config, Mapping):
        raise S209WorkerBlocked("WORKER_CONFIG_OBJECT_REQUIRED")
    if config.get("schema_version") != S209_WORKER_CONFIG_SCHEMA or config.get("formal_eligible") is not True:
        raise S209WorkerBlocked("WORKER_CONFIG_FORMAL_SCHEMA_REQUIRED")
    declared = config.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash({key: item for key, item in config.items() if key != "artifact_hash"}) != declared:
        raise S209WorkerBlocked("WORKER_CONFIG_ARTIFACT_HASH_MISMATCH")
    if config.get("run_id") is not None and config.get("run_id") != task["run_id"]:
        raise S209WorkerBlocked("WORKER_CONFIG_RUN_ID_MISMATCH")
    try:
        with redirect_stdout(sys.stderr):
            result = backend(task=task, config=dict(config))
    except S209WorkerBlocked:
        raise
    except Exception as error:
        raise S209WorkerBlocked(f"PROFILER_BACKEND_FAILED:{type(error).__name__}:{error}") from error
    return _validate_backend_output(result, task=task)


def load_backend(specification: str) -> Callable[..., Mapping[str, Any]]:
    """Load a backend function specified as ``python.module:function``."""

    if not isinstance(specification, str) or specification.count(":") != 1:
        raise S209WorkerBlocked("BACKEND_SPEC_MUST_BE_MODULE_COLON_FUNCTION")
    module_name, function_name = specification.split(":", 1)
    if not module_name or not function_name:
        raise S209WorkerBlocked("BACKEND_SPEC_EMPTY")
    try:
        module = importlib.import_module(module_name)
        backend = getattr(module, function_name)
    except (ImportError, AttributeError) as error:
        raise S209WorkerBlocked("BACKEND_IMPORT_FAILED") from error
    if not callable(backend):
        raise S209WorkerBlocked("BACKEND_CALLABLE_REQUIRED")
    return backend


def execute_cli(*, backend_spec: str, config_path: str | Path) -> dict[str, Any]:
    config = _load_config(config_path, run_id=_required_text(os.environ, "S29_RUN_ID"))
    return run_s209_profiler_worker(backend=load_backend(backend_spec), config=config)


__all__ = [
    "S209_WORKER_CONFIG_SCHEMA",
    "S209WorkerBlocked",
    "execute_cli",
    "load_backend",
    "run_s209_profiler_worker",
]
