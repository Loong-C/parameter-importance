"""Fail-closed S2.9/G2.7a cost and system validation.

This module is a detached consumer.  It does not start a worker, enumerate a
GPU, or launch a CUDA wave.  A caller supplies the already sealed G2.4b
matrix, the S2.7 raw-results manifest, health evidence, and profiler rows.
Scientific rows must carry the paired-run and shared-gradient-pool identity
published by the one serial paired worker; the reducer rejects rows that only
claim the shared semantic.
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
from pathlib import PurePosixPath
import random
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord, GateStatus
from ..core.oracles import compare_tensor_maps_fp64
from ..core.tensors import TensorMap
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import APPROVED_GPU_UUIDS, EXCLUDED_GPU_UUID, EXCLUDED_PCI, S27_RAW_MANIFEST_SCHEMA


S29_SCHEMA = "stage2-s209-g27a-cost-system-validation-v1"
S29_GATE_SCHEMA = "stage2-s209-g27a-gate-v1"
S29_MEASUREMENT_PLAN_SCHEMA = "stage2-s209-g27a-measurement-plan-v1"
S29_SHARED_RUN_SCHEMA = "stage2-s209-g27a-shared-paired-run-v1"
S29_SHARED_POOL_SCHEMA = "stage2-s209-g27a-shared-gradient-pool-v1"
S29_CROSSCHECK_SCHEMA = "stage2-s209-g27a-shared-attribution-crosscheck-v1"
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
S29_STAGE1_NUMERIC_ARTIFACT_SCHEMA = "stage2-s209-stage1-raw-numeric-artifact-v1"
S29_STAGE1_NUMERIC_COMPARISON_SCHEMA = "stage2-s209-stage1-raw-numeric-comparison-v1"
# Formal references identify persisted files, never JSON fragments.  The
# anchor's embedded ``stage1_numeric_artifact`` is checked against this file
# after reload, so a fragment would make the provenance ambiguous.
S29_STAGE1_NUMERIC_REFERENCE_REF = "anchors/single-gpu-anchor.json"
S29_STAGE1_NUMERIC_CANDIDATE_REF = "anchors/four-gpu-stage1-numeric.json"
S29_STAGE1_TOLERANCE_PROFILE = {
    "name": "T64_ORACLE",
    "comparison_dtype": "torch.float64_cpu",
    "natural_scale": 16.0,
    "atol": 1e-12,
    "rtol": 1e-10,
    "normalized_l2_limit": 1e-10,
}
S29_EXECUTION_IDENTITY_SCHEMA = "stage2-s209-execution-identity-v1"
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")


def _valid_stage1_numeric_ref(value: Any, *, suffix: str, field: str) -> str:
    """Validate a DATA_ROOT-relative POSIX artifact reference."""

    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/") or ":" in value:
        raise S29G27ABlocked(f"{field}:DATA_ROOT_POSIX_REF_REQUIRED")
    if "#" in value:
        raise S29G27ABlocked(f"{field}:DATA_ROOT_POSIX_REF_FRAGMENT_FORBIDDEN")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts) or not value.endswith(suffix):
        raise S29G27ABlocked(f"{field}:DATA_ROOT_POSIX_REF_INVALID")
    return value


def _formal_file(root: Path, reference: Any, *, filename: str, field: str) -> tuple[dict[str, Any], str]:
    """Reload one persisted formal anchor; caller mappings never grant authority."""

    if not isinstance(reference, str) or not reference or "#" in reference or "\\" in reference:
        raise S29G27ABlocked(f"{field}:PERSISTED_POSIX_REF_REQUIRED")
    path_ref = PurePosixPath(reference)
    if path_ref.is_absolute() or any(part in {"", ".", ".."} for part in path_ref.parts) or not reference.endswith(filename):
        raise S29G27ABlocked(f"{field}:PERSISTED_POSIX_REF_INVALID")
    current = root.resolve()
    for part in path_ref.parts:
        current = current / part
        if current.is_symlink():
            raise S29G27ABlocked(f"{field}:SYMLINK_COMPONENT_FORBIDDEN")
    path = current.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise S29G27ABlocked(f"{field}:PATH_ESCAPE") from error
    if path.name != filename or not path.is_file():
        raise S29G27ABlocked(f"{field}:PERSISTED_FILE_REQUIRED")
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise S29G27ABlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S29G27ABlocked(f"{field}:OBJECT_REQUIRED")
    payload = dict(value)
    declared = payload.get("artifact_hash")
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise S29G27ABlocked(f"{field}:ARTIFACT_HASH_REQUIRED")
    if canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"}) != declared:
        raise S29G27ABlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return payload, reference


def _validate_execution_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S29G27ABlocked("EXECUTION_IDENTITY_REQUIRED")
    payload = dict(value)
    required = {
        "schema_version",
        "repository_head",
        "launcher_source_sha256",
        "profiler_command_hash",
        "repository_clean",
        "artifact_hash",
    }
    if set(payload) != required or payload.get("schema_version") != S29_EXECUTION_IDENTITY_SCHEMA:
        raise S29G27ABlocked("EXECUTION_IDENTITY_SCHEMA_INVALID")
    if not isinstance(payload.get("repository_head"), str) or _GIT_HEAD.fullmatch(payload["repository_head"]) is None:
        raise S29G27ABlocked("EXECUTION_IDENTITY_HEAD_INVALID")
    for name in ("launcher_source_sha256", "profiler_command_hash", "artifact_hash"):
        _sha(payload.get(name), field=f"execution_identity.{name}")
    if payload.get("repository_clean") is not True:
        raise S29G27ABlocked("EXECUTION_IDENTITY_REPOSITORY_DIRTY")
    if canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"}) != payload["artifact_hash"]:
        raise S29G27ABlocked("EXECUTION_IDENTITY_HASH_MISMATCH")
    return payload
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
S29_SHARED_ROW_FIELDS = (
    "paired_run_id",
    "paired_run_identity_hash",
    "measurement_plan_hash",
    "shared_pool_id",
    "shared_pool_artifact_hash",
    "shared_pool_ref",
    "shared_sample_mapping_hash",
    "shared_gradient_pool_hash",
    "shared_method_order",
    "shared_method_index",
    "shared_sample_sequence_count",
    "shared_sample_token_count",
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


def _stage1_numeric_wire(value: Any) -> dict[str, dict[str, Any]]:
    """Serialize an actual estimator TensorMap without dropping coordinates."""

    try:
        import torch

        items = list(value.items())
    except (AttributeError, ImportError) as error:
        raise S29G27ABlocked("STAGE1_NUMERIC_TENSOR_MAP_REQUIRED") from error
    if not items:
        raise S29G27ABlocked("STAGE1_NUMERIC_TENSOR_MAP_EMPTY")
    output: dict[str, dict[str, Any]] = {}
    for name, tensor in items:
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise S29G27ABlocked("STAGE1_NUMERIC_TENSOR_MAP_INVALID")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise S29G27ABlocked("STAGE1_NUMERIC_TENSOR_NONFINITE")
        output[name] = {
            "dtype": str(tensor.dtype),
            "shape": [int(size) for size in tensor.shape],
            "values": [float(item) for item in tensor.detach().to(dtype=torch.float64).reshape(-1).tolist()],
        }
    return output


def _stage1_numeric_artifact(
    value: Any,
    *,
    role: str,
    fixed_checkpoint_id: str,
    checkpoint_hash: str,
    mapping_hash: str,
    sample_mapping_hash: str,
    state_digest: str,
    state_digest_after: str,
    statistical_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the content-addressed raw-estimator artifact used by S2.9."""

    if role not in {"single_gpu_reference", "four_gpu_candidate"}:
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_ROLE_INVALID")
    if not isinstance(fixed_checkpoint_id, str) or not fixed_checkpoint_id:
        raise S29G27ABlocked("STAGE1_NUMERIC_CHECKPOINT_ID_REQUIRED")
    for name, item in (
        ("checkpoint_hash", checkpoint_hash),
        ("mapping_hash", mapping_hash),
        ("sample_mapping_hash", sample_mapping_hash),
        ("state_digest", state_digest),
        ("state_digest_after", state_digest_after),
    ):
        _sha(item, field=f"stage1_numeric.{name}")
    wire = _stage1_numeric_wire(value)
    binding = dict(statistical_binding or {})
    required_binding = {
        "global_s1_hash",
        "global_s2_hash",
        "estimate_hash",
        "global_weight",
        "global_statistical_unit_count",
    }
    if set(binding) != required_binding:
        raise S29G27ABlocked("STAGE1_NUMERIC_STATISTICAL_BINDING_REQUIRED")
    for name in ("global_s1_hash", "global_s2_hash", "estimate_hash"):
        _sha(binding[name], field=f"stage1_numeric.statistical_binding.{name}")
    global_weight = binding["global_weight"]
    global_count = binding["global_statistical_unit_count"]
    if isinstance(global_weight, bool) or not isinstance(global_weight, (int, float)) or not math.isfinite(float(global_weight)) or float(global_weight) <= 0:
        raise S29G27ABlocked("STAGE1_NUMERIC_GLOBAL_WEIGHT_INVALID")
    if isinstance(global_count, bool) or not isinstance(global_count, int) or global_count <= 0:
        raise S29G27ABlocked("STAGE1_NUMERIC_GLOBAL_COUNT_INVALID")
    body: dict[str, Any] = {
        "schema_version": S29_STAGE1_NUMERIC_ARTIFACT_SCHEMA,
        "role": role,
        "estimator": "raw",
        "fixed_checkpoint_id": fixed_checkpoint_id,
        "checkpoint_hash": checkpoint_hash,
        "mapping_hash": mapping_hash,
        "sample_mapping_hash": sample_mapping_hash,
        "state_digest": state_digest,
        "state_digest_after": state_digest_after,
        "statistical_binding": binding,
        "value": wire,
        "value_hash": canonical_json_hash(wire),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _stage1_numeric_map_from_wire(value: Any) -> TensorMap:
    try:
        import torch

        if not isinstance(value, Mapping) or not value:
            raise S29G27ABlocked("STAGE1_NUMERIC_VALUE_INVALID")
        tensors: dict[str, torch.Tensor] = {}
        for name, entry in value.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                raise S29G27ABlocked("STAGE1_NUMERIC_VALUE_INVALID")
            shape = entry.get("shape")
            values = entry.get("values")
            dtype = entry.get("dtype")
            if not isinstance(dtype, str) or dtype not in {"torch.float32", "torch.float64"}:
                raise S29G27ABlocked("STAGE1_NUMERIC_DTYPE_INVALID")
            if not isinstance(shape, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
                raise S29G27ABlocked("STAGE1_NUMERIC_SHAPE_INVALID")
            if not isinstance(values, list) or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in values):
                raise S29G27ABlocked("STAGE1_NUMERIC_VALUES_INVALID")
            expected_count = 1
            for size in shape:
                expected_count *= size
            if len(values) != expected_count:
                raise S29G27ABlocked("STAGE1_NUMERIC_VALUE_LENGTH_INVALID")
            tensors[name] = torch.tensor(values, dtype=torch.float64).reshape(tuple(shape))
        return TensorMap(tensors, require_finite=True)
    except S29G27ABlocked:
        raise
    except Exception as error:
        raise S29G27ABlocked("STAGE1_NUMERIC_VALUE_INVALID") from error


def _validate_stage1_numeric_artifact(value: Any, *, role: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_REQUIRED")
    output = dict(value)
    required = {
        "schema_version", "role", "estimator", "fixed_checkpoint_id", "checkpoint_hash", "mapping_hash",
        "sample_mapping_hash", "state_digest", "state_digest_after", "value", "value_hash", "artifact_hash",
        "statistical_binding",
    }
    if set(output) != required or output.get("schema_version") != S29_STAGE1_NUMERIC_ARTIFACT_SCHEMA or output.get("estimator") != "raw":
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_SCHEMA_INVALID")
    if role is not None and output.get("role") != role:
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_ROLE_DRIFT")
    if not isinstance(output.get("role"), str) or output["role"] not in {"single_gpu_reference", "four_gpu_candidate"}:
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_ROLE_INVALID")
    if not isinstance(output.get("fixed_checkpoint_id"), str) or not output["fixed_checkpoint_id"]:
        raise S29G27ABlocked("STAGE1_NUMERIC_CHECKPOINT_ID_REQUIRED")
    for name in ("checkpoint_hash", "mapping_hash", "sample_mapping_hash", "state_digest", "state_digest_after", "value_hash", "artifact_hash"):
        _sha(output.get(name), field=f"stage1_numeric.{name}")
    _stage1_numeric_map_from_wire(output["value"])
    binding = output["statistical_binding"]
    if not isinstance(binding, Mapping) or set(binding) != {
        "global_s1_hash", "global_s2_hash", "estimate_hash", "global_weight", "global_statistical_unit_count"
    }:
        raise S29G27ABlocked("STAGE1_NUMERIC_STATISTICAL_BINDING_INVALID")
    for name in ("global_s1_hash", "global_s2_hash", "estimate_hash"):
        _sha(binding.get(name), field=f"stage1_numeric.statistical_binding.{name}")
    if isinstance(binding.get("global_weight"), bool) or not isinstance(binding.get("global_weight"), (int, float)) or not math.isfinite(float(binding["global_weight"])) or float(binding["global_weight"]) <= 0:
        raise S29G27ABlocked("STAGE1_NUMERIC_GLOBAL_WEIGHT_INVALID")
    if isinstance(binding.get("global_statistical_unit_count"), bool) or not isinstance(binding.get("global_statistical_unit_count"), int) or binding["global_statistical_unit_count"] <= 0:
        raise S29G27ABlocked("STAGE1_NUMERIC_GLOBAL_COUNT_INVALID")
    if output["value_hash"] != canonical_json_hash(output["value"]):
        raise S29G27ABlocked("STAGE1_NUMERIC_VALUE_HASH_MISMATCH")
    if output["artifact_hash"] != canonical_json_hash({key: item for key, item in output.items() if key != "artifact_hash"}):
        raise S29G27ABlocked("STAGE1_NUMERIC_ARTIFACT_HASH_MISMATCH")
    return output


def _stage1_numeric_comparison(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    reference_ref: str = S29_STAGE1_NUMERIC_REFERENCE_REF,
) -> dict[str, Any]:
    """Compare candidate/reference with the frozen Stage 1 FP64 comparator."""

    candidate = _validate_stage1_numeric_artifact(candidate, role="four_gpu_candidate")
    reference = _validate_stage1_numeric_artifact(reference, role="single_gpu_reference")
    _valid_stage1_numeric_ref(reference_ref, suffix="anchors/single-gpu-anchor.json", field="stage1_numeric.reference_ref")
    for name in ("fixed_checkpoint_id", "checkpoint_hash", "mapping_hash", "sample_mapping_hash", "state_digest", "state_digest_after"):
        if candidate[name] != reference[name]:
            raise S29G27ABlocked(f"STAGE1_NUMERIC_IDENTITY_DRIFT:{name}")
    try:
        actual = _stage1_numeric_map_from_wire(candidate["value"])
        oracle = _stage1_numeric_map_from_wire(reference["value"])
        settings = {key: value for key, value in S29_STAGE1_TOLERANCE_PROFILE.items() if key in {"natural_scale", "atol", "rtol", "normalized_l2_limit"}}
        global_result = compare_tensor_maps_fp64(actual, oracle, **settings).to_dict()
        per_tensor: list[dict[str, Any]] = []
        for name in actual:
            result = compare_tensor_maps_fp64(
                TensorMap({name: actual[name]}),
                TensorMap({name: oracle[name]}),
                **settings,
            ).to_dict()
            per_tensor.append({"parameter_name": name, **result})
    except Exception as error:
        raise S29G27ABlocked("STAGE1_NUMERIC_COMPARISON_UNCOMPARABLE") from error
    body: dict[str, Any] = {
        "schema_version": S29_STAGE1_NUMERIC_COMPARISON_SCHEMA,
        "estimator": "raw",
        "profile": dict(S29_STAGE1_TOLERANCE_PROFILE),
        "candidate_artifact_hash": candidate["artifact_hash"],
        "reference_artifact_hash": reference["artifact_hash"],
        "reference_ref": reference_ref,
        "global": global_result,
        "per_tensor": per_tensor,
        "passed": bool(global_result["passed"]) and all(bool(item["passed"]) for item in per_tensor),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _validate_stage1_numeric_comparison(
    value: Any,
    *,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_ref: str = S29_STAGE1_NUMERIC_REFERENCE_REF,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S29G27ABlocked("STAGE1_NUMERIC_COMPARISON_REQUIRED")
    expected = _stage1_numeric_comparison(candidate, reference, reference_ref=reference_ref)
    observed = dict(value)
    if observed != expected:
        raise S29G27ABlocked("STAGE1_NUMERIC_COMPARISON_DRIFT")
    if observed.get("passed") is not True:
        raise S29G27ABlocked("STAGE1_NUMERIC_COMPARISON_FAILED")
    return observed


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


def shared_paired_run_identity(
    *,
    run_id: str,
    measurement_plan_hash: str,
    matrix_hash: str,
    raw_manifest_hash: str,
    source_raw_run_id: str,
    anchor_id: str,
    repetition: int,
    gpu_uuid: str,
    device_count: int,
    method_order: Sequence[str],
) -> dict[str, Any]:
    """Return the deterministic identity for one shared scientific pair.

    A shared pair is one serial worker invocation over one sample/gradient
    pool.  The identity includes every frozen input that can change the pool,
    plus the planned method order, so a row cannot be rebound to another
    pooled run by changing only its semantic label.
    """

    run_id = _id(run_id, field="shared_pair.run_id")
    measurement_plan_hash = _sha(measurement_plan_hash, field="shared_pair.measurement_plan_hash")
    matrix_hash = _sha(matrix_hash, field="shared_pair.matrix_hash")
    raw_manifest_hash = _sha(raw_manifest_hash, field="shared_pair.raw_manifest_hash")
    source_raw_run_id = _id(source_raw_run_id, field="shared_pair.source_raw_run_id")
    anchor_id = _id(anchor_id, field="shared_pair.anchor_id")
    if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
        raise S29G27ABlocked("shared_pair.repetition:INTEGER_REQUIRED")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise S29G27ABlocked("shared_pair.gpu_uuid:REQUIRED")
    _positive(device_count, field="shared_pair.device_count")
    normalized_order = list(method_order)
    if len(normalized_order) != len(S29_METHODS) or len(set(normalized_order)) != len(S29_METHODS) or set(normalized_order) != set(S29_METHODS):
        raise S29G27ABlocked("shared_pair.method_order:METHOD_SET_INVALID")
    body: dict[str, Any] = {
        "schema_version": S29_SHARED_RUN_SCHEMA,
        "run_id": run_id,
        "measurement_plan_hash": measurement_plan_hash,
        "matrix_hash": matrix_hash,
        "raw_manifest_hash": raw_manifest_hash,
        "source_raw_run_id": source_raw_run_id,
        "anchor_id": anchor_id,
        "repetition": repetition,
        "gpu_uuid": gpu_uuid,
        "device_count": device_count,
        "method_order": normalized_order,
        "methods": list(S29_METHODS),
    }
    identity_hash = canonical_json_hash(body)
    return {
        **body,
        "paired_run_id": f"paired-{identity_hash}",
        "paired_run_identity_hash": identity_hash,
    }


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
    _id(payload.get("run_id"), field="measurement_plan.run_id")
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
    result["method"] = _method(result.get("method"), field=f"{semantic}.method")
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
    if semantic == S29_COST_SEMANTICS[0]:
        if result["method"] not in S29_METHODS:
            raise S29G27ABlocked(f"{semantic}:METHOD_REQUIRED")
        for name in S29_SHARED_ROW_FIELDS:
            if name not in result:
                raise S29G27ABlocked(f"{semantic}:{name}:REQUIRED")
        _id(result["paired_run_id"], field=f"{semantic}.paired_run_id")
        _sha(result["paired_run_identity_hash"], field=f"{semantic}.paired_run_identity_hash")
        _sha(result["measurement_plan_hash"], field=f"{semantic}.measurement_plan_hash")
        _sha(result["shared_pool_id"], field=f"{semantic}.shared_pool_id")
        _sha(result["shared_pool_artifact_hash"], field=f"{semantic}.shared_pool_artifact_hash")
        _sha(result["shared_sample_mapping_hash"], field=f"{semantic}.shared_sample_mapping_hash")
        _sha(result["shared_gradient_pool_hash"], field=f"{semantic}.shared_gradient_pool_hash")
        ref = result["shared_pool_ref"]
        if not isinstance(ref, str) or "\\" in ref or not re.fullmatch(r"shared-pools/[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\.json", ref):
            raise S29G27ABlocked(f"{semantic}.shared_pool_ref:INVALID")
        order = result["shared_method_order"]
        if not isinstance(order, list) or len(order) != len(S29_METHODS) or len(set(order)) != len(S29_METHODS) or set(order) != set(S29_METHODS):
            raise S29G27ABlocked(f"{semantic}.shared_method_order:METHOD_SET_INVALID")
        index = result["shared_method_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(S29_METHODS) or order[index] != result["method"]:
            raise S29G27ABlocked(f"{semantic}.shared_method_index:INVALID")
        for name in ("shared_sample_sequence_count", "shared_sample_token_count"):
            _positive(result[name], field=f"{semantic}.{name}")
        if result["sequence_count"] != result["shared_sample_sequence_count"] or result["token_count"] != result["shared_sample_token_count"]:
            raise S29G27ABlocked(f"{semantic}:SHARED_SAMPLE_COUNT_MISMATCH")
        expected_pair = shared_paired_run_identity(
            run_id=str(result["run_id"]),
            measurement_plan_hash=str(metadata.get("measurement_plan_hash", result.get("measurement_plan_hash", frozen.plan_hash))),
            matrix_hash=frozen.matrix_hash,
            raw_manifest_hash=frozen.raw_manifest_hash,
            source_raw_run_id=frozen.raw_run_id,
            anchor_id=str(result["anchor_id"]),
            repetition=int(result["repetition"]),
            gpu_uuid=str(result["gpu_uuid"]),
            device_count=int(result["device_count"]),
            method_order=order,
        )
        if result["paired_run_id"] != expected_pair["paired_run_id"] or result["paired_run_identity_hash"] != expected_pair["paired_run_identity_hash"]:
            raise S29G27ABlocked(f"{semantic}:PAIRED_RUN_IDENTITY_MISMATCH")
        declared_plan_hash = metadata.get("measurement_plan_hash")
        if declared_plan_hash is not None and result.get("measurement_plan_hash") != declared_plan_hash:
            raise S29G27ABlocked(f"{semantic}:MEASUREMENT_PLAN_HASH_MISMATCH")
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
        if normalized["run_id"] != self.run_id:
            raise S29G27ABlocked("RUN_ID_MISMATCH")
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


def _shared_pair_checks(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Verify that scientific rows came from one real pooled worker each.

    The reducer cannot inspect a worker's private gradient tensors, so the
    worker must publish a hash-bound pool identity on every method row.  This
    check requires the complete raw/double/U set for each pair, a common pool
    identity, and a metadata manifest that is copied into the G2.7a report.
    """

    shared = [row for row in rows if row["semantic"] == S29_COST_SEMANTICS[0]]
    if not shared:
        return ["SHARED_COST_OBSERVATIONS_MISSING"], []
    blockers: list[str] = []
    if metadata.get("shared_paired_runner_schema") != S29_SHARED_RUN_SCHEMA:
        blockers.append("SHARED_PAIRED_RUN_SCHEMA_REQUIRED")
    if metadata.get("shared_pool_schema") != S29_SHARED_POOL_SCHEMA:
        blockers.append("SHARED_POOL_SCHEMA_REQUIRED")
    entries = metadata.get("shared_paired_runs")
    if not isinstance(entries, list) or any(not isinstance(item, Mapping) for item in entries):
        blockers.append("SHARED_PAIRED_RUN_MANIFEST_REQUIRED")
        entries = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in shared:
        grouped.setdefault(str(row["paired_run_id"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for paired_run_id, group in sorted(grouped.items()):
        methods = {str(row["method"]) for row in group}
        if methods != set(S29_METHODS) or len(group) != len(S29_METHODS):
            blockers.append(f"SHARED_PAIRED_METHOD_SET_INVALID:{paired_run_id}")
            continue
        if any(row.get("cost_io_quiescent") is not True for row in group):
            blockers.append(f"SHARED_PAIRED_IO_NOT_QUIESCENT:{paired_run_id}")
        identity_fields = (
            "paired_run_identity_hash",
            "measurement_plan_hash",
            "anchor_id",
            "repetition",
            "gpu_uuid",
            "device_count",
            "shared_pool_id",
            "shared_pool_artifact_hash",
            "shared_pool_ref",
            "shared_sample_mapping_hash",
            "shared_gradient_pool_hash",
            "shared_method_order",
            "shared_sample_sequence_count",
            "shared_sample_token_count",
        )
        first = group[0]
        for field in identity_fields:
            if any(row.get(field) != first.get(field) for row in group[1:]):
                blockers.append(f"SHARED_PAIRED_IDENTITY_DRIFT:{paired_run_id}:{field}")
        indexes = [row.get("shared_method_index") for row in group]
        if sorted(indexes) != list(range(len(S29_METHODS))):
            blockers.append(f"SHARED_PAIRED_METHOD_INDEX_INVALID:{paired_run_id}")
        order = first.get("shared_method_order")
        if not isinstance(order, list) or any(row.get("method") != order[int(row["shared_method_index"])] for row in group if isinstance(row.get("shared_method_index"), int)):
            blockers.append(f"SHARED_PAIRED_METHOD_ORDER_INVALID:{paired_run_id}")
        try:
            expected = shared_paired_run_identity(
                run_id=str(first["run_id"]),
                measurement_plan_hash=str(first["measurement_plan_hash"]),
                matrix_hash=str(first["matrix_hash"]),
                raw_manifest_hash=str(first["raw_manifest_hash"]),
                source_raw_run_id=str(first["source_raw_run_id"]),
                anchor_id=str(first["anchor_id"]),
                repetition=int(first["repetition"]),
                gpu_uuid=str(first["gpu_uuid"]),
                device_count=int(first["device_count"]),
                method_order=order if isinstance(order, list) else (),
            )
        except Exception:
            blockers.append(f"SHARED_PAIRED_IDENTITY_INVALID:{paired_run_id}")
        else:
            if paired_run_id != expected["paired_run_id"] or first.get("paired_run_identity_hash") != expected["paired_run_identity_hash"]:
                blockers.append(f"SHARED_PAIRED_IDENTITY_MISMATCH:{paired_run_id}")
        summary.append(
            {
                "paired_run_id": paired_run_id,
                "paired_run_identity_hash": first.get("paired_run_identity_hash"),
                "measurement_plan_hash": first.get("measurement_plan_hash"),
                "anchor_id": first.get("anchor_id"),
                "repetition": first.get("repetition"),
                "gpu_uuid": first.get("gpu_uuid"),
                "device_count": first.get("device_count"),
                "shared_pool_id": first.get("shared_pool_id"),
                "shared_pool_artifact_hash": first.get("shared_pool_artifact_hash"),
                "shared_pool_ref": first.get("shared_pool_ref"),
                "shared_sample_mapping_hash": first.get("shared_sample_mapping_hash"),
                "shared_gradient_pool_hash": first.get("shared_gradient_pool_hash"),
                "shared_method_order": list(order) if isinstance(order, list) else order,
                "shared_sample_sequence_count": first.get("shared_sample_sequence_count"),
                "shared_sample_token_count": first.get("shared_sample_token_count"),
            }
        )
    declared = {str(item.get("paired_run_id")): dict(item) for item in entries if isinstance(item, Mapping)}
    if len(declared) != len(entries):
        blockers.append("SHARED_PAIRED_RUN_MANIFEST_DUPLICATE")
    if set(declared) != set(grouped):
        blockers.append("SHARED_PAIRED_RUN_MANIFEST_SET_INVALID")
    manifest_fields = {
        "paired_run_id",
        "paired_run_identity_hash",
        "measurement_plan_hash",
        "anchor_id",
        "repetition",
        "gpu_uuid",
        "device_count",
        "shared_pool_id",
        "shared_pool_artifact_hash",
        "shared_pool_ref",
        "shared_sample_mapping_hash",
        "shared_gradient_pool_hash",
        "shared_method_order",
        "shared_sample_sequence_count",
        "shared_sample_token_count",
    }
    for item in summary:
        declared_item = declared.get(str(item["paired_run_id"]))
        if declared_item is None:
            continue
        if set(declared_item) != manifest_fields or any(declared_item[field] != item[field] for field in manifest_fields):
            blockers.append(f"SHARED_PAIRED_RUN_MANIFEST_MISMATCH:{item['paired_run_id']}")
    return blockers, summary


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
    if snapshot.get("ecc_errors") != 0:
        blockers.append("GPU_ERROR_COUNTER_NONZERO")
    if "xid_errors" in snapshot:
        if snapshot.get("xid_errors") != 0:
            blockers.append("GPU_ERROR_COUNTER_NONZERO")
    else:
        health_states = snapshot.get("health_states")
        if not isinstance(health_states, list) or not health_states or any(str(item).strip().upper() != "HEALTHY" for item in health_states):
            blockers.append("GPU_HEALTH_STATE_NOT_CLEAN")
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


def _anchor_reasons(
    anchor: Any,
    *,
    frozen: S29FrozenInputs,
    expected_devices: int,
    reference_anchor: Mapping[str, Any] | None = None,
) -> list[str]:
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
    numeric_artifact = anchor.get("stage1_numeric_artifact")
    try:
        expected_role = "single_gpu_reference" if expected_devices == 1 else "four_gpu_candidate"
        validated_artifact = _validate_stage1_numeric_artifact(numeric_artifact, role=expected_role)
        if anchor.get("stage1_numeric_artifact_hash") != validated_artifact["artifact_hash"]:
            reasons.append(prefix + "_STAGE1_NUMERIC_ARTIFACT_HASH_MISMATCH")
        stats_binding = validated_artifact["statistical_binding"]
        for name in (
            "global_s1_hash",
            "global_s2_hash",
            "estimate_hash",
            "global_weight",
            "global_statistical_unit_count",
        ):
            if name not in anchor:
                reasons.append(prefix + "_STAGE1_NUMERIC_STATISTICAL_BINDING_MISSING:" + name)
            elif anchor[name] != stats_binding[name]:
                reasons.append(prefix + "_STAGE1_NUMERIC_STATISTICAL_BINDING_MISMATCH:" + name)
        expected_suffix = "anchors/single-gpu-anchor.json" if expected_devices == 1 else "anchors/four-gpu-stage1-numeric.json"
        try:
            _valid_stage1_numeric_ref(anchor.get("stage1_numeric_artifact_ref"), suffix=expected_suffix, field=prefix + ".stage1_numeric_artifact_ref")
        except S29G27ABlocked:
            reasons.append(prefix + "_STAGE1_NUMERIC_ARTIFACT_REF_INVALID")
    except S29G27ABlocked as error:
        reasons.append(prefix + "_STAGE1_NUMERIC_ARTIFACT_INVALID:" + str(error))
        validated_artifact = None
    if expected_devices == 1:
        return reasons
    if anchor.get("batch_size") != frozen.batch_size or anchor.get("microbatch_count") != frozen.microbatch_count:
        reasons.append(prefix + "_B_M_DRIFT")
    if expected_devices == 4:
        required_system_fields = (
            "barrier_seconds",
            "barrier_count",
            "statistical_unit",
            "statistical_unit_count",
            "global_statistical_unit_count",
            "global_weight",
            "system_anchor_mode",
            "communication_mode",
            "rank_partition_mode",
            "all_reduce_s1_seconds",
            "all_reduce_s2_seconds",
            "all_reduce_weight_seconds",
            "all_reduce_count_seconds",
            "all_reduce_s1_bytes",
            "all_reduce_s2_bytes",
            "all_reduce_weight_bytes",
            "all_reduce_count_bytes",
            "all_reduce_s1_count",
            "all_reduce_s2_count",
            "all_reduce_weight_count",
            "all_reduce_count",
            "all_reduce_identity_hash",
            "local_sample_mapping_hashes",
            "local_gradient_pool_hashes",
            "fixed_checkpoint_id",
            "checkpoint_hash",
            "mapping_hash",
            "sample_mapping_hash",
            "state_digest",
            "state_digest_after",
            "four_process_identity_hash",
            "global_s1_hash",
            "global_s2_hash",
            "estimate_hash",
            "gradient_pool_hash",
            "per_device_measurements",
            "four_card_throughput_sequences_per_second",
            "single_card_throughput_sequences_per_second",
            "strong_scaling_speedup",
            "strong_scaling_efficiency",
            "strong_scaling_reference_wall_seconds",
            "strong_scaling_reference_sequence_count",
            "strong_scaling_reference_token_count",
            "strong_scaling_reference_backward_count",
            "single_anchor_identity_hash",
            "stage1_numeric_artifact",
            "stage1_numeric_artifact_hash",
            "stage1_numeric_artifact_ref",
            "stage1_numeric_comparison",
            "stage1_numeric_comparison_hash",
            "stage1_numeric_reference_ref",
            "stage1_numeric_reference_hash",
        )
        missing_system_fields = [name for name in required_system_fields if name not in anchor]
        reasons.extend(prefix + "_SYSTEM_FIELDS_MISSING:" + name for name in missing_system_fields)
        if missing_system_fields:
            return reasons
        try:
            _valid_stage1_numeric_ref(anchor.get("stage1_numeric_reference_ref"), suffix="anchors/single-gpu-anchor.json", field=prefix + ".stage1_numeric_reference_ref")
        except S29G27ABlocked:
            reasons.append(prefix + "_STAGE1_NUMERIC_REFERENCE_REF_INVALID")
        if anchor.get("stage1_numeric_reference_hash") != (
            reference_anchor.get("stage1_numeric_artifact_hash") if isinstance(reference_anchor, Mapping) else None
        ):
            reasons.append(prefix + "_STAGE1_NUMERIC_REFERENCE_HASH_MISMATCH")
        if not isinstance(reference_anchor, Mapping):
            reasons.append(prefix + "_STAGE1_NUMERIC_REFERENCE_REQUIRED")
        else:
            try:
                reference_artifact = _validate_stage1_numeric_artifact(
                    reference_anchor.get("stage1_numeric_artifact"), role="single_gpu_reference"
                )
                if reference_anchor.get("stage1_numeric_artifact_hash") != reference_artifact["artifact_hash"]:
                    reasons.append(prefix + "_STAGE1_REFERENCE_ARTIFACT_HASH_MISMATCH")
                if anchor.get("stage1_numeric_reference_ref") != reference_anchor.get("stage1_numeric_artifact_ref"):
                    reasons.append(prefix + "_STAGE1_NUMERIC_REFERENCE_REF_MISMATCH")
                if validated_artifact is not None:
                    _validate_stage1_numeric_comparison(
                        anchor.get("stage1_numeric_comparison"),
                        candidate=validated_artifact,
                        reference=reference_artifact,
                        reference_ref=str(reference_anchor.get("stage1_numeric_artifact_ref", S29_STAGE1_NUMERIC_REFERENCE_REF)),
                    )
                    if anchor.get("stage1_numeric_comparison_hash") != anchor["stage1_numeric_comparison"].get("artifact_hash"):
                        reasons.append(prefix + "_STAGE1_NUMERIC_COMPARISON_HASH_MISMATCH")
                else:
                    reasons.append(prefix + "_STAGE1_NUMERIC_COMPARISON_UNAVAILABLE")
            except S29G27ABlocked as error:
                reasons.append(prefix + "_STAGE1_NUMERIC_COMPARISON_INVALID:" + str(error))
        if anchor.get("system_anchor_mode") != "synchronized_fixed_state_four_process_nccl":
            reasons.append(prefix + "_NCCL_SYSTEM_MODE_REQUIRED")
        if anchor.get("communication_mode") != "nccl_all_reduce_s1_s2":
            reasons.append(prefix + "_NCCL_COLLECTIVE_REQUIRED")
        if anchor.get("rank_partition_mode") != "disjoint_complete_microbatch_groups":
            reasons.append(prefix + "_DISJOINT_PARTITION_REQUIRED")
        if anchor.get("communication_bytes", 0) <= 0:
            reasons.append(prefix + "_COMMUNICATION_BYTES_REQUIRED")
        if anchor.get("sequence_count") != frozen.batch_size or anchor.get("backward_count") != frozen.microbatch_count:
            reasons.append(prefix + "_GLOBAL_STATISTICS_COUNT_MISMATCH")
        if anchor.get("statistical_unit") != "microbatch" or anchor.get("statistical_unit_count") != frozen.microbatch_count or anchor.get("global_statistical_unit_count") != frozen.microbatch_count:
            reasons.append(prefix + "_STATISTICAL_UNIT_COUNT_INVALID")
        if anchor.get("barrier_count") != 2:
            reasons.append(prefix + "_BARRIER_COUNT_INVALID")
        if anchor.get("all_reduce_s1_count") != 4 or anchor.get("all_reduce_s2_count") != 4 or anchor.get("all_reduce_weight_count") != 4 or anchor.get("all_reduce_count") != 16:
            reasons.append(prefix + "_ALL_REDUCE_COUNT_INVALID")
        for name in ("barrier_seconds", "global_weight", "all_reduce_s1_seconds", "all_reduce_s2_seconds", "all_reduce_weight_seconds", "all_reduce_count_seconds", "four_card_throughput_sequences_per_second", "single_card_throughput_sequences_per_second", "strong_scaling_speedup", "strong_scaling_efficiency", "strong_scaling_reference_wall_seconds"):
            value = anchor.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                reasons.append(prefix + "_SYSTEM_FLOAT_INVALID:" + name)
        for name in ("all_reduce_s1_bytes", "all_reduce_s2_bytes", "all_reduce_weight_bytes", "all_reduce_count_bytes", "all_reduce_weight_count", "statistical_unit_count", "global_statistical_unit_count", "strong_scaling_reference_sequence_count", "strong_scaling_reference_token_count", "strong_scaling_reference_backward_count"):
            value = anchor.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                reasons.append(prefix + "_SYSTEM_COUNT_INVALID:" + name)
        if anchor.get("strong_scaling_reference_sequence_count") != frozen.batch_size or anchor.get("strong_scaling_reference_backward_count") != frozen.microbatch_count:
            reasons.append(prefix + "_SINGLE_REFERENCE_COUNT_MISMATCH")
        local_hashes = anchor.get("local_sample_mapping_hashes")
        if not isinstance(local_hashes, list) or len(local_hashes) != 4 or len(set(local_hashes)) != 4 or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in local_hashes):
            reasons.append(prefix + "_PARTITION_HASHES_INVALID")
        local_gradient_hashes = anchor.get("local_gradient_pool_hashes")
        if not isinstance(local_gradient_hashes, list) or len(local_gradient_hashes) != 4 or len(set(local_gradient_hashes)) != 4 or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None for value in local_gradient_hashes):
            reasons.append(prefix + "_LOCAL_GRADIENT_HASHES_INVALID")
        if not isinstance(anchor.get("fixed_checkpoint_id"), str) or not anchor["fixed_checkpoint_id"]:
            reasons.append(prefix + "_CHECKPOINT_ID_INVALID")
        for name in ("checkpoint_hash", "mapping_hash", "sample_mapping_hash", "state_digest", "state_digest_after", "four_process_identity_hash", "global_s1_hash", "global_s2_hash", "estimate_hash", "gradient_pool_hash", "all_reduce_identity_hash", "single_anchor_identity_hash"):
            if not isinstance(anchor.get(name), str) or _SHA256.fullmatch(anchor[name]) is None:
                reasons.append(prefix + "_HASH_INVALID:" + name)
        devices = anchor.get("per_device_measurements")
        if not isinstance(devices, list) or len(devices) != 4 or any(not isinstance(item, Mapping) for item in devices):
            reasons.append(prefix + "_PER_DEVICE_ROWS_INVALID")
        else:
            expected_devices_set = {(index, uuid) for index, uuid in enumerate(APPROVED_GPU_UUIDS)}
            observed_devices = {(item.get("rank"), item.get("gpu_uuid")) for item in devices}
            if observed_devices != expected_devices_set:
                reasons.append(prefix + "_PER_DEVICE_UUID_BINDING_INVALID")
            if observed_devices == expected_devices_set:
                ranked_devices = sorted(devices, key=lambda item: item["rank"])
                if [item.get("local_sample_mapping_hash") for item in ranked_devices] != local_hashes:
                    reasons.append(prefix + "_PER_DEVICE_PARTITION_HASH_MISMATCH")
                if [item.get("local_gradient_pool_hash") for item in ranked_devices] != local_gradient_hashes:
                    reasons.append(prefix + "_PER_DEVICE_GRADIENT_HASH_MISMATCH")
                if any(not isinstance(item.get("all_reduce_identity_hash"), str) or _SHA256.fullmatch(item["all_reduce_identity_hash"]) is None for item in ranked_devices):
                    reasons.append(prefix + "_PER_DEVICE_ALL_REDUCE_HASH_INVALID")
                for item in ranked_devices:
                    if any(item.get(name) != anchor.get(name) for name in ("fixed_checkpoint_id", "checkpoint_hash", "mapping_hash", "global_s1_hash", "global_s2_hash", "estimate_hash", "state_digest", "state_digest_after")):
                        reasons.append(prefix + "_PER_DEVICE_NUMERIC_IDENTITY_MISMATCH")
                        break
                expected_all_reduce_identity_hash = canonical_json_hash(
                    [
                        {"rank": int(item["rank"]), "gpu_uuid": str(item["gpu_uuid"]), "identity_hash": str(item["all_reduce_identity_hash"])}
                        for item in ranked_devices
                    ]
                )
                if anchor.get("all_reduce_identity_hash") != expected_all_reduce_identity_hash:
                    reasons.append(prefix + "_ALL_REDUCE_IDENTITY_MISMATCH")
                expected_four_process_identity_hash = canonical_json_hash(
                    {
                        "gpu_uuids": list(APPROVED_GPU_UUIDS),
                        "sample_mapping_hash": anchor.get("sample_mapping_hash"),
                        "gradient_pool_hash": anchor.get("gradient_pool_hash"),
                        "estimate_hash": anchor.get("estimate_hash"),
                        "state_digest": anchor.get("state_digest"),
                        "state_digest_after": anchor.get("state_digest_after"),
                        "checkpoint_hash": anchor.get("checkpoint_hash"),
                        "mapping_hash": anchor.get("mapping_hash"),
                    }
                )
                if anchor.get("four_process_identity_hash") != expected_four_process_identity_hash:
                    reasons.append(prefix + "_PROCESS_IDENTITY_MISMATCH")
            local_sequence_total = sum(int(item.get("sequence_count", -1)) for item in devices if isinstance(item.get("sequence_count"), int) and not isinstance(item.get("sequence_count"), bool))
            local_token_total = sum(int(item.get("token_count", -1)) for item in devices if isinstance(item.get("token_count"), int) and not isinstance(item.get("token_count"), bool))
            local_backward_total = sum(int(item.get("backward_count", -1)) for item in devices if isinstance(item.get("backward_count"), int) and not isinstance(item.get("backward_count"), bool))
            if local_sequence_total != anchor.get("sequence_count") or local_token_total != anchor.get("token_count") or local_backward_total != anchor.get("backward_count"):
                reasons.append(prefix + "_PER_DEVICE_GLOBAL_COUNT_MISMATCH")
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


def _crosscheck_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    """Return a deterministic key for the rows sealed into a cross-check."""

    repetition = row.get("repetition", 0)
    return (
        str(row.get("semantic", "")),
        str(row.get("method", "")),
        str(row.get("anchor_id", "")),
        int(repetition) if isinstance(repetition, int) and not isinstance(repetition, bool) else 0,
        str(row.get("run_id", "")),
    )


def _crosscheck_row_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the complete deterministic set of measured rows."""

    ordered = [dict(row) for row in sorted(rows, key=_crosscheck_row_sort_key)]
    return canonical_json_hash(ordered)


def build_s209_shared_attribution_crosscheck(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a content-addressed cross-check from sealed measured rows.

    The reducer must compare the shared and isolated medians from the exact
    rows it just sealed.  This function intentionally has no fallback values:
    missing method rows or non-positive timings are a hard error.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise S29G27ABlocked("SHARED_ATTRIBUTION_CROSSCHECK_ROWS_REQUIRED")
    materialized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(materialized) != len(rows):
        raise S29G27ABlocked("SHARED_ATTRIBUTION_CROSSCHECK_ROW_INVALID")
    values: dict[str, dict[str, float]] = {}
    for method in S29_METHODS:
        shared = [
            float(row["wall_seconds"])
            for row in materialized
            if row.get("semantic") == "scientific_equal_sample_cost"
            and row.get("method") == method
        ]
        isolated = [
            float(row["wall_seconds"])
            for row in materialized
            if row.get("semantic") == "isolated_estimator_cost"
            and row.get("method") == method
        ]
        if not shared or not isolated or any(value <= 0 or not math.isfinite(value) for value in shared + isolated):
            raise S29G27ABlocked(f"SHARED_ATTRIBUTION_CROSSCHECK_ROWS_MISSING:{method}")
        shared_median = float(statistics.median(shared))
        isolated_median = float(statistics.median(isolated))
        relative_difference = abs(shared_median - isolated_median) / isolated_median
        values[method] = {
            "shared_wall_seconds": shared_median,
            "isolated_wall_seconds": isolated_median,
            "relative_difference": float(relative_difference),
        }
    body: dict[str, Any] = {
        "schema_version": S29_CROSSCHECK_SCHEMA,
        "source": "sealed_measured_rows",
        "row_identity_hash": _crosscheck_row_identity_hash(materialized),
        "rows": values,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _crosscheck_rows_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    """Read the new envelope or the legacy method-key mapping."""

    if not isinstance(value, Mapping):
        raise S29G27ABlocked(f"{field}:OBJECT_REQUIRED")
    if value.get("schema_version") == S29_CROSSCHECK_SCHEMA:
        declared = value.get("artifact_hash")
        if not isinstance(declared, str) or canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != declared:
            raise S29G27ABlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
        if value.get("source") != "sealed_measured_rows":
            raise S29G27ABlocked(f"{field}:SOURCE_INVALID")
        row_identity = value.get("row_identity_hash")
        if not isinstance(row_identity, str) or _SHA256.fullmatch(row_identity) is None:
            raise S29G27ABlocked(f"{field}:ROW_IDENTITY_HASH_REQUIRED")
        rows = value.get("rows")
    else:
        # Compatibility for callers that supplied the original method-key
        # mapping.  It remains fail-closed below because every value must
        # exactly equal the reducer-generated sealed-row result.
        rows = value
    if not isinstance(rows, Mapping):
        raise S29G27ABlocked(f"{field}:ROWS_REQUIRED")
    return rows


def _validate_preprovided_crosscheck(
    value: Any,
    *,
    generated: Mapping[str, Any],
) -> None:
    """Require a presealed attestation to match generated medians exactly."""

    supplied = _crosscheck_rows_mapping(value, field="shared_attribution_crosscheck")
    expected = generated.get("rows")
    if not isinstance(expected, Mapping):
        raise S29G27ABlocked("SHARED_ATTRIBUTION_CROSSCHECK_GENERATED_ROWS_REQUIRED")
    for method in S29_METHODS:
        if supplied.get(method) != expected.get(method):
            raise S29G27ABlocked(f"SHARED_ATTRIBUTION_CROSSCHECK_PRESEALED_MISMATCH:{method}")
    if value.get("schema_version") == S29_CROSSCHECK_SCHEMA and value.get("row_identity_hash") != generated.get("row_identity_hash"):
        raise S29G27ABlocked("SHARED_ATTRIBUTION_CROSSCHECK_ROW_IDENTITY_MISMATCH")


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
    execution_identity: Mapping[str, Any] | None = None,
    data_root: str | Path | None = None,
    single_gpu_anchor_ref: str | None = None,
    four_gpu_anchor_ref: str | None = None,
    four_gpu_numeric_ref: str | None = None,
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
    if validated_measurement_plan is not None and validated_measurement_plan["run_id"] != run_id:
        raise S29G27ABlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    validated_execution_identity = (
        _validate_execution_identity(execution_identity)
        if execution_identity is not None
        else None
    )
    if validated_execution_identity is not None:
        if data_root is None or single_gpu_anchor_ref is None or four_gpu_anchor_ref is None or four_gpu_numeric_ref is None:
            raise S29G27ABlocked("FORMAL_ANCHOR_PERSISTED_REFERENCES_REQUIRED")
        formal_root = Path(data_root).resolve()
        persisted_single, persisted_single_ref = _formal_file(
            formal_root,
            single_gpu_anchor_ref,
            filename="single-gpu-anchor.json",
            field="single_gpu_anchor_ref",
        )
        persisted_four, persisted_four_ref = _formal_file(
            formal_root,
            four_gpu_anchor_ref,
            filename="four-gpu-anchor.json",
            field="four_gpu_anchor_ref",
        )
        persisted_numeric, persisted_numeric_ref = _formal_file(
            formal_root,
            four_gpu_numeric_ref,
            filename="four-gpu-stage1-numeric.json",
            field="four_gpu_numeric_ref",
        )
        if isinstance(single_gpu_anchor, Mapping) and dict(single_gpu_anchor) != persisted_single:
            raise S29G27ABlocked("FORMAL_SINGLE_ANCHOR_CALLER_MAPPING_DRIFT")
        if isinstance(four_gpu_anchor, Mapping) and dict(four_gpu_anchor) != persisted_four:
            raise S29G27ABlocked("FORMAL_FOUR_ANCHOR_CALLER_MAPPING_DRIFT")
        single_gpu_anchor = persisted_single
        four_gpu_anchor = persisted_four
        try:
            candidate = _validate_stage1_numeric_artifact(persisted_numeric, role="four_gpu_candidate")
        except Exception as error:
            raise S29G27ABlocked("FORMAL_FOUR_NUMERIC_SIDECAR_INVALID") from error
        if persisted_four.get("stage1_numeric_artifact") != candidate or persisted_four.get("stage1_numeric_artifact_ref") != persisted_numeric_ref:
            raise S29G27ABlocked("FORMAL_FOUR_NUMERIC_SIDECAR_BINDING_DRIFT")
        if persisted_single.get("stage1_numeric_artifact_ref") != persisted_single_ref:
            raise S29G27ABlocked("FORMAL_SINGLE_NUMERIC_REFERENCE_REF_DRIFT")
    if not isinstance(cost_observations, Mapping) or set(cost_observations) != set(S29_COST_SEMANTICS):
        raise S29G27ABlocked("COST_SEMANTICS_EXACTLY_THREE_REQUIRED")
    reducer = StrictS29Reducer(frozen, run_id=run_id)
    metadata: dict[str, Mapping[str, Any]] = {}
    for semantic in S29_COST_SEMANTICS:
        meta, rows = _semantic_rows(cost_observations[semantic], semantic=semantic)
        if semantic == S29_COST_SEMANTICS[0] and validated_measurement_plan is not None:
            # The worker's pair identity is rooted in the immutable plan hash;
            # make the reducer compare against the validated producer rather
            # than trusting a row/metadata value supplied by the caller.
            meta = dict(meta)
            meta["measurement_plan_hash"] = validated_measurement_plan["artifact_hash"]
        metadata[semantic] = meta
        for row in rows:
            reducer.add(row, semantic=semantic, metadata=meta)
    reduced = reducer.seal()
    shared_blockers, shared_pairs = _shared_pair_checks(
        reduced,
        metadata["scientific_equal_sample_cost"],
    )
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
    four_reasons = _anchor_reasons(
        four_gpu_anchor,
        frozen=frozen,
        expected_devices=4,
        reference_anchor=single_gpu_anchor,
    )
    if four_reasons:
        provisional.append("FOUR_CARD_EVIDENCE_MISSING_OR_INVALID")
        blockers.extend(four_reasons)
    consistency = _consistency(reduced, frozen)
    if not consistency["all_pass"]:
        blockers.append("COST_SYSTEM_CONSISTENCY_FAILED")
    online_blockers, online_aggregates, online_ratios = _online_checks(reduced, metadata["online_training_incremental_cost"])
    blockers.extend(online_blockers)
    blockers.extend(shared_blockers)
    generated_crosscheck: dict[str, Any] | None = None
    try:
        generated_crosscheck = build_s209_shared_attribution_crosscheck(reduced)
    except S29G27ABlocked as error:
        blockers.append(f"SHARED_ATTRIBUTION_CROSSCHECK_GENERATION_BLOCKED:{error}")
    if generated_crosscheck is not None:
        if shared_attribution_cross_check is not None:
            try:
                _validate_preprovided_crosscheck(
                    shared_attribution_cross_check,
                    generated=generated_crosscheck,
                )
            except S29G27ABlocked as error:
                blockers.append(str(error))
        # Always reduce the values sealed in this invocation. A caller may
        # preseal an attestation, but it can never replace measured rows.
        blockers.extend(_crosscheck(reduced, generated_crosscheck["rows"]))
    else:
        # A supplied attestation is never a fallback for malformed or
        # incomplete sealed rows.  Without a reducer-generated identity there
        # is no content-addressed cross-check that can be safely compared.
        blockers.append("SHARED_ATTRIBUTION_CROSSCHECK_GENERATION_REQUIRED")
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
        "shared_paired_run_count": len(shared_pairs),
        "shared_pool_artifact_hashes": sorted({str(item["shared_pool_artifact_hash"]) for item in shared_pairs}),
        "shared_attribution_crosscheck_hash": None if generated_crosscheck is None else generated_crosscheck["artifact_hash"],
    }
    if validated_execution_identity is not None:
        measured.update(
            {
                "repository_head": validated_execution_identity["repository_head"],
                "launcher_source_sha256": validated_execution_identity["launcher_source_sha256"],
                "profiler_command_hash": validated_execution_identity["profiler_command_hash"],
                "execution_identity_hash": validated_execution_identity["artifact_hash"],
            }
        )
    gate = GateRecord(
        gate_id="stage2.G2.7a",
        stage=2,
        status=gate_status,
        checked_at=now,
        measured=measured,
        threshold={"online_training_incremental_cost_ratio": S29_DECISION_RATIO, "shared_crosscheck_relative_difference": S29_CROSSCHECK_TOLERANCE, "four_gpu_required": True, "cost_io_quiescent": True, "shared_paired_runner_schema": S29_SHARED_RUN_SCHEMA, "shared_gradient_pool_schema": S29_SHARED_POOL_SCHEMA},
        evidence_refs=(matrix_ref, raw_manifest_ref, f"s209/{run_id}/cost-system-validation.json", "shared-attribution-crosscheck.json"),
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
        if semantic == "scientific_equal_sample_cost":
            semantic_summary[semantic]["shared_paired_runner_schema"] = S29_SHARED_RUN_SCHEMA
            semantic_summary[semantic]["shared_pool_schema"] = S29_SHARED_POOL_SCHEMA
            semantic_summary[semantic]["shared_paired_runs"] = shared_pairs
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
        "shared_attribution_cross_check": None if generated_crosscheck is None else generated_crosscheck["rows"],
        "shared_attribution_crosscheck": generated_crosscheck,
        "shared_paired_runs": shared_pairs,
        "online_training_incremental_cost": {"aggregates": online_aggregates, "ratios": online_ratios, "decision_source": "online_training_incremental_cost"},
        "pareto": pareto,
        "capacity": capacity,
        "measurement_plan": validated_measurement_plan,
        "gate": gate.to_dict(),
        "reasons": list(reasons),
    }
    if validated_execution_identity is not None:
        report["execution_identity"] = validated_execution_identity
    report["artifact_hash"] = canonical_json_hash(report)
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        write_canonical_json(root / "cost-system-validation.json", report)
        write_canonical_json(root / "g2.7a-gate.json", gate.to_dict())
        if generated_crosscheck is not None:
            write_canonical_json(root / "shared-attribution-crosscheck.json", generated_crosscheck)
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
    "S29_CROSSCHECK_SCHEMA",
    "S29_SHARED_POOL_SCHEMA",
    "S29_SHARED_RUN_SCHEMA",
    "S29_SHARED_ROW_FIELDS",
    "S29_STAGE1_NUMERIC_ARTIFACT_SCHEMA",
    "S29_STAGE1_NUMERIC_COMPARISON_SCHEMA",
    "S29_STAGE1_NUMERIC_REFERENCE_REF",
    "S29_STAGE1_NUMERIC_CANDIDATE_REF",
    "S29_STAGE1_TOLERANCE_PROFILE",
    "S29FrozenInputs",
    "S29G27ABlocked",
    "StrictS29Reducer",
    "build_s209_shared_attribution_crosscheck",
    "bind_s209_inputs",
    "orchestrate_s209_g27a",
    "prepare_s209_measurement_plan",
    "run_s209_g27a",
    "shared_paired_run_identity",
    "_stage1_numeric_artifact",
    "_stage1_numeric_comparison",
    "_validate_stage1_numeric_artifact",
    "_validate_stage1_numeric_comparison",
    "validate_g27a",
]
