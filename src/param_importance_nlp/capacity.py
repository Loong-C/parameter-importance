from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .atomic import stable_json_hash


GIB = 1024**3

CAPACITY_PROTOCOL_VERSION = "stage0.capacity-measurement.v1"
PARAMETER_STATE_ENVELOPE_VERSION = "stage0.parameter-state-capacity-envelope.v1"
WORKLOAD_ENVELOPE_VERSION = "stage0.compute-communication-work-envelope.v1"

_DTYPE_BYTES: Mapping[str, int] = MappingProxyType(
    {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "int16": 2,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "float32": 4,
        "int64": 8,
        "float64": 8,
    }
)


def dtype_nbytes(dtype: str) -> int:
    """Return the frozen storage width for one supported scalar dtype."""

    try:
        return _DTYPE_BYTES[dtype.removeprefix("torch.")]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"CAPACITY_DTYPE_UNSUPPORTED:{dtype!r}") from error


@dataclass(frozen=True, slots=True)
class ParameterTensorShape:
    """One real model tensor identity used by every capacity estimate."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool = True

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("CAPACITY_TENSOR_NAME_INVALID")
        normalized = tuple(self.shape)
        if not normalized or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in normalized
        ):
            raise ValueError(f"CAPACITY_TENSOR_SHAPE_INVALID:{self.name}")
        dtype_nbytes(self.dtype)
        if type(self.requires_grad) is not bool:
            raise TypeError("CAPACITY_TENSOR_REQUIRES_GRAD_NOT_BOOLEAN")
        object.__setattr__(self, "shape", normalized)
        object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def storage_bytes(self) -> int:
        return self.element_count * dtype_nbytes(self.dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
            "element_count": self.element_count,
            "storage_bytes": self.storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class ParameterStateBuffer:
    """A full-parameter resident or transient state in the Stage 0 envelope."""

    buffer_id: str
    dtype: str
    location: str
    lifecycle: str
    simultaneously_resident: bool
    checkpointed: bool
    update_cadence: str
    purpose: str

    def __post_init__(self) -> None:
        if (
            not self.buffer_id
            or self.location not in {"gpu", "cpu"}
            or self.lifecycle not in {"resident", "transient_reduction", "transient_serialization"}
            or not self.update_cadence
            or not self.purpose
        ):
            raise ValueError(f"CAPACITY_BUFFER_CONTRACT_INVALID:{self.buffer_id}")
        dtype_nbytes(self.dtype)
        if type(self.simultaneously_resident) is not bool or type(self.checkpointed) is not bool:
            raise TypeError("CAPACITY_BUFFER_BOOLEAN_INVALID")
        object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))

    def bytes_for(self, elements: int) -> int:
        if isinstance(elements, bool) or not isinstance(elements, int) or elements <= 0:
            raise ValueError("CAPACITY_BUFFER_ELEMENTS_INVALID")
        return elements * dtype_nbytes(self.dtype)

    def to_dict(self, *, elements: int) -> dict[str, Any]:
        return {
            "buffer_id": self.buffer_id,
            "shape_binding": "each_trainable_parameter_exact_shape",
            "element_count": elements,
            "dtype": self.dtype,
            "bytes": self.bytes_for(elements),
            "location": self.location,
            "lifecycle": self.lifecycle,
            "simultaneously_resident": self.simultaneously_resident,
            "checkpointed": self.checkpointed,
            "update_cadence": self.update_cadence,
            "purpose": self.purpose,
        }


def default_parameter_state_buffers(*, parameter_dtype: str) -> tuple[ParameterStateBuffer, ...]:
    """Freeze the resource envelope without asserting estimator mathematics.

    The names describe future consumers only.  Stage 0 allocates and updates
    shape-equivalent tensors; it never computes raw/U/path-integral scores.
    """

    return (
        ParameterStateBuffer(
            "model_weights", parameter_dtype, "gpu", "resident", True, True,
            "optimizer_step", "active model parameter storage",
        ),
        ParameterStateBuffer(
            "optimizer_gradient", "float32", "gpu", "resident", True, False,
            "microbatch_then_optimizer_step", "streamed training gradient",
        ),
        ParameterStateBuffer(
            "optimizer_first_moment", "float32", "gpu", "resident", True, True,
            "optimizer_step", "Adam first-moment capacity",
        ),
        ParameterStateBuffer(
            "optimizer_second_moment", "float32", "gpu", "resident", True, True,
            "optimizer_step", "Adam second-moment capacity",
        ),
        ParameterStateBuffer(
            "raw_accumulator", "float32", "gpu", "resident", True, True,
            "optimizer_step", "raw-view capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "signed_accumulator", "float32", "gpu", "resident", True, True,
            "optimizer_step", "signed-view capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "positive_accumulator", "float32", "gpu", "resident", True, True,
            "optimizer_step", "positive-view capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "negative_accumulator", "float32", "gpu", "resident", True, True,
            "optimizer_step", "negative-view capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "u_first_moment", "float32", "gpu", "resident", True, True,
            "optimizer_step", "U-statistic first-moment capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "u_second_moment", "float32", "gpu", "resident", True, True,
            "optimizer_step", "U-statistic second-moment capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "path_integral_accumulator", "float32", "gpu", "resident", True, True,
            "optimizer_step", "path-integral capacity placeholder; no formula",
        ),
        ParameterStateBuffer(
            "distributed_reduction_scratch", "float32", "gpu", "transient_reduction", False, False,
            "checkpoint_boundary", "chunked collective staging upper bound",
        ),
        ParameterStateBuffer(
            "checkpoint_serialization_scratch", "float32", "cpu", "transient_serialization", False, False,
            "checkpoint_boundary", "one-full-buffer host serialization staging upper bound",
        ),
    )


def build_parameter_state_envelope(
    *,
    model_id: str,
    tensors: Sequence[ParameterTensorShape],
    config_hash: str,
    model_manifest_id: str,
    checkpoint_every_steps: int,
    parameter_dtype: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound exact-shape parameter-state capacity envelope."""

    ordered = tuple(tensors)
    if not model_id or not ordered or len({item.name for item in ordered}) != len(ordered):
        raise ValueError("CAPACITY_PARAMETER_SHAPE_SET_INVALID")
    if (
        not isinstance(checkpoint_every_steps, int)
        or isinstance(checkpoint_every_steps, bool)
        or checkpoint_every_steps <= 0
    ):
        raise ValueError("CAPACITY_CHECKPOINT_CADENCE_INVALID")
    trainable = tuple(item for item in ordered if item.requires_grad)
    if not trainable:
        raise ValueError("CAPACITY_NO_TRAINABLE_PARAMETERS")
    total_elements = sum(item.element_count for item in trainable)
    inferred_dtypes = {item.dtype for item in trainable}
    selected_parameter_dtype = parameter_dtype or (
        next(iter(inferred_dtypes)) if len(inferred_dtypes) == 1 else "float32"
    )
    buffers = default_parameter_state_buffers(parameter_dtype=selected_parameter_dtype)
    buffer_rows = [item.to_dict(elements=total_elements) for item in buffers]
    resident_gpu = sum(
        int(item["bytes"])
        for item in buffer_rows
        if item["location"] == "gpu" and item["simultaneously_resident"] is True
    )
    transient_gpu = max(
        (
            int(item["bytes"])
            for item in buffer_rows
            if item["location"] == "gpu" and item["simultaneously_resident"] is False
        ),
        default=0,
    )
    transient_cpu = max(
        (
            int(item["bytes"])
            for item in buffer_rows
            if item["location"] == "cpu" and item["simultaneously_resident"] is False
        ),
        default=0,
    )
    checkpoint_bytes = sum(
        int(item["bytes"]) for item in buffer_rows if item["checkpointed"] is True
    )
    value: dict[str, Any] = {
        "schema_version": PARAMETER_STATE_ENVELOPE_VERSION,
        "model_id": model_id,
        "model_manifest_id": model_manifest_id,
        "config_hash": config_hash,
        "tensor_shapes": [item.to_dict() for item in ordered],
        "tensor_shape_hash": stable_json_hash([item.to_dict() for item in ordered]),
        "trainable_tensor_count": len(trainable),
        "trainable_parameter_count": total_elements,
        "source_dtype_counts": {
            dtype: sum(item.element_count for item in trainable if item.dtype == dtype)
            for dtype in sorted(inferred_dtypes)
        },
        "buffers": buffer_rows,
        "resident_gpu_bytes": resident_gpu,
        "transient_gpu_bytes_max": transient_gpu,
        "peak_parameter_state_gpu_bytes": resident_gpu + transient_gpu,
        "transient_cpu_bytes_max": transient_cpu,
        "checkpoint_parameter_state_bytes": checkpoint_bytes,
        "checkpoint_every_steps": checkpoint_every_steps,
        "mathematics_implemented": False,
        "scope": "shape_dtype_lifecycle_capacity_only",
    }
    value["artifact_hash"] = stable_json_hash(value)
    return value


def build_compute_communication_envelope(
    *,
    model_id: str,
    parameter_count: int,
    world_size: int,
    microbatches_per_optimizer_step: int,
    checkpoint_every_steps: int,
    collective_chunk_bytes: int = 64 * 1024**2,
    compute_dtype: str = "float32",
    sequence_length: int = 1,
    microbatch_size: int = 1,
    candidate_stage: int = 0,
) -> dict[str, Any]:
    """Freeze the synthetic upper-bound work performed by the G8 worker."""

    values = (
        parameter_count,
        world_size,
        microbatches_per_optimizer_step,
        checkpoint_every_steps,
        collective_chunk_bytes,
        sequence_length,
        microbatch_size,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("CAPACITY_WORK_ENVELOPE_POSITIVE_INTEGER_REQUIRED")
    if isinstance(candidate_stage, bool) or not isinstance(candidate_stage, int) or not 0 <= candidate_stage <= 9:
        raise ValueError("CAPACITY_WORK_ENVELOPE_STAGE_INVALID")
    parameter_bytes = parameter_count * dtype_nbytes("float32")
    dtype_nbytes(compute_dtype)
    collective_chunks = (
        0 if world_size == 1 else math.ceil(parameter_bytes / collective_chunk_bytes)
    )
    value: dict[str, Any] = {
        "schema_version": WORKLOAD_ENVELOPE_VERSION,
        "model_id": model_id,
        "world_size": world_size,
        "candidate_stage": candidate_stage,
        "compute_dtype": compute_dtype.removeprefix("torch."),
        "sequence_length": sequence_length,
        "microbatch_size": microbatch_size,
        "per_optimizer_step": {
            "forward_calls": microbatches_per_optimizer_step,
            "backward_calls": microbatches_per_optimizer_step,
            "gradient_evaluations": microbatches_per_optimizer_step,
            "activation_recomputations": 0,
            "synthetic_full_parameter_updates": 7,
            "cpu_aggregation_bytes": 0,
            "serialization_bytes": 0,
            "distributed_gradient_sync": {
                "logical_collective_type": "all_reduce_sum" if world_size > 1 else "none",
                "logical_collective_count": 1 if world_size > 1 else 0,
                "payload_bytes_per_rank": parameter_bytes if world_size > 1 else 0,
                "runtime_bucketization": "framework_bound_and_reported",
            },
        },
        "checkpoint_boundary": {
            "cadence_steps": checkpoint_every_steps,
            "collective_type": "all_reduce_sum",
            "collective_count": collective_chunks,
            "collective_payload_bytes": parameter_bytes,
            "collective_chunk_bytes": collective_chunk_bytes,
            "cpu_aggregation": "per_rank_no_rank0_gather",
            "serialization_scope": "model_optimizer_scheduler_and_resident_capacity_buffers",
        },
        "mathematics_implemented": False,
        "scope": "synthetic_compute_communication_upper_bound",
    }
    value["artifact_hash"] = stable_json_hash(value)
    return value


def estimate_activation_bytes(
    *,
    hidden_size: int,
    layers: int,
    sequence_length: int,
    microbatch_size: int,
    dtype: str,
    activation_factor: int = 16,
    gradient_checkpointing: bool = False,
) -> int:
    """Conservative transformer activation estimate bound to one fixed config."""

    integers = (hidden_size, layers, sequence_length, microbatch_size, activation_factor)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in integers):
        raise ValueError("CAPACITY_ACTIVATION_INPUT_INVALID")
    multiplier = max(4, activation_factor // 2) if gradient_checkpointing else activation_factor
    return (
        hidden_size
        * layers
        * sequence_length
        * microbatch_size
        * dtype_nbytes(dtype)
        * multiplier
    )


def estimate_fixed_model_budget(
    *,
    model_id: str,
    tensors: Sequence[ParameterTensorShape],
    hidden_size: int,
    layers: int,
    sequence_length: int,
    microbatch_size: int,
    compute_dtype: str,
    retained_checkpoints: int,
    checkpoint_every_steps: int,
    seed_count: int,
    parallel_runs: int,
    logs_and_reports_per_run: int,
) -> dict[str, Any]:
    """Estimate one model from its own shapes and config, never another scale."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (retained_checkpoints, seed_count, parallel_runs, logs_and_reports_per_run)
    ):
        raise ValueError("CAPACITY_FIXED_MODEL_BUDGET_INPUT_INVALID")
    ordered = tuple(tensors)
    parameter_count = sum(item.element_count for item in ordered if item.requires_grad)
    source_parameter_bytes = sum(item.storage_bytes for item in ordered if item.requires_grad)
    synthetic = build_parameter_state_envelope(
        model_id=model_id,
        tensors=ordered,
        config_hash="0" * 64,
        model_manifest_id=f"analysis:{model_id}",
        checkpoint_every_steps=checkpoint_every_steps,
    )
    activation_bytes = estimate_activation_bytes(
        hidden_size=hidden_size,
        layers=layers,
        sequence_length=sequence_length,
        microbatch_size=microbatch_size,
        dtype=compute_dtype,
    )
    checkpoint_bytes = int(synthetic["checkpoint_parameter_state_bytes"])
    per_run_storage = (
        checkpoint_bytes * retained_checkpoints
        + logs_and_reports_per_run
    )
    value: dict[str, Any] = {
        "schema_version": "stage0.fixed-model-capacity-budget.v1",
        "model_id": model_id,
        "parameter_count": parameter_count,
        "tensor_count": len(ordered),
        "tensor_shape_hash": synthetic["tensor_shape_hash"],
        "source_parameter_bytes": source_parameter_bytes,
        "activation_bytes_estimate": activation_bytes,
        "resident_gpu_bytes_estimate": int(synthetic["resident_gpu_bytes"]) + activation_bytes,
        "transient_gpu_bytes_estimate": int(synthetic["transient_gpu_bytes_max"]),
        "checkpoint_bytes_estimate": checkpoint_bytes,
        "retained_checkpoints": retained_checkpoints,
        "logs_and_reports_per_run": logs_and_reports_per_run,
        "seed_count": seed_count,
        "parallel_runs": parallel_runs,
        "total_storage_bytes_estimate": per_run_storage * seed_count * parallel_runs,
        "derived_from_other_model_scale": False,
    }
    value["artifact_hash"] = stable_json_hash(value)
    return value


def estimation_error(*, estimated_bytes: int, measured_bytes: int) -> dict[str, float | int]:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (estimated_bytes, measured_bytes)
    ):
        raise ValueError("CAPACITY_ESTIMATION_ERROR_INPUT_INVALID")
    delta = measured_bytes - estimated_bytes
    return {
        "estimated_bytes": estimated_bytes,
        "measured_bytes": measured_bytes,
        "signed_error_bytes": delta,
        "relative_error": delta / measured_bytes,
        "estimate_to_measurement_ratio": estimated_bytes / measured_bytes,
    }


@dataclass(frozen=True, slots=True)
class StorageBudget:
    name: str
    expected_new_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int

    @classmethod
    def from_expected(cls, name: str, expected_new_bytes: int) -> "StorageBudget":
        if expected_new_bytes < 0:
            raise ValueError("expected_new_bytes cannot be negative")
        margin = max((expected_new_bytes + 4) // 5, 100 * GIB)
        return cls(name, expected_new_bytes, margin, expected_new_bytes + margin)

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def check_storage_budget(path: str | Path, budget: StorageBudget) -> dict[str, int | bool]:
    usage = shutil.disk_usage(Path(path))
    return {
        "free_bytes": usage.free,
        "required_free_bytes": budget.required_free_bytes,
        "ok": usage.free >= budget.required_free_bytes,
    }


def check_launch_storage(
    *,
    data_root: str | Path,
    root_filesystem: str | Path,
    budget: StorageBudget,
    root_minimum_free_bytes: int = 10 * GIB,
) -> dict[str, int | bool | None]:
    data_usage = shutil.disk_usage(Path(data_root))
    root_usage = shutil.disk_usage(Path(root_filesystem))
    inode_free: int | None = None
    inode_total: int | None = None
    if hasattr(os, "statvfs"):
        stat = os.statvfs(Path(data_root))
        inode_free = stat.f_favail
        inode_total = stat.f_files
    data_ok = data_usage.free >= budget.required_free_bytes
    root_ok = root_usage.free >= root_minimum_free_bytes
    inode_ok = inode_free is None or inode_free > 0
    return {
        "data_free_bytes": data_usage.free,
        "data_required_free_bytes": budget.required_free_bytes,
        "root_free_bytes": root_usage.free,
        "root_minimum_free_bytes": root_minimum_free_bytes,
        "inode_free": inode_free,
        "inode_total": inode_total,
        "data_ok": data_ok,
        "root_ok": root_ok,
        "inode_ok": inode_ok,
        "ok": data_ok and root_ok and inode_ok,
    }


def estimate_checkpoint_bytes(parameter_count: int, *, safety_factor: float = 1.25) -> int:
    """Conservative BF16 model + FP32 Adam states/RNG/metadata estimate."""

    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    base_bytes_per_parameter = 2 + 4 + 4 + 4
    return int(parameter_count * base_bytes_per_parameter * safety_factor)


def estimate_parameter_statistics_bytes(
    parameter_count: int,
    *,
    resident_fp32_buffers: int,
    transient_fp32_buffers: int = 2,
) -> int:
    if parameter_count <= 0 or resident_fp32_buffers < 0 or transient_fp32_buffers < 0:
        raise ValueError("invalid parameter statistics estimate")
    return parameter_count * 4 * (resident_fp32_buffers + transient_fp32_buffers)


def estimate_experiment_storage(
    *,
    parameter_count: int,
    retained_checkpoints: int,
    resident_fp32_buffers: int,
    seed_count: int,
    parallel_runs: int,
    logs_and_reports_per_run: int,
) -> int:
    values: Iterable[int] = (
        estimate_checkpoint_bytes(parameter_count) * retained_checkpoints,
        estimate_parameter_statistics_bytes(
            parameter_count, resident_fp32_buffers=resident_fp32_buffers
        ),
        logs_and_reports_per_run,
    )
    per_run = sum(values)
    return per_run * seed_count * parallel_runs
