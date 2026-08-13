"""Fresh-process real-asset capacity worker for formal Stage 0 S0.10.

The synthetic tensors in this module reproduce only declared shape, dtype,
lifetime, update, collective, and serialization costs.  They intentionally do
not implement raw importance, U-statistics, or path integration mathematics.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import time
from typing import Any, Final, Mapping, Sequence

import torch
import torch.distributed as dist

try:  # ``resource`` is available on the formal Linux server, not on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - local importability only
    _resource = None

from .atomic import sha256_file
from .capacity import ParameterTensorShape, build_parameter_state_envelope
from .contracts import ResolvedConfigV2, canonical_json_hash, load_canonical_json, write_canonical_json
from .contracts.jsonio import JSONValue
from .experiments.task_runners import _training_resources
from .providers import TorchModelAdapter, configure_batch_cursor
from .runtime import (
    EventRecord,
    EventType,
    JsonlEventSink,
    RunStatus,
    RunStatusStore,
    SessionStatus,
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
    build_optimizer,
    build_scheduler,
    install_training_rng,
    rebuild_tensorboard_from_jsonl,
)
from .runtime.tensor_bundle import publish_tensor_bundle


WORKER_PLAN_SCHEMA = "stage0-g8-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage0-g8-worker-report-v1"
NCCL_P2P_DISABLE_PROTOCOL: Final = 1
SYNTHETIC_BUFFER_IDS = (
    "raw_accumulator",
    "signed_accumulator",
    "positive_accumulator",
    "negative_accumulator",
    "u_first_moment",
    "u_second_moment",
    "path_integral_accumulator",
)


class Stage0G8WorkerError(RuntimeError):
    """A capacity child violated the immutable G8 execution protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G8WorkerError(f"G8_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G8WorkerError(f"G8_WORKER_REF_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G8WorkerError(f"G8_WORKER_REF_ESCAPE:{field}")
    result = root.joinpath(*logical.parts).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise Stage0G8WorkerError(f"G8_WORKER_REF_ESCAPE:{field}") from error
    return result


def _load_hashed(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(path), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage0G8WorkerError(f"G8_WORKER_HASHED_CONTROL_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def _validate_plan(root: Path, plan_ref: str) -> tuple[dict[str, Any], Path, TaskExecutionRequest]:
    path = _logical_path(root, plan_ref, field="plan_ref")
    plan = _load_hashed(path, schema=WORKER_PLAN_SCHEMA, field="plan")
    expected = {
        "schema_version", "run_id", "profile_id", "gate_id", "model_id", "mode",
        "precision_profile", "candidate_stage", "repeat_index", "world_size", "selected_gpu_uuids", "nccl_p2p_disable", "generator_git_commit",
        "config_ref", "config_sha256", "config_hash", "environment_ref",
        "environment_sha256", "environment_hash", "parameter_envelope_ref",
        "parameter_envelope_sha256", "parameter_envelope_hash", "work_envelope_ref",
        "work_envelope_sha256", "work_envelope_hash", "output_root_ref", "report_ref",
        "warmup_steps", "measured_steps", "memory_warmup_steps", "memory_measured_steps",
        "loader_worker_candidates", "run_loader_sweep", "checkpoint_steps",
        "collective_chunk_elements", "timeout_seconds", "artifact_hash",
    }
    if set(plan) != expected:
        raise Stage0G8WorkerError("G8_WORKER_PLAN_FIELDS_INVALID")
    if plan.get("nccl_p2p_disable") != NCCL_P2P_DISABLE_PROTOCOL:
        raise Stage0G8WorkerError("G8_WORKER_NCCL_P2P_PROTOCOL_INVALID")
    if plan.get("profile_id") not in {"g8-c-14m", "g8-s4-160m", "g8-s5-410m"}:
        raise Stage0G8WorkerError("G8_WORKER_PROFILE_INVALID")
    expected_gate = {
        "g8-c-14m": "stage0.G8-C",
        "g8-s4-160m": "stage0.G8-S4",
        "g8-s5-410m": "stage0.G8-S5",
    }[str(plan["profile_id"])]
    if plan.get("gate_id") != expected_gate or plan.get("mode") not in {"minimal", "formal"}:
        raise Stage0G8WorkerError("G8_WORKER_GATE_OR_MODE_INVALID")
    allowed_precisions = {"fp32", "bf16"} if plan["profile_id"] == "g8-c-14m" else {"bf16"}
    if plan.get("precision_profile") not in allowed_precisions:
        raise Stage0G8WorkerError("G8_WORKER_PRECISION_PROFILE_INVALID")
    expected_stage = {"g8-c-14m": 0, "g8-s4-160m": 4, "g8-s5-410m": 5}[str(plan["profile_id"])]
    if plan.get("candidate_stage") != expected_stage:
        raise Stage0G8WorkerError("G8_WORKER_CANDIDATE_STAGE_INVALID")
    if plan.get("repeat_index") not in {0, 1, 2}:
        raise Stage0G8WorkerError("G8_WORKER_REPEAT_INVALID")
    world_size = plan.get("world_size")
    selected = plan.get("selected_gpu_uuids")
    if (
        world_size not in {1, 4}
        or not isinstance(selected, list)
        or len(selected) != world_size
        or len(set(selected)) != world_size
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G8WorkerError("G8_WORKER_DEVICE_SET_INVALID")
    if plan["profile_id"] != "g8-c-14m" and world_size != 4:
        raise Stage0G8WorkerError("G8_WORKER_SCALE_PROFILE_REQUIRES_FOUR_GPU")
    if (
        plan.get("warmup_steps") != 10
        or plan.get("measured_steps") != 30
        or plan.get("memory_warmup_steps") != 5
        or plan.get("memory_measured_steps") != 20
        or plan.get("loader_worker_candidates") != [0, 1, 2, 4]
        or type(plan.get("run_loader_sweep")) is not bool
    ):
        raise Stage0G8WorkerError("G8_WORKER_MEASUREMENT_PROTOCOL_INVALID")
    if plan["run_loader_sweep"] is not (
        plan["profile_id"] == "g8-c-14m"
        and plan["precision_profile"] == "bf16"
        and plan["mode"] == "formal"
        and plan["repeat_index"] == 0
        and world_size == 4
    ):
        raise Stage0G8WorkerError("G8_WORKER_LOADER_SWEEP_ASSIGNMENT_INVALID")
    checkpoint_steps = plan.get("checkpoint_steps")
    if (
        not isinstance(checkpoint_steps, list)
        or not checkpoint_steps
        or checkpoint_steps[-1] != 40
        or checkpoint_steps != sorted(set(checkpoint_steps))
        or any(not isinstance(item, int) or isinstance(item, bool) or not 1 <= item <= 40 for item in checkpoint_steps)
    ):
        raise Stage0G8WorkerError("G8_WORKER_CHECKPOINT_STEPS_INVALID")
    for ref_field, sha_field in (
        ("config_ref", "config_sha256"),
        ("environment_ref", "environment_sha256"),
        ("parameter_envelope_ref", "parameter_envelope_sha256"),
        ("work_envelope_ref", "work_envelope_sha256"),
    ):
        control = _logical_path(root, plan[ref_field], field=ref_field)
        if sha256_file(control) != plan[sha_field]:
            raise Stage0G8WorkerError(f"G8_WORKER_CONTROL_SHA_MISMATCH:{ref_field}")
    config_path = _logical_path(root, plan["config_ref"], field="config_ref")
    environment_path = _logical_path(root, plan["environment_ref"], field="environment_ref")
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(config_path), field="config")
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(load_canonical_json(environment_path), field="environment")
    )
    distributed = _mapping(config.base_config.section("distributed"), field="distributed")
    model = _mapping(config.base_config.section("model"), field="model")
    training = _mapping(config.section("training"), field="training")
    profiling = _mapping(config.section("profiling"), field="profiling")
    if (
        config.task_id != "stage0.10_capacity_and_operations"
        or config.config_hash != plan["config_hash"]
        or environment.environment_hash != plan["environment_hash"]
        or distributed.get("world_size") != world_size
        or model.get("asset_id") != plan["model_id"]
        or training.get("max_steps") != 100
        or profiling != {
            "enabled": True,
            "warmup_steps": 10,
            "measure_steps": 30,
            "repetitions": 3,
            "capture_memory": True,
            "capture_throughput": True,
            "capture_communication": True,
            "synchronize_device": True,
        }
        or _mapping(config.base_config.section("precision"), field="precision").get("compute_dtype")
        != ("float32" if plan["precision_profile"] == "fp32" else "bfloat16")
    ):
        raise Stage0G8WorkerError("G8_WORKER_CONFIG_IDENTITY_INVALID")
    parameter_envelope = _load_hashed(
        _logical_path(root, plan["parameter_envelope_ref"], field="parameter_envelope_ref"),
        schema="stage0.parameter-state-capacity-envelope.v1",
        field="parameter_envelope",
    )
    work_envelope = _load_hashed(
        _logical_path(root, plan["work_envelope_ref"], field="work_envelope_ref"),
        schema="stage0.compute-communication-work-envelope.v1",
        field="work_envelope",
    )
    batching = _mapping(config.base_config.section("batching"), field="batching")
    data = _mapping(config.base_config.section("data"), field="data")
    precision = _mapping(config.base_config.section("precision"), field="precision")
    if (
        parameter_envelope.get("artifact_hash") != plan["parameter_envelope_hash"]
        or work_envelope.get("artifact_hash") != plan["work_envelope_hash"]
        or parameter_envelope.get("model_id") != plan["model_id"]
        or work_envelope.get("model_id") != plan["model_id"]
        or work_envelope.get("world_size") != world_size
        or work_envelope.get("candidate_stage") != plan["candidate_stage"]
        or work_envelope.get("compute_dtype") != precision["compute_dtype"]
        or work_envelope.get("sequence_length") != data["sequence_length"]
        or work_envelope.get("microbatch_size") != batching["microbatch_size"]
        or parameter_envelope.get("mathematics_implemented") is not False
        or work_envelope.get("mathematics_implemented") is not False
    ):
        raise Stage0G8WorkerError("G8_WORKER_ENVELOPE_IDENTITY_INVALID")
    return plan, path, TaskExecutionRequest(config, config.task_definition, environment)


def _rank_identity(plan: Mapping[str, Any]) -> dict[str, JSONValue]:
    world_size = int(plan["world_size"])
    if world_size == 1:
        rank = local_rank = 0
        observed_world = 1
    else:
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            observed_world = int(os.environ["WORLD_SIZE"])
        except (KeyError, ValueError) as error:
            raise Stage0G8WorkerError("G8_WORKER_TORCHRUN_ENV_INVALID") from error
    selected = tuple(str(item) for item in plan["selected_gpu_uuids"])
    if (
        observed_world != world_size
        or not 0 <= rank < world_size
        or local_rank != rank
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(selected)
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != world_size
    ):
        raise Stage0G8WorkerError("G8_WORKER_RANK_MAPPING_INVALID")
    torch.cuda.set_device(local_rank)
    probe = torch.ones(1, device=torch.device("cuda", local_rank))
    probe.add_(rank)
    torch.cuda.synchronize(local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "gpu_uuid": selected[local_rank],
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
    }


def _validate_nccl_transport_environment() -> None:
    if os.environ.get("NCCL_P2P_DISABLE") != str(NCCL_P2P_DISABLE_PROTOCOL):
        raise Stage0G8WorkerError("G8_WORKER_NCCL_P2P_ENVIRONMENT_INVALID")


def _init_distributed(world_size: int, timeout_seconds: int) -> None:
    if world_size == 1:
        return
    if not dist.is_available() or not dist.is_nccl_available():
        raise Stage0G8WorkerError("G8_WORKER_NCCL_UNAVAILABLE")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=timeout_seconds)
    )


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _gather(world_size: int, value: object) -> list[object]:
    if world_size == 1:
        return [value]
    result: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(result, value)
    return result


def _percentile95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("G8_WORKER_PERCENTILE_EMPTY")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _fd_count() -> int:
    path = Path("/proc/self/fd")
    if not path.is_dir():
        raise Stage0G8WorkerError("G8_WORKER_PROC_FD_UNAVAILABLE")
    return len(tuple(path.iterdir()))


def _host_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    status = Path("/proc/self/status")
    if not status.is_file():
        raise Stage0G8WorkerError("G8_WORKER_PROC_STATUS_UNAVAILABLE")
    for line in status.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM"}:
            result[f"{key.lower()}_bytes"] = int(raw.strip().split()[0]) * 1024
    if set(result) != {"vmrss_bytes", "vmhwm_bytes"}:
        raise Stage0G8WorkerError("G8_WORKER_PROC_MEMORY_FIELDS_MISSING")
    return result


def _gpu_processes() -> list[dict[str, JSONValue]]:
    probe = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        raise Stage0G8WorkerError("G8_WORKER_GPU_PROCESS_QUERY_FAILED")
    rows: list[dict[str, JSONValue]] = []
    for line in probe.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) == 3 and fields[1].isdigit():
            rows.append(
                {"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": Path(fields[2]).name}
            )
    return rows


def _gpu_health(selected: Sequence[str]) -> list[dict[str, JSONValue]]:
    fields = (
        "uuid,temperature.gpu,memory.total,memory.used,"
        "ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total"
    )
    probe = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        raise Stage0G8WorkerError("G8_WORKER_GPU_HEALTH_QUERY_FAILED")

    def integer(value: str) -> int | None:
        return int(value) if value.isdigit() else None

    rows: dict[str, dict[str, JSONValue]] = {}
    for line in probe.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 6:
            continue
        rows[values[0]] = {
            "gpu_uuid": values[0],
            "temperature_c": integer(values[1]),
            "memory_total_mib": integer(values[2]),
            "memory_used_mib": integer(values[3]),
            "volatile_uncorrected_ecc": integer(values[4]),
            "aggregate_uncorrected_ecc": integer(values[5]),
        }
    if set(selected) - set(rows):
        raise Stage0G8WorkerError("G8_WORKER_SELECTED_GPU_HEALTH_MISSING")
    return [rows[item] for item in selected]


def _resource_snapshot(selected: Sequence[str], allowed_pids: set[int]) -> dict[str, JSONValue]:
    processes = _gpu_processes()
    selected_processes = [item for item in processes if item["gpu_uuid"] in selected]
    unknown = [item for item in selected_processes if int(item["pid"]) not in allowed_pids]
    return {
        "observed_at": _now(),
        "host_memory": _host_memory(),
        "open_fds": _fd_count(),
        "gpu_health": _gpu_health(selected),
        "selected_gpu_processes": selected_processes,
        "unknown_gpu_processes": unknown,
    }


def _loader_sweep(
    resources: object,
    *,
    seed: int,
    rank: int,
    world_size: int,
    candidates: Sequence[int],
) -> list[dict[str, JSONValue]]:
    dataset = getattr(resources, "dataset")
    rows: list[dict[str, JSONValue]] = []
    for workers in candidates:
        before = _fd_count()
        cursor = configure_batch_cursor(
            dataset.cursor(seed=seed, rank=rank, world_size=world_size),
            num_workers=int(workers),
            prefetch_factor=(None if workers == 0 else 2),
            persistent_workers=workers > 0,
        )
        durations: list[float] = []
        sample_count = 0
        peak = before
        try:
            for _ in range(8):
                started = time.perf_counter()
                batches = cursor.next_microbatches()
                durations.append(time.perf_counter() - started)
                sample_count += sum(len(item.sample_ids) for item in batches)
                peak = max(peak, _fd_count())
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
        wall = sum(durations)
        rows.append(
            {
                "num_workers": workers,
                "prefetch_factor": None if workers == 0 else 2,
                "steps": 8,
                "samples": sample_count,
                "wall_seconds": wall,
                "samples_per_second": sample_count / wall,
                "open_fds_before": before,
                "open_fds_peak": peak,
                "open_fds_after": _fd_count(),
            }
        )
    return rows


def _exact_shapes(module: torch.nn.Module) -> tuple[ParameterTensorShape, ...]:
    return tuple(
        ParameterTensorShape(
            name,
            tuple(int(value) for value in parameter.shape),
            str(parameter.dtype).removeprefix("torch."),
            requires_grad=bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters(remove_duplicate=True)
    )


def _allocate_synthetic_buffers(
    *, total_elements: int, device: torch.device
) -> dict[str, torch.Tensor]:
    try:
        return {
            name: torch.zeros(total_elements, dtype=torch.float32, device=device)
            for name in SYNTHETIC_BUFFER_IDS
        }
    except torch.OutOfMemoryError as error:
        raise Stage0G8WorkerError("G8_WORKER_SYNTHETIC_ENVELOPE_OOM") from error


def _synthetic_update(buffers: Mapping[str, torch.Tensor], step: int) -> float:
    started = time.perf_counter()
    coefficient = float((step % 13) + 1) * 1e-7
    for name in SYNTHETIC_BUFFER_IDS:
        buffers[name].add_(coefficient)
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _synthetic_collective(
    source: torch.Tensor,
    *, world_size: int,
    chunk_elements: int,
) -> tuple[float, int]:
    started = time.perf_counter()
    scratch = torch.empty_like(source)
    scratch.copy_(source)
    calls = 0
    if world_size > 1:
        for start in range(0, scratch.numel(), chunk_elements):
            dist.all_reduce(
                scratch[start : start + chunk_elements], op=dist.ReduceOp.SUM
            )
            calls += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del scratch
    return elapsed, calls


def _scratch_checkpoint(
    root: Path,
    output_root: Path,
    *,
    step: int,
    rank: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    buffers: Mapping[str, torch.Tensor],
    cursor_state: Mapping[str, object],
) -> dict[str, JSONValue]:
    scratch = output_root / "scratch" / f"rank-{rank:04d}-step-{step:08d}"
    audit = output_root / "checkpoint-audit"
    audit.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    bundle = publish_tensor_bundle(
        scratch,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "synthetic_parameter_state_capacity_only": dict(buffers),
            "cursor": dict(cursor_state),
            "step": step,
            "mathematics_implemented": False,
        },
    )
    elapsed = time.perf_counter() - started
    bytes_written = sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file())
    if bytes_written <= 0:
        raise Stage0G8WorkerError("G8_WORKER_SCRATCH_CHECKPOINT_EMPTY")
    intent: dict[str, JSONValue] = {
        "schema_version": "stage0-g8-scratch-purge-intent-v1",
        "rank": rank,
        "step": step,
        "exact_scratch_ref": scratch.relative_to(root).as_posix(),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "authorized_release_bytes": bytes_written,
        "reason": "generated G8 capacity fixture; durable report retains measurements",
        "authorized_at": _now(),
    }
    intent["artifact_hash"] = canonical_json_hash(intent)
    intent_path = audit / f"rank-{rank:04d}-step-{step:08d}-purge-intent.json"
    write_canonical_json(intent_path, intent)
    shutil.rmtree(scratch)
    if scratch.exists():
        raise Stage0G8WorkerError("G8_WORKER_SCRATCH_PURGE_FAILED")
    record: dict[str, JSONValue] = {
        "schema_version": "stage0-g8-scratch-purge-record-v1",
        "rank": rank,
        "step": step,
        "purge_intent_ref": intent_path.relative_to(root).as_posix(),
        "purge_intent_hash": str(intent["artifact_hash"]),
        "released_bytes": bytes_written,
        "objects_deleted": 1,
        "completed_at": _now(),
    }
    record["artifact_hash"] = canonical_json_hash(record)
    record_path = audit / f"rank-{rank:04d}-step-{step:08d}-purge-record.json"
    write_canonical_json(record_path, record)
    return {
        "step": step,
        "save_wall_seconds": elapsed,
        "checkpoint_bytes": bytes_written,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "purge_intent_ref": intent_path.relative_to(root).as_posix(),
        "purge_record_ref": record_path.relative_to(root).as_posix(),
        "released_bytes": bytes_written,
        "scratch_removed": True,
    }


def _health_delta_ok(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    before_rows = {
        row["gpu_uuid"]: row for row in before["gpu_health"] if isinstance(row, Mapping)
    }
    after_rows = {
        row["gpu_uuid"]: row for row in after["gpu_health"] if isinstance(row, Mapping)
    }
    if set(before_rows) != set(after_rows):
        return False
    for uuid, first in before_rows.items():
        last = after_rows[uuid]
        for field in ("volatile_uncorrected_ecc", "aggregate_uncorrected_ecc"):
            old = first[field]
            new = last[field]
            if old is not None and new is not None and int(new) > int(old):
                return False
    return True


def _run_training(
    root: Path,
    plan: Mapping[str, Any],
    request: TaskExecutionRequest,
    identity: Mapping[str, Any],
    allowed_pids: set[int],
) -> dict[str, JSONValue]:
    rank = int(identity["rank"])
    world_size = int(plan["world_size"])
    device = torch.device("cuda", rank)
    output_root = _logical_path(root, plan["output_root_ref"], field="output_root_ref")
    rank_root = output_root / f"rank-{rank:04d}"
    rank_root.mkdir(parents=True, exist_ok=False)
    before = _resource_snapshot(tuple(plan["selected_gpu_uuids"]), allowed_pids)
    if before["unknown_gpu_processes"]:
        raise Stage0G8WorkerError("G8_WORKER_EXTERNAL_GPU_PROCESS_BEFORE")
    base = request.config.base_config
    identity_config = _mapping(base.section("identity"), field="identity")
    seed_plan = install_training_rng(
        int(identity_config["master_seed"]), rank=rank, world_size=world_size
    )
    torch.cuda.reset_peak_memory_stats(device)
    resources = _training_resources(
        request,
        root,
        rank=rank,
        world_size=world_size,
        data_route_stage=int(plan["candidate_stage"]),
    )
    loader_profiles: list[dict[str, JSONValue]] = []
    if bool(plan["run_loader_sweep"]):
        loader_profiles = _loader_sweep(
            resources,
            seed=seed_plan.seed_for("sampler"),
            rank=rank,
            world_size=world_size,
            candidates=tuple(int(item) for item in plan["loader_worker_candidates"]),
        )
    resources.model.module.to(device)
    torch.cuda.synchronize(device)
    model_loaded_memory = int(torch.cuda.memory_allocated(device))
    if world_size > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            resources.model.module,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
        )
        model = TorchModelAdapter(wrapped, task_type=resources.model.task_type)
        checkpoint_model = wrapped.module
    else:
        model = resources.model
        checkpoint_model = resources.model.module
    shapes = _exact_shapes(checkpoint_model)
    envelope = _load_hashed(
        _logical_path(root, plan["parameter_envelope_ref"], field="parameter_envelope_ref"),
        schema="stage0.parameter-state-capacity-envelope.v1",
        field="parameter_envelope",
    )
    rebuilt = build_parameter_state_envelope(
        model_id=str(plan["model_id"]),
        tensors=shapes,
        config_hash=request.config.config_hash,
        model_manifest_id=str(envelope["model_manifest_id"]),
        checkpoint_every_steps=int(envelope["checkpoint_every_steps"]),
    )
    if rebuilt != envelope:
        raise Stage0G8WorkerError("G8_WORKER_REAL_PARAMETER_SHAPES_DRIFT")
    parameter_count = int(envelope["trainable_parameter_count"])
    optimizer = build_optimizer(
        model.module.parameters(),
        _mapping(base.section("optimizer"), field="optimizer"),
        _mapping(request.config.section("optimizer_runtime"), field="optimizer_runtime"),
    )
    scheduler = build_scheduler(
        optimizer, _mapping(request.config.section("scheduler"), field="scheduler")
    )
    data_loader = _mapping(request.config.section("data_loader"), field="data_loader")
    cursor = configure_batch_cursor(
        resources.dataset.cursor(
            seed=seed_plan.seed_for("sampler"), rank=rank, world_size=world_size
        ),
        num_workers=int(data_loader["num_workers"]),
        prefetch_factor=data_loader["prefetch_factor"],
        persistent_workers=bool(data_loader["persistent_workers"]),
    )
    synthetic = _allocate_synthetic_buffers(total_elements=parameter_count, device=device)
    torch.cuda.synchronize(device)
    envelope_allocated_memory = int(torch.cuda.memory_allocated(device))
    if _resource is None:
        raise Stage0G8WorkerError("G8_WORKER_RESOURCE_MODULE_UNAVAILABLE")
    fd_soft, _fd_hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    if fd_soft <= 0:
        raise Stage0G8WorkerError("G8_WORKER_FD_LIMIT_INVALID")
    event_path = rank_root / "events.jsonl"
    status: RunStatusStore | None = None
    if plan["mode"] == "formal":
        status = RunStatusStore(rank_root / "run-status.json", run_id=str(plan["run_id"]))
        status.transition_run(RunStatus.RUNNING, actor="stage0-g8-worker", reason="capacity measurement")
        status.register_attempt(attempt_id="attempt-0001")
        status.transition_attempt("attempt-0001", SessionStatus.RUNNING, actor="stage0-g8-worker", reason="started")
        status.register_session(attempt_id="attempt-0001", session_id="session-0001")
        status.transition_session("session-0001", SessionStatus.RUNNING, actor="stage0-g8-worker", reason="started")
    step_seconds: list[float] = []
    measured_effective_tokens = 0
    measured_samples = 0
    measured_losses: list[float] = []
    synthetic_update_seconds: list[float] = []
    checkpoint_records: list[dict[str, JSONValue]] = []
    fd_peak = _fd_count()
    memory_window_peak: int | None = None
    total_steps = int(plan["warmup_steps"]) + int(plan["measured_steps"])
    compute_dtype = _mapping(base.section("precision"), field="precision")["compute_dtype"]
    autocast_enabled = bool(
        _mapping(request.config.section("precision_runtime"), field="precision_runtime")[
            "autocast_enabled"
        ]
    )
    autocast_dtype = torch.bfloat16 if compute_dtype == "bfloat16" else torch.float32
    max_grad_norm = _mapping(request.config.section("training"), field="training")[
        "gradient_clip_max_norm"
    ]
    checkpoint_steps = set(int(item) for item in plan["checkpoint_steps"])
    mode = str(plan["mode"])
    derived_seconds = 0.0
    sink = JsonlEventSink(event_path)
    try:
        sink.append(
            EventRecord.create(
                experiment_id="stage0-g8-capacity",
                run_id=str(plan["run_id"]),
                attempt_id="attempt-0001",
                session_id="session-0001",
                rank=rank,
                event_type=EventType.RUN_LIFECYCLE,
                sequence=0,
                payload={"status": "RUNNING"},
            ),
            critical=True,
        )
        for step_index in range(total_steps):
            step = step_index + 1
            if step == int(plan["memory_warmup_steps"]) + 1:
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            batches = cursor.next_microbatches()
            optimizer.zero_grad(set_to_none=True)
            local_effective = 0
            local_samples = 0
            local_loss_sum = 0.0
            for micro_index, cpu_batch in enumerate(batches):
                batch = cpu_batch.to(device)
                sync_context = (
                    model.module.no_sync()
                    if world_size > 1
                    and callable(getattr(model.module, "no_sync", None))
                    and micro_index < len(batches) - 1
                    else nullcontext()
                )
                with sync_context:
                    with torch.autocast(
                        device_type="cuda", dtype=autocast_dtype, enabled=autocast_enabled
                    ):
                        loss = model.loss(batch)
                        scaled = loss.mean_loss / len(batches)
                    scaled.backward()
                local_effective += int(loss.effective_count)
                local_samples += len(batch.sample_ids)
                local_loss_sum += float(loss.mean_loss.detach().item())
                del batch, loss, scaled
            gradient_norm = (
                float(torch.nn.utils.clip_grad_norm_(model.module.parameters(), float(max_grad_norm)).item())
                if max_grad_norm is not None
                else 0.0
            )
            if not math.isfinite(gradient_norm):
                raise Stage0G8WorkerError("G8_WORKER_NONFINITE_GRADIENT")
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # type: ignore[attr-defined]
            synthetic_update_seconds.append(_synthetic_update(synthetic, step))
            collective_seconds = 0.0
            collective_calls = 0
            if step in checkpoint_steps:
                _barrier(world_size)
                collective_seconds, collective_calls = _synthetic_collective(
                    synthetic[SYNTHETIC_BUFFER_IDS[0]],
                    world_size=world_size,
                    chunk_elements=int(plan["collective_chunk_elements"]),
                )
                _barrier(world_size)
                checkpoint = _scratch_checkpoint(
                    root,
                    output_root,
                    step=step,
                    rank=rank,
                    model=checkpoint_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    buffers=synthetic,
                    cursor_state=cursor.state_dict(),
                )
                checkpoint["collective_wall_seconds"] = collective_seconds
                checkpoint["collective_calls"] = collective_calls
                checkpoint["host_memory_after"] = _host_memory()
                checkpoint_records.append(checkpoint)
                _barrier(world_size)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            fd_peak = max(fd_peak, _fd_count())
            if step_index >= int(plan["warmup_steps"]):
                step_seconds.append(elapsed)
                measured_effective_tokens += local_effective
                measured_samples += local_samples
                measured_losses.append(local_loss_sum / len(batches))
            if mode == "formal":
                sink.append(
                    EventRecord.create(
                        experiment_id="stage0-g8-capacity",
                        run_id=str(plan["run_id"]),
                        attempt_id="attempt-0001",
                        session_id="session-0001",
                        rank=rank,
                        event_type=EventType.OPTIMIZER_STEP,
                        sequence=step,
                        payload={
                            "global_step": step,
                            "microstep_count": len(batches),
                            "sample_count": local_samples,
                            "effective_token_count": local_effective,
                            "mean_loss": local_loss_sum / len(batches),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "learning_rates_post_step": {
                                f"group_{index:04d}": float(group["lr"])
                                for index, group in enumerate(optimizer.param_groups)
                            },
                            "global_gradient_norm": gradient_norm,
                        },
                    )
                )
                if status is not None and step % 5 == 0:
                    status.heartbeat("session-0001", last_step=step)
            if step == int(plan["memory_warmup_steps"]) + int(plan["memory_measured_steps"]):
                memory_window_peak = int(torch.cuda.max_memory_allocated(device))
        sink.append(
            EventRecord.create(
                experiment_id="stage0-g8-capacity",
                run_id=str(plan["run_id"]),
                attempt_id="attempt-0001",
                session_id="session-0001",
                rank=rank,
                event_type=EventType.RUN_LIFECYCLE,
                sequence=(total_steps + 1 if mode == "formal" else 1),
                payload={"status": "SUCCESS"},
            ),
            critical=True,
        )
    finally:
        sink.close()
    tensorboard_refs: list[str] = []
    if mode == "formal":
        derived_started = time.perf_counter()
        tensorboard_root = rank_root / "tensorboard"
        scalar_count = rebuild_tensorboard_from_jsonl((event_path,), tensorboard_root)
        if scalar_count <= 0:
            raise Stage0G8WorkerError("G8_WORKER_TENSORBOARD_EMPTY")
        tensorboard_refs = [
            path.relative_to(root).as_posix()
            for path in sorted(tensorboard_root.rglob("*"))
            if path.is_file()
        ]
        derived_seconds = time.perf_counter() - derived_started
        assert status is not None
        status.transition_session("session-0001", SessionStatus.SUCCEEDED, actor="stage0-g8-worker", reason="complete")
        status.transition_attempt("attempt-0001", SessionStatus.SUCCEEDED, actor="stage0-g8-worker", reason="complete")
        status.transition_run(RunStatus.SUCCESS, actor="stage0-g8-worker", reason="complete")
        status.close()
    after = _resource_snapshot(tuple(plan["selected_gpu_uuids"]), allowed_pids)
    if after["unknown_gpu_processes"] or not _health_delta_ok(before, after):
        raise Stage0G8WorkerError("G8_WORKER_POSTFLIGHT_GPU_UNHEALTHY")
    measured_wall = sum(step_seconds) + derived_seconds
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    if memory_window_peak is None:
        raise Stage0G8WorkerError("G8_WORKER_MEMORY_WINDOW_NOT_CAPTURED")
    result: dict[str, JSONValue] = {
        "rank": rank,
        "gpu_uuid": str(identity["gpu_uuid"]),
        "parameter_count": parameter_count,
        "tensor_shape_hash": str(envelope["tensor_shape_hash"]),
        "source_parameter_dtype_counts": dict(envelope["source_dtype_counts"]),
        "synthetic_buffer_ids": list(SYNTHETIC_BUFFER_IDS),
        "synthetic_mathematics_implemented": False,
        "step_seconds": step_seconds,
        "step_median_seconds": statistics.median(step_seconds),
        "step_p95_seconds": _percentile95(step_seconds),
        "measured_wall_seconds": measured_wall,
        "measured_effective_tokens": measured_effective_tokens,
        "measured_samples": measured_samples,
        "effective_tokens_per_second": measured_effective_tokens / measured_wall,
        "samples_per_second": measured_samples / measured_wall,
        "mean_loss_median": statistics.median(measured_losses),
        "synthetic_update_seconds_total": sum(synthetic_update_seconds),
        "peak_memory_bytes": peak_memory,
        "memory_window_peak_bytes": memory_window_peak,
        "phase_memory_bytes": {
            "model_loaded_allocated": model_loaded_memory,
            "parameter_envelope_allocated": envelope_allocated_memory,
            "training_window_peak": memory_window_peak,
            "whole_process_peak": peak_memory,
        },
        "total_memory_bytes": int(identity["total_memory_bytes"]),
        "peak_memory_fraction": peak_memory / int(identity["total_memory_bytes"]),
        "host_memory": _host_memory(),
        "fd_soft_limit": int(fd_soft),
        "open_fds_peak": fd_peak,
        "open_fds_fraction": fd_peak / int(fd_soft),
        "loader_profiles": loader_profiles,
        "checkpoint_records": checkpoint_records,
        "checkpoint_pause_seconds_max": max(
            float(item["save_wall_seconds"]) + float(item["collective_wall_seconds"])
            for item in checkpoint_records
        ),
        "checkpoint_bytes_max": max(int(item["checkpoint_bytes"]) for item in checkpoint_records),
        "released_scratch_bytes": sum(int(item["released_bytes"]) for item in checkpoint_records),
        "event_stream_ref": event_path.relative_to(root).as_posix(),
        "event_stream_sha256": sha256_file(event_path),
        "tensorboard_refs": tensorboard_refs,
        "resource_snapshots": {"before": before, "after": after},
    }
    close = getattr(cursor, "close", None)
    if callable(close):
        close()
    dataset_close = getattr(resources.dataset, "close", None)
    if callable(dataset_close):
        dataset_close()
    del synthetic, optimizer, scheduler, model
    torch.cuda.empty_cache()
    return result


def _aggregate(
    plan: Mapping[str, Any],
    plan_path: Path,
    root: Path,
    identities: Sequence[object],
    ranks: Sequence[object],
) -> dict[str, JSONValue]:
    rank_rows = [_mapping(item, field="rank_report") for item in ranks]
    world_size = int(plan["world_size"])
    if len(rank_rows) != world_size or {int(item["rank"]) for item in rank_rows} != set(range(world_size)):
        raise Stage0G8WorkerError("G8_WORKER_RANK_REPORT_SET_INVALID")
    durations = [float(item["measured_wall_seconds"]) for item in rank_rows]
    total_tokens = sum(int(item["measured_effective_tokens"]) for item in rank_rows)
    total_samples = sum(int(item["measured_samples"]) for item in rank_rows)
    peak_values = [int(item["peak_memory_bytes"]) for item in rank_rows]
    report: dict[str, JSONValue] = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "run_id": str(plan["run_id"]),
        "profile_id": str(plan["profile_id"]),
        "gate_id": str(plan["gate_id"]),
        "model_id": str(plan["model_id"]),
        "mode": str(plan["mode"]),
        "precision_profile": str(plan["precision_profile"]),
        "candidate_stage": int(plan["candidate_stage"]),
        "repeat_index": int(plan["repeat_index"]),
        "status": "PASS",
        "completed_at": _now(),
        "generator_git_commit": str(plan["generator_git_commit"]),
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "config_hash": str(plan["config_hash"]),
        "environment_hash": str(plan["environment_hash"]),
        "parameter_envelope_hash": str(plan["parameter_envelope_hash"]),
        "work_envelope_hash": str(plan["work_envelope_hash"]),
        "world_size": world_size,
        "selected_gpu_uuids": list(plan["selected_gpu_uuids"]),
        "nccl_p2p_disable": int(plan["nccl_p2p_disable"]),
        "rank_identities": list(identities),
        "warmup_steps": int(plan["warmup_steps"]),
        "measured_steps": int(plan["measured_steps"]),
        "memory_warmup_steps": int(plan["memory_warmup_steps"]),
        "memory_measured_steps": int(plan["memory_measured_steps"]),
        "rank_reports": rank_rows,
        "effective_tokens_per_second": total_tokens / max(durations),
        "samples_per_second": total_samples / max(durations),
        "step_median_seconds": statistics.median(
            [float(item["step_median_seconds"]) for item in rank_rows]
        ),
        "step_p95_seconds": max(float(item["step_p95_seconds"]) for item in rank_rows),
        "peak_memory_bytes_max": max(peak_values),
        "peak_memory_fraction_max": max(float(item["peak_memory_fraction"]) for item in rank_rows),
        "rank_peak_memory_imbalance_fraction": (
            0.0 if max(peak_values) == 0 else (max(peak_values) - min(peak_values)) / max(peak_values)
        ),
        "checkpoint_pause_seconds_max": max(
            float(item["checkpoint_pause_seconds_max"]) for item in rank_rows
        ),
        "checkpoint_bytes_total_max_boundary": sum(
            int(item["checkpoint_bytes_max"]) for item in rank_rows
        ),
        "released_scratch_bytes": sum(int(item["released_scratch_bytes"]) for item in rank_rows),
        "open_fds_fraction_max": max(float(item["open_fds_fraction"]) for item in rank_rows),
        "synthetic_mathematics_implemented": False,
        "all_scratch_removed": all(
            all(bool(record["scratch_removed"]) for record in item["checkpoint_records"])
            for item in rank_rows
        ),
        "no_unknown_gpu_processes": all(
            not _mapping(item["resource_snapshots"], field="resource")["before"]["unknown_gpu_processes"]
            and not _mapping(item["resource_snapshots"], field="resource")["after"]["unknown_gpu_processes"]
            for item in rank_rows
        ),
    }
    if (
        report["peak_memory_fraction_max"] > 0.80
        or report["open_fds_fraction_max"] > 0.70
        or report["all_scratch_removed"] is not True
        or report["no_unknown_gpu_processes"] is not True
    ):
        raise Stage0G8WorkerError("G8_WORKER_CAPACITY_OR_OPERATION_THRESHOLD_FAILED")
    report["artifact_hash"] = canonical_json_hash(report)
    return report


def run_stage0_g8_worker(*, data_root: str | Path, plan_ref: str) -> dict[str, JSONValue] | None:
    root = Path(data_root).resolve(strict=True)
    plan, plan_path, request = _validate_plan(root, plan_ref)
    _validate_nccl_transport_environment()
    output_root = _logical_path(root, plan["output_root_ref"], field="output_root_ref")
    identity = _rank_identity(plan)
    world_size = int(plan["world_size"])
    _init_distributed(world_size, int(plan["timeout_seconds"]))
    try:
        output_ready = True
        if int(identity["rank"]) == 0:
            try:
                if output_root.exists():
                    raise Stage0G8WorkerError("G8_WORKER_OUTPUT_COLLISION")
                output_root.mkdir(parents=True)
            except Exception:
                output_ready = False
        if world_size > 1:
            flag = torch.tensor(
                [1 if output_ready else 0], dtype=torch.int64,
                device=torch.device("cuda", int(identity["local_rank"])),
            )
            dist.broadcast(flag, src=0)
            output_ready = bool(flag.item())
        if not output_ready:
            raise Stage0G8WorkerError("G8_WORKER_OUTPUT_COLLISION")
        _barrier(world_size)
        identities = _gather(world_size, identity)
        allowed_pids = {int(_mapping(item, field="identity")["pid"]) for item in identities}
        if world_size > 1 and (
            dist.get_world_size() != world_size or str(dist.get_backend()) != "nccl"
        ):
            raise Stage0G8WorkerError("G8_WORKER_PROCESS_GROUP_INVALID")
        rank_report = _run_training(root, plan, request, identity, allowed_pids)
        rank_reports = _gather(world_size, rank_report)
        _barrier(world_size)
        if int(identity["rank"]) != 0:
            return None
        report = _aggregate(plan, plan_path, root, identities, rank_reports)
        report_path = _logical_path(root, plan["report_ref"], field="report_ref")
        if report_path.exists():
            raise Stage0G8WorkerError("G8_WORKER_REPORT_COLLISION")
        write_canonical_json(report_path, report)
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "NCCL_P2P_DISABLE_PROTOCOL",
    "SYNTHETIC_BUFFER_IDS",
    "Stage0G8WorkerError",
    "WORKER_PLAN_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "_validate_nccl_transport_environment",
    "run_stage0_g8_worker",
]
