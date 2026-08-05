"""Fresh ``torchrun`` worker for the formal Stage 0 G6 four-GPU suite.

The parent process owns publication.  This module only performs one bounded
NCCL launch, gathers rank-local facts, and lets rank zero write a hash-bound
report.  A controlled rank-failure mode intentionally produces no success
report; its launcher transcript is validated by the parent.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from .atomic import sha256_file
from .contracts import (
    ResolvedConfigV2,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .runtime import TaskExecutionRequest, TaskRuntimeEnvironment, TorchDistributedReducer


WORKER_PLAN_SCHEMA = "stage0-g6-worker-plan-v2"
WORKER_REPORT_SCHEMA = "stage0-g6-worker-report-v1"
WORLD_SIZE = 4
FAILURE_MARKER = "STAGE0_G6_INJECTED_RANK_FAILURE"


class Stage0G6WorkerError(RuntimeError):
    """A child launch violated a frozen G6 precondition or protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G6WorkerError(f"G6_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G6WorkerError(f"G6_WORKER_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G6WorkerError(f"G6_WORKER_LOGICAL_PATH_ESCAPE:{field}")
    resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage0G6WorkerError(f"G6_WORKER_LOGICAL_PATH_ESCAPE:{field}") from error
    return resolved


def _validate_plan(root: Path, plan_ref: str) -> tuple[dict[str, Any], Path]:
    plan_path = _logical_path(root, plan_ref, field="plan_ref")
    plan = _mapping(load_canonical_json(plan_path), field="plan")
    expected = {
        "schema_version",
        "run_id",
        "run_kind",
        "repeat_index",
        "generator_git_commit",
        "config_ref",
        "config_sha256",
        "config_hash",
        "environment_ref",
        "environment_sha256",
        "environment_hash",
        "selected_gpu_uuids",
        "report_ref",
        "timeout_seconds",
        "collective_protocol",
        "artifact_hash",
    }
    if set(plan) != expected or plan.get("schema_version") != WORKER_PLAN_SCHEMA:
        raise Stage0G6WorkerError("G6_WORKER_PLAN_FIELDS_OR_VERSION_INVALID")
    declared = plan.pop("artifact_hash")
    if declared != canonical_json_hash(plan):
        raise Stage0G6WorkerError("G6_WORKER_PLAN_HASH_MISMATCH")
    plan["artifact_hash"] = declared
    run_kind = plan.get("run_kind")
    if run_kind not in {"collective", "semantic", "recovery", "failure_rank"}:
        raise Stage0G6WorkerError("G6_WORKER_RUN_KIND_INVALID")
    selected = plan.get("selected_gpu_uuids")
    if (
        not isinstance(selected, list)
        or len(selected) != WORLD_SIZE
        or len(set(selected)) != WORLD_SIZE
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G6WorkerError("G6_WORKER_DEVICE_SET_INVALID")
    config_path = _logical_path(root, plan["config_ref"], field="config_ref")
    environment_path = _logical_path(
        root, plan["environment_ref"], field="environment_ref"
    )
    if sha256_file(config_path) != plan.get("config_sha256"):
        raise Stage0G6WorkerError("G6_WORKER_CONFIG_FILE_HASH_MISMATCH")
    if sha256_file(environment_path) != plan.get("environment_sha256"):
        raise Stage0G6WorkerError("G6_WORKER_ENVIRONMENT_FILE_HASH_MISMATCH")
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(config_path), field="config")
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(load_canonical_json(environment_path), field="environment")
    )
    launcher = config.section("launcher")
    if (
        config.task_id != "stage0.07_ddp_and_gradient_semantics"
        or config.config_hash != plan.get("config_hash")
        or environment.environment_hash != plan.get("environment_hash")
        or not isinstance(launcher, Mapping)
        or launcher.get("backend") != "nccl"
        or launcher.get("world_size") != WORLD_SIZE
    ):
        raise Stage0G6WorkerError("G6_WORKER_CONFIG_OR_ENVIRONMENT_IDENTITY_INVALID")
    return plan, plan_path


def _rank_identity(selected: Sequence[str]) -> dict[str, JSONValue]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as error:
        raise Stage0G6WorkerError("G6_WORKER_TORCHRUN_ENVIRONMENT_INVALID") from error
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != ",".join(selected):
        raise Stage0G6WorkerError("G6_WORKER_CUDA_VISIBLE_DEVICES_MISMATCH")
    if world_size != WORLD_SIZE or not 0 <= rank < WORLD_SIZE or local_rank != rank:
        raise Stage0G6WorkerError("G6_WORKER_RANK_MAPPING_INVALID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise Stage0G6WorkerError("G6_WORKER_CUDA_CARDINALITY_INVALID")
    torch.cuda.set_device(local_rank)
    probe = torch.ones(1, device=torch.device("cuda", local_rank))
    probe.add_(rank)
    torch.cuda.synchronize(local_rank)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if query.returncode != 0:
        raise Stage0G6WorkerError("G6_WORKER_NVIDIA_PROCESS_QUERY_FAILED")
    observed = {
        fields[1]
        for line in query.stdout.splitlines()
        if len(fields := [item.strip() for item in line.split(",")]) == 2
        and fields[0].isdigit()
        and int(fields[0]) == os.getpid()
    }
    if observed != {selected[local_rank]}:
        raise Stage0G6WorkerError("G6_WORKER_PROCESS_GPU_UUID_MISMATCH")
    properties = torch.cuda.get_device_properties(local_rank)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "pid": os.getpid(),
        "gpu_uuid": selected[local_rank],
        "logical_device": local_rank,
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "cuda_visible_devices": visible,
    }


def _init_process_group(timeout_seconds: int) -> None:
    if not dist.is_available() or not dist.is_nccl_available():
        raise Stage0G6WorkerError("G6_WORKER_NCCL_UNAVAILABLE")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )


def _gather_objects(value: object) -> list[object]:
    gathered: list[object] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, value)
    return gathered


def _percentile95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _collective_metrics(protocol: Mapping[str, Any]) -> dict[str, JSONValue]:
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    warmup = int(protocol["warmup_iterations"])
    measured = int(protocol["measured_iterations"])
    samples_per_measurement = int(protocol["samples_per_measurement"])
    sizes = protocol["tensor_elements"]
    if (
        warmup != 20
        or measured != 50
        or samples_per_measurement != 3
        or not isinstance(sizes, list)
        or len(sizes) < 3
    ):
        raise Stage0G6WorkerError("G6_WORKER_COLLECTIVE_PROTOCOL_INVALID")
    device = torch.device("cuda", local_rank)
    message_metrics: list[dict[str, JSONValue]] = []
    for raw_elements in sizes:
        elements = int(raw_elements)
        if elements <= 0:
            raise Stage0G6WorkerError("G6_WORKER_COLLECTIVE_SIZE_INVALID")
        tensor = torch.empty(elements, dtype=torch.float32, device=device)
        for _ in range(warmup):
            tensor.fill_(float(rank + 1))
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(local_rank)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(measured):
            timings: list[float] = []
            for _ in range(samples_per_measurement):
                tensor.fill_(float(rank + 1))
                torch.cuda.synchronize(local_rank)
                start_event.record()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                end_event.record()
                torch.cuda.synchronize(local_rank)
                timings.append(start_event.elapsed_time(end_event) / 1000.0)
            samples.append(statistics.median(timings))
        expected_sum = float(sum(range(1, WORLD_SIZE + 1)))
        if float(tensor[0].item()) != expected_sum or float(tensor[-1].item()) != expected_sum:
            raise Stage0G6WorkerError("G6_WORKER_ALL_REDUCE_RESULT_INVALID")
        median = statistics.median(samples)
        p95 = _percentile95(samples)
        message_metrics.append(
            {
                "tensor_elements": elements,
                "message_bytes": elements * 4,
                "warmup_iterations": warmup,
                "measured_iterations": measured,
                "samples_per_measurement": samples_per_measurement,
                "median_seconds": median,
                "p95_seconds": p95,
                "throughput_bytes_per_second": (elements * 4 * WORLD_SIZE) / median,
                "sample_seconds": samples,
                "result_value": expected_sum,
            }
        )
    broadcast = torch.tensor([2026 + rank], dtype=torch.int64, device=device)
    if rank == 0:
        broadcast.fill_(2026)
    dist.broadcast(broadcast, src=0)
    gathered = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(WORLD_SIZE)]
    source = torch.tensor([rank], dtype=torch.int64, device=device)
    dist.all_gather(gathered, source)
    dist.barrier()
    if int(broadcast.item()) != 2026 or [int(item.item()) for item in gathered] != list(
        range(WORLD_SIZE)
    ):
        raise Stage0G6WorkerError("G6_WORKER_AUXILIARY_COLLECTIVE_INVALID")
    return {
        "message_metrics": message_metrics,
        "broadcast_pass": True,
        "all_gather_pass": True,
        "barrier_pass": True,
        "scalar_expected_sum": 10,
    }


class _SemanticModel(torch.nn.Module):
    """Small dense fixture with a tied parameter and an exact zero gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.linspace(-0.4, 0.5, 12, dtype=torch.float32).reshape(3, 4)
        )
        self.tied_weight = self.weight
        self.bias = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.zero = torch.nn.Parameter(torch.tensor([0.7, -0.8, 0.9]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = inputs @ self.weight.t()
        tied = inputs @ self.tied_weight.t()
        return projected + 0.25 * tied + self.bias + self.zero * 0.0


def _samples(sample_ids: Sequence[int], device: torch.device) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for index in sample_ids:
        inputs = torch.tensor(
            [[
                0.2 + 0.03 * index,
                -0.4 + 0.02 * index,
                0.6 - 0.01 * index,
                0.1 + 0.04 * index,
            ]],
            dtype=torch.float32,
            device=device,
        )
        target = torch.tensor(
            [[0.05 * index, -0.03 * index + 0.2, 0.1 - 0.02 * index]],
            dtype=torch.float32,
            device=device,
        )
        count = 1 + index % 3
        mask = torch.tensor(
            [[1.0 if offset < count else 0.0 for offset in range(3)]],
            dtype=torch.float32,
            device=device,
        )
        values.append(
            {
                "sample_id": f"sample-{index:02d}",
                "inputs": inputs,
                "target": target,
                "mask": mask,
            }
        )
    return values


def _named_tensor_state(
    module: torch.nn.Module, *, gradients: bool = False
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, parameter in module.named_parameters():
        canonical = name.removeprefix("module.")
        tensor = parameter.grad if gradients else parameter.detach()
        if tensor is None:
            raise Stage0G6WorkerError(f"G6_WORKER_GRADIENT_MISSING:{canonical}")
        result[canonical] = tensor.detach().float().cpu().reshape(-1).tolist()
    return result


def _loss_numerator(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    prediction = model(batch["inputs"])
    return (((prediction - batch["target"]) ** 2) * batch["mask"]).sum()


def _serial_step(sample_ids: Sequence[int], device: torch.device) -> dict[str, JSONValue]:
    model = _SemanticModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.075)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    batches = _samples(sample_ids, device)
    denominator = int(sum(float(batch["mask"].sum().item()) for batch in batches))
    numerator = sum((_loss_numerator(model, batch) for batch in batches), start=torch.zeros((), device=device))
    mean_loss = numerator / denominator
    mean_loss.backward()
    gradient = _named_tensor_state(model, gradients=True)
    global_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05).item())
    clipped_gradient = _named_tensor_state(model, gradients=True)
    optimizer.step()
    scheduler.step()
    return {
        "mean_loss": float(mean_loss.detach().item()),
        "effective_count": denominator,
        "gradient": gradient,
        "clipped_gradient": clipped_gradient,
        "parameters": _named_tensor_state(model),
        "global_gradient_norm": global_norm,
        "clip_factor": min(1.0, 0.05 / (global_norm + 1e-6)),
        "event_order": ["gradient_ready", "clip", "optimizer", "scheduler"],
        "shared_parameter_identity": model.weight.data_ptr() == model.tied_weight.data_ptr(),
        "zero_gradient_max_abs": max(abs(item) for item in gradient["zero"]),
    }


def _run_ddp_step(
    sample_ids: Sequence[int],
    device: torch.device,
    *,
    merge_local: bool,
    use_no_sync: bool,
) -> dict[str, JSONValue]:
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

    rank = dist.get_rank()
    batches = _samples([sample_ids[index] for index in range(rank, len(sample_ids), WORLD_SIZE)], device)
    if not batches:
        raise Stage0G6WorkerError("G6_WORKER_EMPTY_RANK_SHARD")
    model = _SemanticModel().to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
    )
    hook_state: dict[str, int] = {"calls": 0}

    def counting_hook(state: dict[str, int], bucket):
        state["calls"] += 1
        return allreduce_hook(None, bucket)  # type: ignore[arg-type]

    ddp.register_comm_hook(hook_state, counting_hook)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.075)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    local_count = int(sum(float(batch["mask"].sum().item()) for batch in batches))
    count_tensor = torch.tensor([local_count], dtype=torch.int64, device=device)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    global_count = int(count_tensor.item())
    optimizer.zero_grad(set_to_none=True)
    local_numerator = torch.zeros((), dtype=torch.float64, device=device)
    event_order: list[str] = []
    work: list[dict[str, Any]]
    if merge_local:
        work = [
            {
                "sample_id": "+".join(str(batch["sample_id"]) for batch in batches),
                "inputs": torch.cat([batch["inputs"] for batch in batches], dim=0),
                "target": torch.cat([batch["target"] for batch in batches], dim=0),
                "mask": torch.cat([batch["mask"] for batch in batches], dim=0),
            }
        ]
    else:
        work = batches
    for index, batch in enumerate(work):
        synchronize = not use_no_sync or index == len(work) - 1
        context = nullcontext() if synchronize else ddp.no_sync()
        with context:
            numerator = _loss_numerator(ddp, batch)
            local_numerator += numerator.detach().double()
            (numerator * (WORLD_SIZE / global_count)).backward()
        event_order.append("sync_backward" if synchronize else "no_sync_forward_backward")
    loss_total = local_numerator.clone()
    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
    event_order.append("gradient_sync_complete")
    gradient = _named_tensor_state(ddp, gradients=True)
    global_norm = float(torch.nn.utils.clip_grad_norm_(ddp.parameters(), 0.05).item())
    event_order.append("clip")
    clipped_gradient = _named_tensor_state(ddp, gradients=True)
    optimizer.step()
    event_order.append("optimizer")
    scheduler.step()
    event_order.append("scheduler")
    result: dict[str, JSONValue] = {
        "rank": rank,
        "sample_ids": [str(batch["sample_id"]) for batch in batches],
        "mean_loss": float((loss_total / global_count).item()),
        "effective_count": global_count,
        "gradient": gradient,
        "clipped_gradient": clipped_gradient,
        "parameters": _named_tensor_state(ddp),
        "global_gradient_norm": global_norm,
        "clip_factor": min(1.0, 0.05 / (global_norm + 1e-6)),
        "gradient_collective_calls": hook_state["calls"],
        "microbatch_count": len(work),
        "event_order": event_order,
        "shared_parameter_identity": model.weight.data_ptr() == model.tied_weight.data_ptr(),
        "zero_gradient_max_abs": max(abs(item) for item in gradient["zero"]),
    }
    del ddp
    return result


def _compare_tensor_states(
    expected: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, JSONValue]:
    if set(expected) != set(observed):
        raise Stage0G6WorkerError("G6_WORKER_TENSOR_STATE_KEYS_MISMATCH")
    max_abs = 0.0
    max_rel = 0.0
    numerator = 0.0
    denominator = 0.0
    worst = ""
    for name in sorted(expected):
        left = torch.tensor(expected[name], dtype=torch.float64)
        right = torch.tensor(observed[name], dtype=torch.float64)
        if left.shape != right.shape or not torch.isfinite(right).all():
            raise Stage0G6WorkerError("G6_WORKER_TENSOR_STATE_INVALID")
        difference = (left - right).abs()
        absolute = float(difference.max().item()) if difference.numel() else 0.0
        scale = torch.maximum(left.abs(), right.abs()).clamp_min(1e-30)
        relative = float((difference / scale).max().item()) if difference.numel() else 0.0
        numerator += float(torch.sum(difference * difference).item())
        denominator += float(torch.sum(left * left).item())
        if absolute >= max_abs:
            max_abs = absolute
            worst = name
        max_rel = max(max_rel, relative)
    return {
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "relative_l2_error": math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30),
        "worst_tensor": worst,
    }


def _variant_comparison(
    serial: Mapping[str, Any], observed: Mapping[str, Any]
) -> dict[str, JSONValue]:
    return {
        "loss_absolute_error": abs(float(serial["mean_loss"]) - float(observed["mean_loss"])),
        "gradient": _compare_tensor_states(
            _mapping(serial["gradient"], field="serial.gradient"),
            _mapping(observed["gradient"], field="observed.gradient"),
        ),
        "clipped_gradient": _compare_tensor_states(
            _mapping(serial["clipped_gradient"], field="serial.clipped_gradient"),
            _mapping(observed["clipped_gradient"], field="observed.clipped_gradient"),
        ),
        "parameters": _compare_tensor_states(
            _mapping(serial["parameters"], field="serial.parameters"),
            _mapping(observed["parameters"], field="observed.parameters"),
        ),
        "effective_count_match": serial["effective_count"] == observed["effective_count"],
        "clip_factor_absolute_error": abs(
            float(serial["clip_factor"]) - float(observed["clip_factor"])
        ),
    }


def _nonfinite_sync(device: torch.device) -> dict[str, JSONValue]:
    rank = dist.get_rank()
    model = _SemanticModel().to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], output_device=device.index
    )
    before = _named_tensor_state(ddp)
    batch = _samples([rank], device)[0]
    loss = _loss_numerator(ddp, batch)
    if rank == 1:
        loss = loss * torch.tensor(float("nan"), device=device)
    loss.backward()
    local_finite = int(
        all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in ddp.parameters()
        )
    )
    global_flag = torch.tensor([local_finite], dtype=torch.int64, device=device)
    dist.all_reduce(global_flag, op=dist.ReduceOp.MIN)
    skipped = int(global_flag.item()) == 0
    if not skipped:
        raise Stage0G6WorkerError("G6_WORKER_NONFINITE_NOT_GLOBALIZED")
    ddp.zero_grad(set_to_none=True)
    after = _named_tensor_state(ddp)
    return {
        "rank": rank,
        "local_finite": bool(local_finite),
        "global_finite": bool(global_flag.item()),
        "skipped": skipped,
        "parameters_unchanged": before == after,
    }


def _bf16_real_training(
    root: Path,
    plan: Mapping[str, Any],
    device: torch.device,
) -> dict[str, JSONValue]:
    from .experiments.task_runners import TrainingTaskRunner, _training_resources
    from .providers.training import TorchModelAdapter

    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, plan["config_ref"], field="config_ref")),
            field="config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, plan["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    request = TaskExecutionRequest(config, config.task_definition, environment)
    rank = dist.get_rank()
    resources = _training_resources(request, root, rank=rank, world_size=WORLD_SIZE)
    ddp = torch.nn.parallel.DistributedDataParallel(
        resources.model.module.to(device),
        device_ids=[device.index],
        output_device=device.index,
    )
    adapter = TorchModelAdapter(ddp, task_type=resources.model.task_type)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result, engine, _, assets, _, profiles = TrainingTaskRunner(root)._run_training(
        request,
        rank=rank,
        world_size=WORLD_SIZE,
        reducer=TorchDistributedReducer(integer_device=device),
        resources=resources,
        wrapped_model=adapter,
    )
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    records = result.records
    finite_parameters = all(bool(torch.isfinite(parameter).all()) for parameter in ddp.parameters())
    value: dict[str, JSONValue] = {
        "rank": rank,
        "global_steps": result.state.global_step,
        "attempts": result.state.attempt_index,
        "record_count": len(records),
        "statuses": [record.status for record in records],
        "mean_losses": [record.mean_loss for record in records],
        "effective_counts": [record.effective_count for record in records],
        "batch_ids": [batch_id for record in records for batch_id in record.batch_ids],
        "checkpoint_count": len(result.checkpoint_ids),
        "finite_parameters": finite_parameters,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "wall_seconds": wall,
        "asset_evidence_count": len(assets),
        "resource_profile_count": len(profiles),
        "coordinate_registry_hash": engine.registry.coordinate_registry_hash,
    }
    del ddp, adapter, engine
    torch.cuda.empty_cache()
    return value


def _semantic_metrics(root: Path, plan: Mapping[str, Any]) -> dict[str, JSONValue]:
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    main_ids = tuple(range(8))
    tail_ids = tuple(range(8, 12))
    full = _run_ddp_step(main_ids, device, merge_local=True, use_no_sync=False)
    accumulated = _run_ddp_step(main_ids, device, merge_local=False, use_no_sync=False)
    no_sync = _run_ddp_step(main_ids, device, merge_local=False, use_no_sync=True)
    tail = _run_ddp_step(tail_ids, device, merge_local=False, use_no_sync=True)
    nonfinite = _nonfinite_sync(device)
    local = {
        "rank": rank,
        "full": full,
        "accumulated": accumulated,
        "no_sync": no_sync,
        "tail": tail,
        "nonfinite": nonfinite,
    }
    gathered = [_mapping(item, field="semantic.rank") for item in _gather_objects(local)]
    dist.barrier()
    bf16_local = _bf16_real_training(root, plan, device)
    bf16 = [_mapping(item, field="bf16.rank") for item in _gather_objects(bf16_local)]
    if rank != 0:
        return {"rank_local_complete": True}
    serial = _serial_step(main_ids, device)
    serial_tail = _serial_step(tail_ids, device)
    canonical = gathered[0]
    main_shards = [
        str(sample_id)
        for item in gathered
        for sample_id in _mapping(item["no_sync"], field="no_sync")["sample_ids"]
    ]
    tail_shards = [
        str(sample_id)
        for item in gathered
        for sample_id in _mapping(item["tail"], field="tail")["sample_ids"]
    ]
    comparisons = {
        "ddp_full": _variant_comparison(serial, _mapping(canonical["full"], field="full")),
        "accumulation_sync_each": _variant_comparison(
            serial, _mapping(canonical["accumulated"], field="accumulated")
        ),
        "accumulation_no_sync": _variant_comparison(
            serial, _mapping(canonical["no_sync"], field="no_sync")
        ),
        "incomplete_tail": _variant_comparison(
            serial_tail, _mapping(canonical["tail"], field="tail")
        ),
    }
    return {
        "data_shards": {
            "expected_main": [f"sample-{index:02d}" for index in main_ids],
            "observed_main": main_shards,
            "expected_tail": [f"sample-{index:02d}" for index in tail_ids],
            "observed_tail": tail_shards,
        },
        "comparisons": comparisons,
        "communication": {
            "sync_each_calls": _mapping(canonical["accumulated"], field="accumulated")[
                "gradient_collective_calls"
            ],
            "no_sync_calls": _mapping(canonical["no_sync"], field="no_sync")[
                "gradient_collective_calls"
            ],
            "microbatches_per_rank": _mapping(
                canonical["no_sync"], field="no_sync"
            )["microbatch_count"],
            "event_order": _mapping(canonical["no_sync"], field="no_sync")[
                "event_order"
            ],
        },
        "clipping": {
            "serial_clip_factor": serial["clip_factor"],
            "serial_global_gradient_norm": serial["global_gradient_norm"],
            "shared_parameter_identity": serial["shared_parameter_identity"],
            "zero_gradient_max_abs": serial["zero_gradient_max_abs"],
        },
        "nonfinite": [item["nonfinite"] for item in gathered],
        "bf16": bf16,
    }


def _write_report(
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    rank_records: Sequence[object],
    metrics: Mapping[str, JSONValue],
    *,
    started_at: str,
) -> None:
    report: dict[str, JSONValue] = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "run_id": str(plan["run_id"]),
        "run_kind": str(plan["run_kind"]),
        "repeat_index": int(plan["repeat_index"]),
        "status": "PASS",
        "started_at": started_at,
        "completed_at": _now(),
        "generator_git_commit": str(plan["generator_git_commit"]),
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "config_ref": str(plan["config_ref"]),
        "config_sha256": str(plan["config_sha256"]),
        "config_hash": str(plan["config_hash"]),
        "environment_ref": str(plan["environment_ref"]),
        "environment_sha256": str(plan["environment_sha256"]),
        "environment_hash": str(plan["environment_hash"]),
        "selected_gpu_uuids": list(plan["selected_gpu_uuids"]),
        "backend": "nccl",
        "world_size": WORLD_SIZE,
        "rank_records": list(rank_records),
        "metrics": dict(metrics),
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_path = _logical_path(root, plan["report_ref"], field="report_ref")
    if report_path.exists():
        raise Stage0G6WorkerError("G6_WORKER_REPORT_COLLISION")
    write_canonical_json(report_path, report)


def run_stage0_g6_worker(*, data_root: str | Path, plan_ref: str) -> None:
    """Execute one rank of one formal G6 launch."""

    root = Path(data_root).resolve(strict=True)
    plan, plan_path = _validate_plan(root, plan_ref)
    selected = tuple(str(item) for item in plan["selected_gpu_uuids"])
    started_at = _now()
    identity = _rank_identity(selected)
    timeout_seconds = int(plan["timeout_seconds"])
    _init_process_group(timeout_seconds)
    try:
        if dist.get_world_size() != WORLD_SIZE or str(dist.get_backend()) != "nccl":
            raise Stage0G6WorkerError("G6_WORKER_PROCESS_GROUP_IDENTITY_INVALID")
        rank_records = _gather_objects(identity)
        run_kind = str(plan["run_kind"])
        if run_kind == "failure_rank":
            failure_flag = torch.tensor(
                [1 if dist.get_rank() == 1 else 0],
                dtype=torch.int64,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            dist.all_reduce(failure_flag, op=dist.ReduceOp.SUM)
            dist.barrier()
            if dist.get_rank() == 1:
                print(FAILURE_MARKER, file=sys.stderr, flush=True)
            if int(failure_flag.item()) != 1:
                raise Stage0G6WorkerError(
                    "G6_WORKER_FAILURE_INJECTION_DID_NOT_TERMINATE"
                )
            raise Stage0G6WorkerError(FAILURE_MARKER)
        if run_kind == "collective":
            metrics = _collective_metrics(
                _mapping(plan["collective_protocol"], field="collective_protocol")
            )
        elif run_kind == "semantic":
            metrics = _semantic_metrics(root, plan)
        elif run_kind == "recovery":
            tensor = torch.tensor(
                [dist.get_rank() + 1],
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            dist.barrier()
            metrics = {
                "scalar_sum": int(tensor.item()),
                "healthy_restart": int(tensor.item()) == 10,
            }
        else:  # pragma: no cover - validated before process-group creation
            raise Stage0G6WorkerError("G6_WORKER_RUN_KIND_INVALID")
        dist.barrier()
        if dist.get_rank() == 0:
            _write_report(
                root,
                plan,
                plan_path,
                rank_records,
                metrics,
                started_at=started_at,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "FAILURE_MARKER",
    "Stage0G6WorkerError",
    "WORKER_PLAN_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "WORLD_SIZE",
    "run_stage0_g6_worker",
]
