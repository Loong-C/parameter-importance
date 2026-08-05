"""Fresh-process worker for formal Stage 0 S0.9 recovery trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any, Final, Mapping, Sequence

import torch
import torch.distributed as dist

_IMPORTANCE_ENABLED: Final = False

from .atomic import sha256_file
from .contracts import ResolvedConfigV2, canonical_json_hash, load_canonical_json, write_canonical_json
from .contracts.jsonio import JSONValue
from .experiments.task_runners import _training_resources
from .providers.training import TorchModelAdapter, configure_batch_cursor
from .runtime import (
    AttemptCommitEvent,
    CheckpointGroupStore,
    CheckpointStore,
    GradientReadyEvent,
    JsonlEventSink,
    LocalReducer,
    ParameterPostEvent,
    SkippedAttemptEvent,
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
    TorchDistributedReducer,
    TrainingEngine,
    TrainingRunSpec,
    TrainingState,
    build_canonical_event_lineage,
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    checkpoint_state_sha256,
    install_training_rng,
    read_event_stream,
)


WORKER_PLAN_SCHEMA = "stage0-g7-recovery-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage0-g7-recovery-worker-report-v1"
BOUNDARY_SCHEMA = "stage0-g7-recovery-boundary-v1"
INTERRUPTION_MARKER = "STAGE0_G7_RECOVERY_INTENTIONAL_INTERRUPT"


class Stage0G7RecoveryWorkerError(RuntimeError):
    """A recovery child violated its hash-bound execution protocol."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G7RecoveryWorkerError(f"G7_RECOVERY_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G7RecoveryWorkerError(f"G7_RECOVERY_WORKER_REF_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G7RecoveryWorkerError(f"G7_RECOVERY_WORKER_REF_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G7RecoveryWorkerError(f"G7_RECOVERY_WORKER_REF_ESCAPE:{field}") from error
    return path


def _validate_plan(root: Path, plan_ref: str) -> tuple[dict[str, Any], Path, TaskExecutionRequest]:
    plan_path = _logical_path(root, plan_ref, field="plan_ref")
    plan = _mapping(load_canonical_json(plan_path), field="plan")
    expected = {
        "schema_version",
        "run_id",
        "trajectory",
        "phase",
        "world_size",
        "selected_gpu_uuids",
        "generator_git_commit",
        "config_ref",
        "config_sha256",
        "config_hash",
        "environment_ref",
        "environment_sha256",
        "environment_hash",
        "execution_root_ref",
        "group_root_ref",
        "result_ref",
        "resume_group_checkpoint_id",
        "total_steps",
        "boundary_step",
        "timeout_seconds",
        "artifact_hash",
    }
    if set(plan) != expected or plan.get("schema_version") != WORKER_PLAN_SCHEMA:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_PLAN_FIELDS_OR_VERSION_INVALID")
    declared = plan.pop("artifact_hash")
    if declared != canonical_json_hash(plan):
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_PLAN_HASH_MISMATCH")
    plan["artifact_hash"] = declared
    if plan.get("trajectory") not in {"baseline", "recovery"}:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_TRAJECTORY_INVALID")
    if plan.get("phase") not in {"baseline", "interrupt", "resume"}:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_PHASE_INVALID")
    world_size = plan.get("world_size")
    selected = plan.get("selected_gpu_uuids")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in {1, 4}
        or not isinstance(selected, list)
        or len(selected) != world_size
        or len(set(selected)) != world_size
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_DEVICE_SET_INVALID")
    if plan["phase"] == "baseline" and plan["trajectory"] != "baseline":
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_BASELINE_IDENTITY_INVALID")
    if plan["phase"] != "baseline" and plan["trajectory"] != "recovery":
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_RECOVERY_IDENTITY_INVALID")
    total = plan.get("total_steps")
    boundary = plan.get("boundary_step")
    if total != 4 or boundary != 2:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_STEP_PROTOCOL_INVALID")
    if plan["phase"] == "resume" and not isinstance(plan["resume_group_checkpoint_id"], str):
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_RESUME_ID_MISSING")
    if plan["phase"] != "resume" and plan["resume_group_checkpoint_id"] is not None:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_UNEXPECTED_RESUME_ID")
    config_path = _logical_path(root, plan["config_ref"], field="config_ref")
    environment_path = _logical_path(root, plan["environment_ref"], field="environment_ref")
    if sha256_file(config_path) != plan["config_sha256"]:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_CONFIG_HASH_MISMATCH")
    if sha256_file(environment_path) != plan["environment_sha256"]:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_ENVIRONMENT_HASH_MISMATCH")
    config = ResolvedConfigV2.from_mapping(_mapping(load_canonical_json(config_path), field="config"))
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(load_canonical_json(environment_path), field="environment")
    )
    distributed = config.base_config.section("distributed")
    training = config.section("training")
    if (
        config.task_id != "stage0.09_checkpoint_and_resume"
        or config.config_hash != plan["config_hash"]
        or environment.environment_hash != plan["environment_hash"]
        or not isinstance(distributed, Mapping)
        or distributed.get("world_size") != world_size
        or not isinstance(training, Mapping)
        or training.get("max_steps") != total
    ):
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_CONFIG_IDENTITY_INVALID")
    return plan, plan_path, TaskExecutionRequest(config, config.task_definition, environment)


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
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_TORCHRUN_ENV_INVALID") from error
    selected = tuple(str(item) for item in plan["selected_gpu_uuids"])
    if (
        observed_world != world_size
        or not 0 <= rank < world_size
        or local_rank != rank
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(selected)
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != world_size
    ):
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_RANK_MAPPING_INVALID")
    torch.cuda.set_device(local_rank)
    probe = torch.ones(1, device=torch.device("cuda", local_rank))
    probe.add_(rank)
    torch.cuda.synchronize(local_rank)
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if query.returncode != 0:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_NVIDIA_QUERY_FAILED")
    observed = {
        fields[1]
        for line in query.stdout.splitlines()
        if len(fields := [item.strip() for item in line.split(",")]) == 2
        and fields[0].isdigit()
        and int(fields[0]) == os.getpid()
    }
    if observed != {selected[local_rank]}:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GPU_UUID_MISMATCH")
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


def _init_distributed(world_size: int, timeout_seconds: int) -> None:
    if world_size == 1:
        return
    if not dist.is_available() or not dist.is_nccl_available():
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_NCCL_UNAVAILABLE")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
    )


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _broadcast_success(world_size: int, success: bool) -> None:
    if world_size == 1:
        if not success:
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GROUP_VALIDATION_FAILED")
        return
    value = torch.tensor([1 if success else 0], dtype=torch.int64, device=torch.device("cuda", torch.cuda.current_device()))
    dist.broadcast(value, src=0)
    if int(value.item()) != 1:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GROUP_VALIDATION_FAILED")


def _gather(world_size: int, value: object) -> list[object]:
    if world_size == 1:
        return [value]
    gathered: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(world_size: int, value: object | None) -> object:
    if world_size == 1:
        if value is None:
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_BROADCAST_VALUE_MISSING")
        return value
    values = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(values, src=0)
    if values[0] is None:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_BROADCAST_VALUE_MISSING")
    return values[0]


@dataclass(slots=True)
class _SampleTrace:
    steps: list[dict[str, JSONValue]] = field(default_factory=list)

    def on_gradient_ready(self, event: GradientReadyEvent) -> None:
        self.steps.append(
            {
                "global_step": event.global_step + 1,
                "attempt_index": event.attempt_index,
                "microbatch_ids": list(event.microbatch_ids),
                "sample_ids": list(event.sample_ids),
            }
        )

    def on_parameter_post(self, event: ParameterPostEvent) -> None:
        return None

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None:
        return None

    def on_skip(self, event: SkippedAttemptEvent) -> None:
        return None


def _build_engine(
    request: TaskExecutionRequest,
    root: Path,
    plan: Mapping[str, Any],
    *,
    rank: int,
    event_sink: JsonlEventSink,
    session_id: str,
) -> tuple[TrainingEngine, _SampleTrace, list[object]]:
    world_size = int(plan["world_size"])
    base = request.config.base_config
    identity = _mapping(base.section("identity"), field="identity")
    training = _mapping(request.config.section("training"), field="training")
    scheduler_options = _mapping(request.config.section("scheduler"), field="scheduler")
    precision_runtime = _mapping(request.config.section("precision_runtime"), field="precision_runtime")
    optimizer_runtime = _mapping(request.config.section("optimizer_runtime"), field="optimizer_runtime")
    checkpoint_schedule = _mapping(request.config.section("checkpoint_schedule"), field="checkpoint_schedule")
    data_loader = _mapping(request.config.section("data_loader"), field="data_loader")
    execution = _mapping(request.config.section("execution"), field="execution")
    base_optimizer = _mapping(base.section("optimizer"), field="optimizer")
    base_importance = _mapping(base.section("importance"), field="importance")
    base_data = _mapping(base.section("data"), field="data")
    base_logging = _mapping(base.section("logging"), field="logging")
    seed_plan = install_training_rng(
        int(identity["master_seed"]), rank=rank, world_size=world_size
    )
    resources = _training_resources(request, root, rank=rank, world_size=world_size)
    device = torch.device("cuda", rank)
    resources.model.module.to(device)
    if world_size > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            resources.model.module,
            device_ids=[rank],
            output_device=rank,
        )
        model = TorchModelAdapter(wrapped, task_type=resources.model.task_type)
        reducer: object = TorchDistributedReducer(integer_device=device)
    else:
        model = resources.model
        reducer = LocalReducer()
    optimizer = build_optimizer(model.module.parameters(), base_optimizer, optimizer_runtime)
    scheduler = build_scheduler(optimizer, scheduler_options)
    scaler = build_grad_scaler(precision_runtime, device_type="cuda")
    spec = TrainingRunSpec(
        run_id=f"{plan['run_id']}-rank-{rank:04d}",
        run_intent="formal",
        max_steps=int(training["max_steps"]),
        max_attempts=int(training["max_steps"]) + int(execution["max_attempts"]) - 1,
        importance_enabled=_IMPORTANCE_ENABLED,
        estimator_name=str(base_importance["estimator_name"]),
        accumulation_dtype=str(base.section("precision")["statistic_dtype"]),  # type: ignore[index]
        max_grad_norm=training["gradient_clip_max_norm"],  # type: ignore[arg-type]
        autocast_dtype=(
            str(precision_runtime["autocast_dtype"])
            if bool(precision_runtime["autocast_enabled"])
            else "none"
        ),
        checkpoint_every_steps=int(checkpoint_schedule["segments"][0]["every_steps"]),  # type: ignore[index]
        log_every_steps=int(base_logging["log_every_steps"]),
        weights_exogenous=bool(base_data["weights_exogenous"]),
        common_mean_assumption=bool(base_data["common_mean_assumption"]),
        metadata={
            "task_id": request.task.task_id,
            "config_hash": request.config.config_hash,
            "seed_plan_hash": seed_plan.artifact_hash,
            "rank": rank,
            "world_size": world_size,
            "data_loader": dict(data_loader),
        },
        checkpoint_segments=tuple(dict(item) for item in checkpoint_schedule["segments"]),  # type: ignore[arg-type]
    )
    execution_root = _logical_path(root, plan["execution_root_ref"], field="execution_root_ref")
    checkpoint_root = execution_root / "checkpoints" / f"rank-{rank:04d}"
    cursor = configure_batch_cursor(
        resources.dataset.cursor(
            seed=seed_plan.seed_for("sampler"), rank=rank, world_size=world_size
        ),
        num_workers=int(data_loader["num_workers"]),
        prefetch_factor=data_loader["prefetch_factor"],  # type: ignore[arg-type]
        persistent_workers=bool(data_loader["persistent_workers"]),
    )
    trace = _SampleTrace()
    engine = TrainingEngine(
        spec=spec,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        reducer=reducer,  # type: ignore[arg-type]
        cursor=cursor,
        checkpoint_store=CheckpointStore(checkpoint_root),
        event_sink=event_sink,
        experiment_id="stage0-g7-recovery",
        attempt_id=("attempt-baseline" if plan["phase"] == "baseline" else "attempt-recovery"),
        session_id=session_id,
        rank=rank,
        observers=(trace,),
    )
    return engine, trace, [cursor, resources.dataset]


def _event_pointer(
    root: Path,
    store: CheckpointStore,
    checkpoint_id: str,
    event_path: Path,
) -> dict[str, JSONValue]:
    state, _ = store.load(checkpoint_id)
    control = _mapping(state["training_state"], field="training_state")
    sequence = int(control["event_sequence"]) - 1
    if sequence < 0:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_CHECKPOINT_EVENT_MISSING")
    return {
        "event_ref": event_path.relative_to(root).as_posix(),
        "event_sha256": sha256_file(event_path),
        "checkpoint_event_sequence": sequence,
    }


def _rank_binding(root: Path, plan: Mapping[str, Any], *, rank: int, checkpoint_id: str, event_path: Path) -> dict[str, JSONValue]:
    execution_root = _logical_path(root, plan["execution_root_ref"], field="execution_root_ref")
    store_path = execution_root / "checkpoints" / f"rank-{rank:04d}"
    return {
        "rank": rank,
        "checkpoint_store_ref": store_path.relative_to(root).as_posix(),
        "checkpoint_id": checkpoint_id,
        "event_pointer": _event_pointer(root, CheckpointStore(store_path), checkpoint_id, event_path),
    }


def _checkpoint_bytes(root: Path, bindings: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for binding in bindings:
        store = _logical_path(root, binding["checkpoint_store_ref"], field="checkpoint_store_ref")
        checkpoint_id = str(binding["checkpoint_id"])
        for path in (store / "objects" / checkpoint_id).rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        total += (store / "commits" / f"{checkpoint_id}.json").stat().st_size
    return total


def _group_metadata(
    request: TaskExecutionRequest,
    plan: Mapping[str, Any],
    *,
    generation: int,
    save_wall_seconds: float,
    checkpoint_bytes: int,
    peak_memory_bytes: int,
) -> dict[str, JSONValue]:
    model = _mapping(request.config.base_config.section("model"), field="model")
    data = _mapping(request.config.base_config.section("data"), field="data")
    identity = _mapping(request.config.base_config.section("identity"), field="identity")
    loader = _mapping(request.config.section("data_loader"), field="data_loader")
    return {
        "config_hash": request.config.config_hash,
        "environment_hash": request.environment.environment_hash,
        "model_manifest_id": str(model["asset_id"]),
        "data_manifest_id": str(data["asset_id"]),
        "sampler_seed": int(identity["master_seed"]),
        "epoch": 0,
        "committed_global_batch": generation,
        "next_global_batch": generation,
        "prefetch_policy": (
            "checkpoint_pending_batches_and_source_cursor"
            if int(loader["num_workers"]) > 0
            else "direct_committed_cursor"
        ),
        "snapshot_type": "optimizer_step_checkpoint",
        "state_extension_schema": "online-importance-state-v1",
        "save_wall_seconds": save_wall_seconds,
        "checkpoint_bytes": checkpoint_bytes,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _publish_group(
    root: Path,
    request: TaskExecutionRequest,
    plan: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    *,
    generation: int,
    parent_checkpoint_id: str | None,
    peak_memory_bytes: int,
) -> tuple[str, float]:
    group = CheckpointGroupStore(root, _logical_path(root, plan["group_root_ref"], field="group_root_ref"))
    checkpoint_id = f"{plan['run_id']}-group-step-{generation:08d}"
    metadata = _group_metadata(
        request,
        plan,
        generation=generation,
        save_wall_seconds=0.0,
        checkpoint_bytes=_checkpoint_bytes(root, bindings),
        peak_memory_bytes=peak_memory_bytes,
    )
    started = time.perf_counter()
    commit = group.publish(
        checkpoint_id,
        generation=generation,
        run_id=str(plan["run_id"]),
        world_size=int(plan["world_size"]),
        rank_checkpoints=bindings,
        metadata=metadata,
        parent_checkpoint_id=parent_checkpoint_id,
    )
    elapsed = time.perf_counter() - started
    # The measured duration is evidence, not commit identity: mutating a published
    # commit to insert it would break atomicity.  It is reported by the worker.
    if commit.checkpoint_id != checkpoint_id:
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GROUP_COMMIT_ID_DRIFT")
    return checkpoint_id, elapsed


def _trace_summary(
    engine: TrainingEngine,
    trace: _SampleTrace,
    event_path: Path,
    *,
    rank: int,
) -> dict[str, JSONValue]:
    events = read_event_stream(event_path)
    optimizer_events = [item for item in events if item.event_type == "optimizer_step"]
    return {
        "rank": rank,
        "global_step": engine.state.global_step,
        "attempt_index": engine.state.attempt_index,
        "sample_trace": list(trace.steps),
        "records": [record.to_dict() for record in engine._records],
        "optimizer_event_steps": [int(item.payload["global_step"]) for item in optimizer_events],
        "learning_rates": [dict(item.payload["learning_rates_post_step"]) for item in optimizer_events],
        "event_ref": event_path.as_posix(),
        "event_sha256": sha256_file(event_path),
        "model_sha256": checkpoint_state_sha256(engine.model.module.state_dict()),
        "optimizer_sha256": checkpoint_state_sha256(engine.optimizer.state_dict()),
        "scheduler_sha256": checkpoint_state_sha256(None if engine.scheduler is None else engine.scheduler.state_dict()),
        "scaler_sha256": checkpoint_state_sha256(None if engine.scaler is None else engine.scaler.state_dict()),
        "cursor_sha256": checkpoint_state_sha256(engine.cursor.state_dict()),
        "rng_checkpointed": True,
    }


def _write_boundary(
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    *,
    group_checkpoint_id: str,
    rank_summaries: Sequence[object],
    tail_refs: Sequence[Mapping[str, JSONValue]],
) -> None:
    value: dict[str, JSONValue] = {
        "schema_version": BOUNDARY_SCHEMA,
        "run_id": str(plan["run_id"]),
        "status": "INTERRUPTED_AFTER_COMMITTED_BOUNDARY",
        "generator_git_commit": str(plan["generator_git_commit"]),
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "config_hash": str(plan["config_hash"]),
        "environment_hash": str(plan["environment_hash"]),
        "world_size": int(plan["world_size"]),
        "boundary_step": int(plan["boundary_step"]),
        "group_checkpoint_id": group_checkpoint_id,
        "rank_summaries": list(rank_summaries),
        "superseded_tail_refs": list(tail_refs),
        "recorded_at": _now(),
    }
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(_logical_path(root, plan["result_ref"], field="result_ref"), value)


def _write_report(
    root: Path,
    plan: Mapping[str, Any],
    plan_path: Path,
    identities: Sequence[object],
    *,
    group_checkpoint_id: str,
    rank_summaries: Sequence[object],
    group_save_seconds: float,
    group_load_seconds: float,
    lineage_ref: str | None,
    canonical_event_ref: str | None,
) -> dict[str, JSONValue]:
    report: dict[str, JSONValue] = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "run_id": str(plan["run_id"]),
        "trajectory": str(plan["trajectory"]),
        "phase": str(plan["phase"]),
        "status": "PASS",
        "completed_at": _now(),
        "generator_git_commit": str(plan["generator_git_commit"]),
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "config_hash": str(plan["config_hash"]),
        "environment_hash": str(plan["environment_hash"]),
        "world_size": int(plan["world_size"]),
        "selected_gpu_uuids": list(plan["selected_gpu_uuids"]),
        "rank_identities": list(identities),
        "total_steps": int(plan["total_steps"]),
        "boundary_step": int(plan["boundary_step"]),
        "group_checkpoint_id": group_checkpoint_id,
        "group_root_ref": str(plan["group_root_ref"]),
        "rank_summaries": list(rank_summaries),
        "group_save_seconds": group_save_seconds,
        "group_load_seconds": group_load_seconds,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(torch.cuda.current_device())),
        "lineage_ref": lineage_ref,
        "canonical_event_ref": canonical_event_ref,
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_path = _logical_path(root, plan["result_ref"], field="result_ref")
    if report_path.exists():
        raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_REPORT_COLLISION")
    write_canonical_json(report_path, report)
    return report


def run_stage0_g7_recovery_worker(*, data_root: str | Path, plan_ref: str) -> dict[str, JSONValue] | None:
    root = Path(data_root).resolve(strict=True)
    plan, plan_path, request = _validate_plan(root, plan_ref)
    identity = _rank_identity(plan)
    rank = int(identity["rank"])
    world_size = int(plan["world_size"])
    _init_distributed(world_size, int(plan["timeout_seconds"]))
    cleanup: list[object] = []
    try:
        if world_size > 1 and (dist.get_world_size() != world_size or str(dist.get_backend()) != "nccl"):
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_PROCESS_GROUP_INVALID")
        identities = _gather(world_size, identity)
        torch.cuda.reset_peak_memory_stats(rank)
        phase = str(plan["phase"])
        session = f"session-{phase}-rank-{rank:04d}"
        execution_root = _logical_path(root, plan["execution_root_ref"], field="execution_root_ref")
        event_path = execution_root / "events" / f"rank-{rank:04d}-{session}.jsonl"
        with JsonlEventSink(event_path) as sink:
            engine, trace, cleanup = _build_engine(
                request, root, plan, rank=rank, event_sink=sink, session_id=session
            )
            group_load_seconds = 0.0
            if phase == "resume":
                _barrier(world_size)
                success = True
                validated_bindings: list[Mapping[str, Any]] | None = None
                load_started = time.perf_counter()
                if rank == 0:
                    try:
                        group = CheckpointGroupStore(
                            root, _logical_path(root, plan["group_root_ref"], field="group_root_ref")
                        )
                        commit = group.verify(
                            str(plan["resume_group_checkpoint_id"]),
                            expected_run_id=str(plan["run_id"]),
                            expected_world_size=world_size,
                            expected_config_hash=request.config.config_hash,
                            expected_data_manifest_id=str(request.config.base_config.section("data")["asset_id"]),  # type: ignore[index]
                        )
                        if len(commit.rank_checkpoints) != world_size or commit.generation != int(plan["boundary_step"]):
                            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GROUP_RESUME_BOUNDARY_INVALID")
                        validated_bindings = [dict(item) for item in commit.rank_checkpoints]
                    except Exception:
                        success = False
                group_load_seconds = time.perf_counter() - load_started
                _broadcast_success(world_size, success)
                received = _broadcast_object(world_size, validated_bindings)
                if not isinstance(received, list) or len(received) != world_size:
                    raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_VALIDATED_BINDINGS_INVALID")
                binding = _mapping(received[rank], field="validated_rank_binding")
                engine.resume_checkpoint(str(binding["checkpoint_id"]))
                engine.state = TrainingState(
                    engine.state.global_step,
                    engine.state.attempt_index,
                    engine.state.skipped_steps,
                    0,
                    engine.state.last_checkpoint_id,
                )
            result = engine.run(
                until_step=(int(plan["boundary_step"]) if phase == "interrupt" else None)
            )
        if phase == "interrupt" and result.status != "PAUSED":
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_BOUNDARY_NOT_PAUSED")
        if phase != "interrupt" and result.status != "COMPLETE":
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_TRAJECTORY_NOT_COMPLETE")
        checkpoint_id = str(engine.state.last_checkpoint_id)
        local_binding = _rank_binding(root, plan, rank=rank, checkpoint_id=checkpoint_id, event_path=event_path)
        bindings = [_mapping(item, field="rank_binding") for item in _gather(world_size, local_binding)]
        _barrier(world_size)
        group_checkpoint_id = ""
        group_save_seconds = 0.0
        if rank == 0:
            parent = str(plan["resume_group_checkpoint_id"]) if phase == "resume" else None
            group_checkpoint_id, group_save_seconds = _publish_group(
                root,
                request,
                plan,
                bindings,
                generation=engine.state.global_step,
                parent_checkpoint_id=parent,
                peak_memory_bytes=int(torch.cuda.max_memory_allocated(rank)),
            )
        gathered_group = _gather(world_size, {"rank": rank, "id": group_checkpoint_id, "save_seconds": group_save_seconds})
        owner = _mapping(gathered_group[0], field="group_owner")
        group_checkpoint_id = str(owner["id"])
        group_save_seconds = float(owner["save_seconds"])
        if not group_checkpoint_id:
            raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_GROUP_COMMIT_MISSING")
        local_summary = _trace_summary(engine, trace, event_path, rank=rank)
        local_summary["event_ref"] = event_path.relative_to(root).as_posix()
        rank_summaries = _gather(world_size, local_summary)
        if phase == "interrupt":
            boundary_summaries = rank_summaries
            tail_session = f"session-orphan-tail-rank-{rank:04d}"
            tail_path = execution_root / "events" / f"rank-{rank:04d}-{tail_session}.jsonl"
            engine.checkpoint_store = None
            engine.state = TrainingState(
                engine.state.global_step,
                engine.state.attempt_index,
                engine.state.skipped_steps,
                0,
                engine.state.last_checkpoint_id,
            )
            engine.session_id = tail_session
            tail_trace = _SampleTrace()
            engine.register_observer(tail_trace)
            with JsonlEventSink(tail_path) as tail_sink:
                engine.event_sink = tail_sink
                tail_result = engine.run(until_step=int(plan["boundary_step"]) + 1)
            if tail_result.status != "PAUSED":
                raise Stage0G7RecoveryWorkerError("G7_RECOVERY_WORKER_ORPHAN_TAIL_NOT_WRITTEN")
            tail_ref: dict[str, JSONValue] = {
                "rank": rank,
                "attempt_id": engine.attempt_id,
                "session_id": tail_session,
                "event_ref": tail_path.relative_to(root).as_posix(),
                "event_sha256": sha256_file(tail_path),
                "optimizer_steps": [
                    int(item.payload["global_step"])
                    for item in read_event_stream(tail_path)
                    if item.event_type == "optimizer_step"
                ],
                "sample_trace": list(tail_trace.steps),
            }
            tail_refs = [_mapping(item, field="tail_ref") for item in _gather(world_size, tail_ref)]
            _barrier(world_size)
            if rank == 0:
                _write_boundary(
                    root,
                    plan,
                    plan_path,
                    group_checkpoint_id=group_checkpoint_id,
                    rank_summaries=boundary_summaries,
                    tail_refs=tail_refs,
                )
            _barrier(world_size)
            print(INTERRUPTION_MARKER, file=sys.stderr, flush=True)
            raise Stage0G7RecoveryWorkerError(INTERRUPTION_MARKER)
        lineage_ref: str | None = None
        canonical_event_ref: str | None = None
        if phase == "resume" and rank == 0:
            boundary = _mapping(
                load_canonical_json(_logical_path(root, str(plan["result_ref"]).replace("resume-report.json", "boundary.json"), field="boundary_ref")),
                field="boundary",
            )
            old = _mapping(boundary["rank_summaries"][0], field="boundary.rank0")
            tail = _mapping(boundary["superseded_tail_refs"][0], field="boundary.tail0")
            old_events = read_event_stream(_logical_path(root, old["event_ref"], field="old_event_ref"))
            resume_events = read_event_stream(event_path)
            tail_events = read_event_stream(_logical_path(root, tail["event_ref"], field="tail_event_ref"))
            lineage_ref = f"{plan['execution_root_ref']}/canonical/lineage.json"
            canonical_event_ref = f"{plan['execution_root_ref']}/canonical/events.jsonl"
            build_canonical_event_lineage(
                root=root,
                run_id=engine.spec.run_id,
                segments=(
                    {
                        "attempt_id": old_events[0].attempt_id,
                        "session_id": old_events[0].session_id,
                        "rank": 0,
                        "event_ref": old["event_ref"],
                        "event_sha256": old["event_sha256"],
                        "sequence_start": 0,
                        "sequence_end": len(old_events) - 1,
                        "checkpoint_ref": str(plan["resume_group_checkpoint_id"]),
                    },
                    {
                        "attempt_id": resume_events[0].attempt_id,
                        "session_id": resume_events[0].session_id,
                        "rank": 0,
                        "event_ref": event_path.relative_to(root).as_posix(),
                        "event_sha256": sha256_file(event_path),
                        "sequence_start": 0,
                        "sequence_end": len(resume_events) - 1,
                        "checkpoint_ref": group_checkpoint_id,
                    },
                ),
                superseded_tails=(
                    {
                        "attempt_id": tail_events[0].attempt_id,
                        "session_id": tail_events[0].session_id,
                        "event_ref": tail["event_ref"],
                        "event_sha256": tail["event_sha256"],
                        "sequence_start": 0,
                        "sequence_end": len(tail_events) - 1,
                        "reason": "events after the selected checkpoint were superseded by fresh-process resume",
                    },
                ),
                parent_checkpoint_ref=str(plan["resume_group_checkpoint_id"]),
                output_ref=lineage_ref,
                canonical_event_ref=canonical_event_ref,
            )
        lineage_values = _gather(
            world_size,
            {"rank": rank, "lineage_ref": lineage_ref, "canonical_event_ref": canonical_event_ref},
        )
        owner_lineage = _mapping(lineage_values[0], field="lineage_owner")
        lineage_ref = owner_lineage["lineage_ref"]  # type: ignore[assignment]
        canonical_event_ref = owner_lineage["canonical_event_ref"]  # type: ignore[assignment]
        _barrier(world_size)
        report = None
        if rank == 0:
            report = _write_report(
                root,
                plan,
                plan_path,
                identities,
                group_checkpoint_id=group_checkpoint_id,
                rank_summaries=rank_summaries,
                group_save_seconds=group_save_seconds,
                group_load_seconds=group_load_seconds,
                lineage_ref=lineage_ref,
                canonical_event_ref=canonical_event_ref,
            )
        _barrier(world_size)
        return report
    finally:
        for candidate in cleanup:
            close = getattr(candidate, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = [
    "BOUNDARY_SCHEMA",
    "INTERRUPTION_MARKER",
    "Stage0G7RecoveryWorkerError",
    "WORKER_PLAN_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "run_stage0_g7_recovery_worker",
]
