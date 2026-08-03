"""Formal Stage 0 S0.10 capacity and operations qualification.

The orchestrator is deliberately evidence-first.  It freezes exact real-model
shapes, candidate configs, synthetic parameter-state lifetimes and the allowed
compute/communication work before any CUDA process is created.  Child workers
then execute those immutable controls in fresh processes.  No importance
estimator mathematics is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .atomic import sha256_file
from .capacity import (
    GIB,
    ParameterTensorShape,
    build_compute_communication_envelope,
    build_parameter_state_envelope,
    estimate_fixed_model_budget,
)
from .contracts import (
    GateRecord,
    GateStatus,
    ResolvedConfigV2,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .experiments import build_default_task_runtime
from .experiments.task_runners import _training_resources
from .g3_runtime_assets import FormalG3RuntimeAssets
from .runtime import (
    FailureClass,
    GpuLeaseIdentity,
    LaunchClaimRegistry,
    ProjectGpuLease,
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    classify_stale_heartbeat,
    evaluate_launch_preflight,
    exercise_canary_writer,
    failure_response,
    load_committed_task_artifact,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g7_recovery import (
    Stage0G7RecoveryFormalState,
    load_stage0_g7_recovery_formal_state,
)
from .stage0_g8_worker import WORKER_PLAN_SCHEMA, WORKER_REPORT_SCHEMA
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID = "stage0.10_capacity_and_operations"
GATE_IDS = ("stage0.G8-C", "stage0.G8-S4", "stage0.G8-S5", "stage0.G8")
_OUTPUT_KINDS = {"capacity_envelope", "operations_preflight", "fault_report"}
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_MODEL_SPECS: tuple[tuple[str, str, int], ...] = (
    ("pythia-14m-step0", "g8-c-14m", 0),
    ("pythia-31m-deduped-step0", "budget-31m", 0),
    ("pythia-160m-deduped-step0", "g8-s4-160m", 4),
    ("pythia-410m-deduped-step0", "g8-s5-410m", 5),
)
_CRITICAL_SOURCE_REFS = (
    "ops/stage0/formalize_g8.py",
    "ops/stage0/run_g8_capacity_worker.py",
    "src/param_importance_nlp/capacity.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/experiments/task_runners.py",
    "src/param_importance_nlp/runtime/operations.py",
    "src/param_importance_nlp/stage0_g8.py",
    "src/param_importance_nlp/stage0_g8_worker.py",
    "schemas/stage0-g8-capacity-evidence-v1.json",
    "schemas/stage0-g8-formalization-index-v1.json",
    "schemas/stage0-g8-parameter-envelope-v1.json",
    "schemas/stage0-g8-work-envelope-v1.json",
    "schemas/stage0-g8-worker-plan-v1.json",
    "schemas/stage0-g8-worker-report-v1.json",
)


class Stage0G8Error(RuntimeError):
    """S0.10 capacity, operations, or replay evidence failed closed."""


@dataclass(frozen=True, slots=True)
class G8SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G8FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G8FormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    g7_recovery_index_ref: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _capture_source() -> G8SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    probes = {
        "top": _git(repository, "rev-parse", "--show-toplevel"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "tracked": _git(repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS),
        "status": _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if any(item.returncode != 0 for item in probes.values()):
        raise Stage0G8Error("G8_SOURCE_GIT_PROBE_FAILED")
    if Path(probes["top"].stdout.strip()).resolve() != repository:
        raise Stage0G8Error("G8_SOURCE_GIT_ROOT_MISMATCH")
    commit = probes["head"].stdout.strip()
    branch = probes["branch"].stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch:
        raise Stage0G8Error("G8_SOURCE_IDENTITY_INVALID")
    if probes["status"].stdout.strip():
        raise Stage0G8Error("G8_FORMAL_SOURCE_DIRTY")
    return G8SourceBinding(repository, commit, branch)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G8Error(f"G8_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G8Error(f"G8_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G8Error(f"G8_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G8Error(f"G8_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G8Error("G8_CONTROL_FILE_DRIFT")
        return
    write_canonical_json(path, dict(value))


def _selected_gpu_uuids(state: Stage0G7RecoveryFormalState, root: Path) -> tuple[str, ...]:
    reference = state.environment.evidence_refs.get("g4_provenance")
    if reference is None:
        raise Stage0G8Error("G8_G4_PROVENANCE_REF_MISSING")
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    envelope = _mapping(loaded.payload, field="g4_provenance_envelope")
    provenance = _mapping(envelope.get("provenance_record"), field="g4_provenance")
    selected = provenance.get("device_mapping")
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or len(set(selected)) != 4
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G8Error("G8_G4_DEVICE_MAPPING_INVALID")
    return tuple(selected)


def _config_overrides(
    template: ResolvedConfigV2,
    *,
    model_id: str,
    profile_id: str,
    candidate_stage: int,
    world_size: int,
    precision_profile: str,
) -> tuple[dict[str, object], dict[str, object]]:
    base = template.base_config
    runtime = _mapping(base.section("runtime"), field="runtime")
    model = _mapping(base.section("model"), field="model")
    data = _mapping(base.section("data"), field="data")
    loss = _mapping(base.section("loss"), field="loss")
    optimizer = _mapping(base.section("optimizer"), field="optimizer")
    precision = _mapping(base.section("precision"), field="precision")
    checkpoint = _mapping(base.section("checkpoint"), field="checkpoint")
    if profile_id == "g8-c-14m":
        per_device = 4 if world_size == 1 else 1
        microbatch = per_device
        accumulation = 1
        global_batch = 4
        checkpoint_every = 40
    else:
        per_device = 4
        microbatch = 4
        accumulation = 4
        global_batch = 64
        checkpoint_every = 25 if candidate_stage == 4 else 40
    compute_dtype = "float32" if precision_profile == "fp32" else "bfloat16"
    base_overrides: dict[str, object] = {
        "identity": {"route": f"stage0-g8-{profile_id}-{precision_profile}-w{world_size}"},
        "runtime": {**runtime, "device": "cuda", "allow_dirty_worktree": False},
        "model": {**model, "asset_id": model_id},
        "data": {
            **data,
            "split": "debug" if candidate_stage == 0 else "train",
            "sequence_length": 2048,
            "sampling_design": "without_replacement_frozen_epoch",
        },
        "loss": loss,
        "batching": {
            "global_batch_size": global_batch,
            "per_device_batch_size": per_device,
            "microbatch_size": microbatch,
            "accumulation_steps": accumulation,
            "no_sync": world_size > 1 and accumulation > 1,
        },
        "distributed": {
            "world_size": world_size,
            "backend": "nccl" if world_size > 1 else "local",
            "device_ids": list(range(world_size)),
            "timeout_seconds": 300,
        },
        "precision": {
            **precision,
            "compute_dtype": compute_dtype,
            "gradient_dtype": "float32",
            "statistic_dtype": "float32",
            "amp": precision_profile == "bf16",
        },
        "optimizer": {
            **optimizer,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "fused": False,
            "foreach": False,
        },
        "checkpoint": {**checkpoint, "save_every_steps": checkpoint_every, "max_to_keep": 1},
    }
    v2_overrides: dict[str, object] = {
        "execution": {"timeout_seconds": 14400, "max_attempts": 1},
        "training": {
            "max_steps": 100,
            "deterministic_algorithms": True,
            "gradient_clip_max_norm": 1.0,
        },
        "data_loader": {
            "num_workers": 2,
            "prefetch_factor": 2,
            "persistent_workers": True,
            "drop_last": False,
            "cursor_policy": "checkpoint_commit",
        },
        "evaluation": {
            "enabled": False,
            "split": None,
            "every_steps": None,
            "batch_size": None,
            "max_batches": None,
            "metrics": [],
            "save_predictions": False,
        },
        "profiling": {
            "enabled": True,
            "warmup_steps": 10,
            "measure_steps": 30,
            "repetitions": 3,
            "capture_memory": True,
            "capture_throughput": True,
            "capture_communication": True,
            "synchronize_device": True,
        },
        "checkpoint_schedule": {
            "segments": [{"start_step": 0, "end_step": None, "every_steps": checkpoint_every}],
            "save_on_phase_end": True,
            "save_optimizer": True,
            "save_rng": True,
            "save_data_state": True,
        },
        "precision_runtime": {
            "autocast_enabled": precision_profile == "bf16",
            "autocast_dtype": compute_dtype,
            "grad_scaler_enabled": False,
            "initial_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "global_found_inf_reduce": True,
        },
        "launcher": {
            "kind": "torchrun" if world_size > 1 else "local",
            "backend": "nccl" if world_size > 1 else "local",
            "world_size": world_size,
            "init_method": "env" if world_size > 1 else "local",
            "init_ref": None,
            "rendezvous_id": (
                f"stage0-g8-{profile_id}-{precision_profile}-w{world_size}"
                if world_size > 1
                else None
            ),
            "max_restarts": 0,
        },
        "recovery": {
            "mode": "restart_idempotent",
            "resume_ref": None,
            "max_restarts": 0,
            "safe_boundary": "immutable_publish",
        },
        "scheduler": template.section("scheduler"),
        "providers": template.section("providers"),
        "optimizer_runtime": template.section("optimizer_runtime"),
    }
    return base_overrides, v2_overrides


def _capacity_config(
    *,
    repository: Path,
    input_refs: Sequence[str],
    template: ResolvedConfigV2,
    model_id: str,
    profile_id: str,
    candidate_stage: int,
    world_size: int,
    precision_profile: str,
    output_dir: str,
) -> ResolvedConfigV2:
    base_overrides, v2_overrides = _config_overrides(
        template,
        model_id=model_id,
        profile_id=profile_id,
        candidate_stage=candidate_stage,
        world_size=world_size,
        precision_profile=precision_profile,
    )
    return build_stage0_formal_config(
        repository,
        task_id=TASK_ID,
        input_refs=tuple(input_refs),
        output_dir=output_dir,
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )


def build_stage0_g8_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G7RecoveryFormalState,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    _selected_gpu_uuids(state, root)
    return _capacity_config(
        repository=binding.repository,
        input_refs=tuple(state.task_output_refs.values()),
        template=state.config,
        model_id="pythia-410m-deduped-step0",
        profile_id="g8-s5-410m",
        candidate_stage=5,
        world_size=4,
        precision_profile="bf16",
        output_dir=f"evidence/stage0/tasks/10-{state.gate_artifact_hash}",
    )


def _exact_model_shapes(
    request: TaskExecutionRequest,
    root: Path,
    *,
    candidate_stage: int,
) -> tuple[tuple[ParameterTensorShape, ...], dict[str, int], str]:
    resources = _training_resources(
        request,
        root,
        rank=0,
        world_size=int(request.config.base_config.section("distributed")["world_size"]),
        data_route_stage=candidate_stage,
    )
    module = resources.model.module
    shapes = tuple(
        ParameterTensorShape(
            name=name,
            shape=tuple(int(item) for item in parameter.shape),
            dtype=str(parameter.dtype).removeprefix("torch."),
            requires_grad=bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters()
    )
    config = getattr(module, "config", None)
    architecture = {
        "hidden_size": int(getattr(config, "hidden_size")),
        "layers": int(getattr(config, "num_hidden_layers")),
    }
    assets = FormalG3RuntimeAssets.from_request(request, root)
    model_id = str(request.config.base_config.section("model")["asset_id"])
    model_asset = assets.resolve(model_id, expected_kind="model")
    manifest_id = model_asset.ready_manifest_sha256
    close = getattr(resources.dataset, "close", None)
    if callable(close):
        close()
    del module, resources
    gc.collect()
    return shapes, architecture, manifest_id


def _build_controls(
    request: TaskExecutionRequest,
    root: Path,
    source: G8SourceBinding,
    suite_ref: str,
) -> dict[str, Any]:
    input_refs = tuple(
        str(item)
        for item in _mapping(request.config.section("orchestration"), field="orchestration")[
            "input_result_refs"
        ]
    )
    template = request.config
    config_specs: list[tuple[str, str, str, int, int, str]] = [
        ("g8-c-14m-fp32-w1", "pythia-14m-step0", "g8-c-14m", 0, 1, "fp32"),
        ("g8-c-14m-fp32-w4", "pythia-14m-step0", "g8-c-14m", 0, 4, "fp32"),
        ("g8-c-14m-bf16-w1", "pythia-14m-step0", "g8-c-14m", 0, 1, "bf16"),
        ("g8-c-14m-bf16-w4", "pythia-14m-step0", "g8-c-14m", 0, 4, "bf16"),
        ("budget-31m-bf16-w4", "pythia-31m-deduped-step0", "budget-31m", 0, 4, "bf16"),
        ("g8-s4-160m-bf16-w4", "pythia-160m-deduped-step0", "g8-s4-160m", 4, 4, "bf16"),
        ("g8-s5-410m-bf16-w4", "pythia-410m-deduped-step0", "g8-s5-410m", 5, 4, "bf16"),
    ]
    configs: dict[str, ResolvedConfigV2] = {}
    config_refs: dict[str, str] = {}
    for key, model_id, profile_id, stage, world_size, precision in config_specs:
        config = _capacity_config(
            repository=source.repository,
            input_refs=input_refs,
            template=template,
            model_id=model_id,
            profile_id=profile_id,
            candidate_stage=stage,
            world_size=world_size,
            precision_profile=precision,
            output_dir=f"{suite_ref}/task-output/{key}",
        )
        reference = f"{suite_ref}/controls/config-{key}.json"
        _write_or_verify(_logical_path(root, reference, field="config_ref"), config.to_dict())
        configs[key] = config
        config_refs[key] = reference

    # Probe each scale once through the same qualified offline provider used by
    # the worker.  The 31M model is budgeted but intentionally not substituted
    # for either scale-gate measurement.
    probe_keys = {
        "pythia-14m-step0": "g8-c-14m-bf16-w4",
        "pythia-31m-deduped-step0": "budget-31m-bf16-w4",
        "pythia-160m-deduped-step0": "g8-s4-160m-bf16-w4",
        "pythia-410m-deduped-step0": "g8-s5-410m-bf16-w4",
    }
    shapes_by_model: dict[str, tuple[ParameterTensorShape, ...]] = {}
    architecture_by_model: dict[str, dict[str, int]] = {}
    manifest_by_model: dict[str, str] = {}
    for model_id, key in probe_keys.items():
        spec = next(item for item in config_specs if item[0] == key)
        child_request = TaskExecutionRequest(configs[key], configs[key].task_definition, request.environment)
        shapes, architecture, manifest_id = _exact_model_shapes(
            child_request, root, candidate_stage=spec[3]
        )
        shapes_by_model[model_id] = shapes
        architecture_by_model[model_id] = architecture
        manifest_by_model[model_id] = manifest_id

    budgets: list[dict[str, JSONValue]] = []
    for model_id, profile_id, candidate_stage in _MODEL_SPECS:
        key = probe_keys[model_id]
        config = configs[key]
        batching = _mapping(config.base_config.section("batching"), field="batching")
        data = _mapping(config.base_config.section("data"), field="data")
        precision = _mapping(config.base_config.section("precision"), field="precision")
        checkpoint = _mapping(config.base_config.section("checkpoint"), field="checkpoint")
        architecture = architecture_by_model[model_id]
        budget = estimate_fixed_model_budget(
            model_id=model_id,
            tensors=shapes_by_model[model_id],
            hidden_size=architecture["hidden_size"],
            layers=architecture["layers"],
            sequence_length=int(data["sequence_length"]),
            microbatch_size=int(batching["microbatch_size"]),
            compute_dtype=str(precision["compute_dtype"]),
            retained_checkpoints=1,
            checkpoint_every_steps=int(checkpoint["save_every_steps"]),
            seed_count=3,
            parallel_runs=4 if candidate_stage in {4, 5} else 1,
            logs_and_reports_per_run=GIB,
        )
        budget.update(
            {
                "profile_id": profile_id,
                "candidate_stage": candidate_stage,
                "config_ref": config_refs[key],
                "config_hash": config.config_hash,
                "model_manifest_id": manifest_by_model[model_id],
            }
        )
        declared = budget.pop("artifact_hash")
        del declared
        budget["artifact_hash"] = canonical_json_hash(budget)
        budgets.append(budget)  # type: ignore[arg-type]
    if len({item["tensor_shape_hash"] for item in budgets}) != 4 or any(
        item.get("derived_from_other_model_scale") is not False for item in budgets
    ):
        raise Stage0G8Error("G8_PER_MODEL_BUDGET_INDEPENDENCE_INVALID")
    budget_report = _with_hash(
        {
            "schema_version": "stage0-g8-per-model-budget-report-v1",
            "generator_git_commit": source.git_commit,
            "status": "PASS",
            "models": budgets,
            "scale_extrapolation_used": False,
        }
    )
    budget_ref = f"{suite_ref}/controls/per-model-budgets.json"
    _write_or_verify(_logical_path(root, budget_ref, field="budget_ref"), budget_report)

    parameter_refs: dict[str, str] = {}
    parameter_values: dict[str, dict[str, Any]] = {}
    work_refs: dict[str, str] = {}
    work_values: dict[str, dict[str, Any]] = {}
    for key, model_id, profile_id, stage, world_size, precision_profile in config_specs:
        config = configs[key]
        checkpoint = _mapping(config.base_config.section("checkpoint"), field="checkpoint")
        batching = _mapping(config.base_config.section("batching"), field="batching")
        data = _mapping(config.base_config.section("data"), field="data")
        precision = _mapping(config.base_config.section("precision"), field="precision")
        parameter = build_parameter_state_envelope(
            model_id=model_id,
            tensors=shapes_by_model[model_id],
            config_hash=config.config_hash,
            model_manifest_id=manifest_by_model[model_id],
            checkpoint_every_steps=int(checkpoint["save_every_steps"]),
        )
        microbatches = int(batching["accumulation_steps"]) * (
            int(batching["per_device_batch_size"]) // int(batching["microbatch_size"])
        )
        work = build_compute_communication_envelope(
            model_id=model_id,
            parameter_count=int(parameter["trainable_parameter_count"]),
            world_size=world_size,
            microbatches_per_optimizer_step=microbatches,
            checkpoint_every_steps=int(checkpoint["save_every_steps"]),
            collective_chunk_bytes=64 * 1024**2,
            compute_dtype=str(precision["compute_dtype"]),
            sequence_length=int(data["sequence_length"]),
            microbatch_size=int(batching["microbatch_size"]),
            candidate_stage=stage,
        )
        parameter_ref = f"{suite_ref}/controls/parameter-envelope-{key}.json"
        work_ref = f"{suite_ref}/controls/work-envelope-{key}.json"
        _write_or_verify(_logical_path(root, parameter_ref, field="parameter_ref"), parameter)
        _write_or_verify(_logical_path(root, work_ref, field="work_ref"), work)
        parameter_refs[key] = parameter_ref
        parameter_values[key] = parameter
        work_refs[key] = work_ref
        work_values[key] = work

    environment_ref = f"{suite_ref}/controls/environment.json"
    _write_or_verify(
        _logical_path(root, environment_ref, field="environment_ref"), request.environment.to_dict()
    )
    return {
        "configs": configs,
        "config_refs": config_refs,
        "parameter_refs": parameter_refs,
        "parameter_values": parameter_values,
        "work_refs": work_refs,
        "work_values": work_values,
        "budget_ref": budget_ref,
        "budget_report": budget_report,
        "environment_ref": environment_ref,
        "execution_keys": tuple(key for key, *_ in config_specs if not key.startswith("budget-")),
        "specs": {item[0]: item for item in config_specs},
    }


def _nvidia_query(fields: str) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise Stage0G8Error("G8_NVIDIA_QUERY_FAILED")
    return [[item.strip() for item in line.split(",")] for line in result.stdout.splitlines() if line.strip()]


def _gpu_snapshot(selected: Sequence[str]) -> dict[str, JSONValue]:
    rows = _nvidia_query(
        "uuid,pci.bus_id,memory.total,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total"
    )
    indexed: dict[str, dict[str, JSONValue]] = {}
    for row in rows:
        if len(row) != 7:
            raise Stage0G8Error("G8_NVIDIA_HEALTH_FIELDS_INVALID")
        uuid = row[0]
        indexed[uuid] = {
            "gpu_uuid": uuid,
            "pci_bus_id": row[1],
            "memory_total_mib": int(row[2]),
            "memory_used_mib": int(row[3]),
            "temperature_c": int(row[4]),
            "volatile_uncorrected_ecc": None if row[5] in {"N/A", "[N/A]"} else int(row[5]),
            "aggregate_uncorrected_ecc": None if row[6] in {"N/A", "[N/A]"} else int(row[6]),
        }
    if set(selected) - set(indexed):
        raise Stage0G8Error("G8_SELECTED_GPU_MISSING")
    chosen = [indexed[item] for item in selected]
    health_ok = all(
        int(item["memory_used_mib"]) == 0
        and int(item["temperature_c"]) < 85
        and item["volatile_uncorrected_ecc"] in {0, None}
        and item["aggregate_uncorrected_ecc"] in {0, None}
        for item in chosen
    )
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=False, capture_output=True, text=True, timeout=30
    )
    if topology.returncode != 0:
        raise Stage0G8Error("G8_NVIDIA_TOPOLOGY_QUERY_FAILED")
    return {
        "selected": chosen,
        "health_ok": health_ok,
        "topology_sha256": hashlib.sha256(topology.stdout.encode("utf-8")).hexdigest(),
        "topology_text": topology.stdout,
    }


def _compute_processes(selected: Sequence[str]) -> list[dict[str, JSONValue]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise Stage0G8Error("G8_NVIDIA_PROCESS_QUERY_FAILED")
    rows: list[dict[str, JSONValue]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit() and fields[1] in selected:
            rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1], "used_memory_mib": int(fields[2])})
    return rows


def _active_downloads() -> list[dict[str, JSONValue]]:
    if os.name == "nt":
        return []
    found: list[dict[str, JSONValue]] = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lower = raw.lower()
        if any(token in lower for token in ("curl ", "wget ", "aria2", "acquire_g3", "download")):
            found.append({"pid": int(path.parent.name), "command_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    return sorted(found, key=lambda item: int(item["pid"]))


def _memory_available_bytes() -> int:
    if os.name == "nt":
        raise Stage0G8Error("G8_FORMAL_PREFLIGHT_REQUIRES_LINUX")
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise Stage0G8Error("G8_MEMAVAILABLE_MISSING")


def _fd_soft_limit() -> int:
    try:
        import resource
    except ImportError as error:  # pragma: no cover - formal path is Linux
        raise Stage0G8Error("G8_RESOURCE_MODULE_UNAVAILABLE") from error
    return int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])


def _preflight_snapshot(
    request: TaskExecutionRequest,
    root: Path,
    selected: Sequence[str],
    controls: Mapping[str, Any],
    *,
    suite_ref: str,
) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
    gpu = _gpu_snapshot(selected)
    largest = max(
        int(value["checkpoint_parameter_state_bytes"])
        for value in controls["parameter_values"].values()
    )
    expected_new = largest * 4 * 2 + 8 * GIB
    data_usage = shutil.disk_usage(root)
    root_usage = shutil.disk_usage(Path("/"))
    stat = os.statvfs(root)
    snapshot: dict[str, JSONValue] = {
        "required_gate_ids": ["stage0.G0-G", "stage0.G5", "stage0.G6", "stage0.G7", "stage0.G3-S4", "stage0.G3-S5"],
        "passed_gate_ids": sorted(request.environment.passed_gate_ids),
        "source_clean": True,
        "identity_hashes_match": True,
        "selected_gpu_uuids": list(selected),
        "gpu_health_ok": bool(gpu["health_ok"]),
        "external_gpu_processes": _compute_processes(selected),
        "active_competing_downloads": _active_downloads(),
        "data_cursor_covered": True,
        "data_free_bytes": int(data_usage.free),
        "expected_new_bytes": expected_new,
        "root_free_bytes": int(root_usage.free),
        "inode_free": int(stat.f_favail),
        "fd_soft_limit": _fd_soft_limit(),
        "predicted_open_fds": 512,
        "memory_available_bytes": _memory_available_bytes(),
        "predicted_host_peak_bytes": largest * 4 + 16 * GIB,
        "output_collision": _logical_path(root, f"{suite_ref}/runs", field="runs_ref").exists(),
        "lease_available": True,
        "g1d_status": "ACCEPTED_SINGLE_DISK_RISK",
    }
    return snapshot, gpu


def _wait_for_no_compute_processes(
    selected: Sequence[str], *, timeout_seconds: float = 30.0
) -> list[dict[str, JSONValue]]:
    deadline = time.monotonic() + timeout_seconds
    observed = _compute_processes(selected)
    while observed and time.monotonic() < deadline:
        time.sleep(0.25)
        observed = _compute_processes(selected)
    return observed


def _launch_worker(
    *,
    root: Path,
    source: G8SourceBinding,
    plan_ref: str,
    launch_id: str,
    world_size: int,
    selected: Sequence[str],
    timeout_seconds: float,
    transcript_ref: str,
) -> dict[str, Any]:
    command = [sys.executable]
    if world_size == 4:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=4",
            ]
        )
    command.extend(
        [
            str(source.repository / "ops" / "stage0" / "run_g8_capacity_worker.py"),
            "--data-root",
            str(root),
            "--plan-ref",
            plan_ref,
        ]
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected[:world_size])
    environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(root)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    source_path = str(source.repository / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    started = datetime.now(timezone.utc)
    process = subprocess.Popen(
        command,
        cwd=source.repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - formal path is Linux
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
            stdout, stderr = process.communicate()
    completed = datetime.now(timezone.utc)
    residual = _wait_for_no_compute_processes(selected[:world_size])
    transcript = _with_hash(
        {
            "schema_version": "stage0-g8-launch-transcript-v1",
            "launch_id": launch_id,
            "plan_ref": plan_ref,
            "world_size": world_size,
            "selected_gpu_uuids": list(selected[:world_size]),
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "completed_at": completed.isoformat().replace("+00:00", "Z"),
            "duration_seconds": (completed - started).total_seconds(),
            "return_code": int(process.returncode or 0),
            "timed_out": timed_out,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-20000:],
            "residual_compute_processes": residual,
        }
    )
    _write_or_verify(
        _logical_path(root, transcript_ref, field="transcript_ref"), transcript
    )
    return transcript


def _worker_plan(
    *,
    root: Path,
    request: TaskExecutionRequest,
    source: G8SourceBinding,
    controls: Mapping[str, Any],
    suite_ref: str,
    key: str,
    mode: str,
    repeat_index: int,
    selected: Sequence[str],
) -> tuple[dict[str, JSONValue], str, str, str]:
    _spec_key, model_id, profile_id, candidate_stage, world_size, precision_profile = controls[
        "specs"
    ][key]
    config: ResolvedConfigV2 = controls["configs"][key]
    parameter = controls["parameter_values"][key]
    work = controls["work_values"][key]
    config_ref = controls["config_refs"][key]
    parameter_ref = controls["parameter_refs"][key]
    work_ref = controls["work_refs"][key]
    environment_ref = str(controls["environment_ref"])
    launch_id = f"{key}-{mode}-r{repeat_index}"
    output_root_ref = f"{suite_ref}/runs/{launch_id}"
    report_ref = f"{suite_ref}/reports/{launch_id}.json"
    plan_ref = f"{suite_ref}/plans/{launch_id}.json"
    checkpoint_steps = [25, 40] if profile_id == "g8-s4-160m" else [40]
    plan = _with_hash(
        {
            "schema_version": WORKER_PLAN_SCHEMA,
            "run_id": f"stage0-g8-{launch_id}",
            "profile_id": profile_id,
            "gate_id": {
                "g8-c-14m": "stage0.G8-C",
                "g8-s4-160m": "stage0.G8-S4",
                "g8-s5-410m": "stage0.G8-S5",
            }[profile_id],
            "model_id": model_id,
            "mode": mode,
            "precision_profile": precision_profile,
            "candidate_stage": candidate_stage,
            "repeat_index": repeat_index,
            "world_size": world_size,
            "selected_gpu_uuids": list(selected[:world_size]),
            "generator_git_commit": source.git_commit,
            "config_ref": config_ref,
            "config_sha256": sha256_file(_logical_path(root, config_ref, field="config_ref")),
            "config_hash": config.config_hash,
            "environment_ref": environment_ref,
            "environment_sha256": sha256_file(_logical_path(root, environment_ref, field="environment_ref")),
            "environment_hash": request.environment.environment_hash,
            "parameter_envelope_ref": parameter_ref,
            "parameter_envelope_sha256": sha256_file(_logical_path(root, parameter_ref, field="parameter_ref")),
            "parameter_envelope_hash": parameter["artifact_hash"],
            "work_envelope_ref": work_ref,
            "work_envelope_sha256": sha256_file(_logical_path(root, work_ref, field="work_ref")),
            "work_envelope_hash": work["artifact_hash"],
            "output_root_ref": output_root_ref,
            "report_ref": report_ref,
            "warmup_steps": 10,
            "measured_steps": 30,
            "memory_warmup_steps": 5,
            "memory_measured_steps": 20,
            "loader_worker_candidates": [0, 1, 2, 4],
            "run_loader_sweep": (
                profile_id == "g8-c-14m"
                and precision_profile == "bf16"
                and world_size == 4
                and mode == "formal"
                and repeat_index == 0
            ),
            "checkpoint_steps": checkpoint_steps,
            "collective_chunk_elements": 16 * 1024**2,
            "timeout_seconds": 600,
        }
    )
    _write_or_verify(_logical_path(root, plan_ref, field="plan_ref"), plan)
    return plan, plan_ref, report_ref, launch_id


def _load_hashed_report(root: Path, reference: str, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(_logical_path(root, reference, field=field)), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage0G8Error(f"G8_HASHED_REPORT_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def _validate_worker_report(
    root: Path,
    reference: str,
    *,
    plan: Mapping[str, Any],
    source: G8SourceBinding,
) -> dict[str, Any]:
    report = _load_hashed_report(root, reference, schema=WORKER_REPORT_SCHEMA, field="worker_report")
    plan_path = _logical_path(root, str(report.get("plan_ref")), field="report.plan_ref")
    if (
        report.get("status") != "PASS"
        or report.get("run_id") != plan["run_id"]
        or report.get("profile_id") != plan["profile_id"]
        or report.get("gate_id") != plan["gate_id"]
        or report.get("model_id") != plan["model_id"]
        or report.get("mode") != plan["mode"]
        or report.get("precision_profile") != plan["precision_profile"]
        or report.get("candidate_stage") != plan["candidate_stage"]
        or report.get("repeat_index") != plan["repeat_index"]
        or report.get("world_size") != plan["world_size"]
        or report.get("selected_gpu_uuids") != plan["selected_gpu_uuids"]
        or report.get("generator_git_commit") != source.git_commit
        or report.get("plan_ref") != plan_path.relative_to(root).as_posix()
        or report.get("plan_sha256") != sha256_file(plan_path)
        or report.get("config_hash") != plan["config_hash"]
        or report.get("environment_hash") != plan["environment_hash"]
        or report.get("parameter_envelope_hash") != plan["parameter_envelope_hash"]
        or report.get("work_envelope_hash") != plan["work_envelope_hash"]
        or report.get("peak_memory_fraction_max", 1.0) > 0.80
        or report.get("open_fds_fraction_max", 1.0) > 0.70
        or report.get("all_scratch_removed") is not True
        or report.get("no_unknown_gpu_processes") is not True
        or report.get("synthetic_mathematics_implemented") is not False
    ):
        raise Stage0G8Error("G8_WORKER_REPORT_IDENTITY_OR_THRESHOLD_INVALID")
    rank_reports = report.get("rank_reports")
    if not isinstance(rank_reports, list) or len(rank_reports) != int(plan["world_size"]):
        raise Stage0G8Error("G8_WORKER_REPORT_RANK_SET_INVALID")
    work = _load_hashed_report(
        root,
        str(plan["work_envelope_ref"]),
        schema="stage0.compute-communication-work-envelope.v1",
        field="report.work_envelope",
    )
    expected_collectives = int(work["checkpoint_boundary"]["collective_count"])
    expected_checkpoint_steps = list(plan["checkpoint_steps"])
    identities = report.get("rank_identities")
    if not isinstance(identities, list) or len(identities) != int(plan["world_size"]):
        raise Stage0G8Error("G8_WORKER_REPORT_RANK_IDENTITIES_INVALID")
    for rank, row in enumerate(rank_reports):
        item = _mapping(row, field="rank_report")
        event_ref = item.get("event_stream_ref")
        checkpoints = item.get("checkpoint_records")
        tensorboard_refs = item.get("tensorboard_refs")
        phases = item.get("phase_memory_bytes")
        if (
            item.get("rank") != rank
            or _mapping(identities[rank], field="rank_identity").get("rank") != rank
            or _mapping(identities[rank], field="rank_identity").get("gpu_uuid")
            != plan["selected_gpu_uuids"][rank]
            or item.get("synthetic_mathematics_implemented") is not False
            or item.get("synthetic_buffer_ids")
            != [
                "raw_accumulator",
                "signed_accumulator",
                "positive_accumulator",
                "negative_accumulator",
                "u_first_moment",
                "u_second_moment",
                "path_integral_accumulator",
            ]
            or len(item.get("step_seconds", [])) != 30
            or not isinstance(checkpoints, list)
            or [record.get("step") for record in checkpoints] != expected_checkpoint_steps
            or any(record.get("collective_calls") != expected_collectives for record in checkpoints)
            or any(record.get("scratch_removed") is not True for record in checkpoints)
            or not isinstance(event_ref, str)
            or item.get("event_stream_sha256")
            != sha256_file(_logical_path(root, event_ref, field="event_ref"))
            or not isinstance(tensorboard_refs, list)
            or (plan["mode"] == "formal") != bool(tensorboard_refs)
            or any(not _logical_path(root, value, field="tensorboard_ref").is_file() for value in tensorboard_refs)
            or not isinstance(phases, Mapping)
            or set(phases)
            != {
                "model_loaded_allocated",
                "parameter_envelope_allocated",
                "training_window_peak",
                "whole_process_peak",
            }
        ):
            raise Stage0G8Error("G8_WORKER_REPORT_RANK_CONTENT_INVALID")
        for record in checkpoints:
            for field in ("purge_intent_ref", "purge_record_ref"):
                audit = _load_hashed_report(
                    root,
                    str(record[field]),
                    schema=(
                        "stage0-g8-scratch-purge-intent-v1"
                        if field == "purge_intent_ref"
                        else "stage0-g8-scratch-purge-record-v1"
                    ),
                    field=field,
                )
                if audit.get("rank") != rank or audit.get("step") != record["step"]:
                    raise Stage0G8Error("G8_WORKER_PURGE_AUDIT_IDENTITY_INVALID")
    return report


def _fault_exercises(
    request: TaskExecutionRequest,
    root: Path,
    source: G8SourceBinding,
    snapshot: Mapping[str, Any],
    *,
    suite_ref: str,
) -> tuple[dict[str, JSONValue], str]:
    incompatible = dict(snapshot)
    incompatible["memory_available_bytes"] = 1
    oom_preflight = evaluate_launch_preflight(incompatible)
    canary = exercise_canary_writer(
        lambda: (_ for _ in ()).throw(OSError("injected truth-store failure"))
    )
    stale = classify_stale_heartbeat(heartbeat_stale=True, process_alive=True)
    claims = LaunchClaimRegistry(root)
    fixture_id = f"g8-ssh-fixture-{request.config.config_hash[:12]}"
    claims.claim(
        launch_id=fixture_id,
        run_id="stage0-g8-ssh-fixture",
        config_hash=request.config.config_hash,
        environment_hash=request.environment.environment_hash,
    )
    duplicate_rejected = False
    try:
        claims.claim(
            launch_id=fixture_id,
            run_id="stage0-g8-ssh-fixture",
            config_hash=request.config.config_hash,
            environment_hash=request.environment.environment_hash,
        )
    except RuntimeError as error:
        duplicate_rejected = "LAUNCH_CLAIM_ALREADY_EXISTS" in str(error)
    g6_ref = request.environment.evidence_refs.get("g6_distributed_validation")
    if g6_ref is None:
        raise Stage0G8Error("G8_G6_RANK_FAILURE_EVIDENCE_MISSING")
    g6_envelope = _mapping(
        load_committed_task_artifact(root, g6_ref, require_formal=True).payload,
        field="g6_envelope",
    )
    g6_gate = GateRecord.from_mapping(
        _mapping(g6_envelope.get("gate_record"), field="g6_gate_record")
    )
    g6_validation = _mapping(
        g6_envelope.get("validation_report"), field="g6_validation"
    )
    g6_metrics = _mapping(g6_validation.get("metrics"), field="g6_metrics")
    if (
        g6_gate.gate_id != "stage0.G6"
        or g6_gate.status is not GateStatus.PASS
        or not isinstance(g6_metrics.get("controlled_failure_duration_seconds"), (int, float))
        or float(g6_metrics["controlled_failure_duration_seconds"]) > 60.0
    ):
        raise Stage0G8Error("G8_G6_RANK_FAILURE_REVALIDATION_FAILED")
    responses = [failure_response(item) for item in FailureClass]
    if (
        oom_preflight["status"] != "FAIL"
        or oom_preflight["running_state_may_publish"] is not False
        or canary["status"] != "EXPECTED_FAILURE"
        or canary["stop_required"] is not True
        or stale != "ACTIVE_PROCESS_HEARTBEAT_STALE_DO_NOT_REAP"
        or not duplicate_rejected
        or len(responses) != len(FailureClass)
    ):
        raise Stage0G8Error("G8_CONTROLLED_FAILURE_EXERCISE_FAILED")
    report = _with_hash(
        {
            "schema_version": "stage0-g8-fault-exercise-report-v1",
            "status": "PASS",
            "checked_at": _now(),
            "generator_git_commit": source.git_commit,
            "controlled_oom_or_memory_preflight": oom_preflight,
            "single_rank_failure_revalidated_g6_gate_hash": str(g6_envelope["artifact_hash"]),
            "single_rank_failure_duration_seconds": float(
                g6_metrics["controlled_failure_duration_seconds"]
            ),
            "truth_writer_canary": canary,
            "stale_live_process_classification": stale,
            "ssh_duplicate_launch_rejected": duplicate_rejected,
            "failure_responses": responses,
            "orphan_processes_after_exercises": _compute_processes(
                tuple(snapshot["selected_gpu_uuids"])
            ),
            "unknown_files_deleted": False,
            "same_disk_claimed_as_backup": False,
        }
    )
    if report["orphan_processes_after_exercises"]:
        raise Stage0G8Error("G8_FAULT_EXERCISE_LEFT_GPU_PROCESS")
    reference = f"{suite_ref}/reports/fault-exercises.json"
    _write_or_verify(_logical_path(root, reference, field="fault_ref"), report)
    return report, reference


def _summarize_measurements(
    controls: Mapping[str, Any],
    reports: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, JSONValue]:
    expected_keys = set(controls["execution_keys"])
    if len(reports) != len(expected_keys) * 2 * 3:
        raise Stage0G8Error("G8_MEASUREMENT_REPORT_COUNT_INVALID")
    overheads: dict[str, list[float]] = {}
    for key in sorted(expected_keys):
        values: list[float] = []
        for repeat in range(3):
            minimal = float(reports[(key, "minimal", repeat)]["effective_tokens_per_second"])
            formal = float(reports[(key, "formal", repeat)]["effective_tokens_per_second"])
            values.append(max(0.0, 1.0 - formal / minimal))
        overheads[key] = values
    overhead_medians = {key: statistics.median(values) for key, values in overheads.items()}
    if any(value > 0.10 for value in overhead_medians.values()):
        raise Stage0G8Error("G8_FORMAL_LOGGING_OVERHEAD_EXCEEDED")

    scaling: dict[str, dict[str, float]] = {}
    for precision in ("fp32", "bf16"):
        single_key = f"g8-c-14m-{precision}-w1"
        four_key = f"g8-c-14m-{precision}-w4"
        single = statistics.median(
            float(reports[(single_key, "formal", repeat)]["effective_tokens_per_second"])
            for repeat in range(3)
        )
        four = statistics.median(
            float(reports[(four_key, "formal", repeat)]["effective_tokens_per_second"])
            for repeat in range(3)
        )
        scaling[precision] = {
            "single_effective_tokens_per_second": single,
            "four_gpu_effective_tokens_per_second": four,
            "strong_scaling_efficiency": four / (4.0 * single),
        }

    peak = max(float(item["peak_memory_fraction_max"]) for item in reports.values())
    fd = max(float(item["open_fds_fraction_max"]) for item in reports.values())
    imbalance = max(
        float(item["rank_peak_memory_imbalance_fraction"]) for item in reports.values()
    )
    checkpoint_pause = max(
        float(item["checkpoint_pause_seconds_max"]) for item in reports.values()
    )
    precision_coverage = {
        str(item["precision_profile"])
        for item in reports.values()
        if item["profile_id"] == "g8-c-14m"
    }
    if precision_coverage != {"fp32", "bf16"} or peak > 0.80 or fd > 0.70:
        raise Stage0G8Error("G8_CAPACITY_COVERAGE_OR_THRESHOLD_INVALID")

    loader_rows: dict[int, list[float]] = {0: [], 1: [], 2: [], 4: []}
    loader_source = reports[("g8-c-14m-bf16-w4", "formal", 0)]
    for rank_report in loader_source["rank_reports"]:
        for row in rank_report["loader_profiles"]:
            loader_rows[int(row["num_workers"])].append(float(row["samples_per_second"]))
    if any(len(values) != 4 for values in loader_rows.values()):
        raise Stage0G8Error("G8_LOADER_SWEEP_INCOMPLETE")
    loader_medians = {str(key): statistics.median(value) for key, value in loader_rows.items()}
    best = max(loader_rows, key=lambda key: loader_medians[str(key)])
    saturation = min(
        key for key in sorted(loader_rows) if loader_medians[str(key)] >= loader_medians[str(best)] * 0.95
    )

    estimation_rows: list[dict[str, JSONValue]] = []
    for key in sorted(expected_keys):
        envelope = controls["parameter_values"][key]
        formal_reports = [reports[(key, "formal", repeat)] for repeat in range(3)]
        measured_peak = max(int(item["peak_memory_bytes_max"]) for item in formal_reports)
        measured_checkpoint = max(
            int(item["checkpoint_bytes_total_max_boundary"]) for item in formal_reports
        )
        estimated_peak = int(envelope["peak_parameter_state_gpu_bytes"])
        estimated_checkpoint = int(envelope["checkpoint_parameter_state_bytes"]) * int(
            formal_reports[0]["world_size"]
        )
        estimation_rows.append(
            {
                "config_key": key,
                "gpu_peak_estimated_bytes": estimated_peak,
                "gpu_peak_measured_bytes": measured_peak,
                "gpu_peak_signed_error_bytes": measured_peak - estimated_peak,
                "gpu_peak_estimate_to_measurement_ratio": estimated_peak / measured_peak,
                "checkpoint_estimated_bytes": estimated_checkpoint,
                "checkpoint_measured_bytes": measured_checkpoint,
                "checkpoint_signed_error_bytes": measured_checkpoint - estimated_checkpoint,
                "checkpoint_estimate_to_measurement_ratio": estimated_checkpoint / measured_checkpoint,
            }
        )
    return {
        "status": "PASS",
        "fresh_process_repetitions": 3,
        "warmup_steps": 10,
        "measured_steps": 30,
        "memory_warmup_steps": 5,
        "memory_measured_steps": 20,
        "precision_coverage": sorted(precision_coverage),
        "formal_logging_overhead_by_config": overheads,
        "formal_logging_overhead_median_by_config": overhead_medians,
        "formal_logging_overhead_limit": 0.10,
        "strong_scaling": scaling,
        "peak_memory_fraction_max": peak,
        "peak_memory_fraction_limit": 0.80,
        "rank_peak_memory_imbalance_fraction_max": imbalance,
        "open_fds_fraction_max": fd,
        "open_fds_fraction_limit": 0.70,
        "checkpoint_pause_seconds_max": checkpoint_pause,
        "loader_worker_median_samples_per_second": loader_medians,
        "loader_worker_saturation_choice": saturation,
        "estimation_error_rows": estimation_rows,
        "performance_baseline_kind": "environment_and_topology_bound_initial_baseline",
        "regression_warning_fraction": 0.70,
        "synthetic_mathematics_implemented": False,
    }


def _run_suite(
    request: TaskExecutionRequest,
    root: Path,
    source: G8SourceBinding,
    selected: Sequence[str],
    suite_ref: str,
) -> dict[str, Any]:
    controls = _build_controls(request, root, source, suite_ref)
    lease = ProjectGpuLease(
        root,
        GpuLeaseIdentity(
            run_id=f"stage0-g8-{request.config.config_hash[:12]}",
            lease_id=f"stage0-g8-{request.environment.environment_hash[:12]}",
            gpu_uuids=tuple(selected),
            owner="stage0-g8-capacity-orchestrator",
            config_hash=request.config.config_hash,
            environment_hash=request.environment.environment_hash,
        ),
    )
    lease.acquire()
    lease_history: Path | None = None
    try:
        snapshot, pre_gpu = _preflight_snapshot(
            request, root, selected, controls, suite_ref=suite_ref
        )
        preflight = evaluate_launch_preflight(snapshot)
        preflight_value = _with_hash(
            {
                "schema_version": "stage0-g8-preflight-evidence-v1",
                "checked_at": _now(),
                "generator_git_commit": source.git_commit,
                "environment_hash": request.environment.environment_hash,
                "project_gpu_lease": dict(
                    load_canonical_json(lease.current_path)
                ),
                "gpu_snapshot": pre_gpu,
                "preflight": preflight,
            }
        )
        preflight_ref = f"{suite_ref}/reports/preflight.json"
        _write_or_verify(
            _logical_path(root, preflight_ref, field="preflight_ref"), preflight_value
        )
        if preflight["status"] != "PASS" or preflight["running_state_may_publish"] is not True:
            raise Stage0G8Error("G8_PREFLIGHT_NOT_PASS")

        claims = LaunchClaimRegistry(root)
        reports: dict[tuple[str, str, int], dict[str, Any]] = {}
        report_refs: list[str] = []
        plan_refs: list[str] = []
        transcript_refs: list[str] = []
        for key in controls["execution_keys"]:
            world_size = int(controls["specs"][key][4])
            for repeat in range(3):
                modes = ("minimal", "formal") if repeat % 2 == 0 else ("formal", "minimal")
                for mode in modes:
                    plan, plan_ref, report_ref, launch_id = _worker_plan(
                        root=root,
                        request=request,
                        source=source,
                        controls=controls,
                        suite_ref=suite_ref,
                        key=key,
                        mode=mode,
                        repeat_index=repeat,
                        selected=selected,
                    )
                    claims.claim(
                        launch_id=launch_id,
                        run_id=str(plan["run_id"]),
                        config_hash=str(plan["config_hash"]),
                        environment_hash=request.environment.environment_hash,
                    )
                    transcript_ref = f"{suite_ref}/transcripts/{launch_id}.json"
                    transcript = _launch_worker(
                        root=root,
                        source=source,
                        plan_ref=plan_ref,
                        launch_id=launch_id,
                        world_size=world_size,
                        selected=selected,
                        timeout_seconds=14400.0,
                        transcript_ref=transcript_ref,
                    )
                    if (
                        transcript["return_code"] != 0
                        or transcript["timed_out"] is not False
                        or transcript["residual_compute_processes"]
                    ):
                        raise Stage0G8Error(f"G8_WORKER_LAUNCH_FAILED:{launch_id}")
                    report = _validate_worker_report(
                        root, report_ref, plan=plan, source=source
                    )
                    reports[(key, mode, repeat)] = report
                    plan_refs.append(plan_ref)
                    report_refs.append(report_ref)
                    transcript_refs.append(transcript_ref)
                    lease.heartbeat()

        summary = _summarize_measurements(controls, reports)
        summary_report = _with_hash(
            {
                "schema_version": "stage0-g8-measurement-summary-v1",
                "status": "PASS",
                "checked_at": _now(),
                "generator_git_commit": source.git_commit,
                "environment_hash": request.environment.environment_hash,
                "topology_sha256": pre_gpu["topology_sha256"],
                "metrics": summary,
                "worker_report_refs": report_refs,
            }
        )
        summary_ref = f"{suite_ref}/reports/measurement-summary.json"
        _write_or_verify(_logical_path(root, summary_ref, field="summary_ref"), summary_report)
        fault_report, fault_ref = _fault_exercises(
            request,
            root,
            source,
            snapshot,
            suite_ref=suite_ref,
        )
        post_gpu = _gpu_snapshot(selected)
        post_processes = _wait_for_no_compute_processes(selected)
        if (
            post_processes
            or post_gpu["health_ok"] is not True
            or post_gpu["topology_sha256"] != pre_gpu["topology_sha256"]
            or [item["volatile_uncorrected_ecc"] for item in post_gpu["selected"]]
            != [item["volatile_uncorrected_ecc"] for item in pre_gpu["selected"]]
            or [item["aggregate_uncorrected_ecc"] for item in post_gpu["selected"]]
            != [item["aggregate_uncorrected_ecc"] for item in pre_gpu["selected"]]
        ):
            raise Stage0G8Error("G8_POSTFLIGHT_GPU_OR_PROCESS_INVALID")
        postflight = _with_hash(
            {
                "schema_version": "stage0-g8-postflight-evidence-v1",
                "status": "PASS",
                "checked_at": _now(),
                "generator_git_commit": source.git_commit,
                "gpu_snapshot": post_gpu,
                "residual_compute_processes": post_processes,
                "gpu_contexts_returned_to_zero": True,
                "project_lease_release_pending": True,
            }
        )
        postflight_ref = f"{suite_ref}/reports/postflight.json"
        _write_or_verify(
            _logical_path(root, postflight_ref, field="postflight_ref"), postflight
        )
        lease_history = lease.release(outcome="SUCCESS")
    except BaseException:
        if lease_history is None:
            lease_history = lease.release(outcome="FAILED")
        raise
    assert lease_history is not None
    lease_history_ref = lease_history.relative_to(root).as_posix()
    return {
        "controls": controls,
        "preflight_ref": preflight_ref,
        "preflight": preflight_value,
        "plan_refs": plan_refs,
        "report_refs": report_refs,
        "reports": reports,
        "transcript_refs": transcript_refs,
        "summary_ref": summary_ref,
        "summary": summary_report,
        "fault_ref": fault_ref,
        "fault": fault_report,
        "postflight_ref": postflight_ref,
        "postflight": postflight,
        "lease_history_ref": lease_history_ref,
    }


def _gate_payloads(
    request: TaskExecutionRequest,
    root: Path,
    source: G8SourceBinding,
    suite: Mapping[str, Any],
) -> tuple[dict[str, GateRecord], dict[str, Stage0GateReport], str]:
    checked_at = str(suite["summary"]["checked_at"])
    metrics = _mapping(suite["summary"]["metrics"], field="summary.metrics")
    reports = suite["reports"]
    controls = suite["controls"]
    common_refs = (
        str(controls["budget_ref"]),
        str(suite["preflight_ref"]),
        str(suite["summary_ref"]),
        str(suite["fault_ref"]),
        str(suite["postflight_ref"]),
        str(suite["lease_history_ref"]),
    )
    evidence_schema = {
        str(controls["budget_ref"]): "stage0-g8-per-model-budget-report-v1",
        str(suite["preflight_ref"]): "stage0-g8-preflight-evidence-v1",
        str(suite["summary_ref"]): "stage0-g8-measurement-summary-v1",
        str(suite["fault_ref"]): "stage0-g8-fault-exercise-report-v1",
        str(suite["postflight_ref"]): "stage0-g8-postflight-evidence-v1",
        str(suite["lease_history_ref"]): "runtime.project-gpu-lease-history.v1",
    }
    evidence = tuple(
        Stage0EvidenceRef(
            ref=reference,
            sha256=sha256_file(_logical_path(root, reference, field="gate_evidence_ref")),
            schema_version=evidence_schema[reference],
        )
        for reference in common_refs
    )
    config_hashes = {
        key: config.config_hash for key, config in controls["configs"].items()
    }
    environment_id = f"env-{request.environment.environment_hash[:20]}"

    c_reports = [item for item in reports.values() if item["gate_id"] == "stage0.G8-C"]
    s4_reports = [item for item in reports.values() if item["gate_id"] == "stage0.G8-S4"]
    s5_reports = [item for item in reports.values() if item["gate_id"] == "stage0.G8-S5"]
    if len(c_reports) != 24 or len(s4_reports) != 6 or len(s5_reports) != 6:
        raise Stage0G8Error("G8_COMPONENT_REPORT_COUNTS_INVALID")

    checks_by_gate: dict[str, tuple[Stage0GateCheck, ...]] = {
        "stage0.G8-C": (
            Stage0GateCheck(
                "stage0.G8-C.protocol",
                Stage0CheckClass.CORRECTNESS,
                Stage0CheckStatus.PASS,
                "Exact-shape synthetic capacity protocol ran in fresh real-asset processes without importance mathematics.",
                measurements={
                    "worker_reports": len(c_reports),
                    "fresh_process_repetitions": metrics["fresh_process_repetitions"],
                    "precision_coverage": metrics["precision_coverage"],
                    "synthetic_mathematics_implemented": metrics["synthetic_mathematics_implemented"],
                },
                evidence_refs=(str(suite["summary_ref"]),),
            ),
            Stage0GateCheck(
                "stage0.G8-C.capacity",
                Stage0CheckClass.CAPACITY,
                Stage0CheckStatus.PASS,
                "All observed GPU peaks and file-descriptor peaks remain within hard limits.",
                exception_eligible=True,
                measurements={
                    "peak_memory_fraction": metrics["peak_memory_fraction_max"],
                    "peak_memory_limit": 0.80,
                    "open_fds_fraction": metrics["open_fds_fraction_max"],
                    "open_fds_limit": 0.70,
                },
                evidence_refs=(str(suite["summary_ref"]),),
            ),
            Stage0GateCheck(
                "stage0.G8-C.scaling",
                Stage0CheckClass.PERFORMANCE,
                Stage0CheckStatus.PASS,
                "FP32 and BF16 single/four-GPU strong-scaling baselines are bound to this environment and topology.",
                exception_eligible=True,
                measurements={"strong_scaling": metrics["strong_scaling"]},
                evidence_refs=(str(suite["summary_ref"]),),
            ),
            Stage0GateCheck(
                "stage0.G8-C.logging-overhead",
                Stage0CheckClass.PERFORMANCE,
                Stage0CheckStatus.PASS,
                "Median paired formal logging overhead is at most ten percent for every measured config.",
                exception_eligible=True,
                measurements={
                    "median_by_config": metrics["formal_logging_overhead_median_by_config"],
                    "limit": 0.10,
                },
                evidence_refs=(str(suite["summary_ref"]),),
            ),
            Stage0GateCheck(
                "stage0.G8-C.operations",
                Stage0CheckClass.OPERATIONAL,
                Stage0CheckStatus.PASS,
                "Project lease, launch preflight, loader sweep, postflight and exact scratch purge all passed.",
                measurements={
                    "loader_worker_saturation_choice": metrics["loader_worker_saturation_choice"],
                    "preflight_status": suite["preflight"]["preflight"]["status"],
                    "postflight_status": suite["postflight"]["status"],
                },
                evidence_refs=(
                    str(suite["preflight_ref"]),
                    str(suite["postflight_ref"]),
                    str(suite["lease_history_ref"]),
                ),
            ),
            Stage0GateCheck(
                "stage0.G8-C.faults",
                Stage0CheckClass.SAFETY,
                Stage0CheckStatus.PASS,
                "Controlled memory, rank, writer, stale-heartbeat and duplicate-launch failures fail closed without orphans.",
                measurements={"fault_status": suite["fault"]["status"]},
                evidence_refs=(str(suite["fault_ref"]),),
            ),
        ),
        "stage0.G8-S4": (
            Stage0GateCheck(
                "stage0.G8-S4.real-assets",
                Stage0CheckClass.CORRECTNESS,
                Stage0CheckStatus.PASS,
                "The 160M Stage 4 candidate used its own qualified model shapes and Stage 4 data budget.",
                measurements={
                    "worker_reports": len(s4_reports),
                    "model_id": "pythia-160m-deduped-step0",
                    "candidate_stage": 4,
                    "scale_extrapolation_used": False,
                },
                evidence_refs=(str(controls["budget_ref"]), str(suite["summary_ref"])),
            ),
            Stage0GateCheck(
                "stage0.G8-S4.capacity",
                Stage0CheckClass.CAPACITY,
                Stage0CheckStatus.PASS,
                "The 160M BF16 four-GPU candidate, formal logging, full state envelope and checkpoints stayed within limits.",
                exception_eligible=True,
                measurements={
                    "peak_memory_fraction": max(float(item["peak_memory_fraction_max"]) for item in s4_reports),
                    "checkpoint_cadence_steps": 25,
                    "reports": 6,
                },
                evidence_refs=(str(suite["summary_ref"]),),
            ),
        ),
        "stage0.G8-S5": (
            Stage0GateCheck(
                "stage0.G8-S5.real-assets",
                Stage0CheckClass.CORRECTNESS,
                Stage0CheckStatus.PASS,
                "The 410M Stage 5 candidate used its own qualified model shapes and Stage 5 data budget.",
                measurements={
                    "worker_reports": len(s5_reports),
                    "model_id": "pythia-410m-deduped-step0",
                    "candidate_stage": 5,
                    "scale_extrapolation_used": False,
                },
                evidence_refs=(str(controls["budget_ref"]), str(suite["summary_ref"])),
            ),
            Stage0GateCheck(
                "stage0.G8-S5.capacity",
                Stage0CheckClass.CAPACITY,
                Stage0CheckStatus.PASS,
                "The 410M BF16 four-GPU candidate, formal logging, full state envelope and high-cadence checkpoint stayed within limits.",
                exception_eligible=True,
                measurements={
                    "peak_memory_fraction": max(float(item["peak_memory_fraction_max"]) for item in s5_reports),
                    "checkpoint_cadence_steps": 40,
                    "reports": 6,
                },
                evidence_refs=(str(suite["summary_ref"]),),
            ),
        ),
    }
    reports_by_gate: dict[str, Stage0GateReport] = {}
    records: dict[str, GateRecord] = {}
    for gate_id, checks in checks_by_gate.items():
        report = Stage0GateReport(
            gate_id=gate_id,
            generated_at=checked_at,
            generator_git_commit=source.git_commit,
            environment_id=environment_id,
            checks=checks,
            input_evidence=evidence,
            config_hashes=config_hashes,
        )
        reports_by_gate[gate_id] = report
        records[gate_id] = GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at=checked_at,
            measured={
                "component_status": report.status.value,
                "gate_report_artifact_hash": report.artifact_hash,
            },
            threshold={"required_status": "PASS"},
            evidence_refs=common_refs,
        )

    composition_checks = tuple(
        Stage0GateCheck(
            f"stage0.G8.component-{gate_id.rsplit('-', 1)[-1].lower()}",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            f"Required component {gate_id} is PASS.",
            measurements={"gate_id": gate_id, "status": records[gate_id].status.value},
            evidence_refs=(str(suite["summary_ref"]),),
        )
        for gate_id in ("stage0.G8-C", "stage0.G8-S4", "stage0.G8-S5")
    )
    composition = Stage0GateReport(
        gate_id="stage0.G8",
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        checks=composition_checks,
        input_evidence=evidence,
        config_hashes=config_hashes,
    )
    reports_by_gate["stage0.G8"] = composition
    records["stage0.G8"] = GateRecord(
        gate_id="stage0.G8",
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "component_statuses": {
                gate_id: records[gate_id].status.value
                for gate_id in ("stage0.G8-C", "stage0.G8-S4", "stage0.G8-S5")
            },
            "gate_report_artifact_hash": composition.artifact_hash,
        },
        threshold={"all_components_required": "PASS"},
        evidence_refs=common_refs,
    )
    return records, reports_by_gate, checked_at


def run_formal_g8_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G8Error("G8_FORMAL_REQUEST_INVALID")
    required = {"stage0.G0-G", "stage0.G5", "stage0.G6", "stage0.G7", "stage0.G3-S4", "stage0.G3-S5"}
    if not required <= request.environment.passed_gate_ids:
        raise Stage0G8Error("G8_REQUIRED_GATES_MISSING")
    provenance_ref = request.environment.evidence_refs.get("g4_provenance")
    if provenance_ref is None:
        raise Stage0G8Error("G8_G4_PROVENANCE_REF_MISSING")
    provenance = _mapping(
        load_committed_task_artifact(root, provenance_ref, require_formal=True).payload,
        field="g4_provenance_envelope",
    )
    selected = _mapping(provenance.get("provenance_record"), field="g4_provenance").get(
        "device_mapping"
    )
    if not isinstance(selected, list) or len(selected) != 4:
        raise Stage0G8Error("G8_GPU_SELECTION_INVALID")
    suite_ref = (
        f"evidence/stage0/g8-suite/{request.config.config_hash}/"
        f"{request.environment.environment_hash}"
    )
    suite = _run_suite(
        request, root, source, tuple(str(item) for item in selected), suite_ref
    )
    records, gate_reports, checked_at = _gate_payloads(request, root, source, suite)
    evidence_refs = tuple(
        dict.fromkeys(
            (
                *source_refs,
                str(suite["controls"]["budget_ref"]),
                *suite["controls"]["config_refs"].values(),
                *suite["controls"]["parameter_refs"].values(),
                *suite["controls"]["work_refs"].values(),
                str(suite["preflight_ref"]),
                *suite["plan_refs"],
                *suite["report_refs"],
                *suite["transcript_refs"],
                str(suite["summary_ref"]),
                str(suite["fault_ref"]),
                str(suite["postflight_ref"]),
                str(suite["lease_history_ref"]),
            )
        )
    )
    refs: dict[str, str] = {}
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g8-capacity-evidence-v1",
            "artifact_role": kind,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "gate_ids": list(GATE_IDS),
            "suite_root_ref": suite_ref,
            "budget_report_ref": str(suite["controls"]["budget_ref"]),
            "config_refs": dict(suite["controls"]["config_refs"]),
            "parameter_envelope_refs": dict(suite["controls"]["parameter_refs"]),
            "work_envelope_refs": dict(suite["controls"]["work_refs"]),
            "preflight_ref": str(suite["preflight_ref"]),
            "worker_plan_refs": list(suite["plan_refs"]),
            "worker_report_refs": list(suite["report_refs"]),
            "launch_transcript_refs": list(suite["transcript_refs"]),
            "measurement_summary_ref": str(suite["summary_ref"]),
            "fault_report_ref": str(suite["fault_ref"]),
            "postflight_ref": str(suite["postflight_ref"]),
            "lease_history_ref": str(suite["lease_history_ref"]),
            "gate_records": {gate_id: records[gate_id].to_dict() for gate_id in GATE_IDS},
            "gate_reports": {gate_id: gate_reports[gate_id].to_dict() for gate_id in GATE_IDS},
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        refs[kind] = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
            source_refs=evidence_refs,
        ).commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        message="Stage 0 S0.10 capacity and operations gates passed",
        metadata={"stage0_g8_specialized": True, "gate_ids": list(GATE_IDS)},
    )


def validate_formal_g8_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G8Error("G8_OUTPUT_SET_INVALID")
    envelopes: list[dict[str, Any]] = []
    for kind, reference in outputs.items():
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != request.config.config_hash
        ):
            raise Stage0G8Error("G8_OUTPUT_COMMIT_IDENTITY_INVALID")
        envelope = _mapping(loaded.payload, field=f"output.{kind}")
        declared = envelope.pop("artifact_hash", None)
        if (
            envelope.get("schema_version") != "stage0-g8-capacity-evidence-v1"
            or envelope.get("artifact_role") != kind
            or envelope.get("status") != "PASS"
            or envelope.get("generator_git_commit") != source.git_commit
            or envelope.get("environment_hash") != request.environment.environment_hash
            or envelope.get("gate_ids") != list(GATE_IDS)
            or declared != canonical_json_hash(envelope)
        ):
            raise Stage0G8Error("G8_OUTPUT_ENVELOPE_INVALID")
        envelope["artifact_hash"] = declared
        envelopes.append(envelope)
    canonical = {
        key: value
        for key, value in envelopes[0].items()
        if key not in {"artifact_role", "artifact_hash"}
    }
    if any(
        {
            key: value
            for key, value in envelope.items()
            if key not in {"artifact_role", "artifact_hash"}
        }
        != canonical
        for envelope in envelopes[1:]
    ):
        raise Stage0G8Error("G8_OUTPUT_ROLE_PAYLOAD_DRIFT")

    raw_config_refs = canonical.get("config_refs")
    raw_parameter_refs = canonical.get("parameter_envelope_refs")
    raw_work_refs = canonical.get("work_envelope_refs")
    if not all(isinstance(item, Mapping) for item in (raw_config_refs, raw_parameter_refs, raw_work_refs)):
        raise Stage0G8Error("G8_OUTPUT_CONTROL_REF_MAP_INVALID")
    config_refs = {str(key): str(value) for key, value in raw_config_refs.items()}
    parameter_refs = {str(key): str(value) for key, value in raw_parameter_refs.items()}
    work_refs = {str(key): str(value) for key, value in raw_work_refs.items()}
    if set(config_refs) != set(parameter_refs) or set(config_refs) != set(work_refs) or len(config_refs) != 7:
        raise Stage0G8Error("G8_OUTPUT_CONTROL_REF_KEY_SET_INVALID")
    configs: dict[str, ResolvedConfigV2] = {}
    parameters: dict[str, dict[str, Any]] = {}
    works: dict[str, dict[str, Any]] = {}
    for key in config_refs:
        configs[key] = ResolvedConfigV2.from_mapping(
            _mapping(
                load_canonical_json(_logical_path(root, config_refs[key], field="config_ref")),
                field="config",
            )
        )
        parameters[key] = _load_hashed_report(
            root,
            parameter_refs[key],
            schema="stage0.parameter-state-capacity-envelope.v1",
            field="parameter_envelope",
        )
        works[key] = _load_hashed_report(
            root,
            work_refs[key],
            schema="stage0.compute-communication-work-envelope.v1",
            field="work_envelope",
        )
        if (
            configs[key].task_id != TASK_ID
            or parameters[key]["config_hash"] != configs[key].config_hash
            or parameters[key]["model_id"] != works[key]["model_id"]
            or parameters[key]["mathematics_implemented"] is not False
            or works[key]["mathematics_implemented"] is not False
        ):
            raise Stage0G8Error("G8_OUTPUT_CONTROL_IDENTITY_INVALID")
    budget_ref = canonical.get("budget_report_ref")
    if not isinstance(budget_ref, str):
        raise Stage0G8Error("G8_OUTPUT_BUDGET_REF_INVALID")
    budget = _load_hashed_report(
        root, budget_ref, schema="stage0-g8-per-model-budget-report-v1", field="budget"
    )
    models = budget.get("models")
    if (
        budget.get("status") != "PASS"
        or budget.get("scale_extrapolation_used") is not False
        or not isinstance(models, list)
        or len(models) != 4
        or len({item.get("model_id") for item in models if isinstance(item, Mapping)}) != 4
        or any(not isinstance(item, Mapping) or item.get("derived_from_other_model_scale") is not False for item in models)
    ):
        raise Stage0G8Error("G8_OUTPUT_PER_MODEL_BUDGET_INVALID")

    plan_refs = canonical.get("worker_plan_refs")
    report_refs = canonical.get("worker_report_refs")
    transcript_refs = canonical.get("launch_transcript_refs")
    if (
        not isinstance(plan_refs, list)
        or not isinstance(report_refs, list)
        or not isinstance(transcript_refs, list)
        or len(plan_refs) != 36
        or len(report_refs) != 36
        or len(transcript_refs) != 36
    ):
        raise Stage0G8Error("G8_OUTPUT_EXECUTION_REF_COUNTS_INVALID")
    config_hash_to_key = {config.config_hash: key for key, config in configs.items()}
    if len(config_hash_to_key) != len(configs):
        raise Stage0G8Error("G8_OUTPUT_CONFIG_HASH_COLLISION")
    reports: dict[tuple[str, str, int], dict[str, Any]] = {}
    specs: dict[str, tuple[str, str, str, int, int, str]] = {}
    for plan_ref, report_ref, transcript_ref in zip(plan_refs, report_refs, transcript_refs, strict=True):
        plan = _load_hashed_report(
            root, str(plan_ref), schema=WORKER_PLAN_SCHEMA, field="worker_plan"
        )
        try:
            key = config_hash_to_key[str(plan["config_hash"])]
        except KeyError as error:
            raise Stage0G8Error("G8_OUTPUT_PLAN_CONFIG_UNKNOWN") from error
        if (
            plan.get("parameter_envelope_ref") != parameter_refs[key]
            or plan.get("work_envelope_ref") != work_refs[key]
            or plan.get("parameter_envelope_hash") != parameters[key]["artifact_hash"]
            or plan.get("work_envelope_hash") != works[key]["artifact_hash"]
            or plan.get("environment_hash") != request.environment.environment_hash
        ):
            raise Stage0G8Error("G8_OUTPUT_PLAN_CONTROL_BINDING_INVALID")
        report = _validate_worker_report(
            root, str(report_ref), plan=plan, source=source
        )
        transcript = _load_hashed_report(
            root,
            str(transcript_ref),
            schema="stage0-g8-launch-transcript-v1",
            field="launch_transcript",
        )
        if (
            transcript.get("plan_ref") != plan_ref
            or transcript.get("return_code") != 0
            or transcript.get("timed_out") is not False
            or transcript.get("residual_compute_processes")
        ):
            raise Stage0G8Error("G8_OUTPUT_LAUNCH_TRANSCRIPT_INVALID")
        identity = (key, str(plan["mode"]), int(plan["repeat_index"]))
        if identity in reports:
            raise Stage0G8Error("G8_OUTPUT_MEASUREMENT_IDENTITY_DUPLICATE")
        reports[identity] = report
        specs[key] = (
            key,
            str(plan["model_id"]),
            str(plan["profile_id"]),
            int(plan["candidate_stage"]),
            int(plan["world_size"]),
            str(plan["precision_profile"]),
        )
    execution_keys = tuple(key for key in configs if not key.startswith("budget-"))
    controls = {
        "configs": configs,
        "config_refs": config_refs,
        "parameter_refs": parameter_refs,
        "parameter_values": parameters,
        "work_refs": work_refs,
        "work_values": works,
        "budget_ref": budget_ref,
        "budget_report": budget,
        "execution_keys": execution_keys,
        "specs": specs,
    }
    replayed_metrics = _summarize_measurements(controls, reports)
    summary_ref = canonical.get("measurement_summary_ref")
    if not isinstance(summary_ref, str):
        raise Stage0G8Error("G8_OUTPUT_SUMMARY_REF_INVALID")
    summary = _load_hashed_report(
        root, summary_ref, schema="stage0-g8-measurement-summary-v1", field="summary"
    )
    if (
        summary.get("status") != "PASS"
        or summary.get("generator_git_commit") != source.git_commit
        or summary.get("environment_hash") != request.environment.environment_hash
        or summary.get("metrics") != replayed_metrics
        or summary.get("worker_report_refs") != report_refs
    ):
        raise Stage0G8Error("G8_OUTPUT_SUMMARY_REPLAY_MISMATCH")

    preflight_ref = canonical.get("preflight_ref")
    fault_ref = canonical.get("fault_report_ref")
    postflight_ref = canonical.get("postflight_ref")
    lease_ref = canonical.get("lease_history_ref")
    if not all(isinstance(item, str) for item in (preflight_ref, fault_ref, postflight_ref, lease_ref)):
        raise Stage0G8Error("G8_OUTPUT_OPERATIONAL_REFS_INVALID")
    preflight = _load_hashed_report(
        root, preflight_ref, schema="stage0-g8-preflight-evidence-v1", field="preflight"
    )
    fault = _load_hashed_report(
        root, fault_ref, schema="stage0-g8-fault-exercise-report-v1", field="fault"
    )
    postflight = _load_hashed_report(
        root, postflight_ref, schema="stage0-g8-postflight-evidence-v1", field="postflight"
    )
    lease = _load_hashed_report(
        root, lease_ref, schema="runtime.project-gpu-lease-history.v1", field="lease"
    )
    responses = fault.get("failure_responses")
    if (
        preflight.get("preflight", {}).get("status") != "PASS"
        or preflight.get("preflight", {}).get("running_state_may_publish") is not True
        or fault.get("status") != "PASS"
        or fault.get("orphan_processes_after_exercises")
        or not isinstance(responses, list)
        or responses != [failure_response(item) for item in FailureClass]
        or fault.get("truth_writer_canary", {}).get("status") != "EXPECTED_FAILURE"
        or fault.get("stale_live_process_classification")
        != "ACTIVE_PROCESS_HEARTBEAT_STALE_DO_NOT_REAP"
        or fault.get("ssh_duplicate_launch_rejected") is not True
        or postflight.get("status") != "PASS"
        or postflight.get("residual_compute_processes")
        or postflight.get("gpu_contexts_returned_to_zero") is not True
        or lease.get("outcome") != "SUCCESS"
    ):
        raise Stage0G8Error("G8_OUTPUT_OPERATIONAL_REPORT_INVALID")

    suite = {
        "controls": controls,
        "preflight_ref": preflight_ref,
        "preflight": preflight,
        "summary_ref": summary_ref,
        "summary": summary,
        "fault_ref": fault_ref,
        "fault": fault,
        "postflight_ref": postflight_ref,
        "postflight": postflight,
        "lease_history_ref": lease_ref,
        "reports": reports,
    }
    replayed_records, replayed_reports, replayed_checked_at = _gate_payloads(
        request, root, source, suite
    )
    raw_records = canonical.get("gate_records")
    raw_gate_reports = canonical.get("gate_reports")
    if not isinstance(raw_records, Mapping) or not isinstance(raw_gate_reports, Mapping):
        raise Stage0G8Error("G8_OUTPUT_GATE_MAP_INVALID")
    for gate_id in GATE_IDS:
        gate = GateRecord.from_mapping(_mapping(raw_records.get(gate_id), field=f"gate.{gate_id}"))
        gate_report = Stage0GateReport.from_mapping(
            _mapping(raw_gate_reports.get(gate_id), field=f"gate_report.{gate_id}")
        )
        if (
            gate.to_dict() != replayed_records[gate_id].to_dict()
            or gate_report.to_dict() != replayed_reports[gate_id].to_dict()
            or gate.status is not GateStatus.PASS
            or gate_report.status.value != "PASS"
        ):
            raise Stage0G8Error("G8_OUTPUT_GATE_REPLAY_MISMATCH")
    if canonical.get("checked_at") != replayed_checked_at:
        raise Stage0G8Error("G8_OUTPUT_CHECKED_AT_MISMATCH")
    return replayed_records["stage0.G8"]


def execute_stage0_g8(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g7_recovery_index_ref: str,
) -> Stage0G8FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g7_recovery_formal_state(
        data_root=root,
        index_ref=g7_recovery_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g8_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g8-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G8Error(f"G8_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}")
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, state.environment)
    gate = validate_formal_g8_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g8_capacity": outputs["capacity_envelope"],
            "g8_operations": outputs["operations_preflight"],
            "g8_faults": outputs["fault_report"],
            "gate_stage0_g8_c": outputs["capacity_envelope"],
            "gate_stage0_g8_s4": outputs["capacity_envelope"],
            "gate_stage0_g8_s5": outputs["capacity_envelope"],
            "gate_stage0_g8": outputs["capacity_envelope"],
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids | frozenset(GATE_IDS),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    write_canonical_json(
        _logical_path(root, environment_ref, field="environment_ref"), environment.to_dict()
    )
    gate_payload = load_committed_task_artifact(
        root, outputs["capacity_envelope"], require_formal=True
    ).payload
    gate_hash = _mapping(
        _mapping(gate_payload, field="g8_payload").get("gate_records"), field="g8_gates"
    )["stage0.G8"]["artifact_hash"]
    index: dict[str, JSONValue] = {
        "schema_version": "stage0-g8-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g7_recovery_index_ref": state.index_ref,
        "g7_recovery_index_sha256": state.index_sha256,
        "g7_gate_artifact_hash": state.gate_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_ref": outputs["capacity_envelope"],
        "gate_artifact_hash": gate_hash,
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.11_test_quality_and_replay",
        "next_input_refs": list(outputs.values()),
    }
    index["artifact_hash"] = canonical_json_hash(index)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(_logical_path(root, index_ref, field="index_ref"), index)
    return Stage0G8FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def load_stage0_g8_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G8FormalState:
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g8_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g7_recovery_index_ref",
        "g7_recovery_index_sha256",
        "g7_gate_artifact_hash",
        "config_ref",
        "config_hash",
        "task_output_refs",
        "gate_ref",
        "gate_artifact_hash",
        "environment_ref",
        "environment_hash",
        "next_task_id",
        "next_input_refs",
        "artifact_hash",
    }
    declared = raw.pop("artifact_hash", None)
    if (
        set(raw) != expected - {"artifact_hash"}
        or declared != canonical_json_hash(raw)
        or raw.get("schema_version") != "stage0-g8-formalization-index-v1"
        or raw.get("generator_git_commit") != expected_git_commit
    ):
        raise Stage0G8Error("G8_STATE_INDEX_INVALID")
    config_ref = str(raw["config_ref"])
    environment_ref = str(raw["environment_ref"])
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(_logical_path(root, config_ref, field="config_ref")), field="config")
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, environment_ref, field="environment_ref")),
            field="environment",
        )
    )
    outputs = _mapping(raw.get("task_output_refs"), field="task_output_refs")
    ordered = {kind: str(outputs[kind]) for kind in config.task_definition.artifact_kinds}
    upstream = load_stage0_g7_recovery_formal_state(
        data_root=root,
        index_ref=str(raw["g7_recovery_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    request = TaskExecutionRequest(config, config.task_definition, upstream.environment)
    gate = validate_formal_g8_outputs(request, root, ordered)
    next_inputs = raw.get("next_input_refs")
    if (
        config.config_hash != raw.get("config_hash")
        or environment.environment_hash != raw.get("environment_hash")
        or not set(GATE_IDS) <= environment.passed_gate_ids
        or raw.get("gate_ref") != ordered["capacity_envelope"]
        or raw.get("gate_artifact_hash") != gate.artifact_hash
        or raw.get("next_task_id") != "stage0.11_test_quality_and_replay"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(ordered.values())
        or raw.get("g7_recovery_index_sha256") != upstream.index_sha256
        or raw.get("g7_gate_artifact_hash") != upstream.gate_artifact_hash
    ):
        raise Stage0G8Error("G8_STATE_HANDOFF_INVALID")
    return Stage0G8FormalState(
        environment=environment,
        task_output_refs=ordered,
        config=config,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate.artifact_hash,
        g7_recovery_index_ref=str(raw["g7_recovery_index_ref"]),
    )


__all__ = [
    "GATE_IDS",
    "G8SourceBinding",
    "Stage0G8Error",
    "Stage0G8FormalState",
    "Stage0G8FormalizationResult",
    "TASK_ID",
    "build_stage0_g8_config",
    "execute_stage0_g8",
    "load_stage0_g8_formal_state",
    "run_formal_g8_task",
    "validate_formal_g8_outputs",
]
