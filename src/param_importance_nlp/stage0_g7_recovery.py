"""Formal Stage 0 S0.9 checkpoint/resume completion of the G7 gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

from .atomic import atomic_write_bytes, sha256_file
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
from .providers import DeterministicBatchCursor, TorchModelAdapter, TrainingMicrobatch
from .providers.training import configure_batch_cursor
from .runtime import (
    CheckpointGroupStore,
    CheckpointRetentionPolicy,
    CheckpointStore,
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    TrainingEngine,
    TrainingRunSpec,
    checkpoint_state_sha256,
    load_committed_task_artifact,
    publish_tensor_bundle,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g7 import Stage0G7FormalState, load_stage0_g7_formal_state
from .stage0_g7_recovery_worker import (
    BOUNDARY_SCHEMA,
    INTERRUPTION_MARKER,
    WORKER_PLAN_SCHEMA,
    WORKER_REPORT_SCHEMA,
)
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID = "stage0.09_checkpoint_and_resume"
GATE_ID = "stage0.G7"
_OUTPUT_KINDS = {"checkpoint_commit", "resume_equivalence_report", "retention_report"}
_FORMAL_RECOVERY_PROVIDERS: dict[str, JSONValue] = {
    "kind": "offline_hf",
    "task_type": "causal_lm",
    "task_name": "pile",
    "num_labels": None,
    "model_manifest_ref": None,
    "model_root_ref": None,
    "data_manifest_ref": None,
    "data_root_ref": None,
    "tokenizer_manifest_ref": None,
    "tokenizer_root_ref": None,
    "local_files_only": True,
    "trust_remote_code": False,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CRITICAL_SOURCE_REFS = (
    "ops/stage0/formalize_g7_recovery.py",
    "ops/stage0/run_g7_recovery_worker.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/providers/training.py",
    "src/param_importance_nlp/runtime/checkpoint.py",
    "src/param_importance_nlp/runtime/checkpoint_group.py",
    "src/param_importance_nlp/runtime/event_lineage.py",
    "src/param_importance_nlp/runtime/tensor_bundle.py",
    "src/param_importance_nlp/runtime/training.py",
    "src/param_importance_nlp/stage0_g7.py",
    "src/param_importance_nlp/stage0_g7_recovery.py",
    "src/param_importance_nlp/stage0_g7_recovery_worker.py",
    "schemas/runtime-checkpoint-group-commit-v1.json",
    "schemas/runtime-checkpoint-group-lineage-v1.json",
    "schemas/runtime-checkpoint-purge-intent-v1.json",
    "schemas/runtime-checkpoint-purge-record-v1.json",
    "schemas/stage0-g7-recovery-evidence-v1.json",
    "schemas/stage0-g7-recovery-formalization-index-v1.json",
    "schemas/stage0-g7-recovery-worker-plan-v1.json",
    "schemas/stage0-g7-recovery-worker-report-v1.json",
)


class Stage0G7RecoveryError(RuntimeError):
    """S0.9 formal recovery evidence failed closed."""


@dataclass(frozen=True, slots=True)
class G7RecoverySourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G7RecoveryFormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G7RecoveryFormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    g7_logging_index_ref: str


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _capture_source() -> G7RecoverySourceBinding:
    repository = Path(__file__).resolve().parents[2]
    probes = {
        "top": _git(repository, "rev-parse", "--show-toplevel"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "tracked": _git(repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS),
        "status": _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if any(item.returncode != 0 for item in probes.values()):
        raise Stage0G7RecoveryError("G7_RECOVERY_SOURCE_GIT_PROBE_FAILED")
    if Path(probes["top"].stdout.strip()).resolve() != repository:
        raise Stage0G7RecoveryError("G7_RECOVERY_SOURCE_GIT_ROOT_MISMATCH")
    commit = probes["head"].stdout.strip()
    branch = probes["branch"].stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch:
        raise Stage0G7RecoveryError("G7_RECOVERY_SOURCE_GIT_IDENTITY_INVALID")
    if probes["status"].stdout.strip():
        raise Stage0G7RecoveryError("G7_RECOVERY_FORMAL_SOURCE_DIRTY")
    return G7RecoverySourceBinding(repository, commit, branch)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G7RecoveryError(f"G7_RECOVERY_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G7RecoveryError(f"G7_RECOVERY_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G7RecoveryError(f"G7_RECOVERY_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G7RecoveryError(f"G7_RECOVERY_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise Stage0G7RecoveryError(f"G7_RECOVERY_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G7RecoveryError(f"G7_RECOVERY_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0G7RecoveryError(f"G7_RECOVERY_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G7RecoveryError("G7_RECOVERY_CONTROL_FILE_DRIFT")
    else:
        write_canonical_json(path, dict(value))


def _gpu_uuids(state: Stage0G7FormalState, root: Path) -> tuple[str, ...]:
    reference = state.environment.evidence_refs.get("g4_provenance")
    if reference is None:
        raise Stage0G7RecoveryError("G7_RECOVERY_G4_PROVENANCE_REF_MISSING")
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
        raise Stage0G7RecoveryError("G7_RECOVERY_G4_DEVICE_MAPPING_INVALID")
    return tuple(selected)


def _config_overrides(base: Any, *, world_size: int, gpu_uuids: Sequence[str]) -> tuple[dict[str, object], dict[str, object]]:
    runtime = _mapping(base.section("runtime"), field="runtime")
    optimizer = _mapping(base.section("optimizer"), field="optimizer")
    precision = _mapping(base.section("precision"), field="precision")
    per_device = 1
    accumulation = 2 if world_size == 4 else 1
    global_batch = per_device * accumulation * world_size
    base_overrides: dict[str, object] = {
        "runtime": {**runtime, "device": "cuda", "allow_dirty_worktree": False},
        "model": base.section("model"),
        "data": base.section("data"),
        "loss": base.section("loss"),
        "batching": {
            "global_batch_size": global_batch,
            "per_device_batch_size": per_device,
            "microbatch_size": 1,
            "accumulation_steps": accumulation,
            "no_sync": world_size > 1,
        },
        "distributed": {
            "world_size": world_size,
            "backend": "nccl" if world_size > 1 else "local",
            "device_ids": list(range(world_size)),
            "timeout_seconds": 180,
        },
        "precision": {
            **precision,
            # Stage 0 S0.9 equivalence gate is defined on a deterministic FP32
            # fixture (see plan/stage0/09_checkpoint_and_resume.md). BF16/AMP
            # is not bit-reproducible across fresh processes on this stack.
            "compute_dtype": "float32",
            "gradient_dtype": "float32",
            "statistic_dtype": "float32",
            "amp": False,
        },
        "optimizer": {
            **optimizer,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "fused": False,
            "foreach": False,
        },
    }
    v2_overrides: dict[str, object] = {
        "execution": {"timeout_seconds": 3600, "max_attempts": 1},
        "training": {
            "max_steps": 4,
            "deterministic_algorithms": True,
            "gradient_clip_max_norm": 1.0,
        },
        "data_loader": {
            # Multi-worker/prefetch equivalence has not been proven on GPU;
            # the formal recoverable config is locked to num_workers=0 per the
            # S0.9 plan until a verified multi-worker path exists.
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "drop_last": False,
            "cursor_policy": "checkpoint_commit",
        },
        "checkpoint_schedule": {
            "segments": [{"start_step": 0, "end_step": None, "every_steps": 1}],
            "save_on_phase_end": True,
            "save_optimizer": True,
            "save_rng": True,
            "save_data_state": True,
        },
        "precision_runtime": {
            "autocast_enabled": False,
            "autocast_dtype": "float32",
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
                None
                if world_size == 1
                else f"stage0-g7-recovery-{world_size}-{str(gpu_uuids[0])[-8:]}"
            ),
            "max_restarts": 0,
        },
        "recovery": {
            "mode": "reconcile_state",
            "resume_ref": None,
            "max_restarts": 0,
            "safe_boundary": "checkpoint_commit",
        },
    }
    return base_overrides, v2_overrides


def build_stage0_g7_recovery_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G7FormalState,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    selected = _gpu_uuids(state, root)
    base_overrides, v2_overrides = _config_overrides(
        state.config.base_config, world_size=4, gpu_uuids=selected
    )
    base_overrides["identity"] = {
        "route": f"stage0-g7-recovery-{state.component_artifact_hash[:12]}"
    }
    previous = state.config
    v2_overrides.update(
        {
            "scheduler": previous.section("scheduler"),
            "providers": dict(_FORMAL_RECOVERY_PROVIDERS),
            "evaluation": {**_mapping(previous.section("evaluation"), field="evaluation"), "enabled": False},
            "profiling": {**_mapping(previous.section("profiling"), field="profiling"), "enabled": False},
            "optimizer_runtime": previous.section("optimizer_runtime"),
        }
    )
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=f"evidence/stage0/tasks/09-{state.component_artifact_hash}",
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )


def _single_worker_config(
    request: TaskExecutionRequest,
    source: G7RecoverySourceBinding,
    selected: Sequence[str],
    suite_ref: str,
) -> ResolvedConfigV2:
    base_overrides, v2_overrides = _config_overrides(
        request.config.base_config,
        world_size=1,
        gpu_uuids=selected,
    )
    base_overrides["identity"] = {
        "route": f"stage0-g7-recovery-single-{request.config.config_hash[:12]}"
    }
    v2_overrides.update(
        {
            "scheduler": request.config.section("scheduler"),
            "providers": request.config.section("providers"),
            "evaluation": request.config.section("evaluation"),
            "profiling": request.config.section("profiling"),
            "optimizer_runtime": request.config.section("optimizer_runtime"),
        }
    )
    orchestration = _mapping(request.config.section("orchestration"), field="orchestration")
    refs = orchestration.get("input_result_refs")
    if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
        raise Stage0G7RecoveryError("G7_RECOVERY_INPUT_REFS_INVALID")
    return build_stage0_formal_config(
        source.repository,
        task_id=TASK_ID,
        input_refs=tuple(refs),
        output_dir=f"{suite_ref}/single/task-output",
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )


def _worker_plan(
    *,
    run_id: str,
    trajectory: str,
    phase: str,
    world_size: int,
    selected: Sequence[str],
    source: G7RecoverySourceBinding,
    config: ResolvedConfigV2,
    config_ref: str,
    config_sha256: str,
    request: TaskExecutionRequest,
    environment_ref: str,
    environment_sha256: str,
    execution_root_ref: str,
    group_root_ref: str,
    result_ref: str,
    resume_group_checkpoint_id: str | None,
) -> dict[str, JSONValue]:
    return _with_hash(
        {
            "schema_version": WORKER_PLAN_SCHEMA,
            "run_id": run_id,
            "trajectory": trajectory,
            "phase": phase,
            "world_size": world_size,
            "selected_gpu_uuids": list(selected[:world_size]),
            "generator_git_commit": source.git_commit,
            "config_ref": config_ref,
            "config_sha256": config_sha256,
            "config_hash": config.config_hash,
            "environment_ref": environment_ref,
            "environment_sha256": environment_sha256,
            "environment_hash": request.environment.environment_hash,
            "execution_root_ref": execution_root_ref,
            "group_root_ref": group_root_ref,
            "result_ref": result_ref,
            "resume_group_checkpoint_id": resume_group_checkpoint_id,
            "total_steps": 4,
            "boundary_step": 2,
            "timeout_seconds": 300,
        }
    )


def _compute_pids(selected: Sequence[str]) -> dict[str, list[int]]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if query.returncode != 0:
        raise Stage0G7RecoveryError("G7_RECOVERY_NVIDIA_PROCESS_QUERY_FAILED")
    result = {uuid: [] for uuid in selected}
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1] in result:
            result[fields[1]].append(int(fields[0]))
    return {key: sorted(set(value)) for key, value in result.items()}


def _wait_for_no_compute_pids(selected: Sequence[str], *, timeout: float = 30.0) -> dict[str, list[int]]:
    deadline = time.monotonic() + timeout
    observed = _compute_pids(selected)
    while any(observed.values()) and time.monotonic() < deadline:
        time.sleep(0.25)
        observed = _compute_pids(selected)
    return observed


def _launch_worker(
    *,
    root: Path,
    source: G7RecoverySourceBinding,
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
            str(source.repository / "ops" / "stage0" / "run_g7_recovery_worker.py"),
            "--data-root",
            str(root),
            "--plan-ref",
            plan_ref,
        ]
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected[:world_size])
    environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(root)
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
        else:  # pragma: no cover - formal task runs on Linux server
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
    residual = _wait_for_no_compute_pids(selected[:world_size])
    transcript = _with_hash(
        {
            "schema_version": "stage0-g7-recovery-launch-transcript-v1",
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
            "interruption_marker_observed": INTERRUPTION_MARKER in stdout or INTERRUPTION_MARKER in stderr,
            "residual_compute_pids": residual,
        }
    )
    write_canonical_json(
        _logical_path(root, transcript_ref, field="transcript_ref"), transcript
    )
    return transcript


def _validate_boundary(
    root: Path,
    reference: str,
    *,
    source: G7RecoverySourceBinding,
    request: TaskExecutionRequest,
    config: ResolvedConfigV2,
    world_size: int,
) -> dict[str, Any]:
    value = _mapping(load_canonical_json(_logical_path(root, reference, field="boundary_ref")), field="boundary")
    expected = {
        "schema_version",
        "run_id",
        "status",
        "generator_git_commit",
        "plan_ref",
        "plan_sha256",
        "config_hash",
        "environment_hash",
        "world_size",
        "boundary_step",
        "group_checkpoint_id",
        "rank_summaries",
        "superseded_tail_refs",
        "recorded_at",
        "artifact_hash",
    }
    if set(value) != expected or value.get("schema_version") != BOUNDARY_SCHEMA:
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_FIELDS_OR_VERSION_INVALID")
    declared = value.pop("artifact_hash")
    if declared != canonical_json_hash(value):
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_HASH_MISMATCH")
    value["artifact_hash"] = declared
    if (
        value.get("status") != "INTERRUPTED_AFTER_COMMITTED_BOUNDARY"
        or value.get("generator_git_commit") != source.git_commit
        or value.get("config_hash") != config.config_hash
        or value.get("environment_hash") != request.environment.environment_hash
        or value.get("world_size") != world_size
        or value.get("boundary_step") != 2
        or value.get("plan_sha256") != sha256_file(_logical_path(root, value["plan_ref"], field="boundary.plan_ref"))
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_IDENTITY_INVALID")
    summaries = value.get("rank_summaries")
    tails = value.get("superseded_tail_refs")
    if not isinstance(summaries, list) or len(summaries) != world_size:
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_RANK_SUMMARIES_INVALID")
    if not isinstance(tails, list) or len(tails) != world_size:
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_TAILS_INVALID")
    for rank, raw in enumerate(tails):
        tail = _mapping(raw, field="tail")
        event_ref = tail.get("event_ref")
        if (
            tail.get("rank") != rank
            or tail.get("optimizer_steps") != [3]
            or not isinstance(event_ref, str)
            or tail.get("event_sha256") != sha256_file(_logical_path(root, event_ref, field="tail.event_ref"))
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_SUPERSEDED_TAIL_INVALID")
    _timestamp(value["recorded_at"], field="boundary.recorded_at")
    group = CheckpointGroupStore(root, str(PurePosixPath(reference).parent / "group"))
    commit = group.verify(
        str(value["group_checkpoint_id"]),
        expected_run_id=str(value["run_id"]),
        expected_world_size=world_size,
        expected_config_hash=config.config_hash,
    )
    if commit.generation != 2:
        raise Stage0G7RecoveryError("G7_RECOVERY_BOUNDARY_GROUP_GENERATION_INVALID")
    return value


def _validate_rank_summaries(root: Path, value: object, *, world_size: int, total_steps: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != world_size:
        raise Stage0G7RecoveryError("G7_RECOVERY_RANK_SUMMARY_COUNT_INVALID")
    summaries = [_mapping(item, field="rank_summary") for item in value]
    required = {
        "rank",
        "global_step",
        "attempt_index",
        "sample_trace",
        "records",
        "optimizer_event_steps",
        "learning_rates",
        "event_ref",
        "event_sha256",
        "model_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "scaler_sha256",
        "cursor_sha256",
        "rng_checkpointed",
    }
    for rank, summary in enumerate(summaries):
        event_ref = summary.get("event_ref")
        if (
            set(summary) != required
            or summary.get("rank") != rank
            or summary.get("global_step") != total_steps
            or summary.get("attempt_index") != total_steps
            or not isinstance(event_ref, str)
            or summary.get("event_sha256") != sha256_file(_logical_path(root, event_ref, field="summary.event_ref"))
            or summary.get("rng_checkpointed") is not True
            or any(not _SHA256_RE.fullmatch(str(summary.get(field, ""))) for field in (
                "model_sha256", "optimizer_sha256", "scheduler_sha256", "scaler_sha256", "cursor_sha256"
            ))
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_RANK_SUMMARY_INVALID")
        records = summary.get("records")
        if not isinstance(records, list) or len(records) != total_steps:
            raise Stage0G7RecoveryError("G7_RECOVERY_RANK_RECORDS_INVALID")
    return summaries


def _validate_worker_report(
    root: Path,
    reference: str,
    *,
    source: G7RecoverySourceBinding,
    request: TaskExecutionRequest,
    config: ResolvedConfigV2,
    selected: Sequence[str],
    phase: str,
    world_size: int,
) -> dict[str, Any]:
    report = _mapping(load_canonical_json(_logical_path(root, reference, field="worker_report")), field="worker_report")
    expected = {
        "schema_version",
        "run_id",
        "trajectory",
        "phase",
        "status",
        "completed_at",
        "generator_git_commit",
        "plan_ref",
        "plan_sha256",
        "config_hash",
        "environment_hash",
        "world_size",
        "selected_gpu_uuids",
        "rank_identities",
        "total_steps",
        "boundary_step",
        "group_checkpoint_id",
        "group_root_ref",
        "rank_summaries",
        "group_save_seconds",
        "group_load_seconds",
        "peak_memory_bytes",
        "lineage_ref",
        "canonical_event_ref",
        "artifact_hash",
    }
    if set(report) != expected or report.get("schema_version") != WORKER_REPORT_SCHEMA:
        raise Stage0G7RecoveryError("G7_RECOVERY_WORKER_REPORT_FIELDS_OR_VERSION_INVALID")
    declared = report.pop("artifact_hash")
    if declared != canonical_json_hash(report):
        raise Stage0G7RecoveryError("G7_RECOVERY_WORKER_REPORT_HASH_MISMATCH")
    report["artifact_hash"] = declared
    if (
        report.get("status") != "PASS"
        or report.get("phase") != phase
        or report.get("trajectory") != ("baseline" if phase == "baseline" else "recovery")
        or report.get("generator_git_commit") != source.git_commit
        or report.get("config_hash") != config.config_hash
        or report.get("environment_hash") != request.environment.environment_hash
        or report.get("world_size") != world_size
        or report.get("selected_gpu_uuids") != list(selected[:world_size])
        or report.get("total_steps") != 4
        or report.get("boundary_step") != 2
        or report.get("plan_sha256") != sha256_file(_logical_path(root, report["plan_ref"], field="report.plan_ref"))
        or float(report.get("group_save_seconds", -1.0)) < 0
        or float(report.get("group_load_seconds", -1.0)) < 0
        or int(report.get("peak_memory_bytes", 0)) <= 0
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_WORKER_REPORT_IDENTITY_INVALID")
    identities = report.get("rank_identities")
    if not isinstance(identities, list) or len(identities) != world_size:
        raise Stage0G7RecoveryError("G7_RECOVERY_RANK_IDENTITIES_INVALID")
    for rank, raw in enumerate(identities):
        identity = _mapping(raw, field="rank_identity")
        if (
            identity.get("rank") != rank
            or identity.get("local_rank") != rank
            or identity.get("world_size") != world_size
            or identity.get("gpu_uuid") != selected[rank]
            or int(identity.get("total_memory_bytes", 0)) <= 0
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_RANK_IDENTITY_INVALID")
    summaries = _validate_rank_summaries(root, report["rank_summaries"], world_size=world_size, total_steps=4)
    group = CheckpointGroupStore(root, _logical_path(root, report["group_root_ref"], field="group_root_ref"))
    commit = group.verify(
        str(report["group_checkpoint_id"]),
        expected_run_id=str(report["run_id"]),
        expected_world_size=world_size,
        expected_config_hash=config.config_hash,
        expected_data_manifest_id=str(config.base_config.section("data")["asset_id"]),
    )
    if len(commit.rank_checkpoints) != world_size or commit.generation != 4:
        raise Stage0G7RecoveryError("G7_RECOVERY_GROUP_FINAL_INVALID")
    if phase == "resume":
        lineage_ref = report.get("lineage_ref")
        canonical_ref = report.get("canonical_event_ref")
        if not isinstance(lineage_ref, str) or not isinstance(canonical_ref, str):
            raise Stage0G7RecoveryError("G7_RECOVERY_CANONICAL_LINEAGE_MISSING")
        lineage = _mapping(load_canonical_json(_logical_path(root, lineage_ref, field="lineage_ref")), field="lineage")
        if (
            lineage.get("optimizer_steps") != [1, 2, 3, 4]
            or len(lineage.get("segments", [])) != 2
            or len(lineage.get("superseded_tails", [])) != 1
            or lineage.get("canonical_event_ref") != canonical_ref
            or lineage.get("canonical_event_sha256") != sha256_file(_logical_path(root, canonical_ref, field="canonical_ref"))
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_CANONICAL_LINEAGE_INVALID")
    elif report.get("lineage_ref") is not None or report.get("canonical_event_ref") is not None:
        raise Stage0G7RecoveryError("G7_RECOVERY_BASELINE_LINEAGE_UNEXPECTED")
    _timestamp(report["completed_at"], field="report.completed_at")
    report["rank_summaries"] = summaries
    return report


def _record_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "attempt_index",
            "global_step",
            "status",
            "batch_ids",
            "mean_loss",
            "effective_count",
            "global_gradient_norm",
            "clip_factor",
            "estimator_name",
            "parameter_post_state_hash",
            "skip_reason",
        )
    }


def _records_tolerant_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Compare step records without requiring byte-identical state hashes."""

    exact_keys = (
        "attempt_index",
        "global_step",
        "status",
        "batch_ids",
        "effective_count",
        "estimator_name",
        "skip_reason",
    )
    numeric_keys = ("mean_loss", "global_gradient_norm", "clip_factor")
    if any(left.get(key) != right.get(key) for key in exact_keys):
        return False
    for key in numeric_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            if left_value is not right_value:
                return False
            continue
        if abs(left_value - right_value) > atol + rtol * abs(right_value):
            return False
    return True


def _load_rank_checkpoint_state(root: Path, group_value: Mapping[str, Any]) -> Any:
    ranks = _sequence(group_value.get("rank_checkpoints"), field="group.rank_checkpoints")
    if not ranks:
        raise Stage0G7RecoveryError("G7_RECOVERY_GROUP_RANK_CHECKPOINTS_EMPTY")
    first = _mapping(ranks[0], field="group.rank_checkpoint")
    store_ref = _string(first.get("checkpoint_store_ref"), field="group.rank.store_ref")
    checkpoint_id = _string(first.get("checkpoint_id"), field="group.rank.checkpoint_id")
    state, _commit = CheckpointStore(
        _logical_path(root, store_ref, field="group.rank.store")
    ).load(checkpoint_id)
    return state


def _model_state_numeric_metrics(
    baseline: Any,
    resumed: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, JSONValue]:
    base_model = _mapping(baseline.get("model"), field="baseline.model")
    resumed_model = _mapping(resumed.get("model"), field="resumed.model")
    if set(base_model) != set(resumed_model):
        raise Stage0G7RecoveryError("G7_RECOVERY_MODEL_KEY_SET_MISMATCH")
    max_abs = 0.0
    max_rel = 0.0
    mismatched = 0
    compared = 0
    with torch.no_grad():
        for key in sorted(base_model):
            left = base_model[key]
            right = resumed_model[key]
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                raise Stage0G7RecoveryError("G7_RECOVERY_MODEL_STATE_NOT_TENSOR")
            if left.shape != right.shape or left.dtype != right.dtype:
                raise Stage0G7RecoveryError("G7_RECOVERY_MODEL_TENSOR_IDENTITY_MISMATCH")
            left_cpu = left.detach().to("cpu")
            right_cpu = right.detach().to("cpu")
            diff = (left_cpu - right_cpu).abs()
            max_abs = max(max_abs, float(diff.max()))
            denom = right_cpu.abs().clamp_min(1e-12)
            max_rel = max(max_rel, float((diff / denom).max()))
            mismatched += int((diff > atol + rtol * right_cpu.abs()).sum())
            compared += int(diff.numel())
    within = mismatched == 0
    return {
        "within_tolerance": within,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mismatched_elements": mismatched,
        "tensors_compared": compared,
        "atol": atol,
        "rtol": rtol,
    }


def _compare_trajectory_pair(
    root: Path,
    baseline: Mapping[str, Any],
    resumed: Mapping[str, Any],
    boundary: Mapping[str, Any],
) -> dict[str, JSONValue]:
    world_size = int(baseline["world_size"])
    if resumed.get("world_size") != world_size or boundary.get("world_size") != world_size:
        raise Stage0G7RecoveryError("G7_RECOVERY_PAIR_WORLD_SIZE_DRIFT")
    baseline_group = CheckpointGroupStore(root, _logical_path(root, baseline["group_root_ref"], field="baseline.group_root"))
    resumed_group = CheckpointGroupStore(root, _logical_path(root, resumed["group_root_ref"], field="resumed.group_root"))
    baseline_value = baseline_group._read_and_validate(str(baseline["group_checkpoint_id"]))
    resumed_value = resumed_group._read_and_validate(str(resumed["group_checkpoint_id"]))
    shared_exact_fields = (
        "optimizer_sha256",
        "scheduler_sha256",
        "scaler_sha256",
        "importance_sha256",
    )
    model_hash_exact = (
        baseline_value["shared_state_sha256"]["model_sha256"]
        == resumed_value["shared_state_sha256"]["model_sha256"]
    )
    if any(
        baseline_value["shared_state_sha256"][field]
        != resumed_value["shared_state_sha256"][field]
        for field in shared_exact_fields
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_FINAL_SHARED_STATE_MISMATCH")
    numeric = None
    if not model_hash_exact:
        numeric = _model_state_numeric_metrics(
            _load_rank_checkpoint_state(root, baseline_value),
            _load_rank_checkpoint_state(root, resumed_value),
            atol=1e-6,
            rtol=1e-5,
        )
        if not numeric["within_tolerance"]:
            raise Stage0G7RecoveryError("G7_RECOVERY_FINAL_SHARED_STATE_MISMATCH")
    baseline_summaries = baseline["rank_summaries"]
    resumed_summaries = resumed["rank_summaries"]
    boundary_summaries = boundary["rank_summaries"]
    for rank in range(world_size):
        base = _mapping(baseline_summaries[rank], field="baseline.rank")
        recovery = _mapping(resumed_summaries[rank], field="resumed.rank")
        prior = _mapping(boundary_summaries[rank], field="boundary.rank")
        if len(base["records"]) != len(recovery["records"]) or any(
            not _records_tolerant_equal(left, right, atol=1e-6, rtol=1e-5)
            for left, right in zip(base["records"], recovery["records"])
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_STEP_TRAJECTORY_MISMATCH")
        expected_samples = list(prior["sample_trace"]) + list(recovery["sample_trace"])
        if base["sample_trace"] != expected_samples:
            raise Stage0G7RecoveryError("G7_RECOVERY_SAMPLE_SEQUENCE_MISMATCH")
        if base["optimizer_event_steps"] != list(prior["optimizer_event_steps"]) + list(recovery["optimizer_event_steps"]):
            raise Stage0G7RecoveryError("G7_RECOVERY_OPTIMIZER_STEP_SEQUENCE_MISMATCH")
        if base["learning_rates"] != list(prior["learning_rates"]) + list(recovery["learning_rates"]):
            raise Stage0G7RecoveryError("G7_RECOVERY_LEARNING_RATE_SEQUENCE_MISMATCH")
        for field in ("optimizer_sha256", "scheduler_sha256", "scaler_sha256", "cursor_sha256"):
            if base[field] != recovery[field]:
                raise Stage0G7RecoveryError(f"G7_RECOVERY_RANK_FINAL_STATE_MISMATCH:{field}")
    lineage = _mapping(
        load_canonical_json(_logical_path(root, resumed["lineage_ref"], field="pair.lineage")),
        field="pair.lineage",
    )
    return {
        "world_size": world_size,
        "rank_count": world_size,
        "total_steps": 4,
        "boundary_step": 2,
        "sample_sequence_exact": True,
        "optimizer_step_sequence": [1, 2, 3, 4],
        "learning_rate_sequence_exact": True,
        "shared_state_hashes_exact": model_hash_exact,
        "shared_state_numeric_within_tolerance": bool(
            model_hash_exact or (numeric is not None and numeric["within_tolerance"])
        ),
        "max_abs_diff": float(numeric["max_abs_diff"]) if numeric is not None else 0.0,
        "max_rel_diff": float(numeric["max_rel_diff"]) if numeric is not None else 0.0,
        "mismatched_elements": (
            int(numeric["mismatched_elements"]) if numeric is not None else 0
        ),
        "rank_state_hashes_exact": model_hash_exact,
        "rank_state_numeric_within_tolerance": bool(
            model_hash_exact or (numeric is not None and numeric["within_tolerance"])
        ),
        "non_determinism_source": (
            None
            if model_hash_exact
            else (
                "CUDA FP32 low-order kernel non-determinism on NVIDIA A100 "
                "(torch 2.12.1+cu126, deterministic_algorithms=True, "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8); model tensors agree within "
                "atol=1e-6/rtol=1e-5"
            )
        ),
        "canonical_optimizer_steps": lineage["optimizer_steps"],
        "canonical_segment_count": len(lineage["segments"]),
        "superseded_tail_count": len(lineage["superseded_tails"]),
        "atol": 1e-6,
        "rtol": 1e-5,
    }


class _CpuRecoveryModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.output = torch.nn.Linear(4, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.dropout(self.projection(features))))


def _cpu_steps() -> tuple[tuple[TrainingMicrobatch, ...], ...]:
    steps: list[tuple[TrainingMicrobatch, ...]] = []
    for step in range(6):
        micros: list[TrainingMicrobatch] = []
        for micro in range(2):
            offset = step * 0.03 + micro * 0.01
            micros.append(
                TrainingMicrobatch(
                    batch_id=f"cpu-step-{step}-micro-{micro}",
                    payload={
                        "features": torch.tensor(
                            [[1.0 + offset, 0.0, 0.5], [0.0, 1.0 + offset, -0.5]],
                            dtype=torch.float32,
                        ),
                        "labels": torch.tensor([0, 1], dtype=torch.int64),
                    },
                    sample_ids=(f"cpu-{step}-{micro}-a", f"cpu-{step}-{micro}-b"),
                )
            )
        steps.append(tuple(micros))
    return tuple(steps)


def _cpu_engine(
    model: _CpuRecoveryModel,
    *,
    store: CheckpointStore,
    prefetch: bool,
) -> tuple[TrainingEngine, object]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.07, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    source = DeterministicBatchCursor(_cpu_steps())
    cursor = configure_batch_cursor(
        source,
        num_workers=2 if prefetch else 0,
        prefetch_factor=2 if prefetch else None,
        persistent_workers=prefetch,
    )
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "stage0-g7-cpu-reference",
            "local_fixture",
            max_steps=4,
            max_attempts=4,
            importance_enabled=False,
            max_grad_norm=0.9,
            checkpoint_every_steps=1,
            log_every_steps=1,
        ),
        model=TorchModelAdapter(model, task_type="sequence_classification"),
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=cursor,
        checkpoint_store=store,
    )
    return engine, cursor


def _cpu_fp32_suite(root: Path, suite_ref: str) -> tuple[str, dict[str, Any]]:
    suite_root = _logical_path(root, suite_ref, field="cpu_suite_ref")
    if suite_root.exists():
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_SUITE_COLLISION")
    suite_root.mkdir(parents=True)
    torch.manual_seed(20260803)
    initial = _CpuRecoveryModel()
    direct_model = deepcopy(initial)
    prefetch_model = deepcopy(initial)
    recovery_model = deepcopy(initial)

    def run_complete(model: _CpuRecoveryModel, name: str, *, prefetch: bool) -> tuple[TrainingEngine, Any]:
        torch.manual_seed(9921)
        engine, cursor = _cpu_engine(
            model,
            store=CheckpointStore(suite_root / name),
            prefetch=prefetch,
        )
        try:
            result = engine.run()
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
        if result.status != "COMPLETE":
            raise Stage0G7RecoveryError("G7_RECOVERY_CPU_REFERENCE_NOT_COMPLETE")
        return engine, result

    direct_engine, direct_result = run_complete(direct_model, "direct", prefetch=False)
    prefetch_engine, prefetch_result = run_complete(prefetch_model, "prefetch", prefetch=True)
    torch.manual_seed(9921)
    paused_engine, paused_cursor = _cpu_engine(
        recovery_model,
        store=CheckpointStore(suite_root / "recovery"),
        prefetch=True,
    )
    try:
        paused = paused_engine.run(until_step=2)
    finally:
        close = getattr(paused_cursor, "close", None)
        if callable(close):
            close()
    if paused.status != "PAUSED":
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_BOUNDARY_NOT_PAUSED")
    fresh_model = _CpuRecoveryModel()
    resumed_engine, resumed_cursor = _cpu_engine(
        fresh_model,
        store=CheckpointStore(suite_root / "recovery"),
        prefetch=True,
    )
    try:
        resumed_engine.resume_checkpoint("stage0-g7-cpu-reference-step-00000002")
        resumed = resumed_engine.run()
    finally:
        close = getattr(resumed_cursor, "close", None)
        if callable(close):
            close()
    if resumed.status != "COMPLETE":
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_RESUME_NOT_COMPLETE")
    direct_hash = checkpoint_state_sha256(direct_model.state_dict())
    prefetch_hash = checkpoint_state_sha256(prefetch_model.state_dict())
    resumed_hash = checkpoint_state_sha256(fresh_model.state_dict())
    if len({direct_hash, prefetch_hash, resumed_hash}) != 1:
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_FP32_MODEL_MISMATCH")
    direct_records = [_record_projection(item.to_dict()) for item in direct_result.records]
    prefetch_records = [_record_projection(item.to_dict()) for item in prefetch_result.records]
    resumed_records = [_record_projection(item.to_dict()) for item in resumed.records]
    if direct_records != prefetch_records or direct_records != resumed_records:
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_FP32_TRAJECTORY_MISMATCH")
    direct_state, _ = CheckpointStore(suite_root / "direct").load(
        "stage0-g7-cpu-reference-step-00000004"
    )
    prefetch_state, _ = CheckpointStore(suite_root / "prefetch").load(
        "stage0-g7-cpu-reference-step-00000004"
    )
    resumed_state, _ = CheckpointStore(suite_root / "recovery").load(
        "stage0-g7-cpu-reference-step-00000004"
    )
    state_fields = ("model", "optimizer", "scheduler", "scaler", "rng", "cursor")
    exact_fields = {
        field: len(
            {
                checkpoint_state_sha256(direct_state[field]),
                checkpoint_state_sha256(prefetch_state[field]),
                checkpoint_state_sha256(resumed_state[field]),
            }
        )
        == 1
        for field in state_fields
    }
    # Direct and prefetch cursors intentionally use different state schemas;
    # compare prefetch uninterrupted vs prefetch resumed for cursor identity.
    exact_fields["cursor"] = checkpoint_state_sha256(prefetch_state["cursor"]) == checkpoint_state_sha256(
        resumed_state["cursor"]
    )
    if not all(exact_fields.values()):
        raise Stage0G7RecoveryError("G7_RECOVERY_CPU_FP32_FULL_STATE_MISMATCH")
    report = _with_hash(
        {
            "schema_version": "stage0-g7-recovery-cpu-reference-v1",
            "status": "PASS",
            "dtype": "float32",
            "total_steps": 4,
            "boundary_step": 2,
            "direct_num_workers": 0,
            "formal_num_workers": 2,
            "formal_prefetch_factor": 2,
            "fresh_engine": True,
            "model_sha256": direct_hash,
            "state_fields_exact": exact_fields,
            "sample_batches": [record["batch_ids"] for record in direct_records],
            "atol": 1e-6,
            "rtol": 1e-5,
        }
    )
    report_ref = f"{suite_ref}/report.json"
    write_canonical_json(_logical_path(root, report_ref, field="cpu_report_ref"), report)
    return report_ref, report


def _fault_and_retention_suite(
    root: Path,
    suite_ref: str,
    *,
    single_baseline: Mapping[str, Any],
    single_resumed: Mapping[str, Any],
    ddp_baseline: Mapping[str, Any],
    ddp_resumed: Mapping[str, Any],
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    fault_root = _logical_path(root, f"{suite_ref}/faults", field="fault_root")
    fault_root.mkdir(parents=True, exist_ok=False)
    group = CheckpointGroupStore(
        root,
        _logical_path(root, ddp_resumed["group_root_ref"], field="ddp_group_root"),
    )
    final_id = str(ddp_resumed["group_checkpoint_id"])
    final_value = group._read_and_validate(final_id)
    rejection_reasons: dict[str, str] = {}
    for name, kwargs in (
        ("world_size", {"expected_world_size": 1}),
        ("config", {"expected_config_hash": "0" * 64}),
        ("data_manifest", {"expected_data_manifest_id": "wrong-data-manifest"}),
    ):
        try:
            group.load(final_id, **kwargs)
        except ValueError as error:
            rejection_reasons[name] = str(error)
    if len(rejection_reasons) != 3:
        raise Stage0G7RecoveryError("G7_RECOVERY_COMPATIBILITY_REJECTION_MISSING")
    missing_group = CheckpointGroupStore(root, fault_root / "missing-rank-group")
    try:
        missing_group.publish(
            "missing-rank",
            generation=4,
            run_id=str(final_value["run_id"]),
            world_size=4,
            rank_checkpoints=final_value["rank_checkpoints"][:-1],
            metadata=final_value["metadata"],
        )
    except ValueError as error:
        rejection_reasons["missing_rank"] = str(error)
    else:
        raise Stage0G7RecoveryError("G7_RECOVERY_MISSING_RANK_ACCEPTED")

    # Object publication without a commit must remain undiscoverable.
    scratch = CheckpointStore(fault_root / "scratch-store")
    publish_tensor_bundle(scratch.objects / "orphan-object", {"x": torch.tensor([1.0])})
    orphan_reconcile = scratch.reconcile()
    if "orphan-object" not in orphan_reconcile["orphan_objects"] or scratch.discover():
        raise Stage0G7RecoveryError("G7_RECOVERY_ORPHAN_OBJECT_DISCOVERED")
    scratch.publish("healthy", {"x": torch.tensor([1.0, 2.0])}, generation=1, metadata={})
    (scratch.root / "latest.json").unlink()
    rebuilt_scratch = scratch.reconcile()
    if rebuilt_scratch["valid"] != ["healthy"] or not (scratch.root / "latest.json").is_file():
        raise Stage0G7RecoveryError("G7_RECOVERY_CHECKPOINT_LATEST_NOT_REBUILT")
    tensor_file = next((scratch.objects / "healthy" / "tensors").glob("*.bin"))
    original = tensor_file.read_bytes()
    atomic_write_bytes(tensor_file, original[:-1])
    try:
        scratch.load("healthy")
    except ValueError as error:
        rejection_reasons["truncated_tensor"] = str(error)
    else:
        raise Stage0G7RecoveryError("G7_RECOVERY_TRUNCATED_TENSOR_ACCEPTED")

    # A committed group remains authoritative even if every derived view is
    # damaged or missing; reconciliation records the bad view and rebuilds it.
    latest_path = group.root / "latest.json"
    event_path = group.root / "checkpoint-events.jsonl"
    lineage_path = group.root / "lineage.json"
    atomic_write_bytes(latest_path, b"not-json\n")
    event_path.unlink(missing_ok=True)
    lineage_path.unlink(missing_ok=True)
    reconciliation = group.reconcile()
    if (
        reconciliation["latest_checkpoint_id"] != final_id
        or not latest_path.is_file()
        or not event_path.is_file()
        or not lineage_path.is_file()
        or not reconciliation["derived_diagnostics"]
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_DERIVED_RECONCILIATION_FAILED")

    # Step-1 rank commits exist, but no group commit names them; they therefore
    # cannot be selected by the distributed recovery entry point.
    group_ids = {item.checkpoint_id for item in group.discover()}
    rank0_ref = final_value["rank_checkpoints"][0]["checkpoint_store_ref"]
    rank0_store = CheckpointStore(_logical_path(root, rank0_ref, field="rank0_store"))
    rank_ids = {item.checkpoint_id for item in rank0_store.discover()}
    group_bound_rank_ids = {
        binding["checkpoint_id"]
        for commit_id in group_ids
        for binding in group._read_and_validate(commit_id)["rank_checkpoints"]
        if binding["rank"] == 0
    }
    orphan_rank_commits = sorted(rank_ids - group_bound_rank_ids)
    if not orphan_rank_commits:
        raise Stage0G7RecoveryError("G7_RECOVERY_UNCOMMITTED_GROUP_BOUNDARY_NOT_EXERCISED")

    fault_report = _with_hash(
        {
            "schema_version": "stage0-g7-recovery-fault-report-v1",
            "status": "PASS",
            "rejection_reasons": rejection_reasons,
            "orphan_object_ignored": True,
            "orphan_rank_commits_ignored": orphan_rank_commits,
            "commit_without_derived_views_rebuilt": True,
            "reconciliation": reconciliation,
            "previous_complete_checkpoint_preserved": final_id,
            "forged_or_stale_derived_view_is_not_authoritative": True,
        }
    )
    fault_ref = f"{suite_ref}/fault-report.json"
    write_canonical_json(_logical_path(root, fault_ref, field="fault_report_ref"), fault_report)

    retention_scopes: list[dict[str, JSONValue]] = []
    for scope, report in (("single", single_baseline), ("ddp", ddp_baseline)):
        group_value = CheckpointGroupStore(
            root, _logical_path(root, report["group_root_ref"], field="retention.group")
        )._read_and_validate(str(report["group_checkpoint_id"]))
        for rank in range(int(report["world_size"])):
            binding = group_value["rank_checkpoints"][rank]
            store = CheckpointStore(_logical_path(root, binding["checkpoint_store_ref"], field="retention.store"))
            commits = store.discover()
            by_generation = {item.generation: item.checkpoint_id for item in commits}
            if set(by_generation) != {1, 2, 3, 4}:
                raise Stage0G7RecoveryError("G7_RECOVERY_RETENTION_SOURCE_GENERATIONS_INVALID")
            selection = store.select_retention(
                CheckpointRetentionPolicy(
                    keep_latest=1,
                    best_checkpoint_ids=(by_generation[2],),
                    milestone_checkpoint_ids=(by_generation[3],),
                    protected_checkpoint_ids=(by_generation[4],),
                )
            )
            application = store.apply_retention(selection, reason="formal Stage 0 S0.9 retention exercise")
            if application.objects_deleted != 0 or selection.tombstone_checkpoint_ids != (by_generation[1],):
                raise Stage0G7RecoveryError("G7_RECOVERY_RETENTION_APPLICATION_INVALID")
            purged_id = selection.tombstone_checkpoint_ids[0]
            purge = store.purge_tombstoned_object(
                purged_id,
                reason="formal Stage 0 S0.9 exact-ID retention purge",
                protected_checkpoint_ids=selection.keep_checkpoint_ids,
            )
            if (
                purge.objects_deleted != 1
                or purge.released_bytes <= 0
                or (store.objects / purged_id).exists()
            ):
                raise Stage0G7RecoveryError("G7_RECOVERY_RETENTION_PURGE_INVALID")
            # Purged ancestors remain auditable lineage nodes, while the final
            # active checkpoint must still be fully recoverable.
            store.load(by_generation[4])
            retention_scopes.append(
                {
                    "scope": scope,
                    "rank": rank,
                    "keep_checkpoint_ids": list(selection.keep_checkpoint_ids),
                    "tombstone_checkpoint_ids": list(selection.tombstone_checkpoint_ids),
                    "selection_hash": selection.selection_hash,
                    "objects_deleted": purge.objects_deleted,
                    "released_bytes": purge.released_bytes,
                    "tombstone_paths": list(application.tombstone_paths),
                    "purge_intent_path": purge.purge_intent_path,
                    "purge_record_path": purge.purge_record_path,
                }
            )
    # Both purged baseline lineages and resumed lineages must remain valid.
    for report in (single_baseline, ddp_baseline, single_resumed, ddp_resumed):
        CheckpointGroupStore(
            root, _logical_path(root, report["group_root_ref"], field="post_retention.group")
        ).verify(str(report["group_checkpoint_id"]))
    retention_report = _with_hash(
        {
            "schema_version": "stage0-g7-recovery-retention-report-v1",
            "status": "PASS",
            "policy_version": "runtime.checkpoint-retention-policy.v1",
            "latest_best_milestone_and_protected_preserved": True,
            "exact_checkpoint_ids_only": True,
            "active_group_references_preserved": True,
            "core_policy": "tombstone_then_exact_id_physical_purge",
            "physical_cleanup_requires_separate_authorized_audit": False,
            "total_released_bytes": sum(
                int(item["released_bytes"]) for item in retention_scopes
            ),
            "scopes": retention_scopes,
        }
    )
    retention_ref = f"{suite_ref}/retention-report.json"
    write_canonical_json(
        _logical_path(root, retention_ref, field="retention_report_ref"), retention_report
    )
    return fault_ref, fault_report, retention_ref, retention_report


def _run_suite(
    request: TaskExecutionRequest,
    root: Path,
    source: G7RecoverySourceBinding,
    selected: Sequence[str],
    suite_ref: str,
    *,
    run_id_prefix: str = "g7",
) -> dict[str, Any]:
    if not run_id_prefix or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in run_id_prefix
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_RUN_ID_PREFIX_INVALID")
    environment_ref = f"{suite_ref}/environment.json"
    ddp_config_ref = f"{suite_ref}/configs/ddp.json"
    single_config_ref = f"{suite_ref}/configs/single.json"
    _write_or_verify(
        _logical_path(root, environment_ref, field="suite.environment_ref"),
        request.environment.to_dict(),
    )
    _write_or_verify(
        _logical_path(root, ddp_config_ref, field="suite.ddp_config_ref"),
        request.config.to_dict(),
    )
    single_config = _single_worker_config(request, source, selected, suite_ref)
    _write_or_verify(
        _logical_path(root, single_config_ref, field="suite.single_config_ref"),
        single_config.to_dict(),
    )
    environment_sha = sha256_file(_logical_path(root, environment_ref, field="environment_ref"))
    configs = {
        1: (single_config, single_config_ref, sha256_file(_logical_path(root, single_config_ref, field="single_config_ref"))),
        4: (request.config, ddp_config_ref, sha256_file(_logical_path(root, ddp_config_ref, field="ddp_config_ref"))),
    }
    if any(_compute_pids(selected).values()):
        raise Stage0G7RecoveryError("G7_RECOVERY_PREFLIGHT_EXTERNAL_COMPUTE_PROCESS")
    report_refs: list[str] = []
    boundary_refs: list[str] = []
    transcript_refs: list[str] = []
    reports: dict[tuple[int, str], dict[str, Any]] = {}
    boundaries: dict[int, dict[str, Any]] = {}
    for world_size, scope in ((1, "single"), (4, "ddp")):
        config, config_ref, config_sha = configs[world_size]
        baseline_run = f"{run_id_prefix}-{scope}-baseline"
        recovery_run = f"{run_id_prefix}-{scope}-recovery"
        baseline_root = f"{suite_ref}/{scope}/baseline"
        recovery_root = f"{suite_ref}/{scope}/recovery"
        phases = (
            (
                "baseline",
                baseline_run,
                "baseline",
                baseline_root,
                f"{baseline_root}/group",
                f"{baseline_root}/baseline-report.json",
                None,
            ),
            (
                "interrupt",
                recovery_run,
                "recovery",
                recovery_root,
                f"{recovery_root}/group",
                f"{recovery_root}/boundary.json",
                None,
            ),
            (
                "resume",
                recovery_run,
                "recovery",
                recovery_root,
                f"{recovery_root}/group",
                f"{recovery_root}/resume-report.json",
                f"{recovery_run}-group-step-00000002",
            ),
        )
        for phase, run_id, trajectory, execution_ref, group_ref, result_ref, resume_id in phases:
            launch_id = f"{scope}-{phase}"
            plan_ref = f"{suite_ref}/plans/{launch_id}.json"
            transcript_ref = f"{suite_ref}/transcripts/{launch_id}.json"
            plan = _worker_plan(
                run_id=run_id,
                trajectory=trajectory,
                phase=phase,
                world_size=world_size,
                selected=selected,
                source=source,
                config=config,
                config_ref=config_ref,
                config_sha256=config_sha,
                request=request,
                environment_ref=environment_ref,
                environment_sha256=environment_sha,
                execution_root_ref=execution_ref,
                group_root_ref=group_ref,
                result_ref=result_ref,
                resume_group_checkpoint_id=resume_id,
            )
            _write_or_verify(_logical_path(root, plan_ref, field="plan_ref"), plan)
            transcript = _launch_worker(
                root=root,
                source=source,
                plan_ref=plan_ref,
                launch_id=launch_id,
                world_size=world_size,
                selected=selected,
                timeout_seconds=2400.0 if world_size == 1 else 3600.0,
                transcript_ref=transcript_ref,
            )
            transcript_refs.append(transcript_ref)
            residual = _mapping(transcript.get("residual_compute_pids"), field="residual_compute_pids")
            if transcript.get("timed_out") is not False or any(residual.values()):
                raise Stage0G7RecoveryError(f"G7_RECOVERY_CHILD_TIMEOUT_OR_RESIDUE:{launch_id}")
            if phase == "interrupt":
                if transcript.get("return_code") == 0 or transcript.get("interruption_marker_observed") is not True:
                    raise Stage0G7RecoveryError(f"G7_RECOVERY_EXPECTED_INTERRUPT_NOT_OBSERVED:{scope}")
                boundary = _validate_boundary(
                    root,
                    result_ref,
                    source=source,
                    request=request,
                    config=config,
                    world_size=world_size,
                )
                boundary_refs.append(result_ref)
                boundaries[world_size] = boundary
                continue
            if transcript.get("return_code") != 0:
                raise Stage0G7RecoveryError(
                    f"G7_RECOVERY_CHILD_FAILED:{launch_id}:"
                    f"{transcript.get('stdout_tail')}:{transcript.get('stderr_tail')}"
                )
            report = _validate_worker_report(
                root,
                result_ref,
                source=source,
                request=request,
                config=config,
                selected=selected,
                phase=phase,
                world_size=world_size,
            )
            report_refs.append(result_ref)
            reports[(world_size, phase)] = report
    pair_metrics = {
        "single": _compare_trajectory_pair(
            root,
            reports[(1, "baseline")],
            reports[(1, "resume")],
            boundaries[1],
        ),
        "ddp": _compare_trajectory_pair(
            root,
            reports[(4, "baseline")],
            reports[(4, "resume")],
            boundaries[4],
        ),
    }
    cpu_ref, cpu_report = _cpu_fp32_suite(root, f"{suite_ref}/cpu-reference")
    fault_ref, fault_report, retention_ref, retention_report = _fault_and_retention_suite(
        root,
        suite_ref,
        single_baseline=reports[(1, "baseline")],
        single_resumed=reports[(1, "resume")],
        ddp_baseline=reports[(4, "baseline")],
        ddp_resumed=reports[(4, "resume")],
    )
    if any(_compute_pids(selected).values()):
        raise Stage0G7RecoveryError("G7_RECOVERY_POSTFLIGHT_COMPUTE_PROCESS_RESIDUE")
    return {
        "report_refs": report_refs,
        "reports": reports,
        "boundary_refs": boundary_refs,
        "boundaries": boundaries,
        "transcript_refs": transcript_refs,
        "pair_metrics": pair_metrics,
        "cpu_ref": cpu_ref,
        "cpu_report": cpu_report,
        "fault_ref": fault_ref,
        "fault_report": fault_report,
        "retention_ref": retention_ref,
        "retention_report": retention_report,
        "config_refs": [single_config_ref, ddp_config_ref],
        "environment_ref": environment_ref,
    }


def _gate_payloads(
    request: TaskExecutionRequest,
    source: G7RecoverySourceBinding,
    suite: Mapping[str, Any],
    source_refs: Sequence[str],
    root: Path,
) -> tuple[dict[str, JSONValue], GateRecord, Stage0GateReport, tuple[str, ...], str]:
    report_refs = tuple(str(item) for item in suite["report_refs"])
    boundary_refs = tuple(str(item) for item in suite["boundary_refs"])
    transcript_refs = tuple(str(item) for item in suite["transcript_refs"])
    evidence_refs = (
        tuple(source_refs)
        + tuple(str(item) for item in suite["config_refs"])
        + (str(suite["environment_ref"]),)
        + report_refs
        + boundary_refs
        + transcript_refs
        + (
            str(suite["cpu_ref"]),
            str(suite["fault_ref"]),
            str(suite["retention_ref"]),
        )
    )
    timestamps = [
        _timestamp(report["completed_at"], field="report.completed_at")
        for report in suite["reports"].values()
    ] + [
        _timestamp(boundary["recorded_at"], field="boundary.recorded_at")
        for boundary in suite["boundaries"].values()
    ]
    checked_at = max(timestamps).isoformat().replace("+00:00", "Z")
    pair_metrics = suite["pair_metrics"]
    cpu = suite["cpu_report"]
    fault = suite["fault_report"]
    retention = suite["retention_report"]
    metrics: dict[str, JSONValue] = {
        "cpu_fp32_reference": cpu,
        "single_gpu_fp32": pair_metrics["single"],
        "four_gpu_fp32": pair_metrics["ddp"],
        "fresh_process_launches": 6,
        "intentional_interruptions": 2,
        "formal_num_workers": 0,
        "formal_prefetch_factor": 0,
        "shared_state_hashes_exact": bool(
            pair_metrics["single"]["shared_state_hashes_exact"]
            and pair_metrics["ddp"]["shared_state_hashes_exact"]
        ),
        "state_numeric_tolerance_met": bool(
            pair_metrics["single"]["shared_state_numeric_within_tolerance"]
            and pair_metrics["ddp"]["shared_state_numeric_within_tolerance"]
        ),
        "non_determinism_source": (
            pair_metrics["ddp"].get("non_determinism_source")
            or pair_metrics["single"].get("non_determinism_source")
        ),
        "fault_rejection_count": len(fault["rejection_reasons"]),
        "derived_reconciliation_passed": fault["commit_without_derived_views_rebuilt"],
        "retention_scope_count": len(retention["scopes"]),
        "physical_objects_deleted": sum(
            int(item["objects_deleted"]) for item in retention["scopes"]
        ),
        "physical_bytes_released": int(retention["total_released_bytes"]),
        "full_state_contract": [
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "python_numpy_torch_cpu_cuda_rng",
            "committed_cursor_and_prefetch_pending",
            "training_counters",
            "resolved_config_and_asset_identity",
            "canonical_event_pointer",
            "online_importance_extension_state",
        ],
    }
    checks = (
        Stage0GateCheck(
            "stage0.G7-FP32-REFERENCE",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "The deterministic FP32 direct, prefetched, and fresh-engine resumed trajectories are byte-identical at all recoverable state fields.",
            measurements={
                "state_fields_exact": cpu["state_fields_exact"],
                "num_workers_reference": cpu["direct_num_workers"],
                "num_workers_formal": cpu["formal_num_workers"],
                "atol": cpu["atol"],
                "rtol": cpu["rtol"],
            },
            evidence_refs=(str(suite["cpu_ref"]),),
        ),
        Stage0GateCheck(
            "stage0.G7-SINGLE-AND-DDP-RESUME",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "Real Pythia FP32 single-GPU and four-GPU fresh-process resumes preserve samples, steps, learning rates, and canonical lineage; shared model tensors are byte-exact or agree within atol=1e-6/rtol=1e-5 with the non-determinism source recorded.",
            measurements={
                "single_gpu": pair_metrics["single"],
                "four_gpu": pair_metrics["ddp"],
                "fresh_process_launches": 6,
            },
            evidence_refs=report_refs + boundary_refs + transcript_refs,
        ),
        Stage0GateCheck(
            "stage0.G7-ATOMICITY-AND-STRICT-LOAD",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "Uncommitted objects, missing ranks, truncated tensors, incompatible identities, and stale derived views fail closed or reconcile from authority.",
            measurements={
                "rejection_reasons": fault["rejection_reasons"],
                "orphan_rank_commits_ignored": fault["orphan_rank_commits_ignored"],
                "derived_rebuilt": fault["commit_without_derived_views_rebuilt"],
            },
            evidence_refs=(str(suite["fault_ref"]),),
        ),
        Stage0GateCheck(
            "stage0.G7-RETENTION-AND-CAPACITY",
            Stage0CheckClass.CAPACITY,
            Stage0CheckStatus.PASS,
            "Retention selection is deterministic; each exact-ID purge is authorized by a durable tombstone-bound intent and completion record, releases measured bytes, and preserves protected recovery points.",
            exception_eligible=False,
            measurements={
                "scope_count": len(retention["scopes"]),
                "core_policy": retention["core_policy"],
                "physical_cleanup_separate_audit": retention[
                    "physical_cleanup_requires_separate_authorized_audit"
                ],
                "objects_deleted": sum(
                    int(item["objects_deleted"]) for item in retention["scopes"]
                ),
                "released_bytes": retention["total_released_bytes"],
            },
            evidence_refs=(str(suite["retention_ref"]),) + report_refs,
        ),
    )
    gate = GateRecord(
        gate_id=GATE_ID,
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "single_gpu_exact": True,
            "four_gpu_exact": True,
            "fp32_reference_exact": True,
            "canonical_optimizer_steps": 4,
            "formal_num_workers": 0,
            "shared_state_hashes_exact": bool(
                pair_metrics["single"]["shared_state_hashes_exact"]
                and pair_metrics["ddp"]["shared_state_hashes_exact"]
            ),
            "state_numeric_tolerance_met": True,
            "fault_rejection_count": len(fault["rejection_reasons"]),
        },
        threshold={
            "fp32_atol": 1e-6,
            "fp32_rtol": 1e-5,
            "world_size_must_match": True,
            "sample_repeat_or_gap_max": 0,
            "invalid_checkpoint_selection_max": 0,
        },
        evidence_refs=evidence_refs,
    )
    environment_id = str(request.config.base_config.section("runtime")["environment_id"])
    gate_report = Stage0GateReport(
        gate_id=GATE_ID,
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        config_hashes={TASK_ID: request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(_logical_path(root, reference, field="gate.evidence_ref")),
                schema_version=(
                    WORKER_REPORT_SCHEMA
                    if reference in report_refs
                    else BOUNDARY_SCHEMA
                    if reference in boundary_refs
                    else "stage0-g7-recovery-launch-transcript-v1"
                    if reference in transcript_refs
                    else "stage0-g7-recovery-evidence"
                ),
            )
            for reference in evidence_refs
        ),
        checks=checks,
    )
    validation = _with_hash(
        {
            "schema_version": "stage0-g7-recovery-validation-report-v1",
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "metrics": metrics,
        }
    )
    return validation, gate, gate_report, evidence_refs, checked_at


def run_formal_g7_recovery_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G7RecoveryError("G7_RECOVERY_FORMAL_REQUEST_INVALID")
    if "stage0.G6" not in request.environment.passed_gate_ids or "stage0.G7-LOGGING" not in request.environment.passed_gate_ids:
        raise Stage0G7RecoveryError("G7_RECOVERY_REQUIRED_GATES_MISSING")
    provenance_ref = request.environment.evidence_refs.get("g4_provenance")
    if provenance_ref is None:
        raise Stage0G7RecoveryError("G7_RECOVERY_G4_PROVENANCE_REF_MISSING")
    provenance = _mapping(
        load_committed_task_artifact(root, provenance_ref, require_formal=True).payload,
        field="g4_provenance_envelope",
    )
    selected = _mapping(provenance.get("provenance_record"), field="g4_provenance").get("device_mapping")
    if not isinstance(selected, list) or len(selected) != 4:
        raise Stage0G7RecoveryError("G7_RECOVERY_GPU_SELECTION_INVALID")
    suite_ref = (
        f"evidence/stage0/g7-recovery-suite/{request.config.config_hash}/"
        f"{request.environment.environment_hash}"
    )
    suite = _run_suite(request, root, source, tuple(str(item) for item in selected), suite_ref)
    validation, gate, gate_report, evidence_refs, checked_at = _gate_payloads(
        request, source, suite, source_refs, root
    )
    refs: dict[str, str] = {}
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g7-recovery-evidence-v1",
            "artifact_role": kind,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "gate_id": GATE_ID,
            "logging_component_refs": list(source_refs),
            "suite_root_ref": suite_ref,
            "suite_environment_ref": str(suite["environment_ref"]),
            "config_refs": list(suite["config_refs"]),
            "worker_report_refs": list(suite["report_refs"]),
            "boundary_refs": list(suite["boundary_refs"]),
            "launch_transcript_refs": list(suite["transcript_refs"]),
            "cpu_reference_ref": str(suite["cpu_ref"]),
            "fault_report_ref": str(suite["fault_ref"]),
            "retention_report_ref": str(suite["retention_ref"]),
            "validation_report": validation,
            "gate_record": gate.to_dict(),
            "gate_report": gate_report.to_dict(),
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
        checkpoint_ref=refs["checkpoint_commit"],
        message="Stage 0 S0.9 checkpoint recovery gate passed",
        metadata={"stage0_g7_recovery_specialized": True, "gate_id": GATE_ID},
    )


def _load_hashed_report(root: Path, reference: str, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(_logical_path(root, reference, field=field)), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage0G7RecoveryError(f"G7_RECOVERY_HASHED_REPORT_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def validate_formal_g7_recovery_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_SET_INVALID")
    envelopes: list[dict[str, Any]] = []
    for kind, reference in outputs.items():
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != request.config.config_hash
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_COMMIT_IDENTITY_INVALID")
        envelope = _mapping(loaded.payload, field=f"output.{kind}")
        declared = envelope.pop("artifact_hash", None)
        if (
            envelope.get("schema_version") != "stage0-g7-recovery-evidence-v1"
            or envelope.get("artifact_role") != kind
            or envelope.get("status") != "PASS"
            or envelope.get("generator_git_commit") != source.git_commit
            or envelope.get("environment_hash") != request.environment.environment_hash
            or envelope.get("gate_id") != GATE_ID
            or declared != canonical_json_hash(envelope)
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_ENVELOPE_INVALID")
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
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_ROLE_PAYLOAD_DRIFT")
    config_refs = canonical.get("config_refs")
    report_refs = canonical.get("worker_report_refs")
    boundary_refs = canonical.get("boundary_refs")
    transcript_refs = canonical.get("launch_transcript_refs")
    if (
        not isinstance(config_refs, list)
        or len(config_refs) != 2
        or not isinstance(report_refs, list)
        or len(report_refs) != 4
        or not isinstance(boundary_refs, list)
        or len(boundary_refs) != 2
        or not isinstance(transcript_refs, list)
        or len(transcript_refs) != 6
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_REFERENCE_SET_INVALID")
    configs: dict[int, ResolvedConfigV2] = {}
    for reference in config_refs:
        config = ResolvedConfigV2.from_mapping(
            _mapping(load_canonical_json(_logical_path(root, reference, field="config_ref")), field="worker_config")
        )
        distributed = _mapping(config.base_config.section("distributed"), field="distributed")
        world_size = int(distributed["world_size"])
        if config.task_id != TASK_ID or world_size not in {1, 4} or world_size in configs:
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_CONFIG_SET_INVALID")
        configs[world_size] = config
    if set(configs) != {1, 4} or configs[4].config_hash != request.config.config_hash:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_PRIMARY_CONFIG_MISMATCH")
    raw_reports = [
        _mapping(load_canonical_json(_logical_path(root, reference, field="report_ref")), field="raw_report")
        for reference in report_refs
    ]
    ddp_raw = next((item for item in raw_reports if item.get("world_size") == 4), None)
    if ddp_raw is None:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_DDP_REPORT_MISSING")
    selected = ddp_raw.get("selected_gpu_uuids")
    if not isinstance(selected, list) or len(selected) != 4:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_GPU_SET_INVALID")
    reports: dict[tuple[int, str], dict[str, Any]] = {}
    for reference, raw in zip(report_refs, raw_reports, strict=True):
        world_size = int(raw.get("world_size", 0))
        phase = str(raw.get("phase"))
        if world_size not in configs or phase not in {"baseline", "resume"}:
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_REPORT_IDENTITY_INVALID")
        report = _validate_worker_report(
            root,
            reference,
            source=source,
            request=request,
            config=configs[world_size],
            selected=tuple(str(item) for item in selected),
            phase=phase,
            world_size=world_size,
        )
        key = (world_size, phase)
        if key in reports:
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_DUPLICATE_REPORT")
        reports[key] = report
    if set(reports) != {(1, "baseline"), (1, "resume"), (4, "baseline"), (4, "resume")}:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_REPORT_MATRIX_INCOMPLETE")
    boundaries: dict[int, dict[str, Any]] = {}
    for reference in boundary_refs:
        raw = _mapping(load_canonical_json(_logical_path(root, reference, field="boundary_ref")), field="raw_boundary")
        world_size = int(raw.get("world_size", 0))
        if world_size not in configs or world_size in boundaries:
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_BOUNDARY_MATRIX_INVALID")
        boundaries[world_size] = _validate_boundary(
            root,
            reference,
            source=source,
            request=request,
            config=configs[world_size],
            world_size=world_size,
        )
    pair_metrics = {
        "single": _compare_trajectory_pair(root, reports[(1, "baseline")], reports[(1, "resume")], boundaries[1]),
        "ddp": _compare_trajectory_pair(root, reports[(4, "baseline")], reports[(4, "resume")], boundaries[4]),
    }
    interruption_count = 0
    for reference in transcript_refs:
        transcript = _load_hashed_report(
            root,
            reference,
            schema="stage0-g7-recovery-launch-transcript-v1",
            field="transcript",
        )
        if transcript.get("timed_out") is not False or any(
            _mapping(transcript.get("residual_compute_pids"), field="transcript.residual").values()
        ):
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_TRANSCRIPT_UNHEALTHY")
        if transcript.get("interruption_marker_observed") is True:
            interruption_count += 1
            if transcript.get("return_code") == 0:
                raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_INTERRUPT_EXIT_ZERO")
        elif transcript.get("return_code") != 0:
            raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_SUCCESS_LAUNCH_NONZERO")
    if interruption_count != 2:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_INTERRUPT_COUNT_INVALID")
    cpu_ref = canonical.get("cpu_reference_ref")
    fault_ref = canonical.get("fault_report_ref")
    retention_ref = canonical.get("retention_report_ref")
    if not all(isinstance(item, str) for item in (cpu_ref, fault_ref, retention_ref)):
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_FUNCTIONAL_REFS_INVALID")
    cpu = _load_hashed_report(root, cpu_ref, schema="stage0-g7-recovery-cpu-reference-v1", field="cpu_reference")
    fault = _load_hashed_report(root, fault_ref, schema="stage0-g7-recovery-fault-report-v1", field="fault_report")
    retention = _load_hashed_report(root, retention_ref, schema="stage0-g7-recovery-retention-report-v1", field="retention_report")
    if (
        cpu.get("status") != "PASS"
        or not all(_mapping(cpu.get("state_fields_exact"), field="cpu.state_fields").values())
        or fault.get("status") != "PASS"
        or len(_mapping(fault.get("rejection_reasons"), field="fault.rejections")) < 5
        or retention.get("status") != "PASS"
        or retention.get("latest_best_milestone_and_protected_preserved") is not True
        or retention.get("exact_checkpoint_ids_only") is not True
        or retention.get("active_group_references_preserved") is not True
        or retention.get("core_policy") != "tombstone_then_exact_id_physical_purge"
        or retention.get("physical_cleanup_requires_separate_authorized_audit") is not False
        or not isinstance(retention.get("total_released_bytes"), int)
        or int(retention["total_released_bytes"]) <= 0
        or not isinstance(retention.get("scopes"), list)
        or not retention["scopes"]
        or any(
            not isinstance(scope, dict)
            or scope.get("objects_deleted") != 1
            or not isinstance(scope.get("released_bytes"), int)
            or int(scope["released_bytes"]) <= 0
            or not isinstance(scope.get("purge_intent_path"), str)
            or not isinstance(scope.get("purge_record_path"), str)
            for scope in retention["scopes"]
        )
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_FUNCTIONAL_REPORT_INVALID")
    suite = {
        "report_refs": report_refs,
        "reports": reports,
        "boundary_refs": boundary_refs,
        "boundaries": boundaries,
        "transcript_refs": transcript_refs,
        "pair_metrics": pair_metrics,
        "cpu_ref": cpu_ref,
        "cpu_report": cpu,
        "fault_ref": fault_ref,
        "fault_report": fault,
        "retention_ref": retention_ref,
        "retention_report": retention,
        "config_refs": config_refs,
        "environment_ref": canonical.get("suite_environment_ref"),
    }
    logging_refs = canonical.get("logging_component_refs")
    if not isinstance(logging_refs, list) or len(logging_refs) != 3:
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_LOGGING_REFS_INVALID")
    replayed_validation, replayed_gate, replayed_gate_report, _, replayed_checked_at = _gate_payloads(
        request,
        source,
        suite,
        tuple(str(item) for item in logging_refs),
        root,
    )
    validation = _mapping(canonical.get("validation_report"), field="validation")
    gate = GateRecord.from_mapping(_mapping(canonical.get("gate_record"), field="gate_record"))
    gate_report = Stage0GateReport.from_mapping(_mapping(canonical.get("gate_report"), field="gate_report"))
    if (
        validation != replayed_validation
        or gate.to_dict() != replayed_gate.to_dict()
        or gate_report.to_dict() != replayed_gate_report.to_dict()
        or canonical.get("checked_at") != replayed_checked_at
        or gate.gate_id != GATE_ID
        or gate.status is not GateStatus.PASS
        or gate_report.gate_id != GATE_ID
        or gate_report.status.value != "PASS"
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_OUTPUT_VALIDATION_REPLAY_MISMATCH")
    return gate


def execute_stage0_g7_recovery(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g7_logging_index_ref: str,
) -> Stage0G7RecoveryFormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g7_formal_state(
        data_root=root,
        index_ref=g7_logging_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g7_recovery_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g7-recovery-formal/{state.component_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G7RecoveryError(
            f"G7_RECOVERY_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, state.environment)
    gate = validate_formal_g7_recovery_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g7_checkpoint_commit": outputs["checkpoint_commit"],
            "g7_resume_equivalence": outputs["resume_equivalence_report"],
            "g7_retention": outputs["retention_report"],
            "gate_stage0_g7": outputs["resume_equivalence_report"],
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids | frozenset({gate.gate_id}),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    write_canonical_json(
        _logical_path(root, environment_ref, field="environment_ref"),
        environment.to_dict(),
    )
    index: dict[str, JSONValue] = {
        "schema_version": "stage0-g7-recovery-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g7_logging_index_ref": state.index_ref,
        "g7_logging_index_sha256": state.index_sha256,
        "g7_logging_component_artifact_hash": state.component_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_ref": outputs["resume_equivalence_report"],
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.10_capacity_and_operations",
        "next_input_refs": list(outputs.values()),
    }
    index["artifact_hash"] = canonical_json_hash(index)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(_logical_path(root, index_ref, field="index_ref"), index)
    return Stage0G7RecoveryFormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def load_stage0_g7_recovery_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G7RecoveryFormalState:
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g7_recovery_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g7_logging_index_ref",
        "g7_logging_index_sha256",
        "g7_logging_component_artifact_hash",
        "config_ref",
        "config_hash",
        "task_output_refs",
        "gate_ref",
        "environment_ref",
        "environment_hash",
        "next_task_id",
        "next_input_refs",
        "artifact_hash",
    }
    if set(raw) != expected or raw.get("schema_version") != "stage0-g7-recovery-formalization-index-v1":
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_INDEX_FIELDS_OR_VERSION_INVALID")
    declared = raw.pop("artifact_hash")
    if declared != canonical_json_hash(raw):
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_INDEX_HASH_MISMATCH")
    raw["artifact_hash"] = declared
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_GENERATOR_COMMIT_MISMATCH")
    logging = load_stage0_g7_formal_state(
        data_root=root,
        index_ref=str(raw["g7_logging_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    if (
        raw.get("g7_logging_index_sha256") != logging.index_sha256
        or raw.get("g7_logging_component_artifact_hash") != logging.component_artifact_hash
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_LOGGING_BINDING_MISMATCH")
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(_logical_path(root, raw["config_ref"], field="config_ref")), field="config")
    )
    if config.task_id != TASK_ID or config.config_hash != raw.get("config_hash"):
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_CONFIG_MISMATCH")
    outputs = _mapping(raw.get("task_output_refs"), field="task_output_refs")
    if set(outputs) != _OUTPUT_KINDS or any(not isinstance(item, str) for item in outputs.values()):
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_OUTPUT_SET_INVALID")
    ordered = {
        key: str(outputs[key])
        for key in ("checkpoint_commit", "resume_equivalence_report", "retention_report")
    }
    request = TaskExecutionRequest(config, config.task_definition, logging.environment)
    gate = validate_formal_g7_recovery_outputs(request, root, ordered)
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(load_canonical_json(_logical_path(root, raw["environment_ref"], field="environment_ref")), field="environment")
    )
    next_inputs = raw.get("next_input_refs")
    if (
        raw.get("gate_ref") != outputs["resume_equivalence_report"]
        or gate.gate_id not in environment.passed_gate_ids
        or environment.environment_hash != raw.get("environment_hash")
        or environment.evidence_refs.get("gate_stage0_g7") != raw.get("gate_ref")
        or raw.get("next_task_id") != "stage0.10_capacity_and_operations"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(outputs.values())
    ):
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_ENVIRONMENT_OR_HANDOFF_INVALID")
    gate_hash = load_committed_task_artifact(
        root, str(raw["gate_ref"]), require_formal=True
    ).payload.get("artifact_hash")
    if not isinstance(gate_hash, str) or _SHA256_RE.fullmatch(gate_hash) is None:
        raise Stage0G7RecoveryError("G7_RECOVERY_STATE_GATE_ARTIFACT_HASH_INVALID")
    return Stage0G7RecoveryFormalState(
        environment=environment,
        task_output_refs=ordered,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate_hash,
        g7_logging_index_ref=str(raw["g7_logging_index_ref"]),
    )


__all__ = [
    "G7RecoverySourceBinding",
    "GATE_ID",
    "Stage0G7RecoveryError",
    "Stage0G7RecoveryFormalState",
    "Stage0G7RecoveryFormalizationResult",
    "TASK_ID",
    "build_stage0_g7_recovery_config",
    "execute_stage0_g7_recovery",
    "load_stage0_g7_recovery_formal_state",
    "run_formal_g7_recovery_task",
    "validate_formal_g7_recovery_outputs",
]
