"""Formal Stage 0 G5 single-GPU execution and aggregate gate.

G5 is deliberately a suite, not a synonym for one successful training call.
The aggregate runner launches every numerical repeat in a fresh interpreter,
binds one physical GPU UUID, replays immutable child reports, and only then
publishes the canonical ``stage0.G5`` decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

from .atomic import sha256_file
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
from .runtime import (
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
    load_tensor_bundle,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g4 import Stage0G4FormalState, load_stage0_g4_formal_state
from .stage0_g5_worker import WORKER_PLAN_SCHEMA, WORKER_REPORT_SCHEMA
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID = "stage0.06_single_gpu_smoke"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CRITICAL_SOURCE_REFS = (
    "ops/stage0/formalize_g5.py",
    "ops/stage0/run_g5_worker.py",
    "src/param_importance_nlp/experiments/task_runners.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/stage0_g5.py",
    "src/param_importance_nlp/stage0_g5_worker.py",
    "src/param_importance_nlp/stage0_gate.py",
    "schemas/stage0-g5-evidence-v1.json",
    "schemas/stage0-g5-formalization-index-v1.json",
    "schemas/stage0-g5-worker-plan-v1.json",
    "schemas/stage0-g5-worker-report-v1.json",
)
_OUTPUT_KINDS = {"training_smoke_result", "event_stream", "checkpoint_commit"}
_REPORT_FIELDS = {
    "schema_version",
    "run_id",
    "run_kind",
    "repeat_index",
    "status",
    "pid",
    "parent_pid",
    "started_at",
    "completed_at",
    "generator_git_commit",
    "plan_ref",
    "plan_sha256",
    "config_ref",
    "config_sha256",
    "config_hash",
    "config_full_hash",
    "environment_ref",
    "environment_sha256",
    "environment_hash",
    "device",
    "phase_memory",
    "training_wall_seconds",
    "training_result",
    "event_stream_ref",
    "event_stream_semantic_sha256",
    "checkpoint_commits",
    "asset_evidence",
    "resource_profiles",
    "selected_tensor_bundle_ref",
    "selected_tensor_bundle_sha256",
    "selected_tensor_names",
    "data_boundary",
    "step_telemetry",
    "failure",
    "artifact_hash",
}


class Stage0G5Error(RuntimeError):
    """G5 child execution or aggregate validation failed closed."""


@dataclass(frozen=True, slots=True)
class G5SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G5FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G5FormalState:
    """Revalidated G5 handoff consumed by the four-GPU G6 suite."""

    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    g4_index_ref: str


@dataclass(frozen=True, slots=True)
class _ChildSpec:
    run_kind: str
    repeat_index: int
    max_steps: int
    precision: str
    repeat_fixed_batch: bool
    capture_tensor_step: int | None
    expected_failure_code: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.run_kind}-{self.repeat_index:02d}"


_CHILD_SPECS = (
    *(_ChildSpec("fp32_determinism", index, 1, "fp32", True, 1) for index in range(2)),
    *(_ChildSpec("overfit", index, 50, "fp32", True, None) for index in range(3)),
    _ChildSpec("bf16", 0, 1, "bf16", True, 1),
    *(_ChildSpec("memory", index, 25, "bf16", False, None) for index in range(3)),
    _ChildSpec(
        "failure_invalid_asset", 0, 1, "fp32", False, None,
        "G3_RUNTIME_LOGICAL_ASSET_UNKNOWN",
    ),
    _ChildSpec(
        "failure_out_of_range", 0, 1_000_000_000, "fp32", False, None,
        "G3_RUNTIME_PILE_SPLIT_BUDGET_EXCEEDED",
    ),
    _ChildSpec(
        "failure_nonfinite", 0, 1, "fp32", True, None,
        "STAGE0_G5_INJECTED_NONFINITE_LOSS",
    ),
    _ChildSpec(
        "failure_output_collision", 0, 1, "fp32", False, None,
        "RUN_DIRECTORY_EXISTS",
    ),
    _ChildSpec(
        "failure_checkpoint_write", 0, 1, "fp32", True, None,
        "STAGE0_G5_INJECTED_CHECKPOINT_WRITE_FAILURE",
    ),
)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _capture_source() -> G5SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    top = _git(repository, "rev-parse", "--show-toplevel")
    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    tracked = _git(
        repository,
        "ls-files",
        "--error-unmatch",
        "--",
        *_CRITICAL_SOURCE_REFS,
    )
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if any(item.returncode != 0 for item in (top, head, branch, tracked, status)):
        raise Stage0G5Error("G5_SOURCE_GIT_PROBE_FAILED")
    if Path(top.stdout.strip()).resolve() != repository:
        raise Stage0G5Error("G5_SOURCE_GIT_ROOT_MISMATCH")
    commit = head.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch.stdout.strip():
        raise Stage0G5Error("G5_SOURCE_GIT_IDENTITY_INVALID")
    if status.stdout.strip():
        raise Stage0G5Error("G5_FORMAL_SOURCE_DIRTY")
    return G5SourceBinding(repository, commit, branch.stdout.strip())


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G5Error(f"G5_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G5Error(f"G5_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G5Error(f"G5_LOGICAL_PATH_ESCAPE:{field}")
    resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage0G5Error(f"G5_LOGICAL_PATH_ESCAPE:{field}") from error
    return resolved


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Stage0G5Error(f"G5_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G5Error(f"G5_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0G5Error(f"G5_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _payload_with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _g4_provenance(state: Stage0G4FormalState, root: Path) -> dict[str, Any]:
    loaded = load_committed_task_artifact(
        root, state.task_output_refs["provenance_record"], require_formal=True
    )
    envelope = _mapping(loaded.payload, field="g4_provenance_envelope")
    provenance = _mapping(envelope.get("provenance_record"), field="g4_provenance")
    mapping = provenance.get("device_mapping")
    if (
        not isinstance(mapping, list)
        or len(mapping) != 4
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in mapping)
    ):
        raise Stage0G5Error("G5_G4_DEVICE_MAPPING_INVALID")
    return provenance


def build_stage0_g5_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G4FormalState,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    provenance = _g4_provenance(state, root)
    g4_base = state.config.base_config
    model = g4_base.section("model")
    data = g4_base.section("data")
    runtime = g4_base.section("runtime")
    assert isinstance(model, dict) and isinstance(data, dict) and isinstance(runtime, dict)
    output_dir = f"evidence/stage0/tasks/06-{state.gate_artifact_hash}"
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=output_dir,
        base_overrides={
            "identity": {"route": f"stage0-g5-{state.gate_artifact_hash[:12]}"},
            "runtime": {
                "environment_id": provenance["environment_id"],
                "device": "cuda",
                "dependency_profile": runtime["dependency_profile"],
                "output_root": "runs/stage0",
                "temp_root": "tmp/stage0",
                "cache_root": "cache/stage0",
                "allow_dirty_worktree": False,
            },
            "model": model,
            "data": {
                **data,
                "split": "debug",
                "sampler": "without_replacement",
                "sampling_design": "without_replacement_frozen_epoch",
            },
            "loss": {
                "task_type": "causal_lm",
                "reduction": "mean",
                "ignore_index": -100,
                "weighting": "effective_token",
            },
            "batching": {
                "global_batch_size": 1,
                "per_device_batch_size": 1,
                "microbatch_size": 1,
                "accumulation_steps": 1,
                "no_sync": False,
            },
            "distributed": {
                "world_size": 1,
                "backend": "local",
                "device_ids": [0],
                "timeout_seconds": 180,
            },
            "precision": {
                "compute_dtype": "float32",
                "gradient_dtype": "float32",
                "statistic_dtype": "float32",
                "reference_dtype": "float64",
                "quadrature_weight_dtype": "float64",
                "path_accumulation_dtype": "float64",
                "amp": False,
            },
            "optimizer": {
                "type": "adamw",
                "learning_rate": 0.0003,
                "momentum": 0.0,
                "weight_decay": 0.0,
                "fused": False,
                "foreach": False,
                "parameter_groups": [],
            },
        },
        v2_overrides={
            "execution": {"timeout_seconds": 1800, "max_attempts": 1},
            "training": {
                "max_steps": 1,
                "deterministic_algorithms": True,
                "gradient_clip_max_norm": None,
            },
            "providers": {
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
            },
            "launcher": {
                "kind": "local",
                "backend": "local",
                "world_size": 1,
                "init_method": "local",
                "init_ref": None,
                "rendezvous_id": None,
                "max_restarts": 0,
            },
        },
    )


def _child_config(
    main: ResolvedConfigV2,
    spec: _ChildSpec,
    *,
    output_dir: str,
) -> ResolvedConfigV2:
    base = main.base_config.to_dict()
    identity = _mapping(base["identity"], field="child.identity")
    identity["route"] = f"stage0-g5-{spec.run_kind}"
    precision = _mapping(base["precision"], field="child.precision")
    if spec.precision == "bf16":
        precision.update(
            {
                "compute_dtype": "bfloat16",
                "gradient_dtype": "float32",
                "statistic_dtype": "float32",
                "amp": True,
            }
        )
    else:
        precision.update(
            {
                "compute_dtype": "float32",
                "gradient_dtype": "float32",
                "statistic_dtype": "float32",
                "amp": False,
            }
        )
    if spec.run_kind == "failure_invalid_asset":
        model = _mapping(base["model"], field="child.model")
        model["asset_id"] = "invalid-g5-model-asset"
        base["model"] = model
    base["identity"] = identity
    base["precision"] = precision
    overrides: dict[str, Any] = {
        "execution": main.section("execution"),
        "training": {
            **_mapping(main.section("training"), field="main.training"),
            "max_steps": spec.max_steps,
        },
        "scheduler": {
            **_mapping(main.section("scheduler"), field="main.scheduler"),
            "total_steps": None,
        },
        "data_loader": main.section("data_loader"),
        "providers": main.section("providers"),
        "evaluation": main.section("evaluation"),
        "profiling": main.section("profiling"),
        "checkpoint_schedule": {
            "segments": [
                {
                    "start_step": 0,
                    "end_step": None,
                    "every_steps": spec.max_steps,
                }
            ],
            "save_on_phase_end": True,
            "save_optimizer": True,
            "save_rng": True,
            "save_data_state": True,
        },
        "precision_runtime": {
            **_mapping(main.section("precision_runtime"), field="main.precision_runtime"),
            "autocast_enabled": spec.precision == "bf16",
            "autocast_dtype": "bfloat16" if spec.precision == "bf16" else "float32",
            "grad_scaler_enabled": False,
        },
        "optimizer_runtime": main.section("optimizer_runtime"),
        "launcher": main.section("launcher"),
        "orchestration": main.section("orchestration"),
        "recovery": main.section("recovery"),
        "artifacts": {
            **_mapping(main.section("artifacts"), field="main.artifacts"),
            "output_dir": output_dir,
        },
    }
    return ResolvedConfigV2.resolve(base, task_id=TASK_ID, overrides=overrides)


def _nvidia_query(selected_uuid: str) -> dict[str, JSONValue]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={selected_uuid}",
            "--query-gpu=uuid,index,pci.bus_id,temperature.gpu,memory.used,memory.total,"
            "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if query.returncode != 0:
        raise Stage0G5Error(f"G5_NVIDIA_SMI_GPU_QUERY_FAILED:{query.stderr.strip()}")
    lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise Stage0G5Error("G5_NVIDIA_SMI_GPU_QUERY_CARDINALITY")
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != 8 or fields[0] != selected_uuid:
        raise Stage0G5Error("G5_NVIDIA_SMI_GPU_QUERY_FIELDS_INVALID")
    try:
        integers = [int(value) for value in fields[1:2] + fields[3:]]
    except ValueError as error:
        raise Stage0G5Error("G5_NVIDIA_SMI_GPU_QUERY_NOT_NUMERIC") from error
    return {
        "uuid": fields[0],
        "index": integers[0],
        "pci_bus_id": fields[2],
        "temperature_c": integers[1],
        "memory_used_mib": integers[2],
        "memory_total_mib": integers[3],
        "ecc_corrected_volatile": integers[4],
        "ecc_uncorrected_volatile": integers[5],
    }


def _compute_pids(selected_uuid: str) -> tuple[int, ...]:
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
        raise Stage0G5Error(f"G5_NVIDIA_SMI_PROCESS_QUERY_FAILED:{query.stderr.strip()}")
    pids: list[int] = []
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 2 and fields[1] == selected_uuid:
            try:
                pids.append(int(fields[0]))
            except ValueError as error:
                raise Stage0G5Error("G5_NVIDIA_SMI_PROCESS_PID_INVALID") from error
    return tuple(sorted(set(pids)))


def _validate_worker_report(
    root: Path,
    report_ref: str,
    *,
    expected_commit: str,
    expected_environment_hash: str,
    selected_uuid: str,
) -> dict[str, Any]:
    path = _logical_path(root, report_ref, field="worker_report_ref")
    report = _mapping(load_canonical_json(path), field="worker_report")
    if set(report) != _REPORT_FIELDS or report.get("schema_version") != WORKER_REPORT_SCHEMA:
        raise Stage0G5Error("G5_WORKER_REPORT_FIELDS_OR_VERSION_INVALID")
    declared = report.pop("artifact_hash")
    if declared != canonical_json_hash(report):
        raise Stage0G5Error("G5_WORKER_REPORT_HASH_MISMATCH")
    report["artifact_hash"] = declared
    plan_ref = report.get("plan_ref")
    plan_path = _logical_path(root, plan_ref, field="worker_plan_ref")
    plan = _mapping(load_canonical_json(plan_path), field="worker_plan")
    plan_declared = plan.pop("artifact_hash", None)
    if (
        plan.get("schema_version") != WORKER_PLAN_SCHEMA
        or plan_declared != canonical_json_hash(plan)
        or report.get("plan_sha256") != plan_declared
        or report.get("run_id") != plan.get("run_id")
        or report.get("run_kind") != plan.get("run_kind")
        or report.get("repeat_index") != plan.get("repeat_index")
    ):
        raise Stage0G5Error("G5_WORKER_REPORT_PLAN_BINDING_INVALID")
    device = _mapping(report.get("device"), field="worker.device")
    if (
        report.get("generator_git_commit") != expected_commit
        or report.get("environment_hash") != expected_environment_hash
        or device.get("selected_physical_uuid") != selected_uuid
        or device.get("cuda_visible_devices") != selected_uuid
        or device.get("cuda_device_count") != 1
    ):
        raise Stage0G5Error("G5_WORKER_REPORT_SOURCE_ENV_DEVICE_MISMATCH")
    config_path = _logical_path(root, report.get("config_ref"), field="worker.config_ref")
    if sha256_file(config_path) != report.get("config_sha256"):
        raise Stage0G5Error("G5_WORKER_REPORT_CONFIG_FILE_HASH_MISMATCH")
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(config_path), field="worker.config")
    )
    if (
        config.config_hash != report.get("config_hash")
        or config.full_hash != report.get("config_full_hash")
    ):
        raise Stage0G5Error("G5_WORKER_REPORT_CONFIG_IDENTITY_MISMATCH")
    failure_kind = str(report["run_kind"]).startswith("failure_")
    expected_status = "EXPECTED_FAILURE_CONFIRMED" if failure_kind else "PASS"
    if report.get("status") != expected_status:
        raise Stage0G5Error("G5_WORKER_REPORT_STATUS_INVALID")
    if failure_kind:
        failure = _mapping(report.get("failure"), field="worker.failure")
        if failure.get("success_commit_absent") is not True:
            raise Stage0G5Error("G5_WORKER_FAILURE_SUCCESS_COMMIT_PRESENT")
    else:
        training = _mapping(report.get("training_result"), field="worker.training_result")
        if training.get("status") != "COMPLETE" or report.get("failure") is not None:
            raise Stage0G5Error("G5_WORKER_SUCCESS_TRAINING_INVALID")
    return report


def _tensor_state(root: Path, report: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = report.get("selected_tensor_bundle_ref")
    digest = report.get("selected_tensor_bundle_sha256")
    if not isinstance(reference, str) or _SHA256_RE.fullmatch(str(digest)) is None:
        raise Stage0G5Error("G5_SELECTED_TENSOR_BUNDLE_REF_INVALID")
    state, identity = load_tensor_bundle(_logical_path(root, reference, field="tensor_bundle"))
    if identity.manifest_sha256 != digest or not isinstance(state, Mapping):
        raise Stage0G5Error("G5_SELECTED_TENSOR_BUNDLE_HASH_MISMATCH")
    return state


def _records(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    training = _mapping(report.get("training_result"), field="training_result")
    raw = training.get("records")
    if not isinstance(raw, list) or not raw:
        raise Stage0G5Error("G5_TRAINING_RECORDS_INVALID")
    return [_mapping(item, field="training_record") for item in raw]


def _assert_finite_records(records: Sequence[Mapping[str, Any]], expected: int) -> None:
    if len(records) != expected:
        raise Stage0G5Error(f"G5_RECORD_COUNT_MISMATCH:{len(records)}:{expected}")
    for index, record in enumerate(records, start=1):
        if (
            record.get("status") != "COMMITTED"
            or record.get("global_step") != index
            or not isinstance(record.get("mean_loss"), (int, float))
            or not math.isfinite(float(record["mean_loss"]))
            or not isinstance(record.get("global_gradient_norm"), (int, float))
            or not math.isfinite(float(record["global_gradient_norm"]))
            or not isinstance(record.get("parameter_post_state_hash"), str)
        ):
            raise Stage0G5Error(f"G5_RECORD_NOT_FINITE_COMMITTED:{index}")


def _max_tensor_error(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    left64 = left.detach().cpu().to(torch.float64)
    right64 = right.detach().cpu().to(torch.float64)
    absolute = float((left64 - right64).abs().max().item())
    denominator = torch.maximum(right64.abs(), torch.tensor(1e-30, dtype=torch.float64))
    relative = float(((left64 - right64).abs() / denominator).max().item())
    return absolute, relative


def validate_g5_report_set(
    root: Path,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, JSONValue]:
    """Pure aggregate validator used both at publication and restore."""

    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        by_kind.setdefault(str(report["run_kind"]), []).append(report)
    expected_counts = {
        "fp32_determinism": 2,
        "overfit": 3,
        "bf16": 1,
        "memory": 3,
        "failure_invalid_asset": 1,
        "failure_out_of_range": 1,
        "failure_nonfinite": 1,
        "failure_output_collision": 1,
        "failure_checkpoint_write": 1,
    }
    if {key: len(value) for key, value in by_kind.items()} != expected_counts:
        raise Stage0G5Error("G5_REPORT_SET_CARDINALITY_INVALID")

    fp32 = sorted(by_kind["fp32_determinism"], key=lambda item: int(item["repeat_index"]))
    if fp32[0]["config_hash"] != fp32[1]["config_hash"]:
        raise Stage0G5Error("G5_FP32_CONFIG_HASH_MISMATCH")
    fp32_records = [_records(item) for item in fp32]
    for records in fp32_records:
        _assert_finite_records(records, 1)
    if fp32_records[0][0]["batch_ids"] != fp32_records[1][0]["batch_ids"]:
        raise Stage0G5Error("G5_FP32_BATCH_ID_MISMATCH")
    states = [_tensor_state(root, item) for item in fp32]
    tensors = [_mapping(item.get("tensors"), field="selected_tensors") for item in states]
    if states[0].get("selected_names") != states[1].get("selected_names"):
        raise Stage0G5Error("G5_FP32_SELECTED_NAMES_MISMATCH")
    max_abs = 0.0
    max_rel = 0.0
    for phase in ("parameters_pre", "mean_gradient", "optimizer_gradient", "parameters_post", "update"):
        left = _mapping(tensors[0].get(phase), field=f"tensor.{phase}.left")
        right = _mapping(tensors[1].get(phase), field=f"tensor.{phase}.right")
        if set(left) != set(right):
            raise Stage0G5Error(f"G5_FP32_TENSOR_NAME_MISMATCH:{phase}")
        for name in left:
            if not isinstance(left[name], torch.Tensor) or not isinstance(right[name], torch.Tensor):
                raise Stage0G5Error("G5_FP32_TENSOR_TYPE_INVALID")
            absolute, relative = _max_tensor_error(left[name], right[name])
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            try:
                torch.testing.assert_close(
                    left[name], right[name], atol=1e-6, rtol=1e-5
                )
            except AssertionError as error:
                raise Stage0G5Error(f"G5_FP32_TENSOR_TOLERANCE_FAILED:{phase}:{name}") from error
    loss_delta = abs(
        float(fp32_records[0][0]["mean_loss"])
        - float(fp32_records[1][0]["mean_loss"])
    )
    if loss_delta > 1e-6 + 1e-5 * abs(float(fp32_records[1][0]["mean_loss"])):
        raise Stage0G5Error("G5_FP32_LOSS_TOLERANCE_FAILED")

    overfit_metrics: list[JSONValue] = []
    for report in sorted(by_kind["overfit"], key=lambda item: int(item["repeat_index"])):
        records = _records(report)
        _assert_finite_records(records, 50)
        losses = [float(item["mean_loss"]) for item in records]
        first = statistics.median(losses[:10])
        last = statistics.median(losses[-10:])
        ratio = last / first
        if not last <= first * 0.95:
            raise Stage0G5Error(f"G5_OVERFIT_LOSS_DROP_FAILED:{report['repeat_index']}:{ratio}")
        telemetry = report.get("step_telemetry")
        if not isinstance(telemetry, list) or len(telemetry) != 50:
            raise Stage0G5Error("G5_OVERFIT_TELEMETRY_COUNT_INVALID")
        sample_sequences = [tuple(item["sample_ids"]) for item in telemetry]
        if not sample_sequences[0] or len(set(sample_sequences)) != 1:
            raise Stage0G5Error("G5_OVERFIT_FIXED_SAMPLE_SEQUENCE_INVALID")
        overfit_metrics.append(
            {
                "repeat_index": report["repeat_index"],
                "first10_median": first,
                "last10_median": last,
                "last_to_first_ratio": ratio,
            }
        )

    bf16 = by_kind["bf16"][0]
    _assert_finite_records(_records(bf16), 1)
    boundary = _mapping(bf16.get("data_boundary"), field="bf16.data_boundary")
    if boundary.get("all_valid") is not True or not boundary.get("tokenizer_asset_id"):
        raise Stage0G5Error("G5_BF16_DATA_BOUNDARY_FAILED")
    phase_memory = bf16.get("phase_memory")
    if not isinstance(phase_memory, list) or len(phase_memory) != 3:
        raise Stage0G5Error("G5_BF16_MEMORY_PHASES_INVALID")
    peak_allocated = max(int(item["cuda_peak_allocated_bytes"]) for item in phase_memory)
    if peak_allocated <= 0:
        raise Stage0G5Error("G5_BF16_PEAK_MEMORY_INVALID")
    _tensor_state(root, bf16)

    memory_metrics: list[JSONValue] = []
    for report in sorted(by_kind["memory"], key=lambda item: int(item["repeat_index"])):
        records = _records(report)
        _assert_finite_records(records, 25)
        telemetry = report.get("step_telemetry")
        if not isinstance(telemetry, list) or len(telemetry) != 25:
            raise Stage0G5Error("G5_MEMORY_TELEMETRY_COUNT_INVALID")
        measured = telemetry[5:]
        windows = [measured[index : index + 5] for index in range(0, 20, 5)]
        medians = [
            statistics.median(
                int(_mapping(row["memory"], field="memory.row")["cuda_allocated_bytes"])
                for row in window
            )
            for window in windows
        ]
        monotonic_growth = all(left < right for left, right in zip(medians, medians[1:]))
        final_ratio = medians[-1] / medians[0] if medians[0] else math.inf
        if monotonic_growth or final_ratio > 1.05:
            raise Stage0G5Error(
                f"G5_MEMORY_STABILITY_FAILED:{report['repeat_index']}:{medians}:{final_ratio}"
            )
        memory_metrics.append(
            {
                "repeat_index": report["repeat_index"],
                "window_allocated_medians": medians,
                "strict_monotonic_growth": monotonic_growth,
                "final_to_first_ratio": final_ratio,
            }
        )

    failures: list[JSONValue] = []
    for kind in sorted(key for key in by_kind if key.startswith("failure_")):
        report = by_kind[kind][0]
        failure = _mapping(report.get("failure"), field=f"{kind}.failure")
        if report.get("status") != "EXPECTED_FAILURE_CONFIRMED" or (
            failure.get("success_commit_absent") is not True
        ):
            raise Stage0G5Error(f"G5_FAILURE_PATH_INVALID:{kind}")
        failures.append(
            {
                "run_kind": kind,
                "exception_class": failure.get("exception_class"),
                "last_valid_step": failure.get("last_valid_step"),
                "success_commit_absent": True,
            }
        )

    return {
        "report_count": len(reports),
        "fp32": {
            "config_hash": fp32[0]["config_hash"],
            "loss_absolute_delta": loss_delta,
            "selected_tensor_max_absolute_error": max_abs,
            "selected_tensor_max_relative_error": max_rel,
            "atol": 1e-6,
            "rtol": 1e-5,
        },
        "overfit": overfit_metrics,
        "bf16": {
            "config_hash": bf16["config_hash"],
            "peak_allocated_bytes": peak_allocated,
            "training_wall_seconds": bf16["training_wall_seconds"],
        },
        "memory": memory_metrics,
        "failure_paths": failures,
    }


def _run_child_processes(
    request: TaskExecutionRequest,
    root: Path,
    source: G5SourceBinding,
    *,
    selected_uuid: str,
    suite_root_ref: str,
) -> tuple[list[str], list[dict[str, Any]], dict[str, JSONValue]]:
    suite_root = _logical_path(root, suite_root_ref, field="suite_root")
    suite_root.mkdir(parents=True, exist_ok=True)
    environment_ref = f"{suite_root_ref}/environment.json"
    environment_path = _logical_path(root, environment_ref, field="environment_ref")
    if environment_path.exists():
        if load_canonical_json(environment_path) != request.environment.to_dict():
            raise Stage0G5Error("G5_SUITE_ENVIRONMENT_DRIFT")
    else:
        write_canonical_json(environment_path, request.environment.to_dict())
    environment_sha = sha256_file(environment_path)
    initial_gpu = _nvidia_query(selected_uuid)
    if int(initial_gpu["ecc_uncorrected_volatile"]) != 0:
        raise Stage0G5Error("G5_SELECTED_GPU_UNCORRECTED_ECC_NONZERO")
    if _compute_pids(selected_uuid):
        raise Stage0G5Error("G5_SELECTED_GPU_NOT_EXCLUSIVE_BEFORE_SUITE")

    report_refs: list[str] = []
    reports: list[dict[str, Any]] = []
    process_audit: list[JSONValue] = []
    for spec in _CHILD_SPECS:
        child_root_ref = f"{suite_root_ref}/children/{spec.run_id}"
        config_ref = f"{child_root_ref}/resolved-config.json"
        plan_ref = f"{child_root_ref}/worker-plan.json"
        report_ref = f"{child_root_ref}/worker-report.json"
        output_ref = f"{child_root_ref}/training-output"
        config = _child_config(request.config, spec, output_dir=output_ref)
        config_path = _logical_path(root, config_ref, field="child_config_ref")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.exists():
            if load_canonical_json(config_path) != config.to_dict():
                raise Stage0G5Error("G5_CHILD_CONFIG_DRIFT")
        else:
            write_canonical_json(config_path, config.to_dict())
        plan: dict[str, JSONValue] = {
            "schema_version": WORKER_PLAN_SCHEMA,
            "run_id": spec.run_id,
            "run_kind": spec.run_kind,
            "repeat_index": spec.repeat_index,
            "generator_git_commit": source.git_commit,
            "config_ref": config_ref,
            "config_sha256": sha256_file(config_path),
            "environment_ref": environment_ref,
            "environment_sha256": environment_sha,
            "selected_gpu_uuid": selected_uuid,
            "repeat_fixed_batch": spec.repeat_fixed_batch,
            "capture_tensor_step": spec.capture_tensor_step,
            "memory_warmup_steps": 5 if spec.run_kind == "memory" else None,
            "memory_measure_steps": 20 if spec.run_kind == "memory" else None,
            "expected_failure_code": spec.expected_failure_code,
        }
        plan["artifact_hash"] = canonical_json_hash(plan)
        plan_path = _logical_path(root, plan_ref, field="child_plan_ref")
        if plan_path.exists():
            if load_canonical_json(plan_path) != plan:
                raise Stage0G5Error("G5_CHILD_PLAN_DRIFT")
        else:
            write_canonical_json(plan_path, plan)
        before_pids = _compute_pids(selected_uuid)
        if before_pids:
            raise Stage0G5Error(f"G5_SELECTED_GPU_NOT_EXCLUSIVE:{spec.run_id}:{before_pids}")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = selected_uuid
        environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(root)
        source_path = str(source.repository / "src")
        environment["PYTHONPATH"] = source_path + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        command = [
            sys.executable,
            str(source.repository / "ops/stage0/run_g5_worker.py"),
            "--data-root",
            str(root),
            "--plan-ref",
            plan_ref,
            "--report-ref",
            report_ref,
        ]
        completed = subprocess.run(
            command,
            cwd=source.repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise Stage0G5Error(
                "G5_CHILD_PROCESS_FAILED:"
                f"{spec.run_id}:rc={completed.returncode}:"
                f"stdout={completed.stdout[-2000:]}:stderr={completed.stderr[-4000:]}"
            )
        report = _validate_worker_report(
            root,
            report_ref,
            expected_commit=source.git_commit,
            expected_environment_hash=request.environment.environment_hash,
            selected_uuid=selected_uuid,
        )
        after_pids = _compute_pids(selected_uuid)
        child_pid = int(report["pid"])
        if child_pid in after_pids or after_pids:
            raise Stage0G5Error(
                f"G5_CHILD_GPU_PROCESS_RESIDUE:{spec.run_id}:{after_pids}"
            )
        process_audit.append(
            {
                "run_id": spec.run_id,
                "pid": child_pid,
                "returncode": completed.returncode,
                "gpu_pids_before": list(before_pids),
                "gpu_pids_after": list(after_pids),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            }
        )
        report_refs.append(report_ref)
        reports.append(report)
    final_gpu = _nvidia_query(selected_uuid)
    if (
        final_gpu["ecc_corrected_volatile"] != initial_gpu["ecc_corrected_volatile"]
        or final_gpu["ecc_uncorrected_volatile"] != initial_gpu["ecc_uncorrected_volatile"]
        or _compute_pids(selected_uuid)
    ):
        raise Stage0G5Error("G5_GPU_HEALTH_OR_PROCESS_STATE_DRIFT")
    audit: dict[str, JSONValue] = {
        "selected_gpu_uuid": selected_uuid,
        "initial_gpu": initial_gpu,
        "final_gpu": final_gpu,
        "children": process_audit,
        "no_residual_compute_processes": True,
        "ecc_counters_unchanged": True,
    }
    return report_refs, reports, audit


def _checks(metrics: Mapping[str, JSONValue], audit: Mapping[str, JSONValue]) -> tuple[Stage0GateCheck, ...]:
    refs: tuple[str, ...] = ()
    return (
        Stage0GateCheck(
            "stage0.g5.single_gpu_mapping",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "one physical GPU UUID was explicitly mapped and remained exclusive",
            measurements={
                "selected_gpu_uuid": audit["selected_gpu_uuid"],
                "no_residual_compute_processes": audit["no_residual_compute_processes"],
                "ecc_counters_unchanged": audit["ecc_counters_unchanged"],
            },
            evidence_refs=refs,
        ),
        Stage0GateCheck(
            "stage0.g5.fp32_determinism",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "two fresh FP32 processes matched loss, gradients and updates",
            measurements=_mapping(metrics["fp32"], field="metrics.fp32"),
            evidence_refs=refs,
        ),
        Stage0GateCheck(
            "stage0.g5.overfit",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "three fresh 50-step fixed-real-batch runs exceeded the 5% loss-drop threshold",
            measurements={"repetitions": metrics["overfit"]},
            evidence_refs=refs,
        ),
        Stage0GateCheck(
            "stage0.g5.bf16_data_boundary",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "Pythia-14M BF16 forward/backward/update and real-token boundaries passed",
            measurements=_mapping(metrics["bf16"], field="metrics.bf16"),
            evidence_refs=refs,
        ),
        Stage0GateCheck(
            "stage0.g5.memory_stability",
            Stage0CheckClass.CAPACITY,
            Stage0CheckStatus.PASS,
            "three fresh 20-step measurement windows had no sustained monotonic growth and stayed within 105%",
            exception_eligible=True,
            measurements={"repetitions": metrics["memory"]},
            evidence_refs=refs,
        ),
        Stage0GateCheck(
            "stage0.g5.failure_paths",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "invalid asset, cursor, nonfinite, collision and checkpoint failures published no success",
            measurements={"paths": metrics["failure_paths"]},
            evidence_refs=refs,
        ),
    )


def run_formal_g5_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: tuple[str, ...],
) -> TaskRunResult:
    source = _capture_source()
    if request.config.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G5Error("G5_REQUEST_TASK_OR_SCOPE_INVALID")
    if set(source_refs) != set(request.config.section("orchestration")["input_result_refs"]):  # type: ignore[index]
        raise Stage0G5Error("G5_INPUT_SOURCE_REF_SET_MISMATCH")
    g4_loaded = [
        load_committed_task_artifact(root, reference, require_formal=True)
        for reference in source_refs
    ]
    if (
        {item.identity.task_id for item in g4_loaded}
        != {"stage0.05_config_run_identity_and_seeds"}
        or {item.identity.artifact_kind for item in g4_loaded}
        != {"resolved_config", "run_identity", "seed_plan", "provenance_record"}
    ):
        raise Stage0G5Error("G5_G4_INPUT_SET_INVALID")
    provenance_item = next(
        item for item in g4_loaded if item.identity.artifact_kind == "provenance_record"
    )
    g4_envelope = _mapping(provenance_item.payload, field="g4_envelope")
    provenance = _mapping(g4_envelope.get("provenance_record"), field="g4_provenance")
    mapping = provenance.get("device_mapping")
    if not isinstance(mapping, list) or len(mapping) != 4:
        raise Stage0G5Error("G5_G4_DEVICE_MAPPING_INVALID")
    selected_uuid = str(mapping[0])
    suite_root_ref = (
        f"evidence/stage0/g5-suite/{request.config.config_hash}/"
        f"{request.environment.environment_hash}"
    )
    report_refs, reports, process_audit = _run_child_processes(
        request,
        root,
        source,
        selected_uuid=selected_uuid,
        suite_root_ref=suite_root_ref,
    )
    metrics = validate_g5_report_set(root, reports)
    checks = _checks(metrics, process_audit)
    checked_at = max(
        (_timestamp(item["completed_at"], field="worker.completed_at") for item in reports)
    ).isoformat().replace("+00:00", "Z")
    all_evidence_refs = tuple(source_refs) + tuple(report_refs)
    gate = GateRecord(
        gate_id="stage0.G5",
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "selected_gpu_uuid": selected_uuid,
            "child_process_count": len(reports),
            "fp32_max_absolute_error": metrics["fp32"]["selected_tensor_max_absolute_error"],  # type: ignore[index]
            "overfit_repetitions": 3,
            "memory_repetitions": 3,
            "failure_paths": 5,
        },
        threshold={
            "fp32_atol": 1e-6,
            "fp32_rtol": 1e-5,
            "overfit_last10_vs_first10_max_ratio": 0.95,
            "memory_final_vs_first_window_max_ratio": 1.05,
            "memory_strict_monotonic_growth_allowed": False,
            "residual_compute_processes_allowed": 0,
        },
        evidence_refs=all_evidence_refs,
    )
    environment_id = str(request.config.base_config.section("runtime")["environment_id"])
    report = Stage0GateReport(
        gate_id="stage0.G5",
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        config_hashes={
            f"{item['run_kind']}-{int(item['repeat_index']):02d}": str(item["config_hash"])
            for item in reports
        },
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(_logical_path(root, reference, field="evidence_ref")),
                schema_version=(
                    "stage0-g5-worker-report-v1"
                    if reference in report_refs
                    else "task-output-commit-v1"
                ),
            )
            for reference in all_evidence_refs
        ),
        checks=checks,
    )
    validation = _payload_with_hash(
        {
            "schema_version": "stage0-g5-validation-report-v1",
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "metrics": metrics,
            "process_audit": process_audit,
        }
    )
    refs: dict[str, str] = {}
    publication_sources = (*all_evidence_refs,)
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g5-evidence-v1",
            "artifact_role": kind,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "selected_gpu_uuid": selected_uuid,
            "suite_root_ref": suite_root_ref,
            "worker_report_refs": report_refs,
            "validation_report": validation,
            "gate_record": gate.to_dict(),
            "gate_report": report.to_dict(),
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        refs[kind] = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
            source_refs=publication_sources,
        ).commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        checkpoint_ref=refs["checkpoint_commit"],
        message="Stage 0 G5 fresh-process single-GPU suite passed",
        metadata={"stage0_g5_specialized": True, "gate_id": "stage0.G5"},
    )


def validate_formal_g5_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G5Error("G5_OUTPUT_SET_INVALID")
    envelopes: list[dict[str, Any]] = []
    for kind, reference in outputs.items():
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != request.config.config_hash
        ):
            raise Stage0G5Error("G5_OUTPUT_COMMIT_IDENTITY_INVALID")
        envelope = _mapping(loaded.payload, field=f"output.{kind}")
        declared = envelope.pop("artifact_hash", None)
        if (
            envelope.get("schema_version") != "stage0-g5-evidence-v1"
            or envelope.get("artifact_role") != kind
            or envelope.get("status") != "PASS"
            or envelope.get("generator_git_commit") != source.git_commit
            or envelope.get("environment_hash") != request.environment.environment_hash
            or declared != canonical_json_hash(envelope)
        ):
            raise Stage0G5Error("G5_OUTPUT_ENVELOPE_INVALID")
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
            for key, value in item.items()
            if key not in {"artifact_role", "artifact_hash"}
        }
        != canonical
        for item in envelopes[1:]
    ):
        raise Stage0G5Error("G5_OUTPUT_ROLE_PAYLOAD_DRIFT")
    report_refs = canonical.get("worker_report_refs")
    if not isinstance(report_refs, list) or len(report_refs) != len(_CHILD_SPECS):
        raise Stage0G5Error("G5_OUTPUT_WORKER_REFS_INVALID")
    reports = [
        _validate_worker_report(
            root,
            str(reference),
            expected_commit=source.git_commit,
            expected_environment_hash=request.environment.environment_hash,
            selected_uuid=str(canonical["selected_gpu_uuid"]),
        )
        for reference in report_refs
    ]
    replayed_metrics = validate_g5_report_set(root, reports)
    validation = _mapping(canonical.get("validation_report"), field="validation")
    validation_declared = validation.pop("artifact_hash", None)
    if (
        validation_declared != canonical_json_hash(validation)
        or validation.get("metrics") != replayed_metrics
        or validation.get("status") != "PASS"
    ):
        raise Stage0G5Error("G5_OUTPUT_VALIDATION_REPLAY_MISMATCH")
    gate = GateRecord.from_mapping(
        _mapping(canonical.get("gate_record"), field="gate_record")
    )
    gate_report = Stage0GateReport.from_mapping(
        _mapping(canonical.get("gate_report"), field="gate_report")
    )
    if (
        gate.gate_id != "stage0.G5"
        or gate.status is not GateStatus.PASS
        or gate_report.gate_id != "stage0.G5"
        or gate_report.status.value != "PASS"
    ):
        raise Stage0G5Error("G5_OUTPUT_GATE_INVALID")
    return gate


def execute_stage0_g5(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g4_index_ref: str,
) -> Stage0G5FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g4_formal_state(
        data_root=root,
        index_ref=g4_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g5_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g5-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G5Error(
            f"G5_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=state.environment,
    )
    gate = validate_formal_g5_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g5_training_smoke": outputs["training_smoke_result"],
            "g5_event_stream": outputs["event_stream"],
            "g5_checkpoint_commit": outputs["checkpoint_commit"],
            "gate_stage0_g5": outputs["training_smoke_result"],
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
    index_payload: dict[str, JSONValue] = {
        "schema_version": "stage0-g5-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g4_index_ref": state.index_ref,
        "g4_index_sha256": state.index_sha256,
        "g4_gate_artifact_hash": state.gate_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_ref": outputs["training_smoke_result"],
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.07_ddp_and_gradient_semantics",
        "next_input_refs": list(outputs.values()),
    }
    index_payload["artifact_hash"] = canonical_json_hash(index_payload)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(
        _logical_path(root, index_ref, field="index_ref"), index_payload
    )
    return Stage0G5FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def load_stage0_g5_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G5FormalState:
    """Load G5 only after replaying G4, every worker report, and the gate."""

    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g5_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g4_index_ref",
        "g4_index_sha256",
        "g4_gate_artifact_hash",
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
    if set(raw) != expected or raw.get("schema_version") != (
        "stage0-g5-formalization-index-v1"
    ):
        raise Stage0G5Error("G5_STATE_INDEX_FIELDS_OR_VERSION_INVALID")
    declared = raw.pop("artifact_hash")
    if declared != canonical_json_hash(raw):
        raise Stage0G5Error("G5_STATE_INDEX_HASH_MISMATCH")
    raw["artifact_hash"] = declared
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G5Error("G5_STATE_GENERATOR_COMMIT_MISMATCH")

    g4_state = load_stage0_g4_formal_state(
        data_root=root,
        index_ref=str(raw["g4_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    if (
        raw.get("g4_index_sha256") != g4_state.index_sha256
        or raw.get("g4_gate_artifact_hash") != g4_state.gate_artifact_hash
    ):
        raise Stage0G5Error("G5_STATE_G4_BINDING_MISMATCH")
    config_path = _logical_path(root, raw["config_ref"], field="config_ref")
    config = ResolvedConfigV2.from_mapping(
        _mapping(load_canonical_json(config_path), field="config")
    )
    if config.task_id != TASK_ID or config.config_hash != raw.get("config_hash"):
        raise Stage0G5Error("G5_STATE_CONFIG_MISMATCH")
    outputs = _mapping(raw["task_output_refs"], field="task_output_refs")
    if set(outputs) != _OUTPUT_KINDS or any(
        not isinstance(value, str) for value in outputs.values()
    ):
        raise Stage0G5Error("G5_STATE_OUTPUT_SET_INVALID")
    ordered_outputs = {
        key: str(outputs[key])
        for key in ("training_smoke_result", "event_stream", "checkpoint_commit")
    }
    request = TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=g4_state.environment,
    )
    gate = validate_formal_g5_outputs(request, root, ordered_outputs)
    if raw.get("gate_ref") != outputs["training_smoke_result"]:
        raise Stage0G5Error("G5_STATE_GATE_REF_MISMATCH")

    environment_path = _logical_path(
        root, raw["environment_ref"], field="environment_ref"
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(load_canonical_json(environment_path), field="environment")
    )
    next_inputs = raw.get("next_input_refs")
    if (
        environment.environment_hash != raw.get("environment_hash")
        or gate.gate_id not in environment.passed_gate_ids
        or environment.evidence_refs.get("gate_stage0_g5") != raw.get("gate_ref")
        or raw.get("next_task_id") != "stage0.07_ddp_and_gradient_semantics"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(outputs.values())
    ):
        raise Stage0G5Error("G5_STATE_ENVIRONMENT_OR_HANDOFF_MISMATCH")
    loaded_gate = load_committed_task_artifact(
        root, str(raw["gate_ref"]), require_formal=True
    )
    gate_hash = loaded_gate.payload.get("artifact_hash")
    if not isinstance(gate_hash, str) or _SHA256_RE.fullmatch(gate_hash) is None:
        raise Stage0G5Error("G5_STATE_GATE_ARTIFACT_HASH_INVALID")
    return Stage0G5FormalState(
        environment=environment,
        task_output_refs=ordered_outputs,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate_hash,
        g4_index_ref=str(raw["g4_index_ref"]),
    )


__all__ = [
    "G5SourceBinding",
    "Stage0G5Error",
    "Stage0G5FormalState",
    "Stage0G5FormalizationResult",
    "TASK_ID",
    "build_stage0_g5_config",
    "execute_stage0_g5",
    "load_stage0_g5_formal_state",
    "run_formal_g5_task",
    "validate_formal_g5_outputs",
    "validate_g5_report_set",
]
