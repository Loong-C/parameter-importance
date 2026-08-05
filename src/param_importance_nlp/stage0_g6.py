"""Formal Stage 0 G6 four-GPU DDP/NCCL suite and replayable gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

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
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g5 import Stage0G5FormalState, load_stage0_g5_formal_state
from .stage0_g6_worker import (
    FAILURE_MARKER,
    WORKER_PLAN_SCHEMA,
    WORKER_REPORT_SCHEMA,
    WORLD_SIZE,
)
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID = "stage0.07_ddp_and_gradient_semantics"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OUTPUT_KINDS = {
    "distributed_validation",
    "gradient_semantics_report",
    "communication_report",
}
_CRITICAL_SOURCE_REFS = (
    "ops/stage0/formalize_g6.py",
    "ops/stage0/run_g6_worker.py",
    "src/param_importance_nlp/experiments/task_runners.py",
    "src/param_importance_nlp/stage0_g5.py",
    "src/param_importance_nlp/stage0_g6.py",
    "src/param_importance_nlp/stage0_g6_worker.py",
    "schemas/stage0-g6-evidence-v1.json",
    "schemas/stage0-g6-formalization-index-v1.json",
    "schemas/stage0-g6-worker-plan-v2.json",
    "schemas/stage0-g6-worker-report-v1.json",
)
_REPORT_FIELDS = {
    "schema_version",
    "run_id",
    "run_kind",
    "repeat_index",
    "status",
    "started_at",
    "completed_at",
    "generator_git_commit",
    "plan_ref",
    "plan_sha256",
    "config_ref",
    "config_sha256",
    "config_hash",
    "environment_ref",
    "environment_sha256",
    "environment_hash",
    "selected_gpu_uuids",
    "backend",
    "world_size",
    "rank_records",
    "metrics",
    "artifact_hash",
}


class Stage0G6Error(RuntimeError):
    """G6 launch, numerical replay, or evidence publication failed closed."""


@dataclass(frozen=True, slots=True)
class G6SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G6FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G6FormalState:
    """Revalidated G6 handoff consumed by observability and recovery gates."""

    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    g5_index_ref: str


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


def _capture_source() -> G6SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    top = _git(repository, "rev-parse", "--show-toplevel")
    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    tracked = _git(
        repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS
    )
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if any(item.returncode != 0 for item in (top, head, branch, tracked, status)):
        raise Stage0G6Error("G6_SOURCE_GIT_PROBE_FAILED")
    if Path(top.stdout.strip()).resolve() != repository:
        raise Stage0G6Error("G6_SOURCE_GIT_ROOT_MISMATCH")
    commit = head.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch.stdout.strip():
        raise Stage0G6Error("G6_SOURCE_GIT_IDENTITY_INVALID")
    if status.stdout.strip():
        raise Stage0G6Error("G6_FORMAL_SOURCE_DIRTY")
    return G6SourceBinding(repository, commit, branch.stdout.strip())


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G6Error(f"G6_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G6Error(f"G6_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G6Error(f"G6_LOGICAL_PATH_ESCAPE:{field}")
    resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage0G6Error(f"G6_LOGICAL_PATH_ESCAPE:{field}") from error
    return resolved


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Stage0G6Error(f"G6_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G6Error(f"G6_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0G6Error(f"G6_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _payload_with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G6Error("G6_EXISTING_CONTROL_FILE_DRIFT")
        return
    write_canonical_json(path, dict(value))


def _gpu_uuids(state: Stage0G5FormalState, root: Path) -> tuple[str, ...]:
    reference = state.environment.evidence_refs.get("g4_provenance")
    if not isinstance(reference, str):
        raise Stage0G6Error("G6_G4_PROVENANCE_REF_MISSING")
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    envelope = _mapping(loaded.payload, field="g4_provenance_envelope")
    provenance = _mapping(envelope.get("provenance_record"), field="g4_provenance")
    mapping = provenance.get("device_mapping")
    if (
        not isinstance(mapping, list)
        or len(mapping) != WORLD_SIZE
        or len(set(mapping)) != WORLD_SIZE
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in mapping)
    ):
        raise Stage0G6Error("G6_G4_DEVICE_MAPPING_INVALID")
    return tuple(mapping)


def build_stage0_g6_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G5FormalState,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    gpu_uuids = _gpu_uuids(state, root)
    previous = state.config
    base = previous.base_config
    model = base.section("model")
    data = base.section("data")
    runtime = base.section("runtime")
    optimizer = base.section("optimizer")
    assert all(isinstance(item, dict) for item in (model, data, runtime, optimizer))
    output_dir = f"evidence/stage0/tasks/07-{state.gate_artifact_hash}"
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=output_dir,
        base_overrides={
            "identity": {"route": f"stage0-g6-{state.gate_artifact_hash[:12]}"},
            "runtime": {
                **runtime,
                "device": "cuda",
                "allow_dirty_worktree": False,
            },
            "model": model,
            "data": data,
            "loss": base.section("loss"),
            "batching": {
                "global_batch_size": 8,
                "per_device_batch_size": 1,
                "microbatch_size": 1,
                "accumulation_steps": 2,
                "no_sync": True,
            },
            "distributed": {
                "world_size": WORLD_SIZE,
                "backend": "nccl",
                "device_ids": list(range(WORLD_SIZE)),
                "timeout_seconds": 180,
            },
            "precision": {
                **_mapping(base.section("precision"), field="precision"),
                "compute_dtype": "bfloat16",
                "gradient_dtype": "float32",
                "statistic_dtype": "float32",
                "amp": True,
            },
            "optimizer": {
                **optimizer,
                "learning_rate": 0.0003,
                "weight_decay": 0.0,
                "fused": False,
                "foreach": False,
            },
        },
        v2_overrides={
            "execution": {"timeout_seconds": 3600, "max_attempts": 1},
            "training": {
                "max_steps": 1,
                "deterministic_algorithms": True,
                "gradient_clip_max_norm": 1.0,
            },
            "scheduler": previous.section("scheduler"),
            "data_loader": previous.section("data_loader"),
            "providers": previous.section("providers"),
            "evaluation": {**_mapping(previous.section("evaluation"), field="evaluation"), "enabled": False},
            "profiling": previous.section("profiling"),
            "checkpoint_schedule": {
                "segments": [{"start_step": 0, "end_step": None, "every_steps": 1}],
                "save_on_phase_end": True,
                "save_optimizer": True,
                "save_rng": True,
                "save_data_state": True,
            },
            "precision_runtime": {
                "autocast_enabled": True,
                "autocast_dtype": "bfloat16",
                "grad_scaler_enabled": False,
            },
            "optimizer_runtime": previous.section("optimizer_runtime"),
            "launcher": {
                "kind": "torchrun",
                "backend": "nccl",
                "world_size": WORLD_SIZE,
                "init_method": "env",
                "init_ref": None,
                "rendezvous_id": f"stage0-g6-{state.gate_artifact_hash[:16]}",
                "max_restarts": 0,
            },
            "recovery": previous.section("recovery"),
        },
    )


def _compute_pids(selected: Sequence[str]) -> dict[str, list[int]]:
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
        raise Stage0G6Error("G6_NVIDIA_SMI_PROCESS_QUERY_FAILED")
    result = {uuid: [] for uuid in selected}
    for line in query.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        if fields[1] in result:
            result[fields[1]].append(int(fields[0]))
    return {key: sorted(set(value)) for key, value in result.items()}


def _gpu_snapshot(selected: Sequence[str]) -> dict[str, JSONValue]:
    cards: list[dict[str, JSONValue]] = []
    for uuid in selected:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--id={uuid}",
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
            raise Stage0G6Error("G6_NVIDIA_SMI_GPU_QUERY_FAILED")
        lines = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        fields = [item.strip() for item in lines[0].split(",")] if len(lines) == 1 else []
        if len(fields) != 8 or fields[0] != uuid:
            raise Stage0G6Error("G6_NVIDIA_SMI_GPU_QUERY_FIELDS_INVALID")
        try:
            numeric = [int(value) for value in fields[1:2] + fields[3:]]
        except ValueError as error:
            raise Stage0G6Error("G6_NVIDIA_SMI_GPU_QUERY_NOT_NUMERIC") from error
        cards.append(
            {
                "uuid": uuid,
                "index": numeric[0],
                "pci_bus_id": fields[2],
                "temperature_c": numeric[1],
                "memory_used_mib": numeric[2],
                "memory_total_mib": numeric[3],
                "ecc_corrected_volatile": numeric[4],
                "ecc_uncorrected_volatile": numeric[5],
            }
        )
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if topology.returncode != 0 or not topology.stdout.strip():
        raise Stage0G6Error("G6_NVIDIA_SMI_TOPOLOGY_QUERY_FAILED")
    return {
        "cards": cards,
        "compute_pids": _compute_pids(selected),
        "topology_sha256": hashlib.sha256(topology.stdout.encode("utf-8")).hexdigest(),
        "topology_text": topology.stdout,
    }


def _wait_for_no_compute_pids(selected: Sequence[str], *, timeout: float = 10.0) -> dict[str, list[int]]:
    deadline = time.monotonic() + timeout
    while True:
        observed = _compute_pids(selected)
        if all(not value for value in observed.values()):
            return observed
        if time.monotonic() >= deadline:
            return observed
        time.sleep(0.2)


def _plan_payload(
    *,
    run_id: str,
    run_kind: str,
    repeat_index: int,
    source: G6SourceBinding,
    config_ref: str,
    config_sha256: str,
    config_hash: str,
    environment_ref: str,
    environment_sha256: str,
    environment_hash: str,
    selected: Sequence[str],
    report_ref: str,
    timeout_seconds: int,
) -> dict[str, JSONValue]:
    return _payload_with_hash(
        {
            "schema_version": WORKER_PLAN_SCHEMA,
            "run_id": run_id,
            "run_kind": run_kind,
            "repeat_index": repeat_index,
            "generator_git_commit": source.git_commit,
            "config_ref": config_ref,
            "config_sha256": config_sha256,
            "config_hash": config_hash,
            "environment_ref": environment_ref,
            "environment_sha256": environment_sha256,
            "environment_hash": environment_hash,
            "selected_gpu_uuids": list(selected),
            "report_ref": report_ref,
            "timeout_seconds": timeout_seconds,
            "collective_protocol": {
                "protocol_version": "stage0-unified-measurement-v2",
                "warmup_iterations": 20,
                "measured_iterations": 50,
                "samples_per_measurement": 3,
                "tensor_elements": [65536, 262144, 4194304],
            },
        }
    )


def _run_launch(
    *,
    root: Path,
    source: G6SourceBinding,
    plan_ref: str,
    run_id: str,
    selected: Sequence[str],
    wall_timeout: float,
    transcript_ref: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(source.repository / "ops" / "stage0" / "run_g6_worker.py"),
        "--data-root",
        str(root),
        "--plan-ref",
        plan_ref,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(root)
    current_pythonpath = environment.get("PYTHONPATH")
    source_path = str(source.repository / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    started_at = datetime.now(timezone.utc)
    process = subprocess.Popen(
        command,
        cwd=source.repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=wall_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - formal G6 runs on the Linux GPU server
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
            stdout, stderr = process.communicate()
    completed_at = datetime.now(timezone.utc)
    residual = _wait_for_no_compute_pids(selected)
    transcript = _payload_with_hash(
        {
            "schema_version": "stage0-g6-launch-transcript-v1",
            "run_id": run_id,
            "plan_ref": plan_ref,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
            "duration_seconds": (completed_at - started_at).total_seconds(),
            "return_code": process.returncode,
            "timed_out": timed_out,
            "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-20000:],
            "failure_marker_observed": FAILURE_MARKER in stdout or FAILURE_MARKER in stderr,
            "residual_compute_pids": residual,
        }
    )
    transcript_path = _logical_path(root, transcript_ref, field="transcript_ref")
    if transcript_path.exists():
        raise Stage0G6Error("G6_LAUNCH_TRANSCRIPT_COLLISION")
    write_canonical_json(transcript_path, transcript)
    return transcript


def _validate_rank_records(
    value: object, selected: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != WORLD_SIZE:
        raise Stage0G6Error("G6_REPORT_RANK_RECORDS_INVALID")
    records = [_mapping(item, field="rank_record") for item in value]
    for rank, record in enumerate(records):
        if (
            record.get("rank") != rank
            or record.get("local_rank") != rank
            or record.get("world_size") != WORLD_SIZE
            or record.get("gpu_uuid") != selected[rank]
            or record.get("logical_device") != rank
            or record.get("cuda_visible_devices") != ",".join(selected)
            or not isinstance(record.get("pid"), int)
            or int(record.get("total_memory_bytes", 0)) <= 0
        ):
            raise Stage0G6Error("G6_REPORT_RANK_MAPPING_INVALID")
    return records


def _validate_worker_report(
    root: Path,
    reference: str,
    *,
    source: G6SourceBinding,
    request: TaskExecutionRequest,
    selected: Sequence[str],
) -> dict[str, Any]:
    path = _logical_path(root, reference, field="worker_report_ref")
    report = _mapping(load_canonical_json(path), field="worker_report")
    if set(report) != _REPORT_FIELDS or report.get("schema_version") != WORKER_REPORT_SCHEMA:
        raise Stage0G6Error("G6_REPORT_FIELDS_OR_VERSION_INVALID")
    declared = report.pop("artifact_hash")
    if declared != canonical_json_hash(report):
        raise Stage0G6Error("G6_REPORT_HASH_MISMATCH")
    report["artifact_hash"] = declared
    plan_path = _logical_path(root, report["plan_ref"], field="plan_ref")
    config_path = _logical_path(root, report["config_ref"], field="config_ref")
    environment_path = _logical_path(
        root, report["environment_ref"], field="environment_ref"
    )
    if (
        report.get("status") != "PASS"
        or report.get("generator_git_commit") != source.git_commit
        or report.get("config_hash") != request.config.config_hash
        or report.get("environment_hash") != request.environment.environment_hash
        or report.get("selected_gpu_uuids") != list(selected)
        or report.get("backend") != "nccl"
        or report.get("world_size") != WORLD_SIZE
        or report.get("plan_sha256") != sha256_file(plan_path)
        or report.get("config_sha256") != sha256_file(config_path)
        or report.get("environment_sha256") != sha256_file(environment_path)
    ):
        raise Stage0G6Error("G6_REPORT_IDENTITY_INVALID")
    _timestamp(report["started_at"], field="report.started_at")
    _timestamp(report["completed_at"], field="report.completed_at")
    _validate_rank_records(report["rank_records"], selected)
    _mapping(report["metrics"], field="report.metrics")
    return report


def _comparison_pass(value: object) -> bool:
    comparison = _mapping(value, field="comparison")
    if comparison.get("effective_count_match") is not True:
        return False
    if float(comparison.get("loss_absolute_error", math.inf)) > 1e-6:
        return False
    if float(comparison.get("clip_factor_absolute_error", math.inf)) > 1e-6:
        return False
    for field in ("gradient", "clipped_gradient", "parameters"):
        tensor = _mapping(comparison.get(field), field=f"comparison.{field}")
        absolute = float(tensor.get("max_absolute_error", math.inf))
        relative = float(tensor.get("max_relative_error", math.inf))
        relative_l2 = float(tensor.get("relative_l2_error", math.inf))
        if not math.isfinite(absolute + relative + relative_l2):
            return False
        if absolute > 1e-6 and relative > 1e-5:
            return False
        if relative_l2 > 1e-5:
            return False
    return True


def validate_g6_report_set(
    reports: Sequence[Mapping[str, Any]],
    launch_audits: Sequence[Mapping[str, Any]],
    device_audit: Mapping[str, Any],
) -> dict[str, JSONValue]:
    """Replay all hard G6 criteria without requiring a GPU."""

    collective = sorted(
        (item for item in reports if item.get("run_kind") == "collective"),
        key=lambda item: int(item["repeat_index"]),
    )
    semantic_reports = [item for item in reports if item.get("run_kind") == "semantic"]
    recovery_reports = [item for item in reports if item.get("run_kind") == "recovery"]
    if (
        len(collective) != 3
        or [item.get("repeat_index") for item in collective] != [0, 1, 2]
        or len(semantic_reports) != 1
        or len(recovery_reports) != 1
    ):
        raise Stage0G6Error("G6_REPORT_SUITE_CARDINALITY_INVALID")
    medians_by_size: dict[int, list[float]] = {}
    for report in collective:
        metrics = _mapping(report["metrics"], field="collective.metrics")
        if not all(
            metrics.get(field) is True
            for field in ("broadcast_pass", "all_gather_pass", "barrier_pass")
        ) or metrics.get("scalar_expected_sum") != 10:
            raise Stage0G6Error("G6_COLLECTIVE_FUNCTIONAL_CHECK_FAILED")
        messages = metrics.get("message_metrics")
        if not isinstance(messages, list) or len(messages) != 3:
            raise Stage0G6Error("G6_COLLECTIVE_MESSAGE_SET_INVALID")
        for raw in messages:
            item = _mapping(raw, field="collective.message")
            size = int(item.get("tensor_elements", 0))
            median = float(item.get("median_seconds", math.nan))
            p95 = float(item.get("p95_seconds", math.nan))
            samples = item.get("sample_seconds")
            if (
                item.get("warmup_iterations") != 20
                or item.get("measured_iterations") != 50
                or item.get("samples_per_measurement") != 3
                or not isinstance(samples, list)
                or len(samples) != 50
                or not math.isfinite(median)
                or median <= 0
                or not math.isfinite(p95)
                or p95 > 2.0 * median
                or float(item.get("throughput_bytes_per_second", 0.0)) <= 0
                or item.get("result_value") != 10.0
            ):
                raise Stage0G6Error("G6_COLLECTIVE_MEASUREMENT_THRESHOLD_FAILED")
            medians_by_size.setdefault(size, []).append(median)
    coefficients: dict[str, JSONValue] = {}
    for size, medians in medians_by_size.items():
        if len(medians) != 3 or statistics.mean(medians) <= 0:
            raise Stage0G6Error("G6_COLLECTIVE_REPETITION_MISSING")
        coefficient = statistics.pstdev(medians) / statistics.mean(medians)
        if coefficient > 0.20:
            raise Stage0G6Error("G6_COLLECTIVE_REPEAT_CV_EXCEEDED")
        coefficients[str(size)] = coefficient

    semantic = _mapping(semantic_reports[0]["metrics"], field="semantic.metrics")
    shards = _mapping(semantic.get("data_shards"), field="semantic.data_shards")
    for suffix in ("main", "tail"):
        expected = shards.get(f"expected_{suffix}")
        observed = shards.get(f"observed_{suffix}")
        if (
            not isinstance(expected, list)
            or not isinstance(observed, list)
            or len(observed) != len(set(observed))
            or set(observed) != set(expected)
        ):
            raise Stage0G6Error("G6_DATA_SHARD_DISJOINTNESS_FAILED")
    comparisons = _mapping(semantic.get("comparisons"), field="semantic.comparisons")
    expected_comparisons = {
        "ddp_full",
        "accumulation_sync_each",
        "accumulation_no_sync",
        "incomplete_tail",
    }
    if set(comparisons) != expected_comparisons or any(
        not _comparison_pass(value) for value in comparisons.values()
    ):
        raise Stage0G6Error("G6_FP32_NUMERICAL_EQUIVALENCE_FAILED")
    communication = _mapping(semantic.get("communication"), field="semantic.communication")
    expected_order = [
        "no_sync_forward_backward",
        "sync_backward",
        "gradient_sync_complete",
        "clip",
        "optimizer",
        "scheduler",
    ]
    if (
        communication.get("microbatches_per_rank") != 2
        or communication.get("no_sync_calls") != 1
        or int(communication.get("sync_each_calls", 0)) <= 1
        or communication.get("event_order") != expected_order
    ):
        raise Stage0G6Error("G6_NO_SYNC_COMMUNICATION_SEMANTICS_FAILED")
    clipping = _mapping(semantic.get("clipping"), field="semantic.clipping")
    if (
        not 0 < float(clipping.get("serial_clip_factor", 0.0)) < 1
        or float(clipping.get("serial_global_gradient_norm", 0.0)) <= 0.05
        or clipping.get("shared_parameter_identity") is not True
        or float(clipping.get("zero_gradient_max_abs", math.inf)) != 0.0
    ):
        raise Stage0G6Error("G6_CLIPPING_OR_PARAMETER_EDGE_CASE_FAILED")
    nonfinite = semantic.get("nonfinite")
    if (
        not isinstance(nonfinite, list)
        or len(nonfinite) != WORLD_SIZE
        or any(
            not isinstance(item, Mapping)
            or item.get("global_finite") is not False
            or item.get("skipped") is not True
            or item.get("parameters_unchanged") is not True
            for item in nonfinite
        )
    ):
        raise Stage0G6Error("G6_NONFINITE_GLOBAL_SKIP_FAILED")
    bf16 = semantic.get("bf16")
    if not isinstance(bf16, list) or len(bf16) != WORLD_SIZE:
        raise Stage0G6Error("G6_BF16_RANK_SET_INVALID")
    coordinate_hashes: set[str] = set()
    bf16_batch_ids: list[str] = []
    for raw in bf16:
        item = _mapping(raw, field="bf16.rank")
        losses = item.get("mean_losses")
        statuses = item.get("statuses")
        if (
            item.get("global_steps") != 1
            or item.get("attempts") != 1
            or item.get("record_count") != 1
            or statuses != ["COMMITTED"]
            or not isinstance(losses, list)
            or len(losses) != 1
            or losses[0] is None
            or not math.isfinite(float(losses[0]))
            or item.get("finite_parameters") is not True
            or int(item.get("peak_memory_bytes", 0)) <= 0
            or float(item.get("wall_seconds", 0.0)) <= 0
            or int(item.get("checkpoint_count", 0)) < 1
            or int(item.get("asset_evidence_count", 0)) < 3
        ):
            raise Stage0G6Error("G6_BF16_FUNCTIONAL_PATH_FAILED")
        coordinate_hashes.add(str(item.get("coordinate_registry_hash")))
        batch_ids = item.get("batch_ids")
        if not isinstance(batch_ids, list) or not batch_ids:
            raise Stage0G6Error("G6_BF16_BATCH_IDS_MISSING")
        bf16_batch_ids.extend(str(value) for value in batch_ids)
    if len(coordinate_hashes) != 1 or len(bf16_batch_ids) != len(set(bf16_batch_ids)):
        raise Stage0G6Error("G6_BF16_RANK_IDENTITY_OR_SHARD_FAILED")

    failure_audits = [item for item in launch_audits if item.get("run_id") == "failure-rank-00"]
    if len(failure_audits) != 1:
        raise Stage0G6Error("G6_FAILURE_AUDIT_MISSING")
    failure = failure_audits[0]
    residual = _mapping(failure.get("residual_compute_pids"), field="failure.residual")
    if (
        failure.get("timed_out") is not False
        or not isinstance(failure.get("return_code"), int)
        or int(failure["return_code"]) == 0
        or float(failure.get("duration_seconds", math.inf)) > 60.0
        or failure.get("failure_marker_observed") is not True
        or any(value for value in residual.values())
    ):
        raise Stage0G6Error("G6_CONTROLLED_RANK_FAILURE_FAILED")
    recovery = _mapping(recovery_reports[0]["metrics"], field="recovery.metrics")
    if recovery.get("healthy_restart") is not True or recovery.get("scalar_sum") != 10:
        raise Stage0G6Error("G6_POST_FAILURE_RECOVERY_FAILED")
    for audit in launch_audits:
        audit_residual = _mapping(
            audit.get("residual_compute_pids"), field="launch.residual"
        )
        if any(value for value in audit_residual.values()):
            raise Stage0G6Error("G6_LAUNCH_RESIDUAL_PROCESS_DETECTED")

    before = _mapping(device_audit.get("before"), field="device.before")
    after = _mapping(device_audit.get("after"), field="device.after")
    if any(value for value in _mapping(before.get("compute_pids"), field="before.pids").values()):
        raise Stage0G6Error("G6_PREFLIGHT_EXTERNAL_COMPUTE_PROCESS")
    if any(value for value in _mapping(after.get("compute_pids"), field="after.pids").values()):
        raise Stage0G6Error("G6_POSTFLIGHT_COMPUTE_PROCESS_RESIDUE")
    before_cards = before.get("cards")
    after_cards = after.get("cards")
    if not isinstance(before_cards, list) or not isinstance(after_cards, list):
        raise Stage0G6Error("G6_DEVICE_CARD_AUDIT_INVALID")
    before_ecc = {
        str(item["uuid"]): (
            int(item["ecc_corrected_volatile"]),
            int(item["ecc_uncorrected_volatile"]),
        )
        for item in before_cards
        if isinstance(item, Mapping)
    }
    after_ecc = {
        str(item["uuid"]): (
            int(item["ecc_corrected_volatile"]),
            int(item["ecc_uncorrected_volatile"]),
        )
        for item in after_cards
        if isinstance(item, Mapping)
    }
    if before_ecc != after_ecc or before.get("topology_sha256") != after.get("topology_sha256"):
        raise Stage0G6Error("G6_ECC_OR_TOPOLOGY_DRIFT")
    return {
        "collective_process_group_rebuilds": 3,
        "collective_median_cv_by_elements": coefficients,
        "collective_samples_per_measurement": 3,
        "fp32_comparisons": comparisons,
        "no_sync_gradient_collectives": communication["no_sync_calls"],
        "sync_each_gradient_collectives": communication["sync_each_calls"],
        "bf16_rank_count": WORLD_SIZE,
        "controlled_failure_duration_seconds": failure["duration_seconds"],
        "residual_process_count": 0,
        "ecc_delta": 0,
    }


def _checks(metrics: Mapping[str, JSONValue], refs: Sequence[str]) -> tuple[Stage0GateCheck, ...]:
    evidence = tuple(refs)
    return (
        Stage0GateCheck(
            "stage0.G6-NCCL",
            Stage0CheckClass.PERFORMANCE,
            Stage0CheckStatus.PASS,
            "Three independent NCCL process groups passed the frozen v2 20/50 median-of-3 protocol at 256 KiB/1 MiB/16 MiB.",
            measurements={
                "process_group_rebuilds": metrics["collective_process_group_rebuilds"],
                "median_cv_by_elements": metrics["collective_median_cv_by_elements"],
            },
            evidence_refs=evidence,
        ),
        Stage0GateCheck(
            "stage0.G6-GRADIENT",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "FP32 global mean, accumulation, no_sync, clipping, and tail semantics match the serial oracle.",
            measurements={
                "atol": 1e-6,
                "rtol": 1e-5,
                "relative_l2_max": 1e-5,
                "comparisons": metrics["fp32_comparisons"],
            },
            evidence_refs=evidence,
        ),
        Stage0GateCheck(
            "stage0.G6-BF16",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "All four ranks completed a finite real-asset BF16 update.",
            measurements={"rank_count": metrics["bf16_rank_count"]},
            evidence_refs=evidence,
        ),
        Stage0GateCheck(
            "stage0.G6-FAILURE",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "Injected rank failure exited within 60 seconds, left no GPU process, and a fresh group restarted.",
            measurements={
                "duration_seconds": metrics["controlled_failure_duration_seconds"],
                "residual_process_count": metrics["residual_process_count"],
                "ecc_delta": metrics["ecc_delta"],
            },
            evidence_refs=evidence,
        ),
    )


def _run_suite(
    request: TaskExecutionRequest,
    root: Path,
    source: G6SourceBinding,
    selected: Sequence[str],
    suite_root_ref: str,
) -> tuple[list[str], list[dict[str, Any]], list[str], list[dict[str, Any]], str, dict[str, Any]]:
    config_ref = f"{suite_root_ref}/resolved-config.json"
    environment_ref = f"{suite_root_ref}/environment.json"
    config_path = _logical_path(root, config_ref, field="suite.config_ref")
    environment_path = _logical_path(root, environment_ref, field="suite.environment_ref")
    _write_or_verify(config_path, request.config.to_dict())
    _write_or_verify(environment_path, request.environment.to_dict())
    config_sha = sha256_file(config_path)
    environment_sha = sha256_file(environment_path)
    before = _gpu_snapshot(selected)
    if any(value for value in _mapping(before["compute_pids"], field="before.pids").values()):
        raise Stage0G6Error("G6_PREFLIGHT_EXTERNAL_COMPUTE_PROCESS")
    specs = [
        *( (f"collective-{index:02d}", "collective", index, 300, 420.0) for index in range(3) ),
        ("semantic-00", "semantic", 0, 180, 1800.0),
        ("failure-rank-00", "failure_rank", 0, 45, 60.0),
        ("recovery-00", "recovery", 0, 180, 240.0),
    ]
    report_refs: list[str] = []
    reports: list[dict[str, Any]] = []
    transcript_refs: list[str] = []
    transcripts: list[dict[str, Any]] = []
    for run_id, run_kind, repeat_index, group_timeout, wall_timeout in specs:
        plan_ref = f"{suite_root_ref}/plans/{run_id}.json"
        report_ref = f"{suite_root_ref}/reports/{run_id}.json"
        transcript_ref = f"{suite_root_ref}/transcripts/{run_id}.json"
        plan = _plan_payload(
            run_id=run_id,
            run_kind=run_kind,
            repeat_index=repeat_index,
            source=source,
            config_ref=config_ref,
            config_sha256=config_sha,
            config_hash=request.config.config_hash,
            environment_ref=environment_ref,
            environment_sha256=environment_sha,
            environment_hash=request.environment.environment_hash,
            selected=selected,
            report_ref=report_ref,
            timeout_seconds=group_timeout,
        )
        _write_or_verify(_logical_path(root, plan_ref, field="plan_ref"), plan)
        transcript = _run_launch(
            root=root,
            source=source,
            plan_ref=plan_ref,
            run_id=run_id,
            selected=selected,
            wall_timeout=wall_timeout,
            transcript_ref=transcript_ref,
        )
        transcript_refs.append(transcript_ref)
        transcripts.append(transcript)
        if run_kind == "failure_rank":
            if transcript.get("return_code") == 0 or not transcript.get("failure_marker_observed"):
                raise Stage0G6Error("G6_EXPECTED_FAILURE_LAUNCH_DID_NOT_FAIL")
            if _logical_path(root, report_ref, field="failure.report_ref").exists():
                raise Stage0G6Error("G6_EXPECTED_FAILURE_PUBLISHED_SUCCESS_REPORT")
            continue
        if transcript.get("return_code") != 0 or transcript.get("timed_out") is not False:
            raise Stage0G6Error(f"G6_CHILD_LAUNCH_FAILED:{run_id}")
        report = _validate_worker_report(
            root,
            report_ref,
            source=source,
            request=request,
            selected=selected,
        )
        report_refs.append(report_ref)
        reports.append(report)
    after = _gpu_snapshot(selected)
    device_audit = _payload_with_hash(
        {
            "schema_version": "stage0-g6-device-audit-v1",
            "selected_gpu_uuids": list(selected),
            "before": before,
            "after": after,
        }
    )
    device_audit_ref = f"{suite_root_ref}/device-audit.json"
    write_canonical_json(
        _logical_path(root, device_audit_ref, field="device_audit_ref"),
        device_audit,
    )
    return report_refs, reports, transcript_refs, transcripts, device_audit_ref, device_audit


def run_formal_g6_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G6Error("G6_FORMAL_REQUEST_INVALID")
    if "stage0.G5" not in request.environment.passed_gate_ids:
        raise Stage0G6Error("G6_REQUIRED_G5_GATE_MISSING")
    selected = _gpu_uuids_from_environment(request.environment, root)
    suite_root_ref = (
        f"evidence/stage0/g6-suite/{request.config.config_hash}/"
        f"{request.environment.environment_hash}"
    )
    (
        report_refs,
        reports,
        transcript_refs,
        transcripts,
        device_audit_ref,
        device_audit,
    ) = _run_suite(request, root, source, selected, suite_root_ref)
    metrics = validate_g6_report_set(reports, transcripts, device_audit)
    checked_at = max(
        _timestamp(report["completed_at"], field="report.completed_at")
        for report in reports
    ).isoformat().replace("+00:00", "Z")
    evidence_refs = tuple(source_refs) + tuple(report_refs) + tuple(transcript_refs) + (
        device_audit_ref,
    )
    checks = _checks(metrics, evidence_refs)
    gate = GateRecord(
        gate_id="stage0.G6",
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "world_size": WORLD_SIZE,
            "backend": "nccl",
            "selected_gpu_uuids": list(selected),
            "process_group_rebuilds": 3,
            "bf16_rank_count": WORLD_SIZE,
            "residual_process_count": 0,
        },
        threshold={
            "fp32_atol": 1e-6,
            "fp32_rtol": 1e-5,
            "fp32_relative_l2_max": 1e-5,
            "collective_p95_vs_median_max_ratio": 2.0,
            "collective_repeat_median_cv_max": 0.20,
            "rank_failure_exit_seconds_max": 60,
            "ecc_delta_max": 0,
            "residual_processes_allowed": 0,
        },
        evidence_refs=evidence_refs,
    )
    environment_id = str(request.config.base_config.section("runtime")["environment_id"])
    gate_report = Stage0GateReport(
        gate_id="stage0.G6",
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        config_hashes={"stage0.07_ddp_and_gradient_semantics": request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(_logical_path(root, reference, field="evidence_ref")),
                schema_version=(
                    WORKER_REPORT_SCHEMA
                    if reference in report_refs
                    else "stage0-g6-launch-transcript-v1"
                    if reference in transcript_refs
                    else "stage0-g6-device-audit-v1"
                    if reference == device_audit_ref
                    else "task-output-commit-v1"
                ),
            )
            for reference in evidence_refs
        ),
        checks=checks,
    )
    validation = _payload_with_hash(
        {
            "schema_version": "stage0-g6-validation-report-v1",
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "metrics": metrics,
        }
    )
    refs: dict[str, str] = {}
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g6-evidence-v1",
            "artifact_role": kind,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "selected_gpu_uuids": list(selected),
            "suite_root_ref": suite_root_ref,
            "worker_report_refs": report_refs,
            "launch_transcript_refs": transcript_refs,
            "device_audit_ref": device_audit_ref,
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
        message="Stage 0 G6 four-GPU DDP/NCCL suite passed",
        metadata={"stage0_g6_specialized": True, "gate_id": "stage0.G6"},
    )


def _gpu_uuids_from_environment(
    environment: TaskRuntimeEnvironment, root: Path
) -> tuple[str, ...]:
    reference = environment.evidence_refs.get("g4_provenance")
    if not isinstance(reference, str):
        raise Stage0G6Error("G6_G4_PROVENANCE_REF_MISSING")
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    envelope = _mapping(loaded.payload, field="g4_provenance_envelope")
    provenance = _mapping(envelope.get("provenance_record"), field="g4_provenance")
    mapping = provenance.get("device_mapping")
    if (
        not isinstance(mapping, list)
        or len(mapping) != WORLD_SIZE
        or len(set(mapping)) != WORLD_SIZE
        or any(not isinstance(value, str) or not value.startswith("GPU-") for value in mapping)
    ):
        raise Stage0G6Error("G6_G4_DEVICE_MAPPING_INVALID")
    return tuple(mapping)


def validate_formal_g6_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G6Error("G6_OUTPUT_SET_INVALID")
    selected = _gpu_uuids_from_environment(request.environment, root)
    envelopes: list[dict[str, Any]] = []
    for kind, reference in outputs.items():
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != request.config.config_hash
        ):
            raise Stage0G6Error("G6_OUTPUT_COMMIT_IDENTITY_INVALID")
        envelope = _mapping(loaded.payload, field=f"output.{kind}")
        declared = envelope.pop("artifact_hash", None)
        if (
            envelope.get("schema_version") != "stage0-g6-evidence-v1"
            or envelope.get("artifact_role") != kind
            or envelope.get("status") != "PASS"
            or envelope.get("generator_git_commit") != source.git_commit
            or envelope.get("environment_hash") != request.environment.environment_hash
            or envelope.get("selected_gpu_uuids") != list(selected)
            or declared != canonical_json_hash(envelope)
        ):
            raise Stage0G6Error("G6_OUTPUT_ENVELOPE_INVALID")
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
        raise Stage0G6Error("G6_OUTPUT_ROLE_PAYLOAD_DRIFT")
    report_refs = canonical.get("worker_report_refs")
    transcript_refs = canonical.get("launch_transcript_refs")
    device_audit_ref = canonical.get("device_audit_ref")
    if (
        not isinstance(report_refs, list)
        or len(report_refs) != 5
        or not isinstance(transcript_refs, list)
        or len(transcript_refs) != 6
        or not isinstance(device_audit_ref, str)
    ):
        raise Stage0G6Error("G6_OUTPUT_EVIDENCE_REFS_INVALID")
    reports = [
        _validate_worker_report(
            root,
            str(reference),
            source=source,
            request=request,
            selected=selected,
        )
        for reference in report_refs
    ]
    transcripts = [
        _mapping(
            load_canonical_json(_logical_path(root, reference, field="transcript_ref")),
            field="transcript",
        )
        for reference in transcript_refs
    ]
    for transcript in transcripts:
        declared = transcript.pop("artifact_hash", None)
        if (
            transcript.get("schema_version") != "stage0-g6-launch-transcript-v1"
            or declared != canonical_json_hash(transcript)
        ):
            raise Stage0G6Error("G6_TRANSCRIPT_INVALID")
        transcript["artifact_hash"] = declared
    device_audit = _mapping(
        load_canonical_json(_logical_path(root, device_audit_ref, field="device_audit_ref")),
        field="device_audit",
    )
    declared_device = device_audit.pop("artifact_hash", None)
    if (
        device_audit.get("schema_version") != "stage0-g6-device-audit-v1"
        or declared_device != canonical_json_hash(device_audit)
    ):
        raise Stage0G6Error("G6_DEVICE_AUDIT_INVALID")
    device_audit["artifact_hash"] = declared_device
    replayed = validate_g6_report_set(reports, transcripts, device_audit)
    validation = _mapping(canonical.get("validation_report"), field="validation")
    declared_validation = validation.pop("artifact_hash", None)
    if (
        declared_validation != canonical_json_hash(validation)
        or validation.get("status") != "PASS"
        or validation.get("metrics") != replayed
    ):
        raise Stage0G6Error("G6_OUTPUT_VALIDATION_REPLAY_MISMATCH")
    gate = GateRecord.from_mapping(
        _mapping(canonical.get("gate_record"), field="gate_record")
    )
    gate_report = Stage0GateReport.from_mapping(
        _mapping(canonical.get("gate_report"), field="gate_report")
    )
    if (
        gate.gate_id != "stage0.G6"
        or gate.status is not GateStatus.PASS
        or gate_report.gate_id != "stage0.G6"
        or gate_report.status.value != "PASS"
    ):
        raise Stage0G6Error("G6_OUTPUT_GATE_INVALID")
    return gate


def execute_stage0_g6(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g5_index_ref: str,
) -> Stage0G6FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g5_formal_state(
        data_root=root,
        index_ref=g5_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g6_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g6-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G6Error(
            f"G6_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, state.environment)
    gate = validate_formal_g6_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g6_distributed_validation": outputs["distributed_validation"],
            "g6_gradient_semantics": outputs["gradient_semantics_report"],
            "g6_communication": outputs["communication_report"],
            "gate_stage0_g6": outputs["distributed_validation"],
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
        "schema_version": "stage0-g6-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g5_index_ref": state.index_ref,
        "g5_index_sha256": state.index_sha256,
        "g5_gate_artifact_hash": state.gate_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_ref": outputs["distributed_validation"],
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.08_logging_and_tracking",
        "next_input_refs": list(outputs.values()),
    }
    index_payload["artifact_hash"] = canonical_json_hash(index_payload)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(_logical_path(root, index_ref, field="index_ref"), index_payload)
    return Stage0G6FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def load_stage0_g6_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G6FormalState:
    """Load G6 after replaying G5 and all five successful G6 launch reports."""

    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g6_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g5_index_ref",
        "g5_index_sha256",
        "g5_gate_artifact_hash",
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
        "stage0-g6-formalization-index-v1"
    ):
        raise Stage0G6Error("G6_STATE_INDEX_FIELDS_OR_VERSION_INVALID")
    declared = raw.pop("artifact_hash")
    if declared != canonical_json_hash(raw):
        raise Stage0G6Error("G6_STATE_INDEX_HASH_MISMATCH")
    raw["artifact_hash"] = declared
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G6Error("G6_STATE_GENERATOR_COMMIT_MISMATCH")
    g5_state = load_stage0_g5_formal_state(
        data_root=root,
        index_ref=str(raw["g5_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    if (
        raw.get("g5_index_sha256") != g5_state.index_sha256
        or raw.get("g5_gate_artifact_hash") != g5_state.gate_artifact_hash
    ):
        raise Stage0G6Error("G6_STATE_G5_BINDING_MISMATCH")
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, raw["config_ref"], field="config_ref")),
            field="config",
        )
    )
    if config.task_id != TASK_ID or config.config_hash != raw.get("config_hash"):
        raise Stage0G6Error("G6_STATE_CONFIG_MISMATCH")
    outputs = _mapping(raw["task_output_refs"], field="task_output_refs")
    if set(outputs) != _OUTPUT_KINDS or any(
        not isinstance(value, str) for value in outputs.values()
    ):
        raise Stage0G6Error("G6_STATE_OUTPUT_SET_INVALID")
    ordered_outputs = {
        key: str(outputs[key])
        for key in (
            "distributed_validation",
            "gradient_semantics_report",
            "communication_report",
        )
    }
    request = TaskExecutionRequest(config, config.task_definition, g5_state.environment)
    gate = validate_formal_g6_outputs(request, root, ordered_outputs)
    if raw.get("gate_ref") != outputs["distributed_validation"]:
        raise Stage0G6Error("G6_STATE_GATE_REF_MISMATCH")
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, raw["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    next_inputs = raw.get("next_input_refs")
    if (
        environment.environment_hash != raw.get("environment_hash")
        or gate.gate_id not in environment.passed_gate_ids
        or environment.evidence_refs.get("gate_stage0_g6") != raw.get("gate_ref")
        or raw.get("next_task_id") != "stage0.08_logging_and_tracking"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(outputs.values())
    ):
        raise Stage0G6Error("G6_STATE_ENVIRONMENT_OR_HANDOFF_MISMATCH")
    loaded_gate = load_committed_task_artifact(
        root, str(raw["gate_ref"]), require_formal=True
    )
    gate_hash = loaded_gate.payload.get("artifact_hash")
    if not isinstance(gate_hash, str) or _SHA256_RE.fullmatch(gate_hash) is None:
        raise Stage0G6Error("G6_STATE_GATE_ARTIFACT_HASH_INVALID")
    return Stage0G6FormalState(
        environment=environment,
        task_output_refs=ordered_outputs,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate_hash,
        g5_index_ref=str(raw["g5_index_ref"]),
    )


__all__ = [
    "G6SourceBinding",
    "Stage0G6Error",
    "Stage0G6FormalState",
    "Stage0G6FormalizationResult",
    "TASK_ID",
    "build_stage0_g6_config",
    "execute_stage0_g6",
    "load_stage0_g6_formal_state",
    "run_formal_g6_task",
    "validate_formal_g6_outputs",
    "validate_g6_report_set",
]
