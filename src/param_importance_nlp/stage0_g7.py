"""Formal Stage 0 S0.8 observability component of the G7 recovery gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

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
from .runtime import (
    EventRecord,
    EventType,
    JsonlEventSink,
    RunStatus,
    RunStatusStore,
    SessionStatus,
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    build_canonical_event_lineage,
    canonical_optimizer_steps,
    load_committed_task_artifact,
    read_event_stream,
    rebuild_tensorboard_from_jsonl,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g6 import Stage0G6FormalState, load_stage0_g6_formal_state
from .stage0_g7_worker import WORKER_PLAN_SCHEMA, WORKER_REPORT_SCHEMA
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID = "stage0.08_logging_and_tracking"
COMPONENT_GATE_ID = "stage0.G7-LOGGING"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OUTPUT_KINDS = {"event_stream", "lineage_manifest", "logging_overhead_report"}
_CRITICAL_SOURCE_REFS = (
    "ops/stage0/formalize_g7.py",
    "ops/stage0/run_g7_worker.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/runtime/events.py",
    "src/param_importance_nlp/runtime/event_lineage.py",
    "src/param_importance_nlp/runtime/run_status.py",
    "src/param_importance_nlp/runtime/telemetry.py",
    "src/param_importance_nlp/runtime/training.py",
    "src/param_importance_nlp/stage0_g6.py",
    "src/param_importance_nlp/stage0_g7.py",
    "src/param_importance_nlp/stage0_g7_worker.py",
    "schemas/runtime-canonical-event-lineage-v1.json",
    "schemas/runtime-event-v1.json",
    "schemas/runtime-run-status-v1.json",
    "schemas/stage0-g7-evidence-v1.json",
    "schemas/stage0-g7-formalization-index-v1.json",
    "schemas/stage0-g7-worker-plan-v1.json",
    "schemas/stage0-g7-worker-report-v1.json",
)


class Stage0G7Error(RuntimeError):
    """S0.8 observability evidence failed closed."""


@dataclass(frozen=True, slots=True)
class G7SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G7FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G7FormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    component_artifact_hash: str
    g6_index_ref: str


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


def _capture_source() -> G7SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    probes = {
        "top": _git(repository, "rev-parse", "--show-toplevel"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "tracked": _git(
            repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS
        ),
        "status": _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if any(item.returncode != 0 for item in probes.values()):
        raise Stage0G7Error("G7_SOURCE_GIT_PROBE_FAILED")
    if Path(probes["top"].stdout.strip()).resolve() != repository:
        raise Stage0G7Error("G7_SOURCE_GIT_ROOT_MISMATCH")
    commit = probes["head"].stdout.strip()
    branch = probes["branch"].stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch:
        raise Stage0G7Error("G7_SOURCE_GIT_IDENTITY_INVALID")
    if probes["status"].stdout.strip():
        raise Stage0G7Error("G7_FORMAL_SOURCE_DIRTY")
    return G7SourceBinding(repository, commit, branch)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G7Error(f"G7_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G7Error(f"G7_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G7Error(f"G7_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G7Error(f"G7_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise Stage0G7Error(f"G7_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G7Error(f"G7_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0G7Error(f"G7_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G7Error("G7_CONTROL_FILE_DRIFT")
    else:
        write_canonical_json(path, dict(value))


def build_stage0_g7_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G6FormalState,
) -> ResolvedConfigV2:
    Path(data_root).resolve(strict=True)
    previous = state.config
    base = previous.base_config
    runtime = _mapping(base.section("runtime"), field="runtime")
    output_dir = f"evidence/stage0/tasks/08-{state.gate_artifact_hash}"
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=output_dir,
        base_overrides={
            "identity": {"route": f"stage0-g7-logging-{state.gate_artifact_hash[:12]}"},
            "runtime": {**runtime, "device": "cpu", "allow_dirty_worktree": False},
            "model": base.section("model"),
            "data": base.section("data"),
            "loss": base.section("loss"),
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
                **_mapping(base.section("precision"), field="precision"),
                "compute_dtype": "float32",
                "amp": False,
            },
            "logging": {
                **_mapping(base.section("logging"), field="logging"),
                "event_format": "jsonl-v1",
                "tensorboard": True,
                "log_every_steps": 1,
            },
        },
        v2_overrides={
            "execution": {"timeout_seconds": 3600, "max_attempts": 1},
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


def _event(
    *,
    run_id: str,
    attempt: str,
    session: str,
    rank: int,
    sequence: int,
    event_type: EventType,
    payload: dict[str, Any],
    suffix: str,
) -> EventRecord:
    return EventRecord.create(
        experiment_id="stage0-g7-functional",
        run_id=run_id,
        attempt_id=attempt,
        session_id=session,
        rank=rank,
        event_type=event_type,
        sequence=sequence,
        event_id=f"{session}-{rank}-{suffix}",
        occurred_at=f"2026-08-03T01:{rank:02d}:{sequence:02d}Z",
        payload=payload,
    )


def _functional_suite(root: Path, suite_ref: str) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    functional_root = _logical_path(root, suite_ref, field="functional_root")
    if functional_root.exists():
        raise Stage0G7Error("G7_FUNCTIONAL_ROOT_COLLISION")
    functional_root.mkdir(parents=True)
    run_id = "g7-functional-run"
    rank_event_refs: list[str] = []
    console_refs: list[str] = []
    for rank in range(4):
        session = f"rank-session-{rank}"
        event_path = functional_root / f"rank-{rank:04d}" / "events.jsonl"
        with JsonlEventSink(event_path) as sink:
            sink.append(
                _event(
                    run_id=run_id,
                    attempt="attempt-ranks",
                    session=session,
                    rank=rank,
                    sequence=0,
                    event_type=EventType.SYSTEM,
                    payload={
                        "global_step": 0,
                        "gpu_memory_bytes": 1000 + rank,
                        "gpu_utilization_percent": 50 + rank,
                        "cpu_memory_bytes": 2000 + rank,
                        "disk_free_bytes": 3000 + rank,
                        "throughput_units_per_second": 10.0 + rank,
                    },
                    suffix="system",
                ),
                critical=True,
            )
        console_path = functional_root / f"rank-{rank:04d}" / "console.log"
        atomic_write_bytes(
            console_path,
            f"rank={rank} session={session} status=SUCCEEDED\n".encode("utf-8"),
        )
        rank_event_refs.append(event_path.relative_to(root).as_posix())
        console_refs.append(console_path.relative_to(root).as_posix())
    shared_path = functional_root / "global" / "events.jsonl"
    with JsonlEventSink(shared_path) as shared:
        for step in range(2):
            shared.append(
                _event(
                    run_id=run_id,
                    attempt="attempt-ranks",
                    session="global-session",
                    rank=0,
                    sequence=step,
                    event_type=EventType.OPTIMIZER_STEP,
                    payload={
                        "global_step": step,
                        "microstep_count": 2,
                        "sample_count": 8,
                        "effective_token_count": 8192,
                        "mean_loss": 2.0 - step,
                        "learning_rate": 0.001,
                        "learning_rates_post_step": {"group_0000": 0.001},
                        "global_gradient_norm": 0.5,
                    },
                    suffix=f"global-{step}",
                ),
                critical=True,
            )
        concurrent_writer_rejected = False
        try:
            JsonlEventSink(shared_path)
        except RuntimeError:
            concurrent_writer_rejected = True

    sensitive_rejections = 0
    for value in (
        "Bearer abcdefghijklmnopqrstuvwxyz",
        "https://example.invalid/file?token=secretvalue",
        "password=not-for-logs",
        "-----BEGIN PRIVATE KEY-----",
    ):
        try:
            EventRecord.create(
                experiment_id="e",
                run_id="r",
                attempt_id="a",
                session_id="s",
                rank=0,
                event_type=EventType.ERROR,
                sequence=0,
                payload={
                    "exception_class": "SensitiveFixtureError",
                    "last_valid_step": 0,
                    "affected_ranks": [0],
                    "recoverable": False,
                    "message": value,
                },
            )
        except ValueError:
            sensitive_rejections += 1
    non_json_rejected = False
    try:
        EventRecord.create(
            experiment_id="e",
            run_id="r",
            attempt_id="a",
            session_id="s",
            rank=0,
            event_type=EventType.SYSTEM,
            sequence=0,
            payload={"arbitrary": object()},
        )
    except ValueError:
        non_json_rejected = True
    volume_guard_rejected = False
    tiny_path = functional_root / "failure" / "tiny.jsonl"
    try:
        with JsonlEventSink(tiny_path, max_session_bytes=1) as sink:
            sink.append(
                _event(
                    run_id=run_id,
                    attempt="failure-attempt",
                    session="failure-session",
                    rank=0,
                    sequence=0,
                    event_type=EventType.ERROR,
                    payload={
                        "exception_class": "OSError",
                        "last_valid_step": 0,
                        "affected_ranks": [0],
                        "recoverable": False,
                    },
                    suffix="write-failure",
                )
            )
    except OSError:
        volume_guard_rejected = True

    failure_status = RunStatusStore(
        functional_root / "failure" / "status.json",
        run_id="truth-write-failure",
        created_at="2026-08-03T02:00:00Z",
    )
    failure_status.transition_run(
        RunStatus.RUNNING,
        actor="fixture",
        reason="truth failure injection",
        at="2026-08-03T02:00:01Z",
    )
    failure_status.transition_run(
        RunStatus.FAILED_FINAL,
        actor="fixture",
        reason="JSONL truth unavailable",
        at="2026-08-03T02:00:02Z",
    )

    derivative_truth = functional_root / "tensorboard-failure" / "truth.jsonl"
    with JsonlEventSink(derivative_truth) as sink:
        sink.append(
            _event(
                run_id=run_id,
                attempt="tb-attempt",
                session="tb-session",
                rank=0,
                sequence=0,
                event_type=EventType.OPTIMIZER_STEP,
                payload={
                    "global_step": 0,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 1.0,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                suffix="tb-truth",
            ),
            critical=True,
        )
    truth_before = sha256_file(derivative_truth)
    invalid_tb_root = functional_root / "tensorboard-failure" / "not-a-directory"
    atomic_write_bytes(invalid_tb_root, b"occupied\n")
    tensorboard_failure_warned = False
    try:
        rebuild_tensorboard_from_jsonl((derivative_truth,), invalid_tb_root)
    except Exception:
        tensorboard_failure_warned = True
    truth_after = sha256_file(derivative_truth)

    raw_root = functional_root / "resume"
    first_path = raw_root / "session-0001.jsonl"
    second_path = raw_root / "session-0002.jsonl"
    with JsonlEventSink(first_path) as sink:
        values = (
            (EventType.RUN_LIFECYCLE, {"status": "RUNNING"}, "start"),
            (
                EventType.OPTIMIZER_STEP,
                {
                    "global_step": 0,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 3.0,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                "s0",
            ),
            (
                EventType.OPTIMIZER_STEP,
                {
                    "global_step": 1,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 2.0,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                "s1",
            ),
            (
                EventType.CHECKPOINT,
                {
                    "checkpoint_id": "checkpoint-k",
                    "global_step": 1,
                    "checkpoint_ref": "checkpoints/checkpoint-k.json",
                    "status": "COMMITTED",
                    "manifest_sha256": "a" * 64,
                    "parent_checkpoint_id": None,
                },
                "checkpoint",
            ),
            (
                EventType.OPTIMIZER_STEP,
                {
                    "global_step": 2,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 100.0,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                "orphan",
            ),
        )
        for sequence, (event_type, payload, suffix) in enumerate(values):
            sink.append(
                _event(
                    run_id=run_id,
                    attempt="attempt-0001",
                    session="session-0001",
                    rank=0,
                    sequence=sequence,
                    event_type=event_type,
                    payload=payload,
                    suffix=suffix,
                ),
                critical=True,
            )
    with JsonlEventSink(second_path) as sink:
        values = (
            (EventType.RUN_LIFECYCLE, {"status": "RUNNING"}, "resume"),
            (
                EventType.OPTIMIZER_STEP,
                {
                    "global_step": 2,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 1.5,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                "s2",
            ),
            (
                EventType.SYSTEM,
                {
                    "global_step": 2,
                    "gpu_memory_bytes": 0,
                    "gpu_utilization_percent": 0.0,
                    "cpu_memory_bytes": 1234,
                    "disk_free_bytes": 5678,
                    "throughput_units_per_second": 10.0,
                },
                "system",
            ),
            (
                EventType.OPTIMIZER_STEP,
                {
                    "global_step": 3,
                    "microstep_count": 1,
                    "sample_count": 1,
                    "effective_token_count": 128,
                    "mean_loss": 1.0,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": {"group_0000": 0.001},
                },
                "s3",
            ),
            (EventType.RUN_LIFECYCLE, {"status": "SUCCESS"}, "success"),
        )
        for sequence, (event_type, payload, suffix) in enumerate(values):
            sink.append(
                _event(
                    run_id=run_id,
                    attempt="attempt-0002",
                    session="session-0002",
                    rank=0,
                    sequence=sequence,
                    event_type=event_type,
                    payload=payload,
                    suffix=suffix,
                ),
                critical=True,
            )
    first_ref = first_path.relative_to(root).as_posix()
    second_ref = second_path.relative_to(root).as_posix()
    first_hash = sha256_file(first_path)
    second_hash = sha256_file(second_path)
    lineage_ref = (raw_root / "lineage.json").relative_to(root).as_posix()
    canonical_ref = (raw_root / "canonical.jsonl").relative_to(root).as_posix()
    lineage = build_canonical_event_lineage(
        root=root,
        run_id=run_id,
        segments=(
            {
                "attempt_id": "attempt-0001",
                "session_id": "session-0001",
                "rank": 0,
                "event_ref": first_ref,
                "event_sha256": first_hash,
                "sequence_start": 0,
                "sequence_end": 3,
                "checkpoint_ref": "checkpoints/checkpoint-k.json",
            },
            {
                "attempt_id": "attempt-0002",
                "session_id": "session-0002",
                "rank": 0,
                "event_ref": second_ref,
                "event_sha256": second_hash,
                "sequence_start": 0,
                "sequence_end": 4,
                "checkpoint_ref": None,
            },
        ),
        superseded_tails=(
            {
                "attempt_id": "attempt-0001",
                "session_id": "session-0001",
                "event_ref": first_ref,
                "event_sha256": first_hash,
                "sequence_start": 4,
                "sequence_end": 4,
                "reason": "superseded_after_checkpoint_k",
            },
        ),
        parent_checkpoint_ref="checkpoints/checkpoint-k.json",
        output_ref=lineage_ref,
        canonical_event_ref=canonical_ref,
    )
    raw_unchanged = first_hash == sha256_file(first_path) and second_hash == sha256_file(second_path)
    canonical_events = read_event_stream(_logical_path(root, canonical_ref, field="canonical"))
    canonical_steps = [
        int(event.payload["global_step"])
        for event in canonical_optimizer_steps((canonical_events,))
    ]
    canonical_tb = raw_root / "tensorboard" / "canonical"
    canonical_tb_scalars = rebuild_tensorboard_from_jsonl(
        (_logical_path(root, canonical_ref, field="canonical"),), canonical_tb
    )

    status_store = RunStatusStore(
        raw_root / "run-status.json",
        run_id=run_id,
        created_at="2026-08-03T03:00:00Z",
    )
    status_store.transition_run(
        RunStatus.RUNNING, actor="runtime", reason="initial", at="2026-08-03T03:00:01Z"
    )
    status_store.register_attempt(
        attempt_id="attempt-0001", at="2026-08-03T03:00:02Z"
    )
    status_store.transition_attempt(
        "attempt-0001",
        SessionStatus.RUNNING,
        actor="runtime",
        reason="attempt started",
        at="2026-08-03T03:00:03Z",
    )
    status_store.register_session(
        attempt_id="attempt-0001", session_id="session-0001", at="2026-08-03T03:00:04Z"
    )
    status_store.transition_session(
        "session-0001", SessionStatus.RUNNING, actor="runtime", reason="started", at="2026-08-03T03:00:05Z"
    )
    status_store.heartbeat(
        "session-0001", last_step=1, observed_at="2026-08-03T03:00:06Z"
    )
    status_store.transition_session(
        "session-0001", SessionStatus.FAILED, actor="runtime", reason="injected", at="2026-08-03T03:00:07Z"
    )
    status_store.transition_attempt(
        "attempt-0001",
        SessionStatus.FAILED,
        actor="runtime",
        reason="session failed",
        at="2026-08-03T03:00:08Z",
    )
    status_store.transition_run(
        RunStatus.RESUMABLE,
        actor="runtime",
        reason="checkpoint complete",
        at="2026-08-03T03:00:09Z",
        checkpoint_ref="checkpoints/checkpoint-k.json",
        checkpoint_complete=True,
    )
    status_store.transition_run(
        RunStatus.RUNNING,
        actor="runtime",
        reason="resume",
        at="2026-08-03T03:00:10Z",
        checkpoint_ref="checkpoints/checkpoint-k.json",
        checkpoint_complete=True,
    )
    status_store.register_attempt(
        attempt_id="attempt-0002", at="2026-08-03T03:00:11Z"
    )
    status_store.transition_attempt(
        "attempt-0002",
        SessionStatus.RUNNING,
        actor="runtime",
        reason="resume attempt",
        at="2026-08-03T03:00:12Z",
    )
    status_store.register_session(
        attempt_id="attempt-0002", session_id="session-0002", at="2026-08-03T03:00:13Z"
    )
    status_store.transition_session(
        "session-0002", SessionStatus.RUNNING, actor="runtime", reason="resumed", at="2026-08-03T03:00:14Z"
    )
    status_store.heartbeat(
        "session-0002", last_step=3, observed_at="2026-08-03T03:00:15Z"
    )
    status_store.transition_session(
        "session-0002", SessionStatus.SUCCEEDED, actor="runtime", reason="complete", at="2026-08-03T03:00:16Z"
    )
    status_store.transition_attempt(
        "attempt-0002",
        SessionStatus.SUCCEEDED,
        actor="runtime",
        reason="complete",
        at="2026-08-03T03:00:17Z",
    )
    status_store.transition_run(
        RunStatus.SUCCESS, actor="runtime", reason="complete", at="2026-08-03T03:00:18Z"
    )
    forbidden_terminal_transition = False
    try:
        status_store.transition_run(
            RunStatus.RUNNING, actor="runtime", reason="forbidden", at="2026-08-03T03:00:19Z"
        )
    except ValueError:
        forbidden_terminal_transition = True
    status_snapshot = status_store.snapshot()
    evidence_refs = tuple(
        rank_event_refs
        + console_refs
        + [
            shared_path.relative_to(root).as_posix(),
            derivative_truth.relative_to(root).as_posix(),
            first_ref,
            second_ref,
            lineage_ref,
            canonical_ref,
            (raw_root / "run-status.json").relative_to(root).as_posix(),
            (functional_root / "failure" / "status.json").relative_to(root).as_posix(),
        ]
        + [
            path.relative_to(root).as_posix()
            for path in sorted(canonical_tb.rglob("*"))
            if path.is_file()
        ]
    )
    report = _with_hash(
        {
            "schema_version": "stage0-g7-functional-report-v1",
            "rank_event_refs": rank_event_refs,
            "rank_console_refs": console_refs,
            "shared_metric_ref": shared_path.relative_to(root).as_posix(),
            "shared_metric_writer_rank": 0,
            "concurrent_writer_rejected": concurrent_writer_rejected,
            "sensitive_pattern_rejections": sensitive_rejections,
            "non_json_payload_rejected": non_json_rejected,
            "volume_guard_rejected": volume_guard_rejected,
            "truth_write_failure_run_status": failure_status.snapshot()["run_status"],
            "tensorboard_failure_warned": tensorboard_failure_warned,
            "tensorboard_failure_truth_unchanged": truth_before == truth_after,
            "raw_streams_unchanged": raw_unchanged,
            "canonical_lineage": lineage,
            "canonical_optimizer_steps": canonical_steps,
            "canonical_typed_event_count": len(canonical_events),
            "canonical_tensorboard_scalars": canonical_tb_scalars,
            "run_status": status_snapshot,
            "transition_matrix": status_store.transition_matrix,
            "terminal_transition_rejected": forbidden_terminal_transition,
            "evidence_refs": list(evidence_refs),
        }
    )
    report_ref = f"{suite_ref}/functional-report.json"
    write_canonical_json(_logical_path(root, report_ref, field="functional_report"), report)
    return report_ref, report, evidence_refs


def _worker_plan(
    *,
    run_id: str,
    mode: str,
    repeat: int,
    source: G7SourceBinding,
    request: TaskExecutionRequest,
    config_ref: str,
    config_sha: str,
    environment_ref: str,
    environment_sha: str,
    output_root_ref: str,
    report_ref: str,
) -> dict[str, JSONValue]:
    return _with_hash(
        {
            "schema_version": WORKER_PLAN_SCHEMA,
            "run_id": run_id,
            "mode": mode,
            "repeat_index": repeat,
            "generator_git_commit": source.git_commit,
            "config_ref": config_ref,
            "config_sha256": config_sha,
            "config_hash": request.config.config_hash,
            "environment_ref": environment_ref,
            "environment_sha256": environment_sha,
            "environment_hash": request.environment.environment_hash,
            "output_root_ref": output_root_ref,
            "report_ref": report_ref,
            "warmup_steps": 10,
            "measured_steps": 30,
            "matrix_size": 1536,
            "seed": 20260803 + repeat,
        }
    )


def _validate_worker_report(
    root: Path,
    reference: str,
    *,
    source: G7SourceBinding,
    request: TaskExecutionRequest,
) -> dict[str, Any]:
    report = _mapping(
        load_canonical_json(_logical_path(root, reference, field="worker_report")),
        field="worker_report",
    )
    expected = {
        "schema_version",
        "run_id",
        "mode",
        "repeat_index",
        "status",
        "completed_at",
        "generator_git_commit",
        "plan_ref",
        "plan_sha256",
        "config_hash",
        "environment_hash",
        "warmup_steps",
        "measured_steps",
        "matrix_size",
        "measured_wall_seconds",
        "steps_per_second",
        "step_median_seconds",
        "step_p95_seconds",
        "workload_checksum",
        "event_stream_ref",
        "event_stream_sha256",
        "event_count",
        "optimizer_steps",
        "event_append_median_seconds",
        "event_append_p95_seconds",
        "critical_flush_max_seconds",
        "derived_tracking_seconds",
        "tensorboard_scalar_count",
        "tensorboard_refs",
        "summary_ref",
        "status_ref",
        "resource_snapshots",
        "competition_free",
        "artifact_hash",
    }
    if set(report) != expected or report.get("schema_version") != WORKER_REPORT_SCHEMA:
        raise Stage0G7Error("G7_WORKER_REPORT_FIELDS_OR_VERSION_INVALID")
    declared = report.pop("artifact_hash")
    if declared != canonical_json_hash(report):
        raise Stage0G7Error("G7_WORKER_REPORT_HASH_MISMATCH")
    report["artifact_hash"] = declared
    if (
        report.get("status") != "PASS"
        or report.get("generator_git_commit") != source.git_commit
        or report.get("config_hash") != request.config.config_hash
        or report.get("environment_hash") != request.environment.environment_hash
        or report.get("warmup_steps") != 10
        or report.get("measured_steps") != 30
        or report.get("event_count") != 42
        or report.get("optimizer_steps") != list(range(40))
        or report.get("plan_sha256")
        != sha256_file(_logical_path(root, report["plan_ref"], field="plan_ref"))
        or report.get("event_stream_sha256")
        != sha256_file(
            _logical_path(root, report["event_stream_ref"], field="event_stream_ref")
        )
        or float(report.get("steps_per_second", 0.0)) <= 0
        or float(report.get("step_median_seconds", 0.0)) <= 0
        or float(report.get("step_p95_seconds", 0.0)) <= 0
        or float(report["step_p95_seconds"])
        < float(report["step_median_seconds"])
        or float(report.get("event_append_median_seconds", -1.0)) < 0
        or float(report.get("event_append_p95_seconds", -1.0)) < 0
        or float(report.get("critical_flush_max_seconds", -1.0)) < 0
        or report.get("competition_free") is not True
    ):
        raise Stage0G7Error("G7_WORKER_REPORT_INVARIANT_FAILED")
    snapshots = report.get("resource_snapshots")
    if (
        not isinstance(snapshots, Mapping)
        or set(snapshots) != {"before", "after"}
        or any(not isinstance(snapshots[item], Mapping) for item in snapshots)
    ):
        raise Stage0G7Error("G7_RESOURCE_SNAPSHOTS_INVALID")
    _timestamp(report["completed_at"], field="worker.completed_at")
    mode = report.get("mode")
    if mode == "formal":
        refs = report.get("tensorboard_refs")
        if (
            not isinstance(refs, list)
            or not refs
            or int(report.get("tensorboard_scalar_count", 0)) <= 0
            or not isinstance(report.get("summary_ref"), str)
            or not isinstance(report.get("status_ref"), str)
        ):
            raise Stage0G7Error("G7_FORMAL_DERIVED_OUTPUTS_MISSING")
        for item in (*refs, report["summary_ref"], report["status_ref"]):
            if not _logical_path(root, item, field="derived_ref").is_file():
                raise Stage0G7Error("G7_FORMAL_DERIVED_FILE_MISSING")
    elif mode == "minimal":
        if (
            report.get("tensorboard_refs") != []
            or report.get("tensorboard_scalar_count") != 0
            or report.get("summary_ref") is not None
            or report.get("status_ref") is not None
        ):
            raise Stage0G7Error("G7_MINIMAL_MODE_DERIVED_OUTPUT_PRESENT")
    else:
        raise Stage0G7Error("G7_WORKER_REPORT_MODE_INVALID")
    return report


def validate_g7_report_set(
    reports: Sequence[Mapping[str, Any]], functional: Mapping[str, Any]
) -> dict[str, JSONValue]:
    if len(reports) != 6:
        raise Stage0G7Error("G7_OVERHEAD_REPORT_COUNT_INVALID")
    pairs: list[dict[str, JSONValue]] = []
    for repeat in range(3):
        minimal = [
            item
            for item in reports
            if item.get("repeat_index") == repeat and item.get("mode") == "minimal"
        ]
        formal = [
            item
            for item in reports
            if item.get("repeat_index") == repeat and item.get("mode") == "formal"
        ]
        if len(minimal) != 1 or len(formal) != 1:
            raise Stage0G7Error("G7_OVERHEAD_PAIR_MISSING")
        if float(minimal[0]["workload_checksum"]) != float(formal[0]["workload_checksum"]):
            raise Stage0G7Error("G7_OVERHEAD_WORKLOAD_DRIFT")
        minimal_rate = float(minimal[0]["steps_per_second"])
        formal_rate = float(formal[0]["steps_per_second"])
        overhead = max(0.0, 1.0 - formal_rate / minimal_rate)
        pairs.append(
            {
                "repeat_index": repeat,
                "minimal_steps_per_second": minimal_rate,
                "formal_steps_per_second": formal_rate,
                "throughput_overhead_fraction": overhead,
                "formal_event_append_p95_seconds": formal[0][
                    "event_append_p95_seconds"
                ],
                "formal_critical_flush_max_seconds": formal[0][
                    "critical_flush_max_seconds"
                ],
                "formal_derived_tracking_seconds": formal[0][
                    "derived_tracking_seconds"
                ],
                "minimal_step_median_seconds": minimal[0]["step_median_seconds"],
                "minimal_step_p95_seconds": minimal[0]["step_p95_seconds"],
                "formal_step_median_seconds": formal[0]["step_median_seconds"],
                "formal_step_p95_seconds": formal[0]["step_p95_seconds"],
            }
        )
    minimal_rates = [float(item["minimal_steps_per_second"]) for item in pairs]
    formal_rates = [float(item["formal_steps_per_second"]) for item in pairs]
    minimal_median = statistics.median(minimal_rates)
    formal_median = statistics.median(formal_rates)
    group_overhead = max(0.0, 1.0 - formal_median / minimal_median)
    if group_overhead > 0.10:
        raise Stage0G7Error("G7_TRACKING_OVERHEAD_EXCEEDED")

    def coefficient_of_variation(values: list[float]) -> float:
        mean = statistics.mean(values)
        return 0.0 if len(values) < 2 or mean == 0.0 else statistics.stdev(values) / mean
    required_true = (
        "concurrent_writer_rejected",
        "non_json_payload_rejected",
        "volume_guard_rejected",
        "tensorboard_failure_warned",
        "tensorboard_failure_truth_unchanged",
        "raw_streams_unchanged",
        "terminal_transition_rejected",
    )
    if any(functional.get(field) is not True for field in required_true):
        raise Stage0G7Error("G7_FUNCTIONAL_BOOLEAN_CHECK_FAILED")
    if (
        functional.get("sensitive_pattern_rejections") != 4
        or functional.get("truth_write_failure_run_status") != "FAILED_FINAL"
        or functional.get("canonical_optimizer_steps") != [0, 1, 2, 3]
        or int(functional.get("canonical_typed_event_count", 0)) <= 4
        or int(functional.get("canonical_tensorboard_scalars", 0)) <= 0
        or functional.get("shared_metric_writer_rank") != 0
    ):
        raise Stage0G7Error("G7_FUNCTIONAL_CONTRACT_FAILED")
    rank_events = functional.get("rank_event_refs")
    rank_console = functional.get("rank_console_refs")
    if (
        not isinstance(rank_events, list)
        or len(rank_events) != 4
        or len(set(rank_events)) != 4
        or not isinstance(rank_console, list)
        or len(rank_console) != 4
        or len(set(rank_console)) != 4
    ):
        raise Stage0G7Error("G7_RANK_LOG_PARTITION_FAILED")
    lineage = _mapping(functional.get("canonical_lineage"), field="functional.lineage")
    tails = lineage.get("superseded_tails")
    segments = lineage.get("segments")
    if (
        not isinstance(tails, list)
        or len(tails) != 1
        or tails[0].get("disposition") != "SUPERSEDED"
        or not isinstance(segments, list)
        or len(segments) != 2
        or any(item.get("disposition") != "CANONICAL" for item in segments)
    ):
        raise Stage0G7Error("G7_CANONICAL_LINEAGE_DISPOSITION_FAILED")
    status = _mapping(functional.get("run_status"), field="functional.run_status")
    if status.get("run_status") != "SUCCESS":
        raise Stage0G7Error("G7_RUN_STATUS_FINAL_INVALID")
    matrix = _mapping(functional.get("transition_matrix"), field="transition_matrix")
    if set(_mapping(matrix.get("run"), field="transition.run")) != {
        "CREATED",
        "RUNNING",
        "RESUMABLE",
        "SUCCESS",
        "FAILED_FINAL",
        "ABORTED_FINAL",
    }:
        raise Stage0G7Error("G7_RUN_TRANSITION_MATRIX_INCOMPLETE")
    lifecycle_states = {
        "STARTING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "ABORTED",
        "STALE",
    }
    if (
        set(_mapping(matrix.get("attempt"), field="transition.attempt"))
        != lifecycle_states
        or set(_mapping(matrix.get("session"), field="transition.session"))
        != lifecycle_states
    ):
        raise Stage0G7Error("G7_ATTEMPT_SESSION_TRANSITION_MATRIX_INCOMPLETE")
    return {
        "paired_repetitions": pairs,
        "minimal_throughput_median_steps_per_second": minimal_median,
        "formal_throughput_median_steps_per_second": formal_median,
        "minimal_throughput_cv": coefficient_of_variation(minimal_rates),
        "formal_throughput_cv": coefficient_of_variation(formal_rates),
        "tracking_overhead_fraction": group_overhead,
        "max_paired_tracking_overhead_fraction": max(
            float(item["throughput_overhead_fraction"]) for item in pairs
        ),
        "rank_event_streams": 4,
        "rank_console_logs": 4,
        "sensitive_pattern_hits": 0,
        "canonical_optimizer_steps": [0, 1, 2, 3],
        "superseded_tail_count": 1,
        "tensorboard_failure_preserved_truth": True,
        "truth_failure_final_status": "FAILED_FINAL",
    }


def _run_suite(
    request: TaskExecutionRequest,
    root: Path,
    source: G7SourceBinding,
    suite_ref: str,
) -> tuple[list[str], list[dict[str, Any]], str, dict[str, Any], tuple[str, ...]]:
    config_ref = f"{suite_ref}/resolved-config.json"
    environment_ref = f"{suite_ref}/environment.json"
    config_path = _logical_path(root, config_ref, field="config_ref")
    environment_path = _logical_path(root, environment_ref, field="environment_ref")
    _write_or_verify(config_path, request.config.to_dict())
    _write_or_verify(environment_path, request.environment.to_dict())
    config_sha = sha256_file(config_path)
    environment_sha = sha256_file(environment_path)
    report_refs: list[str] = []
    reports: list[dict[str, Any]] = []
    order = (("minimal", "formal"), ("formal", "minimal"), ("minimal", "formal"))
    for repeat, modes in enumerate(order):
        for mode in modes:
            run_id = f"g7-{mode}-{repeat:02d}"
            plan_ref = f"{suite_ref}/plans/{run_id}.json"
            output_ref = f"{suite_ref}/workers/{run_id}"
            report_ref = f"{suite_ref}/reports/{run_id}.json"
            plan = _worker_plan(
                run_id=run_id,
                mode=mode,
                repeat=repeat,
                source=source,
                request=request,
                config_ref=config_ref,
                config_sha=config_sha,
                environment_ref=environment_ref,
                environment_sha=environment_sha,
                output_root_ref=output_ref,
                report_ref=report_ref,
            )
            _write_or_verify(_logical_path(root, plan_ref, field="plan_ref"), plan)
            environment = os.environ.copy()
            environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(root)
            source_path = str(source.repository / "src")
            environment["PYTHONPATH"] = source_path + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(source.repository / "ops" / "stage0" / "run_g7_worker.py"),
                    "--data-root",
                    str(root),
                    "--plan-ref",
                    plan_ref,
                ],
                cwd=source.repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
            )
            if completed.returncode != 0:
                raise Stage0G7Error(
                    f"G7_WORKER_PROCESS_FAILED:{run_id}:"
                    f"{completed.stdout[-2000:]}:{completed.stderr[-4000:]}"
                )
            report_refs.append(report_ref)
            reports.append(
                _validate_worker_report(
                    root, report_ref, source=source, request=request
                )
            )
    functional_ref, functional, functional_refs = _functional_suite(
        root, f"{suite_ref}/functional"
    )
    return report_refs, reports, functional_ref, functional, functional_refs


def run_formal_g7_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G7Error("G7_FORMAL_REQUEST_INVALID")
    if "stage0.G6" not in request.environment.passed_gate_ids:
        raise Stage0G7Error("G7_G6_GATE_MISSING")
    suite_ref = (
        f"evidence/stage0/g7-suite/{request.config.config_hash}/"
        f"{request.environment.environment_hash}"
    )
    report_refs, reports, functional_ref, functional, functional_refs = _run_suite(
        request, root, source, suite_ref
    )
    metrics = validate_g7_report_set(reports, functional)
    checked_at = max(
        _timestamp(report["completed_at"], field="worker.completed_at")
        for report in reports
    ).isoformat().replace("+00:00", "Z")
    evidence_refs = (
        tuple(source_refs)
        + tuple(report_refs)
        + (functional_ref,)
        + tuple(functional_refs)
    )
    checks = (
        Stage0GateCheck(
            "stage0.G7-LOGGING-CORRECTNESS",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "Typed JSONL, rank isolation, canonical resume lineage, status transitions, and derived views passed.",
            measurements={
                "rank_event_streams": metrics["rank_event_streams"],
                "rank_console_logs": metrics["rank_console_logs"],
                "canonical_optimizer_steps": metrics["canonical_optimizer_steps"],
                "superseded_tail_count": metrics["superseded_tail_count"],
                "sensitive_pattern_hits": metrics["sensitive_pattern_hits"],
            },
            evidence_refs=evidence_refs,
        ),
        Stage0GateCheck(
            "stage0.G7-LOGGING-OVERHEAD",
            Stage0CheckClass.PERFORMANCE,
            Stage0CheckStatus.PASS,
            "Three alternating fresh-process pairs kept formal tracking overhead within ten percent.",
            exception_eligible=True,
            measurements={
                "paired_repetitions": metrics["paired_repetitions"],
                "tracking_overhead_fraction": metrics["tracking_overhead_fraction"],
                "max_paired_overhead_fraction": metrics[
                    "max_paired_tracking_overhead_fraction"
                ],
                "minimal_throughput_cv": metrics["minimal_throughput_cv"],
                "formal_throughput_cv": metrics["formal_throughput_cv"],
                "threshold": 0.10,
            },
            evidence_refs=tuple(report_refs),
        ),
    )
    gate = GateRecord(
        gate_id=COMPONENT_GATE_ID,
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "paired_repetitions": 3,
            "max_tracking_overhead_fraction": metrics[
                "tracking_overhead_fraction"
            ],
            "canonical_optimizer_steps": 4,
            "sensitive_pattern_hits": 0,
        },
        threshold={
            "tracking_overhead_fraction_max": 0.10,
            "sensitive_pattern_hits_max": 0,
            "shared_metric_writers": 1,
        },
        evidence_refs=evidence_refs,
    )
    environment_id = str(request.config.base_config.section("runtime")["environment_id"])
    gate_report = Stage0GateReport(
        gate_id=COMPONENT_GATE_ID,
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        config_hashes={TASK_ID: request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(_logical_path(root, reference, field="evidence_ref")),
                schema_version=(
                    WORKER_REPORT_SCHEMA
                    if reference in report_refs
                    else "stage0-g7-functional-report-v1"
                    if reference == functional_ref
                    else "runtime-observability-evidence"
                    if reference in functional_refs
                    else "task-output-commit-v1"
                ),
            )
            for reference in evidence_refs
        ),
        checks=checks,
    )
    validation = _with_hash(
        {
            "schema_version": "stage0-g7-validation-report-v1",
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
            "schema_version": "stage0-g7-evidence-v1",
            "artifact_role": kind,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "component_gate_id": COMPONENT_GATE_ID,
            "full_g7_status": "PENDING_S0.9",
            "suite_root_ref": suite_ref,
            "worker_report_refs": report_refs,
            "functional_report_ref": functional_ref,
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
        message="Stage 0 S0.8 observability component passed",
        metadata={
            "stage0_g7_logging_specialized": True,
            "component_gate_id": COMPONENT_GATE_ID,
        },
    )


def validate_formal_g7_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G7Error("G7_OUTPUT_SET_INVALID")
    envelopes: list[dict[str, Any]] = []
    for kind, reference in outputs.items():
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != request.config.config_hash
        ):
            raise Stage0G7Error("G7_OUTPUT_COMMIT_IDENTITY_INVALID")
        envelope = _mapping(loaded.payload, field=f"output.{kind}")
        declared = envelope.pop("artifact_hash", None)
        if (
            envelope.get("schema_version") != "stage0-g7-evidence-v1"
            or envelope.get("artifact_role") != kind
            or envelope.get("status") != "PASS"
            or envelope.get("generator_git_commit") != source.git_commit
            or envelope.get("environment_hash") != request.environment.environment_hash
            or envelope.get("component_gate_id") != COMPONENT_GATE_ID
            or envelope.get("full_g7_status") != "PENDING_S0.9"
            or declared != canonical_json_hash(envelope)
        ):
            raise Stage0G7Error("G7_OUTPUT_ENVELOPE_INVALID")
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
        raise Stage0G7Error("G7_OUTPUT_ROLE_PAYLOAD_DRIFT")
    report_refs = canonical.get("worker_report_refs")
    functional_ref = canonical.get("functional_report_ref")
    if (
        not isinstance(report_refs, list)
        or len(report_refs) != 6
        or not isinstance(functional_ref, str)
    ):
        raise Stage0G7Error("G7_OUTPUT_REPORT_REFS_INVALID")
    reports = [
        _validate_worker_report(root, ref, source=source, request=request)
        for ref in report_refs
    ]
    functional = _mapping(
        load_canonical_json(_logical_path(root, functional_ref, field="functional_ref")),
        field="functional",
    )
    functional_declared = functional.pop("artifact_hash", None)
    if (
        functional.get("schema_version") != "stage0-g7-functional-report-v1"
        or functional_declared != canonical_json_hash(functional)
    ):
        raise Stage0G7Error("G7_FUNCTIONAL_REPORT_INVALID")
    functional["artifact_hash"] = functional_declared
    replayed = validate_g7_report_set(reports, functional)
    validation = _mapping(canonical.get("validation_report"), field="validation")
    validation_declared = validation.pop("artifact_hash", None)
    if (
        validation_declared != canonical_json_hash(validation)
        or validation.get("status") != "PASS"
        or validation.get("metrics") != replayed
    ):
        raise Stage0G7Error("G7_OUTPUT_VALIDATION_REPLAY_MISMATCH")
    gate = GateRecord.from_mapping(
        _mapping(canonical.get("gate_record"), field="gate_record")
    )
    report = Stage0GateReport.from_mapping(
        _mapping(canonical.get("gate_report"), field="gate_report")
    )
    if (
        gate.gate_id != COMPONENT_GATE_ID
        or gate.status is not GateStatus.PASS
        or report.gate_id != COMPONENT_GATE_ID
        or report.status.value != "PASS"
    ):
        raise Stage0G7Error("G7_OUTPUT_GATE_INVALID")
    return gate


def execute_stage0_g7(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g6_index_ref: str,
) -> Stage0G7FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g6_formal_state(
        data_root=root,
        index_ref=g6_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g7_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g7-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G7Error(
            f"G7_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, state.environment)
    gate = validate_formal_g7_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g7_logging_event_stream": outputs["event_stream"],
            "g7_logging_lineage": outputs["lineage_manifest"],
            "g7_logging_overhead": outputs["logging_overhead_report"],
            "gate_stage0_g7_logging": outputs["lineage_manifest"],
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids
        | frozenset({gate.gate_id}),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    write_canonical_json(
        _logical_path(root, environment_ref, field="environment_ref"),
        environment.to_dict(),
    )
    index: dict[str, JSONValue] = {
        "schema_version": "stage0-g7-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g6_index_ref": state.index_ref,
        "g6_index_sha256": state.index_sha256,
        "g6_gate_artifact_hash": state.gate_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "component_gate_ref": outputs["lineage_manifest"],
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.09_checkpoint_and_resume",
        "next_input_refs": list(outputs.values()),
    }
    index["artifact_hash"] = canonical_json_hash(index)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(_logical_path(root, index_ref, field="index_ref"), index)
    return Stage0G7FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def load_stage0_g7_formal_state(
    *, data_root: str | Path, index_ref: str, expected_git_commit: str
) -> Stage0G7FormalState:
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g7_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g6_index_ref",
        "g6_index_sha256",
        "g6_gate_artifact_hash",
        "config_ref",
        "config_hash",
        "task_output_refs",
        "component_gate_ref",
        "environment_ref",
        "environment_hash",
        "next_task_id",
        "next_input_refs",
        "artifact_hash",
    }
    if set(raw) != expected or raw.get("schema_version") != "stage0-g7-formalization-index-v1":
        raise Stage0G7Error("G7_STATE_INDEX_FIELDS_OR_VERSION_INVALID")
    declared = raw.pop("artifact_hash")
    if declared != canonical_json_hash(raw):
        raise Stage0G7Error("G7_STATE_INDEX_HASH_MISMATCH")
    raw["artifact_hash"] = declared
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G7Error("G7_STATE_GENERATOR_COMMIT_MISMATCH")
    g6 = load_stage0_g6_formal_state(
        data_root=root,
        index_ref=str(raw["g6_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    if (
        raw.get("g6_index_sha256") != g6.index_sha256
        or raw.get("g6_gate_artifact_hash") != g6.gate_artifact_hash
    ):
        raise Stage0G7Error("G7_STATE_G6_BINDING_MISMATCH")
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, raw["config_ref"], field="config_ref")),
            field="config",
        )
    )
    if config.task_id != TASK_ID or config.config_hash != raw.get("config_hash"):
        raise Stage0G7Error("G7_STATE_CONFIG_MISMATCH")
    outputs = _mapping(raw["task_output_refs"], field="task_output_refs")
    if set(outputs) != _OUTPUT_KINDS or any(
        not isinstance(value, str) for value in outputs.values()
    ):
        raise Stage0G7Error("G7_STATE_OUTPUTS_INVALID")
    ordered = {
        key: str(outputs[key])
        for key in ("event_stream", "lineage_manifest", "logging_overhead_report")
    }
    request = TaskExecutionRequest(config, config.task_definition, g6.environment)
    gate = validate_formal_g7_outputs(request, root, ordered)
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, raw["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    if (
        raw.get("component_gate_ref") != outputs["lineage_manifest"]
        or gate.gate_id not in environment.passed_gate_ids
        or environment.environment_hash != raw.get("environment_hash")
        or environment.evidence_refs.get("gate_stage0_g7_logging")
        != raw.get("component_gate_ref")
        or raw.get("next_task_id") != "stage0.09_checkpoint_and_resume"
        or set(raw.get("next_input_refs", [])) != set(outputs.values())
    ):
        raise Stage0G7Error("G7_STATE_ENVIRONMENT_OR_HANDOFF_INVALID")
    component = load_committed_task_artifact(
        root, str(raw["component_gate_ref"]), require_formal=True
    ).payload.get("artifact_hash")
    if not isinstance(component, str) or _SHA256_RE.fullmatch(component) is None:
        raise Stage0G7Error("G7_STATE_COMPONENT_HASH_INVALID")
    return Stage0G7FormalState(
        environment=environment,
        task_output_refs=ordered,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        component_artifact_hash=component,
        g6_index_ref=str(raw["g6_index_ref"]),
    )


__all__ = [
    "COMPONENT_GATE_ID",
    "G7SourceBinding",
    "Stage0G7Error",
    "Stage0G7FormalState",
    "Stage0G7FormalizationResult",
    "TASK_ID",
    "build_stage0_g7_config",
    "execute_stage0_g7",
    "load_stage0_g7_formal_state",
    "run_formal_g7_task",
    "validate_formal_g7_outputs",
    "validate_g7_report_set",
]
