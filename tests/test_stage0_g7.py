from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from param_importance_nlp.atomic import sha256_file
from param_importance_nlp.contracts import canonical_json_hash, write_canonical_json
from param_importance_nlp.runtime import (
    EventRecord,
    EventType,
    JsonlEventSink,
    RunStatus,
    RunStatusStore,
    SessionStatus,
    build_canonical_event_lineage,
    canonical_optimizer_steps,
    read_event_stream,
)
from param_importance_nlp import stage0_g7
from param_importance_nlp.stage0_g7 import Stage0G7Error, validate_g7_report_set
from param_importance_nlp.stage0_g7_worker import run_stage0_g7_worker


def _event(
    *,
    session: str,
    attempt: str,
    sequence: int,
    event_type: EventType,
    payload: dict[str, object],
) -> EventRecord:
    if event_type is EventType.OPTIMIZER_STEP:
        payload = {
            "global_step": 0,
            "microstep_count": 1,
            "sample_count": 1,
            "effective_token_count": 16,
            "mean_loss": 1.0,
            "global_gradient_norm": 0.5,
            "learning_rates_post_step": {"group_0000": 0.1},
            **payload,
        }
    elif event_type is EventType.SYSTEM:
        payload = {
            "gpu_memory_bytes": 0,
            "gpu_utilization_percent": 0.0,
            "cpu_memory_bytes": 1024,
            "disk_free_bytes": 2048,
            "throughput_units_per_second": 1.0,
            **payload,
        }
    elif event_type is EventType.CHECKPOINT:
        payload = {
            "global_step": 0,
            "checkpoint_ref": "checkpoints/k.json",
            "status": "COMMITTED",
            "manifest_sha256": "a" * 64,
            **payload,
        }
    return EventRecord.create(
        experiment_id="g7-test",
        run_id="run-g7",
        attempt_id=attempt,
        session_id=session,
        rank=0,
        event_type=event_type,
        sequence=sequence,
        payload=payload,
        event_id=f"{session}-{sequence}",
        occurred_at=f"2026-08-03T00:00:{sequence:02d}Z",
    )


def test_run_status_recovery_stale_and_terminal_transitions(tmp_path: Path) -> None:
    store = RunStatusStore(
        tmp_path / "run-status.json",
        run_id="run-g7",
        created_at="2026-08-03T00:00:00Z",
    )
    store.transition_run(
        RunStatus.RUNNING,
        actor="test",
        reason="initial launch",
        at="2026-08-03T00:00:01Z",
    )
    with pytest.raises(RuntimeError, match="RUN_STATUS_WRITER_ALREADY_ACTIVE"):
        RunStatusStore(tmp_path / "run-status.json", run_id="run-g7")
    store.register_attempt(attempt_id="attempt-1", at="2026-08-03T00:00:02Z")
    store.transition_attempt(
        "attempt-1",
        SessionStatus.RUNNING,
        actor="test",
        reason="attempt ready",
        at="2026-08-03T00:00:03Z",
    )
    store.register_session(
        attempt_id="attempt-1",
        session_id="session-1",
        at="2026-08-03T00:00:04Z",
    )
    store.transition_session(
        "session-1",
        SessionStatus.RUNNING,
        actor="test",
        reason="worker ready",
        at="2026-08-03T00:00:05Z",
    )
    store.heartbeat(
        "session-1", last_step=8, observed_at="2026-08-03T00:00:06Z"
    )
    with pytest.raises(ValueError, match="RUN_STATUS_HEARTBEAT_STEP_REGRESSION"):
        store.heartbeat(
            "session-1", last_step=7, observed_at="2026-08-03T00:00:07Z"
        )
    with pytest.raises(ValueError, match="RUN_STATUS_STALE_PRECONDITIONS_FAILED"):
        store.transition_session(
            "session-1",
            SessionStatus.STALE,
            actor="watchdog",
            reason="probe still sees process",
            at="2026-08-03T00:01:00Z",
            process_exists=True,
            stale_after_seconds=30,
        )
    store.transition_session(
        "session-1",
        SessionStatus.STALE,
        actor="watchdog",
        reason="process absent and heartbeat expired",
        at="2026-08-03T00:01:00Z",
        process_exists=False,
        stale_after_seconds=30,
    )
    store.transition_session(
        "session-1",
        SessionStatus.FAILED,
        actor="watchdog",
        reason="stale session finalized",
        at="2026-08-03T00:01:01Z",
    )
    store.transition_attempt(
        "attempt-1",
        SessionStatus.FAILED,
        actor="watchdog",
        reason="session finalized",
        at="2026-08-03T00:01:02Z",
    )
    with pytest.raises(
        ValueError, match="RUN_STATUS_RECOVERY_REQUIRES_COMPLETE_CHECKPOINT"
    ):
        store.transition_run(
            RunStatus.RESUMABLE,
            actor="test",
            reason="missing checkpoint proof",
            at="2026-08-03T00:01:03Z",
        )
    store.transition_run(
        RunStatus.RESUMABLE,
        actor="test",
        reason="complete checkpoint found",
        at="2026-08-03T00:01:04Z",
        checkpoint_ref="checkpoints/k.json",
        checkpoint_complete=True,
    )
    with pytest.raises(
        ValueError, match="RUN_STATUS_RECOVERY_REQUIRES_COMPLETE_CHECKPOINT"
    ):
        store.transition_run(
            RunStatus.RUNNING,
            actor="test",
            reason="unverified resume",
            at="2026-08-03T00:01:05Z",
        )
    store.transition_run(
        RunStatus.RUNNING,
        actor="test",
        reason="verified resume",
        at="2026-08-03T00:01:06Z",
        checkpoint_ref="checkpoints/k.json",
        checkpoint_complete=True,
    )
    store.register_attempt(attempt_id="attempt-2", at="2026-08-03T00:01:07Z")
    store.transition_attempt(
        "attempt-2",
        SessionStatus.RUNNING,
        actor="test",
        reason="resume attempt ready",
        at="2026-08-03T00:01:08Z",
    )
    store.register_session(
        attempt_id="attempt-2",
        session_id="session-2",
        at="2026-08-03T00:01:09Z",
    )
    store.transition_session(
        "session-2",
        SessionStatus.RUNNING,
        actor="test",
        reason="resume ready",
        at="2026-08-03T00:01:10Z",
    )
    store.transition_session(
        "session-2",
        SessionStatus.SUCCEEDED,
        actor="test",
        reason="complete",
        at="2026-08-03T00:01:11Z",
    )
    store.transition_attempt(
        "attempt-2",
        SessionStatus.SUCCEEDED,
        actor="test",
        reason="complete",
        at="2026-08-03T00:01:12Z",
    )
    snapshot = store.transition_run(
        RunStatus.SUCCESS,
        actor="test",
        reason="complete",
        at="2026-08-03T00:01:13Z",
    )
    assert snapshot["run_status"] == "SUCCESS"
    with pytest.raises(ValueError, match="RUN_STATUS_TRANSITION_FORBIDDEN"):
        store.transition_run(
            RunStatus.RUNNING,
            actor="test",
            reason="terminal state is immutable",
            at="2026-08-03T00:01:14Z",
        )


def test_canonical_event_lineage_preserves_raw_tail_and_replays(tmp_path: Path) -> None:
    first = tmp_path / "raw" / "session-1.jsonl"
    second = tmp_path / "raw" / "session-2.jsonl"
    with JsonlEventSink(first) as sink:
        sink.append(
            _event(
                session="session-1",
                attempt="attempt-1",
                sequence=0,
                event_type=EventType.OPTIMIZER_STEP,
                payload={"global_step": 0, "mean_loss": 3.0},
            )
        )
        sink.append(
            _event(
                session="session-1",
                attempt="attempt-1",
                sequence=1,
                event_type=EventType.CHECKPOINT,
                payload={"checkpoint_id": "k", "global_step": 0},
            ),
            critical=True,
        )
        sink.append(
            _event(
                session="session-1",
                attempt="attempt-1",
                sequence=2,
                event_type=EventType.OPTIMIZER_STEP,
                payload={"global_step": 1, "mean_loss": 100.0},
            )
        )
    with JsonlEventSink(second) as sink:
        sink.append(
            _event(
                session="session-2",
                attempt="attempt-2",
                sequence=0,
                event_type=EventType.OPTIMIZER_STEP,
                payload={"global_step": 1, "mean_loss": 2.0},
            )
        )
        sink.append(
            _event(
                session="session-2",
                attempt="attempt-2",
                sequence=1,
                event_type=EventType.SYSTEM,
                payload={"global_step": 1, "cpu_memory_bytes": 1024},
            )
        )
        sink.append(
            _event(
                session="session-2",
                attempt="attempt-2",
                sequence=2,
                event_type=EventType.OPTIMIZER_STEP,
                payload={"global_step": 2, "mean_loss": 1.0},
            )
        )
    first_hash = sha256_file(first)
    second_hash = sha256_file(second)
    lineage = build_canonical_event_lineage(
        root=tmp_path,
        run_id="run-g7",
        segments=(
            {
                "attempt_id": "attempt-1",
                "session_id": "session-1",
                "rank": 0,
                "event_ref": "raw/session-1.jsonl",
                "event_sha256": first_hash,
                "sequence_start": 0,
                "sequence_end": 1,
                "checkpoint_ref": "checkpoints/k.json",
            },
            {
                "attempt_id": "attempt-2",
                "session_id": "session-2",
                "rank": 0,
                "event_ref": "raw/session-2.jsonl",
                "event_sha256": second_hash,
                "sequence_start": 0,
                "sequence_end": 2,
                "checkpoint_ref": None,
            },
        ),
        superseded_tails=(
            {
                "attempt_id": "attempt-1",
                "session_id": "session-1",
                "event_ref": "raw/session-1.jsonl",
                "event_sha256": first_hash,
                "sequence_start": 2,
                "sequence_end": 2,
                "reason": "superseded_after_k",
            },
        ),
        parent_checkpoint_ref="checkpoints/k.json",
        output_ref="derived/lineage.json",
        canonical_event_ref="derived/canonical.jsonl",
    )
    canonical = read_event_stream(tmp_path / "derived" / "canonical.jsonl")
    assert sha256_file(first) == first_hash
    assert sha256_file(second) == second_hash
    assert lineage["optimizer_steps"] == [0, 1, 2]
    assert [event.sequence for event in canonical] == list(range(len(canonical)))
    assert [
        event.payload["global_step"] for event in canonical_optimizer_steps((canonical,))
    ] == [0, 1, 2]
    assert any(
        event.event_type == EventType.SYSTEM.value
        and event.payload["global_step"] == 1
        for event in canonical
    )

    with pytest.raises(ValueError, match="EVENT_LINEAGE_RAW_STREAM_HASH_MISMATCH"):
        build_canonical_event_lineage(
            root=tmp_path,
            run_id="run-g7",
            segments=(
                {
                    "attempt_id": "attempt-1",
                    "session_id": "session-1",
                    "rank": 0,
                    "event_ref": "raw/session-1.jsonl",
                    "event_sha256": "0" * 64,
                    "sequence_start": 0,
                    "sequence_end": 1,
                    "checkpoint_ref": "checkpoints/k.json",
                },
            ),
            superseded_tails=(),
            parent_checkpoint_ref="checkpoints/k.json",
            output_ref="derived/bad-lineage.json",
            canonical_event_ref="derived/bad-canonical.jsonl",
        )


def _functional() -> dict[str, object]:
    return {
        "concurrent_writer_rejected": True,
        "non_json_payload_rejected": True,
        "volume_guard_rejected": True,
        "tensorboard_failure_warned": True,
        "tensorboard_failure_truth_unchanged": True,
        "raw_streams_unchanged": True,
        "terminal_transition_rejected": True,
        "sensitive_pattern_rejections": 4,
        "truth_write_failure_run_status": "FAILED_FINAL",
        "canonical_optimizer_steps": [0, 1, 2, 3],
        "canonical_typed_event_count": 9,
        "canonical_tensorboard_scalars": 12,
        "shared_metric_writer_rank": 0,
        "rank_event_refs": [f"rank/{rank}/events.jsonl" for rank in range(4)],
        "rank_console_refs": [f"rank/{rank}/console.log" for rank in range(4)],
        "canonical_lineage": {
            "segments": [
                {"disposition": "CANONICAL"},
                {"disposition": "CANONICAL"},
            ],
            "superseded_tails": [{"disposition": "SUPERSEDED"}],
        },
        "run_status": {"run_status": "SUCCESS"},
        "transition_matrix": {
            "run": {
                status: []
                for status in (
                    "CREATED",
                    "RUNNING",
                    "RESUMABLE",
                    "SUCCESS",
                    "FAILED_FINAL",
                    "ABORTED_FINAL",
                )
            },
            "attempt": {
                status: []
                for status in (
                    "STARTING",
                    "RUNNING",
                    "SUCCEEDED",
                    "FAILED",
                    "ABORTED",
                    "STALE",
                )
            },
            "session": {
                status: []
                for status in (
                    "STARTING",
                    "RUNNING",
                    "SUCCEEDED",
                    "FAILED",
                    "ABORTED",
                    "STALE",
                )
            },
        },
    }


def _reports(formal_rates: tuple[float, float, float] = (95.0, 94.0, 96.0)) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for repeat, (minimal_rate, formal_rate) in enumerate(
        zip((100.0, 101.0, 99.0), formal_rates, strict=True)
    ):
        for mode, rate in (("minimal", minimal_rate), ("formal", formal_rate)):
            reports.append(
                {
                    "repeat_index": repeat,
                    "mode": mode,
                    "workload_checksum": 123.5 + repeat,
                    "steps_per_second": rate,
                    "event_append_p95_seconds": 0.001,
                    "critical_flush_max_seconds": 0.002,
                    "derived_tracking_seconds": 0.003 if mode == "formal" else 0.0,
                    "step_median_seconds": 1.0 / rate,
                    "step_p95_seconds": 1.2 / rate,
                }
            )
    return reports


def test_g7_report_replay_uses_three_repeat_medians_and_rejects_mutations() -> None:
    metrics = validate_g7_report_set(_reports(), _functional())
    assert metrics["minimal_throughput_median_steps_per_second"] == 100.0
    assert metrics["formal_throughput_median_steps_per_second"] == 95.0
    assert metrics["tracking_overhead_fraction"] == pytest.approx(0.05)

    with pytest.raises(Stage0G7Error, match="G7_TRACKING_OVERHEAD_EXCEEDED"):
        validate_g7_report_set(_reports((80.0, 79.0, 81.0)), _functional())

    functional = _functional()
    functional["rank_event_refs"] = ["same.jsonl"] * 4
    with pytest.raises(Stage0G7Error, match="G7_RANK_LOG_PARTITION_FAILED"):
        validate_g7_report_set(_reports(), functional)


def test_g7_functional_suite_materializes_only_canonical_tensorboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuilt_sources: list[list[list[int]]] = []

    def fake_rebuild(streams, output_root):
        target = Path(output_root)
        if target.is_file():
            raise OSError("injected derived writer failure")
        observed: list[list[int]] = []
        scalar_count = 0
        for stream in streams:
            events = read_event_stream(stream)
            steps = [
                int(event.payload["global_step"])
                for event in events
                if event.event_type == EventType.OPTIMIZER_STEP.value
            ]
            observed.append(steps)
            scalar_count += len(steps)
        target.mkdir(parents=True, exist_ok=True)
        (target / "events.out").write_bytes(b"derived\n")
        rebuilt_sources.append(observed)
        return scalar_count

    monkeypatch.setattr(stage0_g7, "rebuild_tensorboard_from_jsonl", fake_rebuild)
    report_ref, report, evidence_refs = stage0_g7._functional_suite(
        tmp_path, "g7-functional"
    )

    assert report_ref == "g7-functional/functional-report.json"
    assert len(evidence_refs) == len(set(evidence_refs))
    assert report["canonical_optimizer_steps"] == [0, 1, 2, 3]
    assert report["raw_streams_unchanged"] is True
    assert report["tensorboard_failure_warned"] is True
    assert rebuilt_sources == [[[0, 1, 2, 3]]]
    metrics = validate_g7_report_set(_reports(), report)
    assert metrics["canonical_optimizer_steps"] == [0, 1, 2, 3]

    functional = deepcopy(_functional())
    functional["canonical_lineage"]["superseded_tails"] = []
    with pytest.raises(
        Stage0G7Error, match="G7_CANONICAL_LINEAGE_DISPOSITION_FAILED"
    ):
        validate_g7_report_set(_reports(), functional)


def test_event_contract_rejects_missing_checkpoint_id_and_large_payload() -> None:
    with pytest.raises(ValueError, match="EVENT_CHECKPOINT_ID_MISSING"):
        EventRecord.create(
            experiment_id="e",
            run_id="r",
            attempt_id="a",
            session_id="s",
            rank=0,
            event_type=EventType.CHECKPOINT,
            sequence=0,
            payload={"global_step": 0},
        )
    with pytest.raises(ValueError, match="EVENT_(STRING_TOO_LONG|PAYLOAD_SIZE_LIMIT)"):
        EventRecord.create(
            experiment_id="e",
            run_id="r",
            attempt_id="a",
            session_id="s",
            rank=0,
            event_type=EventType.SYSTEM,
            sequence=0,
            payload={"samples": ["x" * 8192 for _ in range(129)]},
        )


def test_g7_minimal_worker_emits_typed_truth_and_measurement_protocol(
    tmp_path: Path,
) -> None:
    config_ref = "control/config.json"
    environment_ref = "control/environment.json"
    config_path = tmp_path / config_ref
    environment_path = tmp_path / environment_ref
    write_canonical_json(config_path, {"schema_version": "test-config-v1"})
    write_canonical_json(environment_path, {"schema_version": "test-environment-v1"})
    plan = {
        "schema_version": "stage0-g7-worker-plan-v1",
        "run_id": "g7-minimal-test",
        "mode": "minimal",
        "repeat_index": 0,
        "generator_git_commit": "a" * 40,
        "config_ref": config_ref,
        "config_sha256": sha256_file(config_path),
        "config_hash": "b" * 64,
        "environment_ref": environment_ref,
        "environment_sha256": sha256_file(environment_path),
        "environment_hash": "c" * 64,
        "output_root_ref": "workers/minimal",
        "report_ref": "reports/minimal.json",
        "warmup_steps": 10,
        "measured_steps": 30,
        "matrix_size": 256,
        "seed": 20260803,
    }
    plan["artifact_hash"] = canonical_json_hash(plan)
    write_canonical_json(tmp_path / "plans/minimal.json", plan)

    report = run_stage0_g7_worker(
        data_root=tmp_path, plan_ref="plans/minimal.json"
    )
    assert report["event_count"] == 42
    assert report["optimizer_steps"] == list(range(40))
    assert report["step_p95_seconds"] >= report["step_median_seconds"] > 0
    assert report["tensorboard_refs"] == []
    assert report["resource_snapshots"]["before"]["disk_free_bytes"] > 0
    events = read_event_stream(tmp_path / str(report["event_stream_ref"]))
    assert len(canonical_optimizer_steps((events,))) == 40
