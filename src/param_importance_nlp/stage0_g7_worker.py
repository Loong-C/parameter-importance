"""Fresh-process paired observability-overhead worker for Stage 0 S0.8."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import time
from typing import Any, Mapping

import torch

from .atomic import sha256_file
from .contracts import canonical_json_hash, load_canonical_json, write_canonical_json
from .contracts.jsonio import JSONValue
from .runtime import (
    EventRecord,
    EventType,
    JsonlEventSink,
    RunStatus,
    RunStatusStore,
    SessionStatus,
    canonical_optimizer_steps,
    read_event_stream,
    rebuild_tensorboard_from_jsonl,
)


WORKER_PLAN_SCHEMA = "stage0-g7-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage0-g7-worker-report-v1"


class Stage0G7WorkerError(RuntimeError):
    """The paired overhead child violated its immutable plan."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G7WorkerError(f"G7_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G7WorkerError(f"G7_WORKER_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G7WorkerError(f"G7_WORKER_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G7WorkerError(f"G7_WORKER_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _validate_plan(root: Path, plan_ref: str) -> tuple[dict[str, Any], Path]:
    path = _logical_path(root, plan_ref, field="plan_ref")
    plan = _mapping(load_canonical_json(path), field="plan")
    expected = {
        "schema_version",
        "run_id",
        "mode",
        "repeat_index",
        "generator_git_commit",
        "config_ref",
        "config_sha256",
        "config_hash",
        "environment_ref",
        "environment_sha256",
        "environment_hash",
        "output_root_ref",
        "report_ref",
        "warmup_steps",
        "measured_steps",
        "matrix_size",
        "seed",
        "artifact_hash",
    }
    if set(plan) != expected or plan.get("schema_version") != WORKER_PLAN_SCHEMA:
        raise Stage0G7WorkerError("G7_WORKER_PLAN_FIELDS_OR_VERSION_INVALID")
    declared = plan.pop("artifact_hash")
    if declared != canonical_json_hash(plan):
        raise Stage0G7WorkerError("G7_WORKER_PLAN_HASH_MISMATCH")
    plan["artifact_hash"] = declared
    if plan.get("mode") not in {"minimal", "formal"}:
        raise Stage0G7WorkerError("G7_WORKER_MODE_INVALID")
    if plan.get("warmup_steps") != 10 or plan.get("measured_steps") != 30:
        raise Stage0G7WorkerError("G7_WORKER_MEASUREMENT_PROTOCOL_INVALID")
    for field in ("config_ref", "environment_ref"):
        control_path = _logical_path(root, plan[field], field=field)
        if sha256_file(control_path) != plan[f"{field.removesuffix('_ref')}_sha256"]:
            raise Stage0G7WorkerError("G7_WORKER_CONTROL_FILE_HASH_MISMATCH")
    return plan, path


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _resource_snapshot(output_root: Path) -> dict[str, JSONValue]:
    """Capture bounded, non-sensitive resource counters for one repetition.

    Process command lines and environments are intentionally excluded.  The
    process list records only PID and the kernel ``comm`` name so this evidence
    cannot leak credentials passed through arguments.
    """

    memory: dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition(":")
            if separator and key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                token = raw.strip().split()[0]
                memory[f"{key.lower()}_bytes"] = int(token) * 1024
    network_rx = 0
    network_tx = 0
    netdev = Path("/proc/net/dev")
    if netdev.is_file():
        for line in netdev.read_text(encoding="utf-8").splitlines()[2:]:
            _, separator, counters = line.partition(":")
            fields = counters.split()
            if separator and len(fields) >= 9:
                network_rx += int(fields[0])
                network_tx += int(fields[8])
    competing: list[dict[str, JSONValue]] = []
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                name = (entry / "comm").read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if name.lower() in {"aria2c", "curl", "wget", "rclone"}:
                competing.append({"pid": int(entry.name), "process_name": name})
    gpu_processes: list[dict[str, JSONValue]] = []
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    ) if shutil.which("nvidia-smi") else None
    if probe is not None and probe.returncode == 0:
        for line in probe.stdout.splitlines():
            fields = [field.strip() for field in line.split(",", maxsplit=2)]
            if len(fields) == 3 and fields[1].isdigit():
                gpu_processes.append(
                    {
                        "gpu_uuid": fields[0],
                        "pid": int(fields[1]),
                        "process_name": Path(fields[2]).name,
                    }
                )
    disk = shutil.disk_usage(output_root)
    load_average = list(os.getloadavg()) if hasattr(os, "getloadavg") else []
    return {
        "observed_at": _now(),
        "cpu_count": os.cpu_count(),
        "load_average_1m_5m_15m": load_average,
        "memory": memory,
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "network_rx_bytes": network_rx,
        "network_tx_bytes": network_tx,
        "download_processes": competing,
        "gpu_compute_processes": gpu_processes,
    }


def run_stage0_g7_worker(
    *, data_root: str | Path, plan_ref: str
) -> dict[str, JSONValue]:
    root = Path(data_root).resolve(strict=True)
    plan, plan_path = _validate_plan(root, plan_ref)
    output_root = _logical_path(root, plan["output_root_ref"], field="output_root_ref")
    if output_root.exists():
        raise Stage0G7WorkerError("G7_WORKER_OUTPUT_COLLISION")
    output_root.mkdir(parents=True)
    before_resources = _resource_snapshot(output_root)
    mode = str(plan["mode"])
    repeat = int(plan["repeat_index"])
    run_id = str(plan["run_id"])
    event_path = output_root / "events.jsonl"
    status_ref: str | None = None
    summary_ref: str | None = None
    tensorboard_refs: list[str] = []
    status: RunStatusStore | None = None
    timestamp = "2026-08-03T00:00:00Z"
    if mode == "formal":
        status_path = output_root / "run-status.json"
        status = RunStatusStore(status_path, run_id=run_id, created_at=timestamp)
        status.transition_run(
            RunStatus.RUNNING,
            actor="stage0-g7-worker",
            reason="paired overhead measurement",
            at="2026-08-03T00:00:01Z",
        )
        status.register_attempt(
            attempt_id="attempt-0001",
            at="2026-08-03T00:00:02Z",
        )
        status.transition_attempt(
            "attempt-0001",
            SessionStatus.RUNNING,
            actor="stage0-g7-worker",
            reason="attempt started",
            at="2026-08-03T00:00:03Z",
        )
        status.register_session(
            attempt_id="attempt-0001",
            session_id="session-0001",
            at="2026-08-03T00:00:04Z",
        )
        status.transition_session(
            "session-0001",
            SessionStatus.RUNNING,
            actor="stage0-g7-worker",
            reason="measurement started",
            at="2026-08-03T00:00:05Z",
        )
        status_ref = status_path.relative_to(root).as_posix()

    torch.set_num_threads(1)
    torch.manual_seed(int(plan["seed"]))
    size = int(plan["matrix_size"])
    if not 256 <= size <= 4096:
        raise Stage0G7WorkerError("G7_WORKER_MATRIX_SIZE_INVALID")
    left = torch.randn(size, size, dtype=torch.float32)
    right = torch.randn(size, size, dtype=torch.float32)
    event_latencies: list[float] = []
    measured_step_seconds: list[float] = []
    critical_flush_seconds: list[float] = []
    warmup = int(plan["warmup_steps"])
    measured = int(plan["measured_steps"])
    total = warmup + measured
    with JsonlEventSink(event_path) as sink:
        start_flush = time.perf_counter()
        sink.append(
            EventRecord.create(
                experiment_id="stage0-g7-overhead",
                run_id=run_id,
                attempt_id="attempt-0001",
                session_id="session-0001",
                rank=0,
                event_type=EventType.RUN_LIFECYCLE,
                sequence=0,
                event_id=f"{run_id}-start",
                occurred_at="2026-08-03T00:00:00Z",
                payload={"status": "RUNNING"},
            ),
            critical=True,
        )
        critical_flush_seconds.append(time.perf_counter() - start_flush)
        checksum = 0.0
        for step in range(total):
            step_started = time.perf_counter()
            product = torch.mm(left, right)
            checksum += float(product[step % size, (step * 7) % size].item())
            append_started = time.perf_counter()
            sink.append(
                EventRecord.create(
                    experiment_id="stage0-g7-overhead",
                    run_id=run_id,
                    attempt_id="attempt-0001",
                    session_id="session-0001",
                    rank=0,
                    event_type=EventType.OPTIMIZER_STEP,
                    sequence=step + 1,
                    event_id=f"{run_id}-step-{step:04d}",
                    occurred_at=f"2026-08-03T00:01:{step:02d}Z",
                    payload={
                        "global_step": step,
                        "microstep_count": 1,
                        "sample_count": 1,
                        "effective_token_count": 128,
                        "mean_loss": 1.0 / (step + 1),
                        "learning_rate": 0.001,
                        "learning_rates_post_step": [0.001],
                        "global_gradient_norm": 0.5,
                    },
                ),
                critical=False,
            )
            event_latencies.append(time.perf_counter() - append_started)
            if status is not None and step % 5 == 0:
                status.heartbeat(
                    "session-0001",
                    last_step=step,
                    observed_at=f"2026-08-03T00:02:{step:02d}Z",
                )
            if step >= warmup:
                measured_step_seconds.append(time.perf_counter() - step_started)
        end_flush = time.perf_counter()
        sink.append(
            EventRecord.create(
                experiment_id="stage0-g7-overhead",
                run_id=run_id,
                attempt_id="attempt-0001",
                session_id="session-0001",
                rank=0,
                event_type=EventType.RUN_LIFECYCLE,
                sequence=total + 1,
                event_id=f"{run_id}-success",
                occurred_at="2026-08-03T00:03:00Z",
                payload={"status": "SUCCESS"},
            ),
            critical=True,
        )
        critical_flush_seconds.append(time.perf_counter() - end_flush)

    derived_seconds = 0.0
    tensorboard_scalar_count = 0
    if status is not None:
        derived_started = time.perf_counter()
        tensorboard_root = output_root / "tensorboard" / "session-0001"
        tensorboard_scalar_count = rebuild_tensorboard_from_jsonl(
            (event_path,), tensorboard_root
        )
        events = read_event_stream(event_path)
        optimizer = canonical_optimizer_steps((events,))
        summary: dict[str, JSONValue] = {
            "schema_version": "stage0-g7-derived-summary-v1",
            "run_id": run_id,
            "optimizer_step_count": len(optimizer),
            "first_step": int(optimizer[0].payload["global_step"]),
            "last_step": int(optimizer[-1].payload["global_step"]),
            "final_loss": float(optimizer[-1].payload["mean_loss"]),
            "source_event_sha256": sha256_file(event_path),
        }
        summary["artifact_hash"] = canonical_json_hash(summary)
        summary_path = output_root / "summary.json"
        write_canonical_json(summary_path, summary)
        summary_ref = summary_path.relative_to(root).as_posix()
        status.transition_session(
            "session-0001",
            SessionStatus.SUCCEEDED,
            actor="stage0-g7-worker",
            reason="measurement complete",
            at="2026-08-03T00:04:00Z",
        )
        status.transition_attempt(
            "attempt-0001",
            SessionStatus.SUCCEEDED,
            actor="stage0-g7-worker",
            reason="measurement complete",
            at="2026-08-03T00:04:01Z",
        )
        status.transition_run(
            RunStatus.SUCCESS,
            actor="stage0-g7-worker",
            reason="measurement complete",
            at="2026-08-03T00:04:02Z",
        )
        tensorboard_refs = [
            path.relative_to(root).as_posix()
            for path in sorted(tensorboard_root.rglob("*"))
            if path.is_file()
        ]
        if not tensorboard_refs:
            raise Stage0G7WorkerError("G7_WORKER_TENSORBOARD_OUTPUT_MISSING")
        derived_seconds = time.perf_counter() - derived_started

    measured_wall = sum(measured_step_seconds) + derived_seconds
    after_resources = _resource_snapshot(output_root)
    competition_free = not before_resources["download_processes"] and not after_resources[
        "download_processes"
    ]
    events = read_event_stream(event_path)
    optimizer_steps = canonical_optimizer_steps((events,))
    report: dict[str, JSONValue] = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "repeat_index": repeat,
        "status": "PASS",
        "completed_at": _now(),
        "generator_git_commit": str(plan["generator_git_commit"]),
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "config_hash": str(plan["config_hash"]),
        "environment_hash": str(plan["environment_hash"]),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "matrix_size": size,
        "measured_wall_seconds": measured_wall,
        "steps_per_second": measured / measured_wall,
        "step_median_seconds": statistics.median(measured_step_seconds),
        "step_p95_seconds": _percentile95(measured_step_seconds),
        "workload_checksum": checksum,
        "event_stream_ref": event_path.relative_to(root).as_posix(),
        "event_stream_sha256": sha256_file(event_path),
        "event_count": len(events),
        "optimizer_steps": [int(event.payload["global_step"]) for event in optimizer_steps],
        "event_append_median_seconds": statistics.median(event_latencies[warmup:]),
        "event_append_p95_seconds": _percentile95(event_latencies[warmup:]),
        "critical_flush_max_seconds": max(critical_flush_seconds),
        "derived_tracking_seconds": derived_seconds,
        "tensorboard_scalar_count": tensorboard_scalar_count,
        "tensorboard_refs": tensorboard_refs,
        "summary_ref": summary_ref,
        "status_ref": status_ref,
        "resource_snapshots": {
            "before": before_resources,
            "after": after_resources,
        },
        "competition_free": competition_free,
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_path = _logical_path(root, plan["report_ref"], field="report_ref")
    write_canonical_json(report_path, report)
    return report


__all__ = [
    "Stage0G7WorkerError",
    "WORKER_PLAN_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "run_stage0_g7_worker",
]
