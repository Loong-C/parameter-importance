"""Append-only bounded S2.4 formal-r20 six-cell GPU work queue.

This launcher owns only process scheduling.  Each child invokes
``run_s204_formal.py --execute`` with one cell, one approved UUID, and the
same explicit execution commit.  A failed cell is recorded once and is never
retried; another pending cell may still use the newly idle GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence


APPROVED_GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)
EXCLUDED_GPU_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
EXCLUDED_PCI = "0000:50:00.0"
RUN_NAME = "formal-r20-g3-v5"


def _hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "stage2-s204-r20-queue-event-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **dict(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(payload)
    value["artifact_hash"] = _hash(value)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"R20_IMMUTABLE_IDENTITY_CONFLICT:{path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _parse_cell_config(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--cell-config must be CELL=PATH: {value!r}")
        cell_id, raw_path = value.split("=", 1)
        if not cell_id or not raw_path or cell_id in result:
            raise ValueError(f"invalid or duplicate --cell-config: {value!r}")
        result[cell_id] = Path(raw_path).resolve()
    if len(result) != 6:
        raise ValueError("r20 queue requires exactly six distinct --cell-config entries")
    return result


def _parse_cell_environment(values: Sequence[str], cells: Mapping[str, Path]) -> dict[str, Path]:
    """Parse the immutable per-cell TaskRuntimeEnvironment refs.

    A base environment is insufficient for formal S2.4: each cell carries a
    distinct sizing-plan, parameter-registry, and delta-sci binding.  Requiring
    the six explicit mappings prevents a queue launch from silently applying
    one cell's evidence to another cell.
    """

    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--runtime-environment must be CELL=PATH: {value!r}")
        cell_id, raw_path = value.split("=", 1)
        if cell_id not in cells or not raw_path or cell_id in result:
            raise ValueError(f"invalid or duplicate --runtime-environment: {value!r}")
        result[cell_id] = Path(raw_path).resolve()
    if set(result) != set(cells):
        raise ValueError("r20 queue requires exactly one runtime environment per cell")
    return result


def lpt_order(cell_estimates: Mapping[str, float]) -> tuple[str, ...]:
    """Return deterministic longest-processing-time-first cell order."""

    if len(cell_estimates) != 6 or not cell_estimates:
        raise ValueError("r20 queue requires six cell estimates")
    for cell_id, estimate in cell_estimates.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("r20 cell IDs must be non-empty strings")
        if not isinstance(estimate, (int, float)) or not float(estimate) > 0:
            raise ValueError(f"r20 cell estimate must be positive: {cell_id}")
    return tuple(sorted(cell_estimates, key=lambda cell: (-float(cell_estimates[cell]), cell)))


def _parse_estimates(values: Sequence[str], cells: Mapping[str, Path]) -> dict[str, float]:
    estimates = {cell: (2.0 if "31m" in cell or "31M" in cell else 1.0) for cell in cells}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--cell-estimate must be CELL=SECONDS: {value!r}")
        cell, raw = value.split("=", 1)
        if cell not in cells:
            raise ValueError(f"estimate references unknown cell: {cell}")
        estimates[cell] = float(raw)
    return estimates


def _child_command(
    *,
    python: str,
    launcher: Path,
    execution_commit: str,
    cell_id: str,
    cell_config: Path,
    gpu_uuid: str,
    g21_evidence: Path,
    asset_resolution: Path,
    data_range: Path,
    data_root: Path,
    output_root: Path,
    runtime_environment: Path,
    heartbeat_seconds: float,
) -> list[str]:
    if gpu_uuid not in APPROVED_GPU_UUIDS or gpu_uuid == EXCLUDED_GPU_UUID:
        raise ValueError(f"GPU is not an approved r20 UUID: {gpu_uuid}")
    return [
        python,
        str(launcher),
        "--g21-evidence",
        str(g21_evidence),
        "--asset-resolution",
        str(asset_resolution),
        "--data-range",
        str(data_range),
        "--output-root",
        str(output_root),
        "--data-root",
        str(data_root),
        "--runtime-config",
        str(cell_config),
        "--runtime-environment",
        str(runtime_environment),
        "--cell-id",
        cell_id,
        "--execution-commit",
        execution_commit,
        "--cuda-visible-devices",
        gpu_uuid,
        "--heartbeat-seconds",
        str(heartbeat_seconds),
        "--execute",
    ]


def run_queue(args: argparse.Namespace) -> int:
    if len(set(APPROVED_GPU_UUIDS)) != 4 or EXCLUDED_GPU_UUID in APPROVED_GPU_UUIDS:
        raise RuntimeError("R20_APPROVED_GPU_SET_INVALID")
    if len(args.execution_commit) != 40 or any(c not in "0123456789abcdef" for c in args.execution_commit):
        raise ValueError("--execution-commit must be 40 lowercase hexadecimal characters")
    cells = _parse_cell_config(args.cell_config)
    environments = _parse_cell_environment(args.runtime_environment, cells)
    estimates = _parse_estimates(args.cell_estimate, cells)
    order = lpt_order(estimates)
    python = str(Path(args.python).resolve())
    launcher = Path(args.s204_launcher).resolve()
    output_root = Path(args.output_root).resolve()
    if RUN_NAME not in output_root.parts:
        raise ValueError(f"r20 output root must be under a {RUN_NAME} directory")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_root = Path(args.queue_root).resolve() / RUN_NAME / run_id
    queue_root.mkdir(parents=True, exist_ok=False)
    event_path = queue_root / "queue-events.jsonl"
    manifest = {
        "schema_version": "stage2-s204-r20-queue-v1",
        "run_name": RUN_NAME,
        "run_id": run_id,
        "execution_commit": args.execution_commit,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "cell_order_lpt": list(order),
        "cell_estimates_seconds": estimates,
        "cell_configs": {cell: path.as_posix() for cell, path in cells.items()},
        "retry_policy": "none",
        "output_root": output_root.as_posix(),
    }
    _publish_once(queue_root / "queue-manifest.json", manifest)
    _append(event_path, {"event": "QUEUE_STARTED", "run_id": run_id, **manifest})

    pending = list(order)
    running: dict[str, tuple[str, subprocess.Popen[str], float]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    stop = threading.Event()
    state_lock = threading.Lock()

    def heartbeat() -> None:
        while not stop.wait(args.heartbeat_seconds):
            with state_lock:
                running_snapshot = {
                    uuid: {"cell_id": cell, "pid": process.pid}
                    for uuid, (cell, process, _) in running.items()
                }
                pending_snapshot = list(pending)
            _append(
                event_path,
                {
                    "event": "QUEUE_HEARTBEAT",
                    "run_id": run_id,
                    "pending_cells": pending_snapshot,
                    "running": running_snapshot,
                },
            )

    heartbeat_thread = threading.Thread(target=heartbeat, name="r20-queue-heartbeat", daemon=True)
    heartbeat_thread.start()
    try:
        while pending or running:
            with state_lock:
                running_uuids = set(running)
            for gpu_uuid in APPROVED_GPU_UUIDS:
                if not pending or gpu_uuid in running_uuids:
                    continue
                cell_id = pending.pop(0)
                cell_root = queue_root / "cells" / cell_id
                cell_root.mkdir(parents=True, exist_ok=False)
                command = _child_command(
                    python=python,
                    launcher=launcher,
                    execution_commit=args.execution_commit,
                    cell_id=cell_id,
                    cell_config=cells[cell_id],
                    gpu_uuid=gpu_uuid,
                    g21_evidence=Path(args.g21_evidence).resolve(),
                    asset_resolution=Path(args.asset_resolution).resolve(),
                    data_range=Path(args.data_range).resolve(),
                    data_root=Path(args.data_root).resolve(),
                    output_root=output_root,
                    runtime_environment=environments[cell_id],
                    heartbeat_seconds=args.child_heartbeat_seconds,
                )
                stdout = (cell_root / "stdout.log").open("x", encoding="utf-8")
                stderr = (cell_root / "stderr.log").open("x", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": gpu_uuid,
                        "PYTHONPATH": os.pathsep.join(
                            item
                            for item in (
                                str(launcher.parents[2] / "src"),
                                os.environ.get("PYTHONPATH", ""),
                            )
                            if item
                        ),
                    },
                    text=True,
                )
                stdout.close()
                stderr.close()
                _publish_once(
                    cell_root / "pid.json",
                    {
                        "schema_version": "stage2-s204-r20-cell-pid-v1",
                        "run_id": run_id,
                        "cell_id": cell_id,
                        "gpu_uuid": gpu_uuid,
                        "pid": process.pid,
                        "command": command,
                        "execution_commit": args.execution_commit,
                    },
                )
                with state_lock:
                    running[gpu_uuid] = (cell_id, process, time.monotonic())
                _append(
                    event_path,
                    {"event": "CELL_STARTED", "cell_id": cell_id, "gpu_uuid": gpu_uuid, "pid": process.pid},
                )

            with state_lock:
                running_items = list(running.items())
            for gpu_uuid, (cell_id, process, started) in running_items:
                returncode = process.poll()
                if returncode is None:
                    continue
                elapsed = time.monotonic() - started
                outcome = {
                    "schema_version": "stage2-s204-r20-cell-status-v1",
                    "run_id": run_id,
                    "cell_id": cell_id,
                    "gpu_uuid": gpu_uuid,
                    "pid": process.pid,
                    "returncode": returncode,
                    "status": "COMPLETE" if returncode == 0 else "FAILED",
                    "elapsed_seconds": elapsed,
                    "execution_commit": args.execution_commit,
                    "retry": False,
                }
                _publish_once(queue_root / "cells" / cell_id / "queue-status.json", outcome)
                _append(event_path, {"event": "CELL_EXITED", **outcome})
                outcomes[cell_id] = outcome
                with state_lock:
                    del running[gpu_uuid]
            if pending or running:
                time.sleep(args.poll_seconds)
    finally:
        stop.set()
        heartbeat_thread.join(timeout=max(1.0, args.heartbeat_seconds))

    status = "COMPLETE" if len(outcomes) == len(cells) and all(item["returncode"] == 0 for item in outcomes.values()) else "FAILED"
    final = {
        "schema_version": "stage2-s204-r20-queue-final-v1",
        "run_name": RUN_NAME,
        "run_id": run_id,
        "status": status,
        "execution_commit": args.execution_commit,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "cell_order_lpt": list(order),
        "outcomes": outcomes,
        "retry_policy": "none",
    }
    _publish_once(queue_root / "queue-final.json", final)
    _append(event_path, {"event": "QUEUE_FINISHED", **final})
    print(json.dumps({**final, "queue_root": queue_root.as_posix()}, ensure_ascii=False, sort_keys=True))
    return 0 if status == "COMPLETE" else 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detached S2.4 formal-r20 four-GPU bounded queue")
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--cell-config", action="append", required=True, help="CELL=resolved-config.json (exactly six)")
    parser.add_argument("--cell-estimate", action="append", default=[], help="CELL=seconds; defaults 31m=2, others=1")
    parser.add_argument("--g21-evidence", type=Path, required=True)
    parser.add_argument("--asset-resolution", type=Path, required=True)
    parser.add_argument("--data-range", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-environment",
        action="append",
        required=True,
        help="CELL=per-cell TaskRuntimeEnvironment JSON (exactly six)",
    )
    parser.add_argument("--s204-launcher", type=Path, default=Path(__file__).with_name("run_s204_formal.py"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-id")
    parser.add_argument("--child-heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.heartbeat_seconds <= 0 or args.child_heartbeat_seconds <= 0 or args.poll_seconds <= 0:
            raise ValueError("heartbeat and poll intervals must be positive")
        return run_queue(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"S2.4 r20 queue blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
