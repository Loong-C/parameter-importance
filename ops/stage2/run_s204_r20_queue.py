"""Append-only bounded S2.4 formal-r20 six-cell GPU work queue.

This launcher owns only process scheduling.  Each child invokes
``run_s204_formal.py --execute`` with one cell, one approved UUID, and the
same explicit execution commit.  A failed cell is recorded once and is never
retried; another pending cell may still use the newly idle GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from param_importance_nlp.runtime import load_committed_task_artifact


APPROVED_GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)
EXCLUDED_GPU_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
EXCLUDED_PCI = "0000:50:00.0"
RUN_NAME = "formal-r20-g3-v5"
EXPECTED_CANDIDATE_SAMPLE_COUNTS = (131072, 262144)
EXPECTED_BLOCK_SIZE = 32
EXPECTED_SEGMENT_START_POSITION = 81920
EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE = 344064


@dataclass(frozen=True, slots=True)
class _SizingBinding:
    candidate_sizes: tuple[int, int]
    amendment_ref: str
    amendment_artifact_hash: str
    block_size: int
    segment_start_position: int
    segment_end_position_exclusive: int
    convergence_tolerance: float
    normalized_l1_threshold: float
    required_consecutive: int
    complete_all_candidates: bool
    optional_stopping: bool
    resume_ref: str | None
    reuse_prior_sizing_prefix: bool


def _hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical_artifact_hash(value: Mapping[str, Any]) -> str:
    """Hash the canonical JSON wire bytes used by committed task artifacts."""

    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalize_candidate_sizes(values: Sequence[int]) -> tuple[int, int]:
    """Require an explicit, strictly increasing two-node sizing contract."""

    try:
        normalized = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError("candidate sizes must be positive integers") from error
    if len(normalized) != 2 or any(value <= 0 for value in normalized):
        raise ValueError("r20 queue requires exactly two positive --candidate-sizes values")
    if normalized[0] >= normalized[1]:
        raise ValueError("--candidate-sizes must be strictly increasing")
    return normalized


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be loaded: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _load_hash_bound_artifact(path: Path, *, label: str) -> dict[str, Any]:
    value = _load_mapping(path, label=label)
    return _verify_hash_bound_mapping(value, label=label)


def _verify_hash_bound_mapping(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if not isinstance(declared, str) or _canonical_artifact_hash(body) != declared:
        raise ValueError(f"{label} artifact_hash is invalid")
    return dict(value)


def _resolve_data_ref(data_root: Path, raw_ref: object, *, label: str) -> Path:
    if not isinstance(raw_ref, str) or not raw_ref:
        raise ValueError(f"{label} must be a non-empty workspace-relative ref")
    root = data_root.resolve()
    candidate = Path(raw_ref)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes data root: {raw_ref!r}") from error
    return resolved


def _candidate_sizes_from_bound_plans(
    data_root: Path,
    environments: Mapping[str, Path],
    requested: Sequence[int],
) -> _SizingBinding:
    """Cross-check CLI nodes against every bound plan and its amendment.

    The CLI is only a selector.  A queue may proceed only when every formal
    environment's committed sizing plan and its round amendment agree with
    the selector, so an old default cannot silently become an r23 launch.
    """

    candidate_sizes = _normalize_candidate_sizes(requested)
    observed: set[tuple[int, int]] = set()
    amendment_refs: set[str] = set()
    amendment_hashes: set[str] = set()
    binding: _SizingBinding | None = None
    for cell_id, environment_path in sorted(environments.items()):
        environment = _load_mapping(environment_path, label=f"runtime environment {cell_id}")
        evidence_refs = environment.get("evidence_refs")
        if not isinstance(evidence_refs, Mapping):
            raise ValueError(f"runtime environment {cell_id} has no evidence_refs")
        sizing_ref = evidence_refs.get("stage2_reference_sizing_plan")
        sizing_path = _resolve_data_ref(
            data_root,
            sizing_ref,
            label=f"{cell_id}.stage2_reference_sizing_plan",
        )
        sizing_commit_ref = sizing_path.relative_to(data_root.resolve()).as_posix()
        try:
            loaded_plan = load_committed_task_artifact(
                data_root,
                sizing_commit_ref,
                require_formal=True,
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"sizing plan {cell_id} TaskArtifact commit is invalid") from error
        if loaded_plan.identity.artifact_kind != "reference_sizing_plan":
            raise ValueError(f"sizing plan {cell_id} TaskArtifact kind is invalid")
        plan = _verify_hash_bound_mapping(loaded_plan.payload, label=f"sizing plan {cell_id}")
        if plan.get("schema_version") != "stage2-reference-sizing-plan-v1":
            raise ValueError(f"sizing plan {cell_id} schema is not formal v1")
        raw_plan_sizes = plan.get("candidate_sample_counts")
        if not isinstance(raw_plan_sizes, list):
            raise ValueError(f"sizing plan {cell_id} has no candidate_sample_counts")
        try:
            plan_sizes = _normalize_candidate_sizes(raw_plan_sizes)
        except ValueError as error:
            raise ValueError(f"sizing plan {cell_id} candidate_sample_counts invalid") from error
        if plan_sizes != candidate_sizes:
            raise ValueError(
                f"candidate sizes drift for {cell_id}: CLI={list(candidate_sizes)} plan={list(plan_sizes)}"
            )
        plan_contract = (
            plan.get("convergence_tolerance"),
            plan.get("required_consecutive"),
            plan.get("require_terminal_convergence"),
        )
        expected_plan_contract = (0.02, 1, True)
        if plan_contract != expected_plan_contract:
            raise ValueError(
                f"sizing plan {cell_id} convergence contract drift: "
                f"observed={plan_contract} expected={expected_plan_contract}"
            )
        if plan.get("block_size") != EXPECTED_BLOCK_SIZE:
            raise ValueError(f"sizing plan {cell_id} block_size is not {EXPECTED_BLOCK_SIZE}")
        plan_segment = (
            plan.get("draw_start_position"),
            plan.get("draw_end_position_exclusive"),
            plan.get("final_stream_start_position"),
            plan.get("final_stream_end_position_exclusive"),
        )
        expected_segment = (
            EXPECTED_SEGMENT_START_POSITION,
            EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE,
            EXPECTED_SEGMENT_START_POSITION,
            EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE,
        )
        if plan_segment != expected_segment:
            raise ValueError(
                f"sizing plan {cell_id} segment drift: observed={plan_segment} expected={expected_segment}"
            )
        amendment_ref = plan.get("round_manifest_ref")
        amendment_path = _resolve_data_ref(
            data_root,
            amendment_ref,
            label=f"{cell_id}.round_manifest_ref",
        )
        amendment_relative_ref = amendment_path.relative_to(data_root.resolve()).as_posix()
        amendment_refs.add(amendment_relative_ref)
        amendment = _load_hash_bound_artifact(amendment_path, label=f"round amendment {cell_id}")
        declared_amendment_hash = amendment.get("artifact_hash")
        if not isinstance(declared_amendment_hash, str):
            raise ValueError(f"round amendment {cell_id} artifact_hash is missing")
        amendment_hashes.add(declared_amendment_hash)
        if amendment.get("schema_version") != "stage2-reference-sizing-amendment-v1":
            raise ValueError(f"round amendment {cell_id} schema is not amendment v1")
        if amendment.get("round_id") != "r23" or amendment.get("amendment_id") != "r23-amend-r1":
            raise ValueError(f"round amendment {cell_id} is not the r23-amend-r1 contract")
        sizing = amendment.get("sizing")
        if not isinstance(sizing, Mapping):
            raise ValueError(f"round amendment {cell_id} has no sizing object")
        amendment_contract = (
            sizing.get("normalized_l1_threshold"),
            sizing.get("required_consecutive"),
            sizing.get("complete_all_candidates"),
            sizing.get("optional_stopping"),
            sizing.get("block_size"),
            sizing.get("resume_ref"),
            sizing.get("reuse_prior_sizing_prefix"),
        )
        expected_amendment_contract = (0.02, 1, True, False, EXPECTED_BLOCK_SIZE, None, False)
        if amendment_contract != expected_amendment_contract:
            raise ValueError(
                f"round amendment {cell_id} sizing contract drift: "
                f"observed={amendment_contract} expected={expected_amendment_contract}"
            )
        amendment_segment = (
            sizing.get("segment_start_position"),
            sizing.get("segment_end_position_exclusive"),
            sizing.get("prior_consumed_end_position"),
        )
        expected_amendment_segment = (
            EXPECTED_SEGMENT_START_POSITION,
            EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE,
            EXPECTED_SEGMENT_START_POSITION,
        )
        if amendment_segment != expected_amendment_segment:
            raise ValueError(
                f"round amendment {cell_id} segment drift: observed={amendment_segment} "
                f"expected={expected_amendment_segment}"
            )
        amendment_streams = sizing.get("final_stream_segments")
        if not isinstance(amendment_streams, Mapping):
            raise ValueError(f"round amendment {cell_id} final_stream_segments are missing")
        for stream_name in ("reference_A", "reference_B"):
            stream_segment = amendment_streams.get(stream_name)
            if not isinstance(stream_segment, Mapping) or (
                stream_segment.get("start_position") != EXPECTED_SEGMENT_START_POSITION
                or stream_segment.get("end_position_exclusive") != EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE
            ):
                raise ValueError(f"round amendment {cell_id} {stream_name} segment drift")
        raw_amendment_sizes = sizing.get("candidate_sample_counts")
        if not isinstance(raw_amendment_sizes, list):
            raise ValueError(f"round amendment {cell_id} has no candidate_sample_counts")
        try:
            amendment_sizes = _normalize_candidate_sizes(raw_amendment_sizes)
        except ValueError as error:
            raise ValueError(f"round amendment {cell_id} candidate_sample_counts invalid") from error
        if amendment_sizes != EXPECTED_CANDIDATE_SAMPLE_COUNTS:
            raise ValueError(
                f"round amendment {cell_id} candidate_sample_counts are not "
                f"{list(EXPECTED_CANDIDATE_SAMPLE_COUNTS)}"
            )
        if amendment_sizes != plan_sizes:
            raise ValueError(
                f"sizing plan/amendment drift for {cell_id}: plan={list(plan_sizes)} "
                f"amendment={list(amendment_sizes)}"
            )
        current_binding = _SizingBinding(
            candidate_sizes=plan_sizes,
            amendment_ref=amendment_relative_ref,
            amendment_artifact_hash=declared_amendment_hash,
            block_size=EXPECTED_BLOCK_SIZE,
            segment_start_position=EXPECTED_SEGMENT_START_POSITION,
            segment_end_position_exclusive=EXPECTED_SEGMENT_END_POSITION_EXCLUSIVE,
            convergence_tolerance=float(plan_contract[0]),
            normalized_l1_threshold=float(amendment_contract[0]),
            required_consecutive=int(plan_contract[1]),
            complete_all_candidates=bool(amendment_contract[2]),
            optional_stopping=bool(amendment_contract[3]),
            resume_ref=amendment_contract[5],
            reuse_prior_sizing_prefix=bool(amendment_contract[6]),
        )
        if binding is not None and current_binding != binding:
            raise ValueError(f"six-cell sizing bindings disagree for {cell_id}")
        binding = current_binding
        observed.add(plan_sizes)
    if len(observed) != 1:
        raise ValueError(f"six-cell sizing plans disagree: {sorted(observed)}")
    if len(amendment_refs) != 1:
        raise ValueError(f"six-cell sizing plans disagree on round_manifest_ref: {sorted(amendment_refs)}")
    if len(amendment_hashes) != 1:
        raise ValueError(f"six-cell sizing plans disagree on amendment artifact_hash: {sorted(amendment_hashes)}")
    if binding is None:
        raise ValueError("six-cell sizing plans are missing")
    return binding


def _queue_manifest(
    *,
    run_id: str,
    execution_commit: str,
    sizing_binding: _SizingBinding,
    order: Sequence[str],
    estimates: Mapping[str, float],
    cells: Mapping[str, Path],
    python: str,
    output_root: Path,
) -> dict[str, Any]:
    """Build the immutable queue identity, including the sizing selector."""

    return {
        "schema_version": "stage2-s204-r20-queue-v1",
        "run_name": RUN_NAME,
        "run_id": run_id,
        "execution_commit": execution_commit,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "candidate_sample_counts": list(sizing_binding.candidate_sizes),
        "round_amendment_ref": sizing_binding.amendment_ref,
        "round_amendment_artifact_hash": sizing_binding.amendment_artifact_hash,
        "block_size": sizing_binding.block_size,
        "segment_start_position": sizing_binding.segment_start_position,
        "segment_end_position_exclusive": sizing_binding.segment_end_position_exclusive,
        "convergence_tolerance": sizing_binding.convergence_tolerance,
        "normalized_l1_threshold": sizing_binding.normalized_l1_threshold,
        "required_consecutive": sizing_binding.required_consecutive,
        "complete_all_candidates": sizing_binding.complete_all_candidates,
        "optional_stopping": sizing_binding.optional_stopping,
        "resume_ref": sizing_binding.resume_ref,
        "reuse_prior_sizing_prefix": sizing_binding.reuse_prior_sizing_prefix,
        "cell_order_lpt": list(order),
        "cell_estimates_seconds": dict(estimates),
        "cell_configs": {cell: path.as_posix() for cell, path in cells.items()},
        "python": python,
        "retry_policy": "none",
        "output_root": output_root.as_posix(),
    }


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


def _absolute_without_resolving(path: str | os.PathLike[str]) -> str:
    """Make a launcher path absolute while preserving symlink spelling."""

    return os.path.abspath(os.fspath(path))


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
    candidate_sizes: Sequence[int],
    heartbeat_seconds: float,
) -> list[str]:
    if gpu_uuid not in APPROVED_GPU_UUIDS or gpu_uuid == EXCLUDED_GPU_UUID:
        raise ValueError(f"GPU is not an approved r20 UUID: {gpu_uuid}")
    normalized_candidate_sizes = _normalize_candidate_sizes(candidate_sizes)
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
        "--candidate-sizes",
        *(str(value) for value in normalized_candidate_sizes),
        "--block-size",
        str(EXPECTED_BLOCK_SIZE),
        "--execute",
    ]


def run_queue(args: argparse.Namespace) -> int:
    if len(set(APPROVED_GPU_UUIDS)) != 4 or EXCLUDED_GPU_UUID in APPROVED_GPU_UUIDS:
        raise RuntimeError("R20_APPROVED_GPU_SET_INVALID")
    if len(args.execution_commit) != 40 or any(c not in "0123456789abcdef" for c in args.execution_commit):
        raise ValueError("--execution-commit must be 40 lowercase hexadecimal characters")
    cells = _parse_cell_config(args.cell_config)
    environments = _parse_cell_environment(args.runtime_environment, cells)
    sizing_binding = _candidate_sizes_from_bound_plans(
        Path(args.data_root).resolve(),
        environments,
        args.candidate_sizes,
    )
    candidate_sizes = sizing_binding.candidate_sizes
    estimates = _parse_estimates(args.cell_estimate, cells)
    order = lpt_order(estimates)
    # Do not use Path.resolve(): the formal venv's ``python`` may be a
    # symlink, and resolving it can escape the venv into a system interpreter
    # without the approved dependencies.  The lexical absolute path is also
    # recorded in the manifest and each immutable cell PID command.
    python = _absolute_without_resolving(args.python)
    launcher = Path(args.s204_launcher).resolve()
    output_root = Path(args.output_root).resolve()
    if RUN_NAME not in output_root.parts:
        raise ValueError(f"r20 output root must be under a {RUN_NAME} directory")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_root = Path(args.queue_root).resolve() / RUN_NAME / run_id
    queue_root.mkdir(parents=True, exist_ok=False)
    event_path = queue_root / "queue-events.jsonl"
    manifest = _queue_manifest(
        run_id=run_id,
        execution_commit=args.execution_commit,
        sizing_binding=sizing_binding,
        order=order,
        estimates=estimates,
        cells=cells,
        python=python,
        output_root=output_root,
    )
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
                    candidate_sizes=candidate_sizes,
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
    parser.add_argument(
        "--candidate-sizes",
        type=int,
        nargs="+",
        required=True,
        help="exactly two explicit sizing nodes; must match every bound plan and amendment",
    )
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
