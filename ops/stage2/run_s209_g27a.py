"""Detached production launcher for S2.9/G2.7a profiler validation.

``--prepare`` creates only a frozen measurement plan after binding G2.4b and
the sealed S2.7 manifest.  ``--preflight`` re-reads every producer artifact,
GPU inventory, and I/O evidence.  ``--execute`` is the sole mode allowed to
start a profiler worker; the worker command must emit actual measured rows.
For ``scientific_equal_sample_cost``, one invocation receives
``S29_METHOD=shared`` and must emit a
``stage2-s209-g27a-shared-paired-run-v1`` envelope containing all three
methods and the hash-bound shared-pool artifact.
``--detach``/``--status``/``--wait`` are control-plane operations and never
invent a result when a profiler or prerequisite is missing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

_LAUNCHER_BOOT_ROOT = Path(__file__).resolve().parents[2]
if str(_LAUNCHER_BOOT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_BOOT_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_s209_runner import (
    S29_ATTEMPT_SCHEMA,
    S29_EXECUTION_IDENTITY_SCHEMA,
    S29RunnerBlocked,
    S29ProfilerRunner,
    S29StatusStore,
    _validate_execution_identity,
    load_s209_preflight,
    prepare_s209_plan,
    subprocess_profiler_executor,
)

S29_DETACHED_ATTEMPT_SCHEMA = "stage2-s209-detached-attempt-v2"
S29_DETACHED_FAILURE_SCHEMA = "stage2-s209-detached-failure-v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S29RunnerBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    parts = reference.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE")
    root = _checked_directory(root, field=f"{field}.root")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise S29RunnerBlocked(f"{field}:SYMLINK_COMPONENT_FORBIDDEN")
    path = current.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE") from error
    return path


def _checked_directory(value: Path | str, *, field: str) -> Path:
    """Resolve a directory only after rejecting every lexical symlink."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink():
        raise S29RunnerBlocked(f"{field}:SYMLINK_COMPONENT_FORBIDDEN")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise S29RunnerBlocked(f"{field}:SYMLINK_COMPONENT_FORBIDDEN")
    if not candidate.exists() or not candidate.is_dir():
        raise S29RunnerBlocked(f"{field}:DIRECTORY_REQUIRED")
    return candidate.resolve()


def _data_root(value: Path | str) -> Path:
    return _checked_directory(value, field="data_root")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(repository: Path | str | None) -> Path:
    if repository is None:
        raise S29RunnerBlocked("REPOSITORY_REQUIRED")
    candidate = Path(repository)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists() or not candidate.is_dir():
        raise S29RunnerBlocked("REPOSITORY_DIRECTORY_REQUIRED")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise S29RunnerBlocked("REPOSITORY_SYMLINK_COMPONENT_FORBIDDEN")
    resolved = candidate.resolve()
    try:
        top_level = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise S29RunnerBlocked("REPOSITORY_GIT_TOP_LEVEL_UNAVAILABLE") from error
    if not top_level or Path(top_level).resolve() != resolved:
        raise S29RunnerBlocked("REPOSITORY_GIT_TOP_LEVEL_MISMATCH")
    return resolved


def _execution_identity(command: Sequence[str], repository: Path | str | None = None) -> dict[str, Any]:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
    repo = _repository_path(repository)
    launcher = Path(__file__).absolute()
    try:
        launcher_ref = launcher.relative_to(repo).as_posix()
    except ValueError as error:
        raise S29RunnerBlocked("LAUNCHER_SOURCE_OUTSIDE_REPOSITORY") from error
    launcher = _logical(repo, launcher_ref, field="launcher_source")
    if not launcher.is_file():
        raise S29RunnerBlocked("LAUNCHER_SOURCE_REGULAR_FILE_REQUIRED")
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise S29RunnerBlocked("REPOSITORY_IDENTITY_UNAVAILABLE") from error
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head) or dirty:
        raise S29RunnerBlocked("REPOSITORY_NOT_CLEAN_OR_HEAD_INVALID")
    body: dict[str, Any] = {
        "schema_version": S29_EXECUTION_IDENTITY_SCHEMA,
        "repository_head": head,
        "launcher_source_sha256": _file_sha256(launcher),
        "profiler_command_hash": canonical_json_hash(list(command)),
        "repository_clean": True,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return _validate_execution_identity(body)


def _verify_hash(payload: Mapping[str, Any], *, field: str) -> str:
    declared = payload.get("artifact_hash")
    if not isinstance(declared, str) or len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
        raise S29RunnerBlocked(f"{field}:ARTIFACT_HASH_REQUIRED")
    if canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"}) != declared:
        raise S29RunnerBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _formal_anchor(root: Path, reference: str | None, *, field: str, filename: str) -> tuple[dict[str, Any], Path]:
    if not isinstance(reference, str) or not reference or "#" in reference or not reference.endswith(filename):
        raise S29RunnerBlocked(f"{field}:REFERENCE_REQUIRED")
    path = _logical(root, reference, field=field)
    if path.name != filename or not path.is_file():
        raise S29RunnerBlocked(f"{field}:REFERENCE_NOT_FOUND")
    value = _load_optional(root, reference, field=field)
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked(f"{field}:OBJECT_REQUIRED")
    payload = dict(value)
    _verify_hash(payload, field=field)
    return payload, path


def _load_optional(root: Path, reference: str | None, *, field: str) -> Any:
    if reference is None:
        return None
    path = _logical(root, reference, field=field)
    value = load_canonical_json(path)
    return value


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    """Publish one detached manifest without overwriting prior provenance."""

    if path.is_symlink():
        raise S29RunnerBlocked(f"{field}:SYMLINK_OUTPUT_FORBIDDEN")
    if path.exists():
        try:
            existing = load_canonical_json(path)
        except Exception as error:
            raise S29RunnerBlocked(f"{field}:INVALID_EXISTING") from error
        if isinstance(existing, Mapping) and dict(existing) == dict(value):
            return
        raise S29RunnerBlocked(f"{field}:IMMUTABLE_OUTPUT_EXISTS")
    write_canonical_json(path, value)


def _attempt_identity(
    *,
    run_id: str,
    run_root_ref: str,
    identity: Mapping[str, Any],
    attempt_id: str,
    child_argv_hash: str,
    parent_pid: int,
) -> dict[str, Any]:
    log_ref = f"{run_root_ref.rstrip('/')}/attempts/{attempt_id}/launcher.log"
    body: dict[str, Any] = {
        "schema_version": S29_DETACHED_ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "run_id": run_id,
        "run_root_ref": run_root_ref,
        "parent_pid": parent_pid,
        "repository_head": identity["repository_head"],
        "launcher_source_sha256": identity["launcher_source_sha256"],
        "profiler_command_hash": identity["profiler_command_hash"],
        "execution_identity_hash": identity["artifact_hash"],
        "child_argv_hash": child_argv_hash,
        "log_ref": log_ref,
        "claimed_at": _now(),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _attempt_failure(attempt: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": S29_DETACHED_FAILURE_SCHEMA,
        "attempt": dict(attempt),
        "reason": reason,
        "failed_at": _now(),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _validate_attempt_receipt(
    value: Any,
    *,
    run_id: str,
    identity: Mapping[str, Any],
    running: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_RECEIPT_INVALID")
    payload = dict(value)
    required = {
        "schema_version", "attempt_id", "run_id", "run_root_ref", "parent_pid",
        "repository_head", "launcher_source_sha256", "profiler_command_hash",
        "execution_identity_hash", "child_argv_hash", "log_ref", "claimed_at", "artifact_hash",
    }
    if running:
        required.update({"status", "child_pid", "started_at"})
    if set(payload) != required or payload.get("schema_version") != S29_DETACHED_ATTEMPT_SCHEMA:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_RECEIPT_SCHEMA_INVALID")
    if running and payload.get("status") != "RUNNING":
        raise S29RunnerBlocked("DETACHED_ATTEMPT_RUNNING_STATUS_INVALID")
    if payload.get("run_id") != run_id:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_RUN_ID_DRIFT")
    for name in ("attempt_id", "run_root_ref", "log_ref", "claimed_at"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise S29RunnerBlocked(f"DETACHED_ATTEMPT_{name.upper()}_INVALID")
    log_ref = payload["log_ref"]
    if "\\" in log_ref or not log_ref.endswith(f"/attempts/{payload['attempt_id']}/launcher.log"):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_LOG_REF_INVALID")
    try:
        log_parts = PurePosixPath(log_ref).parts
    except (TypeError, ValueError) as error:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_LOG_REF_INVALID") from error
    if PurePosixPath(log_ref).is_absolute() or any(part in {"", ".", ".."} for part in log_parts):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_LOG_REF_INVALID")
    if isinstance(payload.get("parent_pid"), bool) or not isinstance(payload.get("parent_pid"), int) or payload["parent_pid"] <= 0:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_PARENT_PID_INVALID")
    if running and (isinstance(payload.get("child_pid"), bool) or not isinstance(payload.get("child_pid"), int) or payload["child_pid"] <= 0):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_CHILD_PID_INVALID")
    for name in ("repository_head",):
        if not isinstance(payload.get(name), str) or len(payload[name]) != 40 or any(char not in "0123456789abcdef" for char in payload[name]):
            raise S29RunnerBlocked("DETACHED_ATTEMPT_REPOSITORY_HEAD_INVALID")
    for name in ("launcher_source_sha256", "profiler_command_hash", "execution_identity_hash", "child_argv_hash", "artifact_hash"):
        if not isinstance(payload.get(name), str) or len(payload[name]) != 64 or any(char not in "0123456789abcdef" for char in payload[name]):
            raise S29RunnerBlocked(f"DETACHED_ATTEMPT_{name.upper()}_INVALID")
    if payload["repository_head"] != identity["repository_head"] or payload["launcher_source_sha256"] != identity["launcher_source_sha256"] or payload["profiler_command_hash"] != identity["profiler_command_hash"] or payload["execution_identity_hash"] != identity["artifact_hash"]:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_EXECUTION_IDENTITY_DRIFT")
    if canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"}) != payload["artifact_hash"]:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_RECEIPT_HASH_MISMATCH")
    return payload


def _validate_attempt_failure(
    value: Any,
    *,
    run_id: str,
    identity: Mapping[str, Any],
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_INVALID")
    payload = dict(value)
    required = {"schema_version", "attempt", "reason", "failed_at", "artifact_hash"}
    if set(payload) != required or payload.get("schema_version") != S29_DETACHED_FAILURE_SCHEMA:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_SCHEMA_INVALID")
    if not isinstance(payload.get("reason"), str) or not payload["reason"]:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_REASON_INVALID")
    if not isinstance(payload.get("failed_at"), str) or not payload["failed_at"]:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_TIME_INVALID")
    nested = _validate_attempt_receipt(payload.get("attempt"), run_id=run_id, identity=identity)
    if receipt is not None and nested != dict(receipt):
        raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_RECEIPT_DRIFT")
    _verify_hash(payload, field="detached_attempt.failure")
    return payload


def _reject_legacy_launch_manifests(run_root: Path) -> None:
    """Reject pre-v2 mutex files instead of reviving their stale-lock semantics."""

    for name in ("launcher.lease.json", "launcher.pid.json"):
        path = run_root / name
        if path.is_symlink() or path.exists():
            raise S29RunnerBlocked("DETACHED_LEGACY_LAUNCH_MANIFEST_FORBIDDEN")


def _validate_running_lineage(
    receipt: Mapping[str, Any],
    running: Mapping[str, Any],
) -> None:
    """Require running to extend the exact immutable claim receipt."""

    for key, value in receipt.items():
        if key == "artifact_hash":
            continue
        if running.get(key) != value:
            raise S29RunnerBlocked("DETACHED_ATTEMPT_RUNNING_RECEIPT_DRIFT")


def _launch_attempts(run_root: Path, *, run_id: str, identity: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    _reject_legacy_launch_manifests(run_root)
    attempts_root = run_root / "attempts"
    if attempts_root.exists() and attempts_root.is_symlink():
        raise S29RunnerBlocked("DETACHED_ATTEMPTS_SYMLINK_FORBIDDEN")
    if not attempts_root.exists():
        return []
    result: list[tuple[Path, dict[str, Any]]] = []
    for entry in sorted(attempts_root.iterdir(), key=lambda path: path.name):
        if entry.is_symlink():
            raise S29RunnerBlocked("DETACHED_ATTEMPT_SYMLINK_FORBIDDEN")
        if entry.is_file():
            # S2.9 measurement attempts share this directory; validate their
            # schema rather than interpreting them as detached launch receipts.
            try:
                value = load_canonical_json(entry)
            except Exception as error:
                raise S29RunnerBlocked("DETACHED_ATTEMPTS_UNKNOWN_ARTIFACT") from error
            if not isinstance(value, Mapping) or value.get("schema_version") != S29_ATTEMPT_SCHEMA:
                raise S29RunnerBlocked("DETACHED_ATTEMPTS_UNKNOWN_ARTIFACT")
            _verify_hash(value, field="detached_measurement_attempt")
            if value.get("run_id") != run_id:
                raise S29RunnerBlocked("DETACHED_MEASUREMENT_ATTEMPT_RUN_ID_DRIFT")
            for name, expected in (
                ("repository_head", identity["repository_head"]),
                ("launcher_source_sha256", identity["launcher_source_sha256"]),
                ("profiler_command_hash", identity["profiler_command_hash"]),
                ("execution_identity_hash", identity["artifact_hash"]),
            ):
                if value.get(name) != expected:
                    raise S29RunnerBlocked("DETACHED_MEASUREMENT_ATTEMPT_EXECUTION_IDENTITY_DRIFT")
            continue
        if not entry.is_dir():
            raise S29RunnerBlocked("DETACHED_ATTEMPTS_UNKNOWN_ARTIFACT")
        for child in entry.iterdir():
            if child.is_symlink():
                raise S29RunnerBlocked("DETACHED_ATTEMPT_SYMLINK_FORBIDDEN")
            if child.name not in {"launch-receipt.json", "launch-running.json", "launch-failure.json"}:
                raise S29RunnerBlocked("DETACHED_ATTEMPT_UNKNOWN_ARTIFACT")
        receipt_path = entry / "launch-receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise S29RunnerBlocked("DETACHED_ATTEMPT_RECEIPT_MISSING")
        receipt = _validate_attempt_receipt(load_canonical_json(receipt_path), run_id=run_id, identity=identity)
        running_path = entry / "launch-running.json"
        if running_path.exists():
            if not running_path.is_file():
                raise S29RunnerBlocked("DETACHED_ATTEMPT_RUNNING_INVALID")
            running_value = _validate_attempt_receipt(
                load_canonical_json(running_path),
                run_id=run_id,
                identity=identity,
                running=True,
            )
            _validate_running_lineage(receipt, running_value)
        failure_path = entry / "launch-failure.json"
        if failure_path.exists():
            if not failure_path.is_file():
                raise S29RunnerBlocked("DETACHED_ATTEMPT_FAILURE_INVALID")
            _validate_attempt_failure(
                load_canonical_json(failure_path),
                run_id=run_id,
                identity=identity,
                receipt=receipt,
            )
        result.append((entry, receipt))
    return result


def _claim_detached_attempt(
    run_root: Path,
    *,
    run_id: str,
    run_root_ref: str,
    identity: Mapping[str, Any],
    child_argv_hash: str,
    attempt_id: str,
) -> tuple[Path, dict[str, Any]]:
    for entry, receipt in _launch_attempts(run_root, run_id=run_id, identity=identity):
        if receipt["run_root_ref"] != run_root_ref:
            raise S29RunnerBlocked("DETACHED_ATTEMPT_RUN_ROOT_DRIFT")
        running_path = entry / "launch-running.json"
        if running_path.is_symlink():
            raise S29RunnerBlocked("DETACHED_ATTEMPT_RUNNING_SYMLINK_FORBIDDEN")
        if running_path.exists():
            running = _validate_attempt_receipt(load_canonical_json(running_path), run_id=run_id, identity=identity, running=True)
            if any(running.get(name) != receipt.get(name) for name in ("attempt_id", "run_root_ref", "child_argv_hash", "execution_identity_hash")):
                raise S29RunnerBlocked("DETACHED_ATTEMPT_RUNNING_IDENTITY_DRIFT")
            if running.get("child_argv_hash") != child_argv_hash:
                raise S29RunnerBlocked("DETACHED_ATTEMPT_COMMAND_DRIFT")
            child_pid = running.get("child_pid")
            if not isinstance(child_pid, int) or child_pid <= 0:
                raise S29RunnerBlocked("DETACHED_ATTEMPT_CHILD_PID_INVALID")
            if _pid_alive(child_pid):
                raise S29RunnerBlocked(f"S29_DETACHED_LAUNCH_ALREADY_RUNNING:{child_pid}")
        elif not (entry / "launch-failure.json").exists() and _pid_alive(int(receipt["parent_pid"])):
            raise S29RunnerBlocked(f"S29_DETACHED_LAUNCH_ALREADY_RUNNING:{receipt['parent_pid']}")
    attempts_root = run_root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise S29RunnerBlocked("DETACHED_ATTEMPTS_ROOT_INVALID")
    if not isinstance(attempt_id, str) or not attempt_id.startswith("launch-") or len(attempt_id) != len("launch-") + 32:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_ID_INVALID")
    attempt_dir = attempts_root / attempt_id
    try:
        attempt_dir.mkdir()
    except FileExistsError as error:
        raise S29RunnerBlocked("DETACHED_ATTEMPT_ID_COLLISION") from error
    receipt = _attempt_identity(
        run_id=run_id,
        run_root_ref=run_root_ref,
        identity=identity,
        attempt_id=attempt_id,
        child_argv_hash=child_argv_hash,
        parent_pid=os.getpid(),
    )
    try:
        _write_once(attempt_dir / "launch-receipt.json", receipt, field="detached_attempt.receipt")
    except BaseException:
        failure = _attempt_failure(receipt, reason="CLAIM_RECEIPT_WRITE_FAILED")
        try:
            _write_once(attempt_dir / "launch-failure.json", failure, field="detached_attempt.failure")
        except BaseException:
            pass
        raise
    return attempt_dir, receipt


def _inventory_envelope(root: Path, reference: str) -> Mapping[str, Any]:
    value = _load_optional(root, reference, field="gpu_inventory")
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("GPU_INVENTORY_ENVELOPE_REQUIRED")
    return dict(value)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    identity = _execution_identity(args.profiler_command, args.repository)
    data_root = _data_root(args.data_root)
    preflight = load_s209_preflight(
        data_root=data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    if preflight.measurement_plan.get("run_id") != args.run_id:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    if preflight.execution_identity != identity:
        raise S29RunnerBlocked("EXECUTION_IDENTITY_PLAN_DRIFT")
    return {
        "schema_version": "stage2-s209-g27a-production-preflight-v1",
        "status": "READY",
        "formal_eligible": True,
        "run_id": args.run_id,
        "measurement_plan_hash": preflight.plan_hash,
        "matrix_hash": preflight.frozen.matrix_hash,
        "raw_manifest_hash": preflight.frozen.raw_manifest_hash,
        "raw_run_id": preflight.frozen.raw_run_id,
        "g25_gate_hash": preflight.frozen.g25_gate_hash,
        "approved_gpu_uuids": preflight.inventory["approved_gpu_uuids"],
        "excluded_gpu_uuid": preflight.inventory["excluded_gpu_uuid"],
        "excluded_pci": preflight.inventory["excluded_pci"],
        "inventory_hash": preflight.inventory["inventory_artifact_hash"],
        "inventory_source_sha256": preflight.inventory["inventory_source_sha256"],
        "inventory_source_ref": preflight.inventory["inventory_source_ref"],
        "io_evidence_hash": preflight.io_evidence["artifact_hash"],
        "cost_io_quiescent": preflight.io_evidence["cost_io_quiescent"],
        "capacity_ref": preflight.capacity_ref,
        "capacity_evidence_hash": preflight.capacity_inputs["capacity_evidence_hash"] if preflight.capacity_inputs else None,
        "ulimit_ref": preflight.ulimit_ref,
        "ulimit_evidence_hash": preflight.ulimit_evidence["ulimit_evidence_hash"] if preflight.ulimit_evidence else None,
        "actual_measurements_required": True,
        "four_gpu_anchor_required": True,
        "resumable_terminal_rows": True,
        "execution_identity": identity,
        "repository_ref": str(_repository_path(args.repository)),
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    identity = _execution_identity(args.profiler_command, args.repository)
    root = _data_root(args.data_root)
    inventory = _inventory_envelope(root, args.gpu_inventory_ref) if args.gpu_inventory_ref else None
    io_value = _load_optional(root, args.io_evidence_ref, field="io_evidence") if args.io_evidence_ref else None
    plan = prepare_s209_plan(
        data_root=root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        run_id=args.run_id,
        anchor_ids=tuple(args.anchor_id),
        repetitions=args.repetitions,
        randomization_seed=args.randomization_seed,
        inventory=inventory,
        inventory_ref=args.gpu_inventory_ref,
        io_evidence=io_value,
        output_ref=None,
    )
    plan = dict(plan)
    plan["execution_identity"] = identity
    plan["artifact_hash"] = canonical_json_hash({key: value for key, value in plan.items() if key != "artifact_hash"})
    if args.measurement_plan_ref is not None:
        _write_once(
            _logical(root, args.measurement_plan_ref, field="measurement_plan_output"),
            plan,
            field="measurement_plan",
        )
    return plan


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    identity = _execution_identity(args.profiler_command, args.repository)
    _require_detached_child(args, identity=identity)
    data_root = _data_root(args.data_root)
    preflight = load_s209_preflight(
        data_root=data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    if preflight.measurement_plan.get("run_id") != args.run_id:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    if preflight.execution_identity != identity:
        raise S29RunnerBlocked("EXECUTION_IDENTITY_PLAN_DRIFT")
    root = data_root
    run_root = _logical(root, args.run_root, field="run_root")
    if (args.single_gpu_anchor_ref is None) != (args.four_gpu_anchor_ref is None):
        raise S29RunnerBlocked("ANCHOR_REFERENCE_PAIR_REQUIRED")
    single_anchor: Mapping[str, Any] | None = None
    four_anchor: Mapping[str, Any] | None = None
    if args.single_gpu_anchor_ref is not None:
        single_anchor, single_anchor_path = _formal_anchor(
            root,
            args.single_gpu_anchor_ref,
            field="single_gpu_anchor",
            filename="single-gpu-anchor.json",
        )
        four_anchor, four_anchor_path = _formal_anchor(
            root,
            args.four_gpu_anchor_ref,
            field="four_gpu_anchor",
            filename="four-gpu-anchor.json",
        )
        if single_anchor_path.parent.parent != run_root or four_anchor_path.parent.parent != run_root:
            raise S29RunnerBlocked("ANCHOR_REFERENCE_RUN_ROOT_MISMATCH")
        candidate_ref = f"{args.run_root.rstrip('/')}/anchors/four-gpu-stage1-numeric.json"
        candidate_path = _logical(root, candidate_ref, field="four_gpu_numeric_sidecar")
        candidate = _load_optional(root, candidate_ref, field="four_gpu_numeric_sidecar")
        if not isinstance(candidate, Mapping) or not candidate_path.is_file():
            raise S29RunnerBlocked("FOUR_GPU_NUMERIC_SIDECAR_REQUIRED")
        if four_anchor.get("stage1_numeric_artifact") != dict(candidate):
            raise S29RunnerBlocked("FOUR_GPU_NUMERIC_SIDECAR_DRIFT")
        if four_anchor.get("stage1_numeric_artifact_ref") != candidate_path.relative_to(root).as_posix():
            raise S29RunnerBlocked("FOUR_GPU_NUMERIC_SIDECAR_REF_DRIFT")
    accuracy = _load_optional(root, args.accuracy_ref, field="accuracy") if args.accuracy_ref else []
    if isinstance(accuracy, Mapping):
        accuracy = accuracy.get("rows", accuracy.get("accuracy_rows", []))
    if not isinstance(accuracy, list) or not all(isinstance(item, Mapping) for item in accuracy):
        raise S29RunnerBlocked("ACCURACY_ROWS_INVALID")
    profiler = subprocess_profiler_executor(args.profiler_command)
    runner = S29ProfilerRunner(
        preflight=preflight,
        run_id=args.run_id,
        run_root=_logical(root, args.run_root, field="run_root"),
        profiler=profiler,
        single_gpu_anchor=single_anchor if isinstance(single_anchor, Mapping) else None,
        four_gpu_anchor=four_anchor if isinstance(four_anchor, Mapping) else None,
        shared_attribution_cross_check=(
            _load_optional(root, args.crosscheck_ref, field="crosscheck")
            if args.crosscheck_ref else None
        ),
        accuracy_rows=[dict(item) for item in accuracy],
        capacity_inputs=dict(preflight.capacity_inputs or {}),
        execution_identity=identity,
    )
    return runner.run()


def _detach(args: argparse.Namespace) -> dict[str, Any]:
    if not args.profiler_command:
        raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
    # No detached child may exist until all immutable input, inventory, I/O,
    # capacity, and ulimit evidence has passed the same launch-time preflight.
    preflight = _preflight(args)
    identity = preflight["execution_identity"]
    root = _data_root(args.data_root)
    run_root = _logical(root, args.run_root, field="run_root")
    run_root_ref = run_root.relative_to(root).as_posix()
    run_root.mkdir(parents=True, exist_ok=True)
    # The detached child is the actual executor.  Replacing the action token
    # (rather than dropping it) is required because the child parser demands
    # exactly one mutually-exclusive action.
    child = [str(item) for item in sys.argv[1:]]
    try:
        action_index = child.index("--detach")
    except ValueError as error:
        raise S29RunnerBlocked("DETACH_ACTION_NOT_FOUND") from error
    # Replace only the launcher action.  A profiler command may legitimately
    # carry its own ``--detach`` argument; rewriting every matching token
    # would silently change the worker's frozen command line.
    child[action_index] = "--execute"
    attempt_id = f"launch-{uuid.uuid4().hex}"
    child.extend(["--detached-child-marker", attempt_id, "--attempt-id", attempt_id])
    child_argv_hash = canonical_json_hash(child)
    attempt_dir, receipt = _claim_detached_attempt(
        run_root,
        run_id=args.run_id,
        run_root_ref=run_root_ref,
        identity=identity,
        child_argv_hash=child_argv_hash,
        attempt_id=attempt_id,
    )
    try:
        log_path = attempt_dir / "launcher.log"
        if log_path.is_symlink():
            raise S29RunnerBlocked("DETACHED_ATTEMPT_LOG_SYMLINK_FORBIDDEN")
        with log_path.open("ab") as handle:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *child],
                cwd=_repository_path(args.repository),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException as error:
        failure = _attempt_failure(receipt, reason=f"SPAWN_FAILED:{type(error).__name__}:{error}")
        try:
            _write_once(attempt_dir / "launch-failure.json", failure, field="detached_attempt.failure")
        except BaseException:
            pass
        raise
    running = dict(receipt)
    running.update(
        {
            "status": "RUNNING",
            "child_pid": int(process.pid),
            "started_at": _now(),
        }
    )
    running["artifact_hash"] = canonical_json_hash({key: value for key, value in running.items() if key != "artifact_hash"})
    try:
        _write_once(attempt_dir / "launch-running.json", running, field="detached_attempt.running")
    except BaseException as error:
        try:
            process.terminate()
        finally:
            try:
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                finally:
                    process.wait(timeout=10)
        failure = _attempt_failure(receipt, reason=f"RUNNING_RECEIPT_WRITE_FAILED:{type(error).__name__}:{error}")
        try:
            _write_once(attempt_dir / "launch-failure.json", failure, field="detached_attempt.failure")
        except BaseException:
            pass
        raise
    # The CLI result is the persisted running receipt itself.  Returning a
    # separately-shaped object (or reusing the receipt hash for another body)
    # would make the handoff unverifiable after stdout is lost.
    return dict(running)


def _require_detached_child(args: argparse.Namespace, *, identity: Mapping[str, Any]) -> None:
    marker = getattr(args, "detached_child_marker", None)
    attempt_id = getattr(args, "attempt_id", None)
    if not isinstance(marker, str) or marker != attempt_id:
        raise S29RunnerBlocked("DETACHED_CHILD_MARKER_REQUIRED")
    root = _data_root(args.data_root)
    run_root = _logical(root, args.run_root, field="run_root")
    _launch_attempts(run_root, run_id=args.run_id, identity=identity)
    attempt_dir = run_root / "attempts" / marker
    if attempt_dir.is_symlink() or not attempt_dir.is_dir():
        raise S29RunnerBlocked("DETACHED_CHILD_ATTEMPT_NOT_FOUND")
    receipt_path = attempt_dir / "launch-receipt.json"
    running_path = attempt_dir / "launch-running.json"
    if receipt_path.is_symlink() or running_path.is_symlink() or not receipt_path.is_file() or not running_path.is_file():
        raise S29RunnerBlocked("DETACHED_CHILD_RECEIPT_MISSING")
    receipt = _validate_attempt_receipt(load_canonical_json(receipt_path), run_id=args.run_id, identity=identity)
    running = _validate_attempt_receipt(load_canonical_json(running_path), run_id=args.run_id, identity=identity, running=True)
    expected_run_root_ref = run_root.relative_to(root).as_posix()
    if receipt["run_root_ref"] != expected_run_root_ref or running["run_root_ref"] != expected_run_root_ref:
        raise S29RunnerBlocked("DETACHED_CHILD_RUN_ROOT_DRIFT")
    if running["attempt_id"] != receipt["attempt_id"] or running["child_argv_hash"] != receipt["child_argv_hash"]:
        raise S29RunnerBlocked("DETACHED_CHILD_RECEIPT_IDENTITY_DRIFT")
    _validate_running_lineage(receipt, running)
    if running["child_pid"] != os.getpid():
        raise S29RunnerBlocked("DETACHED_CHILD_PID_IDENTITY_MISMATCH")
    if canonical_json_hash(list(sys.argv[1:])) != receipt["child_argv_hash"]:
        raise S29RunnerBlocked("DETACHED_CHILD_COMMAND_DRIFT")


def _status(args: argparse.Namespace, *, wait: bool) -> int:
    identity = _execution_identity(args.profiler_command, args.repository)
    # Status/replay consumers must reject inventory or frozen-cost identity drift
    # before trusting an existing detached status file.
    data_root = _data_root(args.data_root)
    preflight = load_s209_preflight(
        data_root=data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    if preflight.measurement_plan.get("run_id") != args.run_id:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    if preflight.execution_identity != identity:
        raise S29RunnerBlocked("EXECUTION_IDENTITY_PLAN_DRIFT")
    inventory_identity = preflight.inventory.get("inventory_identity")
    if not isinstance(inventory_identity, Mapping):
        raise S29RunnerBlocked("STATUS_INVENTORY_IDENTITY_MISSING")
    if (
        inventory_identity.get("artifact_hash") != preflight.inventory.get("inventory_artifact_hash")
        or inventory_identity.get("source_sha256") != preflight.inventory.get("inventory_source_sha256")
    ):
        raise S29RunnerBlocked("STATUS_INVENTORY_IDENTITY_DRIFT")
    run_root = _logical(data_root, args.run_root, field="run_root")
    _launch_attempts(run_root, run_id=args.run_id, identity=identity)
    store = S29StatusStore(
        run_root / "status.json",
        run_id=args.run_id,
        plan_hash=preflight.plan_hash,
        inventory_identity=inventory_identity,
        cost_identity={
            "matrix_hash": preflight.frozen.matrix_hash,
            "raw_manifest_hash": preflight.frozen.raw_manifest_hash,
        },
        execution_identity=identity,
    )
    path = run_root / "status.json"
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    while True:
        if path.is_symlink():
            raise S29RunnerBlocked("STATUS_SYMLINK_FORBIDDEN")
        if path.exists():
            value = store.load().to_dict()
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            if not wait or value.get("status") in {"SEALED", "FAILED", "BLOCKED"}:
                return 0 if value.get("status") == "SEALED" else 3
        elif not wait:
            return 4
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(max(0.1, args.poll_seconds))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.9/G2.7a profiler runner")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--detach", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="s209-g27a-formal")
    parser.add_argument("--matrix-ref", required=True)
    parser.add_argument("--gate-ref", required=True)
    parser.add_argument("--g25-gate-ref", required=True)
    parser.add_argument("--raw-manifest-ref", required=True)
    parser.add_argument("--measurement-plan-ref", required=True)
    parser.add_argument("--gpu-inventory-ref", required=True)
    parser.add_argument("--io-evidence-ref", required=True)
    parser.add_argument("--profiler-command", nargs="+")
    parser.add_argument("--single-gpu-anchor-ref")
    parser.add_argument("--four-gpu-anchor-ref")
    parser.add_argument("--accuracy-ref")
    parser.add_argument("--capacity-ref")
    parser.add_argument("--ulimit-ref")
    parser.add_argument("--crosscheck-ref")
    parser.add_argument("--detached-child-marker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--anchor-id", action="append", default=["method-only-anchor-0", "method-only-anchor-1"])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--randomization-seed", type=int, default=2909)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.detach:
            print(json.dumps(_detach(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.status:
            return _status(args, wait=False)
        if args.wait:
            return _status(args, wait=True)
        if args.prepare:
            print(json.dumps(_prepare(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.preflight:
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.execute:
            if not args.profiler_command:
                raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
            print(json.dumps(_execute(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        raise S29RunnerBlocked("S29_ACTION_REQUIRED")
    except (S29RunnerBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.9/G2.7a blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
