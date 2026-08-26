"""Strict S2.10/G2.7b formal control plane.

The legacy invocation (no action flag) remains a synchronous consumer. Formal
invocations add immutable preflight/launch receipts, hash-bound status, and a
replay comparison. No action fabricates a result: all inputs must be sealed
canonical JSON artifacts and formal launches require a clean repository whose
HEAD and source blobs are recorded in the lineage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (  # noqa: E402
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.experiments.stage2_s210_g27b import (  # noqa: E402
    S210G27BBlocked,
    run_s210_g27b,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_STATUS_SCHEMA = "stage2-s210-g27b-detached-status-v1"
_RECEIPT_SCHEMA = "stage2-s210-g27b-input-receipt-v1"
_PREFLIGHT_SCHEMA = "stage2-s210-g27b-preflight-v1"
_LEASE_SCHEMA = "stage2-s210-g27b-detached-lease-v1"
_PID_SCHEMA = "stage2-s210-g27b-detached-launch-v1"
_COMPLETION_SCHEMA = "stage2-s210-g27b-completion-v1"
_REPLAY_SCHEMA = "stage2-s210-g27b-replay-comparison-v1"
_CODE_FILES = (
    "ops/stage2/run_s210_g27b.py",
    "src/param_importance_nlp/experiments/stage2_s210_g27b.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: Any, *, field: str = "run_id") -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise S210G27BBlocked(f"{field}:SAFE_ID_REQUIRED")
    return value


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S210G27BBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise S210G27BBlocked(f"{field}:COMMIT_REQUIRED")
    return value


def _run_git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise S210G27BBlocked(f"GIT_PROVENANCE_FAILED:{args[0] if args else 'unknown'}") from error
    return result.stdout.strip()


def _git_identity(repo: Path, declared_consumer: str | None, producer: str | None) -> dict[str, Any]:
    if not repo.is_dir():
        raise S210G27BBlocked("REPOSITORY_ROOT_INVALID")
    dirty = _run_git(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise S210G27BBlocked("SOURCE_WORKTREE_DIRTY_OR_UNTRACKED")
    head = _commit(_run_git(repo, "rev-parse", "HEAD"), field="consumer_commit")
    if declared_consumer is not None and _commit(declared_consumer, field="consumer_commit") != head:
        raise S210G27BBlocked("CONSUMER_COMMIT_MISMATCH")
    if producer is None:
        raise S210G27BBlocked("PRODUCER_COMMIT_REQUIRED")
    producer_commit = _commit(producer, field="producer_commit")
    try:
        _run_git(repo, "rev-parse", "--verify", f"{producer_commit}^{{commit}}")
    except S210G27BBlocked as error:
        raise S210G27BBlocked("PRODUCER_COMMIT_NOT_IN_REPOSITORY") from error
    blobs: dict[str, str] = {}
    for relative in _CODE_FILES:
        try:
            blob = _run_git(repo, "rev-parse", f"{head}:{relative}")
        except S210G27BBlocked as error:
            raise S210G27BBlocked(f"SOURCE_BLOB_MISSING:{relative}") from error
        blobs[relative] = _commit(blob, field=f"source_blob.{relative}")
    return {
        "repository_head": head,
        "consumer_commit": head,
        "producer_commit": producer_commit,
        "worktree_clean": True,
        "source_blobs": blobs,
    }


def _under(root: Path, value: str | Path, *, field: str) -> Path:
    path = Path(value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise S210G27BBlocked(f"{field}:OUTSIDE_DATA_ROOT") from error
    return path


def _relative(root: Path, path: Path, *, field: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise S210G27BBlocked(f"{field}:OUTSIDE_DATA_ROOT") from error


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:  # pragma: no cover - stable public error below
        raise S210G27BBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S210G27BBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _payload_hash(payload: Mapping[str, Any], *, field: str) -> tuple[str, str | None]:
    declared = payload.get("artifact_hash")
    if declared is None:
        return canonical_json_hash(payload), None
    digest = _sha(declared, field=f"{field}.artifact_hash")
    body = {key: value for key, value in payload.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != digest:
        raise S210G27BBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return digest, digest


def _collect_run_ids(value: Any, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"run_id", "report_id", "source_run_id", "raw_run_id", "s29_run_id"} and isinstance(item, str):
                found.add(item)
            _collect_run_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_run_ids(item, found)


def _input_receipt(path: Path, *, root: Path, field: str) -> dict[str, Any]:
    if not path.is_file():
        raise S210G27BBlocked(f"{field}:FILE_REQUIRED")
    payload = _load_object(path, field=field)
    artifact_hash, declared = _payload_hash(payload, field=field)
    run_ids: set[str] = set()
    _collect_run_ids(payload, run_ids)
    return {
        "ref": _relative(root, path, field=f"{field}.ref"),
        "schema_version": payload.get("schema_version"),
        "artifact_hash": artifact_hash,
        "declared_artifact_hash": declared,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "run_ids": sorted(run_ids),
    }


def _source_paths(args: argparse.Namespace) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for name in (
        "g26_gate",
        "g26_quality_gates",
        "g26_hypothesis_decisions",
        "g26_statistics_long_table",
        "g26_family_decisions",
        "g27a_report",
        "g27a_gate",
    ):
        value = getattr(args, name)
        if value is None:
            raise S210G27BBlocked(f"{name}:INPUT_REQUIRED")
        values[name] = value
    for name in ("g26_statistics_summary", "g26_raw_calibration", "matrix"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    return values


def _control_context(args: argparse.Namespace, *, require_new: bool) -> dict[str, Any]:
    if args.data_root is None or args.run_root is None or args.operations_root is None:
        raise S210G27BBlocked("DATA_RUN_OPERATIONS_ROOTS_REQUIRED")
    if args.producer_commit is None:
        raise S210G27BBlocked("PRODUCER_COMMIT_REQUIRED")
    root = Path(args.data_root).resolve()
    if not root.is_dir():
        raise S210G27BBlocked("DATA_ROOT_INVALID")
    run_root = _under(root, args.run_root, field="run_root")
    operations_root = _under(root, args.operations_root, field="operations_root")
    output_root = _under(root, args.output_root, field="output_root")
    if len({root, run_root, operations_root, output_root}) != 4:
        raise S210G27BBlocked("CONTROL_ROOT_COLLISION")
    if require_new and (output_root.exists() or run_root.exists()):
        raise S210G27BBlocked("RUN_OR_OUTPUT_ROOT_MUST_BE_NEW")
    identity = _git_identity(Path(args.repo_root).resolve(), args.consumer_commit, args.producer_commit)
    source_paths = _source_paths(args)
    input_artifacts: dict[str, Any] = {}
    for name, value in source_paths.items():
        path = _under(root, value, field=name)
        input_artifacts[name] = _input_receipt(path, root=root, field=name)
    if args.g26_raw_calibration is None:
        # The consumer can make a blocked report without this optional input,
        # but a formal preflight must reject such a launch before any child.
        raise S210G27BBlocked("g26_raw_calibration:FORMAL_INPUT_REQUIRED")
    paths = {
        "data_root": ".",
        "run_root": _relative(root, run_root, field="run_root"),
        "operations_root": _relative(root, operations_root, field="operations_root"),
        "output_root": _relative(root, output_root, field="output_root"),
    }
    return {
        "root": root,
        "run_root": run_root,
        "operations_root": operations_root,
        "output_root": output_root,
        "identity": identity,
        "input_artifacts": input_artifacts,
        "paths": paths,
    }


def _immutable(path: Path, value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    payload = dict(value)
    if path.exists():
        existing = _load_object(path, field=field)
        if existing != payload:
            raise S210G27BBlocked(f"{field}:IMMUTABLE_OUTPUT_EXISTS")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, payload)
    return payload


def _hash_bound(body: Mapping[str, Any]) -> dict[str, Any]:
    return dict(body) | {"artifact_hash": canonical_json_hash(body)}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    context = _control_context(args, require_new=True)
    operations_root = context["operations_root"]
    run_id = _safe_id(args.run_id)
    input_artifacts = context["input_artifacts"]
    reasons: list[str] = []
    for name in ("g26_gate", "g27a_gate"):
        payload = _load_object(_under(context["root"], getattr(args, name), field=name), field=name)
        if payload.get("status") != "PASS":
            reasons.append(f"{name.upper()}_NOT_PASS")
    quality = _load_object(_under(context["root"], args.g26_quality_gates, field="g26_quality_gates"), field="g26_quality_gates")
    if quality.get("status") != "PASS" or quality.get("formal_eligible") is not True:
        reasons.append("G2.6_QUALITY_NOT_FORMAL_PASS")
    cost = _load_object(_under(context["root"], args.g27a_report, field="g27a_report"), field="g27a_report")
    if cost.get("status") != "PASS" or cost.get("formal_eligible") is not True:
        reasons.append("G2.7A_COST_NOT_FORMAL_PASS")
    body = {
        "schema_version": _PREFLIGHT_SCHEMA,
        "status": "READY" if not reasons else "BLOCKED",
        "formal_eligible": not reasons,
        "task_id": "stage2.10_visualization_reporting_and_decision",
        "run_id": run_id,
        "prepared_at": _now(),
        "paths": context["paths"],
        "producer_commit": context["identity"]["producer_commit"],
        "consumer_commit": context["identity"]["consumer_commit"],
        "code_identity": context["identity"],
        "input_artifacts": input_artifacts,
        "reasons": sorted(set(reasons)),
    }
    receipt_body = {
        "schema_version": _RECEIPT_SCHEMA,
        "task_id": body["task_id"],
        "run_id": run_id,
        "input_artifacts": input_artifacts,
        "producer_commit": body["producer_commit"],
        "consumer_commit": body["consumer_commit"],
    }
    operations_root.mkdir(parents=True, exist_ok=True)
    receipt = _immutable(operations_root / "input-receipt.json", _hash_bound(receipt_body), field="input_receipt")
    body["input_receipt_hash"] = receipt["artifact_hash"]
    existing_preflight = operations_root / "preflight.json"
    if existing_preflight.exists():
        old = _load_object(existing_preflight, field="preflight")
        prepared_at = old.get("prepared_at")
        if isinstance(prepared_at, str):
            body["prepared_at"] = prepared_at
    preflight = _immutable(existing_preflight, _hash_bound(body), field="preflight")
    if preflight["status"] != "READY":
        raise S210G27BBlocked("PREFLIGHT_BLOCKED:" + ",".join(preflight["reasons"]))
    return {"context": context, "preflight": preflight, "receipt": receipt}


def _load_preflight(args: argparse.Namespace, *, require_new: bool = False) -> dict[str, Any]:
    context = _control_context(args, require_new=require_new)
    path = context["operations_root"] / "preflight.json"
    if not path.exists():
        return _preflight(args)
    old = _load_object(path, field="preflight")
    declared = _sha(old.get("artifact_hash"), field="preflight.artifact_hash")
    if canonical_json_hash({key: value for key, value in old.items() if key != "artifact_hash"}) != declared:
        raise S210G27BBlocked("PREFLIGHT_HASH_MISMATCH")
    if old.get("run_id") != _safe_id(args.run_id):
        raise S210G27BBlocked("PREFLIGHT_RUN_ID_MISMATCH")
    if old.get("status") != "READY" or old.get("formal_eligible") is not True:
        raise S210G27BBlocked("PREFLIGHT_NOT_READY")
    if old.get("paths") != context["paths"]:
        raise S210G27BBlocked("PREFLIGHT_PATH_IDENTITY_MISMATCH")
    if old.get("input_artifacts") != context["input_artifacts"]:
        raise S210G27BBlocked("PREFLIGHT_INPUT_IDENTITY_MISMATCH")
    identity = context["identity"]
    if old.get("producer_commit") != identity["producer_commit"] or old.get("consumer_commit") != identity["consumer_commit"]:
        raise S210G27BBlocked("PREFLIGHT_COMMIT_IDENTITY_MISMATCH")
    receipt_path = context["operations_root"] / "input-receipt.json"
    if not receipt_path.exists():
        raise S210G27BBlocked("INPUT_RECEIPT_MISSING")
    receipt = _load_object(receipt_path, field="input_receipt")
    receipt_hash = _sha(receipt.get("artifact_hash"), field="input_receipt.artifact_hash")
    if canonical_json_hash({key: value for key, value in receipt.items() if key != "artifact_hash"}) != receipt_hash:
        raise S210G27BBlocked("INPUT_RECEIPT_HASH_MISMATCH")
    if old.get("input_receipt_hash") != receipt_hash:
        raise S210G27BBlocked("PREFLIGHT_RECEIPT_IDENTITY_MISMATCH")
    old["input_receipt_hash"] = receipt_hash
    return {"context": context, "preflight": old, "receipt": receipt}


def _lease(operations_root: Path, *, run_id: str, preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = operations_root / "launcher.lease.json"
    body = {
        "schema_version": _LEASE_SCHEMA,
        "run_id": run_id,
        "owner_pid": os.getpid(),
        "started_at": _now(),
        "preflight_hash": preflight["artifact_hash"],
    }
    payload = _hash_bound(body)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise S210G27BBlocked("DETACHED_LAUNCH_ALREADY_CLAIMED") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_json_bytes(payload))
    return payload


def _status_payload(
    *,
    preflight: Mapping[str, Any],
    run_id: str,
    status: str,
    exit_code: int | None,
    owner_pid: int | None,
    terminal_reason: str | None = None,
    output_ref: str | None = None,
    analysis_hash: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema_version": _STATUS_SCHEMA,
        "task_id": "stage2.10_visualization_reporting_and_decision",
        "stage": 2,
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "owner_pid": owner_pid,
        "updated_at": _now(),
        "terminal_reason": terminal_reason,
        "output_ref": output_ref,
        "analysis_hash": analysis_hash,
        "preflight_hash": preflight["artifact_hash"],
        "input_receipt_hash": preflight.get("input_receipt_hash"),
        "producer_commit": preflight["producer_commit"],
        "consumer_commit": preflight["consumer_commit"],
    }
    return _hash_bound(body)


def _replace_status(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    write_canonical_json(path, payload)
    return dict(payload)


def _publish_status(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        old = _load_object(path, field="status")
        old_status = old.get("status")
        new_status = payload.get("status")
        allowed = {
            "RUNNING": {"RUNNING", "SEALED", "BLOCKED", "FAILED"},
            "SEALED": {"SEALED"},
            "BLOCKED": {"BLOCKED"},
            "FAILED": {"FAILED"},
        }
        if old.get("run_id") != payload.get("run_id") or old.get("preflight_hash") != payload.get("preflight_hash"):
            raise S210G27BBlocked("STATUS_IDENTITY_MISMATCH")
        if new_status not in allowed.get(str(old_status), set()):
            raise S210G27BBlocked(f"STATUS_TRANSITION_FORBIDDEN:{old_status}->{new_status}")
        return _replace_status(path, payload)
    return _immutable(path, payload, field="status")


def _execute(args: argparse.Namespace, prepared: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    prepared = prepared or _load_preflight(args, require_new=False)
    context = prepared["context"]
    preflight = prepared["preflight"]
    run_id = _safe_id(args.run_id)
    run_root: Path = context["run_root"]
    if run_root.exists():
        if not args.launcher_child or (run_root / "status.json").exists():
            raise S210G27BBlocked("RUN_ROOT_MUST_BE_NEW")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    lease_path = context["operations_root"] / "launcher.lease.json"
    if not lease_path.exists():
        _lease(context["operations_root"], run_id=run_id, preflight=preflight)
    if args.launcher_child:
        # The parent writes launcher.pid.json immediately after Popen. Wait a
        # short bounded interval for that immutable identity before allowing a
        # child to reuse the pre-created run root.
        pid_path = context["operations_root"] / "launcher.pid.json"
        deadline = time.monotonic() + 5.0
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        pid = _load_object(pid_path, field="detached_launch") if pid_path.exists() else None
        if (
            not isinstance(pid, Mapping)
            or pid.get("pid") != os.getpid()
            or pid.get("run_id") != run_id
            or pid.get("preflight_hash") != preflight["artifact_hash"]
        ):
            raise S210G27BBlocked("DETACHED_CHILD_IDENTITY_MISMATCH")
    status_path = run_root / "status.json"
    _publish_status(status_path, _status_payload(preflight=preflight, run_id=run_id, status="RUNNING", exit_code=None, owner_pid=os.getpid()))
    try:
        result = run_s210_g27b(
            g26_gate=args.g26_gate,
            g26_quality_gates=args.g26_quality_gates,
            g26_hypothesis_decisions=args.g26_hypothesis_decisions,
            g26_statistics_long_table=args.g26_statistics_long_table,
            g26_statistics_summary=args.g26_statistics_summary,
            g26_raw_calibration=args.g26_raw_calibration,
            g26_family_decisions=args.g26_family_decisions,
            g27a_report=args.g27a_report,
            g27a_gate=args.g27a_gate,
            matrix=args.matrix,
            output_root=context["output_root"],
            run_id=run_id,
            checked_at=args.checked_at,
            producer_commit=preflight["producer_commit"],
            consumer_commit=preflight["consumer_commit"],
            input_refs=preflight["input_artifacts"],
            code_identity=preflight["code_identity"],
        )
    except Exception as error:
        _publish_status(
            status_path,
            _status_payload(
                preflight=preflight,
                run_id=run_id,
                status="BLOCKED",
                exit_code=2,
                owner_pid=os.getpid(),
                terminal_reason=f"{type(error).__name__}:{error}",
            ),
        )
        raise
    status = "SEALED" if result.get("status") == "PASS" else "BLOCKED"
    exit_code = 0 if status == "SEALED" else 2
    output_ref = _relative(context["root"], context["output_root"], field="output_root")
    completion = _hash_bound(
        {
            "schema_version": _COMPLETION_SCHEMA,
            "task_id": "stage2.10_visualization_reporting_and_decision",
            "stage": 2,
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "output_ref": output_ref,
            "analysis_hash": result.get("analysis_hash"),
            "preflight_hash": preflight["artifact_hash"],
            "producer_commit": preflight["producer_commit"],
            "consumer_commit": preflight["consumer_commit"],
            "input_receipt_hash": preflight.get("input_receipt_hash"),
            "status_ref": _relative(context["root"], status_path, field="status_ref"),
        }
    )
    # Publish the immutable completion receipt before the terminal status.  A
    # reader can therefore never observe SEALED without its completion proof.
    _immutable(context["operations_root"] / "completion.json", completion, field="completion")
    _publish_status(
        status_path,
        _status_payload(
            preflight=preflight,
            run_id=run_id,
            status=status,
            exit_code=exit_code,
            owner_pid=os.getpid(),
            terminal_reason=None if status == "SEALED" else "S2.10_G27B_BLOCKED",
            output_ref=output_ref,
            analysis_hash=result.get("analysis_hash"),
        ),
    )
    return exit_code, result


def _detach(args: argparse.Namespace, raw_argv: list[str] | None = None) -> dict[str, Any]:
    prepared = _load_preflight(args, require_new=True)
    context = prepared["context"]
    preflight = prepared["preflight"]
    run_id = _safe_id(args.run_id)
    lease = _lease(context["operations_root"], run_id=run_id, preflight=preflight)
    run_root: Path = context["run_root"]
    run_root.mkdir(parents=True, exist_ok=False)
    log_path = run_root / "runner.log"
    child = [str(item) for item in (sys.argv[1:] if raw_argv is None else raw_argv)]
    try:
        child.remove("--detach")
    except ValueError as error:
        raise S210G27BBlocked("DETACH_ACTION_NOT_FOUND") from error
    if "--execute" not in child:
        child.insert(0, "--execute")
    child.append("--launcher-child")
    command = [sys.executable, str(Path(__file__).resolve()), *child]
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=_REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = _hash_bound(
        {
            "schema_version": _PID_SCHEMA,
            "pid": int(process.pid),
            "run_id": run_id,
            "argv": command,
            "run_ref": _relative(context["root"], run_root, field="run_root"),
            "operations_ref": _relative(context["root"], context["operations_root"], field="operations_root"),
            "log_ref": _relative(context["root"], log_path, field="log_ref"),
            "status_ref": _relative(context["root"], run_root / "status.json", field="status_ref"),
            "preflight_hash": preflight["artifact_hash"],
            "producer_commit": preflight["producer_commit"],
            "consumer_commit": preflight["consumer_commit"],
            "input_receipt_hash": preflight.get("input_receipt_hash"),
            "started_at": _now(),
            "lease_hash": lease["artifact_hash"],
        }
    )
    _immutable(context["operations_root"] / "launcher.pid.json", payload, field="detached_launch")
    return payload


def _read_status(args: argparse.Namespace, *, wait: bool) -> int:
    prepared = _load_preflight(args, require_new=False)
    context = prepared["context"]
    path = context["run_root"] / "status.json"
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    while True:
        if path.exists():
            value = _load_object(path, field="status")
            declared = _sha(value.get("artifact_hash"), field="status.artifact_hash")
            if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != declared:
                raise S210G27BBlocked("STATUS_HASH_MISMATCH")
            if value.get("run_id") != _safe_id(args.run_id) or value.get("preflight_hash") != prepared["preflight"]["artifact_hash"]:
                raise S210G27BBlocked("STATUS_IDENTITY_MISMATCH")
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            state = value.get("status")
            if state == "SEALED":
                completion_path = context["operations_root"] / "completion.json"
                if not completion_path.exists():
                    raise S210G27BBlocked("COMPLETION_RECEIPT_MISSING")
                completion = _load_object(completion_path, field="completion")
                completion_hash = _sha(completion.get("artifact_hash"), field="completion.artifact_hash")
                if canonical_json_hash({key: item for key, item in completion.items() if key != "artifact_hash"}) != completion_hash:
                    raise S210G27BBlocked("COMPLETION_HASH_MISMATCH")
                if (
                    completion.get("run_id") != value.get("run_id")
                    or completion.get("status") != "SEALED"
                    or completion.get("exit_code") != 0
                    or completion.get("preflight_hash") != value.get("preflight_hash")
                    or completion.get("output_ref") != value.get("output_ref")
                ):
                    raise S210G27BBlocked("COMPLETION_IDENTITY_MISMATCH")
            if not wait or state in {"SEALED", "BLOCKED", "FAILED"}:
                return 0 if state == "SEALED" and value.get("exit_code") == 0 else 2
        elif not wait:
            return 4
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(max(0.1, args.poll_seconds))


def _health_check(args: argparse.Namespace) -> int:
    prepared = _load_preflight(args, require_new=False)
    context = prepared["context"]
    cost = _load_object(_under(context["root"], args.g27a_report, field="g27a_report"), field="g27a_report")
    snapshot = cost.get("health_snapshot")
    required = (
        "healthy",
        "idle",
        "same_gpu_class",
        "gpu_class",
        "gpu_uuids",
        "ecc_errors",
        "xid_errors",
        "inventory_artifact_hash",
        "inventory_source_sha256",
        "cost_io_quiescent",
    )
    reasons: list[str] = []
    if not isinstance(snapshot, Mapping):
        reasons.append("HEALTH_SNAPSHOT_REQUIRED")
        snapshot = {}
    missing = [name for name in required if name not in snapshot]
    if missing:
        reasons.append("HEALTH_FIELDS_MISSING:" + ",".join(missing))
    if snapshot.get("healthy") is not True or snapshot.get("idle") is not True or snapshot.get("same_gpu_class") is not True:
        reasons.append("GPU_HEALTH_OR_IDLE_FAILED")
    if snapshot.get("ecc_errors") != 0 or snapshot.get("xid_errors") != 0:
        reasons.append("GPU_ERROR_COUNTER_NONZERO")
    if snapshot.get("cost_io_quiescent") is not True:
        reasons.append("COST_IO_NOT_QUIESCENT")
    try:
        _sha(snapshot.get("inventory_artifact_hash"), field="health.inventory_artifact_hash")
        _sha(snapshot.get("inventory_source_sha256"), field="health.inventory_source_sha256")
    except S210G27BBlocked as error:
        reasons.append(str(error))
    health = _hash_bound(
        {
            "schema_version": "stage2-s210-g27b-health-check-v1",
            "task_id": "stage2.10_visualization_reporting_and_decision",
            "stage": 2,
            "run_id": _safe_id(args.run_id),
            "status": "PASS" if not reasons else "BLOCKED",
            "formal_eligible": not reasons,
            "checked_at": _now(),
            "g27a_report_ref": _relative(context["root"], _under(context["root"], args.g27a_report, field="g27a_report"), field="g27a_report"),
            "g27a_report_hash": prepared["preflight"]["input_artifacts"]["g27a_report"]["artifact_hash"],
            "health_snapshot": dict(snapshot),
            "producer_commit": prepared["preflight"]["producer_commit"],
            "consumer_commit": prepared["preflight"]["consumer_commit"],
            "input_receipt_hash": prepared["preflight"].get("input_receipt_hash"),
            "reasons": sorted(set(reasons)),
        }
    )
    _immutable(context["operations_root"] / "health-check.json", health, field="health_check")
    print(json.dumps(health, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if not reasons else 2


def _semantic_file_hash(path: Path, *, relative: str) -> str:
    if path.suffix.lower() != ".json":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = _load_object(path, field=f"replay.{relative}")
    if payload.get("artifact_hash") is not None:
        _payload_hash(payload, field=f"replay.{relative}")
    ignored: set[str] = set()
    if relative == "report.json":
        ignored = {"report_id", "artifact_hash"}
    elif relative == "lineage_manifest.json":
        ignored = {"run_id", "artifact_hash"}
    elif relative == "g2.7b-gate.json":
        ignored = {"checked_at", "artifact_hash"}
    return canonical_json_hash({key: value for key, value in payload.items() if key not in ignored})


def _output_path(root: Path, relative: str, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or "\\" in relative:
        raise S210G27BBlocked(f"{field}:INVALID_OUTPUT_REFERENCE")
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise S210G27BBlocked(f"{field}:OUTPUT_REFERENCE_ESCAPE") from error
    return candidate


def _write_replay_blocked(
    *,
    prepared: Mapping[str, Any],
    root: Path,
    source_root: Path,
    source_run_id: str,
    replay_id: str,
    source_report_hash: str,
    source_lineage_hash: str,
    reason: str,
) -> int:
    context = prepared["context"]
    preflight = prepared["preflight"]
    comparison = _hash_bound(
        {
            "schema_version": _REPLAY_SCHEMA,
            "status": "BLOCKED",
            "formal_eligible": False,
            "source_run_id": source_run_id,
            "replay_id": replay_id,
            "source_result_ref": _relative(root, source_root, field="source_result"),
            "replay_result_ref": _relative(root, context["output_root"], field="replay_result"),
            "source_report_hash": source_report_hash,
            "replay_report_hash": None,
            "source_lineage_hash": source_lineage_hash,
            "replay_lineage_hash": None,
            "producer_commit": preflight["producer_commit"],
            "consumer_commit": preflight["consumer_commit"],
            "input_artifacts": preflight.get("input_artifacts", {}),
            "compared_files": {},
            "allowed_differences": ["replay id", "output refs", "timestamps", "outer artifact hashes"],
            "reasons": [reason],
        }
    )
    _immutable(context["operations_root"] / "replay-comparison.json", comparison, field="replay_comparison")
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2))
    return 2


def _replay(args: argparse.Namespace) -> int:
    if args.source_result is None or args.data_root is None:
        raise S210G27BBlocked("SOURCE_RESULT_AND_DATA_ROOT_REQUIRED")
    root = Path(args.data_root).resolve()
    source_root = _under(root, args.source_result, field="source_result")
    if not source_root.is_dir():
        raise S210G27BBlocked("SOURCE_RESULT_INVALID")
    replay_id = _safe_id(args.replay_id or args.run_id, field="replay_id")
    source_report = _load_object(source_root / "report.json", field="source_report")
    source_lineage = _load_object(source_root / "lineage_manifest.json", field="source_lineage")
    source_report_hash, _ = _payload_hash(source_report, field="source_report")
    source_lineage_hash, _ = _payload_hash(source_lineage, field="source_lineage")
    source_run_id = source_report.get("report_id")
    if not isinstance(source_run_id, str) or source_run_id == replay_id:
        raise S210G27BBlocked("REPLAY_ID_MUST_BE_NEW")
    # The replay identity is the run identity consumed by the producer.  Keep
    # it separate from the source report while preserving all sealed inputs.
    args.run_id = replay_id
    if source_report.get("status") != "PASS" or source_report.get("formal_eligible") is not True:
        raise S210G27BBlocked("SOURCE_RESULT_NOT_FORMAL_PASS")
    source_inputs = source_lineage.get("input_artifacts")
    source_refs = source_lineage.get("input_refs")
    if not isinstance(source_inputs, Mapping) or not isinstance(source_refs, Mapping):
        raise S210G27BBlocked("SOURCE_LINEAGE_INPUT_IDENTITY_MISSING")
    prepared = _load_preflight(args, require_new=True)
    current_inputs = prepared["preflight"].get("input_artifacts")
    if not isinstance(current_inputs, Mapping):
        raise S210G27BBlocked("REPLAY_CURRENT_INPUT_IDENTITY_MISSING")
    current_hashes = {key: value.get("artifact_hash") for key, value in current_inputs.items() if isinstance(value, Mapping)}
    if any(key not in current_hashes or current_hashes[key] != value for key, value in source_inputs.items()):
        return _write_replay_blocked(
            prepared=prepared,
            root=root,
            source_root=source_root,
            source_run_id=source_run_id,
            replay_id=replay_id,
            source_report_hash=source_report_hash,
            source_lineage_hash=source_lineage_hash,
            reason="REPLAY_INPUT_ARTIFACT_HASH_MISMATCH",
        )
    if dict(source_refs) != current_inputs:
        return _write_replay_blocked(
            prepared=prepared,
            root=root,
            source_root=source_root,
            source_run_id=source_run_id,
            replay_id=replay_id,
            source_report_hash=source_report_hash,
            source_lineage_hash=source_lineage_hash,
            reason="REPLAY_INPUT_REFERENCE_MISMATCH",
        )
    source_code = source_lineage.get("code_identity")
    if not isinstance(source_code, Mapping) or dict(source_code) != prepared["preflight"]["code_identity"]:
        return _write_replay_blocked(
            prepared=prepared,
            root=root,
            source_root=source_root,
            source_run_id=source_run_id,
            replay_id=replay_id,
            source_report_hash=source_report_hash,
            source_lineage_hash=source_lineage_hash,
            reason="REPLAY_CONSUMER_COMMIT_MISMATCH",
        )
    try:
        exit_code, result = _execute(args, prepared=prepared)
    except Exception as error:
        _write_replay_blocked(
            prepared=prepared,
            root=root,
            source_root=source_root,
            source_run_id=source_run_id,
            replay_id=replay_id,
            source_report_hash=source_report_hash,
            source_lineage_hash=source_lineage_hash,
            reason=f"EXECUTION_FAILED:{type(error).__name__}:{error}",
        )
        raise
    replay_root: Path = prepared["context"]["output_root"]
    source_files = source_report.get("output_files")
    if not isinstance(source_files, list) or any(not isinstance(item, str) for item in source_files):
        raise S210G27BBlocked("SOURCE_OUTPUT_FILE_LIST_INVALID")
    replay_report = _load_object(replay_root / "report.json", field="replay_report")
    replay_files = replay_report.get("output_files")
    if not isinstance(replay_files, list) or any(not isinstance(item, str) for item in replay_files):
        raise S210G27BBlocked("REPLAY_OUTPUT_FILE_LIST_INVALID")
    compared: dict[str, Any] = {}
    reasons: list[str] = []
    if source_files != replay_files:
        reasons.append("OUTPUT_FILE_SET_MISMATCH")
    relative_files = sorted(set(source_files) | set(replay_files))
    for relative in relative_files:
        source_path = _output_path(source_root, relative, field="source_output")
        replay_path = _output_path(replay_root, relative, field="replay_output")
        if not source_path.is_file() or not replay_path.is_file():
            reasons.append(f"OUTPUT_FILE_MISSING:{relative}")
            continue
        source_hash = _semantic_file_hash(source_path, relative=relative)
        replay_hash = _semantic_file_hash(replay_path, relative=relative)
        compared[relative] = {"source_hash": source_hash, "replay_hash": replay_hash, "match": source_hash == replay_hash}
        if source_hash != replay_hash:
            reasons.append(f"SEMANTIC_HASH_MISMATCH:{relative}")
    replay_report = _load_object(replay_root / "report.json", field="replay_report")
    replay_lineage = _load_object(replay_root / "lineage_manifest.json", field="replay_lineage")
    replay_report_hash, _ = _payload_hash(replay_report, field="replay_report")
    replay_lineage_hash, _ = _payload_hash(replay_lineage, field="replay_lineage")
    if source_report.get("upstream_artifacts") != replay_report.get("upstream_artifacts"):
        reasons.append("UPSTREAM_ARTIFACT_IDENTITY_MISMATCH")
    if source_lineage.get("producer_commit") != replay_lineage.get("producer_commit") or source_lineage.get("consumer_commit") != replay_lineage.get("consumer_commit"):
        reasons.append("LINEAGE_COMMIT_MISMATCH")
    status = "PASS" if exit_code == 0 and not reasons else "BLOCKED"
    comparison_body = {
        "schema_version": _REPLAY_SCHEMA,
        "status": status,
        "formal_eligible": status == "PASS",
        "source_run_id": source_run_id,
        "replay_id": replay_id,
        "source_result_ref": _relative(root, source_root, field="source_result"),
        "replay_result_ref": _relative(root, replay_root, field="replay_result"),
        "source_report_hash": source_report_hash,
        "replay_report_hash": replay_report_hash,
        "source_lineage_hash": source_lineage_hash,
        "replay_lineage_hash": replay_lineage_hash,
        "producer_commit": prepared["preflight"]["producer_commit"],
        "consumer_commit": prepared["preflight"]["consumer_commit"],
        "input_artifacts": current_inputs,
        "compared_files": compared,
        "allowed_differences": ["replay id", "output refs", "timestamps", "outer artifact hashes"],
        "reasons": sorted(set(reasons)),
    }
    comparison = _hash_bound(comparison_body)
    _immutable(prepared["context"]["operations_root"] / "replay-comparison.json", comparison, field="replay_comparison")
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if status == "PASS" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Stage 2 S2.10/G2.7b report and decision")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    action.add_argument("--replay", action="store_true")
    action.add_argument("--health-check", action="store_true")
    # ``--detach`` is a modifier so both ``--detach`` and
    # ``--execute --detach`` remain unambiguous and accepted.
    parser.add_argument("--detach", action="store_true", help="launch a detached --execute child")
    parser.add_argument("--g26-gate", type=Path, required=True)
    parser.add_argument("--g26-quality-gates", type=Path, required=True)
    parser.add_argument("--g26-hypothesis-decisions", type=Path, required=True)
    parser.add_argument("--g26-statistics-long-table", type=Path, required=True)
    parser.add_argument("--g26-statistics-summary", type=Path)
    parser.add_argument("--g26-raw-calibration", type=Path)
    parser.add_argument("--g26-family-decisions", type=Path, required=True)
    parser.add_argument("--g27a-report", type=Path, required=True)
    parser.add_argument("--g27a-gate", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output-root", type=Path, required=True, help="new append-only report directory")
    parser.add_argument("--run-id", default="s210-g27b")
    parser.add_argument("--checked-at")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--operations-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--producer-commit")
    parser.add_argument("--consumer-commit")
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--replay-id")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--launcher-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    control = any((args.preflight, args.execute, args.detach, args.status, args.wait, args.replay, args.health_check))
    try:
        if not control:
            result = run_s210_g27b(
                g26_gate=args.g26_gate,
                g26_quality_gates=args.g26_quality_gates,
                g26_hypothesis_decisions=args.g26_hypothesis_decisions,
                g26_statistics_long_table=args.g26_statistics_long_table,
                g26_statistics_summary=args.g26_statistics_summary,
                g26_raw_calibration=args.g26_raw_calibration,
                g26_family_decisions=args.g26_family_decisions,
                g27a_report=args.g27a_report,
                g27a_gate=args.g27a_gate,
                matrix=args.matrix,
                output_root=args.output_root,
                run_id=args.run_id,
                checked_at=args.checked_at,
            )
            print(f"S2.10/G2.7b {result['status']}; outputs={len(result['output_files'])}; analysis_hash={result['analysis_hash']}")
            return 0 if result["status"] == "PASS" else 2
        if args.preflight:
            prepared = _preflight(args)
            print(json.dumps(prepared["preflight"], ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.detach:
            print(json.dumps(_detach(args, raw_argv=argv), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.execute:
            exit_code, result = _execute(args)
            print(f"S2.10/G2.7b {result['status']}; outputs={len(result['output_files'])}; analysis_hash={result['analysis_hash']}")
            return exit_code
        if args.status:
            return _read_status(args, wait=False)
        if args.wait:
            return _read_status(args, wait=True)
        if args.replay:
            return _replay(args)
        if args.health_check:
            return _health_check(args)
        raise S210G27BBlocked("S210_ACTION_REQUIRED")
    except (S210G27BBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.10/G2.7b BLOCKED: {error}", file=sys.stderr)
        return (2 if args.execute or args.replay or args.health_check else 3) if control else 2


if __name__ == "__main__":
    raise SystemExit(main())
