#!/usr/bin/env python3
"""Run formal Stage3.08 behind the completed S3.07 handoff boundary.

The scientific task is already rebuild-derived and idempotent.  This wrapper
adds the missing operational authority: a hash-bound start/end journal, a
stable clean-Git identity, and a verified immutable receipt suitable for the
subsequent provenance publisher.  It never launches Stage3.08 until the full
99-unit S3.07 handoff audit reports COMPLETE.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[2]
    for _candidate in (_repository_root, _repository_root / "src"):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

from ops.stage3.publish_stage3_g36_provenance import (
    _confirm_git_snapshot,
    _git_snapshot,
)
from ops.stage3.run_stage3_formal import (
    InstanceLock,
    Stage3OrchestratorError,
    _canonical_hash,
    _load_json,
    _now,
    _write_atomic,
)
from ops.stage3.verify_stage3_s307_handoff import run_audit
from param_importance_nlp.contracts import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_bytes
from param_importance_nlp.runtime import (
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
    publish_canonical_immutable,
)


SCHEMA_VERSION = "stage3-s308-timed-execution-v1"
STATE_SCHEMA_VERSION = "stage3-s308-timed-state-v1"
TASK_ID = "stage3.08_error_analysis_and_stability"
OUTPUT_KINDS = frozenset(
    {"path_error_table", "stability_report", "frozen_source_table"}
)
PHASES = frozenset({"PREPARED", "RUNNING", "S308_COMPLETE", "COMPLETE", "FAILED"})


class Stage3S308TimedError(ValueError):
    """Raised when the timed S3.08 boundary cannot safely advance."""


def _fail(code: str, detail: object | None = None) -> Stage3S308TimedError:
    return Stage3S308TimedError(code if detail is None else f"{code}:{detail}")


def _hash(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail("S308_TIMED_HASH_INVALID", field)
    return value


def _git_commit(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _fail("S308_TIMED_GIT_COMMIT_INVALID")
    return value


def _logical_path(root: Path, value: Path, *, field: str) -> tuple[Path, str]:
    candidate = value if value.is_absolute() else root / value
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise _fail("S308_TIMED_PATH_OUTSIDE_DATA_ROOT", field) from error
    if root.is_symlink():
        raise _fail("S308_TIMED_DATA_ROOT_SYMLINK")
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _fail("S308_TIMED_PATH_SYMLINK", field)
    logical = PurePosixPath(*relative.parts).as_posix()
    return absolute, logical


def _existing_input(root: Path, value: Path, *, field: str) -> tuple[Path, str]:
    path, logical = _logical_path(root, value, field=field)
    if not path.exists():
        raise _fail("S308_TIMED_INPUT_MISSING", field)
    return path, logical


def _parse_time(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise _fail("S308_TIMED_TIME_INVALID", field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _fail("S308_TIMED_TIME_INVALID", field) from error
    if parsed.tzinfo is None:
        raise _fail("S308_TIMED_TIMEZONE_REQUIRED", field)
    return parsed


def _result_mtime(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _launch_identity(arguments: argparse.Namespace, *, refs: Mapping[str, str]) -> str:
    return _canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "expected_git_commit": arguments.expected_git_commit,
            "refs": dict(sorted(refs.items())),
            "command": [
                refs["python_executable"],
                "-m",
                "param_importance_nlp",
                "task",
                "run",
                "--config",
                refs["s308_config"],
                "--environment",
                refs["s308_environment"],
                "--result",
                refs["s308_result"],
            ],
        }
    )


def _new_state(
    *,
    launch_hash: str,
    config_hash: str,
    environment_hash: str,
    git_commit: str,
    git_branch: str,
) -> dict[str, Any]:
    now = _now()
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "launch_hash": launch_hash,
        "task_id": TASK_ID,
        "config_hash": config_hash,
        "environment_hash": environment_hash,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "phase": "PREPARED",
        "started_at": None,
        "ended_at": None,
        "ended_at_source": None,
        "attempts": [],
        "result_hash": None,
        "recovered": False,
        "handoff_audit_hash": None,
        "receipt_ref": None,
        "receipt_hash": None,
        "created_at": now,
        "updated_at": now,
    }
    state["state_hash"] = _canonical_hash(state)
    return state


def _validate_state(
    value: Mapping[str, Any],
    *,
    launch_hash: str,
    config_hash: str,
    environment_hash: str,
    git_commit: str,
    git_branch: str,
) -> dict[str, Any]:
    expected = {
        "schema_version", "launch_hash", "task_id", "config_hash",
        "environment_hash", "git_commit", "git_branch", "phase",
        "started_at", "ended_at", "ended_at_source", "attempts",
        "result_hash", "recovered", "handoff_audit_hash", "receipt_ref",
        "receipt_hash", "created_at", "updated_at", "state_hash",
    }
    if set(value) != expected:
        raise _fail("S308_TIMED_STATE_FIELDS_INVALID")
    declared = value.get("state_hash")
    if declared != _canonical_hash(
        {key: item for key, item in value.items() if key != "state_hash"}
    ):
        raise _fail("S308_TIMED_STATE_HASH_INVALID")
    if (
        value.get("schema_version") != STATE_SCHEMA_VERSION
        or value.get("launch_hash") != launch_hash
        or value.get("task_id") != TASK_ID
        or value.get("config_hash") != config_hash
        or value.get("environment_hash") != environment_hash
        or value.get("git_commit") != git_commit
        or value.get("git_branch") != git_branch
    ):
        raise _fail("S308_TIMED_STATE_IDENTITY_DRIFT")
    if value.get("phase") not in PHASES:
        raise _fail("S308_TIMED_STATE_PHASE_INVALID")
    if not isinstance(value.get("attempts"), list) or type(value.get("recovered")) is not bool:
        raise _fail("S308_TIMED_STATE_ATTEMPTS_INVALID")
    _parse_time(value.get("created_at"), field="state.created_at")
    _parse_time(value.get("updated_at"), field="state.updated_at")
    started, ended = value.get("started_at"), value.get("ended_at")
    if started is not None:
        _parse_time(started, field="state.started_at")
    if ended is not None:
        _parse_time(ended, field="state.ended_at")
    if started is not None and ended is not None and _parse_time(
        ended, field="state.ended_at"
    ) < _parse_time(started, field="state.started_at"):
        raise _fail("S308_TIMED_STATE_TIME_ORDER_INVALID")
    return dict(value)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _now()
    state["state_hash"] = _canonical_hash(
        {key: item for key, item in state.items() if key != "state_hash"}
    )
    _write_atomic(path, state)


def _validate_result(
    root: Path,
    path: Path,
    *,
    config_hash: str,
) -> tuple[TaskRunResult, dict[str, str], dict[str, str]]:
    try:
        result = TaskRunResult.from_mapping(_load_json(path))
    except (OSError, TypeError, ValueError, Stage3OrchestratorError) as error:
        raise _fail("S308_TIMED_RESULT_INVALID") from error
    wire = result.to_dict()
    refs = wire.get("artifact_refs")
    if (
        wire.get("task_id") != TASK_ID
        or wire.get("config_hash") != config_hash
        or wire.get("status") != "PASS"
        or wire.get("formal_eligible") is not True
        or not isinstance(refs, Mapping)
        or set(refs) != OUTPUT_KINDS
        or any(not isinstance(item, str) or not item for item in refs.values())
    ):
        raise _fail("S308_TIMED_RESULT_NOT_FORMAL_PASS")
    artifact_refs = {str(key): str(item) for key, item in refs.items()}
    artifact_hashes: dict[str, str] = {}
    for kind, reference in artifact_refs.items():
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise _fail("S308_TIMED_OUTPUT_COMMIT_INVALID", kind) from error
        if (
            loaded.identity.task_id != TASK_ID
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != config_hash
        ):
            raise _fail("S308_TIMED_OUTPUT_IDENTITY_INVALID", kind)
        artifact_hashes[kind] = loaded.identity.artifact_hash
    return result, artifact_refs, artifact_hashes


def _handoff_audit(
    arguments: argparse.Namespace,
    *,
    data_root: Path,
) -> Mapping[str, Any]:
    report = run_audit(
        SimpleNamespace(
            manifest=arguments.s307_manifest,
            workspace_root=arguments.s307_workspace_root,
            data_root=data_root,
            s307_environment=arguments.s307_environment,
            s308_config=arguments.s308_config,
            s308_environment=arguments.s308_environment,
            python_executable=arguments.python_executable,
        )
    )
    if report.get("status") == "COMPLETE":
        target = report.get("s308_target")
        if not isinstance(target, Mapping):
            raise _fail("S308_TIMED_HANDOFF_TARGET_INVALID")
        config = ResolvedConfigV2.from_mapping(_load_json(arguments.s308_config))
        if target.get("config_hash") != config.config_hash:
            raise _fail("S308_TIMED_HANDOFF_CONFIG_MISMATCH")
    return report


def _task_command(
    python_executable: Path,
    *,
    config: Path,
    environment: Path,
    result: Path,
) -> list[str]:
    return [
        str(python_executable), "-m", "param_importance_nlp", "task", "run",
        "--config", str(config), "--environment", str(environment),
        "--result", str(result),
    ]


def _task_environment(data_root: Path, repository_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    prefixes = [
        str(repository_root),
        str(repository_root / "src"),
        "/usr/lib/python3/dist-packages",
    ]
    current = environment.get("PYTHONPATH")
    if current:
        prefixes.append(current)
    environment["PYTHONPATH"] = os.pathsep.join(prefixes)
    environment["PARAM_IMPORTANCE_DATA_ROOT"] = str(data_root)
    environment["TMPDIR"] = str(data_root / "tmp")
    return environment


def _load_receipt(path: Path, *, launch_hash: str) -> Mapping[str, Any]:
    receipt = _load_json(path)
    supplied = receipt.get("receipt_hash")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "PASS"
        or receipt.get("formal_eligible") is not True
        or receipt.get("launch_hash") != launch_hash
        or supplied != _canonical_hash(
            {key: item for key, item in receipt.items() if key != "receipt_hash"}
        )
    ):
        raise _fail("S308_TIMED_RECEIPT_INVALID")
    return receipt


def run_timed(
    arguments: argparse.Namespace,
    *,
    execute: Callable[..., Any] = subprocess.run,
) -> Mapping[str, Any]:
    data_root = arguments.data_root.resolve()
    if not data_root.is_dir() or data_root.is_symlink():
        raise _fail("S308_TIMED_DATA_ROOT_INVALID")
    repository_root, repository_ref = _existing_input(
        data_root, arguments.repository_root, field="repository_root"
    )
    if not repository_root.is_dir():
        raise _fail("S308_TIMED_REPOSITORY_INVALID")
    paths: dict[str, Path] = {}
    refs: dict[str, str] = {"repository_root": repository_ref}
    for name, value, must_exist in (
        ("s307_manifest", arguments.s307_manifest, True),
        ("s307_workspace_root", arguments.s307_workspace_root, True),
        ("s307_environment", arguments.s307_environment, True),
        ("s308_config", arguments.s308_config, True),
        ("s308_environment", arguments.s308_environment, True),
        ("python_executable", arguments.python_executable, True),
        ("s308_result", arguments.s308_result, False),
        ("state", arguments.state, False),
        ("receipt", arguments.receipt, False),
    ):
        path, ref = (
            _existing_input(data_root, value, field=name)
            if must_exist
            else _logical_path(data_root, value, field=name)
        )
        paths[name], refs[name] = path, ref
    if PurePosixPath(refs["state"]).parts[0] != "runs":
        raise _fail("S308_TIMED_STATE_REF_INVALID")
    if PurePosixPath(refs["receipt"]).parts[0] != "results":
        raise _fail("S308_TIMED_RECEIPT_REF_INVALID")
    if paths["state"] == paths["receipt"] or paths["s308_result"] in {
        paths["state"], paths["receipt"]
    }:
        raise _fail("S308_TIMED_OUTPUT_COLLISION")

    config = ResolvedConfigV2.from_mapping(_load_json(paths["s308_config"]))
    environment = TaskRuntimeEnvironment.from_mapping(
        _load_json(paths["s308_environment"])
    )
    if config.task_id != TASK_ID or config.run_intent != "formal":
        raise _fail("S308_TIMED_CONFIG_INVALID")
    expected_commit = _git_commit(arguments.expected_git_commit)
    git_commit, git_branch = _git_snapshot(repository_root, expected_commit)
    launch_hash = _launch_identity(arguments, refs=refs)
    # Bind the canonicalized interpreter path into both the audit and task
    # launch; a relative or replaced executable cannot escape the launch hash.
    arguments.python_executable = paths["python_executable"]
    audit = _handoff_audit(arguments, data_root=data_root)
    if arguments.dry_validate:
        status = "READY" if audit.get("status") == "COMPLETE" else "IN_PROGRESS"
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "formal_eligible": status == "READY",
            "launch_hash": launch_hash,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "config_hash": config.config_hash,
            "environment_hash": environment.environment_hash,
            "handoff_status": audit.get("status"),
            "handoff_audit_hash": audit.get("audit_hash"),
            "completed_unit_count": audit.get("completed_unit_count"),
            "required_unit_count": audit.get("required_unit_count"),
        }
        report["receipt_hash"] = _canonical_hash(report)
        return report
    if audit.get("status") != "COMPLETE":
        raise _fail("S308_TIMED_HANDOFF_NOT_COMPLETE", audit.get("status"))

    lock = InstanceLock(paths["state"].with_suffix(paths["state"].suffix + ".lock"))
    with lock:
        if paths["state"].exists():
            state = _validate_state(
                _load_json(paths["state"]),
                launch_hash=launch_hash,
                config_hash=config.config_hash,
                environment_hash=environment.environment_hash,
                git_commit=git_commit,
                git_branch=git_branch,
            )
        else:
            if paths["s308_result"].exists() or paths["receipt"].exists():
                raise _fail("S308_TIMED_UNJOURNALED_OUTPUT_PRESENT")
            state = _new_state(
                launch_hash=launch_hash,
                config_hash=config.config_hash,
                environment_hash=environment.environment_hash,
                git_commit=git_commit,
                git_branch=git_branch,
            )
            _save_state(paths["state"], state)

        if state["phase"] == "FAILED":
            raise _fail("S308_TIMED_PREVIOUS_ATTEMPT_FAILED")
        if state["phase"] == "COMPLETE":
            receipt = _load_receipt(paths["receipt"], launch_hash=launch_hash)
            if receipt.get("receipt_hash") != state.get("receipt_hash"):
                raise _fail("S308_TIMED_STATE_RECEIPT_MISMATCH")
            return receipt

        recovered = state["phase"] == "RUNNING" and paths["s308_result"].is_file()
        if state["phase"] == "RUNNING" and not recovered and not arguments.resume_interrupted:
            raise _fail("S308_TIMED_INTERRUPTED_ATTEMPT_REQUIRES_EXPLICIT_RESUME")
        if state["phase"] == "PREPARED" or (
            state["phase"] == "RUNNING" and not recovered
        ):
            started_at = _now()
            if state["started_at"] is None:
                state["started_at"] = started_at
            attempt = {
                "attempt_number": len(state["attempts"]) + 1,
                "started_at": started_at,
                "ended_at": None,
                "returncode": None,
            }
            state["attempts"].append(attempt)
            state["phase"] = "RUNNING"
            state["handoff_audit_hash"] = audit["audit_hash"]
            _save_state(paths["state"], state)
            completed = execute(
                _task_command(
                    paths["python_executable"],
                    config=paths["s308_config"],
                    environment=paths["s308_environment"],
                    result=paths["s308_result"],
                ),
                cwd=repository_root,
                env=_task_environment(data_root, repository_root),
                check=False,
            )
            ended_at = _now()
            attempt["ended_at"] = ended_at
            attempt["returncode"] = int(completed.returncode)
            state["ended_at"] = ended_at
            state["ended_at_source"] = "wrapper_post_wait"
            if completed.returncode != 0:
                state["phase"] = "FAILED"
                _save_state(paths["state"], state)
                raise _fail("S308_TIMED_TASK_EXIT", completed.returncode)
            _save_state(paths["state"], state)
        elif recovered:
            ended_at = _result_mtime(paths["s308_result"])
            if _parse_time(ended_at, field="result_mtime") < _parse_time(
                state["started_at"], field="state.started_at"
            ):
                raise _fail("S308_TIMED_RECOVERED_TIME_ORDER_INVALID")
            state["ended_at"] = ended_at
            state["ended_at_source"] = "result_mtime_recovery"
            state["recovered"] = True
            if state["attempts"]:
                state["attempts"][-1]["ended_at"] = ended_at
                state["attempts"][-1]["returncode"] = 0
            _save_state(paths["state"], state)

        _confirm_git_snapshot(
            repository_root,
            expected_commit=state["git_commit"],
            expected_branch=state["git_branch"],
        )
        result, artifact_refs, artifact_hashes = _validate_result(
            data_root, paths["s308_result"], config_hash=config.config_hash
        )
        state["phase"] = "S308_COMPLETE"
        state["result_hash"] = result.result_hash
        _save_state(paths["state"], state)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "scope": "formal",
            "formal_eligible": True,
            "launch_hash": launch_hash,
            "task_id": TASK_ID,
            "config_ref": refs["s308_config"],
            "config_hash": config.config_hash,
            "environment_ref": refs["s308_environment"],
            "environment_hash": environment.environment_hash,
            "result_ref": refs["s308_result"],
            "result_hash": result.result_hash,
            "artifact_refs": artifact_refs,
            "artifact_hashes": artifact_hashes,
            "git_commit": state["git_commit"],
            "git_branch": state["git_branch"],
            "started_at": state["started_at"],
            "ended_at": state["ended_at"],
            "ended_at_source": state["ended_at_source"],
            "recovered": state["recovered"],
            "handoff_audit_hash": audit["audit_hash"],
            "receipt_ref": refs["receipt"],
        }
        receipt["receipt_hash"] = _canonical_hash(receipt)
        publish_canonical_immutable(paths["receipt"], receipt)
        state["phase"] = "COMPLETE"
        state["receipt_ref"] = refs["receipt"]
        state["receipt_hash"] = receipt["receipt_hash"]
        _save_state(paths["state"], state)
        return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--s307-manifest", type=Path, required=True)
    parser.add_argument("--s307-workspace-root", type=Path, required=True)
    parser.add_argument("--s307-environment", type=Path, required=True)
    parser.add_argument("--s308-config", type=Path, required=True)
    parser.add_argument("--s308-environment", type=Path, required=True)
    parser.add_argument("--s308-result", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--dry-validate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_timed(_parser().parse_args(argv))
    except (
        OSError,
        TypeError,
        ValueError,
        Stage3OrchestratorError,
    ) as error:
        blocked: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        blocked["receipt_hash"] = _canonical_hash(blocked)
        print(canonical_json_bytes(blocked).decode("utf-8"), end="")
        return 2
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report.get("status") in {"PASS", "READY"} else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "Stage3S308TimedError",
    "_git_commit",
    "_new_state",
    "_result_mtime",
    "_save_state",
    "_task_command",
    "_validate_state",
    "main",
    "run_timed",
]
