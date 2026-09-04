"""Materialize one command-backed Stage 3 G3-8 Git synchronization role.

The command never runs Git, pushes, or changes a checkout.  It consumes the
record of an already completed role check, rejects mismatched three-end heads
or dirty worktrees, hashes the exact six Agent policy documents and stdout log,
and publishes one canonical evidence file immutably.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import (  # noqa: E402
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.runtime import publish_canonical_immutable  # noqa: E402


SOURCE_SCHEMA = "stage3-g38-git-sync-source-v1"
OUTPUT_SCHEMA = "stage3-g38-git-sync-evidence-v1"
ROLES = ("branch", "commit", "push", "remote", "server_clean_head", "sync")
AGENT_DOCUMENTS = (
    "Agent/git.md",
    "Agent/local.md",
    "Agent/remote_access.md",
    "Agent/server.md",
    "Agent/sync.md",
    "Agent/worklogs.md",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic|g3[-_.]?9|future)", re.IGNORECASE)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], *, required: set[str], optional: set[str] = frozenset(), field: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise ValueError(f"{field} fields mismatch: missing={missing}, unknown={unknown}")


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None or _FORBIDDEN_RE.search(value):
        raise ValueError(f"{field} is not a safe formal identifier")
    return value


def _branch(value: object) -> str:
    if (
        not isinstance(value, str)
        or _BRANCH_RE.fullmatch(value) is None
        or value.endswith("/")
        or "//" in value
        or ".." in value
        or _FORBIDDEN_RE.search(value)
    ):
        raise ValueError("branch is not a safe formal Git branch")
    return value


def _safe_ref(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "?" in value
        or "://" in value
        or "\\" in value
        or _FORBIDDEN_RE.search(value)
    ):
        raise ValueError(f"{field} is not a stable formal workspace ref")
    ref = PurePosixPath(value)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field} escapes the workspace")
    return ref.as_posix()


def _timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("checked_at must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("checked_at is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("checked_at is not UTC")
    return value


def _regular_workspace_file(root: Path, ref: str, *, field: str) -> tuple[Path, bytes]:
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} contains a symlink")
    try:
        resolved = root.joinpath(*PurePosixPath(ref).parts).resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{field} is not an existing workspace file") from error
    if not resolved.is_file():
        raise ValueError(f"{field} is not a regular file")
    return resolved, resolved.read_bytes()


def _file_record(root: Path, value: object, *, field: str) -> dict[str, JSONValue]:
    if isinstance(value, str):
        source: Mapping[str, object] = {"path": value}
    else:
        source = _mapping(value, field=field)
    _exact_keys(source, required={"path"}, optional={"role", "source_refs"}, field=field)
    ref = _safe_ref(source["path"], field=f"{field}.path")
    _resolved, payload = _regular_workspace_file(root, ref, field=f"{field}.path")
    record: dict[str, JSONValue] = {
        "path": ref,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    if "role" in source:
        record["role"] = _safe_id(source["role"], field=f"{field}.role")
    if "source_refs" in source:
        raw_refs = source["source_refs"]
        if not isinstance(raw_refs, list) or any(not isinstance(item, str) for item in raw_refs):
            raise TypeError(f"{field}.source_refs must be a string list")
        refs = [_safe_ref(item, field=f"{field}.source_refs") for item in raw_refs]
        if len(refs) != len(set(refs)):
            raise ValueError(f"{field}.source_refs contains duplicates")
        record["source_refs"] = refs
    return record


def _agent_document_hashes(root: Path) -> dict[str, JSONValue]:
    hashes: dict[str, JSONValue] = {}
    for ref in AGENT_DOCUMENTS:
        _resolved, payload = _regular_workspace_file(root, ref, field=ref)
        hashes[ref] = hashlib.sha256(payload).hexdigest()
    return hashes


def materialize_stage3_git_sync_evidence(
    *,
    workspace_root: str | Path,
    source: Mapping[str, object],
    output: str | Path,
) -> Mapping[str, JSONValue]:
    root = Path(workspace_root).resolve(strict=True)
    required = {
        "schema_version",
        "evidence_id",
        "role",
        "checked_at",
        "branch",
        "local_commit",
        "remote_commit",
        "server_commit",
        "remote_name",
        "local_delivery_worktree_clean",
        "server_worktree_clean",
        "command",
        "returncode",
        "stdout_log",
    }
    _exact_keys(source, required=required, field="source")
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("STAGE3_G38_GIT_SYNC_SOURCE_SCHEMA_UNSUPPORTED")
    evidence_id = _safe_id(source["evidence_id"], field="evidence_id")
    role = source["role"]
    if role not in ROLES:
        raise ValueError("Git evidence role is unsupported")
    checked_at = _timestamp(source["checked_at"])
    branch = _branch(source["branch"])
    commits: list[str] = []
    for name in ("local_commit", "remote_commit", "server_commit"):
        value = source[name]
        if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
            raise ValueError(f"{name} must be a full Git commit")
        commits.append(value)
    if len(set(commits)) != 1:
        raise ValueError("local, remote, and server commits do not match")
    remote_name = _safe_id(source["remote_name"], field="remote_name")
    if source["local_delivery_worktree_clean"] is not True or source["server_worktree_clean"] is not True:
        raise ValueError("both delivery and server worktrees must be clean")
    command = source["command"]
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise TypeError("command must be a non-empty string list")
    returncode = source["returncode"]
    if isinstance(returncode, bool) or returncode != 0:
        raise ValueError("only a zero-returncode Git check can be materialized")
    stdout_log = _file_record(root, source["stdout_log"], field="stdout_log")
    body: dict[str, JSONValue] = {
        "schema_version": OUTPUT_SCHEMA,
        "evidence_id": evidence_id,
        "role": role,
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "checked_at": checked_at,
        "branch": branch,
        "local_commit": commits[0],
        "remote_commit": commits[1],
        "server_commit": commits[2],
        "remote_name": remote_name,
        "local_delivery_worktree_clean": True,
        "server_worktree_clean": True,
        "agent_document_hashes": _agent_document_hashes(root),
        "command": list(command),
        "returncode": 0,
        "stdout_log": stdout_log,
    }
    payload = body | {"artifact_hash": canonical_json_hash(body)}
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    target = Path(os.path.abspath(target))
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("output must stay inside workspace_root") from error
    publish_canonical_immutable(target, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = load_canonical_json(args.source)
    payload = materialize_stage3_git_sync_evidence(
        workspace_root=args.workspace_root,
        source=_mapping(source, field="source"),
        output=args.output,
    )
    target = args.output
    if target.is_absolute():
        output_ref = target.resolve().relative_to(args.workspace_root.resolve()).as_posix()
    else:
        output_ref = PurePosixPath(target.as_posix()).as_posix()
    print(json.dumps({"status": "PASS", "role": payload["role"], "output_ref": output_ref, "artifact_hash": payload["artifact_hash"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
