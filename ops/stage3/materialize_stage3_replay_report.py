"""Materialize one hash-bound Stage 3 G3-8 replay report.

The command consumes a canonical source record produced after a real replay.
It does not run tests and cannot turn a failed/skipped replay into PASS.  It
reopens every evidence file, computes its byte size/SHA-256, derives the only
allowed cache mode from the replay layer, and publishes the final report
immutably.
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


SOURCE_SCHEMA = "stage3-g38-replay-source-v1"
OUTPUT_SCHEMA = "stage3-g38-replay-report-v1"
LAYERS = ("local_cpu", "server_locked", "frozen_endpoint_uncached")
CACHE_MODES = {
    "local_cpu": "not_applicable",
    "server_locked": "locked_environment",
    "frozen_endpoint_uncached": "uncached",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
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


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _timestamp(value: object, *, field: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field} is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field} is not UTC")
    return value, parsed


def _workspace_payload(root: Path, ref: str, *, field: str) -> bytes:
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
    return resolved.read_bytes()


def _file_record(root: Path, value: object, *, field: str, seen: set[str]) -> dict[str, JSONValue]:
    if isinstance(value, str):
        source: Mapping[str, object] = {"path": value}
    else:
        source = _mapping(value, field=field)
    _exact_keys(source, required={"path"}, optional={"role", "source_refs"}, field=field)
    ref = _safe_ref(source["path"], field=f"{field}.path")
    if ref in seen:
        raise ValueError(f"duplicate replay evidence file: {ref}")
    payload = _workspace_payload(root, ref, field=f"{field}.path")
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
    seen.add(ref)
    return record


def materialize_stage3_replay_report(
    *,
    workspace_root: str | Path,
    source: Mapping[str, object],
    output: str | Path,
) -> Mapping[str, JSONValue]:
    root = Path(workspace_root).resolve(strict=True)
    required = {
        "schema_version",
        "replay_id",
        "layer",
        "implementation_commit",
        "environment_hash",
        "command",
        "returncode",
        "started_at",
        "completed_at",
        "test_summary",
        "input_refs",
        "input_hashes",
        "evidence_files",
    }
    _exact_keys(source, required=required, field="source")
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("STAGE3_G38_REPLAY_SOURCE_SCHEMA_UNSUPPORTED")
    replay_id = _safe_id(source["replay_id"], field="replay_id")
    layer = source["layer"]
    if layer not in LAYERS:
        raise ValueError("replay layer is unsupported")
    commit = source["implementation_commit"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("implementation_commit must be a full Git commit")
    environment_hash = _hash(source["environment_hash"], field="environment_hash")
    command = source["command"]
    if not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command):
        raise TypeError("command must be a non-empty string list")
    returncode = source["returncode"]
    if isinstance(returncode, bool) or returncode != 0:
        raise ValueError("only a zero-returncode replay can be materialized")
    started_at, started = _timestamp(source["started_at"], field="started_at")
    completed_at, completed = _timestamp(source["completed_at"], field="completed_at")
    if completed < started:
        raise ValueError("completed_at precedes started_at")

    summary = _mapping(source["test_summary"], field="test_summary")
    summary_fields = {"collected", "passed", "failed", "errors", "skipped"}
    _exact_keys(summary, required=summary_fields, field="test_summary")
    counts = {name: summary[name] for name in summary_fields}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in counts.values()):
        raise TypeError("test summary values must be integers")
    if (
        counts["collected"] <= 0
        or counts["passed"] != counts["collected"]
        or any(counts[name] != 0 for name in ("failed", "errors", "skipped"))
    ):
        raise ValueError("replay test summary is not an all-pass, zero-skip result")

    input_refs_raw = _mapping(source["input_refs"], field="input_refs")
    input_hashes_raw = _mapping(source["input_hashes"], field="input_hashes")
    if not input_refs_raw or set(input_refs_raw) != set(input_hashes_raw):
        raise ValueError("input_refs/input_hashes must have the same non-empty key set")
    input_refs: dict[str, JSONValue] = {}
    input_hashes: dict[str, JSONValue] = {}
    for name in sorted(input_refs_raw):
        key = _safe_id(name, field="input name")
        ref = _safe_ref(input_refs_raw[name], field=f"input_refs.{name}")
        digest = _hash(input_hashes_raw[name], field=f"input_hashes.{name}")
        payload = _workspace_payload(root, ref, field=f"input_refs.{name}")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"input_refs.{name} SHA-256 does not match the workspace file")
        input_refs[key] = ref
        input_hashes[key] = digest

    evidence_raw = source["evidence_files"]
    if not isinstance(evidence_raw, Sequence) or isinstance(evidence_raw, (str, bytes)) or not evidence_raw:
        raise TypeError("evidence_files must be a non-empty list")
    seen: set[str] = set()
    evidence_files = [
        _file_record(root, value, field=f"evidence_files[{index}]", seen=seen)
        for index, value in enumerate(evidence_raw)
    ]
    body: dict[str, JSONValue] = {
        "schema_version": OUTPUT_SCHEMA,
        "replay_id": replay_id,
        "layer": layer,
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "implementation_commit": commit,
        "environment_hash": environment_hash,
        "command": list(command),
        "returncode": 0,
        "started_at": started_at,
        "completed_at": completed_at,
        "cache_mode": CACHE_MODES[str(layer)],
        "test_summary": {name: counts[name] for name in ("collected", "passed", "failed", "errors", "skipped")},
        "input_refs": input_refs,
        "input_hashes": input_hashes,
        "evidence_files": evidence_files,
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
    payload = materialize_stage3_replay_report(
        workspace_root=args.workspace_root,
        source=_mapping(source, field="source"),
        output=args.output,
    )
    target = args.output
    if target.is_absolute():
        output_ref = target.resolve().relative_to(args.workspace_root.resolve()).as_posix()
    else:
        output_ref = PurePosixPath(target.as_posix()).as_posix()
    print(json.dumps({"status": "PASS", "layer": payload["layer"], "output_ref": output_ref, "artifact_hash": payload["artifact_hash"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
