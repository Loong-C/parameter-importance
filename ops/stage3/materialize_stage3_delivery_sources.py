#!/usr/bin/env python3
"""Snapshot tracked Stage 3 delivery code and worklog into stable evidence.

The Stage 3 task artifacts live below DATA_ROOT while the authoritative Git
checkout lives outside it.  G3-8 must not retain paths into a temporary
worktree, so this command copies an exact, clean-commit-bound set of analysis
scripts and one worklog into an immutable DATA_ROOT evidence directory and
publishes a self-hashed manifest for the copy.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _candidate in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from param_importance_nlp.atomic import atomic_write_bytes  # noqa: E402
from param_importance_nlp.contracts.jsonio import (  # noqa: E402
    JSONValue,
    canonical_json_bytes,
    canonical_json_hash,
)
from param_importance_nlp.runtime import publish_canonical_immutable  # noqa: E402


SCHEMA_VERSION = "stage3-g38-delivery-source-snapshot-v1"
MANIFEST_NAME = "source-snapshot-manifest.json"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic|g3[-_.]?9|future)", re.IGNORECASE)


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None or _FORBIDDEN_RE.search(value):
        raise ValueError(f"{field} is not a safe formal identifier")
    return value


def _safe_ref(
    value: object,
    *,
    field: str,
    suffixes: tuple[str, ...] | None = None,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "?" in value
        or "://" in value
        or "\\" in value
        or _FORBIDDEN_RE.search(value)
    ):
        raise ValueError(f"{field} is not a stable formal ref")
    ref = PurePosixPath(value)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field} escapes its root")
    normalized = ref.as_posix()
    if suffixes is not None and not normalized.casefold().endswith(suffixes):
        raise ValueError(f"{field} must end with {suffixes}")
    return normalized


def _git(repository: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_snapshot(repository: Path, *, expected_commit: str) -> tuple[str, str]:
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("repository_root must be an existing non-symlink directory")
    commit = _git(repository, "rev-parse", "HEAD").stdout.strip()
    branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=no").stdout
    if commit != expected_commit or _COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("repository HEAD does not match expected_git_commit")
    if not branch or branch == "HEAD" or status:
        raise ValueError("repository must be on a named clean branch")
    return commit, branch


def _tracked_file(
    repository: Path,
    value: object,
    *,
    field: str,
    suffixes: tuple[str, ...],
) -> tuple[str, Path, bytes]:
    ref = _safe_ref(value, field=field, suffixes=suffixes)
    tracked = _git(repository, "ls-files", "--error-unmatch", "--", ref, check=False)
    if tracked.returncode != 0 or tracked.stdout.strip() != ref:
        raise ValueError(f"{field} is not one exact tracked file")
    current = repository
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} contains a symlink")
    try:
        source = repository.joinpath(*PurePosixPath(ref).parts).resolve(strict=True)
        source.relative_to(repository)
    except (OSError, ValueError) as error:
        raise ValueError(f"{field} is not an existing repository file") from error
    if not source.is_file():
        raise ValueError(f"{field} is not a regular file")
    return ref, source, source.read_bytes()


def _output_root(root: Path, value: object) -> tuple[str, Path]:
    ref = _safe_ref(value, field="output_dir")
    if PurePosixPath(ref).parts[0] not in {"evidence", "results"}:
        raise ValueError("output_dir must be a stable evidence/results ref")
    target = Path(os.path.abspath(root.joinpath(*PurePosixPath(ref).parts)))
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("output_dir must stay inside data_root") from error
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("output_dir contains a symlink")
    return ref, target


def _publish_bytes_immutable(root: Path, target: Path, payload: bytes) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError("source snapshot target escapes data_root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError("source snapshot target contains a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise ValueError(f"immutable source snapshot drift: {target}")
        return
    atomic_write_bytes(target, payload)


def _record(*, source_ref: str, snapshot_ref: str, payload: bytes) -> dict[str, JSONValue]:
    return {
        "source_ref": source_ref,
        "snapshot_ref": snapshot_ref,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def materialize_stage3_delivery_sources(
    *,
    data_root: str | Path,
    repository_root: str | Path,
    expected_git_commit: str,
    snapshot_id: str,
    output_dir: str,
    analysis_scripts: Sequence[str],
    worklog: str,
) -> Mapping[str, JSONValue]:
    root = Path(data_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("data_root must be an existing non-symlink directory")
    root = root.resolve(strict=True)
    repository = Path(repository_root)
    if not isinstance(expected_git_commit, str) or _COMMIT_RE.fullmatch(expected_git_commit) is None:
        raise ValueError("expected_git_commit must be a full lowercase commit")
    commit, branch = _git_snapshot(repository, expected_commit=expected_git_commit)
    repository = repository.resolve(strict=True)
    name = _safe_id(snapshot_id, field="snapshot_id")
    output_ref, target_root = _output_root(root, output_dir)
    if not analysis_scripts or len(analysis_scripts) != len(set(analysis_scripts)):
        raise ValueError("analysis_scripts must be a non-empty unique list")

    sources = [
        _tracked_file(
            repository,
            value,
            field=f"analysis_scripts[{index}]",
            suffixes=(".py",),
        )
        for index, value in enumerate(analysis_scripts)
    ]
    worklog_ref, _worklog_path, worklog_payload = _tracked_file(
        repository,
        worklog,
        field="worklog",
        suffixes=(".md",),
    )
    if worklog_ref in {ref for ref, _path, _payload in sources}:
        raise ValueError("worklog collides with an analysis script")

    script_records: list[dict[str, JSONValue]] = []
    for source_ref, _source_path, payload in sources:
        snapshot_ref = (
            PurePosixPath(output_ref) / "analysis" / PurePosixPath(source_ref)
        ).as_posix()
        _publish_bytes_immutable(
            root,
            root.joinpath(*PurePosixPath(snapshot_ref).parts),
            payload,
        )
        script_records.append(
            _record(source_ref=source_ref, snapshot_ref=snapshot_ref, payload=payload)
        )
    worklog_snapshot_ref = (
        PurePosixPath(output_ref) / "worklog" / PurePosixPath(worklog_ref)
    ).as_posix()
    _publish_bytes_immutable(
        root,
        root.joinpath(*PurePosixPath(worklog_snapshot_ref).parts),
        worklog_payload,
    )
    worklog_record = _record(
        source_ref=worklog_ref,
        snapshot_ref=worklog_snapshot_ref,
        payload=worklog_payload,
    )

    after_commit, after_branch = _git_snapshot(
        repository, expected_commit=expected_git_commit
    )
    if (after_commit, after_branch) != (commit, branch):
        raise ValueError("repository identity changed during source snapshot")
    for source_ref, source_path, payload in (*sources, (worklog_ref, _worklog_path, worklog_payload)):
        if source_path.read_bytes() != payload:
            raise ValueError(f"repository source changed during snapshot: {source_ref}")

    body: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": name,
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "producer_commit": commit,
        "repository_branch": branch,
        "snapshot_root": output_ref,
        "analysis_scripts": script_records,
        "worklog": worklog_record,
    }
    manifest = body | {"artifact_hash": canonical_json_hash(body)}
    manifest_ref = (PurePosixPath(output_ref) / MANIFEST_NAME).as_posix()
    publish_canonical_immutable(
        root.joinpath(*PurePosixPath(manifest_ref).parts), manifest
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-script", action="append", default=[], required=True)
    parser.add_argument("--worklog", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = materialize_stage3_delivery_sources(
        data_root=arguments.data_root,
        repository_root=arguments.repository_root,
        expected_git_commit=arguments.expected_git_commit,
        snapshot_id=arguments.snapshot_id,
        output_dir=arguments.output_dir,
        analysis_scripts=arguments.analysis_script,
        worklog=arguments.worklog,
    )
    manifest_ref = (
        PurePosixPath(arguments.output_dir) / MANIFEST_NAME
    ).as_posix()
    print(
        canonical_json_bytes(
            {
                "status": "PASS",
                "manifest_ref": manifest_ref,
                "artifact_hash": manifest["artifact_hash"],
                "producer_commit": manifest["producer_commit"],
                "analysis_script_refs": [
                    item["snapshot_ref"]
                    for item in manifest["analysis_scripts"]
                ],
                "worklog_ref": manifest["worklog"]["snapshot_ref"],
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
