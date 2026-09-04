"""Build the complete server-side Stage 3 large-artifact inventory."""

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

from param_importance_nlp.contracts.jsonio import JSONValue, canonical_json_hash
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES,
    STAGE3_G38_LARGE_ARTIFACT_MANIFEST_SCHEMA,
)
from param_importance_nlp.runtime import publish_canonical_immutable


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic|future)", re.IGNORECASE)
_FUTURE_GATE_RE = re.compile(r"(?:g3[-_.]?9|stage3[./_-]?g3[-_.]?9)", re.IGNORECASE)


def _safe_id(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or _ID_RE.fullmatch(value) is None
        or _FORBIDDEN_RE.search(value)
    ):
        raise ValueError(f"{field} must be a stable identifier")
    return value


def _safe_ref(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "?" in value
        or "://" in value
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or _FORBIDDEN_RE.search(value)
        or _FUTURE_GATE_RE.search(value)
    ):
        raise ValueError(f"{field} must be a stable POSIX workspace ref")
    ref = PurePosixPath(value)
    if ref.is_absolute() or not ref.parts or any(part in {"", ".", ".."} for part in ref.parts):
        raise ValueError(f"{field} escapes workspace_root")
    return ref.as_posix()


def _utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("generated_at must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("generated_at must be an explicit UTC timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("generated_at must be UTC")
    return value


def _workspace_path(root: Path, ref: str, *, field: str) -> Path:
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} contains a symlink: {ref}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError(f"{field} is not inside workspace_root: {ref}") from error
    return resolved


def _directory_files(root: Path, root_ref: str) -> tuple[Path, ...]:
    resolved = _workspace_path(root, root_ref, field="artifact_root")
    if not resolved.is_dir():
        raise ValueError(f"artifact root is not a directory: {root_ref}")
    files: list[Path] = []

    def walk_error(error: OSError) -> None:
        raise ValueError(f"artifact root walk failed: {root_ref}") from error

    for directory, dirnames, filenames in os.walk(
        resolved,
        topdown=True,
        onerror=walk_error,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in dirnames:
            if (directory_path / name).is_symlink():
                raise ValueError(f"artifact root contains a directory symlink: {root_ref}")
        for name in filenames:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"artifact root contains a non-regular file: {root_ref}")
            files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"artifact root is empty: {root_ref}")
    return tuple(files)


def _hash_stable_file(root: Path, path: Path) -> dict[str, JSONValue]:
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"artifact changed while hashing: {path.relative_to(root).as_posix()}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hasher.hexdigest(),
        "size": before.st_size,
    }


def _target(root: Path, output: str | Path) -> Path:
    target = Path(output)
    if not target.is_absolute():
        target = root / target
    absolute = Path(os.path.abspath(target))
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise ValueError("output must stay inside workspace_root") from error
    return absolute


def materialize_stage3_large_artifact_manifest(
    *,
    workspace_root: str | Path,
    manifest_id: str,
    generated_at: str,
    artifact_roots: Mapping[str, str],
    source_refs: Mapping[str, str],
    source_hashes: Mapping[str, str],
    output: str | Path,
) -> Mapping[str, JSONValue]:
    root = Path(workspace_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace_root must be a directory")
    manifest_id = _safe_id(manifest_id, field="manifest_id")
    generated_at = _utc_timestamp(generated_at)
    if not source_refs or set(source_refs) != set(source_hashes):
        raise ValueError("source_refs and source_hashes must have identical non-empty keys")
    normalized_sources: dict[str, str] = {}
    normalized_source_hashes: dict[str, str] = {}
    for name in sorted(source_refs):
        stable_name = _safe_id(name, field="source_refs key")
        normalized_sources[stable_name] = _safe_ref(source_refs[name], field=f"source_refs.{name}")
        digest = source_hashes[name]
        if not isinstance(digest, str) or _HASH_RE.fullmatch(digest) is None:
            raise ValueError(f"source_hashes.{name} must be SHA-256")
        normalized_source_hashes[stable_name] = digest

    required_roles = set(REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES)
    if not required_roles.issubset(artifact_roots):
        raise ValueError(f"artifact_roots missing roles: {sorted(required_roles - set(artifact_roots))}")
    normalized_roots: dict[str, str] = {}
    root_paths: list[PurePosixPath] = []
    for role in sorted(artifact_roots):
        stable_role = _safe_id(role, field="artifact_roots role")
        root_ref = _safe_ref(artifact_roots[role], field=f"artifact_roots.{role}")
        path = PurePosixPath(root_ref)
        if any(path == prior or path in prior.parents or prior in path.parents for prior in root_paths):
            raise ValueError(f"artifact roots overlap: {root_ref}")
        root_paths.append(path)
        normalized_roots[stable_role] = root_ref

    target = _target(root, output)
    target_ref = PurePosixPath(target.relative_to(root).as_posix())
    if any(target_ref == path or path in target_ref.parents for path in root_paths):
        raise ValueError("output must not be inside an inventoried artifact root")

    groups: list[dict[str, JSONValue]] = []
    total_count = 0
    total_size = 0
    for role, root_ref in normalized_roots.items():
        before_files = _directory_files(root, root_ref)
        records = [_hash_stable_file(root, path) for path in before_files]
        after_files = _directory_files(root, root_ref)
        if before_files != after_files:
            raise RuntimeError(f"artifact root changed while hashing: {root_ref}")
        group_size = sum(int(record["size"]) for record in records)
        group_body: dict[str, JSONValue] = {
            "role": role,
            "root_ref": root_ref,
            "files": records,
            "file_count": len(records),
            "total_size": group_size,
        }
        groups.append(group_body | {"collection_hash": canonical_json_hash(group_body)})
        total_count += len(records)
        total_size += group_size

    body: dict[str, JSONValue] = {
        "schema_version": STAGE3_G38_LARGE_ARTIFACT_MANIFEST_SCHEMA,
        "manifest_id": manifest_id,
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "generated_at": generated_at,
        "source_refs": normalized_sources,
        "source_hashes": normalized_source_hashes,
        "artifact_roots": groups,
        "file_count": total_count,
        "total_size": total_size,
    }
    manifest = body | {"artifact_hash": canonical_json_hash(body)}
    publish_canonical_immutable(target, manifest)
    return manifest


def _pairs(values: Sequence[str], *, field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value or name in result:
            raise ValueError(f"{field} entries must be unique NAME=VALUE pairs")
        result[name] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hash complete Stage 3 server artifact roots into an immutable manifest"
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--manifest-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--artifact-root", action="append", required=True)
    parser.add_argument("--source-ref", action="append", required=True)
    parser.add_argument("--source-hash", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = materialize_stage3_large_artifact_manifest(
        workspace_root=arguments.workspace_root,
        manifest_id=arguments.manifest_id,
        generated_at=arguments.generated_at,
        artifact_roots=_pairs(arguments.artifact_root, field="artifact_root"),
        source_refs=_pairs(arguments.source_ref, field="source_ref"),
        source_hashes=_pairs(arguments.source_hash, field="source_hash"),
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact_hash": manifest["artifact_hash"],
                "file_count": manifest["file_count"],
                "total_size": manifest["total_size"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
