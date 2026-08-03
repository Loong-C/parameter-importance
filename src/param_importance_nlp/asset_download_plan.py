"""Frozen, URL-free acquisition plan for the missing Stage 0 G3 objects.

The committed plan contains only stable Hugging Face logical source IDs,
immutable revisions, expected sizes/hashes, and DATA_ROOT-relative targets.
Runtime HTTP endpoints are derived in memory and are never persisted in the
plan, reports, command-line arguments, or manifests.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Final
from urllib.parse import quote

from .asset_acquisition import (
    AcquisitionPolicy,
    AssetAcquisitionError,
    AssetObjectSpec,
    acquire_http_asset,
    resolve_approved_asset_target,
)
from .asset_layout import validate_stage0_asset_layout
from .asset_requirements import validate_stage0_asset_requirements
from .assets import validate_asset_path
from .atomic import atomic_write_json
from .contracts.jsonio import (
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
    loads_strict_json,
)
from .storage import is_within, require_data_root


SCHEMA_VERSION: Final = "stage0-g3-download-plan-v1"
REPORT_SCHEMA_VERSION: Final = "stage0-g3-download-report-v1"
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT: Final = re.compile(r"^[0-9a-f]{40}$")
_TOP_FIELDS: Final = frozenset(
    {
        "schema_version",
        "artifact_hash",
        "created_at",
        "generator_git_commit",
        "requirements_ref",
        "requirements_sha256",
        "layout_ref",
        "layout_sha256",
        "entries",
    }
)
_ENTRY_FIELDS: Final = frozenset(
    {"object_id", "spec_ref", "asset_root_ref", "final_path"}
)


class AssetDownloadPlanError(ValueError):
    """Raised when a committed acquisition plan is unsafe or inconsistent."""


def download_plan_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AssetDownloadPlanError(f"{field} must be an object with string keys")
    return dict(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AssetDownloadPlanError(f"{field} must be normalized non-empty text")
    if any(ord(character) < 32 for character in value):
        raise AssetDownloadPlanError(f"{field} contains a control character")
    return value


def _digest(value: Any, *, field: str) -> str:
    result = _text(value, field=field)
    if _SHA256.fullmatch(result) is None:
        raise AssetDownloadPlanError(f"{field} must be a lowercase SHA-256")
    return result


def _source_relative_path(value: Any, *, field: str) -> str:
    result = _text(value, field=field)
    if "\\" in result or ":" in result:
        raise AssetDownloadPlanError(f"{field} must use relative POSIX syntax")
    path = PurePosixPath(result)
    if path.is_absolute() or str(path) != result or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise AssetDownloadPlanError(f"{field} is not a normalized relative path")
    return result


def validate_g3_download_plan(
    value: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
) -> None:
    plan = _mapping(value, field="download plan")
    if set(plan) != _TOP_FIELDS:
        raise AssetDownloadPlanError("download plan fields are not exact")
    if plan["schema_version"] != SCHEMA_VERSION:
        raise AssetDownloadPlanError("unsupported download plan schema_version")
    if _digest(plan["artifact_hash"], field="artifact_hash") != (
        download_plan_artifact_hash(plan)
    ):
        raise AssetDownloadPlanError("download plan artifact_hash mismatch")
    created_at = _text(plan["created_at"], field="created_at")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssetDownloadPlanError("created_at must be ISO-8601") from error
    if timestamp.tzinfo is None:
        raise AssetDownloadPlanError("created_at must include a timezone")
    commit = _text(plan["generator_git_commit"], field="generator_git_commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise AssetDownloadPlanError("generator_git_commit must be a Git commit")
    _source_relative_path(plan["requirements_ref"], field="requirements_ref")
    requirements_hash = _digest(
        plan["requirements_sha256"], field="requirements_sha256"
    )
    _source_relative_path(plan["layout_ref"], field="layout_ref")
    layout_hash = _digest(plan["layout_sha256"], field="layout_sha256")

    entries = plan["entries"]
    if not isinstance(entries, list) or len(entries) != 13:
        raise AssetDownloadPlanError("download plan must contain exactly 13 objects")
    object_ids: set[str] = set()
    spec_refs: set[str] = set()
    targets: set[tuple[str, str]] = set()
    allowed_roots = (
        {entry["asset_root_ref"] for entry in layout["entries"]}
        if layout is not None
        else None
    )
    for index, raw in enumerate(entries):
        entry = _mapping(raw, field=f"entries[{index}]")
        if set(entry) != _ENTRY_FIELDS:
            raise AssetDownloadPlanError(f"entries[{index}] fields are not exact")
        object_id = _text(entry["object_id"], field=f"entries[{index}].object_id")
        if not object_id.startswith("huggingface/") or "://" in object_id or "?" in object_id:
            raise AssetDownloadPlanError(f"entries[{index}].object_id is not stable")
        spec_ref = _source_relative_path(
            entry["spec_ref"], field=f"entries[{index}].spec_ref"
        )
        if not spec_ref.startswith("configs/stage0/http-objects/"):
            raise AssetDownloadPlanError(f"entries[{index}].spec_ref is outside the freeze")
        root_ref = validate_asset_path(entry["asset_root_ref"])
        if root_ref.split("/", 1)[0] not in {"models", "datasets"}:
            raise AssetDownloadPlanError(f"entries[{index}].asset_root_ref is not an asset root")
        if allowed_roots is not None and root_ref not in allowed_roots:
            raise AssetDownloadPlanError(f"entries[{index}].asset_root_ref is absent from layout")
        final_path = validate_asset_path(entry["final_path"])
        identity = (root_ref, final_path)
        if object_id in object_ids or spec_ref in spec_refs or identity in targets:
            raise AssetDownloadPlanError("download plan identities and targets must be unique")
        object_ids.add(object_id)
        spec_refs.add(spec_ref)
        targets.add(identity)

    if requirements is not None:
        validate_stage0_asset_requirements(requirements)
        if requirements_hash != requirements.get("artifact_hash"):
            raise AssetDownloadPlanError("requirements_sha256 does not bind requirements")
    if layout is not None:
        validate_stage0_asset_layout(layout, requirements=requirements)
        if layout_hash != layout.get("artifact_hash"):
            raise AssetDownloadPlanError("layout_sha256 does not bind layout")


def load_g3_download_plan(
    path: str | Path,
    *,
    requirements: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = ensure_json_object(loads_strict_json(Path(path).read_bytes()), field="download plan")
    result = dict(value)
    validate_g3_download_plan(result, requirements=requirements, layout=layout)
    return result


def _huggingface_runtime_url(spec: AssetObjectSpec) -> str:
    prefix = "huggingface/"
    if not spec.source_id.startswith(prefix):
        raise AssetDownloadPlanError("only frozen Hugging Face source IDs are supported")
    parts = spec.source_id[len(prefix) :].split("/")
    if len(parts) < 3 or any(not part for part in parts):
        raise AssetDownloadPlanError("Hugging Face source ID must include owner/repo/path")
    owner, repository, *object_parts = parts
    encoded_path = "/".join(quote(part, safe="") for part in object_parts)
    return (
        f"https://huggingface.co/{quote(owner, safe='')}/"
        f"{quote(repository, safe='')}/resolve/{quote(spec.revision, safe='')}/"
        f"{encoded_path}"
    )


def _safe_asset_root(data_root: Path, relative_root: str) -> Path:
    logical = PurePosixPath(validate_asset_path(relative_root))
    target = data_root.joinpath(*logical.parts)
    if not is_within(target, data_root):
        raise AssetDownloadPlanError("asset root escapes DATA_ROOT")
    parent = target.parent
    if not parent.is_dir() or parent.is_symlink():
        raise AssetDownloadPlanError("asset root parent is missing or link-like")
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise AssetDownloadPlanError("asset root is not a real directory")
    target.mkdir(mode=0o750, exist_ok=True)
    return target


def execute_g3_download_plan(
    *,
    plan: Mapping[str, Any],
    source_root: str | Path,
    data_root: str | Path,
    report_path: str | Path,
    started_at: str,
    policy: AcquisitionPolicy | None = None,
) -> dict[str, Any]:
    """Execute the frozen plan and atomically publish a redacted report."""

    validate_g3_download_plan(plan)
    normalized_started_at = _text(started_at, field="started_at")
    try:
        parsed_started_at = datetime.fromisoformat(
            normalized_started_at.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise AssetDownloadPlanError("started_at must be ISO-8601") from error
    if parsed_started_at.tzinfo is None:
        raise AssetDownloadPlanError("started_at must include a timezone")
    repository_root = Path(source_root).resolve(strict=True)
    approved_data_root = require_data_root(data_root)
    approved_report_root = approved_data_root / "operations"
    if not approved_report_root.is_dir() or approved_report_root.is_symlink():
        raise AssetDownloadPlanError("DATA_ROOT/operations is missing or link-like")
    report_target = Path(report_path)
    if not report_target.is_absolute():
        raise AssetDownloadPlanError("report_path must be absolute")
    report_target = Path(os.path.abspath(report_target))
    if not is_within(report_target, approved_report_root):
        raise AssetDownloadPlanError("report_path must remain below DATA_ROOT/operations")
    current = approved_report_root
    try:
        relative_report = report_target.relative_to(approved_report_root)
    except ValueError as error:  # pragma: no cover - guarded by is_within
        raise AssetDownloadPlanError("report_path escapes operations") from error
    for part in relative_report.parts[:-1]:
        current = current / part
        if current.exists() and (current.is_symlink() or not current.is_dir()):
            raise AssetDownloadPlanError("report_path has an unsafe parent chain")
    selected_policy = policy or AcquisitionPolicy(
        max_attempts=6,
        request_timeout_seconds=60.0,
        overall_timeout_seconds=6 * 60 * 60,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=30.0,
        lock_timeout_seconds=60.0,
        chunk_size=4 * 1024 * 1024,
    )
    results: list[dict[str, Any]] = []
    for raw_entry in plan["entries"]:
        entry = dict(raw_entry)
        spec_path = repository_root.joinpath(*PurePosixPath(entry["spec_ref"]).parts)
        if not is_within(spec_path, repository_root):
            raise AssetDownloadPlanError("spec path escapes source root")
        spec_value = ensure_json_object(load_canonical_json(spec_path), field=str(spec_path))
        spec = AssetObjectSpec.from_mapping(spec_value)
        if spec.source_id != entry["object_id"]:
            raise AssetDownloadPlanError("plan object_id does not match the object spec")
        asset_root = _safe_asset_root(approved_data_root, entry["asset_root_ref"])
        target = resolve_approved_asset_target(asset_root, entry["final_path"])
        try:
            outcome = acquire_http_asset(
                spec,
                _huggingface_runtime_url(spec),
                target,
                policy=selected_policy,
            )
        except AssetAcquisitionError as error:
            failure_payload = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "status": "FAILED",
                "started_at": normalized_started_at,
                "plan_sha256": plan["artifact_hash"],
                "completed_objects": results,
                "failed_object": {
                    "object_id": entry["object_id"],
                    "asset_root_ref": entry["asset_root_ref"],
                    "final_path": entry["final_path"],
                    "failure": error.report.to_dict(),
                },
                "runtime_urls_persisted": False,
            }
            failure = failure_payload | {
                "artifact_hash": canonical_json_hash(failure_payload)
            }
            atomic_write_json(report_target, failure)
            raise
        results.append(
            {
                "object_id": entry["object_id"],
                "asset_root_ref": entry["asset_root_ref"],
                "final_path": entry["final_path"],
                "result": outcome.to_dict(),
            }
        )
    report_payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "PASS",
        "started_at": normalized_started_at,
        "plan_sha256": plan["artifact_hash"],
        "objects": results,
        "runtime_urls_persisted": False,
    }
    report = report_payload | {"artifact_hash": canonical_json_hash(report_payload)}
    atomic_write_json(report_target, report)
    return report


__all__ = [
    "AssetDownloadPlanError",
    "REPORT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "download_plan_artifact_hash",
    "execute_g3_download_plan",
    "load_g3_download_plan",
    "validate_g3_download_plan",
]
