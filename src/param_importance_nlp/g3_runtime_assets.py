"""Fail-closed runtime binding for formally qualified Stage 0 G3 assets.

Formal experiment configuration carries logical asset IDs only.  Physical
manifest/root paths are selected exclusively by the committed Stage 0.04
``asset_resolution`` artifact referenced by
``environment.evidence_refs['g3_resolution']``.  This module validates that
commit against the current requirements/layout, then replays qualified asset
resolution before exposing any file to a provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Final

from .asset_layout import load_stage0_asset_layout
from .asset_requirements import load_stage0_asset_requirements
from .assets import (
    ResolvedAsset,
    load_manifest,
    resolve_qualified_asset,
    validate_asset_path,
    validate_g3_qualification,
)
from .contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    ensure_json_object,
    loads_strict_json,
)
from .data.pythia_mmap import PythiaIndexedDataset, PythiaShardDescriptor
from .g3_gate import GATE_IDS, validate_stage0_g3_resolution
from .providers.pythia_mmap import PythiaSamplingDesign
from .runtime.task_artifacts import load_committed_task_artifact


G3_RESOLUTION_EVIDENCE_KEY: Final = "g3_resolution"
G3_TASK_ID: Final = "stage0.04_assets_and_manifests"
G3_RESOLUTION_KIND: Final = "asset_resolution"
G3_RESOLUTION_SCHEMA_VERSION: Final = "stage0-g3-resolution-audit-v1"
G3_REQUIREMENTS_REF: Final = "configs/stage0/g3-asset-requirements-v1.json"
G3_LAYOUT_REF: Final = "configs/stage0/g3-asset-layout-v1.json"

_G3_CRITICAL_SOURCE_REFS: Final = (
    G3_REQUIREMENTS_REF,
    G3_LAYOUT_REF,
    "configs/stage0/g3-download-plan-v1.json",
    "ops/stage0/attest_g3_materialization.py",
    "ops/stage0/materialize_and_publish_g3.py",
    "ops/stage0/verify_g3_assets.py",
    "schemas/stage0/asset-layout-v1.json",
    "schemas/stage0/asset-requirements-v1.json",
    "schemas/stage0/download-plan-v1.json",
    "schemas/stage0-asset-manifest-v1.json",
    "schemas/stage0-g3-acquisition-report-v1.json",
    "schemas/stage0-g3-verify-only-report-v1.json",
    "src/param_importance_nlp/asset_acquisition.py",
    "src/param_importance_nlp/asset_download_plan.py",
    "src/param_importance_nlp/asset_layout.py",
    "src/param_importance_nlp/asset_requirements.py",
    "src/param_importance_nlp/assets.py",
    "src/param_importance_nlp/atomic.py",
    "src/param_importance_nlp/contracts/__init__.py",
    "src/param_importance_nlp/contracts/jsonio.py",
    "src/param_importance_nlp/data/pythia_mmap.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/g3_asset_publication.py",
    "src/param_importance_nlp/g3_gate.py",
    "src/param_importance_nlp/g3_lifecycle_evidence.py",
    "src/param_importance_nlp/g3_semantic_evidence.py",
    "src/param_importance_nlp/glue_builder.py",
    "src/param_importance_nlp/providers/optional.py",
    "src/param_importance_nlp/runtime/task_artifacts.py",
)

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_G3_REQUIREMENTS_REF_RE: Final = re.compile(
    r"^configs/stage0/g3-asset-requirements-v[1-9][0-9]*\.json$"
)
_G3_LAYOUT_REF_RE: Final = re.compile(
    r"^configs/stage0/g3-asset-layout-v[1-9][0-9]*\.json$"
)
_G3_DOWNLOAD_PLAN_REF_RE: Final = re.compile(
    r"^configs/stage0/g3-download-plan-v[1-9][0-9]*\.json$"
)
_LEGACY_PROVIDER_PATH_FIELDS: Final = (
    "model_manifest_ref",
    "model_root_ref",
    "data_manifest_ref",
    "data_root_ref",
    "tokenizer_manifest_ref",
    "tokenizer_root_ref",
)
_RESOLUTION_ENTRY_FIELDS: Final = frozenset(
    {
        "logical_name",
        "kind",
        "requirement_name",
        "gate_ids",
        "manifest_ref",
        "asset_root_ref",
        "qualification_ref",
        "status",
        "checks",
        "reasons",
        "asset_id",
        "candidate_id",
        "candidate_ref",
        "candidate_sha256",
        "ready_manifest_sha256",
        "qualification_artifact_hash",
        "acquisition_ref",
        "acquisition_sha256",
        "verification_ref",
        "verification_sha256",
        "semantic_evidence_ref",
        "semantic_evidence_sha256",
        "semantic_evidence_artifact_hash",
        "files_checked",
        "bytes_checked",
        "expected_file_policy",
    }
)
_ENTRY_IDENTITY_FIELDS: Final = (
    "logical_name",
    "kind",
    "requirement_name",
    "gate_ids",
    "manifest_ref",
    "asset_root_ref",
    "qualification_ref",
)
_ENTRY_DIGEST_FIELDS: Final = (
    "asset_id",
    "candidate_id",
    "candidate_sha256",
    "ready_manifest_sha256",
    "qualification_artifact_hash",
    "acquisition_sha256",
    "verification_sha256",
    "semantic_evidence_sha256",
    "semantic_evidence_artifact_hash",
)
_ENTRY_REF_FIELDS: Final = (
    "manifest_ref",
    "asset_root_ref",
    "qualification_ref",
    "candidate_ref",
    "acquisition_ref",
    "verification_ref",
    "semantic_evidence_ref",
)
_KNOWN_TRAINING_SPLITS: Final[dict[int, str]] = {
    0: "debug",
    1: "train",
    4: "train",
    5: "train",
}
_KNOWN_FROZEN_SPLITS: Final[dict[int, str]] = {
    2: "sampling_universe",
    3: "probe",
}
_DECLARED_SAMPLING_DESIGNS: Final[dict[int, str]] = {
    0: "without_replacement_frozen_epoch",
    1: "without_replacement_frozen_epoch",
    2: "with_replacement_versioned_universe",
    3: "disjoint_frozen_probe_panel",
    4: "without_replacement_frozen_epoch",
    5: "without_replacement_frozen_epoch",
}


class G3RuntimeAssetError(ValueError):
    """Raised when formal runtime asset lineage is missing, stale, or corrupt."""


def reject_legacy_provider_paths(providers: Mapping[str, Any]) -> None:
    configured = [
        field
        for field in _LEGACY_PROVIDER_PATH_FIELDS
        if providers.get(field) is not None
    ]
    if configured:
        raise G3RuntimeAssetError(
            "G3_RUNTIME_LEGACY_PROVIDER_PATH_FORBIDDEN:" + ",".join(configured)
        )


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise G3RuntimeAssetError(f"G3_RUNTIME_SHA256_INVALID:{field}")
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise G3RuntimeAssetError(f"G3_RUNTIME_OBJECT_INVALID:{field}")
    return dict(value)


def _default_control_plane_paths() -> tuple[Path, Path]:
    source_root = Path(__file__).resolve().parents[2]
    return source_root / G3_REQUIREMENTS_REF, source_root / G3_LAYOUT_REF


def _git_source_root(source_root: str | Path | None = None) -> Path:
    return (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )


def _run_git(
    source_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source_root.as_posix()}",
                "-C",
                str(source_root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_GIT_UNAVAILABLE") from error


def _current_g3_source_head(source_root: str | Path | None = None) -> str:
    root = _git_source_root(source_root)
    completed = _run_git(root, "rev-parse", "HEAD")
    if completed.returncode != 0:
        raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_HEAD_UNAVAILABLE")
    head = completed.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(head) is None:
        raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_HEAD_INVALID")
    return head


def _assert_g3_source_commit_compatible(
    source_commit: str,
    *,
    source_root: str | Path | None = None,
    critical_source_refs: tuple[str, ...] = _G3_CRITICAL_SOURCE_REFS,
) -> None:
    """Accept later unrelated commits, but reject any G3 source drift.

    Stage0.04 binds its manifests and qualification artifacts to the producer
    commit.  Later Stage0 tasks are expected to create commits, so equality to
    the current HEAD would make a valid G3 commit expire immediately.  The
    producer must instead remain an ancestor and every critical G3 source blob
    (including the current index/worktree view) must still be identical.
    """

    if _GIT_COMMIT_RE.fullmatch(source_commit) is None:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_COMMIT_INVALID")
    if (
        type(critical_source_refs) is not tuple
        or any(not isinstance(reference, str) for reference in critical_source_refs)
        or len(set(critical_source_refs)) != len(critical_source_refs)
    ):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE")
    root = _git_source_root(source_root)
    verified = _run_git(root, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
    if verified.returncode != 0 or verified.stdout.strip() != source_commit:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_COMMIT_UNKNOWN")
    head = _current_g3_source_head(root)
    ancestor = _run_git(root, "merge-base", "--is-ancestor", source_commit, head)
    if ancestor.returncode == 1:
        raise G3RuntimeAssetError(
            "G3_RUNTIME_RESOLUTION_SOURCE_COMMIT_NOT_ANCESTOR"
        )
    if ancestor.returncode != 0:
        raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_GIT_UNAVAILABLE")

    for reference in critical_source_refs:
        present = _run_git(root, "cat-file", "-e", f"{source_commit}:{reference}")
        if present.returncode != 0:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_RESOLUTION_SOURCE_REF_ABSENT:{reference}"
            )

    committed = _run_git(
        root,
        "diff",
        "--quiet",
        f"{source_commit}..{head}",
        "--",
        *critical_source_refs,
    )
    if committed.returncode == 1:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_CRITICAL_SOURCE_DRIFT")
    if committed.returncode != 0:
        raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_GIT_UNAVAILABLE")
    for scope in ((), ("--cached",)):
        dirty = _run_git(
            root,
            "diff",
            *scope,
            "--quiet",
            "--",
            *critical_source_refs,
        )
        if dirty.returncode == 1:
            raise G3RuntimeAssetError("G3_RUNTIME_CRITICAL_SOURCE_WORKTREE_DIRTY")
        if dirty.returncode != 0:
            raise G3RuntimeAssetError("G3_RUNTIME_SOURCE_GIT_UNAVAILABLE")


def current_g3_source_refs() -> tuple[str, ...]:
    """Return the critical Stage0.04 source references bound to current HEAD."""

    head = _current_g3_source_head()
    return tuple(f"git-source/{head}/{ref}" for ref in _G3_CRITICAL_SOURCE_REFS)


def _validate_committed_source_refs(
    source_refs: tuple[str, ...],
    resolution: Mapping[str, Any],
) -> str:
    try:
        observed = set(source_refs)
    except TypeError as error:
        raise G3RuntimeAssetError(
            "G3_RUNTIME_RESOLUTION_SOURCE_REF_INVALID"
        ) from error
    if any(not isinstance(reference, str) for reference in source_refs):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REF_INVALID")
    git_refs = tuple(
        reference for reference in source_refs if reference.startswith("git-source/")
    )
    if len(git_refs) != len(set(git_refs)):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE")
    source_commits: set[str] = set()
    git_paths: set[str] = set()
    for reference in git_refs:
        parts = reference.split("/", 2)
        if (
            len(parts) != 3
            or parts[0] != "git-source"
            or _GIT_COMMIT_RE.fullmatch(parts[1]) is None
            or not parts[2]
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REF_INVALID")
        source_commits.add(parts[1])
        git_paths.add(parts[2])
    if len(source_commits) != 1:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_COMMIT_AMBIGUOUS")
    source_commit = next(iter(source_commits))
    requirements_ref = resolution.get("requirements_ref")
    if (
        not isinstance(requirements_ref, str)
        or _G3_REQUIREMENTS_REF_RE.fullmatch(requirements_ref) is None
    ):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REF_INVALID")
    layout_refs = {
        reference for reference in git_paths if _G3_LAYOUT_REF_RE.fullmatch(reference)
    }
    download_plan_refs = {
        reference
        for reference in git_paths
        if _G3_DOWNLOAD_PLAN_REF_RE.fullmatch(reference)
    }
    if len(layout_refs) != 1 or len(download_plan_refs) != 1:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE")
    layout_ref = next(iter(layout_refs))
    download_plan_ref = next(iter(download_plan_refs))
    critical_source_refs = (
        requirements_ref,
        layout_ref,
        download_plan_ref,
        *_G3_CRITICAL_SOURCE_REFS[3:],
    )
    if len(set(critical_source_refs)) != len(critical_source_refs):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE")
    expected_git_refs = {
        f"git-source/{source_commit}/{reference}"
        for reference in critical_source_refs
    }
    if set(git_refs) != expected_git_refs:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_SOURCE_REFS_INCOMPLETE")
    _assert_g3_source_commit_compatible(
        source_commit,
        critical_source_refs=critical_source_refs,
    )

    required_evidence_refs = {requirements_ref}
    raw_entries = resolution.get("entries")
    if not isinstance(raw_entries, list):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_ENTRIES_INVALID")
    for raw_entry in raw_entries:
        entry = _mapping(raw_entry, field="resolution.source_entry")
        required_evidence_refs.update(
            str(entry[field])
            for field in (
                "manifest_ref",
                "asset_root_ref",
                "candidate_ref",
                "qualification_ref",
                "acquisition_ref",
                "verification_ref",
                "semantic_evidence_ref",
            )
        )
    if not required_evidence_refs.issubset(observed):
        raise G3RuntimeAssetError(
            "G3_RUNTIME_RESOLUTION_EVIDENCE_SOURCE_REFS_INCOMPLETE"
        )
    return source_commit


def _length_prefixed(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _manifest_directory_content_hash(manifest: Mapping[str, Any]) -> str:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise G3RuntimeAssetError("G3_RUNTIME_DIRECTORY_INVENTORY_INVALID")
    descriptors = sorted(
        (_mapping(item, field="manifest.files") for item in raw_files),
        key=lambda item: str(item.get("path")),
    )
    digest = hashlib.sha256()
    for descriptor in descriptors:
        path = validate_asset_path(descriptor.get("path"))
        size = descriptor.get("size_bytes")
        sha256 = descriptor.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or _SHA256_RE.fullmatch(sha256) is None
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_DIRECTORY_DESCRIPTOR_INVALID")
        _length_prefixed(digest, path.encode("utf-8"))
        _length_prefixed(digest, str(size).encode("ascii"))
        _length_prefixed(digest, bytes.fromhex(sha256))
    return digest.hexdigest()


def _safe_workspace_path(
    root: Path,
    reference: str,
    *,
    field: str,
    directory: bool,
) -> Path:
    try:
        normalized = validate_asset_path(reference)
    except (TypeError, ValueError) as error:
        raise G3RuntimeAssetError(f"G3_RUNTIME_REF_INVALID:{field}") from error
    logical = PurePosixPath(normalized)
    current = root
    for part in logical.parts:
        current = current / part
        if _is_link_like(current):
            raise G3RuntimeAssetError(f"G3_RUNTIME_LINK_LIKE_FORBIDDEN:{field}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise G3RuntimeAssetError(f"G3_RUNTIME_REF_MISSING_OR_ESCAPE:{field}") from error
    if directory and not resolved.is_dir():
        raise G3RuntimeAssetError(f"G3_RUNTIME_DIRECTORY_REQUIRED:{field}")
    if not directory and not resolved.is_file():
        raise G3RuntimeAssetError(f"G3_RUNTIME_FILE_REQUIRED:{field}")
    return resolved


def _is_link_like(path: Path) -> bool:
    """Return true for symlinks, junctions, and other Windows reparse points."""

    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except (FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _strict_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        return dict(
            ensure_json_object(loads_strict_json(path.read_bytes()), field=field)
        )
    except (OSError, TypeError, ValueError) as error:
        raise G3RuntimeAssetError(f"G3_RUNTIME_JSON_INVALID:{field}") from error


def _validated_resolution_commit_ref(
    root: Path, reference: str
) -> tuple[str, str, str]:
    try:
        normalized = validate_asset_path(reference)
    except (TypeError, ValueError) as error:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_COMMIT_REF_INVALID") from error
    logical = PurePosixPath(normalized)
    if (
        len(logical.parts) < 3
        or logical.name != f"{G3_RESOLUTION_KIND}.json"
        or logical.parent.name != "commits"
    ):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_COMMIT_LAYOUT_INVALID")
    commit_path = _safe_workspace_path(
        root,
        normalized,
        field="g3_resolution.commit_ref",
        directory=False,
    )
    preview = _strict_json_object(commit_path, field="g3_resolution.commit")
    artifact_hash = _require_sha256(
        preview.get("artifact_hash"), field="g3_resolution.commit.artifact_hash"
    )
    object_ref = preview.get("object_ref")
    try:
        normalized_object = validate_asset_path(object_ref)
    except (TypeError, ValueError) as error:
        raise G3RuntimeAssetError(
            "G3_RUNTIME_RESOLUTION_OBJECT_REF_INVALID"
        ) from error
    expected_object = (
        logical.parent.parent
        / "objects"
        / G3_RESOLUTION_KIND
        / f"{artifact_hash}.json"
    )
    if PurePosixPath(normalized_object) != expected_object:
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_STORE_BINDING_INVALID")
    _safe_workspace_path(
        root,
        normalized_object,
        field="g3_resolution.object_ref",
        directory=False,
    )
    return normalized, normalized_object, artifact_hash


def _validated_resolution_entries(
    resolution: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_entries = resolution.get("entries")
    layout_entries = layout.get("entries")
    if not isinstance(raw_entries, list) or not isinstance(layout_entries, list):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_ENTRIES_INVALID")
    if len(raw_entries) != len(layout_entries):
        raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_LAYOUT_COUNT_DRIFT")

    indexed: dict[str, dict[str, Any]] = {}
    for index, (raw_entry, raw_layout_entry) in enumerate(
        zip(raw_entries, layout_entries, strict=True)
    ):
        entry = _mapping(raw_entry, field=f"resolution.entries[{index}]")
        layout_entry = _mapping(
            raw_layout_entry, field=f"layout.entries[{index}]"
        )
        if set(entry) != _RESOLUTION_ENTRY_FIELDS:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_RESOLUTION_ENTRY_FIELDS_INVALID:{index}"
            )
        if any(
            entry[field] != layout_entry[field]
            for field in _ENTRY_IDENTITY_FIELDS
        ):
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_RESOLUTION_LAYOUT_IDENTITY_DRIFT:{index}"
            )
        checks = entry["checks"]
        if (
            entry["status"] != "PASS"
            or entry["reasons"] != []
            or not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_RESOLUTION_ENTRY_NOT_PASS:{index}"
            )
        for field in _ENTRY_DIGEST_FIELDS:
            _require_sha256(entry[field], field=f"entries[{index}].{field}")
        for field in _ENTRY_REF_FIELDS:
            try:
                validate_asset_path(entry[field])
            except (TypeError, ValueError) as error:
                raise G3RuntimeAssetError(
                    f"G3_RUNTIME_RESOLUTION_ENTRY_REF_INVALID:{index}:{field}"
                ) from error
        expected_file_policy = (
            "qualification_bound_derived_inventory"
            if entry["kind"] == "glue_derived"
            else "requirements_exact"
        )
        if (
            isinstance(entry["files_checked"], bool)
            or not isinstance(entry["files_checked"], int)
            or entry["files_checked"] <= 0
            or isinstance(entry["bytes_checked"], bool)
            or not isinstance(entry["bytes_checked"], int)
            or entry["bytes_checked"] <= 0
            or entry["expected_file_policy"] != expected_file_policy
        ):
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_RESOLUTION_ENTRY_AUDIT_INVALID:{index}"
            )
        logical_name = entry["logical_name"]
        if not isinstance(logical_name, str) or logical_name in indexed:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_LOGICAL_ASSET_ID_INVALID:{index}"
            )
        indexed[logical_name] = entry
    return indexed


@dataclass(frozen=True, slots=True)
class QualifiedG3RuntimeAsset:
    """One logical asset replayed through ``resolve_qualified_asset``."""

    logical_name: str
    kind: str
    requirement_name: str
    manifest_ref: str
    asset_root_ref: str
    qualification_ref: str
    resolution_ref: str
    resolution_artifact_hash: str
    source_git_commit: str
    ready_manifest_sha256: str
    qualification_artifact_hash: str
    acquisition_ref: str
    acquisition_sha256: str
    verification_ref: str
    verification_sha256: str
    manifest: Mapping[str, Any]
    qualification: Mapping[str, Any]
    resolved: ResolvedAsset

    @property
    def storage_kind(self) -> str | None:
        metadata = self.manifest.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        storage = metadata.get("storage")
        if not isinstance(storage, Mapping):
            return None
        kind = storage.get("kind")
        return kind if isinstance(kind, str) else None

    @property
    def directory_content_sha256(self) -> str | None:
        if self.storage_kind != "hf_load_from_disk":
            return None
        return _manifest_directory_content_hash(self.manifest)

    @property
    def glue_task_name(self) -> str | None:
        if self.kind != "glue_derived":
            return None
        metadata = _mapping(self.manifest.get("metadata"), field="glue.metadata")
        task_contract = _mapping(
            metadata.get("task_contract"), field="glue.metadata.task_contract"
        )
        task = task_contract.get("task")
        if not isinstance(task, str) or task != self.requirement_name:
            raise G3RuntimeAssetError("G3_RUNTIME_GLUE_TASK_CONTRACT_DRIFT")
        return task

    def require_glue_route(self, *, task_name: str, split: str) -> str:
        expected_task = self.glue_task_name
        if expected_task is None:
            raise G3RuntimeAssetError("G3_RUNTIME_GLUE_DERIVED_ASSET_REQUIRED")
        normalized = task_name.lower().replace("-", "").replace("_", "")
        if normalized != expected_task.lower().replace("-", "").replace("_", ""):
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_GLUE_TASK_ROUTE_MISMATCH:expected={expected_task}:"
                f"observed={task_name}"
            )
        metadata = _mapping(self.manifest.get("metadata"), field="glue.metadata")
        splits = _mapping(metadata.get("splits"), field="glue.metadata.splits")
        if split not in splits:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_GLUE_SPLIT_ROUTE_MISMATCH:{expected_task}:{split}"
            )
        return expected_task

    def provenance(self) -> dict[str, JSONValue]:
        return {
            "schema_version": "qualified-g3-runtime-provenance-v1",
            "g3_resolution_ref": self.resolution_ref,
            "g3_resolution_artifact_hash": self.resolution_artifact_hash,
            "source_git_commit": self.source_git_commit,
            "logical_asset_id": self.logical_name,
            "asset_id": self.resolved.asset_id,
            "manifest_ref": self.manifest_ref,
            "ready_manifest_sha256": self.ready_manifest_sha256,
            "asset_root_ref": self.asset_root_ref,
            "qualification_ref": self.qualification_ref,
            "qualification_artifact_hash": self.qualification_artifact_hash,
            "acquisition_ref": self.acquisition_ref,
            "acquisition_sha256": self.acquisition_sha256,
            "verification_ref": self.verification_ref,
            "verification_sha256": self.verification_sha256,
            "storage_kind": self.storage_kind,
            "directory_content_sha256": self.directory_content_sha256,
        }


class FormalG3RuntimeAssets:
    """Validated Stage0.04 resolution commit and lazily replayed assets."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        resolution_ref: str,
        resolution: Mapping[str, Any],
        requirements: Mapping[str, Any],
        layout: Mapping[str, Any],
        entries: Mapping[str, Mapping[str, Any]],
        source_git_commit: str,
    ) -> None:
        self.workspace_root = workspace_root
        self.resolution_ref = resolution_ref
        self.resolution = deepcopy(dict(resolution))
        self.requirements = deepcopy(dict(requirements))
        self.layout = deepcopy(dict(layout))
        self._entries = {
            name: deepcopy(dict(entry)) for name, entry in entries.items()
        }
        self.source_git_commit = source_git_commit
        self._cache: dict[str, QualifiedG3RuntimeAsset] = {}

    @classmethod
    def from_request(
        cls,
        request: object,
        workspace_root: str | Path,
        *,
        requirements_path: str | Path | None = None,
        layout_path: str | Path | None = None,
    ) -> "FormalG3RuntimeAssets":
        config = getattr(request, "config", None)
        if getattr(config, "run_intent", None) != "formal":
            raise G3RuntimeAssetError("G3_RUNTIME_FORMAL_REQUEST_REQUIRED")
        environment = getattr(request, "environment", None)
        evidence_refs = getattr(environment, "evidence_refs", None)
        if not isinstance(evidence_refs, Mapping):
            raise G3RuntimeAssetError("G3_RUNTIME_EVIDENCE_REFS_INVALID")
        resolution_ref = evidence_refs.get(G3_RESOLUTION_EVIDENCE_KEY)
        if not isinstance(resolution_ref, str) or not resolution_ref:
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_REF_REQUIRED")
        return cls.load(
            workspace_root,
            resolution_ref,
            requirements_path=requirements_path,
            layout_path=layout_path,
        )

    @classmethod
    def load(
        cls,
        workspace_root: str | Path,
        resolution_ref: str,
        *,
        requirements_path: str | Path | None = None,
        layout_path: str | Path | None = None,
    ) -> "FormalG3RuntimeAssets":
        root = Path(workspace_root).resolve()
        default_requirements, default_layout = _default_control_plane_paths()
        requirements_source = Path(requirements_path or default_requirements)
        layout_source = Path(layout_path or default_layout)
        try:
            (
                normalized_resolution_ref,
                preview_object_ref,
                preview_artifact_hash,
            ) = _validated_resolution_commit_ref(root, resolution_ref)
            requirements = load_stage0_asset_requirements(requirements_source)
            layout = load_stage0_asset_layout(
                layout_source, requirements=requirements
            )
            loaded = load_committed_task_artifact(
                root, normalized_resolution_ref, require_formal=True
            )
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, G3RuntimeAssetError):
                raise
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_LOAD_FAILED") from error
        if (
            loaded.identity.task_id != G3_TASK_ID
            or loaded.identity.artifact_kind != G3_RESOLUTION_KIND
            or loaded.identity.commit_ref != normalized_resolution_ref
            or loaded.identity.object_ref != preview_object_ref
            or loaded.identity.artifact_hash != preview_artifact_hash
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_COMMIT_IDENTITY_INVALID")
        resolution = _mapping(loaded.payload, field="g3_resolution")
        try:
            validate_stage0_g3_resolution(resolution)
        except (TypeError, ValueError) as error:
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_PAYLOAD_INVALID") from error
        if (
            resolution.get("schema_version") != G3_RESOLUTION_SCHEMA_VERSION
            or resolution.get("status") != "PASS"
            or resolution.get("requirements_ref") != layout["requirements_ref"]
            or resolution.get("requirements_artifact_hash")
            != requirements["artifact_hash"]
            or resolution.get("layout_artifact_hash") != layout["artifact_hash"]
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_RESOLUTION_STALE_OR_BLOCKED")
        raw_gates = resolution.get("gates")
        if not isinstance(raw_gates, list) or tuple(
            gate.get("gate_id") if isinstance(gate, Mapping) else None
            for gate in raw_gates
        ) != GATE_IDS or any(
            not isinstance(gate, Mapping) or gate.get("status") != "PASS"
            for gate in raw_gates
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_SUBGATES_NOT_PASS")
        entries = _validated_resolution_entries(resolution, layout)
        source_git_commit = _validate_committed_source_refs(
            loaded.source_refs, resolution
        )
        return cls(
            workspace_root=root,
            resolution_ref=normalized_resolution_ref,
            resolution=resolution,
            requirements=requirements,
            layout=layout,
            entries=entries,
            source_git_commit=source_git_commit,
        )

    @property
    def resolution_artifact_hash(self) -> str:
        return str(self.resolution["artifact_hash"])

    def runtime_lineage_sha256(
        self, *assets: QualifiedG3RuntimeAsset
    ) -> str:
        """Bind a runtime/checkpoint to the complete qualified asset session."""

        if len(assets) != 3:
            raise G3RuntimeAssetError("G3_RUNTIME_LINEAGE_REQUIRES_THREE_ASSETS")
        if any(
            not isinstance(asset, QualifiedG3RuntimeAsset) for asset in assets
        ):
            raise TypeError("runtime lineage assets must be qualified G3 assets")
        kinds = [asset.kind for asset in assets]
        if (
            kinds.count("model") != 1
            or kinds.count("tokenizer") != 1
            or sum(kind in {"pile", "glue_derived"} for kind in kinds) != 1
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_LINEAGE_ASSET_KIND_SET_INVALID")
        rows: list[dict[str, str]] = []
        logical_names: set[str] = set()
        for asset in assets:
            if (
                asset.resolution_ref != self.resolution_ref
                or asset.resolution_artifact_hash != self.resolution_artifact_hash
                or asset.source_git_commit != self.source_git_commit
                or asset.logical_name in logical_names
            ):
                raise G3RuntimeAssetError("G3_RUNTIME_LINEAGE_ASSET_BINDING_INVALID")
            logical_names.add(asset.logical_name)
            rows.append(
                {
                    "logical_name": asset.logical_name,
                    "asset_id": asset.resolved.asset_id,
                    "ready_manifest_sha256": asset.ready_manifest_sha256,
                    "qualification_artifact_hash": (
                        asset.qualification_artifact_hash
                    ),
                }
            )
        return canonical_json_hash(
            {
                "schema_version": "formal-g3-runtime-lineage-v1",
                "g3_resolution_artifact_hash": self.resolution_artifact_hash,
                "source_git_commit": self.source_git_commit,
                "assets": sorted(rows, key=lambda item: item["logical_name"]),
            }
        )

    def resolve(
        self,
        logical_asset_id: str,
        *,
        expected_kind: str | None = None,
    ) -> QualifiedG3RuntimeAsset:
        if not isinstance(logical_asset_id, str) or not logical_asset_id:
            raise G3RuntimeAssetError("G3_RUNTIME_LOGICAL_ASSET_ID_REQUIRED")
        cached = self._cache.get(logical_asset_id)
        if cached is not None:
            if expected_kind is not None and cached.kind != expected_kind:
                raise G3RuntimeAssetError("G3_RUNTIME_ASSET_KIND_MISMATCH")
            return cached
        try:
            entry = self._entries[logical_asset_id]
        except KeyError as error:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_LOGICAL_ASSET_UNKNOWN:{logical_asset_id}"
            ) from error
        if expected_kind is not None and entry["kind"] != expected_kind:
            raise G3RuntimeAssetError("G3_RUNTIME_ASSET_KIND_MISMATCH")
        manifest_path = _safe_workspace_path(
            self.workspace_root,
            entry["manifest_ref"],
            field=f"{logical_asset_id}.manifest_ref",
            directory=False,
        )
        qualification_path = _safe_workspace_path(
            self.workspace_root,
            entry["qualification_ref"],
            field=f"{logical_asset_id}.qualification_ref",
            directory=False,
        )
        asset_root = _safe_workspace_path(
            self.workspace_root,
            entry["asset_root_ref"],
            field=f"{logical_asset_id}.asset_root_ref",
            directory=True,
        )
        try:
            manifest = load_manifest(manifest_path)
            qualification = _strict_json_object(
                qualification_path,
                field=f"{logical_asset_id}.qualification",
            )
            validate_g3_qualification(qualification)
            manifest_sha256 = canonical_json_hash(manifest)
            if (
                manifest_sha256 != entry["ready_manifest_sha256"]
                or manifest.get("asset_id") != entry["asset_id"]
                or manifest.get("candidate_id") != entry["candidate_id"]
                or qualification.get("artifact_hash")
                != entry["qualification_artifact_hash"]
                or qualification.get("acquisition_ref") != entry["acquisition_ref"]
                or qualification.get("acquisition_sha256")
                != entry["acquisition_sha256"]
                or qualification.get("verification_ref")
                != entry["verification_ref"]
                or qualification.get("verification_sha256")
                != entry["verification_sha256"]
                or manifest.get("generator_git_commit") != self.source_git_commit
                or qualification.get("generator_git_commit")
                != self.source_git_commit
            ):
                raise G3RuntimeAssetError(
                    f"G3_RUNTIME_RESOLUTION_ASSET_BINDING_DRIFT:{logical_asset_id}"
                )
            resolved = resolve_qualified_asset(
                manifest,
                asset_root,
                qualification,
                qualification_ref=entry["qualification_ref"],
                requirements_artifact_hash=str(
                    self.requirements["artifact_hash"]
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, G3RuntimeAssetError):
                raise
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_QUALIFIED_RESOLUTION_FAILED:{logical_asset_id}"
            ) from error
        result = QualifiedG3RuntimeAsset(
            logical_name=logical_asset_id,
            kind=str(entry["kind"]),
            requirement_name=str(entry["requirement_name"]),
            manifest_ref=str(entry["manifest_ref"]),
            asset_root_ref=str(entry["asset_root_ref"]),
            qualification_ref=str(entry["qualification_ref"]),
            resolution_ref=self.resolution_ref,
            resolution_artifact_hash=self.resolution_artifact_hash,
            source_git_commit=self.source_git_commit,
            ready_manifest_sha256=manifest_sha256,
            qualification_artifact_hash=str(qualification["artifact_hash"]),
            acquisition_ref=str(entry["acquisition_ref"]),
            acquisition_sha256=str(entry["acquisition_sha256"]),
            verification_ref=str(entry["verification_ref"]),
            verification_sha256=str(entry["verification_sha256"]),
            manifest=manifest,
            qualification=qualification,
            resolved=resolved,
        )
        self._cache[logical_asset_id] = result
        return result

    def pile_split_interval(
        self,
        asset: QualifiedG3RuntimeAsset,
        split: str,
    ) -> tuple[int, int]:
        if asset.kind != "pile" or asset.storage_kind != "pythia_mmap_shards":
            raise G3RuntimeAssetError("G3_RUNTIME_PYTHIA_MMAP_ASSET_REQUIRED")
        pile = _mapping(self.requirements.get("pile"), field="requirements.pile")
        intervals = {
            item["name"]: (item["start"], item["stop"])
            for item in pile.get("cursor_intervals", [])
            if isinstance(item, Mapping)
        }
        if split not in intervals:
            raise G3RuntimeAssetError(f"G3_RUNTIME_PILE_SPLIT_UNKNOWN:{split}")
        start, stop = intervals[split]
        metadata = _mapping(asset.manifest.get("metadata"), field="pile.metadata")
        splits = _mapping(metadata.get("splits"), field="pile.metadata.splits")
        split_metadata = _mapping(
            splits.get(split), field=f"pile.metadata.splits.{split}"
        )
        cursor = _mapping(
            split_metadata.get("cursor"),
            field=f"pile.metadata.splits.{split}.cursor",
        )
        if cursor != {"start": start, "stop": stop}:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_PILE_SPLIT_BINDING_DRIFT:{split}"
            )
        return int(start), int(stop)

    def pile_workload(self, stage: int, split: str) -> Mapping[str, Any] | None:
        pile = _mapping(self.requirements.get("pile"), field="requirements.pile")
        matches = [
            dict(item)
            for item in pile.get("workloads", [])
            if isinstance(item, Mapping)
            and item.get("stage") == stage
            and item.get("split") == split
        ]
        if len(matches) > 1:
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_WORKLOAD_DUPLICATE")
        return matches[0] if matches else None

    def validate_pile_budget(
        self,
        *,
        stage: int,
        split: str,
        requested_records: int,
        max_steps: int | None = None,
        global_batch_size: int | None = None,
    ) -> None:
        if (
            isinstance(requested_records, bool)
            or not isinstance(requested_records, int)
            or requested_records <= 0
        ):
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_BUDGET_INVALID")
        pile = _mapping(self.requirements.get("pile"), field="requirements.pile")
        intervals = {
            item["name"]: int(item["stop"]) - int(item["start"])
            for item in pile.get("cursor_intervals", [])
            if isinstance(item, Mapping)
        }
        capacity = intervals.get(split)
        if capacity is None or requested_records > capacity:
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_SPLIT_BUDGET_EXCEEDED")
        workload = self.pile_workload(stage, split)
        if workload is None:
            return
        if requested_records > int(workload["required_unique_records"]):
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_WORKLOAD_BUDGET_EXCEEDED")
        workload_steps = workload.get("max_steps")
        workload_batch = workload.get("global_batch_size")
        if (max_steps is None) != (global_batch_size is None):
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_STEP_BATCH_INCOMPLETE")
        if max_steps is not None and global_batch_size is not None:
            if (
                workload_steps is None
                or workload_batch is None
                or max_steps > int(workload_steps)
                or global_batch_size > int(workload_batch)
                or max_steps * global_batch_size != requested_records
            ):
                raise G3RuntimeAssetError(
                    "G3_RUNTIME_PILE_TRAINING_BUDGET_EXCEEDED"
                )
        if requested_records * 2048 > int(workload["required_target_tokens"]):
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_TOKEN_BUDGET_EXCEEDED")

    def pythia_dataset(
        self,
        asset: QualifiedG3RuntimeAsset,
        *,
        split: str,
    ) -> PythiaIndexedDataset:
        start, stop = self.pile_split_interval(asset, split)
        metadata = _mapping(asset.manifest.get("metadata"), field="pile.metadata")
        storage = _mapping(metadata.get("storage"), field="pile.metadata.storage")
        if storage.get("kind") != "pythia_mmap_shards":
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_STORAGE_KIND_INVALID")
        idx = _mapping(storage.get("idx"), field="pile.metadata.storage.idx")
        raw_shards = storage.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_SHARDS_INVALID")
        shards: list[PythiaShardDescriptor] = []
        for expected_ordinal, raw in enumerate(raw_shards):
            shard = _mapping(
                raw, field=f"pile.metadata.storage.shards[{expected_ordinal}]"
            )
            if shard.get("ordinal") != expected_ordinal:
                raise G3RuntimeAssetError("G3_RUNTIME_PILE_SHARD_ORDER_INVALID")
            shards.append(
                PythiaShardDescriptor(
                    ordinal=expected_ordinal,
                    path=asset.resolved.path_for(str(shard["path"])),
                    size_bytes=int(shard["size_bytes"]),
                    sha256=str(shard["sha256"]),
                )
            )
        try:
            dataset = PythiaIndexedDataset(
                asset.resolved.path_for(str(idx["path"])),
                shards,
                record_start=start,
                record_stop=stop,
                tokens_per_record=int(storage["tokens_per_record"]),
                expected_idx_sha256=str(idx["sha256"]),
            )
            if (
                dataset.index.sequence_count != idx["sequence_count"]
                or dataset.index.document_count != idx["document_count"]
                or dataset.index.dtype_code != idx["dtype_code"]
            ):
                dataset.close()
                raise G3RuntimeAssetError("G3_RUNTIME_PILE_INDEX_IDENTITY_DRIFT")
            return dataset
        except (OSError, TypeError, ValueError) as error:
            if isinstance(error, G3RuntimeAssetError):
                raise
            raise G3RuntimeAssetError("G3_RUNTIME_PILE_DATASET_INVALID") from error


@dataclass(frozen=True, slots=True)
class FormalPileRoute:
    split: str
    sampling_design: PythiaSamplingDesign | None


def formal_pile_route(
    *,
    stage: int,
    evaluation: bool,
    declared_sampling_design: str,
    configured_split: str,
) -> FormalPileRoute:
    """Map a formal stage to one frozen Pile interval and sampling owner."""

    expected_declaration = _DECLARED_SAMPLING_DESIGNS.get(stage)
    if (
        expected_declaration is not None
        and declared_sampling_design != expected_declaration
    ):
        raise G3RuntimeAssetError(
            f"G3_RUNTIME_PILE_SAMPLING_DECLARATION_DRIFT:stage{stage}"
        )
    if stage in _KNOWN_FROZEN_SPLITS:
        if evaluation:
            raise G3RuntimeAssetError("G3_RUNTIME_FROZEN_RESOLVER_EVALUATION_INVALID")
        expected_split = _KNOWN_FROZEN_SPLITS[stage]
        if configured_split != expected_split:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_PILE_SPLIT_DECLARATION_DRIFT:stage{stage}:"
                f"expected={expected_split}:observed={configured_split}"
            )
        return FormalPileRoute(expected_split, None)
    if evaluation:
        if configured_split != "validation":
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_PILE_EVALUATION_SPLIT_DRIFT:{configured_split}"
            )
        return FormalPileRoute("validation", PythiaSamplingDesign.SEQUENTIAL)
    if stage in _KNOWN_TRAINING_SPLITS:
        expected_split = _KNOWN_TRAINING_SPLITS[stage]
        if configured_split != expected_split:
            raise G3RuntimeAssetError(
                f"G3_RUNTIME_PILE_SPLIT_DECLARATION_DRIFT:stage{stage}:"
                f"expected={expected_split}:observed={configured_split}"
            )
        return FormalPileRoute(
            expected_split,
            PythiaSamplingDesign.WITHOUT_REPLACEMENT,
        )
    if configured_split == "validation":
        return FormalPileRoute("validation", PythiaSamplingDesign.SEQUENTIAL)
    if configured_split == "train":
        return FormalPileRoute(
            "train", PythiaSamplingDesign.WITHOUT_REPLACEMENT
        )
    raise G3RuntimeAssetError(
        f"G3_RUNTIME_PILE_STAGE_ROUTE_UNSUPPORTED:stage{stage}:{configured_split}"
    )


__all__ = [
    "FormalG3RuntimeAssets",
    "FormalPileRoute",
    "G3RuntimeAssetError",
    "G3_RESOLUTION_EVIDENCE_KEY",
    "QualifiedG3RuntimeAsset",
    "current_g3_source_refs",
    "formal_pile_route",
    "reject_legacy_provider_paths",
]
