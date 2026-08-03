"""Independent acquisition and verification evidence for Stage 0 G3.

The acquisition attestor consumes the frozen thirteen-object download report,
imports the remaining frozen local assets explicitly, builds the three derived
GLUE datasets, and publishes DOWNLOADED candidates.  The verifier is a separate
zero-network boundary: it consumes only that immutable acquisition chain,
rehashes every declared file, and publishes VERIFIED or INVALID candidates.

Neither artifact persists runtime URLs, credentials, absolute paths, or mutable
"current" pointers.  Publication is create-only and accepts an existing target
only when its canonical bytes are identical, which makes crash recovery safe.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Final

from .asset_acquisition import AssetObjectSpec
from .asset_download_plan import (
    REPORT_SCHEMA_VERSION as DOWNLOAD_REPORT_SCHEMA_VERSION,
    load_g3_download_plan,
    resolve_source_git_commit,
    validate_g3_download_plan,
)
from .asset_layout import load_stage0_asset_layout, validate_stage0_asset_layout
from .asset_requirements import (
    load_stage0_asset_requirements,
    validate_stage0_asset_requirements,
)
from .assets import (
    AssetActorRole,
    AssetFile,
    AssetState,
    build_g3_candidate_manifest,
    compute_asset_id,
    transition_manifest,
    validate_asset_path,
    validate_g3_manifest,
)
from .atomic import sha256_file
from .contracts import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
)
from .glue_builder import (
    GLUE_PREPROCESSING_VERSION,
    MAP_FINGERPRINT_SCHEMA_VERSION,
    GlueDerivedBuildResult,
    build_glue_derived_dataset,
    glue_preprocessing_config_hash,
    normalize_tokenizer_descriptor_inventory,
)


ACQUISITION_SCHEMA_VERSION: Final = "stage0-g3-acquisition-report-v1"
VERIFY_SCHEMA_VERSION: Final = "stage0-g3-verify-only-report-v1"
GENERATOR_VERSION: Final = "stage0-g3-lifecycle-evidence-v1"
EXPECTED_ENTRY_COUNT: Final = 13
EXPECTED_DOWNLOAD_OBJECT_COUNT: Final = 13

# One shared formal source contract for acquisition, verification, and the
# final gate materializer.  Dynamic control-plane refs are added by callers.
G3_CRITICAL_SOURCE_REFS: Final = (
    "ops/stage0/materialize_and_publish_g3.py",
    "ops/stage0/attest_g3_materialization.py",
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
    "src/param_importance_nlp/g3_asset_publication.py",
    "src/param_importance_nlp/g3_gate.py",
    "src/param_importance_nlp/g3_lifecycle_evidence.py",
    "src/param_importance_nlp/g3_semantic_evidence.py",
    "src/param_importance_nlp/glue_builder.py",
    "src/param_importance_nlp/providers/optional.py",
)
G3_CRITICAL_MODULE_ORIGINS: Final = (
    ("param_importance_nlp.asset_acquisition", "src/param_importance_nlp/asset_acquisition.py"),
    ("param_importance_nlp.asset_download_plan", "src/param_importance_nlp/asset_download_plan.py"),
    ("param_importance_nlp.asset_layout", "src/param_importance_nlp/asset_layout.py"),
    ("param_importance_nlp.asset_requirements", "src/param_importance_nlp/asset_requirements.py"),
    ("param_importance_nlp.assets", "src/param_importance_nlp/assets.py"),
    ("param_importance_nlp.atomic", "src/param_importance_nlp/atomic.py"),
    ("param_importance_nlp.contracts", "src/param_importance_nlp/contracts/__init__.py"),
    ("param_importance_nlp.contracts.jsonio", "src/param_importance_nlp/contracts/jsonio.py"),
    ("param_importance_nlp.data.pythia_mmap", "src/param_importance_nlp/data/pythia_mmap.py"),
    (
        "param_importance_nlp.g3_asset_publication",
        "src/param_importance_nlp/g3_asset_publication.py",
    ),
    ("param_importance_nlp.g3_gate", "src/param_importance_nlp/g3_gate.py"),
    (
        "param_importance_nlp.g3_lifecycle_evidence",
        "src/param_importance_nlp/g3_lifecycle_evidence.py",
    ),
    (
        "param_importance_nlp.g3_semantic_evidence",
        "src/param_importance_nlp/g3_semantic_evidence.py",
    ),
    ("param_importance_nlp.glue_builder", "src/param_importance_nlp/glue_builder.py"),
    ("param_importance_nlp.providers.optional", "src/param_importance_nlp/providers/optional.py"),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_ACQUISITION_MODES = frozenset(
    {"canonical-plan", "existing-import", "derived-build"}
)
_RESULT_STATUSES = frozenset(
    {"downloaded", "already_ready", "published_by_peer"}
)
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "credentials",
        "password",
        "api_key",
        "access_token",
        "refresh_token",
        "cookie",
        "runtime_url",
        "signed_url",
        "client_secret",
        "private_key",
        "secret_key",
        "auth_token",
        "presigned_url",
        "x_amz_signature",
    }
)

_ACQUISITION_FIELDS = frozenset(
    {
        "schema_version",
        "formal",
        "status",
        "started_at",
        "completed_at",
        "actor",
        "actor_role",
        "actor_instance_id",
        "source_git_commit",
        "requirements_ref",
        "requirements_sha256",
        "layout_ref",
        "layout_sha256",
        "download_plan_ref",
        "download_plan_sha256",
        "download_report_ref",
        "download_report_sha256",
        "runtime_urls_persisted",
        "entry_count",
        "entries",
        "artifact_hash",
    }
)
_ACQUISITION_ENTRY_FIELDS = frozenset(
    {
        "logical_name",
        "kind",
        "requirement_name",
        "asset_root_ref",
        "mode",
        "asset_type",
        "name",
        "source",
        "revision",
        "asset_id",
        "files",
        "metadata",
        "source_evidence",
    }
)
_VERIFY_FIELDS = frozenset(
    {
        "schema_version",
        "formal",
        "status",
        "checked_at",
        "actor",
        "actor_role",
        "actor_instance_id",
        "generator_git_commit",
        "requirements_ref",
        "requirements_sha256",
        "layout_ref",
        "layout_sha256",
        "acquisition_ref",
        "acquisition_sha256",
        "network_attempts",
        "entry_count",
        "entries",
        "artifact_hash",
    }
)
_VERIFY_ENTRY_FIELDS = frozenset(
    {
        "logical_name",
        "asset_id",
        "candidate_id",
        "downloaded_manifest_ref",
        "downloaded_manifest_sha256",
        "files",
        "files_checked",
        "bytes_checked",
        "status",
    }
)
_VERIFY_FILE_FIELDS = frozenset(
    {
        "path",
        "expected_size",
        "expected_sha256",
        "observed_size",
        "observed_sha256",
        "status",
    }
)


class G3LifecycleEvidenceError(ValueError):
    """Raised when an immutable lifecycle evidence chain is not admissible."""


class G3VerificationFailed(G3LifecycleEvidenceError):
    """Raised after a FAILED verify-only report and INVALID candidates exist."""

    def __init__(self, result: "G3VerificationResult") -> None:
        self.result = result
        super().__init__(
            f"G3_VERIFY_ONLY_FAILED:{result.verification_ref}:"
            f"{result.verification_sha256}"
        )


class G3NetworkEgressAttempt(G3LifecycleEvidenceError):
    """Raised if the independent verifier attempts any socket operation."""


@dataclass(frozen=True, slots=True)
class G3AcquisitionResult:
    status: str
    acquisition_ref: str
    acquisition_sha256: str
    downloaded_candidate_refs: tuple[str, ...]
    candidate_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "acquisition_ref": self.acquisition_ref,
            "acquisition_sha256": self.acquisition_sha256,
            "downloaded_candidate_refs": list(self.downloaded_candidate_refs),
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class G3VerificationResult:
    status: str
    verification_ref: str
    verification_sha256: str
    candidate_refs: tuple[str, ...]
    candidate_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verification_ref": self.verification_ref,
            "verification_sha256": self.verification_sha256,
            "candidate_refs": list(self.candidate_refs),
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class _G3SourceSnapshot:
    source_root: str
    head_commit: str
    files: tuple[tuple[str, str], ...]


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise G3LifecycleEvidenceError(f"{field} must be a JSON object")
    return dict(value)


def _exact(value: Any, *, field: str, fields: frozenset[str]) -> dict[str, Any]:
    result = _mapping(value, field=field)
    if set(result) != fields:
        raise G3LifecycleEvidenceError(
            f"{field} fields are not exact; missing={sorted(fields - set(result))}, "
            f"extra={sorted(set(result) - fields)}"
        )
    return result


def _text(value: Any, *, field: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise G3LifecycleEvidenceError(f"{field} must be normalized non-empty text")
    return value


def _sha256(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=64)
    if _SHA256.fullmatch(result) is None:
        raise G3LifecycleEvidenceError(f"{field} must be a lowercase SHA-256")
    return result


def _git_commit(value: Any, *, field: str) -> str:
    result = _text(value, field=field, maximum=64)
    if _GIT_COMMIT.fullmatch(result) is None:
        raise G3LifecycleEvidenceError(f"{field} must be an immutable Git digest")
    return result


def _timestamp(value: Any, *, field: str) -> tuple[str, datetime]:
    result = _text(value, field=field, maximum=64)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise G3LifecycleEvidenceError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise G3LifecycleEvidenceError(f"{field} must include a timezone")
    return result, parsed.astimezone(timezone.utc)


def _artifact_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def _reject_secrets_and_urls(value: Any, *, field: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).casefold()
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
            allowed_absence_sentinel = (
                normalized_key == "runtime_urls_persisted" and item is False
            )
            if not allowed_absence_sentinel and (
                normalized_key in _SENSITIVE_KEYS
                or any(
                    token in normalized_key
                    for token in (
                        "password",
                        "credential",
                        "authorization",
                        "api_key",
                        "access_token",
                        "refresh_token",
                        "auth_token",
                        "client_secret",
                        "secret_key",
                        "private_key",
                        "runtime_url",
                        "signed_url",
                        "presigned_url",
                        "x_amz_signature",
                    )
                )
            ):
                raise G3LifecycleEvidenceError(
                    f"{field} contains forbidden secret/runtime field {raw_key!r}"
                )
            _reject_secrets_and_urls(item, field=f"{field}.{raw_key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets_and_urls(item, field=f"{field}[{index}]")
        return
    if isinstance(value, str):
        folded = value.casefold()
        absolute_path = bool(
            value.startswith(("/", "\\\\", "//"))
            or re.match(r"^[a-zA-Z]:[\\/]", value)
        )
        signed_query = "?" in value and any(
            token in folded
            for token in (
                "x-amz-signature=",
                "signature=",
                "access_token=",
                "auth_token=",
                "credential=",
                "expires=",
            )
        )
        if (
            "://" in value
            or folded.startswith(("bearer ", "basic "))
            or signed_query
            or absolute_path
        ):
            raise G3LifecycleEvidenceError(
                f"{field} may not persist a URL, absolute path, or authorization value"
            )


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _approved_root(value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    if ".." in supplied.parts:
        raise G3LifecycleEvidenceError(f"{field} may not contain parent traversal")
    root = Path(os.path.abspath(supplied))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3LifecycleEvidenceError(f"{field} may not traverse a link")
    if not root.exists() or not root.is_dir():
        raise G3LifecycleEvidenceError(f"{field} must be an existing directory")
    return root.resolve(strict=True)


def _target_for_ref(root: Path, reference: str) -> Path:
    normalized = validate_asset_path(reference)
    relative = PurePosixPath(normalized)
    current = root
    for part in relative.parts:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3LifecycleEvidenceError(
                f"artifact reference traverses a link: {reference}"
            )
    target = root.joinpath(*relative.parts)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise G3LifecycleEvidenceError(
            f"artifact reference escapes the approved root: {reference}"
        ) from error
    return target


def _git_source(root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "-C",
            str(root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise G3LifecycleEvidenceError(
            f"formal source Git check failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout


def _capture_g3_source_snapshot(
    source_root: str | Path,
    *,
    expected_commit: str,
    control_refs: Sequence[str],
) -> _G3SourceSnapshot:
    """Capture one clean, tracked, exact-origin formal source boundary."""

    root = _approved_root(source_root, field="source_root")
    commit = _git_commit(expected_commit, field="source_git_commit")
    top_level = Path(
        _git_source(root, ("rev-parse", "--show-toplevel")).strip()
    ).resolve(strict=True)
    if top_level != root:
        raise G3LifecycleEvidenceError(
            "source_root must be the exact Git worktree root"
        )
    head = _git_source(root, ("rev-parse", "HEAD")).strip()
    if head != commit:
        raise G3LifecycleEvidenceError(
            "formal source HEAD differs from the declared Git commit"
        )
    if _git_source(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    ):
        raise G3LifecycleEvidenceError(
            "formal source worktree/index must be completely clean"
        )

    refs = tuple(
        dict.fromkeys(
            [
                *(validate_asset_path(item) for item in G3_CRITICAL_SOURCE_REFS),
                *(validate_asset_path(item) for item in control_refs),
            ]
        )
    )
    _git_source(root, ("ls-files", "--error-unmatch", "--", *refs))
    files: list[tuple[str, str]] = []
    for reference in refs:
        target = _target_for_ref(root, reference)
        if not target.exists() or not target.is_file() or _is_link_like(target):
            raise G3LifecycleEvidenceError(
                f"critical formal source is missing or unsafe: {reference}"
            )
        files.append((reference, sha256_file(target)))

    for module_name, reference in G3_CRITICAL_MODULE_ORIGINS:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        expected = _target_for_ref(root, reference).resolve(strict=True)
        if not isinstance(module_file, str):
            raise G3LifecycleEvidenceError(
                f"critical module has no file origin: {module_name}"
            )
        try:
            observed = Path(module_file).resolve(strict=True)
        except OSError as error:
            raise G3LifecycleEvidenceError(
                f"critical module origin is unreadable: {module_name}"
            ) from error
        if observed != expected:
            raise G3LifecycleEvidenceError(
                f"critical module origin differs from source_root: {module_name}"
            )
    return _G3SourceSnapshot(
        source_root=str(root),
        head_commit=head,
        files=tuple(files),
    )


def _revalidate_g3_source_snapshot(snapshot: _G3SourceSnapshot) -> None:
    observed = _capture_g3_source_snapshot(
        snapshot.source_root,
        expected_commit=snapshot.head_commit,
        control_refs=tuple(
            reference
            for reference, _ in snapshot.files
            if reference not in G3_CRITICAL_SOURCE_REFS
        ),
    )
    if observed != snapshot:
        raise G3LifecycleEvidenceError(
            "formal source files drifted during lifecycle execution"
        )


def _load_canonical_ref(root: Path, reference: str) -> tuple[dict[str, Any], str]:
    target = _target_for_ref(root, reference)
    if not target.exists() or not target.is_file() or _is_link_like(target):
        raise G3LifecycleEvidenceError(f"missing canonical artifact: {reference}")
    try:
        value = dict(
            ensure_json_object(load_canonical_json(target), field=reference)
        )
    except (CanonicalJSONError, OSError, TypeError, ValueError) as error:
        raise G3LifecycleEvidenceError(
            f"invalid canonical artifact: {reference}"
        ) from error
    return value, hashlib.sha256(target.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_clobber(
    root: Path,
    reference: str,
    value: Mapping[str, Any],
) -> tuple[bool, str]:
    payload = canonical_json_bytes(dict(value))
    digest = hashlib.sha256(payload).hexdigest()
    target = _target_for_ref(root, reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    _target_for_ref(root, reference)
    if target.exists() or _is_link_like(target):
        if _is_link_like(target) or not target.is_file():
            raise G3LifecycleEvidenceError(
                f"publication target is not a regular file: {reference}"
            )
        if target.read_bytes() != payload:
            raise FileExistsError(
                f"no-clobber publication target differs: {reference}"
            )
        return False, digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.publish-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        try:
            if os.name == "nt":
                os.rename(temporary, target)
            else:
                os.link(temporary, target)
                temporary.unlink()
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            if _is_link_like(target) or not target.is_file():
                raise G3LifecycleEvidenceError(
                    f"publication target raced to a non-file: {reference}"
                )
            if target.read_bytes() != payload:
                raise FileExistsError(
                    f"concurrent no-clobber publication differs: {reference}"
                )
            return False, digest
        _fsync_directory(target.parent)
        return True, digest
    finally:
        temporary.unlink(missing_ok=True)


def g3_acquisition_report_ref(acquisition_sha256: str) -> str:
    digest = _sha256(acquisition_sha256, field="acquisition_sha256")
    return f"manifests/evidence/g3/acquisition/{digest}.json"


def g3_verification_report_ref(acquisition_sha256: str) -> str:
    digest = _sha256(acquisition_sha256, field="acquisition_sha256")
    return f"manifests/evidence/g3/verification/{digest}.json"


def g3_downloaded_candidate_ref(logical_name: str, candidate_id: str) -> str:
    logical = validate_asset_path(logical_name)
    candidate = _sha256(candidate_id, field="candidate_id")
    return f"manifests/candidates/g3/{logical}/{candidate}/downloaded.json"


def g3_verified_candidate_ref(logical_name: str, candidate_id: str) -> str:
    logical = validate_asset_path(logical_name)
    candidate = _sha256(candidate_id, field="candidate_id")
    return f"manifests/candidates/g3/{logical}/{candidate}/verified.json"


def g3_invalid_candidate_ref(logical_name: str, candidate_id: str) -> str:
    logical = validate_asset_path(logical_name)
    candidate = _sha256(candidate_id, field="candidate_id")
    return f"manifests/candidates/g3/{logical}/{candidate}/invalid.json"


def _load_requirements(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        validate_stage0_asset_requirements(result)
        return result
    return load_stage0_asset_requirements(value)


def _load_layout(
    value: Mapping[str, Any] | str | Path,
    *,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        validate_stage0_asset_layout(result, requirements=requirements)
        return result
    return load_stage0_asset_layout(value, requirements=requirements)


def _load_plan(
    value: Mapping[str, Any] | str | Path,
    *,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = deepcopy(dict(value))
        validate_g3_download_plan(
            result, requirements=requirements, layout=layout
        )
        return result
    return load_g3_download_plan(
        value, requirements=requirements, layout=layout
    )


def _requirement_index(
    requirements: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for model in requirements["models"]:
        result[("model", model["name"])] = dict(model)
    tokenizer = requirements["tokenizer"]
    result[("tokenizer", tokenizer["name"])] = dict(tokenizer)
    result[("pile", "pile")] = dict(requirements["pile"])
    for task in requirements["glue"]:
        result[("glue_raw", task["task"])] = dict(task)
        result[("glue_derived", task["task"])] = dict(task)
    return result


def _source_relative_path(root: Path, value: str | Path, *, field: str) -> tuple[Path, str]:
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else root / supplied
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise G3LifecycleEvidenceError(f"{field} is outside source_root") from error
    if ".." in relative.parts:
        raise G3LifecycleEvidenceError(f"{field} contains parent traversal")
    current = root
    for part in relative.parts:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3LifecycleEvidenceError(f"{field} traverses a link")
    if not candidate.exists() or not candidate.is_file():
        raise G3LifecycleEvidenceError(f"{field} is not a regular file")
    return candidate.resolve(strict=True), relative.as_posix()


def _validate_download_report(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    source_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    report = _mapping(value, field="download report")
    expected_top = {
        "schema_version",
        "status",
        "started_at",
        "plan_sha256",
        "objects",
        "runtime_urls_persisted",
        "artifact_hash",
    }
    if set(report) != expected_top:
        raise G3LifecycleEvidenceError("download report fields are not exact")
    if (
        report["schema_version"] != DOWNLOAD_REPORT_SCHEMA_VERSION
        or report["status"] != "PASS"
        or report["runtime_urls_persisted"] is not False
        or report["plan_sha256"] != plan["artifact_hash"]
    ):
        raise G3LifecycleEvidenceError("download report is not a PASS for this plan")
    _timestamp(report["started_at"], field="download report started_at")
    if _sha256(report["artifact_hash"], field="download report artifact_hash") != (
        _artifact_hash(report)
    ):
        raise G3LifecycleEvidenceError("download report artifact_hash mismatch")
    objects = report["objects"]
    if not isinstance(objects, list) or len(objects) != EXPECTED_DOWNLOAD_OBJECT_COUNT:
        raise G3LifecycleEvidenceError("download report must contain exactly 13 objects")
    if len(plan["entries"]) != len(objects):
        raise G3LifecycleEvidenceError("download report object count mismatches plan")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    result_fields = {
        "schema_version",
        "status",
        "source_id",
        "revision",
        "size_bytes",
        "sha256",
        "attempts",
        "resumed",
        "network_accessed",
    }
    for index, (raw_object, raw_plan) in enumerate(zip(objects, plan["entries"])):
        item = _exact(
            raw_object,
            field=f"download report objects[{index}]",
            fields=frozenset({"object_id", "asset_root_ref", "final_path", "result"}),
        )
        plan_entry = dict(raw_plan)
        if any(
            item[field] != plan_entry[field]
            for field in ("object_id", "asset_root_ref", "final_path")
        ):
            raise G3LifecycleEvidenceError(
                f"download report objects[{index}] is out of plan order or identity"
            )
        spec_path = source_root.joinpath(*PurePosixPath(plan_entry["spec_ref"]).parts)
        try:
            spec_value = ensure_json_object(
                load_canonical_json(spec_path), field=plan_entry["spec_ref"]
            )
            spec = AssetObjectSpec.from_mapping(spec_value)
        except (CanonicalJSONError, OSError, TypeError, ValueError) as error:
            raise G3LifecycleEvidenceError(
                f"invalid frozen object spec: {plan_entry['spec_ref']}"
            ) from error
        result = _exact(
            item["result"],
            field=f"download report objects[{index}].result",
            fields=frozenset(result_fields),
        )
        if (
            result["schema_version"] != "stage0-asset-acquisition-result-v1"
            or result["status"] not in _RESULT_STATUSES
            or result["source_id"] != spec.source_id
            or result["revision"] != spec.revision
            or result["size_bytes"] != spec.expected_size
            or result["sha256"] != spec.expected_sha256
            or type(result["attempts"]) is not int
            or result["attempts"] < 0
            or type(result["resumed"]) is not bool
            or type(result["network_accessed"]) is not bool
        ):
            raise G3LifecycleEvidenceError(
                f"download report objects[{index}] result does not match its spec"
            )
        identity = (item["asset_root_ref"], item["final_path"])
        if identity in indexed:
            raise G3LifecycleEvidenceError("download report target identities repeat")
        indexed[identity] = {
            "object_id": item["object_id"],
            "spec_ref": plan_entry["spec_ref"],
            "revision": spec.revision,
            "size_bytes": spec.expected_size,
            "sha256": spec.expected_sha256,
            "result_status": result["status"],
            "network_accessed": result["network_accessed"],
        }
    _reject_secrets_and_urls(report, field="download report")
    return indexed


def _ensure_materialized_files(asset_root: Path, files: Sequence[AssetFile]) -> None:
    for descriptor in files:
        target = asset_root.joinpath(*PurePosixPath(descriptor.path).parts)
        if not target.exists() or not target.is_file() or _is_link_like(target):
            raise G3LifecycleEvidenceError(
                f"declared downloaded file is missing or unsafe: {descriptor.path}"
            )
        try:
            target.resolve(strict=True).relative_to(asset_root)
        except ValueError as error:
            raise G3LifecycleEvidenceError(
                f"declared downloaded file escapes its asset root: {descriptor.path}"
            ) from error


def _stable_build_evidence(build: GlueDerivedBuildResult) -> dict[str, Any]:
    return {
        "schema_version": "stage0-g3-derived-build-source-v1",
        "task": build.task,
        "raw_asset_id": build.raw_asset_id,
        "tokenizer_asset_id": build.tokenizer_asset_id,
        "tokenizer_descriptor_inventory": [
            dict(item) for item in build.tokenizer_descriptor_inventory
        ],
        "tokenizer_descriptor_inventory_hash": (
            build.tokenizer_descriptor_inventory_hash
        ),
        "target_ref": build.target_ref,
        "generator_git_commit": build.generator_git_commit,
        "preprocessing_version": build.preprocessing_version,
        "preprocessing_config_hash": build.preprocessing_config_hash,
        "requirement_hash": build.requirement_hash,
        "derived_splits": list(build.derived_splits),
        "split_counts": dict(build.split_counts),
        "map_fingerprints": dict(build.map_fingerprints),
        "file_inventory": [dict(item) for item in build.file_inventory],
        "network_attempts": build.network_attempts,
    }


def _expected_derived_map_fingerprints(
    requirement: Mapping[str, Any],
    *,
    raw_asset_id: str,
    tokenizer_asset_id: str,
    tokenizer_descriptor_inventory_hash: str,
    generator_git_commit: str,
    preprocessing_config_hash: str,
) -> dict[str, str]:
    """Replay the builder's immutable per-split map identity.

    This deliberately mirrors the small canonical payload used by
    ``glue_builder`` instead of trusting arbitrary 64-character values copied
    into an acquisition report.
    """

    fingerprints: dict[str, str] = {}
    for split in requirement["preprocessing"]["derived_splits"]:
        source_files = [
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in requirement["raw_files"]
            if item["role"] == split
        ]
        fingerprints[split] = canonical_json_hash(
            {
                "schema_version": MAP_FINGERPRINT_SCHEMA_VERSION,
                "task": requirement["task"],
                "split": split,
                "raw_asset_id": raw_asset_id,
                "tokenizer_asset_id": tokenizer_asset_id,
                "tokenizer_descriptor_inventory_hash": (
                    tokenizer_descriptor_inventory_hash
                ),
                "generator_git_commit": generator_git_commit,
                "preprocessing_version": GLUE_PREPROCESSING_VERSION,
                "preprocessing_config_hash": preprocessing_config_hash,
                "source_files": source_files,
                "expected_rows": requirement["split_counts"][split],
            }
        )
    return fingerprints


def _identity_entry(
    *,
    layout_entry: Mapping[str, Any],
    mode: str,
    asset_type: str,
    requirement: Mapping[str, Any],
    files: Sequence[AssetFile],
    metadata: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source = requirement["repository"]
    revision = requirement["revision"]
    logical_name = layout_entry["logical_name"]
    file_values = [
        item.as_dict() for item in sorted(files, key=lambda descriptor: descriptor.path)
    ]
    asset_id = compute_asset_id(
        asset_type=asset_type,
        name=logical_name,
        source=source,
        revision=revision,
        files=file_values,
        metadata=metadata,
    )
    return {
        "logical_name": logical_name,
        "kind": layout_entry["kind"],
        "requirement_name": layout_entry["requirement_name"],
        "asset_root_ref": layout_entry["asset_root_ref"],
        "mode": mode,
        "asset_type": asset_type,
        "name": logical_name,
        "source": source,
        "revision": revision,
        "asset_id": asset_id,
        "files": file_values,
        "metadata": deepcopy(dict(metadata)),
        "source_evidence": deepcopy(dict(source_evidence)),
    }


def validate_g3_acquisition_report(
    value: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
    download_plan: Mapping[str, Any] | None = None,
) -> None:
    """Validate one URL-free, immutable thirteen-logical-asset attestation."""

    report = _exact(value, field="acquisition report", fields=_ACQUISITION_FIELDS)
    if (
        report["schema_version"] != ACQUISITION_SCHEMA_VERSION
        or report["formal"] is not True
        or report["status"] != "PASS"
        or report["actor_role"] != AssetActorRole.FETCHER.value
        or report["runtime_urls_persisted"] is not False
    ):
        raise G3LifecycleEvidenceError("acquisition report formal envelope is invalid")
    started_text, started = _timestamp(
        report["started_at"], field="acquisition report started_at"
    )
    _, completed = _timestamp(
        report["completed_at"], field="acquisition report completed_at"
    )
    if completed < started:
        raise G3LifecycleEvidenceError("acquisition completed_at precedes started_at")
    _text(report["actor"], field="acquisition report actor", maximum=256)
    _text(
        report["actor_instance_id"],
        field="acquisition report actor_instance_id",
        maximum=512,
    )
    _git_commit(report["source_git_commit"], field="source_git_commit")
    for field in (
        "requirements_ref",
        "layout_ref",
        "download_plan_ref",
        "download_report_ref",
    ):
        validate_asset_path(_text(report[field], field=field))
    for field in (
        "requirements_sha256",
        "layout_sha256",
        "download_plan_sha256",
        "download_report_sha256",
        "artifact_hash",
    ):
        _sha256(report[field], field=field)
    if report["artifact_hash"] != _artifact_hash(report):
        raise G3LifecycleEvidenceError("acquisition report artifact_hash mismatch")
    if report["entry_count"] != EXPECTED_ENTRY_COUNT:
        raise G3LifecycleEvidenceError("acquisition report entry_count must be 13")
    entries = report["entries"]
    if not isinstance(entries, list) or len(entries) != EXPECTED_ENTRY_COUNT:
        raise G3LifecycleEvidenceError(
            "acquisition report must contain exactly 13 logical entries"
        )
    logical_names: set[str] = set()
    asset_ids: set[str] = set()
    mode_counts = {mode: 0 for mode in _ACQUISITION_MODES}
    for index, raw_entry in enumerate(entries):
        entry = _exact(
            raw_entry,
            field=f"acquisition entries[{index}]",
            fields=_ACQUISITION_ENTRY_FIELDS,
        )
        logical_name = validate_asset_path(
            _text(entry["logical_name"], field=f"entries[{index}].logical_name")
        )
        if entry["name"] != logical_name or logical_name in logical_names:
            raise G3LifecycleEvidenceError(
                "acquisition logical names must be unique and equal identity names"
            )
        logical_names.add(logical_name)
        validate_asset_path(
            _text(entry["asset_root_ref"], field=f"entries[{index}].asset_root_ref")
        )
        _text(entry["kind"], field=f"entries[{index}].kind")
        _text(
            entry["requirement_name"], field=f"entries[{index}].requirement_name"
        )
        mode = entry["mode"]
        if mode not in _ACQUISITION_MODES:
            raise G3LifecycleEvidenceError(f"entries[{index}].mode is invalid")
        mode_counts[mode] += 1
        if (
            (mode == "derived-build")
            != (entry["kind"] == "glue_derived")
        ):
            raise G3LifecycleEvidenceError(
                "only glue_derived entries may use derived-build mode"
            )
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise G3LifecycleEvidenceError(
                f"entries[{index}].files must be a non-empty array"
            )
        normalized_files = sorted(
            (AssetFile.from_value(item) for item in files),
            key=lambda item: item.path,
        )
        if [item.as_dict() for item in normalized_files] != files:
            raise G3LifecycleEvidenceError(
                f"entries[{index}].files must be sorted and canonical"
            )
        expected_asset_id = compute_asset_id(
            asset_type=entry["asset_type"],
            name=entry["name"],
            source=entry["source"],
            revision=entry["revision"],
            files=files,
            metadata=entry["metadata"],
        )
        if entry["asset_id"] != expected_asset_id:
            raise G3LifecycleEvidenceError(
                f"entries[{index}].asset_id does not match its identity"
            )
        if expected_asset_id in asset_ids:
            raise G3LifecycleEvidenceError("acquisition asset_id values must be unique")
        asset_ids.add(expected_asset_id)
        source_evidence = _mapping(
            entry["source_evidence"],
            field=f"entries[{index}].source_evidence",
        )
        expected_evidence_schema = {
            "canonical-plan": "stage0-g3-canonical-plan-source-v1",
            "existing-import": "stage0-g3-existing-import-source-v1",
            "derived-build": "stage0-g3-derived-build-source-v1",
        }[mode]
        if source_evidence.get("schema_version") != expected_evidence_schema:
            raise G3LifecycleEvidenceError(
                f"entries[{index}].source_evidence schema does not match mode"
            )
        if mode == "canonical-plan":
            if set(source_evidence) != {"schema_version", "object_count", "objects"}:
                raise G3LifecycleEvidenceError(
                    "canonical-plan source_evidence fields are not exact"
                )
            objects = source_evidence["objects"]
            if (
                type(source_evidence["object_count"]) is not int
                or source_evidence["object_count"] < 1
                or not isinstance(objects, list)
                or len(objects) != source_evidence["object_count"]
            ):
                raise G3LifecycleEvidenceError(
                    "canonical-plan source_evidence object count is invalid"
                )
            expected_object_fields = {
                "object_id",
                "spec_ref",
                "revision",
                "size_bytes",
                "sha256",
                "result_status",
                "network_accessed",
            }
            for object_index, raw_object in enumerate(objects):
                item = _exact(
                    raw_object,
                    field=(
                        f"entries[{index}].source_evidence.objects[{object_index}]"
                    ),
                    fields=frozenset(expected_object_fields),
                )
                _text(item["object_id"], field="canonical object_id")
                validate_asset_path(item["spec_ref"])
                _text(item["revision"], field="canonical revision")
                if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
                    raise G3LifecycleEvidenceError(
                        "canonical source object size is invalid"
                    )
                _sha256(item["sha256"], field="canonical object sha256")
                if (
                    item["result_status"] not in _RESULT_STATUSES
                    or type(item["network_accessed"]) is not bool
                ):
                    raise G3LifecycleEvidenceError(
                        "canonical source acquisition outcome is invalid"
                    )
        elif mode == "existing-import":
            if set(source_evidence) != {
                "schema_version",
                "requirements_ref",
                "requirements_sha256",
                "declared_file_count",
                "declared_bytes",
            }:
                raise G3LifecycleEvidenceError(
                    "existing-import source_evidence fields are not exact"
                )
            if (
                source_evidence["requirements_ref"] != report["requirements_ref"]
                or source_evidence["requirements_sha256"]
                != report["requirements_sha256"]
                or source_evidence["declared_file_count"] != len(files)
                or source_evidence["declared_bytes"]
                != sum(item.size_bytes for item in normalized_files)
            ):
                raise G3LifecycleEvidenceError(
                    "existing-import source_evidence does not bind its declaration"
                )
        else:
            derived_fields = {
                "schema_version",
                "task",
                "raw_asset_id",
                "tokenizer_asset_id",
                "tokenizer_descriptor_inventory",
                "tokenizer_descriptor_inventory_hash",
                "target_ref",
                "generator_git_commit",
                "preprocessing_version",
                "preprocessing_config_hash",
                "requirement_hash",
                "derived_splits",
                "split_counts",
                "map_fingerprints",
                "file_inventory",
                "network_attempts",
            }
            if set(source_evidence) != derived_fields:
                raise G3LifecycleEvidenceError(
                    "derived-build source_evidence fields are not exact"
                )
            for hash_field in (
                "raw_asset_id",
                "tokenizer_asset_id",
                "tokenizer_descriptor_inventory_hash",
                "preprocessing_config_hash",
                "requirement_hash",
            ):
                _sha256(
                    source_evidence[hash_field],
                    field=f"derived source {hash_field}",
                )
            if (
                source_evidence["task"] != entry["requirement_name"]
                or source_evidence["target_ref"] != entry["asset_root_ref"]
                or source_evidence["generator_git_commit"]
                != report["source_git_commit"]
                or source_evidence["network_attempts"] != 0
            ):
                raise G3LifecycleEvidenceError(
                    "derived-build source_evidence identity is invalid"
                )
            tokenizer_inventory = source_evidence[
                "tokenizer_descriptor_inventory"
            ]
            if (
                not isinstance(tokenizer_inventory, list)
                or not tokenizer_inventory
                or canonical_json_hash(tokenizer_inventory)
                != source_evidence["tokenizer_descriptor_inventory_hash"]
            ):
                raise G3LifecycleEvidenceError(
                    "derived tokenizer descriptor inventory binding is invalid"
                )
            file_inventory = source_evidence["file_inventory"]
            if (
                not isinstance(file_inventory, list)
                or any(
                    not isinstance(item, Mapping)
                    or set(item) != {"path", "size_bytes"}
                    or type(item["size_bytes"]) is not int
                    or item["size_bytes"] < 0
                    for item in file_inventory
                )
                or sorted(
                    (item.get("path"), item.get("size_bytes"))
                    for item in file_inventory
                    if isinstance(item, Mapping)
                )
                != sorted((item.path, item.size_bytes) for item in normalized_files)
            ):
                raise G3LifecycleEvidenceError(
                    "derived build inventory does not bind acquisition files"
                )
    if mode_counts != {
        "canonical-plan": 4,
        "existing-import": 6,
        "derived-build": 3,
    }:
        raise G3LifecycleEvidenceError(
            f"acquisition mode partition is invalid: {mode_counts}"
        )
    if requirements is not None:
        validate_stage0_asset_requirements(requirements)
        if report["requirements_sha256"] != requirements["artifact_hash"]:
            raise G3LifecycleEvidenceError(
                "acquisition report does not bind the supplied requirements"
            )
        requirement_by_identity = _requirement_index(requirements)
        entry_by_identity = {
            (entry["kind"], entry["requirement_name"]): entry
            for entry in entries
        }
        tokenizer_entry = entry_by_identity[("tokenizer", "pythia-tokenizer")]
        expected_tokenizer_inventory = [
            dict(item)
            for item in normalize_tokenizer_descriptor_inventory(
                requirements["tokenizer"]
            )
        ]
        for entry in entries:
            if entry["mode"] != "derived-build":
                continue
            requirement = requirement_by_identity[
                (entry["kind"], entry["requirement_name"])
            ]
            source_evidence = entry["source_evidence"]
            raw_entry = entry_by_identity[
                ("glue_raw", entry["requirement_name"])
            ]
            metadata = entry["metadata"]
            preprocessing = (
                metadata.get("preprocessing")
                if isinstance(metadata, Mapping)
                else None
            )
            expected_config_hash = glue_preprocessing_config_hash(requirement)
            expected_fingerprints = _expected_derived_map_fingerprints(
                requirement,
                raw_asset_id=raw_entry["asset_id"],
                tokenizer_asset_id=tokenizer_entry["asset_id"],
                tokenizer_descriptor_inventory_hash=canonical_json_hash(
                    expected_tokenizer_inventory
                ),
                generator_git_commit=report["source_git_commit"],
                preprocessing_config_hash=expected_config_hash,
            )
            if (
                source_evidence["requirement_hash"]
                != canonical_json_hash(requirement)
                or source_evidence["raw_asset_id"] != raw_entry["asset_id"]
                or source_evidence["tokenizer_asset_id"]
                != tokenizer_entry["asset_id"]
                or source_evidence["tokenizer_descriptor_inventory"]
                != expected_tokenizer_inventory
                or not isinstance(preprocessing, Mapping)
                or not isinstance(metadata, Mapping)
                or metadata.get("preprocessing_version")
                != GLUE_PREPROCESSING_VERSION
                or source_evidence["preprocessing_version"]
                != GLUE_PREPROCESSING_VERSION
                or preprocessing.get("version") != GLUE_PREPROCESSING_VERSION
                or preprocessing.get("parent_asset_ids") != [raw_entry["asset_id"]]
                or preprocessing.get("tokenizer_asset_id")
                != tokenizer_entry["asset_id"]
                or preprocessing.get("config_hash")
                != source_evidence["preprocessing_config_hash"]
                or source_evidence["preprocessing_config_hash"]
                != expected_config_hash
                or source_evidence["derived_splits"]
                != requirement["preprocessing"]["derived_splits"]
                or source_evidence["split_counts"]
                != {
                    split: requirement["split_counts"][split]
                    for split in requirement["preprocessing"]["derived_splits"]
                }
                or not isinstance(source_evidence["map_fingerprints"], Mapping)
                or set(source_evidence["map_fingerprints"])
                != set(requirement["preprocessing"]["derived_splits"])
                or any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in source_evidence["map_fingerprints"].values()
                )
                or dict(source_evidence["map_fingerprints"])
                != expected_fingerprints
            ):
                raise G3LifecycleEvidenceError(
                    "derived-build source evidence does not replay frozen lineage"
                )
    if layout is not None:
        validate_stage0_asset_layout(layout, requirements=requirements)
        if report["layout_sha256"] != layout["artifact_hash"]:
            raise G3LifecycleEvidenceError(
                "acquisition report does not bind the supplied layout"
            )
        for raw_entry, raw_layout_entry in zip(entries, layout["entries"]):
            entry = dict(raw_entry)
            layout_entry = dict(raw_layout_entry)
            for field in (
                "logical_name",
                "kind",
                "requirement_name",
                "asset_root_ref",
            ):
                if entry[field] != layout_entry[field]:
                    raise G3LifecycleEvidenceError(
                        "acquisition entries are not in exact layout order"
                    )
    if download_plan is not None:
        validate_g3_download_plan(
            download_plan, requirements=requirements, layout=layout
        )
        if report["download_plan_sha256"] != download_plan["artifact_hash"]:
            raise G3LifecycleEvidenceError(
                "acquisition report does not bind the supplied download plan"
            )
    _reject_secrets_and_urls(report, field="acquisition report")
    # Keep the parsed value live so timestamp validation cannot be optimized
    # into a superficial syntax-only check by future refactors.
    if not started_text:
        raise AssertionError("unreachable")


def _candidate_from_acquisition_entry(
    acquisition: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    downloading = build_g3_candidate_manifest(
        asset_type=entry["asset_type"],
        name=entry["name"],
        source=entry["source"],
        revision=entry["revision"],
        files=entry["files"],
        actor=acquisition["actor"],
        actor_role=AssetActorRole.FETCHER,
        actor_instance_id=acquisition["actor_instance_id"],
        evidence_ref=g3_acquisition_report_ref(acquisition["artifact_hash"]),
        evidence_sha256=acquisition["artifact_hash"],
        generator_version=GENERATOR_VERSION,
        generator_git_commit=acquisition["source_git_commit"],
        metadata=entry["metadata"],
        created_at=acquisition["started_at"],
    )
    if downloading["asset_id"] != entry["asset_id"]:
        raise G3LifecycleEvidenceError(
            "acquisition entry identity changed while constructing its candidate"
        )
    downloaded = transition_manifest(
        downloading,
        AssetState.DOWNLOADED,
        actor=acquisition["actor"],
        actor_role=AssetActorRole.FETCHER,
        actor_instance_id=acquisition["actor_instance_id"],
        evidence_ref=g3_acquisition_report_ref(acquisition["artifact_hash"]),
        evidence_sha256=acquisition["artifact_hash"],
        summary="acquisition attestor confirmed all declared local objects materialized",
        at=acquisition["completed_at"],
    )
    validate_g3_manifest(downloaded)
    return downloaded


def attest_stage0_g3_acquisition(
    *,
    source_root: str | Path,
    data_root: str | Path,
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    download_plan: Mapping[str, Any] | str | Path,
    requirements_ref: str,
    layout_ref: str,
    download_plan_ref: str,
    download_report_ref: str,
    actor_instance_id: str,
    source_git_commit: str,
    started_at: str,
    completed_at: str,
    actor: str = "stage0-g3-acquisition-attestor",
) -> G3AcquisitionResult:
    """Build derived assets and publish one exact acquisition/DOWNLOADED chain."""

    source = _approved_root(source_root, field="source_root")
    root = _approved_root(data_root, field="data_root")
    requirements_value = _load_requirements(requirements)
    layout_value = _load_layout(layout, requirements=requirements_value)
    plan_value = _load_plan(
        download_plan, requirements=requirements_value, layout=layout_value
    )
    for ref_name, ref in (
        ("requirements_ref", requirements_ref),
        ("layout_ref", layout_ref),
        ("download_plan_ref", download_plan_ref),
        ("download_report_ref", download_report_ref),
    ):
        validate_asset_path(_text(ref, field=ref_name))
    actor_value = _text(actor, field="actor", maximum=256)
    actor_instance = _text(
        actor_instance_id, field="actor_instance_id", maximum=512
    )
    commit = _git_commit(source_git_commit, field="source_git_commit")
    source_snapshot = _capture_g3_source_snapshot(
        source,
        expected_commit=commit,
        control_refs=(
            requirements_ref,
            layout_ref,
            download_plan_ref,
            *(entry["spec_ref"] for entry in plan_value["entries"]),
        ),
    )
    tracked_requirements, _ = _load_canonical_ref(source, requirements_ref)
    tracked_layout, _ = _load_canonical_ref(source, layout_ref)
    tracked_plan, _ = _load_canonical_ref(source, download_plan_ref)
    if (
        tracked_requirements != requirements_value
        or tracked_layout != layout_value
        or tracked_plan != plan_value
    ):
        raise G3LifecycleEvidenceError(
            "attestor inputs differ from their tracked source refs"
        )
    if resolve_source_git_commit(source) != commit:
        raise G3LifecycleEvidenceError(
            "source_git_commit does not match the attestor source checkout"
        )
    started_text, started_value = _timestamp(started_at, field="started_at")
    completed_text, completed_value = _timestamp(completed_at, field="completed_at")
    if completed_value < started_value:
        raise G3LifecycleEvidenceError("completed_at may not precede started_at")
    download_report, download_report_file_sha = _load_canonical_ref(
        root, download_report_ref
    )
    report_objects = _validate_download_report(
        download_report, plan=plan_value, source_root=source
    )
    if (
        layout_value["requirements_ref"] != requirements_ref
        or plan_value["requirements_ref"] != requirements_ref
        or plan_value["layout_ref"] != layout_ref
    ):
        raise G3LifecycleEvidenceError(
            "control-plane references do not match the attestor inputs"
        )

    # Import private identity builders lazily.  This keeps the evidence module
    # acyclic when the semantic gate consumes its public loaders.
    from .g3_asset_publication import (
        _asset_root as publication_asset_root,
        _files_and_metadata as publication_files_and_metadata,
        _parent_model_asset_id as publication_parent_model_asset_id,
    )

    requirement_by_identity = _requirement_index(requirements_value)
    asset_ids: dict[tuple[str, str], str] = {}
    entries: list[dict[str, Any]] = []
    consumed_plan_targets: set[tuple[str, str]] = set()
    layout_entries = list(layout_value["entries"])
    tokenizer_layout = next(
        item for item in layout_entries if item["kind"] == "tokenizer"
    )
    for raw_layout_entry in layout_entries:
        layout_entry = dict(raw_layout_entry)
        identity = (layout_entry["kind"], layout_entry["requirement_name"])
        try:
            requirement = requirement_by_identity[identity]
        except KeyError as error:
            raise G3LifecycleEvidenceError(
                f"layout entry has no frozen requirement: {identity!r}"
            ) from error
        kind = layout_entry["kind"]
        build: GlueDerivedBuildResult | None = None
        if kind == "glue_derived":
            raw_layout = next(
                item
                for item in layout_entries
                if item["kind"] == "glue_raw"
                and item["requirement_name"] == layout_entry["requirement_name"]
            )
            try:
                raw_asset_id = asset_ids[
                    ("glue_raw", layout_entry["requirement_name"])
                ]
                tokenizer_asset_id = asset_ids[
                    ("tokenizer", tokenizer_layout["requirement_name"])
                ]
            except KeyError as error:
                raise G3LifecycleEvidenceError(
                    "derived layout entry precedes its raw/tokenizer parent"
                ) from error
            build = build_glue_derived_dataset(
                root,
                raw_layout["asset_root_ref"],
                raw_asset_id,
                tokenizer_layout["asset_root_ref"],
                tokenizer_asset_id,
                requirement,
                layout_entry["asset_root_ref"],
                tokenizer_requirement=requirements_value["tokenizer"],
                generator_git_commit=commit,
            )
            if build.network_attempts != 0:
                raise G3LifecycleEvidenceError(
                    "derived build recorded a forbidden network attempt"
                )
        asset_root = publication_asset_root(root, layout_entry["asset_root_ref"])
        parent_model_id = (
            publication_parent_model_asset_id(
                requirement, requirements_value, asset_ids
            )
            if kind == "model"
            else None
        )
        tokenizer_id = asset_ids.get(
            ("tokenizer", tokenizer_layout["requirement_name"])
        )
        raw_parent_id = asset_ids.get(
            ("glue_raw", layout_entry["requirement_name"])
        )
        files, metadata, asset_type = publication_files_and_metadata(
            kind,
            requirement,
            asset_root=asset_root,
            generator_git_commit=commit,
            parent_model_asset_id=parent_model_id,
            tokenizer_asset_id=tokenizer_id,
            raw_glue_parent_asset_id=raw_parent_id,
        )
        _ensure_materialized_files(asset_root, files)
        plan_records: list[dict[str, Any]] = []
        for descriptor in files:
            target_identity = (
                layout_entry["asset_root_ref"],
                descriptor.path,
            )
            if target_identity in report_objects:
                record = report_objects[target_identity]
                if (
                    record["size_bytes"] != descriptor.size_bytes
                    or record["sha256"] != descriptor.sha256
                ):
                    raise G3LifecycleEvidenceError(
                        "download plan object identity disagrees with asset requirements"
                    )
                plan_records.append(deepcopy(record))
                consumed_plan_targets.add(target_identity)
        if build is not None:
            if plan_records:
                raise G3LifecycleEvidenceError(
                    "derived-build assets may not be partially covered by the download plan"
                )
            mode = "derived-build"
            source_evidence = _stable_build_evidence(build)
        elif len(plan_records) == len(files):
            mode = "canonical-plan"
            source_evidence = {
                "schema_version": "stage0-g3-canonical-plan-source-v1",
                "object_count": len(plan_records),
                "objects": sorted(plan_records, key=lambda item: item["object_id"]),
            }
        elif not plan_records:
            mode = "existing-import"
            source_evidence = {
                "schema_version": "stage0-g3-existing-import-source-v1",
                "requirements_ref": requirements_ref,
                "requirements_sha256": requirements_value["artifact_hash"],
                "declared_file_count": len(files),
                "declared_bytes": sum(item.size_bytes for item in files),
            }
        else:
            raise G3LifecycleEvidenceError(
                "download plan partially covers one logical asset; refusing mixed identity"
            )
        entry = _identity_entry(
            layout_entry=layout_entry,
            mode=mode,
            asset_type=asset_type,
            requirement=requirement,
            files=files,
            metadata=metadata,
            source_evidence=source_evidence,
        )
        entries.append(entry)
        asset_ids[identity] = entry["asset_id"]
    if consumed_plan_targets != set(report_objects):
        missing = sorted(set(report_objects) - consumed_plan_targets)
        raise G3LifecycleEvidenceError(
            f"download plan contains objects not covered by any logical asset: {missing}"
        )
    payload: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "formal": True,
        "status": "PASS",
        "started_at": started_text,
        "completed_at": completed_text,
        "actor": actor_value,
        "actor_role": AssetActorRole.FETCHER.value,
        "actor_instance_id": actor_instance,
        "source_git_commit": commit,
        "requirements_ref": requirements_ref,
        "requirements_sha256": requirements_value["artifact_hash"],
        "layout_ref": layout_ref,
        "layout_sha256": layout_value["artifact_hash"],
        "download_plan_ref": download_plan_ref,
        "download_plan_sha256": plan_value["artifact_hash"],
        "download_report_ref": download_report_ref,
        "download_report_sha256": download_report_file_sha,
        "runtime_urls_persisted": False,
        "entry_count": len(entries),
        "entries": entries,
    }
    acquisition = payload | {"artifact_hash": canonical_json_hash(payload)}
    validate_g3_acquisition_report(
        acquisition,
        requirements=requirements_value,
        layout=layout_value,
        download_plan=plan_value,
    )
    _revalidate_g3_source_snapshot(source_snapshot)
    acquisition_ref = g3_acquisition_report_ref(acquisition["artifact_hash"])
    report_created, _ = _publish_no_clobber(root, acquisition_ref, acquisition)
    candidate_refs: list[str] = []
    candidate_ids: list[str] = []
    any_created = report_created
    for entry in acquisition["entries"]:
        candidate = _candidate_from_acquisition_entry(acquisition, entry)
        candidate_ref = g3_downloaded_candidate_ref(
            entry["logical_name"], candidate["candidate_id"]
        )
        created, _ = _publish_no_clobber(root, candidate_ref, candidate)
        any_created = any_created or created
        candidate_refs.append(candidate_ref)
        candidate_ids.append(candidate["candidate_id"])
    _revalidate_g3_source_snapshot(source_snapshot)
    return G3AcquisitionResult(
        status="published" if any_created else "reused",
        acquisition_ref=acquisition_ref,
        acquisition_sha256=acquisition["artifact_hash"],
        downloaded_candidate_refs=tuple(candidate_refs),
        candidate_ids=tuple(candidate_ids),
    )


def _validate_acquisition_plan_partition(
    acquisition: Mapping[str, Any],
    *,
    report_objects: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    consumed: set[tuple[str, str]] = set()
    for raw_entry in acquisition["entries"]:
        entry = dict(raw_entry)
        covered = [
            (entry["asset_root_ref"], file_value["path"])
            for file_value in entry["files"]
            if (entry["asset_root_ref"], file_value["path"]) in report_objects
        ]
        if entry["mode"] == "canonical-plan":
            if len(covered) != len(entry["files"]):
                raise G3LifecycleEvidenceError(
                    "canonical-plan acquisition entry is not fully covered"
                )
            expected_objects = sorted(
                (dict(report_objects[target]) for target in covered),
                key=lambda item: item["object_id"],
            )
            evidence = _mapping(
                entry["source_evidence"], field="canonical-plan source_evidence"
            )
            if (
                evidence.get("object_count") != len(expected_objects)
                or evidence.get("objects") != expected_objects
            ):
                raise G3LifecycleEvidenceError(
                    "canonical-plan source evidence does not match the download report"
                )
            consumed.update(covered)
        elif covered:
            raise G3LifecycleEvidenceError(
                f"{entry['mode']} entry is unexpectedly covered by the download plan"
            )
    if consumed != set(report_objects):
        raise G3LifecycleEvidenceError(
            "acquisition logical entries do not exhaust the 13 download objects"
        )


def load_g3_acquisition_report(
    data_root: str | Path,
    acquisition_ref: str,
    *,
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    download_plan: Mapping[str, Any] | str | Path,
    source_root: str | Path,
) -> dict[str, Any]:
    """Load and replay every immutable binding behind an acquisition report."""

    root = _approved_root(data_root, field="data_root")
    source = _approved_root(source_root, field="source_root")
    requirements_value = _load_requirements(requirements)
    layout_value = _load_layout(layout, requirements=requirements_value)
    plan_value = _load_plan(
        download_plan, requirements=requirements_value, layout=layout_value
    )
    acquisition, _ = _load_canonical_ref(root, acquisition_ref)
    validate_g3_acquisition_report(
        acquisition,
        requirements=requirements_value,
        layout=layout_value,
        download_plan=plan_value,
    )
    if acquisition_ref != g3_acquisition_report_ref(acquisition["artifact_hash"]):
        raise G3LifecycleEvidenceError(
            "acquisition report is not stored at its immutable hash reference"
        )
    download_report, observed_download_sha = _load_canonical_ref(
        root, acquisition["download_report_ref"]
    )
    if observed_download_sha != acquisition["download_report_sha256"]:
        raise G3LifecycleEvidenceError(
            "acquisition download report bytes no longer match their binding"
        )
    report_objects = _validate_download_report(
        download_report, plan=plan_value, source_root=source
    )
    _validate_acquisition_plan_partition(
        acquisition, report_objects=report_objects
    )
    return acquisition


@contextmanager
def _zero_network_guard() -> Iterator[list[int]]:
    attempts = [0]
    socket_methods = tuple(
        name
        for name in ("connect", "connect_ex", "send", "sendall", "sendto", "sendmsg")
        if hasattr(socket.socket, name)
    )
    socket_functions = tuple(
        name
        for name in (
            "create_connection",
            "getaddrinfo",
            "gethostbyname",
            "gethostbyname_ex",
            "gethostbyaddr",
        )
        if hasattr(socket, name)
    )
    original_methods = {
        name: getattr(socket.socket, name) for name in socket_methods
    }
    original_functions = {
        name: getattr(socket, name) for name in socket_functions
    }
    original_popen = subprocess.Popen
    environment_names = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}

    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        attempts[0] += 1
        raise G3NetworkEgressAttempt("verify-only attempted socket egress")

    for name in original_methods:
        setattr(socket.socket, name, blocked)
    for name in original_functions:
        setattr(socket, name, blocked)
    subprocess.Popen = blocked  # type: ignore[assignment]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        yield attempts
    finally:
        subprocess.Popen = original_popen
        for name, method in original_methods.items():
            setattr(socket.socket, name, method)
        for name, function in original_functions.items():
            setattr(socket, name, function)
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def _observe_candidate_files(
    root: Path,
    acquisition_entry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int, int, str]:
    asset_root = _target_for_ref(root, acquisition_entry["asset_root_ref"])
    if (
        not asset_root.exists()
        or not asset_root.is_dir()
        or _is_link_like(asset_root)
    ):
        observations = [
            {
                "path": item["path"],
                "expected_size": item["size_bytes"],
                "expected_sha256": item["sha256"],
                "observed_size": None,
                "observed_sha256": None,
                "status": "missing",
            }
            for item in acquisition_entry["files"]
        ]
        return observations, 0, 0, "FAILED"
    resolved_root = asset_root.resolve(strict=True)
    observations: list[dict[str, Any]] = []
    files_checked = 0
    bytes_checked = 0
    passed = True
    for descriptor in acquisition_entry["files"]:
        target = asset_root.joinpath(*PurePosixPath(descriptor["path"]).parts)
        observed_size: int | None = None
        observed_sha: str | None = None
        status = "missing"
        try:
            if not target.exists() or not target.is_file() or _is_link_like(target):
                status = "missing" if not target.exists() else "unsafe"
            else:
                target.resolve(strict=True).relative_to(resolved_root)
                observed_size = target.stat().st_size
                observed_sha = sha256_file(target)
                files_checked += 1
                bytes_checked += observed_size
                if observed_size != descriptor["size_bytes"]:
                    status = "size_mismatch"
                elif observed_sha != descriptor["sha256"]:
                    status = "sha256_mismatch"
                else:
                    status = "PASS"
        except (OSError, ValueError):
            status = "unsafe"
            observed_size = None
            observed_sha = None
        if status != "PASS":
            passed = False
        observations.append(
            {
                "path": descriptor["path"],
                "expected_size": descriptor["size_bytes"],
                "expected_sha256": descriptor["sha256"],
                "observed_size": observed_size,
                "observed_sha256": observed_sha,
                "status": status,
            }
        )
    return observations, files_checked, bytes_checked, "PASS" if passed else "FAILED"


def validate_g3_verify_report(
    value: Mapping[str, Any],
    *,
    acquisition: Mapping[str, Any] | None = None,
    requirements: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
) -> None:
    """Validate the exact, per-file output of the independent verifier."""

    report = _exact(value, field="verify report", fields=_VERIFY_FIELDS)
    if (
        report["schema_version"] != VERIFY_SCHEMA_VERSION
        or report["formal"] is not True
        or report["status"] not in {"PASS", "FAILED"}
        or report["actor_role"] != AssetActorRole.VERIFIER.value
        or report["network_attempts"] != 0
    ):
        raise G3LifecycleEvidenceError("verify report formal envelope is invalid")
    _, checked_at = _timestamp(report["checked_at"], field="verify checked_at")
    _text(report["actor"], field="verify actor", maximum=256)
    _text(
        report["actor_instance_id"], field="verify actor_instance_id", maximum=512
    )
    _git_commit(report["generator_git_commit"], field="generator_git_commit")
    for field in ("requirements_ref", "layout_ref", "acquisition_ref"):
        validate_asset_path(_text(report[field], field=field))
    for field in (
        "requirements_sha256",
        "layout_sha256",
        "acquisition_sha256",
        "artifact_hash",
    ):
        _sha256(report[field], field=field)
    if report["artifact_hash"] != _artifact_hash(report):
        raise G3LifecycleEvidenceError("verify report artifact_hash mismatch")
    if report["entry_count"] != EXPECTED_ENTRY_COUNT:
        raise G3LifecycleEvidenceError("verify report entry_count must be 13")
    entries = report["entries"]
    if not isinstance(entries, list) or len(entries) != EXPECTED_ENTRY_COUNT:
        raise G3LifecycleEvidenceError("verify report must contain 13 logical entries")
    any_failure = False
    for index, raw_entry in enumerate(entries):
        entry = _exact(
            raw_entry,
            field=f"verify entries[{index}]",
            fields=_VERIFY_ENTRY_FIELDS,
        )
        validate_asset_path(_text(entry["logical_name"], field="logical_name"))
        _sha256(entry["asset_id"], field="asset_id")
        _sha256(entry["candidate_id"], field="candidate_id")
        validate_asset_path(
            _text(entry["downloaded_manifest_ref"], field="downloaded_manifest_ref")
        )
        _sha256(
            entry["downloaded_manifest_sha256"],
            field="downloaded_manifest_sha256",
        )
        if entry["status"] not in {"PASS", "FAILED"}:
            raise G3LifecycleEvidenceError("verify entry status is invalid")
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise G3LifecycleEvidenceError("verify entry files must be non-empty")
        observed_checked = 0
        observed_bytes = 0
        file_failed = False
        for file_index, raw_file in enumerate(files):
            item = _exact(
                raw_file,
                field=f"verify entries[{index}].files[{file_index}]",
                fields=_VERIFY_FILE_FIELDS,
            )
            validate_asset_path(_text(item["path"], field="verify file path"))
            if type(item["expected_size"]) is not int or item["expected_size"] < 0:
                raise G3LifecycleEvidenceError("verify expected_size is invalid")
            _sha256(item["expected_sha256"], field="verify expected_sha256")
            if item["observed_size"] is not None:
                if type(item["observed_size"]) is not int or item["observed_size"] < 0:
                    raise G3LifecycleEvidenceError("verify observed_size is invalid")
                observed_checked += 1
                observed_bytes += item["observed_size"]
            if item["observed_sha256"] is not None:
                _sha256(item["observed_sha256"], field="verify observed_sha256")
            if item["status"] not in {
                "PASS",
                "missing",
                "unsafe",
                "size_mismatch",
                "sha256_mismatch",
            }:
                raise G3LifecycleEvidenceError("verify file status is invalid")
            if item["status"] == "PASS":
                if (
                    item["observed_size"] != item["expected_size"]
                    or item["observed_sha256"] != item["expected_sha256"]
                ):
                    raise G3LifecycleEvidenceError(
                        "PASS file observation does not equal its expectation"
                    )
            else:
                file_failed = True
        if (
            entry["files_checked"] != observed_checked
            or entry["bytes_checked"] != observed_bytes
            or (entry["status"] == "FAILED") != file_failed
        ):
            raise G3LifecycleEvidenceError("verify entry summary is inconsistent")
        any_failure = any_failure or file_failed
    if (report["status"] == "FAILED") != any_failure:
        raise G3LifecycleEvidenceError("verify report status is inconsistent")
    if acquisition is not None:
        validate_g3_acquisition_report(
            acquisition, requirements=requirements, layout=layout
        )
        expected_acquisition_ref = g3_acquisition_report_ref(
            acquisition["artifact_hash"]
        )
        if (
            report["acquisition_ref"] != expected_acquisition_ref
            or report["acquisition_sha256"] != acquisition["artifact_hash"]
            or report["requirements_ref"] != acquisition["requirements_ref"]
            or report["requirements_sha256"] != acquisition["requirements_sha256"]
            or report["layout_ref"] != acquisition["layout_ref"]
            or report["layout_sha256"] != acquisition["layout_sha256"]
            or report["generator_git_commit"] != acquisition["source_git_commit"]
            or report["actor_instance_id"] == acquisition["actor_instance_id"]
        ):
            raise G3LifecycleEvidenceError(
                "verify report does not bind the exact acquisition/control plane"
            )
        _, completed_at = _timestamp(
            acquisition["completed_at"], field="acquisition completed_at"
        )
        if checked_at < completed_at:
            raise G3LifecycleEvidenceError("verify checked_at precedes acquisition")
        for raw_verify, raw_acquisition in zip(entries, acquisition["entries"]):
            verify_entry = dict(raw_verify)
            acquisition_entry = dict(raw_acquisition)
            expected_candidate = _candidate_from_acquisition_entry(
                acquisition, acquisition_entry
            )
            expected_candidate_ref = g3_downloaded_candidate_ref(
                acquisition_entry["logical_name"],
                expected_candidate["candidate_id"],
            )
            expected_candidate_sha = hashlib.sha256(
                canonical_json_bytes(expected_candidate)
            ).hexdigest()
            expected_observations = [
                {
                    "path": descriptor["path"],
                    "expected_size": descriptor["size_bytes"],
                    "expected_sha256": descriptor["sha256"],
                }
                for descriptor in acquisition_entry["files"]
            ]
            observed_expectations = [
                {
                    "path": item["path"],
                    "expected_size": item["expected_size"],
                    "expected_sha256": item["expected_sha256"],
                }
                for item in verify_entry["files"]
            ]
            if (
                verify_entry["logical_name"] != acquisition_entry["logical_name"]
                or verify_entry["asset_id"] != acquisition_entry["asset_id"]
                or verify_entry["candidate_id"]
                != expected_candidate["candidate_id"]
                or verify_entry["downloaded_manifest_ref"]
                != expected_candidate_ref
                or verify_entry["downloaded_manifest_sha256"]
                != expected_candidate_sha
                or observed_expectations != expected_observations
            ):
                raise G3LifecycleEvidenceError(
                    "verify entries do not exactly replay acquisition expectations"
                )
    _reject_secrets_and_urls(report, field="verify report")


def verify_stage0_g3_acquisition(
    *,
    source_root: str | Path,
    data_root: str | Path,
    requirements: Mapping[str, Any] | str | Path,
    layout: Mapping[str, Any] | str | Path,
    download_plan: Mapping[str, Any] | str | Path,
    acquisition_ref: str,
    actor_instance_id: str,
    generator_git_commit: str,
    checked_at: str,
    actor: str = "stage0-g3-verify-only",
) -> G3VerificationResult:
    """Hash every acquisition file and publish VERIFIED/INVALID candidates."""

    root = _approved_root(data_root, field="data_root")
    requirements_value = _load_requirements(requirements)
    layout_value = _load_layout(layout, requirements=requirements_value)
    plan_value = _load_plan(
        download_plan, requirements=requirements_value, layout=layout_value
    )
    commit = _git_commit(generator_git_commit, field="generator_git_commit")
    raw_acquisition, _ = _load_canonical_ref(root, acquisition_ref)
    try:
        source_control_refs = (
            raw_acquisition["requirements_ref"],
            raw_acquisition["layout_ref"],
            raw_acquisition["download_plan_ref"],
        )
    except KeyError as error:
        raise G3LifecycleEvidenceError(
            "acquisition is missing formal source refs"
        ) from error
    source_snapshot = _capture_g3_source_snapshot(
        source_root,
        expected_commit=commit,
        control_refs=(
            *source_control_refs,
            *(entry["spec_ref"] for entry in plan_value["entries"]),
        ),
    )
    source = Path(source_snapshot.source_root)
    tracked_requirements, _ = _load_canonical_ref(
        source, raw_acquisition["requirements_ref"]
    )
    tracked_layout, _ = _load_canonical_ref(
        source, raw_acquisition["layout_ref"]
    )
    tracked_plan, _ = _load_canonical_ref(
        source, raw_acquisition["download_plan_ref"]
    )
    if (
        tracked_requirements != requirements_value
        or tracked_layout != layout_value
        or tracked_plan != plan_value
    ):
        raise G3LifecycleEvidenceError(
            "verifier inputs differ from their tracked source refs"
        )
    acquisition = load_g3_acquisition_report(
        root,
        acquisition_ref,
        requirements=requirements_value,
        layout=layout_value,
        download_plan=plan_value,
        source_root=source_root,
    )
    actor_value = _text(actor, field="actor", maximum=256)
    actor_instance = _text(
        actor_instance_id, field="actor_instance_id", maximum=512
    )
    if actor_instance == acquisition["actor_instance_id"]:
        raise G3LifecycleEvidenceError(
            "verifier actor_instance_id must differ from the fetcher"
        )
    if commit != acquisition["source_git_commit"]:
        raise G3LifecycleEvidenceError(
            "verifier generator Git commit differs from the acquisition source"
        )
    checked_text, checked_value = _timestamp(checked_at, field="checked_at")
    _, completed_value = _timestamp(
        acquisition["completed_at"], field="acquisition completed_at"
    )
    if checked_value < completed_value:
        raise G3LifecycleEvidenceError("checked_at may not precede acquisition")

    report_entries: list[dict[str, Any]] = []
    downloaded_candidates: list[dict[str, Any]] = []
    with _zero_network_guard() as attempts:
        for acquisition_entry in acquisition["entries"]:
            expected_candidate = _candidate_from_acquisition_entry(
                acquisition, acquisition_entry
            )
            downloaded_ref = g3_downloaded_candidate_ref(
                acquisition_entry["logical_name"],
                expected_candidate["candidate_id"],
            )
            observed_candidate, downloaded_file_sha = _load_canonical_ref(
                root, downloaded_ref
            )
            try:
                validate_g3_manifest(observed_candidate)
            except ValueError as error:
                raise G3LifecycleEvidenceError(
                    f"invalid DOWNLOADED candidate: {downloaded_ref}"
                ) from error
            if observed_candidate != expected_candidate:
                raise G3LifecycleEvidenceError(
                    f"DOWNLOADED candidate identity/history mismatch: {downloaded_ref}"
                )
            observations, files_checked, bytes_checked, status = (
                _observe_candidate_files(root, acquisition_entry)
            )
            report_entries.append(
                {
                    "logical_name": acquisition_entry["logical_name"],
                    "asset_id": acquisition_entry["asset_id"],
                    "candidate_id": expected_candidate["candidate_id"],
                    "downloaded_manifest_ref": downloaded_ref,
                    "downloaded_manifest_sha256": downloaded_file_sha,
                    "files": observations,
                    "files_checked": files_checked,
                    "bytes_checked": bytes_checked,
                    "status": status,
                }
            )
            downloaded_candidates.append(expected_candidate)
    if attempts[0] != 0:
        raise G3LifecycleEvidenceError(
            "verify-only recorded a forbidden network attempt"
        )
    overall_status = (
        "PASS"
        if all(entry["status"] == "PASS" for entry in report_entries)
        else "FAILED"
    )
    payload: dict[str, Any] = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "formal": True,
        "status": overall_status,
        "checked_at": checked_text,
        "actor": actor_value,
        "actor_role": AssetActorRole.VERIFIER.value,
        "actor_instance_id": actor_instance,
        "generator_git_commit": commit,
        "requirements_ref": acquisition["requirements_ref"],
        "requirements_sha256": acquisition["requirements_sha256"],
        "layout_ref": acquisition["layout_ref"],
        "layout_sha256": acquisition["layout_sha256"],
        "acquisition_ref": acquisition_ref,
        "acquisition_sha256": acquisition["artifact_hash"],
        "network_attempts": 0,
        "entry_count": len(report_entries),
        "entries": report_entries,
    }
    report = payload | {"artifact_hash": canonical_json_hash(payload)}
    validate_g3_verify_report(
        report,
        acquisition=acquisition,
        requirements=requirements_value,
        layout=layout_value,
    )
    _revalidate_g3_source_snapshot(source_snapshot)
    verification_ref = g3_verification_report_ref(acquisition["artifact_hash"])
    report_created, _ = _publish_no_clobber(root, verification_ref, report)
    candidate_refs: list[str] = []
    candidate_ids: list[str] = []
    any_created = report_created
    for acquisition_entry, report_entry, downloaded in zip(
        acquisition["entries"], report_entries, downloaded_candidates
    ):
        if report_entry["status"] == "PASS" and overall_status == "PASS":
            target_state = AssetState.VERIFIED
            candidate_ref = g3_verified_candidate_ref(
                acquisition_entry["logical_name"], downloaded["candidate_id"]
            )
            summary = "independent verify-only process matched every declared file"
        elif report_entry["status"] == "FAILED":
            target_state = AssetState.INVALID
            candidate_ref = g3_invalid_candidate_ref(
                acquisition_entry["logical_name"], downloaded["candidate_id"]
            )
            summary = "independent verify-only process found a file integrity failure"
        else:
            candidate_refs.append(report_entry["downloaded_manifest_ref"])
            candidate_ids.append(downloaded["candidate_id"])
            continue
        transitioned = transition_manifest(
            downloaded,
            target_state,
            actor=actor_value,
            actor_role=AssetActorRole.VERIFIER,
            actor_instance_id=actor_instance,
            evidence_ref=verification_ref,
            evidence_sha256=report["artifact_hash"],
            summary=summary,
            at=checked_text,
        )
        created, _ = _publish_no_clobber(root, candidate_ref, transitioned)
        any_created = any_created or created
        candidate_refs.append(candidate_ref)
        candidate_ids.append(downloaded["candidate_id"])
    _revalidate_g3_source_snapshot(source_snapshot)
    result = G3VerificationResult(
        status=(
            "published"
            if any_created and overall_status == "PASS"
            else "reused"
            if overall_status == "PASS"
            else "failed"
        ),
        verification_ref=verification_ref,
        verification_sha256=report["artifact_hash"],
        candidate_refs=tuple(candidate_refs),
        candidate_ids=tuple(candidate_ids),
    )
    if overall_status == "FAILED":
        raise G3VerificationFailed(result)
    return result


def load_g3_verify_report(
    data_root: str | Path,
    verification_ref: str,
    *,
    acquisition: Mapping[str, Any],
    requirements: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _approved_root(data_root, field="data_root")
    report, _ = _load_canonical_ref(root, verification_ref)
    validate_g3_verify_report(
        report,
        acquisition=acquisition,
        requirements=requirements,
        layout=layout,
    )
    if verification_ref != g3_verification_report_ref(
        acquisition["artifact_hash"]
    ):
        raise G3LifecycleEvidenceError(
            "verify report is not stored at the acquisition-bound immutable ref"
        )
    return report


__all__ = [
    "ACQUISITION_SCHEMA_VERSION",
    "EXPECTED_ENTRY_COUNT",
    "GENERATOR_VERSION",
    "G3AcquisitionResult",
    "G3LifecycleEvidenceError",
    "G3NetworkEgressAttempt",
    "G3VerificationFailed",
    "G3VerificationResult",
    "VERIFY_SCHEMA_VERSION",
    "attest_stage0_g3_acquisition",
    "g3_acquisition_report_ref",
    "g3_downloaded_candidate_ref",
    "g3_invalid_candidate_ref",
    "g3_verification_report_ref",
    "g3_verified_candidate_ref",
    "load_g3_acquisition_report",
    "load_g3_verify_report",
    "validate_g3_acquisition_report",
    "validate_g3_verify_report",
    "verify_stage0_g3_acquisition",
]
