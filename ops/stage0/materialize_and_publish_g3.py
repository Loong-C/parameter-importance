"""Gate-only semantic qualification and READY publication for Stage 0 G3.

This process consumes immutable acquisition and independent verify-only
reports.  It has no download, existing-import, derived-build, hashing, or
VERIFIED-publication fallback.  Its only asset mutations are qualifications
and READY manifests after offline semantic probes pass.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Final

from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_download_plan import load_g3_download_plan
from param_importance_nlp.asset_requirements import (
    load_stage0_asset_requirements,
)
from param_importance_nlp.contracts import (
    canonical_json_bytes,
    canonical_json_hash,
)
from param_importance_nlp.g3_asset_publication import (
    G3AssetPublicationResult,
    gate_stage0_g3_assets_from_evidence,
)
from param_importance_nlp.g3_gate import (
    GATE_IDS,
    evaluate_stage0_g3,
    validate_stage0_g3_resolution,
)
from param_importance_nlp.g3_lifecycle_evidence import (
    G3_CRITICAL_MODULE_ORIGINS,
    G3_CRITICAL_SOURCE_REFS,
    load_g3_acquisition_report,
    load_g3_verify_report,
)


INDEX_SCHEMA_VERSION: Final = "stage0-g3-materialization-index-v2"
AUDIT_SCHEMA_VERSION: Final = "stage0-g3-materialization-audit-v3"
REPORT_ROOT_REF: Final = "reports/stage0/g3"
INDEX_NAME: Final = "asset-index.json"
AUDIT_NAME: Final = "asset-audit.json"
RESOLUTION_NAME: Final = "asset-resolution.json"

_EXPECTED_ENTRY_COUNT: Final = 13
_REQUIRED_SOURCE_REFS: Final = G3_CRITICAL_SOURCE_REFS
_REQUIRED_MODULE_ORIGINS: Final = G3_CRITICAL_MODULE_ORIGINS
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class Stage0G3MaterializationError(RuntimeError):
    """Raised when formal gate-only publication must fail closed."""


@dataclass(frozen=True, slots=True)
class GitSourceBinding:
    source_root: str
    head_commit: str
    requirements_ref: str
    requirements_file_sha256: str
    layout_ref: str
    layout_file_sha256: str
    download_plan_ref: str
    download_plan_file_sha256: str

    def to_dict(self) -> dict[str, str]:
        """Return only the immutable formal projection.

        ``source_root`` is deliberately retained on the runtime object for
        drift checks but must never be persisted into a report bundle.
        """

        return {
            "head_commit": self.head_commit,
            "requirements_ref": self.requirements_ref,
            "requirements_file_sha256": self.requirements_file_sha256,
            "layout_ref": self.layout_ref,
            "layout_file_sha256": self.layout_file_sha256,
            "download_plan_ref": self.download_plan_ref,
            "download_plan_file_sha256": self.download_plan_file_sha256,
        }


@dataclass(frozen=True, slots=True)
class G3ReportBundleResult:
    status: str
    directory_ref: str
    index_ref: str
    index_sha256: str
    audit_ref: str
    audit_sha256: str
    resolution_ref: str
    resolution_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"published", "reused"}:
            raise ValueError("G3_REPORT_BUNDLE_STATUS_INVALID")
        for field in ("index_sha256", "audit_sha256", "resolution_sha256"):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"G3_REPORT_BUNDLE_DIGEST_INVALID:{field}")

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "directory_ref": self.directory_ref,
            "index_ref": self.index_ref,
            "index_sha256": self.index_sha256,
            "audit_ref": self.audit_ref,
            "audit_sha256": self.audit_sha256,
            "resolution_ref": self.resolution_ref,
            "resolution_sha256": self.resolution_sha256,
        }


@dataclass(frozen=True, slots=True)
class Stage0G3MaterializationResult:
    status: str
    generator_git_commit: str
    checked_at: str
    source_binding: GitSourceBinding
    acquisition_ref: str
    acquisition_sha256: str
    verification_ref: str
    verification_sha256: str
    publications: tuple[G3AssetPublicationResult, ...]
    resolution_artifact_hash: str
    reports: G3ReportBundleResult

    def __post_init__(self) -> None:
        if self.status != "PASS":
            raise ValueError("G3_MATERIALIZATION_RESULT_MUST_BE_PASS")
        if len(self.publications) != _EXPECTED_ENTRY_COUNT:
            raise ValueError("G3_MATERIALIZATION_PUBLICATION_COUNT_INVALID")
        for field in (
            "acquisition_sha256",
            "verification_sha256",
            "resolution_artifact_hash",
        ):
            if _SHA256.fullmatch(getattr(self, field)) is None:
                raise ValueError(f"G3_MATERIALIZATION_{field.upper()}_INVALID")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "generator_git_commit": self.generator_git_commit,
            "checked_at": self.checked_at,
            "source_binding": self.source_binding.to_dict(),
            "acquisition_ref": self.acquisition_ref,
            "acquisition_sha256": self.acquisition_sha256,
            "verification_ref": self.verification_ref,
            "verification_sha256": self.verification_sha256,
            "publications": [item.to_dict() for item in self.publications],
            "resolution_artifact_hash": self.resolution_artifact_hash,
            "reports": self.reports.to_dict(),
        }


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _absolute_directory(value: str | Path, *, field: str) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute():
        raise Stage0G3MaterializationError(f"{field}_MUST_BE_ABSOLUTE")
    if ".." in supplied.parts:
        raise Stage0G3MaterializationError(f"{field}_PARENT_TRAVERSAL")
    absolute = Path(os.path.abspath(supplied))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise Stage0G3MaterializationError(f"{field}_LINK_FORBIDDEN:{current}")
    if not absolute.exists() or not absolute.is_dir():
        raise Stage0G3MaterializationError(f"{field}_NOT_DIRECTORY:{absolute}")
    return absolute.resolve(strict=True)


def _control_plane_path(
    source_root: Path,
    value: str | Path,
    *,
    field: str,
) -> tuple[Path, str]:
    supplied = Path(value)
    if ".." in supplied.parts:
        raise Stage0G3MaterializationError(f"{field}_PARENT_TRAVERSAL")
    candidate = supplied if supplied.is_absolute() else source_root / supplied
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(source_root)
    except ValueError as error:
        raise Stage0G3MaterializationError(f"{field}_OUTSIDE_SOURCE_ROOT") from error
    current = source_root
    for part in relative.parts:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise Stage0G3MaterializationError(f"{field}_LINK_FORBIDDEN:{current}")
    if not candidate.exists() or not candidate.is_file():
        raise Stage0G3MaterializationError(f"{field}_NOT_REGULAR_FILE:{candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(source_root)
    except ValueError as error:
        raise Stage0G3MaterializationError(f"{field}_ESCAPES_SOURCE_ROOT") from error
    return resolved, relative.as_posix()


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _git(
    source_root: Path,
    arguments: Sequence[str],
) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={source_root.as_posix()}",
        "-C",
        str(source_root),
        *arguments,
    ]
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise Stage0G3MaterializationError(
            f"GIT_SOURCE_BINDING_FAILED:{' '.join(arguments[:2])}"
        ) from error
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture_git_source_binding(
    source_root: Path,
    requirements_path: Path,
    requirements_ref: str,
    layout_path: Path,
    layout_ref: str,
    download_plan_path: Path,
    download_plan_ref: str,
    *,
    generator_git_commit: str,
) -> GitSourceBinding:
    top_level = _git(source_root, ("rev-parse", "--show-toplevel")).strip()
    try:
        observed_root = Path(top_level).resolve(strict=True)
    except OSError as error:
        raise Stage0G3MaterializationError("GIT_TOP_LEVEL_INVALID") from error
    if observed_root != source_root:
        raise Stage0G3MaterializationError("GIT_TOP_LEVEL_MISMATCH")
    head = _git(source_root, ("rev-parse", "HEAD")).strip()
    if head != generator_git_commit:
        raise Stage0G3MaterializationError(
            "GENERATOR_GIT_COMMIT_DOES_NOT_MATCH_SOURCE_HEAD"
        )
    for reference in _REQUIRED_SOURCE_REFS:
        path = source_root.joinpath(*PurePosixPath(reference).parts)
        if _is_link_like(path) or not path.is_file():
            raise Stage0G3MaterializationError(
                f"GIT_REQUIRED_SOURCE_MISSING:{reference}"
            )
    _git(
        source_root,
        (
            "ls-files",
            "--error-unmatch",
            "--",
            requirements_ref,
            layout_ref,
            download_plan_ref,
            *_REQUIRED_SOURCE_REFS,
        ),
    )
    dirty = _git(
        source_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    if dirty.strip():
        raise Stage0G3MaterializationError("GIT_SOURCE_ROOT_NOT_CLEAN")
    return GitSourceBinding(
        source_root=str(source_root),
        head_commit=head,
        requirements_ref=requirements_ref,
        requirements_file_sha256=_sha256(requirements_path),
        layout_ref=layout_ref,
        layout_file_sha256=_sha256(layout_path),
        download_plan_ref=download_plan_ref,
        download_plan_file_sha256=_sha256(download_plan_path),
    )


def _bind_sources(
    source_root: str | Path,
    requirements_path: str | Path,
    layout_path: str | Path,
    download_plan_path: str | Path,
    *,
    generator_git_commit: str,
) -> tuple[GitSourceBinding, Path, Path, Path]:
    if _GIT_COMMIT.fullmatch(generator_git_commit) is None:
        raise Stage0G3MaterializationError("GENERATOR_GIT_COMMIT_INVALID")
    root = _absolute_directory(source_root, field="SOURCE_ROOT")
    requirements, requirements_ref = _control_plane_path(
        root, requirements_path, field="REQUIREMENTS"
    )
    layout, layout_ref = _control_plane_path(root, layout_path, field="LAYOUT")
    download_plan, download_plan_ref = _control_plane_path(
        root, download_plan_path, field="DOWNLOAD_PLAN"
    )
    binding = _capture_git_source_binding(
        root,
        requirements,
        requirements_ref,
        layout,
        layout_ref,
        download_plan,
        download_plan_ref,
        generator_git_commit=generator_git_commit,
    )
    return binding, requirements, layout, download_plan


def _revalidate_source_binding(binding: GitSourceBinding) -> None:
    source_root = Path(binding.source_root)
    observed = _capture_git_source_binding(
        source_root,
        source_root.joinpath(*PurePosixPath(binding.requirements_ref).parts),
        binding.requirements_ref,
        source_root.joinpath(*PurePosixPath(binding.layout_ref).parts),
        binding.layout_ref,
        source_root.joinpath(*PurePosixPath(binding.download_plan_ref).parts),
        binding.download_plan_ref,
        generator_git_commit=binding.head_commit,
    )
    if observed != binding:
        raise Stage0G3MaterializationError("GIT_SOURCE_BINDING_DRIFTED")


def _assert_imported_module_origins(source_root: Path) -> None:
    expected_current = source_root.joinpath(
        *PurePosixPath("ops/stage0/materialize_and_publish_g3.py").parts
    ).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_current:
        raise Stage0G3MaterializationError(
            "IMPORTED_MODULE_ORIGIN_MISMATCH:ops.stage0.materialize_and_publish_g3"
        )
    for module_name, reference in _REQUIRED_MODULE_ORIGINS:
        module = importlib.import_module(module_name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            raise Stage0G3MaterializationError(
                f"IMPORTED_MODULE_ORIGIN_MISSING:{module_name}"
            )
        try:
            observed = Path(raw_origin).resolve(strict=True)
            expected = source_root.joinpath(
                *PurePosixPath(reference).parts
            ).resolve(strict=True)
        except OSError as error:
            raise Stage0G3MaterializationError(
                f"IMPORTED_MODULE_ORIGIN_INVALID:{module_name}"
            ) from error
        if observed != expected:
            raise Stage0G3MaterializationError(
                f"IMPORTED_MODULE_ORIGIN_MISMATCH:{module_name}"
            )


def _require_publication_order(
    results: Sequence[G3AssetPublicationResult],
    entries: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    if len(results) != len(entries):
        raise Stage0G3MaterializationError(f"{phase}_COUNT_INVALID")
    for result, entry in zip(results, entries):
        if (
            result.logical_name != entry["logical_name"]
            or result.kind != entry["kind"]
            or result.state != "ready"
        ):
            raise Stage0G3MaterializationError(f"{phase}_ORDER_OR_STATE_INVALID")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Stage0G3MaterializationError(f"{field}_INVALID")
    return value


def _require_formal_pass(
    resolution: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
    checked_at: str,
) -> None:
    try:
        validate_stage0_g3_resolution(resolution)
    except Exception as error:
        raise Stage0G3MaterializationError("G3_RESOLUTION_INVALID") from error
    entries = resolution.get("entries")
    gates = resolution.get("gates")
    if (
        resolution.get("status") != "PASS"
        or resolution.get("checked_at") != checked_at
        or resolution.get("requirements_ref") != layout["requirements_ref"]
        or resolution.get("requirements_artifact_hash")
        != requirements["artifact_hash"]
        or resolution.get("layout_artifact_hash") != layout["artifact_hash"]
        or not isinstance(entries, list)
        or len(entries) != _EXPECTED_ENTRY_COUNT
        or not isinstance(gates, list)
        or len(gates) != len(GATE_IDS)
    ):
        raise Stage0G3MaterializationError("G3_FORMAL_PASS_INCOMPLETE")
    if tuple(gate.get("gate_id") for gate in gates if isinstance(gate, Mapping)) != GATE_IDS:
        raise Stage0G3MaterializationError("G3_FORMAL_GATE_ORDER_INVALID")
    if any(
        not isinstance(gate, Mapping) or gate.get("status") != "PASS"
        for gate in gates
    ):
        raise Stage0G3MaterializationError("G3_FORMAL_GATE_BLOCKED")
    for entry, expected in zip(entries, layout["entries"]):
        if not isinstance(entry, Mapping):
            raise Stage0G3MaterializationError("G3_FORMAL_ENTRY_INVALID")
        checks = entry.get("checks")
        if (
            entry.get("logical_name") != expected["logical_name"]
            or entry.get("kind") != expected["kind"]
            or entry.get("requirement_name") != expected["requirement_name"]
            or entry.get("status") != "PASS"
            or entry.get("reasons") != []
            or not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
        ):
            raise Stage0G3MaterializationError("G3_FORMAL_ENTRY_BLOCKED")
        for field in (
            "asset_id",
            "candidate_id",
            "ready_manifest_sha256",
            "qualification_artifact_hash",
            "acquisition_sha256",
            "verification_sha256",
        ):
            _require_sha256(entry.get(field), field=f"G3_ENTRY_{field.upper()}")


def _payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _stable_publication_projection(
    item: G3AssetPublicationResult,
) -> dict[str, str]:
    """Project immutable identity while excluding published/reused run status."""

    value = item.to_dict()
    value.pop("status")
    return value


def _build_report_payloads(
    *,
    binding: GitSourceBinding,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
    generator_git_commit: str,
    checked_at: str,
    acquisition_ref: str,
    acquisition_sha256: str,
    verification_ref: str,
    verification_sha256: str,
    publications: Sequence[G3AssetPublicationResult],
    resolution: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    resolution_hash = _require_sha256(
        resolution.get("artifact_hash"), field="G3_RESOLUTION_ARTIFACT_HASH"
    )
    directory_ref = f"{REPORT_ROOT_REF}/{resolution_hash}"
    resolution_ref = f"{directory_ref}/{RESOLUTION_NAME}"
    audit_ref = f"{directory_ref}/{AUDIT_NAME}"
    index_ref = f"{directory_ref}/{INDEX_NAME}"
    resolution_payload = dict(resolution)
    resolution_sha = hashlib.sha256(
        canonical_json_bytes(resolution_payload)
    ).hexdigest()
    audit = _payload_with_hash(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": generator_git_commit,
            "source_binding": binding.to_dict(),
            "requirements_artifact_hash": requirements["artifact_hash"],
            "layout_artifact_hash": layout["artifact_hash"],
            "acquisition_ref": acquisition_ref,
            "acquisition_sha256": acquisition_sha256,
            "verification_ref": verification_ref,
            "verification_sha256": verification_sha256,
            "publication_count": len(publications),
            "publications": [
                _stable_publication_projection(item) for item in publications
            ],
            "gate_ids": list(GATE_IDS),
            "resolution_ref": resolution_ref,
            "resolution_sha256": resolution_sha,
            "resolution_artifact_hash": resolution_hash,
        }
    )
    audit_sha = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    resolution_entries = resolution["entries"]
    index = _payload_with_hash(
        {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": generator_git_commit,
            "source_git_commit": binding.head_commit,
            "requirements_ref": layout["requirements_ref"],
            "requirements_artifact_hash": requirements["artifact_hash"],
            "layout_ref": binding.layout_ref,
            "layout_artifact_hash": layout["artifact_hash"],
            "download_plan_ref": binding.download_plan_ref,
            "acquisition_ref": acquisition_ref,
            "acquisition_sha256": acquisition_sha256,
            "verification_ref": verification_ref,
            "verification_sha256": verification_sha256,
            "entry_count": len(resolution_entries),
            "entries": [
                {
                    "logical_name": entry["logical_name"],
                    "kind": entry["kind"],
                    "requirement_name": entry["requirement_name"],
                    "asset_id": entry["asset_id"],
                    "candidate_id": entry["candidate_id"],
                    "manifest_ref": entry["manifest_ref"],
                    "ready_manifest_sha256": entry["ready_manifest_sha256"],
                    "acquisition_ref": entry["acquisition_ref"],
                    "acquisition_sha256": entry["acquisition_sha256"],
                    "verification_ref": entry["verification_ref"],
                    "verification_sha256": entry["verification_sha256"],
                }
                for entry in resolution_entries
            ],
            "audit_ref": audit_ref,
            "audit_sha256": audit_sha,
            "resolution_ref": resolution_ref,
            "resolution_sha256": resolution_sha,
            "resolution_artifact_hash": resolution_hash,
        }
    )
    return directory_ref, {
        INDEX_NAME: index,
        AUDIT_NAME: audit,
        RESOLUTION_NAME: resolution_payload,
    }


def _safe_child_directory(root: Path, reference: str, *, create: bool) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference or ":" in reference:
        raise Stage0G3MaterializationError("REPORT_REFERENCE_INVALID")
    relative = PurePosixPath(reference)
    if (
        relative.is_absolute()
        or relative.as_posix() != reference
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise Stage0G3MaterializationError("REPORT_REFERENCE_INVALID")
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current) or not current.is_dir():
                raise Stage0G3MaterializationError(
                    f"REPORT_PATH_COMPONENT_INVALID:{current}"
                )
        elif create:
            current.mkdir()
        else:
            break
    candidate = root.joinpath(*relative.parts)
    if candidate.exists():
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise Stage0G3MaterializationError("REPORT_PATH_ESCAPE") from error
    return candidate


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise Stage0G3MaterializationError("REPORT_STAGING_CROSS_DEVICE")
    if os.name == "nt":
        os.rename(source, target)
        return
    if sys.platform.startswith("linux"):
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise Stage0G3MaterializationError("ATOMIC_NOREPLACE_UNAVAILABLE")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    if sys.platform == "darwin":
        import ctypes

        library = ctypes.CDLL(None, use_errno=True)
        renamex = getattr(library, "renamex_np", None)
        if renamex is None:
            raise Stage0G3MaterializationError("ATOMIC_NOREPLACE_UNAVAILABLE")
        renamex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex.restype = ctypes.c_int
        result = renamex(os.fsencode(source), os.fsencode(target), 0x00000004)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, os.strerror(error_number), target)
        raise OSError(error_number, os.strerror(error_number), target)
    raise Stage0G3MaterializationError("ATOMIC_NOREPLACE_UNAVAILABLE")


def _validate_existing_report_bundle(
    target: Path,
    expected: Mapping[str, bytes],
) -> None:
    if _is_link_like(target) or not target.is_dir():
        raise Stage0G3MaterializationError("REPORT_BUNDLE_TARGET_CONFLICT")
    observed: set[str] = set()
    for child in target.iterdir():
        if _is_link_like(child) or not child.is_file():
            raise Stage0G3MaterializationError("REPORT_BUNDLE_MEMBER_INVALID")
        observed.add(child.name)
    if observed != set(expected):
        raise Stage0G3MaterializationError("REPORT_BUNDLE_FILE_SET_MISMATCH")
    for name, payload in expected.items():
        if (target / name).read_bytes() != payload:
            raise Stage0G3MaterializationError(
                f"REPORT_BUNDLE_NO_CLOBBER_MISMATCH:{name}"
            )


def _publish_report_bundle(
    data_root: Path,
    directory_ref: str,
    payloads: Mapping[str, Mapping[str, Any]],
) -> G3ReportBundleResult:
    if set(payloads) != {INDEX_NAME, AUDIT_NAME, RESOLUTION_NAME}:
        raise Stage0G3MaterializationError("REPORT_BUNDLE_FILE_SET_INVALID")
    expected = {
        name: canonical_json_bytes(dict(payload)) for name, payload in payloads.items()
    }
    target = _safe_child_directory(data_root, directory_ref, create=False)
    if target.exists() or _is_link_like(target):
        _validate_existing_report_bundle(target, expected)
        status = "reused"
    else:
        parent_ref = PurePosixPath(directory_ref).parent.as_posix()
        parent = _safe_child_directory(data_root, parent_ref, create=True)
        tmp_root = _safe_child_directory(data_root, "tmp", create=True)
        staging = Path(tempfile.mkdtemp(prefix="stage0-g3-report-", dir=tmp_root))
        renamed = False
        try:
            for name, payload in expected.items():
                with (staging / name).open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            _fsync_directory(staging)
            try:
                _rename_directory_noreplace(staging, target)
                renamed = True
                _fsync_directory(parent)
                status = "published"
            except FileExistsError:
                _validate_existing_report_bundle(target, expected)
                status = "reused"
        finally:
            if not renamed and staging.exists():
                if (
                    staging.parent != tmp_root
                    or not staging.name.startswith("stage0-g3-report-")
                ):
                    raise RuntimeError("REFUSING_TO_REMOVE_UNOWNED_REPORT_STAGING")
                shutil.rmtree(staging)
                _fsync_directory(tmp_root)
    directory = PurePosixPath(directory_ref)
    return G3ReportBundleResult(
        status=status,
        directory_ref=directory.as_posix(),
        index_ref=(directory / INDEX_NAME).as_posix(),
        index_sha256=hashlib.sha256(expected[INDEX_NAME]).hexdigest(),
        audit_ref=(directory / AUDIT_NAME).as_posix(),
        audit_sha256=hashlib.sha256(expected[AUDIT_NAME]).hexdigest(),
        resolution_ref=(directory / RESOLUTION_NAME).as_posix(),
        resolution_sha256=hashlib.sha256(expected[RESOLUTION_NAME]).hexdigest(),
    )


def materialize_and_publish_stage0_g3(
    *,
    source_root: str | Path,
    data_root: str | Path,
    requirements_path: str | Path,
    layout_path: str | Path,
    download_plan_path: str | Path,
    acquisition_ref: str,
    verification_ref: str,
    gate_actor_instance_id: str,
    generator_git_commit: str,
    checked_at: str,
) -> Stage0G3MaterializationResult:
    """Consume VERIFIED evidence and perform only semantic/READY admission."""

    (
        binding,
        frozen_requirements_path,
        frozen_layout_path,
        frozen_download_plan_path,
    ) = _bind_sources(
        source_root,
        requirements_path,
        layout_path,
        download_plan_path,
        generator_git_commit=generator_git_commit,
    )
    _assert_imported_module_origins(Path(binding.source_root))
    root = _absolute_directory(data_root, field="DATA_ROOT")
    if _paths_overlap(Path(binding.source_root), root):
        raise Stage0G3MaterializationError("SOURCE_ROOT_DATA_ROOT_OVERLAP")
    requirements = load_stage0_asset_requirements(frozen_requirements_path)
    layout = load_stage0_asset_layout(
        frozen_layout_path,
        requirements=requirements,
    )
    download_plan = load_g3_download_plan(
        frozen_download_plan_path,
        requirements=requirements,
        layout=layout,
    )
    if layout["requirements_ref"] != binding.requirements_ref:
        raise Stage0G3MaterializationError(
            "LAYOUT_REQUIREMENTS_REF_DOES_NOT_MATCH_TRACKED_INPUT"
        )
    if (
        download_plan["requirements_ref"] != binding.requirements_ref
        or download_plan["layout_ref"] != binding.layout_ref
        or binding.download_plan_ref != Path(frozen_download_plan_path).relative_to(
            Path(binding.source_root)
        ).as_posix()
    ):
        raise Stage0G3MaterializationError(
            "DOWNLOAD_PLAN_REFS_DO_NOT_MATCH_TRACKED_INPUTS"
        )
    entries = layout["entries"]
    if len(entries) != _EXPECTED_ENTRY_COUNT:
        raise Stage0G3MaterializationError("G3_LAYOUT_ENTRY_COUNT_INVALID")

    def source_guard() -> None:
        _revalidate_source_binding(binding)

    source_guard()
    acquisition = load_g3_acquisition_report(
        root,
        acquisition_ref,
        requirements=requirements,
        layout=layout,
        download_plan=download_plan,
        source_root=binding.source_root,
    )
    verification = load_g3_verify_report(
        root,
        verification_ref,
        acquisition=acquisition,
        requirements=requirements,
        layout=layout,
    )
    if verification["status"] != "PASS":
        raise Stage0G3MaterializationError("VERIFY_ONLY_REPORT_NOT_PASS")
    source_guard()
    publications = gate_stage0_g3_assets_from_evidence(
        frozen_requirements_path,
        frozen_layout_path,
        frozen_download_plan_path,
        binding.source_root,
        root,
        acquisition_ref=acquisition_ref,
        verification_ref=verification_ref,
        generator_git_commit=generator_git_commit,
        checked_at=checked_at,
        gate_actor_instance_id=gate_actor_instance_id,
        pre_exposure_check=source_guard,
    )
    _require_publication_order(
        publications,
        entries,
        phase="G3_COMPLETE_PUBLICATION",
    )
    source_guard()
    resolution = evaluate_stage0_g3(
        frozen_requirements_path,
        frozen_layout_path,
        root,
        checked_at=checked_at,
    )
    _require_formal_pass(
        resolution,
        requirements=requirements,
        layout=layout,
        checked_at=checked_at,
    )
    source_guard()
    directory_ref, report_payloads = _build_report_payloads(
        binding=binding,
        requirements=requirements,
        layout=layout,
        generator_git_commit=generator_git_commit,
        checked_at=checked_at,
        acquisition_ref=acquisition_ref,
        acquisition_sha256=acquisition["artifact_hash"],
        verification_ref=verification_ref,
        verification_sha256=verification["artifact_hash"],
        publications=publications,
        resolution=resolution,
    )
    reports = _publish_report_bundle(root, directory_ref, report_payloads)
    return Stage0G3MaterializationResult(
        status="PASS",
        generator_git_commit=generator_git_commit,
        checked_at=checked_at,
        source_binding=binding,
        acquisition_ref=acquisition_ref,
        acquisition_sha256=acquisition["artifact_hash"],
        verification_ref=verification_ref,
        verification_sha256=verification["artifact_hash"],
        publications=tuple(publications),
        resolution_artifact_hash=resolution["artifact_hash"],
        reports=reports,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate-only semantic qualification and READY publication for Stage 0 G3"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--download-plan", type=Path, required=True)
    parser.add_argument("--acquisition-ref", required=True)
    parser.add_argument("--verification-ref", required=True)
    parser.add_argument("--gate-actor-instance-id", required=True)
    parser.add_argument("--generator-git-commit", required=True)
    parser.add_argument("--checked-at", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = materialize_and_publish_stage0_g3(
        source_root=arguments.source_root,
        data_root=arguments.data_root,
        requirements_path=arguments.requirements,
        layout_path=arguments.layout,
        download_plan_path=arguments.download_plan,
        acquisition_ref=arguments.acquisition_ref,
        verification_ref=arguments.verification_ref,
        gate_actor_instance_id=arguments.gate_actor_instance_id,
        generator_git_commit=arguments.generator_git_commit,
        checked_at=arguments.checked_at,
    )
    print(
        f"status={result.status} assets={len(result.publications)} "
        f"resolution={result.resolution_artifact_hash} "
        f"index={result.reports.index_ref}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "G3ReportBundleResult",
    "GitSourceBinding",
    "INDEX_SCHEMA_VERSION",
    "Stage0G3MaterializationError",
    "Stage0G3MaterializationResult",
    "main",
    "materialize_and_publish_stage0_g3",
]
