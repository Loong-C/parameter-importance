"""Replay a G3 materialization bundle and publish canonical Stage 0.04 evidence.

The Stage 0 task catalog intentionally requires ``model_assets``,
``data_assets`` and ``tokenizer_assets`` before the formal asset task may run.
Those capabilities must not be asserted by configuration.  This module derives
them from the immutable, independently verified G3 report bundle, replays the
current assets, uses the resulting capability commits to execute Stage 0.04,
then rebinds the capabilities and all G3 gates to the formal task outputs.

No acquisition or network operation is available here.  Any report, source,
asset or task-output drift fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

from .asset_download_plan import load_g3_download_plan
from .asset_layout import load_stage0_asset_layout
from .asset_requirements import load_stage0_asset_requirements
from .atomic import sha256_file
from .contracts import (
    GateRecord,
    GateStatus,
    RuntimeCapabilityEvidence,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .experiments import build_default_task_runtime
from .g3_gate import GATE_IDS, evaluate_stage0_g3, validate_stage0_g3_resolution
from .runtime import (
    TaskArtifactStore,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from .stage0_bootstrap import (
    Stage0SourceBinding,
    build_stage0_formal_config,
)
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


_TASK_ID = "stage0.04_assets_and_manifests"
_REQUIREMENTS_REF = "configs/stage0/g3-asset-requirements-v1.json"
_LAYOUT_REF = "configs/stage0/g3-asset-layout-v1.json"
_DOWNLOAD_PLAN_REF = "configs/stage0/g3-download-plan-v1.json"
_INDEX_SCHEMA = "stage0-g3-materialization-index-v2"
_AUDIT_SCHEMA = "stage0-g3-materialization-audit-v3"
_RESOLUTION_SCHEMA = "stage0-g3-resolution-audit-v1"
_EXPECTED_ENTRY_COUNT = 13
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_BOOTSTRAP_INDEX_FIELDS = {
    "schema_version",
    "generator_git_commit",
    "checked_at",
    "environment_ref",
    "environment_hash",
    "config_refs",
    "task_output_refs",
    "next_task_id",
    "next_input_refs",
    "artifact_hash",
}
_MATERIALIZATION_INDEX_FIELDS = {
    "schema_version",
    "status",
    "checked_at",
    "generator_git_commit",
    "source_git_commit",
    "requirements_ref",
    "requirements_artifact_hash",
    "layout_ref",
    "layout_artifact_hash",
    "download_plan_ref",
    "acquisition_ref",
    "acquisition_sha256",
    "verification_ref",
    "verification_sha256",
    "entry_count",
    "entries",
    "audit_ref",
    "audit_sha256",
    "resolution_ref",
    "resolution_sha256",
    "resolution_artifact_hash",
    "artifact_hash",
}
_INDEX_ENTRY_FIELDS = {
    "logical_name",
    "kind",
    "requirement_name",
    "asset_id",
    "candidate_id",
    "manifest_ref",
    "ready_manifest_sha256",
    "acquisition_ref",
    "acquisition_sha256",
    "verification_ref",
    "verification_sha256",
}
_AUDIT_FIELDS = {
    "schema_version",
    "status",
    "checked_at",
    "generator_git_commit",
    "source_binding",
    "requirements_artifact_hash",
    "layout_artifact_hash",
    "acquisition_ref",
    "acquisition_sha256",
    "verification_ref",
    "verification_sha256",
    "publication_count",
    "publications",
    "gate_ids",
    "resolution_ref",
    "resolution_sha256",
    "resolution_artifact_hash",
    "artifact_hash",
}
_SOURCE_BINDING_FIELDS = {
    "head_commit",
    "requirements_ref",
    "requirements_file_sha256",
    "layout_ref",
    "layout_file_sha256",
    "download_plan_ref",
    "download_plan_file_sha256",
}
_PUBLICATION_FIELDS = {
    "logical_name",
    "kind",
    "asset_id",
    "candidate_id",
    "state",
    "manifest_ref",
    "candidate_ref",
    "qualification_ref",
    "verification_ref",
    "semantic_evidence_ref",
}
_FORMALIZATION_INDEX_FIELDS = {
    "schema_version",
    "generator_git_commit",
    "checked_at",
    "bootstrap_index_ref",
    "bootstrap_index_sha256",
    "materialization_index_ref",
    "materialization_index_sha256",
    "materialization_resolution_artifact_hash",
    "unlock_environment_ref",
    "unlock_environment_hash",
    "config_ref",
    "config_hash",
    "task_output_refs",
    "gate_report_ref",
    "gate_refs",
    "capability_refs",
    "environment_ref",
    "environment_hash",
    "next_task_id",
    "next_input_refs",
    "artifact_hash",
}


class Stage0G3FormalizationError(RuntimeError):
    """The materialized G3 evidence cannot authorize formal Stage 0.04."""


@dataclass(frozen=True, slots=True)
class Stage0G3MaterializationBinding:
    checked_at: str
    generator_git_commit: str
    index_ref: str
    index_sha256: str
    index_artifact_hash: str
    audit_ref: str
    audit_sha256: str
    audit_artifact_hash: str
    resolution_ref: str
    resolution_sha256: str
    resolution_artifact_hash: str
    acquisition_ref: str
    acquisition_sha256: str
    verification_ref: str
    verification_sha256: str
    resolution: Mapping[str, JSONValue]

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        return (
            self.index_ref,
            self.audit_ref,
            self.resolution_ref,
            self.acquisition_ref,
            self.verification_ref,
        )


@dataclass(frozen=True, slots=True)
class Stage0G3FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    capability_refs: Mapping[str, str]
    gate_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str
    materialization: Stage0G3MaterializationBinding


@dataclass(frozen=True, slots=True)
class Stage0G3FormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    index_ref: str
    index_sha256: str
    generator_git_commit: str
    checked_at: str
    resolution_artifact_hash: str


def _is_link_like(path: Path) -> bool:
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


def _strict_root(value: str | Path) -> Path:
    supplied = Path(value)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise Stage0G3FormalizationError("G3_FORMAL_DATA_ROOT_INVALID")
    current = Path(supplied.anchor)
    for part in supplied.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise Stage0G3FormalizationError("G3_FORMAL_DATA_ROOT_LINK_FORBIDDEN")
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise Stage0G3FormalizationError("G3_FORMAL_DATA_ROOT_MISSING") from error
    if not root.is_dir():
        raise Stage0G3FormalizationError("G3_FORMAL_DATA_ROOT_NOT_DIRECTORY")
    return root


def _logical_ref(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise Stage0G3FormalizationError(f"G3_FORMAL_REF_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G3FormalizationError(f"G3_FORMAL_REF_INVALID:{field}")
    if logical.as_posix() != value:
        raise Stage0G3FormalizationError(f"G3_FORMAL_REF_NONCANONICAL:{field}")
    return value


def _safe_file(root: Path, reference: object, *, field: str) -> Path:
    ref = _logical_ref(reference, field=field)
    current = root
    for part in PurePosixPath(ref).parts:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise Stage0G3FormalizationError(f"G3_FORMAL_LINK_FORBIDDEN:{field}")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Stage0G3FormalizationError(f"G3_FORMAL_FILE_MISSING:{field}") from error
    if not resolved.is_file():
        raise Stage0G3FormalizationError(f"G3_FORMAL_FILE_INVALID:{field}")
    return resolved


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G3FormalizationError(f"G3_FORMAL_OBJECT_INVALID:{field}")
    return dict(value)


def _array(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise Stage0G3FormalizationError(f"G3_FORMAL_ARRAY_INVALID:{field}")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage0G3FormalizationError(f"G3_FORMAL_TEXT_INVALID:{field}")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    if _SHA256_RE.fullmatch(text) is None:
        raise Stage0G3FormalizationError(f"G3_FORMAL_SHA256_INVALID:{field}")
    return text


def _require_fields(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise Stage0G3FormalizationError(f"G3_FORMAL_FIELDS_INVALID:{field}")


def _validate_self_hash(value: Mapping[str, Any], *, field: str) -> str:
    claimed = _sha256(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    payload = dict(value)
    payload.pop("artifact_hash")
    if canonical_json_hash(payload) != claimed:
        raise Stage0G3FormalizationError(f"G3_FORMAL_ARTIFACT_HASH_MISMATCH:{field}")
    return claimed


def _load_object(root: Path, reference: object, *, field: str) -> tuple[dict[str, Any], str]:
    path = _safe_file(root, reference, field=field)
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise Stage0G3FormalizationError(f"G3_FORMAL_CANONICAL_JSON_INVALID:{field}") from error
    return _mapping(value, field=field), sha256_file(path)


def _load_bootstrap(
    root: Path,
    *,
    binding: Stage0SourceBinding,
    bootstrap_index_ref: str,
) -> tuple[TaskRuntimeEnvironment, tuple[str, ...], str]:
    index, index_sha = _load_object(
        root, bootstrap_index_ref, field="bootstrap_index"
    )
    _require_fields(index, _BOOTSTRAP_INDEX_FIELDS, field="bootstrap_index")
    if (
        index.get("schema_version") != "stage0-formal-bootstrap-index-v1"
        or index.get("generator_git_commit") != binding.git_commit
        or index.get("next_task_id") != _TASK_ID
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_BOOTSTRAP_IDENTITY_MISMATCH")
    _validate_self_hash(index, field="bootstrap_index")
    environment_ref = _logical_ref(
        index.get("environment_ref"), field="bootstrap_index.environment_ref"
    )
    environment_value, _ = _load_object(
        root, environment_ref, field="bootstrap_environment"
    )
    try:
        environment = TaskRuntimeEnvironment.from_mapping(environment_value)
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_BOOTSTRAP_ENVIRONMENT_INVALID") from error
    if environment.environment_hash != index.get("environment_hash"):
        raise Stage0G3FormalizationError("G3_FORMAL_BOOTSTRAP_ENVIRONMENT_DRIFT")

    outputs = _mapping(index.get("task_output_refs"), field="bootstrap.task_output_refs")
    if set(outputs) != {
        "stage0.01_baseline_and_safety",
        "stage0.02_storage_and_layout",
        "stage0.03_runtime_and_dependencies",
    }:
        raise Stage0G3FormalizationError("G3_FORMAL_BOOTSTRAP_TASK_SET_INVALID")
    last = _mapping(
        outputs["stage0.03_runtime_and_dependencies"],
        field="bootstrap.stage0_03_outputs",
    )
    raw_next = _array(index.get("next_input_refs"), field="bootstrap.next_input_refs")
    next_refs = tuple(
        _logical_ref(item, field=f"bootstrap.next_input_refs[{position}]")
        for position, item in enumerate(raw_next)
    )
    if (
        len(next_refs) != 3
        or len(set(next_refs)) != 3
        or set(next_refs) != set(last.values())
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_BOOTSTRAP_NEXT_INPUT_DRIFT")
    return environment, next_refs, index_sha


def load_and_replay_g3_materialization(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    materialization_index_ref: str,
) -> Stage0G3MaterializationBinding:
    """Validate the complete materializer bundle and replay all current assets."""

    root = _strict_root(data_root)
    index_ref = _logical_ref(materialization_index_ref, field="materialization_index")
    index, index_sha = _load_object(root, index_ref, field="materialization_index")
    _require_fields(index, _MATERIALIZATION_INDEX_FIELDS, field="materialization_index")
    index_artifact_hash = _validate_self_hash(index, field="materialization_index")
    if (
        index.get("schema_version") != _INDEX_SCHEMA
        or index.get("status") != "PASS"
        or index.get("generator_git_commit") != binding.git_commit
        or index.get("source_git_commit") != binding.git_commit
        or index.get("entry_count") != _EXPECTED_ENTRY_COUNT
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_MATERIALIZATION_IDENTITY_MISMATCH")

    index_path = PurePosixPath(index_ref)
    audit_ref = _logical_ref(index.get("audit_ref"), field="index.audit_ref")
    resolution_ref = _logical_ref(
        index.get("resolution_ref"), field="index.resolution_ref"
    )
    if (
        index_path.name != "asset-index.json"
        or PurePosixPath(audit_ref).parent != index_path.parent
        or PurePosixPath(audit_ref).name != "asset-audit.json"
        or PurePosixPath(resolution_ref).parent != index_path.parent
        or PurePosixPath(resolution_ref).name != "asset-resolution.json"
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_REPORT_BUNDLE_LAYOUT_INVALID")

    audit, audit_sha = _load_object(root, audit_ref, field="materialization_audit")
    resolution, resolution_sha = _load_object(
        root, resolution_ref, field="materialization_resolution"
    )
    if audit_sha != _sha256(index.get("audit_sha256"), field="index.audit_sha256"):
        raise Stage0G3FormalizationError("G3_FORMAL_AUDIT_FILE_HASH_MISMATCH")
    if resolution_sha != _sha256(
        index.get("resolution_sha256"), field="index.resolution_sha256"
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_RESOLUTION_FILE_HASH_MISMATCH")

    _require_fields(audit, _AUDIT_FIELDS, field="materialization_audit")
    audit_artifact_hash = _validate_self_hash(audit, field="materialization_audit")
    try:
        validate_stage0_g3_resolution(resolution)
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_RESOLUTION_INVALID") from error
    resolution_artifact_hash = _sha256(
        resolution.get("artifact_hash"), field="resolution.artifact_hash"
    )
    checked_at = _text(resolution.get("checked_at"), field="resolution.checked_at")
    if (
        resolution.get("schema_version") != _RESOLUTION_SCHEMA
        or resolution.get("status") != "PASS"
        or index.get("checked_at") != checked_at
        or audit.get("checked_at") != checked_at
        or index.get("resolution_artifact_hash") != resolution_artifact_hash
        or audit.get("resolution_artifact_hash") != resolution_artifact_hash
        or audit.get("resolution_ref") != resolution_ref
        or audit.get("resolution_sha256") != resolution_sha
        or audit.get("schema_version") != _AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("generator_git_commit") != binding.git_commit
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_REPORT_CROSS_BINDING_INVALID")

    source_binding = _mapping(audit.get("source_binding"), field="audit.source_binding")
    _require_fields(source_binding, _SOURCE_BINDING_FIELDS, field="audit.source_binding")
    if source_binding.get("head_commit") != binding.git_commit:
        raise Stage0G3FormalizationError("G3_FORMAL_REPORT_SOURCE_COMMIT_MISMATCH")
    requirements_ref = _logical_ref(
        index.get("requirements_ref"), field="index.requirements_ref"
    )
    layout_ref = _logical_ref(index.get("layout_ref"), field="index.layout_ref")
    download_plan_ref = _logical_ref(
        index.get("download_plan_ref"), field="index.download_plan_ref"
    )
    if (
        requirements_ref != _REQUIREMENTS_REF
        or layout_ref != _LAYOUT_REF
        or download_plan_ref != _DOWNLOAD_PLAN_REF
        or source_binding.get("requirements_ref") != requirements_ref
        or source_binding.get("layout_ref") != layout_ref
        or source_binding.get("download_plan_ref") != download_plan_ref
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_CONTROL_PLANE_REF_MISMATCH")

    requirements_path = binding.repository.joinpath(*PurePosixPath(requirements_ref).parts)
    layout_path = binding.repository.joinpath(*PurePosixPath(layout_ref).parts)
    download_plan_path = binding.repository.joinpath(*PurePosixPath(download_plan_ref).parts)
    if (
        sha256_file(requirements_path) != source_binding.get("requirements_file_sha256")
        or sha256_file(layout_path) != source_binding.get("layout_file_sha256")
        or sha256_file(download_plan_path)
        != source_binding.get("download_plan_file_sha256")
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_CONTROL_PLANE_FILE_DRIFT")
    try:
        requirements = load_stage0_asset_requirements(requirements_path)
        layout = load_stage0_asset_layout(layout_path, requirements=requirements)
        load_g3_download_plan(
            download_plan_path, requirements=requirements, layout=layout
        )
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_CONTROL_PLANE_INVALID") from error
    if (
        requirements.get("artifact_hash") != index.get("requirements_artifact_hash")
        or requirements.get("artifact_hash") != audit.get("requirements_artifact_hash")
        or requirements.get("artifact_hash")
        != resolution.get("requirements_artifact_hash")
        or layout.get("artifact_hash") != index.get("layout_artifact_hash")
        or layout.get("artifact_hash") != audit.get("layout_artifact_hash")
        or layout.get("artifact_hash") != resolution.get("layout_artifact_hash")
        or layout.get("requirements_ref") != resolution.get("requirements_ref")
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_CONTROL_PLANE_HASH_MISMATCH")

    acquisition_ref = _logical_ref(
        index.get("acquisition_ref"), field="index.acquisition_ref"
    )
    acquisition_sha = _sha256(
        index.get("acquisition_sha256"), field="index.acquisition_sha256"
    )
    verification_ref = _logical_ref(
        index.get("verification_ref"), field="index.verification_ref"
    )
    verification_sha = _sha256(
        index.get("verification_sha256"), field="index.verification_sha256"
    )
    if (
        audit.get("acquisition_ref") != acquisition_ref
        or audit.get("acquisition_sha256") != acquisition_sha
        or audit.get("verification_ref") != verification_ref
        or audit.get("verification_sha256") != verification_sha
        or audit.get("publication_count") != _EXPECTED_ENTRY_COUNT
        or audit.get("gate_ids") != list(GATE_IDS)
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_LIFECYCLE_BINDING_MISMATCH")

    index_entries = _array(index.get("entries"), field="index.entries")
    audit_publications = _array(audit.get("publications"), field="audit.publications")
    resolution_entries = _array(resolution.get("entries"), field="resolution.entries")
    if not (
        len(index_entries)
        == len(audit_publications)
        == len(resolution_entries)
        == _EXPECTED_ENTRY_COUNT
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_ENTRY_COUNT_MISMATCH")
    for position, (raw_index, raw_publication, raw_resolution) in enumerate(
        zip(index_entries, audit_publications, resolution_entries, strict=True)
    ):
        entry = _mapping(raw_index, field=f"index.entries[{position}]")
        publication = _mapping(
            raw_publication, field=f"audit.publications[{position}]"
        )
        resolved = _mapping(
            raw_resolution, field=f"resolution.entries[{position}]"
        )
        _require_fields(entry, _INDEX_ENTRY_FIELDS, field=f"index.entries[{position}]")
        _require_fields(
            publication,
            _PUBLICATION_FIELDS,
            field=f"audit.publications[{position}]",
        )
        for field in _INDEX_ENTRY_FIELDS:
            if entry[field] != resolved.get(field):
                raise Stage0G3FormalizationError(
                    f"G3_FORMAL_INDEX_RESOLUTION_DRIFT:{position}:{field}"
                )
        for field in (
            "logical_name",
            "kind",
            "asset_id",
            "candidate_id",
            "manifest_ref",
            "candidate_ref",
            "qualification_ref",
            "verification_ref",
            "semantic_evidence_ref",
        ):
            if publication[field] != resolved.get(field):
                raise Stage0G3FormalizationError(
                    f"G3_FORMAL_AUDIT_RESOLUTION_DRIFT:{position}:{field}"
                )
        if (
            publication.get("state") != "ready"
            or resolved.get("status") != "PASS"
            or resolved.get("acquisition_ref") != acquisition_ref
            or resolved.get("acquisition_sha256") != acquisition_sha
            or resolved.get("verification_ref") != verification_ref
            or resolved.get("verification_sha256") != verification_sha
        ):
            raise Stage0G3FormalizationError(
                f"G3_FORMAL_ENTRY_NOT_VERIFIED_READY:{position}"
            )

    gates = _array(resolution.get("gates"), field="resolution.gates")
    try:
        parsed_gates = tuple(GateRecord.from_mapping(_mapping(item, field="gate")) for item in gates)
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_GATE_RECORD_INVALID") from error
    if (
        tuple(item.gate_id for item in parsed_gates) != GATE_IDS
        or any(item.status is not GateStatus.PASS for item in parsed_gates)
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_SUBGATE_NOT_PASS")

    try:
        replayed = evaluate_stage0_g3(
            requirements_path,
            layout_path,
            root,
            checked_at=checked_at,
        )
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_ASSET_REPLAY_FAILED") from error
    if replayed != resolution:
        raise Stage0G3FormalizationError("G3_FORMAL_ASSET_REPLAY_DRIFT")

    return Stage0G3MaterializationBinding(
        checked_at=checked_at,
        generator_git_commit=binding.git_commit,
        index_ref=index_ref,
        index_sha256=index_sha,
        index_artifact_hash=index_artifact_hash,
        audit_ref=audit_ref,
        audit_sha256=audit_sha,
        audit_artifact_hash=audit_artifact_hash,
        resolution_ref=resolution_ref,
        resolution_sha256=resolution_sha,
        resolution_artifact_hash=resolution_artifact_hash,
        acquisition_ref=acquisition_ref,
        acquisition_sha256=acquisition_sha,
        verification_ref=verification_ref,
        verification_sha256=verification_sha,
        resolution=dict(resolution),
    )


def _asset_groups(resolution: Mapping[str, JSONValue]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {
        "model_assets": [],
        "data_assets": [],
        "tokenizer_assets": [],
    }
    entries = _array(resolution.get("entries"), field="resolution.entries")
    for position, raw in enumerate(entries):
        entry = _mapping(raw, field=f"resolution.entries[{position}]")
        kind = _text(entry.get("kind"), field=f"entries[{position}].kind")
        capability = (
            "model_assets"
            if kind == "model"
            else "tokenizer_assets"
            if kind == "tokenizer"
            else "data_assets"
        )
        groups[capability].append(
            {
                "logical_name": _text(
                    entry.get("logical_name"), field=f"entries[{position}].logical_name"
                ),
                "asset_id": _sha256(
                    entry.get("asset_id"), field=f"entries[{position}].asset_id"
                ),
                "ready_manifest_sha256": _sha256(
                    entry.get("ready_manifest_sha256"),
                    field=f"entries[{position}].ready_manifest_sha256",
                ),
            }
        )
    if any(not values for values in groups.values()):
        raise Stage0G3FormalizationError("G3_FORMAL_CAPABILITY_GROUP_EMPTY")
    return groups


def _publish_capabilities(
    *,
    store: TaskArtifactStore,
    config_hash: str,
    checked_at: str,
    resolution_hash: str,
    groups: Mapping[str, list[dict[str, str]]],
    evidence_refs: tuple[str, ...],
    phase: str,
) -> dict[str, str]:
    refs: dict[str, str] = {}
    for capability in ("model_assets", "data_assets", "tokenizer_assets"):
        assets = groups[capability]
        evidence = RuntimeCapabilityEvidence(
            capability=capability,
            status="VERIFIED",
            checked_at=checked_at,
            evidence_refs=evidence_refs,
            metadata={
                "phase": phase,
                "g3_resolution_artifact_hash": resolution_hash,
                "asset_count": len(assets),
                "assets": assets,
            },
        )
        refs[capability] = store.publish(
            task_id=_TASK_ID,
            artifact_kind=f"capability_{capability}",
            config_hash=config_hash,
            run_intent="formal",
            payload=evidence.to_dict(),
            formal_eligible=True,
            source_refs=evidence_refs,
        ).commit_ref
    return refs


def _environment_with_capabilities(
    base: TaskRuntimeEnvironment,
    capability_refs: Mapping[str, str],
) -> TaskRuntimeEnvironment:
    refs = dict(base.evidence_refs)
    refs.update(
        {f"capability_{capability}": reference for capability, reference in capability_refs.items()}
    )
    return TaskRuntimeEnvironment(
        capabilities=base.capabilities | frozenset(capability_refs),
        frozen_contract_stages=base.frozen_contract_stages,
        passed_gate_ids=base.passed_gate_ids,
        estimator_decision_ref=base.estimator_decision_ref,
        evidence_refs=refs,
    )


def _gate_key(gate_id: str) -> str:
    return "gate_" + re.sub(r"[^a-z0-9]+", "_", gate_id.casefold()).strip("_")


def load_stage0_g3_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str | None = None,
) -> Stage0G3FormalState:
    """Load the durable S0.4 state used by every downstream formal task."""

    root = _strict_root(data_root)
    normalized_ref = _logical_ref(index_ref, field="g3_formalization_index")
    index, index_sha = _load_object(
        root, normalized_ref, field="g3_formalization_index"
    )
    _require_fields(index, _FORMALIZATION_INDEX_FIELDS, field="g3_formalization_index")
    _validate_self_hash(index, field="g3_formalization_index")
    generator = _text(
        index.get("generator_git_commit"), field="g3_formalization.generator_git_commit"
    )
    if (
        index.get("schema_version") != "stage0-g3-formalization-index-v1"
        or index.get("next_task_id") != "stage0.05_config_run_identity_and_seeds"
        or (expected_git_commit is not None and generator != expected_git_commit)
    ):
        raise Stage0G3FormalizationError("G3_FORMALIZATION_INDEX_IDENTITY_INVALID")
    environment_ref = _logical_ref(
        index.get("environment_ref"), field="g3_formalization.environment_ref"
    )
    environment_value, _ = _load_object(
        root, environment_ref, field="g3_formalization.environment"
    )
    try:
        environment = TaskRuntimeEnvironment.from_mapping(environment_value)
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMALIZATION_ENVIRONMENT_INVALID") from error
    if environment.environment_hash != index.get("environment_hash"):
        raise Stage0G3FormalizationError("G3_FORMALIZATION_ENVIRONMENT_HASH_MISMATCH")

    outputs = _mapping(index.get("task_output_refs"), field="g3_formalization.outputs")
    if tuple(outputs) != ("asset_audit", "asset_manifest", "asset_resolution"):
        # Canonical JSON sorts object keys.  Validate the exact set here and use
        # next_input_refs as the authoritative task-catalog order below.
        if set(outputs) != {"asset_manifest", "asset_audit", "asset_resolution"}:
            raise Stage0G3FormalizationError("G3_FORMALIZATION_OUTPUT_SET_INVALID")
    for kind, reference in outputs.items():
        ref = _logical_ref(reference, field=f"g3_formalization.outputs.{kind}")
        try:
            loaded = load_committed_task_artifact(root, ref, require_formal=True)
        except Exception as error:
            raise Stage0G3FormalizationError(
                f"G3_FORMALIZATION_OUTPUT_COMMIT_INVALID:{kind}"
            ) from error
        if loaded.identity.task_id != _TASK_ID or loaded.identity.artifact_kind != kind:
            raise Stage0G3FormalizationError(
                f"G3_FORMALIZATION_OUTPUT_IDENTITY_INVALID:{kind}"
            )
    raw_next = _array(index.get("next_input_refs"), field="g3_formalization.next_input_refs")
    next_refs = tuple(
        _logical_ref(item, field=f"g3_formalization.next_input_refs[{position}]")
        for position, item in enumerate(raw_next)
    )
    ordered_outputs = {
        kind: next_refs[position]
        for position, kind in enumerate(("asset_manifest", "asset_audit", "asset_resolution"))
    }
    if (
        len(next_refs) != 3
        or len(set(next_refs)) != 3
        or set(next_refs) != set(outputs.values())
        or any(
            load_committed_task_artifact(root, reference, require_formal=True).identity.artifact_kind
            != kind
            for kind, reference in ordered_outputs.items()
        )
    ):
        raise Stage0G3FormalizationError("G3_FORMALIZATION_NEXT_INPUT_DRIFT")
    if environment.evidence_refs.get("g3_resolution") != ordered_outputs["asset_resolution"]:
        raise Stage0G3FormalizationError("G3_FORMALIZATION_RESOLUTION_ENVIRONMENT_DRIFT")
    required_capabilities = {"model_assets", "data_assets", "tokenizer_assets"}
    required_gates = set(GATE_IDS) | {"stage0.G3"}
    if (
        not required_capabilities <= environment.capabilities
        or not required_gates <= environment.passed_gate_ids
    ):
        raise Stage0G3FormalizationError("G3_FORMALIZATION_ENVIRONMENT_INCOMPLETE")
    return Stage0G3FormalState(
        environment=environment,
        task_output_refs=ordered_outputs,
        index_ref=normalized_ref,
        index_sha256=index_sha,
        generator_git_commit=generator,
        checked_at=_text(index.get("checked_at"), field="g3_formalization.checked_at"),
        resolution_artifact_hash=_sha256(
            index.get("materialization_resolution_artifact_hash"),
            field="g3_formalization.resolution_artifact_hash",
        ),
    )


def formalize_stage0_g3(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    bootstrap_index_ref: str,
    materialization_index_ref: str,
) -> Stage0G3FormalizationResult:
    """Execute formal S0.4 and publish the durable G3 environment snapshot."""

    root = _strict_root(data_root)
    base_environment, input_refs, bootstrap_index_sha = _load_bootstrap(
        root,
        binding=binding,
        bootstrap_index_ref=bootstrap_index_ref,
    )
    materialization = load_and_replay_g3_materialization(
        binding=binding,
        data_root=root,
        materialization_index_ref=materialization_index_ref,
    )
    groups = _asset_groups(materialization.resolution)
    unlock_hash = canonical_json_hash(
        {
            "schema_version": "stage0-g3-capability-unlock-v1",
            "generator_git_commit": binding.git_commit,
            "bootstrap_index_ref": bootstrap_index_ref,
            "bootstrap_index_sha256": bootstrap_index_sha,
            "materialization_index_ref": materialization.index_ref,
            "materialization_index_sha256": materialization.index_sha256,
            "resolution_artifact_hash": materialization.resolution_artifact_hash,
        }
    )
    unlock_dir = (
        "evidence/stage0/g3-unlock/" + materialization.resolution_artifact_hash
    )
    unlock_store = TaskArtifactStore(root, unlock_dir)
    unlock_capability_refs = _publish_capabilities(
        store=unlock_store,
        config_hash=unlock_hash,
        checked_at=materialization.checked_at,
        resolution_hash=materialization.resolution_artifact_hash,
        groups=groups,
        evidence_refs=materialization.evidence_refs,
        phase="materialization_replay",
    )
    unlock_environment = _environment_with_capabilities(
        base_environment, unlock_capability_refs
    )
    unlock_environment_ref = f"{unlock_dir}/environment.json"
    write_canonical_json(root / unlock_environment_ref, unlock_environment.to_dict())

    task_output_dir = (
        "evidence/stage0/tasks/04-" + materialization.resolution_artifact_hash
    )
    config = build_stage0_formal_config(
        binding.repository,
        task_id=_TASK_ID,
        input_refs=input_refs,
        output_dir=task_output_dir,
    )
    formal_dir = (
        "evidence/stage0/g3-formal/" + materialization.resolution_artifact_hash
    )
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(root / config_ref, config.to_dict())
    runtime = build_default_task_runtime(root)
    result = runtime.execute(config, environment=unlock_environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G3FormalizationError(
            f"G3_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    if tuple(outputs) != ("asset_manifest", "asset_audit", "asset_resolution"):
        raise Stage0G3FormalizationError("G3_FORMAL_TASK_OUTPUT_SET_INVALID")
    try:
        committed_resolution = load_committed_task_artifact(
            root, outputs["asset_resolution"], require_formal=True
        )
    except Exception as error:
        raise Stage0G3FormalizationError("G3_FORMAL_RESOLUTION_COMMIT_INVALID") from error
    if (
        committed_resolution.identity.task_id != _TASK_ID
        or committed_resolution.identity.artifact_kind != "asset_resolution"
        or dict(committed_resolution.payload) != dict(materialization.resolution)
    ):
        raise Stage0G3FormalizationError("G3_FORMAL_RESOLUTION_COMMIT_DRIFT")

    final_store = TaskArtifactStore(root, formal_dir)
    report = Stage0GateReport(
        gate_id="stage0.G3",
        generated_at=materialization.checked_at,
        generator_git_commit=binding.git_commit,
        environment_id=unlock_environment.environment_hash,
        config_hashes={"stage0.04": config.config_hash},
        input_evidence=(
            Stage0EvidenceRef(
                materialization.index_ref,
                materialization.index_sha256,
                _INDEX_SCHEMA,
            ),
            Stage0EvidenceRef(
                materialization.audit_ref,
                materialization.audit_sha256,
                _AUDIT_SCHEMA,
            ),
            Stage0EvidenceRef(
                materialization.resolution_ref,
                materialization.resolution_sha256,
                _RESOLUTION_SCHEMA,
            ),
        ),
        checks=tuple(
            Stage0GateCheck(
                check_id=gate_id,
                check_class=Stage0CheckClass.CORRECTNESS,
                status=Stage0CheckStatus.PASS,
                summary=f"{gate_id} passed in the replayed formal G3 resolution",
                measurements={"status": "PASS"},
                evidence_refs=(outputs["asset_resolution"],),
            )
            for gate_id in GATE_IDS
        ),
    )
    formal_sources = tuple(outputs.values()) + materialization.evidence_refs
    gate_report_ref = final_store.publish(
        task_id=_TASK_ID,
        artifact_kind="gate_report",
        config_hash=config.config_hash,
        run_intent="formal",
        payload=report.to_dict(),
        formal_eligible=True,
        source_refs=formal_sources,
    ).commit_ref

    gate_refs: dict[str, str] = {}
    raw_gates = _array(
        materialization.resolution.get("gates"), field="resolution.gates"
    )
    for raw_gate in raw_gates:
        original = GateRecord.from_mapping(_mapping(raw_gate, field="resolution.gate"))
        rebound = GateRecord(
            gate_id=original.gate_id,
            stage=original.stage,
            status=original.status,
            checked_at=original.checked_at,
            measured=original.measured,
            threshold=original.threshold,
            evidence_refs=tuple(
                dict.fromkeys(
                    (*original.evidence_refs, outputs["asset_resolution"], gate_report_ref)
                )
            ),
            reasons=original.reasons,
            conditions=original.conditions,
            expires_at=original.expires_at,
        )
        kind = _gate_key(original.gate_id).removeprefix("gate_stage0_")
        gate_refs[original.gate_id] = final_store.publish(
            task_id=_TASK_ID,
            artifact_kind=f"gate_{kind}",
            config_hash=config.config_hash,
            run_intent="formal",
            payload=rebound.to_dict(),
            formal_eligible=True,
            source_refs=(outputs["asset_resolution"], gate_report_ref),
        ).commit_ref
    aggregate = GateRecord(
        gate_id="stage0.G3",
        stage=0,
        status=GateStatus.PASS,
        checked_at=materialization.checked_at,
        measured={"subgates": {gate_id: "PASS" for gate_id in GATE_IDS}},
        threshold={"required": "all G3 subgates PASS"},
        evidence_refs=(outputs["asset_resolution"], gate_report_ref),
    )
    gate_refs[aggregate.gate_id] = final_store.publish(
        task_id=_TASK_ID,
        artifact_kind="gate_g3",
        config_hash=config.config_hash,
        run_intent="formal",
        payload=aggregate.to_dict(),
        formal_eligible=True,
        source_refs=(outputs["asset_resolution"], gate_report_ref),
    ).commit_ref

    final_capability_refs = _publish_capabilities(
        store=final_store,
        config_hash=config.config_hash,
        checked_at=materialization.checked_at,
        resolution_hash=materialization.resolution_artifact_hash,
        groups=groups,
        evidence_refs=(outputs["asset_resolution"], gate_report_ref),
        phase="formal_stage0_04",
    )
    refs = dict(base_environment.evidence_refs)
    refs.update(
        {
            f"capability_{capability}": reference
            for capability, reference in final_capability_refs.items()
        }
    )
    refs.update({_gate_key(gate_id): reference for gate_id, reference in gate_refs.items()})
    refs.update(
        {
            "g3_asset_manifest": outputs["asset_manifest"],
            "g3_asset_audit": outputs["asset_audit"],
            "g3_resolution": outputs["asset_resolution"],
            "g3_gate_report": gate_report_ref,
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=base_environment.capabilities | frozenset(final_capability_refs),
        frozen_contract_stages=base_environment.frozen_contract_stages,
        passed_gate_ids=base_environment.passed_gate_ids
        | frozenset(gate_refs),
        estimator_decision_ref=base_environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    write_canonical_json(root / environment_ref, environment.to_dict())

    index_payload: dict[str, JSONValue] = {
        "schema_version": "stage0-g3-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": materialization.checked_at,
        "bootstrap_index_ref": bootstrap_index_ref,
        "bootstrap_index_sha256": bootstrap_index_sha,
        "materialization_index_ref": materialization.index_ref,
        "materialization_index_sha256": materialization.index_sha256,
        "materialization_resolution_artifact_hash": materialization.resolution_artifact_hash,
        "unlock_environment_ref": unlock_environment_ref,
        "unlock_environment_hash": unlock_environment.environment_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_report_ref": gate_report_ref,
        "gate_refs": dict(gate_refs),
        "capability_refs": dict(final_capability_refs),
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.05_config_run_identity_and_seeds",
        "next_input_refs": list(outputs.values()),
    }
    index_payload["artifact_hash"] = canonical_json_hash(index_payload)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(root / index_ref, index_payload)
    return Stage0G3FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        capability_refs=final_capability_refs,
        gate_refs=gate_refs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
        materialization=materialization,
    )


__all__ = [
    "Stage0G3FormalizationError",
    "Stage0G3FormalizationResult",
    "Stage0G3FormalState",
    "Stage0G3MaterializationBinding",
    "formalize_stage0_g3",
    "load_and_replay_g3_materialization",
    "load_stage0_g3_formal_state",
]
