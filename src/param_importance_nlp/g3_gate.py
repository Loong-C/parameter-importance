"""Fail-closed Stage 0 G3 asset resolution and sub-gate aggregation.

The requirements and layout artifacts are the trusted control plane.  Each
layout entry is resolved beneath one approved ``DATA_ROOT`` and admitted only
after the existing G3 resolver has replayed manifest/qualification bindings
and performed full size and SHA-256 verification.  This module then compares
the resolved candidate with the frozen requirement identity and aggregates
the five G3 sub-gates without turning an exception into a passing result.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from .asset_layout import load_stage0_asset_layout
from .asset_requirements import load_stage0_asset_requirements
from .assets import (
    AssetActorRole,
    AssetManifestError,
    AssetState,
    AssetType,
    ResolvedAsset,
    load_asset_manifest,
    resolve_qualified_asset,
    transition_manifest,
    validate_asset_path,
    validate_g3_manifest,
    validate_g3_qualification,
)
from .contracts import (
    CanonicalJSONError,
    GateRecord,
    GateStatus,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    loads_strict_json,
)
from .g3_semantic_evidence import (
    G3SemanticEvidenceError,
    validate_semantic_evidence,
)


SCHEMA_VERSION: Final = "stage0-g3-resolution-audit-v1"
GLUE_PREPROCESSING_PAYLOAD_SCHEMA_VERSION: Final = (
    "stage0-glue-preprocessing-config-v1"
)
GLUE_PREPROCESSING_VERSION: Final = "stage0-glue-preprocessing-v1"
GATE_IDS: Final = (
    "stage0.G3-S1",
    "stage0.G3-S2",
    "stage0.G3-S4",
    "stage0.G3-S5",
    "stage0.G3-S6",
)
QUALIFICATION_CHECK_IDS_BY_KIND: Final[Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "model": (
                "full_file_integrity",
                "model_semantic_contract",
                "offline_model_load",
            ),
            "tokenizer": (
                "full_file_integrity",
                "offline_tokenizer_load",
                "tokenizer_semantic_contract",
            ),
            "pile": (
                "full_file_integrity",
                "pile_causal_lm_contract",
                "pile_cursor_coverage",
                "pile_index_contract",
            ),
            "glue_raw": (
                "full_file_integrity",
                "glue_task_contract",
                "offline_glue_raw_load",
            ),
            "glue_derived": (
                "full_file_integrity",
                "glue_preprocessing_lineage",
                "glue_task_contract",
                "offline_glue_derived_load",
            ),
        }
    )
)
_CHECK_IDS: Final = (
    "input_paths_safe",
    "candidate_artifact_matches",
    "qualified_resolution",
    "qualification_check_set_matches",
    "acquisition_report_matches",
    "verification_report_matches",
    "semantic_evidence_matches",
    "identity_matches",
    "revision_matches",
    "expected_files_match",
    "semantic_metadata_matches",
    "cross_asset_binding_matches",
)
_VERIFICATION_REPORT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "asset_id",
        "state",
        "files_checked",
        "bytes_checked",
        "ok",
    }
)


class G3GateAggregationError(ValueError):
    """Raised when the trusted requirements/layout control plane is invalid."""


def qualification_check_ids_for(kind: str) -> tuple[str, ...]:
    """Return the exact ordered qualification check IDs for one layout kind."""

    try:
        return QUALIFICATION_CHECK_IDS_BY_KIND[kind]
    except (KeyError, TypeError) as error:
        raise G3GateAggregationError("G3_ENTRY_KIND_INVALID") from error


def g3_candidate_artifact_ref(logical_name: str, candidate_id: str) -> str:
    """Return the immutable run-scoped VERIFIED candidate reference."""

    if (
        not isinstance(candidate_id, str)
        or len(candidate_id) != 64
        or any(character not in "0123456789abcdef" for character in candidate_id)
    ):
        raise G3GateAggregationError("G3_CANDIDATE_ID_INVALID")
    return validate_asset_path(
        f"manifests/candidates/g3/{logical_name}/{candidate_id}/verified.json"
    )


def glue_preprocessing_config_payload(
    requirement: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shared, task-bound GLUE preprocessing identity payload."""

    if not isinstance(requirement, Mapping):
        raise TypeError("GLUE requirement must be a mapping")
    try:
        payload = {
            "schema_version": GLUE_PREPROCESSING_PAYLOAD_SCHEMA_VERSION,
            "task": deepcopy(requirement["task"]),
            "text_fields": deepcopy(requirement["text_fields"]),
            "label_mapping": deepcopy(requirement["label_mapping"]),
            "unlabeled_test_policy": deepcopy(
                requirement["unlabeled_test_policy"]
            ),
            "preprocessing": deepcopy(requirement["preprocessing"]),
        }
    except KeyError as error:
        raise G3GateAggregationError(
            "GLUE_PREPROCESSING_REQUIREMENT_INCOMPLETE"
        ) from error
    # Reject aliases and values outside the canonical JSON data model at this
    # public boundary, before a builder persists the payload or its hash.
    canonical_json_bytes(payload)
    return payload


def glue_preprocessing_config_hash(requirement: Mapping[str, Any]) -> str:
    """Hash the shared GLUE preprocessing identity payload canonically."""

    return canonical_json_hash(glue_preprocessing_config_payload(requirement))


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _approved_data_root(data_root: str | Path) -> Path:
    supplied = Path(data_root)
    if ".." in supplied.parts:
        raise G3GateAggregationError("DATA_ROOT_PARENT_TRAVERSAL")
    root = Path(os.path.abspath(supplied))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if (current.exists() or _is_link_like(current)) and _is_link_like(current):
            raise G3GateAggregationError("DATA_ROOT_LINK_FORBIDDEN")
    if not root.exists() or not root.is_dir():
        raise G3GateAggregationError("DATA_ROOT_MISSING_OR_NOT_DIRECTORY")
    return root


def _resolve_data_root_ref(root: Path, reference: str, *, directory: bool) -> Path:
    normalized = validate_asset_path(reference)
    relative = PurePosixPath(normalized)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise G3GateAggregationError("DATA_ROOT_REFERENCE_LINK_FORBIDDEN")
    if not candidate.exists():
        raise FileNotFoundError("DATA_ROOT_REFERENCE_MISSING")
    if directory and not candidate.is_dir():
        raise G3GateAggregationError("ASSET_ROOT_NOT_DIRECTORY")
    if not directory and not candidate.is_file():
        raise G3GateAggregationError("EVIDENCE_REF_NOT_REGULAR_FILE")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise G3GateAggregationError("DATA_ROOT_REFERENCE_ESCAPE") from error
    return resolved


def load_pile_reference_reader_oracle(
    data_root: str | Path,
    requirement: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and validate the frozen official-reader comparison artifact.

    The comparison is an externally produced diagnostic, so its raw bytes are
    hash-bound but are not required to use this repository's canonical JSON
    whitespace.  Its semantic shape and every official batch result remain
    exact and fail closed.
    """

    root = _approved_data_root(data_root)
    oracle = requirement.get("reference_reader_oracle")
    if not isinstance(oracle, Mapping):
        raise G3GateAggregationError("PILE_REFERENCE_ORACLE_REQUIREMENT_MISSING")
    try:
        artifact_ref = oracle["artifact_ref"]
        artifact_sha256 = oracle["artifact_sha256"]
        official_source_ref = oracle["official_source_ref"]
    except KeyError as error:
        raise G3GateAggregationError(
            "PILE_REFERENCE_ORACLE_REQUIREMENT_INCOMPLETE"
        ) from error
    path = _resolve_data_root_ref(root, artifact_ref, directory=False)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact_sha256:
        raise G3GateAggregationError("PILE_REFERENCE_ORACLE_HASH_MISMATCH")
    value = ensure_json_object(
        loads_strict_json(raw),
        field="Pile official-reader comparison",
    )
    if set(value) != {"official_source", "documents", "batches", "status"}:
        raise G3GateAggregationError("PILE_REFERENCE_ORACLE_FIELDS_INVALID")
    official_source = value["official_source"]
    if (
        not isinstance(official_source, str)
        or not official_source.startswith("/")
        or not official_source.endswith(f"/{official_source_ref}")
        or type(value["documents"]) is not int
        or value["documents"] != 1
        or not isinstance(value["batches"], list)
        or value["status"] != "ok"
    ):
        raise G3GateAggregationError("PILE_REFERENCE_ORACLE_CONTENT_INVALID")
    expected_hashes = requirement["reference_batch_sha256"]
    expected_steps = ("0", "1", "511")
    if len(value["batches"]) != len(expected_steps):
        raise G3GateAggregationError("PILE_REFERENCE_ORACLE_BATCH_SET_INVALID")
    batches: list[dict[str, Any]] = []
    batch_fields = {
        "step",
        "shape",
        "independent_sha256",
        "official_batch_viewer_sha256",
        "equal",
    }
    for raw_step, batch in zip(expected_steps, value["batches"], strict=True):
        if not isinstance(batch, Mapping) or set(batch) != batch_fields:
            raise G3GateAggregationError("PILE_REFERENCE_ORACLE_BATCH_FIELDS_INVALID")
        expected_digest = expected_hashes[raw_step]
        if (
            type(batch["step"]) is not int
            or batch["step"] != int(raw_step)
            or batch["shape"] != [requirement["reference_batch_size"], 2049]
            or batch["independent_sha256"] != expected_digest
            or batch["official_batch_viewer_sha256"] != expected_digest
            or batch["equal"] is not True
        ):
            raise G3GateAggregationError(
                "PILE_REFERENCE_ORACLE_BATCH_CONTENT_INVALID"
            )
        batches.append(deepcopy(dict(batch)))
    reader = requirement["reference_reader"]
    revision = reader["revision"]
    return {
        "artifact_ref": artifact_ref,
        "artifact_sha256": artifact_sha256,
        "official_source_ref": official_source_ref,
        "official_reader": {
            "repository": reader["repository"],
            "revision": revision,
            "commit": revision,
        },
        "documents": value["documents"],
        "batches": batches,
        "status": value["status"],
    }


def load_legacy_model_manifest_diagnostic(
    data_root: str | Path,
    requirement: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the two immutable BOM-prefixed 31M legacy manifest copies."""

    root = _approved_data_root(data_root)
    diagnostic = requirement.get("legacy_manifest_diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_REQUIREMENT_MISSING")
    try:
        refs = diagnostic["refs"]
        size_bytes = diagnostic["size_bytes"]
        digest = diagnostic["sha256"]
        condition = diagnostic["condition"]
        replacement_ref = diagnostic["replacement_manifest_ref"]
    except KeyError as error:
        raise G3GateAggregationError(
            "LEGACY_MODEL_MANIFEST_REQUIREMENT_INCOMPLETE"
        ) from error
    if not isinstance(refs, list) or len(refs) != 2:
        raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_REFS_INVALID")
    payloads: list[bytes] = []
    for reference in refs:
        path = _resolve_data_root_ref(root, reference, directory=False)
        raw = path.read_bytes()
        if len(raw) != size_bytes or hashlib.sha256(raw).hexdigest() != digest:
            raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_HASH_MISMATCH")
        payloads.append(raw)
    if payloads[0] != payloads[1]:
        raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_COPIES_DIFFER")
    raw = payloads[0]
    if not raw.startswith(b"\xef\xbb\xbf"):
        raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_BOM_MISSING")
    try:
        loads_strict_json(raw)
    except CanonicalJSONError:
        pass
    else:
        raise G3GateAggregationError(
            "LEGACY_MODEL_MANIFEST_STRICT_JSON_NOT_REJECTED"
        )
    # Prove the diagnostic is a valid legacy JSON document whose only accepted
    # compatibility exception is the already-observed UTF-8 BOM.
    ensure_json_object(
        loads_strict_json(raw, allow_bom=True),
        field="31M legacy model manifest",
    )
    if condition != "utf8_bom_strict_json_rejected" or replacement_ref in refs:
        raise G3GateAggregationError("LEGACY_MODEL_MANIFEST_CONDITION_INVALID")
    return {
        "refs": list(refs),
        "size_bytes": size_bytes,
        "sha256": digest,
        "condition": condition,
        "replacement_manifest_ref": replacement_ref,
    }


def _load_qualification(path: Path) -> dict[str, Any]:
    value = dict(
        ensure_json_object(
            loads_strict_json(path.read_bytes()),
            field="G3 qualification",
        )
    )
    validate_g3_qualification(value)
    return value


def _qualification_check_set_matches(
    kind: str,
    qualification: Mapping[str, Any],
) -> bool:
    observed = tuple(check["check_id"] for check in qualification["checks"])
    return observed == qualification_check_ids_for(kind)


def _load_lifecycle_reports(
    root: Path,
    entry: Mapping[str, Any],
    qualification: Mapping[str, Any],
    verified: Mapping[str, Any],
    *,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the exact acquisition -> DOWNLOADED -> VERIFIED chain."""

    from .g3_lifecycle_evidence import (
        _candidate_from_acquisition_entry,
        g3_acquisition_report_ref,
        g3_downloaded_candidate_ref,
        g3_verification_report_ref,
        validate_g3_acquisition_report,
        validate_g3_verify_report,
    )

    acquisition_ref = qualification["acquisition_ref"]
    acquisition_path = _resolve_data_root_ref(
        root, acquisition_ref, directory=False
    )
    acquisition_raw = acquisition_path.read_bytes()
    acquisition = dict(
        ensure_json_object(
            loads_strict_json(acquisition_raw), field="G3 acquisition report"
        )
    )
    if acquisition_raw != canonical_json_bytes(acquisition):
        raise G3GateAggregationError("ACQUISITION_REPORT_NOT_CANONICAL")
    validate_g3_acquisition_report(
        acquisition, requirements=requirements, layout=layout
    )
    if (
        acquisition_ref
        != g3_acquisition_report_ref(acquisition["artifact_hash"])
        or qualification["acquisition_sha256"] != acquisition["artifact_hash"]
    ):
        raise G3GateAggregationError("ACQUISITION_REPORT_BINDING_MISMATCH")
    download_report_path = _resolve_data_root_ref(
        root, acquisition["download_report_ref"], directory=False
    )
    if (
        hashlib.sha256(download_report_path.read_bytes()).hexdigest()
        != acquisition["download_report_sha256"]
    ):
        raise G3GateAggregationError("DOWNLOAD_REPORT_BINDING_MISMATCH")

    acquisition_entry = next(
        (
            dict(item)
            for item in acquisition["entries"]
            if item["logical_name"] == entry["logical_name"]
        ),
        None,
    )
    if acquisition_entry is None:
        raise G3GateAggregationError("ACQUISITION_ENTRY_MISSING")
    downloaded = _candidate_from_acquisition_entry(acquisition, acquisition_entry)
    downloaded_ref = g3_downloaded_candidate_ref(
        entry["logical_name"], downloaded["candidate_id"]
    )
    downloaded_path = _resolve_data_root_ref(root, downloaded_ref, directory=False)
    downloaded_raw = downloaded_path.read_bytes()
    downloaded_value = dict(
        ensure_json_object(
            loads_strict_json(downloaded_raw), field="G3 DOWNLOADED candidate"
        )
    )
    if (
        downloaded_raw != canonical_json_bytes(downloaded_value)
        or downloaded_value != downloaded
    ):
        raise G3GateAggregationError("DOWNLOADED_CANDIDATE_CONTENT_MISMATCH")

    verification_ref = qualification["verification_ref"]
    verification_path = _resolve_data_root_ref(
        root, verification_ref, directory=False
    )
    verification_raw = verification_path.read_bytes()
    verification = dict(
        ensure_json_object(
            loads_strict_json(verification_raw), field="G3 verify-only report"
        )
    )
    if verification_raw != canonical_json_bytes(verification):
        raise G3GateAggregationError("VERIFICATION_REPORT_NOT_CANONICAL")
    validate_g3_verify_report(
        verification,
        acquisition=acquisition,
        requirements=requirements,
        layout=layout,
    )
    if (
        verification["status"] != "PASS"
        or verification_ref
        != g3_verification_report_ref(acquisition["artifact_hash"])
        or qualification["verification_sha256"]
        != verification["artifact_hash"]
    ):
        raise G3GateAggregationError("VERIFICATION_REPORT_BINDING_MISMATCH")
    verification_entry = next(
        (
            dict(item)
            for item in verification["entries"]
            if item["logical_name"] == entry["logical_name"]
        ),
        None,
    )
    if (
        verification_entry is None
        or verification_entry["status"] != "PASS"
        or verification_entry["candidate_id"] != downloaded["candidate_id"]
        or verification_entry["asset_id"] != downloaded["asset_id"]
        or verification_entry["downloaded_manifest_ref"] != downloaded_ref
        or verification_entry["downloaded_manifest_sha256"]
        != hashlib.sha256(downloaded_raw).hexdigest()
        or any(item["status"] != "PASS" for item in verification_entry["files"])
    ):
        raise G3GateAggregationError("VERIFICATION_ENTRY_CONTENT_INVALID")
    expected_verified = transition_manifest(
        downloaded,
        AssetState.VERIFIED,
        actor=verification["actor"],
        actor_role=AssetActorRole.VERIFIER,
        actor_instance_id=verification["actor_instance_id"],
        evidence_ref=verification_ref,
        evidence_sha256=verification["artifact_hash"],
        summary="independent verify-only process matched every declared file",
        at=verification["checked_at"],
    )
    if expected_verified != verified:
        raise G3GateAggregationError("VERIFIED_CANDIDATE_HISTORY_MISMATCH")
    return acquisition, verification


def _load_candidate_artifact(
    root: Path,
    entry: Mapping[str, Any],
    ready: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    reference = g3_candidate_artifact_ref(
        entry["logical_name"], ready["candidate_id"]
    )
    path = _resolve_data_root_ref(root, reference, directory=False)
    raw = path.read_bytes()
    candidate = dict(
        ensure_json_object(
            loads_strict_json(raw),
            field="G3 VERIFIED candidate artifact",
        )
    )
    if raw != canonical_json_bytes(candidate):
        raise G3GateAggregationError("CANDIDATE_ARTIFACT_NOT_CANONICAL")
    validate_g3_manifest(candidate)
    verified = deepcopy(dict(ready))
    verified["state"] = "verified"
    verified["state_history"].pop()
    validate_g3_manifest(verified)
    if candidate != verified:
        raise G3GateAggregationError("CANDIDATE_ARTIFACT_CONTENT_MISMATCH")
    return candidate, reference, hashlib.sha256(raw).hexdigest()


def _load_semantic_evidence(
    root: Path,
    entry: Mapping[str, Any],
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    qualification: Mapping[str, Any],
    asset_root: Path,
    *,
    requirements_ref: str,
    requirements_hash: str,
) -> dict[str, Any]:
    """Load and bind the semantic evidence named by every non-file check."""

    expected_all = qualification_check_ids_for(entry["kind"])
    expected_semantic = tuple(
        check_id for check_id in expected_all if check_id != "full_file_integrity"
    )
    checks = qualification["checks"]
    observed = tuple(check["check_id"] for check in checks)
    by_id = {check["check_id"]: check for check in checks}
    if observed != expected_all or len(by_id) != len(checks):
        raise G3GateAggregationError("QUALIFICATION_CHECK_SET_MISMATCH")

    full_integrity = by_id["full_file_integrity"]
    if (
        full_integrity["evidence_ref"] != qualification["verification_ref"]
        or full_integrity["evidence_sha256"]
        != qualification["verification_sha256"]
    ):
        raise G3GateAggregationError("FULL_INTEGRITY_EVIDENCE_BINDING_MISMATCH")

    semantic_bindings = {
        (by_id[check_id]["evidence_ref"], by_id[check_id]["evidence_sha256"])
        for check_id in expected_semantic
    }
    if len(semantic_bindings) != 1:
        raise G3GateAggregationError("SEMANTIC_EVIDENCE_BINDING_SET_MISMATCH")
    semantic_ref, semantic_sha256 = next(iter(semantic_bindings))
    path = _resolve_data_root_ref(root, semantic_ref, directory=False)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != semantic_sha256:
        raise G3GateAggregationError("SEMANTIC_EVIDENCE_HASH_MISMATCH")
    evidence = dict(
        ensure_json_object(
            loads_strict_json(raw),
            field="G3 semantic evidence",
        )
    )
    if raw != canonical_json_bytes(evidence):
        raise G3GateAggregationError("SEMANTIC_EVIDENCE_NOT_CANONICAL")
    validate_semantic_evidence(evidence, expected_check_ids=expected_semantic)

    expected_binding = {
        "asset_id": manifest["asset_id"],
        "candidate_id": manifest["candidate_id"],
        "logical_name": entry["logical_name"],
        "kind": entry["kind"],
        "requirements_ref": requirements_ref,
        "requirements_sha256": requirements_hash,
        "checked_at": qualification["checked_at"],
        "generator_git_commit": manifest["generator_git_commit"],
    }
    if (
        qualification["generator_git_commit"]
        != manifest["generator_git_commit"]
    ):
        raise G3GateAggregationError("QUALIFICATION_GENERATOR_BINDING_MISMATCH")
    if any(
        evidence.get(field) != expected
        for field, expected in expected_binding.items()
    ):
        raise G3GateAggregationError("SEMANTIC_EVIDENCE_IDENTITY_BINDING_MISMATCH")
    if not semantic_observations_match(
        entry["kind"],
        requirement,
        manifest,
        evidence,
        asset_root=asset_root,
        data_root=root,
    ):
        raise G3GateAggregationError("SEMANTIC_OBSERVATION_BINDING_MISMATCH")
    return evidence


def _requirement_index(requirements: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for model in requirements["models"]:
        index[("model", model["name"])] = model
    tokenizer = requirements["tokenizer"]
    index[("tokenizer", tokenizer["name"])] = tokenizer
    index[("pile", "pile")] = requirements["pile"]
    for task in requirements["glue"]:
        index[("glue_raw", task["task"])] = task
        index[("glue_derived", task["task"])] = task
    return index


def _expected_asset_type(kind: str) -> AssetType:
    if kind == "model":
        return AssetType.MODEL
    if kind == "tokenizer":
        return AssetType.TOKENIZER
    return AssetType.DATASET


def _requirement_control_evidence_refs(
    kind: str,
    requirement: Mapping[str, Any],
) -> tuple[str, ...]:
    if kind == "pile":
        return (requirement["reference_reader_oracle"]["artifact_ref"],)
    if kind == "model" and "legacy_manifest_diagnostic" in requirement:
        return tuple(requirement["legacy_manifest_diagnostic"]["refs"])
    return ()


def _expected_files(
    kind: str,
    requirement: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    if kind in {"model", "tokenizer"}:
        values = requirement["files"]
    elif kind == "pile":
        values = [requirement["index"], *requirement["selected_shards"]]
    elif kind == "glue_raw":
        values = requirement["raw_files"]
    else:
        # Derived GLUE output files are admitted by their qualification and
        # bound preprocessing lineage; the requirements freeze has no byte
        # inventory for them.
        return None
    return sorted(
        (
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "role": item["role"],
            }
            for item in values
        ),
        key=lambda item: item["path"],
    )


def _manifest_files(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "role": item.get("role"),
            }
            for item in manifest["files"]
        ),
        key=lambda item: item["path"],
    )


def _model_semantics_match(
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    metadata = manifest["metadata"]
    dtype_counts = requirement["dtype_counts"]
    if not isinstance(dtype_counts, Mapping) or not dtype_counts:
        return False
    expected_dtype_summary = (
        next(iter(dtype_counts)) if len(dtype_counts) == 1 else "mixed"
    )
    config = next(
        (item for item in requirement["files"] if item["role"] == "config"),
        None,
    )
    checkpoint = requirement["checkpoint"]
    if not isinstance(checkpoint, str) or not checkpoint.startswith("step"):
        return False
    try:
        training_step = int(checkpoint[4:])
    except ValueError:
        return False
    expected_kind = (
        "base_initialization" if training_step == 0 else "trained_checkpoint"
    )
    expected_initialization_id = (
        f"{requirement['repository']}@{requirement['revision']}:"
        f"{requirement['checkpoint']}"
    )
    return bool(
        config is not None
        and metadata.get("contract_version") == "stage0-model-metadata-v1"
        and metadata.get("architecture") == requirement["architecture"]
        and metadata.get("parameter_count") == requirement["parameter_count"]
        and metadata.get("tensor_count") == requirement["tensor_count"]
        and metadata.get("dtype") == expected_dtype_summary
        and metadata.get("dtype_counts") == dtype_counts
        and metadata.get("max_position_embeddings")
        == requirement["max_position_embeddings"]
        and metadata.get("config_path") == config["path"]
        and metadata.get("config_sha256") == config["sha256"]
        and metadata.get("initialization_kind") == expected_kind
        and metadata.get("initialization_id") == expected_initialization_id
        and metadata.get("training_step") == training_step
        and (
            requirement["initialization_kind"] == "base_initialization"
        )
        == (training_step == 0)
    )


def _tokenizer_semantics_match(
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    metadata = manifest["metadata"]
    config = next(
        (
            item
            for item in requirement["files"]
            if item["role"] == "tokenizer_config"
        ),
        None,
    )
    special_tokens = metadata.get("special_tokens")
    if config is None or not isinstance(special_tokens, Mapping):
        return False
    if set(special_tokens) != {
        "bos_token",
        "eos_token",
        "unk_token",
        "glue_padding_token",
    }:
        return False
    expected_ids = requirement["special_token_ids"]
    for name, token_id in expected_ids.items():
        identity = special_tokens.get(f"{name}_token")
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"token", "token_id"}
            or not isinstance(identity.get("token"), str)
            or not identity["token"]
            or identity.get("token_id") != token_id
        ):
            return False
    padding = special_tokens.get("glue_padding_token")
    expected_padding = requirement["glue_padding_token"]
    return bool(
        isinstance(padding, Mapping)
        and metadata.get("contract_version")
        == "stage0-tokenizer-metadata-v1"
        and padding.get("token") == expected_padding["token"]
        and padding.get("token_id") == expected_padding["token_id"]
        and metadata.get("tokenizer_class") == requirement["tokenizer_class"]
        and metadata.get("vocab_size") == requirement["vocab_size"]
        and metadata.get("token_count_with_added_tokens")
        == requirement["token_count_with_added_tokens"]
        and metadata.get("vocab_mapping_sha256")
        == requirement["vocab_mapping_sha256"]
        and metadata.get("glue_padding_policy")
        == requirement["glue_padding_token"]["policy"]
        and isinstance(metadata.get("implementation_version"), str)
        and metadata["implementation_version"].startswith("tokenizer-json-")
        and isinstance(metadata.get("normalization"), str)
        and bool(metadata["normalization"])
        and metadata.get("config_path") == config["path"]
        and metadata.get("config_sha256") == config["sha256"]
    )


def _pile_semantics_match(
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    metadata = manifest["metadata"]
    if (
        metadata.get("dataset_kind") != "raw_indexed_mmap"
        or metadata.get("raw_revision") != requirement["revision"]
    ):
        return False
    expected_splits = {
        item["name"]: {
            "sample_count": item["stop"] - item["start"],
            "fields": ["tokens"],
            "cursor": {"start": item["start"], "stop": item["stop"]},
        }
        for item in requirement["cursor_intervals"]
    }
    if metadata.get("splits") != expected_splits:
        return False
    storage = metadata.get("storage")
    if not isinstance(storage, Mapping):
        return False
    index = requirement["index"]
    index_contract = requirement["index_contract"]
    expected_magic = bytes.fromhex(index_contract["magic_hex"]).rstrip(b"\0").decode(
        "ascii"
    )
    expected_idx = {
        "path": index["path"],
        "sha256": index["sha256"],
        "magic": expected_magic,
        "version": index_contract["version"],
        "dtype_code": index_contract["dtype_code"],
        "itemsize_bytes": index_contract["dtype_bytes"],
        "sequence_count": index_contract["sequence_count"],
        "document_count": index_contract["document_count"],
    }
    expected_shards = [
        {
            "ordinal": item["ordinal"],
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "byte_start": item["byte_start"],
            "byte_stop": item["byte_stop"],
        }
        for item in requirement["selected_shards"]
    ]
    coverage_stop = expected_shards[-1]["byte_stop"] if expected_shards else 0
    causal = requirement["causal_lm_contract"]
    expected_causal_mapping = {
        "labels_alignment": causal["labels_alignment"],
        "source_tokens_per_record": causal["source_tokens_per_record"],
        "input_sequence_length": causal["input_sequence_length"],
        "label_sequence_length": causal["target_sequence_length"],
        "input_slice": causal["input_slice"],
        "label_slice": causal["target_slice"],
        "attention_mask_policy": causal["attention_mask_policy"],
        "effective_target_tokens": causal["effective_target_tokens_per_record"],
        "loss_adapter_id": causal["loss_adapter_id"],
    }
    return bool(
        storage.get("kind") == "pythia_mmap_shards"
        and storage.get("idx") == expected_idx
        and storage.get("tokens_per_record")
        == requirement["causal_lm_contract"]["source_tokens_per_record"]
        and storage.get("global_byte_coverage")
        == {"start": 0, "stop": coverage_stop}
        and storage.get("required_cursor_stop")
        == requirement["required_cursor_stop"]
        and storage.get("causal_lm_mapping") == expected_causal_mapping
        and storage.get("reference_reader") == requirement["reference_reader"]
        and storage.get("reference_batch_size")
        == requirement["reference_batch_size"]
        and storage.get("reference_batch_sha256")
        == requirement["reference_batch_sha256"]
        and storage.get("last_required_record_sha256")
        == requirement["last_required_record_sha256"]
        and storage.get("cross_shard_policy")
        == requirement["cross_shard_policy"]
        and storage.get("shards") == expected_shards
    )


def _glue_split_metadata(
    requirement: Mapping[str, Any],
    *,
    derived: bool,
) -> dict[str, Any]:
    selected = (
        requirement["preprocessing"]["derived_splits"]
        if derived
        else list(requirement["split_counts"])
    )
    fields = (
        ["input_ids", "attention_mask", "labels"]
        if derived
        else [*requirement["text_fields"], "label"]
    )
    return {
        name: {
            "sample_count": requirement["split_counts"][name],
            "fields": fields,
            "cursor": {
                "start": 0,
                "stop": requirement["split_counts"][name],
            },
        }
        for name in selected
    }


def _glue_semantics_match(
    kind: str,
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    metadata = manifest["metadata"]
    derived = kind == "glue_derived"
    expected_kind = "derived_pretokenized" if derived else "hf_raw_parquet"
    if (
        metadata.get("dataset_kind") != expected_kind
        or metadata.get("raw_revision") != requirement["revision"]
        or metadata.get("splits")
        != _glue_split_metadata(requirement, derived=derived)
    ):
        return False
    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        return False
    expected_config_hash = glue_preprocessing_config_hash(requirement)
    if (
        metadata.get("preprocessing_version") != GLUE_PREPROCESSING_VERSION
        or preprocessing.get("version") != GLUE_PREPROCESSING_VERSION
        or preprocessing.get("config_hash") != expected_config_hash
    ):
        return False
    expected_task_contract = {
        "task": requirement["task"],
        "text_fields": requirement["text_fields"],
        "label_mapping": requirement["label_mapping"],
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
    }
    if metadata.get("task_contract") != expected_task_contract:
        return False
    storage = metadata.get("storage")
    if not isinstance(storage, Mapping):
        return False
    if derived:
        return storage.get("kind") == "hf_load_from_disk"
    expected_storage = {
        "kind": "hf_raw_parquet_files",
        "splits": {
            split: [
                {"path": item["path"], "sha256": item["sha256"]}
                for item in requirement["raw_files"]
                if item["role"] == split
            ]
            for split in requirement["split_counts"]
        },
    }
    return storage == expected_storage


def _semantic_metadata_matches(
    kind: str,
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    if kind == "model":
        return _model_semantics_match(requirement, manifest)
    if kind == "tokenizer":
        return _tokenizer_semantics_match(requirement, manifest)
    if kind == "pile":
        return _pile_semantics_match(requirement, manifest)
    return _glue_semantics_match(kind, requirement, manifest)


def _semantic_check_details(
    evidence: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        check["check_id"]: check["details"]
        for check in evidence["checks"]
    }


def _model_observations_match(
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    details: Mapping[str, Mapping[str, Any]],
    data_root: Path | None,
) -> bool:
    config = next(
        (item for item in requirement["files"] if item["role"] == "config"),
        None,
    )
    architecture = requirement["architecture"]
    if config is None or not architecture.endswith("ForCausalLM"):
        return False
    legacy_replacement = None
    if "legacy_manifest_diagnostic" in requirement:
        if data_root is None:
            return False
        legacy_replacement = load_legacy_model_manifest_diagnostic(
            data_root, requirement
        )
    expected_semantic = {
        "architecture": architecture,
        "max_position_embeddings": requirement["max_position_embeddings"],
        "config_sha256": config["sha256"],
        "legacy_manifest_replacement": legacy_replacement,
    }
    expected_offline = {
        "config_class": f"{architecture.removesuffix('ForCausalLM')}Config",
        "model_class": architecture,
        "parameter_count": requirement["parameter_count"],
        "tensor_count": requirement["tensor_count"],
        "max_position_embeddings": requirement["max_position_embeddings"],
        "local_files_only": True,
    }
    metadata = manifest["metadata"]
    expected_initialization_id = (
        f"{requirement['repository']}@{requirement['revision']}:"
        f"{requirement['checkpoint']}"
    )
    return bool(
        details.get("model_semantic_contract") == expected_semantic
        and details.get("offline_model_load") == expected_offline
        and metadata.get("initialization_id") == expected_initialization_id
    )


def _tokenizer_observations_match(
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    details: Mapping[str, Mapping[str, Any]],
    asset_root: Path | None,
) -> bool:
    if asset_root is None:
        return False
    metadata = manifest["metadata"]
    special_tokens = metadata.get("special_tokens")
    implementation = metadata.get("implementation_version")
    normalization = metadata.get("normalization")
    if (
        not isinstance(special_tokens, Mapping)
        or not isinstance(implementation, str)
        or not implementation.startswith("tokenizer-json-")
        or not isinstance(normalization, str)
        or not normalization
    ):
        return False
    special_descriptor = next(
        (
            item
            for item in requirement["files"]
            if item["role"] == "special_tokens"
        ),
        None,
    )
    tokenizer_descriptor = next(
        (
            item
            for item in requirement["files"]
            if item["role"] == "tokenizer_model"
        ),
        None,
    )
    if special_descriptor is None or tokenizer_descriptor is None:
        return False
    special_payload = ensure_json_object(
        loads_strict_json(
            (asset_root / PurePosixPath(special_descriptor["path"])).read_bytes()
        ),
        field="tokenizer special_tokens_map",
    )
    tokenizer_payload = ensure_json_object(
        loads_strict_json(
            (asset_root / PurePosixPath(tokenizer_descriptor["path"])).read_bytes()
        ),
        field="tokenizer.json",
    )
    tokenizer_version = tokenizer_payload.get("version")
    if not isinstance(tokenizer_version, str) or not tokenizer_version:
        return False
    expected_implementation = f"tokenizer-json-{tokenizer_version}"
    normalizer = tokenizer_payload.get("normalizer")
    expected_normalization = (
        "identity"
        if normalizer is None
        else f"tokenizer-json:{canonical_json_hash(normalizer)}"
    )
    if (
        implementation != expected_implementation
        or normalization != expected_normalization
    ):
        return False
    expected_special: dict[str, Any] = {}
    for name, token_id in requirement["special_token_ids"].items():
        key = f"{name}_token"
        raw_token = special_payload.get(key)
        token = (
            raw_token
            if isinstance(raw_token, str)
            else raw_token.get("content")
            if isinstance(raw_token, Mapping)
            else None
        )
        identity = special_tokens.get(key)
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(identity, Mapping)
            or set(identity) != {"token", "token_id"}
            or identity.get("token") != token
            or identity.get("token_id") != token_id
        ):
            return False
        expected_special[key] = dict(identity)
    padding = requirement["glue_padding_token"]
    expected_padding = {
        "token": padding["token"],
        "token_id": padding["token_id"],
        "policy": padding["policy"],
    }
    if special_tokens.get("glue_padding_token") != {
        "token": padding["token"],
        "token_id": padding["token_id"],
    }:
        return False
    expected_semantic = {
        "implementation_version": expected_implementation,
        "normalization": expected_normalization,
        "special_tokens": expected_special,
        "glue_padding_token": expected_padding,
    }
    expected_offline = {
        "tokenizer_class": requirement["tokenizer_class"],
        "vocab_size": requirement["vocab_size"],
        "token_count_with_added_tokens": requirement[
            "token_count_with_added_tokens"
        ],
        "vocab_mapping_sha256": requirement["vocab_mapping_sha256"],
    }
    return bool(
        details.get("offline_tokenizer_load") == expected_offline
        and details.get("tokenizer_semantic_contract") == expected_semantic
    )


def _pile_observations_match(
    requirement: Mapping[str, Any],
    details: Mapping[str, Mapping[str, Any]],
    data_root: Path | None,
) -> bool:
    if data_root is None:
        return False
    reference_reader_oracle = load_pile_reference_reader_oracle(
        data_root, requirement
    )
    causal = requirement["causal_lm_contract"]
    mapping = {
        "labels_alignment": causal["labels_alignment"],
        "source_tokens_per_record": causal["source_tokens_per_record"],
        "input_sequence_length": causal["input_sequence_length"],
        "label_sequence_length": causal["target_sequence_length"],
        "input_slice": causal["input_slice"],
        "label_slice": causal["target_slice"],
        "attention_mask_policy": causal["attention_mask_policy"],
        "effective_target_tokens": causal[
            "effective_target_tokens_per_record"
        ],
        "loss_adapter_id": causal["loss_adapter_id"],
    }
    required_stop = requirement["required_cursor_stop"]
    coverage_stop = requirement["selected_shards"][-1]["byte_stop"]
    last_end = (
        required_stop
        * causal["source_tokens_per_record"]
        * requirement["index_contract"]["dtype_bytes"]
    )
    expected_cursor = {
        "required_cursor_stop": required_stop,
        "last_required_byte_stop": last_end,
        "selected_coverage_stop": coverage_stop,
    }
    expected_causal = {
        "mapping": mapping,
        "reference_reader": requirement["reference_reader"],
        "reference_batch_size": requirement["reference_batch_size"],
        "reference_batch_sha256": requirement["reference_batch_sha256"],
        "reference_reader_oracle": reference_reader_oracle,
        "last_required_record": {
            "record_index": required_stop - 1,
            "sha256": requirement["last_required_record_sha256"],
        },
    }
    return bool(
        details.get("pile_index_contract") == requirement["index_contract"]
        and details.get("pile_cursor_coverage") == expected_cursor
        and details.get("pile_causal_lm_contract") == expected_causal
    )


def _glue_observations_match(
    kind: str,
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    details: Mapping[str, Mapping[str, Any]],
) -> bool:
    task_contract = {
        "task": requirement["task"],
        "text_fields": requirement["text_fields"],
        "label_mapping": requirement["label_mapping"],
        "unlabeled_test_policy": requirement["unlabeled_test_policy"],
    }
    if details.get("glue_task_contract") != task_contract:
        return False
    labels = sorted(requirement["label_mapping"].values())
    if kind == "glue_raw":
        split_details = {
            split: {
                "sample_count": count,
                "fields": sorted({*requirement["text_fields"], "label"}),
                "labels": [-1] if split.startswith("test") else labels,
            }
            for split, count in requirement["split_counts"].items()
        }
        return details.get("offline_glue_raw_load") == {
            "splits": split_details
        }
    preprocessing = manifest["metadata"].get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        return False
    lineage = {
        "config_hash": glue_preprocessing_config_hash(requirement),
        "tokenizer_asset_id": preprocessing.get("tokenizer_asset_id"),
        "parent_asset_ids": preprocessing.get("parent_asset_ids"),
    }
    split_details = {
        split: {
            "sample_count": requirement["split_counts"][split],
            "fields": ["attention_mask", "input_ids", "labels"],
            "labels": labels,
            "sequence_length": requirement["preprocessing"]["max_length"],
        }
        for split in requirement["preprocessing"]["derived_splits"]
    }
    return bool(
        details.get("glue_preprocessing_lineage") == lineage
        and details.get("offline_glue_derived_load")
        == {"splits": split_details}
    )


def semantic_observations_match(
    kind: str,
    requirement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    asset_root: Path | None = None,
    data_root: Path | None = None,
) -> bool:
    """Bind every semantic observation to the frozen requirement/manifest."""

    try:
        details = _semantic_check_details(evidence)
        if kind == "model":
            return _model_observations_match(
                requirement, manifest, details, data_root
            )
        if kind == "tokenizer":
            return _tokenizer_observations_match(
                requirement, manifest, details, asset_root
            )
        if kind == "pile":
            return _pile_observations_match(requirement, details, data_root)
        if kind in {"glue_raw", "glue_derived"}:
            return _glue_observations_match(
                kind, requirement, manifest, details
            )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return False


@dataclass(slots=True)
class _EntryState:
    entry: dict[str, Any]
    checks: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in _CHECK_IDS}
    )
    reasons: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None
    candidate_manifest: dict[str, Any] | None = None
    candidate_ref: str | None = None
    candidate_sha256: str | None = None
    qualification: dict[str, Any] | None = None
    acquisition_report: dict[str, Any] | None = None
    verification_report: dict[str, Any] | None = None
    semantic_evidence: dict[str, Any] | None = None
    resolved: ResolvedAsset | None = None
    ready_manifest_sha256: str | None = None
    files_checked: int = 0
    bytes_checked: int = 0
    expected_file_policy: str = "requirements_exact"

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) and not self.reasons

    def fail(self, check: str, reason: str) -> None:
        self.checks[check] = False
        if reason not in self.reasons:
            self.reasons.append(reason)

    def to_dict(self) -> dict[str, Any]:
        manifest = self.manifest or {}
        qualification = self.qualification or {}
        semantic = self.semantic_evidence or {}
        semantic_checks = [
            check
            for check in qualification.get("checks", [])
            if check.get("check_id") != "full_file_integrity"
        ]
        semantic_ref = (
            semantic_checks[0].get("evidence_ref") if semantic_checks else None
        )
        semantic_sha256 = (
            semantic_checks[0].get("evidence_sha256") if semantic_checks else None
        )
        return {
            "logical_name": self.entry["logical_name"],
            "kind": self.entry["kind"],
            "requirement_name": self.entry["requirement_name"],
            "gate_ids": list(self.entry["gate_ids"]),
            "manifest_ref": self.entry["manifest_ref"],
            "asset_root_ref": self.entry["asset_root_ref"],
            "qualification_ref": self.entry["qualification_ref"],
            "status": "PASS" if self.passed else "BLOCKED",
            "checks": dict(sorted(self.checks.items())),
            "reasons": sorted(self.reasons),
            "asset_id": manifest.get("asset_id"),
            "candidate_id": manifest.get("candidate_id"),
            "candidate_ref": self.candidate_ref,
            "candidate_sha256": self.candidate_sha256,
            "ready_manifest_sha256": self.ready_manifest_sha256,
            "qualification_artifact_hash": qualification.get("artifact_hash"),
            "acquisition_ref": qualification.get("acquisition_ref"),
            "acquisition_sha256": qualification.get("acquisition_sha256"),
            "verification_ref": qualification.get("verification_ref"),
            "verification_sha256": qualification.get("verification_sha256"),
            "semantic_evidence_ref": semantic_ref,
            "semantic_evidence_sha256": semantic_sha256,
            "semantic_evidence_artifact_hash": semantic.get("artifact_hash"),
            "files_checked": self.files_checked,
            "bytes_checked": self.bytes_checked,
            "expected_file_policy": self.expected_file_policy,
        }


def _evaluate_entry(
    root: Path,
    entry: Mapping[str, Any],
    requirement: Mapping[str, Any],
    *,
    requirements_ref: str,
    requirements_hash: str,
    requirements: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> _EntryState:
    state = _EntryState(entry=deepcopy(dict(entry)))
    if entry["kind"] == "glue_derived":
        state.expected_file_policy = "qualification_bound_derived_inventory"
    try:
        manifest_path = _resolve_data_root_ref(
            root, entry["manifest_ref"], directory=False
        )
        qualification_path = _resolve_data_root_ref(
            root, entry["qualification_ref"], directory=False
        )
        asset_root = _resolve_data_root_ref(
            root, entry["asset_root_ref"], directory=True
        )
        state.checks["input_paths_safe"] = True

        state.manifest = load_asset_manifest(manifest_path)
        state.qualification = _load_qualification(qualification_path)

        try:
            (
                state.candidate_manifest,
                state.candidate_ref,
                state.candidate_sha256,
            ) = _load_candidate_artifact(root, entry, state.manifest)
            state.checks["candidate_artifact_matches"] = True
        except FileNotFoundError:
            state.fail(
                "candidate_artifact_matches",
                "CANDIDATE_ARTIFACT_MISSING",
            )
        except (
            CanonicalJSONError,
            G3GateAggregationError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ):
            state.fail(
                "candidate_artifact_matches",
                "CANDIDATE_ARTIFACT_INVALID",
            )

        qualification_checks_match = _qualification_check_set_matches(
            entry["kind"], state.qualification
        )
        state.checks["qualification_check_set_matches"] = (
            qualification_checks_match
        )
        if not qualification_checks_match:
            state.reasons.append("QUALIFICATION_CHECK_SET_MISMATCH")

        try:
            lifecycle_verified = state.candidate_manifest
            if lifecycle_verified is None:
                # A missing VERIFIED artifact is recorded by the independent
                # candidate check.  The reports can still be replayed against
                # the READY manifest's exact VERIFIED parent so failure
                # attribution does not falsely invalidate acquisition.
                lifecycle_verified = deepcopy(state.manifest)
                lifecycle_verified["state"] = AssetState.VERIFIED.value
                lifecycle_verified["state_history"].pop()
                validate_g3_manifest(lifecycle_verified)
            (
                state.acquisition_report,
                state.verification_report,
            ) = _load_lifecycle_reports(
                root,
                entry,
                state.qualification,
                lifecycle_verified,
                requirements=requirements,
                layout=layout,
            )
            state.checks["acquisition_report_matches"] = True
            state.checks["verification_report_matches"] = True
        except FileNotFoundError:
            state.fail(
                "acquisition_report_matches",
                "LIFECYCLE_REPORT_MISSING",
            )
            state.fail(
                "verification_report_matches",
                "LIFECYCLE_REPORT_MISSING",
            )
        except (
            CanonicalJSONError,
            G3GateAggregationError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ) as error:
            message = str(error).casefold()
            if "verification" in message or "verify" in message:
                state.checks["acquisition_report_matches"] = True
                state.fail(
                    "verification_report_matches",
                    "VERIFICATION_REPORT_INVALID",
                )
            else:
                state.fail(
                    "acquisition_report_matches",
                    "ACQUISITION_REPORT_INVALID",
                )
                state.fail(
                    "verification_report_matches",
                    "VERIFICATION_REPORT_INVALID",
                )

        try:
            state.semantic_evidence = _load_semantic_evidence(
                root,
                entry,
                requirement,
                state.manifest,
                state.qualification,
                asset_root,
                requirements_ref=requirements_ref,
                requirements_hash=requirements_hash,
            )
            state.checks["semantic_evidence_matches"] = True
        except FileNotFoundError:
            state.fail(
                "semantic_evidence_matches",
                "SEMANTIC_EVIDENCE_MISSING",
            )
        except (
            CanonicalJSONError,
            G3GateAggregationError,
            G3SemanticEvidenceError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
        ):
            state.fail(
                "semantic_evidence_matches",
                "SEMANTIC_EVIDENCE_INVALID",
            )

        state.resolved = resolve_qualified_asset(
            state.manifest,
            asset_root,
            state.qualification,
            qualification_ref=entry["qualification_ref"],
            requirements_artifact_hash=requirements_hash,
        )
        state.checks["qualified_resolution"] = True
        state.ready_manifest_sha256 = canonical_json_hash(state.manifest)
        state.files_checked = len(state.resolved.files)
        state.bytes_checked = sum(item.size_bytes for item in state.resolved.files)

        if state.qualification["requirements_ref"] != requirements_ref:
            state.fail(
                "qualified_resolution",
                "QUALIFICATION_REQUIREMENTS_REF_MISMATCH",
            )

        expected_type = _expected_asset_type(entry["kind"])
        identity_matches = bool(
            state.manifest["asset_type"] == expected_type.value
            and state.manifest["name"] == entry["logical_name"]
            and state.manifest["source"] == requirement["repository"]
            and state.resolved.asset_type is expected_type
            and state.resolved.name == entry["logical_name"]
            and state.resolved.asset_id == state.manifest["asset_id"]
        )
        state.checks["identity_matches"] = identity_matches
        if not identity_matches:
            state.reasons.append("REQUIREMENT_IDENTITY_MISMATCH")

        revision_matches = bool(
            state.manifest["revision"] == requirement["revision"]
            and state.resolved.revision == requirement["revision"]
        )
        state.checks["revision_matches"] = revision_matches
        if not revision_matches:
            state.reasons.append("REQUIREMENT_REVISION_MISMATCH")

        expected_files = _expected_files(entry["kind"], requirement)
        files_match = (
            bool(state.manifest["files"])
            if expected_files is None
            else _manifest_files(state.manifest) == expected_files
        )
        state.checks["expected_files_match"] = files_match
        if not files_match:
            state.reasons.append("REQUIREMENT_FILE_INVENTORY_MISMATCH")

        semantics_match = _semantic_metadata_matches(
            entry["kind"], requirement, state.manifest
        )
        state.checks["semantic_metadata_matches"] = semantics_match
        if not semantics_match:
            state.reasons.append("REQUIREMENT_SEMANTIC_METADATA_MISMATCH")

        state.checks["cross_asset_binding_matches"] = True
    except FileNotFoundError:
        state.reasons.append("INPUT_MISSING")
    except (AssetManifestError, CanonicalJSONError):
        state.reasons.append("QUALIFIED_RESOLUTION_FAILED")
    except (G3GateAggregationError, OSError, TypeError, ValueError, KeyError):
        state.reasons.append("ENTRY_VALIDATION_FAILED")
    return state


def _apply_cross_asset_bindings(
    states: list[_EntryState],
    requirements: Mapping[str, Any],
) -> None:
    by_identity = {
        (state.entry["kind"], state.entry["requirement_name"]): state
        for state in states
    }
    models = {item["name"]: item for item in requirements["models"]}
    for state in states:
        if state.manifest is None or state.resolved is None:
            continue
        kind = state.entry["kind"]
        requirement_name = state.entry["requirement_name"]
        if kind == "model":
            requirement = models[requirement_name]
            checkpoint = requirement["checkpoint"]
            if checkpoint == "step0":
                expected_parent = None
            else:
                parent_requirement = next(
                    (
                        item
                        for item in requirements["models"]
                        if item["repository"] == requirement["repository"]
                        and item["checkpoint"] == "step0"
                    ),
                    None,
                )
                parent_state = (
                    None
                    if parent_requirement is None
                    else by_identity.get(("model", parent_requirement["name"]))
                )
                expected_parent = (
                    None
                    if parent_state is None or parent_state.resolved is None
                    else parent_state.resolved.asset_id
                )
            observed_parent = state.manifest["metadata"].get(
                "parent_model_asset_id"
            )
            if observed_parent != expected_parent:
                state.fail(
                    "cross_asset_binding_matches",
                    "MODEL_PARENT_ASSET_BINDING_MISMATCH",
                )
        elif kind == "glue_derived":
            raw_state = by_identity.get(("glue_raw", requirement_name))
            tokenizer_state = by_identity.get(("tokenizer", "pythia-tokenizer"))
            preprocessing = state.manifest["metadata"].get("preprocessing")
            expected_parent = (
                None
                if raw_state is None or raw_state.resolved is None
                else raw_state.resolved.asset_id
            )
            expected_tokenizer = (
                None
                if tokenizer_state is None or tokenizer_state.resolved is None
                else tokenizer_state.resolved.asset_id
            )
            if not isinstance(preprocessing, Mapping) or (
                preprocessing.get("parent_asset_ids") != [expected_parent]
                or expected_parent is None
                or preprocessing.get("tokenizer_asset_id") != expected_tokenizer
                or expected_tokenizer is None
            ):
                state.fail(
                    "cross_asset_binding_matches",
                    "DERIVED_ASSET_LINEAGE_MISMATCH",
                )


def g3_resolution_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return canonical_json_hash(payload)


def validate_stage0_g3_resolution(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "scope",
        "status",
        "checked_at",
        "requirements_ref",
        "requirements_artifact_hash",
        "layout_artifact_hash",
        "entries",
        "gates",
        "artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise G3GateAggregationError("G3_RESOLUTION_FIELDS_INVALID")
    if value["schema_version"] != SCHEMA_VERSION or value["scope"] != "formal":
        raise G3GateAggregationError("G3_RESOLUTION_SCHEMA_INVALID")
    if value["status"] not in {"PASS", "BLOCKED"}:
        raise G3GateAggregationError("G3_RESOLUTION_STATUS_INVALID")
    if not isinstance(value["entries"], list) or not isinstance(value["gates"], list):
        raise G3GateAggregationError("G3_RESOLUTION_ARRAYS_INVALID")
    gates = [GateRecord.from_mapping(dict(item)) for item in value["gates"]]
    if tuple(gate.gate_id for gate in gates) != GATE_IDS:
        raise G3GateAggregationError("G3_RESOLUTION_GATE_SET_INVALID")
    expected_status = (
        "PASS" if all(gate.status is GateStatus.PASS for gate in gates) else "BLOCKED"
    )
    if value["status"] != expected_status:
        raise G3GateAggregationError("G3_RESOLUTION_OVERALL_STATUS_MISMATCH")
    if value["artifact_hash"] != g3_resolution_artifact_hash(value):
        raise G3GateAggregationError("G3_RESOLUTION_ARTIFACT_HASH_MISMATCH")


def evaluate_stage0_g3(
    requirements_path: str | Path,
    layout_path: str | Path,
    data_root: str | Path,
    *,
    checked_at: str,
) -> dict[str, Any]:
    """Resolve all layout entries and return deterministic formal G3 gates.

    Invalid requirements/layout artifacts raise :class:`G3GateAggregationError`
    because no trustworthy gate matrix exists.  Entry-level problems become
    stable ``BLOCKED`` audit reasons and can never produce a PASS gate.
    """

    try:
        requirements = load_stage0_asset_requirements(requirements_path)
        layout = load_stage0_asset_layout(layout_path, requirements=requirements)
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise G3GateAggregationError("G3_CONTROL_PLANE_INVALID") from error
    root = _approved_data_root(data_root)
    requirements_hash = requirements["artifact_hash"]
    requirements_ref = layout["requirements_ref"]
    requirement_index = _requirement_index(requirements)

    states = [
        _evaluate_entry(
            root,
            entry,
            requirement_index[(entry["kind"], entry["requirement_name"])],
            requirements_ref=requirements_ref,
            requirements_hash=requirements_hash,
            requirements=requirements,
            layout=layout,
        )
        for entry in layout["entries"]
    ]
    _apply_cross_asset_bindings(states, requirements)

    gates: list[GateRecord] = []
    for gate_id in GATE_IDS:
        required_states = [
            state for state in states if gate_id in state.entry["gate_ids"]
        ]
        blocked_reasons = sorted(
            {
                f"{state.entry['logical_name']}:{reason}"
                for state in required_states
                for reason in state.reasons
            }
        )
        passed = bool(required_states) and all(state.passed for state in required_states)
        if not passed and not blocked_reasons:
            blocked_reasons = ["GATE_REQUIREMENT_NOT_EXACTLY_SATISFIED"]
        evidence_ref_values: list[str] = [requirements_ref]
        for state in required_states:
            evidence_ref_values.extend(
                (
                    state.entry["manifest_ref"],
                    state.entry["qualification_ref"],
                )
            )
            if state.candidate_ref is not None:
                evidence_ref_values.append(state.candidate_ref)
            if state.qualification is not None:
                evidence_ref_values.append(
                    state.qualification["acquisition_ref"]
                )
                evidence_ref_values.append(
                    state.qualification["verification_ref"]
                )
                evidence_ref_values.extend(
                    check["evidence_ref"]
                    for check in state.qualification.get("checks", [])
                )
            requirement = requirement_index[
                (state.entry["kind"], state.entry["requirement_name"])
            ]
            evidence_ref_values.extend(
                _requirement_control_evidence_refs(
                    state.entry["kind"], requirement
                )
            )
        evidence_refs = tuple(dict.fromkeys(evidence_ref_values))
        gates.append(
            GateRecord(
                gate_id=gate_id,
                stage=0,
                status=GateStatus.PASS if passed else GateStatus.BLOCKED,
                checked_at=checked_at,
                measured={
                    "required_entries": len(required_states),
                    "passed_entries": sum(state.passed for state in required_states),
                    "files_checked": sum(
                        state.files_checked for state in required_states
                    ),
                    "bytes_checked": sum(
                        state.bytes_checked for state in required_states
                    ),
                },
                threshold={
                    "required_entries": len(required_states),
                    "all_exactly_matched": True,
                },
                evidence_refs=evidence_refs,
                reasons=() if passed else tuple(blocked_reasons),
            )
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "formal",
        "status": (
            "PASS"
            if all(gate.status is GateStatus.PASS for gate in gates)
            else "BLOCKED"
        ),
        "checked_at": checked_at,
        "requirements_ref": requirements_ref,
        "requirements_artifact_hash": requirements_hash,
        "layout_artifact_hash": layout["artifact_hash"],
        "entries": [state.to_dict() for state in states],
        "gates": [gate.to_dict() for gate in gates],
    }
    payload["artifact_hash"] = g3_resolution_artifact_hash(payload)
    validate_stage0_g3_resolution(payload)
    return payload


__all__ = [
    "G3GateAggregationError",
    "GATE_IDS",
    "GLUE_PREPROCESSING_PAYLOAD_SCHEMA_VERSION",
    "GLUE_PREPROCESSING_VERSION",
    "QUALIFICATION_CHECK_IDS_BY_KIND",
    "SCHEMA_VERSION",
    "evaluate_stage0_g3",
    "g3_candidate_artifact_ref",
    "g3_resolution_artifact_hash",
    "glue_preprocessing_config_hash",
    "glue_preprocessing_config_payload",
    "load_legacy_model_manifest_diagnostic",
    "load_pile_reference_reader_oracle",
    "qualification_check_ids_for",
    "semantic_observations_match",
    "validate_stage0_g3_resolution",
]
