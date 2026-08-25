"""Strict preparation and rebinding contract for formal S2.5.

This module is deliberately separate from the S2.4 producer and from the
fixed-state provider.  It only discovers a *complete* fresh S2.4 output,
validates an independently published six-cell G2.3 PASS, and returns the
immutable inputs needed by the S2.5 launcher.  It never reads gradients or
creates a confirmatory mapping.

``S25RebindSpec.execution_commit`` is the fresh S2.4 execution identity.  It
must be written explicitly into every COMPLETE final-status artifact.  The
upstream ``FormalExecutionEvidence.metadata.execution_commit`` is a different
authorization/producer identity; this module validates and reports it without
mistaking it for the fresh S2.4 execution commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.g3_gate import (
    g3_resolution_artifact_hash,
    validate_stage0_g3_resolution,
)
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact


EXPECTED_CELL_IDS = (
    "pythia-14m:initialization",
    "pythia-14m:early",
    "pythia-14m:mid_late",
    "pythia-31m-deduped:initialization",
    "pythia-31m-deduped:early",
    "pythia-31m-deduped:mid_late",
)

CELL_COMPONENTS = {
    "pythia-14m:initialization": "pythia-14m__initialization",
    "pythia-14m:early": "pythia-14m__early",
    "pythia-14m:mid_late": "pythia-14m__mid_late",
    "pythia-31m-deduped:initialization": "pythia-31m-deduped__initialization",
    "pythia-31m-deduped:early": "pythia-31m-deduped__early",
    "pythia-31m-deduped:mid_late": "pythia-31m-deduped__mid_late",
}

APPROVED_GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)
EXCLUDED_PCI = "0000:50:00.0"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_G23_GATE_ID = "stage2.G2.3"
_S204_STATUS_SCHEMA = "stage2-s204-cell-final-status-v3"
_TASK_RESULT_SCHEMA = "task-run-result-v2"
_FORMAL_EXECUTION_SCHEMA = "formal-execution-evidence-v1"
_ARTIFACT_SCHEMAS = {
    "reference_result": "reference-result-v1",
    "reference_convergence_report": "stage2-reference-convergence-report-v1",
    "gate_record": "stage23-task-gate-candidate-v1",
}


class S25RebindBlocked(RuntimeError):
    """Raised when preparation cannot prove the S2.5 source contract."""


def _canonical_hash_without_artifact(value: Mapping[str, Any]) -> str:
    return canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})


def _logical_path(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S25RebindBlocked(f"{field}:INVALID_LOGICAL_PATH")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S25RebindBlocked(f"{field}:PATH_ESCAPE")
    candidate = root / path
    resolved_root = root.resolve()
    try:
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise S25RebindBlocked(f"{field}:PATH_ESCAPE_OR_UNREADABLE") from error
    # Evidence is content-addressed; a symlink is not a stable input identity.
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise S25RebindBlocked(f"{field}:SYMLINK_FORBIDDEN")
    return resolved


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S25RebindBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(value, dict):
        raise S25RebindBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise S25RebindBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise S25RebindBlocked(f"{field}:COMMIT_REQUIRED")
    return value


def _validate_hashed_object(value: Mapping[str, Any], *, field: str) -> str:
    declared = _require_hash(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    if declared != _canonical_hash_without_artifact(value):
        raise S25RebindBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _validate_g3_resolution(spec: "S25RebindSpec") -> dict[str, str]:
    """Load the immutable G3 commit and validate both envelope and payload IDs."""

    root = spec.data_root.resolve()
    try:
        loaded = load_committed_task_artifact(root, spec.g3_ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise S25RebindBlocked(f"G3_RESOLUTION_LOAD_FAILED:{type(error).__name__}") from error
    if (
        loaded.identity.commit_ref != spec.g3_ref
        or loaded.identity.task_id != "stage0.04_assets_and_manifests"
        or loaded.identity.artifact_kind != "asset_resolution"
        or loaded.identity.formal_eligible is not True
        or loaded.run_intent != "formal"
    ):
        raise S25RebindBlocked("G3_RESOLUTION_COMMIT_IDENTITY_INVALID")
    if loaded.identity.artifact_hash != spec.g3_artifact_hash:
        raise S25RebindBlocked("G3_RESOLUTION_COMMIT_HASH_MISMATCH")
    payload = dict(loaded.payload)
    try:
        validate_stage0_g3_resolution(payload)
    except (TypeError, ValueError, KeyError) as error:
        raise S25RebindBlocked("G3_RESOLUTION_PAYLOAD_INVALID") from error
    if payload.get("status") != "PASS":
        raise S25RebindBlocked("G3_RESOLUTION_PASS_REQUIRED")
    payload_hash = g3_resolution_artifact_hash(payload)
    if payload.get("artifact_hash") != payload_hash:
        raise S25RebindBlocked("G3_RESOLUTION_PAYLOAD_HASH_MISMATCH")
    return {
        "commit_ref": loaded.identity.commit_ref,
        "commit_hash": loaded.identity.artifact_hash,
        "resolution_hash": payload_hash,
    }


def _validate_formal_lineage(
    root: Path,
    prepared_root: Path,
    environment: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, str, str, str]:
    evidence = environment.get("evidence_refs")
    if not isinstance(evidence, Mapping):
        raise S25RebindBlocked(f"{field}:EVIDENCE_REFS_REQUIRED")
    formal_ref = evidence.get("formal_execution")
    if not isinstance(formal_ref, str) or not formal_ref:
        raise S25RebindBlocked(f"{field}:FORMAL_EXECUTION_REF_REQUIRED")
    formal_path = _absolute_or_logical(root, formal_ref, field=f"{field}.formal_execution")
    try:
        formal_path.relative_to(prepared_root)
    except ValueError as error:
        raise S25RebindBlocked(f"{field}:FORMAL_EXECUTION_OUTSIDE_PREPARED_ROOT") from error
    lineage = _load_object(formal_path, field=f"{field}.formal_execution")
    if lineage.get("schema_version") != _FORMAL_EXECUTION_SCHEMA:
        raise S25RebindBlocked(f"{field}:FORMAL_EXECUTION_SCHEMA_INVALID")
    if lineage.get("run_intent") != "formal":
        raise S25RebindBlocked(f"{field}:FORMAL_EXECUTION_FORMAL_REQUIRED")
    lineage_hash = _validate_hashed_object(lineage, field=f"{field}.formal_execution")
    metadata = lineage.get("metadata")
    if not isinstance(metadata, Mapping):
        raise S25RebindBlocked(f"{field}.formal_execution:METADATA_REQUIRED")
    # This is upstream authorization provenance.  It is deliberately kept
    # separate from the fresh S2.4 execution commit carried by final-status.
    authorization_commit = _require_commit(
        metadata.get("execution_commit"),
        field=f"{field}.formal_execution.metadata.execution_commit",
    )
    authorization_producer_commit = _require_commit(
        metadata.get("producer_commit"),
        field=f"{field}.formal_execution.metadata.producer_commit",
    )
    return (
        formal_path.relative_to(root).as_posix(),
        lineage_hash,
        authorization_commit,
        authorization_producer_commit,
    )


def _absolute_or_logical(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise S25RebindBlocked(f"{field}:PATH_REQUIRED")
    path = Path(value)
    if path.is_absolute():
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError) as error:
            raise S25RebindBlocked(f"{field}:ABSOLUTE_PATH_OUTSIDE_DATA_ROOT") from error
        # Convert to a logical path before applying symlink checks.
        return _logical_path(root, resolved.relative_to(root.resolve()).as_posix(), field=field)
    return _logical_path(root, value, field=field)


@dataclass(frozen=True, slots=True)
class S25RebindSpec:
    """Paths and fixed identities for one fresh S2.4 -> S2.5 handoff."""

    data_root: Path
    s204_run_root: str
    s204_prepared_root: str
    g23_evaluation_root: str
    s205_output_root: str
    operations_root: str
    g3_ref: str
    g3_artifact_hash: str
    execution_commit: str

    def __post_init__(self) -> None:
        for field in (
            "s204_run_root",
            "s204_prepared_root",
            "g23_evaluation_root",
            "s205_output_root",
            "operations_root",
        ):
            _logical_path(self.data_root.resolve(), getattr(self, field), field=field)
        _logical_path(self.data_root.resolve(), self.g3_ref, field="g3_ref")
        if not isinstance(self.g3_artifact_hash, str) or _HASH_RE.fullmatch(self.g3_artifact_hash) is None:
            raise S25RebindBlocked("g3_artifact_hash:INVALID")
        if not isinstance(self.execution_commit, str) or _COMMIT_RE.fullmatch(self.execution_commit) is None:
            raise S25RebindBlocked("execution_commit:INVALID")


def _latest_g23(spec: S25RebindSpec) -> tuple[str, dict[str, Any]]:
    root = spec.data_root.resolve()
    evaluation_root = _logical_path(root, spec.g23_evaluation_root, field="g23_evaluation_root")
    candidates = sorted(evaluation_root.glob("g2.3-attempts/*/evaluation.json"))
    if not candidates:
        raise S25RebindBlocked("G2.3_EVALUATION_MISSING")
    valid: list[tuple[str, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            value = _load_object(candidate, field="g23_evaluation")
        except S25RebindBlocked:
            continue
        if value.get("schema_version") != "stage2-g23-reference-evaluation-v1":
            continue
        if value.get("gate_id") != _G23_GATE_ID:
            continue
        if value.get("status") != "PASS" or value.get("formal_eligible") is not True:
            continue
        if value.get("required_cell_count") != 6 or value.get("complete_cell_count") != 6:
            continue
        if value.get("expected_cell_ids") != list(EXPECTED_CELL_IDS):
            continue
        cells = value.get("cells")
        if (
            not isinstance(cells, list)
            or len(cells) != len(EXPECTED_CELL_IDS)
            or tuple(
                item.get("cell_id") if isinstance(item, dict) else None
                for item in cells
            )
            != EXPECTED_CELL_IDS
        ):
            continue
        if any(
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or not isinstance(item.get("identities"), dict)
            or _HASH_RE.fullmatch(str(item["identities"].get("result_hash", ""))) is None
            or _HASH_RE.fullmatch(str(item["identities"].get("config_hash", ""))) is None
            for item in cells
        ):
            continue
        if value.get("artifact_hash") != _canonical_hash_without_artifact(value):
            continue
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        valid.append((relative, value))
    if not valid:
        raise S25RebindBlocked("G2.3_STRICT_PASS_NOT_FOUND")
    return valid[-1]


def _validate_task_result(
    root: Path,
    status: Mapping[str, Any],
    refs: Mapping[str, str],
    *,
    field: str,
) -> dict[str, Any]:
    result_path = _absolute_or_logical(root, status.get("task_result_ref"), field=f"{field}.task_result_ref")
    result = _load_object(result_path, field=f"{field}.task_result")
    if result.get("schema_version") != _TASK_RESULT_SCHEMA:
        raise S25RebindBlocked(f"{field}:TASK_RESULT_SCHEMA_INVALID")
    result_hash = _require_hash(result.get("result_hash"), field=f"{field}.result_hash")
    if result_hash != canonical_json_hash(
        {key: item for key, item in result.items() if key != "result_hash"}
    ):
        raise S25RebindBlocked(f"{field}:TASK_RESULT_HASH_MISMATCH")
    if (
        result.get("task_id") != "stage2.04_reference_target"
        or result.get("stage") != 2
        or result.get("run_intent") != "formal"
        or result.get("status") != "PASS"
        or result.get("formal_eligible") is not True
        or result.get("config_hash") != status.get("config_hash")
        or result.get("artifact_refs") != dict(refs)
    ):
        raise S25RebindBlocked(f"{field}:TASK_RESULT_IDENTITY_INVALID")
    return result


def _validate_status_artifacts(
    root: Path,
    status: Mapping[str, Any],
    *,
    field: str,
) -> dict[str, str]:
    refs = status.get("artifact_refs")
    if not isinstance(refs, dict) or set(refs) != set(_ARTIFACT_SCHEMAS):
        raise S25RebindBlocked(f"{field}:ARTIFACT_SET_INVALID")
    normalized: dict[str, str] = {}
    for kind, expected_schema in _ARTIFACT_SCHEMAS.items():
        reference = refs.get(kind)
        if not isinstance(reference, str) or not reference:
            raise S25RebindBlocked(f"{field}.{kind}:ARTIFACT_REF_REQUIRED")
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S25RebindBlocked(
                f"{field}.{kind}:COMMITTED_ARTIFACT_INVALID:{type(error).__name__}"
            ) from error
        if (
            loaded.identity.task_id != "stage2.04_reference_target"
            or loaded.identity.artifact_kind != kind
            or loaded.identity.config_hash != status.get("config_hash")
            or loaded.identity.formal_eligible is not True
            or loaded.run_intent != "formal"
            or loaded.payload.get("schema_version") != expected_schema
            or not loaded.source_refs
        ):
            raise S25RebindBlocked(f"{field}.{kind}:ARTIFACT_IDENTITY_INVALID")
        if kind == "reference_result":
            _validate_hashed_object(loaded.payload, field=f"{field}.{kind}.payload")
        normalized[kind] = reference
    return normalized


def _complete_statuses(spec: S25RebindSpec, g23: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = spec.data_root.resolve()
    run_root = _logical_path(root, spec.s204_run_root, field="s204_run_root")
    prep_root = _logical_path(root, spec.s204_prepared_root, field="s204_prepared_root")
    rows: list[dict[str, Any]] = []
    cells = g23.get("cells")
    if not isinstance(cells, list):
        raise S25RebindBlocked("G2.3_CELLS_MISSING")
    by_cell = {item.get("cell_id"): item for item in cells if isinstance(item, dict)}
    lineage_identity: tuple[str, str, str, str] | None = None
    for index, cell_id in enumerate(EXPECTED_CELL_IDS):
        component = CELL_COMPONENTS[cell_id]
        statuses = sorted((run_root / component).rglob("final-status.json"))
        complete: list[dict[str, Any]] = []
        complete_paths: list[tuple[Path, dict[str, Any]]] = []
        for status_path in statuses:
            try:
                status = _load_object(status_path, field=f"{component}.final_status")
            except S25RebindBlocked:
                continue
            if status.get("status") == "COMPLETE" and status.get("cell_id") == cell_id:
                required = {
                    "schema_version",
                    "cell_id",
                    "config_path",
                    "config_hash",
                    "status",
                    "formal_eligible",
                    "task_result_hash",
                    "task_result_ref",
                    "artifact_refs",
                    "artifact_hash",
                    "execution_commit",
                }
                if not required.issubset(status):
                    raise S25RebindBlocked(f"{component}:STATUS_SCHEMA_INVALID")
                if status.get("schema_version") != _S204_STATUS_SCHEMA:
                    raise S25RebindBlocked(f"{component}:STATUS_SCHEMA_INVALID")
                if status.get("formal_eligible") is not True:
                    raise S25RebindBlocked(f"{component}:STATUS_FORMAL_ELIGIBILITY_INVALID")
                _require_hash(status.get("config_hash"), field=f"{component}.config_hash")
                _require_hash(status.get("task_result_hash"), field=f"{component}.task_result_hash")
                _validate_hashed_object(status, field=f"{component}.final_status")
                if status.get("execution_commit") != spec.execution_commit:
                    raise S25RebindBlocked(f"{component}:EXECUTION_COMMIT_MISMATCH")
                _require_commit(
                    status.get("execution_commit"),
                    field=f"{component}.final_status.execution_commit",
                )
                validated_refs = _validate_status_artifacts(
                    root,
                    status,
                    field=f"{component}.final_status",
                )
                _validate_task_result(
                    root,
                    status,
                    validated_refs,
                    field=f"{component}.final_status",
                )
                complete.append(status)
                complete_paths.append((status_path, status))
        if len(complete) != 1:
            raise S25RebindBlocked(f"S204_STATUS_NOT_UNIQUE:{component}:{len(complete)}")
        status_path, status = complete_paths[0]
        evaluated = by_cell.get(cell_id)
        if not isinstance(evaluated, dict) or evaluated.get("status") != "PASS":
            raise S25RebindBlocked(f"{component}:G2.3_CELL_NOT_PASS")
        identities = evaluated.get("identities")
        if not isinstance(identities, dict):
            raise S25RebindBlocked(f"{component}:G2.3_CELL_IDENTITY_MISSING")
        if identities.get("result_hash") != status.get("task_result_hash"):
            raise S25RebindBlocked(f"{component}:G2.3_TASK_RESULT_BINDING_MISMATCH")
        if identities.get("config_hash") != status.get("config_hash"):
            raise S25RebindBlocked(f"{component}:G2.3_CONFIG_BINDING_MISMATCH")
        config_path = _absolute_or_logical(root, status.get("config_path"), field=f"{component}.config_path")
        config = _load_object(config_path, field=f"{component}.config")
        if config.get("schema_version") not in {"resolved-config-v1", "resolved-config-v2"}:
            raise S25RebindBlocked(f"{component}:CONFIG_SCHEMA_INVALID")
        if config.get("config_hash") != status.get("config_hash"):
            raise S25RebindBlocked(f"{component}:CONFIG_STATUS_HASH_MISMATCH")
        env_path = _logical_path(prep_root, f"environments/{component}.json", field=f"{component}.environment")
        environment = _load_object(env_path, field=f"{component}.environment")
        if environment.get("schema_version") != "task-runtime-environment-v1":
            raise S25RebindBlocked(f"{component}:ENVIRONMENT_SCHEMA_INVALID")
        environment_hash = _require_hash(
            environment.get("environment_hash"),
            field=f"{component}.environment_hash",
        )
        if environment_hash != canonical_json_hash(
            {key: item for key, item in environment.items() if key != "environment_hash"}
        ):
            raise S25RebindBlocked(f"{component}:ENVIRONMENT_HASH_MISMATCH")
        evidence = environment.get("evidence_refs")
        if not isinstance(evidence, dict) or evidence.get("g3_resolution") != spec.g3_ref:
            raise S25RebindBlocked(f"{component}:G3_REF_MISMATCH")
        lineage = _validate_formal_lineage(
            root,
            prep_root,
            environment,
            field=f"{component}.environment",
        )
        if lineage_identity is None:
            lineage_identity = lineage
        elif lineage_identity != lineage:
            raise S25RebindBlocked(f"{component}:FORMAL_EXECUTION_LINEAGE_MISMATCH")
        orchestration = config.get("orchestration")
        if not isinstance(orchestration, dict):
            raise S25RebindBlocked(f"{component}:ORCHESTRATION_MISSING")
        input_refs = orchestration.get("input_result_refs", ())
        if not isinstance(input_refs, list):
            raise S25RebindBlocked(f"{component}:INPUT_RESULT_REFS_MISSING")
        # The formal S2.4 config must carry its Stage 2 handoff and assets
        # predecessors.  Keep this check narrow: the S2.5 launcher adds the
        # exact S2.3/S2.4 commit refs after this preflight.
        required_fragments = ("handoff_manifest.json", "fixed_state_contract.json")
        if not all(any(str(ref).endswith(fragment) for ref in input_refs) for fragment in required_fragments):
            raise S25RebindBlocked(f"{component}:S22_INPUT_BINDING_MISSING")
        refs = {str(key): str(value) for key, value in status["artifact_refs"].items()}
        rows.append(
            {
                "cell_id": cell_id,
                "component": component,
                "gpu_uuid": APPROVED_GPU_UUIDS[index % len(APPROVED_GPU_UUIDS)],
                "task_result_status_path": status_path.relative_to(root).as_posix(),
                "config_ref": config_path.relative_to(root).as_posix(),
                "environment_ref": env_path.relative_to(root).as_posix(),
                "reference_artifact_refs": refs,
                "config_hash": config.get("config_hash"),
                "status_artifact_hash": status.get("artifact_hash"),
                "result_hash": status.get("task_result_hash"),
                "execution_commit": spec.execution_commit,
                "formal_execution_ref": lineage[0],
                "formal_execution_hash": lineage[1],
                "authorization_execution_commit": lineage[2],
                "authorization_producer_commit": lineage[3],
            }
        )
    return rows


def prepare_s25_rebind(spec: S25RebindSpec) -> dict[str, Any]:
    """Return a strict, read-only S2.5 rebind plan.

    The function does not create ``s205_output_root`` and does not publish any
    plan or result.  A caller may persist the returned plan separately, but
    formal execution must only start after this function returns ``READY``.
    """

    root = spec.data_root.resolve()
    g3 = _validate_g3_resolution(spec)
    evaluation_ref, evaluation = _latest_g23(spec)
    rows = _complete_statuses(spec, evaluation)
    output = _logical_path(root, spec.s205_output_root, field="s205_output_root")
    if output.exists():
        raise S25RebindBlocked("S205_OUTPUT_ROOT_ALREADY_EXISTS")
    operations = _logical_path(root, spec.operations_root, field="operations_root")
    if operations.exists() and any(operations.iterdir()):
        raise S25RebindBlocked("S205_OPERATIONS_ROOT_NOT_EMPTY")
    return {
        "schema_version": "stage2-s205-rebind-plan-v1",
        "status": "READY",
        "formal_eligible": True,
        "execution_commit": spec.execution_commit,
        "formal_execution_ref": rows[0]["formal_execution_ref"],
        "formal_execution_hash": rows[0]["formal_execution_hash"],
        "authorization_execution_commit": rows[0]["authorization_execution_commit"],
        "authorization_producer_commit": rows[0]["authorization_producer_commit"],
        "g3_ref": spec.g3_ref,
        "g3_artifact_hash": g3["commit_hash"],
        "g3_resolution_payload_hash": g3["resolution_hash"],
        "excluded_pci": EXCLUDED_PCI,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "g23_evaluation_ref": evaluation_ref,
        "g23_evaluation_hash": evaluation["artifact_hash"],
        "s204_run_root": spec.s204_run_root,
        "s204_prepared_root": spec.s204_prepared_root,
        "s205_output_root": spec.s205_output_root,
        "operations_root": spec.operations_root,
        "cells": rows,
    }


__all__ = [
    "APPROVED_GPU_UUIDS",
    "CELL_COMPONENTS",
    "EXPECTED_CELL_IDS",
    "EXCLUDED_PCI",
    "S25RebindBlocked",
    "S25RebindSpec",
    "prepare_s25_rebind",
]
