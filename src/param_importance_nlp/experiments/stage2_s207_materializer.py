"""Strict S2.7 formal-plan materialization.

This module is the small control-plane bridge between the immutable S2.6
freeze and :func:`prepare_s27_plan`.  It only reads already published
artifacts, canonicalizes S2.6's historical absolute references, and publishes
content-addressed direct G2.3 ``GateRecord`` objects for the S2.7 reference
loader.  It never creates draws, runs a provider, or starts a worker.

The S2.6 freeze predates the S2.7 operational failure-stop rule.  The rule is
therefore an explicit, hash-bound formal preregistration amendment input;
there is deliberately no fallback threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import LoadedTaskArtifact, load_committed_task_artifact
from ..runtime.task_lifecycle import publish_canonical_immutable
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import (
    S27_CORRECTED_DELTA_BATCH_SIZES,
    S27_CORRECTED_DELTA_SOURCE,
    S27Plan,
    prepare_s27_plan,
)
from .stage2_s207_runner import (
    load_s27_gpu_inventory_envelope,
    load_s27_materialized_inputs,
)
from .stage2_s25_rebind import validate_g23_evaluation
from .stage2_s25_formal import load_s25_rebind_plan


FAILURE_RULE_SCHEMA = "stage2-s207-failure-rule-amendment-v1"
FAILURE_RULE = "stop_after_failure_fraction_exceeds_max"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class S27MaterializationBlocked(RuntimeError):
    """Raised when a formal S2.7 input cannot be proven immutable."""


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise S27MaterializationBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _logical(root: Path, value: object, *, field: str) -> tuple[str, Path]:
    """Normalize an in-root absolute/relative ref to a safe logical POSIX ref."""

    if not isinstance(value, str) or not value:
        raise S27MaterializationBlocked(f"{field}:LOGICAL_REFERENCE_REQUIRED")
    supplied = Path(value)
    # S2.6 freeze commits were written with ``str(Path)`` and therefore use
    # native Windows separators for absolute refs.  Accept those only for the
    # initial in-root resolution; every returned identity is canonical POSIX.
    if "\\" in value and not supplied.is_absolute():
        raise S27MaterializationBlocked(f"{field}:LOGICAL_REFERENCE_REQUIRED")
    if supplied.is_absolute():
        try:
            resolved = supplied.resolve()
            relative = resolved.relative_to(root.resolve()).as_posix()
        except (OSError, ValueError) as error:
            raise S27MaterializationBlocked(f"{field}:ABSOLUTE_PATH_OUTSIDE_DATA_ROOT") from error
    else:
        relative = PurePosixPath(value).as_posix()
    logical = PurePosixPath(relative)
    if logical.is_absolute() or not logical.parts or any(part in {"", ".", ".."} for part in logical.parts):
        raise S27MaterializationBlocked(f"{field}:PATH_ESCAPE")
    current = root
    try:
        if current.is_symlink():
            raise S27MaterializationBlocked(f"{field}:SYMLINK_FORBIDDEN")
        for part in logical.parts:
            current = current / part
            if current.is_symlink():
                raise S27MaterializationBlocked(f"{field}:SYMLINK_FORBIDDEN")
        resolved = current.resolve()
        resolved.relative_to(root.resolve())
    except S27MaterializationBlocked:
        raise
    except (OSError, ValueError) as error:
        raise S27MaterializationBlocked(f"{field}:PATH_ESCAPE_OR_UNREADABLE") from error
    return logical.as_posix(), resolved


def _load_json(root: Path, value: object, *, field: str) -> tuple[str, dict[str, object]]:
    logical, path = _logical(root, value, field=field)
    try:
        loaded = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S27MaterializationBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(loaded, Mapping):
        raise S27MaterializationBlocked(f"{field}:OBJECT_REQUIRED")
    return logical, dict(loaded)


def _hashed(value: Mapping[str, object], *, field: str) -> str:
    declared = _sha(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != declared:
        raise S27MaterializationBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return declared


def _load_ref_payload(
    root: Path,
    value: object,
    *,
    field: str,
) -> tuple[str, dict[str, object], str, LoadedTaskArtifact | None]:
    logical, path = _logical(root, value, field=field)
    try:
        raw = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S27MaterializationBlocked(f"{field}:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(raw, Mapping):
        raise S27MaterializationBlocked(f"{field}:OBJECT_REQUIRED")
    if raw.get("schema_version") == "task-output-commit-v1":
        try:
            loaded = load_committed_task_artifact(root, logical, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S27MaterializationBlocked(f"{field}:FORMAL_TASK_COMMIT_INVALID") from error
        if not isinstance(loaded.payload, Mapping):
            raise S27MaterializationBlocked(f"{field}:TASK_PAYLOAD_INVALID")
        return logical, dict(loaded.payload), loaded.identity.artifact_hash, loaded
    return logical, dict(raw), _hashed(raw, field=field), None


def _require_iso(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise S27MaterializationBlocked(f"{field}:ISO_TIMESTAMP_REQUIRED")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise S27MaterializationBlocked(f"{field}:ISO_TIMESTAMP_INVALID") from error
    if parsed.tzinfo is None:
        raise S27MaterializationBlocked(f"{field}:TIMEZONE_REQUIRED")
    return value


def validate_failure_rule(
    data_root: str | Path,
    failure_rule_ref: object,
) -> tuple[str, str, float]:
    """Validate the explicit approved S2.7 failure-stop amendment.

    The parent is a formal S2.1 preregistration TaskArtifact commit.  The
    amendment itself is a direct canonical object and its own artifact hash is
    the identity stored in the S2.7 plan.
    """

    root = Path(data_root).resolve()
    logical, value = _load_json(root, failure_rule_ref, field="failure_rule")
    digest = _hashed(value, field="failure_rule")
    required = {
        "schema_version",
        "amendment_id",
        "task_id",
        "scope",
        "status",
        "formal_eligible",
        "approval_status",
        "approved_at",
        "rule",
        "parent_preregistration_ref",
        "parent_preregistration_hash",
        "parent_preregistration_payload_hash",
        "max_failure_fraction",
        "artifact_hash",
    }
    if set(value) != required or value.get("schema_version") != FAILURE_RULE_SCHEMA:
        raise S27MaterializationBlocked("failure_rule:FIELDS_OR_SCHEMA_INVALID")
    if (
        value.get("task_id") != "stage2.07_main_sweep"
        or value.get("scope") != "formal"
        or value.get("status") != "FROZEN"
        or value.get("formal_eligible") is not True
        or value.get("approval_status") != "APPROVED"
        or value.get("rule") != FAILURE_RULE
    ):
        raise S27MaterializationBlocked("failure_rule:FORMAL_APPROVED_FROZEN_REQUIRED")
    if not isinstance(value.get("amendment_id"), str) or _SAFE_ID.fullmatch(str(value["amendment_id"])) is None:
        raise S27MaterializationBlocked("failure_rule:AMENDMENT_ID_INVALID")
    _require_iso(value.get("approved_at"), field="failure_rule.approved_at")
    parent_ref, parent_payload, parent_hash, parent_commit = _load_ref_payload(
        root,
        value.get("parent_preregistration_ref"),
        field="failure_rule.parent_preregistration",
    )
    if parent_commit is None or parent_commit.identity.artifact_kind != "preregistration" or parent_commit.identity.task_id != "stage2.01_scope_hypotheses_and_preregistration":
        raise S27MaterializationBlocked("failure_rule:PARENT_PREREGISTRATION_TASK_COMMIT_REQUIRED")
    if value.get("parent_preregistration_hash") != parent_hash:
        raise S27MaterializationBlocked("failure_rule:PARENT_PREREGISTRATION_HASH_MISMATCH")
    payload_hash = _sha(parent_payload.get("preregistration_hash"), field="failure_rule.parent_preregistration_payload_hash")
    if value.get("parent_preregistration_payload_hash") != payload_hash or canonical_json_hash({key: item for key, item in parent_payload.items() if key != "preregistration_hash"}) != payload_hash:
        raise S27MaterializationBlocked("failure_rule:PARENT_PREREGISTRATION_PAYLOAD_HASH_MISMATCH")
    if parent_payload.get("schema_version") != "stage2-preregistration-v1" or parent_payload.get("scope") != "formal" or parent_payload.get("state") != "FROZEN":
        raise S27MaterializationBlocked("failure_rule:PARENT_PREREGISTRATION_NOT_FORMAL_FROZEN")
    raw_fraction = value.get("max_failure_fraction")
    if isinstance(raw_fraction, bool) or not isinstance(raw_fraction, (int, float)):
        raise S27MaterializationBlocked("failure_rule.MAX_FAILURE_FRACTION_REQUIRED")
    fraction = float(raw_fraction)
    if fraction != fraction or fraction in (float("inf"), float("-inf")) or not 0.0 <= fraction < 1.0:
        raise S27MaterializationBlocked("failure_rule.MAX_FAILURE_FRACTION_OUT_OF_RANGE")
    # Make sure the normalized ref is the value represented by the object;
    # this also prevents a caller from using an absolute path in the plan.
    if parent_ref != value.get("parent_preregistration_ref"):
        raise S27MaterializationBlocked("failure_rule:PARENT_REF_MUST_BE_LOGICAL")
    return logical, digest, fraction


def _load_execution(root: Path, ref: object) -> tuple[str, FormalExecutionEvidence]:
    logical, value = _load_json(root, ref, field="execution_evidence")
    try:
        evidence = FormalExecutionEvidence.from_mapping(dict(value))
        evidence.require_for_stage(2)
    except (TypeError, ValueError, RuntimeError) as error:
        raise S27MaterializationBlocked(f"execution_evidence:INVALID:{error}") from error
    return logical, evidence


def _load_freeze(root: Path, ref: object) -> tuple[str, dict[str, object], str]:
    logical, value = _load_json(root, ref, field="s206_freeze")
    digest = _hashed(value, field="s206_freeze")
    if value.get("schema_version") != "stage2-s206-formal-freeze-commit-v1" or value.get("status") != "PASS" or value.get("formal_eligible") is not True or value.get("confirmatory_gradients_started") is not False:
        raise S27MaterializationBlocked("s206_freeze:FORMAL_PASS_REQUIRED")
    return logical, value, digest


def _load_optional_freeze_sources(
    root: Path,
    freeze: Mapping[str, object],
) -> list[str]:
    """Revalidate optional S2.6 cost/retry source identities for provenance."""

    refs: list[str] = []
    for ref_field, hash_field in (
        ("cost_semantics_ref", "cost_semantics_hash"),
        ("retry_policy_ref", "retry_policy_hash"),
    ):
        raw_ref = freeze.get(ref_field)
        if raw_ref is None:
            continue
        logical, value = _load_json(root, raw_ref, field=f"s206_freeze.{ref_field}")
        expected_hash = _sha(freeze.get(hash_field), field=f"s206_freeze.{hash_field}")
        observed_hash = (
            _hashed(value, field=f"s206.{ref_field}")
            if "artifact_hash" in value
            else canonical_json_hash(value)
        )
        if observed_hash != expected_hash:
            raise S27MaterializationBlocked(f"s206_freeze:{ref_field.upper()}_HASH_BINDING_INVALID")
        refs.append(logical)
    return refs


def _validate_s206_artifacts(
    root: Path,
    freeze: Mapping[str, object],
    *,
    execution_ref: object,
    inventory_json: str | Path,
) -> tuple[str, dict[str, object], str, FormalExecutionEvidence, str, dict[str, object], str, str, dict[str, object], str]:
    """Load freeze-bound matrix/mapping/G2.4b/execution/inventory artifacts."""

    matrix_ref, matrix, matrix_hash = _load_json(root, freeze.get("matrix_ref"), field="s206_freeze.matrix_ref")
    if _hashed(matrix, field="s206.matrix") != matrix_hash or matrix_hash != freeze.get("matrix_hash"):
        raise S27MaterializationBlocked("s206_freeze:MATRIX_HASH_BINDING_INVALID")
    mapping_ref, mapping, mapping_hash = _load_json(root, freeze.get("confirmatory_mapping_ref"), field="s206_freeze.confirmatory_mapping_ref")
    if _hashed(mapping, field="s206.mapping") != mapping_hash or mapping_hash != freeze.get("confirmatory_mapping_hash"):
        raise S27MaterializationBlocked("s206_freeze:MAPPING_HASH_BINDING_INVALID")
    gate_ref, gate_value = _load_json(root, freeze.get("gate_ref"), field="s206_freeze.gate_ref")
    try:
        gate = GateRecord.from_mapping(dict(gate_value))
    except (TypeError, ValueError, RuntimeError) as error:
        raise S27MaterializationBlocked("s206_freeze:G24B_GATE_INVALID") from error
    if gate.gate_id != "stage2.G2.4b" or gate.status is not GateStatus.PASS or gate.artifact_hash != freeze.get("gate_hash"):
        raise S27MaterializationBlocked("s206_freeze:G24B_GATE_HASH_BINDING_INVALID")
    execution_ref_logical, execution = _load_execution(root, execution_ref)
    if execution.artifact_hash != freeze.get("execution_evidence_hash"):
        raise S27MaterializationBlocked("s206_freeze:EXECUTION_HASH_BINDING_INVALID")
    inventory_path = Path(inventory_json).resolve()
    try:
        inventory_path.relative_to(root)
    except ValueError as error:
        raise S27MaterializationBlocked("gpu_inventory:OUTSIDE_DATA_ROOT") from error
    try:
        summary, identity = load_s27_gpu_inventory_envelope(inventory_path, data_root=root)
    except Exception as error:
        raise S27MaterializationBlocked(f"gpu_inventory:INVALID:{error}") from error
    if (
        identity.get("source_ref") != freeze.get("gpu_inventory_ref")
        or identity.get("artifact_hash") != freeze.get("gpu_inventory_artifact_hash")
        or identity.get("source_sha256") != freeze.get("gpu_inventory_source_sha256")
    ):
        raise S27MaterializationBlocked("s206_freeze:GPU_INVENTORY_HASH_BINDING_INVALID")
    freeze_inventory_path = freeze.get("gpu_inventory_path")
    try:
        _freeze_inventory_ref, freeze_inventory_path_resolved = _logical(
            root,
            freeze_inventory_path,
            field="s206_freeze.gpu_inventory_path",
        )
    except S27MaterializationBlocked:
        raise
    if freeze_inventory_path_resolved != inventory_path:
        raise S27MaterializationBlocked("s206_freeze:GPU_INVENTORY_PATH_BINDING_INVALID")
    return (
        matrix_ref,
        matrix,
        matrix_hash,
        execution,
        execution_ref_logical,
        summary,
        str(identity["source_ref"]),
        str(identity["artifact_hash"]),
        dict(identity),
        gate_ref,
    )


def _load_materialized_checkpoints(
    root: Path,
    materialization_ref: object,
    s205_rows: Mapping[str, Mapping[str, object]],
) -> tuple[str, str, str, dict[str, dict[str, object]], list[str]]:
    logical, index_path = _logical(root, materialization_ref, field="materialization_index")
    try:
        raw_index = load_canonical_json(index_path)
    except (OSError, TypeError, ValueError) as error:
        raise S27MaterializationBlocked("materialization_index:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(raw_index, Mapping):
        raise S27MaterializationBlocked("materialization_index:OBJECT_REQUIRED")
    index_is_commit = raw_index.get("schema_version") == "task-output-commit-v1"
    if index_is_commit:
        try:
            loaded_index = load_committed_task_artifact(root, logical, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S27MaterializationBlocked("materialization_index:FORMAL_TASK_COMMIT_INVALID") from error
        if not isinstance(loaded_index.payload, Mapping):
            raise S27MaterializationBlocked("materialization_index:TASK_PAYLOAD_INVALID")
        index = dict(loaded_index.payload)
        index_hash = loaded_index.identity.artifact_hash
    else:
        index = dict(raw_index)
        index_hash = _hashed(index, field="materialization_index") if "index_hash" not in index else _sha(index.get("index_hash"), field="materialization_index.index_hash")
    if not index_is_commit and "index_hash" in index and index_hash != canonical_json_hash({key: item for key, item in index.items() if key != "index_hash"}):
        raise S27MaterializationBlocked("materialization_index:INDEX_HASH_MISMATCH")
    try:
        materialized = load_s27_materialized_inputs(root, logical)
    except Exception as error:
        raise S27MaterializationBlocked(f"materialization_index:INVALID:{error}") from error
    manifest_ref, manifest_path = _logical(root, index.get("six_cell_manifest_ref"), field="six_cell_manifest")
    try:
        raw_manifest = load_canonical_json(manifest_path)
    except (OSError, TypeError, ValueError) as error:
        raise S27MaterializationBlocked("six_cell_manifest:CANONICAL_JSON_REQUIRED") from error
    if not isinstance(raw_manifest, Mapping):
        raise S27MaterializationBlocked("six_cell_manifest:OBJECT_REQUIRED")
    if raw_manifest.get("schema_version") == "task-output-commit-v1":
        try:
            loaded_manifest = load_committed_task_artifact(root, manifest_ref, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S27MaterializationBlocked("six_cell_manifest:FORMAL_TASK_COMMIT_INVALID") from error
        if not isinstance(loaded_manifest.payload, Mapping):
            raise S27MaterializationBlocked("six_cell_manifest:TASK_PAYLOAD_INVALID")
        manifest = dict(loaded_manifest.payload)
    else:
        manifest = dict(raw_manifest)
    if manifest.get("schema_version") != "stage2-s204-six-cell-manifest-v1" or manifest.get("status") != "READY" or manifest.get("scope") != "formal":
        raise S27MaterializationBlocked("six_cell_manifest:FORMAL_READY_REQUIRED")
    declared_manifest = _sha(manifest.get("manifest_hash"), field="six_cell_manifest.manifest_hash")
    if declared_manifest != canonical_json_hash({key: item for key, item in manifest.items() if key != "manifest_hash"}):
        raise S27MaterializationBlocked("six_cell_manifest:HASH_MISMATCH")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or tuple(item.get("cell_id") for item in checkpoints if isinstance(item, Mapping)) != EXPECTED_CELL_IDS:
        raise S27MaterializationBlocked("six_cell_manifest:SIX_CELL_ORDER_INVALID")
    checkpoint_by_cell: dict[str, dict[str, object]] = {}
    source_refs = [logical, manifest_ref]
    for cell_id, row in zip(EXPECTED_CELL_IDS, checkpoints, strict=True):
        if not isinstance(row, Mapping) or row.get("cell_id") != cell_id:
            raise S27MaterializationBlocked(f"six_cell_manifest:CHECKPOINT_ROW_INVALID:{cell_id}")
        checkpoint_ref, checkpoint_path = _logical(root, row.get("checkpoint_manifest_ref"), field=f"checkpoint.{cell_id}.manifest_ref")
        checkpoint_hash = _sha(row.get("checkpoint_hash"), field=f"checkpoint.{cell_id}.hash")
        checkpoint_id = row.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise S27MaterializationBlocked(f"checkpoint.{cell_id}:ID_REQUIRED")
        try:
            checkpoint_raw = load_canonical_json(checkpoint_path)
        except (OSError, TypeError, ValueError) as error:
            raise S27MaterializationBlocked(f"checkpoint.{cell_id}:MANIFEST_UNREADABLE") from error
        if isinstance(checkpoint_raw, Mapping):
            direct_hash = checkpoint_raw.get("artifact_hash")
            if isinstance(direct_hash, str):
                if _hashed(checkpoint_raw, field=f"checkpoint.{cell_id}") != checkpoint_hash:
                    raise S27MaterializationBlocked(f"checkpoint.{cell_id}:HASH_MISMATCH")
            elif canonical_json_hash(dict(checkpoint_raw)) != checkpoint_hash:
                raise S27MaterializationBlocked(f"checkpoint.{cell_id}:HASH_MISMATCH")
        input_row = materialized.get(cell_id)
        if input_row is None or input_row.checkpoint_root_ref != row.get("checkpoint_root_ref"):
            raise S27MaterializationBlocked(f"checkpoint.{cell_id}:ROOT_BINDING_INVALID")
        s205 = s205_rows.get(cell_id)
        if s205 is None or s205.get("config_ref") != input_row.config_ref:
            raise S27MaterializationBlocked(f"materialization.{cell_id}:CONFIG_REF_BINDING_INVALID")
        checkpoint_by_cell[cell_id] = {
            "checkpoint_manifest_ref": checkpoint_ref,
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_id": checkpoint_id,
        }
        source_refs.extend((checkpoint_ref, input_row.config_ref, input_row.environment_ref, input_row.registry_ref, input_row.checkpoint_root_ref))
    return logical, manifest_ref, declared_manifest, checkpoint_by_cell, source_refs


def _publish_g23_gates(
    root: Path,
    *,
    gate_output_root: object,
    checked_at: str,
    evaluation_ref: str,
    evaluation_hash: str,
    evaluation: Mapping[str, object],
    s205_ref: str,
    s205_hash: str,
    s205_rows: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    _output_root_ref, output_root = _logical(root, gate_output_root, field="g23_gate_output_root")
    _require_iso(checked_at, field="checked_at")
    if (
        evaluation.get("schema_version") != "stage2-g23-reference-evaluation-v1"
        or evaluation.get("gate_id") != "stage2.G2.3"
        or evaluation.get("status") != "PASS"
        or evaluation.get("scope") != "formal"
        or evaluation.get("formal_eligible") is not True
    ):
        raise S27MaterializationBlocked("g23_evaluation:FORMAL_PASS_REQUIRED")
    cells = evaluation.get("cells")
    if not isinstance(cells, list) or tuple(item.get("cell_id") for item in cells if isinstance(item, Mapping)) != EXPECTED_CELL_IDS:
        raise S27MaterializationBlocked("g23_evaluation:CELL_ORDER_INVALID")
    result: dict[str, dict[str, str]] = {}
    refs: list[str] = []
    for cell_id, raw_cell in zip(EXPECTED_CELL_IDS, cells, strict=True):
        if not isinstance(raw_cell, Mapping) or raw_cell.get("status") != "PASS":
            raise S27MaterializationBlocked(f"g23_evaluation:CELL_PASS_REQUIRED:{cell_id}")
        identities = raw_cell.get("identities")
        metrics = raw_cell.get("metrics")
        if not isinstance(identities, Mapping) or not isinstance(metrics, Mapping):
            raise S27MaterializationBlocked(f"g23_evaluation:CELL_IDENTITIES_REQUIRED:{cell_id}")
        row = s205_rows.get(cell_id)
        if row is None:
            raise S27MaterializationBlocked(f"s205_rebind:CELL_MISSING:{cell_id}")
        row_result_hash = _sha(row.get("result_hash"), field=f"s205.{cell_id}.result_hash")
        row_config_hash = _sha(row.get("config_hash"), field=f"s205.{cell_id}.config_hash")
        if identities.get("result_hash") != row_result_hash or identities.get("config_hash") != row_config_hash:
            raise S27MaterializationBlocked(f"g23_evaluation:S205_IDENTITY_MISMATCH:{cell_id}")
        refs_map = row.get("reference_artifact_refs")
        if not isinstance(refs_map, Mapping) or set(refs_map) != {"reference_result", "reference_convergence_report", "gate_record"}:
            raise S27MaterializationBlocked(f"s205.{cell_id}:REFERENCE_RESULT_REF_REQUIRED")
        status_ref, status = _load_json(
            root,
            row.get("task_result_status_path"),
            field=f"task_result_status.{cell_id}",
        )
        if status.get("task_result_hash") != row_result_hash or status.get("cell_id") != cell_id:
            raise S27MaterializationBlocked(f"task_result_status.{cell_id}:IDENTITY_MISMATCH")
        task_result_ref, task_result = _load_json(
            root,
            status.get("task_result_ref"),
            field=f"task_result.{cell_id}",
        )
        if task_result.get("result_hash") != row_result_hash or canonical_json_hash({key: item for key, item in task_result.items() if key != "result_hash"}) != row_result_hash:
            raise S27MaterializationBlocked(f"task_result.{cell_id}:HASH_MISMATCH")
        reference_ref, reference_payload, reference_hash, loaded_reference = _load_ref_payload(
            root,
            refs_map.get("reference_result"),
            field=f"reference_result.{cell_id}",
        )
        if (
            loaded_reference is None
            or loaded_reference.identity.task_id != "stage2.04_reference_target"
            or loaded_reference.identity.artifact_kind != "reference_result"
            or loaded_reference.identity.config_hash != row_config_hash
            or loaded_reference.identity.formal_eligible is not True
            or loaded_reference.run_intent != "formal"
        ):
            raise S27MaterializationBlocked(f"reference_result.{cell_id}:FORMAL_TASK_COMMIT_REQUIRED")
        reference_payload_hash = _hashed(reference_payload, field=f"reference_result.{cell_id}.payload")
        if reference_payload.get("schema_version") != "reference-result-v1" or reference_payload.get("scope") != "formal" or reference_payload.get("formal_eligible") is not False:
            raise S27MaterializationBlocked(f"reference_result.{cell_id}:FORMAL_SCOPE_REQUIRED")
        config_ref, config = _load_json(root, row.get("config_ref"), field=f"config.{cell_id}")
        if config.get("config_hash") != row_config_hash:
            raise S27MaterializationBlocked(f"config.{cell_id}:HASH_BINDING_INVALID")
        sidecar_hash = _sha(identities.get("corrected_delta_sci_hash"), field=f"g23.{cell_id}.corrected_delta_sci_hash")
        sidecar_ref, sidecar_path = _logical(root, identities.get("corrected_delta_sci_ref"), field=f"g23.{cell_id}.corrected_delta_sci_ref")
        sidecar = load_canonical_json(sidecar_path)
        if not isinstance(sidecar, Mapping) or _hashed(sidecar, field=f"corrected_delta_sci.{cell_id}") != sidecar_hash or sidecar_ref != metrics.get("corrected_delta_sci_ref") or metrics.get("corrected_delta_sci_hash") != sidecar_hash or metrics.get("corrected_delta_sci_batch_sizes") != list(S27_CORRECTED_DELTA_BATCH_SIZES) or metrics.get("delta_sci_source") != S27_CORRECTED_DELTA_SOURCE:
            raise S27MaterializationBlocked(f"g23.{cell_id}:CORRECTED_SIDECAR_BINDING_INVALID")
        cell_hash = canonical_json_hash(dict(raw_cell))
        measured: dict[str, object] = {
            "producer": "stage2-s207-plan-materializer-v1",
            "evaluation": {"ref": evaluation_ref, "artifact_hash": evaluation_hash},
            "evaluation_cell": {"cell_id": cell_id, "artifact_hash": cell_hash},
            "s205_rebind": {"ref": s205_ref, "artifact_hash": s205_hash},
            "result": {"ref": task_result_ref, "artifact_hash": row_result_hash, "status_ref": status_ref},
            "config": {"ref": config_ref, "config_hash": row_config_hash},
            "reference_result": {"ref": reference_ref, "artifact_hash": reference_hash, "payload_artifact_hash": reference_payload_hash},
            "corrected_delta_sci": {"ref": sidecar_ref, "artifact_hash": sidecar_hash},
        }
        evidence_refs = tuple(dict.fromkeys((evaluation_ref, s205_ref, status_ref, task_result_ref, config_ref, reference_ref, sidecar_ref)))
        gate = GateRecord(
            gate_id="stage2.G2.3",
            stage=2,
            status=GateStatus.PASS,
            checked_at=checked_at,
            measured=measured,
            threshold={"evaluation_status": "PASS", "formal_eligible": True, "cell_id": cell_id},
            evidence_refs=evidence_refs,
        )
        target = output_root / cell_id.replace(":", "__") / f"{gate.artifact_hash}.json"
        publish_canonical_immutable(target, gate.to_dict())
        reread = GateRecord.from_mapping(dict(load_canonical_json(target)))
        if reread.artifact_hash != gate.artifact_hash:
            raise S27MaterializationBlocked(f"g23_gate.{cell_id}:ROUND_TRIP_DRIFT")
        gate_ref = target.relative_to(root).as_posix()
        result[cell_id] = {"gate_ref": gate_ref, "gate_hash": gate.artifact_hash}
        refs.extend((gate_ref,))
    return result, refs


def _validate_matrix_g23_bindings(
    matrix: Mapping[str, object],
    evaluation: Mapping[str, object],
    s205_rows: Mapping[str, Mapping[str, object]],
) -> None:
    """Require the S2.6 frozen sidecar table to equal G2.3 PASS identities."""

    matrix_rows = matrix.get("corrected_delta_sci_bindings")
    eval_rows = evaluation.get("cells")
    if not isinstance(matrix_rows, list) or len(matrix_rows) != len(EXPECTED_CELL_IDS) or not isinstance(eval_rows, list) or len(eval_rows) != len(EXPECTED_CELL_IDS):
        raise S27MaterializationBlocked("g24b_g23_binding:SIX_CELL_ROWS_REQUIRED")
    for cell_id, binding, evaluated in zip(EXPECTED_CELL_IDS, matrix_rows, eval_rows, strict=True):
        if not isinstance(binding, Mapping) or not isinstance(evaluated, Mapping):
            raise S27MaterializationBlocked(f"g24b_g23_binding:ROW_INVALID:{cell_id}")
        identities = evaluated.get("identities")
        metrics = evaluated.get("metrics")
        row = s205_rows.get(cell_id)
        if not isinstance(identities, Mapping) or not isinstance(metrics, Mapping) or row is None:
            raise S27MaterializationBlocked(f"g24b_g23_binding:IDENTITY_INVALID:{cell_id}")
        expected = {
            "cell_id": cell_id,
            "config_hash": row.get("config_hash"),
            "result_hash": row.get("result_hash"),
            "corrected_delta_sci_hash": identities.get("corrected_delta_sci_hash"),
            "corrected_delta_sci_ref": identities.get("corrected_delta_sci_ref"),
            "corrected_delta_sci_batch_sizes": list(S27_CORRECTED_DELTA_BATCH_SIZES),
            "delta_sci_source": S27_CORRECTED_DELTA_SOURCE,
        }
        if dict(binding) != expected or metrics.get("corrected_delta_sci_hash") != binding.get("corrected_delta_sci_hash") or metrics.get("corrected_delta_sci_ref") != binding.get("corrected_delta_sci_ref"):
            raise S27MaterializationBlocked(f"g24b_g23_binding:DRIFT:{cell_id}")


def materialize_s27_plan(
    data_root: str | Path,
    *,
    plan_id: str,
    plan_output: object,
    s206_freeze_ref: object,
    s205_rebind_ref: object,
    g23_evaluation_ref: object,
    materialization_index_ref: object,
    execution_evidence_ref: object,
    gpu_inventory_json: str | Path,
    failure_rule_ref: object,
    g23_gate_output_root: object,
    checked_at: str,
    cost_required: bool = True,
) -> S27Plan:
    """Materialize and immutably publish a preflight-ready S2.7 plan."""

    root = Path(data_root).resolve()
    if not isinstance(plan_id, str) or _SAFE_ID.fullmatch(plan_id) is None:
        raise S27MaterializationBlocked("plan_id:INVALID")
    plan_ref, _plan_path = _logical(root, plan_output, field="plan_output")
    freeze_ref, freeze, freeze_hash = _load_freeze(root, s206_freeze_ref)
    s205_ref, s205_raw = _load_json(root, s205_rebind_ref, field="s205_rebind")
    s205_hash = _hashed(s205_raw, field="s205_rebind") if "artifact_hash" in s205_raw else canonical_json_hash(s205_raw)
    if s205_raw.get("schema_version") != "stage2-s205-rebind-plan-v1" or s205_raw.get("status") != "READY" or s205_raw.get("formal_eligible") is not True:
        raise S27MaterializationBlocked("s205_rebind:FORMAL_READY_REQUIRED")
    try:
        validated_s205 = load_s25_rebind_plan(root, s205_ref)
    except Exception as error:
        raise S27MaterializationBlocked(f"s205_rebind:INVALID:{error}") from error
    if validated_s205.get("_rebind_ref") != s205_ref:
        raise S27MaterializationBlocked("s205_rebind:REF_NORMALIZATION_INVALID")
    raw_rows = s205_raw.get("cells")
    if not isinstance(raw_rows, list) or tuple(row.get("cell_id") for row in raw_rows if isinstance(row, Mapping)) != EXPECTED_CELL_IDS:
        raise S27MaterializationBlocked("s205_rebind:SIX_CELL_ORDER_INVALID")
    s205_rows = {str(row["cell_id"]): dict(row) for row in raw_rows if isinstance(row, Mapping)}
    eval_ref, evaluation = _load_json(root, g23_evaluation_ref, field="g23_evaluation")
    eval_hash = _sha(evaluation.get("artifact_hash"), field="g23_evaluation.artifact_hash")
    if eval_hash != canonical_json_hash({key: item for key, item in evaluation.items() if key != "artifact_hash"}):
        raise S27MaterializationBlocked("g23_evaluation:HASH_MISMATCH")
    validated_eval_ref, evaluation = validate_g23_evaluation(root, eval_ref, eval_hash)
    if validated_eval_ref != eval_ref or s205_raw.get("g23_evaluation_ref") != eval_ref or s205_raw.get("g23_evaluation_hash") != eval_hash:
        raise S27MaterializationBlocked("g23_evaluation:S205_REF_HASH_BINDING_INVALID")
    rule_ref, rule_hash, max_failure_fraction = validate_failure_rule(root, failure_rule_ref)
    material_ref, manifest_ref, manifest_hash, checkpoints, material_sources = _load_materialized_checkpoints(root, materialization_index_ref, s205_rows)
    (
        matrix_ref,
        matrix,
        matrix_hash,
        execution,
        execution_ref,
        _inventory_summary,
        inventory_ref,
        inventory_hash,
        _inventory_identity,
        g24b_gate_ref,
    ) = _validate_s206_artifacts(root, freeze, execution_ref=execution_evidence_ref, inventory_json=gpu_inventory_json)
    freeze_source_refs = _load_optional_freeze_sources(root, freeze)
    _validate_matrix_g23_bindings(matrix, evaluation, s205_rows)
    if freeze.get("confirmatory_mapping_ref") is None:
        raise S27MaterializationBlocked("s206_freeze:CONFIRMATORY_MAPPING_REF_REQUIRED")
    mapping_ref, mapping = _load_json(root, freeze.get("confirmatory_mapping_ref"), field="confirmatory_mapping")
    mapping_hash = _hashed(mapping, field="confirmatory_mapping")
    if mapping_hash != freeze.get("confirmatory_mapping_hash"):
        raise S27MaterializationBlocked("s206_freeze:CONFIRMATORY_MAPPING_HASH_INVALID")
    gate_map, gate_sources = _publish_g23_gates(
        root,
        gate_output_root=g23_gate_output_root,
        checked_at=checked_at,
        evaluation_ref=eval_ref,
        evaluation_hash=eval_hash,
        evaluation=evaluation,
        s205_ref=s205_ref,
        s205_hash=s205_hash,
        s205_rows=s205_rows,
    )
    references: dict[str, dict[str, object]] = {}
    source_refs: list[str] = [freeze_ref, matrix_ref, mapping_ref, g24b_gate_ref, s205_ref, eval_ref, material_ref, manifest_ref, execution_ref, inventory_ref, rule_ref, *freeze_source_refs, *material_sources, *gate_sources]
    for cell_id in EXPECTED_CELL_IDS:
        row = s205_rows[cell_id]
        refs_map = row.get("reference_artifact_refs")
        assert isinstance(refs_map, Mapping)
        reference_ref, reference_payload, reference_hash, loaded_reference = _load_ref_payload(root, refs_map.get("reference_result"), field=f"reference_result.{cell_id}")
        assert loaded_reference is not None
        if reference_payload.get("schema_version") != "reference-result-v1" or loaded_reference.identity.artifact_kind != "reference_result":
            raise S27MaterializationBlocked(f"reference.{cell_id}:IDENTITY_INVALID")
        config_ref, _config = _load_json(root, row.get("config_ref"), field=f"config.{cell_id}")
        gate = gate_map[cell_id]
        references[cell_id] = {
            "reference_ref": reference_ref,
            "reference_hash": reference_hash,
            "gate_ref": gate["gate_ref"],
            "gate_hash": gate["gate_hash"],
            "task_id": "stage2.04_reference_target",
            "gate_id": "stage2.G2.3",
            "gate_status": "PASS",
            "scope": "formal",
            "formal_eligible": True,
            "independent": True,
        }
        environment_ref, _ = _logical(root, row.get("environment_ref"), field=f"environment.{cell_id}")
        registry_ref, _ = _logical(root, row.get("registry_ref"), field=f"registry.{cell_id}")
        convergence_ref, _ = _logical(root, refs_map.get("reference_convergence_report"), field=f"reference_convergence.{cell_id}")
        # The S2.5 ``gate_record`` is a candidate envelope (often NOT_RUN),
        # not the direct G2.3 GateRecord consumed by S2.7.  It is validated by
        # ``load_s25_rebind_plan`` but intentionally never promoted to the
        # S2.7 source list or evidence refs.
        source_refs.extend((reference_ref, config_ref, environment_ref, registry_ref, convergence_ref))
    source_refs = list(dict.fromkeys(source_refs))
    try:
        plan = prepare_s27_plan(
            plan_id=plan_id,
            plan_ref=plan_ref,
            matrix_ref=matrix_ref,
            matrix=matrix,
            mapping_ref=mapping_ref,
            mapping=mapping,
            g24b_gate_ref=g24b_gate_ref,
            g24b_gate=load_canonical_json(root / PurePosixPath(g24b_gate_ref)),
            checkpoints=checkpoints,
            references=references,
            source_artifact_refs=source_refs,
            max_failure_fraction=max_failure_fraction,
            cost_required=cost_required,
            failure_rule_ref=rule_ref,
            failure_rule_hash=rule_hash,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise S27MaterializationBlocked(f"prepare_s27_plan:{error}") from error
    publish_canonical_immutable(_plan_path, plan.to_dict())
    reread = S27Plan.from_mapping(dict(load_canonical_json(_plan_path)))
    if reread.artifact_hash != plan.artifact_hash:
        raise S27MaterializationBlocked("s27_plan:ROUND_TRIP_DRIFT")
    # The freeze itself is a source identity; keep the local variable alive in
    # the function to make the exact hash check above visible to reviewers.
    _sha(freeze_hash, field="s206_freeze.artifact_hash")
    return reread


__all__ = [
    "FAILURE_RULE",
    "FAILURE_RULE_SCHEMA",
    "S27MaterializationBlocked",
    "materialize_s27_plan",
    "validate_failure_rule",
]
