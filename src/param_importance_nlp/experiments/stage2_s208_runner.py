"""Production S2.8/G2.6 runner.

This is a control-plane adapter only.  It never creates draws or loads a
model.  The only scientific inputs it opens are the sealed S2.7 raw manifest,
the frozen matrix/contracts, the S2.4 candidate reference bundle and the
independent G2.3 PASS supplied to :func:`load_s208_reference_bundle`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import uuid
from typing import Any, Mapping

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import load_committed_task_artifact
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_runner import (
    S27ExecutionBlocked,
    load_s27_frozen_mappings,
    load_s27_gpu_inventory_envelope,
    load_s27_plan,
    load_s27_raw_records,
    validate_s27_shard_seals,
)
from .stage2_s208_g26 import S28G26Blocked, analyze_s208_g26
from .stage2_s208_production import (
    S208ProductionBlocked,
    _safe_path,
    load_s208_reference_bundle,
    materialize_s208_matrix,
)


S208_RUNNER_SCHEMA = "stage2-s208-g26-production-runner-v1"


def _new_descendant(path: str | Path, parent: Path, *, field: str) -> Path:
    candidate = Path(path).resolve()
    boundary = parent.resolve()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as error:
        raise S208ProductionBlocked(f"{field}:OUTSIDE_REQUIRED_BOUNDARY") from error
    if not relative.parts:
        raise S208ProductionBlocked(f"{field}:UNIQUE_NAMESPACE_REQUIRED")
    if candidate.exists():
        raise S208ProductionBlocked(f"{field}:NAMESPACE_ALREADY_EXISTS")
    return candidate


def _validate_production_paths(
    data_root: str | Path,
    memmap_root: str | Path | None,
    output_root: str | Path,
) -> tuple[Path, Path, Path]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise S208ProductionBlocked("data_root:DIRECTORY_REQUIRED")
    if memmap_root is None:
        raise S208ProductionBlocked("S208_EXPLICIT_MEMMAP_ROOT_REQUIRED")
    scratch = _new_descendant(memmap_root, root / "tmp", field="memmap_root")
    destination = _new_descendant(
        output_root,
        root / "results" / "stage2" / "derived",
        field="output_root",
    )
    return root, scratch, destination


def _resolve_analysis_gate_refs(root: Path, gates: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve DATA_ROOT-relative gate refs before the detached analyzer reads them."""

    resolved: dict[str, Any] = {}
    for gate_id, value in gates.items():
        if isinstance(value, (str, Path)):
            logical = _root_relative_ref(root, value, field=f"{gate_id}.ref")
            resolved[str(gate_id)] = _safe_path(root, logical, f"{gate_id}.ref")
        else:
            resolved[str(gate_id)] = value
    return resolved


def _root_relative_ref(root: Path, value: str | Path, *, field: str) -> str:
    """Normalize one formal CLI ref while retaining the POSIX contract.

    Production artifacts are DATA_ROOT-relative logical references.  The
    runner accepts ``Path`` only as a convenience for callers; it must never
    turn an absolute path, a Windows separator, or a traversal component into
    an accepted formal input.
    """

    if isinstance(value, Path):
        if value.is_absolute():
            raise S208ProductionBlocked(f"{field}:DATA_ROOT_RELATIVE_REF_REQUIRED")
        logical = value.as_posix()
    elif isinstance(value, str):
        logical = value
    else:
        raise S208ProductionBlocked(f"{field}:DATA_ROOT_RELATIVE_REF_REQUIRED")
    if not logical or "\\" in logical:
        raise S208ProductionBlocked(f"{field}:POSIX_REFERENCE_REQUIRED")
    parts = PurePosixPath(logical).parts
    if PurePosixPath(logical).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise S208ProductionBlocked(f"{field}:PATH_ESCAPE")
    _safe_path(root, logical, field)
    return logical


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_s27_source(root: Path, reference: str, *, field: str) -> tuple[dict[str, Any], str]:
    """Load a content-addressed S2.7 source payload without discovery fallback."""

    path = _safe_path(root, reference, field)
    try:
        raw = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S208ProductionBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(raw, Mapping):
        raise S208ProductionBlocked(f"{field}:OBJECT_REQUIRED")
    if raw.get("schema_version") == "task-output-commit-v1":
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S208ProductionBlocked(f"{field}:FORMAL_TASK_COMMIT_INVALID") from error
        payload = loaded.payload
        if not isinstance(payload, Mapping):
            raise S208ProductionBlocked(f"{field}:TASK_PAYLOAD_INVALID")
        return dict(payload), loaded.identity.artifact_hash
    declared = raw.get("artifact_hash")
    if not isinstance(declared, str) or declared != canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"}):
        raise S208ProductionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return dict(raw), declared


def _validate_s27_wave_seal(
    run_root: Path,
    plan: Any,
    records: Mapping[str, Any],
    *,
    run_id: str,
    cell_id: str,
    shards: tuple[Any, ...],
) -> None:
    path = run_root / "wave-seals" / f"{cell_id.replace(':', '__')}.json"
    if not path.is_file() or path.is_symlink():
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_MISSING:{cell_id}")
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_INVALID:{cell_id}") from error
    required_fields = {"schema_version", "run_id", "plan_hash", "cell_id", "gpu_uuid", "gpu_uuids", "expected_unit_count", "completed_unit_count", "failed_unit_count", "units", "sealed", "checked_at", "artifact_hash"}
    if not isinstance(value, Mapping) or set(value) != required_fields or value.get("schema_version") != "stage2-s27-wave-seal-v1" or value.get("artifact_hash") != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) or value.get("run_id") != run_id or value.get("plan_hash") != plan.artifact_hash or value.get("cell_id") != cell_id or value.get("sealed") is not True:
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_INVALID:{cell_id}")
    expected = {unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id}
    expected_gpu_uuids = [shard.gpu_uuid for shard in shards]
    if value.get("gpu_uuids") != expected_gpu_uuids or value.get("gpu_uuid") is not None or value.get("expected_unit_count") != len(expected) or value.get("completed_unit_count") != len(expected) or value.get("failed_unit_count") != sum(record.status == "FAILED" for record in records.values()):
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_DERIVED_FIELDS_INVALID:{cell_id}")
    descriptors = value.get("units")
    if not isinstance(descriptors, list) or len(descriptors) != len(expected):
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_UNIT_COUNT_INVALID:{cell_id}")
    seen: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise S208ProductionBlocked(f"S27_WAVE_SEAL_DESCRIPTOR_INVALID:{cell_id}")
        unit_id = descriptor.get("unit_id")
        if not isinstance(unit_id, str) or unit_id in seen:
            raise S208ProductionBlocked(f"S27_WAVE_SEAL_DESCRIPTOR_INVALID:{cell_id}")
        seen.add(unit_id)
        record = records.get(unit_id)
        if record is None or descriptor.get("status") != record.status or descriptor.get("attempt_id") != record.attempt_id or descriptor.get("unit_artifact_hash") != record.artifact_hash:
            raise S208ProductionBlocked(f"S27_WAVE_SEAL_RAW_BINDING_INVALID:{cell_id}")
    if seen != expected or set(records) != expected:
        raise S208ProductionBlocked(f"S27_WAVE_SEAL_COVERAGE_INVALID:{cell_id}")


def _validate_s27_gpu_inventory_identity(root: Path, plan: Any, status: Mapping[str, Any]) -> dict[str, Any]:
    """Reload the exact five-field S2.6 inventory identity from S2.7 status."""

    identity = status.get("gpu_inventory_identity")
    required = {"source_ref", "artifact_ref", "artifact_hash", "source_sha256", "schema_version"}
    if not isinstance(identity, Mapping) or set(identity) != required:
        raise S208ProductionBlocked("s27_launcher_status:GPU_INVENTORY_IDENTITY_REQUIRED")
    artifact_ref = identity.get("artifact_ref")
    source_ref = identity.get("source_ref")
    if not isinstance(artifact_ref, str) or not isinstance(source_ref, str) or artifact_ref == source_ref:
        raise S208ProductionBlocked("s27_launcher_status:GPU_INVENTORY_PLAN_BINDING_INVALID")
    try:
        artifact_ref = _root_relative_ref(root, artifact_ref, field="s27.gpu_inventory.artifact_ref")
        source_ref = _root_relative_ref(root, source_ref, field="s27.gpu_inventory.source_ref")
    except S208ProductionBlocked as error:
        raise S208ProductionBlocked(f"s27_launcher_status:GPU_INVENTORY_REF_INVALID:{error}") from error
    if artifact_ref not in plan.source_artifact_refs or source_ref not in plan.source_artifact_refs:
        raise S208ProductionBlocked("s27_launcher_status:GPU_INVENTORY_PLAN_BINDING_INVALID")
    try:
        _summary, observed = load_s27_gpu_inventory_envelope(artifact_ref, data_root=root)
    except (S27ExecutionBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        raise S208ProductionBlocked(f"s27_gpu_inventory:{error}") from error
    if dict(observed) != dict(identity):
        raise S208ProductionBlocked("s27_gpu_inventory:STATUS_IDENTITY_MISMATCH")
    return dict(identity)


def _audit_s27_sealed_run(root: Path, raw_manifest: str, raw_manifest_path: Path, raw_root: Path) -> dict[str, Any]:
    """Revalidate the complete S2.7 append-only run before S2.8 statistics."""

    if raw_root != root:
        raise S208ProductionBlocked("raw_root:DATA_ROOT_REQUIRED")
    if raw_manifest_path.name != "raw-results-manifest.json" or raw_manifest_path.parent.name != "sealed":
        raise S208ProductionBlocked("raw_manifest:S27_SEALED_MANIFEST_REF_REQUIRED")
    run_root = raw_manifest_path.parent.parent
    if not run_root.is_relative_to(root):
        raise S208ProductionBlocked("raw_manifest:S27_RUN_ROOT_OUTSIDE_DATA_ROOT")
    try:
        manifest = load_canonical_json(raw_manifest_path)
    except (OSError, TypeError, ValueError) as error:
        raise S208ProductionBlocked("raw_manifest:CANONICAL_READ_FAILED") from error
    if not isinstance(manifest, Mapping):
        raise S208ProductionBlocked("raw_manifest:OBJECT_REQUIRED")
    manifest = dict(manifest)
    if manifest.get("schema_version") != "stage2-s27-sealed-raw-manifest-v1" or manifest.get("task_id") != "stage2.07_main_sweep" or manifest.get("scope") != "formal" or manifest.get("stream") != "confirmatory" or manifest.get("status") != "SEALED" or manifest.get("formal_eligible") is not True:
        raise S208ProductionBlocked("raw_manifest:S27_FORMAL_SEALED_REQUIRED")
    manifest_hash = manifest.get("artifact_hash")
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_json_hash({key: item for key, item in manifest.items() if key != "artifact_hash"}):
        raise S208ProductionBlocked("raw_manifest:ARTIFACT_HASH_MISMATCH")
    run_id = manifest.get("run_id")
    plan_ref = manifest.get("plan_ref")
    if not isinstance(run_id, str) or not run_id or not isinstance(plan_ref, str):
        raise S208ProductionBlocked("raw_manifest:RUN_AND_PLAN_IDENTITY_REQUIRED")
    plan_ref = _root_relative_ref(root, plan_ref, field="raw_manifest.plan_ref")
    try:
        plan = load_s27_plan(root, plan_ref)
        load_s27_frozen_mappings(root, plan)
        reducer = load_s27_raw_records(root, plan, run_root, run_id=run_id)
    except (S27ExecutionBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        raise S208ProductionBlocked(f"s27_append_only_audit:{error}") from error
    if manifest.get("plan_hash") != plan.artifact_hash or manifest.get("matrix_ref") != plan.frozen_inputs.matrix_ref or manifest.get("matrix_hash") != plan.frozen_inputs.matrix_hash or manifest.get("mapping_ref") != plan.frozen_inputs.mapping_ref or manifest.get("mapping_hash") != plan.frozen_inputs.mapping_hash or manifest.get("sampling_plan_hash") != plan.frozen_inputs.sampling_plan_hash or manifest.get("cell_order") != list(EXPECTED_CELL_IDS):
        raise S208ProductionBlocked("raw_manifest:S27_PLAN_LINEAGE_MISMATCH")
    expected_ids = set(plan.expected_unit_ids)
    records = {record.unit_id: record for record in reducer.records}
    if set(records) != expected_ids or manifest.get("expected_unit_count") != len(expected_ids) or manifest.get("completed_unit_count") != len(expected_ids) or manifest.get("failed_unit_count") != 0 or manifest.get("failure_fraction") != 0.0:
        raise S208ProductionBlocked("raw_manifest:S27_COMPLETENESS_MISMATCH")
    descriptors = manifest.get("units")
    if not isinstance(descriptors, list) or len(descriptors) != len(expected_ids):
        raise S208ProductionBlocked("raw_manifest:S27_UNIT_DESCRIPTORS_REQUIRED")
    seen: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise S208ProductionBlocked("raw_manifest:S27_UNIT_DESCRIPTOR_INVALID")
        unit_id = descriptor.get("unit_id")
        if not isinstance(unit_id, str) or unit_id in seen:
            raise S208ProductionBlocked("raw_manifest:S27_UNIT_DESCRIPTOR_INVALID")
        seen.add(unit_id)
        record = records.get(unit_id)
        expected_cell = next((cell for cell in plan.cells if cell.cell_id == record.cell_id), None) if record is not None else None
        expected_binding = expected_cell.corrected_delta_sci_binding if expected_cell is not None else None
        if record is None or not isinstance(expected_binding, Mapping) or descriptor.get("cell_id") != record.cell_id or descriptor.get("repetition_id") != record.repetition_id or descriptor.get("status") != record.status or descriptor.get("attempt_id") != record.attempt_id or descriptor.get("raw_artifact_ref") != record.raw_artifact_ref or descriptor.get("raw_artifact_hash") != record.raw_artifact_hash or descriptor.get("unit_artifact_hash") != record.artifact_hash or descriptor.get("draw_id_hash") != canonical_json_hash(list(record.draw_ids)) or descriptor.get("corrected_delta_sci_binding") != dict(expected_binding):
            raise S208ProductionBlocked(f"raw_manifest:S27_UNIT_LINEAGE_MISMATCH:{unit_id}")
    if seen != expected_ids:
        raise S208ProductionBlocked("raw_manifest:S27_UNIT_COVERAGE_MISMATCH")
    by_cell = {cell_id: {unit_id: record for unit_id, record in records.items() if record.cell_id == cell_id} for cell_id in EXPECTED_CELL_IDS}
    for cell_id in EXPECTED_CELL_IDS:
        try:
            shards = validate_s27_shard_seals(root, plan, run_root, run_id=run_id, cell_id=cell_id)
        except (S27ExecutionBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
            raise S208ProductionBlocked(f"s27_shard_audit:{cell_id}:{error}") from error
        _validate_s27_wave_seal(run_root, plan, by_cell[cell_id], run_id=run_id, cell_id=cell_id, shards=shards)
    marker_path = run_root / "sealed" / "sealed.marker.json"
    gate_path = run_root / "sealed" / "g2.5-gate.json"
    marker = load_canonical_json(marker_path) if marker_path.is_file() and not marker_path.is_symlink() else None
    if not isinstance(marker, Mapping) or marker.get("schema_version") != "stage2-s27-sealed-marker-v1" or marker.get("run_id") != run_id or marker.get("manifest_ref") != "raw-results-manifest.json" or marker.get("manifest_hash") != manifest_hash or marker.get("manifest_file_sha256") != _sha256_file(raw_manifest_path) or marker.get("sealed") is not True or marker.get("artifact_hash") != canonical_json_hash({key: item for key, item in marker.items() if key != "artifact_hash"}):
        raise S208ProductionBlocked("s27_seal_marker:IDENTITY_INVALID")
    gate_payload = load_canonical_json(gate_path) if gate_path.is_file() and not gate_path.is_symlink() else None
    try:
        g25 = GateRecord.from_mapping(dict(gate_payload)) if isinstance(gate_payload, Mapping) else None
    except (TypeError, ValueError, RuntimeError) as error:
        raise S208ProductionBlocked("s27_g25_gate:INVALID") from error
    if g25 is None or g25.gate_id != "stage2.G2.5" or g25.effective_status() is not GateStatus.PASS or g25.measured.get("raw_manifest_hash") != manifest_hash or g25.measured.get("expected_unit_count") != len(expected_ids) or g25.measured.get("failed_unit_count") != 0:
        raise S208ProductionBlocked("s27_g25_gate:PASS_LINEAGE_INVALID")
    status_path = run_root / "launcher-status.json"
    status = load_canonical_json(status_path) if status_path.is_file() and not status_path.is_symlink() else None
    status_fields = {"schema_version", "run_id", "plan_ref", "plan_hash", "gpu_inventory_identity", "status", "wave_order", "waves", "updated_at", "reason", "artifact_hash"}
    if not isinstance(status, Mapping) or set(status) not in {status_fields, status_fields | {"executor_identity"}} or status.get("schema_version") != "stage2-s27-launch-status-v1" or status.get("run_id") != run_id or status.get("plan_ref") != plan.plan_ref or status.get("plan_hash") != plan.artifact_hash or status.get("status") != "SEALED" or status.get("artifact_hash") != canonical_json_hash({key: item for key, item in status.items() if key != "artifact_hash"}):
        raise S208ProductionBlocked("s27_launcher_status:SEALED_IDENTITY_INVALID")
    if status.get("wave_order") != list(EXPECTED_CELL_IDS):
        raise S208ProductionBlocked("s27_launcher_status:WAVE_ORDER_INVALID")
    waves = status.get("waves")
    if not isinstance(waves, Mapping) or set(waves) != set(EXPECTED_CELL_IDS):
        raise S208ProductionBlocked("s27_launcher_status:SIX_WAVE_COMPLETENESS_INVALID")
    for cell_id in EXPECTED_CELL_IDS:
        wave = waves[cell_id]
        expected_units = len(tuple(unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id))
        expected_wave_ref = (run_root / "wave-seals" / f"{cell_id.replace(':', '__')}.json").relative_to(root).as_posix()
        wave_seal = wave.get("wave_seal") if isinstance(wave, Mapping) else None
        if (
            not isinstance(wave, Mapping)
            or wave.get("cell_id") != cell_id
            or wave.get("status") != "COMPLETE"
            or wave.get("shard_count") != len(plan.approved_gpu_uuids)
            or not isinstance(wave_seal, Mapping)
            or wave_seal.get("status") != "SEALED"
            or wave_seal.get("cell_id") != cell_id
            or wave_seal.get("wave_seal_ref") != expected_wave_ref
            or wave_seal.get("expected_units") != expected_units
            or wave_seal.get("failed_units") != 0
            or wave_seal.get("shard_count") != len(plan.approved_gpu_uuids)
        ):
            raise S208ProductionBlocked(f"s27_launcher_status:WAVE_RECEIPT_INVALID:{cell_id}")
    inventory_identity = _validate_s27_gpu_inventory_identity(root, plan, status)
    inventory_ref = str(inventory_identity["artifact_ref"])
    source_ref = str(inventory_identity["source_ref"])
    execution_candidates: list[tuple[str, FormalExecutionEvidence]] = []
    for source_artifact_ref in plan.source_artifact_refs:
        if source_artifact_ref in {inventory_ref, source_ref}:
            continue
        source_path = _safe_path(root, source_artifact_ref, "s27.source")
        if source_path.is_dir():
            continue
        source, source_hash = _load_s27_source(root, source_artifact_ref, field="s27.source")
        if source.get("schema_version") != "formal-execution-evidence-v1":
            continue
        try:
            evidence = FormalExecutionEvidence.from_mapping(source)
            evidence.require_for_stage(2)
        except (TypeError, ValueError, RuntimeError) as error:
            raise S208ProductionBlocked(f"s27.execution_evidence:INVALID:{error}") from error
        if source_hash != evidence.artifact_hash:
            raise S208ProductionBlocked("s27.execution_evidence:HASH_MISMATCH")
        execution_candidates.append((source_artifact_ref, evidence))
    if len(execution_candidates) != 1:
        raise S208ProductionBlocked("s27.execution_evidence:UNIQUE_SOURCE_REQUIRED")
    execution_ref, execution = execution_candidates[0]
    matrix_payload, matrix_hash = _load_s27_source(root, plan.frozen_inputs.matrix_ref, field="s27.matrix")
    if matrix_hash != plan.frozen_inputs.matrix_hash or matrix_payload.get("execution_evidence_hash") != execution.artifact_hash or matrix_payload.get("b_primary") != plan.frozen_inputs.batch_size or matrix_payload.get("m_primary") != plan.frozen_inputs.microbatch_count or matrix_payload.get("r_primary") != plan.frozen_inputs.repetitions or matrix_payload.get("completion_denominator") != plan.frozen_inputs.completion_denominator:
        raise S208ProductionBlocked("s27.matrix:EXECUTION_OR_DIMENSION_LINEAGE_INVALID")
    # The S2.7 launcher status currently has no executor clean-HEAD receipt;
    # reject that omission rather than treating the S2.6 authorization as an
    # executor identity.
    executor_identity = status.get("executor_identity")
    if not isinstance(executor_identity, Mapping) or set(executor_identity) != {"execution_commit", "launcher_source_sha256", "worktree_clean"} or not isinstance(executor_identity.get("execution_commit"), str) or len(executor_identity["execution_commit"]) != 40 or any(char not in "0123456789abcdef" for char in executor_identity["execution_commit"]) or not isinstance(executor_identity.get("launcher_source_sha256"), str) or len(executor_identity["launcher_source_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in executor_identity["launcher_source_sha256"]) or executor_identity.get("worktree_clean") is not True:
        raise S208ProductionBlocked("s27_launcher_status:EXECUTOR_CLEAN_IDENTITY_REQUIRED")
    return {"manifest_hash": manifest_hash, "plan_hash": plan.artifact_hash, "run_root": run_root, "execution_ref": execution_ref, "execution_hash": execution.artifact_hash, "inventory_identity": dict(inventory_identity), "executor_identity": dict(executor_identity)}


def _atomic_publish(destination: Path, files: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if destination.exists():
        raise S28G26Blocked("OUTPUT_ANALYSIS_DIRECTORY_MUST_BE_NEW")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.stage-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for name, value in files.items():
            write_canonical_json(staging / name, value)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return tuple(files)


def _blocked_gate(*, reason: str, source_lineage: Mapping[str, Any], upstream_gates: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "stage2-s208-g26-gate-v1",
        "gate_id": "stage2.G2.6",
        "stage": 2,
        "status": "BLOCKED",
        "quality_gate_dependency": False,
        "measured": {"production_runner": S208_RUNNER_SCHEMA},
        "threshold": {"frozen_thresholds": True},
        "reasons": [reason],
        "upstream_gate_hashes": {str(key): value.get("artifact_hash") for key, value in upstream_gates.items() if isinstance(value, Mapping)},
        "reference_lineage": dict(source_lineage),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _normalize_production_refs(
    root: Path,
    *,
    raw_manifest: str | Path,
    raw_root: str | Path,
    reference_bundle: str | Path,
    g23_gate: str | Path,
    materialization_index: str | Path,
    matrix: str | Path,
    preregistration: str | Path,
    hypothesis_contract: str | Path,
    reference_root: str | Path | None,
) -> dict[str, Any]:
    """Normalize the complete formal CLI boundary before loading any source."""

    raw_manifest_ref = _root_relative_ref(root, raw_manifest, field="raw_manifest")
    reference_root_ref = None if reference_root is None else _root_relative_ref(root, reference_root, field="reference_root")
    return {
        "raw_manifest_ref": raw_manifest_ref,
        "raw_manifest_path": _safe_path(root, raw_manifest_ref, "raw_manifest"),
        "raw_root_path": Path(raw_root).resolve(),
        "reference_bundle_ref": _root_relative_ref(root, reference_bundle, field="reference_bundle"),
        "g23_gate_ref": _root_relative_ref(root, g23_gate, field="g23_gate"),
        "materialization_index_ref": _root_relative_ref(root, materialization_index, field="materialization_index"),
        "matrix_ref": _root_relative_ref(root, matrix, field="matrix"),
        "preregistration_ref": _root_relative_ref(root, preregistration, field="preregistration"),
        "hypothesis_contract_ref": _root_relative_ref(root, hypothesis_contract, field="hypothesis_contract"),
        "reference_root_path": None if reference_root_ref is None else _safe_path(root, reference_root_ref, "reference_root"),
    }


def run_s208_g26_production(
    *,
    data_root: str | Path,
    raw_manifest: str | Path,
    raw_root: str | Path,
    reference_bundle: str | Path,
    g23_gate: str | Path,
    materialization_index: str | Path,
    matrix: str | Path,
    preregistration: str | Path,
    hypothesis_contract: str | Path,
    upstream_gates: Mapping[str, Mapping[str, Any] | str | Path],
    output_root: str | Path,
    reference_root: str | Path | None = None,
    memmap_root: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260825,
) -> dict[str, Any]:
    """Run S2.8 from real sealed refs and publish PASS/BLOCKED atomically."""

    root, scratch, destination = _validate_production_paths(data_root, memmap_root, output_root)
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=False, exist_ok=False)
    loaded: dict[str, Any] | None = None
    try:
        refs = _normalize_production_refs(
            root,
            raw_manifest=raw_manifest,
            raw_root=raw_root,
            reference_bundle=reference_bundle,
            g23_gate=g23_gate,
            materialization_index=materialization_index,
            matrix=matrix,
            preregistration=preregistration,
            hypothesis_contract=hypothesis_contract,
            reference_root=reference_root,
        )
        raw_manifest_ref = refs["raw_manifest_ref"]
        raw_manifest_path = refs["raw_manifest_path"]
        raw_root_path = refs["raw_root_path"]
        reference_bundle_ref = refs["reference_bundle_ref"]
        g23_gate_ref = refs["g23_gate_ref"]
        materialization_index_ref = refs["materialization_index_ref"]
        matrix_ref = refs["matrix_ref"]
        preregistration_ref = refs["preregistration_ref"]
        hypothesis_contract_ref = refs["hypothesis_contract_ref"]
        reference_root_path = refs["reference_root_path"]
        s27_audit = _audit_s27_sealed_run(root, raw_manifest_ref, raw_manifest_path, raw_root_path)
        loaded = load_s208_reference_bundle(
            root,
            reference_bundle_ref,
            g23_gate_ref,
            reference_root=reference_root_path,
            memmap_root=scratch,
        )
        gates = dict(upstream_gates)
        # The G2.3 object consumed by the strict reference loader is the only
        # accepted G2.3 identity; a caller cannot substitute a second mapping.
        gates["stage2.G2.3"] = loaded["g23_gate"]
        loaded_g23_gate_ref = loaded.get("lineage", {}).get("g23_gate_ref") if isinstance(loaded.get("lineage"), Mapping) else None
        if not isinstance(loaded_g23_gate_ref, (str, Path)) or not loaded_g23_gate_ref:
            raise S208ProductionBlocked("stage2.G2.3:REFERENCE_REQUIRED")
        g24a_input = gates.get("stage2.G2.4a")
        g24b_input = gates.get("stage2.G2.4b")
        if g24a_input is None:
            raise S208ProductionBlocked("stage2.G2.4a:INPUT_REQUIRED")
        if g24b_input is None:
            raise S208ProductionBlocked("stage2.G2.4b:INPUT_REQUIRED")
        if not isinstance(g24a_input, (str, Path)):
            raise S208ProductionBlocked("stage2.G2.4a:REFERENCE_REQUIRED")
        if not isinstance(g24b_input, (str, Path)):
            raise S208ProductionBlocked("stage2.G2.4b:REFERENCE_REQUIRED")
        g24a_input = _root_relative_ref(root, g24a_input, field="stage2.G2.4a")
        g24b_input = _root_relative_ref(root, g24b_input, field="stage2.G2.4b")
        analysis_gates = _resolve_analysis_gate_refs(root, gates)
        matrix_materialization = materialize_s208_matrix(
            root,
            materialization_index_ref,
            matrix=matrix_ref,
            preregistration=preregistration_ref,
            g23_gate=g23_gate_ref,
            g24a_gate=g24a_input,
            g24b_gate=g24b_input,
            references=loaded,
        )
        result = analyze_s208_g26(
            raw_manifest=raw_manifest_path,
            raw_root=raw_root_path,
            references=loaded,
            matrix=matrix_ref,
            matrix_materialization=matrix_materialization,
            preregistration=preregistration_ref,
            hypothesis_contract=hypothesis_contract_ref,
            upstream_gates=analysis_gates,
            output_root=None,
            memmap_root=scratch,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        source_lineage = dict(loaded["lineage"])
        source_lineage["matrix_materialization_hash"] = matrix_materialization["artifact_hash"]
        source_lineage["matrix_materialization_ref"] = "matrix_materialization.json"
        source_lineage["artifact_hash"] = canonical_json_hash({key: value for key, value in source_lineage.items() if key != "artifact_hash"})
        derived_artifacts = result["lineage_manifest"].get("derived_artifacts", [])
        if not isinstance(derived_artifacts, list):
            derived_artifacts = []
        result["lineage_manifest"]["derived_artifacts"] = [
            *derived_artifacts,
            "matrix_materialization.json",
        ]
        result["lineage_manifest"]["derived_artifact_hashes"] = {
            "matrix_materialization.json": matrix_materialization["artifact_hash"],
        }
        result["lineage_manifest"]["reference_source_lineage"] = source_lineage
        result["lineage_manifest"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["lineage_manifest"].items() if key != "artifact_hash"})
        result["input_audit"]["reference_source_lineage"] = source_lineage
        result["input_audit"]["matrix_materialization"] = {
            "ref": "matrix_materialization.json",
            "artifact_hash": matrix_materialization["artifact_hash"],
        }
        result["input_audit"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["input_audit"].items() if key != "artifact_hash"})
        result["input_audit"]["s27_provenance"] = {
            "manifest_hash": s27_audit["manifest_hash"],
            "plan_hash": s27_audit["plan_hash"],
            "execution_hash": s27_audit["execution_hash"],
            "inventory_identity": s27_audit["inventory_identity"],
            "executor_identity": s27_audit["executor_identity"],
        }
        result["input_audit"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["input_audit"].items() if key != "artifact_hash"})
        result["lineage_manifest"]["s27_provenance"] = {
            "manifest_hash": s27_audit["manifest_hash"],
            "plan_hash": s27_audit["plan_hash"],
            "execution_hash": s27_audit["execution_hash"],
            "inventory_identity": s27_audit["inventory_identity"],
            "executor_identity": s27_audit["executor_identity"],
        }
        result["lineage_manifest"]["artifact_hash"] = canonical_json_hash({key: value for key, value in result["lineage_manifest"].items() if key != "artifact_hash"})
        files: dict[str, Mapping[str, Any]] = {
            "analysis_input_audit.json": result["input_audit"],
            "statistics_long_table.json": {"schema_version": result["schema_version"], "rows": result["statistics_long_table"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["statistics_long_table"]})},
            "statistics_summary.json": {"schema_version": result["schema_version"], "rows": result["statistics_summary"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["statistics_summary"]})},
            "raw_calibration.json": {"schema_version": result["schema_version"], "rows": result["raw_calibration"], "artifact_hash": canonical_json_hash({"schema_version": result["schema_version"], "rows": result["raw_calibration"]})},
            "confirmatory_family_decisions.json": result["confirmatory_family_decisions"],
            "quality_gates.json": result["quality_gates"],
            "hypothesis_decisions.json": result["hypothesis_decisions"],
            "lineage_manifest.json": result["lineage_manifest"],
            "matrix_materialization.json": matrix_materialization,
            "g2.6-gate.json": result["g2_6_gate"],
        }
        result["output_files"] = list(_atomic_publish(destination, files))
        result["runner"] = {"schema_version": S208_RUNNER_SCHEMA, "published_at": datetime.now(timezone.utc).isoformat(), "reference_lineage_hash": source_lineage["artifact_hash"]}
        result["analysis_hash"] = canonical_json_hash(result)
        return result
    except (S208ProductionBlocked, S28G26Blocked, OSError, TypeError, ValueError) as error:
        lineage = loaded.get("lineage", {}) if loaded is not None else {"reference_bundle_ref": str(reference_bundle), "g23_gate_ref": str(g23_gate)}
        gate = _blocked_gate(reason=f"{type(error).__name__}:{error}", source_lineage=lineage, upstream_gates=upstream_gates)
        _atomic_publish(destination, {"g2.6-gate.json": gate})
        return {"schema_version": S208_RUNNER_SCHEMA, "status": "BLOCKED", "g2_6_gate": gate, "output_files": ["g2.6-gate.json"], "lineage": lineage}


__all__ = ["S208_RUNNER_SCHEMA", "run_s208_g26_production"]
