"""Strict materialization of the formal S2.5 development inputs.

S2.5 is a runner/invariant gate, not the S2.6 scientific pilot.  Its formal
input therefore covers the complete preregistered development grid exactly
once and records no primary B/M/R selection.  Confirmatory and reference
streams are never touched here.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..runtime.task_artifacts import load_committed_task_artifact
from .preregistration import validate_stage2_preregistration
from .sampling import SamplingPlan
from .stage2_formal import FormalExperimentPlan


S205_SWEEP_SCHEMA = "stage2-s205-development-sweep-plan-v1"
S205_INPUT_INDEX_SCHEMA = "stage2-s205-formal-input-index-v1"
S205_TASK_ID = "stage2.05_paired_estimator_runner"
S201_TASK_ID = "stage2.01_scope_hypotheses_and_preregistration"
S203_TASK_ID = "stage2.03_assets_checkpoints_and_sampling"
EXPECTED_BATCH_SIZES = (32, 64, 128, 256)
EXPECTED_MICROBATCH_COUNTS = (2, 4, 8, 16, 32)


class S205InputBlocked(RuntimeError):
    """Raised when the frozen S2.5 input lineage is incomplete or ambiguous."""


def _logical(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S205InputBlocked(f"{field}:INVALID_LOGICAL_REF")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S205InputBlocked(f"{field}:PATH_ESCAPE")
    return path.as_posix()


def _artifact_payload(
    root: Path,
    ref: str,
    *,
    task_id: str,
    artifact_kind: str,
) -> tuple[dict[str, Any], str]:
    logical = _logical(ref, field=artifact_kind)
    try:
        loaded = load_committed_task_artifact(root, logical, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise S205InputBlocked(f"{artifact_kind}:FORMAL_TASK_ARTIFACT_INVALID:{error}") from error
    identity = loaded.identity
    if (
        identity.commit_ref != logical
        or identity.task_id != task_id
        or identity.artifact_kind != artifact_kind
        or identity.formal_eligible is not True
        or loaded.run_intent != "formal"
    ):
        raise S205InputBlocked(f"{artifact_kind}:TASK_ARTIFACT_IDENTITY_INVALID")
    return dict(loaded.payload), identity.artifact_hash


def _formal_execution(root: Path, ref: str) -> FormalExecutionEvidence:
    logical = _logical(ref, field="formal_execution_ref")
    path = (root / PurePosixPath(logical)).resolve()
    try:
        path.relative_to(root)
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise TypeError("object required")
        evidence = FormalExecutionEvidence.from_mapping(value)
        evidence.require_for_stage(2)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise S205InputBlocked(f"FORMAL_EXECUTION_INVALID:{error}") from error
    return evidence


def build_s205_formal_inputs(
    data_root: str | Path,
    *,
    preregistration_ref: str,
    sampling_plan_ref: str,
    formal_execution_ref: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Extract a direct sampling plan and freeze the exhaustive S2.5 grid."""

    root = Path(data_root).resolve()
    preregistration, prereg_hash = _artifact_payload(
        root,
        preregistration_ref,
        task_id=S201_TASK_ID,
        artifact_kind="preregistration",
    )
    try:
        validate_stage2_preregistration(preregistration)
    except (TypeError, ValueError) as error:
        raise S205InputBlocked(f"PREREGISTRATION_INVALID:{error}") from error
    factors = preregistration.get("factors")
    if not isinstance(factors, Mapping):
        raise S205InputBlocked("PREREGISTRATION_FACTORS_MISSING")
    batches = tuple(factors.get("candidate_batch_sizes", ()))
    microbatches = tuple(factors.get("candidate_microbatch_counts", ()))
    if batches != EXPECTED_BATCH_SIZES or tuple(preregistration.get("candidate_batch_sizes", ())) != batches:
        raise S205InputBlocked("PREREGISTRATION_BATCH_GRID_DRIFT")
    if microbatches != EXPECTED_MICROBATCH_COUNTS or tuple(preregistration.get("candidate_microbatch_counts", ())) != microbatches:
        raise S205InputBlocked("PREREGISTRATION_MICROBATCH_GRID_DRIFT")
    if preregistration.get("formal_primary_values_status") != "FROZEN_RULES_VALUES_PILOT_SELECTED":
        raise S205InputBlocked("PREREGISTRATION_PRIMARY_SELECTION_SEMANTICS_DRIFT")

    sampling_payload, _sampling_artifact_hash = _artifact_payload(
        root,
        sampling_plan_ref,
        task_id=S203_TASK_ID,
        artifact_kind="sampling_plan",
    )
    try:
        sampling = SamplingPlan.from_mapping(sampling_payload)
    except (TypeError, ValueError) as error:
        raise S205InputBlocked(f"SAMPLING_PLAN_PAYLOAD_INVALID:{error}") from error
    sampling_wire = sampling.to_dict()
    if sampling_wire != sampling_payload:
        raise S205InputBlocked("SAMPLING_PLAN_EXTRACTION_NOT_CANONICAL")

    execution = _formal_execution(root, formal_execution_ref)
    source_refs = tuple(sorted({
        _logical(preregistration_ref, field="preregistration_ref"),
        _logical(sampling_plan_ref, field="sampling_plan_ref"),
        _logical(formal_execution_ref, field="formal_execution_ref"),
    }))
    entries: list[dict[str, object]] = []
    start = 0
    for batch_size in batches:
        allowed_m = tuple(item for item in microbatches if batch_size % item == 0)
        plan = FormalExperimentPlan(
            plan_id=f"s205-development-b{batch_size}-v1",
            task_id=S205_TASK_ID,
            wave_id=f"s205-development-b{batch_size}",
            cell_id="all-formal-cells",
            stream="pilot",
            batch_size=batch_size,
            microbatch_counts=allowed_m,
            repetitions=1,
            sampling_plan_hash=sampling.digest,
            execution_evidence_hash=execution.artifact_hash,
            source_artifact_refs=source_refs,
            selection_basis="preregistered_development",
        )
        end = start + batch_size
        entries.append({
            "start_position": start,
            "end_position_exclusive": end,
            "plan": plan.to_dict(),
        })
        start = end

    sweep: dict[str, object] = {
        "schema_version": S205_SWEEP_SCHEMA,
        "task_id": S205_TASK_ID,
        "scope": "formal",
        "state": "FROZEN",
        "formal_eligible": True,
        "selection_basis": "preregistered_development_exhaustive_grid",
        "stream": "pilot",
        "candidate_batch_sizes": list(batches),
        "candidate_microbatch_counts": list(microbatches),
        "repetitions_per_candidate": 1,
        "entries": entries,
        "pilot_draw_count": start,
        "sampling_plan_hash": sampling.digest,
        "execution_evidence_hash": execution.artifact_hash,
        "source_artifact_refs": list(source_refs),
        "preregistration_payload_hash": canonical_json_hash(preregistration),
        "preregistration_artifact_hash": prereg_hash,
        "primary_parameters_selected": False,
        "confirmatory_draws_generated": False,
        "reference_draws_generated": False,
    }
    sweep["artifact_hash"] = canonical_json_hash(sweep)
    validate_s205_development_sweep(sweep, sampling=sampling)
    return sampling_wire, sweep


def validate_s205_development_sweep(
    value: Mapping[str, object],
    *,
    sampling: SamplingPlan | None = None,
) -> dict[str, object]:
    """Strictly validate the exhaustive, non-selecting S2.5 development plan."""

    required = {
        "schema_version", "task_id", "scope", "state", "formal_eligible",
        "selection_basis", "stream", "candidate_batch_sizes",
        "candidate_microbatch_counts", "repetitions_per_candidate", "entries",
        "pilot_draw_count", "sampling_plan_hash", "execution_evidence_hash",
        "source_artifact_refs", "preregistration_payload_hash",
        "preregistration_artifact_hash", "primary_parameters_selected",
        "confirmatory_draws_generated", "reference_draws_generated", "artifact_hash",
    }
    if set(value) != required:
        raise S205InputBlocked("S205_SWEEP_FIELDS_MISMATCH")
    if (
        value.get("schema_version") != S205_SWEEP_SCHEMA
        or value.get("task_id") != S205_TASK_ID
        or value.get("scope") != "formal"
        or value.get("state") != "FROZEN"
        or value.get("formal_eligible") is not True
        or value.get("selection_basis") != "preregistered_development_exhaustive_grid"
        or value.get("stream") != "pilot"
        or value.get("repetitions_per_candidate") != 1
        or value.get("primary_parameters_selected") is not False
        or value.get("confirmatory_draws_generated") is not False
        or value.get("reference_draws_generated") is not False
    ):
        raise S205InputBlocked("S205_SWEEP_SEMANTICS_INVALID")
    if tuple(value.get("candidate_batch_sizes", ())) != EXPECTED_BATCH_SIZES:
        raise S205InputBlocked("S205_SWEEP_BATCH_GRID_INVALID")
    if tuple(value.get("candidate_microbatch_counts", ())) != EXPECTED_MICROBATCH_COUNTS:
        raise S205InputBlocked("S205_SWEEP_MICROBATCH_GRID_INVALID")
    declared = value.get("artifact_hash")
    if not isinstance(declared, str) or declared != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
        raise S205InputBlocked("S205_SWEEP_ARTIFACT_HASH_INVALID")
    if sampling is not None and value.get("sampling_plan_hash") != sampling.digest:
        raise S205InputBlocked("S205_SWEEP_SAMPLING_HASH_MISMATCH")
    refs = value.get("source_artifact_refs")
    if not isinstance(refs, list) or refs != sorted(set(refs)) or len(refs) != 3:
        raise S205InputBlocked("S205_SWEEP_SOURCE_REFS_INVALID")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_BATCH_SIZES):
        raise S205InputBlocked("S205_SWEEP_ENTRY_COUNT_INVALID")
    expected_start = 0
    normalized: list[dict[str, object]] = []
    for batch_size, raw in zip(EXPECTED_BATCH_SIZES, entries):
        if not isinstance(raw, Mapping) or set(raw) != {"start_position", "end_position_exclusive", "plan"}:
            raise S205InputBlocked("S205_SWEEP_ENTRY_FIELDS_INVALID")
        if raw.get("start_position") != expected_start or raw.get("end_position_exclusive") != expected_start + batch_size:
            raise S205InputBlocked("S205_SWEEP_DRAW_INTERVAL_INVALID")
        plan_value = raw.get("plan")
        if not isinstance(plan_value, Mapping):
            raise S205InputBlocked("S205_SWEEP_PLAN_OBJECT_REQUIRED")
        try:
            plan = FormalExperimentPlan.from_mapping(plan_value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise S205InputBlocked(f"S205_SWEEP_PLAN_INVALID:{error}") from error
        if (
            plan.task_id != S205_TASK_ID
            or plan.stream != "pilot"
            or plan.selection_basis != "preregistered_development"
            or plan.batch_size != batch_size
            or plan.microbatch_counts != EXPECTED_MICROBATCH_COUNTS
            or plan.repetitions != 1
            or plan.sampling_plan_hash != value.get("sampling_plan_hash")
            or plan.execution_evidence_hash != value.get("execution_evidence_hash")
            or list(plan.source_artifact_refs) != refs
        ):
            raise S205InputBlocked("S205_SWEEP_PLAN_BINDING_INVALID")
        normalized.append(dict(raw))
        expected_start += batch_size
    if value.get("pilot_draw_count") != expected_start:
        raise S205InputBlocked("S205_SWEEP_PILOT_DRAW_COUNT_INVALID")
    return dict(value)


__all__ = [
    "S205InputBlocked", "S205_INPUT_INDEX_SCHEMA", "S205_SWEEP_SCHEMA",
    "build_s205_formal_inputs", "validate_s205_development_sweep",
]
