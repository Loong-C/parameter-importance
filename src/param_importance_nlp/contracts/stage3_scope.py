"""Strict Stage 3 entry authority derived from the explicit user direction.

The authority is deliberately narrower than an ``EstimatorDecision``: it may
unlock Stage 3 only, records the already selected U-32/B=32 route, and never
promotes historical Stage 2 artifacts or waives any Stage 3 scientific Gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .jsonio import JSONValue, canonical_json_hash
from .status import GateRecord, GateStatus


STAGE3_SCOPE_DECISION_SCHEMA_VERSION = "stage3-g30-user-scope-decision-v1"
STAGE3_SCOPE_GATE_ID = "stage3.G3-0"
STAGE3_SCOPE_TASK_ID = "stage3.01_prerequisites_and_scope"
STAGE3_SCOPE_ARTIFACT_KIND = "scope_authority"


def validate_stage3_scope_decision(
    value: Mapping[str, object],
) -> Mapping[str, JSONValue]:
    """Validate the immutable user decision and return its Stage 2 selection."""

    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise ValueError("STAGE3_SCOPE_DECISION_HASH_MISMATCH")
    downstream = value.get("downstream_effect")
    non_claims = value.get("non_claims")
    accepted = value.get("accepted_stage_inputs")
    stage0 = accepted.get("stage0") if isinstance(accepted, Mapping) else None
    stage1 = accepted.get("stage1") if isinstance(accepted, Mapping) else None
    stage2 = accepted.get("stage2") if isinstance(accepted, Mapping) else None
    if (
        value.get("schema_version") != STAGE3_SCOPE_DECISION_SCHEMA_VERSION
        or value.get("authority") != "explicit_user_direction"
        or value.get("gate_id") != STAGE3_SCOPE_GATE_ID
        or value.get("scope") != "formal"
        or value.get("status") != "PASS"
        or value.get("formal_eligible") is not True
        or not isinstance(downstream, Mapping)
        or downstream.get("stage3_real_experiments_authorized") is not True
        or downstream.get("g3_0_prerequisite_authorized") is not True
        or downstream.get("stage2_formal_replay_required_for_stage3_start") is not False
        or not isinstance(stage0, Mapping)
        or stage0.get("availability") != "ASSUMED_PASSED_AND_STILL_AVAILABLE"
        or not isinstance(stage1, Mapping)
        or stage1.get("availability") != "ASSUMED_PASSED_AND_STILL_AVAILABLE"
        or not isinstance(stage2, Mapping)
        or stage2.get("default_estimator") != "U-32"
        or stage2.get("batch_size") != 32
        or stage2.get("sensitivity_control") != "Raw"
        or stage2.get("run_id") != "pythia-grid-20260826T145530Z"
        or stage2.get("source_branch") != "exp/stage2-direct-all-20260826"
        or stage2.get("source_commit")
        != "000ce1e79af791ce1eae2e2b62da221a10dd3c9a"
        or not isinstance(non_claims, Sequence)
        or isinstance(non_claims, (str, bytes, bytearray))
        or not any(
            isinstance(item, str) and "does not relabel" in item
            for item in non_claims
        )
        or not any(
            isinstance(item, str) and "does not waive" in item
            for item in non_claims
        )
    ):
        raise ValueError("STAGE3_SCOPE_DECISION_NOT_AUTHORIZED")
    return dict(stage2)  # type: ignore[return-value]


def validate_stage3_scope_authority(
    decision: Mapping[str, object],
    gate: GateRecord,
    *,
    decision_ref: str,
) -> Mapping[str, JSONValue]:
    """Validate the decision/G3-0 pair and their immutable reference binding."""

    stage2 = validate_stage3_scope_decision(decision)
    expected_measured = {
        "authority": "explicit_user_direction",
        "stage0": "assumed_pass_and_available",
        "stage1": "assumed_pass_and_available",
        "stage2_batch_size": 32,
        "stage2_estimator": "U-32",
        "stage2_run_id": "pythia-grid-20260826T145530Z",
        "stage2_sensitivity_control": "Raw",
        "stage3_start_authorized": True,
    }
    expected_threshold = {
        "stage0_available": True,
        "stage1_available": True,
        "stage2_decision_available": True,
        "stage3_start_authorized": True,
    }
    if (
        gate.gate_id != STAGE3_SCOPE_GATE_ID
        or gate.stage != 3
        or gate.status is not GateStatus.PASS
        or decision_ref not in gate.evidence_refs
        or dict(gate.measured) != expected_measured
        or dict(gate.threshold) != expected_threshold
        or gate.reasons
        or gate.conditions
    ):
        raise ValueError("STAGE3_SCOPE_G30_GATE_INVALID")
    return stage2


__all__ = [
    "STAGE3_SCOPE_ARTIFACT_KIND",
    "STAGE3_SCOPE_DECISION_SCHEMA_VERSION",
    "STAGE3_SCOPE_GATE_ID",
    "STAGE3_SCOPE_TASK_ID",
    "validate_stage3_scope_authority",
    "validate_stage3_scope_decision",
]
