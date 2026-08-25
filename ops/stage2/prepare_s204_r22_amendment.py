#!/usr/bin/env python3
"""Freeze the smallest append-only S2.4 recovery amendment.

The original r22 round is intentionally immutable.  This command publishes a
versioned amendment only after a machine-readable terminal diagnostic proves
that r22's 65536 node missed the already frozen 0.02 criterion.  The
amendment starts a fresh independent r23 round on a disjoint segment and
pre-registers the next doubling nodes through a conservative terminal;
it does not relax the tolerance, block size, consecutive-count rule, margin
contract, one-shot schema, or downstream evaluator thresholds.

No draw, provider, or final A/B artifact is created here.  ``--data-root``
verification is deliberately required so the amendment cannot be frozen from
an unverified prose claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


SCHEMA_VERSION = "stage2-reference-sizing-amendment-v1"
PARENT_SCHEMA_VERSION = "stage2-reference-sizing-round-v1"
AMENDMENT_ID = "r23-amend-r1"
PARENT_ROUND_ID = "r22"
NEW_ROUND_ID = "r23"
STREAM = "reference_sizing"
OLD_CANDIDATE_SAMPLE_COUNTS = (32768, 65536)
AMENDED_CANDIDATE_SAMPLE_COUNTS = (131072, 262144)
BLOCK_SIZE = 32
NORMALIZED_L1_THRESHOLD = 0.02
REQUIRED_CONSECUTIVE = 1
PRIOR_CONSUMED_END_POSITION = 81920
SEGMENT_START_POSITION = PRIOR_CONSUMED_END_POSITION
SEGMENT_END_POSITION_EXCLUSIVE = SEGMENT_START_POSITION + AMENDED_CANDIDATE_SAMPLE_COUNTS[-1]
TERMINAL_DIAGNOSTIC_VALUE = 0.028604503208326363
PROJECTED_REQUIRED_SAMPLE_COUNT = OLD_CANDIDATE_SAMPLE_COUNTS[-1] * (TERMINAL_DIAGNOSTIC_VALUE / NORMALIZED_L1_THRESHOLD) ** 2
CONSERVATIVE_TERMINAL_PROJECTED_VALUE = TERMINAL_DIAGNOSTIC_VALUE * math.sqrt(
    OLD_CANDIDATE_SAMPLE_COUNTS[-1] / AMENDED_CANDIDATE_SAMPLE_COUNTS[-1]
)
FINAL_REFERENCE_PLAN_SCHEMA = "schemas/shared/stage2-reference-one-shot-plan-v2.json"
CONTINUATION_CONTROL = "precommitted_new_round_disjoint_segment_no_pooling_with_prior_r22"
MULTI_ROUND_CONTROL_SCHEMA = "stage2-multi-round-control-v1"
PROJECTION_MODEL = "normalized_l1_planning_projection=observed_gap*sqrt(base_n/n)"
PROJECTION_SEMANTICS = "descriptive_planning_only_not_an_upper_bound_or_selection_criterion;terminal_must_be_measured"
AMENDMENT_REASON = "r22_terminal_65536_gap_above_threshold"
AMENDMENT_CHANGED_FIELDS = [
    "round_id",
    "sizing.candidate_sample_counts",
    "sizing.segment_start_position",
    "sizing.segment_end_position_exclusive",
    "multi_round_control",
]
AMENDMENT_NON_POSTHOC_BASIS = "bounded_checkpoint_machine_diagnostic_before_r23_draws"
AMENDMENT_AFFECTED_GATES = ["G2.3", "G2.4"]
R23_REFERENCE_STUDY_ID = "stage2-s204-r23-independent-reference-study"
R23_SIZING_RUN_ID = "r23-g3-v5-independent-segment-amend-r1"
R23_FRESH_ATTEMPT_ID = "fresh-r23-amend-r1"
PARENT_SEGMENT_START_POSITION = 16384
PARENT_SEGMENT_END_POSITION_EXCLUSIVE = 81920
UNCHANGED_SCIENTIFIC_CONTRACT = [
    "threshold=0.02",
    "block_size=32",
    "required_consecutive=1",
    "margin_schema_unchanged",
    "evaluator_schema_unchanged",
    "final_A_B_schema_unchanged",
]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9._/-]+$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _canonical_wire_hash(value: Mapping[str, Any]) -> str:
    """Hash the canonical JSON wire bytes used by generator boundaries."""

    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_INVALID")
    return value


def _require_commit(value: object, field: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_INVALID")
    return value


def _require_ref(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(value).parts)
        or _SAFE_REF.fullmatch(value) is None
    ):
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_INVALID")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_INVALID")
    return value


def _validate_contract(value: Mapping[str, Any]) -> None:
    if value.get("append_only") is not True:
        raise ValueError("S204_R22_AMENDMENT_APPEND_ONLY_REQUIRED")
    if value.get("version") != 1:
        raise ValueError("S204_R22_AMENDMENT_VERSION_INVALID")
    if value.get("created_before_new_sizing_draws") is not True:
        raise ValueError("S204_R22_AMENDMENT_ORDER_INVALID")
    if value.get("candidate_expansion") != "append_next_two_doubling_nodes_after_terminal":
        raise ValueError("S204_R22_AMENDMENT_EXPANSION_POLICY_INVALID")
    if value.get("unchanged_scientific_contract") != UNCHANGED_SCIENTIFIC_CONTRACT:
        raise ValueError("S204_R22_AMENDMENT_SCIENTIFIC_CONTRACT_DRIFT")
    if value.get("reason") != AMENDMENT_REASON:
        raise ValueError("S204_R22_AMENDMENT_REASON_INVALID")
    if value.get("changed_fields") != AMENDMENT_CHANGED_FIELDS:
        raise ValueError("S204_R22_AMENDMENT_CHANGED_FIELDS_INVALID")
    if value.get("non_posthoc_basis") != AMENDMENT_NON_POSTHOC_BASIS:
        raise ValueError("S204_R22_AMENDMENT_NON_POSTHOC_BASIS_INVALID")
    if value.get("affected_gates") != AMENDMENT_AFFECTED_GATES:
        raise ValueError("S204_R22_AMENDMENT_AFFECTED_GATES_INVALID")
    review = value.get("review")
    if not isinstance(review, Mapping) or set(review) != {
        "status", "reviewer_role", "reviewed_before_new_draws", "reviewed_at"
    }:
        raise ValueError("S204_R22_AMENDMENT_REVIEW_INVALID")
    if review.get("status") != "ACCEPTED" or review.get("reviewer_role") != "root":
        raise ValueError("S204_R22_AMENDMENT_REVIEW_INVALID")
    if review.get("reviewed_before_new_draws") is not True:
        raise ValueError("S204_R22_AMENDMENT_REVIEW_ORDER_INVALID")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.endswith("Z"):
        raise ValueError("S204_R22_AMENDMENT_REVIEW_TIMESTAMP_INVALID")
    try:
        parsed_reviewed_at = datetime.fromisoformat(reviewed_at[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("S204_R22_AMENDMENT_REVIEW_TIMESTAMP_INVALID") from error
    if parsed_reviewed_at.tzinfo is None or parsed_reviewed_at.utcoffset() != timezone.utc.utcoffset(parsed_reviewed_at):
        raise ValueError("S204_R22_AMENDMENT_REVIEW_INVALID")


def _validate_multi_round_control(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "method", "horizon_round_ids", "allowed_recovery_round_ids",
        "prior_round_ids", "prior_rounds_not_pooled", "prior_final_ab_created",
        "execution_summary_ref", "final_ab_policy", "max_final_ab_rounds",
        "r23_failure_policy", "anytime_valid", "multiplicity_policy",
        "r23_is_only_recovery", "r23_is_only_confirmatory_look", "confirmatory_look_policy",
    }
    if set(value) != required:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_FIELDS_INVALID")
    if value.get("schema_version") != MULTI_ROUND_CONTROL_SCHEMA:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_SCHEMA_INVALID")
    if value.get("method") != "single_confirmatory_look_control_v1":
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_METHOD_INVALID")
    if value.get("horizon_round_ids") != ["r22", "r23"]:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_HORIZON_INVALID")
    if value.get("allowed_recovery_round_ids") != ["r23"]:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_ALLOWED_RETRIES_INVALID")
    if value.get("prior_round_ids") != ["r21", "r22"]:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_PRIORS_INVALID")
    if value.get("prior_rounds_not_pooled") is not True:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_POOLING_FORBIDDEN")
    if value.get("prior_final_ab_created") != {"r21": False, "r22": False}:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_PRIOR_AB_INVALID")
    item = value.get("execution_summary_ref")
    if not isinstance(item, Mapping) or set(item) != {"ref", "sha256"}:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_EVIDENCE_SOURCE_INVALID")
    _require_ref(item.get("ref"), "execution_summary_ref")
    _require_hash(item.get("sha256"), "execution_summary_sha256")
    if value.get("final_ab_policy") != "single_research_one_shot_only":
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_FINAL_AB_POLICY_INVALID")
    if value.get("max_final_ab_rounds") != 1:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_FINAL_AB_LIMIT_INVALID")
    if value.get("r23_failure_policy") != "INCONCLUSIVE_NO_FURTHER_RETRY":
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_FAILURE_POLICY_INVALID")
    if value.get("anytime_valid") is not False:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_ANYTIME_POLICY_INVALID")
    if value.get("multiplicity_policy") != "no_cross_round_pooling;confirmatory_claims_withheld_until_single_final_A_B":
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_MULTIPLICITY_POLICY_INVALID")
    if value.get("r23_is_only_recovery") is not True or value.get("r23_is_only_confirmatory_look") is not True:
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_SINGLE_LOOK_INVALID")
    if value.get("confirmatory_look_policy") != "r23_only_single_final_A_B_look;failure_is_final_INCONCLUSIVE":
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_LOOK_POLICY_INVALID")


def _validate_machine_diagnostics(value: Mapping[str, Any]) -> None:
    if value.get("basis_type") != "bounded_checkpoint_terminal_failure":
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_BASIS_INVALID")
    cell_id = value.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CELL_INVALID")
    if not isinstance(value.get("attempt_id"), str) or not value["attempt_id"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_ATTEMPT_INVALID")
    if value.get("retryable") is not False or value.get("formal_eligible") is not False:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_FORMAL_CONTROL_INVALID")
    _require_commit(value.get("execution_commit"), "diagnostic_execution_commit")
    _require_ref(value.get("config_path"), "diagnostic_config_path")
    for name in ("config_hash", "config_full_hash", "parameter_registry_hash"):
        _require_hash(value.get(name), f"diagnostic_{name}")
    for name in ("checkpoint_revision", "input_checkpoint_id"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"S204_R22_AMENDMENT_DIAGNOSTIC_{name.upper()}_INVALID")
    plan_identity = value.get("plan_identity")
    if not isinstance(plan_identity, Mapping) or set(plan_identity) != {"reference_id", "artifact_hash"}:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PLAN_IDENTITY_INVALID")
    if not isinstance(plan_identity.get("reference_id"), str) or not plan_identity["reference_id"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PLAN_REFERENCE_INVALID")
    _require_hash(plan_identity.get("artifact_hash"), "diagnostic_plan_artifact_hash")
    registry_identity = value.get("registry_identity")
    if not isinstance(registry_identity, Mapping) or set(registry_identity) != {"registry_hash", "parameter_registry_artifact_hash"}:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_REGISTRY_IDENTITY_INVALID")
    _require_hash(registry_identity.get("registry_hash"), "diagnostic_registry_hash")
    _require_hash(registry_identity.get("parameter_registry_artifact_hash"), "diagnostic_registry_artifact_hash")
    draw_identity = value.get("draw_identity")
    if not isinstance(draw_identity, Mapping) or set(draw_identity) != {"parent_sampling_plan_hash", "sizing_draw_hash", "sizing_identity_hash", "seed_namespace", "segment_start_position", "segment_end_position_exclusive"}:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_DRAW_IDENTITY_INVALID")
    _require_hash(draw_identity.get("parent_sampling_plan_hash"), "diagnostic_parent_sampling_plan_hash")
    _require_hash(draw_identity.get("sizing_draw_hash"), "diagnostic_sizing_draw_hash")
    _require_hash(draw_identity.get("sizing_identity_hash"), "diagnostic_sizing_identity_hash")
    if not isinstance(draw_identity.get("seed_namespace"), str) or not draw_identity["seed_namespace"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_DRAW_SEED_INVALID")
    if draw_identity.get("segment_start_position") != PARENT_SEGMENT_START_POSITION or draw_identity.get("segment_end_position_exclusive") != PARENT_SEGMENT_END_POSITION_EXCLUSIVE:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_DRAW_SEGMENT_INVALID")
    sizing_points = value.get("sizing_points")
    if not isinstance(sizing_points, list) or len(sizing_points) != 2:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINTS_COMPLETE_REQUIRED")
    if [item.get("sample_count_per_stream") for item in sizing_points if isinstance(item, Mapping)] != [32768, 65536]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINTS_ORDER_INVALID")
    for point in sizing_points:
        if not isinstance(point, Mapping) or set(point) != {
            "sample_count_per_stream", "block_count_total", "bias_reference_hash", "cross_reference_hash", "ranking_reference_hash",
            "comparison_defined", "comparison_reason", "normalized_l1_from_previous", "convergence_streak",
        }:
            raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINT_FIELDS_INVALID")
        if point.get("convergence_streak") != 0:
            raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINT_STATE_INVALID")
        if isinstance(point.get("block_count_total"), bool) or not isinstance(point.get("block_count_total"), int) or point["block_count_total"] < 2:
            raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINT_BLOCK_COUNT_INVALID")
        for key in ("bias_reference_hash", "cross_reference_hash", "ranking_reference_hash"):
            _require_hash(point.get(key), f"diagnostic_point_{key}")
    refs = value.get("source_refs")
    if not isinstance(refs, Mapping) or set(refs) != {"final_status", "bounded_checkpoint_manifest"}:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_REFS_INVALID")
    for name in ("final_status", "bounded_checkpoint_manifest"):
        item = refs.get(name)
        if not isinstance(item, Mapping) or set(item) != {"ref", "sha256"}:
            raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_SOURCE_INVALID")
        _require_ref(item.get("ref"), f"diagnostic_{name}_ref")
        _require_hash(item.get("sha256"), f"diagnostic_{name}_sha256")
    if value.get("terminal_sample_count_per_stream") != OLD_CANDIDATE_SAMPLE_COUNTS[-1]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_TERMINAL_NODE_INVALID")
    if value.get("terminal_comparison_defined") is not True:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_COMPARISON_UNDEFINED")
    observed = value.get("terminal_normalized_l1_from_previous")
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_GAP_INVALID")
    if float(observed) <= NORMALIZED_L1_THRESHOLD:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_GAP_NOT_ABOVE_THRESHOLD")
    if value.get("terminal_convergence_streak") != 0:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_STREAK_INVALID")
    if value.get("terminal_selected_sample_count_per_stream") is not None:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_SELECTION_INVALID")
    if value.get("threshold") != NORMALIZED_L1_THRESHOLD:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_THRESHOLD_DRIFT")
    if value.get("projection_model") != PROJECTION_MODEL:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PROJECTION_MODEL_INVALID")
    if value.get("projection_semantics") != PROJECTION_SEMANTICS:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PROJECTION_SEMANTICS_INVALID")
    projected_required = value.get("projected_required_sample_count")
    if (
        isinstance(projected_required, bool)
        or not isinstance(projected_required, (int, float))
        or not math.isfinite(float(projected_required))
        or not math.isclose(float(projected_required), PROJECTED_REQUIRED_SAMPLE_COUNT, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_REQUIRED_COUNT_INVALID")
    if value.get("conservative_terminal_sample_count") != AMENDED_CANDIDATE_SAMPLE_COUNTS[-1]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CONSERVATIVE_TERMINAL_INVALID")
    projected_terminal = value.get("projected_terminal_normalized_l1")
    if (
        isinstance(projected_terminal, bool)
        or not isinstance(projected_terminal, (int, float))
        or not math.isfinite(float(projected_terminal))
        or not math.isclose(float(projected_terminal), CONSERVATIVE_TERMINAL_PROJECTED_VALUE, rel_tol=0.0, abs_tol=1e-15)
        or float(projected_terminal) >= NORMALIZED_L1_THRESHOLD
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PROJECTED_TERMINAL_INVALID")


def _validate_sizing(value: Mapping[str, Any]) -> None:
    if value.get("stream") != STREAM:
        raise ValueError("S204_R22_AMENDMENT_STREAM_MISMATCH")
    if value.get("reference_study_id") != R23_REFERENCE_STUDY_ID:
        raise ValueError("S204_R22_AMENDMENT_REFERENCE_STUDY_ID_INVALID")
    if value.get("sizing_run_id") != R23_SIZING_RUN_ID:
        raise ValueError("S204_R22_AMENDMENT_SIZING_RUN_ID_INVALID")
    if value.get("fresh_attempt_id") != R23_FRESH_ATTEMPT_ID:
        raise ValueError("S204_R22_AMENDMENT_FRESH_ATTEMPT_INVALID")
    if value.get("resume_ref") is not None:
        raise ValueError("S204_R22_AMENDMENT_RESUME_REF_FORBIDDEN")
    if value.get("seed_namespaces") != {
        "reference_sizing": "reference_sizing:r23-independent-segment-amend-r1",
        "reference_A": "reference_A:r23-independent-segment-amend-r1",
        "reference_B": "reference_B:r23-independent-segment-amend-r1",
    }:
        raise ValueError("S204_R22_AMENDMENT_SEED_NAMESPACES_INVALID")
    _require_commit(value.get("producer_commit"), "producer_commit")
    if not isinstance(value.get("seed_namespace"), str) or not value["seed_namespace"]:
        raise ValueError("S204_R22_AMENDMENT_SEED_NAMESPACE_REQUIRED")
    if value.get("seed_namespace_mode") != "same_frozen_seed_disjoint_segment":
        raise ValueError("S204_R22_AMENDMENT_SEED_NAMESPACE_MODE_INVALID")
    _require_hash(value.get("parent_sampling_plan_hash"), "parent_sampling_plan_hash")
    if tuple(value.get("candidate_sample_counts", ())) != AMENDED_CANDIDATE_SAMPLE_COUNTS:
        raise ValueError("S204_R22_AMENDMENT_CANDIDATE_NODES_INVALID")
    if value.get("block_size") != BLOCK_SIZE:
        raise ValueError("S204_R22_AMENDMENT_BLOCK_SIZE_INVALID")
    if value.get("normalized_l1_threshold") != NORMALIZED_L1_THRESHOLD:
        raise ValueError("S204_R22_AMENDMENT_THRESHOLD_INVALID")
    if value.get("required_consecutive") != REQUIRED_CONSECUTIVE:
        raise ValueError("S204_R22_AMENDMENT_CONSECUTIVE_INVALID")
    if value.get("complete_all_candidates") is not True:
        raise ValueError("S204_R22_AMENDMENT_ALL_NODES_REQUIRED")
    if value.get("optional_stopping") is not False:
        raise ValueError("S204_R22_AMENDMENT_OPTIONAL_STOPPING_FORBIDDEN")
    if value.get("reuse_prior_sizing_prefix") is not False:
        raise ValueError("S204_R22_AMENDMENT_PREFIX_REUSE_FORBIDDEN")
    if value.get("segment_start_position") != SEGMENT_START_POSITION:
        raise ValueError("S204_R22_AMENDMENT_SEGMENT_START_INVALID")
    if value.get("segment_end_position_exclusive") != SEGMENT_END_POSITION_EXCLUSIVE:
        raise ValueError("S204_R22_AMENDMENT_SEGMENT_END_INVALID")
    if value.get("prior_consumed_end_position") != PRIOR_CONSUMED_END_POSITION:
        raise ValueError("S204_R22_AMENDMENT_PRIOR_BOUNDARY_INVALID")
    final_segments = value.get("final_stream_segments")
    expected_segments = {
        "reference_A": {"start_position": SEGMENT_START_POSITION, "end_position_exclusive": SEGMENT_END_POSITION_EXCLUSIVE},
        "reference_B": {"start_position": SEGMENT_START_POSITION, "end_position_exclusive": SEGMENT_END_POSITION_EXCLUSIVE},
    }
    if final_segments != expected_segments:
        raise ValueError("S204_R22_AMENDMENT_FINAL_SEGMENTS_INVALID")


def validate_r22_amendment(value: Mapping[str, Any]) -> None:
    """Validate the immutable amendment without reading draws or a provider."""

    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("S204_R22_AMENDMENT_SCHEMA_UNSUPPORTED")
    allowed = {
        "schema_version", "round_id", "prior_round_id", "prior_round_status",
        "amendment_id", "amendment_version", "parent_round_id",
        "parent_round_ref", "parent_round_artifact_hash", "parent_preregistration_hash", "parent_execution_status",
        "parent_blocker_code", "sizing", "machine_diagnostics", "run_identity",
        "output_namespace", "new_draws_before_freeze", "final_reference_created",
        "final_reference_plan_schema", "continuation_control", "multi_round_control", "amendment", "status",
        "formal_reference_status", "execution_contract", "artifact_hash",
    }
    if set(value) - allowed:
        raise ValueError("S204_R22_AMENDMENT_UNKNOWN_FIELD")
    if value.get("amendment_id") != AMENDMENT_ID or value.get("amendment_version") != 1:
        raise ValueError("S204_R22_AMENDMENT_ID_OR_VERSION_INVALID")
    if value.get("round_id") != NEW_ROUND_ID or value.get("prior_round_id") != PARENT_ROUND_ID:
        raise ValueError("S204_R22_AMENDMENT_ROUND_LINEAGE_INVALID")
    if value.get("prior_round_status") != "BLOCKED":
        raise ValueError("S204_R22_AMENDMENT_PRIOR_STATUS_INVALID")
    if value.get("parent_round_id") != PARENT_ROUND_ID:
        raise ValueError("S204_R22_AMENDMENT_PARENT_ROUND_INVALID")
    _require_ref(value.get("parent_round_ref"), "parent_round_ref")
    _require_hash(value.get("parent_round_artifact_hash"), "parent_round_artifact_hash")
    _require_hash(value.get("parent_preregistration_hash"), "parent_preregistration_hash")
    if value.get("parent_execution_status") != "BLOCKED":
        raise ValueError("S204_R22_AMENDMENT_PARENT_STATUS_INVALID")
    if value.get("parent_blocker_code") != "contract_unfrozen":
        raise ValueError("S204_R22_AMENDMENT_PARENT_BLOCKER_INVALID")
    sizing = value.get("sizing")
    if not isinstance(sizing, Mapping):
        raise ValueError("S204_R22_AMENDMENT_SIZING_REQUIRED")
    _validate_sizing(sizing)
    diagnostics = value.get("machine_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTICS_REQUIRED")
    _validate_machine_diagnostics(diagnostics)
    run_identity = value.get("run_identity")
    if not isinstance(run_identity, str) or not run_identity or not run_identity.startswith("r23-") or "amend-r1" not in run_identity:
        raise ValueError("S204_R22_AMENDMENT_RUN_ID_INVALID")
    output_namespace = value.get("output_namespace")
    if (
        not isinstance(output_namespace, str)
        or not output_namespace
        or "\\" in output_namespace
        or PurePosixPath(output_namespace).is_absolute()
        or any(part in {"", ".", ".."} for part in PurePosixPath(output_namespace).parts)
        or "r23-" not in output_namespace
        or "amend-r1" not in output_namespace
        or _SAFE_REF.fullmatch(output_namespace) is None
    ):
        raise ValueError("S204_R22_AMENDMENT_OUTPUT_NAMESPACE_INVALID")
    if value.get("new_draws_before_freeze") is not False:
        raise ValueError("S204_R22_AMENDMENT_FREEZE_ORDER_INVALID")
    if value.get("final_reference_created") is not False:
        raise ValueError("S204_R22_AMENDMENT_FINAL_REFERENCE_FORBIDDEN")
    if value.get("final_reference_plan_schema") != FINAL_REFERENCE_PLAN_SCHEMA:
        raise ValueError("S204_R22_AMENDMENT_FINAL_SCHEMA_INVALID")
    if value.get("continuation_control") != CONTINUATION_CONTROL:
        raise ValueError("S204_R22_AMENDMENT_CONTINUATION_CONTROL_INVALID")
    multi_round_control = value.get("multi_round_control")
    if not isinstance(multi_round_control, Mapping):
        raise ValueError("S204_R22_AMENDMENT_MULTI_ROUND_CONTROL_REQUIRED")
    _validate_multi_round_control(multi_round_control)
    if "execution_contract" in value:
        execution_contract = value.get("execution_contract")
        if (
            not isinstance(execution_contract, Mapping)
            or execution_contract.get("multi_round_control") != dict(multi_round_control)
        ):
            raise ValueError("S204_R22_AMENDMENT_EXECUTION_MULTI_ROUND_CONTROL_MISMATCH")
    amendment = value.get("amendment")
    if not isinstance(amendment, Mapping):
        raise ValueError("S204_R22_AMENDMENT_METADATA_REQUIRED")
    _validate_contract(amendment)
    if "status" in value and value.get("status") != "FROZEN_BEFORE_NEW_SIZING_DRAWS":
        raise ValueError("S204_R22_AMENDMENT_STATUS_INVALID")
    if "formal_reference_status" in value and value.get("formal_reference_status") != "NOT_CREATED_UNTIL_SIZING_GATE":
        raise ValueError("S204_R22_AMENDMENT_FORMAL_STATUS_INVALID")
    if "artifact_hash" in value:
        artifact_hash = value.get("artifact_hash")
        if artifact_hash != _canonical_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
            raise ValueError("S204_R22_AMENDMENT_HASH_INVALID")


def _safe_under(root: Path, ref: str, field: str) -> Path:
    path = (root / PurePosixPath(ref)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_OUTSIDE_DATA_ROOT") from error
    if path.is_symlink():
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_LINK_LIKE")
    return path


def _normalise_config_ref(root: Path, value: object, field: str) -> str:
    """Compare a logical amendment ref with a wire absolute DATA_ROOT path.

    Runtime final-status files record the server's absolute config path while
    the amendment deliberately stores only a logical, portable ref.  An
    absolute wire path is accepted only when it resolves inside DATA_ROOT;
    paths outside the root and link-like paths remain fail-closed.
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_INVALID")
    raw = Path(value)
    if raw.is_absolute():
        resolved = _safe_under(root, value, field)
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_OUTSIDE_DATA_ROOT") from error
        return PurePosixPath(relative.as_posix()).as_posix()
    _require_ref(value, field)
    _safe_under(root, value, field)
    return PurePosixPath(value).as_posix()


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_UNREADABLE") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"S204_R22_AMENDMENT_{field.upper()}_OBJECT_REQUIRED")
    return value


def _decode_wire(value: object) -> object:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "dict" and set(value) == {"kind", "items"} and isinstance(value["items"], list):
            result: dict[object, object] = {}
            for pair in value["items"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("S204_R22_AMENDMENT_CHECKPOINT_WIRE_INVALID")
                result[_decode_wire(pair[0])] = _decode_wire(pair[1])
            return result
        if kind in {"list", "tuple"} and set(value) == {"kind", "items"} and isinstance(value["items"], list):
            return [_decode_wire(item) for item in value["items"]]
        if kind == "tensor_ref":
            return "<tensor_ref>"
    return value


def verify_amendment_sources(value: Mapping[str, Any], data_root: str | Path) -> None:
    """Verify parent and diagnostic files before allowing new sizing draws."""

    validate_r22_amendment(value)
    root = Path(data_root).resolve()
    parent_path = _safe_under(root, str(value["parent_round_ref"]), "parent_round_ref")
    parent = _load_json(parent_path, "parent_round")
    if parent.get("schema_version") != PARENT_SCHEMA_VERSION or parent.get("round_id") != PARENT_ROUND_ID:
        raise ValueError("S204_R22_AMENDMENT_PARENT_MANIFEST_INVALID")
    if parent.get("artifact_hash") != _canonical_hash({key: item for key, item in parent.items() if key != "artifact_hash"}):
        raise ValueError("S204_R22_AMENDMENT_PARENT_HASH_INVALID")
    if parent.get("artifact_hash") != value.get("parent_round_artifact_hash"):
        raise ValueError("S204_R22_AMENDMENT_PARENT_HASH_MISMATCH")
    if parent.get("parent_preregistration_hash") != value.get("parent_preregistration_hash"):
        raise ValueError("S204_R22_AMENDMENT_PARENT_PREREGISTRATION_HASH_MISMATCH")
    if parent.get("final_reference_plan_schema") != FINAL_REFERENCE_PLAN_SCHEMA:
        raise ValueError("S204_R22_AMENDMENT_PARENT_FINAL_SCHEMA_INVALID")
    parent_sizing = parent.get("sizing")
    if not isinstance(parent_sizing, Mapping):
        raise ValueError("S204_R22_AMENDMENT_PARENT_SIZING_INVALID")
    if tuple(parent_sizing.get("candidate_sample_counts", ())) != OLD_CANDIDATE_SAMPLE_COUNTS:
        raise ValueError("S204_R22_AMENDMENT_PARENT_CANDIDATES_INVALID")
    if parent_sizing.get("segment_end_position_exclusive") != SEGMENT_START_POSITION:
        raise ValueError("S204_R22_AMENDMENT_PARENT_SEGMENT_BOUNDARY_INVALID")
    if parent_sizing.get("parent_sampling_plan_hash") != value["sizing"]["parent_sampling_plan_hash"]:
        raise ValueError("S204_R22_AMENDMENT_SAMPLING_PLAN_HASH_MISMATCH")

    multi_round_control = value["multi_round_control"]
    assert isinstance(multi_round_control, Mapping)
    source = multi_round_control["execution_summary_ref"]
    assert isinstance(source, Mapping)
    path = _safe_under(root, str(source["ref"]), "execution_summary_ref")
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_HASH_MISMATCH")
    summary = _load_json(path, "execution_summary")
    try:
        from ops.stage2.produce_s204_multi_round_execution_summary import validate_multi_round_execution_summary  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        from produce_s204_multi_round_execution_summary import validate_multi_round_execution_summary  # type: ignore[no-redef]
    try:
        validate_multi_round_execution_summary(summary)
    except (TypeError, ValueError) as error:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_INVALID") from error
    summary_source = summary.get("source_manifest")
    if not isinstance(summary_source, Mapping) or set(summary_source) != {"ref", "sha256"}:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_SOURCE_REQUIRED")
    _require_ref(summary_source.get("ref"), "execution_summary_source_manifest_ref")
    _require_hash(summary_source.get("sha256"), "execution_summary_source_manifest_sha256")
    summary_source_path = _safe_under(root, str(summary_source["ref"]), "execution_summary_source_manifest")
    if not summary_source_path.is_file() or hashlib.sha256(summary_source_path.read_bytes()).hexdigest() != summary_source["sha256"]:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_SOURCE_HASH_MISMATCH")
    try:
        from ops.stage2.produce_s204_multi_round_execution_summary import produce_multi_round_execution_summary  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        from produce_s204_multi_round_execution_summary import produce_multi_round_execution_summary  # type: ignore[no-redef]
    try:
        regenerated_summary = produce_multi_round_execution_summary(root, summary_source_path)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_REPRODUCTION_FAILED") from error
    if regenerated_summary != dict(summary):
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_PROVENANCE_MISMATCH")
    if summary.get("retryable") is not False or summary.get("final_reference_created") is not False or summary.get("one_shot_ab_created") is not False:
        raise ValueError("S204_R22_AMENDMENT_EXECUTION_SUMMARY_CONTROL_INVALID")

    diagnostics = value["machine_diagnostics"]
    assert isinstance(diagnostics, Mapping)
    refs = diagnostics["source_refs"]
    assert isinstance(refs, Mapping)
    final_source = refs["final_status"]
    checkpoint_source = refs["bounded_checkpoint_manifest"]
    assert isinstance(final_source, Mapping) and isinstance(checkpoint_source, Mapping)
    final_path = _safe_under(root, str(final_source["ref"]), "diagnostic_final_status_ref")
    checkpoint_path = _safe_under(root, str(checkpoint_source["ref"]), "diagnostic_checkpoint_ref")
    for path, source, field in (
        (final_path, final_source, "final_status"),
        (checkpoint_path, checkpoint_source, "bounded_checkpoint_manifest"),
    ):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"S204_R22_AMENDMENT_DIAGNOSTIC_{field.upper()}_HASH_MISMATCH")
    final_status = _load_json(final_path, "diagnostic_final_status")
    if final_status.get("status") != "BLOCKED" or final_status.get("cell_id") != diagnostics["cell_id"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_FINAL_STATUS_INVALID")
    if final_status.get("attempt_id") != diagnostics["attempt_id"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_ATTEMPT_MISMATCH")
    if final_status.get("formal_eligible") is not False:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_FORMAL_ELIGIBILITY_MISMATCH")
    if final_status.get("execution_commit") != diagnostics["execution_commit"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_EXECUTION_COMMIT_MISMATCH")
    lineage = final_status.get("execution_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("execution_commit") != diagnostics["execution_commit"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_EXECUTION_LINEAGE_MISSING")
    observed_config_ref = _normalise_config_ref(root, final_status.get("config_path"), "diagnostic_config_path")
    if observed_config_ref != diagnostics["config_path"]:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CONFIG_PATH_MISMATCH")
    for field in ("config_hash", "config_full_hash", "checkpoint_revision", "input_checkpoint_id", "parameter_registry_hash"):
        if final_status.get(field) != diagnostics[field]:
            raise ValueError(f"S204_R22_AMENDMENT_DIAGNOSTIC_{field.upper()}_MISMATCH")
    blockers = final_status.get("blockers")
    if not isinstance(blockers, list) or not any(
        isinstance(item, Mapping) and item.get("code") == "contract_unfrozen" for item in blockers
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_BLOCKER_INVALID")
    if any(not isinstance(item, Mapping) or item.get("retryable") is not False for item in blockers):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_RETRYABLE_BLOCKER")
    checkpoint_manifest = _load_json(checkpoint_path, "diagnostic_checkpoint")
    if (
        checkpoint_manifest.get("schema_version") != "runtime.tensor-bundle.v1"
        or checkpoint_manifest.get("codec") != "raw-tensor-bundle.v1"
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CHECKPOINT_MANIFEST_INVALID")
    state = _decode_wire(checkpoint_manifest.get("state"))
    if not isinstance(state, Mapping):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CHECKPOINT_STATE_INVALID")
    if (
        state.get("checkpoint_schema") != "stage2-reference-bounded-checkpoint-v1"
        or state.get("schema_version") != "stage2-reference-progress-state-v1"
        or state.get("block_size") != BLOCK_SIZE
        or state.get("processed_block_pairs") != OLD_CANDIDATE_SAMPLE_COUNTS[-1] // BLOCK_SIZE
        or state.get("convergence_streak") != 0
        or state.get("selected_sample_count_per_stream") is not None
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CHECKPOINT_CORE_INVALID")
    plan_identity = diagnostics["plan_identity"]
    registry_identity = diagnostics["registry_identity"]
    draw_identity = diagnostics["draw_identity"]
    assert isinstance(plan_identity, Mapping)
    assert isinstance(registry_identity, Mapping)
    assert isinstance(draw_identity, Mapping)
    if (
        draw_identity.get("parent_sampling_plan_hash") != parent_sizing.get("parent_sampling_plan_hash")
        or draw_identity.get("parent_sampling_plan_hash") != value["sizing"].get("parent_sampling_plan_hash")
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_PARENT_SAMPLING_PLAN_MISMATCH")
    if (
        state.get("plan_hash") != plan_identity.get("artifact_hash")
        or state.get("registry_hash") != registry_identity.get("registry_hash")
        or state.get("sizing_draw_hash") != draw_identity.get("sizing_draw_hash")
        or state.get("sizing_identity_hash") != draw_identity.get("sizing_identity_hash")
        or state.get("sizing_stream") is not True
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_CHECKPOINT_IDENTITY_MISMATCH")
    rng_state = state.get("rng_state")
    if not isinstance(rng_state, Mapping) or set(rng_state) != {
        "algorithm_version", "stream", "count", "state_before", "state_after",
        "state_before_sha256", "state_after_sha256",
    }:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_RNG_BOUNDARY_INVALID")
    if (
        not isinstance(rng_state.get("algorithm_version"), str)
        or not rng_state["algorithm_version"]
        or rng_state.get("stream") != "reference_sizing"
        or rng_state.get("count") != PARENT_SEGMENT_END_POSITION_EXCLUSIVE
        or rng_state.get("state_before_sha256") != _canonical_wire_hash(
            {"algorithm_version": rng_state["algorithm_version"], "state": rng_state["state_before"]}
        )
        or rng_state.get("state_after_sha256") != _canonical_wire_hash(
            {"algorithm_version": rng_state["algorithm_version"], "state": rng_state["state_after"]}
        )
        or state.get("rng_state_digest") != _canonical_wire_hash(dict(rng_state))
        or rng_state["count"] - state["processed_block_pairs"] * BLOCK_SIZE
        != draw_identity.get("segment_start_position")
    ):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_RNG_BOUNDARY_MISMATCH")
    points = state.get("points")
    if not isinstance(points, list):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINTS_INVALID")
    diagnostic_points = diagnostics["sizing_points"]
    assert isinstance(diagnostic_points, list)
    source_points = [item for item in points if isinstance(item, Mapping) and item.get("sample_count_per_stream") in {32768, 65536}]
    if len(source_points) != 2:
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_TWO_POINTS_MISSING")
    for expected_point, observed_point in zip(diagnostic_points, source_points, strict=True):
        assert isinstance(expected_point, Mapping)
        for key in expected_point:
            if observed_point.get(key) != expected_point[key]:
                raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_POINT_MISMATCH")
    terminal = next((item for item in points if isinstance(item, Mapping) and item.get("sample_count_per_stream") == OLD_CANDIDATE_SAMPLE_COUNTS[-1]), None)
    if not isinstance(terminal, Mapping):
        raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_TERMINAL_POINT_MISSING")
    for field in (
        "terminal_sample_count_per_stream",
        "terminal_comparison_defined",
        "terminal_normalized_l1_from_previous",
        "terminal_convergence_streak",
    ):
        expected = diagnostics[field]
        observed = {
            "terminal_sample_count_per_stream": terminal.get("sample_count_per_stream"),
            "terminal_comparison_defined": terminal.get("comparison_defined"),
            "terminal_normalized_l1_from_previous": terminal.get("normalized_l1_from_previous"),
            "terminal_convergence_streak": terminal.get("convergence_streak"),
        }[field]
        if field == "terminal_normalized_l1_from_previous":
            if not isinstance(observed, (int, float)) or not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1e-15):
                raise ValueError("S204_R22_AMENDMENT_DIAGNOSTIC_TERMINAL_GAP_MISMATCH")
        elif observed != expected:
            raise ValueError(f"S204_R22_AMENDMENT_DIAGNOSTIC_{field.upper()}_MISMATCH")


def prepare_r22_amendment(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build the frozen amendment without reading draws or running a provider."""

    validate_r22_amendment(value)
    body = dict(value)
    for field in ("status", "formal_reference_status", "execution_contract", "artifact_hash"):
        body.pop(field, None)
    body["status"] = "FROZEN_BEFORE_NEW_SIZING_DRAWS"
    body["formal_reference_status"] = "NOT_CREATED_UNTIL_SIZING_GATE"
    body["execution_contract"] = {
        "parent_round_is_read_only": True,
        "must_complete_candidate_nodes": list(AMENDED_CANDIDATE_SAMPLE_COUNTS),
        "node_order": list(AMENDED_CANDIDATE_SAMPLE_COUNTS),
        "candidate_expansion_basis": "bounded_checkpoint_terminal_failure_with_descriptive_sqrt_projection_only",
        "reference_study_id": R23_REFERENCE_STUDY_ID,
        "sizing_run_id": R23_SIZING_RUN_ID,
        "fresh_attempt_id": R23_FRESH_ATTEMPT_ID,
        "resume_ref": None,
        "seed_namespaces": {
            "reference_sizing": "reference_sizing:r23-independent-segment-amend-r1",
            "reference_A": "reference_A:r23-independent-segment-amend-r1",
            "reference_B": "reference_B:r23-independent-segment-amend-r1",
        },
        "producer_commit": body["sizing"]["producer_commit"],
        "projection_semantics": PROJECTION_SEMANTICS,
        "multi_round_control": dict(body["multi_round_control"]),
        "resume_requires_same_amendment_hash": True,
        "resume_requires_same_seed_namespace": True,
        "resume_requires_same_candidate_nodes": True,
        "resume_requires_same_segment_start": True,
        "resume_requires_same_segment_end": True,
        "draw_positions": {
            "start": SEGMENT_START_POSITION,
            "end_exclusive": SEGMENT_END_POSITION_EXCLUSIVE,
            "prior_sizing_segment_is_read_only": True,
        },
        "final_stream_segments": {
            "reference_A": {"start_position": SEGMENT_START_POSITION, "end_position_exclusive": SEGMENT_END_POSITION_EXCLUSIVE},
            "reference_B": {"start_position": SEGMENT_START_POSITION, "end_position_exclusive": SEGMENT_END_POSITION_EXCLUSIVE},
        },
        "claim_scope": "r23-independent-round-segment-only-no-pooling-with-prior-r22",
    }
    body["artifact_hash"] = _canonical_hash(body)
    validate_r22_amendment(body)
    return body


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("S204_R22_AMENDMENT_OUTPUT_OVERWRITE_FORBIDDEN")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the append-only S2.4 r22 candidate-expansion amendment")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("S204_R22_AMENDMENT_INPUT_OBJECT_REQUIRED")
        result = prepare_r22_amendment(value)
        verify_amendment_sources(result, args.data_root)
        output = args.output.resolve()
        try:
            output.relative_to(args.data_root.resolve())
        except ValueError as error:
            raise ValueError("S204_R22_AMENDMENT_OUTPUT_OUTSIDE_DATA_ROOT") from error
        _write_immutable(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"S2.4 r23 amendment blocked: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMENDED_CANDIDATE_SAMPLE_COUNTS",
    "prepare_r22_amendment",
    "validate_r22_amendment",
    "verify_amendment_sources",
]
