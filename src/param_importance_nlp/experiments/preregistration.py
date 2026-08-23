"""Stage 2.1 的机器可读预注册合同。

这个模块只描述固定状态局部梯度空间研究的口径；它不会读取模型、生成 draw，
也不会根据 pilot 或确认性结果修改阈值。S2.1 runner、离线配置和单元测试共用
这里的 builder，避免 Markdown、JSON 和运行时合同逐渐漂移。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any, Mapping

from ..contracts.jsonio import JSONValue, canonical_json_hash


PREREGISTRATION_SCHEMA_VERSION = "stage2-preregistration-v1"
HYPOTHESIS_SCHEMA_VERSION = "stage2-hypothesis-contract-v1"
AMENDMENT_SCHEMA_VERSION = "stage2-preregistration-amendment-v1"
FORMULA_VERSION = "stage2-fixed-state-local-gradient-square-v1"

MODELS = ("pythia-14m", "pythia-31m-deduped")
TRAINING_STAGES = ("initialization", "early", "mid_late")
BATCH_SIZES = (32, 64, 128, 256)
MICROBATCH_COUNTS = (2, 4, 8, 16, 32)
MICROBATCH_SELECTION_ORDER = (32, 16, 8, 4)
STREAM_NAMES = (
    "reference_sizing",
    "reference_A",
    "reference_B",
    "pilot",
    "confirmatory",
)
TOP_FRACTIONS = (0.0001, 0.001, 0.01, 0.05)
PRIMARY_CELLS = tuple(
    {"model": model, "stage": stage}
    for model in MODELS
    for stage in TRAINING_STAGES
)

# These are absolute, native-unit floors fixed by the synthetic Stage 1 fixture before
# any real draw. They are intentionally conservative positive numbers and are part of
# the identity of this registration, not knobs that a pilot may tune.
ABSOLUTE_FLOORS = {
    "tau_model": 1.0e-12,
    "tau_layer": 1.0e-12,
    "tau_module": 1.0e-12,
    "tau_coord": 1.0e-12,
    "tau_nmse": 1.0e-12,
}

RESOURCE_LIMITS = {
    "max_repetitions": 64,
    "max_a100_hours": 64.0,
    "max_peak_memory_gib": 75.0,
    "max_storage_gib": 2000.0,
}


def _canonical_without_hash(value: Mapping[str, JSONValue], field: str) -> str:
    body = {key: item for key, item in value.items() if key != field}
    return canonical_json_hash(body)


def _hash_binding(value: Mapping[str, JSONValue], field: str) -> dict[str, JSONValue]:
    result = deepcopy(dict(value))
    result[field] = _canonical_without_hash(result, field)
    return result


def _endpoint(
    endpoint_id: str,
    *,
    primary: bool,
    level: str,
    statistic: str,
    independent_unit: str,
    aggregation: str,
    interval: str,
    threshold: str,
    multiplicity: str,
    failure: str,
    tie_breaker: str,
) -> dict[str, JSONValue]:
    return {
        "endpoint_id": endpoint_id,
        "primary": primary,
        "level": level,
        "statistic": statistic,
        "independent_unit": independent_unit,
        "aggregation_order": aggregation,
        "interval": interval,
        "threshold": threshold,
        "multiplicity": multiplicity,
        "failure_propagation": failure,
        "tie_breaker": tie_breaker,
    }


def build_stage2_preregistration(
    *,
    seed_plan_hash: str,
    producer_commit: str,
    mathematics_hash: str | None = None,
    stage1_report_hash: str | None = None,
    upstream_binding_hash: str | None = None,
    stage1_handoff: Mapping[str, JSONValue] | None = None,
    scope: str = "local_fixture",
) -> dict[str, JSONValue]:
    """Build the immutable S2.1 registration payload.

    ``seed_plan_hash`` is retained even for a fixture: the same namespace must be used
    when the formal execution is later constructed. Provenance hashes may be null only
    while a caller is preparing a local draft; the published runner always fills them
    from the current repository/report identity when available.
    """

    if not isinstance(seed_plan_hash, str) or len(seed_plan_hash) != 64:
        raise ValueError("seed_plan_hash must be a SHA-256 string")
    if not isinstance(producer_commit, str) or not producer_commit:
        raise ValueError("producer_commit must be non-empty")
    if scope not in {"local_fixture", "formal"}:
        raise ValueError("scope must be local_fixture or formal")

    estimand: dict[str, JSONValue] = {
        "name": "fixed_state_local_gradient_space_mean_square",
        "theoretical_target": "C_star=mu_k^2",
        "formula_version": FORMULA_VERSION,
        "equation": "C_star_k=eta_eval*mu_k^2; mu_k=E_F[g_k(Theta;z)]",
        "eta_eval": 1.0,
        "checkpoint_policy": "immutable_within_reference_and_repetition",
        "state_policy": "eval_read_only_no_optimizer_scheduler_or_data_cursor_mutation",
        "loss_reduction": "fixed_loss_contract_from_stage1",
        "gradient_dtype": "float32",
        "reference_accumulation_dtype": "float64",
        "gradient_clipping": "disabled",
        "weight_decay": "disabled",
        "optimizer_step": "forbidden",
    }
    estimators: dict[str, JSONValue] = {
        "raw": {
            "name": "local_gradient_space_importance_raw",
            "formula": "(mean_B(g_k))^2",
            "sampling": "one_shared_total_batch",
            "signed_output": True,
            "unbiasedness_claim": "none_same_batch_noise_bias",
        },
        "double": {
            "name": "double_sample_gradient_importance",
            "formula": "mean_A(g_k)*mean_B_prime(g_k)",
            "sampling": "two_independent_half_batches_from_one_total_pool",
            "signed_output": True,
            "unbiasedness_claim": "unbiased_for_fixed_state_target",
        },
        "u": {
            "name": "microbatch_u_statistic",
            "formula": "(S1_k^2-S2_k)/(M*(M-1))",
            "sampling": "M_independent_equal_weight_microbatches",
            "signed_output": True,
            "unbiasedness_claim": "unbiased_for_fixed_state_target",
            "weighted_capability": "implemented_for_exogenous_effective_token_weights",
            "main_experiment_weight_rule": "equal_sequence_weights_only",
        },
        "independent_probe_loss": {
            "name": "independent_probe_loss_drop",
            "formula": "L_probe(Theta)-L_probe(Theta_prime)",
            "status": "prohibited_from_stage2_estimator_comparison",
            "reason": "different_total_loss_estimand_and_must_not_be_named_double_or_u",
        },
    }
    endpoints = [
        _endpoint(
            "model_total_signed_bias",
            primary=True,
            level="model",
            statistic="signed_bias",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_parameter_id_then_model_sum",
            interval="two-sided_90_percent_joint_estimator_reference",
            threshold="[-delta_sci(c,e,B_primary),+delta_sci(c,e,B_primary)]",
            multiplicity="intersection_union_across_six_primary_cells",
            failure="one_failed_cell_disqualifies_method; precision_failure_is_inconclusive",
            tie_breaker="none",
        ),
        _endpoint(
            "layer_total_l1_bias",
            primary=True,
            level="layer",
            statistic="l1_bias",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_non_overlapping_layer_registry_then_l1",
            interval="one-sided_95_percent_upper",
            threshold="upper_bound<delta_sci(c,e,B_primary)",
            multiplicity="intersection_union_across_six_primary_cells",
            failure="one_failed_cell_disqualifies_method; precision_failure_is_inconclusive",
            tie_breaker="none",
        ),
        _endpoint(
            "module_total_l1_bias",
            primary=True,
            level="module",
            statistic="l1_bias",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_non_overlapping_module_registry_then_l1",
            interval="one-sided_95_percent_upper",
            threshold="upper_bound<delta_sci(c,e,B_primary)",
            multiplicity="intersection_union_across_six_primary_cells",
            failure="one_failed_cell_disqualifies_method; precision_failure_is_inconclusive",
            tie_breaker="none",
        ),
        _endpoint(
            "parameter_spearman",
            primary=True,
            level="parameter",
            statistic="spearman_with_average_ties",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_parameter_id",
            interval="paired_95_percent",
            threshold="u_minus_double>=-0.02",
            multiplicity="intersection_union_for_primary_cells",
            failure="secondary_if_bias_not_qualified",
            tie_breaker="none",
        ),
        _endpoint(
            "parameter_overlap_at_1_percent",
            primary=True,
            level="parameter",
            statistic="overlap_ratio_K=max(1,ceil(0.01*P))",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_parameter_id",
            interval="paired_95_percent",
            threshold="u_minus_double>=-0.03",
            multiplicity="intersection_union_for_primary_cells",
            failure="secondary_if_bias_not_qualified",
            tie_breaker="none",
        ),
        _endpoint(
            "corrected_parameter_nmse",
            primary=True,
            level="parameter",
            statistic="NMSE_observed_minus_V_ref",
            independent_unit="repetition/checkpoint/model",
            aggregation="canonical_parameter_id",
            interval="paired_95_percent",
            threshold="u/double<=1.10_only_when_double_corrected_nmse>tau_nmse",
            multiplicity="intersection_union_for_primary_cells",
            failure="inconclusive_if_double_floor_not_cleared",
            tie_breaker="none",
        ),
        _endpoint(
            "online_training_incremental_cost_ratio",
            primary=True,
            level="model_stage",
            statistic="wall_time_and_peak_memory_ratio",
            independent_unit="checkpoint/model",
            aggregation="method_independent_anchor",
            interval="pointwise_with_resource_budget",
            threshold="<=1.25_for_wall_time_and_peak_memory",
            multiplicity="all_six_primary_cells",
            failure="method_not_selected_if_resource_budget_exceeded",
            tie_breaker="double_if_only_double_passes_bias_gate",
        ),
        _endpoint(
            "raw_bias_calibration",
            primary=False,
            level="layer",
            statistic="bias_on_1_over_B_regression",
            independent_unit="repetition/checkpoint/model",
            aggregation="layer_then_hierarchical_slope",
            interval="95_percent_hierarchical",
            threshold="slope in [0.8,1.2], intercept within precision budget",
            multiplicity="all_preregistered_layers_with_power",
            failure="reverse_powerful_cell_is_not_supported",
            tie_breaker="none",
        ),
    ]
    provenance: dict[str, JSONValue] = {
        "producer_commit": producer_commit,
        "seed_plan_hash": seed_plan_hash,
        "mathematics_path": "docs/mathematics.md",
        "mathematics_hash": mathematics_hash,
        # Retained as a non-authoritative compatibility field for older local
        # drafts.  Formal G2.0 requires ``stage1_handoff`` below and never
        # treats this tracked report as Stage1 evidence.
        "stage1_report_path": "reports/stage1/cpu-evidence-20260814-s12-r2/stage1.11_reporting_and_exit_gate/stage_report.json",
        "stage1_report_hash": stage1_report_hash,
        "upstream_binding_hash": upstream_binding_hash,
        "sample_generation_allowed_after": "this_payload_and_hash_committed",
        "amendment_policy": "append_only; never_overwrite_original_registration",
    }
    if stage1_handoff is not None:
        if not isinstance(stage1_handoff, Mapping):
            raise ValueError("stage1_handoff must be a mapping")
        provenance["stage1_handoff"] = dict(stage1_handoff)

    body: dict[str, JSONValue] = {
        "schema_version": PREREGISTRATION_SCHEMA_VERSION,
        "registration_id": "stage2-s2.1-fixed-state-estimator-v1",
        "scope": scope,
        "state": "FROZEN",
        "gate_id": "stage2.G2.0",
        "stage": 2,
        "formula_version": FORMULA_VERSION,
        # Compatibility aliases consumed by the existing Stage 2 handoff adapter;
        # the canonical definitions live under ``factors`` and ``sampling`` below.
        "candidate_batch_sizes": list(BATCH_SIZES),
        "candidate_microbatch_counts": list(MICROBATCH_COUNTS),
        "microbatch_selection_order": list(MICROBATCH_SELECTION_ORDER),
        "sampling_stream_names": list(STREAM_NAMES),
        "formal_primary_values_status": "FROZEN_RULES_VALUES_PILOT_SELECTED",
        "formal_eligible": False,
        "estimand": estimand,
        "estimators": estimators,
        "invariants": {
            "u_m2_equals_double": "M=2_for_same_two_half_draw_mapping_coordinatewise_exact_within_float_tolerance",
            "u_signed_values_preserved": True,
            "u_may_be_negative": True,
            "nested_M_reuses_maximum_gradient_pool": True,
            "same_total_draw_budget_across_methods": True,
        },
        "estimand_exclusions": [
            "full_parameter_path_integral",
            "actual_optimizer_update_contribution",
            "independent_probe_loss_alignment",
            "training_cumulative_positive_negative_or_absolute_importance",
            "pruning_validation",
            "160M_or_410M_formal_training_conclusions",
        ],
        "signed_analysis_policy": {
            "use_signed_raw_estimates": True,
            "clamp_min_zero": False,
            "absolute_value_before_bias": False,
            "positive_negative_split_before_bias": False,
            "negative_u_values_preserved": True,
        },
        "factors": {
            "models": list(MODELS),
            "training_stages": list(TRAINING_STAGES),
            "candidate_batch_sizes": list(BATCH_SIZES),
            "candidate_microbatch_counts": list(MICROBATCH_COUNTS),
            "top_fractions": list(TOP_FRACTIONS),
            "primary_cells": list(PRIMARY_CELLS),
            "stage1_step0_and_synthetic_fixture": "calibration_only",
            "confirmatory_cells": "14m_three_stages_and_all_31m_primary_cells",
        },
        "sampling": {
            "universe": "frozen_non_overlapping_fixed_length_packed_sequences",
            "distribution": "empirical_F_with_replacement",
            "stream_names": list(STREAM_NAMES),
            "draw_id_reuse": False,
            "sample_id_collision_policy": "retain_and_report; do_not_deduplicate",
            "reference_sizing_before_final_reference": True,
            "reference_A_B_one_shot": True,
            "confirmatory_manifest_before_gradient": True,
            "same_total_draw_budget": True,
            "double_split": "pre_registered_disjoint_half_pool",
            "nested_M": "max_M_microbatch_gradients_then_fixed_nested_partitions",
        },
        "statistics": {
            "independent_unit": "repetition_checkpoint_model",
            "metrics": [
                "signed_bias",
                "absolute_bias",
                "variance",
                "mse",
                "mae",
                "pearson",
                "spearman",
                "overlap_at_K",
                "jaccard_at_K",
            ],
            "ranking_tie_policy": "average_rank",
            "ranking_views": ["single_repetition_vs_reference", "repetition_mean_vs_reference"],
            "top_k_rule": "K=max(1,ceil(q*P))",
            "scope_levels": ["parameter", "tensor", "layer", "module"],
            "scope_primary_quantity": "total_sum",
            "scope_secondary_quantity": "mean_per_parameter",
            "near_zero_policy": "reference_signal_or_gradient_snr_bins; no_relative_bias_only",
            "bias_definition": "E_r[estimate]-C_star_reference",
            "absolute_bias_definition": "abs(E_r[estimate]-C_star_reference)",
            "mae_definition": "E_r[abs(estimate-C_star_reference)]",
            "bootstrap_unit": "repetition_with_checkpoint_and_model_strata",
        },
        "equivalence_and_precision": {
            "scientific_margin_formula": "max(0.10*Delta_c_e(B),0.01*S_c_e)",
            "signal_definition": "S_model=max(abs(sum(a_k)),tau_model); S_q=max(sum_g(abs(sum_g(a_k))),tau_q)",
            "noise_definition": "Delta_model=abs(sum_k(d_k(B))); Delta_q=sum_g(abs(sum_g(d_k(B))))",
            "sizing_vectors": {"a": "mu_sizing^2", "d": "sigma_squared_over_B"},
            "group_registry": "canonical_non_overlapping_layer_and_module_registry",
            "absolute_floors": dict(ABSOLUTE_FLOORS),
            "reference_half_width_gate": "h_ref<=min_over_all_candidate_B(delta_sci)/4",
            "numerical_error_gate": "epsilon_num<=min_over_all_candidate_B(delta_sci)/10",
            "precision_failure_state": "inconclusive_blocked",
            "margin_excludes_reference_and_numerical_error": True,
            "model_total_equivalence": "joint_90_percent_interval_inside_signed_margin",
            "layer_module_equivalence": "one_sided_95_percent_upper_below_margin",
        },
        "selection_algorithm": {
            "m_scan_order": list(MICROBATCH_SELECTION_ORDER),
            "b_scan_order": list(BATCH_SIZES),
            "m_candidate_constraints": [
                "M divides B",
                "every_microbatch_contains_one_complete_sequence",
                "six_anchor_no_oom",
                "finite_estimator_outputs",
                "aggregation_overhead<=0.25*shared_gradient_time",
            ],
            "primary_pair_constraints": [
                "six_anchor_runability",
                "worst_case_R_required<=R_max",
                "resource_limits_not_exceeded",
            ],
            "selection_inputs_allowed": ["runability", "finite_values", "variance_for_R", "resource_usage"],
            "selection_inputs_forbidden": ["bias_direction", "method_mean", "nmse", "ranking", "significance", "checkpoint_preference"],
            "r_max": RESOURCE_LIMITS["max_repetitions"],
            "no_candidate_action": "blocked_new_preregistered_round_required",
            "tie_breaker": "first_in_declared_M_then_B_order",
        },
        "resource_limits": dict(RESOURCE_LIMITS),
        "endpoints": endpoints,
        "primary": {
            "cells": list(PRIMARY_CELLS),
            "method_bias_qualification": "intersection_union_all_six_cells_and_three_bias_endpoints",
            "ranking_endpoints": ["parameter_spearman", "parameter_overlap_at_1_percent"],
            "mse_endpoint": "corrected_parameter_nmse",
            "cost_endpoint": "online_training_incremental_cost_ratio",
        },
        "thresholds": {
            "raw_slope_interval": [0.8, 1.2],
            "u_double_nmse_noninferiority_ratio": 1.10,
            "u_double_spearman_difference_lower": -0.02,
            "u_double_overlap_1pct_difference_lower": -0.03,
            "u_double_cost_ratio_upper": 1.25,
            "precision_margin_reference_divisor": 4.0,
            "precision_margin_numeric_divisor": 10.0,
            "confidence_levels": {"equivalence_model_total": 0.90, "layer_module_upper": 0.95, "noninferiority": 0.95},
        },
        "quality_gates": [
            {"gate": "fixed_state", "failure_status": "blocked", "required": True},
            {"gate": "sample_independence", "failure_status": "blocked", "required": True},
            {"gate": "reference_convergence", "failure_status": "inconclusive", "required": True},
            {"gate": "result_completeness", "failure_status": "blocked", "required": True},
            {"gate": "finite_numeric_values", "failure_status": "blocked", "required": True},
            {"gate": "fair_total_draw_budget", "failure_status": "blocked", "required": True},
            {"gate": "replayability", "failure_status": "blocked", "required": True},
        ],
        "hypothesis_decision_states": ["supported", "not_supported", "inconclusive"],
        "decision_tree": {
            "u_and_double_bias_qualified": "select_u_if_u_nmse_ranking_and_online_cost_noninferior_else_select_double",
            "only_double_bias_qualified": "select_double_if_absolute_resource_budget_passes",
            "u_needs_adjustment": "blocked; preserve_results_and_open_new_preregistration",
            "neither_or_reference_failure": "blocked; do_not_fallback_to_raw",
            "quality_failure": "do_not_interpret_hypothesis_results",
            "hypothesis_failure": "retain_complete_results_and_follow_method_branch",
        },
        "provenance": provenance,
    }
    body["preregistration_hash"] = _canonical_without_hash(body, "preregistration_hash")
    return body


def build_stage2_hypothesis_contract(
    preregistration: Mapping[str, JSONValue],
    *,
    upstream_binding_hash: str,
) -> dict[str, JSONValue]:
    """Build the H1--H6 decision contract from one frozen registration."""

    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("hypothesis contract requires stage2-preregistration-v1")
    if preregistration.get("state") != "FROZEN":
        raise ValueError("hypothesis contract requires frozen preregistration")
    hypotheses: list[dict[str, JSONValue]] = [
        {
            "id": "H1",
            "claim": "raw mean signed bias is positive and calibrated to sigma_squared_over_B",
            "primary_evidence": ["raw_bias_calibration", "model_total_signed_bias"],
            "decision_rule": "direction_and_scale_agree_with_sigma_squared_over_B; disclose_low_power",
        },
        {
            "id": "H2",
            "claim": "raw bias decreases with 1/B",
            "primary_evidence": ["raw_bias_calibration"],
            "decision_rule": "hierarchical_slope_in_[0.8,1.2]_and_no_powerful_reverse_cell",
        },
        {
            "id": "H3",
            "claim": "double and microbatch U are equivalent to fixed-state mu_squared",
            "primary_evidence": ["model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias"],
            "decision_rule": "intersection_union_all_six_cells_with_registered_intervals",
        },
        {
            "id": "H4",
            "claim": "U mean is invariant to M at fixed total B; M changes variance/cost",
            "primary_evidence": ["model_total_signed_bias", "corrected_parameter_nmse"],
            "decision_rule": "no_systematic_M_drift; signed_values_and_negative_fraction_reported",
        },
        {
            "id": "H5",
            "claim": "at M>2 U has no higher variance or MSE than equal-budget double; M=2 is equal",
            "primary_evidence": ["corrected_parameter_nmse", "online_training_incremental_cost_ratio"],
            "decision_rule": "apply_only_after_independence_weight_and_reference_diagnostics",
        },
        {
            "id": "H6",
            "claim": "repeat-mean U/double rankings are closer to reference",
            "primary_evidence": ["parameter_spearman", "parameter_overlap_at_1_percent"],
            "decision_rule": "single_draw_ranking_is_secondary_when_small_B_variance_is_high",
        },
    ]
    body: dict[str, JSONValue] = {
        "schema_version": HYPOTHESIS_SCHEMA_VERSION,
        "contract_id": "stage2-hypotheses-h1-h6-v1",
        "preregistration_hash": preregistration["preregistration_hash"],
        "upstream_binding_hash": upstream_binding_hash,
        "statistical_unit": "independent_repetition",
        "null_hypotheses": [
            "mean_signed_bias_interval_contains_zero",
            "candidate_corrected_nmse_not_better_than_double",
        ],
        "hypotheses": hypotheses,
        "decision_states": ["supported", "not_supported", "inconclusive"],
        "initial_decisions": {f"H{i}": "inconclusive" for i in range(1, 7)},
        "quality_gate_dependency": "quality_gates_must_pass_before_scientific_interpretation",
        "multiplicity_policy": "preregistered_family_no_posthoc_promotion; intersection_union_for_primary_cells; FDR_or_simultaneous_for_exploratory_layers",
        "method_decision_policy": "do_not_select_raw_as_fallback",
    }
    body["hypothesis_contract_hash"] = _canonical_without_hash(body, "hypothesis_contract_hash")
    return body


def build_stage2_amendment_template(*, preregistration_hash: str) -> dict[str, JSONValue]:
    """Return an append-only amendment wire object; it cannot silently replace S2.1."""

    body: dict[str, JSONValue] = {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "amendment_id": "REPLACE_WITH_APPEND_ONLY_ID",
        "parent_preregistration_hash": preregistration_hash,
        "state": "DRAFT",
        "created_before_confirmatory_draws": True,
        "reason": "REQUIRED_NON_EMPTY_JUSTIFICATION",
        "changed_fields": [],
        "unchanged_fields": ["estimand", "estimators", "primary_endpoints", "decision_tree"],
        "non_posthoc_basis": "REQUIRED_IMPLEMENTATION_OR_ASSET_OR_RESOURCE_FACT",
        "affected_gates": [],
        "new_hash": None,
        "review": {"reviewer": None, "reviewed_at": None, "decision": "PENDING"},
        "append_only": True,
    }
    body["amendment_hash"] = _canonical_without_hash(body, "amendment_hash")
    return body


def validate_stage2_preregistration(value: Mapping[str, Any]) -> None:
    """Fail closed on the semantic invariants hidden JSON Schema cannot express."""

    if value.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("STAGE2_PREREG_SCHEMA_UNSUPPORTED")
    if value.get("state") != "FROZEN" or value.get("gate_id") != "stage2.G2.0":
        raise ValueError("STAGE2_PREREG_NOT_FROZEN_OR_WRONG_GATE")
    if value.get("formula_version") != FORMULA_VERSION:
        raise ValueError("STAGE2_PREREG_FORMULA_VERSION_MISMATCH")
    estimand = value.get("estimand")
    if not isinstance(estimand, Mapping) or estimand.get("eta_eval") != 1.0:
        raise ValueError("STAGE2_PREREG_ETA_EVAL_MUST_BE_ONE")
    if estimand.get("optimizer_step") != "forbidden" or estimand.get("gradient_clipping") != "disabled":
        raise ValueError("STAGE2_PREREG_MUTATION_POLICY_NOT_FIXED")
    signed = value.get("signed_analysis_policy")
    if not isinstance(signed, Mapping) or any(
        signed.get(field) is not expected
        for field, expected in (
            ("use_signed_raw_estimates", True),
            ("clamp_min_zero", False),
            ("absolute_value_before_bias", False),
            ("positive_negative_split_before_bias", False),
            ("negative_u_values_preserved", True),
        )
    ):
        raise ValueError("STAGE2_PREREG_SIGNED_POLICY_INVALID")
    factors = value.get("factors")
    if not isinstance(factors, Mapping):
        raise ValueError("STAGE2_PREREG_FACTORS_MISSING")
    if tuple(factors.get("candidate_batch_sizes", ())) != BATCH_SIZES:
        raise ValueError("STAGE2_PREREG_BATCH_GRID_MISMATCH")
    if tuple(factors.get("candidate_microbatch_counts", ())) != MICROBATCH_COUNTS:
        raise ValueError("STAGE2_PREREG_MICROBATCH_GRID_MISMATCH")
    if tuple(value.get("sampling", {}).get("stream_names", ())) != STREAM_NAMES:  # type: ignore[union-attr]
        raise ValueError("STAGE2_PREREG_STREAMS_MISMATCH")
    thresholds = value.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("STAGE2_PREREG_THRESHOLDS_MISSING")
    for key in (
        "u_double_nmse_noninferiority_ratio",
        "u_double_spearman_difference_lower",
        "u_double_overlap_1pct_difference_lower",
        "u_double_cost_ratio_upper",
    ):
        number = thresholds.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(float(number)):
            raise ValueError(f"STAGE2_PREREG_THRESHOLD_INVALID:{key}")
    primary = value.get("primary")
    if not isinstance(primary, Mapping) or tuple(primary.get("cells", ())) != PRIMARY_CELLS:
        raise ValueError("STAGE2_PREREG_PRIMARY_CELLS_MISMATCH")
    supplied_hash = value.get("preregistration_hash")
    if supplied_hash != _canonical_without_hash(value, "preregistration_hash"):
        raise ValueError("STAGE2_PREREG_HASH_MISMATCH")


def validate_stage2_hypothesis_contract(
    value: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any] | None = None,
) -> None:
    """Fail-closed validation for the H1--H6 contract.

    The original producer only needed to build this object, while formal G2.0
    qualification is a consumer boundary.  Rebuilding the contract from its
    declared parent is intentional: a caller cannot change a claim, decision
    rule, or multiplicity policy and merely recompute ``hypothesis_contract_hash``.
    """

    if not isinstance(value, Mapping):
        raise ValueError("STAGE2_HYPOTHESIS_CONTRACT_NOT_OBJECT")
    if value.get("schema_version") != HYPOTHESIS_SCHEMA_VERSION:
        raise ValueError("STAGE2_HYPOTHESIS_SCHEMA_UNSUPPORTED")
    if value.get("contract_id") != "stage2-hypotheses-h1-h6-v1":
        raise ValueError("STAGE2_HYPOTHESIS_CONTRACT_ID_MISMATCH")
    parent_hash = value.get("preregistration_hash")
    if not isinstance(parent_hash, str) or not _is_sha256(parent_hash):
        raise ValueError("STAGE2_HYPOTHESIS_PREREGISTRATION_HASH_INVALID")
    upstream = value.get("upstream_binding_hash")
    if not isinstance(upstream, str) or not _is_sha256(upstream):
        raise ValueError("STAGE2_HYPOTHESIS_UPSTREAM_BINDING_HASH_INVALID")
    if preregistration is None:
        raise ValueError("STAGE2_HYPOTHESIS_PREREGISTRATION_REQUIRED")
    validate_stage2_preregistration(preregistration)
    if parent_hash != preregistration.get("preregistration_hash"):
        raise ValueError("STAGE2_HYPOTHESIS_PARENT_HASH_MISMATCH")
    provenance = preregistration.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("STAGE2_HYPOTHESIS_PREREGISTRATION_PROVENANCE_MISSING")
    if upstream != provenance.get("upstream_binding_hash"):
        raise ValueError("STAGE2_HYPOTHESIS_UPSTREAM_BINDING_MISMATCH")
    expected = build_stage2_hypothesis_contract(
        preregistration,
        upstream_binding_hash=upstream,
    )
    if dict(value) != expected:
        raise ValueError("STAGE2_HYPOTHESIS_CONTRACT_CONTENT_MISMATCH")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ABSOLUTE_FLOORS",
    "AMENDMENT_SCHEMA_VERSION",
    "BATCH_SIZES",
    "FORMULA_VERSION",
    "HYPOTHESIS_SCHEMA_VERSION",
    "MICROBATCH_COUNTS",
    "MICROBATCH_SELECTION_ORDER",
    "MODELS",
    "PRIMARY_CELLS",
    "PREREGISTRATION_SCHEMA_VERSION",
    "RESOURCE_LIMITS",
    "STREAM_NAMES",
    "TOP_FRACTIONS",
    "build_stage2_amendment_template",
    "build_stage2_hypothesis_contract",
    "build_stage2_preregistration",
    "validate_stage2_hypothesis_contract",
    "validate_stage2_preregistration",
]
