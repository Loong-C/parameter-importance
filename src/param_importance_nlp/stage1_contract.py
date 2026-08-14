"""Machine-readable Stage 1 S1.1 mathematics and traceability contract.

The entry Gate is a freeze, not a numerical result.  This module therefore
keeps the complete field dictionary and the requirement-to-test registration
in one import-safe place, and validates them before a formal Gate can be
published or consumed.  The downstream test IDs are deliberately registered
here before their numerical producers execute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MATH_CONTRACT_SCHEMA = "stage1-math-contract-v1"
REQUIREMENTS_MATRIX_SCHEMA = "stage1-requirements-matrix-v2"


class Stage1ContractError(ValueError):
    """The S1.1 public contract or its verification matrix is incomplete."""


_ESTIMATOR_FIELDS = {
    "raw": "local_gradient_space_importance_raw",
    "raw_clipped": "local_gradient_space_importance_raw_clipped",
    "double_sample": "double_sample_gradient_importance",
    "equal_u": "local_gradient_space_importance_u",
    "weighted_u": "local_gradient_space_importance_u_weighted",
    "u_clipped": "local_gradient_space_importance_u_clipped",
    "actual_update_raw": "actual_update_raw_importance",
}
_ACCUMULATOR_FIELDS = {
    "signed": "importance_signed",
    "positive": "importance_positive",
    "negative_mass": "importance_negative_mass",
    "absolute": "importance_absolute",
    "movement_data": "parameter_movement_data",
    "net_movement_data": "parameter_net_movement_data",
    "movement_total": "parameter_movement_total",
    "net_movement_total": "parameter_net_movement_total",
    "movement_weight_decay": "parameter_movement_weight_decay",
    "net_movement_weight_decay": "parameter_net_movement_weight_decay",
    "magnitude": "parameter_magnitude",
}
_ENTRY_REQUIREMENT_IDS = (
    "entry.stage0_g10",
    "entry.stage0_g10_reuse",
    "entry.repository_clean",
    "entry.agent_files",
    "entry.data_root",
    "entry.cache_policy",
    "entry.write_paths",
    "entry.scientific_config",
)
_NON_FIELD_CONTRACT_REQUIREMENT_IDS = (
    "contract.target_quantity",
    "contract.loss_reduction",
    "contract.lifecycle",
)
_FIELD_REQUIREMENT_IDS = {
    field_name: f"contract.field.{field_name}"
    for field_name in (*_ESTIMATOR_FIELDS.values(), *_ACCUMULATOR_FIELDS.values())
}
STAGE1_REQUIREMENT_IDS = (
    *_ENTRY_REQUIREMENT_IDS,
    *_NON_FIELD_CONTRACT_REQUIREMENT_IDS,
    *(_FIELD_REQUIREMENT_IDS[field_name] for field_name in _FIELD_REQUIREMENT_IDS),
)


_ENTRY_PROFILES = (
    {
        "device": "cpu",
        "dtype": "not_applicable",
        "tolerance_profile": "not_applicable",
    },
)
_NUMERICAL_PROFILES = (
    {
        "device": "cpu",
        "dtype": "float64",
        "tolerance_profile": "T64_ORACLE",
    },
    {
        "device": "cuda:single_gpu",
        "dtype": "float32",
        "tolerance_profile": "T32_SINGLE",
    },
)


def _traceability(
    *,
    math_field: str | None,
    implementation_modules: Sequence[str],
    independent_oracles: Sequence[str],
    test_ids: Sequence[str],
    artifact_roles: Sequence[str],
    expected_gate_ids: Sequence[str],
    fixture_id: str,
    command_id: str,
    device_dtype_profiles: Sequence[Mapping[str, str]] = _NUMERICAL_PROFILES,
) -> dict[str, object]:
    """Build one immutable-in-practice row of the frozen traceability registry."""

    gates = list(expected_gate_ids)
    return {
        "math_field": math_field,
        "implementation_modules": list(implementation_modules),
        "independent_oracles": list(independent_oracles),
        "device_dtype_profiles": [dict(profile) for profile in device_dtype_profiles],
        "test_ids": list(test_ids),
        "artifact_roles": list(artifact_roles),
        "downstream_gate_ids": gates,
        "minimal_repro_bundle": {
            "fixture_id": fixture_id,
            "command_id": command_id,
            "expected_gate_ids": gates,
        },
    }


# This is a registration of downstream verification work, not evidence that a
# future Gate has passed.  Every public mathematical field has a concrete row;
# the validator below uses exact equality so a generic S1.1 publisher/loader
# cannot be substituted for the module, oracle, artifact or Gate that will
# actually establish its numerical semantics.
STAGE1_TRACEABILITY_REGISTRY: dict[str, dict[str, object]] = {
    "entry.stage0_g10": _traceability(
        math_field=None,
        implementation_modules=(
            "src/param_importance_nlp/experiments/stage01_task_runners.py",
            "src/param_importance_nlp/stage1_s1_1.py",
        ),
        independent_oracles=("stage0.G10:immutable-index-and-hash-loader",),
        test_ids=("stage1.s1.1.formal-entry-preflight",),
        artifact_roles=("stage0_g10_handoff", "entry_snapshot", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.stage0_g10_reuse": _traceability(
        math_field=None,
        implementation_modules=(
            "src/param_importance_nlp/experiments/stage01_task_runners.py",
            "src/param_importance_nlp/stage1_s1_1.py",
        ),
        independent_oracles=("policy:evidence-validity-and-rerun:reuse-attestation-loader",),
        test_ids=("stage1.s1.1.g10-reuse-attestation-oracle",),
        artifact_roles=("g10_reuse_attestation", "entry_snapshot", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.repository_clean": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/experiments/stage01_task_runners.py",),
        independent_oracles=("git:status-porcelain-v1",),
        test_ids=("stage1.s1.1.repository-clean-preflight",),
        artifact_roles=("entry_snapshot", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.agent_files": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/experiments/stage01_task_runners.py",),
        independent_oracles=("Agent:five-file-sha256-manifest",),
        test_ids=("stage1.s1.1.agent-file-hash-oracle",),
        artifact_roles=("agent_hash_manifest", "entry_snapshot", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.data_root": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/experiments/stage01_task_runners.py",),
        independent_oracles=("Agent/server.md:data-root-boundary-audit",),
        test_ids=("stage1.s1.1.data-root-preflight",),
        artifact_roles=("entry_snapshot", "write_path_audit", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.cache_policy": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/experiments/stage01_task_runners.py",),
        independent_oracles=("Agent/server.md:cache-and-tmp-boundary-audit",),
        test_ids=("stage1.s1.1.cache-policy-preflight",),
        artifact_roles=("entry_snapshot", "write_path_audit", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.write_paths": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/experiments/stage01_task_runners.py",),
        independent_oracles=("Agent/sync.md:approved-write-path-audit",),
        test_ids=("stage1.s1.1.write-path-preflight",),
        artifact_roles=("entry_snapshot", "write_path_audit", "g1_entry_record"),
        expected_gate_ids=("stage1.G1-ENTRY",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-entry-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "entry.scientific_config": _traceability(
        math_field=None,
        implementation_modules=(
            "src/param_importance_nlp/experiments/stage01_task_runners.py",
            "src/param_importance_nlp/stage1_s1_1.py",
        ),
        independent_oracles=("stage1.s1.1.resolved-scientific-config-loader",),
        test_ids=("stage1.s1.1.scientific-config-binding-oracle",),
        artifact_roles=("resolved_config", "stage_contract", "g1_contract_record"),
        expected_gate_ids=("stage1.G1-CONTRACT",),
        fixture_id="stage1-s1-1-contract-freeze",
        command_id="stage1.s1.1.formal-contract-preflight",
        device_dtype_profiles=_ENTRY_PROFILES,
    ),
    "contract.target_quantity": _traceability(
        math_field=None,
        implementation_modules=("src/param_importance_nlp/core/estimators.py",),
        independent_oracles=("stage1.s1.5.fp64-target-quantity-oracle",),
        test_ids=("stage1.s1.5.target-quantity-contract-oracle",),
        artifact_roles=("estimator_contract_snapshot", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    "contract.loss_reduction": _traceability(
        math_field=None,
        implementation_modules=(
            "src/param_importance_nlp/core/losses.py",
            "src/param_importance_nlp/runtime/gradients.py",
            "src/param_importance_nlp/runtime/training.py",
        ),
        independent_oracles=("stage1.s1.4.fp64-loss-and-gradient-oracle",),
        test_ids=("stage1.s1.4.loss-reduction-gradient-oracle",),
        artifact_roles=("loss_reduction_report", "gradient_oracle_report", "g1_grad_report"),
        expected_gate_ids=("stage1.G1-GRAD",),
        fixture_id="stage1-s1-4-gradient-fixture",
        command_id="stage1.s1.4.gradient-loss-preflight",
    ),
    "contract.lifecycle": _traceability(
        math_field=None,
        implementation_modules=(
            "src/param_importance_nlp/core/losses.py",
            "src/param_importance_nlp/runtime/gradients.py",
            "src/param_importance_nlp/runtime/training.py",
            "src/param_importance_nlp/runtime/optimizer.py",
        ),
        independent_oracles=(
            "stage1.s1.4.gradient-lifecycle-oracle",
            "stage1.s1.6.training-attempt-replay-oracle",
        ),
        test_ids=(
            "stage1.s1.4.gradient-lifecycle-oracle",
            "stage1.s1.6.training-attempt-lifecycle-oracle",
        ),
        artifact_roles=("gradient_lifecycle_trace", "step_lifecycle_trace", "g1_grad_report", "g1_step_report"),
        expected_gate_ids=("stage1.G1-GRAD", "stage1.G1-STEP"),
        fixture_id="stage1-s1-6-training-step-fixture",
        command_id="stage1.s1.6.step-lifecycle-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["raw"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["raw"],
        implementation_modules=("src/param_importance_nlp/core/estimators.py",),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:local_gradient_space_importance_raw",),
        test_ids=("stage1.s1.5.estimator-oracle:local_gradient_space_importance_raw",),
        artifact_roles=("estimator_tensor_bundle", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["raw_clipped"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["raw_clipped"],
        implementation_modules=("src/param_importance_nlp/core/estimators.py",),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:local_gradient_space_importance_raw_clipped",),
        test_ids=("stage1.s1.5.estimator-oracle:local_gradient_space_importance_raw_clipped",),
        artifact_roles=("estimator_tensor_bundle", "clip_factor_trace", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["double_sample"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["double_sample"],
        implementation_modules=("src/param_importance_nlp/core/estimators.py",),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:double_sample_gradient_importance",),
        test_ids=("stage1.s1.5.estimator-oracle:double_sample_gradient_importance",),
        artifact_roles=("estimator_tensor_bundle", "sampling_provenance", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["equal_u"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["equal_u"],
        implementation_modules=(
            "src/param_importance_nlp/core/estimators.py",
            "src/param_importance_nlp/core/sufficient_statistics.py",
        ),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:local_gradient_space_importance_u",),
        test_ids=("stage1.s1.5.estimator-oracle:local_gradient_space_importance_u",),
        artifact_roles=("sufficient_statistics_bundle", "estimator_tensor_bundle", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["weighted_u"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["weighted_u"],
        implementation_modules=(
            "src/param_importance_nlp/core/estimators.py",
            "src/param_importance_nlp/core/sufficient_statistics.py",
        ),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:local_gradient_space_importance_u_weighted",),
        test_ids=("stage1.s1.5.estimator-oracle:local_gradient_space_importance_u_weighted",),
        artifact_roles=("sufficient_statistics_bundle", "estimator_tensor_bundle", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["u_clipped"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["u_clipped"],
        implementation_modules=(
            "src/param_importance_nlp/core/estimators.py",
            "src/param_importance_nlp/core/sufficient_statistics.py",
        ),
        independent_oracles=("stage1.s1.5.fp64-independent-oracle:local_gradient_space_importance_u_clipped",),
        test_ids=("stage1.s1.5.estimator-oracle:local_gradient_space_importance_u_clipped",),
        artifact_roles=("sufficient_statistics_bundle", "clip_factor_trace", "estimator_tensor_bundle", "fp64_estimator_oracle_report", "g1_est_report"),
        expected_gate_ids=("stage1.G1-EST",),
        fixture_id="stage1-s1-5-estimator-fixture",
        command_id="stage1.s1.5.fp64-estimator-preflight",
    ),
    _FIELD_REQUIREMENT_IDS[_ESTIMATOR_FIELDS["actual_update_raw"]]: _traceability(
        math_field=_ESTIMATOR_FIELDS["actual_update_raw"],
        implementation_modules=(
            "src/param_importance_nlp/runtime/optimizer.py",
            "src/param_importance_nlp/runtime/training.py",
            "src/param_importance_nlp/core/accumulator.py",
        ),
        independent_oracles=("stage1.s1.6.adamw-data-delta-independent-replay",),
        test_ids=("stage1.s1.6.actual-update-raw-sign-oracle",),
        artifact_roles=("optimizer_movement_decomposition_trace", "actual_update_oracle_report", "g1_step_report"),
        expected_gate_ids=("stage1.G1-STEP",),
        fixture_id="stage1-s1-6-training-step-fixture",
        command_id="stage1.s1.6.actual-update-raw-preflight",
    ),
}

for _identifier in ("signed", "positive", "negative_mass", "absolute"):
    _field_name = _ACCUMULATOR_FIELDS[_identifier]
    STAGE1_TRACEABILITY_REGISTRY[_FIELD_REQUIREMENT_IDS[_field_name]] = _traceability(
        math_field=_field_name,
        implementation_modules=(
            "src/param_importance_nlp/core/accumulator.py",
            "src/param_importance_nlp/runtime/training.py",
            "src/param_importance_nlp/runtime/optimizer.py",
        ),
        independent_oracles=(f"stage1.s1.6.step-replay-oracle:{_field_name}",),
        test_ids=(f"stage1.s1.6.accumulator-oracle:{_field_name}",),
        artifact_roles=("importance_state", "step_replay_report", "g1_step_report"),
        expected_gate_ids=("stage1.G1-STEP",),
        fixture_id="stage1-s1-6-training-step-fixture",
        command_id="stage1.s1.6.accumulator-preflight",
    )

for _identifier in (
    "movement_data",
    "net_movement_data",
    "movement_total",
    "net_movement_total",
    "movement_weight_decay",
    "net_movement_weight_decay",
    "magnitude",
):
    _field_name = _ACCUMULATOR_FIELDS[_identifier]
    STAGE1_TRACEABILITY_REGISTRY[_FIELD_REQUIREMENT_IDS[_field_name]] = _traceability(
        math_field=_field_name,
        implementation_modules=(
            "src/param_importance_nlp/core/accumulator.py",
            "src/param_importance_nlp/runtime/training.py",
            "src/param_importance_nlp/runtime/optimizer.py",
        ),
        independent_oracles=(f"stage1.s1.6.step-replay-oracle:{_field_name}",),
        test_ids=(f"stage1.s1.6.movement-oracle:{_field_name}",),
        artifact_roles=("movement_decomposition_trace", "step_replay_report", "g1_step_report"),
        expected_gate_ids=("stage1.G1-STEP",),
        fixture_id="stage1-s1-6-training-step-fixture",
        command_id="stage1.s1.6.movement-preflight",
    )

if set(STAGE1_TRACEABILITY_REGISTRY) != set(STAGE1_REQUIREMENT_IDS):
    raise RuntimeError("Stage 1 traceability registry requirement coverage drift")


def _definition(
    *,
    field_name: str,
    formula: str,
    unit: str,
    parameter_time: str,
    gradient_time: str,
    sample_time: str,
    interpretation_boundary: str,
) -> dict[str, str]:
    return {
        "field_name": field_name,
        "formula": formula,
        "unit": unit,
        "parameter_time": parameter_time,
        "gradient_time": gradient_time,
        "sample_time": sample_time,
        "interpretation_boundary": interpretation_boundary,
    }


def build_stage1_math_contract() -> dict[str, object]:
    """Return the full frozen S1.1 field contract.

    Formula text is intentionally symbolic: the stage is fixing the identity,
    ordering and interpretation, while S1.3--S1.10 register the independent
    numerical oracles that exercise each formula.
    """

    estimators = {
        "raw": _definition(
            field_name=_ESTIMATOR_FIELDS["raw"],
            formula="eta_g(k),t * mean_gradient_k,t**2",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="same successful attempt global mean gradient",
            sample_time="after global mean and before optimizer.step",
            interpretation_boundary="unclipped same-batch baseline; not an AdamW path integral",
        ),
        "raw_clipped": _definition(
            field_name=_ESTIMATOR_FIELDS["raw_clipped"],
            formula="s_t * eta_g(k),t * mean_gradient_k,t**2",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="same successful attempt global mean gradient",
            sample_time="after one global clip factor and before optimizer.step",
            interpretation_boundary="diagnostic only; raw remains the unmodified baseline",
        ),
        "double_sample": _definition(
            field_name=_ESTIMATOR_FIELDS["double_sample"],
            formula="eta_g(k),t * mean_gradient_A,k,t * mean_gradient_B,k,t",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="two independent data-stream mean gradients at Theta_t",
            sample_time="before optimizer.step",
            interpretation_boundary="two independent factors; never a probe-loss alignment score",
        ),
        "equal_u": _definition(
            field_name=_ESTIMATOR_FIELDS["equal_u"],
            formula="eta_g(k),t * (S1_k,t**2 - S2_k,t)/(M_t*(M_t-1))",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="M_t >= 2 independent local microbatch mean gradients",
            sample_time="after global sufficient-statistic aggregation and before optimizer.step",
            interpretation_boundary="unclipped equal-microbatch U-statistic core metric",
        ),
        "weighted_u": _definition(
            field_name=_ESTIMATOR_FIELDS["weighted_u"],
            formula="eta_g(k),t * (G1_k,t**2 - G2_k,t)/(W_t**2 - W2_t)",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="independent microbatch means weighted by effective target-token counts",
            sample_time="after global sufficient-statistic aggregation and before optimizer.step",
            interpretation_boundary="requires positive denominator; zero-token microbatches are rejected",
        ),
        "u_clipped": _definition(
            field_name=_ESTIMATOR_FIELDS["u_clipped"],
            formula="s_t * eta_g(k),t * U_k,t",
            unit="loss-decrease contribution per optimizer step",
            parameter_time="Theta_t (before optimizer step)",
            gradient_time="same microbatch population as U_k,t",
            sample_time="after one global clip factor and before optimizer.step",
            interpretation_boundary="plug-in online score; unbiasedness_claim=none because s_t shares randomness with U_k,t",
        ),
        "actual_update_raw": _definition(
            field_name=_ESTIMATOR_FIELDS["actual_update_raw"],
            formula="-data_delta_k,t * mean_gradient_k,t",
            unit="signed first-order loss-decrease proxy per optimizer step",
            parameter_time="Theta_t and post-step parameter state",
            gradient_time="same successful attempt global mean gradient",
            sample_time="after optimizer.step and data/weight-decay decomposition",
            interpretation_boundary="data_delta = total_delta - decoupled_weight_decay_delta; diagnostic only, with no unbiasedness or path-integral claim",
        ),
    }
    accumulators = {
        "signed": _definition(
            field_name=_ACCUMULATOR_FIELDS["signed"],
            formula="sum_t score_k,t",
            unit="cumulative loss-decrease contribution",
            parameter_time="per-score contract parameter time",
            gradient_time="per-score contract gradient time",
            sample_time="after each successful optimizer step",
            interpretation_boundary="main cumulative view; a single score may be negative",
        ),
        "positive": _definition(
            field_name=_ACCUMULATOR_FIELDS["positive"],
            formula="sum_t max(score_k,t, 0)",
            unit="cumulative positive loss-decrease contribution",
            parameter_time="per-score contract parameter time",
            gradient_time="per-score contract gradient time",
            sample_time="after each successful optimizer step",
            interpretation_boundary="derived non-negative mass",
        ),
        "negative_mass": _definition(
            field_name=_ACCUMULATOR_FIELDS["negative_mass"],
            formula="sum_t max(-score_k,t, 0)",
            unit="cumulative negative-contribution mass",
            parameter_time="per-score contract parameter time",
            gradient_time="per-score contract gradient time",
            sample_time="after each successful optimizer step",
            interpretation_boundary="derived non-negative mass; never store a signed negative value",
        ),
        "absolute": _definition(
            field_name=_ACCUMULATOR_FIELDS["absolute"],
            formula="sum_t abs(score_k,t)",
            unit="cumulative absolute contribution mass",
            parameter_time="per-score contract parameter time",
            gradient_time="per-score contract gradient time",
            sample_time="after each successful optimizer step",
            interpretation_boundary="derived magnitude view; equals positive plus negative_mass",
        ),
        "movement_data": _definition(
            field_name=_ACCUMULATOR_FIELDS["movement_data"],
            formula="sum_t abs(data_delta_k,t)",
            unit="cumulative parameter-coordinate displacement",
            parameter_time="pre/post successful optimizer-step parameter states",
            gradient_time="not applicable after optimizer bridge decomposition",
            sample_time="after each successful optimizer step",
            interpretation_boundary="main movement baseline; excludes decoupled weight decay",
        ),
        "net_movement_data": _definition(
            field_name=_ACCUMULATOR_FIELDS["net_movement_data"],
            formula="abs(sum_t data_delta_k,t)",
            unit="net parameter-coordinate displacement",
            parameter_time="pre/post successful optimizer-step parameter states",
            gradient_time="not applicable after optimizer bridge decomposition",
            sample_time="at the requested terminal successful step",
            interpretation_boundary="signed data_delta values are accumulated before absolute value",
        ),
        "movement_total": _definition(
            field_name=_ACCUMULATOR_FIELDS["movement_total"],
            formula="sum_t abs(total_delta_k,t)",
            unit="cumulative parameter-coordinate displacement",
            parameter_time="pre/post successful optimizer-step parameter states",
            gradient_time="not applicable after optimizer bridge decomposition",
            sample_time="after each successful optimizer step",
            interpretation_boundary="diagnostic path length containing every actual update source, including decoupled weight decay",
        ),
        "net_movement_total": _definition(
            field_name=_ACCUMULATOR_FIELDS["net_movement_total"],
            formula="abs(Theta_T,k - Theta_0,k)",
            unit="net parameter-coordinate displacement",
            parameter_time="Theta_0 and terminal Theta_T",
            gradient_time="not applicable",
            sample_time="at the requested terminal training-attempt boundary",
            interpretation_boundary="diagnostic endpoint difference; includes all actual update sources",
        ),
        "movement_weight_decay": _definition(
            field_name=_ACCUMULATOR_FIELDS["movement_weight_decay"],
            formula="sum_t abs(weight_decay_delta_k,t)",
            unit="cumulative parameter-coordinate displacement",
            parameter_time="pre/post successful optimizer-step parameter states",
            gradient_time="not applicable after optimizer bridge decomposition",
            sample_time="after each successful optimizer step",
            interpretation_boundary="diagnostic decoupled AdamW weight-decay path length; zero for accepted no-decay steps",
        ),
        "net_movement_weight_decay": _definition(
            field_name=_ACCUMULATOR_FIELDS["net_movement_weight_decay"],
            formula="abs(sum_t weight_decay_delta_k,t)",
            unit="net parameter-coordinate displacement",
            parameter_time="pre/post successful optimizer-step parameter states",
            gradient_time="not applicable after optimizer bridge decomposition",
            sample_time="at the requested terminal successful step",
            interpretation_boundary="diagnostic signed decoupled weight-decay displacement, accumulated before absolute value",
        ),
        "magnitude": _definition(
            field_name=_ACCUMULATOR_FIELDS["magnitude"],
            formula="abs(Theta_t,k)",
            unit="parameter-coordinate magnitude",
            parameter_time="requested current or terminal parameter state",
            gradient_time="not applicable",
            sample_time="at the explicitly recorded parameter snapshot",
            interpretation_boundary="parameter scale diagnostic, not movement or importance",
        ),
    }
    return {
        "schema_version": MATH_CONTRACT_SCHEMA,
        "formula_version": "stage1-entry-contract-v3",
        "target": {
            "name": "local_gradient_space_contribution",
            "formula": "eta_g(k),t * E[g_k,t]**2 estimated by the named field",
            "unit": "loss-decrease contribution per optimizer step",
            "interpretation_boundary": "local gradient-space quantity, not a complete AdamW path integral",
        },
        "sign_convention": "positive_means_loss_decrease_contribution",
        "loss_reduction": {
            "task": "causal_lm",
            "numerator": "sum(valid_target_token_loss)",
            "denominator": "valid_target_token_count",
            "reduction": "global_valid_target_token_mean",
            "zero_valid": "reject_or_skip",
            "local_microbatch": "independent_local_microbatch_backward",
        },
        "estimators": estimators,
        "accumulators": accumulators,
        "step_lifecycle": [
            "optimizer_before",
            "consume_actual_lr",
            "unscale_local_gradients",
            "aggregate_global_sufficient_statistics",
            "compute_one_global_clip_factor",
            "compute_score",
            "accumulate_successful_step_views",
            "apply_optimizer_step",
            "decompose_data_and_weight_decay_movement",
            "publish_training_attempt_boundary",
        ],
        "nonfinite_policy": "skipped attempts consume batch/RNG and increment attempt counters but do not change successful-step scores, movement or scheduler",
        "exclusions": [
            "BatchNorm",
            "in_batch_negatives",
            "cross_sample_losses",
            "shared_random_augmentation",
        ],
    }


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise Stage1ContractError(
            f"{field} fields mismatch: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}"
        )


def _require_nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage1ContractError(f"{field} must be non-empty text")
    return value


def validate_stage1_math_contract(value: Mapping[str, object]) -> None:
    """Reject a formal contract with omitted, ambiguous or duplicate public fields."""

    _require_exact_keys(
        value,
        {
            "schema_version", "formula_version", "target", "sign_convention",
            "loss_reduction", "estimators", "accumulators", "step_lifecycle",
            "nonfinite_policy", "exclusions",
        },
        "math_contract",
    )
    if value.get("schema_version") != MATH_CONTRACT_SCHEMA:
        raise Stage1ContractError("math_contract schema_version is not current")
    _require_nonempty_text(value.get("formula_version"), "math_contract.formula_version")
    if value.get("sign_convention") != "positive_means_loss_decrease_contribution":
        raise Stage1ContractError("math_contract sign convention is not frozen")

    target = value.get("target")
    if not isinstance(target, Mapping):
        raise Stage1ContractError("math_contract.target must be an object")
    _require_exact_keys(target, {"name", "formula", "unit", "interpretation_boundary"}, "math_contract.target")
    for key, item in target.items():
        _require_nonempty_text(item, f"math_contract.target.{key}")

    loss = value.get("loss_reduction")
    if not isinstance(loss, Mapping):
        raise Stage1ContractError("math_contract.loss_reduction must be an object")
    _require_exact_keys(
        loss,
        {"task", "numerator", "denominator", "reduction", "zero_valid", "local_microbatch"},
        "math_contract.loss_reduction",
    )
    for key, item in loss.items():
        _require_nonempty_text(item, f"math_contract.loss_reduction.{key}")

    seen_field_names: set[str] = set()
    for section, expected_ids in (("estimators", set(_ESTIMATOR_FIELDS)), ("accumulators", set(_ACCUMULATOR_FIELDS))):
        definitions = value.get(section)
        if not isinstance(definitions, Mapping):
            raise Stage1ContractError(f"math_contract.{section} must be an object")
        _require_exact_keys(definitions, expected_ids, f"math_contract.{section}")
        for identifier, field_name in ({**_ESTIMATOR_FIELDS, **_ACCUMULATOR_FIELDS}).items():
            if identifier not in definitions:
                continue
            definition = definitions[identifier]
            if not isinstance(definition, Mapping):
                raise Stage1ContractError(f"math_contract.{section}.{identifier} must be an object")
            _require_exact_keys(
                definition,
                {"field_name", "formula", "unit", "parameter_time", "gradient_time", "sample_time", "interpretation_boundary"},
                f"math_contract.{section}.{identifier}",
            )
            for key, item in definition.items():
                _require_nonempty_text(item, f"math_contract.{section}.{identifier}.{key}")
            if definition["field_name"] != field_name:
                raise Stage1ContractError(f"math_contract.{section}.{identifier} field name drift")
            if field_name in seen_field_names:
                raise Stage1ContractError(f"math_contract duplicate public field: {field_name}")
            seen_field_names.add(field_name)

    lifecycle = value.get("step_lifecycle")
    if not isinstance(lifecycle, list) or len(lifecycle) != len(set(lifecycle)):
        raise Stage1ContractError("math_contract.step_lifecycle must be a unique ordered list")
    required_lifecycle = {
        "optimizer_before", "consume_actual_lr", "unscale_local_gradients",
        "aggregate_global_sufficient_statistics", "compute_one_global_clip_factor",
        "compute_score", "accumulate_successful_step_views", "apply_optimizer_step",
        "decompose_data_and_weight_decay_movement", "publish_training_attempt_boundary",
    }
    if set(lifecycle) != required_lifecycle or not all(isinstance(item, str) for item in lifecycle):
        raise Stage1ContractError("math_contract.step_lifecycle is incomplete")
    for list_name in ("exclusions",):
        listed = value.get(list_name)
        if not isinstance(listed, list) or not listed or not all(isinstance(item, str) and item for item in listed):
            raise Stage1ContractError(f"math_contract.{list_name} must be a non-empty string list")
    _require_nonempty_text(value.get("nonfinite_policy"), "math_contract.nonfinite_policy")
    if dict(value) != build_stage1_math_contract():
        raise Stage1ContractError("math_contract frozen formula, unit or lifecycle payload drift")


def build_stage1_requirements_matrix(
    *,
    scope: str,
    requirement_checks: Mapping[str, bool],
    local_checks: Mapping[str, bool],
    source_refs: Mapping[str, str],
    external_refs: Sequence[str],
) -> dict[str, object]:
    """Pre-register the required verifier, profile and failure bundle per requirement."""

    if scope not in {"formal", "local_fixture"}:
        raise Stage1ContractError("requirements matrix scope is invalid")
    if set(requirement_checks) != set(STAGE1_REQUIREMENT_IDS) or set(local_checks) != set(STAGE1_REQUIREMENT_IDS):
        raise Stage1ContractError("requirements matrix requirement set drift")
    if set(source_refs) != set(STAGE1_REQUIREMENT_IDS):
        raise Stage1ContractError("requirements matrix source mapping drift")
    entries: list[dict[str, object]] = []
    for requirement_id in STAGE1_REQUIREMENT_IDS:
        traceability = STAGE1_TRACEABILITY_REGISTRY[requirement_id]
        entries.append(
            {
                "requirement_id": requirement_id,
                "source_ref": source_refs[requirement_id],
                **traceability,
                "local_status": "PASS" if local_checks[requirement_id] else "NOT_RUN",
                "formal_status": "PASS" if scope == "formal" and requirement_checks[requirement_id] else "BLOCKED" if scope == "formal" else "NOT_RUN",
                "evidence_refs": list(external_refs),
                "notes": "registration only: local fixture never unlocks a formal Gate",
            }
        )
    matrix = {
        "schema_version": REQUIREMENTS_MATRIX_SCHEMA,
        "task_id": "stage1.01_entry_and_contract",
        "scope": scope,
        "requirements": entries,
        "summary": {
            "total": len(entries),
            "local_pass": sum(1 for entry in entries if entry["local_status"] == "PASS"),
            "formal_pass": sum(1 for entry in entries if entry["formal_status"] == "PASS"),
            "formal_blocked": sum(1 for entry in entries if entry["formal_status"] == "BLOCKED"),
        },
    }
    validate_stage1_requirements_matrix(matrix)
    return matrix


def validate_stage1_requirements_matrix(value: Mapping[str, object]) -> None:
    """Require complete test/oracle/profile/artifact/repro registration for every row."""

    _require_exact_keys(value, {"schema_version", "task_id", "scope", "requirements", "summary"}, "requirements_matrix")
    if value.get("schema_version") != REQUIREMENTS_MATRIX_SCHEMA:
        raise Stage1ContractError("requirements_matrix schema_version is not current")
    if value.get("task_id") != "stage1.01_entry_and_contract" or value.get("scope") not in {"formal", "local_fixture"}:
        raise Stage1ContractError("requirements_matrix task identity is invalid")
    rows = value.get("requirements")
    if not isinstance(rows, list) or len(rows) != len(STAGE1_REQUIREMENT_IDS):
        raise Stage1ContractError("requirements_matrix row count is invalid")
    seen: set[str] = set()
    required_row_fields = {
        "requirement_id", "math_field", "source_ref", "implementation_modules", "independent_oracles",
        "device_dtype_profiles", "test_ids", "artifact_roles", "downstream_gate_ids",
        "minimal_repro_bundle", "local_status", "formal_status", "evidence_refs", "notes",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Stage1ContractError(f"requirements_matrix row {index} must be an object")
        _require_exact_keys(row, required_row_fields, f"requirements_matrix row {index}")
        requirement_id = row.get("requirement_id")
        if not isinstance(requirement_id, str) or requirement_id not in STAGE1_REQUIREMENT_IDS or requirement_id in seen:
            raise Stage1ContractError(f"requirements_matrix invalid requirement id at row {index}")
        seen.add(requirement_id)
        expected_traceability = STAGE1_TRACEABILITY_REGISTRY[requirement_id]
        for field, expected in expected_traceability.items():
            if row.get(field) != expected:
                raise Stage1ContractError(
                    f"requirements_matrix.{requirement_id}.{field} traceability drift"
                )
        _require_nonempty_text(row.get("source_ref"), f"requirements_matrix.{requirement_id}.source_ref")
        for field in (
            "implementation_modules",
            "independent_oracles",
            "test_ids",
            "artifact_roles",
            "downstream_gate_ids",
            "evidence_refs",
        ):
            items = row.get(field)
            if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
                raise Stage1ContractError(f"requirements_matrix.{requirement_id}.{field} must be a string list")
            if field != "evidence_refs" and not items:
                raise Stage1ContractError(f"requirements_matrix.{requirement_id}.{field} must not be empty")
        profiles = row.get("device_dtype_profiles")
        if not isinstance(profiles, list) or not profiles:
            raise Stage1ContractError(
                f"requirements_matrix.{requirement_id}.device_dtype_profiles must be a non-empty list"
            )
        for profile_index, profile in enumerate(profiles):
            if not isinstance(profile, Mapping):
                raise Stage1ContractError(
                    f"requirements_matrix.{requirement_id}.device_dtype_profiles[{profile_index}] must be an object"
                )
            _require_exact_keys(
                profile,
                {"device", "dtype", "tolerance_profile"},
                f"requirements_matrix.{requirement_id}.device_dtype_profiles[{profile_index}]",
            )
            for key, item in profile.items():
                _require_nonempty_text(
                    item,
                    f"requirements_matrix.{requirement_id}.device_dtype_profiles[{profile_index}].{key}",
                )
        repro = row.get("minimal_repro_bundle")
        if not isinstance(repro, Mapping):
            raise Stage1ContractError(f"requirements_matrix.{requirement_id}.minimal_repro_bundle must be an object")
        _require_exact_keys(repro, {"fixture_id", "command_id", "expected_gate_ids"}, f"requirements_matrix.{requirement_id}.minimal_repro_bundle")
        _require_nonempty_text(repro.get("fixture_id"), f"requirements_matrix.{requirement_id}.minimal_repro_bundle.fixture_id")
        _require_nonempty_text(repro.get("command_id"), f"requirements_matrix.{requirement_id}.minimal_repro_bundle.command_id")
        expected_gates = repro.get("expected_gate_ids")
        if not isinstance(expected_gates, list) or not all(
            isinstance(item, str) and item for item in expected_gates
        ):
            raise Stage1ContractError(
                f"requirements_matrix.{requirement_id}.minimal_repro_bundle.expected_gate_ids must be a string list"
            )
        if row.get("local_status") not in {"PASS", "NOT_RUN"} or row.get("formal_status") not in {"PASS", "BLOCKED", "NOT_RUN"}:
            raise Stage1ContractError(f"requirements_matrix.{requirement_id} status is invalid")
        _require_nonempty_text(row.get("notes"), f"requirements_matrix.{requirement_id}.notes")
    if seen != set(STAGE1_REQUIREMENT_IDS):
        raise Stage1ContractError("requirements_matrix requirement coverage is incomplete")
    summary = value.get("summary")
    if not isinstance(summary, Mapping):
        raise Stage1ContractError("requirements_matrix.summary must be an object")
    _require_exact_keys(summary, {"total", "local_pass", "formal_pass", "formal_blocked"}, "requirements_matrix.summary")
    expected_summary = {
        "total": len(rows),
        "local_pass": sum(1 for row in rows if row["local_status"] == "PASS"),
        "formal_pass": sum(1 for row in rows if row["formal_status"] == "PASS"),
        "formal_blocked": sum(1 for row in rows if row["formal_status"] == "BLOCKED"),
    }
    if dict(summary) != expected_summary:
        raise Stage1ContractError("requirements_matrix summary does not match rows")


__all__ = [
    "MATH_CONTRACT_SCHEMA",
    "REQUIREMENTS_MATRIX_SCHEMA",
    "STAGE1_REQUIREMENT_IDS",
    "STAGE1_TRACEABILITY_REGISTRY",
    "Stage1ContractError",
    "build_stage1_math_contract",
    "build_stage1_requirements_matrix",
    "validate_stage1_math_contract",
    "validate_stage1_requirements_matrix",
]
