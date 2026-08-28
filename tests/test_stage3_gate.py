from __future__ import annotations

from datetime import datetime, timezone

import pytest

from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    ProvenanceRecord,
    ProvenanceStatus,
    RunIdentity,
    derive_experiment_id,
    validate_stage23_artifact,
)
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.stage3_formal import (
    QuadratureObservation,
    QuadratureRecommendation,
    QuadratureRecommendationEngine,
    QuadratureThresholds,
)
from param_importance_nlp.experiments.stage3_gate import Stage3GateEvaluator
from param_importance_nlp.contracts.errors import FormalRunRejected


HASHES = tuple(chr(ord("a") + index) * 64 for index in range(6))
SCOPE_DECISION_REF = "evidence/stage3/g30-scope-decision.json"
SCOPE_GATE_REF = "evidence/stage3/g30-scope-gate.json"
GATE_REFS = (SCOPE_DECISION_REF,) + tuple(
    f"evidence/g3-{index}.json" for index in range(1, 6)
)


def _scope_decision() -> dict[str, object]:
    decision: dict[str, object] = {
        "schema_version": "stage3-g30-user-scope-decision-v1",
        "decision_id": "stage3-g30-test-decision",
        "authority": "explicit_user_direction",
        "decided_at": "2026-08-28T00:00:00Z",
        "gate_id": "stage3.G3-0",
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "user_direction": "enter Stage 3 directly",
        "downstream_effect": {
            "stage3_real_experiments_authorized": True,
            "g3_0_prerequisite_authorized": True,
            "stage2_formal_replay_required_for_stage3_start": False,
        },
        "accepted_stage_inputs": {
            "stage0": {"availability": "ASSUMED_PASSED_AND_STILL_AVAILABLE"},
            "stage1": {"availability": "ASSUMED_PASSED_AND_STILL_AVAILABLE"},
            "stage2": {
                "default_estimator": "U-32",
                "batch_size": 32,
                "sensitivity_control": "Raw",
                "run_id": "pythia-grid-20260826T145530Z",
                "source_branch": "exp/stage2-direct-all-20260826",
                "source_commit": "000ce1e79af791ce1eae2e2b62da221a10dd3c9a",
            },
        },
        "preserved_evidence_refs": ["evidence/stage2/direct-delivery.json"],
        "non_claims": [
            "This decision does not relabel any Stage 2 direct-unvalidated artifact as formal.",
            "This decision does not waive any Stage 3 scientific threshold, tolerance, margin, or Gate after G3-0.",
        ],
    }
    decision["artifact_hash"] = canonical_json_hash(decision)
    return decision


def _scope_gate() -> GateRecord:
    return GateRecord(
        gate_id="stage3.G3-0",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:00:00Z",
        measured={
            "authority": "explicit_user_direction",
            "stage0": "assumed_pass_and_available",
            "stage1": "assumed_pass_and_available",
            "stage2_batch_size": 32,
            "stage2_estimator": "U-32",
            "stage2_run_id": "pythia-grid-20260826T145530Z",
            "stage2_sensitivity_control": "Raw",
            "stage3_start_authorized": True,
        },
        threshold={
            "stage0_available": True,
            "stage1_available": True,
            "stage2_decision_available": True,
            "stage3_start_authorized": True,
        },
        evidence_refs=(SCOPE_DECISION_REF,),
    )


def _scope_kwargs() -> dict[str, object]:
    return {
        "stage3_scope_decision": _scope_decision(),
        "stage3_scope_gate": _scope_gate(),
        "stage3_scope_decision_ref": SCOPE_DECISION_REF,
        "stage3_scope_gate_ref": SCOPE_GATE_REF,
    }


def _gates() -> tuple[GateRecord, ...]:
    return tuple(
        _scope_gate()
        if index == 0
        else GateRecord(
            gate_id=f"stage3.G3-{index}",
            stage=3,
            status=GateStatus.PASS,
            checked_at="2026-08-28T00:00:00Z",
            evidence_refs=(GATE_REFS[index],),
        )
        for index in range(6)
    )


def _execution(gates: tuple[GateRecord, ...]) -> FormalExecutionEvidence:
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="f" * 64,
        asset_manifest_hashes=("e" * 64,),
        prerequisite_gates=gates,
    )


def _provenance(gates: tuple[GateRecord, ...]) -> ProvenanceRecord:
    config_hash = "1" * 64
    identity = RunIdentity.create(
        experiment_id=derive_experiment_id(
            stage=3,
            task="stage3-gate",
            model_identity="model-v1",
            route="formal",
            master_seed=1,
            config_hash=config_hash,
        ),
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        collision_code="0123abcd",
    )
    return ProvenanceRecord(
        identity=identity,
        config_hash=config_hash,
        resolved_config_ref="configs/formal.json",
        seed_plan_hash="2" * 64,
        git_commit="3" * 40,
        git_branch="formal-stage3",
        worktree_clean=True,
        environment_id="gpu-formal-v1",
        hardware_snapshot_ref="evidence/hardware.json",
        device_mapping=("cuda:0",),
        model_manifest_id="manifests/model.json",
        data_manifest_id="manifests/data.json",
        started_at="2026-08-28T00:00:00Z",
        ended_at="2026-08-28T00:01:00Z",
        status=ProvenanceStatus.COMPLETED,
        scope="formal",
        artifact_refs=GATE_REFS + (SCOPE_GATE_REF, "evidence/metrics.json"),
    )


def _thresholds() -> dict[str, object]:
    return {
        "max_normalized_l1_error": 0.01,
        "max_normalized_l2_error": 0.01,
        "max_normalized_linf_error": 0.01,
        "max_completeness_absolute_residual": 0.01,
        "max_completeness_relative_residual": 0.01,
        "max_completeness_l1_scaled_residual": 0.01,
        "min_active_spearman": 0.99,
        "min_spearman": 0.99,
        "min_cosine_similarity": 0.99,
        "min_sign_consistency": 0.99,
        "min_topk_overlap": 0.95,
        "min_topq_overlap": 0.95,
        "min_topq_jaccard": 0.95,
        "max_layer_quality_tv": 0.01,
        "max_module_quality_tv": 0.01,
        "max_reference_normalized_l1_error": 0.001,
        "completeness_stability_epsilon": 1e-12,
        "active_set_threshold": 0.0,
        "max_unique_nodes": 16,
        "top_q_values": [0.001, 0.01, 0.05],
        "required_strata": ["model", "stage", "update", "probe"],
        "require_worst_case": True,
    }


def _formal_plan(
    execution: FormalExecutionEvidence,
    *,
    rules: tuple[str, ...] = ("midpoint",),
    units: tuple[str, ...] = ("model14-early-update1-probe1",),
    thresholds: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "stage3-formal-pilot-plan-v1",
        "plan_id": "stage3-formal-plan",
        "scope": "formal",
        "candidate_rules": list(rules),
        "required_unit_ids": list(units),
        "unit_strata": {
            unit: {
                "model": "14M",
                "stage": "early",
                "update": "u1",
                "probe": "p1",
            }
            for unit in units
        },
        "thresholds": _thresholds() if thresholds is None else thresholds,
        "execution_evidence_hash": execution.artifact_hash,
        "formal_eligible": True,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def _row(
    *,
    complete: bool = True,
    rule_name: str = "midpoint",
    normalized_l1_error: float = 0.001,
    unique_nodes: int = 4,
) -> dict[str, object]:
    row: dict[str, object] = {
        "unit_id": "model14-early-update1-probe1",
        "rule_name": rule_name,
        "unique_nodes": unique_nodes,
        "normalized_l1_error": normalized_l1_error,
        "normalized_l2_error": 0.001,
        "normalized_linf_error": 0.001,
        "completeness_absolute_residual": 0.001,
        "completeness_relative_residual": 0.001,
        "completeness_l1_scaled_residual": 0.001,
        "spearman": 1.0,
        "active_spearman": 1.0,
        "cosine_similarity": 1.0,
        "sign_consistency": 1.0,
        "topk_overlap": 1.0,
        "top_q": {
            "0.001": {"overlap": 1.0, "jaccard": 1.0},
            "0.01": {"overlap": 1.0, "jaccard": 1.0},
            "0.05": {"overlap": 1.0, "jaccard": 1.0},
        },
        "layer_quality_tv": 0.001,
        "module_quality_tv": 0.001,
        "reference_normalized_l1_error": 0.0001,
        "strata": {"model": "14M", "stage": "early", "update": "u1", "probe": "p1"},
        "worst_case": True,
        "evidence_refs": ["evidence/metrics.json"],
        "scope": "formal",
    }
    if not complete:
        row.pop("normalized_l2_error")
    return row


def test_stage3_evaluator_requires_all_metrics_and_real_gates() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-audit",
        execution=execution,
        observations=[_row(complete=False)],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.formal_eligible is False
    assert result.status == "BLOCKED"
    assert result.rule_evaluations["midpoint"]["missing_metrics"] == [
        "normalized_l2_error"
    ]
    with pytest.raises(FormalRunRejected):
        result.require_pass()


def test_stage3_evaluator_requires_cosine_as_a_complete_diagnostic() -> None:
    gates = _gates()
    execution = _execution(gates)
    row = _row()
    row.pop("cosine_similarity")
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-cosine-required",
        execution=execution,
        observations=[row],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "BLOCKED"
    assert "cosine_similarity" in result.rule_evaluations["midpoint"][
        "missing_metrics"
    ]


@pytest.mark.parametrize(
    ("metric", "value", "expected_violation"),
    (
        ("spearman", 0.98, "spearman:0.98<0.99"),
        ("cosine_similarity", 0.98, "cosine_similarity:0.98<0.99"),
        ("topk_overlap", 0.94, "topk_overlap:0.94<0.95"),
    ),
)
def test_stage3_evaluator_enforces_frozen_ranking_similarity_thresholds(
    metric: str,
    value: float,
    expected_violation: str,
) -> None:
    gates = _gates()
    execution = _execution(gates)
    row = _row()
    row[metric] = value
    result = Stage3GateEvaluator().evaluate(
        evaluation_id=f"stage3-gate-{metric}-threshold",
        execution=execution,
        observations=[row],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=(
            "evidence/metrics.json",
            GATE_REFS[5],
            SCOPE_DECISION_REF,
            SCOPE_GATE_REF,
        ),
        **_scope_kwargs(),
    )
    assert result.status == "BLOCKED"
    assert expected_violation in result.rule_evaluations["midpoint"]["violations"]


def test_stage3_evaluator_passes_only_complete_six_gate_chain() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-pass",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.formal_eligible is True
    assert result.status == "PASS"
    assert len(result.gate_hashes) == 6
    assert result.stage3_scope_decision_ref == SCOPE_DECISION_REF
    assert result.stage3_scope_decision_hash == _scope_decision()["artifact_hash"]
    assert result.stage3_scope_gate_ref == SCOPE_GATE_REF
    assert result.stage3_scope_gate_hash == _scope_gate().artifact_hash
    assert result.from_mapping(result.to_dict()).artifact_hash == result.artifact_hash


def test_stage3_evaluator_rejects_a_plain_g30_pass_without_user_scope_authority() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-plain-g30-only",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5]),
    )
    assert result.status == "BLOCKED"
    assert "STAGE3_G30_SCOPE_DECISION_REQUIRED" in result.reasons


def test_stage3_evaluator_rejects_scope_decision_ref_or_gate_drift() -> None:
    gates = _gates()
    execution = _execution(gates)
    decision = _scope_decision()
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-scope-binding-drift",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        stage3_scope_decision=decision,
        stage3_scope_gate=_scope_gate(),
        stage3_scope_decision_ref="evidence/stage3/other-decision.json",
        stage3_scope_gate_ref=SCOPE_GATE_REF,
    )
    assert result.status == "BLOCKED"
    assert any("STAGE3_G30_SCOPE_AUTHORITY_INVALID" in item for item in result.reasons)


def test_stage3_evaluator_derives_worst_case_and_ignores_producer_flag() -> None:
    gates = _gates()
    execution = _execution(gates)
    row = _row()
    row["worst_case"] = False
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-derived-worst-case",
        execution=execution,
        observations=[row],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "PASS"
    evaluation = result.rule_evaluations["midpoint"]
    assert evaluation["worst_case_source"] == "derived_from_complete_frozen_table"
    assert evaluation["worst_case_unit_ids"] == [
        "model14-early-update1-probe1"
    ]


def test_stage3_evaluator_rejects_observation_strata_drift_from_plan() -> None:
    gates = _gates()
    execution = _execution(gates)
    row = _row()
    row["strata"] = {
        "model": "14M",
        "stage": "late",
        "update": "u1",
        "probe": "p1",
    }
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-strata-drift",
        execution=execution,
        observations=[row],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "BLOCKED"
    assert "strata_plan_mismatch:model14-early-update1-probe1" in (
        result.rule_evaluations["midpoint"]["violations"]
    )


def test_stage3_evaluator_eliminates_failed_candidate_without_blocking_pass() -> None:
    gates = _gates()
    execution = _execution(gates)
    rules = ("left", "trapezoid")
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-mixed-candidates",
        execution=execution,
        observations=[
            _row(rule_name="left", normalized_l1_error=0.02),
            _row(rule_name="trapezoid"),
        ],
        formal_plan=_formal_plan(execution, rules=rules),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "PASS"
    assert result.reasons == ()
    assert result.rule_evaluations["left"]["passing"] is False
    assert result.rule_evaluations["trapezoid"]["passing"] is True


def test_stage3_evaluator_blocks_when_no_candidate_passes() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-no-passing-rule",
        execution=execution,
        observations=[_row(normalized_l1_error=0.02)],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "BLOCKED"
    assert "STAGE3_NO_PASSING_RULES" in result.reasons


def test_stage3_evaluator_enforces_unique_node_budget() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-node-budget",
        execution=execution,
        observations=[_row(unique_nodes=17)],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "BLOCKED"
    assert result.rule_evaluations["midpoint"]["violations"] == ["unique_nodes:17>16"]


def test_stage3_evaluator_does_not_invent_completeness_scale_bound() -> None:
    gates = _gates()
    execution = _execution(gates)
    thresholds = _thresholds()
    thresholds["max_completeness_absolute_residual"] = 2.0
    thresholds["max_completeness_relative_residual"] = 3.0
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-pilot-calibrated-completeness",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution, thresholds=thresholds),
        formal_plan_ref=GATE_REFS[5],
        thresholds=thresholds,
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    assert result.status == "PASS"


def test_stage3_gate_mapping_cannot_forge_pass_without_source_refs() -> None:
    gates = _gates()
    execution = _execution(gates)
    result = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-source-binding",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=_thresholds(),
        provenance=_provenance(gates),
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    forged = result.to_dict()
    forged["source_artifact_refs"] = []
    payload = {key: value for key, value in forged.items() if key != "artifact_hash"}
    forged["artifact_hash"] = canonical_json_hash(payload)
    with pytest.raises(FormalRunRejected, match="PASS_REQUIRES_PROVENANCE_AND_SOURCES"):
        result.from_mapping(forged)


def test_formal_recommendation_qualification_binds_independent_evaluation() -> None:
    gates = _gates()
    execution = _execution(gates)
    provenance = _provenance(gates)
    threshold_values = _thresholds()
    thresholds = QuadratureThresholds(**threshold_values)  # type: ignore[arg-type]
    observation = QuadratureObservation(
        unit_id="model14-early-update1-probe1",
        rule_name="midpoint",
        unique_nodes=4,
        normalized_l1_error=0.001,
        completeness_absolute_residual=0.001,
        spearman=1.0,
        topk_overlap=1.0,
        wall_seconds=0.1,
        normalized_l2_error=0.001,
        normalized_linf_error=0.001,
        completeness_relative_residual=0.001,
        completeness_l1_scaled_residual=0.001,
        active_spearman=1.0,
        cosine_similarity=1.0,
        sign_consistency=1.0,
        topq_overlap={"0.001": 1.0, "0.01": 1.0, "0.05": 1.0},
        topq_jaccard={"0.001": 1.0, "0.01": 1.0, "0.05": 1.0},
        layer_quality_tv=0.001,
        module_quality_tv=0.001,
        reference_normalized_l1_error=0.0001,
        strata={"model": "14M", "stage": "early", "update": "u1", "probe": "p1"},
        evidence_refs=("evidence/metrics.json",),
        scope="formal",
    )
    recommendation = QuadratureRecommendationEngine().recommend(
        recommendation_id="stage3-formal-recommendation",
        observations=(observation,),
        required_unit_ids=("model14-early-update1-probe1",),
        thresholds=thresholds,
        execution=execution,
    )
    evaluation = Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-gate-qualified-recommendation",
        execution=execution,
        observations=[_row()],
        formal_plan=_formal_plan(execution),
        formal_plan_ref=GATE_REFS[5],
        thresholds=threshold_values,
        provenance=provenance,
        source_artifact_refs=("evidence/metrics.json", GATE_REFS[5], SCOPE_DECISION_REF, SCOPE_GATE_REF),
        **_scope_kwargs(),
    )
    evaluation_ref = "evidence/stage3-gate-evaluation.json"
    g37 = GateRecord(
        gate_id="stage3.G3-7",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:02:00Z",
        evidence_refs=(evaluation_ref,),
    )
    qualified = recommendation.qualify(
        execution=execution,
        gate=g37,
        artifact_ref=evaluation_ref,
        gate_evaluation=evaluation,
        provenance=provenance,
    )
    assert qualified.status == "QUALIFIED"
    assert qualified.qualification.formal_eligible is True
    assert qualified.gate_evaluation_hash == evaluation.artifact_hash
    assert qualified.provenance_hash == provenance.artifact_hash
    reloaded = QuadratureRecommendation.from_mapping_authorized(
        recommendation.to_dict(),
        execution=execution,
        gate=g37,
        artifact_ref=evaluation_ref,
        gate_evaluation=evaluation,
        provenance=provenance,
    )
    assert reloaded.artifact_hash == qualified.artifact_hash
    with pytest.raises(FormalRunRejected, match="AUTHORITY_AWARE_LOADER"):
        qualified.from_mapping(qualified.to_dict())
    with pytest.raises(FormalRunRejected, match="AUTHORITY_AWARE_VALIDATION"):
        validate_stage23_artifact(qualified.to_dict())
