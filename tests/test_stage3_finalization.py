from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from param_importance_nlp.analysis.report import FrozenSourceTable
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
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.contracts.immutable import thaw_json_value
from param_importance_nlp.experiments.stage3_finalization import Stage3Finalizer
from param_importance_nlp.experiments.stage3_formal import (
    QuadratureRecommendationEngine,
    QuadratureThresholds,
)
from param_importance_nlp.experiments.stage3_gate import Stage3GateEvaluator


DECISION_REF = "evidence/stage3/g30-scope-decision.json"
SCOPE_GATE_REF = "evidence/stage3/g30-scope-gate.json"
GATE_REFS = (DECISION_REF,) + tuple(f"evidence/stage3/g3-{i}.json" for i in range(1, 6))
TABLE_REF = "evidence/stage3/formal-frozen-table.json"
PLAN_REF = "evidence/stage3/formal-plan.json"
EVAL_REF = "evidence/stage3/gate-evaluation.json"
PROV_REF = "evidence/stage3/provenance.json"
RECOMMENDATION_REF = "evidence/stage3/recommendation.json"


def _scope_decision() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "stage3-g30-user-scope-decision-v1",
        "decision_id": "stage3-g30-finalization-test",
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
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _g0() -> GateRecord:
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
        evidence_refs=(DECISION_REF,),
    )


def _gates() -> tuple[GateRecord, ...]:
    return tuple(
        _g0()
        if i == 0
        else GateRecord(
            gate_id=f"stage3.G3-{i}",
            stage=3,
            status=GateStatus.PASS,
            checked_at="2026-08-28T00:00:00Z",
            evidence_refs=(GATE_REFS[i], PLAN_REF)
            if i == 5
            else (GATE_REFS[i],),
        )
        for i in range(6)
    )


def _execution(gates: tuple[GateRecord, ...]) -> FormalExecutionEvidence:
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="f" * 64,
        asset_manifest_hashes=("e" * 64,),
        prerequisite_gates=gates,
    )


def _provenance(gates: tuple[GateRecord, ...], refs: tuple[str, ...]) -> ProvenanceRecord:
    config_hash = "1" * 64
    identity = RunIdentity.create(
        experiment_id=derive_experiment_id(
            stage=3,
            task="stage3-finalization",
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
        artifact_refs=tuple(dict.fromkeys((*GATE_REFS, SCOPE_GATE_REF, *refs))),
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


def _plan(execution: FormalExecutionEvidence) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "stage3-formal-pilot-plan-v1",
        "plan_id": "stage3-finalization-plan",
        "scope": "formal",
        "candidate_rules": ["midpoint", "trapezoid"],
        "required_unit_ids": ["model14-early-update1-probe1"],
        "unit_strata": {
            "model14-early-update1-probe1": {
                "model": "14M",
                "stage": "early",
                "update": "u1",
                "probe": "p1",
            }
        },
        "thresholds": _thresholds(),
        "execution_evidence_hash": execution.artifact_hash,
        "formal_eligible": True,
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _row(rule: str) -> dict[str, object]:
    return {
        "unit_id": "model14-early-update1-probe1",
        "rule_name": rule,
        "unique_nodes": 4 if rule == "midpoint" else 8,
        "deterministic_node_cost_units": 4 if rule == "midpoint" else 8,
        "normalized_l1_error": 0.001,
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
        "topq_overlap": {"0.001": 1.0, "0.01": 1.0, "0.05": 1.0},
        "topq_jaccard": {"0.001": 1.0, "0.01": 1.0, "0.05": 1.0},
        "layer_quality_tv": 0.001,
        "module_quality_tv": 0.001,
        "reference_normalized_l1_error": 0.0001,
        "strata": {"model": "14M", "stage": "early", "update": "u1", "probe": "p1"},
        "evidence_refs": [TABLE_REF, "evidence/metrics.json"],
        "scope": "formal",
        "wall_seconds": 0.1 if rule == "midpoint" else 0.2,
        "gradient_evaluations": 1,
        "loss_evaluations": 1,
        "forward_evaluations": 1,
        "backward_evaluations": 1,
        "peak_gpu_memory_bytes": 1024,
    }


def _evaluation(execution: FormalExecutionEvidence, gates: tuple[GateRecord, ...], plan: dict[str, object], provenance: ProvenanceRecord):
    rows = [_row("midpoint"), _row("trapezoid")]
    from param_importance_nlp.experiments.stage3_gate import Stage3GateEvaluator

    return Stage3GateEvaluator().evaluate(
        evaluation_id="stage3-finalization-evaluation",
        execution=execution,
        observations=rows,
        formal_plan=plan,
        formal_plan_ref=PLAN_REF,
        thresholds=plan["thresholds"],
        provenance=provenance,
        source_artifact_refs=(TABLE_REF, PLAN_REF, "evidence/metrics.json", DECISION_REF, SCOPE_GATE_REF),
        gates=gates,
        stage3_scope_decision=_scope_decision(),
        stage3_scope_gate=_g0(),
        stage3_scope_decision_ref=DECISION_REF,
        stage3_scope_gate_ref=SCOPE_GATE_REF,
    )


def _build_inputs():
    gates = _gates()
    execution = _execution(gates)
    plan = _plan(execution)
    refs = (TABLE_REF, PLAN_REF, "evidence/metrics.json", DECISION_REF)
    provenance = _provenance(gates, refs)
    evaluation = _evaluation(execution, gates, plan, provenance)
    table = FrozenSourceTable.from_rows(
        name="stage3_formal_quadrature_observations",
        schema_version="stage3-formal-quadrature-observation-table-v1",
        rows=(_row("midpoint"), _row("trapezoid")),
    )
    thresholds = QuadratureThresholds(**_thresholds())  # type: ignore[arg-type]
    # Build the exact unqualified candidate which the live loader must qualify.
    # The formal evaluator and recommendation engine intentionally consume the
    # same complete table, but only the finalizer may publish a selection.
    from param_importance_nlp.experiments.stage3_formal import QuadratureObservation

    observations = tuple(
        QuadratureObservation(
            unit_id=row["unit_id"],
            rule_name=row["rule_name"],
            unique_nodes=row["unique_nodes"],
            normalized_l1_error=row["normalized_l1_error"],
            completeness_absolute_residual=row["completeness_absolute_residual"],
            spearman=row["spearman"],
            topk_overlap=row["topk_overlap"],
            wall_seconds=row["wall_seconds"],
            normalized_l2_error=row["normalized_l2_error"],
            normalized_linf_error=row["normalized_linf_error"],
            completeness_relative_residual=row["completeness_relative_residual"],
            completeness_l1_scaled_residual=row["completeness_l1_scaled_residual"],
            active_spearman=row["active_spearman"],
            cosine_similarity=row["cosine_similarity"],
            sign_consistency=row["sign_consistency"],
            topq_overlap=row["topq_overlap"],
            topq_jaccard=row["topq_jaccard"],
            layer_quality_tv=row["layer_quality_tv"],
            module_quality_tv=row["module_quality_tv"],
            reference_normalized_l1_error=row["reference_normalized_l1_error"],
            strata=row["strata"],
            evidence_refs=tuple(row["evidence_refs"]),
            scope="formal",
            gradient_evaluations=1,
            loss_evaluations=1,
            forward_evaluations=1,
            backward_evaluations=1,
            peak_gpu_memory_bytes=1024,
        )
        for row in (_row("midpoint"), _row("trapezoid"))
    )
    candidate = QuadratureRecommendationEngine().recommend(
        recommendation_id="stage3-finalization-recommendation",
        observations=observations,
        required_unit_ids=("model14-early-update1-probe1",),
        thresholds=thresholds,
        execution=execution,
    )
    return gates, execution, plan, provenance, evaluation, table, candidate


def _finalizer_kwargs():
    gates, execution, plan, provenance, evaluation, table, candidate = _build_inputs()
    return dict(
        finalization_id="stage3-finalization-test",
        frozen_table=table,
        frozen_table_ref=TABLE_REF,
        formal_plan=plan,
        formal_plan_ref=PLAN_REF,
        execution=execution,
        prerequisite_gates=gates,
        gate_evaluation=evaluation,
        evaluation_ref=EVAL_REF,
        provenance=provenance,
        provenance_ref=PROV_REF,
        recommendation=candidate.to_dict(),
        recommendation_ref=RECOMMENDATION_REF,
        checked_at="2026-08-28T00:02:00Z",
        g3_6_ref="evidence/stage3/g3-6.json",
        g3_7_ref="evidence/stage3/g3-7.json",
    )


def test_finalizer_constructs_g36_g37_and_qualified_method_selection() -> None:
    result = Stage3Finalizer().finalize(**_finalizer_kwargs())
    assert result.status == "PASS"
    assert result.formal_eligible is True
    assert result.g3_6_gate is not None and result.g3_6_gate.status is GateStatus.PASS
    assert result.g3_7_gate is not None and result.g3_7_gate.status is GateStatus.PASS
    assert result.recommendation is not None and result.recommendation.status == "QUALIFIED"
    assert result.method_selection is not None
    payload = result.to_dict()
    assert payload["provenance_ref"] == PROV_REF
    assert payload["provenance_hash"] == result.provenance_hash
    assert payload["evaluation_ref"] == EVAL_REF
    assert payload["evaluation_hash"] == result.evaluation_hash
    assert payload["formal_plan_ref"] == PLAN_REF
    assert payload["formal_plan_hash"] == result.formal_plan_hash


def test_finalizer_blocks_incomplete_frozen_table_and_never_selects_partial_rule() -> None:
    kwargs = _finalizer_kwargs()
    table = kwargs["frozen_table"]
    assert isinstance(table, FrozenSourceTable)
    kwargs["frozen_table"] = FrozenSourceTable.from_rows(
        name=table.name,
        schema_version=table.schema_version,
        rows=(thaw_json_value(dict(table.rows[0])),),
    )
    result = Stage3Finalizer().finalize(**kwargs)
    assert result.status == "BLOCKED"
    assert result.formal_eligible is False
    assert result.recommendation is None
    assert any("incomplete" in reason for reason in result.reasons)


def test_finalizer_keeps_g36_pass_but_blocks_g37_when_fallback_is_not_explicit() -> None:
    kwargs = _finalizer_kwargs()
    candidate = dict(kwargs["recommendation"])
    candidate["fallback_rule"] = None
    candidate["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in candidate.items() if key != "artifact_hash"}
    )
    kwargs["recommendation"] = candidate
    result = Stage3Finalizer().finalize(**kwargs)
    assert result.status == "BLOCKED"
    assert result.g3_6_gate is not None and result.g3_6_gate.status is GateStatus.PASS
    assert result.g3_7_gate is not None and result.g3_7_gate.status is GateStatus.BLOCKED
    assert result.recommendation is None


def test_finalizer_reuses_stage308_g36_without_rehashing_it() -> None:
    finalizer = Stage3Finalizer()
    producer_result = finalizer.finalize(**_finalizer_kwargs())
    assert producer_result.g3_6_gate is not None
    kwargs = _finalizer_kwargs()
    kwargs["existing_g3_6_gate"] = producer_result.g3_6_gate
    kwargs["existing_g3_6_ref"] = "evidence/stage3/g3-6.json"
    reused = finalizer.finalize(**kwargs)
    assert reused.status == "PASS"
    assert reused.g3_6_gate is not None
    assert reused.g3_6_gate.artifact_hash == producer_result.g3_6_gate.artifact_hash
    assert reused.g3_6_ref == "evidence/stage3/g3-6.json"
    assert reused.g3_7_gate is not None
    assert reused.g3_7_gate.measured["g3_6_hash"] == producer_result.g3_6_gate.artifact_hash


def test_finalizer_rejects_drifted_existing_g36_measured_binding() -> None:
    producer_result = Stage3Finalizer().finalize(**_finalizer_kwargs())
    assert producer_result.g3_6_gate is not None
    drifted = replace(
        producer_result.g3_6_gate,
        measured={**dict(producer_result.g3_6_gate.measured), "source_table_hash": "a" * 64},
    )
    kwargs = _finalizer_kwargs()
    kwargs["existing_g3_6_gate"] = drifted
    kwargs["existing_g3_6_ref"] = "evidence/stage3/g3-6.json"
    result = Stage3Finalizer().finalize(**kwargs)
    assert result.status == "BLOCKED"
    assert any("EXISTING_G3_6_MEASURED_DRIFT" in reason for reason in result.reasons)


def test_finalizer_live_reload_rejects_payload_or_authority_drift() -> None:
    finalizer = Stage3Finalizer()
    kwargs = _finalizer_kwargs()
    result = finalizer.finalize(**kwargs)
    assert result.status == "PASS"
    gates, execution, plan, provenance, evaluation, table, _candidate = _build_inputs()
    reloaded = finalizer.reload_live(
        result.to_dict(),
        frozen_table=table,
        formal_plan=plan,
        execution=execution,
        prerequisite_gates=gates,
        gate_evaluation=evaluation,
        provenance=provenance,
    )
    assert reloaded.artifact_hash == result.artifact_hash
    tampered = result.to_dict()
    tampered["formal_plan_hash"] = "a" * 64
    tampered["artifact_hash"] = canonical_json_hash({key: value for key, value in tampered.items() if key != "artifact_hash"})
    with pytest.raises((FormalRunRejected, ValueError)):
        finalizer.reload_live(
            tampered,
            frozen_table=table,
            formal_plan=plan,
            execution=execution,
            prerequisite_gates=gates,
            gate_evaluation=evaluation,
            provenance=provenance,
        )


def test_finalization_wire_loader_roundtrips_qualified_payload_without_self_qualification() -> None:
    from param_importance_nlp.experiments import (
        STAGE3_FINALIZATION_SCHEMA,
        Stage3Finalizer,
    )

    result = Stage3Finalizer().finalize(**_finalizer_kwargs())
    payload = result.to_dict()
    assert payload["schema_version"] == STAGE3_FINALIZATION_SCHEMA
    validated = validate_stage23_artifact(payload)
    assert validated.kind == "stage3-finalization"
    assert validated.artifact_hash == result.artifact_hash

    tampered = dict(payload)
    tampered["method_selection"] = dict(payload["method_selection"])
    tampered["method_selection"]["default_rule"] = "left"  # type: ignore[index]
    tampered["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in tampered.items() if key != "artifact_hash"}
    )
    with pytest.raises((FormalRunRejected, ValueError)):
        validate_stage23_artifact(tampered)
