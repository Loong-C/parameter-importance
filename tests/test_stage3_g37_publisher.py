"""Focused contract tests for the independent G3-7 publisher.

The more involved Stage3.08 fixture is intentionally reused only as input
construction.  The tests exercise the new downstream boundary: formal commit
reload, deterministic config binding, a real BLOCKED G3-7 for a missing
fallback, and row-integrity rejection.
"""

from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from param_importance_nlp.analysis.report import FrozenSourceTable
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.immutable import thaw_json_value
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.experiments.stage3_formal import (
    QuadratureObservation,
    QuadratureRecommendationEngine,
    QuadratureThresholds,
)
from param_importance_nlp.experiments.stage3_g36_publisher import Stage3G36Publisher
from param_importance_nlp.experiments.stage3_g37_publisher import (
    Stage3G37Publication,
    Stage3G37Publisher,
)
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


CONFIG_HASH = "c" * 64
STAGE309_CONFIG_HASH = "d" * 64


def _g36_test_module():
    path = Path(__file__).with_name("test_stage3_g36_publisher.py")
    spec = importlib.util.spec_from_file_location("stage3_g36_test_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load Stage3.08 fixture helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(
    tmp_path: Path,
    *,
    upstream_config_hash: str = CONFIG_HASH,
) -> tuple[dict[str, str], object, object]:
    helper = _g36_test_module()
    refs = helper._publish_inputs(
        tmp_path,
        upstream_config_hash=upstream_config_hash,
    )
    g36 = Stage3G36Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g36",
        config_hash=CONFIG_HASH,
        **refs,
    )
    source = FrozenSourceTable.from_mapping(
        dict(load_committed_task_artifact(tmp_path, refs["frozen_source_table_ref"], require_formal=True).payload)
    )
    input_store = TaskArtifactStore(tmp_path, "artifacts/g37-input")
    cost = input_store.publish(
        task_id="stage3.09_g3_7_publisher",
        artifact_kind="cost_accuracy_table",
        config_hash=STAGE309_CONFIG_HASH,
        run_intent="formal",
        payload=source.to_dict(),
        formal_eligible=True,
        source_refs=(refs["frozen_source_table_ref"],),
    )
    raw = helper._row()
    top_q = raw.pop("top_q")
    raw["topq_overlap"] = {key: value["overlap"] for key, value in top_q.items()}
    raw["topq_jaccard"] = {key: value["jaccard"] for key, value in top_q.items()}
    raw["wall_seconds"] = 4.0
    observation = QuadratureObservation(**raw)
    execution = FormalExecutionEvidence.from_mapping(
        dict(load_committed_task_artifact(tmp_path, refs["execution_evidence_ref"], require_formal=True).payload)
    )
    candidate = QuadratureRecommendationEngine().recommend(
        recommendation_id="g37-candidate",
        observations=(observation,),
        required_unit_ids=("model14-early-update1-probe1",),
        thresholds=QuadratureThresholds(**helper._thresholds()),
        execution=execution,
    )
    candidate_commit = input_store.publish(
        task_id="stage3.09_g3_7_publisher",
        artifact_kind="quadrature_decision",
        config_hash=STAGE309_CONFIG_HASH,
        run_intent="formal",
        payload=candidate.to_dict(),
        formal_eligible=True,
        source_refs=(refs["frozen_source_table_ref"], cost.commit_ref),
    )
    refs = dict(refs)
    refs.update(
        cost_accuracy_table_ref=cost.commit_ref,
        quadrature_decision_ref=candidate_commit.commit_ref,
        evaluation_ref=g36.evaluation_ref,
        g3_6_ref=g36.g3_6_ref,
    )
    return refs, g36, source


def _pass_inputs(tmp_path: Path) -> dict[str, str]:
    """Build the smallest two-rule formal handoff for the positive path."""
    helper = _g36_test_module()
    base = helper._publish_inputs(tmp_path)
    scope_decision_ref = base["stage3_scope_decision_ref"]
    scope_gate_ref = base["stage3_scope_gate_ref"]
    def publish(
        directory: str,
        kind: str,
        payload: object,
        source_refs: tuple[str, ...] = (),
        *,
        config_hash: str = CONFIG_HASH,
    ) -> str:
        return TaskArtifactStore(tmp_path, directory).publish(
            task_id="stage3.09_g3_7_publisher", artifact_kind=kind,
            config_hash=config_hash, run_intent="formal",
            payload=payload.to_dict() if hasattr(payload, "to_dict") else payload,  # type: ignore[union-attr]
            formal_eligible=True, source_refs=source_refs,
        ).commit_ref
    plan_ref = "artifacts/two-plan/commits/formal_plan.json"
    prereq = [helper._scope_gate()]
    for index in range(1, 6):
        prereq.append(
            helper.GateRecord(
                gate_id=f"stage3.G3-{index}", stage=3, status=helper.GateStatus.PASS,
                checked_at="2026-08-28T00:00:00Z",
                evidence_refs=(plan_ref,) if index == 5 else (f"evidence/g3-{index}.json",),
            )
        )
    execution = FormalExecutionEvidence(
        "formal", contract_freeze_hash="f" * 64, asset_manifest_hashes=("e" * 64,),
        prerequisite_gates=tuple(prereq),
    )
    execution_ref = publish("artifacts/two-execution", "formal_execution_evidence", execution)
    units = ["model14-early-update1-probe1"]
    rules = ["midpoint", "trapezoid"]
    strata = {units[0]: {"model": "14M", "stage": "early", "update": "u1", "probe": "p1"}}
    plan = {
        "schema_version": "stage3-formal-pilot-plan-v1", "plan_id": "two-rule-plan",
        "scope": "formal", "candidate_rules": rules, "required_unit_ids": units,
        "unit_strata": strata, "thresholds": helper._thresholds(),
        "execution_evidence_hash": execution.artifact_hash, "formal_eligible": True,
    }
    plan["artifact_hash"] = helper.canonical_json_hash(plan)
    plan_ref = publish("artifacts/two-plan", "formal_plan", plan)
    rows: list[dict[str, object]] = []
    for rule, nodes in (("midpoint", 4), ("trapezoid", 5)):
        row = copy.deepcopy(helper._row())
        row["rule_name"] = rule; row["unique_nodes"] = nodes
        row["strata"] = strata[units[0]]; row["evidence_refs"] = [f"evidence/{rule}.json"]
        rows.append(row)
    table = FrozenSourceTable.from_rows(
        name="stage3_formal_quadrature_observations",
        schema_version="stage3-formal-quadrature-observation-table-v1", rows=rows,
    )
    table_ref = publish("artifacts/two-table", "frozen_source_table", table, ("evidence/midpoint.json", "evidence/trapezoid.json"))
    provenance_refs = (table_ref, execution_ref, plan_ref, scope_decision_ref, scope_gate_ref,
                       "evidence/g3-1.json", "evidence/g3-2.json", "evidence/g3-3.json",
                       "evidence/g3-4.json", "evidence/g3-5.json", "evidence/midpoint.json", "evidence/trapezoid.json")
    provenance_ref = publish("artifacts/two-provenance", "provenance_record", helper._provenance(provenance_refs))
    g36 = Stage3G36Publisher().publish(
        workspace_root=tmp_path, output_dir="artifacts/two-g36", config_hash=CONFIG_HASH,
        frozen_source_table_ref=table_ref, provenance_ref=provenance_ref,
        formal_plan_ref=plan_ref, execution_evidence_ref=execution_ref,
        stage3_scope_decision_ref=scope_decision_ref, stage3_scope_gate_ref=scope_gate_ref,
    )
    observations = []
    for raw in rows:
        item = dict(raw); top_q = item.pop("top_q")
        item["topq_overlap"] = {key: value["overlap"] for key, value in top_q.items()}  # type: ignore[union-attr]
        item["topq_jaccard"] = {key: value["jaccard"] for key, value in top_q.items()}  # type: ignore[union-attr]
        item["wall_seconds"] = float(item["unique_nodes"])
        observations.append(QuadratureObservation(**item))
    candidate = QuadratureRecommendationEngine().recommend(
        recommendation_id="two-rule-candidate", observations=tuple(observations),
        required_unit_ids=tuple(units), thresholds=QuadratureThresholds(**helper._thresholds()), execution=execution,
    )
    candidate_ref = publish(
        "artifacts/two-candidate",
        "quadrature_decision",
        candidate,
        (table_ref, plan_ref, execution_ref),
        config_hash=STAGE309_CONFIG_HASH,
    )
    cost_ref = publish(
        "artifacts/two-cost",
        "cost_accuracy_table",
        table,
        (table_ref,),
        config_hash=STAGE309_CONFIG_HASH,
    )
    return {
        "frozen_source_table_ref": table_ref, "cost_accuracy_table_ref": cost_ref,
        "quadrature_decision_ref": candidate_ref, "formal_plan_ref": plan_ref,
        "execution_evidence_ref": execution_ref, "provenance_ref": provenance_ref,
        "evaluation_ref": g36.evaluation_ref, "g3_6_ref": g36.g3_6_ref,
        "stage3_scope_decision_ref": scope_decision_ref, "stage3_scope_gate_ref": scope_gate_ref,
    }


def test_g37_commits_real_blocked_gate_for_missing_fallback_and_roundtrips(tmp_path: Path) -> None:
    refs, _g36, _source = _inputs(tmp_path)
    result = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37",
        **refs,
    )
    assert result.status == "BLOCKED"
    assert result.formal_eligible is False
    assert result.g3_7_gate.status.value == "BLOCKED"
    assert "EXPLICIT_FALLBACK_MISSING" in result.reasons
    assert result.recommendation_ref is None
    assert result.finalization_ref is None
    assert Stage3G37Publication.from_mapping(result.to_dict()).artifact_hash == result.artifact_hash
    # The Gate commit exists even though qualification correctly stopped.
    assert Path(tmp_path, result.g3_7_ref).exists()


def test_g37_binds_real_upstream_config_identities_without_relabelling(
    tmp_path: Path,
) -> None:
    refs, _g36, _source = _inputs(
        tmp_path,
        upstream_config_hash="e" * 64,
    )
    result = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37",
        **refs,
    )
    assert result.input_config_hashes["execution"] == "e" * 64
    assert result.input_config_hashes["scope_decision"] == "e" * 64
    assert result.input_config_hashes["scope_gate"] == "e" * 64
    assert result.input_config_hashes["frozen_source_table"] == CONFIG_HASH


def test_g37_qualifies_and_publishes_recommendation_finalization_and_receipt(tmp_path: Path) -> None:
    refs = _pass_inputs(tmp_path)
    result = Stage3G37Publisher().publish(workspace_root=tmp_path, output_dir="artifacts/g37", **refs)
    assert result.status == "PASS"
    assert result.formal_eligible is True
    assert result.g3_7_gate.status.value == "PASS"
    assert result.recommendation_ref is not None
    assert result.finalization_ref is not None
    assert result.recommendation is not None and result.recommendation["status"] == "QUALIFIED"
    assert result.finalization is not None and result.finalization["status"] == "PASS"
    assert result.input_config_hashes["frozen_source_table"] == CONFIG_HASH
    assert result.input_config_hashes["cost_accuracy_table"] == STAGE309_CONFIG_HASH
    assert result.quadrature_decision_ref in result.g3_7_gate.evidence_refs
    assert result.g3_7_ref not in result.g3_7_gate.evidence_refs
    schema = json.loads((Path(__file__).parents[1] / "schemas/shared/stage3-g37-publication-v1.json").read_text())
    Draft202012Validator(schema).validate(result.to_dict())


def test_g37_config_binding_ignores_untrusted_caller_hash(tmp_path: Path) -> None:
    refs, _g36, _source = _inputs(tmp_path)
    first = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37-a",
        config_hash="1" * 64,
        **refs,
    )
    second = Stage3G37Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/g37-b",
        config_hash="2" * 64,
        **refs,
    )
    assert first.config_hash == second.config_hash
    assert first.config_hash not in {"1" * 64, "2" * 64}


def test_g37_rejects_cost_row_drift_before_gate_commit(tmp_path: Path) -> None:
    refs, _g36, source = _inputs(tmp_path)
    drifted = dict(thaw_json_value(dict(source.rows[0])))
    drifted["unique_nodes"] = int(drifted["unique_nodes"]) + 1
    bad_cost = FrozenSourceTable.from_rows(
        name=source.name,
        schema_version="stage3-cost-accuracy-table-v1",
        rows=[drifted],
    )
    store = TaskArtifactStore(tmp_path, "artifacts/g37-bad-input")
    bad = store.publish(
        task_id="stage3.09_g3_7_publisher",
        artifact_kind="cost_accuracy_table",
        config_hash=STAGE309_CONFIG_HASH,
        run_intent="formal",
        payload=bad_cost.to_dict(),
        formal_eligible=True,
        source_refs=(refs["frozen_source_table_ref"],),
    )
    refs["cost_accuracy_table_ref"] = bad.commit_ref
    with pytest.raises(FormalRunRejected, match="COST_FROZEN_ROW_MISMATCH"):
        Stage3G37Publisher().publish(workspace_root=tmp_path, output_dir="artifacts/g37", **refs)


def test_g37_rejects_stage309_cost_candidate_config_drift(tmp_path: Path) -> None:
    refs, _g36, source = _inputs(tmp_path)
    drifted = TaskArtifactStore(tmp_path, "artifacts/g37-config-drift").publish(
        task_id="stage3.09_cost_and_method_selection",
        artifact_kind="cost_accuracy_table",
        config_hash="e" * 64,
        run_intent="formal",
        payload=source.to_dict(),
        formal_eligible=True,
        source_refs=(refs["frozen_source_table_ref"],),
    )
    refs["cost_accuracy_table_ref"] = drifted.commit_ref
    with pytest.raises(FormalRunRejected, match="STAGE309_CONFIG_HASH_MISMATCH"):
        Stage3G37Publisher().publish(
            workspace_root=tmp_path,
            output_dir="artifacts/g37",
            **refs,
        )
