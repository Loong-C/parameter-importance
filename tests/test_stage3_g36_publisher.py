from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import param_importance_nlp.experiments.stage3_g36_publisher as g36_publisher
from param_importance_nlp.analysis.report import FrozenSourceTable
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    ProvenanceRecord,
    ProvenanceStatus,
    RunIdentity,
    canonical_json_hash,
    derive_experiment_id,
    validate_stage23_artifact,
)
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.contracts.stage3_scope import validate_stage3_scope_decision
from param_importance_nlp.experiments.stage3_g36_publisher import (
    Stage3G36Publication,
    Stage3G36Publisher,
)
from param_importance_nlp.experiments.stage3_raw_storage import (
    RAW_AGGREGATE_SCHEMA,
    VECTOR_DERIVATION_HASH,
    VECTOR_DERIVATION_SCHEMA,
)
from param_importance_nlp.experiments.stage3_protocol import DEFAULT_CANDIDATE_RULES
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


CONFIG_HASH = "c" * 64
DECISION_REF = "artifacts/scope/commits/scope_authority.json"
SCOPE_GATE_REF = "artifacts/scope/commits/gate_record.json"
TABLE_REF = "artifacts/table/commits/frozen_source_table.json"
EXECUTION_REF = "artifacts/execution/commits/formal_execution_evidence.json"
PLAN_REF = "artifacts/plan/commits/formal_plan.json"
PROVENANCE_REF = "artifacts/provenance/commits/provenance_record.json"
ROW_REF = "evidence/stage3/formal-row.json"


def _scope_decision() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "stage3-g30-user-scope-decision-v1",
        "decision_id": "stage3-g36-publisher-test",
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
        evidence_refs=(DECISION_REF,),
    )


def _row() -> dict[str, object]:
    return {
        "unit_id": "model14-early-update1-probe1",
        "rule_name": "midpoint",
        "unique_nodes": 4,
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
        "top_q": {
            "0.001": {"overlap": 1.0, "jaccard": 1.0},
            "0.01": {"overlap": 1.0, "jaccard": 1.0},
            "0.05": {"overlap": 1.0, "jaccard": 1.0},
        },
        "layer_quality_tv": 0.001,
        "module_quality_tv": 0.001,
        "reference_normalized_l1_error": 0.0001,
        "strata": {"model": "14M", "stage": "early", "update": "u1", "probe": "p1"},
        "worst_case": False,
        "evidence_refs": [ROW_REF],
        "scope": "formal",
    }


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


def _provenance(refs: tuple[str, ...]) -> ProvenanceRecord:
    identity = RunIdentity.create(
        experiment_id=derive_experiment_id(
            stage=3,
            task="stage3-g36-publisher",
            model_identity="model-v1",
            route="formal",
            master_seed=1,
            config_hash=CONFIG_HASH,
        ),
        created_at=datetime(2026, 8, 28, tzinfo=timezone.utc),
        collision_code="0123abcd",
    )
    return ProvenanceRecord(
        identity=identity,
        config_hash=CONFIG_HASH,
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
        artifact_refs=refs,
    )


def _publish_inputs(
    root: Path,
    *,
    self_bind_provenance: bool = False,
    matrix_plan: bool = False,
) -> dict[str, str]:
    def store(name: str) -> TaskArtifactStore:
        return TaskArtifactStore(root, f"artifacts/{name}")

    decision = _scope_decision()
    validate_stage3_scope_decision(decision)
    decision_commit = store("scope").publish(
        task_id="stage3.01_prerequisites_and_scope",
        artifact_kind="scope_authority",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=decision,
        formal_eligible=True,
    )
    assert decision_commit.commit_ref == DECISION_REF
    scope_gate = _scope_gate()
    scope_gate_commit = store("scope").publish(
        task_id="stage3.01_prerequisites_and_scope",
        artifact_kind="gate_record",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=scope_gate.to_dict(),
        formal_eligible=True,
        source_refs=(DECISION_REF,),
    )
    assert scope_gate_commit.commit_ref == SCOPE_GATE_REF

    prereq = [scope_gate]
    for index in range(1, 6):
        evidence = (PLAN_REF,) if index == 5 else (f"evidence/stage3/g3-{index}.json",)
        prereq.append(
            GateRecord(
                gate_id=f"stage3.G3-{index}",
                stage=3,
                status=GateStatus.PASS,
                checked_at="2026-08-28T00:00:00Z",
                evidence_refs=evidence,
            )
        )
    execution = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="f" * 64,
        asset_manifest_hashes=("e" * 64,),
        prerequisite_gates=tuple(prereq),
    )
    execution_commit = store("execution").publish(
        task_id="stage3.08_g3_6_publisher",
        artifact_kind="formal_execution_evidence",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=execution.to_dict(),
        formal_eligible=True,
    )
    assert execution_commit.commit_ref == EXECUTION_REF

    plan: dict[str, object] = {
        "schema_version": "stage3-formal-pilot-plan-v1",
        "plan_id": "stage3-g36-publisher-plan",
        "scope": "formal",
        "candidate_rules": ["midpoint"],
        "required_unit_ids": ["model14-early-update1-probe1"],
        "unit_strata": {
            "model14-early-update1-probe1": {
                "model": "14M", "stage": "early", "update": "u1", "probe": "p1"
            }
        },
        "thresholds": _thresholds(),
        "execution_evidence_hash": execution.artifact_hash,
        "formal_eligible": True,
    }
    if matrix_plan:
        plan.update(
            {
                "plan_kind": "matrix",
                "production_unit_index_scope": "formal",
                "production_unit_index_ref": "plans/stage3/production-unit-index.json",
                "production_unit_index_hash": "2" * 64,
            }
        )
    plan["artifact_hash"] = canonical_json_hash(plan)
    plan_commit = store("plan").publish(
        task_id="stage3.07_formal_experiment_matrix",
        artifact_kind="formal_plan",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=plan,
        formal_eligible=True,
    )
    assert plan_commit.commit_ref == PLAN_REF

    table = FrozenSourceTable.from_rows(
        name="stage3_formal_quadrature_observations",
        schema_version="stage3-formal-quadrature-observation-table-v1",
        rows=[_row()],
    )
    table_commit = store("table").publish(
        task_id="stage3.08_g3_6_publisher",
        artifact_kind="frozen_source_table",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=table.to_dict(),
        formal_eligible=True,
        source_refs=(ROW_REF,),
    )
    assert table_commit.commit_ref == TABLE_REF

    evaluator_sources = (
        TABLE_REF,
        EXECUTION_REF,
        PLAN_REF,
        DECISION_REF,
        SCOPE_GATE_REF,
        "evidence/stage3/g3-1.json",
        "evidence/stage3/g3-2.json",
        "evidence/stage3/g3-3.json",
        "evidence/stage3/g3-4.json",
        ROW_REF,
    )
    provenance_refs = evaluator_sources + ((PROVENANCE_REF,) if self_bind_provenance else ())
    provenance = _provenance(provenance_refs)
    provenance_commit = store("provenance").publish(
        task_id="stage3.08_g3_6_publisher",
        artifact_kind="provenance_record",
        config_hash=CONFIG_HASH,
        run_intent="formal",
        payload=provenance.to_dict(),
        formal_eligible=True,
    )
    assert provenance_commit.commit_ref == PROVENANCE_REF
    return {
        "frozen_source_table_ref": TABLE_REF,
        "execution_evidence_ref": EXECUTION_REF,
        "formal_plan_ref": PLAN_REF,
        "provenance_ref": PROVENANCE_REF,
        "stage3_scope_decision_ref": DECISION_REF,
        "stage3_scope_gate_ref": SCOPE_GATE_REF,
    }


def test_stage3_g36_publisher_loads_commits_evaluates_and_publishes_acyclic_gate(tmp_path: Path) -> None:
    refs = _publish_inputs(tmp_path)
    result = Stage3G36Publisher().publish(
        workspace_root=tmp_path,
        output_dir="artifacts/publisher",
        config_hash=CONFIG_HASH,
        **refs,
    )
    assert result.status == "PASS"
    assert result.formal_eligible is True
    assert result.gate_evaluation.status == "PASS"
    assert result.g3_6_gate.status is GateStatus.PASS
    assert result.provenance_ref not in result.gate_evaluation.source_artifact_refs
    assert result.provenance_ref not in result.source_artifact_refs[: len(result.gate_evaluation.source_artifact_refs)]
    assert result.g3_6_gate.evidence_refs == (
        TABLE_REF,
        result.evaluation_ref,
        PROVENANCE_REF,
        PLAN_REF,
    )
    assert Stage3G36Publication.from_mapping(result.to_dict()).artifact_hash == result.artifact_hash
    assert validate_stage23_artifact(result.to_dict()).artifact_hash == result.artifact_hash
    receipt = load_committed_task_artifact(
        tmp_path,
        "artifacts/publisher/commits/g36_publication.json",
        require_formal=True,
    )
    assert receipt.payload == result.to_dict()


def test_stage3_g36_publisher_rejects_cross_config_input_mix(tmp_path: Path) -> None:
    refs = _publish_inputs(tmp_path)
    with pytest.raises(FormalRunRejected, match="INPUT_CONFIG_HASH_MISMATCH"):
        Stage3G36Publisher().publish(
            workspace_root=tmp_path,
            output_dir="artifacts/publisher",
            config_hash="d" * 64,
            **refs,
        )


def test_stage3_g36_publisher_rejects_provenance_self_binding(tmp_path: Path) -> None:
    refs = _publish_inputs(tmp_path, self_bind_provenance=True)
    with pytest.raises(FormalRunRejected, match="PROVENANCE_SELF_BINDING"):
        Stage3G36Publisher().publish(
            workspace_root=tmp_path,
            output_dir="artifacts/publisher",
            config_hash=CONFIG_HASH,
            **refs,
        )


def test_stage3_g36_publisher_requires_streaming_coverage_for_matrix_plan(tmp_path: Path) -> None:
    refs = _publish_inputs(tmp_path, matrix_plan=True)
    with pytest.raises(FormalRunRejected, match="STREAMING_COVERAGE_REQUIRED"):
        Stage3G36Publisher().publish(
            workspace_root=tmp_path,
            output_dir="artifacts/publisher",
            config_hash=CONFIG_HASH,
            **refs,
        )


def test_raw_streaming_verifier_returns_metadata_without_retaining_tensor_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G3-6 must not accumulate the matrix's TensorBundle states in memory."""

    units = tuple(f"unit-{index:03d}" for index in range(99))
    rules = tuple(DEFAULT_CANDIDATE_RULES)
    execution_hash = "a" * 64
    binding_hash = "b" * 64
    entries: dict[str, dict[str, str]] = {}
    for index, unit_id in enumerate(units):
        entries[unit_id] = {
            "shard_ref": f"raw/shards/{unit_id}.json",
            "shard_hash": f"{index + 1:064x}",
            "bundle_ref": f"raw/bundles/{unit_id}.bundle",
            "bundle_manifest_hash": f"{index + 101:064x}",
            "path_identity_hash": f"{index + 201:064x}",
            "reference_artifact_hash": f"{index + 301:064x}",
            "reference_identity_hash": f"{index + 401:064x}",
        }
    aggregate: dict[str, object] = {
        "schema_version": RAW_AGGREGATE_SCHEMA,
        "execution_evidence_hash": execution_hash,
        "reference_binding_hash": binding_hash,
        "required_unit_ids": list(units),
        "candidate_rule_names": sorted(rules),
        "vector_derivation_schema": VECTOR_DERIVATION_SCHEMA,
        "vector_derivation_contract_hash": VECTOR_DERIVATION_HASH,
        "complete_unit_ids": list(units),
        "missing_unit_ids": [],
        "unit_shards": entries,
    }
    aggregate["artifact_hash"] = canonical_json_hash(aggregate)
    aggregate_path = tmp_path / "raw" / "aggregate.json"
    aggregate_path.parent.mkdir(parents=True)
    write_canonical_json(aggregate_path, aggregate)

    calls = 0

    def fake_load_shard(*, root: Path, shard_ref: object, expected: object):
        nonlocal calls
        calls += 1
        assert root == tmp_path
        assert isinstance(expected, dict)
        unit_id = str(expected["unit_id"])
        entry = entries[unit_id]
        # These stand in for the large state/bundle objects.  The verifier
        # must release them per iteration and return only aggregate metadata.
        shard = {
            "artifact_hash": entry["shard_hash"],
            "bundle_ref": entry["bundle_ref"],
            "bundle_manifest_hash": entry["bundle_manifest_hash"],
            "path_identity_hash": entry["path_identity_hash"],
            "reference_artifact_hash": entry["reference_artifact_hash"],
            "reference_identity_hash": entry["reference_identity_hash"],
        }
        return shard, {"large": bytearray(1024)}, object()

    monkeypatch.setattr(g36_publisher, "_load_raw_shard", fake_load_shard)
    raw, metadata = g36_publisher._load_raw_aggregate_for_streaming(
        tmp_path,
        "raw/aggregate.json",
        declared_hash=aggregate["artifact_hash"],
        expected_units=units,
        expected_rules=rules,
        expected_execution_hash=execution_hash,
        expected_binding_hash=binding_hash,
    )

    assert calls == len(units)
    assert raw is not None
    assert metadata == entries
    assert all(set(value) == set(entries[units[0]]) for value in metadata.values())
