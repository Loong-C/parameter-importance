"""Stage 2–9 公共 artifact 的严格 wire 边界。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from param_importance_nlp.analysis import (
    AnalysisReport,
    AnalysisReportBuilder,
    ChartArtifact,
    ChartSpec,
    FrozenSourceTable,
)
from param_importance_nlp.analysis.metrics import MetricResult
from param_importance_nlp.cli import _validate_known_artifact
from param_importance_nlp.contracts import (
    canonical_json_hash,
    validate_path_integral_result_artifact,
    validate_path_spec_artifact,
    validate_reference_result_artifact,
)
from param_importance_nlp.core import PruningPlan, trapezoid_rule
from param_importance_nlp.experiments import (
    AblationFactor,
    AblationMatrix,
    EstimatorDecision,
    QuadratureDecision,
    SamplingPlan,
    SamplingUniverse,
    TrainingPhaseSpec,
    TrainingRouteSpec,
    build_fixture_quadrature_decision,
)


def _bind(payload: dict[str, object]) -> dict[str, object]:
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def test_all_public_artifact_schemas_are_strict_json_objects() -> None:
    schema_root = Path("schemas/shared")
    names = {
        "sampling-plan-v1.json",
        "reference-result-v1.json",
        "estimator-decision-v1.json",
        "quadrature-decision-v1.json",
        "quadrature-rule-v1.json",
        "path-spec-v1.json",
        "path-integral-result-v1.json",
        "training-route-v1.json",
        "pruning-plan-v1.json",
        "ablation-matrix-v1.json",
        "analysis-report-v1.json",
        "analysis-chart-spec-v1.json",
        "analysis-chart-artifact-v1.json",
    }
    for name in names:
        value = json.loads((schema_root / name).read_text(encoding="utf-8"))
        assert value["$schema"].endswith("2020-12/schema")
        assert value["type"] == "object"
        assert value["additionalProperties"] is False


def test_sampling_plan_roundtrip_and_hash_tamper_rejected() -> None:
    plan = SamplingPlan(
        universe=SamplingUniverse("fixture-universe", ("a", "b", "c")),
        stream_seeds={
            "reference_sizing": 1,
            "reference_A": 2,
            "reference_B": 3,
            "pilot": 4,
            "confirmatory": 5,
        },
    )
    wire = plan.to_dict()
    assert SamplingPlan.from_mapping(wire).digest == plan.digest
    assert _validate_known_artifact(wire) == ("sampling_plan", plan.digest)
    tampered = copy.deepcopy(wire)
    tampered["stream_seeds"]["pilot"] = 99  # type: ignore[index]
    with pytest.raises(ValueError, match="plan_hash"):
        SamplingPlan.from_mapping(tampered)


def test_estimator_decision_loader_rejects_bool_as_integer() -> None:
    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "fixture-decision",
        "selected_estimator": "u",
        "scope": "local_fixture",
        "status": "FIXTURE_ONLY",
        "state": "UNFROZEN",
        "batch_size": 32,
        "microbatch_count": 4,
        "repetitions": 2,
        "gate_id": "stage2.G2.7b",
        "gate_status": "NOT_RUN",
        "artifact_ref": None,
        "metadata": {"formal_eligible": False},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    assert EstimatorDecision.from_mapping(payload).batch_size == 32
    invalid = copy.deepcopy(payload)
    invalid["batch_size"] = True
    invalid["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in invalid.items() if key != "artifact_hash"}
    )
    with pytest.raises(TypeError, match="不能是 bool"):
        EstimatorDecision.from_mapping(invalid)


def test_reference_and_path_manifests_bind_every_field_and_reject_escape() -> None:
    reference = _bind(
        {
            "schema_version": "reference-result-v1",
            "reference_id": "reference-fixture",
            "bias_reference_hash": "1" * 64,
            "cross_reference_hash": "2" * 64,
            "ranking_reference_hash": "3" * 64,
            "sample_count_a": 64,
            "sample_count_b": 64,
            "block_size": 8,
            "registry_hash": "4" * 64,
            "scope": "local_fixture",
            "formal_eligible": False,
            "metadata": {"source": "synthetic"},
            "tensor_bundle_ref": "artifacts/reference/tensors",
            "tensor_bundle_manifest_hash": "5" * 64,
        }
    )
    result = validate_reference_result_artifact(reference)
    assert (result.kind, result.artifact_hash) == (
        "reference_result_manifest",
        reference["artifact_hash"],
    )
    assert _validate_known_artifact(reference)[0] == "reference_result_manifest"
    escaped = copy.deepcopy(reference)
    escaped["tensor_bundle_ref"] = "../outside"
    escaped["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in escaped.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="路径逃逸"):
        validate_reference_result_artifact(escaped)

    path = _bind(
        {
            "schema_version": "path-spec-v1",
            "path_id": "linear-path",
            "probe_id": "probe-1",
            "loss_id": "loss-1",
            "accumulation_dtype": "float64",
            "registry_hash": "6" * 64,
            "parameter_names": ["layer.weight"],
            "parameter_pre_bundle_ref": "artifacts/path/pre",
            "parameter_pre_bundle_manifest_hash": "7" * 64,
            "parameter_post_bundle_ref": "artifacts/path/post",
            "parameter_post_bundle_manifest_hash": "8" * 64,
            "path_identity_hash": "9" * 64,
        }
    )
    assert validate_path_spec_artifact(path).kind == "path_spec_manifest"
    assert _validate_known_artifact(path)[0] == "path_spec_manifest"


def test_path_integral_manifest_validates_rule_and_scalar_contract() -> None:
    wire = _bind(
        {
            "schema_version": "path-integral-result-v1",
            "path_identity_hash": "1" * 64,
            "rule": {
                "name": "trapezoid",
                "version": "1",
                "kind": "single_interval",
                "nodes": [0.0, 1.0],
                "weights": [0.5, 0.5],
                "subintervals": None,
                "exact_polynomial_degree": 1,
                "theoretical_order": 2,
            },
            "contribution_bundle_ref": "artifacts/path/result",
            "contribution_bundle_manifest_hash": "2" * 64,
            "views": ["signed", "positive", "negative_mass", "absolute"],
            "endpoint_loss_pre": 2.0,
            "endpoint_loss_post": 1.0,
            "loss_drop": 1.0,
            "completeness_absolute_residual": 0.0,
            "completeness_relative_residual": 0.0,
            "completeness_l1_scaled_residual": 0.0,
            "node_losses": [2.0, 1.0],
            "unique_gradient_evaluations": 2,
        }
    )
    assert validate_path_integral_result_artifact(wire).kind == "path_integral_result_manifest"
    assert _validate_known_artifact(wire)[0] == "path_integral_result_manifest"
    invalid = copy.deepcopy(wire)
    invalid["rule"]["weights"] = [0.4, 0.4]  # type: ignore[index]
    invalid["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in invalid.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="之和必须为 1"):
        validate_path_integral_result_artifact(invalid)


def test_decision_route_pruning_ablation_and_report_roundtrip() -> None:
    decision = build_fixture_quadrature_decision(
        passing_rules_by_cost=("midpoint", "simpson")
    )
    assert QuadratureDecision.from_mapping(decision.to_dict()) == decision
    assert _validate_known_artifact(decision.to_dict())[0] == "quadrature_decision"
    rule = trapezoid_rule()
    loaded_rule = type(rule).from_mapping(rule.to_dict())
    assert loaded_rule.artifact_hash == rule.artifact_hash
    assert _validate_known_artifact(rule.to_dict())[0] == "quadrature_rule"

    phase = TrainingPhaseSpec(
        phase_id="pretrain",
        phase_type="pretrain",
        base_initialization_id="base-1",
        model_asset_id="model-asset",
        dataset_asset_id="dataset-asset",
        output_checkpoint_id="checkpoint-1",
        checkpoint_frequency_steps=10,
        importance_enabled=False,
    )
    route = TrainingRouteSpec("route-1", (phase,), "local_fixture", None)
    assert TrainingRouteSpec.from_mapping(route.to_dict()).lineage_hash == route.lineage_hash
    assert _validate_known_artifact(route.to_dict())[0] == "training_route"

    pruning = PruningPlan(0.25, "high", score_view="positive")
    assert PruningPlan.from_mapping(pruning.to_dict()) == pruning
    assert _validate_known_artifact(pruning.to_dict())[0] == "pruning_plan"

    matrix = AblationMatrix.compile(
        matrix_id="matrix-1",
        base_config={"optimizer": {"lr": 0.1}},
        factors=(
            AblationFactor("learning_rate", ("optimizer", "lr"), 0.1, (0.01,)),
        ),
        base_seed=7,
    )
    assert AblationMatrix.from_mapping(matrix.to_dict()).digest == matrix.digest
    assert _validate_known_artifact(matrix.to_dict())[0] == "ablation_matrix"

    source = FrozenSourceTable.from_rows(
        name="frozen-source",
        schema_version="source-table-v1",
        rows=({"step": 0, "value": 1.0},),
    )
    builder = AnalysisReportBuilder(report_id="report-1")
    builder.add_source(source)
    builder.add_metric(
        "metric",
        MetricResult(True, 1.0),
        source=source,
        derivation_id="stage9.test_metric.v1",
        input_columns=("value",),
    )
    report = builder.build(metadata={"derived_only": True})
    assert AnalysisReport.from_mapping(report.to_dict()).report_hash == report.report_hash
    assert _validate_known_artifact(report.to_dict())[0] == "analysis_report"

    chart = ChartSpec.from_table(
        source,
        chart_id="chart-1",
        chart_type="line",
        x_column="step",
        y_columns=("value",),
        sort_columns=("step",),
    )
    assert ChartSpec.from_mapping(chart.to_dict()).spec_hash == chart.spec_hash
    assert _validate_known_artifact(chart.to_dict()) == (
        "analysis_chart_spec",
        chart.spec_hash,
    )
    chart_artifact = ChartArtifact.from_spec(chart)
    assert (
        ChartArtifact.from_mapping(chart_artifact.to_dict()).artifact_hash
        == chart_artifact.artifact_hash
    )
    assert _validate_known_artifact(chart_artifact.to_dict()) == (
        "analysis_chart_artifact",
        chart_artifact.artifact_hash,
    )


def test_strict_loaders_reject_unknown_fields() -> None:
    plan = PruningPlan(0.5, "low").to_dict()
    plan["unknown"] = True
    with pytest.raises(Exception, match="字段集合"):
        PruningPlan.from_mapping(plan)
