"""Stage 7--9 默认执行资源、ETL、统计与渲染的 run-ready 验收。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from param_importance_nlp.analysis import (
    BoundSourceTable,
    CompositeFigureArtifact,
    FrozenSourceTable,
    render_composite_figure_set,
)
from param_importance_nlp.cli import _load_mapping
from param_importance_nlp.contracts import ResolvedConfig, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.core import TensorMap, produce_baseline_scores
from param_importance_nlp.experiments import build_default_task_runtime
from param_importance_nlp.experiments.ablation import AblationFactor, AblationMatrix
from param_importance_nlp.experiments.stage789_task_runners import (
    AblationMatrixDeclaration,
    LoadedInputArtifact,
    _rows_from_documents,
)
from param_importance_nlp.runtime import TaskArtifactStore


ROOT = Path(__file__).resolve().parents[1]


def _base(task_id: str) -> ResolvedConfig:
    stage = int(task_id.removeprefix("stage").split(".", 1)[0])
    value = deepcopy(load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json"))
    value["identity"].update({"stage": stage, "task": task_id})
    value["loss"].update({"task_type": "sequence_classification", "weighting": "sample"})
    value["data"].update({"statistical_unit": "sample", "weight_unit": "sample"})
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def _v2(task_id: str, output: str, **sections: object) -> ResolvedConfigV2:
    overrides = {"providers": {"num_labels": 3}, "artifacts": {"output_dir": output}}
    overrides.update(sections)
    return ResolvedConfigV2.resolve(_base(task_id), task_id=task_id, overrides=overrides)


def _hashed(value: dict[str, object]) -> dict[str, object]:
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def test_baseline_producer_exposes_real_methods_and_fail_closed_availability() -> None:
    parameters = TensorMap({"w": torch.tensor([2.0, -1.0], dtype=torch.float64)})
    state = {
        "version": 2,
        "magnitude": {"w": torch.tensor([2.0, 1.0])},
        "data_movement": {"w": torch.tensor([0.4, 0.2])},
        "raw": {"w": torch.tensor([0.5, 0.25])},
        "positive": {"w": torch.tensor([0.3, 0.1])},
        "negative_mass": {"w": torch.tensor([0.05, 0.2])},
        "initial_parameters": {"w": torch.tensor([1.5, -0.7])},
        "last_parameters": {"w": torch.tensor([2.0, -1.0])},
        "has_initial_parameters": True,
    }
    gradients = (
        TensorMap({"w": torch.tensor([1.0, 2.0])}),
        TensorMap({"w": torch.tensor([3.0, 4.0])}),
    )
    updates = (
        TensorMap({"w": torch.tensor([-0.1, -0.2])}),
        TensorMap({"w": torch.tensor([-0.3, -0.4])}),
    )
    produced = produce_baseline_scores(
        parameters,
        state,
        estimator_name="u",
        per_unit_gradients=gradients,
        si_gradients=gradients,
        si_data_updates=updates,
    )

    assert set(produced.scores) == {
        "magnitude",
        "movement",
        "raw",
        "u",
        "empirical_fisher",
        "si",
    }
    assert produced.unavailable == {"double": "training_estimator_not_double"}
    assert torch.allclose(produced.scores["u"]["w"], torch.tensor([0.25, -0.1], dtype=torch.float64))

    produced_double = produce_baseline_scores(
        parameters,
        state,
        estimator_name="double",
        per_unit_gradients=gradients,
    )
    assert "double" in produced_double.scores
    assert produced_double.unavailable["u"] == "training_estimator_not_u"


def test_default_runtime_builds_pruning_resources_from_training_output(tmp_path: Path) -> None:
    training = _v2(
        "stage0.06_single_gpu_smoke",
        "outputs/training",
        training={"max_steps": 2},
        evaluation={
            "enabled": True,
            "split": "validation",
            "every_steps": 1,
            "batch_size": 2,
            "max_batches": 1,
            "metrics": ["loss", "accuracy"],
        },
        checkpoint_schedule={
            "segments": [{"start_step": 0, "end_step": None, "every_steps": 1}]
        },
    )
    runtime = build_default_task_runtime(tmp_path)
    training_result = runtime.execute(training)
    assert training_result.status.value == "PASS", training_result.message
    training_ref = training_result.artifact_refs["training_smoke_result"]

    base = _base("stage7.matrix").to_dict()
    base["pruning"].update(
        {"enabled": True, "strategy": "highest", "ratios": [0.25], "random_repetitions": 1}
    )
    matrix = ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(base),
        task_id="stage7.matrix",
        overrides={
            "training": {"max_steps": 1},
            "providers": {"num_labels": 3},
            "evaluation": {
                "enabled": True,
                "split": "validation",
                "every_steps": 1,
                "batch_size": 2,
                "max_batches": 1,
                "metrics": ["loss", "accuracy"],
            },
            "orchestration": {"input_result_refs": [training_ref]},
            "artifacts": {"output_dir": "outputs/stage7-matrix"},
        },
    )
    matrix_result = runtime.execute(matrix)

    assert matrix_result.status.value == "PASS", matrix_result.message
    published = TaskArtifactStore(tmp_path, "outputs/stage7-matrix").load_commit(
        matrix_result.artifact_refs["pruning_matrix"]
    )
    payload = load_canonical_json(tmp_path / published.object_ref)["payload"]
    methods = {item["method"] for item in payload["sources"]}
    assert {
        "magnitude",
        "movement",
        "raw",
        "u",
        "empirical_fisher",
        "si",
    }.issubset(methods)
    assert all(item["metadata"]["tensor_bundle_ref"] for item in payload["sources"])


def test_default_ablation_executor_consumes_manifest_without_python_injection(tmp_path: Path) -> None:
    matrix = AblationMatrix.compile(
        matrix_id="run-ready-ablation",
        base_config={"experiment": {"variant": "baseline"}},
        factors=(
            AblationFactor(
                name="variant",
                config_path=("experiment", "variant"),
                baseline_value="baseline",
                alternatives=("candidate",),
            ),
        ),
        base_seed=7,
    )
    write_canonical_json(tmp_path / "inputs/matrix.json", matrix.to_dict())
    checkpoint = _hashed(
        {
            "schema_version": "test-checkpoint-resource-v1",
            "formal_eligible": False,
            "checkpoint_id": "tiny-checkpoint",
        }
    )
    write_canonical_json(tmp_path / "inputs/checkpoint.json", checkpoint)
    evidence = {
        "schema_version": "ablation-cell-evidence-manifest-v1",
        "matrix_hash": matrix.digest,
        "checkpoint_artifact_hash": checkpoint["artifact_hash"],
        "scope": "local_fixture",
        "formal_eligible": False,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "config_hash": cell.config_hash,
                "metrics": {"quality": 1.0 if cell.parent_cell_id is None else 0.8},
                "metric_directions": {"quality": "higher_is_better"},
                "evidence_hash": canonical_json_hash({"cell": cell.cell_id}),
                "result_ref": f"results/{cell.cell_id}.json",
            }
            for cell in matrix.cells
        ],
    }
    _hashed(evidence)
    write_canonical_json(tmp_path / "inputs/evidence.json", evidence)
    config = _v2(
        "stage8.execute",
        "outputs/ablation",
        orchestration={
            "matrix_ref": "inputs/matrix.json",
            "input_result_refs": ["inputs/checkpoint.json", "inputs/evidence.json"],
            "paired_design": {
                "enabled": True,
                "design": "matched_seeds",
                "mapping_ref": "inputs/evidence.json",
                "budget_unit": "samples",
            },
        },
    )
    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS", result.message
    assert tuple(result.artifact_refs) == config.task_definition.artifact_kinds


def test_cross_stage_etl_adapts_training_stage2_stage3_and_route_envelopes() -> None:
    training = _hashed(
        {
            "schema_version": "training-run-result-v1",
            "run_id": "train-a",
            "status": "COMPLETE",
            "state": {},
            "registry_hash": "a" * 64,
            "optimizer_contract_hash": "b" * 64,
            "records": [
                {"attempt_index": 1, "global_step": 1, "status": "COMMITTED", "mean_loss": 0.7}
            ],
            "checkpoint_ids": ["ckpt-1"],
            "importance_snapshot": None,
            "importance_trajectory": None,
        }
    )
    stage2 = _hashed(
        {
            "schema_version": "stage2-paired-wave-summary-v1",
            "method_statistics": {"u": {"mse": 0.1, "mae": 0.2}},
        }
    )
    stage3 = _hashed(
        {
            "schema_version": "path-integral-result-v1",
            "path_identity_hash": "c" * 64,
            "rule": {"name": "simpson"},
            "loss_drop": 0.3,
            "completeness_absolute_residual": 0.01,
            "completeness_relative_residual": 0.02,
            "completeness_l1_scaled_residual": 0.03,
            "unique_gradient_evaluations": 5,
        }
    )
    route = {
        "schema_version": "stage456-route-execution-v1",
        "training_metrics": [
            {"route_id": "r", "phase_id": "p", "phase_type": "finetune", "mean_training_loss": 0.4, "final_training_loss": 0.3}
        ],
        "evaluation_metrics": [
            {"route_id": "r", "phase_id": "p", "phase_type": "finetune", "metrics": {"accuracy": 0.9}}
        ],
    }
    documents = tuple(
        LoadedInputArtifact(
            ref=f"inputs/{index}.json",
            value=value,
            artifact_hash=canonical_json_hash({"document": index}),
            formal_eligible=False,
        )
        for index, value in enumerate((training, stage2, stage3, route))
    )
    rows = _rows_from_documents(documents, scope="local_fixture")

    assert {item.stage for item in rows} == {1, 2, 3, 6}
    assert {item.adapter_id for item in rows} == {
        "stage9.training_run_result.v1",
        "stage9.stage2_estimator_summary.v1",
        "stage9.stage3_path_integral.v1",
        "stage9.stage456_route_execution.v1",
    }
    assert all(item.source_artifact_hash in {doc.artifact_hash for doc in documents} for item in rows)


def test_stage9_statistics_runs_paired_holm_and_ci(tmp_path: Path) -> None:
    config = _v2("stage9.statistics", "outputs/statistics")
    result = build_default_task_runtime(tmp_path).execute(config)
    assert result.status.value == "PASS", result.message
    commit = TaskArtifactStore(tmp_path, "outputs/statistics").load_commit(
        result.artifact_refs["statistics_table"]
    )
    payload = load_canonical_json(tmp_path / commit.object_ref)["payload"]
    paired = payload["paired_statistics_holm"]
    assert paired["table"]["schema_version"] == "paired-statistics-holm-v1"
    assert paired["table"]["rows"][0]["p_value_holm"] is not None
    assert paired["table"]["rows"][0]["ci_lower"] is not None


def test_composite_heatmap_errorbar_facet_render_is_deterministic(tmp_path: Path) -> None:
    table = FrozenSourceTable.from_rows(
        name="figure-source",
        schema_version="figure-source-v1",
        rows=(
            {"condition": "baseline", "replicate": 0, "ratio": 0.0, "value": 1.0},
            {"condition": "baseline", "replicate": 1, "ratio": 0.0, "value": 0.9},
            {"condition": "candidate", "replicate": 0, "ratio": 0.5, "value": 0.8},
            {"condition": "candidate", "replicate": 1, "ratio": 0.5, "value": 0.7},
        ),
    )
    source = BoundSourceTable.create(
        table,
        role="test",
        scope="local_fixture",
        parent_artifact_hashes=("d" * 64,),
        derivation_id="tests.figure.v1",
    )
    first = render_composite_figure_set(source, tmp_path / "first")
    second = render_composite_figure_set(source, tmp_path / "second")

    assert first.artifact_hash == second.artifact_hash
    assert first.producer_source_hash == second.producer_source_hash
    assert [item["content_sha256"] for item in first.files] == [
        item["content_sha256"] for item in second.files
    ]
    assert {item["kind"] for item in first.files} == {"heatmap", "errorbar", "facet"}
    assert CompositeFigureArtifact.from_mapping(first.to_dict()).artifact_hash == first.artifact_hash


def test_default_stage9_table_runner_binds_producer_source_hash(tmp_path: Path) -> None:
    config = _v2("stage9.tables", "outputs/tables")
    result = build_default_task_runtime(tmp_path).execute(config)
    assert result.status.value == "PASS", result.message
    commit = TaskArtifactStore(tmp_path, "outputs/tables").load_commit(
        result.artifact_refs["table_artifacts"]
    )
    payload = load_canonical_json(tmp_path / commit.object_ref)["payload"]
    assert len(payload["producer_source_hash"]) == 64
    assert payload["artifact"]["schema_version"] == "analysis-table-artifact-v1"


def test_default_stage9_chart_runner_publishes_rendered_figure_set(tmp_path: Path) -> None:
    config = _v2("stage9.charts", "outputs/charts")
    result = build_default_task_runtime(tmp_path).execute(config)
    assert result.status.value == "PASS", result.message
    commit = TaskArtifactStore(tmp_path, "outputs/charts").load_commit(
        result.artifact_refs["chart_artifacts"]
    )
    payload = load_canonical_json(tmp_path / commit.object_ref)["payload"]
    composite = payload["composite_figure_artifact"]
    assert composite["schema_version"] == "analysis-composite-figure-artifact-v1"
    assert {item["kind"] for item in composite["files"]} == {
        "heatmap",
        "errorbar",
        "facet",
    }
    assert len(payload["file_refs"]) == 3
    assert all((tmp_path / reference).is_file() for reference in payload["file_refs"])


def test_ablation_declaration_create_removes_manual_hash_step() -> None:
    declaration = AblationMatrixDeclaration.create(
        matrix_id="builder-test",
        base_config={"method": "u"},
        factors=(
            AblationFactor(
                name="method",
                config_path=("method",),
                baseline_value="u",
                alternatives=("double",),
            ),
        ),
        base_seed=42,
        seed_namespace="tests.builder.v1",
        scope="formal",
    )
    assert declaration.formal_eligible is True
    assert AblationMatrixDeclaration.from_mapping(declaration.to_dict()).artifact_hash == declaration.artifact_hash


def test_stage7_to_9_formal_templates_are_strictly_resolvable() -> None:
    base_path = ROOT / "configs/local-fixtures/resolved-config-v1.json"
    cases = (
        ("stage7.matrix", "formal-stage7.yaml", "stage7-matrix-formal.yaml"),
        ("stage7.evaluate", "formal-stage7.yaml", "stage7-evaluate-formal.yaml"),
        ("stage7.reduce", "formal-stage7.yaml", "stage7-reduce-formal.yaml"),
        ("stage7.report", "formal-stage7.yaml", "stage7-report-formal.yaml"),
        ("stage8.freeze", "formal-stage8.yaml", "stage8-freeze-formal.yaml"),
        ("stage8.execute", "formal-stage8.yaml", "stage8-execute-formal.yaml"),
        ("stage8.reduce", "formal-stage8.yaml", "stage8-reduce-formal.yaml"),
        ("stage8.recommend", "formal-stage8.yaml", "stage8-recommend-formal.yaml"),
        ("stage8.report", "formal-stage8.yaml", "stage8-report-formal.yaml"),
        ("stage9.ingest", "formal-stage9.yaml", "stage9-ingest-formal.yaml"),
        ("stage9.statistics", "formal-stage9.yaml", "stage9-statistics-formal.yaml"),
        ("stage9.tables", "formal-stage9.yaml", "stage9-tables-formal.yaml"),
        ("stage9.charts", "formal-stage9.yaml", "stage9-charts-formal.yaml"),
        ("stage9.report", "formal-stage9.yaml", "stage9-report-formal.yaml"),
        ("stage9.bundle", "formal-stage9.yaml", "stage9-bundle-formal.yaml"),
        ("stage9.replay", "formal-stage9.yaml", "stage9-replay-formal.yaml"),
    )
    for task_id, layer, override in cases:
        base = ResolvedConfig.resolve(
            _load_mapping(base_path),
            _load_mapping(ROOT / "configs/run-ready/layers/formal-pythia160m-stage4.yaml"),
            _load_mapping(ROOT / "configs/run-ready/layers" / layer),
        )
        config = ResolvedConfigV2.resolve(
            base,
            task_id=task_id,
            overrides=_load_mapping(ROOT / "configs/run-ready/v2" / override),
        )
        assert config.run_intent == "formal"
        assert config.task_id == task_id
