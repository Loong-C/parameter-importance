"""Stage 7--9 复合 TaskRuntime runner 的聚焦测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from param_importance_nlp.analysis.pipeline import StageArtifactRows
from param_importance_nlp.contracts import (
    ResolvedConfig,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.contracts.task_catalog import RunnerKind
from param_importance_nlp.experiments.stage789_task_runners import (
    Stage789CompositeTaskRunner,
    build_stage789_composite_runners,
    register_stage789_task_runners,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRuntime,
    TaskRuntimeEnvironment,
)


def _base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = deepcopy(load_canonical_json("configs/local-fixtures/resolved-config-v1.json"))
    value["identity"]["stage"] = stage
    value["identity"]["task"] = task_id
    value["loss"]["task_type"] = "sequence_classification"
    value["loss"]["weighting"] = "sample"
    value["data"]["statistical_unit"] = "sample"
    value["data"]["weight_unit"] = "sample"
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def _formal_base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = deepcopy(load_canonical_json("configs/local-fixtures/resolved-config-v1.json"))
    value["identity"]["stage"] = stage
    value["identity"]["task"] = task_id
    value["identity"]["run_intent"] = "formal"
    value["identity"]["formal_eligible"] = True
    value["runtime"]["allow_dirty_worktree"] = False
    value["loss"]["task_type"] = "sequence_classification"
    value["loss"]["weighting"] = "sample"
    value["data"]["statistical_unit"] = "sample"
    value["data"]["weight_unit"] = "sample"
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def _config(tmp_path, task_id: str, *, suffix: str = "run", overrides=None):
    stage = int(task_id.removeprefix("stage").split(".", 1)[0])
    merged = {
        "providers": {"num_labels": 2},
        "artifacts": {"output_dir": f"outputs/{suffix}"},
    }
    for section, value in (overrides or {}).items():
        merged[section] = value
    return ResolvedConfigV2.resolve(
        _base_config(stage=stage, task_id=task_id),
        task_id=task_id,
        overrides=merged,
    )


def _execute_single_kind(tmp_path, config):
    runtime = TaskRuntime()
    runner = Stage789CompositeTaskRunner(
        runner_kind=config.task_definition.runner_kind,
        workspace_root=tmp_path,
    )
    runtime.register(runner)
    return runtime.execute(config)


def test_composite_factory_covers_shared_and_stage9_auxiliary_kinds(tmp_path) -> None:
    runners = build_stage789_composite_runners(tmp_path)
    assert {runner.runner_kind for runner in runners} == {
        RunnerKind.CONTRACT,
        RunnerKind.PRUNING,
        RunnerKind.ABLATION,
        RunnerKind.ANALYSIS,
        RunnerKind.STATISTICS,
        RunnerKind.REPORTING,
        RunnerKind.DELIVERY,
        RunnerKind.TEST_MATRIX,
    }
    runtime = TaskRuntime()
    registered = register_stage789_task_runners(runtime, tmp_path)
    assert tuple(registered) == runners
    assert set(runtime.registered_kinds) == {runner.runner_kind for runner in runners}


def test_stage8_freeze_compiles_declaration_and_publishes_validation(tmp_path) -> None:
    declaration = {
        "schema_version": "ablation-matrix-declaration-v1",
        "matrix_id": "declared-stage8-matrix",
        "base_config": {"method": {"estimator": "u", "clip": False}},
        "factors": [
            {
                "name": "estimator",
                "config_path": ["method", "estimator"],
                "baseline_value": "u",
                "alternatives": ["double", "raw"],
            }
        ],
        "base_seed": 314,
        "seed_namespace": "tests-stage8-freeze-v1",
        "scope": "local_fixture",
        "formal_eligible": False,
    }
    declaration["artifact_hash"] = canonical_json_hash(declaration)
    write_canonical_json(tmp_path / "inputs" / "ablation-declaration.json", declaration)
    config = _config(
        tmp_path,
        "stage8.freeze",
        suffix="stage8-freeze",
        overrides={
            "orchestration": {"matrix_ref": "inputs/ablation-declaration.json"}
        },
    )

    result = _execute_single_kind(tmp_path, config)

    assert result.status.value == "PASS"
    assert tuple(result.artifact_refs) == config.task_definition.artifact_kinds
    store = TaskArtifactStore(tmp_path, "outputs/stage8-freeze")
    matrix_object = store.load_commit(result.artifact_refs["ablation_matrix"])
    validation_object = store.load_commit(
        result.artifact_refs["single_factor_validation"]
    )
    matrix_value = load_canonical_json(tmp_path / matrix_object.object_ref)["payload"]
    validation_value = load_canonical_json(
        tmp_path / validation_object.object_ref
    )["payload"]
    assert matrix_value["matrix_id"] == "declared-stage8-matrix"
    assert len(matrix_value["cells"]) == 3
    assert validation_value["all_valid"] is True
    assert {
        row["expected_leaf_difference_count"] for row in validation_value["cells"]
    } == {0, 1}


def test_stage7_and_stage8_compatibility_tasks_use_real_core_runners(tmp_path) -> None:
    stage7 = _config(tmp_path, "stage7.functional_pruning_validation", suffix="stage7")
    stage8 = _config(tmp_path, "stage8.ablation_and_robustness", suffix="stage8")

    result7 = _execute_single_kind(tmp_path, stage7)
    result8 = _execute_single_kind(tmp_path, stage8)

    assert result7.status.value == "PASS"
    assert result8.status.value == "PASS"
    assert tuple(result7.artifact_refs) == stage7.task_definition.artifact_kinds
    assert tuple(result8.artifact_refs) == stage8.task_definition.artifact_kinds
    value7 = load_canonical_json(
        tmp_path
        / TaskArtifactStore(tmp_path, "outputs/stage7")
        .load_commit(result7.artifact_refs["pruning_results"])
        .object_ref
    )
    value8 = load_canonical_json(
        tmp_path
        / TaskArtifactStore(tmp_path, "outputs/stage8")
        .load_commit(result8.artifact_refs["ablation_results"])
        .object_ref
    )
    assert value7["payload"]["schema_version"] == (
        "stage789-pruning-results-task-payload-v1"
    )
    assert value7["payload"]["study_result"]["schema_version"] == "pruning-study-result-v1"
    assert value8["payload"]["schema_version"] == "ablation-study-result-v1"


def test_stage7_partial_cells_require_explicit_resume_ref(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        "stage7.functional_pruning_validation",
        suffix="stage7-explicit-resume",
    )
    runner = Stage789CompositeTaskRunner(
        runner_kind=RunnerKind.PRUNING,
        workspace_root=tmp_path,
    )
    runtime = TaskRuntime()
    runtime.register(runner)
    first = runtime.execute(config)
    assert first.status.value == "PASS"

    # 模拟所有 cell 已提交、但任务级聚合 commit 发布前进程退出。
    task_commits = tmp_path / "outputs/stage7-explicit-resume/commits"
    for path in task_commits.glob("*.json"):
        path.unlink()
    implicit = runtime.execute(config)
    assert implicit.status.value == "FAIL"
    assert "STAGE789_RESUME_REF_REQUIRED" in implicit.message

    resumed_config = _config(
        tmp_path,
        "stage7.functional_pruning_validation",
        suffix="stage7-explicit-resume",
        overrides={
            "recovery": {
                "resume_ref": "outputs/stage7-explicit-resume/core/pruning/commits"
            }
        },
    )
    assert resumed_config.config_hash == config.config_hash
    resumed = runtime.execute(resumed_config)
    assert resumed.status.value == "PASS"
    assert resumed.to_dict() == first.to_dict()


def test_stage4_pruning_validation_is_dispatched_to_pruning_core(tmp_path) -> None:
    config = _config(tmp_path, "stage4.pruning_validation", suffix="stage4-pruning")
    runner = Stage789CompositeTaskRunner(
        runner_kind=RunnerKind.PRUNING,
        workspace_root=tmp_path,
    )
    assert "stage4.pruning_validation" in runner.handled_task_ids
    runtime = TaskRuntime()
    runtime.register(runner)

    result = runtime.execute(config)

    assert result.status.value == "PASS"
    assert tuple(result.artifact_refs) == config.task_definition.artifact_kinds
    store = TaskArtifactStore(tmp_path, "outputs/stage4-pruning")
    published = store.load_commit(result.artifact_refs["pruning_results"])
    value = load_canonical_json(tmp_path / published.object_ref)
    assert value["payload"]["schema_version"] == (
        "stage789-pruning-results-task-payload-v1"
    )
    assert value["payload"]["stage_rows"]["stage"] == 4
    assert value["payload"]["study_result"]["formal_eligible"] is False


def test_leaf_evaluators_consume_frozen_local_matrix_refs(tmp_path) -> None:
    matrix_config = _config(tmp_path, "stage7.matrix", suffix="stage7-matrix")
    matrix_result = _execute_single_kind(tmp_path, matrix_config)
    assert matrix_result.status.value == "PASS"
    evaluate_config = _config(
        tmp_path,
        "stage7.evaluate",
        suffix="stage7-evaluate",
        overrides={
            "orchestration": {
                "matrix_ref": matrix_result.artifact_refs["pruning_matrix"]
            }
        },
    )
    evaluate_result = _execute_single_kind(tmp_path, evaluate_config)
    assert evaluate_result.status.value == "PASS"
    reduce_config = _config(
        tmp_path,
        "stage7.reduce",
        suffix="stage7-reduce",
        overrides={
            "orchestration": {
                "matrix_ref": matrix_result.artifact_refs["pruning_matrix"],
                "input_result_refs": [
                    evaluate_result.artifact_refs["pruning_evaluation_results"]
                ],
            }
        },
    )
    reduce_result = _execute_single_kind(tmp_path, reduce_config)
    assert reduce_result.status.value == "PASS", reduce_result.message
    auc_object = TaskArtifactStore(tmp_path, "outputs/stage7-reduce").load_commit(
        reduce_result.artifact_refs["damage_auc_table"]
    )
    auc_value = load_canonical_json(tmp_path / auc_object.object_ref)
    assert auc_value["payload"]["schema_version"] == "bound-frozen-source-table-v1"

    # Stage 8 execute 接受独立 canonical matrix；direct matrix 没有 formal scope，
    # 因而只允许本机 fixture 使用。
    from param_importance_nlp.experiments.ablation import AblationFactor, AblationMatrix

    matrix = AblationMatrix.compile(
        matrix_id="test-stage8-local-matrix",
        base_config={
            "optimizer": {"learning_rate": 0.08},
            "training": {"max_steps": 1},
        },
        factors=(
            AblationFactor(
                name="learning_rate",
                config_path=("optimizer", "learning_rate"),
                baseline_value=0.08,
                alternatives=(0.16,),
            ),
        ),
        base_seed=91,
    )
    write_canonical_json(tmp_path / "inputs" / "ablation-matrix.json", matrix.to_dict())
    execute_config = _config(
        tmp_path,
        "stage8.execute",
        suffix="stage8-execute",
        overrides={
            "orchestration": {
                "matrix_ref": "inputs/ablation-matrix.json",
                "paired_design": {
                    "enabled": True,
                    "design": "matched_seeds",
                    "mapping_ref": "inputs/ablation-matrix.json",
                    "budget_unit": "samples",
                },
            }
        },
    )
    execute_result = _execute_single_kind(tmp_path, execute_config)
    assert execute_result.status.value == "PASS"


def test_stage9_ingest_consumes_strict_hash_bound_input_ref(tmp_path) -> None:
    rows = StageArtifactRows.create(
        artifact_id="external-stage8",
        source_schema_version="external-stage8-v1",
        source_artifact_hash="a" * 64,
        stage=8,
        adapter_id="tests.external.v1",
        rows=(
            {"condition": "baseline", "value": 1.0, "replicate": 0},
            {"condition": "candidate", "value": 0.8, "replicate": 0},
        ),
        scope="local_fixture",
        formal_eligible=False,
    )
    write_canonical_json(tmp_path / "inputs" / "rows.json", rows.to_dict())
    config = _config(
        tmp_path,
        "stage9.ingest",
        suffix="ingest",
        overrides={"orchestration": {"input_result_refs": ["inputs/rows.json"]}},
    )

    result = _execute_single_kind(tmp_path, config)

    assert result.status.value == "PASS"
    store = TaskArtifactStore(tmp_path, "outputs/ingest")
    published = store.load_commit(result.artifact_refs["frozen_source_table"])
    value = load_canonical_json(tmp_path / published.object_ref)
    assert value["source_refs"] == ["inputs/rows.json"]
    assert value["payload"]["parent_artifact_hashes"] == [rows.artifact_hash]


def test_stage9_ingest_recursively_rebuilds_stage8_report_lineage(tmp_path) -> None:
    """报告只提供 claim；Stage 9 必须沿 commit source_refs 回到原始冻结行。"""

    rows = StageArtifactRows.create(
        artifact_id="stage8-report-lineage-source",
        source_schema_version="stage8-report-lineage-source-v1",
        source_artifact_hash="c" * 64,
        stage=8,
        adapter_id="tests.stage8.report_lineage.v1",
        rows=(
            {"condition": "baseline", "value": 1.0, "replicate": 0},
            {"condition": "candidate", "value": 0.75, "replicate": 0},
        ),
        scope="local_fixture",
        formal_eligible=False,
    )
    write_canonical_json(tmp_path / "inputs" / "stage8-source-rows.json", rows.to_dict())

    reduce_result = _execute_single_kind(
        tmp_path,
        _config(
            tmp_path,
            "stage8.reduce",
            suffix="stage8-lineage-reduce",
            overrides={
                "orchestration": {
                    "input_result_refs": ["inputs/stage8-source-rows.json"],
                    "matrix_ref": "inputs/stage8-source-rows.json",
                }
            },
        ),
    )
    assert reduce_result.status.value == "PASS", reduce_result.message
    recommend_result = _execute_single_kind(
        tmp_path,
        _config(
            tmp_path,
            "stage8.recommend",
            suffix="stage8-lineage-recommend",
            overrides={
                "orchestration": {
                    "input_result_refs": [
                        reduce_result.artifact_refs["ablation_summary_table"]
                    ],
                    "matrix_ref": reduce_result.artifact_refs[
                        "ablation_summary_table"
                    ],
                }
            },
        ),
    )
    assert recommend_result.status.value == "PASS", recommend_result.message
    report_result = _execute_single_kind(
        tmp_path,
        _config(
            tmp_path,
            "stage8.report",
            suffix="stage8-lineage-report",
            overrides={
                "orchestration": {
                    "input_result_refs": [
                        recommend_result.artifact_refs["configuration_recommendation"]
                    ],
                    "matrix_ref": recommend_result.artifact_refs[
                        "configuration_recommendation"
                    ],
                }
            },
        ),
    )
    assert report_result.status.value == "PASS", report_result.message
    ingest_result = _execute_single_kind(
        tmp_path,
        _config(
            tmp_path,
            "stage9.ingest",
            suffix="stage9-lineage-ingest",
            overrides={
                "orchestration": {
                    "input_result_refs": [report_result.artifact_refs["stage_report"]]
                }
            },
        ),
    )
    assert ingest_result.status.value == "PASS", ingest_result.message
    lineage_commit = TaskArtifactStore(
        tmp_path, "outputs/stage9-lineage-ingest"
    ).load_commit(ingest_result.artifact_refs["source_lineage_manifest"])
    lineage = load_canonical_json(tmp_path / lineage_commit.object_ref)["payload"]
    assert "inputs/stage8-source-rows.json" in lineage["input_refs"]


def test_stage9_composite_and_replay_publish_catalog_exact_outputs(tmp_path) -> None:
    composite = _config(
        tmp_path,
        "stage9.analysis_visualization_reporting",
        suffix="analysis",
    )
    replay = _config(tmp_path, "stage9.replay", suffix="replay")

    composite_result = _execute_single_kind(tmp_path, composite)
    replay_result = _execute_single_kind(tmp_path, replay)

    assert composite_result.status.value == "PASS"
    assert replay_result.status.value == "PASS"
    assert tuple(composite_result.artifact_refs) == composite.task_definition.artifact_kinds
    assert tuple(replay_result.artifact_refs) == replay.task_definition.artifact_kinds
    store = TaskArtifactStore(tmp_path, "outputs/replay")
    comparison = store.load_commit(replay_result.artifact_refs["hash_comparison"])
    value = load_canonical_json(tmp_path / comparison.object_ref)
    assert value["payload"]["artifact_hashes_equal"] is True


def test_stage9_charts_rectangularize_without_dropping_nonpaired_rows(
    tmp_path: Path,
) -> None:
    rows = StageArtifactRows.create(
        artifact_id="nonrectangular-stage7",
        source_schema_version="nonrectangular-stage7-v1",
        source_artifact_hash="b" * 64,
        stage=7,
        adapter_id="tests.nonrectangular.v1",
        rows=(
            {"condition": "a", "replicate": 0, "value": 1.0},
            {"condition": "a", "replicate": 1, "value": 2.0},
            {"condition": "b", "replicate": 0, "value": 3.0},
        ),
        scope="local_fixture",
        formal_eligible=False,
    )
    write_canonical_json(tmp_path / "inputs" / "nonrectangular-rows.json", rows.to_dict())
    config = _config(
        tmp_path,
        "stage9.charts",
        suffix="nonrectangular-charts",
        overrides={
            "orchestration": {
                "input_result_refs": ["inputs/nonrectangular-rows.json"]
            }
        },
    )

    result = _execute_single_kind(tmp_path, config)

    assert result.status.value == "PASS", result.message
    store = TaskArtifactStore(tmp_path, "outputs/nonrectangular-charts")
    spec_ref = store.load_commit(result.artifact_refs["chart_specs"])
    artifact_ref = store.load_commit(result.artifact_refs["chart_artifacts"])
    spec_payload = load_canonical_json(tmp_path / spec_ref.object_ref)["payload"]
    artifact_payload = load_canonical_json(tmp_path / artifact_ref.object_ref)["payload"]
    original = spec_payload["original_frozen_source"]
    derived = spec_payload["visualization_source"]
    assert spec_payload["rectangularization_applied"] is True
    assert derived["parent_artifact_hashes"] == [original["artifact_hash"]]
    assert len(derived["table"]["rows"]) == len(rows.rows)
    assert [item["source_row"]["value"] for item in derived["table"]["rows"]] == [
        1.0,
        2.0,
        3.0,
    ]
    composite = artifact_payload["composite_figure_artifact"]
    assert composite["source_table_hash"] == derived["artifact_hash"]
    assert artifact_payload["original_source_artifact_hash"] == original["artifact_hash"]


def test_shared_runner_kind_without_fallback_blocks_unhandled_earlier_task(tmp_path) -> None:
    config = _config(tmp_path, "stage2.08_statistics_and_robustness", suffix="earlier")
    result = _execute_single_kind(tmp_path, config)

    assert result.status.value == "BLOCKED"
    assert result.blockers[0].code.value == "capability_unavailable"
    assert result.blockers[0].requirement == "fallback:statistics"


def test_formal_analysis_is_blocked_by_uncommitted_preflight_inputs_before_fallback(
    tmp_path,
) -> None:
    task_id = "stage9.ingest"
    config = ResolvedConfigV2.resolve(
        _formal_base_config(stage=9, task_id=task_id),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 2},
            "artifacts": {"output_dir": "outputs/formal-ingest"},
        },
    )
    policy = config.task_definition.formal_eligibility
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(policy.required_capabilities),
        frozen_contract_stages=frozenset(policy.required_contract_stages),
        passed_gate_ids=frozenset(policy.required_gate_ids),
        estimator_decision_ref="evidence/estimator-decision.json",
    )
    runtime = TaskRuntime()
    runtime.register(
        Stage789CompositeTaskRunner(
            runner_kind=RunnerKind.ANALYSIS,
            workspace_root=tmp_path,
        )
    )

    result = runtime.execute(config, environment=environment)

    assert result.status.value == "BLOCKED"
    requirements = {blocker.requirement for blocker in result.blockers}
    assert any(item.startswith("input:upstream_01:") for item in requirements)
    assert all(blocker.code.value != "runner_unavailable" for blocker in result.blockers)


def test_formal_stage8_freeze_is_blocked_by_uncommitted_matrix_predecessor(
    tmp_path,
) -> None:
    task_id = "stage8.freeze"
    config = ResolvedConfigV2.resolve(
        _formal_base_config(stage=8, task_id=task_id),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 2},
            "artifacts": {"output_dir": "outputs/formal-stage8-freeze"},
        },
    )
    policy = config.task_definition.formal_eligibility
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(policy.required_capabilities),
        frozen_contract_stages=frozenset(policy.required_contract_stages),
        passed_gate_ids=frozenset(policy.required_gate_ids),
        estimator_decision_ref="evidence/formal-estimator-decision.json",
    )
    runtime = TaskRuntime()
    runtime.register(
        Stage789CompositeTaskRunner(
            runner_kind=RunnerKind.CONTRACT,
            workspace_root=tmp_path,
        )
    )

    result = runtime.execute(config, environment=environment)

    assert result.status.value == "BLOCKED"
    requirements = {blocker.requirement for blocker in result.blockers}
    assert any(item.startswith("input:upstream_01:") for item in requirements)
    assert all(blocker.code.value != "runner_unavailable" for blocker in result.blockers)
