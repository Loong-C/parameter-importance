"""Stage 4--6 专用路线 runner 的本机真实执行与恢复测试。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from param_importance_nlp.contracts import ResolvedConfig, load_canonical_json
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.task_catalog import RunnerKind
from param_importance_nlp.experiments.routes import (
    TrainingPhaseSpec,
    TrainingRouteSpec,
)
from param_importance_nlp.experiments.stage2 import EstimatorDecision
from param_importance_nlp.experiments.stage456_task_runners import (
    Stage456TaskRunner,
    build_stage456_runner_overrides,
)
from param_importance_nlp.experiments.artifact_lineage import load_input_artifact
from param_importance_nlp.experiments.stage7_training_source import (
    load_route_training_source,
)
from param_importance_nlp.runtime import TaskArtifactStore, TaskRuntime


ROOT = Path(__file__).resolve().parents[1]


def _base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = deepcopy(
        load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json")
    )
    value["identity"]["stage"] = stage
    value["identity"]["task"] = task_id
    value["loss"]["task_type"] = "sequence_classification"
    value["loss"]["weighting"] = "sample"
    value["data"]["statistical_unit"] = "sample"
    value["data"]["weight_unit"] = "sample"
    value["model"]["architecture"] = "tiny-sequence-classifier"
    # 每步保留两个独立 microbatch，满足 U 统计量分母；缩小 global batch 使路线
    # 测试把时间花在事务/恢复边界而非重复的 tiny autograd 上。
    value["batching"]["global_batch_size"] = 2
    value["batching"]["per_device_batch_size"] = 2
    value["batching"]["microbatch_size"] = 1
    value["batching"]["accumulation_steps"] = 1
    return ResolvedConfig.from_mapping(value)


def _fixture_decision() -> EstimatorDecision:
    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "stage456-local-fixture",
        "selected_estimator": "u",
        "scope": "local_fixture",
        "status": "FIXTURE_ONLY",
        "state": "UNFROZEN",
        "batch_size": 32,
        "microbatch_count": 2,
        "repetitions": 2,
        "gate_id": "stage2.G2.7b",
        "gate_status": "NOT_RUN",
        "artifact_ref": None,
        "metadata": {"formal_eligible": False},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return EstimatorDecision.from_mapping(payload)


def _fixture_route(route_id: str = "tiny-stage456-route") -> TrainingRouteSpec:
    pretrain = TrainingPhaseSpec(
        "pretrain",
        "pretrain",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-pretrain-data-v1",
        "logical-pretrain-checkpoint",
        1,
        metadata={"runtime": {"data_seed_offset": 11}},
    )
    direct = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-supervised-data-v1",
        "logical-direct-checkpoint",
        1,
        task_id="tiny-classification",
        metadata={"runtime": {"data_seed_offset": 23}},
    )
    finetune = TrainingPhaseSpec(
        "finetune",
        "finetune",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-supervised-data-v1",
        "logical-finetune-checkpoint",
        1,
        parent_phase_id="pretrain",
        input_checkpoint_id="logical-pretrain-checkpoint",
        task_id="tiny-classification",
        metadata={"runtime": {"data_seed_offset": 23}},
    )
    return TrainingRouteSpec(
        route_id,
        (finetune, direct, pretrain),
        "local_fixture",
        _fixture_decision(),
    )


def _route_config(
    tmp_path: Path,
    route: TrainingRouteSpec,
    *,
    task_id: str = "stage4.minimal_complete_loop",
    output_dir: str = "runs/stage456",
    resume_ref: str | None = None,
    profiling: bool = False,
    max_steps: int = 1,
    evaluation_every_steps: int | None = None,
) -> ResolvedConfigV2:
    route_ref = f"routes/{route.route_id}.json"
    write_canonical_json(tmp_path / route_ref, route.to_dict())
    return ResolvedConfigV2.resolve(
        _base_config(stage=int(task_id.removeprefix("stage").split(".", 1)[0]), task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": max_steps},
            "providers": {"num_labels": 3},
            "profiling": {
                "enabled": profiling,
                "warmup_steps": 0,
                "measure_steps": 1 if profiling else None,
                "repetitions": 1,
                "capture_memory": profiling,
                "capture_throughput": profiling,
            },
            "evaluation": {
                "enabled": True,
                "split": "fixture",
                "every_steps": evaluation_every_steps,
                "batch_size": 2,
                "max_batches": 1,
                "metrics": ["loss", "accuracy"],
            },
            "orchestration": {"route_spec_ref": route_ref},
            "recovery": {"resume_ref": resume_ref},
            "artifacts": {"output_dir": output_dir},
        },
    )


def test_route_phases_publish_recoverable_resource_windows(tmp_path: Path) -> None:
    route = _fixture_route("tiny-stage456-profiled-route")
    config = _route_config(tmp_path, route, profiling=True)
    runtime = TaskRuntime()
    runtime.register(Stage456TaskRunner(RunnerKind.ROUTE_TRAINING, tmp_path))

    result = runtime.execute(config)

    assert result.status.value == "PASS"
    store = TaskArtifactStore(tmp_path, "runs/stage456")
    ref = store.load_commit(result.artifact_refs["training_route"])
    payload = load_canonical_json(tmp_path / ref.object_ref)["payload"]
    assert {item["phase_id"] for item in payload["resource_profiles"]} == {
        "pretrain",
        "direct",
        "finetune",
    }
    for item in payload["resource_profiles"]:
        window = item["window"]
        assert window["start_step"] == 0
        assert window["end_step"] == 1
        assert window["profile"]["completed_steps"] == 1
        assert window["artifact_hash"]
        commit = (
            tmp_path
            / "runs/stage456/route-execution"
            / route.lineage_hash
            / "rank-0000/resource-profiles"
            / item["phase_id"]
            / "commits/window-0000.json"
        )
        assert commit.is_file()


def test_route_evaluates_each_declared_checkpoint_boundary(tmp_path: Path) -> None:
    """评估频率必须对应真实 checkpoint，而不能只重复评估最终模型。"""

    route = _fixture_route("tiny-stage456-checkpoint-evaluation-route")
    config = _route_config(
        tmp_path,
        route,
        max_steps=2,
        evaluation_every_steps=1,
    )
    runtime = TaskRuntime()
    runtime.register(Stage456TaskRunner(RunnerKind.ROUTE_TRAINING, tmp_path))

    result = runtime.execute(config)

    assert result.status.value == "PASS"
    store = TaskArtifactStore(tmp_path, "runs/stage456")
    ref = store.load_commit(result.artifact_refs["training_route"])
    payload = load_canonical_json(tmp_path / ref.object_ref)["payload"]
    rows = payload["evaluation_metrics"]
    for phase_id in route.topological_order:
        phase_rows = [row for row in rows if row["phase_id"] == phase_id]
        assert [row["global_step"] for row in phase_rows] == [1, 2]
        assert [row["checkpoint_id"].rsplit("-", 1)[-1] for row in phase_rows] == [
            "00000001",
            "00000002",
        ]
        assert all(row["checkpoint_manifest_sha256"] for row in phase_rows)
        assert all(
            row["schema_version"] == "stage456-route-evaluation-v1"
            and set(row["metrics"]) == {"loss", "accuracy"}
            and row["panel_sample_ids_hash"]
            for row in phase_rows
        )


def test_tiny_route_executes_all_phases_and_is_task_idempotent(tmp_path: Path) -> None:
    route = _fixture_route()
    config = _route_config(tmp_path, route)
    runtime = TaskRuntime()
    runtime.register(Stage456TaskRunner(RunnerKind.ROUTE_TRAINING, tmp_path))

    first = runtime.execute(config)
    second = runtime.execute(config)

    assert first.status.value == "PASS"
    assert first.to_dict() == second.to_dict()
    assert tuple(first.artifact_refs) == config.task_definition.artifact_kinds
    store = TaskArtifactStore(tmp_path, "runs/stage456")
    training_ref = store.load_commit(first.artifact_refs["training_route"])
    value = load_canonical_json(tmp_path / training_ref.object_ref)
    payload = value["payload"]
    assert payload["route_result"]["status"] == "COMPLETE"
    assert {row["phase_type"] for row in payload["checkpoint_lineage"]} == {
        "pretrain",
        "direct_supervised",
        "finetune",
    }
    assert len(payload["training_metrics"]) == 3
    assert len(payload["evaluation_metrics"]) == 3
    assert payload["route_comparison"]["defined"] is True
    assert payload["importance_trajectory"]
    assert payload["layer_module_summary"]
    assert payload["topk_sets"]
    assert {row["strategy"] for row in payload["pruning_results"]} == {
        "baseline",
        "high",
        "low",
        "random",
    }
    assert payload["estimator_decision_audit"]["qualified"] is False
    assert payload["quadrature_decision_audit"]["enabled"] is False
    assert value["formal_eligible"] is False

    # Stage 7 必须能仅凭上游冻结的 commit ref/稳定 identity/manifest，严格恢复
    # 最深 finetune 终点的真实模型与在线重要性累计器。
    route_document = load_input_artifact(
        tmp_path,
        first.artifact_refs["training_route"],
    )
    stage7_source = load_route_training_source(tmp_path, route_document)
    assert stage7_source.training_result["run_id"].endswith("-finetune")
    assert stage7_source.checkpoint_state["model"]
    assert stage7_source.accumulator["magnitude"]
    assert len(stage7_source.checkpoint_identity_hash) == 64

    phase_commit_root = (
        tmp_path
        / "runs/stage456/route-execution"
        / route.lineage_hash
        / "rank-0000/phase-results/commits"
    )
    before = {
        path.name: load_canonical_json(path) for path in phase_commit_root.glob("*.json")
    }
    # 模拟进程在 route phase 全部提交后、task commit 发布期间退出。对象与 phase
    # commit 保留；这里只移走可重建的任务级 commit。
    task_commit_root = tmp_path / "runs/stage456/commits"
    for path in task_commit_root.glob("*.json"):
        path.unlink()

    implicit_resume = runtime.execute(config)
    assert implicit_resume.status.value == "FAIL"
    assert "TRAINING_RESUME_REF_REQUIRED" in implicit_resume.message

    resume_config = _route_config(
        tmp_path,
        route,
        resume_ref=f"runs/stage456/route-execution/{route.lineage_hash}",
    )
    assert resume_config.config_hash == config.config_hash
    assert resume_config.full_hash != config.full_hash
    resumed = runtime.execute(resume_config)
    after = {
        path.name: load_canonical_json(path) for path in phase_commit_root.glob("*.json")
    }
    assert resumed.status.value == "PASS"
    assert resumed.to_dict() == first.to_dict()
    assert after == before

    trajectory_task = "stage4.importance_trajectory"
    trajectory_config = ResolvedConfigV2.resolve(
        _base_config(stage=4, task_id=trajectory_task),
        task_id=trajectory_task,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {
                "input_result_refs": [first.artifact_refs["training_route"]]
            },
            "artifacts": {"output_dir": "runs/stage4-trajectory"},
        },
    )
    statistics_runtime = TaskRuntime()
    statistics_runtime.register(
        Stage456TaskRunner(RunnerKind.STATISTICS, tmp_path)
    )
    trajectory_result = statistics_runtime.execute(trajectory_config)
    assert trajectory_result.status.value == "PASS"
    assert tuple(trajectory_result.artifact_refs) == (
        "importance_trajectory_table",
        "layer_module_summary",
    )

    report_task = "stage4.report"
    report_config = ResolvedConfigV2.resolve(
        _base_config(stage=4, task_id=report_task),
        task_id=report_task,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {
                "input_result_refs": [
                    trajectory_result.artifact_refs["importance_trajectory_table"]
                ]
            },
            "artifacts": {"output_dir": "runs/stage4-report"},
        },
    )
    report_runtime = TaskRuntime()
    report_runtime.register(Stage456TaskRunner(RunnerKind.REPORTING, tmp_path))
    report_result = report_runtime.execute(report_config)
    assert report_result.status.value == "PASS"
    assert tuple(report_result.artifact_refs) == (
        "stage_report",
        "chart_artifacts",
        "gate_summary",
    )


def test_route_contract_rejects_declared_parent_lineage_drift() -> None:
    with pytest.raises(ValueError, match="input checkpoint"):
        TrainingRouteSpec(
            "bad-route",
            (
                TrainingPhaseSpec(
                    "pretrain",
                    "pretrain",
                    "init",
                    "model",
                    "data",
                    "pretrain-out",
                    1,
                ),
                TrainingPhaseSpec(
                    "finetune",
                    "finetune",
                    "init",
                    "model",
                    "task-data",
                    "finetune-out",
                    1,
                    parent_phase_id="pretrain",
                    input_checkpoint_id="wrong-parent-output",
                    task_id="task",
                ),
            ),
            "local_fixture",
            _fixture_decision(),
        )


def test_route_contract_and_stage6_matrix_are_hash_bound(tmp_path: Path) -> None:
    """Stage 6 必须消费两条真实路线执行与其可恢复评估 checkpoint。"""

    direct_phase = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-supervised-data-v1",
        "logical-direct-checkpoint",
        1,
        task_id="tiny-classification",
        metadata={"runtime": {"data_seed_offset": 23}},
    )
    pretrain_phase = TrainingPhaseSpec(
        "pretrain",
        "pretrain",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-pretrain-data-v1",
        "logical-pretrain-checkpoint",
        1,
        metadata={"runtime": {"data_seed_offset": 11}},
    )
    finetune_phase = TrainingPhaseSpec(
        "finetune",
        "finetune",
        "tiny-init-v1",
        "tiny-model-v1",
        "tiny-supervised-data-v1",
        "logical-finetune-checkpoint",
        1,
        parent_phase_id="pretrain",
        input_checkpoint_id="logical-pretrain-checkpoint",
        task_id="tiny-classification",
        metadata={"runtime": {"data_seed_offset": 23}},
    )
    direct_route = TrainingRouteSpec(
        "route-matrix-direct", (direct_phase,), "local_fixture", _fixture_decision()
    )
    finetune_route = TrainingRouteSpec(
        "route-matrix-finetune",
        (finetune_phase, pretrain_phase),
        "local_fixture",
        _fixture_decision(),
    )
    training_runtime = TaskRuntime()
    training_runtime.register(
        Stage456TaskRunner(RunnerKind.ROUTE_TRAINING, tmp_path)
    )
    direct_result = training_runtime.execute(
        _route_config(
            tmp_path,
            direct_route,
            task_id="stage4.direct_supervised",
            output_dir="runs/stage4-direct",
        )
    )
    finetune_result = training_runtime.execute(
        _route_config(
            tmp_path,
            finetune_route,
            task_id="stage4.finetune",
            output_dir="runs/stage4-finetune",
        )
    )
    assert direct_result.status.value == finetune_result.status.value == "PASS"
    direct_payload_ref = TaskArtifactStore(tmp_path, "runs/stage4-direct").load_commit(
        direct_result.artifact_refs["checkpoint_lineage"]
    )
    direct_payload = load_canonical_json(tmp_path / direct_payload_ref.object_ref)[
        "payload"
    ]
    checkpoint_row = direct_payload["checkpoint_lineage"][0]
    assert checkpoint_row["physical_checkpoint_commit_ref"]
    assert checkpoint_row["commit_identity_sha256"]
    assert checkpoint_row["bundle_manifest_sha256"]

    matrix_task = "stage6.route_matrix"
    matrix_config = ResolvedConfigV2.resolve(
        _base_config(stage=6, task_id=matrix_task),
        task_id=matrix_task,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {
                "input_result_refs": [
                    *direct_result.artifact_refs.values(),
                    *finetune_result.artifact_refs.values(),
                ]
            },
            "artifacts": {"output_dir": "runs/stage6-route-matrix"},
        },
    )
    contract_runtime = TaskRuntime()
    contract_runtime.register(Stage456TaskRunner(RunnerKind.CONTRACT, tmp_path))
    matrix_result = contract_runtime.execute(matrix_config)
    assert matrix_result.status.value == "PASS"
    matrix_store = TaskArtifactStore(tmp_path, "runs/stage6-route-matrix")
    matrix_ref = matrix_store.load_commit(matrix_result.artifact_refs["route_matrix"])
    matrix = load_canonical_json(tmp_path / matrix_ref.object_ref)["payload"]["route_matrix"]
    assert matrix["base_initialization_id"] == "tiny-init-v1"
    assert matrix["route_lineage_hashes"] == sorted(
        [direct_route.lineage_hash, finetune_route.lineage_hash]
    )
    assert len(matrix["route_execution_sources"]) == 2

    evaluate_task = "stage6.evaluate"
    evaluate_config = ResolvedConfigV2.resolve(
        _base_config(stage=6, task_id=evaluate_task),
        task_id=evaluate_task,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {
                "input_result_refs": list(matrix_result.artifact_refs.values()),
                "matrix_ref": matrix_result.artifact_refs["route_matrix"],
                "paired_design": {
                    "enabled": True,
                    "design": "matched_seeds",
                    "mapping_ref": matrix_result.artifact_refs["route_matrix"],
                    "budget_unit": "samples",
                },
            },
            "artifacts": {"output_dir": "runs/stage6-evaluate"},
        },
    )
    validation_runtime = TaskRuntime()
    validation_runtime.register(Stage456TaskRunner(RunnerKind.VALIDATION, tmp_path))
    evaluate_result = validation_runtime.execute(evaluate_config)
    assert evaluate_result.status.value == "PASS"
    paired_ref = TaskArtifactStore(tmp_path, "runs/stage6-evaluate").load_commit(
        evaluate_result.artifact_refs["paired_route_metrics"]
    )
    paired_payload = load_canonical_json(tmp_path / paired_ref.object_ref)["payload"]
    assert paired_payload["data"]
    assert paired_payload["route_audit"]["verified_checkpoint_count"] == 2


def test_override_factory_composes_fallbacks_by_runner_kind(tmp_path: Path) -> None:
    class _Fallback:
        runner_kind = RunnerKind.CONTRACT

        def run(self, request):  # pragma: no cover - 本测试只验证组合身份
            raise AssertionError(request)

    overrides = build_stage456_runner_overrides(
        tmp_path, fallbacks={RunnerKind.CONTRACT: _Fallback()}
    )
    assert set(overrides) == {
        RunnerKind.ROUTE_TRAINING,
        RunnerKind.CONTRACT,
        RunnerKind.STATISTICS,
        RunnerKind.VALIDATION,
        RunnerKind.ANALYSIS,
        RunnerKind.REPORTING,
    }
    assert overrides[RunnerKind.CONTRACT].fallback is not None
