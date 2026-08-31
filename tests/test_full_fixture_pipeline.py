"""Stage 0--9 缩小本机流水线的跨输出根确定性验收。"""

from __future__ import annotations

from pathlib import Path

from param_importance_nlp.contracts import load_canonical_json
from param_importance_nlp.experiments.full_fixture_pipeline import (
    run_full_fixture_pipeline,
)
from param_importance_nlp.runtime import TaskArtifactStore


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/local-fixtures/resolved-config-v1.json"


def _payload(root: Path, output_dir: str, commit_ref: str):
    published = TaskArtifactStore(root, output_dir).load_commit(commit_ref)
    return load_canonical_json(root / published.object_ref)["payload"]


def test_two_clean_roots_rebuild_identical_stage0_to_9_pipeline(tmp_path: Path) -> None:
    first_root = tmp_path / "first-clean-root"
    second_root = tmp_path / "second-clean-root"

    first = run_full_fixture_pipeline(
        workspace_root=first_root,
        base_config_path=BASE_CONFIG,
    )
    second = run_full_fixture_pipeline(
        workspace_root=second_root,
        base_config_path=BASE_CONFIG,
    )

    # 物理根、mtime 与执行耗时均不在 wire object 中；整份结果可逐字段比较。
    assert first.to_dict() == second.to_dict()
    assert (first_root / "full-fixture-result.json").read_bytes() == (
        second_root / "full-fixture-result.json"
    ).read_bytes()
    assert first.coordinate_registry_hashes
    assert first.seed_plan_hashes
    assert len(first.config_hashes) == len(second.config_hashes)

    expected_tasks = {
        "stage0.06_single_gpu_smoke",
        "stage1.01_entry_and_contract",
        "stage2_3.fixture_core",
        "stage4.minimal_complete_loop",
        "stage5.checkpoint_analysis",
        "stage6.importance_reuse",
        "stage7.functional_pruning_validation",
        "stage8.ablation_and_robustness",
        "stage9.ingest",
        "stage9.tables",
        "stage9.charts",
        "stage9.report",
        "stage9.analysis_visualization_reporting",
        "stage9.replay",
    }
    assert set(first.task_artifact_hashes) == expected_tasks

    # Stage 0 的实际训练产物必须包含已提交在线重要性；Stage 2/3 不是占位状态。
    smoke_commit = "runs/stage0-06-single-gpu-smoke/commits/training_smoke_result.json"
    smoke = _payload(
        first_root,
        "runs/stage0-06-single-gpu-smoke",
        smoke_commit,
    )
    training_result = smoke["training_result"]
    assert training_result["state"]["global_step"] == 1
    assert sum(
        record["status"] == "COMMITTED" for record in training_result["records"]
    ) == 1
    assert smoke["importance_snapshot"] is not None
    stage23 = load_canonical_json(
        first_root / "fixture-output/stage23/local-fixture-result.json"
    )
    # Stage 2 的 decision 自身仍是 FIXTURE_ONLY；完整 fixture 状态机成功收口后，
    # 顶层执行状态按冻结契约推进为 FIXTURE_COMPLETE。
    assert stage23["stage2"]["fixture_status"] == "FIXTURE_COMPLETE"
    assert stage23["stage3"]["fixture_status"] == "FIXTURE_RECOMMENDATION"

    # Stage 4 的真实路线训练被 Stage 5/6 派生任务消费；两个阶段均有权威 commit。
    for output_dir, kind in (
        ("runs/stage5-checkpoint-analysis", "checkpoint_analysis_table"),
        ("runs/stage6-importance-reuse", "importance_reuse_table"),
    ):
        ref = f"{output_dir}/commits/{kind}.json"
        TaskArtifactStore(first_root, output_dir).load_commit(ref)

    # Stage 9 同时发布冻结源、表、图、报告与 deterministic replay 摘要。
    assert first.source_table_hash == second.source_table_hash
    assert first.table_hash == second.table_hash
    assert first.chart_hash == second.chart_hash
    assert first.report_hash == second.report_hash
    assert first.replay_hash == second.replay_hash
    replay = _payload(
        first_root,
        "runs/stage9-replay",
        "runs/stage9-replay/commits/hash_comparison.json",
    )
    assert replay["artifact_hashes_equal"] is True

    # 同一根再次调用走各 task 的 resume/discovery 路径，结果不能发生漂移。
    resumed = run_full_fixture_pipeline(
        workspace_root=first_root,
        base_config_path=BASE_CONFIG,
    )
    assert resumed.to_dict() == first.to_dict()
