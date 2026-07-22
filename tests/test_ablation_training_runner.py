"""Stage 8 真实 cell 训练、证据发布和故障恢复测试。"""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest

from param_importance_nlp.contracts import (
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.experiments import (
    AblationCellEvidenceManifest,
    AblationCellResult,
    AblationCellTrainingEvidence,
    AblationFactor,
    AblationMatrix,
    AblationStudyRunner,
    AblationTrainingCellRuntime,
    ConfiguredAblationTrainingBuilder,
    TinyAblationTrainingBuilder,
)


def _matrix() -> AblationMatrix:
    return AblationMatrix.compile(
        matrix_id="stage8-real-training-fixture",
        base_config={
            "optimizer": {"learning_rate": 0.08},
            "training": {"max_steps": 2},
        },
        factors=(
            AblationFactor(
                name="learning_rate",
                config_path=("optimizer", "learning_rate"),
                baseline_value=0.08,
                alternatives=(0.16,),
            ),
        ),
        base_seed=20260722,
        seed_namespace="stage8-real-training-test",
    )


def _stage8_v2() -> ResolvedConfigV2:
    value = deepcopy(load_canonical_json("configs/local-fixtures/resolved-config-v1.json"))
    value["identity"].update({"stage": 8, "task": "stage8.execute"})
    value["loss"].update(
        {"task_type": "sequence_classification", "weighting": "sample"}
    )
    value["data"].update({"statistical_unit": "sample", "weight_unit": "sample"})
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(value),
        task_id="stage8.execute",
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
            "orchestration": {
                "matrix_ref": "inputs/ablation-matrix.json",
                "paired_design": {
                    "enabled": True,
                    "design": "matched_seeds",
                    "mapping_ref": "inputs/paired-seeds.json",
                    "budget_unit": "samples",
                },
            },
            "artifacts": {"output_dir": "outputs/stage8-configured"},
        },
    )


def test_configured_builder_consumes_v2_without_python_factory(tmp_path: Path) -> None:
    config = _stage8_v2()
    matrix = AblationMatrix.compile(
        matrix_id="stage8-configured-v2",
        base_config=config.to_dict(),
        factors=(
            AblationFactor(
                name="max_steps",
                config_path=("training", "max_steps"),
                baseline_value=1,
                alternatives=(2,),
            ),
        ),
        base_seed=17,
    )
    baseline = next(cell for cell in matrix.cells if cell.parent_cell_id is None)
    # v2 的两个摘要是派生字段，矩阵只保存可重新严格解析的 payload。
    assert "config_hash" not in baseline.config
    resolved = ResolvedConfigV2(baseline.to_dict()["config"])  # type: ignore[arg-type]
    runtime = ConfiguredAblationTrainingBuilder(tmp_path)(
        baseline,
        resolved_config=resolved,
        context={
            "cell_root": tmp_path / "ablation" / baseline.cell_id,
            "cell_artifact_ref_prefix": (
                f"ablation/{baseline.cell_id}/checkpoints"
            ),
        },
    )
    result = runtime.engine.run()
    assert result.status == "COMPLETE"
    assert result.checkpoint_ids
    assert set(runtime.evaluator.evaluate(runtime.engine.model, runtime.evaluation_microbatches)) == {
        "loss",
        "accuracy",
    }


class _CountingBuilder:
    def __init__(self) -> None:
        self.delegate = TinyAblationTrainingBuilder()
        self.calls: list[str] = []

    def __call__(self, cell, *, resolved_config, context):
        self.calls.append(cell.cell_id)
        return self.delegate(
            cell,
            resolved_config=resolved_config,
            context=context,
        )


class _CrashAfterCheckpointBuilder(_CountingBuilder):
    """第一次让底层 TrainingEngine 完成 checkpoint 后模拟进程崩溃。"""

    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    def __call__(self, cell, *, resolved_config, context):
        runtime = super().__call__(
            cell,
            resolved_config=resolved_config,
            context=context,
        )
        owner = self

        class _Engine:
            model = runtime.engine.model

            def run(self):
                result = runtime.engine.run()
                if not owner.crashed:
                    owner.crashed = True
                    raise RuntimeError("injected-after-training-checkpoint")
                return result

        return AblationTrainingCellRuntime(
            engine=_Engine(),
            evaluator=runtime.evaluator,
            evaluation_microbatches=runtime.evaluation_microbatches,
            metric_directions=runtime.metric_directions,
            checkpoint_ref_resolver=runtime.checkpoint_ref_resolver,
            seed=runtime.seed,
            runtime_id=runtime.runtime_id,
            run_intent=runtime.run_intent,
        )


def _runner(tmp_path: Path, builder: object) -> AblationStudyRunner:
    return AblationStudyRunner(
        _matrix(),
        builder=builder,  # type: ignore[arg-type]
        result_root=tmp_path / "ablation",
        artifact_ref_prefix="ablation",
        run_intent="local_fixture",
        source_checkpoint_artifact_hash=canonical_json_hash(
            {"source": "tiny-deterministic-initialization"}
        ),
        config_validator=lambda value: dict(value),
    )


def test_ablation_study_runner_trains_cells_and_publishes_checkpoint_metric_refs(
    tmp_path: Path,
) -> None:
    builder = _CountingBuilder()
    runner = _runner(tmp_path, builder)
    output = runner.run()

    assert len(builder.calls) == len(_matrix().cells)
    assert output.study_result.scope == "local_fixture"
    assert len(output.study_result.cell_result_hashes) == 2
    manifest = AblationCellEvidenceManifest.from_mapping(
        output.evidence_manifest.to_dict()
    )
    assert manifest.to_dict() == output.evidence_manifest.to_dict()
    assert (tmp_path / output.evidence_manifest_ref).is_file()

    observed_seeds: set[int] = set()
    for row in manifest.cells:
        result_ref = str(row["result_ref"])
        assert (tmp_path / result_ref).is_file()
        result_id = Path(result_ref).stem
        result = AblationCellResult.from_mapping(
            runner.matrix_runner.store.load(result_id)
        )
        metadata = result.metadata["outcome"]
        evidence_ref = str(metadata["training_evidence_ref"])
        metrics_ref = str(metadata["metrics_ref"])
        assert (tmp_path / evidence_ref).is_file()
        assert (tmp_path / metrics_ref).is_file()
        evidence = AblationCellTrainingEvidence.from_mapping(
            runner.executor.evidence_store.load(f"evidence-{result.cell_id}")
        )
        assert evidence.artifact_hash == row["evidence_hash"]
        assert evidence.metrics_ref == metrics_ref
        assert evidence.training_result_hash == metadata["training_result_hash"]
        assert tuple(metadata["checkpoint_refs"]) == evidence.checkpoint_refs
        assert all((tmp_path / ref).is_file() for ref in evidence.checkpoint_refs)
        observed_seeds.add(evidence.seed)
    assert len(observed_seeds) == len(_matrix().cells)

    # 完整 cell result commit 存在时，重复调用只做严格恢复，不再构造训练引擎。
    second = runner.run()
    assert second.to_dict() == output.to_dict()
    assert len(builder.calls) == len(_matrix().cells)


def test_ablation_training_evidence_is_reused_after_result_commit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _CountingBuilder()
    runner = _runner(tmp_path, builder)
    original_publish = runner.matrix_runner.store.publish
    failed = False

    def fail_first_result_publish(result_id: str, value: object) -> bool:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected-after-training-evidence")
        return original_publish(result_id, value)  # type: ignore[arg-type]

    monkeypatch.setattr(runner.matrix_runner.store, "publish", fail_first_result_publish)
    with pytest.raises(RuntimeError, match="injected-after-training-evidence"):
        runner.run()
    assert len(builder.calls) == 1
    baseline = next(
        cell for cell in _matrix().cells if cell.parent_cell_id is None
    )
    # training evidence/metrics 已有权威 commit，而 cell result 尚未提交。
    assert (
        runner.executor.evidence_store.commits / f"evidence-{baseline.cell_id}.json"
    ).is_file()
    assert not tuple(runner.matrix_runner.store.commits.glob("*.json"))

    monkeypatch.setattr(runner.matrix_runner.store, "publish", original_publish)
    resumed = runner.run()
    assert len(resumed.study_result.cell_result_hashes) == 2
    # baseline 直接复用 evidence；仅尚未执行的 child 新建一个训练运行时。
    assert len(builder.calls) == 2
    # 至少一个结果明确记录从已提交 training evidence 恢复；不依赖 commit 枚举顺序。
    restored = [
        AblationCellResult.from_mapping(value)
        for value in runner.matrix_runner.store.restore()
    ]
    assert any(
        result.metadata["outcome"]["resumed_training_evidence"] is True
        for result in restored
    )


def test_ablation_training_retry_resumes_engine_checkpoint_before_evidence(
    tmp_path: Path,
) -> None:
    builder = _CrashAfterCheckpointBuilder()
    runner = _runner(tmp_path, builder)
    baseline = next(cell for cell in _matrix().cells if cell.parent_cell_id is None)
    checkpoint_commits = (
        tmp_path
        / "ablation/training/cells"
        / baseline.cell_id
        / "checkpoints/commits"
    )

    with pytest.raises(RuntimeError, match="injected-after-training-checkpoint"):
        runner.run()
    first_checkpoint_names = tuple(
        path.name for path in sorted(checkpoint_commits.glob("*.json"))
    )
    assert first_checkpoint_names
    assert not tuple(runner.executor.evidence_store.commits.glob("*.json"))

    output = runner.run()
    assert len(output.study_result.cell_result_hashes) == 2
    # baseline 第二次 builder 调用先 resume_latest；完整 checkpoint 不会被重新发布。
    assert tuple(path.name for path in sorted(checkpoint_commits.glob("*.json"))) == (
        first_checkpoint_names
    )
    # baseline 构造两次（崩溃、恢复），child 构造一次。
    assert len(builder.calls) == 3
