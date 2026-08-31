from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.experiments.ablation import AblationFactor, AblationMatrix
from param_importance_nlp.experiments.ablation_runner import (
    AblationCellResult,
    AblationMatrixRunner,
    AblationStudyResult,
)
from param_importance_nlp.experiments.pruning_runner import EvaluationOutcome


AUTH_HASH = "f" * 64


class _ConfigEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, cell, *, resolved_config, context):
        self.calls += 1
        return EvaluationOutcome(
            evaluator_id="ablation-evaluator-v1",
            metrics={
                "quality": float(resolved_config["optimizer"]["learning_rate"]),
                "loss": 1.0 / float(resolved_config["optimizer"]["learning_rate"]),
            },
            metric_directions={
                "quality": "higher_is_better",
                "loss": "lower_is_better",
            },
            scope="local_fixture",
            formal_eligible=False,
        )


def _matrix() -> AblationMatrix:
    return AblationMatrix.compile(
        matrix_id="ablation-matrix-v1",
        base_config={"optimizer": {"learning_rate": 1.0}},
        factors=(
            AblationFactor(
                "learning_rate",
                ("optimizer", "learning_rate"),
                1.0,
                (0.5, 2.0),
            ),
        ),
        base_seed=42,
    )


def test_ablation_runner_binds_children_to_baseline_and_resumes(tmp_path: Path) -> None:
    evaluator = _ConfigEvaluator()
    runner = AblationMatrixRunner(
        _matrix(),
        executor=evaluator,
        result_root=tmp_path / "ablation",
        run_intent="local_fixture",
        config_validator=lambda value: value,
    )
    first = runner.run()
    assert AblationStudyResult.from_mapping(first.to_dict()).to_dict() == first.to_dict()
    assert evaluator.calls == 3
    assert len(first.rows) == 6  # 三个 cell × 两个 metric

    restored = [AblationCellResult.from_mapping(value) for value in runner.store.restore()]
    baseline = next(result for result in restored if result.parent_cell_id is None)
    children = [result for result in restored if result.parent_cell_id is not None]
    assert all(result.parent_result_hash == baseline.artifact_hash for result in children)
    improved = next(result for result in children if result.metrics["quality"] == 2.0)
    assert improved.deltas["quality"] == pytest.approx(1.0)
    assert improved.directed_effects["loss"] == pytest.approx(0.5)

    second = runner.run()
    assert evaluator.calls == 3
    assert first.artifact_hash == second.artifact_hash


def test_formal_ablation_requires_strict_resolved_config_result(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ResolvedConfig"):
        AblationMatrixRunner(
            _matrix(),
            executor=_ConfigEvaluator(),
            result_root=tmp_path / "formal",
            run_intent="formal",
            config_validator=lambda value: value,
            formal_authorization_hash=AUTH_HASH,
        )
