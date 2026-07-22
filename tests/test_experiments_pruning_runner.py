from __future__ import annotations

from pathlib import Path

import pytest
import torch

from param_importance_nlp.core import TensorMap
from param_importance_nlp.experiments.pruning import ImportanceSourceSpec, PruningStudySpec
from param_importance_nlp.experiments.pruning_runner import (
    EvaluationOutcome,
    PruningEvaluationResult,
    PruningStudyResult,
    PruningStudyRunner,
)


SOURCE_HASH = "a" * 64
REGISTRY_HASH = "b" * 64
CHECKPOINT_HASH = "c" * 64
AUTH_HASH = "d" * 64
EVIDENCE_HASH = "e" * 64


class _SumEvaluator:
    def __init__(self, *, scope: str = "local_fixture") -> None:
        self.scope = scope
        self.calls = 0

    def evaluate(self, parameters, *, run_id, context):
        self.calls += 1
        return EvaluationOutcome(
            evaluator_id="sum-evaluator-v1",
            metrics={"quality": float(parameters["weight"].sum())},
            metric_directions={"quality": "higher_is_better"},
            scope=self.scope,
            formal_eligible=self.scope == "formal",
            evidence_hash=EVIDENCE_HASH if self.scope == "formal" else None,
            metadata={"run_id": run_id, "condition": context["condition"] if "condition" in context else "pruned"},
        )


def _study(*, scope: str = "local_fixture") -> PruningStudySpec:
    return PruningStudySpec(
        study_id="pruning-study-v1",
        sources=(
            ImportanceSourceSpec(
                method="magnitude",
                artifact_id="importance-v1",
                artifact_hash=SOURCE_HASH,
                coordinate_registry_hash=REGISTRY_HASH,
                score_view="absolute",
                scope=scope,
            ),
        ),
        ratios=(0.5,),
        pruning_scopes=("global",),
        random_mask_seeds=(7,),
        run_intent=scope,
    )


def test_pruning_runner_restores_parameters_and_resumes_committed_units(tmp_path: Path) -> None:
    parameters = TensorMap({"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])})
    original = parameters["weight"].clone()
    scores = TensorMap({"weight": torch.tensor([1.0, 4.0, 2.0, 3.0])})
    evaluator = _SumEvaluator()
    runner = PruningStudyRunner(
        _study(),
        parameters=parameters,
        scores_by_artifact_hash={SOURCE_HASH: scores},
        evaluator=evaluator,
        result_root=tmp_path / "pruning",
        model_checkpoint_hash=CHECKPOINT_HASH,
        coordinate_registry_hash=REGISTRY_HASH,
    )

    first = runner.run()
    assert PruningStudyResult.from_mapping(first.to_dict()).to_dict() == first.to_dict()
    # baseline + high + low + one random mask
    assert len(first.result_ids) == 4
    assert evaluator.calls == 4
    torch.testing.assert_close(parameters["weight"], original)

    restored = [
        PruningEvaluationResult.from_mapping(value) for value in runner.store.restore()
    ]
    high = next(result for result in restored if result.direction == "high")
    low = next(result for result in restored if result.direction == "low")
    assert high.damage["quality"] > low.damage["quality"]
    assert high.selected_count == low.selected_count == 2

    second = runner.run()
    assert evaluator.calls == 4
    assert first.artifact_hash == second.artifact_hash
    assert runner.store.reconcile()["invalid_commits"] == []


def test_pruning_formal_path_requires_authorization_and_formal_evaluator(tmp_path: Path) -> None:
    parameters = TensorMap({"weight": torch.tensor([1.0, 2.0])})
    scores = TensorMap({"weight": torch.tensor([1.0, 2.0])})
    with pytest.raises(ValueError, match="formal_authorization_hash"):
        PruningStudyRunner(
            _study(scope="formal"),
            parameters=parameters,
            scores_by_artifact_hash={SOURCE_HASH: scores},
            evaluator=_SumEvaluator(),
            result_root=tmp_path / "missing-auth",
            model_checkpoint_hash=CHECKPOINT_HASH,
            coordinate_registry_hash=REGISTRY_HASH,
        )

    runner = PruningStudyRunner(
        _study(scope="formal"),
        parameters=parameters,
        scores_by_artifact_hash={SOURCE_HASH: scores},
        evaluator=_SumEvaluator(),
        result_root=tmp_path / "fixture-evaluator",
        model_checkpoint_hash=CHECKPOINT_HASH,
        coordinate_registry_hash=REGISTRY_HASH,
        formal_authorization_hash=AUTH_HASH,
    )
    with pytest.raises(ValueError, match="scope"):
        runner.run()
