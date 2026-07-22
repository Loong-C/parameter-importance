from __future__ import annotations

import torch

from param_importance_nlp.providers import (
    ClassificationEvaluator,
    DeterministicBatchCursor,
    TorchModelAdapter,
    TrainingMicrobatch,
)
from param_importance_nlp.runtime import TrainingEngine, TrainingRunSpec


class _OverfitClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def test_tiny_classification_training_overfits_repeated_batch() -> None:
    """训练执行器必须真的优化 loss，而不是只发布生命周期 artifact。"""

    torch.manual_seed(20260722)
    batch = TrainingMicrobatch(
        batch_id="overfit-batch",
        payload={
            "features": torch.tensor(
                [[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        },
        sample_ids=("a", "b", "c", "d"),
    )
    model = _OverfitClassifier()
    adapter = TorchModelAdapter(model, task_type="sequence_classification")
    evaluator = ClassificationEvaluator()
    before = evaluator.evaluate(adapter, (batch,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "tiny-overfit",
            run_intent="local_fixture",
            max_steps=6,
            max_attempts=6,
            importance_enabled=False,
            estimator_name="u",
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=adapter,
        optimizer=optimizer,
        cursor=DeterministicBatchCursor(tuple((batch,) for _ in range(6))),
    )
    result = engine.run()
    after = evaluator.evaluate(adapter, (batch,))

    assert result.status == "COMPLETE"
    assert after["loss"] < before["loss"] * 0.35
    assert after["accuracy"] == 1.0
    assert result.importance_snapshot is None
