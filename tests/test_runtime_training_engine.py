"""真实 autograd 训练、在线重要性和 fresh-process 恢复测试。"""

from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch

from param_importance_nlp.contracts import canonical_json_bytes
from param_importance_nlp.providers import (
    ClassificationEvaluator,
    DeterministicBatchCursor,
    TorchModelAdapter,
    TrainingMicrobatch,
)
from param_importance_nlp.runtime import CheckpointStore, ReducerCapabilities
from param_importance_nlp.runtime.training import (
    ImportanceTrajectory,
    TrainingEngine,
    TrainingRunSpec,
    install_training_rng,
)


class TinyClassifier(torch.nn.Module):
    """无 dropout 的小模型，使更新差异只能来自训练执行语义。"""

    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class _InfiniteReducedLossReducer:
    """只让跨 rank 的 loss 诊断溢出，梯度归约仍保持有限。"""

    capabilities = ReducerCapabilities("fixture-overflow", 1, "cpu")

    def sum_tensors(
        self, values: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result = {name: value.clone() for name, value in values.items()}
        if set(result) == {"loss_numerator"}:
            result["loss_numerator"].fill_(float("inf"))
        return result

    def sum_int(self, value: int) -> int:
        return int(value)


def test_install_training_rng_restarts_all_runtime_domains_per_rank() -> None:
    first_plan = install_training_rng(20260722, rank=0, world_size=2)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))

    # 任意推进三个全局 RNG；重新安装同一 seed plan 后必须回到完全相同的起点。
    random.random()
    np.random.random()
    torch.rand(())
    second_plan = install_training_rng(20260722, rank=0, world_size=2)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert first_plan.artifact_hash == second_plan.artifact_hash
    assert first == second

    install_training_rng(20260722, rank=1, world_size=2)
    rank_one = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert rank_one != first


def _steps(count: int = 5) -> tuple[tuple[TrainingMicrobatch, ...], ...]:
    result: list[tuple[TrainingMicrobatch, ...]] = []
    for step in range(count):
        micros: list[TrainingMicrobatch] = []
        for micro in range(2):
            offset = step * 0.05 + micro * 0.02
            features = torch.tensor(
                [
                    [1.0 + offset, 0.0, 0.5],
                    [0.0, 1.0 + offset, -0.5],
                ],
                dtype=torch.float32,
            )
            labels = torch.tensor([0, 1], dtype=torch.int64)
            micros.append(
                TrainingMicrobatch(
                    batch_id=f"s{step}-m{micro}",
                    payload={"features": features, "labels": labels},
                    sample_ids=(f"s{step}-m{micro}-a", f"s{step}-m{micro}-b"),
                )
            )
        result.append(tuple(micros))
    return tuple(result)


def _engine(
    model: TinyClassifier,
    spec: TrainingRunSpec,
    *,
    store: CheckpointStore | None = None,
) -> TrainingEngine:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    return TrainingEngine(
        spec=spec,
        model=TorchModelAdapter(model, task_type="sequence_classification"),
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=DeterministicBatchCursor(_steps()),
        checkpoint_store=store,
    )


def _parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def test_training_with_online_u_does_not_perturb_optimizer_updates() -> None:
    torch.manual_seed(17)
    observed = TinyClassifier()
    control = deepcopy(observed)
    common = dict(
        run_intent="local_fixture",
        max_steps=4,
        max_attempts=4,
        max_grad_norm=0.9,
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    observed_result = _engine(
        observed,
        TrainingRunSpec("observed", importance_enabled=True, estimator_name="u", **common),
    ).run()
    control_result = _engine(
        control,
        TrainingRunSpec("control", importance_enabled=False, estimator_name="u", **common),
    ).run()

    assert observed_result.status == control_result.status == "COMPLETE"
    assert observed_result.importance_snapshot is not None
    assert observed_result.importance_snapshot.successful_steps == 4
    assert control_result.importance_snapshot is None
    for name, observed_value in _parameter_state(observed).items():
        assert torch.equal(observed_value, _parameter_state(control)[name])
    assert all(record.parameter_post_state_hash for record in observed_result.records)
    assert all(record.attempt_commit_state_hash for record in observed_result.records)


def test_training_checkpoint_resume_matches_uninterrupted(tmp_path) -> None:
    torch.manual_seed(31)
    uninterrupted_model = TinyClassifier()
    resumed_model = deepcopy(uninterrupted_model)
    spec = TrainingRunSpec(
        "resume-fixture",
        "local_fixture",
        max_steps=4,
        max_attempts=4,
        importance_enabled=True,
        estimator_name="double",
        checkpoint_every_steps=1,
        weights_exogenous=True,
        common_mean_assumption=True,
    )

    uninterrupted = _engine(uninterrupted_model, spec).run()
    store = CheckpointStore(tmp_path / "checkpoints")
    paused = _engine(resumed_model, spec, store=store).run(until_step=2)
    assert paused.status == "PAUSED"

    # 模拟 fresh process：重新构造 model、optimizer、scheduler、cursor 和 engine，
    # 只通过权威 checkpoint 恢复，不能复用旧 Python 对象。
    fresh_model = TinyClassifier()
    resumed = _engine(fresh_model, spec, store=store).run(resume=True)
    assert resumed.status == "COMPLETE"
    assert resumed.state.global_step == 4
    assert len(store.discover()) == 4
    for name, expected in _parameter_state(uninterrupted_model).items():
        assert torch.equal(expected, _parameter_state(fresh_model)[name])
    assert uninterrupted.importance_snapshot is not None
    assert resumed.importance_snapshot is not None
    assert (
        uninterrupted.importance_snapshot.state_hash
        == resumed.importance_snapshot.state_hash
    )


def test_segmented_checkpoint_schedule_and_importance_trajectory_roundtrip(
    tmp_path,
) -> None:
    """分段频率采用左闭右开边界，phase 末尾总会留下可恢复提交。"""

    torch.manual_seed(47)
    model = TinyClassifier()
    spec = TrainingRunSpec(
        "segmented-fixture",
        "local_fixture",
        max_steps=5,
        max_attempts=5,
        importance_enabled=True,
        estimator_name="u",
        weights_exogenous=True,
        common_mean_assumption=True,
        checkpoint_every_steps=99,
        checkpoint_segments=(
            {"start_step": 0, "end_step": 2, "every_steps": 1},
            {"start_step": 2, "end_step": None, "every_steps": 2},
        ),
    )
    store = CheckpointStore(tmp_path / "segmented-checkpoints")
    result = _engine(model, spec, store=store).run()

    assert result.status == "COMPLETE"
    assert result.checkpoint_ids == (
        "segmented-fixture-step-00000001",
        "segmented-fixture-step-00000002",
        "segmented-fixture-step-00000004",
        # step 5 不命中分段整除规则，但 final checkpoint 规则必须补齐。
        "segmented-fixture-step-00000005",
    )
    assert [commit.generation for commit in store.discover()] == [1, 2, 4, 5]
    trajectory = result.importance_trajectory
    assert trajectory is not None
    assert [point.global_step for point in trajectory.points] == [1, 2, 4, 5]
    restored = ImportanceTrajectory.from_mapping(trajectory.to_dict())
    assert restored == trajectory
    assert restored.to_dict()["artifact_hash"] == trajectory.to_dict()["artifact_hash"]
    canonical_json_bytes(result.to_dict())


def test_nonfinite_reduced_loss_is_recorded_as_null_not_json_nan_or_inf() -> None:
    """诊断 loss 溢出不能污染 canonical artifact，也不能伪造成有限数。"""

    torch.manual_seed(53)
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    result = TrainingEngine(
        spec=TrainingRunSpec(
            "nonfinite-loss-fixture",
            "local_fixture",
            max_steps=1,
            max_attempts=1,
            importance_enabled=False,
        ),
        model=TorchModelAdapter(model, task_type="sequence_classification"),
        optimizer=optimizer,
        cursor=DeterministicBatchCursor(_steps(1)),
        reducer=_InfiniteReducedLossReducer(),
    ).run()

    assert result.status == "COMPLETE"
    assert result.records[0].mean_loss is None
    encoded = canonical_json_bytes(result.to_dict())
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded


def test_classification_evaluator_restores_training_mode() -> None:
    model = TinyClassifier()
    adapter = TorchModelAdapter(model, task_type="sequence_classification")
    model.train(True)
    result = ClassificationEvaluator().evaluate(adapter, _steps(1)[0])
    assert model.training is True
    assert set(result) == {"loss", "accuracy"}
    assert 0.0 <= result["accuracy"] <= 1.0


def test_formal_training_rejects_missing_decision_and_gate() -> None:
    try:
        TrainingRunSpec("formal", "formal", 1, 1)
    except ValueError as exc:
        assert str(exc) == "FORMAL_TRAINING_ESTIMATOR_DECISION_REQUIRED"
    else:  # pragma: no cover - fail-closed 防线
        raise AssertionError("formal training 不应在缺少 decision 时构造成功")
