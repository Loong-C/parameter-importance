"""Stage 4–6 路线执行、checkpoint lineage 与幂等恢复。"""

from __future__ import annotations

import torch

from param_importance_nlp.experiments.routes import (
    TrainingPhaseSpec,
    TrainingRouteSpec,
)
from param_importance_nlp.experiments.training_routes import (
    TrainingPhaseRuntime,
    TrainingRouteRunner,
)
from param_importance_nlp.providers import build_tiny_training_fixture
from param_importance_nlp.runtime import CheckpointStore
from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec


def _route() -> TrainingRouteSpec:
    pretrain = TrainingPhaseSpec(
        "pretrain",
        "pretrain",
        "base-init",
        "tiny-architecture",
        "tiny-pretrain-data",
        "logical-pretrain-checkpoint",
        1,
        importance_enabled=False,
    )
    finetune = TrainingPhaseSpec(
        "finetune",
        "finetune",
        "base-init",
        "tiny-architecture",
        "tiny-finetune-data",
        "logical-finetune-checkpoint",
        1,
        parent_phase_id="pretrain",
        input_checkpoint_id="logical-pretrain-checkpoint",
        task_id="tiny-classification",
        importance_enabled=False,
    )
    direct = TrainingPhaseSpec(
        "direct",
        "direct_supervised",
        "base-init",
        "tiny-architecture",
        "tiny-direct-data",
        "logical-direct-checkpoint",
        1,
        task_id="tiny-classification",
        importance_enabled=False,
    )
    return TrainingRouteSpec(
        "tiny-route",
        (pretrain, finetune, direct),
        "local_fixture",
        None,
    )


def test_training_route_runner_executes_and_resumes_without_rebuilding(tmp_path) -> None:
    route = _route()
    build_calls: list[str] = []

    def builder(phase, parent):
        build_calls.append(phase.phase_id)
        fixture = build_tiny_training_fixture(
            task_type="sequence_classification",
            seed=100 + len(build_calls),
            steps=1,
        )
        optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.02)
        store = CheckpointStore(tmp_path / "phase-checkpoints" / phase.phase_id)
        engine = TrainingEngine(
            spec=TrainingRunSpec(
                f"route-{phase.phase_id}",
                "local_fixture",
                1,
                1,
                importance_enabled=False,
            ),
            model=fixture.model,
            optimizer=optimizer,
            cursor=fixture.dataset.cursor(seed=1),
            checkpoint_store=store,
        )
        return TrainingPhaseRuntime(
            engine,
            phase.input_checkpoint_id,
            None if parent is None else parent["artifact_hash"],
        )

    runner = TrainingRouteRunner(route, builder, result_root=tmp_path / "route")
    first = runner.run()
    assert first.status == "COMPLETE"
    assert set(first.phase_results) == {"pretrain", "direct", "finetune"}
    assert len(build_calls) == 3

    def forbidden_builder(phase, parent):  # pragma: no cover - 不应被 resume 调用
        raise AssertionError((phase, parent))

    resumed = TrainingRouteRunner(
        route,
        forbidden_builder,
        result_root=tmp_path / "route",
    ).run(resume=True)
    assert resumed.to_dict() == first.to_dict()
