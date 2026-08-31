"""训练事务与 Stage 3 endpoint 两阶段发布的集成测试。"""

from __future__ import annotations

import hashlib

import torch

from param_importance_nlp.experiments import EndpointState, TrainingEndpointObserver
from param_importance_nlp.providers import build_tiny_training_fixture
from param_importance_nlp.runtime import CheckpointStore, TrainingEngine, TrainingRunSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _endpoint_state(phase: str, step: int, attempt: int) -> EndpointState:
    """构造满足 pre/post/commit 分层约束的确定性 fixture 身份。"""

    common = f"{step}:{attempt}"
    parameter = _hash(f"parameter:{'pre' if phase == 'pre' else 'post'}:{common}")
    optimizer = _hash(f"optimizer:{'pre' if phase == 'pre' else 'post'}:{common}")
    committed = phase == "attempt_commit"
    return EndpointState(
        artifact_id=f"{phase}-{step}-{attempt}",
        artifact_hash=_hash(f"artifact:{phase}:{common}"),
        parameter_hash=parameter,
        buffer_hash=_hash(f"buffer:{common}"),
        optimizer_hash=optimizer,
        scheduler_hash=_hash(f"scheduler:{'commit' if committed else 'pre'}:{common}"),
        scaler_hash=_hash(f"scaler:{'commit' if committed else 'pre'}:{common}"),
        rng_hash=_hash(f"rng:{'commit' if committed else 'pre'}:{common}"),
        data_cursor_hash=_hash(f"cursor:{'commit' if committed else 'pre'}:{common}"),
        model_mode_hash=_hash("model-mode:train"),
    )


def test_training_endpoint_observer_publishes_replay_verified_bundle(tmp_path) -> None:
    fixture = build_tiny_training_fixture(
        task_type="sequence_classification",
        seed=29,
        steps=1,
    )
    observer = TrainingEndpointObserver(
        source_run_id="tiny-endpoint",
        parameter_registry_hash=_hash("registry"),
        selected_steps={1},
        state_capture=_endpoint_state,
        replay_verifier=lambda record: (
            record.replay_verified is False
            and record.parameter_post_state.parameter_hash
            == record.attempt_commit_state.parameter_hash
        ),
        probe_buffer_hash=lambda: _hash("buffer:0:1"),
        output_root=tmp_path / "endpoints",
    )
    optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.05)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "tiny-endpoint",
            "local_fixture",
            max_steps=1,
            max_attempts=1,
            importance_enabled=True,
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=fixture.model,
        optimizer=optimizer,
        cursor=fixture.dataset.cursor(seed=29),
        observers=(observer,),
    )

    result = engine.run()

    assert result.status == "COMPLETE"
    assert len(observer.bundles) == 1
    assert observer.bundles[0].formal_eligible is False
    assert observer.reconcile() == {"valid": ["tiny-endpoint-step-00000001"], "invalid": []}


def _real_engine(tmp_path, *, endpoint_root, checkpoint_root):
    fixture = build_tiny_training_fixture(
        task_type="sequence_classification",
        seed=71,
        steps=2,
    )
    optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.05, momentum=0.9)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "tiny-real-endpoint",
            "local_fixture",
            max_steps=2,
            max_attempts=2,
            importance_enabled=True,
            checkpoint_every_steps=1,
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=fixture.model,
        optimizer=optimizer,
        cursor=fixture.dataset.cursor(seed=71),
        checkpoint_store=CheckpointStore(checkpoint_root),
    )
    observer = TrainingEndpointObserver(
        source_run_id="tiny-real-endpoint",
        parameter_registry_hash=engine.registry.coordinate_registry_hash,
        selected_steps={1, 2},
        output_root=endpoint_root,
        workspace_root=tmp_path,
    )
    observer.bind_engine(engine)
    engine.register_observer(observer)
    return engine, observer


def test_real_endpoint_bundles_replay_optimizer_and_resume_without_recapture(tmp_path) -> None:
    endpoint_root = tmp_path / "run" / "endpoints"
    checkpoint_root = tmp_path / "run" / "checkpoints"
    first, first_observer = _real_engine(
        tmp_path,
        endpoint_root=endpoint_root,
        checkpoint_root=checkpoint_root,
    )
    paused = first.run(until_step=1)
    assert paused.status == "PAUSED"
    assert first_observer.captured_steps == frozenset({1})

    resumed, resumed_observer = _real_engine(
        tmp_path,
        endpoint_root=endpoint_root,
        checkpoint_root=checkpoint_root,
    )
    # 构造时只能从权威 commit 发现 step 1；孤立 state bundle 不会推进集合。
    assert resumed_observer.captured_steps == frozenset({1})
    resumed.resume_latest()
    complete = resumed.run()

    assert complete.status == "COMPLETE"
    assert resumed_observer.captured_steps == frozenset({1, 2})
    assert resumed_observer.reconcile() == {
        "valid": [
            "tiny-real-endpoint-step-00000001",
            "tiny-real-endpoint-step-00000002",
        ],
        "invalid": [],
    }


def test_orphan_endpoint_object_is_reused_before_authoritative_commit(tmp_path) -> None:
    """模拟 object 已发布、commit 前进程退出；重跑不得要求手工删产物。"""

    endpoint_root = tmp_path / "orphan-run" / "endpoints"
    checkpoint_root = tmp_path / "orphan-run" / "checkpoints"
    first, first_observer = _real_engine(
        tmp_path,
        endpoint_root=endpoint_root,
        checkpoint_root=checkpoint_root,
    )
    assert first.run(until_step=1).status == "PAUSED"
    endpoint_id = "tiny-real-endpoint-step-00000001"
    commit_path = endpoint_root / "commits" / f"{endpoint_id}.json"
    object_path = endpoint_root / "objects" / f"{endpoint_id}.json"
    assert commit_path.is_file() and object_path.is_file()

    # 删除临时目录中的 commit 等价于在 object 发布后、权威提交前故障。
    commit_path.unlink()
    retried, retried_observer = _real_engine(
        tmp_path,
        endpoint_root=endpoint_root,
        checkpoint_root=tmp_path / "orphan-retry" / "checkpoints",
    )
    assert retried_observer.captured_steps == frozenset()
    assert retried.run(until_step=1).status == "PAUSED"

    assert retried_observer.captured_steps == frozenset({1})
    assert retried_observer.reconcile() == {
        "valid": [endpoint_id],
        "invalid": [],
    }
