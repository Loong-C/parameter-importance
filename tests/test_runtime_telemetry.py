from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts import DependencyUnavailable
from param_importance_nlp.runtime import EventRecord, EventType, JsonlEventSink
from param_importance_nlp.runtime import telemetry
from param_importance_nlp.runtime.telemetry import (
    ResourceSampler,
    rebuild_tensorboard_from_jsonl,
)


def test_resource_sampler_records_units_and_explicit_memory_scope() -> None:
    sampler = ResourceSampler(capture_memory=True)
    sampler.start()
    _values = [index * index for index in range(128)]
    time.sleep(0.001)
    profile = sampler.stop(completed_steps=2, effective_units=16)
    value = profile.to_dict()

    assert _values
    assert value["schema_version"] == "training-resource-profile-v1"
    assert value["wall_seconds"] > 0
    assert value["completed_steps"] == 2
    assert value["effective_units"] == 16
    assert value["steps_per_second"] is not None
    assert value["units_per_second"] is not None
    assert value["python_peak_bytes"] is not None
    assert value["memory_scope"] == "python_tracemalloc"


def test_resource_sampler_lifecycle_fails_closed() -> None:
    sampler = ResourceSampler(capture_memory=False)
    with pytest.raises(RuntimeError, match="NOT_STARTED"):
        sampler.stop(completed_steps=0, effective_units=0)
    sampler.start()
    with pytest.raises(RuntimeError, match="ALREADY_STARTED"):
        sampler.start()
    sampler.stop(completed_steps=0, effective_units=0)


def test_tensorboard_rebuild_is_explicit_optional_dependency(tmp_path) -> None:
    with pytest.raises(DependencyUnavailable, match="tensorboard"):
        rebuild_tensorboard_from_jsonl((), tmp_path / "tensorboard")


def test_tensorboard_rebuild_uses_global_step_and_stable_nested_tags(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, float, int]] = []

    class Writer:
        def __init__(self, *, log_dir: str) -> None:
            assert log_dir.endswith("tensorboard")

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            calls.append((tag, value, step))

        def close(self) -> None:
            return None

    module = SimpleNamespace(
        summary=SimpleNamespace(writer=SimpleNamespace(SummaryWriter=Writer))
    )
    monkeypatch.setattr(telemetry, "load_tensorboard_module", lambda: module)
    event_path = tmp_path / "events.jsonl"
    with JsonlEventSink(event_path) as sink:
        sink.append(
            EventRecord.create(
                experiment_id="telemetry-test",
                run_id="run-1",
                attempt_id="attempt-1",
                session_id="session-1",
                rank=0,
                event_type=EventType.OPTIMIZER_STEP,
                sequence=0,
                payload={
                    "global_step": 7,
                    "microstep_count": 1,
                    "sample_count": 2,
                    "effective_token_count": 32,
                    "mean_loss": 1.25,
                    "global_gradient_norm": 0.5,
                    "learning_rates_post_step": [0.01, 0.001],
                },
            )
        )

    count = rebuild_tensorboard_from_jsonl((event_path,), tmp_path / "tensorboard")
    assert count == len(calls)
    assert ("train/mean_loss", 1.25, 7) in calls
    assert ("train/learning_rates_post_step/0", 0.01, 7) in calls
    assert ("train/learning_rates_post_step/1", 0.001, 7) in calls
