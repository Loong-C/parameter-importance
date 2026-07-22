from __future__ import annotations

import time

import pytest

from param_importance_nlp.contracts import DependencyUnavailable
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
