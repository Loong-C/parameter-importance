from __future__ import annotations

import json
from pathlib import Path

import pytest

from param_importance_nlp.cache import (
    runtime_cache_environment,
    validate_runtime_cache_environment,
)
from param_importance_nlp.capacity import (
    GIB,
    StorageBudget,
    check_launch_storage,
    estimate_checkpoint_bytes,
    estimate_experiment_storage,
)
from param_importance_nlp.lifecycle import RunDirectory, RunState, validate_identifier
from param_importance_nlp.storage import REQUIRED_DIRECTORIES, StorageLayout


def _layout(tmp_path: Path) -> StorageLayout:
    root = tmp_path / "data-root"
    root.mkdir()
    for name in REQUIRED_DIRECTORIES:
        (root / name).mkdir()
    return StorageLayout(root)


def test_run_directory_never_overwrites(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    run = RunDirectory.create(layout, "stage0-fixture")
    assert run.state() is RunState.CREATED
    with pytest.raises(FileExistsError):
        RunDirectory.create(layout, "stage0-fixture")


def test_attempt_and_session_are_isolated(tmp_path: Path) -> None:
    run = RunDirectory.create(_layout(tmp_path), "stage0-fixture")
    first = run.create_attempt()
    second = run.create_attempt()
    assert first.attempt_id == "attempt-000001"
    assert second.attempt_id == "attempt-000002"
    session = first.create_session("session-a")
    assert (session / "events").is_dir()
    assert json.loads((session / "session-state.json").read_text())["state"] == "STARTING"
    with pytest.raises(FileExistsError):
        first.create_session("session-a")


def test_terminal_run_state_cannot_transition(tmp_path: Path) -> None:
    run = RunDirectory.create(_layout(tmp_path), "stage0-fixture")
    run.transition(RunState.RUNNING, reason="started")
    run.transition(RunState.SUCCESS, reason="done")
    with pytest.raises(ValueError, match="Forbidden state transition"):
        run.transition(RunState.RUNNING, reason="illegal restart")


def test_identifier_rejects_path_syntax() -> None:
    for value in ("../escape", "nested/name", "", " space"):
        with pytest.raises(ValueError):
            validate_identifier(value, field="run_id")


def test_cache_environment_stays_inside_data_root(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    environment = runtime_cache_environment(
        layout,
        run_id="run-a",
        attempt_id="attempt-000001",
        session_id="session-a",
    )
    assert validate_runtime_cache_environment(environment, layout) == []
    environment["HF_HOME"] = str(tmp_path / "outside")
    assert validate_runtime_cache_environment(environment, layout) == [
        f"outside_data_root:HF_HOME:{tmp_path / 'outside'}"
    ]


def test_storage_budget_uses_twenty_percent_or_100_gib() -> None:
    small = StorageBudget.from_expected("small", 10 * GIB)
    assert small.safety_margin_bytes == 100 * GIB
    large = StorageBudget.from_expected("large", 1000 * GIB)
    assert large.safety_margin_bytes == 200 * GIB


def test_launch_storage_check_fails_closed_when_budget_exceeds_disk(tmp_path: Path) -> None:
    budget = StorageBudget.from_expected("impossible", 10**30)
    report = check_launch_storage(
        data_root=tmp_path,
        root_filesystem=tmp_path,
        budget=budget,
        root_minimum_free_bytes=0,
    )
    assert report["data_ok"] is False
    assert report["root_ok"] is True
    assert report["ok"] is False


def test_capacity_estimates_scale_with_models_and_repeats() -> None:
    assert estimate_checkpoint_bytes(410_000_000) > estimate_checkpoint_bytes(160_000_000)
    one = estimate_experiment_storage(
        parameter_count=160_000_000,
        retained_checkpoints=4,
        resident_fp32_buffers=8,
        seed_count=1,
        parallel_runs=1,
        logs_and_reports_per_run=GIB,
    )
    repeated = estimate_experiment_storage(
        parameter_count=160_000_000,
        retained_checkpoints=4,
        resident_fp32_buffers=8,
        seed_count=3,
        parallel_runs=2,
        logs_and_reports_per_run=GIB,
    )
    assert repeated == one * 6
