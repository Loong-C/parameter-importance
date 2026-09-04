"""Focused tests for the formal Stage3.08 timed boundary journal."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ops.stage3.run_stage3_s308_timed import (
    Stage3S308TimedError,
    _executable_input,
    _git_commit,
    _new_state,
    _result_mtime,
    _save_state,
    _task_command,
    _validate_state,
)
from ops.stage3.run_stage3_formal import _load_json


HASH = "a" * 64


def _state() -> dict[str, object]:
    return _new_state(
        launch_hash=HASH,
        config_hash="b" * 64,
        environment_hash="c" * 64,
        git_commit="d" * 40,
        git_branch="codex/stage3",
    )


def _validate(value: dict[str, object]) -> dict[str, object]:
    return _validate_state(
        value,
        launch_hash=HASH,
        config_hash="b" * 64,
        environment_hash="c" * 64,
        git_commit="d" * 40,
        git_branch="codex/stage3",
    )


def test_state_round_trips_and_detects_tampering(tmp_path: Path) -> None:
    state = _state()
    path = tmp_path / "state.json"
    _save_state(path, state)  # type: ignore[arg-type]
    loaded = dict(_load_json(path))
    assert _validate(loaded)["phase"] == "PREPARED"
    loaded["phase"] = "COMPLETE"
    with pytest.raises(Stage3S308TimedError, match="STATE_HASH_INVALID"):
        _validate(loaded)


def test_state_rejects_git_or_launch_identity_drift() -> None:
    state = _state()
    with pytest.raises(Stage3S308TimedError, match="STATE_IDENTITY_DRIFT"):
        _validate_state(
            state,
            launch_hash="e" * 64,
            config_hash="b" * 64,
            environment_hash="c" * 64,
            git_commit="d" * 40,
            git_branch="codex/stage3",
        )


def test_git_commit_accepts_sha1_and_sha256_but_rejects_other_hashes() -> None:
    assert _git_commit("d" * 40) == "d" * 40
    assert _git_commit("e" * 64) == "e" * 64
    with pytest.raises(Stage3S308TimedError, match="GIT_COMMIT_INVALID"):
        _git_commit("f" * 39)


def test_executable_allows_only_final_venv_symlink(tmp_path: Path) -> None:
    environment = tmp_path / "env/bin"
    environment.mkdir(parents=True)
    executable = environment / "python"
    executable.symlink_to("/bin/sh")
    path, logical = _executable_input(tmp_path, executable)
    assert path == executable
    assert logical == "env/bin/python"
    linked_parent = tmp_path / "linked-env"
    linked_parent.symlink_to(tmp_path / "env", target_is_directory=True)
    with pytest.raises(Stage3S308TimedError, match="PATH_SYMLINK"):
        _executable_input(tmp_path, linked_parent / "bin/python")


def test_result_mtime_is_timezone_aware_and_task_command_is_exact(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.json"
    result.write_text("{}\n", encoding="utf-8")
    timestamp = datetime(2026, 9, 4, 12, 34, 56, tzinfo=timezone.utc).timestamp()
    result.touch()
    import os

    os.utime(result, (timestamp, timestamp))
    recovered = _result_mtime(result)
    assert recovered == "2026-09-04T12:34:56Z"
    command = _task_command(
        Path("/env/bin/python"),
        config=Path("/data/s308.json"),
        environment=Path("/data/s308-env.json"),
        result=Path("/data/s308-result.json"),
    )
    assert command == [
        "/env/bin/python", "-m", "param_importance_nlp", "task", "run",
        "--config", "/data/s308.json",
        "--environment", "/data/s308-env.json",
        "--result", "/data/s308-result.json",
    ]
