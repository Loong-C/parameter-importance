"""Focused tests for the formal Stage3.08 timed boundary journal."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from ops.stage3.run_stage3_s308_timed import (
    Stage3S308TimedError,
    _executable_input,
    _git_commit,
    _new_state,
    _result_mtime,
    _save_state,
    _task_command,
    _validate_materialization_receipt,
    _validate_state,
)
from ops.stage3.run_stage3_formal import _canonical_hash, _load_json


HASH = "a" * 64


def _timed_receipt() -> dict[str, object]:
    return {
        "schema_version": "stage3-s308-timed-execution-v2",
        "status": "PASS",
        "scope": "formal",
        "formal_eligible": True,
        "launch_hash": "1" * 64,
        "task_id": "stage3.08_error_analysis_and_stability",
        "materialization_receipt_ref": "results/stage3/s308/materialization.json",
        "materialization_receipt_hash": "2" * 64,
        "config_ref": "configs/stage3/s308.json",
        "config_hash": "3" * 64,
        "environment_ref": "configs/stage3/s308-environment.json",
        "environment_hash": "4" * 64,
        "result_ref": "results/stage3/s308/result.json",
        "result_hash": "5" * 64,
        "artifact_refs": {
            "path_error_table": "results/stage3/s308/path-error.json",
            "stability_report": "results/stage3/s308/stability.json",
            "frozen_source_table": "results/stage3/s308/source-table.json",
        },
        "artifact_hashes": {
            "path_error_table": "6" * 64,
            "stability_report": "7" * 64,
            "frozen_source_table": "8" * 64,
        },
        "git_commit": "9" * 40,
        "git_branch": "codex/stage3",
        "started_at": "2026-09-04T10:00:00Z",
        "ended_at": "2026-09-04T10:01:00Z",
        "ended_at_source": "wrapper_post_wait",
        "recovered": False,
        "handoff_audit_hash": "a" * 64,
        "receipt_ref": "results/stage3/s308/timed-execution.json",
        "receipt_hash": "b" * 64,
    }


def test_timed_receipt_v2_has_strict_machine_schema() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "schemas/shared/stage3-s308-timed-execution-v2.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(_timed_receipt())
    inconsistent = _timed_receipt()
    inconsistent["ended_at_source"] = "result_mtime_recovery"
    with pytest.raises(ValidationError):
        validator.validate(inconsistent)


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


def test_materialization_receipt_binds_canonical_result_and_outputs() -> None:
    output_dir = "results/stage3/s308/artifacts"
    receipt = {
        "schema_version": "stage3-task-materialization-receipt-v1",
        "task_id": "stage3.08_error_analysis_and_stability",
        "config_hash": "b" * 64,
        "config_ref": "configs/stage3/s308.json",
        "result_ref": "results/stage3/s308/result.json",
        "artifact_output_dir": output_dir,
        "authority_output_dir": "evidence/stage3/s308/authority",
        "output_refs": {
            kind: f"{output_dir}/commits/{kind}.json"
            for kind in (
                "path_error_table", "stability_report", "frozen_source_table"
            )
        },
        "evidence_refs": {"formal_execution": "evidence/execution.json"},
        "external_gate_ref": None,
        "command": [
            "{python}", "-m", "param_importance_nlp", "task", "run",
            "--config", "{config}", "--environment", "{environment}",
            "--result", "{result}",
        ],
    }
    receipt["artifact_hash"] = _canonical_hash(receipt)
    assert _validate_materialization_receipt(
        receipt,
        receipt_ref="results/stage3/s308/materialization-receipt.json",
        config_ref="configs/stage3/s308.json",
        result_ref="results/stage3/s308/result.json",
        config_hash="b" * 64,
        artifact_output_dir=output_dir,
        environment_evidence_refs={"formal_execution": "evidence/execution.json"},
    ) == receipt
    with pytest.raises(Stage3S308TimedError, match="MATERIALIZATION_RECEIPT_INVALID"):
        _validate_materialization_receipt(
            receipt,
            receipt_ref="results/stage3/s308/materialization-receipt.json",
            config_ref="configs/stage3/s308.json",
            result_ref="results/stage3/s308/task-result.json",
            config_hash="b" * 64,
            artifact_output_dir=output_dir,
            environment_evidence_refs={"formal_execution": "evidence/execution.json"},
        )
