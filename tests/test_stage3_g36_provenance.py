"""Tests for clean formal provenance construction at the G3-6 boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import pytest

from ops.stage3.publish_stage3_g36_provenance import (
    Stage3ProvenancePublicationError,
    _build_record,
    _apply_timed_execution_receipt,
    _confirm_git_snapshot,
    _device_mapping,
    _git_snapshot,
)
from param_importance_nlp.contracts import ProvenanceRecord
from ops.stage3.run_stage3_formal import _canonical_hash


HASH = "a" * 64


class _Base:
    def __init__(self, *, device: str = "cuda") -> None:
        self.sections = {
            "identity": {
                "master_seed": 3301,
                "input_checkpoint_id": "pythia-14m-step0-56079904bb80",
                "input_run_id": "pythia-grid-20260826T145530Z",
                "parent_experiment_id": None,
                "route": "path_integration",
            },
            "model": {"initialization_id": "pythia-14m-step0-56079904bb80"},
            "runtime": {"device": device, "environment_id": "server-cuda-formal-v1"},
            "distributed": {"device_ids": [0]},
        }

    def section(self, name: str):
        return self.sections[name]


class _Config:
    task_id = "stage3.08_error_analysis_and_stability"
    config_hash = HASH

    def __init__(self, *, device: str = "cuda") -> None:
        self.base_config = _Base(device=device)

    def section(self, name: str):
        assert name == "launcher"
        return {"world_size": 1}


def _environment():
    return SimpleNamespace(
        evidence_refs={
            "gpu_health": "reports/gpu-health.json",
            "stage2_model_manifest": "evidence/model/commits/model_manifest.json",
            "stage2_tokenizer_manifest": "evidence/model/commits/tokenizer_manifest.json",
            "stage2_data_manifest": "evidence/data/commits/data_manifest.json",
        }
    )


def test_build_record_is_formal_clean_and_round_trips() -> None:
    record = _build_record(
        config=_Config(),  # type: ignore[arg-type]
        environment=_environment(),  # type: ignore[arg-type]
        config_ref="configs/s308.json",
        evaluator_sources=("evidence/table.json", "evidence/execution.json"),
        git_commit="b" * 40,
        git_branch="codex/stage3",
        started_at="2026-09-04T10:00:00Z",
        ended_at="2026-09-04T10:01:00Z",
    )
    assert record.formal_eligible
    assert record.worktree_clean
    assert record.device_mapping == ("cuda:0",)
    assert record.artifact_refs == ("evidence/table.json", "evidence/execution.json")
    assert ProvenanceRecord.from_mapping(record.to_dict()) == record


def test_device_mapping_supports_cpu_and_rejects_unknown_device() -> None:
    assert _device_mapping(_Config(device="cpu")) == ("cpu",)  # type: ignore[arg-type]
    with pytest.raises(Stage3ProvenancePublicationError, match="DEVICE_INVALID"):
        _device_mapping(_Config(device="xpu"))  # type: ignore[arg-type]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_git_snapshot_requires_clean_named_branch(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-b", "codex/test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Stage3 Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    assert _git_snapshot(tmp_path, commit) == (commit, "codex/test")
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(Stage3ProvenancePublicationError, match="GIT_DIRTY"):
        _git_snapshot(tmp_path, commit)


def test_confirm_git_snapshot_rejects_same_commit_on_different_branch(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-b", "codex/original")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Stage3 Test")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "switch", "-c", "codex/drifted")
    with pytest.raises(Stage3ProvenancePublicationError, match="GIT_BRANCH_DRIFT"):
        _confirm_git_snapshot(
            tmp_path, expected_commit=commit, expected_branch="codex/original"
        )


def test_build_record_rejects_inverted_times() -> None:
    with pytest.raises(Stage3ProvenancePublicationError, match="TIME_ORDER_INVALID"):
        _build_record(
            config=_Config(),  # type: ignore[arg-type]
            environment=_environment(),  # type: ignore[arg-type]
            config_ref="configs/s308.json",
            evaluator_sources=("evidence/table.json",),
            git_commit="b" * 40,
            git_branch="codex/stage3",
            started_at=datetime(2026, 9, 4, 10, 1, tzinfo=timezone.utc).isoformat(),
            ended_at=datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc).isoformat(),
        )


def test_timed_receipt_binds_all_provenance_execution_inputs(tmp_path: Path) -> None:
    refs = {
        "config_ref": "configs/s308.json",
        "environment_ref": "configs/s308-environment.json",
        "result_ref": "results/s308-result.json",
    }
    for ref in refs.values():
        path = tmp_path / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    receipt_ref = "results/s308-timed-receipt.json"
    receipt = {
        "schema_version": "stage3-s308-timed-execution-v1",
        "status": "PASS",
        "scope": "formal",
        "formal_eligible": True,
        "launch_hash": "1" * 64,
        "task_id": "stage3.08_error_analysis_and_stability",
        **refs,
        "config_hash": "2" * 64,
        "environment_hash": "3" * 64,
        "result_hash": "4" * 64,
        "artifact_refs": {
            "path_error_table": "evidence/error.json",
            "stability_report": "evidence/stability.json",
            "frozen_source_table": "evidence/table.json",
        },
        "artifact_hashes": {
            "path_error_table": "5" * 64,
            "stability_report": "6" * 64,
            "frozen_source_table": "7" * 64,
        },
        "git_commit": "8" * 40,
        "git_branch": "codex/stage3",
        "started_at": "2026-09-04T10:00:00Z",
        "ended_at": "2026-09-04T10:01:00Z",
        "ended_at_source": "wrapper_post_wait",
        "recovered": False,
        "handoff_audit_hash": "9" * 64,
        "receipt_ref": receipt_ref,
    }
    receipt["receipt_hash"] = _canonical_hash(receipt)
    receipt_path = tmp_path / receipt_ref
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    arguments = Namespace(
        timed_execution_receipt=receipt_path,
        config=None,
        environment=None,
        task_result=None,
        started_at=None,
        ended_at=None,
        expected_git_commit=None,
    )
    loaded, logical = _apply_timed_execution_receipt(arguments, tmp_path)
    assert loaded == receipt
    assert logical == receipt_ref
    assert arguments.config == tmp_path / refs["config_ref"]
    assert arguments.environment == tmp_path / refs["environment_ref"]
    assert arguments.task_result == tmp_path / refs["result_ref"]
    assert arguments.started_at == receipt["started_at"]
    assert arguments.ended_at == receipt["ended_at"]
    assert arguments.expected_git_commit == receipt["git_commit"]

    receipt["ended_at_source"] = "result_mtime_recovery"
    receipt["receipt_hash"] = _canonical_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage3ProvenancePublicationError, match="TIMED_RECEIPT_INVALID"):
        _apply_timed_execution_receipt(
            Namespace(timed_execution_receipt=receipt_path), tmp_path
        )


def test_timed_receipt_rejects_tampering(tmp_path: Path) -> None:
    receipt_path = tmp_path / "results/timed.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        '{"schema_version":"stage3-s308-timed-execution-v1",'
        '"receipt_hash":"' + "0" * 64 + '"}\n',
        encoding="utf-8",
    )
    arguments = Namespace(timed_execution_receipt=receipt_path)
    with pytest.raises(Stage3ProvenancePublicationError, match="TIMED_RECEIPT_INVALID"):
        _apply_timed_execution_receipt(arguments, tmp_path)
