from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.test_stage2_s210_g27b import _formal_inputs, _g27a_gate
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json


def _cli(monkeypatch):
    spec = importlib.util.spec_from_file_location("s210_control_cli", "ops/stage2/run_s210_g27b.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_git_identity",
        lambda repo, declared, producer: {
            "repository_head": "a" * 40,
            "consumer_commit": "a" * 40,
            "producer_commit": producer,
            "worktree_clean": True,
            "source_blobs": {name: "b" * 40 for name in module._CODE_FILES},
        },
    )
    return module


def _args(root: Path, *, run_id: str, output: Path, operations: Path, run: Path) -> list[str]:
    values = _formal_inputs(double_qualified=True)
    payloads = {
        "g26_gate": values["g26_gate"],
        "g26_quality_gates": values["quality"],
        "g26_hypothesis_decisions": values["hypothesis"],
        "g26_statistics_long_table": values["long"],
        "g26_raw_calibration": values["raw_calibration"],
        "g26_family_decisions": values["family"],
        "g27a_report": values["cost"],
        "g27a_gate": _g27a_gate(),
    }
    result: list[str] = []
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        if not path.exists():
            write_canonical_json(path, payload)
        result.extend((f"--{name.replace('_', '-')}", str(path)))
    result.extend(
        (
            "--output-root",
            str(output),
            "--data-root",
            str(root),
            "--run-root",
            str(run),
            "--operations-root",
            str(operations),
            "--repo-root",
            str(root),
            "--producer-commit",
            "c" * 40,
            "--consumer-commit",
            "a" * 40,
            "--run-id",
            run_id,
        )
    )
    return result


def test_control_plane_publishes_status_and_provenance(tmp_path: Path, monkeypatch) -> None:
    cli = _cli(monkeypatch)
    args = _args(tmp_path, run_id="formal-1", output=tmp_path / "results", operations=tmp_path / "operations", run=tmp_path / "runs")
    assert cli.main(["--preflight", *args]) == 0
    assert cli.main(["--execute", *args]) == 0
    assert cli.main(["--status", *args]) == 0
    lineage = load_canonical_json(tmp_path / "results" / "lineage_manifest.json")
    assert lineage["producer_commit"] == "c" * 40
    assert lineage["consumer_commit"] == "a" * 40
    assert lineage["input_refs"]["g27a_report"]["ref"] == "g27a_report.json"
    status = load_canonical_json(tmp_path / "runs" / "status.json")
    assert status["status"] == "SEALED" and status["exit_code"] == 0


def test_control_plane_replay_is_new_and_semantically_equal(tmp_path: Path, monkeypatch) -> None:
    cli = _cli(monkeypatch)
    source_args = _args(tmp_path, run_id="formal-1", output=tmp_path / "results" / "source", operations=tmp_path / "operations" / "source", run=tmp_path / "runs" / "source")
    assert cli.main(["--execute", *source_args]) == 0
    replay_args = _args(tmp_path, run_id="replay-1", output=tmp_path / "results" / "replay", operations=tmp_path / "operations" / "replay", run=tmp_path / "runs" / "replay")
    assert cli.main(["--replay", *replay_args, "--source-result", str(tmp_path / "results" / "source")]) == 0
    comparison = load_canonical_json(tmp_path / "operations" / "replay" / "replay-comparison.json")
    assert comparison["status"] == "PASS"
    assert comparison["source_run_id"] == "formal-1"
    assert comparison["replay_id"] == "replay-1"


def test_control_plane_detach_records_child_identity(tmp_path: Path, monkeypatch) -> None:
    cli = _cli(monkeypatch)
    args = _args(tmp_path, run_id="detached-1", output=tmp_path / "results", operations=tmp_path / "operations", run=tmp_path / "runs")

    class _Process:
        pid = 4242

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *command, **kwargs: _Process())
    assert cli.main(["--detach", *args]) == 0
    launch = load_canonical_json(tmp_path / "operations" / "launcher.pid.json")
    assert launch["pid"] == 4242
    assert "--detach" not in launch["argv"]
    assert "--execute" in launch["argv"]
    assert "--launcher-child" in launch["argv"]


def test_control_plane_replay_mismatch_is_canonical_blocked(tmp_path: Path, monkeypatch) -> None:
    cli = _cli(monkeypatch)
    source_args = _args(tmp_path, run_id="formal-1", output=tmp_path / "results" / "source", operations=tmp_path / "operations" / "source", run=tmp_path / "runs" / "source")
    assert cli.main(["--execute", *source_args]) == 0
    lineage_path = tmp_path / "results" / "source" / "lineage_manifest.json"
    lineage = load_canonical_json(lineage_path)
    lineage.pop("artifact_hash")
    lineage["input_refs"]["g27a_report"]["ref"] = "forged.json"
    lineage["artifact_hash"] = canonical_json_hash(lineage)
    write_canonical_json(lineage_path, lineage)
    replay_args = _args(tmp_path, run_id="replay-1", output=tmp_path / "results" / "replay", operations=tmp_path / "operations" / "replay", run=tmp_path / "runs" / "replay")
    assert cli.main(["--replay", *replay_args, "--source-result", str(tmp_path / "results" / "source")]) == 2
    comparison = load_canonical_json(tmp_path / "operations" / "replay" / "replay-comparison.json")
    assert comparison["status"] == "BLOCKED"
    assert comparison["reasons"] == ["REPLAY_INPUT_REFERENCE_MISMATCH"]
