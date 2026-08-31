from __future__ import annotations

import json
from pathlib import Path

import pytest

from param_importance_nlp.cli import main


def test_storage_budget_cli_reports_success(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "param_importance_nlp.cli.check_launch_storage",
        lambda **_: {
            "data_free_bytes": 200 * 1024**3,
            "data_required_free_bytes": 100 * 1024**3 + 1,
            "root_free_bytes": 20 * 1024**3,
            "root_minimum_free_bytes": 0,
            "inode_free": 1000,
            "inode_total": 2000,
            "data_ok": True,
            "root_ok": True,
            "inode_ok": True,
            "ok": True,
        },
    )
    result = main(
        [
            "storage-budget-check",
            "--data-root",
            str(tmp_path),
            "--root-filesystem",
            str(tmp_path),
            "--name",
            "fixture",
            "--expected-new-bytes",
            "1",
            "--root-minimum-free-bytes",
            "0",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert result == 0
    assert report["ok"] is True
    assert report["budget"]["safety_margin_bytes"] == 100 * 1024**3
