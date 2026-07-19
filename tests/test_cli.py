from __future__ import annotations

import json
from pathlib import Path

from param_importance_nlp.cli import main


def test_storage_budget_cli_reports_success(tmp_path: Path, capsys) -> None:
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
