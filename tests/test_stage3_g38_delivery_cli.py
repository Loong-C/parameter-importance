from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from param_importance_nlp.contracts.jsonio import write_canonical_json

from ops.stage3 import publish_stage3_delivery_manifest as command


def test_delivery_manifest_command_binds_all_stage310_refs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest_path = tmp_path / "delivery-manifest.json"
    write_canonical_json(manifest_path, {"schema_version": "example"})
    refs = {
        "analysis_report": "evidence/stage3/s310/analysis.json",
        "chart_artifacts": "evidence/stage3/s310/charts.json",
        "handoff_manifest": "evidence/stage3/s310/handoff.json",
        "gate_summary": "evidence/stage3/s310/gates.json",
    }
    captured: dict[str, object] = {}

    def fake_publish(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            commit_ref="evidence/stage3/manifest/commits/delivery_manifest.json",
            identity=SimpleNamespace(artifact_hash="a" * 64),
        )

    monkeypatch.setattr(command, "publish_stage3_delivery_manifest", fake_publish)
    result = command.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--output-dir",
            "evidence/stage3/manifest",
            "--config-hash",
            "b" * 64,
            "--manifest",
            str(manifest_path),
            "--analysis-report-ref",
            refs["analysis_report"],
            "--chart-artifacts-ref",
            refs["chart_artifacts"],
            "--handoff-manifest-ref",
            refs["handoff_manifest"],
            "--gate-summary-ref",
            refs["gate_summary"],
            "--source-ref",
            "evidence/stage3/upstream.json",
            "--source-ref",
            refs["analysis_report"],
        ]
    )

    assert result == 0
    assert captured["workspace_root"] == tmp_path
    assert captured["stage3_10_refs"] == refs
    assert captured["source_refs"] == (
        "evidence/stage3/upstream.json",
        refs["analysis_report"],
        refs["chart_artifacts"],
        refs["handoff_manifest"],
        refs["gate_summary"],
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "artifact_hash": "a" * 64,
        "commit_ref": "evidence/stage3/manifest/commits/delivery_manifest.json",
        "source_refs": list(captured["source_refs"]),
        "status": "PASS",
    }
