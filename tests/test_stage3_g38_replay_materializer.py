from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import validate_stage3_replay_reports
from param_importance_nlp.runtime import TaskLifecycleError


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ops" / "stage3" / "materialize_stage3_replay_report.py"
if not MODULE_PATH.is_file():
    MODULE_PATH = ROOT / ".agent-temp" / "materialize_stage3_replay_report.py"
SPEC = importlib.util.spec_from_file_location("stage3_replay_materializer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(root: Path, layer: str) -> dict[str, object]:
    log = root / "evidence" / f"{layer}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{layer}: 3 passed, 0 skipped\n", encoding="utf-8")
    return {
        "schema_version": MODULE.SOURCE_SCHEMA,
        "replay_id": f"stage3-replay-{layer}",
        "layer": layer,
        "implementation_commit": "1" * 40,
        "environment_hash": "2" * 64,
        "command": ["python", "-m", "pytest", "-q", "tests/stage3-replay"],
        "returncode": 0,
        "started_at": "2026-09-04T01:00:00Z",
        "completed_at": "2026-09-04T01:01:00Z",
        "test_summary": {"collected": 3, "passed": 3, "failed": 0, "errors": 0, "skipped": 0},
        "input_refs": {"delivery": "results/stage3/delivery.json"},
        "input_hashes": {"delivery": "3" * 64},
        "evidence_files": [{"path": log.relative_to(root).as_posix(), "role": "pytest_log", "source_refs": ["results/stage3/delivery.json"]}],
    }


def _record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def test_materialized_reports_are_accepted_by_g38_consumer(tmp_path: Path) -> None:
    records: dict[str, dict[str, object]] = {}
    for layer in MODULE.LAYERS:
        output = tmp_path / "reports" / f"{layer}.json"
        payload = MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=_source(tmp_path, layer), output=output)
        assert payload["cache_mode"] == MODULE.CACHE_MODES[layer]
        assert payload["artifact_hash"] == canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
        assert load_canonical_json(output) == payload
        records[layer] = _record(tmp_path, output)
    validate_stage3_replay_reports(tmp_path, SimpleNamespace(replay_reports=records))


def test_materializer_rejects_failed_skipped_missing_and_changed_retry(tmp_path: Path) -> None:
    source = _source(tmp_path, "server_locked")
    skipped = dict(source)
    skipped["test_summary"] = {"collected": 3, "passed": 2, "failed": 0, "errors": 0, "skipped": 1}
    with pytest.raises(ValueError, match="zero-skip"):
        MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=skipped, output="reports/skipped.json")

    failed = dict(source, returncode=1)
    with pytest.raises(ValueError, match="zero-returncode"):
        MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=failed, output="reports/failed.json")

    missing = dict(source, evidence_files=["evidence/missing.log"])
    with pytest.raises(ValueError, match="existing workspace file"):
        MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=missing, output="reports/missing.json")

    output = tmp_path / "reports" / "locked.json"
    MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=source, output=output)
    (tmp_path / "evidence" / "server_locked.log").write_text("changed\n", encoding="utf-8")
    with pytest.raises(TaskLifecycleError, match="内容不同|different"):
        MODULE.materialize_stage3_replay_report(workspace_root=tmp_path, source=source, output=output)
