from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from ops.stage3.materialize_stage3_delivery_sources import (
    MANIFEST_NAME,
    materialize_stage3_delivery_sources,
)
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(root: Path) -> tuple[str, list[str], str]:
    root.mkdir()
    _git(root, "init", "-b", "codex/stage3-delivery")
    _git(root, "config", "user.email", "stage3@example.invalid")
    _git(root, "config", "user.name", "Stage3 Test")
    scripts = [
        "ops/stage3/analyze_formal_results.py",
        "src/param_importance_nlp/experiments/stage3_reporting.py",
    ]
    for index, ref in enumerate(scripts):
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
    worklog = "worklogs/2026-09-04-stage3-formal.md"
    path = root / worklog
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Stage 3 formal worklog\n", encoding="utf-8")
    _git(root, "add", "--", *scripts, worklog)
    _git(root, "commit", "-m", "Prepare Stage3 delivery sources")
    return _git(root, "rev-parse", "HEAD"), scripts, worklog


def test_snapshot_copies_clean_tracked_sources_to_stable_data_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    data_root = tmp_path / "data"
    data_root.mkdir()
    commit, scripts, worklog = _repository(repository)
    output_dir = f"evidence/stage3/delivery-sources/{commit}"
    manifest = materialize_stage3_delivery_sources(
        data_root=data_root,
        repository_root=repository,
        expected_git_commit=commit,
        snapshot_id="stage3-formal-delivery-sources-r1",
        output_dir=output_dir,
        analysis_scripts=scripts,
        worklog=worklog,
    )
    assert manifest["artifact_hash"] == canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "artifact_hash"}
    )
    assert load_canonical_json(data_root / output_dir / MANIFEST_NAME) == manifest
    for record in (*manifest["analysis_scripts"], manifest["worklog"]):
        source = repository / record["source_ref"]
        snapshot = data_root / record["snapshot_ref"]
        assert snapshot.read_bytes() == source.read_bytes()
        assert record["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert record["size"] == snapshot.stat().st_size
        assert not str(record["snapshot_ref"]).startswith("tmp/")

    same = materialize_stage3_delivery_sources(
        data_root=data_root,
        repository_root=repository,
        expected_git_commit=commit,
        snapshot_id="stage3-formal-delivery-sources-r1",
        output_dir=output_dir,
        analysis_scripts=scripts,
        worklog=worklog,
    )
    assert same == manifest


def test_snapshot_rejects_dirty_untracked_and_immutable_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    data_root = tmp_path / "data"
    data_root.mkdir()
    commit, scripts, worklog = _repository(repository)
    output_dir = f"evidence/stage3/delivery-sources/{commit}"
    source = repository / scripts[0]
    original = source.read_bytes()
    source.write_bytes(b"VALUE = 99\n")
    with pytest.raises(ValueError, match="clean branch"):
        materialize_stage3_delivery_sources(
            data_root=data_root,
            repository_root=repository,
            expected_git_commit=commit,
            snapshot_id="stage3-formal-delivery-sources-r1",
            output_dir=output_dir,
            analysis_scripts=scripts,
            worklog=worklog,
        )
    source.write_bytes(original)

    with pytest.raises(ValueError, match="tracked file"):
        materialize_stage3_delivery_sources(
            data_root=data_root,
            repository_root=repository,
            expected_git_commit=commit,
            snapshot_id="stage3-formal-delivery-sources-r1",
            output_dir=output_dir,
            analysis_scripts=[*scripts, "ops/stage3/not-tracked.py"],
            worklog=worklog,
        )

    manifest = materialize_stage3_delivery_sources(
        data_root=data_root,
        repository_root=repository,
        expected_git_commit=commit,
        snapshot_id="stage3-formal-delivery-sources-r1",
        output_dir=output_dir,
        analysis_scripts=scripts,
        worklog=worklog,
    )
    snapshot = data_root / manifest["analysis_scripts"][0]["snapshot_ref"]
    snapshot.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="immutable source snapshot drift"):
        materialize_stage3_delivery_sources(
            data_root=data_root,
            repository_root=repository,
            expected_git_commit=commit,
            snapshot_id="stage3-formal-delivery-sources-r1",
            output_dir=output_dir,
            analysis_scripts=scripts,
            worklog=worklog,
        )
