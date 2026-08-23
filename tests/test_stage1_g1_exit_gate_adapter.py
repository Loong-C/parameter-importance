"""Focused contract test for the Stage 1 G1-EXIT TaskArtifact adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from param_importance_nlp.contracts import canonical_json_hash, write_canonical_json
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from ops.stage1 import publish_s1_g1_exit_gate as adapter


def _group(root: Path, task_id: str, output: str, kinds: tuple[str, ...]) -> dict[str, str]:
    source_refs = ("evidence/stage1/source.json",)
    config_body = {
        "schema_version": "producer-config-v1",
        "task_id": task_id,
        "run_intent": "formal",
        "formal_eligible": True,
        "artifact_kinds": list(kinds),
        "source_refs": list(source_refs),
        "commit_refs": {kind: f"{output}/commits/{kind}.json" for kind in kinds},
    }
    config = {**config_body, "config_hash": canonical_json_hash(config_body)}
    store = TaskArtifactStore(root, output)
    published = {}
    for kind in kinds:
        published[kind] = store.publish(
            task_id=task_id,
            artifact_kind=kind,
            config_hash=config["config_hash"],
            run_intent="formal",
            formal_eligible=True,
            source_refs=source_refs,
            payload={"schema_version": f"{kind}-v1", "status": "PASS"},
        )
    hashes = {kind: item.artifact_hash for kind, item in published.items()}
    manifest_body = {
        "schema_version": "stage1-task-artifact-group-manifest-v1",
        "status": "PUBLISHING",
        "task_id": task_id,
        "artifact_kinds": list(kinds),
        "config_hash": config["config_hash"],
        "source_refs_hash": canonical_json_hash(list(source_refs)),
        "source_refs": list(source_refs),
        "expected_artifact_hashes": hashes,
        "commit_refs": config["commit_refs"],
    }
    manifest = {**manifest_body, "artifact_hash": canonical_json_hash(manifest_body)}
    manifest_path = root / output / "group-manifest.json"
    write_canonical_json(manifest_path, manifest)
    success_body = {
        "schema_version": "stage1-task-artifact-group-success-v1",
        "status": "PASS",
        "task_id": task_id,
        "artifact_kinds": list(kinds),
        "config_hash": config["config_hash"],
        "group_manifest_sha256": adapter._sha(manifest_path),
        "commit_refs": config["commit_refs"],
        "commit_artifact_hashes": hashes,
    }
    success = {**success_body, "artifact_hash": canonical_json_hash(success_body)}
    write_canonical_json(root / output / "producer-config.json", config)
    write_canonical_json(root / output / "success.json", success)
    return config["config_hash"]


def test_publish_stage1_exit_gate_validates_seven_commits_and_reuses(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "evidence/stage1/source.json"
    source.parent.mkdir(parents=True)
    write_canonical_json(source, {"schema_version": "source-v1"})
    for task_id, output, kinds in adapter.S1_GROUPS:
        _group(tmp_path, task_id, output, kinds)
    archived_files_before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for output in (adapter.S110_OUTPUT, adapter.S111_OUTPUT)
        for path in (tmp_path / output).rglob("*")
        if path.is_file()
    }
    fake_exit = SimpleNamespace(
        index_sha256="a" * 64,
        index_artifact_hash="b" * 64,
        producer_commit=adapter.STAGE1_G1_EXIT_PRODUCER_COMMIT,
        execution_commit=adapter.STAGE1_G1_EXIT_PRODUCER_COMMIT,
    )
    calls = {"validate": 0, "r4": 0, "s110": 0}
    monkeypatch.setattr(adapter, "validate_stage1_exit_evidence", lambda *_args: calls.__setitem__("validate", calls["validate"] + 1) or fake_exit)
    monkeypatch.setattr(adapter.canonical_stage1, "_emit_load_r4", lambda **_kwargs: calls.__setitem__("r4", calls["r4"] + 1) or {})
    monkeypatch.setattr(adapter.canonical_stage1, "_emit_load_s110", lambda **_kwargs: calls.__setitem__("s110", calls["s110"] + 1) or {})

    first = adapter.publish_stage1_exit_gate(repository_root=Path.cwd(), data_root=tmp_path)
    second = adapter.publish_stage1_exit_gate(repository_root=Path.cwd(), data_root=tmp_path)
    assert first["commit_ref"] == second["commit_ref"]
    assert second["reused"] is True
    gate = first["gate_record"]
    assert gate["gate_id"] == adapter.GATE_ID
    assert gate["stage"] == 1
    assert gate["status"] == "PASS"
    assert len(gate["evidence_refs"]) == 8
    assert gate["measured"]["commit_count"] == 7
    assert gate["measured"]["adapter_repository_head"]
    assert gate["measured"]["adapter_source_ref"] == adapter.ADAPTER_SOURCE_REF
    assert len(gate["measured"]["adapter_source_sha256"]) == 64
    assert len(gate["measured"]["adapter_source_git_object"]) == 40
    assert first["commit_ref"].startswith(adapter.DEFAULT_OUTPUT_DIR)
    archived_files_after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for output in (adapter.S110_OUTPUT, adapter.S111_OUTPUT)
        for path in (tmp_path / output).rglob("*")
        if path.is_file()
    }
    assert archived_files_after == archived_files_before
    assert calls == {"validate": 2, "r4": 2, "s110": 2}
