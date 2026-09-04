from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import validate_stage3_git_sync_evidence
from param_importance_nlp.runtime import TaskLifecycleError


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ops" / "stage3" / "materialize_stage3_git_sync_evidence.py"
if not MODULE_PATH.is_file():
    MODULE_PATH = ROOT / ".agent-temp" / "materialize_stage3_git_sync_evidence.py"
SPEC = importlib.util.spec_from_file_location("stage3_git_sync_materializer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prepare_workspace(root: Path) -> None:
    for ref in MODULE.AGENT_DOCUMENTS:
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"policy:{ref}\n", encoding="utf-8")


def _source(root: Path, role: str, *, commit: str = "4" * 40) -> dict[str, object]:
    log = root / "evidence" / f"git-{role}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"{role}: clean and synchronized at {commit}\n", encoding="utf-8")
    return {
        "schema_version": MODULE.SOURCE_SCHEMA,
        "evidence_id": f"stage3-git-{role}",
        "role": role,
        "checked_at": "2026-09-04T02:00:00Z",
        "branch": "codex/stage3-delivery",
        "local_commit": commit,
        "remote_commit": commit,
        "server_commit": commit,
        "remote_name": "origin",
        "local_delivery_worktree_clean": True,
        "server_worktree_clean": True,
        "command": ["git", "status", "--porcelain=v1"],
        "returncode": 0,
        "stdout_log": {"path": log.relative_to(root).as_posix(), "role": "git_stdout"},
    }


def _record(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def test_six_materialized_roles_are_accepted_by_g38_consumer(tmp_path: Path) -> None:
    delivery_root = tmp_path / "data-root"
    document_root = tmp_path / "repository"
    delivery_root.mkdir()
    document_root.mkdir()
    _prepare_workspace(document_root)
    records: dict[str, dict[str, object]] = {}
    expected_documents = {
        ref: hashlib.sha256((document_root / ref).read_bytes()).hexdigest()
        for ref in MODULE.AGENT_DOCUMENTS
    }
    for role in MODULE.ROLES:
        output = delivery_root / "git-sync" / f"{role}.json"
        payload = MODULE.materialize_stage3_git_sync_evidence(
            workspace_root=delivery_root,
            agent_document_root=document_root,
            source=_source(delivery_root, role),
            output=output,
        )
        assert payload["agent_document_hashes"] == expected_documents
        assert payload["artifact_hash"] == canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
        assert load_canonical_json(output) == payload
        records[role] = _record(delivery_root, output)
    assert not (delivery_root / "Agent").exists()
    validate_stage3_git_sync_evidence(delivery_root, SimpleNamespace(git_sync=records))


def test_materializer_rejects_head_drift_dirty_missing_policy_and_changed_retry(tmp_path: Path) -> None:
    delivery_root = tmp_path / "data-root"
    document_root = tmp_path / "repository"
    delivery_root.mkdir()
    document_root.mkdir()
    _prepare_workspace(document_root)
    source = _source(delivery_root, "sync")
    mismatch = dict(source, remote_commit="5" * 40)
    with pytest.raises(ValueError, match="do not match"):
        MODULE.materialize_stage3_git_sync_evidence(workspace_root=delivery_root, agent_document_root=document_root, source=mismatch, output="git-sync/mismatch.json")

    dirty = dict(source, server_worktree_clean=False)
    with pytest.raises(ValueError, match="must be clean"):
        MODULE.materialize_stage3_git_sync_evidence(workspace_root=delivery_root, agent_document_root=document_root, source=dirty, output="git-sync/dirty.json")

    (document_root / MODULE.AGENT_DOCUMENTS[-1]).unlink()
    with pytest.raises(ValueError, match="existing workspace file"):
        MODULE.materialize_stage3_git_sync_evidence(workspace_root=delivery_root, agent_document_root=document_root, source=source, output="git-sync/missing.json")
    (document_root / MODULE.AGENT_DOCUMENTS[-1]).write_text("restored\n", encoding="utf-8")

    output = delivery_root / "git-sync" / "sync.json"
    MODULE.materialize_stage3_git_sync_evidence(workspace_root=delivery_root, agent_document_root=document_root, source=source, output=output)
    (delivery_root / "evidence" / "git-sync.log").write_text("changed\n", encoding="utf-8")
    with pytest.raises(TaskLifecycleError, match="内容不同|different"):
        MODULE.materialize_stage3_git_sync_evidence(workspace_root=delivery_root, agent_document_root=document_root, source=source, output=output)
