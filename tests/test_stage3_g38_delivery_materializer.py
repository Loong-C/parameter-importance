from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ops.stage3.materialize_stage3_delivery_manifest import (
    INPUT_SCHEMA,
    materialize_stage3_delivery_manifest,
)
from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import (
    Stage3G38DeliveryManifest,
)
from param_importance_nlp.runtime import TaskLifecycleError


def _file(root: Path, name: str) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"real:{name}".encode())
    return name


def _inventory(root: Path) -> dict[str, object]:
    counter = 0

    def f(suffix: str) -> str:
        nonlocal counter
        counter += 1
        return _file(root, f"delivery/item-{counter}{suffix}")

    return {
        "schema_version": INPUT_SCHEMA,
        "manifest_id": "stage3-formal-delivery-r1",
        "source_tables": {"csv": [f(".csv")], "json": [f(".json")]},
        "analysis_scripts": [f(".py")],
        "figures": [{"id": "overview", "png": f(".png"), "svg": f(".svg")}],
        "chinese_report": {"tex": f(".tex"), "pdf": f(".pdf")},
        "beamer": {
            "tex": f("-slides.tex"),
            "pdf": f("-slides.pdf"),
            "notes": [f("-notes.md")],
            "backups": [f("-backup-1.tex"), f("-backup-2.tex"), f("-backup-3.tex")],
        },
        "replay_reports": {
            "local_cpu": f("-local.json"),
            "server_locked": f("-server.json"),
            "frozen_endpoint_uncached": f("-endpoint.json"),
        },
        "server_large_artifact_manifest": f("-large.json"),
        "git_sync": {
            role: f(f"-{role}.json")
            for role in ("branch", "commit", "push", "remote", "server_clean_head", "sync")
        },
        "worklog": f(".md"),
    }


def test_materializer_hashes_real_files_and_publishes_immutably(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    manifest = materialize_stage3_delivery_manifest(
        workspace_root=tmp_path,
        inventory=inventory,
        output="delivery/manifest.json",
    )

    loaded = load_canonical_json(tmp_path / "delivery/manifest.json")
    parsed = Stage3G38DeliveryManifest.from_mapping(loaded)
    assert parsed == manifest
    assert len(parsed.file_records()) == 24
    for record in parsed.file_records():
        payload = (tmp_path / str(record["path"])).read_bytes()
        assert record["size"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()

    same = materialize_stage3_delivery_manifest(
        workspace_root=tmp_path,
        inventory=inventory,
        output="delivery/manifest.json",
    )
    assert same.artifact_hash == manifest.artifact_hash


def test_materializer_rejects_duplicates_missing_files_and_changed_retry(tmp_path: Path) -> None:
    inventory = _inventory(tmp_path)
    duplicate = dict(inventory)
    duplicate["analysis_scripts"] = [inventory["source_tables"]["csv"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="duplicate delivery file"):
        materialize_stage3_delivery_manifest(
            workspace_root=tmp_path,
            inventory=duplicate,
            output="delivery/duplicate.json",
        )

    missing = dict(inventory)
    missing["worklog"] = "delivery/does-not-exist.md"
    with pytest.raises(ValueError, match="not an existing workspace file"):
        materialize_stage3_delivery_manifest(
            workspace_root=tmp_path,
            inventory=missing,
            output="delivery/missing.json",
        )

    manifest = materialize_stage3_delivery_manifest(
        workspace_root=tmp_path,
        inventory=inventory,
        output="delivery/manifest.json",
    )
    changed = tmp_path / str(manifest.csv_tables[0]["path"])
    changed.write_bytes(b"changed")
    with pytest.raises(TaskLifecycleError, match="内容不同|different"):
        materialize_stage3_delivery_manifest(
            workspace_root=tmp_path,
            inventory=inventory,
            output="delivery/manifest.json",
        )
