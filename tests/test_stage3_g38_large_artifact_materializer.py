from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage3.materialize_stage3_large_artifact_manifest import (
    materialize_stage3_large_artifact_manifest,
)
from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES,
    validate_stage3_large_artifact_manifest,
)


def _roots(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, role in enumerate(REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES):
        ref = f"large/{role}"
        path = root / ref / "payload.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload-{index}".encode())
        result[role] = ref
    return result


def _record(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def test_large_artifact_materializer_is_complete_deterministic_and_validated(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    kwargs = {
        "workspace_root": tmp_path,
        "manifest_id": "stage3-large-r1",
        "generated_at": "2026-09-04T08:00:00Z",
        "artifact_roots": roots,
        "source_refs": {"formal_execution": "evidence/stage3/execution.json"},
        "source_hashes": {"formal_execution": "a" * 64},
        "output": "delivery/large-artifact-manifest.json",
    }
    manifest = materialize_stage3_large_artifact_manifest(**kwargs)
    repeated = materialize_stage3_large_artifact_manifest(**kwargs)
    assert repeated == manifest
    assert manifest["file_count"] == len(REQUIRED_STAGE3_G38_LARGE_ARTIFACT_ROLES)
    assert load_canonical_json(tmp_path / str(kwargs["output"])) == manifest

    record = _record(tmp_path / str(kwargs["output"]), tmp_path)
    delivery = SimpleNamespace(server_large_artifact_manifest=record)
    validate_stage3_large_artifact_manifest(tmp_path, delivery)


def test_large_artifact_materializer_rejects_incomplete_overlap_and_inner_output(
    tmp_path: Path,
) -> None:
    roots = _roots(tmp_path)
    common = {
        "workspace_root": tmp_path,
        "manifest_id": "stage3-large-r1",
        "generated_at": "2026-09-04T08:00:00Z",
        "source_refs": {"formal_execution": "evidence/stage3/execution.json"},
        "source_hashes": {"formal_execution": "a" * 64},
    }
    incomplete = dict(roots)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="missing roles"):
        materialize_stage3_large_artifact_manifest(
            **common,
            artifact_roots=incomplete,
            output="delivery/incomplete.json",
        )

    overlap = dict(roots)
    overlap["additional_role"] = next(iter(roots.values())) + "/nested"
    with pytest.raises(ValueError, match="overlap"):
        materialize_stage3_large_artifact_manifest(
            **common,
            artifact_roots=overlap,
            output="delivery/overlap.json",
        )

    with pytest.raises(ValueError, match="must not be inside"):
        materialize_stage3_large_artifact_manifest(
            **common,
            artifact_roots=roots,
            output=next(iter(roots.values())) + "/manifest.json",
        )
