"""Publish the formal Stage 1 ``stage1.G1-EXIT`` TaskArtifact.

This is a narrow adapter around the released Stage 1 loaders.  It does not
rebuild Stage 1 evidence or accept caller supplied status, metrics, or hashes.
The only writable object is the content-addressed ``gate_record`` commit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from param_importance_nlp.contracts import (  # noqa: E402
    GateRecord,
    GateStatus,
    STAGE1_G1_EXIT_PRODUCER_COMMIT,
    STAGE1_G1_EXIT_TASK_ID,
    canonical_json_hash,
    load_canonical_json,
    validate_stage1_exit_evidence,
)
from param_importance_nlp.runtime.task_artifacts import (  # noqa: E402
    LoadedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from ops.stage1 import formalize_s1_11 as canonical_stage1  # noqa: E402


GATE_ID = "stage1.G1-EXIT"
ARTIFACT_KIND = "gate_record"
S111_OUTPUT = "evidence/stage1/tasks/11-s1-11-r4-20260821"
S110_OUTPUT = "evidence/stage1/tasks/10-s1-10-r12-20260821"
INDEX_REF = canonical_stage1.S111_R4_INDEX_REF
S110_TASK_ID = "stage1.10_checkpoint_resume_and_artifacts"
S110_KINDS = tuple(canonical_stage1.S110_TASK_ARTIFACT_KINDS)
S111_KINDS = tuple(canonical_stage1.S111_TASK_ARTIFACT_KINDS)
S1_GROUPS = (
    (S110_TASK_ID, S110_OUTPUT, S110_KINDS),
    (STAGE1_G1_EXIT_TASK_ID, S111_OUTPUT, S111_KINDS),
)


class Stage1ExitGateAdapterError(RuntimeError):
    """The immutable Stage 1 authority cannot be promoted to a Gate."""


def _logical(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage1ExitGateAdapterError(f"{field}:INVALID_REF")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1ExitGateAdapterError(f"{field}:INVALID_REF")
    return logical.as_posix()


def _path(root: Path, ref: str, *, field: str) -> Path:
    logical = _logical(ref, field=field)
    candidate = root.joinpath(*PurePosixPath(logical).parts)
    if any(part.is_symlink() for part in (root, *candidate.parents, candidate)):
        raise Stage1ExitGateAdapterError(f"{field}:SYMLINK_FORBIDDEN")
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise Stage1ExitGateAdapterError(f"{field}:PATH_ESCAPE") from error
    return candidate


def _sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Stage1ExitGateAdapterError(f"FILE_UNREADABLE:{path}") from error


def _obj(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = load_canonical_json(path)
    except Exception as error:
        raise Stage1ExitGateAdapterError(f"{field}:JSON_INVALID") from error
    if not isinstance(value, dict):
        raise Stage1ExitGateAdapterError(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _self_hash(value: Mapping[str, object], *, field: str, hash_field: str = "artifact_hash") -> str:
    body = dict(value)
    declared = body.pop(hash_field, None)
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise Stage1ExitGateAdapterError(f"{field}:SELF_HASH_INVALID")
    return declared


def _load_group(
    root: Path,
    *,
    task_id: str,
    output_dir: str,
    kinds: Sequence[str],
) -> tuple[dict[str, LoadedTaskArtifact], dict[str, object]]:
    """Read one complete producer group and verify config/manifest/success closure."""

    output = _path(root, output_dir, field=f"{task_id}.output")
    config_ref = f"{output_dir}/producer-config.json"
    manifest_ref = f"{output_dir}/group-manifest.json"
    success_ref = f"{output_dir}/success.json"
    config = _obj(_path(root, config_ref, field=f"{task_id}.config"), field=f"{task_id}.config")
    manifest = _obj(_path(root, manifest_ref, field=f"{task_id}.manifest"), field=f"{task_id}.manifest")
    success = _obj(_path(root, success_ref, field=f"{task_id}.success"), field=f"{task_id}.success")
    if config.get("task_id") != task_id or config.get("run_intent") != "formal" or config.get("formal_eligible") is not True:
        raise Stage1ExitGateAdapterError(f"{task_id}:CONFIG_IDENTITY_INVALID")
    config_hash = config.get("config_hash")
    if not isinstance(config_hash, str) or canonical_json_hash({k: v for k, v in config.items() if k != "config_hash"}) != config_hash:
        raise Stage1ExitGateAdapterError(f"{task_id}:CONFIG_HASH_INVALID")
    expected_kinds = list(kinds)
    commit_refs = {kind: f"{output_dir}/commits/{kind}.json" for kind in kinds}
    source_refs = config.get("source_refs")
    if not isinstance(source_refs, list) or any(not isinstance(ref, str) for ref in source_refs) or len(source_refs) != len(set(source_refs)):
        raise Stage1ExitGateAdapterError(f"{task_id}:CONFIG_SOURCE_CLOSURE_INVALID")
    source_tuple = tuple(_logical(ref, field=f"{task_id}.source_ref") for ref in source_refs)
    for source_ref in source_tuple:
        source_path = _path(root, source_ref, field=f"{task_id}.source_ref")
        if not source_path.is_file():
            raise Stage1ExitGateAdapterError(f"{task_id}:SOURCE_MISSING:{source_ref}")
    if config.get("artifact_kinds") != expected_kinds or config.get("commit_refs") != commit_refs:
        raise Stage1ExitGateAdapterError(f"{task_id}:CONFIG_COMMIT_WIRE_INVALID")
    if manifest.get("schema_version") != "stage1-task-artifact-group-manifest-v1" or manifest.get("status") != "PUBLISHING":
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_MANIFEST_INVALID")
    if manifest.get("task_id") != task_id or manifest.get("artifact_kinds") != expected_kinds or manifest.get("config_hash") != config_hash:
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_MANIFEST_IDENTITY_INVALID")
    if manifest.get("source_refs") != list(source_tuple) or manifest.get("source_refs_hash") != canonical_json_hash(list(source_tuple)) or manifest.get("commit_refs") != commit_refs:
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_SOURCE_CLOSURE_INVALID")
    manifest_hash = _self_hash(manifest, field=f"{task_id}.manifest")
    if success.get("schema_version") != "stage1-task-artifact-group-success-v1" or success.get("status") != "PASS":
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_SUCCESS_INVALID")
    if success.get("task_id") != task_id or success.get("artifact_kinds") != expected_kinds or success.get("config_hash") != config_hash:
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_SUCCESS_IDENTITY_INVALID")
    if success.get("group_manifest_sha256") != _sha(_path(root, manifest_ref, field=f"{task_id}.manifest")) or success.get("commit_refs") != commit_refs:
        raise Stage1ExitGateAdapterError(f"{task_id}:GROUP_SUCCESS_CLOSURE_INVALID")
    _self_hash(success, field=f"{task_id}.success")
    loaded: dict[str, LoadedTaskArtifact] = {}
    hashes: dict[str, str] = {}
    for kind in kinds:
        ref = commit_refs[kind]
        try:
            item = load_committed_task_artifact(root, ref, require_formal=True)
        except Exception as error:
            raise Stage1ExitGateAdapterError(f"{task_id}.{kind}:COMMIT_INVALID") from error
        if item.identity.task_id != task_id or item.identity.artifact_kind != kind or item.identity.config_hash != config_hash or item.source_refs != source_tuple:
            raise Stage1ExitGateAdapterError(f"{task_id}.{kind}:COMMIT_IDENTITY_INVALID")
        if any(str(item.payload.get(key, "")).upper() in {"FAIL", "FAILED", "BLOCKED", "NOT_RUN", "FORMAL_CANDIDATE"} for key in ("status", "gate_status", "overall_status", "exit_verdict")):
            raise Stage1ExitGateAdapterError(f"{task_id}.{kind}:NOT_SUCCESS")
        loaded[kind] = item
        hashes[kind] = item.identity.artifact_hash
    if manifest.get("expected_artifact_hashes") != hashes or success.get("commit_artifact_hashes") != hashes:
        raise Stage1ExitGateAdapterError(f"{task_id}:COMMIT_HASH_CLOSURE_INVALID")
    return loaded, {
        "task_id": task_id,
        "output_dir": output_dir,
        "config_hash": config_hash,
        "config_sha256": _sha(_path(root, config_ref, field=f"{task_id}.config")),
        "group_manifest_ref": manifest_ref,
        "group_manifest_sha256": _sha(_path(root, manifest_ref, field=f"{task_id}.manifest")),
        "group_manifest_artifact_hash": manifest_hash,
        "success_ref": success_ref,
        "success_sha256": _sha(_path(root, success_ref, field=f"{task_id}.success")),
        "success_artifact_hash": str(success["artifact_hash"]),
        "source_refs": list(source_tuple),
        "commit_refs": commit_refs,
        "commit_artifact_hashes": hashes,
    }


def _gate_payload(
    *,
    index_ref: str,
    exit_evidence: object,
    groups: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    evidence_refs = (index_ref, *[ref for group in groups.values() for ref in group["commit_refs"].values()])
    measured = {
        "index_ref": index_ref,
        "index_sha256": exit_evidence.index_sha256,
        "index_artifact_hash": exit_evidence.index_artifact_hash,
        "producer_commit": exit_evidence.producer_commit,
        "execution_commit": exit_evidence.execution_commit,
        "consumer_commit": STAGE1_G1_EXIT_PRODUCER_COMMIT,
        "task_groups": dict(groups),
        "commit_count": len(evidence_refs) - 1,
    }
    gate = GateRecord(
        gate_id=GATE_ID,
        stage=1,
        status=GateStatus.PASS,
        checked_at="2026-08-21T00:00:00Z",
        measured=measured,
        threshold={
            "required_status": "PASS",
            "required_index_schema": canonical_stage1.S111_R4_INDEX_SCHEMA if hasattr(canonical_stage1, "S111_R4_INDEX_SCHEMA") else "stage1-s1-11-formalization-index-v1",
            "required_task_groups": [S110_TASK_ID, STAGE1_G1_EXIT_TASK_ID],
            "required_commit_count": 7,
        },
        evidence_refs=evidence_refs,
    )
    return gate.to_dict()


def publish_stage1_exit_gate(
    *,
    repository_root: str | Path,
    data_root: str | Path,
    output_dir: str = S111_OUTPUT,
) -> dict[str, object]:
    """Validate released Stage 1, then publish and strictly reread one Gate commit."""

    repository = Path(repository_root).resolve()
    root = Path(data_root).resolve()
    if not repository.is_dir() or not root.is_dir():
        raise Stage1ExitGateAdapterError("ROOT_INVALID")
    # Both calls are intentionally mandatory: the generic consumer contract
    # and the released producer loaders protect different identity surfaces.
    exit_evidence = validate_stage1_exit_evidence(root, INDEX_REF)
    canonical_stage1._emit_load_r4(
        repository=repository, evidence_root=root, evidence_ref=INDEX_REF, approved_data_root=root,
    )
    canonical_stage1._emit_load_s110(
        repository=repository, evidence_root=root, approved_data_root=root,
    )
    loaded_groups: dict[str, dict[str, object]] = {}
    for task_id, group_output, kinds in S1_GROUPS:
        _, metadata = _load_group(root, task_id=task_id, output_dir=group_output, kinds=kinds)
        loaded_groups[task_id] = metadata
    payload = _gate_payload(index_ref=INDEX_REF, exit_evidence=exit_evidence, groups=loaded_groups)
    refs = tuple(payload["evidence_refs"])
    store = TaskArtifactStore(root, output_dir)
    commit_ref = f"{store.output_dir}/commits/{ARTIFACT_KIND}.json"
    existing_path = _path(root, commit_ref, field="gate_record.commit")
    if existing_path.exists():
        loaded = load_committed_task_artifact(root, commit_ref, require_formal=True)
        if loaded.identity.task_id != STAGE1_G1_EXIT_TASK_ID or loaded.identity.artifact_kind != ARTIFACT_KIND or loaded.source_refs != refs or dict(loaded.payload) != payload:
            raise Stage1ExitGateAdapterError("EXISTING_GATE_IDENTITY_DRIFT")
        return {"commit_ref": commit_ref, "artifact_hash": loaded.identity.artifact_hash, "reused": True, "gate_record": dict(loaded.payload)}
    published = store.publish(
        task_id=STAGE1_G1_EXIT_TASK_ID,
        artifact_kind=ARTIFACT_KIND,
        config_hash=canonical_json_hash({"index_ref": INDEX_REF, "index_artifact_hash": exit_evidence.index_artifact_hash, "commit_artifact_hashes": {task: data["commit_artifact_hashes"] for task, data in loaded_groups.items()}}),
        run_intent="formal",
        payload=payload,
        formal_eligible=True,
        source_refs=refs,
    )
    reread = load_committed_task_artifact(root, published.commit_ref, require_formal=True)
    if dict(reread.payload) != payload or reread.source_refs != refs:
        raise Stage1ExitGateAdapterError("GATE_READBACK_INVALID")
    return {"commit_ref": published.commit_ref, "artifact_hash": published.artifact_hash, "reused": False, "gate_record": dict(reread.payload)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", default=S111_OUTPUT)
    args = parser.parse_args(argv)
    print(publish_stage1_exit_gate(repository_root=args.repository, data_root=args.data_root, output_dir=args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
