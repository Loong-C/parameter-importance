"""Publish the explicit Stage 3 scope authority as formal task envelopes."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..contracts.jsonio import JSONValue, load_canonical_json
from ..contracts.stage3_scope import (
    STAGE3_SCOPE_ARTIFACT_KIND,
    STAGE3_SCOPE_TASK_ID,
    validate_stage3_scope_authority,
    validate_stage3_scope_decision,
)
from ..contracts.status import GateRecord
from ..runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


def _workspace_ref(root: Path, value: str | Path, *, field: str) -> str:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"STAGE3_SCOPE_SOURCE_OUTSIDE_WORKSPACE:{field}") from error


def publish_stage3_scope_authority(
    *,
    workspace_root: str | Path,
    output_dir: str,
    decision_path: str | Path,
    gate_path: str | Path,
    config_hash: str,
) -> Mapping[str, JSONValue]:
    """Wrap the reviewed user decision and a rebound G3-0 Gate in commits.

    The source Gate remains untouched.  The published Gate copies its measured
    facts and binds the new decision commit, which is the reference consumed by
    formal runtime preflight.
    """

    root = Path(workspace_root).resolve()
    decision_ref = _workspace_ref(root, decision_path, field="decision_path")
    source_gate_ref = _workspace_ref(root, gate_path, field="gate_path")
    decision_value = load_canonical_json(root / Path(decision_ref))
    gate_value = load_canonical_json(root / Path(source_gate_ref))
    if not isinstance(decision_value, Mapping) or not isinstance(gate_value, Mapping):
        raise TypeError("STAGE3_SCOPE_SOURCE_NOT_OBJECT")
    validate_stage3_scope_decision(decision_value)
    source_gate = GateRecord.from_mapping(dict(gate_value))
    validate_stage3_scope_authority(
        decision_value,
        source_gate,
        decision_ref=decision_ref,
    )

    store = TaskArtifactStore(root, output_dir)
    decision = store.publish(
        task_id=STAGE3_SCOPE_TASK_ID,
        artifact_kind=STAGE3_SCOPE_ARTIFACT_KIND,
        config_hash=config_hash,
        run_intent="formal",
        payload=decision_value,  # type: ignore[arg-type]
        formal_eligible=True,
        source_refs=(decision_ref,),
    )
    rebound_gate = GateRecord(
        gate_id=source_gate.gate_id,
        stage=source_gate.stage,
        status=source_gate.status,
        checked_at=source_gate.checked_at,
        measured=source_gate.measured,
        threshold=source_gate.threshold,
        evidence_refs=(decision.commit_ref, decision_ref, source_gate_ref),
        reasons=source_gate.reasons,
        conditions=source_gate.conditions,
        expires_at=source_gate.expires_at,
    )
    gate = store.publish(
        task_id=STAGE3_SCOPE_TASK_ID,
        artifact_kind="gate_record",
        config_hash=config_hash,
        run_intent="formal",
        payload=rebound_gate.to_dict(),
        formal_eligible=True,
        source_refs=(source_gate_ref, decision.commit_ref),
    )
    loaded_decision = load_committed_task_artifact(
        root,
        decision.commit_ref,
        require_formal=True,
    )
    loaded_gate = load_committed_task_artifact(
        root,
        gate.commit_ref,
        require_formal=True,
    )
    parsed_gate = GateRecord.from_mapping(dict(loaded_gate.payload))
    validate_stage3_scope_authority(
        loaded_decision.payload,
        parsed_gate,
        decision_ref=decision.commit_ref,
    )
    return {
        "schema_version": "stage3-scope-authority-publication-v1",
        "task_id": STAGE3_SCOPE_TASK_ID,
        "config_hash": config_hash,
        "decision_commit_ref": decision.commit_ref,
        "decision_artifact_hash": decision.artifact_hash,
        "gate_commit_ref": gate.commit_ref,
        "gate_artifact_hash": gate.artifact_hash,
        "source_decision_ref": decision_ref,
        "source_gate_ref": source_gate_ref,
        "stage2_artifacts_relabelled": False,
    }


__all__ = ["publish_stage3_scope_authority"]
