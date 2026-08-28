"""Regression tests for append-only Stage 3 execution-evidence lineage."""

from __future__ import annotations

from types import SimpleNamespace

from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
)
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.stage23_task_runners import (
    _stage3_execution_is_append_only_extension,
    _stage3_execution_matches_or_extends_plan,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def _hash(label: str) -> str:
    return canonical_json_hash({"label": label})


def _gate(index: int, *, label: str | None = None) -> GateRecord:
    return GateRecord(
        gate_id=f"stage3.G3-{index}",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:00:00Z",
        measured={"identity": label or f"g{index}"},
        threshold={"required": True},
        evidence_refs=(f"evidence/stage3/g3-{index}.json",),
    )


def _execution(*gates: GateRecord, metadata: str = "same") -> FormalExecutionEvidence:
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_hash("contract"),
        asset_manifest_hashes=(_hash("model"), _hash("data")),
        prerequisite_gates=gates,
        metadata={"run_family": metadata},
    )


def test_plan_execution_allows_only_content_bound_gate_extension(tmp_path) -> None:
    base = _execution(*(_gate(index) for index in range(4)))
    current = _execution(*(_gate(index) for index in range(5)))
    assert _stage3_execution_is_append_only_extension(base, current)

    execution_commit = TaskArtifactStore(tmp_path, "authority/base-execution").publish(
        task_id="stage3.formal_execution_authority",
        artifact_kind="formal_execution_evidence",
        config_hash=_hash("base-config"),
        run_intent="formal",
        payload=base.to_dict(),
        formal_eligible=True,
    )
    plan_commit = TaskArtifactStore(tmp_path, "authority/pilot-plan").publish(
        task_id="stage3.formal_plan_authority",
        artifact_kind="stage3_formal_plan",
        config_hash=_hash("plan-config"),
        run_intent="formal",
        payload={"schema_version": "test-stage3-plan-v1"},
        formal_eligible=True,
        source_refs=(execution_commit.commit_ref,),
    )
    request = SimpleNamespace(
        environment=SimpleNamespace(evidence_refs={})
    )
    assert _stage3_execution_matches_or_extends_plan(
        request,  # type: ignore[arg-type]
        tmp_path,
        current=current,
        declared_hash=base.artifact_hash,
        plan_ref=plan_commit.commit_ref,
        plan_kind="pilot",
    )

    changed_old_gate = _execution(
        _gate(0),
        _gate(1, label="changed"),
        _gate(2),
        _gate(3),
        _gate(4),
    )
    assert not _stage3_execution_is_append_only_extension(base, changed_old_gate)
    changed_metadata = _execution(
        *(_gate(index) for index in range(5)), metadata="drifted"
    )
    assert not _stage3_execution_is_append_only_extension(base, changed_metadata)


def test_plan_execution_rejects_bare_hash_without_original_commit(tmp_path) -> None:
    base = _execution(*(_gate(index) for index in range(4)))
    current = _execution(*(_gate(index) for index in range(5)))
    plan_commit = TaskArtifactStore(tmp_path, "authority/unbound-plan").publish(
        task_id="stage3.formal_plan_authority",
        artifact_kind="stage3_formal_plan",
        config_hash=_hash("unbound-plan-config"),
        run_intent="formal",
        payload={"schema_version": "test-stage3-plan-v1"},
        formal_eligible=True,
    )
    request = SimpleNamespace(
        environment=SimpleNamespace(evidence_refs={})
    )
    assert not _stage3_execution_matches_or_extends_plan(
        request,  # type: ignore[arg-type]
        tmp_path,
        current=current,
        declared_hash=base.artifact_hash,
        plan_ref=plan_commit.commit_ref,
        plan_kind="pilot",
    )
