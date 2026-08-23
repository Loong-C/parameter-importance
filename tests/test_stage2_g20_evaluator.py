"""Formal, independent G2.0 evaluator contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.preregistration import (
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
)
from param_importance_nlp.experiments.stage2_g20_evaluator import (
    ARTIFACT_KINDS,
    TASK_ID,
    evaluate_formal_g20,
)
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT / "docs/mathematics.md"
STAGE1 = ROOT / "reports/stage1/cpu-evidence-20260814-s12-r2/stage1.11_reporting_and_exit_gate/stage_report.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, candidate_status: str = "NOT_RUN", local: bool = False):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "reports/stage1/cpu-evidence-20260814-s12-r2/stage1.11_reporting_and_exit_gate").mkdir(parents=True)
    shutil.copy2(MATH, root / "docs/mathematics.md")
    shutil.copy2(STAGE1, root / "reports/stage1/cpu-evidence-20260814-s12-r2/stage1.11_reporting_and_exit_gate/stage_report.json")
    (root / "plan/stage2").mkdir(parents=True)
    shutil.copy2(ROOT / "plan/stage2/01_scope_hypotheses_and_preregistration.md", root / "plan/stage2/01_scope_hypotheses_and_preregistration.md")
    (root / "evidence").mkdir()
    write_canonical_json(root / "evidence/upstream.json", {"schema_version": "upstream-v1", "ok": True})
    config_hash = "a" * 64
    prereg = build_stage2_preregistration(
        seed_plan_hash="b" * 64,
        producer_commit="c" * 40,
        mathematics_hash=_sha(root / "docs/mathematics.md"),
        stage1_report_hash=_sha(root / "reports/stage1/cpu-evidence-20260814-s12-r2/stage1.11_reporting_and_exit_gate/stage_report.json"),
        upstream_binding_hash="d" * 64,
        scope="formal",
    )
    hypothesis = build_stage2_hypothesis_contract(prereg, upstream_binding_hash="d" * 64)
    candidate = {
        "schema_version": "stage23-task-gate-candidate-v1",
        "task_id": TASK_ID,
        "gate_ids": ["stage1.G1-EXIT"],
        "gate_status": candidate_status,
        "local_validation_status": "NOT_RUN" if not local else "PASS",
        "formal_eligible": False,
        "reason": "formal_gate_requires_independent_review",
        "gate_id": "stage2.G2.0",
        "preregistration_hash": prereg["preregistration_hash"],
        "hypothesis_contract_hash": hypothesis["hypothesis_contract_hash"],
        "quality_gate_status": "NOT_RUN",
        "sample_generation_status": "FORBIDDEN_UNTIL_COMMITTED",
    }
    store = TaskArtifactStore(root, "runs/stage2-01")
    refs = {}
    for kind, payload in (("preregistration", prereg), ("hypothesis_contract", hypothesis), ("gate_record", candidate)):
        refs[kind] = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=config_hash,
            run_intent="local_fixture" if local else "formal",
            payload=payload,
            formal_eligible=not local,
            source_refs=("evidence/upstream.json",),
        ).commit_ref
    return root, refs


def test_real_three_commit_formal_fixture_publishes_gate_record(tmp_path: Path) -> None:
    root, refs = _fixture(tmp_path)
    result = evaluate_formal_g20(root, refs, output_root=root / "evaluations")
    assert result["status"] == "PASS"
    assert result["formal_eligible"] is True
    assert isinstance(result["commit_ref"], str)
    loaded = load_committed_task_artifact(root, result["commit_ref"], require_formal=True)  # type: ignore[arg-type]
    gate = GateRecord.from_mapping(dict(loaded.payload))
    assert gate.gate_id == "stage2.G2.0"
    assert gate.status is GateStatus.PASS
    assert loaded.source_refs == tuple(refs[kind] for kind in ARTIFACT_KINDS)
    assert loaded.payload["measured"]["runner_candidate_role"] == "non_self_signed_lineage_only"  # type: ignore[index]


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda refs: (refs["preregistration"], refs["preregistration"], refs["gate_record"]), "BLOCKED"),
        (lambda refs: (refs["preregistration"], refs["hypothesis_contract"], "wrong/path.json"), "BLOCKED"),
    ],
)
def test_missing_duplicate_and_wrong_refs_fail_closed(tmp_path: Path, mutator, expected: str) -> None:
    root, refs = _fixture(tmp_path)
    result = evaluate_formal_g20(root, mutator(refs), output_root=root / "evaluations")
    assert result["status"] == expected
    assert result["formal_eligible"] is False
    assert result["commit_ref"] is None


def test_mixed_config_and_local_fixture_are_rejected(tmp_path: Path) -> None:
    root, refs = _fixture(tmp_path)
    original_gate = TaskArtifactStore(root, "runs/stage2-01").load_commit(refs["gate_record"])
    original_payload = load_canonical_json(root / original_gate.object_ref)["payload"]
    published = TaskArtifactStore(root, "runs/other").publish(
        task_id=TASK_ID,
        artifact_kind="gate_record",
        config_hash="e" * 64,
        run_intent="formal",
        formal_eligible=True,
        source_refs=("evidence/upstream.json",),
        payload=original_payload,  # type: ignore[arg-type]
    )
    result = evaluate_formal_g20(root, (refs["preregistration"], refs["hypothesis_contract"], published.commit_ref))
    assert result["status"] == "BLOCKED"
    local_root, local_refs = _fixture(tmp_path / "local", local=True)
    local_result = evaluate_formal_g20(local_root, local_refs, output_root=local_root / "evaluations")
    assert local_result["status"] == "BLOCKED"


def test_runner_candidate_pass_is_not_a_numeric_gate(tmp_path: Path) -> None:
    root, refs = _fixture(tmp_path, candidate_status="PASS")
    result = evaluate_formal_g20(root, refs, output_root=root / "evaluations")
    assert result["status"] == "FAIL"
    assert result["formal_eligible"] is False
    assert "SELF_SIGNED" in " ".join(result["gate_record"]["reasons"])  # type: ignore[index]


def test_payload_and_document_tamper_are_rejected(tmp_path: Path) -> None:
    root, refs = _fixture(tmp_path)
    commit = TaskArtifactStore(root, "runs/stage2-01").load_commit(refs["hypothesis_contract"])
    object_path = root / commit.object_ref
    payload = json.loads(object_path.read_text(encoding="utf-8"))
    payload["payload"]["hypotheses"][0]["claim"] = "tampered"
    object_path.write_text(json.dumps(payload), encoding="utf-8")
    result = evaluate_formal_g20(root, refs, output_root=root / "evaluations")
    assert result["status"] == "BLOCKED"
    assert result["commit_ref"] is None

    root2, refs2 = _fixture(tmp_path / "docs")
    (root2 / "docs/mathematics.md").write_text("tampered\n", encoding="utf-8")
    result2 = evaluate_formal_g20(root2, refs2, output_root=root2 / "evaluations")
    assert result2["status"] == "BLOCKED"
    assert isinstance(result2["commit_ref"], str)
    blocked = load_committed_task_artifact(root2, result2["commit_ref"], require_formal=True)  # type: ignore[arg-type]
    assert GateRecord.from_mapping(dict(blocked.payload)).status is GateStatus.BLOCKED

    root3, refs3 = _fixture(tmp_path / "plan")
    (root3 / "plan/stage2/01_scope_hypotheses_and_preregistration.md").write_text("tampered\n", encoding="utf-8")
    result3 = evaluate_formal_g20(root3, refs3, output_root=root3 / "evaluations")
    assert result3["status"] == "BLOCKED"
