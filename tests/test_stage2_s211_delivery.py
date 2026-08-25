from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s211_delivery import S211DeliveryBlocked, run_s211_g28


TASKS = [
    "stage2.01_scope_hypotheses_and_preregistration",
    "stage2.02_stage1_handoff_and_fixed_state_contract",
    "stage2.03_assets_checkpoints_and_sampling",
    "stage2.04_reference_target",
    "stage2.05_paired_estimator_runner",
    "stage2.06_pilot_and_matrix_freeze",
    "stage2.07_main_sweep",
    "stage2.08_statistics_and_robustness",
    "stage2.09_cost_and_system_validation",
    "stage2.10_visualization_reporting_and_decision",
]


def _hashed(body: dict[str, object]) -> dict[str, object]:
    return body | {"artifact_hash": canonical_json_hash(body)}


def _gate() -> dict[str, object]:
    return GateRecord(
        gate_id="stage2.G2.7b",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00Z",
        measured={"decision_hash": "0" * 64},
        threshold={},
        evidence_refs=("decision.json",),
    ).to_dict()


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    decision = _hashed({
        "schema_version": "estimator-decision-v1",
        "selected_estimator": "u",
        "scope": "formal",
        "status": "SELECTED",
        "gate_status": "PASS",
    })
    gate = GateRecord(
        gate_id="stage2.G2.7b", stage=2, status="PASS",
        checked_at="2026-08-25T00:00:00Z",
        measured={"decision_hash": decision["artifact_hash"]}, threshold={},
        evidence_refs=("decision.json",),
    ).to_dict()
    lineage = _hashed({
        "schema_version": "stage2-s211-lineage-v1",
        "tasks": {
            task: {"task_id": task, "status": "PASS", "formal_eligible": True, "artifact_refs": [task + ".json"]}
            for task in TASKS
        },
    })
    boundaries = {
        role: _hashed({"schema_version": f"{role}-v1", "status": "PASS", "formal_eligible": True})
        for role in ("environment", "assets", "reference", "pilot", "formal_14m", "formal_31m", "analysis", "decision")
    }
    replay = _hashed({
        "schema_version": "stage2-s211-replay-audit-v1",
        "audit_type": "confirmatory_31m_repetition",
        "model": "Pythia-31M",
        "status": "PASS",
        "formal_eligible": True,
        "replay_executed": True,
        "equivalent": True,
        "repetition_id": "confirmatory-r0001",
        "source_result_hash": "1" * 64,
        "replay_result_hash": "1" * 64,
        "source_artifact_hash": boundaries["formal_31m"]["artifact_hash"],
    })
    return gate, decision, lineage, boundaries | {"replay_audit_31m": replay}


def test_s211_missing_refs_publishes_blocked_append_only_bundle(tmp_path: Path) -> None:
    result = run_s211_g28(
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
        stage2_lineage={},
    )
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert (tmp_path / "delivery" / "g2.8-gate.json").exists()
    with pytest.raises(S211DeliveryBlocked, match="OUTPUT_ROOT_MUST_BE_NEW"):
        run_s211_g28(output_root=tmp_path / "delivery", producer_commit="a" * 40)


def test_s211_pass_requires_all_boundaries_and_replay(tmp_path: Path) -> None:
    gate, decision, lineage, values = _inputs()
    boundaries = {key: value for key, value in values.items() if key != "replay_audit_31m"}
    # The gate fixture is created after the decision so its measured hash is bound.
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs=boundaries,
        replay_audit_31m=values["replay_audit_31m"],
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["gate"]["status"] == "PASS"
    manifest = load_canonical_json(tmp_path / "delivery" / "delivery_manifest.json")
    assert manifest["boundary_hashes"]["formal_31m"] == boundaries["formal_31m"]["artifact_hash"]
    assert load_canonical_json(tmp_path / "delivery" / "replay_instructions.json")["status"] == "READY"


def test_s211_tampered_g27b_hash_blocks_without_overwriting(tmp_path: Path) -> None:
    gate, decision, lineage, values = _inputs()
    forged = dict(gate)
    forged["status"] = "BLOCKED"
    result = run_s211_g28(
        g27b_gate=forged,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert any("ARTIFACT_HASH_MISMATCH" in reason for reason in result["delivery_manifest"]["reasons"])


def test_s211_replay_cannot_be_inferred_from_31m_boundary(tmp_path: Path) -> None:
    gate, decision, lineage, values = _inputs()
    boundaries = {key: value for key, value in values.items() if key != "replay_audit_31m"}
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs=boundaries,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert result["replay_audit_31m"]["replay_executed"] is False
    assert "REPLAY_AUDIT_MISSING" in result["delivery_manifest"]["reasons"]
