from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s211_delivery import S211DeliveryBlocked, run_s211_g28
from param_importance_nlp.experiments.stage2 import EstimatorDecision


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


def _inputs(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    decision = _hashed({
        "schema_version": "estimator-decision-v1",
        "decision_id": "decision-formal-r0001",
        "selected_estimator": "u",
        "scope": "formal",
        "status": "SELECTED",
        "state": "SELECTED",
        "batch_size": 8,
        "microbatch_count": 2,
        "repetitions": 3,
        "gate_id": "stage2.G2.7b",
        "gate_status": "PASS",
        "artifact_ref": "decision.json",
        "metadata": {},
    })
    gate = GateRecord(
        gate_id="stage2.G2.7b", stage=2, status="PASS",
        checked_at="2026-08-25T00:00:00Z",
        measured={"decision_hash": decision["artifact_hash"]}, threshold={},
        evidence_refs=("decision.json",),
    ).to_dict()
    lineage_tasks = {
        task: _hashed({"task_id": task, "status": "PASS", "formal_eligible": True, "artifact_refs": [task + ".json"]})
        for task in TASKS
    }
    lineage = _hashed({
        "schema_version": "stage2-s211-lineage-v1",
        "tasks": lineage_tasks,
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
    upstream: dict[str, object] = {}
    for gate_id in ("stage2.G2.0", "stage2.G2.1", "stage2.G2.2", "stage2.G2.3", "stage2.G2.4a", "stage2.G2.4b", "stage2.G2.5", "stage2.G2.6", "stage2.G2.7a"):
        upstream[gate_id] = GateRecord(
            gate_id=gate_id,
            stage=2,
            status="PASS",
            checked_at="2026-08-25T00:00:00Z",
            measured={},
            threshold={},
            evidence_refs=("evidence.json",),
        ).to_dict()
    gate_summary_gates = {gate_id: {"status": "PASS"} for gate_id in upstream}
    gate_summary_gates["stage2.G2.7b"] = {"status": "PASS"}
    large = tmp_path / "large-artifact.bin"
    large.write_bytes(b"formal-large-artifact")
    agent_docs_root = tmp_path / "agent-docs"
    agent_docs_root.mkdir()
    agent_document_refs = {}
    agent_documents = {}
    for document_name in ("git.md", "local.md", "remote_access.md", "server.md", "sync.md", "worklogs.md"):
        document = agent_docs_root / document_name
        document.write_bytes(f"fixture-agent-document:{document_name}\n".encode())
        agent_document_refs[document_name] = f"agent-docs/{document_name}"
        agent_documents[document_name] = hashlib.sha256(document.read_bytes()).hexdigest()
    role_bodies = {
        "plan": {"schema_version": "stage2-s211-plan-v1", "task_id": "stage2.11_delivery_and_exit_gate", "gate_id": "stage2.G2.8", "status": "PASS", "formal_eligible": True},
        "task_catalog": {"schema_version": "task-catalog-v1", "task_id": "stage2.11_delivery_and_exit_gate", "status": "PASS", "formal_eligible": True, "outputs": ["delivery_manifest", "replay_report", "gate_summary", "sync_report"], "gates": ["stage2.G2.7b"]},
        "replay_report": {"schema_version": "stage2-s211-replay-report-v1", "status": "PASS", "formal_eligible": True, "replay_executed": True, "equivalent": True},
        "gate_summary": {"schema_version": "stage2-s211-gate-summary-v1", "status": "PASS", "formal_eligible": True, "gates": gate_summary_gates},
        "sync_report": {"schema_version": "stage2-s211-sync-report-v1", "status": "PASS", "formal_eligible": True, "agent_documents": agent_documents, "agent_document_refs": agent_document_refs, "worktree_clean": True, "server_worktree_clean": True, "user_files_excluded": True, "github_commit": "b" * 40, "server_commit": "c" * 40, "target_execution_commit": "a" * 40},
        "estimator_decision": decision,
        "large_artifact_index": {"schema_version": "stage2-s211-large-artifact-index-v1", "status": "PASS", "formal_eligible": True, "entries": [{"path": "large-artifact.bin", "sha256": hashlib.sha256(large.read_bytes()).hexdigest()}]},
        "worklog": {"schema_version": "stage2-s211-worklog-v1", "status": "PASS", "formal_eligible": True, "entries": [{"event": "delivery"}]},
        "dirty_head_evidence": {"schema_version": "stage2-s211-dirty-head-v1", "status": "PASS", "formal_eligible": True, "worktree_clean_for_delivery": True, "excluded_files": ["presentation/user.pdf"]},
        "failure_retry_amendment_history": {"schema_version": "stage2-s211-history-v1", "status": "PASS", "formal_eligible": True, "failures": [], "retries": [], "amendments": []},
    }
    roles = {name: _hashed(body) if "artifact_hash" not in body else body for name, body in role_bodies.items()}
    return gate, decision, lineage, boundaries | {"replay_audit_31m": replay}, upstream, roles


def test_s211_missing_refs_publishes_blocked_append_only_bundle(tmp_path: Path) -> None:
    result = run_s211_g28(
        data_root=tmp_path,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
        stage2_lineage={},
    )
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert (tmp_path / "delivery" / "g2.8-gate.json").exists()
    assert result["delivery_roles"]["replay_report.json"]["schema_version"] == "stage2-s211-replay-report-v1"
    with pytest.raises(S211DeliveryBlocked, match="OUTPUT_ROOT_MUST_BE_NEW"):
        run_s211_g28(data_root=tmp_path, output_root=tmp_path / "delivery", producer_commit="a" * 40)


def test_s211_pass_requires_all_boundaries_and_replay(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    boundaries = {key: value for key, value in values.items() if key != "replay_audit_31m"}
    # The gate fixture is created after the decision so its measured hash is bound.
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs=boundaries,
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["gate"]["status"] == "PASS"
    manifest = load_canonical_json(tmp_path / "delivery" / "delivery_manifest.json")
    assert manifest["boundary_hashes"]["formal_31m"] == boundaries["formal_31m"]["artifact_hash"]
    assert load_canonical_json(tmp_path / "delivery" / "replay_instructions.json")["status"] == "READY"
    replay_report = load_canonical_json(tmp_path / "delivery" / "replay_report.json")
    assert replay_report["status"] == "PASS"
    assert EstimatorDecision.from_mapping(load_canonical_json(tmp_path / "delivery" / "estimator_decision.json")).formal_eligible
    assert load_canonical_json(tmp_path / "delivery" / "output_inventory.json")["files"]


def test_s211_tampered_g27b_hash_blocks_without_overwriting(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    forged = dict(gate)
    forged["status"] = "BLOCKED"
    result = run_s211_g28(
        g27b_gate=forged,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert any("ARTIFACT_HASH_MISMATCH" in reason for reason in result["delivery_manifest"]["reasons"])


def test_s211_replay_cannot_be_inferred_from_31m_boundary(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    boundaries = {key: value for key, value in values.items() if key != "replay_audit_31m"}
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs=boundaries,
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert result["replay_audit_31m"]["replay_executed"] is False
    assert "REPLAY_AUDIT_MISSING" in result["delivery_manifest"]["reasons"]


def test_s211_lineage_missing_formal_eligibility_blocks(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    row = dict(lineage["tasks"][TASKS[0]])
    row.pop("formal_eligible")
    lineage["tasks"][TASKS[0]] = _hashed({key: value for key, value in row.items() if key != "artifact_hash"})
    lineage = _hashed({key: value for key, value in lineage.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert f"STAGE2_LINEAGE_NOT_FORMAL:{TASKS[0]}" in result["delivery_manifest"]["reasons"]


def test_s211_missing_or_blocked_upstream_gate_cannot_be_replaced_by_g28(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    missing = dict(upstream)
    missing.pop("stage2.G2.3")
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=missing, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert "UPSTREAM_GATE_MISSING:stage2.G2.3" in result["delivery_manifest"]["reasons"]
    blocked_gate = GateRecord(
        gate_id="stage2.G2.3", stage=2, status="BLOCKED",
        checked_at="2026-08-25T00:00:00Z", measured={}, threshold={},
        evidence_refs=("evidence.json",), reasons=("upstream-blocked",),
    ).to_dict()
    blocked = dict(upstream)
    blocked["stage2.G2.3"] = blocked_gate
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=blocked, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert "UPSTREAM_GATE_NOT_PASS:stage2.G2.3" in result["delivery_manifest"]["reasons"]


def test_s211_path_escape_and_large_artifact_tamper_block(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    escaped = dict(values["formal_31m"])
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={**{key: value for key, value in values.items() if key != "replay_audit_31m"}, "formal_31m": r"D:\outside\formal.json"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert any("boundary.formal_31m:DATA_ROOT_PATH_ESCAPE" in reason for reason in result["delivery_manifest"]["reasons"])
    large_path = tmp_path / roles["large_artifact_index"]["entries"][0]["path"]
    large_path.write_bytes(b"tampered")
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert "LARGE_ARTIFACT_INDEX_SHA_MISMATCH:0" in result["delivery_manifest"]["reasons"]


def test_s211_missing_delivery_role_blocks_stage3_handoff(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    roles = dict(roles)
    roles.pop("sync_report")
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "DELIVERY_ROLE_MISSING:sync_report" in result["delivery_manifest"]["reasons"]


def test_s211_sync_report_recomputes_agent_document_sha(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    roles = dict(roles)
    sync = dict(roles["sync_report"])
    sync["agent_documents"] = dict(sync["agent_documents"])
    sync["agent_documents"]["git.md"] = "0" * 64
    roles["sync_report"] = _hashed({key: value for key, value in sync.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles, producer_commit="a" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "AGENT_DOCUMENT_SHA_MISMATCH:git.md" in result["delivery_manifest"]["reasons"]


def test_s211_sync_report_missing_legacy_and_escape_refs_block(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    base_sync = dict(roles["sync_report"])
    sync = dict(base_sync)
    sync["agent_document_refs"] = dict(sync["agent_document_refs"])
    sync["agent_document_refs"]["git.md"] = "agent-docs/missing-git.md"
    roles_missing = dict(roles)
    roles_missing["sync_report"] = _hashed({key: value for key, value in sync.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles_missing, producer_commit="a" * 40,
    )
    assert "AGENT_DOCUMENT_REF_INVALID:git.md" in result["delivery_manifest"]["reasons"]

    escaped_sync = dict(base_sync)
    escaped_sync["agent_document_refs"] = dict(escaped_sync["agent_document_refs"])
    escaped_sync["agent_document_refs"]["git.md"] = r"D:\outside\git.md"
    roles_escaped = dict(roles)
    roles_escaped["sync_report"] = _hashed({key: value for key, value in escaped_sync.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles_escaped, producer_commit="a" * 40,
    )
    assert "AGENT_DOCUMENT_REF_INVALID:git.md" in result["delivery_manifest"]["reasons"]

    legacy_sync = dict(base_sync)
    legacy_sync["agent_documents"] = dict(legacy_sync["agent_documents"])
    legacy_sync["agent_document_refs"] = dict(legacy_sync["agent_document_refs"])
    legacy_sync["agent_documents"]["local_temp.md"] = "0" * 64
    legacy_sync["agent_document_refs"]["local_temp.md"] = "agent-docs/local_temp.md"
    roles_legacy = dict(roles)
    roles_legacy["sync_report"] = _hashed({key: value for key, value in legacy_sync.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate, g27b_decision=decision, stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"], data_root=tmp_path,
        predecessor_gates=upstream, delivery_refs=roles_legacy, producer_commit="a" * 40,
    )
    assert "LEGACY_LOCAL_TEMP_MUST_NOT_BE_BOUND" in result["delivery_manifest"]["reasons"]


def test_s211_accepts_s210_producer_gate_and_task_output_wrapper(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    producer_body = dict(gate)
    producer_body["schema_version"] = "stage2-s210-g27b-gate-v1"
    producer_body["task_id"] = "stage2.10_visualization_reporting_and_decision"
    producer_body["formal_eligible"] = True
    producer_gate = _hashed({key: value for key, value in producer_body.items() if key != "artifact_hash"})
    wrapper = _hashed({
        "schema_version": "task-output-artifact-v1",
        "task_id": "stage2.10_visualization_reporting_and_decision",
        "artifact_kind": "g27b_gate",
        "config_hash": "d" * 64,
        "run_intent": "formal",
        "formal_eligible": True,
        "source_refs": ["s210/g2.7b-gate.json"],
        "payload": producer_gate,
    })
    result = run_s211_g28(
        g27b_gate=wrapper,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["delivery_manifest"]["g27b_gate_hash"] == producer_gate["artifact_hash"]


def test_s211_relative_refs_are_rooted_at_data_root(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    write_canonical_json(tmp_path / "g27b.json", gate)
    write_canonical_json(tmp_path / "decision.json", decision)
    result = run_s211_g28(
        g27b_gate=Path("g27b.json"),
        g27b_decision=Path("decision.json"),
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert not any(reason.startswith("g27b_gate:") for reason in result["delivery_manifest"]["reasons"])
    assert not any(reason.startswith("g27b_decision:") for reason in result["delivery_manifest"]["reasons"])


def test_s211_replay_source_hash_is_required(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    replay = dict(values["replay_audit_31m"])
    replay.pop("source_artifact_hash")
    replay = _hashed({key: value for key, value in replay.items() if key != "artifact_hash"})
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=replay,
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "REPLAY_31M_SOURCE_HASH_MISSING" in result["delivery_manifest"]["reasons"]


def test_s211_duplicate_lineage_and_role_schema_mismatch_block(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    lineage_body = {key: value for key, value in lineage.items() if key != "artifact_hash"}
    lineage_body["tasks"] = list(lineage_body["tasks"].values()) + [
        dict(next(iter(lineage_body["tasks"].values())))
    ]
    duplicate_lineage = _hashed(lineage_body)
    broken_roles = dict(roles)
    broken_replay_report = dict(broken_roles["replay_report"])
    broken_replay_report["schema_version"] = "wrong-role-schema-v1"
    broken_roles["replay_report"] = _hashed({
        key: value for key, value in broken_replay_report.items() if key != "artifact_hash"
    })
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=duplicate_lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=broken_roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert f"STAGE2_LINEAGE_DUPLICATE:{TASKS[0]}" in result["delivery_manifest"]["reasons"]
    assert "DELIVERY_ROLE_SCHEMA_MISMATCH:replay_report" in result["delivery_manifest"]["reasons"]


def test_s211_unknown_upstream_gate_is_not_ignored(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    extra = GateRecord(
        gate_id="stage2.G9.9",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00Z",
        measured={},
        threshold={},
        evidence_refs=("evidence.json",),
    ).to_dict()
    upstream = dict(upstream)
    upstream["stage2.G9.9"] = extra
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "UPSTREAM_GATE_UNSUPPORTED:stage2.G9.9" in result["delivery_manifest"]["reasons"]


def test_s211_unknown_lineage_task_is_not_ignored(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    lineage_body = {key: value for key, value in lineage.items() if key != "artifact_hash"}
    lineage_body["tasks"] = dict(lineage_body["tasks"])
    lineage_body["tasks"]["stage2.99_unknown"] = _hashed({
        "task_id": "stage2.99_unknown",
        "status": "PASS",
        "formal_eligible": True,
        "artifact_refs": ["unknown.json"],
    })
    unknown_lineage = _hashed(lineage_body)
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=unknown_lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "STAGE2_LINEAGE_UNSUPPORTED_TASK:stage2.99_unknown" in result["delivery_manifest"]["reasons"]


def test_s211_lineage_mapping_key_must_match_task_id(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    lineage_body = {key: value for key, value in lineage.items() if key != "artifact_hash"}
    lineage_body["tasks"] = dict(lineage_body["tasks"])
    first_task = TASKS[0]
    first_entry = dict(lineage_body["tasks"][first_task])
    first_entry.pop("artifact_hash", None)
    first_entry["task_id"] = TASKS[1]
    lineage_body["tasks"][first_task] = _hashed(first_entry)
    mismatched_lineage = _hashed(lineage_body)
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=mismatched_lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "STAGE2_LINEAGE_TASK_KEY_MISMATCH" in result["delivery_manifest"]["reasons"]


def test_s211_replay_hash_mismatch_cannot_claim_equivalence(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    replay = dict(values["replay_audit_31m"])
    replay.pop("artifact_hash", None)
    replay["replay_result_hash"] = "2" * 64
    replay = _hashed(replay)
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=replay,
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "REPLAY_RESULT_HASH_MISMATCH" in result["delivery_manifest"]["reasons"]


def test_s211_accepts_actual_producer_specific_g23_g24a_g26_gates(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    cell_ids = (
        "pythia-14m:initialization",
        "pythia-14m:early",
        "pythia-14m:mid_late",
        "pythia-31m-deduped:initialization",
        "pythia-31m-deduped:early",
        "pythia-31m-deduped:mid_late",
    )
    g23 = _hashed({
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "gate_id": "stage2.G2.3",
        "status": "PASS",
        "formal_eligible": True,
        "required_cell_count": 6,
        "complete_cell_count": 6,
        "expected_cell_ids": list(cell_ids),
        "cells": [{"cell_id": cell_id, "status": "PASS"} for cell_id in cell_ids],
    })
    g24a = _hashed({
        "schema_version": "stage2-g24a-formal-evaluation-v1",
        "gate_id": "stage2.G2.4a",
        "status": "PASS",
        "formal_eligible": True,
        "cell_count": 6,
        "results": [
            {"cell_id": cell_id, "status": "PASS", "formal_eligible": True}
            for cell_id in cell_ids
        ],
        "rebind_plan_hash": "3" * 64,
        "g23_evaluation_hash": g23["artifact_hash"],
        "execution_commit": "d" * 40,
        "evidence_refs": ["evidence.json"],
        "confirmatory_draws_generated": False,
    })
    g26 = _hashed({
        "schema_version": "stage2-s208-g26-gate-v1",
        "gate_id": "stage2.G2.6",
        "stage": 2,
        "status": "PASS",
        "quality_gate_dependency": True,
        "measured": {"six_primary_cells": True},
        "threshold": {"frozen_thresholds": True},
        "reasons": [],
        "upstream_gate_hashes": {
            "stage2.G2.3": g23["artifact_hash"],
            "stage2.G2.4a": g24a["artifact_hash"],
            "stage2.G2.4b": upstream["stage2.G2.4b"]["artifact_hash"],
            "stage2.G2.5": upstream["stage2.G2.5"]["artifact_hash"],
        },
    })
    custom_upstream = dict(upstream)
    custom_upstream.update({"stage2.G2.3": g23, "stage2.G2.4a": g24a, "stage2.G2.6": g26})
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=custom_upstream,
        delivery_refs=roles,
        output_root=tmp_path / "delivery",
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["delivery_manifest"]["upstream_gate_hashes"]["stage2.G2.6"] == g26["artifact_hash"]


def test_s211_accepts_actual_s210_decision_schema(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    s210_decision = dict(decision)
    s210_decision["schema_version"] = "stage2-s210-estimator-decision-v1"
    s210_decision.pop("artifact_hash", None)
    s210_decision = _hashed(s210_decision)
    s210_gate = GateRecord(
        gate_id="stage2.G2.7b",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00Z",
        measured={"decision_hash": s210_decision["artifact_hash"]},
        threshold={},
        evidence_refs=("decision.json",),
    ).to_dict()
    roles = dict(roles)
    roles["estimator_decision"] = s210_decision
    result = run_s211_g28(
        g27b_gate=s210_gate,
        g27b_decision=s210_decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["delivery_roles"]["estimator_decision.json"]["schema_version"] == "stage2-s210-estimator-decision-v1"


def test_s211_cross_binds_g26_upstream_hashes(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    g26 = _hashed({
        "schema_version": "stage2-s208-g26-gate-v1",
        "gate_id": "stage2.G2.6",
        "stage": 2,
        "status": "PASS",
        "quality_gate_dependency": True,
        "measured": {"six_primary_cells": True},
        "threshold": {"frozen_thresholds": True},
        "reasons": [],
        "upstream_gate_hashes": {
            "stage2.G2.3": "0" * 64,
            "stage2.G2.4a": upstream["stage2.G2.4a"]["artifact_hash"],
            "stage2.G2.4b": upstream["stage2.G2.4b"]["artifact_hash"],
            "stage2.G2.5": upstream["stage2.G2.5"]["artifact_hash"],
        },
    })
    upstream = dict(upstream)
    upstream["stage2.G2.6"] = g26
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs={key: value for key, value in values.items() if key != "replay_audit_31m"},
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "G2.6_UPSTREAM_HASH_MISMATCH:stage2.G2.3" in result["delivery_manifest"]["reasons"]


def test_s211_explicit_local_fixture_marker_cannot_pass(tmp_path: Path) -> None:
    gate, decision, lineage, values, upstream, roles = _inputs(tmp_path)
    boundary = dict(values["formal_31m"])
    boundary.pop("artifact_hash", None)
    boundary["local_fixture"] = True
    boundary = _hashed(boundary)
    boundaries = {key: value for key, value in values.items() if key != "replay_audit_31m"}
    boundaries["formal_31m"] = boundary
    result = run_s211_g28(
        g27b_gate=gate,
        g27b_decision=decision,
        stage2_lineage=lineage,
        boundary_refs=boundaries,
        replay_audit_31m=values["replay_audit_31m"],
        data_root=tmp_path,
        predecessor_gates=upstream,
        delivery_refs=roles,
        producer_commit="a" * 40,
        consumer_commit="b" * 40,
    )
    assert result["status"] == "BLOCKED"
    assert "BOUNDARY_NOT_FORMAL:formal_31m" in result["delivery_manifest"]["reasons"]
