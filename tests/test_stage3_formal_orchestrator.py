"""Control-plane tests for the detached Stage 3 production entry point.

These tests exercise only contracts, locks, ledgers, and authority publishing;
they never stand in for a model endpoint, probe, or formal experiment result.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage3 import run_stage3_formal as orchestrator


def _hash(value: object) -> str:
    return orchestrator._canonical_hash(value)


def _write_health(path: Path, *, gpu_uuid: str, pci_bus_id: str = "0000:18:00.0") -> None:
    payload = {
        "schema_version": orchestrator.HEALTH_SCHEMA,
        "checked_at": "2026-08-28T00:00:00Z",
        "selected_gpu_uuid": gpu_uuid,
        "selected_pci_bus_id": pci_bus_id,
        "devices": [
            {
                "uuid": gpu_uuid,
                "pci_bus_id": pci_bus_id,
                "health": "PASS",
                "active_processes": [],
                "uncorrected_ecc": 0,
            }
        ],
    }
    payload["artifact_hash"] = _hash(payload)
    orchestrator._write_atomic(path, payload)


def test_stage2_identity_requires_real_u32_raw_identity() -> None:
    payload = {
        "run_id": orchestrator.EXPECTED_STAGE2_RUN_ID,
        "default_estimator": "U-32",
        "batch_size": 32,
        "data_variant": "Raw",
    }
    orchestrator.validate_stage2_identity(payload)

    with pytest.raises(orchestrator.Stage3OrchestratorError, match="STAGE2_RUN_ID_MISMATCH"):
        orchestrator.validate_stage2_identity({**payload, "run_id": "fixture-stage2"})


def test_phase_dag_runs_real_stage301_before_pilot_and_keeps_formal_post_gate() -> None:
    assert orchestrator.PILOT_TASK_ORDER == (
        "stage3.01_prerequisites_and_scope",
        "stage3.02_math_and_metric_contract",
        "stage3.03_endpoint_and_probe_pipeline",
        "stage3.04_quadrature_engine_and_unit_tests",
        "stage3.05_reference_integral_and_precision",
        "stage3.06_pilot_and_threshold_freeze",
    )
    assert orchestrator.FORMAL_TASK_ORDER == (
        "stage3.07_formal_experiment_matrix",
        "stage3.08_error_analysis_and_stability",
        "stage3.09_cost_and_method_selection",
        "stage3.10_reports_visualizations_and_handoff",
    )
    assert "stage3.10_reports_visualizations_and_handoff" not in (
        orchestrator.EXTERNAL_GATE_BY_TASK
    )


def test_gpu_health_rejects_permanently_excluded_uuid(tmp_path: Path) -> None:
    path = tmp_path / "gpu-health.json"
    _write_health(path, gpu_uuid=orchestrator.EXCLUDED_GPU_UUID, pci_bus_id=orchestrator.EXCLUDED_PCI_BUS_ID)
    with pytest.raises(orchestrator.Stage3OrchestratorError, match="GPU_NOT_APPROVED"):
        orchestrator.verify_gpu_health_once(path)


def test_unit_ledger_is_atomic_and_resume_safe(tmp_path: Path) -> None:
    unit = orchestrator.UnitRecord(
        unit_id="pilot-u-0001",
        model="pythia-14m",
        seed=7,
        stage="early",
        endpoint_hash="e" * 64,
        probe_id="probe-0",
    )
    ledger_path = tmp_path / "state" / "unit-ledger.json"
    ledger = orchestrator.UnitLedger(
        ledger_path,
        scope="pilot",
        config_hash="a" * 64,
        unit_index_hash="b" * 64,
        units=(unit,),
    )
    ledger.record_attempt("pilot-u-0001", "RUNNING", attempt_id="attempt-1")
    ledger.record_attempt("pilot-u-0001", "PASS", attempt_id="attempt-1", metadata={"real": True})
    resumed = orchestrator.UnitLedger(
        ledger_path,
        scope="pilot",
        config_hash="a" * 64,
        unit_index_hash="b" * 64,
        units=(unit,),
    )
    assert resumed.complete
    assert len(resumed.value["attempts"]) == 2


def test_unit_status_reconcile_rejects_duplicate_coverage(tmp_path: Path) -> None:
    unit = orchestrator.UnitRecord("u-1", "pythia-14m", 1, "early", "e" * 64, "probe-0")
    ledger = orchestrator.UnitLedger(
        tmp_path / "ledger.json",
        scope="pilot",
        config_hash="a" * 64,
        unit_index_hash="b" * 64,
        units=(unit,),
    )
    status = {
        "schema_version": "stage3-unit-status-v1",
        "config_hash": "a" * 64,
        "unit_index_hash": "b" * 64,
        "units": [{"unit_id": "u-1", "status": "PASS"}, {"unit_id": "u-1", "status": "PASS"}],
    }
    path = tmp_path / "status.json"
    orchestrator._write_atomic(path, status)
    with pytest.raises(orchestrator.Stage3OrchestratorError, match="UNIT_STATUS_COVERAGE_INVALID"):
        ledger.reconcile(path)


def test_live_pid_lock_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.lock"
    orchestrator._write_atomic(path, {"pid": __import__("os").getpid(), "started_at": "2026-08-28T00:00:00Z"})
    with pytest.raises(orchestrator.Stage3OrchestratorError, match="ORCHESTRATOR_ALREADY_RUNNING"):
        orchestrator.InstanceLock(path).acquire()


def test_task_spec_cannot_declare_candidate_gate_output() -> None:
    with pytest.raises(orchestrator.Stage3OrchestratorError, match="TASK_SPEC_OUTPUT_COVERAGE_INVALID"):
        orchestrator.TaskSpec.from_mapping(
            {
                "task_id": "stage3.02_math_and_metric_contract",
                "config_ref": "configs/stage3.json",
                "config_hash": "a" * 64,
                "environment_ref": "environment/stage3.json",
                "evidence_refs": {},
                "command": ["{python}", "script.py"],
                "output_refs": {
                    "path_math_contract": "outputs/path_math_contract.json",
                    "metric_contract": "outputs/metric_contract.json",
                    "gate_record": "outputs/candidate-gate.json",
                },
                "output_dir": "operations/stage3/s302",
                "result_ref": None,
                "unit_status_ref": None,
                "external_gate_ref": None,
            }
        )


def test_gate_authority_publishes_independent_formal_gate_and_evidence(tmp_path: Path) -> None:
    from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
    from param_importance_nlp.contracts.status import GateRecord, GateStatus
    from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore, load_committed_task_artifact

    config_hash = "a" * 64
    contract_hash = "b" * 64
    asset_hash = "c" * 64
    g30 = GateRecord(
        gate_id="stage3.G3-0",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:00:00Z",
        measured={"source": "stage2-authority"},
        threshold={"required": True},
        evidence_refs=("inputs/stage2-authority.json",),
    )
    initial_evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=contract_hash,
        asset_manifest_hashes=(asset_hash,),
        prerequisite_gates=(g30,),
        metadata={"scope": "formal"},
    )
    initial_ref = TaskArtifactStore(tmp_path, "initial").publish(
        task_id=orchestrator.CONTROL_TASK_ID,
        artifact_kind="formal_execution_evidence",
        config_hash=config_hash,
        run_intent="formal",
        payload=initial_evidence.to_dict(),
        formal_eligible=True,
    ).commit_ref
    output_refs: dict[str, str] = {}
    store = TaskArtifactStore(tmp_path, "s302-outputs")
    payloads = {
        "path_math_contract": {
            "schema_version": "stage3-task-path-math-contract-v1",
            "signed_contribution": "-delta_theta*integral_0^1 gradient(theta(alpha)) d_alpha",
            "parameter_post_state_distinct_from_attempt_commit_state": True,
            "quadrature_weight_dtype": "float64",
        },
        "metric_contract": {
            "schema_version": "stage3-task-metric-contract-v1",
            "undefined_policy": "defined_false_with_reason_no_epsilon",
            "strata": ["model", "stage", "update", "probe"],
            "metrics": [
                "normalized_l1_error", "normalized_l2_error",
                "normalized_linf_error", "cosine_similarity",
                "active_spearman", "sign_consistency",
                "completeness_absolute_residual",
                "completeness_relative_residual",
                "completeness_l1_scaled_residual", "spearman",
                "top_q_overlap", "top_q_jaccard",
                "layer_total_variation", "module_total_variation",
                "real_gradient_evaluation_cost",
            ],
        },
    }
    for kind, payload in payloads.items():
        output_refs[kind] = store.publish(
            task_id="stage3.02_math_and_metric_contract",
            artifact_kind=kind,
            config_hash=config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
        ).commit_ref

    gate_ref, evidence_ref, gate = orchestrator.GateAuthorityPublisher(tmp_path).publish(
        output_dir="authority/s302",
        config_hash=config_hash,
        task_id="stage3.02_math_and_metric_contract",
        gate_id="stage3.G3-1",
        output_refs=output_refs,
        previous_evidence_ref=initial_ref,
        contract_freeze_hash=contract_hash,
        asset_manifest_hashes=(asset_hash,),
    )
    assert gate.status is GateStatus.PASS
    loaded_gate = load_committed_task_artifact(tmp_path, gate_ref, require_formal=True)
    assert loaded_gate.payload["schema_version"] == "gate-record-v1"
    loaded_evidence = load_committed_task_artifact(tmp_path, evidence_ref, require_formal=True)
    parsed = FormalExecutionEvidence.from_mapping(dict(loaded_evidence.payload))
    assert [item.gate_id for item in parsed.prerequisite_gates] == ["stage3.G3-0", "stage3.G3-1"]
    assert dict(parsed.metadata) == dict(initial_evidence.metadata)


def test_detached_launcher_uses_argument_vector_and_new_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_popen(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(pid=31415)

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    pid = orchestrator.launch_detached(["python", "runner.py", "--formal"], log_path=tmp_path / "run.log", cwd=tmp_path)
    assert pid == 31415
    assert calls and calls[0]["shell"] is False
    assert calls[0]["args"] == ["python", "runner.py", "--formal"]
    assert calls[0]["start_new_session"] is True
