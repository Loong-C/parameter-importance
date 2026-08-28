from __future__ import annotations

from pathlib import Path

from ops.stage3 import materialize_stage3_phase as phase
from ops.stage3 import run_stage3_formal as formal


def _write(path: Path, value: dict[str, object]) -> None:
    formal._write_atomic(path, value)


def _seal(value: dict[str, object]) -> dict[str, object]:
    value["artifact_hash"] = formal._canonical_hash(value)
    return value


def _direct_receipt(task_id: str, index: int) -> dict[str, object]:
    root = f"runs/stage3/s3{index:02d}"
    return _seal(
        {
            "schema_version": phase.DIRECT_RECEIPT_SCHEMA,
            "task_id": task_id,
            "config_ref": f"configs/s3{index:02d}.json",
            "config_hash": f"{index:064x}",
            "result_ref": f"results/s3{index:02d}.json",
            "output_refs": {
                kind: f"{root}/commits/{kind}.json"
                for kind in formal.EXPECTED_OUTPUTS[task_id]
            },
            "artifact_output_dir": root,
            "authority_output_dir": f"authority/s3{index:02d}",
            "evidence_refs": {},
            "external_gate_ref": None,
            "command": [
                "{python}",
                "-m",
                "param_importance_nlp",
                "task",
                "run",
                "--config",
                "{config}",
                "--environment",
                "{environment}",
                "--result",
                "{result}",
            ],
        }
    )


def _fanout_receipt(
    root: Path, task_id: str, index: int, run_hash: str
) -> dict[str, object]:
    output_root = f"runs/stage3/s3{index:02d}"
    manifest_ref = f"manifests/s3{index:02d}.json"
    fanout_manifest: dict[str, object] = {
        "schema_version": "stage3-fanout-manifest-v1",
        "task_id": task_id,
        "scope": "pilot",
        "run_config_hash": run_hash,
        "unit_index_ref": "unit-index.json",
        "unit_index_hash": "d" * 64,
        "state_dir": f"fanout-state/s3{index:02d}",
        "status_ref": f"status/s3{index:02d}.json",
        "steps": [],
    }
    fanout_manifest["manifest_hash"] = formal._canonical_hash(fanout_manifest)
    _write(root / manifest_ref, fanout_manifest)
    kinds = formal.EXPECTED_OUTPUTS[task_id]
    return _seal(
        {
            "schema_version": phase.FANOUT_RECEIPT_SCHEMA,
            "task_id": task_id,
            "scope": "pilot",
            "unit_count": 12,
            "step_count": 12,
            "manifest_ref": manifest_ref,
            "manifest_hash": fanout_manifest["manifest_hash"],
            "final_config_ref": f"configs/s3{index:02d}-final.json",
            "final_config_hash": f"{index:064x}",
            "final_result_ref": f"results/s3{index:02d}-final.json",
            "status_ref": f"status/s3{index:02d}.json",
            "artifact_output_dir": output_root,
            "expected_output_refs": {
                kind: f"{output_root}/commits/{kind}.json" for kind in kinds
            },
        }
    )


def test_phase_materializer_assembles_exact_pilot_dag(
    tmp_path: Path, monkeypatch
) -> None:
    run_hash = "c" * 64
    task_refs: dict[str, str] = {}
    for index, task_id in enumerate(formal.PILOT_TASK_ORDER, start=1):
        receipt = (
            _fanout_receipt(tmp_path, task_id, index, run_hash)
            if task_id in phase.FANOUT_TASKS
            else _direct_receipt(task_id, index)
        )
        ref = f"receipts/s3{index:02d}.json"
        _write(tmp_path / ref, receipt)
        task_refs[task_id] = ref
    _write(tmp_path / "unit-index.json", {})
    units = tuple(
        formal.UnitRecord(
            f"pilot-unit-{index:02d}",
            "14M",
            7,
            ("early", "middle", "late")[index // 4],
            f"{index + 1:064x}",
            f"probe-{index:02d}",
        )
        for index in range(12)
    )
    monkeypatch.setattr(formal, "load_unit_index", lambda *_args, **_kwargs: ("d" * 64, units))
    source = {
        "schema_version": phase.SOURCE_SCHEMA,
        "scope": "pilot",
        "config_hash": run_hash,
        "state_dir": "phase-state/pilot",
        "unit_index_ref": "unit-index.json",
        "stage2_authority_ref": "stage2.json",
        "initial_execution_evidence_ref": "authority/initial-evidence.json",
        "initial_execution_config_hash": "a" * 64,
        "initial_environment_ref": "authority/initial-environment.json",
        "g30_gate_ref": "authority/g30.json",
        "health_snapshot_ref": "authority/health.json",
        "scope_decision_ref": "authority/scope.json",
        "contract_freeze_hash": "b" * 64,
        "asset_manifest_hashes": ["e" * 64],
        "task_receipt_refs": task_refs,
    }
    manifest = phase.materialize(
        source, workspace_root=tmp_path, data_root=tmp_path
    )
    assert [task["task_id"] for task in manifest["tasks"]] == list(
        formal.PILOT_TASK_ORDER
    )
    specs = {task["task_id"]: task for task in manifest["tasks"]}
    assert specs["stage3.05_reference_integral_and_precision"]["unit_status_ref"] is None
    assert (
        specs["stage3.06_pilot_and_threshold_freeze"]["unit_status_ref"]
        == "status/s306.json"
    )
    assert specs["stage3.06_pilot_and_threshold_freeze"]["command"][2] == (
        "ops.stage3.run_stage3_fanout"
    )


def test_phase_materializer_rejects_fanout_run_hash_drift(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        phase,
        "Stage3Orchestrator",
        lambda *_args, **_kwargs: None,
    )
    task_refs: dict[str, str] = {}
    for index, task_id in enumerate(formal.PILOT_TASK_ORDER, start=1):
        receipt = (
            _fanout_receipt(tmp_path, task_id, index, "f" * 64)
            if task_id in phase.FANOUT_TASKS
            else _direct_receipt(task_id, index)
        )
        ref = f"receipts/s3{index:02d}.json"
        _write(tmp_path / ref, receipt)
        task_refs[task_id] = ref
    source = {
        "schema_version": phase.SOURCE_SCHEMA,
        "scope": "pilot",
        "config_hash": "c" * 64,
        "state_dir": "phase-state/pilot",
        "unit_index_ref": "unit-index.json",
        "stage2_authority_ref": "stage2.json",
        "initial_execution_evidence_ref": "authority/initial-evidence.json",
        "initial_execution_config_hash": "a" * 64,
        "initial_environment_ref": "authority/initial-environment.json",
        "g30_gate_ref": "authority/g30.json",
        "health_snapshot_ref": "authority/health.json",
        "scope_decision_ref": "authority/scope.json",
        "contract_freeze_hash": "b" * 64,
        "asset_manifest_hashes": ["e" * 64],
        "task_receipt_refs": task_refs,
    }
    try:
        phase.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    except formal.Stage3OrchestratorError as error:
        assert "PHASE_FANOUT_MANIFEST_DRIFT" in str(error)
    else:
        raise AssertionError("fanout run-config drift was accepted")
