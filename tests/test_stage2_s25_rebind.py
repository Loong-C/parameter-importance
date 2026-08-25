"""Read-only S2.5 fresh-run/G2.3 rebind preparation tests.

These tests build only tiny canonical control-plane artifacts.  They do not
start a runner, touch CUDA, or exercise the S2.4 provider.
"""

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.experiments.stage2_s25_rebind import (
    APPROVED_GPU_UUIDS,
    CELL_COMPONENTS,
    EXPECTED_CELL_IDS,
    S25RebindBlocked,
    S25RebindSpec,
    prepare_s25_rebind,
)


G3_REF = "evidence/stage0/tasks/g3-v5/commits/asset_resolution.json"
EXECUTION_COMMIT = "1" * 40
CONFIG_HASH = "b" * 64


def _spec(tmp_path: Path) -> S25RebindSpec:
    g3_commit = load_canonical_json(tmp_path / G3_REF)["artifact_hash"]
    return S25RebindSpec(
        data_root=tmp_path,
        s204_run_root="evidence/stage2/s204/formal-r19-g3-v5",
        s204_prepared_root="evidence/stage2/s204/prepared-r18-g3-v5",
        g23_evaluation_root="evidence/stage2/s204/formal-r19-g3-v5/g23-evaluation",
        s205_output_root="evidence/stage2/s205-formal-r19-g3-v5",
        operations_root="operations/stage2/s205-formal-r19-g3-v5",
        g3_ref=G3_REF,
        g3_artifact_hash=g3_commit,
        execution_commit=EXECUTION_COMMIT,
    )


def _g3(tmp_path: Path) -> str:
    gate_ids = (
        "stage0.G3-S1",
        "stage0.G3-S2",
        "stage0.G3-S4",
        "stage0.G3-S5",
        "stage0.G3-S6",
    )
    gates = [
        GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at="2026-08-25T00:00:00+00:00",
            evidence_refs=("evidence/g3.json",),
        ).to_dict()
        for gate_id in gate_ids
    ]
    payload = {
        "schema_version": "stage0-g3-resolution-audit-v1",
        "scope": "formal",
        "status": "PASS",
        "checked_at": "2026-08-25T00:00:00+00:00",
        "requirements_ref": "requirements.json",
        "requirements_artifact_hash": "a" * 64,
        "layout_artifact_hash": "b" * 64,
        "entries": [],
        "gates": gates,
    }
    store = TaskArtifactStore(tmp_path, "evidence/stage0/tasks/g3-v5")
    published = store.publish(
        task_id="stage0.04_assets_and_manifests",
        artifact_kind="asset_resolution",
        config_hash="c" * 64,
        run_intent="formal",
        formal_eligible=True,
        payload={**payload, "artifact_hash": canonical_json_hash(payload)},
        source_refs=("requirements.json", "layout.json"),
    )
    return published.commit_ref


def _formal_lineage(tmp_path: Path, spec: S25RebindSpec) -> None:
    value = {
        "schema_version": "formal-execution-evidence-v1",
        "run_intent": "formal",
        "contract_freeze_hash": "d" * 64,
        "asset_manifest_hashes": ["e" * 64],
        "prerequisite_gates": [],
        "metadata": {
            "execution_commit": EXECUTION_COMMIT,
            "producer_commit": EXECUTION_COMMIT,
        },
    }
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(
        tmp_path / spec.s204_prepared_root / "formal-execution-g22.json", value
    )


def _write_ready_inputs(tmp_path: Path, *, g23: bool = True) -> None:
    g3_ref = _g3(tmp_path)
    spec = S25RebindSpec(
        data_root=tmp_path,
        s204_run_root="evidence/stage2/s204/formal-r19-g3-v5",
        s204_prepared_root="evidence/stage2/s204/prepared-r18-g3-v5",
        g23_evaluation_root="evidence/stage2/s204/formal-r19-g3-v5/g23-evaluation",
        s205_output_root="evidence/stage2/s205-formal-r19-g3-v5",
        operations_root="operations/stage2/s205-formal-r19-g3-v5",
        g3_ref=g3_ref,
        g3_artifact_hash=load_canonical_json(tmp_path / g3_ref)["artifact_hash"],
        execution_commit=EXECUTION_COMMIT,
    )
    _formal_lineage(tmp_path, spec)
    g23_cells: list[dict[str, object]] = []
    for cell_id in EXPECTED_CELL_IDS:
        component = CELL_COMPONENTS[cell_id]
        config_ref = f"configs/{component}.json"
        write_canonical_json(
            tmp_path / config_ref,
            {
                "schema_version": "resolved-config-v2",
                "config_hash": CONFIG_HASH,
                "orchestration": {
                    "input_result_refs": [
                        "evidence/stage2/s204/materialized-task-inputs-r7/task-outputs/stage2-02/commits/handoff_manifest.json",
                        "evidence/stage2/s204/materialized-task-inputs-r7/task-outputs/stage2-02/commits/fixed_state_contract.json",
                    ]
                },
            },
        )
        store = TaskArtifactStore(tmp_path, f"evidence/artifacts/{component}")
        refs: dict[str, str] = {}
        for kind, schema in (
            ("reference_result", "reference-result-v1"),
            ("reference_convergence_report", "stage2-reference-convergence-report-v1"),
            ("gate_record", "stage23-task-gate-candidate-v1"),
        ):
            payload = {"schema_version": schema, "cell_id": cell_id}
            if kind == "reference_result":
                payload["artifact_hash"] = canonical_json_hash(payload)
            published = store.publish(
                task_id="stage2.04_reference_target",
                artifact_kind=kind,
                config_hash=CONFIG_HASH,
                run_intent="formal",
                formal_eligible=True,
                payload=payload,
                source_refs=("evidence/source.json",),
            )
            refs[kind] = published.commit_ref
        result = {
            "schema_version": "task-run-result-v2",
            "task_id": "stage2.04_reference_target",
            "stage": 2,
            "runner_kind": "reference",
            "run_intent": "formal",
            "status": "PASS",
            "config_hash": CONFIG_HASH,
            "formal_eligible": True,
            "artifact_refs": refs,
            "checkpoint_ref": None,
            "blockers": [],
            "error_code": None,
            "message": "ok",
            "recovery_mode": "restart_idempotent",
            "metadata": {},
        }
        result["result_hash"] = canonical_json_hash(result)
        result_ref = (
            f"evidence/stage2/s204/formal-r19-g3-v5/{component}/attempts/fresh/"
            f"task-results/{result['result_hash']}.json"
        )
        write_canonical_json(tmp_path / result_ref, result)
        status = {
            "schema_version": "stage2-s204-cell-final-status-v3",
            "cell_id": cell_id,
            "config_path": config_ref,
            "config_hash": CONFIG_HASH,
            "status": "COMPLETE",
            "formal_eligible": True,
            "task_result_hash": result["result_hash"],
            "task_result_ref": result_ref,
            "artifact_refs": refs,
            "execution_commit": EXECUTION_COMMIT,
        }
        status["artifact_hash"] = canonical_json_hash(status)
        write_canonical_json(
            tmp_path
            / f"evidence/stage2/s204/formal-r19-g3-v5/{component}/attempts/fresh/final-status.json",
            status,
        )
        environment = {
            "schema_version": "task-runtime-environment-v1",
            "capabilities": [],
            "frozen_contract_stages": [0, 1, 2],
            "passed_gate_ids": ["stage2.G2.0", "stage2.G2.1", "stage2.G2.2"],
            "estimator_decision_ref": None,
            "evidence_refs": {
                "g3_resolution": g3_ref,
                "formal_execution": f"{spec.s204_prepared_root}/formal-execution-g22.json",
            },
        }
        environment["environment_hash"] = canonical_json_hash(environment)
        write_canonical_json(
            tmp_path / spec.s204_prepared_root / "environments" / f"{component}.json",
            environment,
        )
        g23_cells.append(
            {
                "cell_id": cell_id,
                "status": "PASS",
                "identities": {
                    "result_hash": result["result_hash"],
                    "config_hash": CONFIG_HASH,
                },
            }
        )
    if g23:
        evaluation = {
            "schema_version": "stage2-g23-reference-evaluation-v1",
            "gate_id": "stage2.G2.3",
            "status": "PASS",
            "formal_eligible": True,
            "required_cell_count": 6,
            "complete_cell_count": 6,
            "expected_cell_ids": list(EXPECTED_CELL_IDS),
            "cells": g23_cells,
        }
        evaluation["artifact_hash"] = canonical_json_hash(evaluation)
        write_canonical_json(
            tmp_path
            / spec.g23_evaluation_root
            / "g2.3-attempts"
            / evaluation["artifact_hash"]
            / "evaluation.json",
            evaluation,
        )


def test_rebind_plan_requires_fresh_s204_and_strict_g23_pass(tmp_path: Path) -> None:
    _write_ready_inputs(tmp_path)
    plan = prepare_s25_rebind(_spec(tmp_path))
    assert plan["status"] == "READY"
    assert plan["formal_eligible"] is True
    assert plan["g3_artifact_hash"] == load_canonical_json(tmp_path / G3_REF)["artifact_hash"]
    assert plan["g3_resolution_payload_hash"]
    assert plan["g23_evaluation_hash"]
    assert [row["cell_id"] for row in plan["cells"]] == list(EXPECTED_CELL_IDS)
    assert [row["gpu_uuid"] for row in plan["cells"]] == [
        APPROVED_GPU_UUIDS[index % len(APPROVED_GPU_UUIDS)] for index in range(6)
    ]


def test_rebind_blocks_until_g23_pass(tmp_path: Path) -> None:
    _write_ready_inputs(tmp_path, g23=False)
    with pytest.raises(S25RebindBlocked, match="G2.3_EVALUATION_MISSING"):
        prepare_s25_rebind(_spec(tmp_path))


def test_rebind_blocks_duplicate_complete_s204_status(tmp_path: Path) -> None:
    _write_ready_inputs(tmp_path)
    spec = _spec(tmp_path)
    component = CELL_COMPONENTS[EXPECTED_CELL_IDS[0]]
    original = load_canonical_json(
        tmp_path
        / spec.s204_run_root
        / component
        / "attempts"
        / "fresh"
        / "final-status.json"
    )
    duplicate = (
        tmp_path
        / spec.s204_run_root
        / component
        / "attempts"
        / "duplicate"
        / "final-status.json"
    )
    write_canonical_json(duplicate, original)
    with pytest.raises(S25RebindBlocked, match="S204_STATUS_NOT_UNIQUE"):
        prepare_s25_rebind(spec)


def test_rebind_requires_explicit_fresh_execution_commit_in_status(tmp_path: Path) -> None:
    _write_ready_inputs(tmp_path)
    spec = _spec(tmp_path)
    component = CELL_COMPONENTS[EXPECTED_CELL_IDS[0]]
    status_path = (
        tmp_path
        / spec.s204_run_root
        / component
        / "attempts"
        / "fresh"
        / "final-status.json"
    )
    status = load_canonical_json(status_path)
    del status["execution_commit"]
    status["artifact_hash"] = canonical_json_hash(status)
    write_canonical_json(status_path, status)
    with pytest.raises(S25RebindBlocked, match="STATUS_SCHEMA_INVALID"):
        prepare_s25_rebind(spec)


def test_rebind_does_not_reuse_output_root(tmp_path: Path) -> None:
    _write_ready_inputs(tmp_path)
    output = tmp_path / _spec(tmp_path).s205_output_root
    output.mkdir(parents=True)
    with pytest.raises(S25RebindBlocked, match="S205_OUTPUT_ROOT_ALREADY_EXISTS"):
        prepare_s25_rebind(_spec(tmp_path))
