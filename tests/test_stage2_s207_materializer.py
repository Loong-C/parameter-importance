from __future__ import annotations

from pathlib import Path
import json

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s207_materializer import (
    S27MaterializationBlocked,
    _logical,
    _publish_g23_gates,
    validate_failure_rule,
)
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def test_s206_native_absolute_refs_are_canonicalized_to_logical_posix(tmp_path: Path) -> None:
    target = tmp_path / "freeze" / "commit.json"
    target.parent.mkdir()
    write_canonical_json(target, {"schema_version": "fixture-v1"})
    logical, resolved = _logical(tmp_path, str(target), field="freeze")
    assert logical == "freeze/commit.json"
    assert resolved == target.resolve()


def _parent_preregistration(tmp_path: Path) -> tuple[str, str, str]:
    body = {
        "schema_version": "stage2-preregistration-v1",
        "scope": "formal",
        "state": "FROZEN",
    }
    body["preregistration_hash"] = canonical_json_hash(body)
    store = TaskArtifactStore(tmp_path, "artifacts/s201")
    published = store.publish(
        task_id="stage2.01_scope_hypotheses_and_preregistration",
        artifact_kind="preregistration",
        config_hash="a" * 64,
        run_intent="formal",
        payload=body,
        formal_eligible=True,
    )
    return published.commit_ref, published.artifact_hash, str(body["preregistration_hash"])


def test_failure_rule_requires_explicit_hash_bound_formal_parent(tmp_path: Path) -> None:
    parent_ref, parent_hash, parent_payload_hash = _parent_preregistration(tmp_path)
    body = {
        "schema_version": "stage2-s207-failure-rule-amendment-v1",
        "amendment_id": "s207-failure-rule-v1",
        "task_id": "stage2.07_main_sweep",
        "scope": "formal",
        "status": "FROZEN",
        "formal_eligible": True,
        "approval_status": "APPROVED",
        "approved_at": "2026-08-26T00:00:00+00:00",
        "rule": "stop_after_failure_fraction_exceeds_max",
        "parent_preregistration_ref": parent_ref,
        "parent_preregistration_hash": parent_hash,
        "parent_preregistration_payload_hash": parent_payload_hash,
        "max_failure_fraction": 0.01,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    ref = "amendments/s207-failure-rule.json"
    write_canonical_json(tmp_path / ref, body)
    assert validate_failure_rule(tmp_path, ref) == (ref, body["artifact_hash"], 0.01)

    tampered = dict(body)
    tampered["max_failure_fraction"] = 0.0
    write_canonical_json(tmp_path / ref, tampered)
    with pytest.raises(S27MaterializationBlocked, match="ARTIFACT_HASH_MISMATCH"):
        validate_failure_rule(tmp_path, ref)


def test_g23_gate_is_direct_and_binds_eval_result_config_reference_and_sidecar(tmp_path: Path) -> None:
    config_hash = "b" * 64
    sidecar_refs: dict[str, str] = {}
    sidecar_hashes: dict[str, str] = {}
    for index, current_cell_id in enumerate(EXPECTED_CELL_IDS):
        sidecar_body = {
            "schema_version": "stage2-g23-corrected-delta-sci-v1",
            "cell_id": current_cell_id,
            "delta_sci_batch_sizes": [32, 64, 128, 256],
            "marker": index,
        }
        current_hash = canonical_json_hash(sidecar_body)
        current_ref = "g23/g2.3-corrected-delta-sci/" + current_hash + ".json"
        write_canonical_json(tmp_path / current_ref, {**sidecar_body, "artifact_hash": current_hash})
        sidecar_refs[current_cell_id] = current_ref
        sidecar_hashes[current_cell_id] = current_hash

    reference_body = {
        "schema_version": "reference-result-v1",
        "scope": "formal",
        "formal_eligible": False,
        "artifact_hash": canonical_json_hash({
            "schema_version": "reference-result-v1",
            "scope": "formal",
            "formal_eligible": False,
        }),
    }
    store = TaskArtifactStore(tmp_path, "artifacts/s204")
    reference = store.publish(
        task_id="stage2.04_reference_target",
        artifact_kind="reference_result",
        config_hash=config_hash,
        run_intent="formal",
        payload=reference_body,
        formal_eligible=True,
    )
    cells = []
    rows = {}
    for index, cell_id in enumerate(EXPECTED_CELL_IDS):
        task_result_ref = f"s204/task-result-{index}.json"
        task_result_body = {
            "schema_version": "task-run-result-v2",
            "task_id": "stage2.04_reference_target",
            "cell_id": cell_id,
        }
        current_result_hash = canonical_json_hash(task_result_body)
        write_canonical_json(tmp_path / task_result_ref, {**task_result_body, "result_hash": current_result_hash})
        status_ref = f"s204/status-{index}.json"
        write_canonical_json(
            tmp_path / status_ref,
            {"schema_version": "stage2-s204-cell-final-status-v3", "cell_id": cell_id, "task_result_ref": task_result_ref, "task_result_hash": current_result_hash},
        )
        cell = {
            "cell_id": cell_id,
            "status": "PASS",
            "identities": {
                "result_hash": current_result_hash,
                "config_hash": config_hash,
                "corrected_delta_sci_hash": sidecar_hashes[cell_id],
                "corrected_delta_sci_ref": sidecar_refs[cell_id],
            },
            "metrics": {
                "corrected_delta_sci_hash": sidecar_hashes[cell_id],
                "corrected_delta_sci_ref": sidecar_refs[cell_id],
                "corrected_delta_sci_batch_sizes": [32, 64, 128, 256],
                "delta_sci_source": "g23_output_derived_corrected_sidecar",
            },
        }
        cells.append(cell)
        current_config_ref = f"configs/cell-{index}.json"
        write_canonical_json(tmp_path / current_config_ref, {"config_hash": config_hash, "schema_version": "resolved-config-v2"})
        rows[cell_id] = {
            "result_hash": current_result_hash,
            "config_hash": config_hash,
            "config_ref": current_config_ref,
            "task_result_status_path": status_ref,
            "reference_artifact_refs": {
                "reference_result": reference.commit_ref,
                "reference_convergence_report": "artifacts/s204/commits/reference_convergence_report.json",
                "gate_record": "artifacts/s204/commits/gate_record.json",
            },
        }
    gates, _ = _publish_g23_gates(
        tmp_path,
        gate_output_root="g23/direct-gates",
        checked_at="2026-08-26T00:00:00+00:00",
        evaluation_ref="g23/g2.3-attempts/" + "d" * 64 + "/evaluation.json",
        evaluation_hash="d" * 64,
        evaluation={
            "schema_version": "stage2-g23-reference-evaluation-v1",
            "gate_id": "stage2.G2.3",
            "status": "PASS",
            "scope": "formal",
            "formal_eligible": True,
            "cells": cells,
        },
        s205_ref="s205/rebind.json",
        s205_hash="e" * 64,
        s205_rows=rows,
    )
    cell_id = EXPECTED_CELL_IDS[0]
    gate_ref = gates[cell_id]["gate_ref"]
    gate = GateRecord.from_mapping(dict(json.loads((tmp_path / gate_ref).read_text(encoding="utf-8"))))
    assert gate.gate_id == "stage2.G2.3"
    assert reference.commit_ref in gate.evidence_refs
    assert gate.measured["evaluation"]["artifact_hash"] == "d" * 64  # type: ignore[index]
    assert gate.measured["corrected_delta_sci"]["artifact_hash"] == sidecar_hashes[cell_id]  # type: ignore[index]
