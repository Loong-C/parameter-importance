from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.stage2_s206_formal import ANCHOR_IDS, APPROVED_GPU_BINDINGS, EXCLUDED_PCI, EXCLUDED_UUID
from param_importance_nlp.experiments.stage2_s206_inputs import (
    S206FormalInputError,
    build_cost_semantics_contract,
    build_formal_execution_evidence,
    build_retry_policy_contract,
    collect_gpu_inventory,
)
from param_importance_nlp.experiments.stage2_s207_runner import load_s27_gpu_inventory_envelope
from ops.stage2.produce_s206_formal_inputs import main as produce_s206_inputs
from ops.stage2.run_s206_formal import _load_formal_execution


def _gpu_output() -> str:
    rows: list[str] = []
    for pci, uuid in APPROVED_GPU_BINDINGS:
        rows.append(f"{uuid},{pci},A100-SXM4-80GB,42,0,81920,0,Default,0,0,0,0,None")
    rows.append(f"{EXCLUDED_UUID},{EXCLUDED_PCI},A100-SXM4-80GB,55,32,81920,0,Default,3,7,0,1,None")
    for index, pci in enumerate(("0000:4F:00.0", "0000:51:00.0", "0000:57:00.0")):
        rows.append(f"GPU-extra-{index},{pci},A100-SXM4-80GB,40,0,81920,0,Default,0,0,0,0,None")
    return "\n".join(rows) + "\n"


def test_live_collector_is_explicit_and_non_self_referential(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return type("Completed", (), {"stdout": "" if "query-compute-apps" in command[1] else _gpu_output(), "stderr": ""})()

    output = tmp_path / "evidence/gpu-inventory.json"
    source = tmp_path / "evidence/gpu-inventory.capture.txt"
    payload = collect_gpu_inventory(
        output=output,
        source_output=source,
        data_root=tmp_path,
        runner=runner,
    )
    assert len(calls) == 2
    assert payload["artifact_ref"] == "evidence/gpu-inventory.json"
    assert payload["source_ref"] == "evidence/gpu-inventory.capture.txt"
    assert payload["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["source_sha256"] != hashlib.sha256(output.read_bytes()).hexdigest()
    loaded = load_canonical_json(output)
    assert loaded == payload
    _summary, identity = load_s27_gpu_inventory_envelope(output, data_root=tmp_path)
    assert identity["source_ref"] == "evidence/gpu-inventory.capture.txt"
    assert identity["source_sha256"] == payload["source_sha256"]
    with pytest.raises(S206FormalInputError, match="IMMUTABLE_SOURCE_CAPTURE_CONFLICT"):
        collect_gpu_inventory(
            output=output,
            source_output=source,
            data_root=tmp_path,
            runner=runner,
            checked_at="2026-08-26T00:00:01+00:00",
        )


def test_formal_inventory_loader_requires_distinct_source_binding(tmp_path: Path) -> None:
    path = tmp_path / "gpu-inventory.json"
    payload = {
        "schema_version": "stage2-s206-gpu-inventory-v1",
        "source_ref": "gpu-inventory.json",
        "rows": [],
        "compute_apps": [],
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(path, payload)
    from ops.stage2 import run_s206_formal as launcher

    with pytest.raises(Exception, match="GPU_INVENTORY_ARTIFACT_REF_REQUIRED"):
        launcher._load_inventory_snapshot(path, data_root=tmp_path)

    source = tmp_path / "capture.txt"
    source.write_bytes(b"raw capture\n")
    payload["artifact_ref"] = "gpu-inventory.json"
    payload["source_ref"] = "capture.txt"
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(path, payload)
    with pytest.raises(Exception, match="GPU_INVENTORY_SOURCE_SHA256_REQUIRED"):
        launcher._load_inventory_snapshot(path, data_root=tmp_path)

    payload["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload["source_ref"] = "capture.txt//"
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(path, payload)
    with pytest.raises(Exception, match="GPU_INVENTORY_SOURCE_REF_NOT_CANONICAL"):
        launcher._load_inventory_snapshot(path, data_root=tmp_path)


def test_cost_and_retry_producers_are_frozen_without_guessed_seconds() -> None:
    cost = build_cost_semantics_contract()
    assert cost["status"] == "FROZEN"
    for name in ("scientific_equal_sample_cost", "isolated_estimator_cost", "online_training_incremental_cost"):
        observation = cost[name]
        assert observation["measurement_status"] == "PENDING_S2.9"
        assert "seconds" not in observation
    assert cost["artifact_hash"] == canonical_json_hash({key: value for key, value in cost.items() if key != "artifact_hash"})
    retry = build_retry_policy_contract(max_cell_attempts=1)
    assert retry["new_pilot_draws_on_retry"] is False
    assert retry["artifact_hash"] == canonical_json_hash({key: value for key, value in retry.items() if key != "artifact_hash"})
    with pytest.raises(S206FormalInputError, match="MAX_CELL_ATTEMPTS_EXPLICIT_REQUIRED"):
        build_retry_policy_contract()


def test_formal_input_cli_publishes_immutable_contracts(tmp_path: Path) -> None:
    cost_path = tmp_path / "contracts/cost.json"
    retry_path = tmp_path / "contracts/retry.json"
    assert produce_s206_inputs(["cost-semantics", "--output", str(cost_path)]) == 0
    assert produce_s206_inputs(["retry-policy", "--output", str(retry_path), "--max-cell-attempts", "2"]) == 0
    assert produce_s206_inputs(["cost-semantics", "--output", str(cost_path)]) == 0
    assert produce_s206_inputs(["cost-semantics", "--output", str(cost_path), "--cost-io-quiescent"]) == 3
    observed_path = tmp_path / "contracts/cost-observed.json"
    assert produce_s206_inputs(["cost-semantics", "--output", str(observed_path), "--cost-io-quiescent"]) == 0
    observed = load_canonical_json(observed_path)
    assert isinstance(observed, dict) and observed["cost_io_quiescent"] is True


def _write_hashed(root: Path, ref: str, payload: dict[str, object]) -> str:
    payload = dict(payload)
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(root / ref, payload)
    return str(payload["artifact_hash"])


def test_execution_evidence_binds_exact_g23_g24a_and_s205(tmp_path: Path) -> None:
    anchors = ANCHOR_IDS
    g23_cells = [anchor.replace(".", ":", 1) for anchor in anchors]
    g23_ref = "evidence/g23.json"
    g23_hash = _write_hashed(
        tmp_path,
        g23_ref,
        {
            "schema_version": "stage2-g23-reference-evaluation-v1",
            "status": "PASS",
            "formal_eligible": True,
            "required_cell_count": 6,
            "complete_cell_count": 6,
            "cells": [{"cell_id": cell, "status": "PASS", "formal_eligible": True} for cell in g23_cells],
        },
    )
    g24a_ref = "evidence/g24a.json"
    _write_hashed(
        tmp_path,
        g24a_ref,
        {
            "schema_version": "stage2-g24a-formal-evaluation-v1",
            "gate_id": "stage2.G2.4a",
            "status": "PASS",
            "formal_eligible": True,
            "cell_count": 6,
            "g23_evaluation_ref": g23_ref,
            "g23_evaluation_hash": g23_hash,
            "results": [{"cell_id": cell, "status": "PASS", "formal_eligible": True, "metrics": {"h_ref_model_total": 0.01}} for cell in g23_cells],
        },
    )
    s205_ref = "evidence/s205-rebind.json"
    _write_hashed(
        tmp_path,
        s205_ref,
        {
            "schema_version": "stage2-s205-rebind-plan-v1",
            "status": "READY",
            "formal_eligible": True,
            "g23_evaluation_ref": g23_ref,
            "g23_evaluation_hash": g23_hash,
            "cells": [
                {
                    "cell_id": cell,
                    "config_ref": f"prepared/{index}.json",
                    "environment_ref": f"prepared/{index}-env.json",
                    "formal_execution_ref": "evidence/parent.json",
                    "reference_artifact_refs": {},
                    "task_result_status_path": f"prepared/{index}-status.json",
                    "config_hash": "a" * 64,
                    "result_hash": "b" * 64,
                }
                for index, cell in enumerate(g23_cells)
            ],
        },
    )
    parent = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="c" * 64,
        asset_manifest_hashes=("d" * 64,),
        prerequisite_gates=(
            GateRecord(
                gate_id="stage2.G2.2",
                stage=2,
                status=GateStatus.PASS,
                checked_at="2026-08-26T00:00:00+00:00",
                measured={"status": "PASS"},
                threshold={"status": "PASS"},
                evidence_refs=("evidence/g22.json",),
            ),
        ),
        metadata={"legacy_g22": True},
    )
    parent_ref = "evidence/parent.json"
    write_canonical_json(tmp_path / parent_ref, parent.to_dict())
    output = build_formal_execution_evidence(
        data_root=tmp_path,
        g23_evaluation_ref=g23_ref,
        g24a_evaluation_ref=g24a_ref,
        s205_rebind_ref=s205_ref,
        parent_evidence_ref=parent_ref,
        checked_at="2026-08-26T01:00:00+00:00",
    )
    assert output["artifact_hash"] == canonical_json_hash({key: value for key, value in output.items() if key != "artifact_hash"})
    metadata = output["metadata"]
    assert metadata["g23_evaluation_ref"] == g23_ref
    assert metadata["g23_evaluation_hash"] == g23_hash
    assert metadata["g24a_evaluation_ref"] == g24a_ref
    assert metadata["s205_rebind_ref"] == s205_ref
    evidence = FormalExecutionEvidence.from_mapping(output)
    assert {gate.gate_id for gate in evidence.prerequisite_gates} >= {"stage2.G2.2", "stage2.G2.3", "stage2.G2.4a"}

    execution_ref = "evidence/execution.json"
    write_canonical_json(tmp_path / execution_ref, output)
    args = SimpleNamespace(
        data_root=tmp_path,
        execution_evidence_ref=execution_ref,
        g23_evaluation=g23_ref,
        g24a_evaluation=g24a_ref,
        s205_rebind_ref=s205_ref,
    )
    assert _load_formal_execution(args).artifact_hash == output["artifact_hash"]

    # A second valid object is still a different amendment when its ref is
    # changed; the consumer must reject metadata substitution even when the
    # content hash happens to be identical.
    alternate_g23_ref = "evidence/g23-other.json"
    alternate_g23 = load_canonical_json(tmp_path / g23_ref)
    assert isinstance(alternate_g23, dict)
    write_canonical_json(tmp_path / alternate_g23_ref, alternate_g23)
    substituted = dict(output)
    substituted_metadata = dict(output["metadata"])  # type: ignore[arg-type]
    substituted_metadata["g23_evaluation_ref"] = alternate_g23_ref
    substituted["metadata"] = substituted_metadata
    substituted["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in substituted.items() if key != "artifact_hash"}
    )
    substituted_ref = "evidence/execution-substituted.json"
    write_canonical_json(tmp_path / substituted_ref, substituted)
    with pytest.raises(Exception, match="EXECUTION_EVIDENCE_G23_EVALUATION_REF_MISMATCH"):
        _load_formal_execution(SimpleNamespace(**{**vars(args), "execution_evidence_ref": substituted_ref}))

    missing_hash = dict(output)
    missing_metadata = dict(output["metadata"])  # type: ignore[arg-type]
    missing_metadata.pop("g24a_evaluation_hash")
    missing_hash["metadata"] = missing_metadata
    missing_hash["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in missing_hash.items() if key != "artifact_hash"}
    )
    missing_ref = "evidence/execution-missing-hash.json"
    write_canonical_json(tmp_path / missing_ref, missing_hash)
    with pytest.raises(Exception, match="EXECUTION_EVIDENCE_G24A_EVALUATION_HASH_REQUIRED"):
        _load_formal_execution(SimpleNamespace(**{**vars(args), "execution_evidence_ref": missing_ref}))

    blocked_g23 = load_canonical_json(tmp_path / g23_ref)
    assert isinstance(blocked_g23, dict)
    blocked_g23["status"] = "BLOCKED"
    blocked_g23["artifact_hash"] = canonical_json_hash({key: value for key, value in blocked_g23.items() if key != "artifact_hash"})
    write_canonical_json(tmp_path / "evidence/g23-blocked.json", blocked_g23)
    with pytest.raises(S206FormalInputError, match="PASS_FORMAL_REQUIRED"):
        build_formal_execution_evidence(
            data_root=tmp_path,
            g23_evaluation_ref="evidence/g23-blocked.json",
            g24a_evaluation_ref=g24a_ref,
            s205_rebind_ref=s205_ref,
            parent_evidence_ref=parent_ref,
        )
