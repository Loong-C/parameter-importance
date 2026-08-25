from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.preregistration import (
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
)
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s208_g26 import (
    S28G26Blocked,
    analyze_s208_g26,
    two_stage_bootstrap,
)


def _gate(gate_id: str) -> dict[str, object]:
    return GateRecord(
        gate_id=gate_id,
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={},
        threshold={},
        evidence_refs=(f"{gate_id}/evidence.json",),
    ).to_dict()


def _formal_inputs(tmp_path: Path) -> dict[str, object]:
    prereg = build_stage2_preregistration(
        seed_plan_hash="1" * 64,
        producer_commit="2" * 40,
        mathematics_hash="3" * 64,
        stage1_report_hash="4" * 64,
        upstream_binding_hash="5" * 64,
        scope="formal",
    )
    hypothesis = build_stage2_hypothesis_contract(prereg, upstream_binding_hash="5" * 64)
    gates = {gate_id: _gate(gate_id) for gate_id in ("stage2.G2.3", "stage2.G2.4a", "stage2.G2.4b", "stage2.G2.5")}
    cells: dict[str, object] = {}
    refs: dict[str, object] = {}
    units: list[dict[str, object]] = []
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for cell_index, cell in enumerate(EXPECTED_CELL_IDS):
        coordinate_ids = ["a", "b", "c"]
        margins = {
            "model_total_signed_bias": 10.0,
            "layer_total_l1_bias": 10.0,
            "module_total_l1_bias": 10.0,
        }
        cells[cell] = {
            "nmse_denominator": 1.0,
            "margins": margins,
            "reference_half_width": 0.0,
            "numeric_error": 0.0,
        }
        ref_body: dict[str, object] = {
            "schema_version": "reference-result-v1",
            "reference_hash": f"{cell_index + 20:064x}",
            "formal_eligible": True,
            "coordinate_ids": coordinate_ids,
            "vectors": {"bias": [1.0, 1.0, 1.0], "cross": [1.0, 1.0, 1.0], "ranking": [1.0, 1.0, 1.0]},
            "reference_blocks": {
                "bias": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                "cross": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                "ranking": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            },
            "metadata": {"qualification_gate_hash": gates["stage2.G2.3"]["artifact_hash"]},
        }
        ref_body["artifact_hash"] = canonical_json_hash(ref_body)
        refs[cell] = ref_body
        for rep in range(3):
            payload: dict[str, object] = {
                "schema_version": "stage2-wave-unit-state-v1",
                "unit_id": f"{cell}::rep-{rep}",
                "cell_id": cell,
                "reference_hash": ref_body["reference_hash"],
                "batch_size": 32,
                "microbatch_count": 16,
                "coordinate_ids": coordinate_ids,
                "vectors": {"raw": [1.0, 1.0, 1.0], "double": [1.0, 1.0, 1.0], "u_m16": [1.0, 1.0, 1.0]},
            }
            ref = raw_root / f"{cell.replace(':', '_')}-{rep}.json"
            write_canonical_json(ref, payload)
            units.append({
                "unit_id": payload["unit_id"],
                "cell_id": cell,
                "repetition_id": f"rep-{rep}",
                "status": "SUCCESS",
                "raw_artifact_ref": ref.name,
                "raw_artifact_hash": canonical_json_hash(payload),
            })
    matrix: dict[str, object] = {
        "schema_version": "stage2-formal-pilot-matrix-freeze-v1",
        "status": "FORMAL_FROZEN",
        "formal_eligible": True,
        "b_primary": 32,
        "m_primary": 16,
        "qualification_gate_hash": gates["stage2.G2.4b"]["artifact_hash"],
        "cells": cells,
        "group_registry": {"layer0": ["a", "b"], "layer1": ["c"]},
    }
    matrix["artifact_hash"] = canonical_json_hash(matrix)
    manifest: dict[str, object] = {
        "schema_version": "stage2-s27-sealed-raw-manifest-v1",
        "status": "SEALED",
        "formal_eligible": True,
        "matrix_hash": matrix["artifact_hash"],
        "expected_unit_count": len(units),
        "completed_unit_count": len(units),
        "failed_unit_count": 0,
        "units": units,
    }
    manifest["artifact_hash"] = canonical_json_hash(manifest)
    gates["stage2.G2.5"]["measured"] = {"raw_manifest_hash": manifest["artifact_hash"]}
    # Rebuild the G2.5 hash after adding its manifest binding.
    gates["stage2.G2.5"] = _gate("stage2.G2.5")
    # The strict consumer intentionally requires this binding, so make a
    # properly bound PASS record directly.
    g25 = GateRecord(
        gate_id="stage2.G2.5", stage=2, status="PASS", checked_at="2026-08-25T00:00:00+00:00",
        measured={"raw_manifest_hash": manifest["artifact_hash"]}, threshold={}, evidence_refs=("g25/evidence.json",),
    ).to_dict()
    gates["stage2.G2.5"] = g25
    gates_payload = {key: value for key, value in gates.items()}
    return {"prereg": prereg, "hypothesis": hypothesis, "gates": gates_payload, "manifest": manifest, "refs": {"cells": refs}, "matrix": matrix, "raw_root": raw_root}


def test_two_stage_bootstrap_never_resamples_coordinates() -> None:
    result = two_stage_bootstrap([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], replicates=100, seed=7)
    assert result["unit"] == "repetition_with_reference_block_strata"
    assert result["parameter_coordinate_resampling"] is False


def test_s208_requires_valid_preregistration_before_upstream_gates() -> None:
    with pytest.raises(S28G26Blocked, match="PREREGISTRATION_OR_HYPOTHESIS_CONTRACT_INVALID"):
        analyze_s208_g26(
            raw_manifest={}, raw_root=".", references={}, matrix={}, preregistration={}, hypothesis_contract={}, upstream_gates={}
        )


def test_s208_formal_fixture_publishes_machine_artifacts(tmp_path: Path) -> None:
    values = _formal_inputs(tmp_path)
    output = tmp_path / "derived" / "analysis-1"
    result = analyze_s208_g26(
        raw_manifest=values["manifest"], raw_root=values["raw_root"], references=values["refs"], matrix=values["matrix"],
        preregistration=values["prereg"], hypothesis_contract=values["hypothesis"], upstream_gates=values["gates"],
        output_root=output, bootstrap_replicates=100, bootstrap_seed=3,
    )
    assert result["status"] == "PASS"
    assert (output / "quality_gates.json").exists()
    assert (output / "hypothesis_decisions.json").exists()
    family = result["confirmatory_family_decisions"]
    assert len(family["rows"]) == 6 * 2 * 3
    assert family["global"]["double"]["bias_qualified"] is True
