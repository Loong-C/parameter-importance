from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.preregistration import (
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
)
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s208_g26 import (
    S28G26Blocked,
    _top_metrics,
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


def _formal_inputs(tmp_path: Path, *, bounded: bool = False) -> dict[str, object]:
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
            "metadata": {"qualification_gate_hash": gates["stage2.G2.3"]["artifact_hash"]},
        }
        if bounded:
            ref_body.update({
                "reference_uncertainty_mode": "independent_reference_variance_combination",
                "reference_variances": {
                    "bias": [0.0, 0.0, 0.0],
                    "cross": [0.0, 0.0, 0.0],
                    "ranking": [0.0, 0.0, 0.0],
                },
                "reference_uncertainty": {
                    "schema_version": "stage2-reference-uncertainty-v1",
                    "estimator": "block_u_delete_one_jackknife",
                    "confidence_level": 0.95,
                    "block_count_a": 3,
                    "block_count_b": 3,
                },
                "sequence_variance": [0.0, 0.0, 0.0],
            })
        else:
            ref_body["reference_blocks"] = {
                "bias": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                "cross": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                "ranking": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
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


def test_top_k_primary_ranking_preserves_signed_scores() -> None:
    # With absolute-value ranking both vectors would select coordinate 0;
    # signed ranking must select coordinate 1 for the observed vector and
    # coordinate 0 for the reference vector.
    metrics = _top_metrics(
        np.asarray([-10.0, 9.0]),
        np.asarray([8.0, -7.0]),
    )
    assert metrics["overlap_at_0.0001"] == 0.0


def test_nmse_observed_averages_per_repetition_coordinate_sums(tmp_path: Path) -> None:
    values = _formal_inputs(tmp_path)
    cell = EXPECTED_CELL_IDS[0]
    # Keep every bootstrap resample's corrected denominator positive.  The
    # raw method is displaced in one repetition, while U/double remain
    # displaced in all repetitions so a valid resample cannot collapse their
    # non-inferiority ratio to 0/0.
    descriptors = [item for item in values["manifest"]["units"] if item["cell_id"] == cell]
    assert len(descriptors) == 3
    for index, descriptor in enumerate(descriptors):
        payload_path = values["raw_root"] / str(descriptor["raw_artifact_ref"])
        payload = load_canonical_json(payload_path)
        assert isinstance(payload, dict)
        payload["vectors"] = {
            "raw": [3.0, 3.0, 3.0] if index == 0 else [1.0, 1.0, 1.0],
            "double": [2.0, 2.0, 2.0],
            "u_m16": [2.0, 2.0, 2.0],
        }
        write_canonical_json(payload_path, payload)
        descriptor["raw_artifact_hash"] = canonical_json_hash(payload)
    values["manifest"]["artifact_hash"] = canonical_json_hash({key: item for key, item in values["manifest"].items() if key != "artifact_hash"})
    values["gates"]["stage2.G2.5"] = GateRecord(
        gate_id="stage2.G2.5",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={"raw_manifest_hash": values["manifest"]["artifact_hash"]},
        threshold={},
        evidence_refs=("g25/evidence.json",),
    ).to_dict()

    result = analyze_s208_g26(
        raw_manifest=values["manifest"], raw_root=values["raw_root"], references=values["refs"], matrix=values["matrix"],
        preregistration=values["prereg"], hypothesis_contract=values["hypothesis"], upstream_gates=values["gates"],
        bootstrap_replicates=100, bootstrap_seed=3,
    )
    summary = next(item for item in result["statistics_summary"] if item["cell_id"] == cell and item["method"] == "raw")
    # Three coordinates are displaced by +2 in one of three repetitions:
    # (3 * 2^2) / 3 / D_c(=1) = 4, not the squared repetition-mean error 4/3.
    assert summary["corrected_nmse"] == pytest.approx(4.0)


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


def test_s208_bounded_uncertainty_uses_independent_variance_combination(tmp_path: Path) -> None:
    values = _formal_inputs(tmp_path, bounded=True)
    result = analyze_s208_g26(
        raw_manifest=values["manifest"], raw_root=values["raw_root"], references=values["refs"], matrix=values["matrix"],
        preregistration=values["prereg"], hypothesis_contract=values["hypothesis"], upstream_gates=values["gates"],
        bootstrap_replicates=100, bootstrap_seed=11,
    )
    assert result["status"] == "PASS"
    assert all(
        row["bootstrap"]["reference_uncertainty_mode"] == "independent_reference_variance_combination"
        for row in result["confirmatory_family_decisions"]["rows"]
    )
    assert all(row["sequence_variance_source"] == "s204_hash_bound_sequence_variance" for row in result["raw_calibration"])


def test_s208_bounded_uncertainty_missing_variance_fails_closed(tmp_path: Path) -> None:
    values = _formal_inputs(tmp_path, bounded=True)
    cell = EXPECTED_CELL_IDS[0]
    refs = values["refs"]["cells"]
    refs[cell]["reference_variances"].pop("cross")
    refs[cell]["artifact_hash"] = canonical_json_hash({key: value for key, value in refs[cell].items() if key != "artifact_hash"})
    with pytest.raises(S28G26Blocked, match="cross_variance:VECTOR_REQUIRED"):
        analyze_s208_g26(
            raw_manifest=values["manifest"], raw_root=values["raw_root"], references=values["refs"], matrix=values["matrix"],
            preregistration=values["prereg"], hypothesis_contract=values["hypothesis"], upstream_gates=values["gates"],
            bootstrap_replicates=100, bootstrap_seed=11,
        )


def test_s208_bounded_nonzero_ranking_variance_is_inconclusive_not_fake_pass(tmp_path: Path) -> None:
    values = _formal_inputs(tmp_path, bounded=True)
    for reference in values["refs"]["cells"].values():
        reference["reference_variances"]["ranking"] = [1.0e-8, 1.0e-8, 1.0e-8]
        reference["artifact_hash"] = canonical_json_hash({key: value for key, value in reference.items() if key != "artifact_hash"})
    result = analyze_s208_g26(
        raw_manifest=values["manifest"], raw_root=values["raw_root"], references=values["refs"], matrix=values["matrix"],
        preregistration=values["prereg"], hypothesis_contract=values["hypothesis"], upstream_gates=values["gates"],
        bootstrap_replicates=100, bootstrap_seed=11,
    )
    ranking_rows = [
        row for row in result["confirmatory_family_decisions"]["noninferiority_rows"]
        if row["endpoint"] in {"parameter_spearman_noninferiority", "parameter_overlap_at_1_percent_noninferiority"}
    ]
    assert ranking_rows and all(row["state"] == "INCONCLUSIVE" for row in ranking_rows)
    assert all(row["interval"]["raw_reference_blocks_reconstructed"] is False for row in ranking_rows)
