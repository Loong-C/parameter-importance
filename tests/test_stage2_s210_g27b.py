from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.artifacts import validate_estimator_decision_artifact
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s210_g27b import S210G27BBlocked, run_s210_g27b


def _hashed(body: dict[str, object]) -> dict[str, object]:
    return body | {"artifact_hash": canonical_json_hash(body)}


def _g26_gate(status: str = "BLOCKED") -> dict[str, object]:
    body = {
        "schema_version": "stage2-s208-g26-gate-v1",
        "gate_id": "stage2.G2.6",
        "stage": 2,
        "status": status,
        "quality_gate_dependency": status == "PASS",
        "measured": {"matrix_hash": "1" * 64, "raw_manifest_hash": "2" * 64},
        "threshold": {},
        "reasons": [] if status == "PASS" else ["FIXTURE_BLOCKED"],
        "upstream_gate_hashes": {},
    }
    return _hashed(body)


def _g27a_gate() -> dict[str, object]:
    return GateRecord(
        gate_id="stage2.G2.7a",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={},
        threshold={},
        evidence_refs=("cost-system-validation.json",),
    ).to_dict()


def _inputs() -> dict[str, object]:
    methods = ("raw", "double", "u_m16")
    rows: list[dict[str, object]] = []
    for cell in EXPECTED_CELL_IDS:
        model, stage = cell.split(":", 1)
        for method in methods:
            for view in ("bias", "ranking"):
                rows.append(
                    {
                        "cell_id": cell,
                        "model": model,
                        "training_stage": stage,
                        "batch_size": 32,
                        "microbatch_count": 16,
                        "method": method,
                        "scope": "parameter",
                        "reference_view": view,
                        "repetitions": 2,
                        "signed_bias": 0.0,
                        "variance_s2": 1.0,
                        "mse_observed": 1.0,
                        "spearman": 0.9,
                        "negative_fraction": 0.1,
                        "negative_mass": 1.0,
                        "positive_mass": 2.0,
                    }
                )
    long = _hashed({"schema_version": "stage2-s208-g26-analysis-v1", "rows": rows})
    family_rows = [
        {"cell_id": cell, "method": method, "endpoint": endpoint, "state": "FAIL"}
        for cell in EXPECTED_CELL_IDS
        for method in ("double", "u_m16")
        for endpoint in ("model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias")
    ]
    family = _hashed(
        {
            "schema_version": "stage2-s208-confirmatory-family-v1",
            "primary_cells": list(EXPECTED_CELL_IDS),
            "b_primary": 32,
            "m_primary": 16,
            "multiplicity": "intersection_union_across_six_primary_cells",
            "global": {"double": {"bias_qualified": False}, "u_m16": {"bias_qualified": False}},
            "rows": family_rows,
            "noninferiority_rows": [],
        }
    )
    quality = _hashed(
        {
            "schema_version": "stage2-s208-quality-gates-v1",
            "gate_id": "stage2.G2.6",
            "status": "BLOCKED",
            "gates": [{"gate": "reference_convergence", "status": "BLOCKED"}],
            "formal_eligible": False,
        }
    )
    hypothesis = {"schema_version": "stage2-s208-hypothesis-decisions-v1", "decision_states": {"H1": "inconclusive"}}
    cost_body = {
        "schema_version": "stage2-s209-g27a-cost-system-validation-v1",
        "status": "PASS",
        "formal_eligible": True,
        "run_id": "s209-fixture",
        "cost_io_quiescent": True,
        "four_gpu_anchor": {"status": "PASS"},
        "consistency": {"all_pass": True},
        "pareto": {"status": "PASS", "rows": [{"cell_id": EXPECTED_CELL_IDS[0], "method": "raw", "corrected_nmse": 1.0, "mse": 1.0, "spearman": 0.9, "overlap_1pct": 0.8}]},
        "online_training_incremental_cost": {"ratios": {"source": "online_training_incremental_cost", "threshold": 1.25, "methods": {"double": {"wall_seconds": 1.1}, "u": {"wall_seconds": 1.1}}}},
    }
    return {"quality": quality, "hypothesis": hypothesis, "long": long, "family": family, "cost": _hashed(cost_body)}


def _formal_inputs(*, double_qualified: bool) -> dict[str, object]:
    values = _inputs()
    values["g26_gate"] = _g26_gate("PASS")

    quality = dict(values["quality"])
    quality.pop("artifact_hash", None)
    quality["status"] = "PASS"
    quality["formal_eligible"] = True
    quality["gates"] = [{"gate": "reference_convergence", "status": "PASS"}]
    values["quality"] = _hashed(quality)

    family = dict(values["family"])
    family.pop("artifact_hash", None)
    family["global"] = {
        "double": {"bias_qualified": double_qualified},
        "u_m16": {"bias_qualified": True},
    }
    family["rows"] = [
        {
            "cell_id": cell,
            "method": method,
            "endpoint": endpoint,
            "state": "PASS" if method == "u_m16" or double_qualified else "FAIL",
        }
        for cell in EXPECTED_CELL_IDS
        for method in ("double", "u_m16")
        for endpoint in ("model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias")
    ]
    noninferiority_endpoints = (
        "corrected_parameter_nmse_noninferiority",
        "parameter_spearman_noninferiority",
        "parameter_overlap_at_1_percent_noninferiority",
    )
    family["noninferiority_rows"] = [
        {"cell_id": cell, "method": "u_m16", "endpoint": endpoint, "state": "PASS"}
        for cell in EXPECTED_CELL_IDS
        for endpoint in noninferiority_endpoints
    ]
    family["noninferiority_global"] = {
        endpoint: {"all_cells": True} for endpoint in noninferiority_endpoints
    }
    values["family"] = _hashed(family)

    cost = dict(values["cost"])
    cost.pop("artifact_hash", None)
    cost["health_snapshot"] = {"healthy": True, "cost_io_quiescent": True}
    values["cost"] = _hashed(cost)
    values["raw_calibration"] = _hashed(
        {
            "schema_version": "stage2-s208-raw-calibration-v1",
            "rows": [{"cell_id": EXPECTED_CELL_IDS[0], "slope": 1.0, "intercept": 0.0}],
        }
    )
    return values


def test_s210_rejects_missing_sources() -> None:
    with pytest.raises(S210G27BBlocked, match="INPUT_REQUIRED"):
        run_s210_g27b()


def test_s210_publishes_blocked_append_only_report(tmp_path: Path) -> None:
    values = _inputs()
    output = tmp_path / "s210"
    result = run_s210_g27b(
        g26_gate=_g26_gate(),
        g26_quality_gates=values["quality"],
        g26_hypothesis_decisions=values["hypothesis"],
        g26_statistics_long_table=values["long"],
        g26_family_decisions=values["family"],
        g27a_report=values["cost"],
        g27a_gate=_g27a_gate(),
        output_root=output,
        checked_at="2026-08-25T00:00:00+00:00",
    )
    assert result["status"] == "BLOCKED"
    assert result["gate"]["status"] == "BLOCKED"
    assert result["decision"]["selected_estimator"] is None
    assert result["decision"]["gate_status"] == "BLOCKED"
    assert result["report"]["formal_eligible"] is False
    assert (output / "estimator_decision.json").exists()
    with pytest.raises(S210G27BBlocked, match="OUTPUT_ROOT_MUST_BE_NEW"):
        run_s210_g27b(
            g26_gate=_g26_gate(),
            g26_quality_gates=values["quality"],
            g26_hypothesis_decisions=values["hypothesis"],
            g26_statistics_long_table=values["long"],
            g26_family_decisions=values["family"],
            g27a_report=values["cost"],
            g27a_gate=_g27a_gate(),
            output_root=output,
        )


def test_s210_u_only_is_nonformal_provisional_and_blocked(tmp_path: Path) -> None:
    values = _formal_inputs(double_qualified=False)
    result = run_s210_g27b(
        g26_gate=values["g26_gate"],
        g26_quality_gates=values["quality"],
        g26_hypothesis_decisions=values["hypothesis"],
        g26_statistics_long_table=values["long"],
        g26_raw_calibration=values["raw_calibration"],
        g26_family_decisions=values["family"],
        g27a_report=values["cost"],
        g27a_gate=_g27a_gate(),
        output_root=tmp_path / "u-provisional",
        checked_at="2026-08-25T00:00:00+00:00",
    )

    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert result["gate"]["status"] == "BLOCKED"
    decision = result["decision"]
    assert decision["scope"] == "local_fixture"
    assert decision["status"] == "BLOCKED"
    assert decision["state"] == "UNFROZEN"
    assert decision["selected_estimator"] is None
    assert decision["gate_status"] == "BLOCKED"
    assert decision["batch_size"] is None
    assert decision["microbatch_count"] is None
    assert decision["repetitions"] is None
    assert decision["metadata"]["decision_rule"] == "u_provisional_revalidate_double"
    assert decision["metadata"]["provisional"] is True
    assert "DOUBLE_REVALIDATION_REQUIRED" in decision["metadata"]["reasons"]
    assert result["charts"] == []
    validate_estimator_decision_artifact(decision)


def test_s210_formal_selection_still_passes_after_provisional_guard(tmp_path: Path) -> None:
    values = _formal_inputs(double_qualified=True)
    result = run_s210_g27b(
        g26_gate=values["g26_gate"],
        g26_quality_gates=values["quality"],
        g26_hypothesis_decisions=values["hypothesis"],
        g26_statistics_long_table=values["long"],
        g26_raw_calibration=values["raw_calibration"],
        g26_family_decisions=values["family"],
        g27a_report=values["cost"],
        g27a_gate=_g27a_gate(),
        output_root=tmp_path / "formal-pass",
        checked_at="2026-08-25T00:00:00+00:00",
    )

    assert result["status"] == "PASS"
    assert result["formal_eligible"] is True
    assert result["gate"]["status"] == "PASS"
    decision = result["decision"]
    assert decision["scope"] == "formal"
    assert decision["status"] == "SELECTED"
    assert decision["selected_estimator"] == "u"
    assert decision["gate_status"] == "PASS"
    assert decision["metadata"]["provisional"] is False
    assert len(result["charts"]) == 5
    validate_estimator_decision_artifact(decision, require_formal=True)


def test_s210_path_inputs_reject_hash_forgery(tmp_path: Path) -> None:
    values = _inputs()
    paths: dict[str, Path] = {}
    for name, value in (
        ("g26_gate", _g26_gate()),
        ("quality", values["quality"]),
        ("hypothesis", values["hypothesis"]),
        ("long", values["long"]),
        ("family", values["family"]),
        ("cost", values["cost"]),
        ("g27a_gate", _g27a_gate()),
    ):
        path = tmp_path / f"{name}.json"
        write_canonical_json(path, value)
        paths[name] = path
    forged = dict(values["quality"])
    forged["status"] = "PASS"
    # Canonical bytes alone are insufficient: the producer hash must still
    # match the payload body.
    (tmp_path / "quality.json").write_bytes(canonical_json_bytes(forged))
    with pytest.raises(S210G27BBlocked, match="g26_quality_gates:ARTIFACT_HASH_MISMATCH"):
        run_s210_g27b(
            g26_gate=paths["g26_gate"],
            g26_quality_gates=paths["quality"],
            g26_hypothesis_decisions=paths["hypothesis"],
            g26_statistics_long_table=paths["long"],
            g26_family_decisions=paths["family"],
            g27a_report=paths["cost"],
            g27a_gate=paths["g27a_gate"],
        )
